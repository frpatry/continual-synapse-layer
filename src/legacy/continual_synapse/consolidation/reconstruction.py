"""Reconstructive retrieval from cold storage.

The other half of the consolidation cycle: at forward time, the
synapse layer asks "have I seen anything like this activation
pattern before?" and reconstructs an approximate strengths matrix
from cold storage to augment its current correction.

Two design points matter for the project hypothesis:

1. **Context-dependent.** The query is the current activation
   pattern. If the current input is similar to past inputs we
   archived, we recover their strengths; if not, we get back
   near-zero. This is the mechanism that should resolve the
   Phase-3.5 "universal correction conflicts with per-task heads"
   problem — under multi-head, each task produces a different
   activation distribution, so each task pulls a different slice
   of cold storage.
2. **Approximate by design.** Compression at 16/8/4-bit precision
   loses information. Reconstruction returns a weighted sum of
   nearest entries, so even at 32-bit precision the result is a
   blend, not an exact replay. That is the desired behaviour: the
   archive holds gist, not detail.

A side-effect of retrieval is updating each retrieved entry's
``access_count`` metadata. This feeds back into the compression
schedule (rarely-accessed entries are squeezed first) and into
the pressure metric of any future consolidation cycle.
"""

from __future__ import annotations

import base64
import math
from typing import Any

import torch

from continual_synapse.cold_storage.compression import dequantize
from continual_synapse.cold_storage.store import ColdStorage, StoredEntry


def reconstruct_strengths(
    store: ColdStorage,
    query_activation: torch.Tensor,
    *,
    k: int,
    n_neurons: int,
    weighting: str = "similarity",
    bump_access_count: bool = True,
    confidence_exponent: float = 0.0,
    out_retrieved_meta: list[tuple[str, int]] | None = None,
    current_task_id: int | None = None,
    task_recency_decay: float = 0.0,
) -> torch.Tensor:
    """Return a ``(n_neurons, n_neurons)`` reconstructed strengths matrix.

    Args:
        store: Cold-storage backend to query.
        query_activation: ``(n_neurons,)`` activation pattern used
            as the query embedding.
        k: Maximum number of cold-storage entries to combine.
        n_neurons: Expected shape; entries with a different
            ``n_neurons`` in their metadata are skipped.
        weighting: ``"similarity"`` (default) weights each entry
            by ``1 / (1 + distance)``; ``"uniform"`` weights all
            retrieved entries equally.
        bump_access_count: When True (the default), each retrieved
            entry's ``access_count`` metadata is incremented by 1.
            Set to False for diagnostic queries that should not
            disturb the schedule.
        confidence_exponent: Power applied to ``(1 + access_count)``
            before mixing it into the per-entry weight. ``0.0`` (the
            default) leaves weights bit-exact equivalent to the
            pre-amplification path because ``x**0 == 1``. Positive
            values upweight frequently-accessed (i.e. proven-useful)
            entries; 0.5 is the amplified-variant default. The factor
            multiplies the similarity weight, so the combined form
            is ``weight = (1 / (1 + d)) * (1 + access_count) ** k``
            in similarity mode, and ``weight = (1 + access_count) ** k``
            in uniform mode.
        out_retrieved_meta: Optional list the function appends
            ``(entry_id, pre_bump_access_count)`` tuples to for each
            entry actually used in the reconstruction. Lets the caller
            measure how stale or how popular the retrieved entries are
            without re-querying the store. Pre-bump means the value
            recorded is the access_count *before* ``bump_access_count``
            increments it on this call.
        current_task_id: Identifier of the task the caller is
            currently training on. Combined with ``task_recency_decay``
            (and the per-entry ``task_id`` metadata written by
            ``consolidate_to_storage``) to scale each entry's weight
            by ``exp(-decay * max(current_task_id - entry_task_id, 0))``.
            ``None`` (the default) disables the task-recency factor
            entirely.
        task_recency_decay: Decay constant in the recency formula
            above. ``0.0`` (the default) leaves weights bit-exact
            equivalent to the pre-task-aware path because
            ``exp(0) == 1``. A value of ``0.5`` means entries from the
            previous task contribute at ~60% weight, entries two
            tasks back at ~37%, etc. Entries that were stored without
            a task_id (metadata value ``-1``) are not scaled.

    Returns:
        A float32 ``(n_neurons, n_neurons)`` tensor on the same
        device as ``query_activation``. Zeros if the store is empty
        or no entries had a matching ``n_neurons``.
    """
    if query_activation.ndim != 1:
        raise ValueError(
            f"query_activation must be 1-D, got shape "
            f"{tuple(query_activation.shape)}"
        )
    if query_activation.shape[0] != n_neurons:
        raise ValueError(
            f"query_activation dim {query_activation.shape[0]} does not "
            f"match n_neurons={n_neurons}"
        )
    if weighting not in ("similarity", "uniform"):
        raise ValueError(
            f"weighting must be 'similarity' or 'uniform', got {weighting!r}"
        )

    device = query_activation.device
    out = torch.zeros(n_neurons, n_neurons, device=device, dtype=torch.float32)
    if store.count() == 0 or k <= 0:
        return out

    entries = store.retrieve_similar(query_activation.detach().cpu().tolist(), k)
    if not entries:
        return out

    weights: list[float] = []
    matrices: list[torch.Tensor] = []
    bumped: list[tuple[str, dict[str, Any]]] = []
    for entry in entries:
        meta = entry.metadata
        n_entry = int(meta.get("n_neurons", -1))
        if n_entry != n_neurons:
            continue
        precision = int(meta.get("precision", 32))
        try:
            blob = base64.b64decode(entry.document)
        except Exception:  # pragma: no cover — would indicate store corruption
            continue
        matrices.append(
            dequantize(blob, precision=precision, shape=(n_neurons, n_neurons))
        )
        pre_bump_access = int(meta.get("access_count", 0))
        if weighting == "uniform":
            w = 1.0
        else:
            d = entry.distance if entry.distance is not None else 0.0
            w = 1.0 / (1.0 + max(float(d), 0.0))
        # Confidence weighting: ``(1 + access_count) ** k``. ``k = 0``
        # collapses to a constant factor of 1.0 so the weighted sum is
        # bit-exact equivalent to the pre-amplification path.
        if confidence_exponent != 0.0:
            w = w * (1.0 + pre_bump_access) ** float(confidence_exponent)
        # Task-recency weighting: ``exp(-decay * task_distance)``.
        # decay=0 collapses to exp(0)=1; entries without a task_id
        # (-1 = "untagged") are skipped so older runs' archives are
        # used as-is.
        if (
            task_recency_decay != 0.0
            and current_task_id is not None
        ):
            entry_task_id = int(meta.get("task_id", -1))
            if entry_task_id >= 0:
                distance = max(int(current_task_id) - entry_task_id, 0)
                w = w * math.exp(
                    -float(task_recency_decay) * float(distance)
                )
        weights.append(w)

        if out_retrieved_meta is not None:
            out_retrieved_meta.append((entry.id, pre_bump_access))

        if bump_access_count:
            new_meta = dict(meta)
            new_meta["access_count"] = pre_bump_access + 1
            bumped.append((entry.id, new_meta))

    if not matrices:
        return out

    total = sum(weights)
    if total <= 0.0:
        return out
    for w, m in zip(weights, matrices):
        out.add_(m.to(device=device, dtype=torch.float32), alpha=w / total)

    for entry_id, new_meta in bumped:
        store.update_metadata(entry_id, new_meta)

    return out


def fetch_entries_for_query(
    store: ColdStorage,
    query_activation: torch.Tensor,
    *,
    k: int,
) -> list[StoredEntry]:
    """Diagnostic helper: return raw retrieved entries without reconstruction.

    Useful for tests and notebooks that want to inspect which
    archive entries were matched. Does not bump ``access_count``.
    """
    if query_activation.ndim != 1:
        raise ValueError("query_activation must be 1-D")
    if store.count() == 0 or k <= 0:
        return []
    return store.retrieve_similar(
        query_activation.detach().cpu().tolist(), k
    )
