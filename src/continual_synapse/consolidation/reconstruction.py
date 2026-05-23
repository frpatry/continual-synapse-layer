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
        if weighting == "uniform":
            w = 1.0
        else:
            d = entry.distance if entry.distance is not None else 0.0
            w = 1.0 / (1.0 + max(float(d), 0.0))
        weights.append(w)

        if bump_access_count:
            new_meta = dict(meta)
            new_meta["access_count"] = int(meta.get("access_count", 0)) + 1
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
