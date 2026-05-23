"""Consolidation pipeline — synapse working memory to cold storage.

When a :class:`ConsolidationTrigger` decides the working set is
saturated, this module takes the high-pressure candidate synapses,
compresses them, archives them to cold storage, and drains the
candidate cells so the layer can keep learning fresh patterns.

Phase 4 v1 keeps the design intentionally simple:

- One archive entry per consolidation cycle (no k-means clustering
  yet — DESIGN.md §3.5 lists it as a refinement; in v1 a single
  entry per cycle still demonstrates the storage / retrieval loop
  and is easier to reason about for the tests).
- The entry's embedding is the *current activation pattern* (the
  caller supplies it). At retrieval time, similar activation
  patterns will match.
- The entry's document is the strengths matrix with non-candidate
  entries zeroed, compressed at the schedule's "fresh" precision
  (32-bit by default).
- Drain semantics follow the spec literally: reset ``strength``,
  ``evidence``, ``access_count`` on candidate synapses. ``age``
  and ``confidence`` are left intact — they track the synapse cell
  itself, not the pattern stored there.
"""

from __future__ import annotations

import base64
from typing import Any

import torch

from continual_synapse.cold_storage.compression import (
    CompressionSchedule,
    quantize,
)
from continual_synapse.cold_storage.store import ColdStorage
from continual_synapse.consolidation.trigger import ConsolidationTrigger
from continual_synapse.synapse_layer.layer import SynapseLayer


def consolidate_to_storage(
    synapse: SynapseLayer,
    store: ColdStorage,
    trigger: ConsolidationTrigger,
    activation_embedding: torch.Tensor,
    *,
    schedule: CompressionSchedule | None = None,
    force: bool = False,
) -> str | None:
    """Run one consolidation cycle. Returns the new entry id or ``None``.

    Args:
        synapse: The working-memory layer being drained.
        store: Cold-storage backend.
        trigger: Decides whether to fire (unless ``force=True``).
        activation_embedding: ``(n_neurons,)`` tensor that becomes
            the new entry's embedding key. Detached and cloned
            before storage.
        schedule: Compression schedule. Defaults to a fresh
            :class:`CompressionSchedule` if omitted.
        force: Skip the trigger check. Useful for tests and for
            end-of-task forced consolidations.

    Returns:
        The id of the new cold-storage entry on a successful cycle,
        or ``None`` if the trigger declined to fire.
    """
    if activation_embedding.ndim != 1:
        raise ValueError(
            f"activation_embedding must be 1-D, got shape "
            f"{tuple(activation_embedding.shape)}"
        )
    if activation_embedding.shape[0] != synapse.n_neurons:
        raise ValueError(
            f"activation_embedding has dim "
            f"{activation_embedding.shape[0]}; expected "
            f"{synapse.n_neurons}"
        )

    if not force and not trigger.should_fire(synapse):
        return None

    schedule = schedule or CompressionSchedule()

    mask = trigger.candidate_mask(synapse)
    if not mask.any():
        # Nothing to archive — refuse to write an empty entry.
        return None

    with torch.no_grad():
        strengths_to_archive = (
            synapse.strengths * mask.to(synapse.strengths.dtype)
        )
        embedding = activation_embedding.detach().to(torch.float32).cpu()

    precision = schedule.precision_for(age=0, access_count=0)
    blob = quantize(strengths_to_archive, precision=precision)
    document = base64.b64encode(blob).decode("ascii")

    n_neurons = synapse.n_neurons
    step = int(synapse.global_step.item())
    metadata: dict[str, Any] = {
        "precision": int(precision),
        "n_neurons": int(n_neurons),
        "age": 0,
        "access_count": 0,
        "created_at_step": int(step),
        "num_candidates": int(mask.sum().item()),
    }
    entry_id = store.store_cluster(
        embedding=embedding.tolist(),
        metadata=metadata,
        document=document,
    )

    _drain_candidates(synapse, mask)
    trigger.mark_fired(synapse)
    return entry_id


def _drain_candidates(
    synapse: SynapseLayer, mask: torch.Tensor
) -> None:
    """Reset strength, evidence, access_count where ``mask`` is True.

    ``age`` and ``confidence`` are deliberately left alone — they
    track the synapse cell's history, which survives the
    consolidation event.
    """
    with torch.no_grad():
        keep = (~mask).to(synapse.strengths.dtype)
        synapse.strengths.mul_(keep)
        synapse.evidence.mul_(keep)
        keep_long = (~mask).to(synapse.access_count.dtype)
        synapse.access_count.mul_(keep_long)
