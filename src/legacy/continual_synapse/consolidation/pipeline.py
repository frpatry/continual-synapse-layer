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
import json
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from continual_synapse.cold_storage.compression import (
    CompressionSchedule,
    quantize,
)
from continual_synapse.cold_storage.store import ColdStorage
from continual_synapse.consolidation.trigger import ConsolidationTrigger
from continual_synapse.synapse_layer.layer import SynapseLayer


@dataclass
class ConsolidationOutcome:
    """Result of one call to :func:`consolidate_to_storage`.

    Replaces the prior ``str | None`` return so callers can tell
    apart "trigger declined", "stored a new entry", and "merged
    into an existing entry" — the third case becomes possible with
    the amplification-variant ``merge_threshold`` flag.

    Attributes:
        entry_id: The id of the freshly-stored cold-storage entry,
            or ``None`` if no new entry was created. Truthy only on
            new-entry outcomes.
        merged_into: When ``was_merged`` is True, the id of the
            existing entry whose access_count (and possibly
            embedding) was updated. ``None`` otherwise.
        was_merged: True iff the cycle resolved by merging into an
            existing entry rather than creating a new one.
    """

    entry_id: str | None
    merged_into: str | None
    was_merged: bool

    @property
    def fired(self) -> bool:
        """True iff a consolidation cycle actually completed (new or merged)."""
        return self.entry_id is not None or self.was_merged


def consolidate_to_storage(
    synapse: SynapseLayer,
    store: ColdStorage,
    trigger: ConsolidationTrigger,
    activation_embedding: torch.Tensor,
    *,
    schedule: CompressionSchedule | None = None,
    force: bool = False,
    drain: bool = True,
    merge_threshold: float = 1.0,
    merge_access_bump: float = 10.0,
    embedding_running_average_decay: float = 0.9,
    task_id: int = -1,
    true_label: int | None = None,
    label_histogram: Sequence[int] | None = None,
) -> ConsolidationOutcome:
    """Run one consolidation cycle.

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
        drain: When True (the default), reset strength / evidence /
            access_count on the candidate synapses after archival —
            the pre-amplification behaviour. When False (the
            amplified variant), source synapses stay live in working
            memory after their pattern is copied to cold storage; the
            archive is purely additive.
        merge_threshold: Cosine-similarity-like cut-off
            (``1 / (1 + distance) > merge_threshold``) above which a
            proposed new entry collapses into the nearest existing
            entry rather than spawning a new row in the store.
            ``1.0`` (the default) is effectively never-merge because
            cosine similarity is at most 1 — preserving the
            pre-amplification one-new-entry-per-cycle behaviour bit-
            exact. ``0.85`` is the amplified-variant default.
        merge_access_bump: When a merge fires, the existing entry's
            ``access_count`` is incremented by this amount. The
            recommended value (``10``) is large enough that one merge
            event meaningfully strengthens the entry in the
            consolidation-trigger pressure metric, but not so large
            that one merge dominates.
        embedding_running_average_decay: When a merge fires, the
            existing entry's embedding is replaced with
            ``decay * old + (1 - decay) * new``. Set to ``1.0`` to
            keep the original embedding untouched.
        task_id: Identifier of the task currently being trained.
            Stored in the new entry's metadata under ``"task_id"``
            so that retrieval can apply a task-recency weighting.
            Default ``-1`` means "untagged" — readers treat untagged
            entries as having no task affinity and skip the recency
            adjustment for them.
        true_label: Path-A label storage. Dominant ground-truth class
            from the batch that triggered this consolidation. When
            supplied, written into metadata as ``"true_label"`` (int).
            Default ``None`` omits the field entirely so older
            checkpoints stay schema-compatible. Readers must use
            ``metadata.get("true_label")`` and fall back to derived
            labels when absent.
        label_histogram: Optional per-class count vector for the
            batch (length = num_classes). When supplied, JSON-encoded
            into metadata as ``"label_histogram_json"`` (Chroma
            metadata only accepts scalar values, so the list is
            serialised). Default ``None`` omits the field.

    Returns:
        A :class:`ConsolidationOutcome` describing what happened.
        ``outcome.fired`` is False when the trigger declined or
        there were no candidates; otherwise it is True and
        ``entry_id`` xor ``merged_into`` is set.
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
        return ConsolidationOutcome(
            entry_id=None, merged_into=None, was_merged=False
        )

    schedule = schedule or CompressionSchedule()

    mask = trigger.candidate_mask(synapse)
    if not mask.any():
        # Nothing to archive — refuse to write an empty entry.
        return ConsolidationOutcome(
            entry_id=None, merged_into=None, was_merged=False
        )

    with torch.no_grad():
        strengths_to_archive = (
            synapse.strengths * mask.to(synapse.strengths.dtype)
        )
        embedding = activation_embedding.detach().to(torch.float32).cpu()

    # Repeat-consolidation merging. When the proposed embedding is
    # close to an existing entry, fold this consolidation into that
    # entry instead of growing the store. merge_threshold=1.0 (the
    # default) makes this a no-op because the similarity metric is
    # bounded above by 1.
    if merge_threshold < 1.0 and store.count() > 0:
        nearest = store.retrieve_similar(embedding.tolist(), k=1)
        if nearest:
            top = nearest[0]
            d = top.distance if top.distance is not None else float("inf")
            similarity = 1.0 / (1.0 + max(float(d), 0.0))
            if similarity > merge_threshold:
                _merge_into_existing(
                    store,
                    existing=top,
                    new_embedding=embedding,
                    access_bump=merge_access_bump,
                    embedding_decay=embedding_running_average_decay,
                )
                if drain:
                    _drain_candidates(synapse, mask)
                trigger.mark_fired(synapse)
                return ConsolidationOutcome(
                    entry_id=None, merged_into=top.id, was_merged=True
                )

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
        "task_id": int(task_id),
    }
    if true_label is not None:
        metadata["true_label"] = int(true_label)
    if label_histogram is not None:
        metadata["label_histogram_json"] = json.dumps(
            [int(c) for c in label_histogram]
        )
    entry_id = store.store_cluster(
        embedding=embedding.tolist(),
        metadata=metadata,
        document=document,
    )

    if drain:
        _drain_candidates(synapse, mask)
    trigger.mark_fired(synapse)
    return ConsolidationOutcome(
        entry_id=entry_id, merged_into=None, was_merged=False
    )


def _merge_into_existing(
    store: ColdStorage,
    *,
    existing,  # StoredEntry
    new_embedding: torch.Tensor,
    access_bump: float,
    embedding_decay: float,
) -> None:
    """Update ``existing`` in place: bump access_count, optionally update
    embedding via running average.

    ColdStorage does not natively support changing an entry's
    embedding (update_entry is fixed-embedding by design); we
    delete-and-reinsert with the same id when embedding_decay < 1.0.
    """
    new_meta = dict(existing.metadata)
    new_meta["access_count"] = int(
        new_meta.get("access_count", 0) + access_bump
    )
    if embedding_decay >= 1.0:
        # Embedding stays fixed; metadata update is enough.
        store.update_metadata(existing.id, new_meta)
        return
    old_embedding = torch.tensor(existing.embedding, dtype=torch.float32)
    blended = (
        embedding_decay * old_embedding
        + (1.0 - embedding_decay) * new_embedding.to(torch.float32)
    )
    document = existing.document
    store.delete_cluster(existing.id)
    store.store_cluster(
        embedding=blended.tolist(),
        metadata=new_meta,
        document=document,
        entry_id=existing.id,
    )


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
