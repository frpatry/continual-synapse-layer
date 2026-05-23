"""Tests for the consolidation pipeline (synapse → cold storage)."""

from __future__ import annotations

import base64
import pytest
import torch

from continual_synapse.cold_storage.compression import dequantize
from continual_synapse.cold_storage.store import ColdStorage
from continual_synapse.consolidation.pipeline import consolidate_to_storage
from continual_synapse.consolidation.trigger import ConsolidationTrigger
from continual_synapse.synapse_layer.layer import SynapseLayer


def _populated_layer(
    n: int = 4, strength_scale: float = 2.0, seed: int = 0
) -> SynapseLayer:
    g = torch.Generator().manual_seed(seed)
    layer = SynapseLayer(n_neurons=n)
    with torch.no_grad():
        layer.strengths.copy_(torch.randn(n, n, generator=g) * strength_scale)
        layer.evidence.copy_(torch.rand(n, n, generator=g) + 1.0)
        layer.access_count.copy_(
            torch.randint(0, 3, (n, n), generator=g)
        )
        layer.global_step.fill_(100)
    return layer


def _trigger(threshold: float = 0.0) -> ConsolidationTrigger:
    return ConsolidationTrigger(
        avg_pressure_threshold=threshold,
        min_steps_between=0,
        candidate_quantile=0.25,
    )


def test_consolidate_fires_and_stores_entry() -> None:
    layer = _populated_layer()
    store = ColdStorage(collection_name="pipe_fire")
    embedding = torch.arange(4, dtype=torch.float32)
    eid = consolidate_to_storage(
        layer, store, _trigger(), activation_embedding=embedding
    )
    assert isinstance(eid, str)
    assert store.count() == 1
    entry = store.get_by_id(eid)
    assert entry.embedding == embedding.tolist()
    assert entry.metadata["precision"] == 32
    assert entry.metadata["n_neurons"] == 4
    assert entry.metadata["num_candidates"] > 0


def test_consolidate_returns_none_when_trigger_declines() -> None:
    layer = _populated_layer()
    store = ColdStorage(collection_name="pipe_no_fire")
    # Threshold absurdly high → won't fire.
    trigger = ConsolidationTrigger(
        avg_pressure_threshold=1e9, min_steps_between=0
    )
    out = consolidate_to_storage(
        layer, store, trigger, activation_embedding=torch.zeros(4)
    )
    assert out is None
    assert store.count() == 0


def test_force_bypasses_trigger() -> None:
    layer = _populated_layer()
    store = ColdStorage(collection_name="pipe_force")
    trigger = ConsolidationTrigger(
        avg_pressure_threshold=1e9, min_steps_between=0
    )
    eid = consolidate_to_storage(
        layer, store, trigger, activation_embedding=torch.zeros(4), force=True
    )
    assert eid is not None


def test_drain_resets_strength_evidence_access_on_candidates() -> None:
    layer = _populated_layer()
    store = ColdStorage(collection_name="pipe_drain")
    # Snapshot the trigger's candidate mask before firing, so we can verify
    # exactly which entries should have been drained.
    trigger = _trigger()
    mask = trigger.candidate_mask(layer).clone()
    consolidate_to_storage(
        layer, store, trigger, activation_embedding=torch.zeros(4)
    )
    # Candidate positions: zero in strength, evidence, access_count.
    assert torch.all(layer.strengths[mask] == 0)
    assert torch.all(layer.evidence[mask] == 0)
    assert torch.all(layer.access_count[mask] == 0)


def test_drain_preserves_non_candidate_state() -> None:
    layer = _populated_layer()
    store = ColdStorage(collection_name="pipe_preserve")
    trigger = _trigger()
    mask = trigger.candidate_mask(layer).clone()
    keep_strengths = layer.strengths[~mask].clone()
    keep_evidence = layer.evidence[~mask].clone()
    keep_access = layer.access_count[~mask].clone()
    consolidate_to_storage(
        layer, store, trigger, activation_embedding=torch.zeros(4)
    )
    torch.testing.assert_close(layer.strengths[~mask], keep_strengths)
    torch.testing.assert_close(layer.evidence[~mask], keep_evidence)
    assert torch.equal(layer.access_count[~mask], keep_access)


def test_drain_leaves_age_and_confidence_intact() -> None:
    """Per spec, only strength/evidence/access_count are drained."""
    layer = _populated_layer()
    with torch.no_grad():
        layer.age.fill_(7)
        layer.confidence.fill_(3.5)
    store = ColdStorage(collection_name="pipe_age_conf")
    consolidate_to_storage(
        layer, store, _trigger(), activation_embedding=torch.zeros(4)
    )
    assert torch.all(layer.age == 7)
    assert torch.all(layer.confidence == 3.5)


def test_stored_document_round_trips_through_dequantize() -> None:
    """The archive entry should be reconstructable into the candidate-only
    strengths matrix that was archived."""
    layer = _populated_layer()
    store = ColdStorage(collection_name="pipe_round_trip")
    trigger = _trigger()
    mask = trigger.candidate_mask(layer).clone()
    expected = (layer.strengths * mask.to(layer.strengths.dtype)).clone()
    eid = consolidate_to_storage(
        layer, store, trigger, activation_embedding=torch.zeros(4)
    )
    entry = store.get_by_id(eid)
    blob = base64.b64decode(entry.document)
    recovered = dequantize(
        blob, precision=entry.metadata["precision"], shape=(4, 4)
    )
    # 32-bit precision is exact.
    torch.testing.assert_close(recovered, expected)


def test_consolidate_rejects_bad_embedding_shape() -> None:
    layer = _populated_layer()
    store = ColdStorage(collection_name="pipe_bad_emb")
    with pytest.raises(ValueError, match="1-D"):
        consolidate_to_storage(
            layer, store, _trigger(), activation_embedding=torch.zeros(2, 4)
        )
    with pytest.raises(ValueError, match="expected"):
        consolidate_to_storage(
            layer, store, _trigger(), activation_embedding=torch.zeros(5)
        )


def test_consolidate_refuses_empty_candidate_mask() -> None:
    """If the quantile boundary places no synapse above the cutoff
    (uniformly-zero pressure), the pipeline returns None."""
    layer = SynapseLayer(n_neurons=3)
    # All zeros → all pressures zero → candidate mask via quantile equals
    # the entire tensor, but should_fire would also be False at threshold=0.
    # Use force=True to bypass and verify the empty-mask early return.
    with torch.no_grad():
        # Make pressure exactly zero everywhere so the mask is all-True
        # — but pre-loaded strengths zero means the archived matrix is zero.
        # That's still a valid (if useless) entry, so we test the *truly*
        # empty mask case by mocking the trigger.
        pass
    # Instead, exercise the empty-mask branch via a custom trigger.
    layer = _populated_layer()

    class _EmptyTrigger(ConsolidationTrigger):
        def candidate_mask(self, _syn):  # type: ignore[override]
            return torch.zeros_like(_syn.strengths, dtype=torch.bool)

    store = ColdStorage(collection_name="pipe_empty_mask")
    trigger = _EmptyTrigger(
        avg_pressure_threshold=0.0, min_steps_between=0
    )
    out = consolidate_to_storage(
        layer, store, trigger, activation_embedding=torch.zeros(4)
    )
    assert out is None
    assert store.count() == 0
