"""Tests for the consolidation pipeline (synapse → cold storage)."""

from __future__ import annotations

import base64
import json

import pytest
import torch

from continual_synapse.cold_storage.compression import dequantize, quantize
from continual_synapse.cold_storage.store import ColdStorage
from continual_synapse.consolidation.pipeline import (
    ConsolidationOutcome,
    consolidate_to_storage,
)
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
    outcome = consolidate_to_storage(
        layer, store, _trigger(), activation_embedding=embedding
    )
    assert outcome.fired
    assert not outcome.was_merged
    assert isinstance(outcome.entry_id, str)
    assert store.count() == 1
    entry = store.get_by_id(outcome.entry_id)
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
    outcome = consolidate_to_storage(
        layer, store, trigger, activation_embedding=torch.zeros(4)
    )
    assert not outcome.fired
    assert outcome.entry_id is None
    assert outcome.merged_into is None
    assert store.count() == 0


def test_force_bypasses_trigger() -> None:
    layer = _populated_layer()
    store = ColdStorage(collection_name="pipe_force")
    trigger = ConsolidationTrigger(
        avg_pressure_threshold=1e9, min_steps_between=0
    )
    outcome = consolidate_to_storage(
        layer, store, trigger, activation_embedding=torch.zeros(4), force=True
    )
    assert outcome.fired and outcome.entry_id is not None


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
    outcome = consolidate_to_storage(
        layer, store, trigger, activation_embedding=torch.zeros(4)
    )
    entry = store.get_by_id(outcome.entry_id)
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
    outcome = consolidate_to_storage(
        layer, store, trigger, activation_embedding=torch.zeros(4)
    )
    assert not outcome.fired
    assert outcome.entry_id is None and outcome.merged_into is None
    assert store.count() == 0


# ---- Amplification variant: change 3 (no-drain) ----


def test_drain_default_true_zeros_candidates() -> None:
    """Regression check: the default drain=True preserves the original
    behaviour where candidate synapses are reset post-archival."""
    layer = _populated_layer()
    store = ColdStorage(collection_name="pipe_drain_default")
    trigger = _trigger()
    mask = trigger.candidate_mask(layer).clone()
    consolidate_to_storage(
        layer, store, trigger, activation_embedding=torch.zeros(4)
    )
    assert torch.all(layer.strengths[mask] == 0)
    assert torch.all(layer.evidence[mask] == 0)


def test_no_drain_keeps_synapse_state_intact() -> None:
    """With drain=False, candidate synapses survive post-archival."""
    layer = _populated_layer()
    store = ColdStorage(collection_name="pipe_no_drain")
    trigger = _trigger()
    mask = trigger.candidate_mask(layer).clone()
    pre_strengths = layer.strengths.clone()
    pre_evidence = layer.evidence.clone()
    pre_access = layer.access_count.clone()
    outcome = consolidate_to_storage(
        layer, store, trigger,
        activation_embedding=torch.zeros(4),
        drain=False,
    )
    assert outcome.fired and not outcome.was_merged
    # Source synapses are bit-exact untouched.
    torch.testing.assert_close(layer.strengths, pre_strengths)
    torch.testing.assert_close(layer.evidence, pre_evidence)
    assert torch.equal(layer.access_count, pre_access)
    # But the archive still received the candidate-only matrix.
    assert store.count() == 1


# ---- Amplification variant: change 4 (repeat-consolidation merging) ----


def _seed_one_entry(
    store: ColdStorage,
    embedding: list[float],
    strengths: torch.Tensor,
    access_count: int = 3,
    entry_id: str = "seed",
) -> None:
    blob = quantize(strengths, precision=32)
    doc = base64.b64encode(blob).decode("ascii")
    store.store_cluster(
        embedding=embedding,
        metadata={
            "precision": 32, "n_neurons": int(strengths.shape[0]),
            "age": 0, "access_count": int(access_count),
            "created_at_step": 0, "num_candidates": 0,
        },
        document=doc,
        entry_id=entry_id,
    )


def test_default_threshold_one_never_merges() -> None:
    """merge_threshold=1.0 (default) means similarity must exceed 1, which
    cosine cannot — every cycle creates a new entry. Backward compat."""
    layer = _populated_layer()
    store = ColdStorage(collection_name="pipe_merge_default")
    _seed_one_entry(store, embedding=[1.0, 2.0, 3.0, 4.0],
                    strengths=torch.zeros(4, 4))
    outcome = consolidate_to_storage(
        layer, store, _trigger(),
        # Exact embedding match → similarity = 1, but threshold = 1
        # strictly excludes equality (we use > not >=).
        activation_embedding=torch.tensor([1.0, 2.0, 3.0, 4.0]),
    )
    assert outcome.fired
    assert outcome.entry_id is not None
    assert not outcome.was_merged
    assert store.count() == 2  # original + new


def test_merge_fires_when_similarity_above_threshold() -> None:
    """A close embedding bumps access_count of the existing entry and
    does not create a new row."""
    layer = _populated_layer()
    store = ColdStorage(collection_name="pipe_merge_hit")
    _seed_one_entry(store, embedding=[1.0, 0.0, 0.0, 0.0],
                    strengths=torch.zeros(4, 4), access_count=5)
    pre_count = store.count()
    outcome = consolidate_to_storage(
        layer, store, _trigger(),
        activation_embedding=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        merge_threshold=0.5,  # Threshold permissive enough to fire.
        merge_access_bump=10,
        embedding_running_average_decay=1.0,  # Keep embedding pinned for assert.
    )
    assert outcome.fired and outcome.was_merged
    assert outcome.entry_id is None
    assert outcome.merged_into == "seed"
    assert store.count() == pre_count  # No new entry.
    # access_count bumped by 10.
    assert store.get_by_id("seed").metadata["access_count"] == 15


def test_merge_does_not_fire_when_similarity_below_threshold() -> None:
    """A far embedding does not merge and falls through to new-entry."""
    layer = _populated_layer()
    store = ColdStorage(collection_name="pipe_merge_miss")
    _seed_one_entry(store, embedding=[10.0, 10.0, 10.0, 10.0],
                    strengths=torch.zeros(4, 4))
    outcome = consolidate_to_storage(
        layer, store, _trigger(),
        activation_embedding=torch.tensor([0.0, 0.0, 0.0, 0.0]),
        merge_threshold=0.9,
    )
    assert outcome.fired
    assert outcome.entry_id is not None
    assert not outcome.was_merged
    assert store.count() == 2


def test_merge_updates_embedding_via_running_average() -> None:
    """With decay < 1, the merged entry's embedding moves toward the new one."""
    layer = _populated_layer()
    store = ColdStorage(collection_name="pipe_merge_blend")
    _seed_one_entry(store, embedding=[1.0, 0.0, 0.0, 0.0],
                    strengths=torch.zeros(4, 4))
    new_embedding = torch.tensor([1.0, 0.01, 0.0, 0.0])  # Close enough to merge.
    consolidate_to_storage(
        layer, store, _trigger(),
        activation_embedding=new_embedding,
        merge_threshold=0.5,
        embedding_running_average_decay=0.9,
    )
    blended = store.get_by_id("seed").embedding
    # Expected: 0.9 * [1,0,0,0] + 0.1 * [1,0.01,0,0] = [1, 0.001, 0, 0].
    assert abs(blended[0] - 1.0) < 1e-5
    assert abs(blended[1] - 0.001) < 1e-5


def test_outcome_dataclass_fired_property() -> None:
    not_fired = ConsolidationOutcome(entry_id=None, merged_into=None, was_merged=False)
    new_entry = ConsolidationOutcome(entry_id="a", merged_into=None, was_merged=False)
    merged = ConsolidationOutcome(entry_id=None, merged_into="b", was_merged=True)
    assert not not_fired.fired
    assert new_entry.fired
    assert merged.fired


# ---- Task-aware variant: task_id tagging on consolidation ----


def test_task_id_default_is_minus_one_in_metadata() -> None:
    layer = _populated_layer()
    store = ColdStorage(collection_name="pipe_task_default")
    outcome = consolidate_to_storage(
        layer, store, _trigger(), activation_embedding=torch.zeros(4)
    )
    entry = store.get_by_id(outcome.entry_id)
    assert entry.metadata["task_id"] == -1


def test_task_id_stored_in_metadata_when_supplied() -> None:
    layer = _populated_layer()
    store = ColdStorage(collection_name="pipe_task_tagged")
    outcome = consolidate_to_storage(
        layer, store, _trigger(),
        activation_embedding=torch.zeros(4),
        task_id=7,
    )
    entry = store.get_by_id(outcome.entry_id)
    assert entry.metadata["task_id"] == 7


# ---- Path-A: true_label / label_histogram metadata ----


def test_consolidate_to_storage_writes_true_label_when_provided() -> None:
    layer = _populated_layer()
    store = ColdStorage(collection_name="pipe_true_label_yes")
    outcome = consolidate_to_storage(
        layer, store, _trigger(),
        activation_embedding=torch.zeros(4),
        true_label=7,
    )
    entry = store.get_by_id(outcome.entry_id)
    assert entry.metadata["true_label"] == 7


def test_consolidate_to_storage_omits_true_label_when_none() -> None:
    layer = _populated_layer()
    store = ColdStorage(collection_name="pipe_true_label_no")
    outcome = consolidate_to_storage(
        layer, store, _trigger(),
        activation_embedding=torch.zeros(4),
    )
    entry = store.get_by_id(outcome.entry_id)
    assert "true_label" not in entry.metadata


def test_consolidate_to_storage_writes_label_histogram_when_provided() -> None:
    layer = _populated_layer()
    store = ColdStorage(collection_name="pipe_hist_yes")
    # Permuted-MNIST-like: 10 classes, batch size 16.
    histogram = [0, 2, 0, 5, 0, 3, 1, 0, 4, 1]
    assert sum(histogram) == 16
    outcome = consolidate_to_storage(
        layer, store, _trigger(),
        activation_embedding=torch.zeros(4),
        label_histogram=histogram,
    )
    entry = store.get_by_id(outcome.entry_id)
    decoded = json.loads(entry.metadata["label_histogram_json"])
    assert decoded == histogram
    assert len(decoded) == 10
    assert sum(decoded) == 16


def test_label_histogram_omitted_when_target_none() -> None:
    layer = _populated_layer()
    store = ColdStorage(collection_name="pipe_hist_no")
    outcome = consolidate_to_storage(
        layer, store, _trigger(),
        activation_embedding=torch.zeros(4),
    )
    entry = store.get_by_id(outcome.entry_id)
    assert "label_histogram_json" not in entry.metadata
