"""Tests for the synapse-augmented MLP and its runner integration."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import TensorDataset

from continual_synapse.baselines.naive_finetune import MLPClassifier, MLPConfig
from continual_synapse.baselines.synapse_finetune import SynapseAugmentedMLP
from continual_synapse.evaluation.benchmarks import Task
from continual_synapse.evaluation.runner import ContinualRunner, set_seed
from continual_synapse.synapse_layer.layer import SynapseLayer
from continual_synapse.synapse_layer.modulation import SynapseModulation


def _build_augmented(
    hidden_dim: int = 8,
    *,
    init_gate: float = 0.0,
    lr_synapse: float = 1e-3,
) -> tuple[SynapseAugmentedMLP, MLPClassifier]:
    cfg = MLPConfig(input_dim=4, hidden_dim=hidden_dim, num_classes=2)
    set_seed(0)
    base = MLPClassifier(cfg)
    base_copy = MLPClassifier(cfg)
    # Make `base_copy` byte-identical to `base` so output comparison is clean.
    base_copy.load_state_dict(base.state_dict())
    synapse = SynapseLayer(n_neurons=hidden_dim, learning_rate=lr_synapse)
    mod = SynapseModulation(init_gate=init_gate)
    return SynapseAugmentedMLP(base, synapse, mod), base_copy


def test_augmented_matches_base_at_init() -> None:
    """Gate=0 and strengths=0 -> identical logits to the underlying MLP."""
    aug, base = _build_augmented(init_gate=0.0)
    x = torch.randn(5, 4)
    torch.testing.assert_close(aug(x), base(x))


def test_augmented_matches_base_after_hebbian_until_gate_moves() -> None:
    """Even with non-zero strengths, gate=0 keeps the correction zero."""
    aug, base = _build_augmented(init_gate=0.0)
    x = torch.randn(8, 4)
    # Forward + several Hebbian updates -> strengths are non-zero.
    aug(x)
    aug.apply_hebbian_update()
    aug(x)
    aug.apply_hebbian_update()
    assert torch.any(aug.synapse.strengths != 0.0)
    torch.testing.assert_close(aug(x), base(x))


def test_dim_mismatch_is_rejected() -> None:
    base = MLPClassifier(MLPConfig(input_dim=4, hidden_dim=8, num_classes=2))
    bad_synapse = SynapseLayer(n_neurons=16)
    with pytest.raises(ValueError, match="does not match"):
        SynapseAugmentedMLP(base, bad_synapse)


def test_apply_hebbian_update_requires_forward_first() -> None:
    aug, _ = _build_augmented()
    with pytest.raises(RuntimeError, match="forward pass"):
        aug.apply_hebbian_update()


def test_apply_hebbian_update_consumes_cache() -> None:
    """Two updates without an intervening forward must error on the second."""
    aug, _ = _build_augmented()
    aug(torch.randn(2, 4))
    aug.apply_hebbian_update()
    with pytest.raises(RuntimeError, match="forward pass"):
        aug.apply_hebbian_update()


def test_apply_hebbian_update_uses_pre_correction_features() -> None:
    """The features fed into the synapse must be the base output, not the
    base output plus the synapse correction. Otherwise the synapse would
    reinforce its own signal."""
    aug, _ = _build_augmented(init_gate=0.5)
    # Manually inflate strengths so the correction is non-trivial.
    with torch.no_grad():
        aug.synapse.strengths.fill_(0.5)
    x = torch.randn(3, 4)
    expected_features = aug.base.features(x).detach()
    aug(x)
    # The cached features captured during forward must equal the
    # pre-correction base features.
    torch.testing.assert_close(aug._last_features, expected_features)


def test_optimizer_does_not_touch_strengths() -> None:
    """The Hebbian buffer must remain outside the optimizer's reach."""
    aug, _ = _build_augmented()
    opt = torch.optim.SGD(aug.parameters(), lr=0.1)
    # Sanity: gate is a Parameter and should be in the optimizer's view.
    param_ids = {id(p) for group in opt.param_groups for p in group["params"]}
    assert id(aug.modulator.gate) in param_ids
    assert id(aug.synapse.strengths) not in param_ids


def _two_task_benchmark() -> "_Benchmark":
    g = torch.Generator().manual_seed(0)

    def make(name: str, offset: float) -> Task:
        x0 = torch.randn(32, 4, generator=g) - offset
        x1 = torch.randn(32, 4, generator=g) + offset
        x = torch.cat([x0, x1])
        y = torch.cat([torch.zeros(32, dtype=torch.int64), torch.ones(32, dtype=torch.int64)])
        idx = torch.randperm(x.shape[0], generator=g)
        x, y = x[idx], y[idx]
        return Task(
            name=name,
            train=TensorDataset(x[:48], y[:48]),
            test=TensorDataset(x[48:], y[48:]),
            classes=(0, 1),
        )

    class _Benchmark:
        name = "synapse_smoke"
        num_classes_per_task = 2
        input_shape = (4,)

        def tasks(self) -> list[Task]:
            return [make("a", 1.5), make("b", -1.5)]

    return _Benchmark()


def test_runner_after_batch_hook_fires_once_per_batch() -> None:
    """on_after_batch is invoked exactly once per optimizer step."""
    calls: list[tuple[int, str]] = []

    def cb(i, task, model, x, y):
        calls.append((i, task.name))

    runner = ContinualRunner(
        optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.05),
        epochs_per_task=2,
        batch_size=16,  # 48 train samples / 16 = 3 batches/epoch, 6 per task
        eval_batch_size=16,
        seed=0,
        on_after_batch=cb,
    )
    model = MLPClassifier(MLPConfig(input_dim=4, hidden_dim=8, num_classes=2))
    runner.run(model, _two_task_benchmark())

    # Two tasks * 6 batches each = 12 invocations.
    assert len(calls) == 12
    assert [i for i, _ in calls[:6]] == [0] * 6
    assert [i for i, _ in calls[6:]] == [1] * 6


# ---- Phase 3: reward computer wiring ----


def test_apply_hebbian_update_uses_fixed_reward_when_no_computer() -> None:
    aug, _ = _build_augmented()
    aug(torch.randn(2, 4))
    applied = aug.apply_hebbian_update()
    assert applied == 1.0


def test_apply_hebbian_update_uses_explicit_reward_when_provided() -> None:
    aug, _ = _build_augmented()
    aug(torch.randn(2, 4))
    applied = aug.apply_hebbian_update(reward=0.25)
    assert applied == 0.25


def test_apply_hebbian_update_uses_reward_computer_when_no_explicit_value() -> None:
    """The reward computer is called with the cached features."""
    received: list[torch.Tensor] = []

    def reward_fn(features: torch.Tensor) -> float:
        received.append(features.clone())
        return 0.3

    cfg = MLPConfig(input_dim=4, hidden_dim=8, num_classes=2)
    set_seed(0)
    base = MLPClassifier(cfg)
    synapse = SynapseLayer(n_neurons=8, learning_rate=0.1)
    aug = SynapseAugmentedMLP(
        base, synapse, SynapseModulation(), reward_computer=reward_fn
    )
    x = torch.randn(3, 4)
    aug(x)
    applied = aug.apply_hebbian_update()

    assert applied == 0.3
    assert len(received) == 1
    # The computer must see the same pre-correction features used by
    # `consolidate`, i.e. base.features(x).detach().
    torch.testing.assert_close(received[0], base.features(x).detach())


def test_set_active_head_forwards_to_base_when_supported() -> None:
    """SynapseAugmentedMLP wrapping a multi-head base delegates head selection."""
    from continual_synapse.baselines.multi_head import MultiHeadMLPClassifier

    cfg = MLPConfig(input_dim=4, hidden_dim=8, num_classes=2)
    base = MultiHeadMLPClassifier(num_tasks=3, config=cfg)
    synapse = SynapseLayer(n_neurons=8)
    aug = SynapseAugmentedMLP(base, synapse, SynapseModulation())
    aug.set_active_head(2)
    assert base.active_head == 2


def test_set_active_head_raises_on_single_head_base() -> None:
    aug, _ = _build_augmented()
    with pytest.raises(AttributeError, match="multi-head"):
        aug.set_active_head(0)


def test_explicit_reward_overrides_computer() -> None:
    """A caller-supplied reward bypasses the configured computer."""
    calls: list[None] = []

    def reward_fn(features: torch.Tensor) -> float:
        calls.append(None)
        return 0.9

    aug, _ = _build_augmented()
    aug.reward_computer = reward_fn
    aug(torch.randn(2, 4))
    applied = aug.apply_hebbian_update(reward=0.1)
    assert applied == 0.1
    assert calls == []  # computer not invoked


# ---- Phase 4: cold storage integration ----


def _build_with_cold_storage(
    *,
    threshold: float = 0.0,
    quantile: float = 0.25,
    collection: str = "aug_cold_test",
):
    from continual_synapse.cold_storage.store import ColdStorage
    from continual_synapse.consolidation.trigger import ConsolidationTrigger

    cfg = MLPConfig(input_dim=4, hidden_dim=8, num_classes=2)
    set_seed(0)
    base = MLPClassifier(cfg)
    synapse = SynapseLayer(n_neurons=8, learning_rate=0.1)
    store = ColdStorage(collection_name=collection)
    trigger = ConsolidationTrigger(
        avg_pressure_threshold=threshold,
        min_steps_between=0,
        candidate_quantile=quantile,
    )
    aug = SynapseAugmentedMLP(
        base,
        synapse,
        SynapseModulation(init_gate=0.5),
        cold_storage=store,
        consolidation_trigger=trigger,
        retrieval_k=2,
    )
    return aug, store, trigger


def test_cold_storage_off_preserves_existing_behaviour() -> None:
    """Without cold_storage configured, the model matches Phase-3 behaviour bit-for-bit."""
    set_seed(0)
    a, _ = _build_augmented(init_gate=0.5)
    x = torch.randn(3, 4)
    with torch.no_grad():
        a.synapse.strengths.fill_(0.1)
    out_no_cs = a(x).clone()

    set_seed(0)
    b, _ = _build_augmented(init_gate=0.5)
    with torch.no_grad():
        b.synapse.strengths.fill_(0.1)
    # b also has cold_storage=None by construction; output must match exactly.
    torch.testing.assert_close(b(x), out_no_cs)


def test_cold_storage_with_empty_store_does_not_affect_forward() -> None:
    """Empty cold store -> reconstruction returns zeros -> forward unchanged."""
    aug, store, _ = _build_with_cold_storage(collection="aug_cold_empty")
    assert store.count() == 0
    x = torch.randn(2, 4)
    with torch.no_grad():
        aug.synapse.strengths.fill_(0.05)
    out_with_empty = aug(x).clone()

    # Compare against a fresh aug with no cold storage at all.
    set_seed(0)
    base = MLPClassifier(MLPConfig(input_dim=4, hidden_dim=8, num_classes=2))
    syn = SynapseLayer(n_neurons=8, learning_rate=0.1)
    plain = SynapseAugmentedMLP(base, syn, SynapseModulation(init_gate=0.5))
    with torch.no_grad():
        plain.synapse.strengths.fill_(0.05)
    # Note: the two models have different base parameters (different
    # construction order means different RNG state). We can't compare
    # outputs directly; instead, check the cold-storage path is
    # functionally inert by comparing aug to itself with cold_storage=None.
    plain_cs = aug.cold_storage
    aug.cold_storage = None
    out_no_cs = aug(x).clone()
    aug.cold_storage = plain_cs

    torch.testing.assert_close(out_with_empty, out_no_cs)


def test_consolidation_fires_when_trigger_says_so() -> None:
    aug, store, _ = _build_with_cold_storage(
        threshold=0.0, collection="aug_cold_fire"
    )
    # Pre-populate the synapse with high-pressure state.
    with torch.no_grad():
        aug.synapse.strengths.fill_(2.0)
        aug.synapse.evidence.fill_(2.0)
        aug.synapse.global_step.fill_(50)

    aug(torch.randn(2, 4))
    aug.apply_hebbian_update()
    assert aug.consolidation_count == 1
    assert store.count() == 1


def test_no_consolidation_when_trigger_or_store_missing() -> None:
    """Consolidation requires both cold_storage and consolidation_trigger."""
    cfg = MLPConfig(input_dim=4, hidden_dim=8, num_classes=2)
    set_seed(0)
    base = MLPClassifier(cfg)
    synapse = SynapseLayer(n_neurons=8, learning_rate=0.1)
    # No trigger -> no consolidation even when store present.
    from continual_synapse.cold_storage.store import ColdStorage

    store = ColdStorage(collection_name="aug_no_trigger")
    aug = SynapseAugmentedMLP(
        base, synapse, SynapseModulation(init_gate=0.0), cold_storage=store
    )
    aug(torch.randn(2, 4))
    aug.apply_hebbian_update()
    assert aug.consolidation_count == 0
    assert store.count() == 0


def test_cold_storage_alters_forward_after_consolidation() -> None:
    """Once an entry lives in cold storage, the retrieved strengths reach
    the modulator and change the output for a similar query."""
    aug, store, _ = _build_with_cold_storage(
        threshold=0.0,
        quantile=1.0,  # archive everything for a definitive test
        collection="aug_cold_alters",
    )
    # Build a clear archived pattern: large strengths, all zero before.
    with torch.no_grad():
        aug.synapse.strengths.copy_(torch.full((8, 8), 0.5))
        aug.synapse.evidence.copy_(torch.full((8, 8), 1.0))
        aug.synapse.global_step.fill_(100)

    x = torch.randn(2, 4)
    aug(x)
    aug.apply_hebbian_update()
    assert store.count() == 1
    # After consolidation, synapse strengths were drained (the entire
    # tensor was a candidate at quantile=1.0).
    assert torch.all(aug.synapse.strengths == 0)

    # New forward: retrieval should bring something back through the
    # modulator. Compare against a fresh model with no cold storage:
    # same drained synapse strengths but the cold-storage retrieval is
    # active here.
    out_with_cs = aug(x).clone()
    aug.cold_storage = None
    out_no_cs = aug(x).clone()

    # The retrieval-augmented forward differs from the plain forward
    # because retrieved strengths get added before the modulator.
    assert not torch.allclose(out_with_cs, out_no_cs)


def test_retrieval_cache_reused_within_interval() -> None:
    """Retrieval caching avoids one Chroma query per forward when the
    cache hasn't been invalidated and the interval has not elapsed."""
    from unittest.mock import patch

    aug, store, _ = _build_with_cold_storage(
        threshold=0.0,
        quantile=1.0,
        collection="aug_cache_reuse",
    )
    aug.retrieval_refresh_interval = 4
    # Seed the store with one entry so retrieval has something to query.
    with torch.no_grad():
        aug.synapse.strengths.copy_(torch.full((8, 8), 0.5))
        aug.synapse.evidence.copy_(torch.full((8, 8), 1.0))
        aug.synapse.global_step.fill_(100)
    aug(torch.randn(2, 4))
    aug.apply_hebbian_update()
    assert store.count() == 1

    # Spy on reconstruct_strengths to count refreshes.
    import continual_synapse.baselines.synapse_finetune as mod_under_test

    refresh_count = [0]
    real_reconstruct = mod_under_test.reconstruct_strengths

    def counted(*a, **kw):
        refresh_count[0] += 1
        return real_reconstruct(*a, **kw)

    with patch.object(mod_under_test, "reconstruct_strengths", counted):
        # First forward after invalidation -> refresh.
        aug(torch.randn(2, 4))
        # Three more forwards inside the interval -> reuse cache.
        aug(torch.randn(2, 4))
        aug(torch.randn(2, 4))
        aug(torch.randn(2, 4))
        # Fifth forward crosses the interval -> refresh again.
        aug(torch.randn(2, 4))
    assert refresh_count[0] == 2  # exactly two real queries in 5 forwards


def test_consolidation_invalidates_retrieval_cache() -> None:
    """A fresh consolidation should make the very next forward refresh."""
    from unittest.mock import patch

    aug, store, _ = _build_with_cold_storage(
        threshold=0.0,
        quantile=1.0,
        collection="aug_cache_invalidate",
    )
    aug.retrieval_refresh_interval = 1000  # long enough that the interval
    # alone would never refresh during the test.

    # Seed the store with an initial entry.
    with torch.no_grad():
        aug.synapse.strengths.copy_(torch.full((8, 8), 0.5))
        aug.synapse.evidence.copy_(torch.full((8, 8), 1.0))
        aug.synapse.global_step.fill_(100)
    aug(torch.randn(2, 4))
    aug.apply_hebbian_update()
    assert store.count() == 1

    # Warm the cache.
    aug(torch.randn(2, 4))

    import continual_synapse.baselines.synapse_finetune as mod_under_test

    refresh_count = [0]
    real_reconstruct = mod_under_test.reconstruct_strengths

    def counted(*a, **kw):
        refresh_count[0] += 1
        return real_reconstruct(*a, **kw)

    with patch.object(mod_under_test, "reconstruct_strengths", counted):
        # Reset synapse so consolidate has something fresh to archive.
        with torch.no_grad():
            aug.synapse.strengths.copy_(torch.full((8, 8), 0.5))
            aug.synapse.evidence.copy_(torch.full((8, 8), 1.0))
        aug(torch.randn(2, 4))  # cache hit (no refresh)
        # Trigger another consolidation — this should invalidate the cache.
        aug.apply_hebbian_update()
        # Next forward: cache invalidated -> refresh.
        aug(torch.randn(2, 4))
    assert refresh_count[0] == 1


def test_retrieval_refresh_interval_validated() -> None:
    from continual_synapse.cold_storage.store import ColdStorage

    cfg = MLPConfig(input_dim=4, hidden_dim=8, num_classes=2)
    base = MLPClassifier(cfg)
    syn = SynapseLayer(n_neurons=8)
    with pytest.raises(ValueError, match="retrieval_refresh_interval"):
        SynapseAugmentedMLP(
            base,
            syn,
            SynapseModulation(),
            cold_storage=ColdStorage(collection_name="aug_refresh_validation"),
            retrieval_refresh_interval=0,
        )


def test_retrieval_k_validated() -> None:
    from continual_synapse.cold_storage.store import ColdStorage

    cfg = MLPConfig(input_dim=4, hidden_dim=8, num_classes=2)
    base = MLPClassifier(cfg)
    syn = SynapseLayer(n_neurons=8)
    with pytest.raises(ValueError, match="retrieval_k"):
        SynapseAugmentedMLP(
            base,
            syn,
            SynapseModulation(),
            cold_storage=ColdStorage(collection_name="aug_k_validation"),
            retrieval_k=0,
        )


def test_runner_end_to_end_with_synapse_smoke() -> None:
    """End-to-end: the synapse-augmented MLP trains without crashing and
    the strengths grow as Hebbian updates accumulate."""
    set_seed(0)
    base = MLPClassifier(MLPConfig(input_dim=4, hidden_dim=8, num_classes=2))
    synapse = SynapseLayer(n_neurons=8, learning_rate=1e-3)
    model = SynapseAugmentedMLP(base, synapse, SynapseModulation(init_gate=0.0))

    runner = ContinualRunner(
        optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.05),
        epochs_per_task=1,
        batch_size=16,
        eval_batch_size=16,
        seed=0,
        on_after_batch=lambda i, t, m, x, y: m.apply_hebbian_update(),
    )
    result = runner.run(model, _two_task_benchmark())
    # 2 tasks recorded.
    assert result.accuracy_matrix.shape == (2, 2)
    # Strengths moved off zero.
    assert torch.any(synapse.strengths != 0.0)
    # global_step counts batches: 2 tasks * 1 epoch * 3 batches = 6.
    assert synapse.global_step.item() == 6
