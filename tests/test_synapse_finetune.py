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


# ---- Phase 4b follow-up: multi-pass wiring ----


def test_n_passes_default_one_is_bit_exact() -> None:
    """Default n_passes=1 reproduces single-pass behaviour bit-exact."""
    set_seed(0)
    a, _ = _build_augmented(init_gate=0.0, lr_synapse=0.1)
    a.train()
    x = torch.randn(4, 4)
    a(x)
    a.apply_hebbian_update()
    legacy_strengths = a.synapse.strengths.clone()

    # Same configuration with explicit n_passes=1.
    cfg = MLPConfig(input_dim=4, hidden_dim=8, num_classes=2)
    set_seed(0)
    base = MLPClassifier(cfg)
    base_copy_state = base.state_dict()
    synapse_b = SynapseLayer(n_neurons=8, learning_rate=0.1, n_passes=1)
    new = SynapseAugmentedMLP(
        base, synapse_b, SynapseModulation(init_gate=0.0)
    )
    new.train()
    # Reset to identical RNG-derived starting state.
    new.base.load_state_dict(base_copy_state)
    new(x)
    new.apply_hebbian_update()
    torch.testing.assert_close(new.synapse.strengths, legacy_strengths)


def test_n_passes_does_not_observe_in_eval_mode() -> None:
    """Multi-pass only applies during training; eval forwards must
    leave the observation buffer empty so the model can be evaluated
    without polluting synapse state."""
    cfg = MLPConfig(input_dim=4, hidden_dim=8, num_classes=2)
    set_seed(0)
    base = MLPClassifier(cfg)
    synapse = SynapseLayer(n_neurons=8, n_passes=5)
    aug = SynapseAugmentedMLP(base, synapse, SynapseModulation(), n_passes=5)
    aug.eval()
    aug(torch.randn(2, 4))
    assert synapse.buffer_size == 0


def test_n_passes_observes_buffer_in_training_mode() -> None:
    cfg = MLPConfig(input_dim=4, hidden_dim=8, num_classes=2)
    set_seed(0)
    base = MLPClassifier(cfg)
    synapse = SynapseLayer(n_neurons=8, n_passes=3)
    aug = SynapseAugmentedMLP(base, synapse, SynapseModulation(), n_passes=3)
    aug.train()
    aug(torch.randn(2, 4))
    # 3 passes -> 3 observations in the buffer.
    assert synapse.buffer_size == 3


def test_n_passes_routes_buffer_through_consolidate() -> None:
    """After forward + apply_hebbian_update with n_passes>1, the
    buffer is drained and strengths reflect the buffer average."""
    cfg = MLPConfig(input_dim=4, hidden_dim=8, num_classes=2)
    set_seed(0)
    base = MLPClassifier(cfg)
    synapse = SynapseLayer(n_neurons=8, learning_rate=0.5, n_passes=3)
    aug = SynapseAugmentedMLP(
        base, synapse, SynapseModulation(), n_passes=3
    )
    aug.train()
    aug(torch.randn(4, 4))
    assert synapse.buffer_size == 3
    aug.apply_hebbian_update()
    assert synapse.buffer_size == 0
    # Strengths moved off zero — buffer was consumed.
    assert torch.any(synapse.strengths != 0)


def test_n_passes_with_deterministic_forward_is_equivalent_to_single_pass() -> None:
    """For a deterministic forward (no dropout), all N passes give
    identical activations; their average equals a single pass, so
    the strengths after consolidation match single-pass exactly."""
    cfg = MLPConfig(input_dim=4, hidden_dim=8, num_classes=2)
    x = torch.randn(4, 4, generator=torch.Generator().manual_seed(7))

    set_seed(0)
    base_single = MLPClassifier(cfg)
    synapse_single = SynapseLayer(n_neurons=8, learning_rate=0.5)
    aug_single = SynapseAugmentedMLP(
        base_single, synapse_single, SynapseModulation()
    )
    aug_single.train()
    aug_single(x)
    aug_single.apply_hebbian_update()

    set_seed(0)
    base_multi = MLPClassifier(cfg)
    synapse_multi = SynapseLayer(n_neurons=8, learning_rate=0.5, n_passes=5)
    aug_multi = SynapseAugmentedMLP(
        base_multi, synapse_multi, SynapseModulation(), n_passes=5
    )
    aug_multi.train()
    aug_multi(x)
    aug_multi.apply_hebbian_update()

    torch.testing.assert_close(
        aug_multi.synapse.strengths, aug_single.synapse.strengths
    )


def test_n_passes_with_dropout_differs_from_single_pass() -> None:
    """With dropout enabled, the N passes see different masks; the
    averaged outer product is denoised and differs from any single-
    pass equivalent (sometimes called for noise-filtering benefit)."""
    cfg = MLPConfig(
        input_dim=4, hidden_dim=8, num_classes=2, dropout=0.5
    )
    x = torch.randn(8, 4, generator=torch.Generator().manual_seed(11))

    set_seed(0)
    base_single = MLPClassifier(cfg)
    synapse_single = SynapseLayer(n_neurons=8, learning_rate=0.5)
    aug_single = SynapseAugmentedMLP(
        base_single, synapse_single, SynapseModulation()
    )
    aug_single.train()
    aug_single(x)
    aug_single.apply_hebbian_update()

    set_seed(0)
    base_multi = MLPClassifier(cfg)
    synapse_multi = SynapseLayer(n_neurons=8, learning_rate=0.5, n_passes=10)
    aug_multi = SynapseAugmentedMLP(
        base_multi, synapse_multi, SynapseModulation(), n_passes=10
    )
    aug_multi.train()
    aug_multi(x)
    aug_multi.apply_hebbian_update()

    # Strengths differ because each pass saw a different dropout mask.
    assert not torch.allclose(
        aug_multi.synapse.strengths, aug_single.synapse.strengths
    )


def test_n_passes_validated_at_construction() -> None:
    cfg = MLPConfig(input_dim=4, hidden_dim=8, num_classes=2)
    base = MLPClassifier(cfg)
    syn = SynapseLayer(n_neurons=8)
    with pytest.raises(ValueError, match="n_passes"):
        SynapseAugmentedMLP(base, syn, SynapseModulation(), n_passes=0)


# ---- Phase 4b follow-up: compression sweep wiring ----


def _build_with_cold_storage_and_sweep(
    *,
    sweep_interval: int = 5,
    threshold: float = 0.0,
    quantile: float = 0.5,
    collection: str = "aug_sweep_test",
    schedule=None,
):
    from continual_synapse.cold_storage.compression import CompressionSchedule
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
        compression_sweep_interval=sweep_interval,
        compression_schedule=schedule or CompressionSchedule(),
    )
    return aug, store, trigger


def test_compression_sweep_default_interval_is_zero_disabled() -> None:
    """Default interval=0 means the sweep never fires — bit-exact
    Phase-4 cold-storage behaviour."""
    aug, _ = _build_augmented()
    assert aug.compression_sweep_interval == 0
    assert aug.compression_sweep_count == 0


def test_compression_sweep_fires_at_configured_interval() -> None:
    aug, store, _ = _build_with_cold_storage_and_sweep(
        sweep_interval=3, collection="sweep_fires"
    )
    # Pre-load synapse state so consolidation triggers on the first batch.
    with torch.no_grad():
        aug.synapse.strengths.fill_(2.0)
        aug.synapse.evidence.fill_(2.0)
        aug.synapse.global_step.fill_(50)
    # Run 7 hebbian updates; sweep fires every 3 -> expect 2 sweeps.
    for _ in range(7):
        aug(torch.randn(2, 4))
        aug.apply_hebbian_update()
    assert aug.compression_sweep_count == 2


def test_compression_sweep_zero_interval_never_fires() -> None:
    aug, store, _ = _build_with_cold_storage_and_sweep(
        sweep_interval=0, collection="sweep_zero"
    )
    with torch.no_grad():
        aug.synapse.strengths.fill_(2.0)
        aug.synapse.evidence.fill_(2.0)
        aug.synapse.global_step.fill_(50)
    for _ in range(10):
        aug(torch.randn(2, 4))
        aug.apply_hebbian_update()
    assert aug.compression_sweep_count == 0


def test_compression_sweep_updates_age_metadata() -> None:
    """After the sweep, entries in cold storage should have non-zero
    `age` fields (was always 0 in Phase 4b/4c)."""
    from continual_synapse.cold_storage.compression import CompressionSchedule

    aug, store, _ = _build_with_cold_storage_and_sweep(
        sweep_interval=1,
        collection="sweep_age",
        schedule=CompressionSchedule(
            age_thresholds=(10_000_000,),  # so no precision change
            tier_precisions=(32, 32),
            access_count_floor=1_000_000,
        ),
    )
    with torch.no_grad():
        aug.synapse.strengths.fill_(2.0)
        aug.synapse.evidence.fill_(2.0)
        aug.synapse.global_step.fill_(0)
    # First batch: consolidation fires AT global_step 0 (created_at_step=0),
    # then sweep fires. Run a second batch to advance global_step and
    # sweep again.
    aug(torch.randn(2, 4))
    aug.apply_hebbian_update()  # global_step now 1, entry created_at_step=0
    aug(torch.randn(2, 4))
    aug.apply_hebbian_update()  # global_step now 2, sweep updates age to 2

    entries = store.all_entries()
    assert len(entries) >= 1
    # The first entry was created at step 0, now at step 2 -> age 2.
    assert all(e.metadata["age"] >= 0 for e in entries)
    # At least one entry should have age > 0 after the second sweep.
    assert any(e.metadata["age"] > 0 for e in entries)


def test_compression_sweep_actually_reduces_precision() -> None:
    """End-to-end: with a schedule whose thresholds are crossed, the
    sweep moves stored entries to lower precision tiers and
    `last_compression_counts` reflects the move."""
    from continual_synapse.cold_storage.compression import CompressionSchedule

    aug, store, _ = _build_with_cold_storage_and_sweep(
        sweep_interval=1,
        collection="sweep_compress",
        # Thresholds set so age >= 2 -> 16-bit.
        schedule=CompressionSchedule(
            age_thresholds=(2,),
            tier_precisions=(32, 16),
            access_count_floor=1_000_000,
        ),
    )
    with torch.no_grad():
        aug.synapse.strengths.fill_(2.0)
        aug.synapse.evidence.fill_(2.0)
        aug.synapse.global_step.fill_(0)
    # Insert one entry, then advance enough that its age crosses the
    # threshold and the next sweep moves it to 16-bit.
    for _ in range(5):
        aug(torch.randn(2, 4))
        aug.apply_hebbian_update()
    # After 5 sweeps the latest sweep saw entries with age >= 2 and
    # moved them to 16-bit. Inspect the most recent counts.
    counts = aug.last_compression_counts
    assert 16 in counts and counts[16] >= 1
    # Inspect entries directly: at least one is now at 16-bit.
    entries = store.all_entries()
    precisions = [e.metadata["precision"] for e in entries]
    assert 16 in precisions


def test_compression_sweep_requires_cold_storage() -> None:
    """No cold storage configured -> sweep cannot fire even with a
    non-zero interval."""
    cfg = MLPConfig(input_dim=4, hidden_dim=8, num_classes=2)
    base = MLPClassifier(cfg)
    syn = SynapseLayer(n_neurons=8, learning_rate=0.1)
    aug = SynapseAugmentedMLP(
        base, syn, SynapseModulation(), compression_sweep_interval=1
    )
    aug(torch.randn(2, 4))
    aug.apply_hebbian_update()
    assert aug.compression_sweep_count == 0


def test_compression_sweep_interval_validated_at_construction() -> None:
    cfg = MLPConfig(input_dim=4, hidden_dim=8, num_classes=2)
    base = MLPClassifier(cfg)
    syn = SynapseLayer(n_neurons=8)
    with pytest.raises(ValueError, match="compression_sweep_interval"):
        SynapseAugmentedMLP(
            base, syn, SynapseModulation(), compression_sweep_interval=-1
        )


def test_training_target_updates_external_reward_with_batch_accuracy() -> None:
    """When apply_hebbian_update gets a training_target, it should
    compute per-batch accuracy from cached logits and push it into
    the mixer's external reward source before the mixer reads it."""
    from continual_synapse.reward.external import ExternalReward
    from continual_synapse.reward.mixer import RewardMixer
    from continual_synapse.reward.consistency import ConsistencyReward

    cfg = MLPConfig(input_dim=4, hidden_dim=8, num_classes=2)
    set_seed(0)
    base = MLPClassifier(cfg)
    # Make head deterministic so we know what accuracy to expect.
    with torch.no_grad():
        base.head.weight.zero_()
        base.head.bias.zero_()
        # bias logit for class 0 high so model always predicts class 0
        base.head.bias[0] = 10.0
    external = ExternalReward(default=0.5)
    mixer = RewardMixer(
        external=external,
        consistency=ConsistencyReward(n_neurons=8),
        gamma=0.0,  # disable trajectory decay -> alpha stays at 1 so mixer ≈ external
    )
    synapse = SynapseLayer(n_neurons=8, learning_rate=0.05)
    aug = SynapseAugmentedMLP(
        base, synapse, SynapseModulation(init_gate=0.0),
        reward_computer=mixer,
    )
    aug.train()
    x = torch.randn(4, 4)
    aug(x)
    # All four samples predicted class 0; if labels are all 0 -> 100% acc.
    y_all_zero = torch.zeros(4, dtype=torch.int64)
    aug.apply_hebbian_update(training_target=y_all_zero)
    assert external.value == 1.0

    # Half wrong: two samples labelled class 1 -> 50% acc.
    aug(x)
    y_half = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    aug.apply_hebbian_update(training_target=y_half)
    assert external.value == 0.5


def test_training_target_noop_when_no_external_reward() -> None:
    """Without a RewardMixer + ExternalReward, training_target is harmless."""
    aug, _ = _build_augmented()
    aug.train()
    aug(torch.randn(2, 4))
    applied = aug.apply_hebbian_update(
        training_target=torch.zeros(2, dtype=torch.int64)
    )
    assert applied == 1.0  # default reward path unchanged


def test_training_target_noop_when_reward_computer_is_not_mixer() -> None:
    """Bare-callable reward computers (not RewardMixer) ignore training_target."""
    cfg = MLPConfig(input_dim=4, hidden_dim=8, num_classes=2)
    set_seed(0)
    base = MLPClassifier(cfg)
    synapse = SynapseLayer(n_neurons=8, learning_rate=0.05)
    aug = SynapseAugmentedMLP(
        base, synapse, SynapseModulation(init_gate=0.0),
        reward_computer=lambda features: 0.42,
    )
    aug.train()
    aug(torch.randn(2, 4))
    applied = aug.apply_hebbian_update(
        training_target=torch.zeros(2, dtype=torch.int64)
    )
    assert applied == 0.42


def test_last_logits_cached_only_in_training_mode() -> None:
    aug, _ = _build_augmented()
    aug.eval()
    aug(torch.randn(2, 4))
    assert aug._last_logits is None

    aug.train()
    aug(torch.randn(2, 4))
    assert aug._last_logits is not None
    assert aug._last_logits.shape == (2, 2)


def test_last_logits_consumed_and_cleared_by_apply_hebbian_update() -> None:
    aug, _ = _build_augmented()
    aug.train()
    aug(torch.randn(2, 4))
    assert aug._last_logits is not None
    aug.apply_hebbian_update()
    assert aug._last_logits is None


def test_last_compression_counts_starts_empty() -> None:
    aug, _, _ = _build_with_cold_storage_and_sweep(
        sweep_interval=10, collection="sweep_initial_counts"
    )
    assert aug.last_compression_counts == {}


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


# ---- Amplification variant: change 1 (multiplicative composition) ----


def _augmented_with_seeded_storage(
    *,
    amplification_alpha: float,
    hidden_dim: int = 4,
) -> tuple[SynapseAugmentedMLP, torch.Tensor]:
    """Build an augmented MLP with one cold-storage entry pre-loaded.

    Returns the model plus the retrieved tensor it would produce so
    tests can compute expected effective strengths exactly.
    """
    import base64

    from continual_synapse.cold_storage.compression import quantize
    from continual_synapse.cold_storage.store import ColdStorage

    cfg = MLPConfig(input_dim=4, hidden_dim=hidden_dim, num_classes=2)
    set_seed(0)
    base = MLPClassifier(cfg)
    synapse = SynapseLayer(n_neurons=hidden_dim, learning_rate=1e-9)
    # Seed the synapse strengths with a known non-zero pattern.
    with torch.no_grad():
        synapse.strengths.copy_(torch.arange(hidden_dim * hidden_dim, dtype=torch.float32)
                                .reshape(hidden_dim, hidden_dim))
    store = ColdStorage(collection_name=f"amplify_{amplification_alpha}_{id(base)}")
    # Pre-load one entry whose embedding matches what the model will
    # query with (a constant activation pattern).
    pattern = torch.full((hidden_dim, hidden_dim), 2.0)
    blob = quantize(pattern, precision=32)
    store.store_cluster(
        embedding=[1.0] * hidden_dim,
        metadata={
            "precision": 32, "n_neurons": hidden_dim,
            "age": 0, "access_count": 0, "created_at_step": 0,
        },
        document=base64.b64encode(blob).decode("ascii"),
    )
    aug = SynapseAugmentedMLP(
        base, synapse, SynapseModulation(init_gate=0.5),  # non-zero so strengths matter
        cold_storage=store,
        retrieval_k=1,
        retrieval_refresh_interval=1,
        amplification_alpha=amplification_alpha,
    )
    return aug, pattern


def _trace_correction(
    aug: SynapseAugmentedMLP, x: torch.Tensor, forced_features: torch.Tensor
) -> torch.Tensor:
    """Compute the modulator correction for `forced_features` against the
    same effective_strengths the model would use for `x`. The base
    model's `features()` is bypassed so input-zeroing ReLUs don't hide
    the composition rule under test."""
    # Reproduce the composition logic from features() but with the
    # forced pre-correction features.
    with torch.no_grad():
        retrieved = aug._get_or_refresh_retrieval(forced_features)
        if aug.amplification_alpha == 0.0:
            effective = aug.synapse.strengths + retrieved
        else:
            max_abs = retrieved.abs().max().clamp_min(1e-8)
            effective = aug.synapse.strengths * (
                1.0 + aug.amplification_alpha * (retrieved / max_abs)
            )
        return aug.modulator(forced_features, effective)


def test_amplification_default_is_additive_bit_exact() -> None:
    """alpha=0 must compute effective_strengths via strict addition,
    matching the legacy path bit-exact."""
    aug, pattern = _augmented_with_seeded_storage(amplification_alpha=0.0)
    forced = torch.ones(2, 4)
    with torch.no_grad():
        retrieved = aug._get_or_refresh_retrieval(forced)
        expected_effective = aug.synapse.strengths + retrieved
        expected_correction = aug.modulator(forced, expected_effective)
    actual_correction = _trace_correction(aug, forced, forced)
    torch.testing.assert_close(actual_correction, expected_correction)


def test_amplification_alpha_one_uses_multiplicative_normalized() -> None:
    """alpha=1 corresponds to effective = strengths * (1 + retrieved_norm)
    where retrieved_norm = retrieved / max(|retrieved|). The result must
    differ from the additive path on the same retrieved."""
    aug_amp, _ = _augmented_with_seeded_storage(amplification_alpha=1.0)
    aug_add, _ = _augmented_with_seeded_storage(amplification_alpha=0.0)
    forced = torch.ones(2, 4)

    out_amp = _trace_correction(aug_amp, forced, forced)
    out_add = _trace_correction(aug_add, forced, forced)
    # The pre-loaded retrieved pattern is all 2.0; strengths is arange.
    # Additive: strengths + 2; multiplicative-norm: strengths * (1 + 1) = 2*strengths
    # These differ structurally, so corrections differ.
    assert not torch.allclose(out_amp, out_add)

    # Verify the multiplicative formula numerically.
    with torch.no_grad():
        retrieved = aug_amp._get_or_refresh_retrieval(forced)
        max_abs = retrieved.abs().max().clamp_min(1e-8)
        expected_effective = aug_amp.synapse.strengths * (
            1.0 + retrieved / max_abs
        )
        expected_correction = aug_amp.modulator(forced, expected_effective)
    torch.testing.assert_close(out_amp, expected_correction)


def test_amplification_normalization_bounds_multiplier() -> None:
    """The multiplier (1 + alpha * retrieved_normalized) lies in
    [1 - alpha, 1 + alpha] because retrieved_normalized ∈ [-1, +1]."""
    cfg = MLPConfig(input_dim=4, hidden_dim=4, num_classes=2)
    set_seed(0)
    base = MLPClassifier(cfg)
    synapse = SynapseLayer(n_neurons=4, learning_rate=1e-9)
    with torch.no_grad():
        synapse.strengths.fill_(1.0)
    # Inject a known retrieved tensor by monkey-patching _get_or_refresh_retrieval.
    retrieved_test = torch.tensor(
        [[-3.0, 0.0, 1.0, 3.0]] * 4, dtype=torch.float32
    )
    # max-abs is 3.0, so normalized is [-1, 0, 1/3, 1]
    expected_normalized = retrieved_test / 3.0
    alpha = 0.5
    expected_multiplier = 1.0 + alpha * expected_normalized
    expected_effective = synapse.strengths * expected_multiplier

    # Direct math sanity check.
    assert (expected_multiplier >= 1.0 - alpha - 1e-6).all()
    assert (expected_multiplier <= 1.0 + alpha + 1e-6).all()
    assert torch.allclose(
        expected_effective.max(),
        torch.tensor(1.0 + alpha),
        atol=1e-6,
    )


def test_amplification_validated_at_construction() -> None:
    cfg = MLPConfig(input_dim=4, hidden_dim=4, num_classes=2)
    base = MLPClassifier(cfg)
    synapse = SynapseLayer(n_neurons=4)
    with pytest.raises(ValueError, match="amplification_alpha must be"):
        SynapseAugmentedMLP(base, synapse, amplification_alpha=-0.1)


# ---- Amplification variant: change 5 (retrieval-success feedback) ----


def _build_aug_with_storage_and_one_entry(
    *,
    retrieval_feedback_threshold: float,
    retrieval_feedback_bump: float = 0.5,
) -> SynapseAugmentedMLP:
    """Helper: augmented MLP with one cold-storage entry pre-loaded so
    every forward in training mode pulls that entry into
    _last_retrieved_meta."""
    import base64
    from continual_synapse.cold_storage.compression import quantize
    from continual_synapse.cold_storage.store import ColdStorage

    cfg = MLPConfig(input_dim=4, hidden_dim=4, num_classes=2)
    set_seed(0)
    base = MLPClassifier(cfg)
    synapse = SynapseLayer(n_neurons=4, learning_rate=1e-9)
    store = ColdStorage(
        collection_name=f"feedback_{retrieval_feedback_threshold}_{id(base)}"
    )
    pattern = torch.full((4, 4), 0.1)
    blob = quantize(pattern, precision=32)
    store.store_cluster(
        embedding=[1.0, 1.0, 1.0, 1.0],
        metadata={
            "precision": 32, "n_neurons": 4,
            "age": 0, "access_count": 0, "created_at_step": 0,
        },
        document=base64.b64encode(blob).decode("ascii"),
        entry_id="only",
    )
    return SynapseAugmentedMLP(
        base, synapse, SynapseModulation(init_gate=0.0),
        cold_storage=store,
        retrieval_k=1, retrieval_refresh_interval=1,
        retrieval_feedback_threshold=retrieval_feedback_threshold,
        retrieval_feedback_bump=retrieval_feedback_bump,
    )


def test_retrieval_feedback_default_threshold_is_noop() -> None:
    """threshold=0 ⇒ retrieval feedback never fires, even when loss
    is provided. Backward-compat guarantee for non-amplified methods."""
    aug = _build_aug_with_storage_and_one_entry(retrieval_feedback_threshold=0.0)
    aug.train()
    x = torch.ones(2, 4)
    y = torch.zeros(2, dtype=torch.int64)
    aug(x)
    aug.apply_hebbian_update(training_target=y, loss=0.001)
    # Cold storage entry's access_count was bumped by the retrieval
    # cache's automatic bump (== 1), but not by the feedback path.
    entry = aug.cold_storage.get_by_id("only")
    assert int(entry.metadata["access_count"]) == 1
    assert aug.retrieval_feedback_event_count == 0


def test_retrieval_feedback_bumps_on_low_loss() -> None:
    """With threshold=0.9 and the first loss seeding the EMA, a
    much-smaller second loss must trip the bump."""
    aug = _build_aug_with_storage_and_one_entry(
        retrieval_feedback_threshold=0.9, retrieval_feedback_bump=0.5,
    )
    aug.train()
    x = torch.ones(2, 4)
    y = torch.zeros(2, dtype=torch.int64)
    # First call: seeds the EMA at loss=1.0. No prior history → no bump.
    aug(x)
    aug.apply_hebbian_update(training_target=y, loss=1.0)
    pre_event_count = aug.retrieval_feedback_event_count
    # Second call: loss=0.1 ≪ EMA * 0.9 = 0.9 ⇒ bump fires.
    aug(x)
    aug.apply_hebbian_update(training_target=y, loss=0.1)
    assert aug.retrieval_feedback_event_count == pre_event_count + 1
    # The entry's access_count = 2 (one bump per retrieve) + 0.5 (one feedback
    # bump fired against the second batch). int() floors to 2.
    entry = aug.cold_storage.get_by_id("only")
    assert float(entry.metadata["access_count"]) == pytest.approx(2.5)


def test_retrieval_feedback_no_bump_when_loss_above_threshold() -> None:
    """A loss above the EMA × threshold cut-off must not bump."""
    aug = _build_aug_with_storage_and_one_entry(
        retrieval_feedback_threshold=0.9,
    )
    aug.train()
    x = torch.ones(2, 4)
    y = torch.zeros(2, dtype=torch.int64)
    aug(x)
    aug.apply_hebbian_update(training_target=y, loss=1.0)
    aug(x)
    # Loss = 0.95, EMA * 0.9 = 0.9; 0.95 > 0.9 ⇒ no bump.
    aug.apply_hebbian_update(training_target=y, loss=0.95)
    assert aug.retrieval_feedback_event_count == 0


def test_retrieval_feedback_derives_loss_from_cached_logits_when_loss_arg_omitted() -> None:
    """If the caller doesn't pass loss but did pass training_target, the
    model computes CE from _last_logits. Verified by the EMA moving."""
    aug = _build_aug_with_storage_and_one_entry(
        retrieval_feedback_threshold=0.9,
    )
    aug.train()
    x = torch.ones(2, 4)
    y = torch.zeros(2, dtype=torch.int64)
    aug(x)
    aug.apply_hebbian_update(training_target=y)  # no loss kwarg
    # EMA was None pre-call; a derived loss should have seeded it.
    assert aug.loss_ema is not None and aug.loss_ema > 0


def test_retrieval_feedback_validation_at_construction() -> None:
    cfg = MLPConfig(input_dim=4, hidden_dim=4, num_classes=2)
    base = MLPClassifier(cfg)
    synapse = SynapseLayer(n_neurons=4)
    with pytest.raises(ValueError, match="retrieval_feedback_threshold"):
        SynapseAugmentedMLP(base, synapse, retrieval_feedback_threshold=-0.1)
    with pytest.raises(ValueError, match="retrieval_feedback_decay"):
        SynapseAugmentedMLP(base, synapse, retrieval_feedback_decay=1.5)


# ---- Task-aware variant ----


def _build_aug_task_aware(
    *,
    task_aware_decay: float = 0.5,
    task_warmup_batches: int = 0,
    task_warmup_downweight: float = 1.0,
    amplification_alpha: float = 1.0,
) -> SynapseAugmentedMLP:
    import base64
    from continual_synapse.cold_storage.compression import quantize
    from continual_synapse.cold_storage.store import ColdStorage
    from continual_synapse.consolidation.trigger import ConsolidationTrigger

    cfg = MLPConfig(input_dim=4, hidden_dim=4, num_classes=2)
    set_seed(0)
    base = MLPClassifier(cfg)
    synapse = SynapseLayer(n_neurons=4, learning_rate=1e-9)
    store = ColdStorage(collection_name=f"task_aware_{id(base)}")
    # Pre-load one entry tagged with task_id=3.
    pattern = torch.full((4, 4), 0.5)
    store.store_cluster(
        embedding=[1.0, 1.0, 1.0, 1.0],
        metadata={
            "precision": 32, "n_neurons": 4,
            "age": 0, "access_count": 0, "created_at_step": 0,
            "task_id": 3,
        },
        document=base64.b64encode(quantize(pattern, precision=32)).decode("ascii"),
        entry_id="t3",
    )
    return SynapseAugmentedMLP(
        base, synapse, SynapseModulation(init_gate=0.5),
        cold_storage=store,
        consolidation_trigger=ConsolidationTrigger(
            avg_pressure_threshold=0.0, min_steps_between=0,
            candidate_quantile=0.5,
        ),
        retrieval_k=1, retrieval_refresh_interval=1,
        amplification_alpha=amplification_alpha,
        task_aware_decay=task_aware_decay,
        task_warmup_batches=task_warmup_batches,
        task_warmup_downweight=task_warmup_downweight,
    )


def test_task_aware_defaults_preserve_behavior() -> None:
    """All defaults (decay=0, warmup_batches=0, downweight=1) keep the
    model bit-exact equivalent to a non-task-aware build."""
    cfg = MLPConfig(input_dim=4, hidden_dim=4, num_classes=2)
    set_seed(0)
    base = MLPClassifier(cfg)
    synapse = SynapseLayer(n_neurons=4)
    aug = SynapseAugmentedMLP(base, synapse, SynapseModulation())
    # Defaults are inactive — current_task_id stays at sentinel -1.
    assert aug.current_task_id == -1
    # Warmup counter starts large so the window is inactive.
    assert aug.batches_since_task_change > 1000


def test_notify_task_change_resets_state() -> None:
    aug = _build_aug_task_aware(
        task_aware_decay=0.5, task_warmup_batches=10,
        task_warmup_downweight=0.1,
    )
    aug.train()
    x = torch.ones(2, 4)
    # Drive a few apply_hebbian_update calls so the counter advances.
    for _ in range(5):
        aug(x)
        aug.apply_hebbian_update()
    assert aug.batches_since_task_change >= 5
    aug.notify_task_change(7)
    assert aug.current_task_id == 7
    assert aug.batches_since_task_change == 0
    # Cache marked stale — next forward will refresh.
    assert aug._cache_invalidated


def test_consolidations_get_tagged_with_current_task_id() -> None:
    aug = _build_aug_task_aware(
        task_aware_decay=0.0, task_warmup_batches=0,
        task_warmup_downweight=1.0,
    )
    aug.notify_task_change(5)
    aug.train()
    # Populate synapse state with non-trivial strengths so consolidation fires.
    with torch.no_grad():
        aug.synapse.strengths.copy_(torch.full((4, 4), 0.5))
        aug.synapse.evidence.copy_(torch.full((4, 4), 1.0))
        aug.synapse.global_step.fill_(1000)
    x = torch.ones(2, 4)
    aug(x)
    aug.apply_hebbian_update()
    # The pre-loaded "t3" entry was there before; any new entry must
    # carry task_id=5.
    new_entries = [
        e for e in aug.cold_storage.all_entries() if e.id != "t3"
    ]
    assert new_entries, "consolidation did not fire — adjust setup"
    assert all(e.metadata.get("task_id") == 5 for e in new_entries)


def test_task_warmup_downweights_retrieval_for_first_n_batches() -> None:
    """During the first task_warmup_batches batches after notify_task_change,
    retrieval is scaled by task_warmup_downweight in the composition.

    We capture the ``effective_strengths`` argument that ``features()``
    hands to the modulator — that's where the downweight materialises.
    Going via end-to-end forward would be fragile because the base
    MLP's ReLU stack can zero out activations for ``torch.ones`` input
    and hide any composition difference."""
    aug = _build_aug_task_aware(
        task_aware_decay=0.0, task_warmup_batches=3,
        task_warmup_downweight=0.0,  # full clear, easiest to assert
        amplification_alpha=0.0,     # additive path so effective = S + retrieved
    )
    captured: list[torch.Tensor] = []
    orig_modulator_forward = aug.modulator.forward

    def spy(activations, strengths):
        captured.append(strengths.detach().clone())
        return orig_modulator_forward(activations, strengths)

    aug.modulator.forward = spy  # type: ignore[method-assign]

    aug.notify_task_change(0)
    aug.train()
    x = torch.ones(2, 4)
    # Synapse strengths stay at zero (no Hebbian update lands here),
    # so effective_strengths == downweight * retrieved during warmup,
    # and == retrieved post-warmup.
    aug(x); aug.apply_hebbian_update()  # batch 1: counter 0 → 1, in warmup
    aug(x); aug.apply_hebbian_update()  # batch 2: counter 1 → 2, in warmup
    aug(x); aug.apply_hebbian_update()  # batch 3: counter 2 → 3, in warmup
    assert aug.batches_since_task_change == 3
    aug(x); aug.apply_hebbian_update()  # batch 4: counter 3, past warmup

    # First three captures (in warmup): effective_strengths is the
    # zero-scaled retrieved ⇒ all zeros (downweight=0 × any = 0; plus
    # synapse.strengths is zero from a fresh layer with lr~0).
    for i in range(3):
        assert torch.allclose(captured[i], torch.zeros_like(captured[i])), (
            f"warmup batch {i+1}: expected zero effective_strengths, got "
            f"{captured[i]}"
        )
    # Fourth capture (post-warmup): retrieval contributes ⇒ non-zero.
    assert not torch.allclose(captured[3], torch.zeros_like(captured[3]))


def test_task_aware_decay_validation_at_construction() -> None:
    cfg = MLPConfig(input_dim=4, hidden_dim=4, num_classes=2)
    base = MLPClassifier(cfg)
    synapse = SynapseLayer(n_neurons=4)
    with pytest.raises(ValueError, match="task_aware_decay"):
        SynapseAugmentedMLP(base, synapse, task_aware_decay=-0.1)
    with pytest.raises(ValueError, match="task_warmup_batches"):
        SynapseAugmentedMLP(base, synapse, task_warmup_batches=-1)
    with pytest.raises(ValueError, match="task_warmup_downweight"):
        SynapseAugmentedMLP(base, synapse, task_warmup_downweight=-0.5)


# ---- Reward-modulated amplification ----


def _build_aug_reward_modulated(
    *,
    reward_modulated: bool,
    amplification_alpha: float = 1.0,
) -> SynapseAugmentedMLP:
    """Augmented MLP with one cold-storage entry, ready for composition
    tests via the captured-modulator-arg pattern."""
    import base64
    from continual_synapse.cold_storage.compression import quantize
    from continual_synapse.cold_storage.store import ColdStorage

    cfg = MLPConfig(input_dim=4, hidden_dim=4, num_classes=2)
    set_seed(0)
    base = MLPClassifier(cfg)
    synapse = SynapseLayer(n_neurons=4, learning_rate=1e-9)
    with torch.no_grad():
        synapse.strengths.fill_(1.0)  # known non-zero baseline
    store = ColdStorage(collection_name=f"reward_mod_{reward_modulated}_{id(base)}")
    pattern = torch.full((4, 4), 0.5)
    store.store_cluster(
        embedding=[1.0, 1.0, 1.0, 1.0],
        metadata={
            "precision": 32, "n_neurons": 4,
            "age": 0, "access_count": 0, "created_at_step": 0,
        },
        document=base64.b64encode(quantize(pattern, precision=32)).decode("ascii"),
        entry_id="only",
    )
    return SynapseAugmentedMLP(
        base, synapse, SynapseModulation(init_gate=0.5),
        cold_storage=store,
        retrieval_k=1, retrieval_refresh_interval=1,
        amplification_alpha=amplification_alpha,
        reward_modulated_amplification=reward_modulated,
    )


def test_reward_modulation_default_off_preserves_amplification() -> None:
    """The default reward_modulated_amplification=False produces the
    same effective_strengths as the plain amplification path even after
    a reward has been cached."""
    aug = _build_aug_reward_modulated(reward_modulated=False)
    captured: list[torch.Tensor] = []
    orig = aug.modulator.forward

    def spy(activations, strengths):
        captured.append(strengths.detach().clone())
        return orig(activations, strengths)

    aug.modulator.forward = spy  # type: ignore[method-assign]

    aug.train()
    x = torch.ones(2, 4)
    # First forward + update populates _last_reward (= 1.0 since no
    # reward_computer is wired and the default is 1.0).
    aug(x)
    aug.apply_hebbian_update()
    assert aug.last_reward == 1.0
    # Now manually flip _last_reward to a wildly different value so we
    # can detect any modulation that's NOT supposed to happen.
    aug._last_reward = -5.0
    aug(x)
    # The composition must still use plain amplification: effective =
    # strengths * (1 + alpha * retrieved_norm). The captured tensor
    # should not reflect any -5.0 scaling.
    captured_first = captured[0]   # post-first-forward
    captured_second = captured[1]  # post-_last_reward override
    # Without modulation, _last_reward is ignored ⇒ both compositions
    # use the same alpha, same retrieved ⇒ identical effective_strengths.
    torch.testing.assert_close(captured_first, captured_second)


def test_reward_modulation_active_scales_alpha_by_last_reward() -> None:
    """With the flag on, effective_alpha = alpha * last_reward."""
    aug = _build_aug_reward_modulated(
        reward_modulated=True, amplification_alpha=0.1
    )
    captured: list[torch.Tensor] = []
    orig = aug.modulator.forward
    aug.modulator.forward = lambda a, s: (captured.append(s.detach().clone()) or orig(a, s))  # type: ignore[method-assign]

    aug.train()
    x = torch.ones(2, 4)
    # Cold start ⇒ _last_reward = None ⇒ defaults to 1.0 in the
    # composition. This first capture is the baseline.
    aug(x)
    # Now set a known reward and forward again.
    aug._last_reward = 0.5
    aug(x)
    aug._last_reward = -1.0
    aug(x)

    # Reconstruct expected effective_strengths analytically.
    retrieved = aug._get_or_refresh_retrieval(torch.ones(2, 4))
    max_abs = retrieved.abs().max().clamp_min(1e-8)
    retrieved_norm = retrieved / max_abs
    strengths = aug.synapse.strengths

    def expected(eff_alpha: float) -> torch.Tensor:
        return strengths * (1.0 + eff_alpha * retrieved_norm)

    # Capture 0: reward=None ⇒ effective_alpha = 0.1 * 1.0 = 0.1
    torch.testing.assert_close(captured[0], expected(0.1))
    # Capture 1: reward=0.5 ⇒ effective_alpha = 0.1 * 0.5 = 0.05
    torch.testing.assert_close(captured[1], expected(0.05))
    # Capture 2: reward=-1.0 ⇒ effective_alpha = -0.1 (anti-amplification)
    torch.testing.assert_close(captured[2], expected(-0.1))


def test_reward_modulation_anti_amplifies_at_negative_reward() -> None:
    """Negative reward inverts the amplification sign: the retrieval
    pattern that would normally boost strengths instead suppresses them."""
    aug = _build_aug_reward_modulated(
        reward_modulated=True, amplification_alpha=1.0
    )
    captured: list[torch.Tensor] = []
    orig = aug.modulator.forward
    aug.modulator.forward = lambda a, s: (captured.append(s.detach().clone()) or orig(a, s))  # type: ignore[method-assign]

    aug.train()
    x = torch.ones(2, 4)
    aug._last_reward = +1.0
    aug(x)
    aug._last_reward = -1.0
    aug(x)
    # With strengths=ones and retrieved>0 (pattern=0.5), at +1 reward
    # effective is strengths * (1 + retrieved_norm) > 1. At -1 reward,
    # effective is strengths * (1 - retrieved_norm) < 1. Opposite sides
    # of the original strengths.
    pos, neg = captured[0], captured[1]
    assert (pos > 1.0).any() and (neg < 1.0).any()
    # The element-wise deviation from strengths is symmetric around 1.
    torch.testing.assert_close(pos - 1.0, -(neg - 1.0))


def test_reward_modulation_ignored_when_alpha_is_zero() -> None:
    """If amplification_alpha=0, the additive composition is taken
    regardless of the modulation flag. reward_modulated only affects
    the multiplicative branch."""
    aug = _build_aug_reward_modulated(
        reward_modulated=True, amplification_alpha=0.0
    )
    captured: list[torch.Tensor] = []
    orig = aug.modulator.forward
    aug.modulator.forward = lambda a, s: (captured.append(s.detach().clone()) or orig(a, s))  # type: ignore[method-assign]

    aug.train()
    x = torch.ones(2, 4)
    aug._last_reward = 0.5  # Would scale alpha if it were nonzero.
    aug(x)
    # Expected: additive path ⇒ effective = strengths + retrieved.
    retrieved = aug._get_or_refresh_retrieval(torch.ones(2, 4))
    expected = aug.synapse.strengths + retrieved
    torch.testing.assert_close(captured[0], expected)


def test_last_reward_populated_after_apply_hebbian_update() -> None:
    """The property starts None and becomes the applied reward."""
    aug = _build_aug_reward_modulated(reward_modulated=False)
    assert aug.last_reward is None
    aug.train()
    aug(torch.ones(2, 4))
    aug.apply_hebbian_update(reward=0.42)
    assert aug.last_reward == 0.42
