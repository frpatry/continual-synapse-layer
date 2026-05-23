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
