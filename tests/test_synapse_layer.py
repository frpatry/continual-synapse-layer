"""Tests for SynapseLayer v1."""

from __future__ import annotations

import pytest
import torch

from continual_synapse.synapse_layer.layer import SynapseLayer


def test_strengths_initialised_to_zero() -> None:
    layer = SynapseLayer(n_neurons=5)
    assert layer.strengths.shape == (5, 5)
    assert torch.all(layer.strengths == 0.0)
    assert layer.global_step.item() == 0


def test_consolidate_applies_hebbian_outer_product() -> None:
    """Single-batch update equals η · (aᵀa) / B exactly."""
    layer = SynapseLayer(n_neurons=3, learning_rate=0.1)
    a = torch.tensor([[1.0, 2.0, -1.0]])
    layer.consolidate(a)
    expected = 0.1 * (a.transpose(-1, -2) @ a) / 1.0
    torch.testing.assert_close(layer.strengths, expected)
    assert layer.global_step.item() == 1


def test_consolidate_averages_over_batch() -> None:
    """Update with B samples equals the mean of B single-sample updates."""
    lr = 0.05
    layer_batch = SynapseLayer(n_neurons=2, learning_rate=lr)
    layer_loop = SynapseLayer(n_neurons=2, learning_rate=lr)
    a = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])

    layer_batch.consolidate(a)

    # Equivalent: B independent batch-1 updates with lr scaled by 1/B,
    # then summed. Or directly compute the mean outer product:
    expected = lr * (a.transpose(-1, -2) @ a) / a.shape[0]
    torch.testing.assert_close(layer_batch.strengths, expected)

    # Cross-check against a loop using batch=1 with manually averaged lr.
    for row in a:
        layer_loop.consolidate(row.unsqueeze(0), reward=1.0 / a.shape[0])
    torch.testing.assert_close(layer_loop.strengths, expected)


def test_reward_scales_update_linearly() -> None:
    layer_a = SynapseLayer(n_neurons=2, learning_rate=0.1)
    layer_b = SynapseLayer(n_neurons=2, learning_rate=0.1)
    activations = torch.tensor([[1.0, 1.0]])
    layer_a.consolidate(activations, reward=1.0)
    layer_b.consolidate(activations, reward=3.0)
    torch.testing.assert_close(layer_b.strengths, 3.0 * layer_a.strengths)


def test_zero_activations_produce_zero_update() -> None:
    layer = SynapseLayer(n_neurons=4, learning_rate=1.0)
    layer.consolidate(torch.zeros(8, 4))
    assert torch.all(layer.strengths == 0.0)
    # Global step still advances — we did do an update, it just had no effect.
    assert layer.global_step.item() == 1


def test_empty_batch_is_a_noop() -> None:
    layer = SynapseLayer(n_neurons=3)
    layer.consolidate(torch.zeros(0, 3))
    assert layer.global_step.item() == 0


def test_consolidate_rejects_wrong_dim() -> None:
    layer = SynapseLayer(n_neurons=3)
    with pytest.raises(ValueError, match="n_neurons=3"):
        layer.consolidate(torch.zeros(2, 5))


def test_consolidate_rejects_non_2d() -> None:
    layer = SynapseLayer(n_neurons=3)
    with pytest.raises(ValueError, match="2-D"):
        layer.consolidate(torch.zeros(3))


def test_constructor_rejects_bad_args() -> None:
    with pytest.raises(ValueError, match="n_neurons"):
        SynapseLayer(n_neurons=0)
    with pytest.raises(ValueError, match="learning_rate"):
        SynapseLayer(n_neurons=3, learning_rate=0.0)


def test_consolidate_does_not_track_gradients() -> None:
    """Strengths must never accumulate autograd history."""
    layer = SynapseLayer(n_neurons=2, learning_rate=0.1)
    a = torch.randn(4, 2, requires_grad=True)
    layer.consolidate(a)
    assert not layer.strengths.requires_grad
    assert layer.strengths.grad_fn is None


def test_strengths_remain_finite_under_repeated_updates() -> None:
    """Numerical stability check with a small learning rate.

    100 updates with normalised activations should keep strengths
    within roughly the order of magnitude predicted by the
    Hebbian sum, with no NaNs or infinities.
    """
    layer = SynapseLayer(n_neurons=8, learning_rate=1e-3)
    g = torch.Generator().manual_seed(0)
    for _ in range(100):
        a = torch.randn(16, 8, generator=g)
        layer.consolidate(a)
    s = layer.strengths
    assert torch.isfinite(s).all()
    # Per-update outer-product norm is roughly E[||aaᵀ||] ~ n; with
    # lr=1e-3 and 100 updates, strengths magnitude stays modest.
    assert s.abs().max().item() < 5.0
    assert layer.global_step.item() == 100


def test_reset_zeroes_state() -> None:
    layer = SynapseLayer(n_neurons=2, learning_rate=0.5)
    layer.consolidate(torch.tensor([[1.0, 1.0]]))
    assert layer.global_step.item() == 1
    assert torch.any(layer.strengths != 0.0)
    layer.reset()
    assert layer.global_step.item() == 0
    assert torch.all(layer.strengths == 0.0)


def test_state_dict_roundtrip() -> None:
    layer = SynapseLayer(n_neurons=4, learning_rate=0.05)
    layer.consolidate(torch.randn(8, 4, generator=torch.Generator().manual_seed(1)))

    other = SynapseLayer(n_neurons=4, learning_rate=0.05)
    other.load_state_dict(layer.state_dict())
    torch.testing.assert_close(other.strengths, layer.strengths)
    assert other.global_step.item() == layer.global_step.item()


def test_to_device_moves_buffers() -> None:
    layer = SynapseLayer(n_neurons=3)
    moved = layer.to(torch.device("cpu"))  # CPU is the available device in CI
    assert moved.strengths.device.type == "cpu"
    assert moved.global_step.device.type == "cpu"
