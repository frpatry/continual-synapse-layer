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
    assert moved.evidence.device.type == "cpu"


# ---- Phase 3: evidence buffer + resistance ----


def test_evidence_initialised_to_zero() -> None:
    layer = SynapseLayer(n_neurons=4)
    assert layer.evidence.shape == (4, 4)
    assert torch.all(layer.evidence == 0.0)


def test_evidence_accumulates_absolute_co_activations() -> None:
    """Evidence ← mean_b(|a_i| · |a_j|), regardless of sign."""
    layer = SynapseLayer(n_neurons=3, learning_rate=0.1)
    a = torch.tensor([[1.0, -2.0, 0.0]])
    layer.consolidate(a)
    expected = a.abs().transpose(-1, -2) @ a.abs() / a.shape[0]
    torch.testing.assert_close(layer.evidence, expected)


def test_evidence_grows_monotonically() -> None:
    layer = SynapseLayer(n_neurons=3, learning_rate=0.1)
    g = torch.Generator().manual_seed(0)
    layer.consolidate(torch.randn(8, 3, generator=g))
    snapshot = layer.evidence.clone()
    layer.consolidate(torch.randn(8, 3, generator=g))
    # Strict inequality everywhere both batches had non-zero activations,
    # weak inequality otherwise. Either way: never decreases.
    assert torch.all(layer.evidence >= snapshot)


def test_beta_zero_strength_update_matches_v1_exactly() -> None:
    """With β=0 the strength path is bit-identical to Phase 2 v1."""
    g = torch.Generator().manual_seed(1)
    a = torch.randn(16, 5, generator=g)

    layer = SynapseLayer(n_neurons=5, learning_rate=0.05, resistance_beta=0.0)
    layer.consolidate(a)

    # Hand-computed v1 update:
    expected = 0.05 * (a.transpose(-1, -2) @ a) / a.shape[0]
    torch.testing.assert_close(layer.strengths, expected)


def test_resistance_dampens_high_evidence_updates() -> None:
    """A synapse with high evidence gets a smaller update than a fresh one."""
    layer = SynapseLayer(n_neurons=2, learning_rate=1.0, resistance_beta=1.0)
    # Manually pre-load evidence to put one entry far above the other.
    with torch.no_grad():
        layer.evidence[0, 0] = 10.0  # 1/(1+1*10) = 1/11 resistance factor
        layer.evidence[1, 1] = 0.0   # full update on this position

    a = torch.tensor([[1.0, 1.0]])
    layer.consolidate(a, reward=1.0)

    # raw_outer is all-ones for this input; expected[0,0] = 1/11,
    # expected[1,1] = 1, expected[0,1] = 1/(1+5) = 1/6 (mean evidence ~5).
    s = layer.strengths
    assert s[1, 1].item() > s[0, 0].item()  # high-evidence resists more
    assert abs(s[0, 0].item() - 1.0 / 11.0) < 1e-5
    assert abs(s[1, 1].item() - 1.0) < 1e-5


def test_resistance_reduces_cumulative_drift() -> None:
    """Over many updates, β>0 must keep strength magnitudes smaller than β=0."""
    g = torch.Generator().manual_seed(2)
    activations = [torch.randn(16, 4, generator=g) for _ in range(50)]

    layer_off = SynapseLayer(
        n_neurons=4, learning_rate=0.1, resistance_beta=0.0
    )
    layer_on = SynapseLayer(
        n_neurons=4, learning_rate=0.1, resistance_beta=1.0
    )
    for a in activations:
        layer_off.consolidate(a)
        layer_on.consolidate(a)

    # Both saw identical inputs; resistance must produce smaller strengths.
    assert layer_on.strengths.abs().max() < layer_off.strengths.abs().max()
    # Evidence accumulation is identical for both (resistance does not
    # affect the evidence path).
    torch.testing.assert_close(layer_on.evidence, layer_off.evidence)


def test_constructor_rejects_negative_beta() -> None:
    with pytest.raises(ValueError, match="resistance_beta"):
        SynapseLayer(n_neurons=3, resistance_beta=-0.1)


def test_reset_clears_evidence_too() -> None:
    layer = SynapseLayer(n_neurons=3, resistance_beta=0.5)
    layer.consolidate(torch.tensor([[1.0, 1.0, 1.0]]))
    assert torch.any(layer.evidence != 0.0)
    layer.reset()
    assert torch.all(layer.evidence == 0.0)
