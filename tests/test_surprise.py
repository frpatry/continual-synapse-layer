"""Tests for the SurpriseReward component."""

from __future__ import annotations

import pytest
import torch

from continual_synapse.reward.surprise import SurpriseReward


def test_first_call_returns_zero() -> None:
    sr = SurpriseReward(n_neurons=4)
    r = sr(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    assert r == 0.0
    assert sr.has_history


def test_second_call_with_zero_predictor_is_max_surprise() -> None:
    """The predictor starts zero-initialised, so the first real prediction
    is the zero vector; cosine distance against a non-zero target is
    undefined and the implementation treats that as max surprise."""
    sr = SurpriseReward(n_neurons=3)
    sr(torch.tensor([[1.0, 0.0, 0.0]]))
    r = sr(torch.tensor([[0.0, 1.0, 0.0]]))
    assert r == 1.0


def test_surprise_decreases_with_constant_input() -> None:
    """Feed the same input repeatedly; the predictor should learn to
    match it and surprise should trend toward zero."""
    sr = SurpriseReward(n_neurons=2, predictor_lr=0.5)
    a = torch.tensor([[1.0, 0.5]])
    sr(a)  # call 1: returns 0, seeds
    surprises = [sr(a) for _ in range(50)]
    # Early-call surprise (right after the zero-init misprediction) is high;
    # late-call surprise should be much lower.
    assert surprises[0] >= surprises[-1]
    assert surprises[-1] < 0.1


def test_surprise_spikes_on_regime_change() -> None:
    """Train on one pattern, then switch — surprise should jump."""
    sr = SurpriseReward(n_neurons=2, predictor_lr=0.5)
    sr(torch.tensor([[1.0, 0.0]]))  # seed
    # Warm the predictor on the same pattern.
    for _ in range(40):
        sr(torch.tensor([[1.0, 0.0]]))
    settled = sr(torch.tensor([[1.0, 0.0]]))
    # Now switch to an orthogonal pattern; surprise should be much higher.
    spike = sr(torch.tensor([[0.0, 1.0]]))
    assert spike > settled
    assert spike > 0.5


def test_reset_drops_history_and_predictor() -> None:
    sr = SurpriseReward(n_neurons=3)
    sr(torch.randn(2, 3))
    sr(torch.randn(2, 3))
    sr.reset()
    assert not sr.has_history
    assert torch.all(sr.predictor.weight == 0.0)


def test_surprise_does_not_propagate_grad_to_inputs() -> None:
    sr = SurpriseReward(n_neurons=3)
    a = torch.randn(2, 3, requires_grad=True)
    sr(a)
    sr(a)
    # The predictor was updated, but the input tensor's grad must stay None.
    assert a.grad is None


def test_surprise_uses_batch_mean() -> None:
    """A batch with average (1, 0) should be equivalent to a single
    (1, 0) sample for the purpose of surprise."""
    sr1 = SurpriseReward(n_neurons=2, predictor_lr=0.1)
    sr2 = SurpriseReward(n_neurons=2, predictor_lr=0.1)
    a_batch = torch.tensor([[2.0, 0.0], [0.0, 0.0]])  # mean = (1, 0)
    a_single = torch.tensor([[1.0, 0.0]])
    # Same seed of internal state: identical updates -> identical surprise.
    s1 = (sr1(a_batch), sr1(a_batch))
    s2 = (sr2(a_single), sr2(a_single))
    assert s1 == s2


def test_rejects_bad_shape_and_args() -> None:
    sr = SurpriseReward(n_neurons=3)
    with pytest.raises(ValueError, match="2-D"):
        sr(torch.zeros(3))
    with pytest.raises(ValueError, match="does not match"):
        sr(torch.zeros(2, 4))
    with pytest.raises(ValueError):
        SurpriseReward(n_neurons=0)
    with pytest.raises(ValueError, match="predictor_lr"):
        SurpriseReward(n_neurons=3, predictor_lr=0.0)
