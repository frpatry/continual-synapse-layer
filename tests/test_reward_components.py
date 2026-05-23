"""Tests for the external + consistency reward components."""

from __future__ import annotations

import math

import pytest
import torch

from continual_synapse.reward.consistency import ConsistencyReward
from continual_synapse.reward.external import ExternalReward


# ---- ExternalReward ----


def test_external_reward_returns_default() -> None:
    er = ExternalReward()
    assert er() == 1.0
    assert er(torch.zeros(2, 3)) == 1.0


def test_external_reward_custom_default() -> None:
    er = ExternalReward(default=0.25)
    assert er() == 0.25


def test_external_reward_can_be_updated() -> None:
    er = ExternalReward()
    er.set(-0.5)
    assert er() == -0.5
    er.set(2.0)
    assert er.value == 2.0


# ---- ConsistencyReward ----


def test_consistency_first_call_returns_one() -> None:
    cr = ConsistencyReward(n_neurons=3)
    r = cr(torch.tensor([[1.0, 2.0, 3.0]]))
    assert r == 1.0
    # EMA was seeded with the first observation.
    assert cr.ema is not None
    torch.testing.assert_close(cr.ema, torch.tensor([1.0, 2.0, 3.0]))


def test_consistency_identical_inputs_yield_one() -> None:
    cr = ConsistencyReward(n_neurons=3)
    a = torch.tensor([[1.0, 1.0, 1.0]])
    cr(a)
    r = cr(a)
    assert math.isclose(r, 1.0, abs_tol=1e-6)


def test_consistency_orthogonal_inputs_yield_zero() -> None:
    cr = ConsistencyReward(n_neurons=2)
    cr(torch.tensor([[1.0, 0.0]]))
    r = cr(torch.tensor([[0.0, 1.0]]))
    assert math.isclose(r, 0.0, abs_tol=1e-6)


def test_consistency_opposite_inputs_yield_minus_one() -> None:
    """Cosine sim is signed; opposite-direction inputs give -1."""
    cr = ConsistencyReward(n_neurons=2)
    cr(torch.tensor([[1.0, 1.0]]))
    r = cr(torch.tensor([[-1.0, -1.0]]))
    assert math.isclose(r, -1.0, abs_tol=1e-6)


def test_consistency_ema_drifts_toward_current() -> None:
    cr = ConsistencyReward(n_neurons=2, decay=0.5)
    cr(torch.tensor([[1.0, 0.0]]))  # EMA = (1, 0)
    cr(torch.tensor([[0.0, 1.0]]))  # EMA <- 0.5*(1,0) + 0.5*(0,1) = (0.5, 0.5)
    torch.testing.assert_close(cr.ema, torch.tensor([0.5, 0.5]))


def test_consistency_averages_batch_before_comparing() -> None:
    """The mixer feeds whole batches; mean across batch is the relevant signal."""
    cr = ConsistencyReward(n_neurons=2)
    cr(torch.tensor([[1.0, 1.0]]))
    # A batch with average (1, 1) should match the EMA exactly.
    batch = torch.tensor([[2.0, 0.0], [0.0, 2.0]])  # mean = (1, 1)
    r = cr(batch)
    assert math.isclose(r, 1.0, abs_tol=1e-6)


def test_consistency_reset_drops_history() -> None:
    cr = ConsistencyReward(n_neurons=2)
    cr(torch.tensor([[1.0, 0.0]]))
    cr.reset()
    assert cr.ema is None
    # The next call should behave like the first call: returns 1.0 and seeds.
    r = cr(torch.tensor([[0.5, 0.5]]))
    assert r == 1.0


def test_consistency_rejects_bad_shape() -> None:
    cr = ConsistencyReward(n_neurons=3)
    with pytest.raises(ValueError, match="2-D"):
        cr(torch.zeros(3))
    with pytest.raises(ValueError, match="does not match"):
        cr(torch.zeros(2, 4))


def test_consistency_rejects_bad_constructor_args() -> None:
    with pytest.raises(ValueError):
        ConsistencyReward(n_neurons=0)
    with pytest.raises(ValueError, match="decay"):
        ConsistencyReward(n_neurons=3, decay=1.0)
    with pytest.raises(ValueError, match="decay"):
        ConsistencyReward(n_neurons=3, decay=-0.5)


def test_consistency_does_not_track_gradients() -> None:
    cr = ConsistencyReward(n_neurons=3)
    a = torch.randn(4, 3, requires_grad=True)
    cr(a)
    cr(a)
    # The internal EMA buffer must not be tied to the autograd graph.
    assert cr.ema is not None
    assert not cr.ema.requires_grad
