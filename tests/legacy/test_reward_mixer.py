"""Tests for the RewardMixer."""

from __future__ import annotations

import math

import pytest
import torch

from continual_synapse.reward.consistency import ConsistencyReward
from continual_synapse.reward.external import ExternalReward
from continual_synapse.reward.mixer import RewardMixer
from continual_synapse.reward.surprise import SurpriseReward


def test_mixer_rejects_empty_configuration() -> None:
    with pytest.raises(ValueError, match="at least one component"):
        RewardMixer()


def test_mixer_rejects_negative_gamma() -> None:
    with pytest.raises(ValueError, match="gamma"):
        RewardMixer(external=ExternalReward(), gamma=-1.0)


def test_alpha_starts_at_one_and_decays() -> None:
    mixer = RewardMixer(
        external=ExternalReward(),
        consistency=ConsistencyReward(n_neurons=3),
        gamma=0.1,
    )
    assert mixer.alpha == 1.0
    a = torch.ones(1, 3)
    mixer(a)
    mixer(a)
    mixer(a)
    # step=3 → 1/(1+0.1*3) = 1/1.3 ≈ 0.769
    assert math.isclose(mixer.alpha, 1.0 / 1.3, rel_tol=1e-6)
    assert mixer.step == 3


def test_external_only_returns_external_value() -> None:
    """With only external configured, alpha decay does not zero out reward."""
    er = ExternalReward(default=0.7)
    mixer = RewardMixer(external=er, gamma=10.0)  # large gamma -> small alpha
    for _ in range(20):
        r = mixer(torch.ones(1, 3))
    assert r == 0.7


def test_internal_only_ignores_alpha() -> None:
    """No external -> only internal contributes, no alpha weighting."""
    cr = ConsistencyReward(n_neurons=2)
    mixer = RewardMixer(consistency=cr, w_consistency=2.0, gamma=10.0)
    a = torch.tensor([[1.0, 0.0]])
    r = mixer(a)  # first call seeds EMA, consistency returns 1.0
    assert math.isclose(r, 2.0, abs_tol=1e-6)


def test_external_plus_consistency_blend() -> None:
    """At α=1 the reward equals external; with α<1 it interpolates."""
    er = ExternalReward(default=2.0)
    cr = ConsistencyReward(n_neurons=2)
    mixer = RewardMixer(external=er, consistency=cr, gamma=0.0)
    # gamma=0 -> alpha=1 forever -> reward equals external.
    a = torch.tensor([[1.0, 0.0]])
    assert math.isclose(mixer(a), 2.0, abs_tol=1e-6)
    assert math.isclose(mixer(a), 2.0, abs_tol=1e-6)

    mixer2 = RewardMixer(external=er, consistency=cr, gamma=1.0)
    # First call: alpha=1, reward = 2.0
    r1 = mixer2(a)
    assert math.isclose(r1, 2.0, abs_tol=1e-6)
    # Second call: alpha=1/2, consistency=1.0 -> reward = 0.5*2 + 0.5*1 = 1.5
    r2 = mixer2(a)
    assert math.isclose(r2, 1.5, abs_tol=1e-6)


def test_mixer_applies_component_weights() -> None:
    cr = ConsistencyReward(n_neurons=2)
    sr = SurpriseReward(n_neurons=2)
    mixer = RewardMixer(
        consistency=cr,
        surprise=sr,
        w_consistency=0.5,
        w_surprise=0.5,
        gamma=0.0,
    )
    a = torch.tensor([[1.0, 0.0]])
    # First call: consistency=1.0, surprise=0.0 -> r = 0.5*1 + 0.5*0 = 0.5
    r = mixer(a)
    assert math.isclose(r, 0.5, abs_tol=1e-6)


def test_reset_clears_step_and_components() -> None:
    cr = ConsistencyReward(n_neurons=2)
    sr = SurpriseReward(n_neurons=2)
    mixer = RewardMixer(consistency=cr, surprise=sr)
    a = torch.tensor([[1.0, 1.0]])
    mixer(a)
    mixer(a)
    assert mixer.step == 2
    assert cr.ema is not None
    assert sr.has_history
    mixer.reset()
    assert mixer.step == 0
    assert cr.ema is None
    assert not sr.has_history


def test_mixer_returns_float() -> None:
    """Downstream code passes mixer output to SynapseLayer.consolidate
    which expects a Python float, not a tensor."""
    mixer = RewardMixer(external=ExternalReward())
    r = mixer(torch.ones(1, 3))
    assert isinstance(r, float)
