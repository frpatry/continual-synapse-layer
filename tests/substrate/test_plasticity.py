"""Tests for plasticity: covariance Hebbian + age-modulated decay."""

from __future__ import annotations

import numpy as np

from substrate.connectivity import ConnectivityMatrix
from substrate.plasticity import (
    age_modulated_decay,
    apply_plasticity,
    covariance_hebbian_update,
)


# ---------- covariance_hebbian_update ----------

def test_covariance_hebbian_zero_for_all_zero_activations():
    """All zeros → mean is 0 → outer product is 0 → delta is 0."""
    a = np.zeros(8, dtype=np.float32)
    d = covariance_hebbian_update(a, eta=0.1)
    assert (d == 0.0).all()


def test_covariance_hebbian_no_diagonal():
    """The diagonal must be exactly zero regardless of input."""
    a = np.array([1.0, 0.5, 0.0, 0.8], dtype=np.float32)
    d = covariance_hebbian_update(a, eta=0.1)
    assert (np.diag(d) == 0.0).all()


def test_covariance_hebbian_positive_for_co_active_pair():
    """Two neurons co-activated above the mean → positive delta."""
    # Activations [0.9, 0.9, 0.0, 0.0] — mean=0.45.
    # Pair (0,1) co-active → outer=0.81; baseline=0.45²=0.2025;
    # delta = η * (0.81 − 0.2025) > 0.
    a = np.array([0.9, 0.9, 0.0, 0.0], dtype=np.float32)
    d = covariance_hebbian_update(a, eta=0.1)
    assert d[0, 1] > 0.0
    assert d[1, 0] > 0.0


def test_covariance_hebbian_negative_for_anti_pair():
    """Pair where only one is high and one is low (anti-correlated
    relative to mean) gets a negative delta."""
    a = np.array([0.9, 0.0, 0.9, 0.0], dtype=np.float32)
    d = covariance_hebbian_update(a, eta=0.1)
    # (0, 1) → outer = 0; baseline = 0.45² = 0.2025; delta < 0.
    assert d[0, 1] < 0.0


def test_covariance_hebbian_returns_float32():
    a = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    d = covariance_hebbian_update(a)
    assert d.dtype == np.float32


# ---------- age_modulated_decay ----------

def test_decay_at_age_zero_uses_full_rate():
    """At age 0: factor = 1, delta = -λ_base * W."""
    W = np.array([[0.0, 0.5], [0.3, 0.0]], dtype=np.float32)
    d = age_modulated_decay(W, system_age=0.0, lambda_base=0.01)
    expected = -0.01 * W
    assert np.allclose(d, expected)


def test_decay_at_large_age_is_smaller_than_at_age_zero():
    """At a much older age, the decay magnitude is reduced."""
    W = np.ones((3, 3), dtype=np.float32)
    young = age_modulated_decay(W, system_age=0.0, lambda_base=0.01)
    old = age_modulated_decay(W, system_age=1000.0, lambda_base=0.01)
    # ``decay`` values are negative; "smaller in magnitude" → closer to 0.
    assert np.abs(old).mean() < np.abs(young).mean()


def test_decay_age_growth_monotonic():
    """Decay magnitude strictly decreases with age."""
    W = np.ones((2, 2), dtype=np.float32)
    ages = [0, 1, 10, 100, 1000, 10000]
    mags = [float(np.abs(age_modulated_decay(W, a)).mean()) for a in ages]
    for prev, cur in zip(mags, mags[1:]):
        assert prev >= cur


def test_decay_only_signed_negative():
    """Decay is always non-positive (it reduces weights)."""
    W = np.array([[0.0, 0.5], [0.3, 0.0]], dtype=np.float32)
    d = age_modulated_decay(W, system_age=5.0, lambda_base=0.01)
    assert (d <= 0.0).all()


# ---------- apply_plasticity ----------

def test_apply_plasticity_modifies_weights():
    """Co-activated neurons that are in the mask should see their
    weight grow after a plasticity step."""
    c = ConnectivityMatrix(n_neurons=4, k=2, seed=0)
    # Find a (source, target) pair that IS in the mask.
    src, tgt = np.argwhere(c.mask)[0]
    initial_w = c.W[src, tgt]
    # Co-activate them strongly, others quiet.
    a = np.zeros(4, dtype=np.float32)
    a[src] = 1.0
    a[tgt] = 1.0
    apply_plasticity(c, a, system_age=0.0, eta=0.5, lambda_base=0.0)
    assert c.W[src, tgt] > initial_w


def test_apply_plasticity_does_not_change_offmask_entries():
    """Even with a Hebbian-positive pair, weights off the mask
    must remain zero (the topology is fixed in Phase 1)."""
    c = ConnectivityMatrix(n_neurons=6, k=2, seed=1)
    off_mask_idx = np.argwhere(~c.mask)[0]
    a = np.ones(6, dtype=np.float32) * 0.9
    apply_plasticity(c, a, system_age=0.0, eta=0.5, lambda_base=0.0)
    assert c.W[off_mask_idx[0], off_mask_idx[1]] == 0.0
