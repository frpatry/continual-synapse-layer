"""Tests for s_dynamics — S-level propagation + adaptive k-WTA."""

from __future__ import annotations

import numpy as np
import pytest

from substrate.p_entity import PEntity
from substrate.s_dynamics import (
    compute_s_k,
    propagate_s_activations,
    s_winners_take_all,
)
from substrate.s_entity import SEntity


# ---------- compute_s_k (adaptive k bounds) ----------


def test_compute_s_k_zero_n_returns_zero():
    assert compute_s_k(0) == 0


def test_compute_s_k_respects_min_bound():
    """Tiny pool with low sparsity must still have min_active winners."""
    assert compute_s_k(2, sparsity=0.05, min_active=1, max_active=3) == 1
    assert compute_s_k(5, sparsity=0.05, min_active=1, max_active=3) == 1


def test_compute_s_k_respects_max_bound():
    """Large pool with high sparsity caps at max_active."""
    assert compute_s_k(100, sparsity=0.20, min_active=1, max_active=3) == 3
    assert compute_s_k(50, sparsity=0.50, min_active=1, max_active=3) == 3


def test_compute_s_k_natural_range():
    """In the unbounded sweet spot, k is naïve int(sparsity·n)."""
    # n=10, sparsity=0.30 → naïve k=3, fits [1,5]
    assert compute_s_k(10, sparsity=0.30, min_active=1, max_active=5) == 3


# ---------- s_winners_take_all ----------


def test_s_wta_returns_zeros_when_k_zero():
    out = s_winners_take_all(np.array([0.5, 0.3, 0.8], dtype=np.float32), k=0)
    assert (out == 0.0).all()


def test_s_wta_returns_copy_when_k_ge_n():
    a = np.array([0.5, 0.3, 0.8], dtype=np.float32)
    out = s_winners_take_all(a, k=5)
    assert np.array_equal(out, a)
    assert out is not a  # copy, not the same array


def test_s_wta_keeps_top_k():
    a = np.array([0.1, 0.5, 0.3, 0.9, 0.2], dtype=np.float32)
    out = s_winners_take_all(a, k=2)
    # top 2 are 0.9 and 0.5 → kept; others zeroed.
    assert out[3] == pytest.approx(0.9)
    assert out[1] == pytest.approx(0.5)
    assert out[0] == 0.0
    assert out[2] == 0.0
    assert out[4] == 0.0


# ---------- propagate_s_activations ----------


def test_propagate_s_no_entities_returns_empty():
    p = {0: PEntity(id=0, components=(0, 1), activation=0.5)}
    result = propagate_s_activations(
        s_entities={}, p_entities=p,
        rng=np.random.default_rng(0),
    )
    assert result == {}


def test_propagate_s_input_from_component_p_mean():
    """An S's raw input is α · mean(P.activation for P in contents)."""
    p_entities = {
        0: PEntity(id=0, components=(0, 1), activation=0.6),
        1: PEntity(id=1, components=(2, 3), activation=0.8),
        2: PEntity(id=2, components=(4, 5), activation=0.0),  # unrelated
    }
    s_entities = {
        100: SEntity(id=100, contents={0, 1}),
    }
    result = propagate_s_activations(
        s_entities=s_entities, p_entities=p_entities,
        alpha_p_to_s=1.0, s_threshold=0.0,
        s_sparsity_target=1.0, s_min_active=1, s_max_active=3,
        s_background_noise_sigma=0.0,
        rng=np.random.default_rng(0),
    )
    # Mean of P0=0.6 + P1=0.8 → 0.7, threshold=0 → 0.7, k=1, kept.
    assert result[100] == pytest.approx(0.7, abs=1e-5)


def test_propagate_s_threshold_zeroes_low_input():
    """If raw S input is below threshold, S activation = 0."""
    p_entities = {0: PEntity(id=0, components=(0, 1), activation=0.1)}
    s_entities = {100: SEntity(id=100, contents={0})}
    result = propagate_s_activations(
        s_entities=s_entities, p_entities=p_entities,
        alpha_p_to_s=0.3, s_threshold=0.5,
        s_min_active=1, s_max_active=3,
        s_background_noise_sigma=0.0,
        rng=np.random.default_rng(0),
    )
    # raw = 0.3 · 0.1 = 0.03 < 0.5 → 0.
    assert result[100] == 0.0


def test_propagate_s_skips_dissolved_member_p():
    """An S whose contents include a dissolved P (not in p_entities)
    just averages over the remaining live members."""
    p_entities = {0: PEntity(id=0, components=(0, 1), activation=0.6)}
    s_entities = {
        100: SEntity(id=100, contents={0, 99}),  # 99 dissolved
    }
    result = propagate_s_activations(
        s_entities=s_entities, p_entities=p_entities,
        alpha_p_to_s=1.0, s_threshold=0.0,
        s_min_active=1, s_max_active=3,
        s_background_noise_sigma=0.0,
        rng=np.random.default_rng(0),
    )
    # Only live member P0=0.6 → mean=0.6, post-threshold 0.6, kept.
    assert result[100] == pytest.approx(0.6, abs=1e-5)


def test_propagate_s_k_wta_enforces_max():
    """5 S all with high input but max_active=2 → only 2 kept positive."""
    p_entities = {
        i: PEntity(id=i, components=(i, i + 100), activation=0.9)
        for i in range(5)
    }
    s_entities = {
        100 + i: SEntity(id=100 + i, contents={i})
        for i in range(5)
    }
    result = propagate_s_activations(
        s_entities=s_entities, p_entities=p_entities,
        alpha_p_to_s=1.0, s_threshold=0.0,
        s_sparsity_target=0.5, s_min_active=1, s_max_active=2,
        s_background_noise_sigma=0.01,  # small noise to break ties
        rng=np.random.default_rng(0),
    )
    n_active = sum(1 for v in result.values() if v > 0.0)
    assert n_active == 2
