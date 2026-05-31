"""Tests for age-dependent plasticity infrastructure (Phase 3, P4).

These cover the *substrate's age machinery* itself — the formula
behaviour, the ctor parameter, and the contrast between a young and a
mature substrate's retention. Whether the **biological** critical-
period story (fast/volatile young, slow/stable adult) actually emerges
under THEORY.md §3.2's specific formula ``1/(1+log(1+age))`` is an
empirical question answered by the Phase 3 experiment, not these unit
tests.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from substrate.plasticity import age_modulated_decay
from substrate.substrate import Substrate


# ---------- starting_age plumbing ----------


def test_substrate_starting_age_default_zero():
    sub = Substrate(n_neurons=20, k_connectivity=4)
    assert sub.system_age == 0.0


def test_substrate_starting_age_custom():
    sub = Substrate(n_neurons=20, k_connectivity=4, starting_age=1000.0)
    assert sub.system_age == 1000.0


def test_substrate_step_advances_age_from_starting():
    """system_age increments by 1.0 per step regardless of starting offset."""
    sub = Substrate(n_neurons=20, k_connectivity=4, starting_age=500.0)
    for _ in range(10):
        sub.step()
    assert sub.system_age == 510.0


# ---------- age_modulated_decay formula behaviour ----------


def test_age_modulated_decay_magnitude_decreases_with_age():
    """Higher age → smaller decay magnitude (THEORY §3.2)."""
    W = np.ones((5, 5), dtype=np.float32)
    decay_young = age_modulated_decay(W, system_age=0.0, lambda_base=0.01)
    decay_middle = age_modulated_decay(W, system_age=100.0, lambda_base=0.01)
    decay_mature = age_modulated_decay(W, system_age=10000.0, lambda_base=0.01)

    mag_young = float(-decay_young.sum())
    mag_middle = float(-decay_middle.sum())
    mag_mature = float(-decay_mature.sum())

    assert mag_young > mag_middle > mag_mature, (
        f"Expected decreasing decay magnitude with age, got "
        f"young={mag_young}, middle={mag_middle}, mature={mag_mature}"
    )


def test_age_modulated_decay_formula_values():
    """Direct sanity check of ``1/(1+log(1+age))`` at three ages."""
    # age=0 → factor = 1/(1+log(1)) = 1.0
    assert 1.0 / (1.0 + math.log(1.0)) == pytest.approx(1.0)
    # age=10 → factor = 1/(1+log(11)) ≈ 0.294
    assert 1.0 / (1.0 + math.log(11.0)) == pytest.approx(0.294, abs=0.01)
    # age=1000 → factor = 1/(1+log(1001)) ≈ 0.126
    assert 1.0 / (1.0 + math.log(1001.0)) == pytest.approx(0.126, abs=0.01)


# ---------- end-to-end retention contrast ----------


def test_young_substrate_pattern_decays_faster_than_mature():
    """Train both, then idle. Mature should retain a larger fraction
    of its post-training weight than young.

    This is the *retention* half of P4 — the half we have strong
    theoretical reason to expect from the math. (The "young learns
    faster" half is a separate empirical question handled by the
    Phase 3 experiment, not this unit test.)
    """
    seed = 42
    n = 100
    pattern = np.array([3, 7, 12, 24, 30])
    external = np.zeros(n, dtype=np.float32)
    external[pattern] = 0.7

    sub_young = Substrate(
        n_neurons=n, k_connectivity=10,
        starting_age=0.0, seed=seed,
    )
    sub_mature = Substrate(
        n_neurons=n, k_connectivity=10,
        starting_age=10000.0, seed=seed,
    )

    for _ in range(50):
        sub_young.step(external_input=external)
        sub_mature.step(external_input=external)

    w_young_post_train = float(sub_young.connectivity.W.sum())
    w_mature_post_train = float(sub_mature.connectivity.W.sum())

    for _ in range(500):
        sub_young.step()
        sub_mature.step()

    w_young_post_idle = float(sub_young.connectivity.W.sum())
    w_mature_post_idle = float(sub_mature.connectivity.W.sum())

    retention_young = w_young_post_idle / max(w_young_post_train, 1e-6)
    retention_mature = w_mature_post_idle / max(w_mature_post_train, 1e-6)

    assert retention_mature > retention_young, (
        f"Mature substrate should retain more than young: "
        f"mature={retention_mature:.3f}, young={retention_young:.3f}"
    )
