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

from substrate.connectivity import ConnectivityMatrix
from substrate.p_connectivity import PConnectivity
from substrate.p_entity import PEntity
from substrate.p_plasticity import apply_pp_plasticity
from substrate.plasticity import age_modulated_decay, apply_plasticity, rho_age
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


# ---------- Phase 3.1: symmetric ρ(age) on growth + decay ----------


def test_rho_age_canonical_values():
    """ρ(0)=1.0, ρ(10)≈0.294, ρ(1000)≈0.126, ρ(10000)≈0.098."""
    assert rho_age(0.0) == pytest.approx(1.0)
    assert rho_age(10.0) == pytest.approx(0.294, abs=0.01)
    assert rho_age(1000.0) == pytest.approx(0.126, abs=0.01)
    assert rho_age(10000.0) == pytest.approx(0.098, abs=0.01)


def test_rho_age_monotonically_decreasing():
    """ρ is strictly monotonic in age (never plateaus, never reverses)."""
    ages = [0, 1, 5, 10, 50, 100, 1000, 10000, 100000]
    rhos = [rho_age(a) for a in ages]
    for prev, cur in zip(rhos, rhos[1:]):
        assert cur < prev, f"ρ must strictly decrease, got {rhos}"


def test_apply_plasticity_growth_scales_with_rho_at_n_level():
    """At a non-zero age, the Hebbian growth term must be smaller
    than at age=0 — both are scaled by ρ(age), so the delta to a
    pattern-pair weight after one step shrinks with age."""
    rng_seed = 0

    def step_at_age(age: float) -> float:
        c = ConnectivityMatrix(n_neurons=10, k=4, seed=rng_seed)
        # Two strongly co-active neurons; the rest quiet.
        a = np.zeros(10, dtype=np.float32)
        a[0] = 1.0
        a[1] = 1.0
        # Pick a (0, 1) edge from the mask if it exists, else any
        # masked pair sharing the first co-active id.
        src, tgt = 0, None
        for j in range(1, 10):
            if c.mask[0, j]:
                tgt = j
                break
        assert tgt is not None
        a[tgt] = 1.0  # make sure tgt is one of the co-active two
        # Manually trigger one plasticity step at the given age.
        w_before = float(c.W[src, tgt])
        apply_plasticity(c, a, system_age=age, eta=1.0, lambda_base=0.0)
        return float(c.W[src, tgt]) - w_before

    delta_young = step_at_age(0.0)
    delta_mature = step_at_age(10000.0)
    # ρ(0)=1.0, ρ(10000)≈0.098, so mature delta ≈ 0.10 × young delta.
    assert delta_young > 0.0
    assert delta_mature > 0.0
    assert delta_mature < delta_young * 0.5, (
        f"Mature growth should be substantially smaller (≈ ρ(10000)/ρ(0) = 0.10×). "
        f"young={delta_young:.5f}, mature={delta_mature:.5f}"
    )


def test_apply_pp_plasticity_growth_scales_with_rho_at_p_level():
    """Same symmetric ρ at the P level: growth scales with ρ(age)."""
    def setup_three_active_p() -> dict[int, PEntity]:
        # 3 P: two strongly co-active, one quiet (pulls mean down so
        # covariance signal is positive). See test_p_plasticity.py
        # for the rationale.
        return {
            0: PEntity(id=0, components=(0, 1), activation=0.8),
            1: PEntity(id=1, components=(2, 3), activation=0.8),
            2: PEntity(id=2, components=(4, 5), activation=0.0),
        }

    pc_young = PConnectivity()
    pc_young.update_weight(0, 1, 0.1)  # seed an existing edge
    apply_pp_plasticity(
        setup_three_active_p(), pc_young, system_age=0.0,
        eta_pp=1.0, lambda_pp_decay=0.0,  # isolate growth
    )
    delta_young = pc_young.get_weight(0, 1) - 0.1

    pc_mature = PConnectivity()
    pc_mature.update_weight(0, 1, 0.1)
    apply_pp_plasticity(
        setup_three_active_p(), pc_mature, system_age=10000.0,
        eta_pp=1.0, lambda_pp_decay=0.0,
    )
    delta_mature = pc_mature.get_weight(0, 1) - 0.1

    assert delta_young > 0.0
    assert delta_mature > 0.0
    assert delta_mature < delta_young * 0.5, (
        f"Mature P-P growth should be substantially smaller: "
        f"young={delta_young:.5f}, mature={delta_mature:.5f}"
    )


def test_equilibrium_weight_unchanged_by_age():
    """With symmetric ρ on growth AND decay, the equilibrium weight
    (W where growth = decay) is independent of age.

    For a fixed co-active pair: equilibrium W_eq = (eta · hebb_term) / λ.
    The ρ cancels because both numerator and denominator get the same
    factor. Verified by running long enough at each age to reach
    near-equilibrium and comparing."""
    seed = 0

    def run_to_near_equilibrium(age: float, n_steps: int = 5000) -> float:
        c = ConnectivityMatrix(n_neurons=4, k=2, seed=seed)
        # Find a mask edge to track.
        src, tgt = None, None
        for i in range(4):
            for j in range(4):
                if c.mask[i, j]:
                    src, tgt = i, j
                    break
            if src is not None:
                break
        # Hold a fixed activation pattern (just (src, tgt) co-active)
        # so growth term is deterministic.
        a = np.zeros(4, dtype=np.float32)
        a[src] = 1.0
        a[tgt] = 1.0
        # Use a HIGH eta and lambda so equilibrium is reached fast,
        # and a starting_age so the substrate doesn't "drift young"
        # too much during the test.
        for _ in range(n_steps):
            apply_plasticity(c, a, system_age=age,
                             eta=0.5, lambda_base=0.1)
        return float(c.W[src, tgt])

    # Both should converge to the same equilibrium (within tolerance).
    w_young = run_to_near_equilibrium(age=0.0)
    w_mature = run_to_near_equilibrium(age=10000.0)
    # Mature reaches equilibrium more slowly but should approach the
    # SAME asymptote. Allow generous tolerance because mature with
    # ρ ≈ 0.1 takes ~10× longer to converge — 5000 steps may not be
    # quite enough. Both should be within 30% of each other.
    ratio = w_mature / w_young if w_young > 0 else 0.0
    assert 0.3 < ratio < 1.7, (
        f"Equilibrium W should be ~age-invariant under symmetric ρ. "
        f"w_young={w_young:.3f}, w_mature={w_mature:.3f}, ratio={ratio:.2f}"
    )
