"""Tests for the Phase 6a ρ floor — minimum plasticity factor."""

from __future__ import annotations

import math

import numpy as np
import pytest

from substrate.connectivity import ConnectivityMatrix
from substrate.plasticity import apply_plasticity, rho_age
from substrate.substrate import Substrate


# ---------- rho_age behaviour ----------


def test_rho_age_no_floor_default():
    """Default floor=0.0 → pre-Phase-6a behaviour preserved."""
    assert rho_age(0.0, floor=0.0) == pytest.approx(1.0)
    assert rho_age(10000.0, floor=0.0) == pytest.approx(0.098, abs=0.01)


def test_rho_age_respects_floor_at_high_age():
    """At high age the floor is the lower bound on ρ."""
    assert rho_age(1e9, floor=0.3) == pytest.approx(0.3)
    assert rho_age(1e9, floor=0.5) == pytest.approx(0.5)
    assert rho_age(1e9, floor=0.1) == pytest.approx(0.1)


def test_rho_age_unaffected_when_above_floor():
    """When raw ρ > floor, the raw value is returned."""
    # age=0 → raw ρ = 1.0, well above floor=0.3
    assert rho_age(0.0, floor=0.3) == pytest.approx(1.0)
    # age=10 → raw ρ ≈ 0.294, just below floor=0.3 → clamps
    assert rho_age(10.0, floor=0.3) == pytest.approx(0.3)


def test_rho_age_floor_transition_point():
    """The transition happens where raw ρ = floor, derivable as
    age = exp(1/floor − 1) − 1."""
    floor = 0.3
    transition_age = math.exp(1.0 / floor - 1.0) - 1.0
    # Just below transition — still on the raw curve.
    assert rho_age(transition_age * 0.5, floor=floor) > floor
    # Well past transition — clamped to floor.
    assert rho_age(transition_age * 10.0, floor=floor) == pytest.approx(floor)


def test_rho_age_floor_zero_matches_raw():
    """floor=0.0 matches the raw expression exactly across the range."""
    for age in (0.0, 1.0, 10.0, 100.0, 10000.0, 1e6):
        assert rho_age(age, floor=0.0) == pytest.approx(
            1.0 / (1.0 + math.log(1.0 + age))
        )


# ---------- Substrate ctor / default ----------


def test_substrate_accepts_rho_floor():
    sub = Substrate(n_neurons=10, k_connectivity=3, rho_floor=0.3)
    assert sub.rho_floor == 0.3


def test_substrate_default_rho_floor():
    """Substrate's default is the Phase 6a recommended floor (0.3)."""
    sub = Substrate(n_neurons=10, k_connectivity=3)
    assert sub.rho_floor == 0.3


def test_substrate_rho_floor_zero_matches_no_floor():
    """rho_floor=0.0 reproduces pre-Phase-6a behaviour exactly."""
    sub = Substrate(
        n_neurons=20, k_connectivity=3,
        rho_floor=0.0, starting_age=10000.0, seed=0,
    )
    effective = rho_age(sub.system_age, floor=sub.rho_floor)
    raw = rho_age(sub.system_age, floor=0.0)
    assert effective == pytest.approx(raw)


# ---------- apply_plasticity integration ----------


def test_apply_plasticity_with_floor_at_high_age_grows_more():
    """At very high age, a floored substrate sees substantially more
    weight change (in magnitude) per plasticity step than an un-floored
    one. Direction of change depends on the activation pattern (anti-
    Hebbian + decay can dominate when most pairs are uncorrelated);
    what we test is the SIZE of the change, since both growth and
    decay scale with ρ."""
    # Capture initial W for both substrates (same seed → identical).
    initial_W = ConnectivityMatrix(n_neurons=20, k=10, seed=0).W.copy()

    conn_no_floor = ConnectivityMatrix(n_neurons=20, k=10, seed=0)
    conn_floored = ConnectivityMatrix(n_neurons=20, k=10, seed=0)

    activations = np.zeros(20, dtype=np.float32)
    activations[:5] = 0.7

    high_age = 1e6  # well past floor activation point

    apply_plasticity(
        conn_no_floor, activations, system_age=high_age,
        eta=0.01, lambda_base=0.001, rho_floor=0.0,
    )
    apply_plasticity(
        conn_floored, activations, system_age=high_age,
        eta=0.01, lambda_base=0.001, rho_floor=0.3,
    )

    # Magnitude of delta from the initial state.
    delta_no_floor = float(np.abs(conn_no_floor.W - initial_W).sum())
    delta_floored = float(np.abs(conn_floored.W - initial_W).sum())

    # ρ(1e6, floor=0) ≈ 0.067; ρ(1e6, floor=0.3) = 0.3 → 4.5× ratio.
    # Both growth AND decay scale with ρ, so the per-step delta
    # magnitude scales with ρ regardless of direction. We assert > 3×
    # to leave room for nonlinear floor / clipping at zero.
    assert delta_floored > 3.0 * delta_no_floor, (
        f"|Δ| with floor should be > 3× |Δ| without: "
        f"floored={delta_floored:.5f}, no-floor={delta_no_floor:.5f}"
    )
