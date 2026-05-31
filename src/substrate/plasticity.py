"""Plasticity rules: covariance Hebbian update + age-modulated decay.

Implements decision D3 / THEORY.md §3.2:

    Δw[i, j] = ρ(age) · η · cov(a_i, a_j) − ρ(age) · λ · w[i, j]
    ρ(age)   = max(floor, 1 / (1 + log(1 + age)))

The age modulator ``ρ(age)`` scales BOTH growth and decay
(symmetric modulation per Phase 3.1's corrected §3.2). The floor
(Phase 6a) keeps plasticity from collapsing below a minimum so
substrates aged beyond ~age=10 can still emerge new patterns —
fixes Phase 5b's "late patterns can't form" bottleneck (a) at the
cost of a partially-reduced critical-period asymmetry.

* Young (age=0, ρ=1.0): fast plasticity in both directions.
* Pre-floor mature (age=10000, no floor, ρ≈0.1): very slow.
* Floored mature (age=10000, floor=0.3, ρ=0.3): preserves
  emergence capacity at the cost of slightly slower equilibrium.

Phase 3 (asymmetric — decay-only) produced a WEAK verdict because
mature substrates had *less* decay competing with the unscaled
growth, so mature actually reached equilibrium faster. Phase 3.1's
symmetric formulation produces the biological critical-period story.
Phase 6a's floor addresses the practical fall-out: an asymmetric-
free, ρ-floored substrate retains the critical-period direction in
its early window (steps 0–~10 where raw ρ > floor) while still
permitting late-life emergence.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .connectivity import ConnectivityMatrix


def rho_age(system_age: float, floor: float = 0.0) -> float:
    """Symmetric age modulator with optional floor (Phase 6a).

    ρ(age) = max(floor, 1 / (1 + log(1 + age)))

    Returns 1.0 at age=0, ≈0.29 at age=10, ≈0.13 at age=1000,
    ≈0.10 at age=10000 (without floor). With floor=0.3, ρ stays at
    0.3 from age ≈ 9.3 onward (the age at which the raw expression
    equals 0.3 — solve ``1/(1+log(1+a)) = 0.3``).

    Applied SYMMETRICALLY to both Hebbian growth and decay:
        Δw = ρ · (η · hebb_term − λ · w)

    Args:
        system_age: substrate age in steps (clamped at 0 if negative).
        floor: minimum value of ρ. Default 0.0 (pre-Phase-6a behaviour);
            Substrate now defaults this to 0.3 in its constructor.
    """
    raw = 1.0 / (1.0 + math.log(1.0 + max(0.0, float(system_age))))
    return max(float(floor), raw)


def covariance_hebbian_update(
    activations: np.ndarray,
    eta: float = 0.01,
) -> np.ndarray:
    """Covariance-form Hebbian update.

    ``Δw[i, j] = η · (a_i · a_j − ⟨a⟩²)``

    Reinforces ``(i, j)`` when both co-activate above the mean,
    weakens when one or both are below — more stable than pure
    Hebb (no run-away growth from low-correlated firing). The
    diagonal is zeroed out (no self-connection updates; the
    connectivity mask also forbids self-connections).
    """
    mean_a = float(activations.mean())
    outer = np.outer(activations, activations)
    delta = eta * (outer - mean_a * mean_a)
    np.fill_diagonal(delta, 0.0)
    return delta.astype(np.float32)


def age_modulated_decay(
    W: np.ndarray,
    system_age: float,
    lambda_base: float = 0.001,
    rho_floor: float = 0.0,
) -> np.ndarray:
    """Decay term scaled by ``ρ(age, floor)`` — ``Δw_decay = −ρ · λ · W``.

    Args mirror :func:`apply_plasticity`'s ``rho_floor`` so callers can
    invoke this in isolation with the same age-modulation contract.
    Returns a non-positive ``(n, n)`` delta.
    """
    rho = rho_age(system_age, floor=rho_floor)
    return (-rho * lambda_base * W).astype(np.float32)


def apply_plasticity(
    connectivity: "ConnectivityMatrix",
    activations: np.ndarray,
    system_age: float,
    eta: float = 0.01,
    lambda_base: float = 0.001,
    rho_floor: float = 0.0,
) -> None:
    """Combined plasticity step — Hebbian growth + age decay,
    BOTH scaled by ρ(age, floor) (Phase 3.1 symmetric + Phase 6a floor).

    Modifies ``connectivity`` in place. Per H3, runs every timestep
    (forward-pass-as-learning).

    Args:
        rho_floor: minimum age-modulation factor. 0.0 (default at this
            function level) reproduces pre-Phase-6a behaviour; the
            Substrate class defaults its own ctor to 0.3 and threads
            that through here.
    """
    rho = rho_age(system_age, floor=rho_floor)
    hebb = covariance_hebbian_update(activations, eta=eta * rho)
    # Reuse the same ρ via the explicit floor — the helper would
    # recompute it but we want a single source of truth per step.
    decay = age_modulated_decay(
        connectivity.W, system_age, lambda_base, rho_floor=rho_floor,
    )
    connectivity.update_weights(hebb + decay)
