"""Plasticity rules: covariance Hebbian update + age-modulated decay.

Implements decision D3 / THEORY.md §3.2:

    Δw[i, j] = ρ(age) · η · cov(a_i, a_j) − ρ(age) · λ · w[i, j]

The age modulator ``ρ(age) = 1 / (1 + log(1 + age))`` scales BOTH
growth and decay (Phase 3.1: symmetric modulation per the corrected
§3.2). This preserves the equilibrium weight ``W_eq = η·hebb / λ``
across ages but changes the *timescale* of convergence:

* Young (``age=0``, ρ=1.0): fast plasticity in both directions →
  fast learning AND fast forgetting (biological volatility).
* Mature (``age≫1``, ρ≈0.1): slow plasticity in both directions →
  slow learning AND slow forgetting (biological stability).

Phase 3 (asymmetric — decay-only) produced a WEAK verdict because
mature substrates had *less* decay competing with the unscaled
growth, so mature actually reached equilibrium faster. Phase 3.1's
symmetric formulation produces the biological critical-period story.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .connectivity import ConnectivityMatrix


def rho_age(system_age: float) -> float:
    """Symmetric age modulator (THEORY.md §3.2, Phase 3.1).

    ρ(age) = 1 / (1 + log(1 + age))

    Returns 1.0 at age=0, ≈0.29 at age=10, ≈0.13 at age=1000,
    ≈0.10 at age=10000. Asymptotes toward 0 but never reaches it.

    Applied SYMMETRICALLY to both Hebbian growth and decay, so:
        Δw = ρ · (η · hebb_term − λ · w)
    """
    return 1.0 / (1.0 + math.log(1.0 + max(0.0, float(system_age))))


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
) -> np.ndarray:
    """Decay term whose rate is slowed by system age — ``Δw_decay = −ρ(age) · λ · W``.

    - Young (``age = 0``): ρ = 1, decay at full ``λ`` rate.
    - Mature (``age ≫ 1``): ρ → 0, decay rate approaches zero.

    Returns a non-positive ``(n, n)`` delta. (Same as Phase 1.x;
    unchanged in 3.1 — symmetric modulation is achieved by
    :func:`apply_plasticity` also scaling ``eta`` by ``ρ``.)
    """
    return (-rho_age(system_age) * lambda_base * W).astype(np.float32)


def apply_plasticity(
    connectivity: "ConnectivityMatrix",
    activations: np.ndarray,
    system_age: float,
    eta: float = 0.01,
    lambda_base: float = 0.001,
) -> None:
    """Combined plasticity step — Hebbian growth + age decay,
    BOTH scaled by ρ(age) (Phase 3.1, symmetric modulation).

    Modifies ``connectivity`` in place. Per H3, runs every timestep
    (forward-pass-as-learning).
    """
    rho = rho_age(system_age)
    hebb = covariance_hebbian_update(activations, eta=eta * rho)
    # ``age_modulated_decay`` already scales by ρ internally.
    decay = age_modulated_decay(connectivity.W, system_age, lambda_base)
    connectivity.update_weights(hebb + decay)
