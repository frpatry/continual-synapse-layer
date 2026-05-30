"""Plasticity rules: covariance Hebbian update + age-modulated decay.

Implements decision D3 in the spec / THEORY.md §3.2:

    Δw[i, j] = η · cov(a_i, a_j) − g(w[i, j], age)

where ``cov`` is the mean-subtracted product (allows both
strengthening and weakening), and ``g`` is a decay term whose
rate is slowed by ``system_age`` (critical-period dynamics, P4).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .connectivity import ConnectivityMatrix


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
    """Decay term whose rate is slowed by system age.

    ``Δw_decay = −(1 / (1 + log(1 + age))) · λ_base · W``

    - Young (``age = 0``): factor = 1, decay applied at the full
      ``λ_base`` rate.
    - Mature (``age ≫ 1``): factor → 0, decay rate approaches
      zero, established weights persist (P4 / critical period).

    Returns a non-positive ``(n, n)`` delta.
    """
    age_factor = 1.0 / (1.0 + math.log(1.0 + max(0.0, float(system_age))))
    return (-age_factor * lambda_base * W).astype(np.float32)


def apply_plasticity(
    connectivity: "ConnectivityMatrix",
    activations: np.ndarray,
    system_age: float,
    eta: float = 0.01,
    lambda_base: float = 0.001,
) -> None:
    """Combined plasticity step — Hebbian growth + age decay.

    Modifies ``connectivity`` in place. Per H3, this runs every
    timestep (forward-pass-as-learning).
    """
    hebb = covariance_hebbian_update(activations, eta)
    decay = age_modulated_decay(connectivity.W, system_age, lambda_base)
    connectivity.update_weights(hebb + decay)
