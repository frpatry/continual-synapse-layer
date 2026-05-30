"""Activation propagation + metastable background dynamics.

Implements:
- ``soft_threshold`` activation function (decision D2 in the
  spec) — encourages the sparse-distributed regime H5.
- :class:`GlobalBackground` — H4 spontaneous activity via a slow
  global drift plus per-neuron noise.
- ``propagate_activation`` — synchronous single-timestep update
  (decision D5) per THEORY.md §3.1.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def soft_threshold(x: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """Soft threshold: ``clip(x - threshold, 0, 1)``.

    Subthreshold input → 0 (sparsity, H5). Suprathreshold input is
    linear up to a cap of 1 (saturation; activations live in
    ``[0, 1]`` per THEORY §2.3).
    """
    return np.clip(x - threshold, 0.0, 1.0)


class GlobalBackground:
    """Generates the H4 metastable background drive.

    Implementation choice (D4): a global stochastic wave —
    ``base_level + amplitude * drift_state(t) + local_noise``,
    where ``drift_state`` is a smooth random walk shared across
    all neurons and ``local_noise`` is independent per-neuron
    Gaussian. The substrate is therefore never fully silent and
    the per-step drive is correlated across neurons (network-wide
    metastable fluctuations).
    """

    def __init__(
        self,
        base_level: float = 0.1,
        drift_amplitude: float = 0.05,
        drift_rate: float = 0.02,
        local_noise_sigma: float = 0.02,
        seed: int = 42,
    ) -> None:
        self.base_level = float(base_level)
        self.drift_amplitude = float(drift_amplitude)
        self.drift_rate = float(drift_rate)
        self.local_noise_sigma = float(local_noise_sigma)
        self.rng = np.random.default_rng(seed)
        self.drift_state: float = 0.0

    def step(self, n_neurons: int) -> np.ndarray:
        """One step of the background process. Returns a
        ``(n_neurons,)`` array to be added to the propagated
        weighted input before thresholding."""
        self.drift_state += float(self.rng.normal(0.0, self.drift_rate))
        # Cap the drift so it doesn't wander unboundedly over long
        # runs.
        if self.drift_state > 1.0:
            self.drift_state = 1.0
        elif self.drift_state < -1.0:
            self.drift_state = -1.0
        global_mod = self.base_level + self.drift_amplitude * self.drift_state
        local_noise = self.rng.normal(
            0.0, self.local_noise_sigma, size=n_neurons,
        )
        return (global_mod + local_noise).astype(np.float32)


def propagate_activation(
    current_activations: np.ndarray,
    connectivity_W: np.ndarray,
    neuron_weights: np.ndarray,
    background: np.ndarray,
    external_input: Optional[np.ndarray] = None,
    threshold: float = 0.3,
) -> np.ndarray:
    """One synchronous timestep of activation propagation.

    Per THEORY §3.1::

        N.activation(t+1) = soft_threshold(
            Σ_{j} W[j, i] * N_j.activation(t)
            + background_drive(i, t)
            + external_input(i, t)
        )

    Convention: ``W[j, i]`` = weight from source ``j`` to target
    ``i``. The matrix-vector form is ``W.T @ activations``.

    The result is *modulated* by ``neuron_weights[i]`` — N's
    intrinsic excitability per THEORY §2.3.

    Args:
        current_activations: ``(n,)`` activations at time ``t``.
        connectivity_W: ``(n, n)`` structural weights between N.
        neuron_weights: ``(n,)`` intrinsic excitability of each N.
        background: ``(n,)`` background drive for this step.
        external_input: optional ``(n,)`` external clamping; added
            after the weighted sum.
        threshold: soft-threshold value (default 0.3).

    Returns:
        ``(n,)`` activations at time ``t + 1``.
    """
    # W.T @ a gives the incoming sum at each target.
    weighted_input = connectivity_W.T @ current_activations
    # Modulate by intrinsic excitability (per-target gain).
    weighted_input = weighted_input * neuron_weights
    total = weighted_input + background
    if external_input is not None:
        total = total + external_input
    return soft_threshold(total, threshold).astype(np.float32)
