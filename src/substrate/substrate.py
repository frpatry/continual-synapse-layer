"""Substrate — unified orchestrator for Phase 1 dynamics.

Per H1 (unified substrate), this is the single container that
holds both *computation* (current activations) and *memory*
(structural weights). Every call to :meth:`step` advances both
together — there is no separate "train" mode vs "infer" mode.
That distinction is exactly what H1 / H3 argue against.

Phase 1 scope:
- N entities only (no P / S / C).
- Sparse implicit connectivity, plasticity over edge weights.
- Background dynamics keep the substrate non-silent (H4).
- Soft-threshold activation pushes toward sparse codes (H5).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .connectivity import ConnectivityMatrix
from .dynamics import DEFAULT_SPARSITY_TARGET, GlobalBackground, propagate_activation
from .neuron import N
from .plasticity import apply_plasticity


class Substrate:
    """Brain-aligned substrate, Phase 1 (N only).

    Public attributes that callers / tests inspect:
        n_neurons (int): substrate size.
        neurons (list[N]): per-neuron entity objects (id + weight).
        activations (np.ndarray): ``(n,)`` current activations.
        neuron_weights (np.ndarray): ``(n,)`` per-neuron intrinsic
            excitability (modulates incoming weighted sum).
        connectivity (ConnectivityMatrix): sparse weights between
            neurons.
        background (GlobalBackground): H4 background dynamics.
        system_age (float): increments by 1.0 on every step;
            drives age-modulated plasticity decay.
        step_count (int): total steps since construction.
    """

    def __init__(
        self,
        n_neurons: int = 500,
        k_connectivity: int = 50,
        threshold: float = 0.3,
        eta: float = 0.01,
        lambda_decay: float = 0.001,
        sparsity_target: float = DEFAULT_SPARSITY_TARGET,
        background_base: float = 0.1,
        background_drift_amp: float = 0.05,
        background_drift_rate: float = 0.02,
        local_noise_sigma: float = 0.02,
        seed: int = 42,
    ) -> None:
        self.n_neurons = int(n_neurons)
        self.threshold = float(threshold)
        self.eta = float(eta)
        self.lambda_decay = float(lambda_decay)
        self.sparsity_target = float(sparsity_target)
        self.seed = int(seed)

        # Per-N entity objects (mostly carry the id + intrinsic
        # weight; canonical state lives in the numpy arrays below).
        self.neurons: List[N] = [N(id=i) for i in range(self.n_neurons)]

        self.activations: np.ndarray = np.zeros(self.n_neurons, dtype=np.float32)
        self.neuron_weights: np.ndarray = np.ones(self.n_neurons, dtype=np.float32)

        self.connectivity = ConnectivityMatrix(
            n_neurons=self.n_neurons,
            k=k_connectivity,
            seed=seed,
        )
        self.background = GlobalBackground(
            base_level=background_base,
            drift_amplitude=background_drift_amp,
            drift_rate=background_drift_rate,
            local_noise_sigma=local_noise_sigma,
            seed=seed + 1,
        )
        self.system_age: float = 0.0
        self.step_count: int = 0

    # ---------- core update ----------

    def step(self, external_input: Optional[np.ndarray] = None) -> np.ndarray:
        """One synchronous timestep.

        Order:
          1. Generate background drive.
          2. Propagate activations.
          3. Apply plasticity (H3 — every step).
          4. Advance age + step counter.

        Returns a copy of the new activations so the caller can
        store snapshots without seeing later steps' mutations.
        """
        bg = self.background.step(self.n_neurons)
        new_acts = propagate_activation(
            current_activations=self.activations,
            connectivity_W=self.connectivity.W,
            neuron_weights=self.neuron_weights,
            background=bg,
            external_input=external_input,
            threshold=self.threshold,
            sparsity_target=self.sparsity_target,
        )
        self.activations = new_acts
        apply_plasticity(
            connectivity=self.connectivity,
            activations=self.activations,
            system_age=self.system_age,
            eta=self.eta,
            lambda_base=self.lambda_decay,
        )
        self.system_age += 1.0
        self.step_count += 1
        return new_acts.copy()

    # ---------- diagnostics ----------

    def sparsity(self, threshold: float = 0.1) -> float:
        """Fraction of neurons currently above ``threshold``
        activation. Sparsity target per H5 is roughly ~5 %."""
        return float((self.activations > threshold).mean())

    def total_weight(self) -> float:
        """Sum of all structural weights — proxy for accumulated
        learning."""
        return float(self.connectivity.W.sum())
