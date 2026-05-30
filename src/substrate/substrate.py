"""Substrate — unified orchestrator for Phase 1 + 2a dynamics.

Per H1 (unified substrate), this is the single container that
holds both *computation* (current activations) and *memory*
(structural weights). Every call to :meth:`step` advances both
together — there is no separate "train" mode vs "infer" mode.
That distinction is exactly what H1 / H3 argue against.

Phase 1 / 1.1 scope:
- N entities only, sparse implicit connectivity, plasticity over
  edge weights.
- Background dynamics keep the substrate non-silent (H4).
- Soft-threshold + k-WTA push toward sparse codes (H5, §3.6).

Phase 2a scope (observational P):
- :class:`PassTracker` accumulates emergence candidacy for every
  connected pair on every step.
- When a pair clears the γ structural threshold AND has been
  validated across enough distinct passes, a :class:`PEntity` is
  emerged and tracked in ``self.p_entities``.
- P weights grow with use, decay with age, and dissolve when they
  fall below ``p_viability_threshold``.
- P does NOT yet influence N-level dynamics — that arrives in
  Phase 2b. Here P is purely observational: we measure whether
  emergence concentrates correctly on pattern-pair structure (P2).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .connectivity import ConnectivityMatrix
from .dynamics import DEFAULT_SPARSITY_TARGET, GlobalBackground, propagate_activation
from .neuron import N
from .p_entity import PEntity
from .pass_tracker import PassTracker
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
        # ---- Phase 2a P-emergence parameters ----
        theta_emergence: float = 0.5,
        n_min_passes: int = 3,
        pass_boost: float = 0.1,
        pass_decay: float = 0.95,
        pass_theta_high: float = 0.1,
        pass_theta_low: float = 0.05,
        p_weight_decay: float = 0.005,
        p_viability_threshold: float = 0.1,
        # -----------------------------------------
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

        # ---------- Phase 2a P-level state ----------
        self.theta_emergence: float = float(theta_emergence)
        self.n_min_passes: int = int(n_min_passes)
        self.p_weight_decay: float = float(p_weight_decay)
        self.p_viability_threshold: float = float(p_viability_threshold)

        self.pass_tracker = PassTracker(
            connectivity_mask=self.connectivity.mask,
            boost_factor=pass_boost,
            decay_factor=pass_decay,
            theta_quiet_high=pass_theta_high,
            theta_quiet_low=pass_theta_low,
        )

        # P entities keyed by their id; ``p_pairs_emerged`` is the
        # fast lookup set used to suppress re-emergence of pairs we
        # already have a live P for.
        self.p_entities: Dict[int, PEntity] = {}
        self.p_pairs_emerged: Set[Tuple[int, int]] = set()
        self._next_p_id: int = 0
        # Diagnostic log of (step, p_id, components, system_age) for
        # every emergence event — observed by experiments / tests.
        self.p_emergence_history: List[dict] = []

    # ---------- core update ----------

    def step(self, external_input: Optional[np.ndarray] = None) -> np.ndarray:
        """One synchronous timestep.

        Order:
          1. Generate background drive.
          2. Propagate activations.
          3. Apply N-level plasticity (H3 — every step).
          4. Update P-level pass tracker on the new activations.
          5. Check for new P emergences.
          6. Update P activations (derived from components in 2a).
          7. Decay P weights, dissolve non-viable.
          8. Advance age + step counter.

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

        # ---------- Phase 2a: P-level pass ----------
        self.pass_tracker.update(self.activations)
        candidates = self.pass_tracker.find_emergence_candidates(
            connectivity_W=self.connectivity.W,
            theta_emergence=self.theta_emergence,
            n_min_passes=self.n_min_passes,
            existing_p_pairs=self.p_pairs_emerged,
        )
        for (i, j) in candidates:
            self._emerge_p(i, j)
        # P activations are purely derived from components in 2a.
        for p in self.p_entities.values():
            i, j = p.components
            p.activation = float(
                (self.activations[i] + self.activations[j]) / 2.0,
            )
        self._decay_and_dissolve_p()

        self.system_age += 1.0
        self.step_count += 1
        return new_acts.copy()

    # ---------- P-level helpers ----------

    def _emerge_p(self, i: int, j: int) -> None:
        """Create a fresh PEntity for the canonical pair (i, j)."""
        p = PEntity(
            id=self._next_p_id,
            components=(int(i), int(j)),
            activation=float(
                (self.activations[i] + self.activations[j]) / 2.0,
            ),
            weight=1.0,
            age_at_emergence=self.system_age,
        )
        self.p_entities[self._next_p_id] = p
        self.p_pairs_emerged.add(p.components)
        self.p_emergence_history.append(
            {
                "step": self.step_count,
                "p_id": p.id,
                "components": p.components,
                "system_age": self.system_age,
            }
        )
        self._next_p_id += 1

    def _decay_and_dissolve_p(self) -> None:
        """Hebb-like growth + age-modulated decay on every P; dissolve
        those whose weight falls below the viability threshold.

        Mirrors the N-level rule: growth scales with the entity's own
        activation², decay is age-modulated like ``age_modulated_decay``
        on the N edges. Phase 2a doesn't use ``p.weight`` for anything
        downstream, but the dynamics are wired now so Phase 2b can
        gate P-P propagation on weight.
        """
        # Age factor mirrors :func:`age_modulated_decay` on N edges.
        age_factor = 1.0 / (1.0 + math.log(1.0 + self.system_age))
        to_dissolve: List[int] = []
        for p_id, p in self.p_entities.items():
            growth = self.eta * p.activation * p.activation
            decay = age_factor * self.p_weight_decay * p.weight
            p.weight = max(0.0, p.weight + growth - decay)
            if p.weight < self.p_viability_threshold:
                to_dissolve.append(p_id)
        for p_id in to_dissolve:
            p = self.p_entities[p_id]
            self.p_pairs_emerged.discard(p.components)
            del self.p_entities[p_id]

    # ---------- diagnostics ----------

    def sparsity(self, threshold: float = 0.1) -> float:
        """Fraction of neurons currently above ``threshold``
        activation. Sparsity target per H5 is roughly ~5 %."""
        return float((self.activations > threshold).mean())

    def total_weight(self) -> float:
        """Sum of all structural weights — proxy for accumulated
        learning."""
        return float(self.connectivity.W.sum())

    def p_count(self) -> int:
        """Number of live P entities at this moment."""
        return len(self.p_entities)
