"""Substrate — unified orchestrator for Phase 1 + 2a + 2b dynamics.

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

Phase 2b scope (P-level dynamics + P-P connections):
- Each step, after N-level work and emergence, every live P entity
  gets its own activation update via :func:`propagate_p_activations`
  (soft-threshold + k-WTA on P-P input + N components + small noise).
- P-P connections form / strengthen / decay via
  :func:`apply_pp_plasticity` (covariance Hebbian with a creation
  gate so noise-floor co-activations don't seed edges).
- When a P dissolves, every P-P connection touching it is removed.
- Still no P → N feedback (that arrives in Phase 2c).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .connectivity import ConnectivityMatrix
from .dynamics import DEFAULT_SPARSITY_TARGET, GlobalBackground, propagate_activation
from .neuron import N
from .p_connectivity import PConnectivity
from .p_dynamics import propagate_p_activations
from .p_entity import PEntity
from .p_plasticity import apply_pp_plasticity
from .p_to_n_feedback import compute_p_to_n_feedback
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
        # ---- Phase 2b P-level dynamics + P-P plasticity ----
        alpha_n_to_p: float = 0.3,
        p_threshold: float = 0.3,
        p_sparsity_target: float = DEFAULT_SPARSITY_TARGET,
        p_background_noise_sigma: float = 0.01,
        eta_pp: float = 0.005,
        lambda_pp_decay: float = 0.001,
        min_coactivation_to_create_pp: float = 0.1,
        # ---- Phase 2c top-down P→N feedback ----
        gamma_p_to_n: float = 0.1,
        enable_feedback_p_to_n: bool = True,
        # ---- Phase 3 critical periods ----
        starting_age: float = 0.0,
        # ---- Phase 6a plasticity floor ----
        rho_floor: float = 0.7,
        # ---- Phase 6f fresh-pattern protection ----
        k_protect: int = 5000,
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
        # ``system_age`` is what drives age-modulated decay everywhere in
        # the substrate. Defaulting it to 0.0 is the canonical "young"
        # substrate. ``starting_age`` lets experiments instantiate a
        # substrate as if it had already aged — used by Phase 3 to test
        # the critical-period prediction P4 (does age-modulated decay
        # produce the biological fast-young / slow-mature distinction).
        self.system_age: float = float(starting_age)
        self.step_count: int = 0

        # Phase 6a: floor on the age-modulation factor so plasticity
        # never collapses to zero, even at very high age. The Phase 6a
        # default 0.3 preserved some critical-period asymmetry in the
        # substrate's first ~10 steps; Phase 6f raised the default to
        # 0.7 to fully eliminate artificial late-learning suppression.
        # Stated tradeoff: Phase 3.1's biological critical-periods
        # demonstration no longer shows young-vs-mature asymmetry —
        # acceptable for a Bio-Inspired AI substrate whose goal is
        # learning + retention rather than developmental fidelity.
        # Set rho_floor=0.0 to reproduce pre-Phase-6a behaviour.
        self.rho_floor: float = float(rho_floor)

        # Phase 6f: fresh-pattern protection window. When a P emerges,
        # its ``protected_until`` is set to ``step_count + k_protect``.
        # Inside that window, the P entity is exempt from dissolution
        # even if its weight drops below ``p_viability_threshold``.
        # Bio analog: protein-synthesis-dependent LTP late phase +
        # synaptic tagging in mammalian neurons — newly-potentiated
        # synapses are actively maintained against competition for
        # ~30 min – 1 h post-induction. Set ``k_protect=0`` to disable
        # the protection mechanism.
        self.k_protect: int = int(k_protect)

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

        # ---------- Phase 2b P-level dynamics + P-P plasticity ----------
        self.alpha_n_to_p: float = float(alpha_n_to_p)
        self.p_threshold: float = float(p_threshold)
        self.p_sparsity_target: float = float(p_sparsity_target)
        self.p_background_noise_sigma: float = float(p_background_noise_sigma)
        self.eta_pp: float = float(eta_pp)
        self.lambda_pp_decay: float = float(lambda_pp_decay)
        self.min_coactivation_to_create_pp: float = float(
            min_coactivation_to_create_pp,
        )

        # Dynamic, dict-backed P-P connection store. Starts empty;
        # grows only when co-active P entities clear the creation gate.
        self.p_connectivity = PConnectivity()

        # Independent RNG so P-level noise is reproducible *and*
        # decoupled from the GlobalBackground draw sequence.
        self.p_rng: np.random.Generator = np.random.default_rng(self.seed + 2)

        # ---------- Phase 2c top-down feedback ----------
        self.gamma_p_to_n: float = float(gamma_p_to_n)
        self.enable_feedback_p_to_n: bool = bool(enable_feedback_p_to_n)

    # ---------- core update ----------

    def step(self, external_input: Optional[np.ndarray] = None) -> np.ndarray:
        """One synchronous timestep.

        Order:
          0. (Phase 2c) Compute P → N feedback from the *current* P
             activations and fold it into ``external_input`` before
             N propagation. Feedback is computed BEFORE the N
             propagation step so cause and effect are correctly
             ordered: P at time t boosts N at time t+1.
          1. Generate background drive.
          2. Propagate N activations (sees external_input + feedback).
          3. Apply N-level plasticity (H3 — every step).
          4. Update P-level pass tracker on the new N activations.
          5. Check for new P emergences (Phase 2a).
          6. Propagate P activations using P-P channel + N components
             (Phase 2b) — synchronous update, neighbours read OLD state.
          7. Apply P-P Hebbian plasticity on the freshly-updated P
             activations (Phase 2b).
          8. Decay P weights, dissolve non-viable (cleanup removes
             every P-P connection touching a dissolved P).
          9. Advance age + step counter.

        Returns a copy of the new N activations so the caller can
        store snapshots without seeing later steps' mutations.
        """
        # ---------- Phase 2c: P → N feedback ----------
        # Compute from CURRENT (previous step's) P activations so the
        # boost reaches N before propagation. ``compute_p_to_n_feedback``
        # short-circuits to zeros for gamma=0 or empty p_entities, so
        # there's no penalty pre-emergence.
        if self.enable_feedback_p_to_n and self.p_entities:
            feedback = compute_p_to_n_feedback(
                self.p_entities, self.n_neurons, gamma=self.gamma_p_to_n,
            )
        else:
            feedback = None

        if external_input is not None and feedback is not None:
            effective_input = external_input + feedback
        elif external_input is not None:
            effective_input = external_input
        elif feedback is not None and feedback.any():
            effective_input = feedback
        else:
            effective_input = None

        bg = self.background.step(self.n_neurons)
        new_acts = propagate_activation(
            current_activations=self.activations,
            connectivity_W=self.connectivity.W,
            neuron_weights=self.neuron_weights,
            background=bg,
            external_input=effective_input,
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
            rho_floor=self.rho_floor,
        )

        # ---------- Phase 2a: emergence detection ----------
        self.pass_tracker.update(self.activations)
        candidates = self.pass_tracker.find_emergence_candidates(
            connectivity_W=self.connectivity.W,
            theta_emergence=self.theta_emergence,
            n_min_passes=self.n_min_passes,
            existing_p_pairs=self.p_pairs_emerged,
        )
        for (i, j) in candidates:
            self._emerge_p(i, j)

        # ---------- Phase 2b: P-level dynamics ----------
        # Synchronous update: each P sees its neighbours' OLD activations.
        if self.p_entities:
            new_p_activations = propagate_p_activations(
                p_entities=self.p_entities,
                p_connectivity=self.p_connectivity,
                n_activations=self.activations,
                alpha_n_to_p=self.alpha_n_to_p,
                p_threshold=self.p_threshold,
                p_sparsity_target=self.p_sparsity_target,
                p_background_noise_sigma=self.p_background_noise_sigma,
                rng=self.p_rng,
            )
            for p_id, new_act in new_p_activations.items():
                self.p_entities[p_id].activation = new_act

        # ---------- Phase 2b: P-P plasticity (Hebbian + creation gate) ----
        apply_pp_plasticity(
            p_entities=self.p_entities,
            p_connectivity=self.p_connectivity,
            system_age=self.system_age,
            eta_pp=self.eta_pp,
            lambda_pp_decay=self.lambda_pp_decay,
            min_coactivation_to_create=self.min_coactivation_to_create_pp,
            rho_floor=self.rho_floor,
        )

        # ---------- P weight decay + dissolution ----------
        self._decay_and_dissolve_p()

        self.system_age += 1.0
        self.step_count += 1
        return new_acts.copy()

    # ---------- P-level helpers ----------

    def _emerge_p(self, i: int, j: int) -> None:
        """Create a fresh PEntity for the canonical pair (i, j).

        Phase 6f: the new P starts with ``protected_until = step_count
        + k_protect``, giving it a guaranteed window during which
        dissolution is suppressed even if its weight drops below
        viability. This lets fresh attractors stabilize through
        subsequent consolidation cycles instead of being out-competed
        by older, more-established attractors.
        """
        p = PEntity(
            id=self._next_p_id,
            components=(int(i), int(j)),
            activation=float(
                (self.activations[i] + self.activations[j]) / 2.0,
            ),
            weight=1.0,
            age_at_emergence=self.system_age,
            protected_until=self.step_count + self.k_protect,
        )
        self.p_entities[self._next_p_id] = p
        self.p_pairs_emerged.add(p.components)
        self.p_emergence_history.append(
            {
                "step": self.step_count,
                "p_id": p.id,
                "components": p.components,
                "system_age": self.system_age,
                "protected_until": p.protected_until,
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
            # Phase 6f: protected P entities still see weight decay
            # applied above, but the dissolution check is skipped
            # until protection ends. Their weight may temporarily dip
            # below ``p_viability_threshold`` during the window.
            if p.is_protected(self.step_count):
                continue
            if p.weight < self.p_viability_threshold:
                to_dissolve.append(p_id)
        for p_id in to_dissolve:
            p = self.p_entities[p_id]
            self.p_pairs_emerged.discard(p.components)
            # Phase 2b: cleanup every P-P connection touching this id
            # so the dynamic connectivity doesn't leak references.
            self.p_connectivity.remove_entity(p_id)
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

    def p_connection_count(self) -> int:
        """Number of live P-P connections (Phase 2b)."""
        return self.p_connectivity.connection_count()

    def p_sparsity(self) -> float:
        """Fraction of live P entities with activation > 0 right now."""
        if not self.p_entities:
            return 0.0
        active = sum(1 for p in self.p_entities.values() if p.activation > 0.0)
        return active / len(self.p_entities)
