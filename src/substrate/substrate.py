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
from .plasticity import apply_plasticity, rho_age
from .s_dynamics import propagate_s_activations
from .s_entity import SEntity
from .s_pass_tracker import SPassTracker
from .s_to_p_feedback import compute_s_to_p_feedback


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
        # ---- Phase 6i S-level (third ontological tier) ----
        theta_s_emergence: float = 0.5,
        s_pass_boost: float = 0.1,
        s_pass_decay: float = 0.95,
        s_pass_theta_high: float = 0.1,
        s_pass_theta_low: float = 0.05,
        theta_s_growth: float = 0.3,
        alpha_p_to_s: float = 0.3,
        s_threshold: float = 0.2,
        s_sparsity_target: float = 0.20,
        s_min_active: int = 1,
        s_max_active: int = 3,
        s_background_noise_sigma: float = 0.01,
        gamma_s_to_p: float = 1.0,
        eta_s: float = 0.005,
        lambda_s_decay: float = 0.001,
        s_viability_threshold: float = 0.1,
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

        # ---------- Phase 6i S-level (third ontological tier) ----------
        self.theta_s_emergence: float = float(theta_s_emergence)
        self.theta_s_growth: float = float(theta_s_growth)
        self.alpha_p_to_s: float = float(alpha_p_to_s)
        self.s_threshold: float = float(s_threshold)
        self.s_sparsity_target: float = float(s_sparsity_target)
        self.s_min_active: int = int(s_min_active)
        self.s_max_active: int = int(s_max_active)
        self.s_background_noise_sigma: float = float(s_background_noise_sigma)
        self.gamma_s_to_p: float = float(gamma_s_to_p)
        self.eta_s: float = float(eta_s)
        self.lambda_s_decay: float = float(lambda_s_decay)
        self.s_viability_threshold: float = float(s_viability_threshold)

        # Live S entities, keyed by id. Emerges/dissolves dynamically.
        self.s_entities: Dict[int, SEntity] = {}
        self._next_s_id: int = 0
        # Canonical (min_p_id, max_p_id) pairs that have already produced
        # an S — prevents emergence of two S from the same P pair.
        self.s_pairs_emerged: Set[Tuple[int, int]] = set()
        # Emergence log for diagnostics.
        self.s_emergence_history: List[dict] = []

        # Tracker for P-P co-activation candidacy (for S emergence) and
        # P-S co-activation candidacy (for S growth).
        self.s_pass_tracker = SPassTracker(
            boost=s_pass_boost,
            decay=s_pass_decay,
            theta_high=s_pass_theta_high,
            theta_low=s_pass_theta_low,
        )

        # Independent RNG for S-level noise (reproducible + decoupled
        # from N background and P-level RNGs).
        self.s_rng: np.random.Generator = np.random.default_rng(self.seed + 3)

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

        # ---------- Phase 6i: S → P feedback (computed BEFORE P propagation) ----
        # Captures CURRENT (previous step's) S activations so S boost
        # reflects the schema state at start-of-step. Empty dict if no S.
        s_to_p_boost = compute_s_to_p_feedback(
            s_entities=self.s_entities,
            p_entities=self.p_entities,
            gamma_s_to_p=self.gamma_s_to_p,
        )

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
            # Apply S → P feedback post-threshold (cf. s_to_p_feedback
            # docstring: S boosts P that already cleared their own gate,
            # rather than injecting new activity from scratch). Clip to
            # the canonical [0, 1] activation range.
            for p_id, new_act in new_p_activations.items():
                boosted = new_act + s_to_p_boost.get(p_id, 0.0)
                self.p_entities[p_id].activation = float(
                    max(0.0, min(1.0, boosted))
                )

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
        dissolved_p_ids = self._decay_and_dissolve_p()

        # ---------- Phase 6i: S-level update ----------
        # 1. Update S pass tracker on current P + S activations.
        # 2. Detect new S emergences (P-P pair candidacy) + growth
        #    (P-S candidacy → add P to existing S).
        # 3. Propagate S activations from (now-updated) P contents.
        # 4. S plasticity (growth + age-modulated decay).
        # 5. Dissolve unviable S.
        # 6. Clean up s_pass_tracker for dissolved P + S.
        self.s_pass_tracker.update(self.p_entities, self.s_entities)

        s_emergence = self.s_pass_tracker.find_s_emergence_candidates(
            theta_s_emergence=self.theta_s_emergence,
            n_min_passes=self.n_min_passes,
            existing_s_pairs=self.s_pairs_emerged,
        )
        for (p_a, p_b) in s_emergence:
            self._emerge_s(p_a, p_b)

        s_growth = self.s_pass_tracker.find_s_growth_candidates(
            theta_growth=self.theta_s_growth,
            n_min_passes=self.n_min_passes,
        )
        for (p_id, s_id) in s_growth:
            if s_id in self.s_entities and p_id in self.p_entities:
                self.s_entities[s_id].add_member(p_id)

        if self.s_entities:
            new_s_activations = propagate_s_activations(
                s_entities=self.s_entities,
                p_entities=self.p_entities,
                alpha_p_to_s=self.alpha_p_to_s,
                s_threshold=self.s_threshold,
                s_sparsity_target=self.s_sparsity_target,
                s_min_active=self.s_min_active,
                s_max_active=self.s_max_active,
                s_background_noise_sigma=self.s_background_noise_sigma,
                rng=self.s_rng,
            )
            for s_id, new_act in new_s_activations.items():
                self.s_entities[s_id].activation = new_act

        self._update_s_plasticity()
        dissolved_s_ids = self._dissolve_unviable_s()

        if dissolved_p_ids:
            self.s_pass_tracker.cleanup_dissolved_p(dissolved_p_ids)
        if dissolved_s_ids:
            self.s_pass_tracker.cleanup_dissolved_s(dissolved_s_ids)

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

    def _decay_and_dissolve_p(self) -> List[int]:
        """Hebb-like growth + age-modulated decay on every P; dissolve
        those whose weight falls below the viability threshold.

        Mirrors the N-level rule: growth scales with the entity's own
        activation², decay is age-modulated like ``age_modulated_decay``
        on the N edges. Phase 2a doesn't use ``p.weight`` for anything
        downstream, but the dynamics are wired now so Phase 2b can
        gate P-P propagation on weight.

        Returns the list of dissolved P ids so Phase 6i's S-level
        bookkeeping can clean up dependent entries (S contents membership
        + s_pass_tracker pair/p_to_s candidacy).
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
            # Phase 6i: also remove this P from any S that contains it.
            for s in self.s_entities.values():
                s.remove_member(p_id)
            del self.p_entities[p_id]
        # Return the dissolved id list so the substrate's S-level
        # cleanup pass can prune dependent tracker entries.
        return to_dissolve

    # ---------- Phase 6i S-level helpers ----------

    def _emerge_s(self, p_a: int, p_b: int) -> None:
        """Create a fresh SEntity from a co-active P pair."""
        contents = {int(p_a), int(p_b)}
        # Derive initial activation as mean of the two seeding P entities.
        seed_act_sum = 0.0
        seed_n = 0
        for p_id in contents:
            if p_id in self.p_entities:
                seed_act_sum += float(self.p_entities[p_id].activation)
                seed_n += 1
        seed_activation = seed_act_sum / seed_n if seed_n > 0 else 0.0

        s = SEntity(
            id=self._next_s_id,
            contents=contents,
            activation=float(seed_activation),
            weight=1.0,
            age_at_emergence=self.system_age,
        )
        self.s_entities[self._next_s_id] = s
        canonical = (min(int(p_a), int(p_b)), max(int(p_a), int(p_b)))
        self.s_pairs_emerged.add(canonical)
        self.s_emergence_history.append(
            {
                "step": self.step_count,
                "s_id": s.id,
                "seed_pair": canonical,
                "system_age": self.system_age,
            }
        )
        self._next_s_id += 1

    def _update_s_plasticity(self) -> None:
        """Hebbian-like growth (eta_s · ρ · activation²) + age-modulated
        decay (ρ · λ_s · weight) on every S weight."""
        if not self.s_entities:
            return
        rho = rho_age(self.system_age, floor=self.rho_floor)
        for s in self.s_entities.values():
            growth = self.eta_s * rho * s.activation * s.activation
            decay = rho * self.lambda_s_decay * s.weight
            s.weight = max(0.0, s.weight + growth - decay)

    def _dissolve_unviable_s(self) -> List[int]:
        """Dissolve S whose weight has dropped below the viability
        threshold. Also dissolves S with empty contents (e.g. when all
        their member P were dissolved). Returns dissolved S ids."""
        to_dissolve: List[int] = []
        for s_id, s in self.s_entities.items():
            if s.weight < self.s_viability_threshold or not s.contents:
                to_dissolve.append(s_id)
        for s_id in to_dissolve:
            s = self.s_entities[s_id]
            # Remove every pair-permutation among its contents from the
            # emerged-pairs set so the same seed pair can re-emerge later.
            contents_list = sorted(int(x) for x in s.contents)
            for i, a in enumerate(contents_list):
                for b in contents_list[i + 1:]:
                    self.s_pairs_emerged.discard((a, b))
            del self.s_entities[s_id]
        return to_dissolve

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

    def s_count(self) -> int:
        """Number of live S entities at this moment (Phase 6i)."""
        return len(self.s_entities)

    def s_sparsity(self) -> float:
        """Fraction of live S entities with activation > 0 right now."""
        if not self.s_entities:
            return 0.0
        active = sum(1 for s in self.s_entities.values() if s.activation > 0.0)
        return active / len(self.s_entities)
