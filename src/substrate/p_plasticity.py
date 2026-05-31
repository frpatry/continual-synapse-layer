"""P-P Hebbian plasticity (Phase 3.1: symmetric ρ(age) on growth + decay).

Mirrors the N-level rule (covariance Hebbian + age-modulated decay,
both scaled by ρ(age)) but operates on the dynamic
:class:`PConnectivity` dict instead of a fixed matrix:

* Existing connections track
  ``Δw = ρ · η · (a_i · a_j − ⟨a⟩²) − ρ · λ · w``.
* Non-existing connections are *created* only when the raw co-activation
  ``a_i · a_j`` exceeds ``min_coactivation_to_create``. This gate
  prevents noise-floor co-activations from spawning a long tail of
  near-zero P-P edges that would have to be tracked and decayed every
  step. The creation gate uses RAW co-activation (independent of ρ)
  so the gate threshold has consistent meaning across ages.

Per Q2 = A, the same mechanism that grows N-N edges grows P-P edges —
emergence is hierarchical, not parameterised separately. Per the
Phase 3.1 correction (THEORY.md §3.2), ρ(age) applies symmetrically
at every plasticity scale.

Covariance baseline note:
    With ``len(p_entities) == 2`` the covariance baseline
    ``mean²`` exactly cancels ``a_i · a_j`` and Δ = 0, so no connection
    ever forms in a 2-P pool. Tests for creation use ≥ 3 P with at least
    one quiet entity to pull the mean down. In the live substrate this
    is not an issue: the live P pool is larger and there are always
    quiescent P entities pulling the mean below the co-active pair.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .p_connectivity import PConnectivity
from .p_entity import PEntity
from .plasticity import rho_age


def apply_pp_plasticity(
    p_entities: Dict[int, PEntity],
    p_connectivity: PConnectivity,
    system_age: float,
    eta_pp: float = 0.005,
    lambda_pp_decay: float = 0.001,
    min_coactivation_to_create: float = 0.1,
) -> None:
    """One plasticity step on every (a, b) P pair (in-place on ``p_connectivity``).

    For each ordered pair ``(a, b)`` with ``a.id < b.id``:

    * Compute covariance Hebbian delta:
      ``δ = η · (a.act · b.act − ⟨act⟩²)``
    * If a connection already exists: apply ``δ − age_decay(W)``.
    * If not: only insert when raw co-activation ``a.act · b.act``
      exceeds ``min_coactivation_to_create`` AND ``δ > 0``.

    The plasticity rule is symmetric, which matches the symmetric
    storage in :class:`PConnectivity`.
    """
    n_p = len(p_entities)
    if n_p < 2:
        return

    p_ids = sorted(p_entities.keys())
    activations = np.array(
        [p_entities[pid].activation for pid in p_ids],
        dtype=np.float32,
    )
    mean_act = float(activations.mean())
    mean_sq = mean_act * mean_act

    # Phase 3.1: symmetric age modulation. Same ρ scales BOTH the
    # Hebbian (growth) and decay terms. Equilibrium W stays the same;
    # the timescale to reach it slows with age.
    rho = rho_age(system_age)
    eta_effective = eta_pp * rho

    for i in range(n_p):
        a_i = float(activations[i])
        for j in range(i + 1, n_p):
            a_j = float(activations[j])
            id_a = p_ids[i]
            id_b = p_ids[j]

            coact_signal = a_i * a_j - mean_sq
            delta = eta_effective * coact_signal

            current_w = p_connectivity.get_weight(id_a, id_b)
            if current_w > 0.0:
                # Existing connection: Hebbian + age-modulated decay,
                # both scaled by ρ.
                decay = rho * lambda_pp_decay * current_w
                p_connectivity.update_weight(id_a, id_b, delta - decay)
            else:
                # Creation gate: require BOTH a positive Hebbian delta
                # (covariance) AND a strong RAW co-activation (so we
                # don't insert edges off the noise floor). The gate
                # uses raw a_i · a_j (independent of ρ) so the
                # threshold has consistent meaning across ages.
                if delta > 0.0 and a_i * a_j > min_coactivation_to_create:
                    p_connectivity.update_weight(id_a, id_b, delta)
