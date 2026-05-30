"""P-P Hebbian plasticity.

Mirrors the N-level rule (covariance Hebbian + age-modulated decay)
but operates on the dynamic :class:`PConnectivity` dict instead of a
fixed matrix:

* Existing connections track ``Δw = η · (a_i · a_j − ⟨a⟩²) − age_decay``.
* Non-existing connections are *created* only when the raw co-activation
  ``a_i · a_j`` exceeds ``min_coactivation_to_create``. This gate
  prevents noise-floor co-activations from spawning a long tail of
  near-zero P-P edges that would have to be tracked and decayed every
  step.

Per Q2 = A, the same mechanism that grows N-N edges grows P-P edges —
emergence is hierarchical, not parameterised separately.

Covariance baseline note:
    With ``len(p_entities) == 2`` the covariance baseline
    ``mean²`` exactly cancels ``a_i · a_j`` and Δ = 0, so no connection
    ever forms in a 2-P pool. Tests for creation use ≥ 3 P with at least
    one quiet entity to pull the mean down. In the live substrate this
    is not an issue: the live P pool is larger and there are always
    quiescent P entities pulling the mean below the co-active pair.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

from .p_connectivity import PConnectivity
from .p_entity import PEntity


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

    age_factor = 1.0 / (1.0 + math.log(1.0 + system_age))

    for i in range(n_p):
        a_i = float(activations[i])
        for j in range(i + 1, n_p):
            a_j = float(activations[j])
            id_a = p_ids[i]
            id_b = p_ids[j]

            coact_signal = a_i * a_j - mean_sq
            delta = eta_pp * coact_signal

            current_w = p_connectivity.get_weight(id_a, id_b)
            if current_w > 0.0:
                # Existing connection: Hebbian + age-modulated decay.
                decay = age_factor * lambda_pp_decay * current_w
                p_connectivity.update_weight(id_a, id_b, delta - decay)
            else:
                # Creation gate: require BOTH a positive Hebbian delta
                # (covariance) AND a strong raw co-activation (so we
                # don't insert edges off the noise floor).
                if delta > 0.0 and a_i * a_j > min_coactivation_to_create:
                    p_connectivity.update_weight(id_a, id_b, delta)
