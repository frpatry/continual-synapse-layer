"""Top-down P → N feedback (Phase 2c).

A live, active :class:`PEntity` represents the substrate's
"recognition" that two specific N have been pair-encoded. Phase 2c
lets that recognition speak back: an active ``P_ij`` contributes
``γ · P_ij.activation`` to both of its component N as an additive
boost on the next N propagation step.

This is the mechanism that turns the bottom-up emergence machinery of
Phase 2a / 2b into an *associative memory*: a partial cue (some of a
pattern's N clamped) propagates to the matching P entities, which
then feed back to the *missing* N and complete the pattern.

Design choice (Q4 = proportional): each active P contributes linearly
to its components; multiple P sharing a component sum their
contributions (no max, no normalisation, no winner-take-all at the
feedback). The proportional rule keeps the architecture trivially
inspectable and biologically plausible (excitatory synapse-like
addition).

Per Q5 = single substrate: feedback is delivered as an additive
component to the N's existing external_input — there is no separate
"feedback channel". From the N side, feedback is indistinguishable
from any other external drive.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .p_entity import PEntity


def compute_p_to_n_feedback(
    p_entities: Dict[int, PEntity],
    n_neurons: int,
    gamma: float = 0.1,
) -> np.ndarray:
    """Build the per-N feedback boost vector contributed by live P.

    Args:
        p_entities: live P entities keyed by id (matches
            :attr:`Substrate.p_entities`).
        n_neurons: total N count (output array size).
        gamma: feedback gain. ``gamma=0`` short-circuits to an
            all-zeros vector and is the canonical way to disable
            feedback at the function level.

    Returns:
        ``(n_neurons,)`` float32 vector. Entry ``i`` is
        ``γ · Σ_{P with i in components} P.activation``. Returns the
        zero vector when ``gamma == 0`` or ``p_entities`` is empty.
    """
    boost = np.zeros(n_neurons, dtype=np.float32)
    if gamma <= 0.0 or not p_entities:
        return boost

    for p in p_entities.values():
        if p.activation > 0.0:
            contribution = float(gamma) * float(p.activation)
            # Multiple P sharing a component sum their contributions
            # (proportional rule, no normalisation).
            boost[p.components[0]] += contribution
            boost[p.components[1]] += contribution
    return boost
