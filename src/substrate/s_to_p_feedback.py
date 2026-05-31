"""Top-down S → P feedback (Phase 6i).

When an S entity is active, it contributes
``γ_s_to_p · S.activation`` to every P entity in its ``contents``.
Multiple S sharing a P sum their contributions (same proportional
rule as :func:`compute_p_to_n_feedback`).

In the substrate's :meth:`step`, this boost is computed from the
*current* S activations (the substrate hasn't propagated S yet this
step) and is added to the P activations *after* the P propagation
step's soft-threshold + k-WTA — equivalent in spirit to how P → N
feedback augments the external_input pathway but post-applied at the
P level (we don't pass it into ``propagate_p_activations``'s
threshold-and-kWTA loop because S → P is meant to *boost* P that
already cleared their own gate, not to inject new activity from
scratch).
"""

from __future__ import annotations

from typing import Dict


def compute_s_to_p_feedback(
    s_entities: Dict,
    p_entities: Dict,
    gamma_s_to_p: float = 1.0,
) -> Dict[int, float]:
    """Build the per-P boost vector contributed by live S.

    Returns a dict keyed by ALL live P ids (zero-filled if no S
    boost). Callers can iterate ``p_entities`` and pick out the
    boost for each — also tolerant of S referencing dissolved P
    (just skipped)."""
    boost = {pid: 0.0 for pid in p_entities.keys()}
    if gamma_s_to_p <= 0.0 or not s_entities:
        return boost

    for s in s_entities.values():
        if s.activation <= 0.0:
            continue
        contribution = float(gamma_s_to_p) * float(s.activation)
        for p_id in s.contents:
            if p_id in boost:
                boost[p_id] += contribution
    return boost
