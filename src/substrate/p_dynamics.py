"""P-level activation propagation + structural sparsity enforcement.

Phase 2b: every live P entity gets its own activation update each step,
not just a derived ``(N_i + N_j) / 2`` value. The update mirrors the
N-level recipe so the substrate is self-similar across levels — only
the source of input differs:

    P.activation(t+1) = soft_threshold(
            Σ_{Q ∈ neighbors(P)} W_PP[P, Q] · Q.activation(t)
            + alpha_n_to_p · (N_i.activation + N_j.activation) / 2
            + small_p_background_noise
        )

then k-WTA is applied across the whole P pool so only the top
``p_sparsity_target`` fraction survives. This is the recursive H5
enforcement at the P level (D2 + §3.6 at a higher abstraction).

This module is pure-function w.r.t. ``p_entities`` — it returns a
``{p_id: new_activation}`` dict and leaves the caller responsible for
writing back the values, so the activation update is synchronous
(each P sees the *old* activations of its neighbors, not partially
updated state).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .dynamics import DEFAULT_SPARSITY_TARGET, k_winners_take_all, soft_threshold
from .p_connectivity import PConnectivity
from .p_entity import PEntity


def compute_p_input(
    p: PEntity,
    p_connectivity: PConnectivity,
    p_entities: Dict[int, PEntity],
    n_activations: np.ndarray,
    alpha_n_to_p: float = 0.3,
) -> float:
    """Raw scalar input to one P (before soft-threshold + k-WTA).

    Two additive contributions:

    1. P-P channel: ``Σ W_PP[p, q] · q.activation`` over every live
       neighbour ``q`` listed in ``p_connectivity``. Neighbours that
       were dropped from ``p_entities`` (already dissolved) are skipped.
    2. N channel: ``alpha · (N_i + N_j) / 2`` — the mean activation of
       p's two component N, scaled by ``alpha_n_to_p`` (Phase 2b
       default 0.3, weak coupling so P doesn't simply mirror N).

    No background here — that's added by :func:`propagate_p_activations`
    so the per-call randomness is centralised.
    """
    pp_input = 0.0
    for neighbor_id, w in p_connectivity.neighbors_of(p.id).items():
        neighbour = p_entities.get(neighbor_id)
        if neighbour is not None:
            pp_input += w * neighbour.activation

    n_i_act = float(n_activations[p.components[0]])
    n_j_act = float(n_activations[p.components[1]])
    n_input = alpha_n_to_p * (n_i_act + n_j_act) / 2.0

    return float(pp_input + n_input)


def propagate_p_activations(
    p_entities: Dict[int, PEntity],
    p_connectivity: PConnectivity,
    n_activations: np.ndarray,
    alpha_n_to_p: float = 0.3,
    p_threshold: float = 0.3,
    p_sparsity_target: float = DEFAULT_SPARSITY_TARGET,
    p_background_noise_sigma: float = 0.01,
    rng: Optional[np.random.Generator] = None,
) -> Dict[int, float]:
    """Synchronous one-step update of all P activations.

    Args:
        p_entities: live P entities keyed by id.
        p_connectivity: current P-P weights.
        n_activations: ``(n_neurons,)`` N activations *after* their own
            propagation step (so P sees the same N state the rest of
            the substrate sees).
        alpha_n_to_p: N→P channel gain.
        p_threshold: soft-threshold for P.
        p_sparsity_target: fraction of the P pool kept active after k-WTA.
        p_background_noise_sigma: per-P Gaussian noise std deviation —
            small so it doesn't drown P-P signal but breaks ties in
            the k-WTA.
        rng: numpy ``Generator`` for reproducibility. If ``None``, a
            fresh default RNG is used (avoid in production code paths —
            the substrate threads its own seeded RNG through here).

    Returns:
        ``{p_id: new_activation}`` — the caller writes these back to
        ``p_entities[p_id].activation`` *after* this function returns,
        so within-step neighbour reads see the OLD state.
    """
    if not p_entities:
        return {}

    if rng is None:
        rng = np.random.default_rng()

    # Stable ordering for deterministic k-WTA / RNG draw.
    p_ids = sorted(p_entities.keys())
    n_p = len(p_ids)

    raw_inputs = np.zeros(n_p, dtype=np.float32)
    for idx, p_id in enumerate(p_ids):
        raw_inputs[idx] = compute_p_input(
            p=p_entities[p_id],
            p_connectivity=p_connectivity,
            p_entities=p_entities,
            n_activations=n_activations,
            alpha_n_to_p=alpha_n_to_p,
        )

    if p_background_noise_sigma > 0.0:
        noise = rng.normal(
            0.0, p_background_noise_sigma, size=n_p,
        ).astype(np.float32)
        raw_inputs = raw_inputs + noise

    thresholded = soft_threshold(raw_inputs, p_threshold).astype(np.float32)
    final = k_winners_take_all(thresholded, sparsity_target=p_sparsity_target)

    return {p_id: float(final[idx]) for idx, p_id in enumerate(p_ids)}
