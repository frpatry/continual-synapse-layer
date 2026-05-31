"""S-level activation dynamics: bottom-up propagation + adaptive k-WTA.

Per Phase 6i design:

  S.raw_input  = α_p_to_s · mean(P.activation for P in S.contents)
  S.activation = k_WTA(soft_threshold(S.raw_input + small_noise), k_adaptive)

The k-WTA at S level is *adaptive* rather than a fixed fraction —
the live S pool is small (few per pattern) so a pure-fraction k-WTA
either keeps no S active or keeps them all. We use a sparsity target
of ~0.20 with explicit ``min_active=1`` and ``max_active=3`` bounds:
small enough to enforce selectivity, large enough that S has any
expressive role at all.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .dynamics import soft_threshold
from .s_entity import SEntity


def compute_s_k(
    n_s: int,
    sparsity: float = 0.20,
    min_active: int = 1,
    max_active: int = 3,
) -> int:
    """Adaptive k for the S-level k-WTA.

    Returns max(min_active, min(max_active, int(sparsity · n_s))),
    bounded to ``[0, n_s]``. When ``n_s == 0`` returns 0 (no k-WTA to do)."""
    if n_s <= 0:
        return 0
    k_naive = int(sparsity * n_s)
    k = max(int(min_active), min(int(max_active), k_naive))
    return min(k, n_s)


def s_winners_take_all(
    activations: np.ndarray, k: int,
) -> np.ndarray:
    """k-WTA on the S-level activation vector.

    Mirrors :func:`dynamics.k_winners_take_all` but takes ``k`` directly
    rather than a sparsity fraction (k is already computed via
    :func:`compute_s_k`). Returns an all-zeros array if ``k == 0``;
    returns a copy if ``k >= n``."""
    n = len(activations)
    if k <= 0:
        return np.zeros_like(activations)
    if k >= n:
        return activations.copy()

    sorted_vals = np.partition(activations, -k)
    cutoff = sorted_vals[-k]
    result = np.where(activations >= cutoff, activations, 0.0)
    return result.astype(activations.dtype)


def propagate_s_activations(
    s_entities: Dict[int, SEntity],
    p_entities: Dict[int, "PEntity"],  # noqa: F821 — avoid import cycle
    alpha_p_to_s: float = 0.3,
    s_threshold: float = 0.2,
    s_sparsity_target: float = 0.20,
    s_min_active: int = 1,
    s_max_active: int = 3,
    s_background_noise_sigma: float = 0.01,
    rng: Optional[np.random.Generator] = None,
) -> Dict[int, float]:
    """Synchronous one-step update of all S activations.

    Returns ``{s_id: new_activation}``; caller writes back to
    ``s_entities[s_id].activation`` so within-step S neighbours read
    the OLD state (consistent with how propagate_p_activations is used)."""
    if not s_entities:
        return {}
    if rng is None:
        rng = np.random.default_rng()

    s_ids = sorted(s_entities.keys())
    n_s = len(s_ids)

    # Raw input per S = α · mean(P.activation for P in S.contents)
    raw_inputs = np.zeros(n_s, dtype=np.float32)
    for idx, s_id in enumerate(s_ids):
        s = s_entities[s_id]
        if not s.contents:
            continue
        # Skip any dissolved-but-not-yet-cleaned P ids — they may
        # appear in contents transiently between the dissolution
        # call and the substrate's cleanup pass.
        live_member_acts = [
            p_entities[p_id].activation
            for p_id in s.contents
            if p_id in p_entities
        ]
        if not live_member_acts:
            continue
        raw_inputs[idx] = (
            alpha_p_to_s * sum(live_member_acts) / len(live_member_acts)
        )

    if s_background_noise_sigma > 0.0:
        noise = rng.normal(
            0.0, s_background_noise_sigma, size=n_s,
        ).astype(np.float32)
        raw_inputs = raw_inputs + noise

    thresholded = soft_threshold(raw_inputs, s_threshold).astype(np.float32)
    k = compute_s_k(
        n_s, s_sparsity_target, s_min_active, s_max_active,
    )
    final = s_winners_take_all(thresholded, k)

    return {s_id: float(final[idx]) for idx, s_id in enumerate(s_ids)}
