"""Reward-as-confidence: per-sample informativeness signal.

The Hebbian update in :class:`SynapseAugmentedMLP` has historically
used a single scalar ``R`` (defaulting to ``1.0``) for every sample
in a batch. Every sample contributes equally to synapse state
updates, whether it's a routine correct prediction or a confidently-
wrong edge case.

This module produces a per-sample replacement: a vector ``R[i]`` of
shape ``(B,)`` capturing how informative each sample is. The
intuition: an uncertain, wrong, or miscalibrated sample carries
more signal about what the model still needs to learn than a
confidently-correct one does. Weighting the Hebbian update by
``R[i]`` lets the synapse layer learn proportionally more from the
samples that matter.

The formula combines three terms (per sample ``i``, with class
probabilities ``p = softmax(logits)`` and ``K`` classes):

- ``error_i = 1 - p(y_i | x_i)`` — straight error probability,
  ``∈ [0, 1]``.
- ``uncertainty_i = -Σ_c p_c log p_c  /  log K`` — normalised
  entropy, ``∈ [0, 1]``.
- ``calibration_i = |normalised_max_prob_i - correct_i|`` —
  absolute mismatch between (rescaled) confidence and 0/1
  correctness, ``∈ [0, 1]``.

These are combined as:

    R_i = (1 - γ) * [α * error_i + (1 - α) * uncertainty_i]
          + γ * calibration_i

The ``α`` weight is **developmental** (see
:func:`developmental_alpha`): low when the system is "young" (so
the reward mostly tracks uncertainty — useful when error is noisy
because the classifier hasn't stabilised), rising to a capped
maximum as it matures (so the reward mostly tracks error — useful
once the classifier is reliable enough that being wrong actually
means something). The cap (default ``0.85``) keeps some weight on
uncertainty even at full maturity, preventing late-stage
stagnation where the system stops noticing distribution shift
because its confident predictions are usually right.

The output ``R`` is normalised within the batch by
:func:`normalize_reward_batch` so its mean equals ``1`` — that
preserves the existing scalar contract with the synapse layer
(the average per-batch update magnitude stays unchanged) while
making the variance across samples the informative signal.

This module is **standalone**: it returns tensors. Integration
into ``apply_hebbian_update`` lands in a follow-up commit.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-8


def compute_reward_signal(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    gamma: float = 0.3,
) -> torch.Tensor:
    """Compute the per-sample informativeness reward.

    Args:
        logits: Model output logits, shape ``(B, K)``. Pre-softmax.
        targets: True class indices, shape ``(B,)``, dtype long.
        alpha: Weight on error vs uncertainty in the non-calibration
            term. ``0.2`` early in training, ``0.85`` mature. Pass
            the value returned by :func:`developmental_alpha` when
            you want the developmental schedule.
        gamma: Weight on the calibration term. ``0.3`` by default
            (so the non-calibration portion gets ``0.7``). Pass
            ``0.0`` to ablate the calibration component.

    Returns:
        ``R`` of shape ``(B,)``. Values are approximately in
        ``[0, 1]`` (each component is in ``[0, 1]`` and they're
        combined with non-negative weights summing to one), with
        higher values indicating a more informative sample.
    """
    if logits.ndim != 2:
        raise ValueError(
            f"logits must be 2-D (B, K), got shape {tuple(logits.shape)}"
        )
    if targets.ndim != 1:
        raise ValueError(
            f"targets must be 1-D (B,), got shape {tuple(targets.shape)}"
        )
    if logits.shape[0] != targets.shape[0]:
        raise ValueError(
            f"logits batch {logits.shape[0]} disagrees with targets "
            f"batch {targets.shape[0]}"
        )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")

    B, K = logits.shape
    probs = F.softmax(logits, dim=-1)

    # Error: 1 - p(true class). ∈ [0, 1].
    p_true = probs.gather(1, targets.unsqueeze(1).to(torch.long)).squeeze(1)
    error = 1.0 - p_true

    # Uncertainty: normalised entropy. ∈ [0, 1]; equals 1 at the
    # uniform distribution and 0 at any one-hot.
    entropy = -(probs * (probs + EPS).log()).sum(dim=-1)
    log_K = torch.log(
        torch.tensor(float(K), dtype=entropy.dtype, device=entropy.device)
    )
    uncertainty = entropy / log_K

    # Calibration: |confidence - correctness|. Confidence is the
    # max class probability rescaled to ``[0, 1]`` (so a uniform
    # prediction maps to 0 and a perfectly confident one to 1);
    # correctness is the 0/1 indicator that the argmax matches the
    # target. The absolute difference is in ``[0, 1]``.
    max_prob = probs.max(dim=-1).values
    normalised_confidence = (max_prob - 1.0 / K) / (1.0 - 1.0 / K + EPS)
    pred = probs.argmax(dim=-1)
    correctness = (pred == targets).to(probs.dtype)
    calibration = (normalised_confidence - correctness).abs()

    non_calib = (1.0 - gamma) * (alpha * error + (1.0 - alpha) * uncertainty)
    R = non_calib + gamma * calibration
    return R


def normalize_reward_batch(
    R: torch.Tensor,
    floor: float = 0.01,
) -> torch.Tensor:
    """Normalise per-batch so ``mean(R) == 1``.

    Preserves the existing scalar contract with the synapse layer
    (per-batch average update magnitude unchanged); the
    informative content moves into the variance.

    A floor on the mean prevents amplifying noise when the entire
    batch is uninformative: if every sample is routine (low
    error, low uncertainty, well-calibrated) the per-sample
    differences are mostly noise, and dividing by a tiny mean
    would explode that noise into large multipliers. In that
    regime we just return a uniform ``R = 1`` so the update
    behaves like the historical constant-R path.

    Args:
        R: Per-sample reward, shape ``(B,)``. Must be non-negative
            (the function does not check, but
            :func:`compute_reward_signal`'s output satisfies this
            by construction).
        floor: Minimum mean below which the function returns a
            uniform vector. ``0.01`` by default.

    Returns:
        ``R_norm`` of shape ``(B,)`` with ``mean(R_norm) == 1``
        when ``R.mean() >= floor``, otherwise a tensor of ones.
    """
    if R.ndim != 1:
        raise ValueError(
            f"R must be 1-D (B,), got shape {tuple(R.shape)}"
        )
    if R.numel() == 0:
        return R
    mean = R.mean()
    if float(mean) < floor:
        return torch.ones_like(R)
    return R / mean


def developmental_alpha(
    maturity: float,
    alpha_min: float = 0.2,
    alpha_max: float = 0.85,
) -> float:
    """Map a developmental maturity scalar to the reward's ``α``.

    Args:
        maturity: Scalar in ``[0, 1]``. Values outside that range
            are clamped. ``0`` represents a brand-new system that
            has consolidated nothing; ``1`` represents a system
            that has hit its consolidation target.
        alpha_min: ``α`` at ``maturity = 0``. Default ``0.2``
            means a young system weights uncertainty 4× more than
            error.
        alpha_max: ``α`` at ``maturity = 1``. Default ``0.85`` —
            note this is **below** ``1.0`` on purpose: even a
            fully mature system keeps 15% weight on uncertainty
            so it doesn't stop responding to distribution shift
            once its confident predictions are usually correct.

    Returns:
        ``α`` to pass to :func:`compute_reward_signal`.
    """
    if not 0.0 <= alpha_min <= 1.0:
        raise ValueError(f"alpha_min must be in [0, 1], got {alpha_min}")
    if not 0.0 <= alpha_max <= 1.0:
        raise ValueError(f"alpha_max must be in [0, 1], got {alpha_max}")
    if alpha_max < alpha_min:
        raise ValueError(
            f"alpha_max ({alpha_max}) must be >= alpha_min ({alpha_min})"
        )
    clamped = max(0.0, min(1.0, float(maturity)))
    return alpha_min + (alpha_max - alpha_min) * clamped
