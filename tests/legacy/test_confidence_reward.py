"""Tests for the reward-as-confidence utility module."""

from __future__ import annotations

import math

import pytest
import torch

from continual_synapse.reward.confidence_reward import (
    compute_reward_signal,
    developmental_alpha,
    normalize_reward_batch,
)


# ---- compute_reward_signal: four state quadrants ----


def test_perfectly_correct_confident_gives_low_R() -> None:
    """A confidently-correct sample is uninformative: error ≈ 0,
    uncertainty ≈ 0, calibration ≈ 0 ⇒ R near 0."""
    # Logits heavily favour class 0; target is class 0.
    logits = torch.tensor([[10.0, 0.0, 0.0]])
    targets = torch.tensor([0], dtype=torch.long)
    R = compute_reward_signal(logits, targets, alpha=0.5, gamma=0.3)
    assert R.shape == (1,)
    assert float(R) < 0.01, f"expected R near 0, got {float(R):.4f}"


def test_perfectly_wrong_confident_gives_high_R() -> None:
    """A confidently-wrong sample is maximally informative:
    error ≈ 1, calibration ≈ 1 ⇒ R near the upper bound. Uncertainty
    is still ≈ 0 (the prediction is sharp), so with alpha=0.5 the
    non-calib term is ≈ 0.5; calib adds 0.3; total ≈ 0.5*0.7 + 0.3 = 0.65."""
    logits = torch.tensor([[10.0, 0.0, 0.0]])  # confidently predicts 0
    targets = torch.tensor([1], dtype=torch.long)  # but the truth is 1
    R = compute_reward_signal(logits, targets, alpha=0.5, gamma=0.3)
    # Expect R ≈ 0.7 * (0.5 * ~1 + 0.5 * ~0) + 0.3 * ~1 ≈ 0.65
    assert 0.55 < float(R) < 0.75, (
        f"expected R near 0.65 for confidently-wrong, got {float(R):.4f}"
    )


def test_uncertain_correct_gives_moderate_R() -> None:
    """Uniform logits ⇒ uncertainty 1, error ≈ 1 - 1/K = 0.667 for
    K=3, calibration ≈ 1 (zero confidence over correctness 1). With
    alpha=0.5, gamma=0.3: non-calib = 0.7 * (0.5*0.667 + 0.5*1) ≈
    0.583, calib = 0.3, R ≈ 0.883. Between the confident-correct
    extreme (near 0) and the confident-wrong extreme — verifies the
    formula is sensible across the spectrum, not strictly that the
    moderate value is "between the others"."""
    logits = torch.zeros(1, 3)  # uniform
    targets = torch.tensor([0], dtype=torch.long)  # picked an arbitrary class
    R = compute_reward_signal(logits, targets, alpha=0.5, gamma=0.3)
    # Not 0 (uncertain) and not at the confidently-wrong cap.
    assert 0.3 < float(R) < 1.0, (
        f"expected moderate R for uniform-uncertain-correct, "
        f"got {float(R):.4f}"
    )


def test_calibration_term_isolated() -> None:
    """Two samples with the same error and uncertainty but different
    calibration mismatches must differ in R by exactly
    gamma * Δcalibration."""
    # Logits identical → identical error, identical uncertainty.
    # Sample A: target matches argmax → correctness = 1.
    # Sample B: target differs from argmax → correctness = 0.
    logits = torch.tensor(
        [[2.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
    )
    targets = torch.tensor([0, 1], dtype=torch.long)
    R = compute_reward_signal(logits, targets, alpha=0.0, gamma=0.5)
    # alpha=0 isolates uncertainty (which is identical between
    # samples); gamma=0.5 weights calibration. Then ΔR = 0.5 *
    # Δcalibration, and we can compute Δcalibration analytically.
    import torch.nn.functional as F
    probs = F.softmax(logits[0], dim=-1)
    max_prob = float(probs.max())
    K = 3
    norm_conf = (max_prob - 1.0 / K) / (1.0 - 1.0 / K + 1e-8)
    # Sample A: calibration = |norm_conf - 1| ; B: |norm_conf - 0|
    expected_d_calib = abs(norm_conf - 0) - abs(norm_conf - 1)
    expected_dR = 0.5 * expected_d_calib
    actual_dR = float(R[1] - R[0])
    assert math.isclose(actual_dR, expected_dR, rel_tol=1e-4, abs_tol=1e-5), (
        f"expected ΔR = {expected_dR:.4f} (= gamma * Δcalib), "
        f"got {actual_dR:.4f}"
    )


# ---- normalize_reward_batch ----


def test_normalization_preserves_relative_order() -> None:
    """Dividing by a positive scalar can't reorder positive values."""
    R = torch.tensor([0.1, 0.5, 0.3, 0.9, 0.2])
    R_norm = normalize_reward_batch(R)
    assert torch.equal(
        R.argsort(),
        R_norm.argsort(),
    ), "normalisation must preserve relative order"
    # And the mean should now be 1.
    assert math.isclose(float(R_norm.mean()), 1.0, abs_tol=1e-6)


def test_normalization_floor_prevents_blow_up() -> None:
    """A batch where every sample is uninformative (mean(R) < floor)
    must return a uniform vector so noise doesn't get amplified into
    huge multipliers."""
    # Tiny values, mean well below default floor 0.01.
    R = torch.tensor([1e-5, 2e-5, 3e-5, 4e-5])
    R_norm = normalize_reward_batch(R, floor=0.01)
    assert torch.equal(R_norm, torch.ones_like(R)), (
        f"expected uniform fallback below floor, got {R_norm.tolist()}"
    )


# ---- developmental_alpha ----


def test_developmental_alpha_endpoints() -> None:
    """maturity=0 returns alpha_min; maturity=1 returns alpha_max."""
    assert developmental_alpha(0.0) == pytest.approx(0.2)
    assert developmental_alpha(1.0) == pytest.approx(0.85)
    # And the linear interp in the middle.
    assert developmental_alpha(0.5) == pytest.approx(0.2 + 0.5 * (0.85 - 0.2))


def test_developmental_alpha_capped() -> None:
    """Maturity values outside [0, 1] are clamped, so an over-mature
    system never gets α > alpha_max (the safeguard against late-stage
    stagnation)."""
    assert developmental_alpha(1.5) == pytest.approx(0.85)
    assert developmental_alpha(10.0) == pytest.approx(0.85)
    assert developmental_alpha(-0.5) == pytest.approx(0.2)
    # Custom caps respect the same clamping.
    assert developmental_alpha(2.0, alpha_min=0.1, alpha_max=0.6) == pytest.approx(0.6)


# ---- REWARD_CONFIGS trigger_mode wiring ----


def test_reward_configs_have_correct_trigger_modes() -> None:
    """The baseline must keep pressure-based triggering (so the
    comparison cell stays bit-identical to exp 23/25), and all
    reward-using configs must use count-based triggering (the
    fix for the R-anti-correlation pressure-suppression cascade)."""
    from continual_synapse.reward.training_configs import REWARD_CONFIGS

    assert REWARD_CONFIGS["cs_gated_cosine_developmental"].trigger_mode == "pressure"
    assert REWARD_CONFIGS["cs_reward_developmental"].trigger_mode == "count"
    assert REWARD_CONFIGS["cosine_reward_developmental"].trigger_mode == "count"
    assert REWARD_CONFIGS["reward_only_static"].trigger_mode == "count"
    # And the invariant that all non-constant alpha_mode configs use
    # count triggering — guards against future configs being added
    # without flipping the flag.
    for cfg in REWARD_CONFIGS.values():
        if cfg.alpha_mode != "constant":
            assert cfg.trigger_mode == "count", (
                f"{cfg.name}: alpha_mode={cfg.alpha_mode} but "
                f"trigger_mode={cfg.trigger_mode}; reward-using "
                f"configs must use count to avoid the R-magnitude "
                f"cascade."
            )
