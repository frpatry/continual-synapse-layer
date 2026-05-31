"""Tests for compute_s_to_p_feedback (Phase 6i)."""

from __future__ import annotations

import pytest

from substrate.p_entity import PEntity
from substrate.s_entity import SEntity
from substrate.s_to_p_feedback import compute_s_to_p_feedback


# ---------- short-circuits ----------


def test_feedback_empty_s_returns_zero_per_p():
    p = {0: PEntity(id=0, components=(0, 1)),
         1: PEntity(id=1, components=(2, 3))}
    boost = compute_s_to_p_feedback({}, p, gamma_s_to_p=1.0)
    assert boost == {0: 0.0, 1: 0.0}


def test_feedback_gamma_zero_returns_zero():
    p = {0: PEntity(id=0, components=(0, 1))}
    s = {10: SEntity(id=10, contents={0}, activation=0.7)}
    boost = compute_s_to_p_feedback(s, p, gamma_s_to_p=0.0)
    assert boost == {0: 0.0}


def test_feedback_inactive_s_no_contribution():
    p = {0: PEntity(id=0, components=(0, 1))}
    s = {10: SEntity(id=10, contents={0}, activation=0.0)}
    boost = compute_s_to_p_feedback(s, p, gamma_s_to_p=1.0)
    assert boost[0] == 0.0


# ---------- contribution semantics ----------


def test_active_s_boosts_each_component_p():
    p = {
        0: PEntity(id=0, components=(0, 1)),
        1: PEntity(id=1, components=(2, 3)),
        2: PEntity(id=2, components=(4, 5)),
    }
    s = {10: SEntity(id=10, contents={0, 1}, activation=0.5)}
    boost = compute_s_to_p_feedback(s, p, gamma_s_to_p=1.0)
    assert boost[0] == pytest.approx(0.5)
    assert boost[1] == pytest.approx(0.5)
    assert boost[2] == 0.0  # P2 not in S10.contents


def test_multiple_s_sharing_p_sum_contributions():
    """Two S each containing P0 each at activation 0.4 with γ=1 →
    P0 boost = 0.4 + 0.4 = 0.8 (sum, not max)."""
    p = {0: PEntity(id=0, components=(0, 1)),
         1: PEntity(id=1, components=(2, 3))}
    s = {
        10: SEntity(id=10, contents={0, 1}, activation=0.4),
        20: SEntity(id=20, contents={0}, activation=0.4),
    }
    boost = compute_s_to_p_feedback(s, p, gamma_s_to_p=1.0)
    assert boost[0] == pytest.approx(0.8)
    assert boost[1] == pytest.approx(0.4)


def test_feedback_skips_p_not_in_pool():
    """If an S references a P id that's already been dissolved (not
    in p_entities), the boost is silently skipped."""
    p = {0: PEntity(id=0, components=(0, 1))}
    s = {10: SEntity(id=10, contents={0, 99}, activation=0.5)}
    boost = compute_s_to_p_feedback(s, p, gamma_s_to_p=1.0)
    # P0 gets its share; P99 is not in the boost dict (since it doesn't exist).
    assert boost == {0: pytest.approx(0.5)}
