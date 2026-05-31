"""Tests for top-down P → N feedback (Phase 2c)."""

from __future__ import annotations

import numpy as np
import pytest

from substrate.p_entity import PEntity
from substrate.p_to_n_feedback import compute_p_to_n_feedback
from substrate.substrate import Substrate


# ---------- compute_p_to_n_feedback ----------


def test_feedback_empty_p_returns_zeros():
    boost = compute_p_to_n_feedback({}, n_neurons=10, gamma=0.1)
    assert boost.shape == (10,)
    assert (boost == 0.0).all()


def test_feedback_gamma_zero_returns_zeros():
    """gamma=0 short-circuits to the zero vector even with active P."""
    p_entities = {0: PEntity(id=0, components=(1, 2), activation=0.8)}
    boost = compute_p_to_n_feedback(p_entities, n_neurons=10, gamma=0.0)
    assert (boost == 0.0).all()


def test_feedback_inactive_p_no_contribution():
    """P with activation=0 contributes nothing even when gamma>0."""
    p_entities = {0: PEntity(id=0, components=(1, 2), activation=0.0)}
    boost = compute_p_to_n_feedback(p_entities, n_neurons=10, gamma=0.5)
    assert (boost == 0.0).all()


def test_feedback_active_p_boosts_both_components():
    """An active P contributes γ·activation to *each* component N
    and leaves every other N at zero."""
    p_entities = {0: PEntity(id=0, components=(3, 7), activation=0.8)}
    boost = compute_p_to_n_feedback(p_entities, n_neurons=10, gamma=0.5)
    expected = 0.5 * 0.8  # 0.4
    assert boost[3] == pytest.approx(expected)
    assert boost[7] == pytest.approx(expected)
    untouched = np.delete(boost, [3, 7])
    assert (untouched == 0.0).all()


def test_feedback_multiple_p_sharing_component_sums():
    """When two P share a component N, their contributions add (linear)."""
    p_entities = {
        0: PEntity(id=0, components=(1, 2), activation=0.5),
        1: PEntity(id=1, components=(1, 3), activation=0.7),
    }
    boost = compute_p_to_n_feedback(p_entities, n_neurons=5, gamma=1.0)
    # N1 is in both → 0.5 + 0.7 = 1.2
    # N2 only in p0 → 0.5
    # N3 only in p1 → 0.7
    assert boost[1] == pytest.approx(1.2)
    assert boost[2] == pytest.approx(0.5)
    assert boost[3] == pytest.approx(0.7)
    assert boost[0] == 0.0 and boost[4] == 0.0


def test_feedback_returns_float32():
    p_entities = {0: PEntity(id=0, components=(0, 1), activation=0.5)}
    boost = compute_p_to_n_feedback(p_entities, n_neurons=5, gamma=0.1)
    assert boost.dtype == np.float32


# ---------- Substrate-level integration ----------


def test_substrate_feedback_disabled_when_flag_false():
    """With ``enable_feedback_p_to_n=False``, a manually-injected
    active P must NOT influence N propagation.

    We compare two substrates with identical N-level seeds — one has
    a manually-added P + feedback DISABLED, the other has no P. Both
    should produce identical N activations because feedback is the
    only channel through which P can affect N in Phase 2c."""
    # With P, feedback DISABLED → P inert from N's perspective.
    sub_with_p_disabled = Substrate(
        n_neurons=50, k_connectivity=5,
        enable_feedback_p_to_n=False, gamma_p_to_n=0.5,
        seed=0,
    )
    sub_with_p_disabled.p_entities[0] = PEntity(
        id=0, components=(5, 10), activation=1.0,
    )

    # No P at all → no possible feedback anyway.
    sub_no_p = Substrate(
        n_neurons=50, k_connectivity=5,
        enable_feedback_p_to_n=True, gamma_p_to_n=0.5,
        seed=0,
    )

    sub_with_p_disabled.step()
    sub_no_p.step()

    # N activations identical: the only channel by which the
    # injected P could influence N is feedback, which is disabled.
    np.testing.assert_array_equal(
        sub_with_p_disabled.activations, sub_no_p.activations,
    )


def test_substrate_feedback_enabled_changes_dynamics():
    """With identical seeds + identical injected P, the only difference
    between two substrates is the ``enable_feedback_p_to_n`` flag.
    Their N activations after one step MUST differ."""
    sub_off = Substrate(
        n_neurons=50, k_connectivity=5,
        enable_feedback_p_to_n=False, gamma_p_to_n=0.5,
        seed=0,
    )
    sub_off.p_entities[0] = PEntity(
        id=0, components=(5, 10), activation=1.0,
    )

    sub_on = Substrate(
        n_neurons=50, k_connectivity=5,
        enable_feedback_p_to_n=True, gamma_p_to_n=0.5,
        seed=0,
    )
    sub_on.p_entities[0] = PEntity(
        id=0, components=(5, 10), activation=1.0,
    )

    sub_off.step()
    sub_on.step()

    # ``sub_on`` injects γ·1.0 = 0.5 into N5 and N10 before propagation;
    # ``sub_off`` injects nothing → activations diverge.
    assert not np.array_equal(sub_off.activations, sub_on.activations)
    # And specifically: N5 and N10 must be MORE active in sub_on
    # (or at least non-zero where sub_off is zero).
    assert sub_on.activations[5] >= sub_off.activations[5]
    assert sub_on.activations[10] >= sub_off.activations[10]
