"""Tests for SPassTracker — recursive emergence at P level."""

from __future__ import annotations

import pytest

from substrate.p_entity import PEntity
from substrate.s_entity import SEntity
from substrate.s_pass_tracker import SPassTracker


def _make_p(pid: int, activation: float) -> PEntity:
    """Convenience: P entity with custom id and activation."""
    # components is required; use (pid, pid+1000) so canonical key
    # never collides with another test's P.
    return PEntity(id=pid, components=(pid, pid + 1000), activation=activation)


def _make_s(sid: int, contents: set[int], activation: float) -> SEntity:
    return SEntity(id=sid, contents=set(contents), activation=activation)


# ---------- init / validation ----------


def test_s_pass_tracker_init_zero_state():
    pt = SPassTracker()
    assert pt.pair_candidacy == {}
    assert pt.pair_passes == {}
    assert pt.pair_in_pass == {}
    assert pt.p_to_s_candidacy == {}


def test_s_pass_tracker_rejects_invalid_hysteresis():
    """th_low must be strictly less than th_high."""
    with pytest.raises(ValueError):
        SPassTracker(theta_high=0.1, theta_low=0.1)
    with pytest.raises(ValueError):
        SPassTracker(theta_high=0.05, theta_low=0.1)


# ---------- pair candidacy ----------


def test_co_active_p_pair_boosts_pair_candidacy():
    pt = SPassTracker(boost=0.5, decay=1.0,
                      theta_high=10.0, theta_low=0.001)
    p_entities = {
        0: _make_p(0, 0.8),
        1: _make_p(1, 0.8),
        2: _make_p(2, 0.0),  # below min_active_p
    }
    pt.update(p_entities, s_entities={})
    # (0, 1) co-active → candidacy > 0; (0, 2) and (1, 2) shouldn't.
    assert pt.pair_candidacy[(0, 1)] > 0.0
    assert (0, 2) not in pt.pair_candidacy
    assert (1, 2) not in pt.pair_candidacy


def test_pair_candidacy_decays_per_step():
    pt = SPassTracker(boost=0.5, decay=0.5,
                      theta_high=10.0, theta_low=0.001)
    p = {0: _make_p(0, 0.8), 1: _make_p(1, 0.8)}
    pt.update(p, s_entities={})
    initial = pt.pair_candidacy[(0, 1)]
    # Update with all-quiet P (no new boost) — decay only.
    p_quiet = {0: _make_p(0, 0.0), 1: _make_p(1, 0.0)}
    pt.update(p_quiet, s_entities={})
    assert pt.pair_candidacy[(0, 1)] == pytest.approx(initial * 0.5, abs=1e-6)


# ---------- p_to_s candidacy ----------


def test_active_p_and_active_s_boosts_p_to_s_candidacy():
    """If P co-fires with S and P is not in S.contents, candidacy boosts."""
    pt = SPassTracker(boost=0.5, decay=1.0,
                      theta_high=10.0, theta_low=0.001)
    p = {0: _make_p(0, 0.8), 1: _make_p(1, 0.8), 2: _make_p(2, 0.8)}
    s = {10: _make_s(10, {1, 2}, activation=0.7)}
    pt.update(p, s)
    # P0 not in S10 → candidacy should rise.
    assert pt.p_to_s_candidacy[(0, 10)] > 0.0
    # P1 IS in S10's contents → should NOT be tracked.
    assert (1, 10) not in pt.p_to_s_candidacy
    assert (2, 10) not in pt.p_to_s_candidacy


# ---------- pass detection (hysteresis) ----------


def test_rising_edge_increments_pass_count():
    """Pair candidacy rising above th_high increments validation_passes."""
    pt = SPassTracker(boost=0.5, decay=0.5,
                      theta_high=0.2, theta_low=0.05)
    p = {0: _make_p(0, 0.8), 1: _make_p(1, 0.8)}
    pt.update(p, s_entities={})
    # boost = 0.5 * 0.64 = 0.32 > 0.2 → pass count 1.
    assert pt.pair_passes[(0, 1)] == 1
    assert pt.pair_in_pass[(0, 1)] is True


def test_hysteresis_prevents_double_count_on_oscillation():
    """A pair staying above th_low between updates doesn't get
    re-counted on the next rise — hysteresis behaviour."""
    pt = SPassTracker(boost=0.4, decay=0.5,
                      theta_high=0.3, theta_low=0.05)
    # Use activation=1.0 so a single update boost (0.4·1·1 = 0.4)
    # immediately crosses theta_high=0.3.
    p = {0: _make_p(0, 1.0), 1: _make_p(1, 1.0)}
    pt.update(p, s_entities={})
    assert pt.pair_passes[(0, 1)] == 1
    # Several oscillating updates that stay above th_low.
    for _ in range(5):
        pt.update({0: _make_p(0, 0.0), 1: _make_p(1, 0.0)}, s_entities={})
        pt.update(p, s_entities={})
    assert pt.pair_passes[(0, 1)] == 1  # NO new count


# ---------- candidate queries ----------


def test_find_s_emergence_candidates_joint_criterion():
    """Emergence requires candidacy > theta_s_emergence AND passes >= n_min."""
    pt = SPassTracker(boost=1.0, decay=1.0,
                      theta_high=0.1, theta_low=0.01)
    p = {0: _make_p(0, 0.8), 1: _make_p(1, 0.8)}
    # Drive (0, 1) up several rises to get 3 passes.
    for _ in range(3):
        pt.update(p, s_entities={})
        # Quiet (decay can't lower us below th_low=0.01 with decay=1.0,
        # so manually clear in_pass to force a new rising edge next step.
        pt.pair_in_pass[(0, 1)] = False
    # Now candidacy is huge; should be candidate.
    candidates = pt.find_s_emergence_candidates(
        theta_s_emergence=0.5, n_min_passes=3,
    )
    assert candidates == [(0, 1)]
    # Stricter threshold blocks it.
    assert pt.find_s_emergence_candidates(
        theta_s_emergence=10000.0, n_min_passes=3,
    ) == []
    # Too few passes blocks it.
    assert pt.find_s_emergence_candidates(
        theta_s_emergence=0.5, n_min_passes=10,
    ) == []


def test_find_s_emergence_candidates_skips_existing_s_pairs():
    """If a (p_a, p_b) pair already gave rise to an S, don't return it."""
    pt = SPassTracker(boost=1.0, decay=1.0,
                      theta_high=0.1, theta_low=0.01)
    p = {0: _make_p(0, 0.8), 1: _make_p(1, 0.8)}
    for _ in range(3):
        pt.update(p, s_entities={})
        pt.pair_in_pass[(0, 1)] = False
    assert pt.find_s_emergence_candidates(
        theta_s_emergence=0.5, n_min_passes=3,
        existing_s_pairs={(0, 1)},
    ) == []


# ---------- cleanup ----------


def test_cleanup_dissolved_p_removes_related_tracking():
    pt = SPassTracker(boost=1.0, decay=1.0,
                      theta_high=10.0, theta_low=0.001)
    p = {
        0: _make_p(0, 0.8), 1: _make_p(1, 0.8), 2: _make_p(2, 0.8),
    }
    s = {10: _make_s(10, {99}, activation=0.7)}  # contents {99} so P0 can be candidate
    pt.update(p, s)
    # We have pair_candidacy on (0,1), (0,2), (1,2).
    assert len(pt.pair_candidacy) >= 3
    assert (0, 10) in pt.p_to_s_candidacy

    pt.cleanup_dissolved_p({0})
    # All entries containing 0 should be gone.
    assert not any(0 in k for k in pt.pair_candidacy)
    assert not any(k[0] == 0 for k in pt.p_to_s_candidacy)


def test_cleanup_dissolved_s_removes_related_tracking():
    pt = SPassTracker(boost=1.0, decay=1.0,
                      theta_high=10.0, theta_low=0.001)
    p = {0: _make_p(0, 0.8)}
    s = {10: _make_s(10, {99}, activation=0.7),
         20: _make_s(20, {99}, activation=0.7)}
    pt.update(p, s)
    assert (0, 10) in pt.p_to_s_candidacy
    assert (0, 20) in pt.p_to_s_candidacy
    pt.cleanup_dissolved_s({20})
    assert (0, 10) in pt.p_to_s_candidacy
    assert (0, 20) not in pt.p_to_s_candidacy
