"""Tests for PassTracker — candidacy + validation_passes."""

from __future__ import annotations

import numpy as np
import pytest

from substrate.pass_tracker import PassTracker


def _full_mask(n: int) -> np.ndarray:
    """All-to-all mask except diagonal."""
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    return mask


# ---------- init ----------


def test_pass_tracker_init_zero_state():
    """All tracking arrays start at zero / False."""
    pt = PassTracker(_full_mask(10))
    assert (pt.candidacy_strength == 0.0).all()
    assert (pt.validation_passes == 0).all()
    assert not pt.in_pass.any()


def test_pass_tracker_init_validates_hysteresis_gap():
    """th_low must be strictly less than th_high."""
    with pytest.raises(ValueError):
        PassTracker(_full_mask(4), theta_quiet_high=0.1, theta_quiet_low=0.1)
    with pytest.raises(ValueError):
        PassTracker(_full_mask(4), theta_quiet_high=0.05, theta_quiet_low=0.1)


def test_pass_tracker_init_validates_square_mask():
    """Mask must be 2D and square."""
    with pytest.raises(ValueError):
        PassTracker(np.ones((5, 6), dtype=bool))
    with pytest.raises(ValueError):
        PassTracker(np.ones(10, dtype=bool))


# ---------- update ----------


def test_pass_tracker_co_activation_increments_candidacy():
    """Co-activated pairs in the mask see candidacy_strength rise.
    Uncoactivated pairs stay at zero."""
    pt = PassTracker(
        _full_mask(5), boost_factor=0.5, decay_factor=1.0,
        theta_quiet_high=10.0, theta_quiet_low=0.001,
    )
    pt.update(np.array([1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    # N0 & N1 co-active → candidacy[0,1] = 0.5 * 1*1 = 0.5.
    assert pt.candidacy_strength[0, 1] > 0.0
    assert pt.candidacy_strength[1, 0] > 0.0
    # N0 / N2 — N2 inactive → no rise.
    assert pt.candidacy_strength[0, 2] == 0.0


def test_pass_tracker_no_self_pairs_in_candidacy():
    """Diagonal must stay zero — no pair (i, i) is tracked."""
    pt = PassTracker(_full_mask(4), boost_factor=1.0, decay_factor=1.0,
                     theta_quiet_high=10.0, theta_quiet_low=0.001)
    pt.update(np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
    assert (np.diag(pt.candidacy_strength) == 0.0).all()


def test_pass_tracker_off_mask_entries_stay_zero():
    """Pairs masked out of the connectivity see no candidacy build-up."""
    mask = _full_mask(4)
    mask[0, 1] = False  # explicitly remove the 0→1 edge
    mask[1, 0] = False
    pt = PassTracker(mask, boost_factor=1.0, decay_factor=1.0,
                     theta_quiet_high=10.0, theta_quiet_low=0.001)
    pt.update(np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32))
    assert pt.candidacy_strength[0, 1] == 0.0
    assert pt.candidacy_strength[1, 0] == 0.0


def test_pass_tracker_pass_count_increments_on_rising_edge():
    """validation_passes counts distinct entries into the pass regime.

    Uses fast decay (0.7) so 10 quiet steps drop candidacy below
    th_low — required to allow a clean second pass."""
    pt = PassTracker(
        _full_mask(3), boost_factor=0.2, decay_factor=0.7,
        theta_quiet_high=0.1, theta_quiet_low=0.05,
    )
    # Pass 1: drive co-activation up.
    for _ in range(5):
        pt.update(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    assert pt.validation_passes[0, 1] == 1
    assert pt.in_pass[0, 1]
    # Decay back to quiet.
    for _ in range(10):
        pt.update(np.zeros(3, dtype=np.float32))
    assert not pt.in_pass[0, 1]
    # Pass 2.
    for _ in range(5):
        pt.update(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    assert pt.validation_passes[0, 1] == 2


def test_pass_tracker_hysteresis_prevents_chatter():
    """Oscillations that re-cross th_high but never fall below th_low
    must NOT be counted as new passes."""
    pt = PassTracker(
        _full_mask(3), boost_factor=0.4, decay_factor=0.5,
        theta_quiet_high=0.3, theta_quiet_low=0.05,
    )
    # Single push above th_high — one pass.
    pt.update(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    assert pt.validation_passes[0, 1] == 1
    # A series of partial decays + boosts where candidacy stays above
    # th_low between cycles. The pair is "in pass" throughout, so no
    # new pass should be counted.
    for _ in range(5):
        pt.update(np.zeros(3, dtype=np.float32))
        pt.update(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    assert pt.validation_passes[0, 1] == 1
    # The pair is still in the pass regime.
    assert pt.in_pass[0, 1]


# ---------- find_emergence_candidates ----------


def test_find_emergence_candidates_requires_both_conditions():
    """Emergence needs W > theta AND passes >= n_min. Either alone is
    insufficient."""
    pt = PassTracker(_full_mask(4))
    # Pre-populate validation_passes for ONE pair only.
    pt.validation_passes[0, 1] = 5
    pt.validation_passes[1, 0] = 5

    W = np.zeros((4, 4), dtype=np.float32)
    W[0, 1] = 0.8        # high W, has passes → emerges
    W[1, 0] = 0.8
    W[1, 2] = 0.8        # high W, no passes → does NOT emerge
    W[0, 2] = 0.3        # low W, even with passes wouldn't qualify

    candidates = pt.find_emergence_candidates(
        connectivity_W=W,
        theta_emergence=0.5,
        n_min_passes=3,
    )
    assert candidates == [(0, 1)]


def test_find_emergence_candidates_skips_existing_p_pairs():
    """If a pair is already represented by a live P, don't suggest it
    again."""
    pt = PassTracker(_full_mask(4))
    pt.validation_passes[2, 3] = 5
    pt.validation_passes[3, 2] = 5
    W = np.zeros((4, 4), dtype=np.float32)
    W[2, 3] = 0.9
    W[3, 2] = 0.9

    cand_all = pt.find_emergence_candidates(W, theta_emergence=0.5, n_min_passes=3)
    assert cand_all == [(2, 3)]

    cand_filtered = pt.find_emergence_candidates(
        W, theta_emergence=0.5, n_min_passes=3,
        existing_p_pairs={(2, 3)},
    )
    assert cand_filtered == []


def test_find_emergence_candidates_returns_canonical_pairs():
    """Even if the symmetric W has both (i, j) and (j, i) qualifying,
    only the canonical (min, max) form is returned."""
    pt = PassTracker(_full_mask(3))
    pt.validation_passes[0, 2] = 5
    pt.validation_passes[2, 0] = 5
    W = np.zeros((3, 3), dtype=np.float32)
    W[0, 2] = 0.9
    W[2, 0] = 0.9

    candidates = pt.find_emergence_candidates(W, theta_emergence=0.5, n_min_passes=3)
    assert candidates == [(0, 2)]
