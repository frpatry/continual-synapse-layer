"""Tests for the Phase 6f fresh-pattern protection mechanism.

When a P entity emerges, the substrate sets ``p.protected_until =
step_count + k_protect``. Within that window:
  - decay STILL applies to the P's weight (the dissolution check just
    doesn't act on it);
  - dissolution is suppressed even if weight < viability_threshold.

After the window expires (``step_count >= protected_until``), normal
dissolution dynamics resume.
"""

from __future__ import annotations

import numpy as np

from substrate.p_entity import PEntity
from substrate.substrate import Substrate


# ---------- PEntity-level ----------


def test_pentity_protected_until_default_zero():
    """A PEntity with no explicit protected_until argument is born
    unprotected. (The Substrate sets the field on emergence.)"""
    p = PEntity(id=0, components=(1, 2))
    assert p.protected_until == 0
    # current_step=0 is NOT < protected_until=0 → not protected.
    assert not p.is_protected(0)


def test_pentity_is_protected_method():
    """is_protected returns True iff current_step < protected_until."""
    p = PEntity(id=0, components=(1, 2), protected_until=100)
    assert p.is_protected(0)
    assert p.is_protected(50)
    assert p.is_protected(99)
    # Boundary: at == protected_until, protection has expired.
    assert not p.is_protected(100)
    assert not p.is_protected(150)


# ---------- Substrate defaults ----------


def test_substrate_default_k_protect():
    """Phase 6f default for the protection window is 5000 steps."""
    sub = Substrate(n_neurons=20, k_connectivity=5)
    assert sub.k_protect == 5000


def test_substrate_accepts_custom_k_protect():
    sub = Substrate(n_neurons=20, k_connectivity=5, k_protect=1234)
    assert sub.k_protect == 1234


# ---------- Emergence sets protection ----------


def test_emerged_p_has_protected_until_set():
    """When a P emerges via _emerge_p, its protected_until reflects
    the substrate's step_count + k_protect at the time of emergence."""
    sub = Substrate(n_neurons=20, k_connectivity=5, k_protect=1000)
    # Advance the substrate by a few steps so step_count > 0.
    for _ in range(7):
        sub.step()
    pre_step = sub.step_count
    sub._emerge_p(0, 1)
    # exactly one P, with protected_until = pre_step + k_protect.
    assert sub.p_count() == 1
    p = next(iter(sub.p_entities.values()))
    assert p.protected_until == pre_step + 1000
    assert p.is_protected(pre_step)
    assert not p.is_protected(pre_step + 1000)


# ---------- Dissolution is suppressed during protection ----------


def test_protected_p_not_dissolved_below_viability():
    """Even with weight far below the viability threshold, a protected
    P entity survives the dissolution pass. Weight is still updated
    (decay can pull it further down) — the check just doesn't act."""
    sub = Substrate(n_neurons=20, k_connectivity=5, k_protect=1000)
    p = PEntity(
        id=0, components=(1, 2),
        weight=0.05,        # far below default viability 0.1
        protected_until=500,
    )
    sub.p_entities[0] = p
    sub.p_pairs_emerged.add(p.components)
    sub._next_p_id = 1

    # step_count is 0, well within protection window.
    sub._decay_and_dissolve_p()

    # P should still exist — protection suppressed dissolution.
    assert 0 in sub.p_entities
    # And weight still updated (decay applied).
    # At step_count=0, system_age=0 → age_factor=1.0; decay term
    # = 1.0 · p_weight_decay · 0.05 = 0.0005 · 0.05 (after eta·act²=0)
    # so weight should be slightly below 0.05 after the step.
    assert sub.p_entities[0].weight < 0.05


def test_unprotected_p_dissolved_below_viability():
    """Past the protection window, normal dissolution applies."""
    sub = Substrate(n_neurons=20, k_connectivity=5)
    # Fast-forward step counter past any plausible protection.
    sub.step_count = 10_000
    p = PEntity(
        id=0, components=(1, 2),
        weight=0.05,
        protected_until=5_000,   # already expired
    )
    sub.p_entities[0] = p
    sub.p_pairs_emerged.add(p.components)

    sub._decay_and_dissolve_p()

    # Unprotected + below viability → dissolved.
    assert 0 not in sub.p_entities
    assert (1, 2) not in sub.p_pairs_emerged


def test_protected_p_dissolved_immediately_after_window_expires():
    """The two-step sequence: protection-window step (survives),
    post-window step (dissolved). Confirms the window boundary is
    actually load-bearing."""
    sub = Substrate(n_neurons=20, k_connectivity=5, k_protect=10)
    p = PEntity(
        id=0, components=(1, 2),
        weight=0.05,        # far below viability
        protected_until=5,   # very short window
    )
    sub.p_entities[0] = p
    sub.p_pairs_emerged.add(p.components)

    # step_count starts at 0, protected_until = 5 → protected.
    sub.step_count = 3
    sub._decay_and_dissolve_p()
    assert 0 in sub.p_entities  # protected

    # step_count = 5 → protection EXPIRED (boundary is strict <).
    sub.step_count = 5
    sub._decay_and_dissolve_p()
    assert 0 not in sub.p_entities  # dissolved


def test_protection_does_not_prevent_growth():
    """If a protected P sees activation > 0 and the Hebbian growth term
    pushes its weight above viability, it survives and is still
    growing — protection isn't anti-growth, just anti-dissolution."""
    sub = Substrate(n_neurons=20, k_connectivity=5, k_protect=1000)
    p = PEntity(
        id=0, components=(1, 2),
        weight=0.05,
        activation=0.8,    # strong activation → growth term contributes
        protected_until=500,
    )
    sub.p_entities[0] = p
    sub.p_pairs_emerged.add(p.components)
    pre_weight = p.weight

    sub._decay_and_dissolve_p()

    # Growth term: eta·activation² = 0.01·0.64 = 0.0064
    # Decay term: ~0.0005·0.05 ≈ 0.000025 at step_count=0
    # Net positive → weight up.
    assert sub.p_entities[0].weight > pre_weight
