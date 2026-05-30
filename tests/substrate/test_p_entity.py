"""Tests for the PEntity dataclass."""

from __future__ import annotations

from substrate.p_entity import PEntity


def test_p_entity_canonical_order():
    """Components are always stored with min first, max second."""
    p1 = PEntity(id=0, components=(7, 3))
    assert p1.components == (3, 7)

    p2 = PEntity(id=1, components=(2, 5))
    assert p2.components == (2, 5)


def test_p_entity_defaults():
    """activation=0, weight=1 (mature), age_at_emergence=0 by default."""
    p = PEntity(id=0, components=(1, 2))
    assert p.activation == 0.0
    assert p.weight == 1.0
    assert p.age_at_emergence == 0.0


def test_p_entity_reset_activation():
    """reset_activation zeroes the activation, leaving weight intact."""
    p = PEntity(id=0, components=(1, 2), activation=0.7, weight=0.4)
    p.reset_activation()
    assert p.activation == 0.0
    assert p.weight == 0.4


def test_p_entity_equal_components_preserved():
    """Even (n, n) (a self-loop, would never emerge in practice)
    should not crash __post_init__."""
    p = PEntity(id=0, components=(3, 3))
    assert p.components == (3, 3)
