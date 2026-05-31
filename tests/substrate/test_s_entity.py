"""Tests for SEntity — schema dataclass (Phase 6i)."""

from __future__ import annotations

from substrate.s_entity import SEntity


def test_s_entity_defaults():
    """Default ctor produces empty contents, zero activation,
    weight=1.0, age=0."""
    s = SEntity(id=0)
    assert s.id == 0
    assert s.contents == set()
    assert s.activation == 0.0
    assert s.weight == 1.0
    assert s.age_at_emergence == 0.0


def test_s_entity_contents_coerced_to_set():
    """contents argument as list/tuple/iterable is normalised to a set."""
    s = SEntity(id=1, contents=[3, 7, 7, 5])
    assert isinstance(s.contents, set)
    assert s.contents == {3, 5, 7}


def test_s_entity_add_and_remove_member():
    s = SEntity(id=2, contents={1, 2})
    s.add_member(3)
    assert s.contents == {1, 2, 3}
    s.remove_member(2)
    assert s.contents == {1, 3}
    # Removing non-existent is a no-op (discard, not remove).
    s.remove_member(99)
    assert s.contents == {1, 3}


def test_s_entity_size():
    s = SEntity(id=3, contents={10, 20, 30})
    assert s.size() == 3
    s.add_member(40)
    assert s.size() == 4


def test_s_entity_reset_activation():
    s = SEntity(id=4, contents={1, 2}, activation=0.8, weight=0.5)
    s.reset_activation()
    assert s.activation == 0.0
    # Weight and contents untouched.
    assert s.weight == 0.5
    assert s.contents == {1, 2}
