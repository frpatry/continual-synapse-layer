"""Tests for the N (neuron) entity."""

from __future__ import annotations

from substrate.neuron import N


def test_N_default_values():
    n = N(id=0)
    assert n.activation == 0.0
    assert n.weight == 1.0


def test_N_explicit_values():
    n = N(id=42, activation=0.5, weight=1.3)
    assert n.id == 42
    assert n.activation == 0.5
    assert n.weight == 1.3


def test_N_unique_ids():
    """Different ids → different objects."""
    a = N(id=1)
    b = N(id=2)
    assert a.id != b.id


def test_N_reset_activation_clears_only_activation():
    """``reset_activation`` zeroes ``activation`` but leaves
    ``weight`` (the accumulated structural property) untouched."""
    n = N(id=0, activation=0.8, weight=2.5)
    n.reset_activation()
    assert n.activation == 0.0
    assert n.weight == 2.5
