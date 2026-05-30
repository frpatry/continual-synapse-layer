"""Tests for the dynamic P-P PConnectivity store."""

from __future__ import annotations

from substrate.p_connectivity import PConnectivity


def test_pconnectivity_init_empty():
    pc = PConnectivity()
    assert pc.connection_count() == 0
    assert list(pc.all_pairs()) == []


def test_pconnectivity_add_connection_symmetric():
    """add_connection(a, b) and add_connection(b, a) refer to the same entry."""
    pc = PConnectivity()
    pc.add_connection(3, 7, weight=0.5)
    assert pc.get_weight(3, 7) == 0.5
    assert pc.get_weight(7, 3) == 0.5
    # Setting the reverse-order pair overwrites the same entry.
    pc.add_connection(7, 3, weight=0.9)
    assert pc.get_weight(3, 7) == 0.9
    assert pc.connection_count() == 1


def test_pconnectivity_no_self_connection():
    """Self-connections are a no-op."""
    pc = PConnectivity()
    pc.add_connection(5, 5, weight=1.0)
    assert pc.get_weight(5, 5) == 0.0
    assert pc.connection_count() == 0


def test_pconnectivity_get_weight_zero_if_absent():
    pc = PConnectivity()
    assert pc.get_weight(0, 1) == 0.0
    assert pc.get_weight(100, 200) == 0.0


def test_pconnectivity_update_weight_adds_delta():
    pc = PConnectivity()
    pc.update_weight(2, 4, 0.3)
    assert abs(pc.get_weight(2, 4) - 0.3) < 1e-6
    pc.update_weight(2, 4, 0.2)
    assert abs(pc.get_weight(2, 4) - 0.5) < 1e-6


def test_pconnectivity_update_weight_clipped_at_zero():
    """Large negative delta should clip to zero (and drop the entry)."""
    pc = PConnectivity()
    pc.update_weight(0, 1, 0.4)
    pc.update_weight(0, 1, -100.0)
    assert pc.get_weight(0, 1) == 0.0
    # Dropped entry → connection_count goes back to 0.
    assert pc.connection_count() == 0


def test_pconnectivity_zeroed_connection_removed_from_dict():
    """Explicitly setting a connection to zero via update drops it."""
    pc = PConnectivity()
    pc.update_weight(1, 2, 0.5)
    assert pc.connection_count() == 1
    # Exactly cancel.
    pc.update_weight(1, 2, -0.5)
    assert pc.connection_count() == 0
    assert pc.get_weight(1, 2) == 0.0


def test_pconnectivity_remove_entity_removes_all_its_connections():
    pc = PConnectivity()
    pc.add_connection(0, 1, 0.3)
    pc.add_connection(0, 2, 0.4)
    pc.add_connection(0, 3, 0.5)
    pc.add_connection(1, 2, 0.6)  # does NOT involve 0
    assert pc.connection_count() == 4
    pc.remove_entity(0)
    # 3 connections touching 0 removed; (1,2) survives.
    assert pc.connection_count() == 1
    assert pc.get_weight(0, 1) == 0.0
    assert pc.get_weight(0, 2) == 0.0
    assert pc.get_weight(0, 3) == 0.0
    assert pc.get_weight(1, 2) == 0.6


def test_pconnectivity_remove_entity_with_no_connections_is_noop():
    """Removing an id that has no connections is fine (no error, no change)."""
    pc = PConnectivity()
    pc.add_connection(1, 2, 0.5)
    pc.remove_entity(999)
    assert pc.connection_count() == 1


def test_pconnectivity_neighbors_of_returns_correct_dict():
    pc = PConnectivity()
    pc.add_connection(5, 7, 0.1)
    pc.add_connection(5, 9, 0.2)
    pc.add_connection(7, 9, 0.3)  # does NOT involve 5
    neighbors = pc.neighbors_of(5)
    assert neighbors == {7: 0.1, 9: 0.2}
    # Symmetric.
    neighbors_of_7 = pc.neighbors_of(7)
    assert neighbors_of_7 == {5: 0.1, 9: 0.3}


def test_pconnectivity_all_pairs_iterates_correctly():
    pc = PConnectivity()
    pc.add_connection(0, 1, 0.1)
    pc.add_connection(2, 3, 0.2)
    pairs = list(pc.all_pairs())
    assert len(pairs) == 2
    # Each pair is canonical (lower, higher, weight)
    pair_set = {(a, b, round(w, 6)) for a, b, w in pairs}
    assert pair_set == {(0, 1, 0.1), (2, 3, 0.2)}
