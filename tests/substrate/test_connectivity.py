"""Tests for ConnectivityMatrix."""

from __future__ import annotations

import numpy as np
import pytest

from substrate.connectivity import ConnectivityMatrix


def test_connectivity_correct_shape():
    c = ConnectivityMatrix(n_neurons=10, k=3)
    assert c.mask.shape == (10, 10)
    assert c.W.shape == (10, 10)


def test_connectivity_k_connections_per_neuron():
    """Each row of the mask should sum to k (the connection count
    per source N)."""
    c = ConnectivityMatrix(n_neurons=20, k=5, seed=0)
    per_row = c.mask.sum(axis=1)
    assert (per_row == 5).all()


def test_connectivity_no_self_connections():
    """Diagonal of the mask must be all False — no self-loops."""
    c = ConnectivityMatrix(n_neurons=12, k=4, seed=1)
    assert not c.mask.diagonal().any()


def test_connectivity_weights_only_on_mask():
    """Off-mask entries must be exactly zero at init."""
    c = ConnectivityMatrix(n_neurons=15, k=3, seed=2)
    assert (c.W[~c.mask] == 0.0).all()


def test_connectivity_initial_weights_non_negative():
    c = ConnectivityMatrix(n_neurons=10, k=3, seed=3)
    assert (c.W >= 0.0).all()


def test_update_weights_respects_mask():
    """A delta with non-zero entries off the mask shouldn't move
    those off-mask weights."""
    c = ConnectivityMatrix(n_neurons=8, k=2, seed=4)
    delta = np.ones((8, 8), dtype=np.float32) * 0.5
    before_off_mask = c.W[~c.mask].copy()
    c.update_weights(delta)
    after_off_mask = c.W[~c.mask]
    assert np.allclose(before_off_mask, after_off_mask)


def test_update_weights_clips_to_non_negative():
    """A large negative delta on the masked entries should clip
    weights to 0, not push them negative."""
    c = ConnectivityMatrix(n_neurons=6, k=2, seed=5)
    delta = -np.ones((6, 6), dtype=np.float32) * 100.0
    c.update_weights(delta)
    assert (c.W >= 0.0).all()


def test_update_weights_wrong_shape_raises():
    c = ConnectivityMatrix(n_neurons=8, k=2)
    with pytest.raises(ValueError):
        c.update_weights(np.zeros((4, 4), dtype=np.float32))


def test_connection_count_matches_mask():
    c = ConnectivityMatrix(n_neurons=20, k=4, seed=6)
    assert c.connection_count() == int(c.mask.sum()) == 20 * 4


def test_invalid_construction_raises():
    with pytest.raises(ValueError):
        ConnectivityMatrix(n_neurons=0, k=1)
    with pytest.raises(ValueError):
        ConnectivityMatrix(n_neurons=10, k=15)  # k >= n
    with pytest.raises(ValueError):
        ConnectivityMatrix(n_neurons=10, k=0)
