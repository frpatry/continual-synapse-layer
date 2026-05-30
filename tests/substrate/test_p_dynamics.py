"""Tests for P-level propagation + structural sparsity (k-WTA at P)."""

from __future__ import annotations

import numpy as np

from substrate.p_connectivity import PConnectivity
from substrate.p_dynamics import compute_p_input, propagate_p_activations
from substrate.p_entity import PEntity


# ---------- compute_p_input ----------


def test_compute_p_input_only_n_contribution_when_no_pp():
    """No P-P connections → input = alpha · (N_i + N_j) / 2."""
    p = PEntity(id=0, components=(2, 5))
    pc = PConnectivity()
    n_acts = np.zeros(10, dtype=np.float32)
    n_acts[2] = 0.8
    n_acts[5] = 0.4
    inp = compute_p_input(p, pc, {0: p}, n_acts, alpha_n_to_p=0.5)
    # Expected: 0.5 · (0.8 + 0.4) / 2 = 0.3.
    assert abs(inp - 0.3) < 1e-6


def test_compute_p_input_pp_contribution_added():
    """P-P channel adds Σ W · neighbour.activation on top of N contribution."""
    p_a = PEntity(id=0, components=(2, 5))
    p_b = PEntity(id=1, components=(3, 7), activation=0.6)
    pc = PConnectivity()
    pc.update_weight(0, 1, 0.4)
    # Zero N activations so the only contribution is the P-P edge.
    n_acts = np.zeros(10, dtype=np.float32)
    inp = compute_p_input(p_a, pc, {0: p_a, 1: p_b}, n_acts, alpha_n_to_p=0.5)
    # P-P: 0.4 (weight) · 0.6 (b activation) = 0.24
    assert abs(inp - 0.24) < 1e-6


def test_compute_p_input_skips_dissolved_neighbour():
    """A neighbour referenced in p_connectivity but absent from
    p_entities (e.g. dissolved this step) contributes nothing."""
    p_a = PEntity(id=0, components=(2, 5))
    pc = PConnectivity()
    pc.update_weight(0, 99, 0.5)  # neighbour 99 doesn't exist in entities
    n_acts = np.zeros(10, dtype=np.float32)
    inp = compute_p_input(p_a, pc, {0: p_a}, n_acts, alpha_n_to_p=0.0)
    assert inp == 0.0


# ---------- propagate_p_activations ----------


def test_propagate_p_no_entities_returns_empty():
    rng = np.random.default_rng(0)
    result = propagate_p_activations(
        {}, PConnectivity(), np.zeros(5, dtype=np.float32), rng=rng,
    )
    assert result == {}


def test_propagate_p_respects_kwta():
    """20 P, sparsity_target=0.05 → k=1 → exactly one winner."""
    rng = np.random.default_rng(0)
    # 20 P, each with distinct N components.
    p_entities = {i: PEntity(id=i, components=(2 * i, 2 * i + 1)) for i in range(20)}
    pc = PConnectivity()
    n_acts = np.zeros(50, dtype=np.float32)
    # Per-P distinct N input so values are not tied at the cutoff.
    for i in range(20):
        n_acts[2 * i] = 0.5 + 0.01 * i
        n_acts[2 * i + 1] = 0.5 + 0.01 * i

    new_acts = propagate_p_activations(
        p_entities, pc, n_acts,
        alpha_n_to_p=1.0,
        p_threshold=0.1,
        p_sparsity_target=0.05,
        p_background_noise_sigma=0.0,
        rng=rng,
    )
    n_active = sum(1 for v in new_acts.values() if v > 0)
    assert n_active == 1


def test_propagate_p_soft_threshold_below_returns_zero():
    """Low input + high threshold → all P at activation 0."""
    rng = np.random.default_rng(0)
    p_entities = {
        0: PEntity(id=0, components=(0, 1)),
        1: PEntity(id=1, components=(2, 3)),
    }
    n_acts = np.array([0.1, 0.1, 0.1, 0.1, 0.0], dtype=np.float32)
    result = propagate_p_activations(
        p_entities, PConnectivity(), n_acts,
        alpha_n_to_p=0.3, p_threshold=0.5,  # high — won't be cleared
        p_sparsity_target=0.5,
        p_background_noise_sigma=0.0,
        rng=rng,
    )
    # Each P input: 0.3 · (0.1 + 0.1) / 2 = 0.03 ≪ 0.5 → 0.
    assert all(v == 0.0 for v in result.values())


def test_propagate_p_reproducible_with_same_rng():
    """Two seeded RNGs with identical seed produce identical output."""
    p_entities = {
        0: PEntity(id=0, components=(0, 1)),
        1: PEntity(id=1, components=(2, 3)),
    }
    n_acts = np.array([0.4, 0.4, 0.4, 0.4, 0.0], dtype=np.float32)

    r1 = propagate_p_activations(
        p_entities, PConnectivity(), n_acts,
        p_background_noise_sigma=0.05,
        rng=np.random.default_rng(42),
    )
    r2 = propagate_p_activations(
        p_entities, PConnectivity(), n_acts,
        p_background_noise_sigma=0.05,
        rng=np.random.default_rng(42),
    )
    assert r1 == r2


def test_propagate_p_writes_back_synchronously():
    """The function must return without mutating ``p_entities`` —
    synchronous-update semantics rely on the caller applying the
    returned values *after* the dict computation finishes, so any
    neighbour read inside compute_p_input sees the OLD activations."""
    p_a = PEntity(id=0, components=(0, 1), activation=0.7)
    p_b = PEntity(id=1, components=(2, 3), activation=0.7)
    p_entities = {0: p_a, 1: p_b}
    n_acts = np.zeros(5, dtype=np.float32)
    _ = propagate_p_activations(
        p_entities, PConnectivity(), n_acts,
        p_background_noise_sigma=0.0,
        rng=np.random.default_rng(0),
    )
    # Original PEntity.activation values must be untouched.
    assert p_a.activation == 0.7
    assert p_b.activation == 0.7
