"""Tests for the Substrate orchestrator."""

from __future__ import annotations

import numpy as np

from substrate.substrate import Substrate


def test_substrate_initialization_shapes():
    s = Substrate(n_neurons=20, k_connectivity=4, seed=0)
    assert s.n_neurons == 20
    assert len(s.neurons) == 20
    assert s.activations.shape == (20,)
    assert s.neuron_weights.shape == (20,)
    assert s.connectivity.W.shape == (20, 20)
    assert s.system_age == 0.0
    assert s.step_count == 0


def test_substrate_step_advances_age_and_counter():
    s = Substrate(n_neurons=10, k_connectivity=3, seed=0)
    for i in range(5):
        s.step()
    assert s.step_count == 5
    assert s.system_age == 5.0


def test_substrate_step_returns_copy_not_view():
    """The returned snapshot should not move when the substrate's
    activations are updated by a subsequent step."""
    s = Substrate(n_neurons=10, k_connectivity=3, seed=0)
    snap = s.step()
    s.step()
    # ``snap`` should be a frozen copy of the state at step 1.
    assert snap is not s.activations


def test_substrate_no_external_input_still_has_dynamics():
    """H4: with a metastable-tuned background (base + noise able
    to occasionally cross the threshold), the substrate exhibits
    spontaneous activity even without external input.

    The DEFAULT background params keep the substrate silent at
    threshold=0.3 (base=0.1, total reach ≈ 0.2); H4 is about the
    *capacity* for spontaneous activity given tuning, so we use
    bumped noise here to exercise the propagation path. See the
    P6 prediction work in later phases for the calibration sweep.
    """
    s = Substrate(
        n_neurons=30, k_connectivity=4, seed=0,
        background_base=0.2, local_noise_sigma=0.15,
        threshold=0.3,
    )
    saw_activity = False
    for _ in range(200):
        out = s.step()
        if (out > 0.0).any():
            saw_activity = True
            break
    assert saw_activity


def test_substrate_external_input_propagates():
    """Strong external input on neuron 0 should make it active in
    the next step (subject to soft-threshold)."""
    s = Substrate(n_neurons=20, k_connectivity=4, seed=0)
    ext = np.zeros(20, dtype=np.float32)
    ext[0] = 1.0
    out = s.step(external_input=ext)
    assert out[0] > 0.5  # 1.0 + small bg, soft-thresh(>=0.7) ≈ ≥ 0.7


def test_substrate_sparsity_diagnostic_returns_fraction():
    s = Substrate(n_neurons=10, k_connectivity=3, seed=0)
    s.activations = np.array(
        [0.0, 0.5, 0.0, 0.2, 0.05, 0.9, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    # > 0.1 → entries 1, 3, 5 → 3 / 10.
    assert s.sparsity(threshold=0.1) == 0.3


def test_substrate_pattern_pair_weights_grow_under_co_activation():
    """Sustained co-activation of a pattern should raise the
    sum of weights ON PATTERN-PATTERN connections specifically.

    Total weight across the whole substrate can drift either way
    because covariance Hebbian *weakens* uncorrelated (non-
    pattern) pairs at the same time it strengthens correlated
    (pattern) ones. The right thing to verify is the targeted
    pattern footprint, which is what P1 cares about.
    """
    s = Substrate(
        n_neurons=40, k_connectivity=10, seed=0,
        eta=0.1, lambda_decay=0.0,
    )
    pattern = np.arange(12)
    pattern_mask = np.zeros((40, 40), dtype=bool)
    for i in pattern:
        for j in pattern:
            if i != j:
                pattern_mask[i, j] = True
    pattern_connected = pattern_mask & s.connectivity.mask
    if not pattern_connected.any():
        # Vanishingly unlikely with k=10 and 12 pattern nodes,
        # but guard against the degenerate-random-seed corner.
        return
    initial = float(s.connectivity.W[pattern_connected].sum())
    ext = np.zeros(40, dtype=np.float32)
    ext[pattern] = 1.0
    for _ in range(50):
        s.step(external_input=ext)
    final = float(s.connectivity.W[pattern_connected].sum())
    assert final > initial, (
        f"pattern-pair weight should grow under co-activation: "
        f"{initial:.4f} → {final:.4f}"
    )


def test_substrate_neurons_have_correct_ids():
    """Each neuron in the ``neurons`` list has id matching its
    index in the substrate."""
    s = Substrate(n_neurons=15, k_connectivity=3, seed=0)
    for i, n in enumerate(s.neurons):
        assert n.id == i
