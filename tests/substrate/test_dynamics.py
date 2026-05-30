"""Tests for dynamics: soft_threshold, GlobalBackground, propagate."""

from __future__ import annotations

import numpy as np

from substrate.dynamics import (
    GlobalBackground,
    propagate_activation,
    soft_threshold,
)


# ---------- soft_threshold ----------

def test_soft_threshold_zeroes_subthreshold():
    """Inputs ≤ threshold yield 0."""
    out = soft_threshold(np.array([0.0, 0.1, 0.3, 0.2]), threshold=0.3)
    assert np.allclose(out, [0.0, 0.0, 0.0, 0.0])


def test_soft_threshold_linear_above_threshold():
    """Above threshold, output = x - threshold (until the cap)."""
    out = soft_threshold(np.array([0.4, 0.5, 0.8]), threshold=0.3)
    assert np.allclose(out, [0.1, 0.2, 0.5])


def test_soft_threshold_caps_at_one():
    """Output is clipped to ≤ 1.0 even for very large inputs."""
    out = soft_threshold(np.array([1.5, 5.0, 100.0]), threshold=0.3)
    assert np.all(out <= 1.0)


def test_soft_threshold_pushes_toward_sparsity():
    """With ``threshold=0.3``, a vector uniformly distributed in
    [0, 1] should have most entries zeroed (~30 % zero by chance,
    so we just check at least some are zero)."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.0, size=1000)
    out = soft_threshold(x, threshold=0.3)
    assert (out == 0.0).sum() > 200


# ---------- GlobalBackground ----------

def test_background_step_returns_correct_shape():
    bg = GlobalBackground(seed=0)
    out = bg.step(n_neurons=42)
    assert out.shape == (42,)


def test_background_never_silent_over_many_steps():
    """H4: substrate is never fully silent. Across many steps,
    max background value should be positive."""
    bg = GlobalBackground(base_level=0.1, drift_amplitude=0.05, seed=0)
    maxes = [bg.step(50).max() for _ in range(100)]
    assert max(maxes) > 0.0


def test_background_centred_near_base_level():
    """Mean across many steps should sit near ``base_level``
    (drift is a bounded random walk centred at 0; local noise
    averages out)."""
    bg = GlobalBackground(
        base_level=0.1, drift_amplitude=0.05,
        drift_rate=0.01, local_noise_sigma=0.02, seed=0,
    )
    samples = np.concatenate([bg.step(20) for _ in range(500)])
    # Loose bound — drift can take a while to mean-revert.
    assert 0.02 < float(samples.mean()) < 0.18


def test_background_drift_state_bounded():
    """Cumulative drift must stay in [-1, 1] even after many
    steps."""
    bg = GlobalBackground(drift_rate=0.5, seed=0)
    for _ in range(1000):
        bg.step(1)
    assert -1.0 <= bg.drift_state <= 1.0


# ---------- propagate_activation ----------

def test_propagate_basic_computation():
    """Tiny 3-neuron substrate with hand-computed expectation."""
    # 0 → 1, 1 → 2 (so activation at 0 reaches 1, then 2 next step)
    W = np.zeros((3, 3), dtype=np.float32)
    W[0, 1] = 1.0
    W[1, 2] = 1.0
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    neuron_weights = np.ones(3, dtype=np.float32)
    background = np.zeros(3, dtype=np.float32)
    new = propagate_activation(
        current_activations=a,
        connectivity_W=W,
        neuron_weights=neuron_weights,
        background=background,
        threshold=0.3,
    )
    # Neuron 1 receives 1.0 from neuron 0; soft_threshold(1.0, 0.3) = 0.7.
    assert new[0] == 0.0
    assert abs(new[1] - 0.7) < 1e-6
    assert new[2] == 0.0


def test_propagate_respects_threshold():
    """Weighted input below threshold should yield 0."""
    W = np.array([[0, 0.1], [0, 0]], dtype=np.float32)
    a = np.array([1.0, 0.0], dtype=np.float32)
    nw = np.ones(2, dtype=np.float32)
    bg = np.zeros(2, dtype=np.float32)
    out = propagate_activation(a, W, nw, bg, threshold=0.3)
    # 0.1 < 0.3 → both targets zero
    assert (out == 0.0).all()


def test_propagate_external_input_adds_to_drive():
    """External input is summed in before thresholding."""
    W = np.zeros((2, 2), dtype=np.float32)
    a = np.zeros(2, dtype=np.float32)
    nw = np.ones(2, dtype=np.float32)
    bg = np.zeros(2, dtype=np.float32)
    ext = np.array([0.5, 0.1], dtype=np.float32)
    out = propagate_activation(a, W, nw, bg, external_input=ext, threshold=0.3)
    # 0.5 + 0 > 0.3 → 0.2.  0.1 + 0 < 0.3 → 0.
    assert abs(out[0] - 0.2) < 1e-6
    assert out[1] == 0.0


def test_propagate_returns_float32():
    W = np.zeros((4, 4), dtype=np.float32)
    a = np.zeros(4, dtype=np.float32)
    nw = np.ones(4, dtype=np.float32)
    bg = np.zeros(4, dtype=np.float32)
    out = propagate_activation(a, W, nw, bg)
    assert out.dtype == np.float32
