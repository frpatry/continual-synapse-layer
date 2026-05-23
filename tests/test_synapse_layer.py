"""Tests for SynapseLayer v1."""

from __future__ import annotations

import pytest
import torch

from continual_synapse.synapse_layer.layer import SynapseLayer


def test_strengths_initialised_to_zero() -> None:
    layer = SynapseLayer(n_neurons=5)
    assert layer.strengths.shape == (5, 5)
    assert torch.all(layer.strengths == 0.0)
    assert layer.global_step.item() == 0


def test_consolidate_applies_hebbian_outer_product() -> None:
    """Single-batch update equals η · (aᵀa) / B exactly."""
    layer = SynapseLayer(n_neurons=3, learning_rate=0.1)
    a = torch.tensor([[1.0, 2.0, -1.0]])
    layer.consolidate(a)
    expected = 0.1 * (a.transpose(-1, -2) @ a) / 1.0
    torch.testing.assert_close(layer.strengths, expected)
    assert layer.global_step.item() == 1


def test_consolidate_averages_over_batch() -> None:
    """Update with B samples equals the mean of B single-sample updates."""
    lr = 0.05
    layer_batch = SynapseLayer(n_neurons=2, learning_rate=lr)
    layer_loop = SynapseLayer(n_neurons=2, learning_rate=lr)
    a = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])

    layer_batch.consolidate(a)

    # Equivalent: B independent batch-1 updates with lr scaled by 1/B,
    # then summed. Or directly compute the mean outer product:
    expected = lr * (a.transpose(-1, -2) @ a) / a.shape[0]
    torch.testing.assert_close(layer_batch.strengths, expected)

    # Cross-check against a loop using batch=1 with manually averaged lr.
    for row in a:
        layer_loop.consolidate(row.unsqueeze(0), reward=1.0 / a.shape[0])
    torch.testing.assert_close(layer_loop.strengths, expected)


def test_reward_scales_update_linearly() -> None:
    layer_a = SynapseLayer(n_neurons=2, learning_rate=0.1)
    layer_b = SynapseLayer(n_neurons=2, learning_rate=0.1)
    activations = torch.tensor([[1.0, 1.0]])
    layer_a.consolidate(activations, reward=1.0)
    layer_b.consolidate(activations, reward=3.0)
    torch.testing.assert_close(layer_b.strengths, 3.0 * layer_a.strengths)


def test_zero_activations_produce_zero_update() -> None:
    layer = SynapseLayer(n_neurons=4, learning_rate=1.0)
    layer.consolidate(torch.zeros(8, 4))
    assert torch.all(layer.strengths == 0.0)
    # Global step still advances — we did do an update, it just had no effect.
    assert layer.global_step.item() == 1


def test_empty_batch_is_a_noop() -> None:
    layer = SynapseLayer(n_neurons=3)
    layer.consolidate(torch.zeros(0, 3))
    assert layer.global_step.item() == 0


def test_consolidate_rejects_wrong_dim() -> None:
    layer = SynapseLayer(n_neurons=3)
    with pytest.raises(ValueError, match="n_neurons=3"):
        layer.consolidate(torch.zeros(2, 5))


def test_consolidate_rejects_non_2d() -> None:
    layer = SynapseLayer(n_neurons=3)
    with pytest.raises(ValueError, match="2-D"):
        layer.consolidate(torch.zeros(3))


def test_constructor_rejects_bad_args() -> None:
    with pytest.raises(ValueError, match="n_neurons"):
        SynapseLayer(n_neurons=0)
    with pytest.raises(ValueError, match="learning_rate"):
        SynapseLayer(n_neurons=3, learning_rate=0.0)


def test_consolidate_does_not_track_gradients() -> None:
    """Strengths must never accumulate autograd history."""
    layer = SynapseLayer(n_neurons=2, learning_rate=0.1)
    a = torch.randn(4, 2, requires_grad=True)
    layer.consolidate(a)
    assert not layer.strengths.requires_grad
    assert layer.strengths.grad_fn is None


def test_strengths_remain_finite_under_repeated_updates() -> None:
    """Numerical stability check with a small learning rate.

    100 updates with normalised activations should keep strengths
    within roughly the order of magnitude predicted by the
    Hebbian sum, with no NaNs or infinities.
    """
    layer = SynapseLayer(n_neurons=8, learning_rate=1e-3)
    g = torch.Generator().manual_seed(0)
    for _ in range(100):
        a = torch.randn(16, 8, generator=g)
        layer.consolidate(a)
    s = layer.strengths
    assert torch.isfinite(s).all()
    # Per-update outer-product norm is roughly E[||aaᵀ||] ~ n; with
    # lr=1e-3 and 100 updates, strengths magnitude stays modest.
    assert s.abs().max().item() < 5.0
    assert layer.global_step.item() == 100


def test_reset_zeroes_state() -> None:
    layer = SynapseLayer(n_neurons=2, learning_rate=0.5)
    layer.consolidate(torch.tensor([[1.0, 1.0]]))
    assert layer.global_step.item() == 1
    assert torch.any(layer.strengths != 0.0)
    layer.reset()
    assert layer.global_step.item() == 0
    assert torch.all(layer.strengths == 0.0)


def test_state_dict_roundtrip() -> None:
    layer = SynapseLayer(n_neurons=4, learning_rate=0.05)
    layer.consolidate(torch.randn(8, 4, generator=torch.Generator().manual_seed(1)))

    other = SynapseLayer(n_neurons=4, learning_rate=0.05)
    other.load_state_dict(layer.state_dict())
    torch.testing.assert_close(other.strengths, layer.strengths)
    assert other.global_step.item() == layer.global_step.item()


def test_to_device_moves_buffers() -> None:
    layer = SynapseLayer(n_neurons=3)
    moved = layer.to(torch.device("cpu"))  # CPU is the available device in CI
    assert moved.strengths.device.type == "cpu"
    assert moved.global_step.device.type == "cpu"
    assert moved.evidence.device.type == "cpu"


# ---- Phase 3: evidence buffer + resistance ----


def test_evidence_initialised_to_zero() -> None:
    layer = SynapseLayer(n_neurons=4)
    assert layer.evidence.shape == (4, 4)
    assert torch.all(layer.evidence == 0.0)


def test_evidence_accumulates_absolute_co_activations() -> None:
    """Evidence ← mean_b(|a_i| · |a_j|), regardless of sign."""
    layer = SynapseLayer(n_neurons=3, learning_rate=0.1)
    a = torch.tensor([[1.0, -2.0, 0.0]])
    layer.consolidate(a)
    expected = a.abs().transpose(-1, -2) @ a.abs() / a.shape[0]
    torch.testing.assert_close(layer.evidence, expected)


def test_evidence_grows_monotonically() -> None:
    layer = SynapseLayer(n_neurons=3, learning_rate=0.1)
    g = torch.Generator().manual_seed(0)
    layer.consolidate(torch.randn(8, 3, generator=g))
    snapshot = layer.evidence.clone()
    layer.consolidate(torch.randn(8, 3, generator=g))
    # Strict inequality everywhere both batches had non-zero activations,
    # weak inequality otherwise. Either way: never decreases.
    assert torch.all(layer.evidence >= snapshot)


def test_beta_zero_strength_update_matches_v1_exactly() -> None:
    """With β=0 the strength path is bit-identical to Phase 2 v1."""
    g = torch.Generator().manual_seed(1)
    a = torch.randn(16, 5, generator=g)

    layer = SynapseLayer(n_neurons=5, learning_rate=0.05, resistance_beta=0.0)
    layer.consolidate(a)

    # Hand-computed v1 update:
    expected = 0.05 * (a.transpose(-1, -2) @ a) / a.shape[0]
    torch.testing.assert_close(layer.strengths, expected)


def test_resistance_dampens_high_evidence_updates() -> None:
    """High-normalised-evidence synapses get a smaller update.

    Evidence is normalised by its current max before applying β,
    so the most-evidenced synapse always sees ``1/(1+β)`` resistance
    and zero-evidence synapses see no dampening.
    """
    layer = SynapseLayer(n_neurons=2, learning_rate=1.0, resistance_beta=1.0)
    # Pre-load evidence: max is 10, normalised becomes 1 for (0,0)
    # and 0 for (1,1).
    with torch.no_grad():
        layer.evidence[0, 0] = 10.0
        layer.evidence[1, 1] = 0.0

    a = torch.tensor([[1.0, 1.0]])
    layer.consolidate(a, reward=1.0)

    # raw_outer is all-ones; with β=1 and max_ev=10:
    #   resistance[0,0] = 1/(1+1*1) = 1/2
    #   resistance[1,1] = 1/(1+1*0) = 1
    s = layer.strengths
    assert s[1, 1].item() > s[0, 0].item()
    assert abs(s[0, 0].item() - 0.5) < 1e-5
    assert abs(s[1, 1].item() - 1.0) < 1e-5


def test_resistance_is_dataset_scale_independent() -> None:
    """Doubling the evidence scale should leave the strength update unchanged.

    With normalised evidence, only the *relative* magnitudes matter,
    so β has the same effective meaning regardless of how large
    evidence has grown.
    """
    layer_a = SynapseLayer(n_neurons=2, learning_rate=1.0, resistance_beta=1.0)
    layer_b = SynapseLayer(n_neurons=2, learning_rate=1.0, resistance_beta=1.0)
    with torch.no_grad():
        layer_a.evidence.fill_(10.0)
        layer_a.evidence[0, 0] = 5.0
        layer_b.evidence.fill_(10000.0)
        layer_b.evidence[0, 0] = 5000.0  # same ratio
    a = torch.tensor([[1.0, 1.0]])
    layer_a.consolidate(a)
    layer_b.consolidate(a)
    torch.testing.assert_close(layer_a.strengths, layer_b.strengths)


def test_resistance_reduces_cumulative_drift() -> None:
    """Over many updates, β>0 must keep strength magnitudes smaller than β=0."""
    g = torch.Generator().manual_seed(2)
    activations = [torch.randn(16, 4, generator=g) for _ in range(50)]

    layer_off = SynapseLayer(
        n_neurons=4, learning_rate=0.1, resistance_beta=0.0
    )
    layer_on = SynapseLayer(
        n_neurons=4, learning_rate=0.1, resistance_beta=1.0
    )
    for a in activations:
        layer_off.consolidate(a)
        layer_on.consolidate(a)

    # Both saw identical inputs; resistance must produce smaller strengths.
    assert layer_on.strengths.abs().max() < layer_off.strengths.abs().max()
    # Evidence accumulation is identical for both (resistance does not
    # affect the evidence path).
    torch.testing.assert_close(layer_on.evidence, layer_off.evidence)


def test_constructor_rejects_negative_beta() -> None:
    with pytest.raises(ValueError, match="resistance_beta"):
        SynapseLayer(n_neurons=3, resistance_beta=-0.1)


def test_reset_clears_evidence_too() -> None:
    layer = SynapseLayer(n_neurons=3, resistance_beta=0.5)
    layer.consolidate(torch.tensor([[1.0, 1.0, 1.0]]))
    assert torch.any(layer.evidence != 0.0)
    layer.reset()
    assert torch.all(layer.evidence == 0.0)


# ---- Phase 3 follow-up: confidence, age, access_count ----


def test_new_state_buffers_initialised_to_zero() -> None:
    layer = SynapseLayer(n_neurons=4)
    for name in ("confidence", "age", "access_count"):
        buf = getattr(layer, name)
        assert buf.shape == (4, 4)
        assert torch.all(buf == 0)
    assert layer.age.dtype == torch.int64
    assert layer.access_count.dtype == torch.int64
    assert layer.confidence.dtype == torch.float32


def test_age_ticks_every_consolidate_call() -> None:
    layer = SynapseLayer(n_neurons=2)
    a = torch.tensor([[1.0, 1.0]])
    for expected in range(1, 6):
        layer.consolidate(a)
        assert torch.all(layer.age == expected)


def test_confidence_stays_zero_on_first_batch() -> None:
    """No previous-batch reference exists, so confidence cannot grow."""
    layer = SynapseLayer(n_neurons=3)
    layer.consolidate(torch.tensor([[1.0, 2.0, -1.0]]))
    assert torch.all(layer.confidence == 0.0)


def test_confidence_grows_for_sustained_co_activation() -> None:
    """Confidence ← min(prev_abs_outer, curr_abs_outer) per pair."""
    layer = SynapseLayer(n_neurons=2)
    # Both batches have a = (1, 1) → abs_outer = [[1, 1], [1, 1]]
    a = torch.tensor([[1.0, 1.0]])
    layer.consolidate(a)  # confidence still zero (first batch)
    layer.consolidate(a)  # confidence += min(1, 1) = 1
    expected = torch.ones(2, 2)
    torch.testing.assert_close(layer.confidence, expected)
    # Third call adds another 1.
    layer.consolidate(a)
    torch.testing.assert_close(layer.confidence, expected * 2)


def test_confidence_truncates_to_weaker_of_two_batches() -> None:
    layer = SynapseLayer(n_neurons=2)
    layer.consolidate(torch.tensor([[2.0, 2.0]]))   # |outer| = 4 everywhere
    layer.consolidate(torch.tensor([[0.5, 0.5]]))   # |outer| = 0.25 everywhere
    # min(4, 0.25) = 0.25 everywhere; confidence grows by 0.25.
    expected = torch.full((2, 2), 0.25)
    torch.testing.assert_close(layer.confidence, expected)


def test_confidence_zero_when_one_batch_has_zero_pair() -> None:
    layer = SynapseLayer(n_neurons=2)
    layer.consolidate(torch.tensor([[1.0, 0.0]]))  # only (0,0) non-zero outer
    layer.consolidate(torch.tensor([[0.0, 1.0]]))  # only (1,1) non-zero outer
    # min across the two batches is zero everywhere.
    assert torch.all(layer.confidence == 0.0)


def test_record_access_counts_non_trivial_contributions() -> None:
    layer = SynapseLayer(n_neurons=3)
    with torch.no_grad():
        # Make (0, 0) strong, others zero.
        layer.strengths[0, 0] = 1.0
    # Activations: neuron 0 has mean |a| = 1.0; contribution at (0,0) = 1.0
    layer.record_access(torch.tensor([[1.0, 0.0, 0.0]]), threshold=0.1)
    assert layer.access_count[0, 0].item() == 1
    # All other entries below threshold (no strength or no activation).
    other = layer.access_count.clone()
    other[0, 0] = 0
    assert torch.all(other == 0)


def test_record_access_threshold_excludes_small_contributions() -> None:
    layer = SynapseLayer(n_neurons=2)
    with torch.no_grad():
        layer.strengths.fill_(0.01)
    layer.record_access(torch.tensor([[1.0, 1.0]]), threshold=0.5)
    # mean|a| = 1.0, |s| = 0.01, contribution = 0.01 < threshold 0.5
    assert torch.all(layer.access_count == 0)


def test_record_access_validates_shape() -> None:
    layer = SynapseLayer(n_neurons=3)
    with pytest.raises(ValueError):
        layer.record_access(torch.zeros(2, 5))


# ---- Phase 3 follow-up: sparse top-k partner selection ----


def test_sparse_false_unchanged() -> None:
    """sparse=False default reproduces dense behaviour bit-for-bit."""
    g = torch.Generator().manual_seed(0)
    a = torch.randn(8, 5, generator=g)

    dense = SynapseLayer(n_neurons=5, learning_rate=0.1)
    sparse_off = SynapseLayer(n_neurons=5, learning_rate=0.1, sparse=False)
    for _ in range(3):
        dense.consolidate(a)
        sparse_off.consolidate(a)
    torch.testing.assert_close(dense.strengths, sparse_off.strengths)
    torch.testing.assert_close(dense.evidence, sparse_off.evidence)


def test_sparse_top_k_equal_to_n_is_dense() -> None:
    """sparse=True with top_k=n keeps everything; equivalent to dense."""
    g = torch.Generator().manual_seed(1)
    a = torch.randn(6, 4, generator=g)

    dense = SynapseLayer(n_neurons=4, learning_rate=0.1)
    sparse = SynapseLayer(
        n_neurons=4, learning_rate=0.1, sparse=True, top_k=4
    )
    for _ in range(3):
        dense.consolidate(a)
        sparse.consolidate(a)
    torch.testing.assert_close(dense.strengths, sparse.strengths)


def test_sparse_keeps_at_most_k_partners_per_row() -> None:
    layer = SynapseLayer(
        n_neurons=8, learning_rate=0.5, sparse=True, top_k=3
    )
    g = torch.Generator().manual_seed(2)
    for _ in range(5):
        layer.consolidate(torch.randn(16, 8, generator=g))
    non_zero_per_row = (layer.strengths != 0).sum(dim=1)
    assert torch.all(non_zero_per_row <= 3)


def test_sparse_evicts_weakest_partner() -> None:
    """A new strong co-activation should displace the weakest partner."""
    layer = SynapseLayer(
        n_neurons=4, learning_rate=1.0, sparse=True, top_k=2
    )
    # Hand-place strengths so partners (0,1) and (0,2) are weak (~0.1, 0.2)
    # and (0,0), (0,3) are zero. The next consolidate should fill (0,0)
    # and (0,3) with strong co-activations and evict (0,1), (0,2).
    with torch.no_grad():
        layer.strengths[0, 1] = 0.1
        layer.strengths[0, 2] = 0.2
        # Pre-populate matching evidence so the resistance path also
        # has consistent state.
        layer.evidence[0, 1] = 0.1
        layer.evidence[0, 2] = 0.2
    # Activations: neuron 0 fires strongly with 3, very weakly with 1, 2.
    a = torch.tensor([[3.0, 0.01, 0.01, 3.0]])
    layer.consolidate(a)
    row0 = layer.strengths[0]
    # Top-2 should now be the new (0, 0) and (0, 3) co-activations.
    assert row0[0] != 0
    assert row0[3] != 0
    assert row0[1] == 0  # evicted
    assert row0[2] == 0  # evicted
    # All co-state for evicted positions should also be zero.
    assert layer.evidence[0, 1] == 0
    assert layer.evidence[0, 2] == 0


def test_sparse_state_buffers_all_zero_where_strengths_zero() -> None:
    """After an eviction every state field follows the strength mask."""
    layer = SynapseLayer(
        n_neurons=6, learning_rate=0.1, sparse=True, top_k=2
    )
    g = torch.Generator().manual_seed(3)
    for _ in range(4):
        layer.consolidate(torch.randn(8, 6, generator=g))
    mask = layer.strengths != 0
    for name in ("evidence", "confidence", "_prev_abs_outer"):
        buf = getattr(layer, name)
        # Wherever strength is zero, the other float buffers are zero.
        assert torch.all(buf[~mask] == 0), name
    for name in ("age", "access_count"):
        buf = getattr(layer, name)
        assert torch.all(buf[~mask] == 0), name


def test_constructor_rejects_bad_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        SynapseLayer(n_neurons=8, top_k=0)


def test_reset_clears_all_phase3_buffers() -> None:
    layer = SynapseLayer(n_neurons=2)
    layer.consolidate(torch.tensor([[1.0, 1.0]]))
    layer.consolidate(torch.tensor([[1.0, 1.0]]))
    layer.record_access(torch.tensor([[1.0, 1.0]]), threshold=0.001)
    # All five state buffers should be non-zero somewhere now.
    for name in ("strengths", "evidence", "confidence", "age", "access_count"):
        assert torch.any(getattr(layer, name) != 0), name
    layer.reset()
    for name in (
        "strengths",
        "evidence",
        "confidence",
        "age",
        "access_count",
        "_prev_abs_outer",
    ):
        assert torch.all(getattr(layer, name) == 0), name
    assert layer.global_step.item() == 0
