"""Tests for the pressure metric and consolidation trigger."""

from __future__ import annotations

import math

import pytest
import torch

from continual_synapse.consolidation.trigger import (
    ConsolidationTrigger,
    compute_pressure,
)
from continual_synapse.synapse_layer.layer import SynapseLayer


def _populate(layer: SynapseLayer, *, strength, evidence, access_count) -> None:
    with torch.no_grad():
        layer.strengths.copy_(torch.as_tensor(strength, dtype=torch.float32))
        layer.evidence.copy_(torch.as_tensor(evidence, dtype=torch.float32))
        layer.access_count.copy_(torch.as_tensor(access_count, dtype=torch.int64))


def test_pressure_uses_design_formula() -> None:
    """|s| * e / (1 + a). Hand-computed on a 2×2."""
    layer = SynapseLayer(n_neurons=2)
    _populate(
        layer,
        strength=[[2.0, -3.0], [0.0, 1.0]],
        evidence=[[4.0, 4.0], [4.0, 4.0]],
        access_count=[[0, 1], [3, 0]],
    )
    p = compute_pressure(layer)
    # |2|*4/(1+0)=8, |-3|*4/(1+1)=6, |0|*4/(1+3)=0, |1|*4/(1+0)=4
    expected = torch.tensor([[8.0, 6.0], [0.0, 4.0]])
    torch.testing.assert_close(p, expected)


def test_pressure_is_non_negative() -> None:
    g = torch.Generator().manual_seed(0)
    layer = SynapseLayer(n_neurons=4)
    _populate(
        layer,
        strength=torch.randn(4, 4, generator=g),
        evidence=torch.rand(4, 4, generator=g),
        access_count=torch.randint(0, 5, (4, 4), generator=g),
    )
    p = compute_pressure(layer)
    assert torch.all(p >= 0)


def test_pressure_zero_when_strength_or_evidence_zero() -> None:
    layer = SynapseLayer(n_neurons=3)
    # Strength zero everywhere → pressure zero.
    _populate(
        layer,
        strength=torch.zeros(3, 3),
        evidence=torch.full((3, 3), 5.0),
        access_count=torch.zeros(3, 3, dtype=torch.int64),
    )
    assert torch.all(compute_pressure(layer) == 0)
    # Evidence zero everywhere → pressure zero.
    _populate(
        layer,
        strength=torch.full((3, 3), 2.0),
        evidence=torch.zeros(3, 3),
        access_count=torch.zeros(3, 3, dtype=torch.int64),
    )
    assert torch.all(compute_pressure(layer) == 0)


def test_pressure_drops_with_access_count() -> None:
    """Holding strength and evidence constant, pressure decreases as
    access_count grows."""
    layer = SynapseLayer(n_neurons=2)
    _populate(
        layer,
        strength=torch.ones(2, 2),
        evidence=torch.ones(2, 2),
        access_count=[[0, 1], [5, 99]],
    )
    p = compute_pressure(layer)
    assert p[0, 0] > p[0, 1] > p[1, 0] > p[1, 1]


# ---- trigger semantics ----


def test_trigger_constructor_validates_args() -> None:
    with pytest.raises(ValueError, match="avg_pressure_threshold"):
        ConsolidationTrigger(avg_pressure_threshold=-1.0)
    with pytest.raises(ValueError, match="min_steps_between"):
        ConsolidationTrigger(min_steps_between=-1)
    with pytest.raises(ValueError, match="candidate_quantile"):
        ConsolidationTrigger(candidate_quantile=0.0)
    with pytest.raises(ValueError, match="candidate_quantile"):
        ConsolidationTrigger(candidate_quantile=1.5)


def test_should_fire_returns_false_for_empty_state() -> None:
    layer = SynapseLayer(n_neurons=3)
    trigger = ConsolidationTrigger(avg_pressure_threshold=0.01)
    # Without any updates, all buffers are zero → pressure zero → no fire.
    assert not trigger.should_fire(layer)


def test_should_fire_triggers_above_threshold() -> None:
    layer = SynapseLayer(n_neurons=2)
    _populate(
        layer,
        strength=torch.ones(2, 2),
        evidence=torch.ones(2, 2),
        access_count=torch.zeros(2, 2, dtype=torch.int64),
    )
    # Bump global_step so the refractory check passes.
    with torch.no_grad():
        layer.global_step.fill_(100)
    # Mean pressure = 1.0 here. Threshold below should fire.
    trigger = ConsolidationTrigger(
        avg_pressure_threshold=0.5, min_steps_between=0
    )
    assert trigger.should_fire(layer)
    trigger_high = ConsolidationTrigger(
        avg_pressure_threshold=2.0, min_steps_between=0
    )
    assert not trigger_high.should_fire(layer)


def test_min_steps_between_blocks_consecutive_fires() -> None:
    layer = SynapseLayer(n_neurons=2)
    _populate(
        layer,
        strength=torch.ones(2, 2),
        evidence=torch.ones(2, 2),
        access_count=torch.zeros(2, 2, dtype=torch.int64),
    )
    with torch.no_grad():
        layer.global_step.fill_(50)
    trigger = ConsolidationTrigger(
        avg_pressure_threshold=0.1, min_steps_between=10
    )
    assert trigger.should_fire(layer)
    trigger.mark_fired(layer)
    # Same step → no fire allowed.
    assert not trigger.should_fire(layer)
    # 9 steps later → still refractory.
    with torch.no_grad():
        layer.global_step.fill_(59)
    assert not trigger.should_fire(layer)
    # 10 steps later → cleared.
    with torch.no_grad():
        layer.global_step.fill_(60)
    assert trigger.should_fire(layer)


def test_candidate_mask_selects_top_quantile() -> None:
    """With 16 synapses and quantile=0.25, exactly 4 should be selected."""
    layer = SynapseLayer(n_neurons=4)
    # Linear pressure: 0..15 across the flattened tensor.
    s = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    _populate(
        layer,
        strength=s,
        evidence=torch.ones(4, 4),
        access_count=torch.zeros(4, 4, dtype=torch.int64),
    )
    trigger = ConsolidationTrigger(candidate_quantile=0.25)
    mask = trigger.candidate_mask(layer)
    assert mask.dtype == torch.bool
    # Top 4 entries (values 12, 13, 14, 15) should be True.
    selected = mask.flatten().nonzero().flatten().tolist()
    assert sorted(selected) == [12, 13, 14, 15]


def test_candidate_mask_includes_at_least_one_when_uniform() -> None:
    """If all pressures are equal, the quantile cutoff includes them all."""
    layer = SynapseLayer(n_neurons=3)
    _populate(
        layer,
        strength=torch.ones(3, 3),
        evidence=torch.ones(3, 3),
        access_count=torch.zeros(3, 3, dtype=torch.int64),
    )
    trigger = ConsolidationTrigger(candidate_quantile=0.1)
    mask = trigger.candidate_mask(layer)
    # Quantile cutoff equals 1.0; every entry passes the >= check.
    assert mask.all()


def test_pressure_does_not_propagate_gradients() -> None:
    layer = SynapseLayer(n_neurons=2)
    _populate(
        layer,
        strength=torch.ones(2, 2),
        evidence=torch.ones(2, 2),
        access_count=torch.zeros(2, 2, dtype=torch.int64),
    )
    p = compute_pressure(layer)
    assert not p.requires_grad
    assert p.grad_fn is None


# ---- Active-mask pressure normalization (sparse-mode pathology fix) ----


def test_should_fire_mean_pressure_active_mask_matches_dense_at_full_density() -> None:
    """Dense mode (all strengths non-zero) ⇒ active mask is all-True
    ⇒ masked mean equals unmasked mean. Bit-exact backward compat."""
    layer = SynapseLayer(n_neurons=4)
    _populate(
        layer,
        strength=torch.full((4, 4), 0.5),  # all entries non-zero
        evidence=torch.full((4, 4), 1.0),
        access_count=torch.zeros(4, 4, dtype=torch.int64),
    )
    with torch.no_grad():
        layer.global_step.fill_(1000)
    # Mean pressure = 0.5 * 1 / 1 = 0.5 everywhere; mean = 0.5.
    # Threshold of 0.4 must fire; threshold of 0.6 must not.
    assert ConsolidationTrigger(
        avg_pressure_threshold=0.4, min_steps_between=0
    ).should_fire(layer)
    assert not ConsolidationTrigger(
        avg_pressure_threshold=0.6, min_steps_between=0
    ).should_fire(layer)


def test_should_fire_not_artificially_deflated_in_sparse_mode() -> None:
    """The regression. Equivalent active synapses produce equivalent
    mean pressure regardless of how many zero entries surround them.

    Setup: two 4×4 layers with the same 4 active synapses at the same
    strength/evidence/access_count. Layer A has the other 12 entries
    set to zero (mimics sparse top-k masking). Layer B has the other
    12 entries set to the same active values (full density). Before
    the fix, layer A's mean was ~4× lower than B's. After the fix,
    the masked mean is identical."""
    layer_sparse = SynapseLayer(n_neurons=4)
    layer_dense = SynapseLayer(n_neurons=4)
    # Active synapses: top-left 2×2 block at strength 0.5, evidence 1, access 0.
    s_sparse = torch.zeros(4, 4)
    s_sparse[:2, :2] = 0.5
    e_sparse = torch.zeros(4, 4)
    e_sparse[:2, :2] = 1.0
    s_dense = torch.full((4, 4), 0.5)
    e_dense = torch.full((4, 4), 1.0)
    _populate(
        layer_sparse, strength=s_sparse, evidence=e_sparse,
        access_count=torch.zeros(4, 4, dtype=torch.int64),
    )
    _populate(
        layer_dense, strength=s_dense, evidence=e_dense,
        access_count=torch.zeros(4, 4, dtype=torch.int64),
    )
    for layer in (layer_sparse, layer_dense):
        with torch.no_grad():
            layer.global_step.fill_(1000)

    trigger = ConsolidationTrigger(
        avg_pressure_threshold=0.4, min_steps_between=0
    )
    # Both must fire at threshold 0.4 — sparse should not be artificially
    # below threshold despite the 12 zero entries.
    assert trigger.should_fire(layer_sparse)
    assert trigger.should_fire(layer_dense)

    # Crucially, the threshold sensitivity must be the same — try a
    # threshold that's just above the active-synapse pressure (0.5).
    trigger_borderline = ConsolidationTrigger(
        avg_pressure_threshold=0.51, min_steps_between=0
    )
    assert not trigger_borderline.should_fire(layer_sparse)
    assert not trigger_borderline.should_fire(layer_dense)
    # And one just below — both fire.
    trigger_below = ConsolidationTrigger(
        avg_pressure_threshold=0.49, min_steps_between=0
    )
    assert trigger_below.should_fire(layer_sparse)
    assert trigger_below.should_fire(layer_dense)


def test_should_fire_false_when_no_active_synapses() -> None:
    """If every strength is zero (e.g. immediately after a full drain),
    there's nothing to consolidate; the trigger must decline."""
    layer = SynapseLayer(n_neurons=3)
    _populate(
        layer,
        strength=torch.zeros(3, 3),
        # Non-zero evidence — but pressure is strength * evidence / (1+access)
        # so it's still zero, and the active mask is all-False either way.
        evidence=torch.full((3, 3), 5.0),
        access_count=torch.zeros(3, 3, dtype=torch.int64),
    )
    with torch.no_grad():
        layer.global_step.fill_(1000)
    trigger = ConsolidationTrigger(
        avg_pressure_threshold=0.0, min_steps_between=0
    )
    # Even with threshold=0 (which would have fired pre-fix because
    # mean of zeros == 0 >= 0), the all-zero active mask short-circuits.
    assert not trigger.should_fire(layer)
