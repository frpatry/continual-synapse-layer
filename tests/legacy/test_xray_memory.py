"""Unit tests for XRayMemory + nt_xent_multi_prototype_loss (Phase 5.7.0)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from continual_synapse.memory import (
    XRayMemory, nt_xent_multi_prototype_loss,
)


# ---------- helpers ----------


def _correct(n: int) -> torch.Tensor:
    return torch.ones(n, dtype=torch.bool)


def _make_memory(**kwargs) -> XRayMemory:
    """Small-feature-dim memory for fast tests. Schedule defaults
    are explicit so tests don't depend on the class defaults
    drifting."""
    defaults = dict(
        num_classes=5,
        feature_dim=16,
        prototypes_per_class=3,
        sparsity_start_refinements=10,
        sparsity_end_refinements=50,
        sparsity_max_drop_fraction=0.5,
        temp_start_refinements=50,
        temp_end_refinements=200,
        temperature_initial=1.0,
        temperature_final=0.3,
    )
    defaults.update(kwargs)
    return XRayMemory(**defaults)


# ---------- 1: memory state ----------


def test_init_empty():
    mem = _make_memory()
    assert mem.num_occupied() == 0
    assert mem.per_class_counts() == [0] * 5
    assert mem.prototypes.shape == (5, 3, 16)
    assert mem.is_occupied.shape == (5, 3)
    assert mem.refinement_counts.shape == (5, 3)
    assert not mem.is_occupied.any()


def test_allocate_first_prototype_for_new_class():
    mem = _make_memory()
    feat = torch.randn(1, 16)
    labels = torch.tensor([2])
    mem.update(feat, labels, _correct(1))
    # Class 2 slot 0 should now hold an exact copy of feat[0].
    assert mem.is_occupied[2, 0]
    assert not mem.is_occupied[2, 1]
    assert not mem.is_occupied[2, 2]
    assert torch.allclose(mem.prototypes[2, 0], feat[0])
    assert int(mem.refinement_counts[2, 0]) == 1
    assert mem.num_occupied() == 1


def test_correct_classification_refines_prototype():
    mem = _make_memory()
    torch.manual_seed(0)
    feat1 = torch.randn(1, 16) * 5  # large so feat1 != feat1+noise
    feat2 = feat1 + torch.randn(1, 16) * 0.1  # similar (high cosine sim)
    labels = torch.tensor([0])
    mem.update(feat1, labels, _correct(1))
    proto_after_1 = mem.prototypes[0, 0].clone()
    mem.update(feat2, labels, _correct(1))
    proto_after_2 = mem.prototypes[0, 0].clone()
    # EMA should produce a NEW prototype, neither equal to feat1
    # alone nor to feat2 alone.
    assert not torch.allclose(proto_after_2, proto_after_1)
    assert not torch.allclose(proto_after_2, feat2[0])
    # Refinement count incremented to 2.
    assert int(mem.refinement_counts[0, 0]) == 2
    # Still only one occupied slot (similar features → EMA-update,
    # not new slot).
    assert mem.num_occupied() == 1


def test_incorrect_classification_no_update():
    mem = _make_memory()
    feat1 = torch.randn(1, 16)
    feat2 = torch.randn(1, 16)
    labels = torch.tensor([3])
    mem.update(feat1, labels, _correct(1))
    proto_before = mem.prototypes[3, 0].clone()
    # Second update marked WRONG — must not change the prototype.
    mem.update(feat2, labels, torch.tensor([False]))
    proto_after = mem.prototypes[3, 0]
    assert torch.allclose(proto_after, proto_before)
    assert int(mem.refinement_counts[3, 0]) == 1  # unchanged


def test_multi_prototype_allocation():
    mem = _make_memory(distinctness_threshold=0.6)
    # Two intentionally orthogonal features for the same class →
    # cosine similarity 0, well below 0.6 threshold.
    feat1 = torch.zeros(1, 16)
    feat1[0, 0] = 1.0
    feat2 = torch.zeros(1, 16)
    feat2[0, 1] = 1.0
    labels = torch.tensor([1])
    mem.update(feat1, labels, _correct(1))
    mem.update(feat2, labels, _correct(1))
    # Both slots 0 AND 1 should now be occupied.
    assert mem.is_occupied[1, 0]
    assert mem.is_occupied[1, 1]
    assert not mem.is_occupied[1, 2]
    assert mem.num_occupied() == 2


def test_multi_prototype_cap_at_k():
    mem = _make_memory(prototypes_per_class=3, distinctness_threshold=0.6)
    labels = torch.tensor([4])
    # Four orthogonal features — first three fill slots, fourth
    # should EMA-update its nearest existing slot (not allocate).
    feats = []
    for d in range(4):
        f = torch.zeros(1, 16)
        f[0, d] = 1.0
        feats.append(f)
    for f in feats:
        mem.update(f, labels, _correct(1))
    # Three occupied, not four.
    assert int(mem.is_occupied[4].sum()) == 3
    # One of the slots got two updates (the original allocation +
    # the EMA from the fourth feature) — total refinements = 4
    # across the three slots.
    total_refinements = int(mem.refinement_counts[4].sum())
    assert total_refinements == 4


# ---------- 2: schedules ----------


def test_sparsification_kicks_in_at_threshold():
    mem = _make_memory(
        feature_dim=128,
        sparsity_start_refinements=10,
        sparsity_end_refinements=50,
        sparsity_max_drop_fraction=0.5,
    )
    labels = torch.tensor([0])
    torch.manual_seed(7)
    # Drive a single prototype to high refinement count by feeding
    # tightly clustered features (so they all EMA-update slot 0
    # rather than allocate new slots).
    base = torch.randn(1, 128) * 3.0
    for i in range(60):
        feat = base + torch.randn(1, 128) * 0.01
        mem.update(feat, labels, _correct(1))

    refinements = int(mem.refinement_counts[0, 0])
    assert refinements >= 50, f"expected ≥ 50 refinements, got {refinements}"

    # By refinement 50+, sparsity_max_drop=0.5 ⇒ ~50% zeros.
    proto = mem.prototypes[0, 0]
    zero_frac = float((proto == 0).float().mean())
    assert 0.40 <= zero_frac <= 0.55, (
        f"expected ~50% zeros after full sparsification; got {zero_frac:.2%}"
    )

    # And a brand-new prototype on a fresh class with 1 refinement
    # must have zero zeros (pre-sparsity).
    feat_new = torch.randn(1, 128)
    mem.update(feat_new, torch.tensor([3]), _correct(1))
    fresh_proto = mem.prototypes[3, 0]
    # The random feature has zero zeros almost surely.
    assert float((fresh_proto == 0).float().mean()) == 0.0


def test_temperature_schedule_decreases():
    mem = _make_memory(
        temp_start_refinements=50,
        temp_end_refinements=200,
        temperature_initial=1.0,
        temperature_final=0.3,
    )
    # No occupied slots → default to temp_init.
    assert mem.temperature() == 1.0
    # Mock the mean refinement count by passing it explicitly.
    assert mem.temperature(0.0) == 1.0
    assert mem.temperature(49.0) == 1.0
    # Midway between 50 and 200 → halfway between 1.0 and 0.3.
    mid = mem.temperature(125.0)
    assert math.isclose(mid, (1.0 + 0.3) / 2.0, abs_tol=1e-6)
    # Past the end → clamped at temp_final.
    assert math.isclose(mem.temperature(200.0), 0.3, abs_tol=1e-6)
    assert math.isclose(mem.temperature(999.0), 0.3, abs_tol=1e-6)


# ---------- 3: output format ----------


def test_get_all_prototypes_returns_correct_format():
    mem = _make_memory()
    # Empty memory.
    p, l = mem.get_all_prototypes()
    assert p.shape == (0, 16)
    assert l.shape == (0,)
    assert l.dtype == torch.long

    # Populate 4 prototypes across 3 classes.
    feats = torch.randn(4, 16)
    labels = torch.tensor([0, 0, 2, 4])
    mem.update(feats, labels, _correct(4))
    # Class 0 has 1 or 2 prototypes depending on distinctness; at
    # least one each across {0, 2, 4}.
    p, l = mem.get_all_prototypes()
    assert p.shape[1] == 16
    assert p.shape[0] == mem.num_occupied()
    assert l.shape == (p.shape[0],)
    # Labels recovered correspond to occupied slots.
    seen = set(l.tolist())
    assert seen.issubset({0, 2, 4})
    assert {0, 2, 4}.issubset(seen)


def test_no_raw_input_stored():
    """Privacy contract: XRayMemory must not expose any attribute
    that smells like raw inputs. The fields it DOES expose are
    prototypes (EMA-blended feature vectors, never raw), refinement
    counts, and the occupied mask."""
    mem = _make_memory()
    for forbidden in ("inputs", "raw_inputs", "raw", "samples"):
        assert not hasattr(mem, forbidden), (
            f"XRayMemory must not expose {forbidden!r} — privacy by design"
        )
    # Sanity: the expected fields ARE present.
    for expected in ("prototypes", "refinement_counts", "is_occupied"):
        assert hasattr(mem, expected), f"missing buffer {expected!r}"


# ---------- 4: NT-Xent loss ----------


def test_nt_xent_loss_zero_when_features_match_correct_prototype():
    """Query features perfectly aligned with the correct class's
    prototype should give a *low* loss (not exactly zero — there
    are other prototypes in memory contributing to the softmax
    denominator)."""
    torch.manual_seed(0)
    D = 16
    # 3 classes, one prototype each, all orthogonal.
    protos = torch.eye(3, D)  # shape (3, 16)
    proto_labels = torch.tensor([0, 1, 2])
    # Query exactly matches class 1's prototype.
    feats = protos[1:2].clone()
    labels = torch.tensor([1])
    loss = nt_xent_multi_prototype_loss(
        feats, labels, protos, proto_labels, temperature=0.5,
    )
    # With sim=1 for the positive and sim=0 for the two negatives,
    # at temperature 0.5: logits = [0, 2, 0]; softmax denominator
    # = e^0 + e^2 + e^0 = 1 + 7.389 + 1 = 9.389; positive prob ≈
    # 7.389 / 9.389 ≈ 0.787; -log(0.787) ≈ 0.24. Should be < 1.0.
    assert float(loss) < 1.0, f"expected low loss for matching feature, got {float(loss):.3f}"


def test_nt_xent_loss_high_when_features_match_wrong_prototype():
    """Query features close to a wrong-class prototype should give
    a *high* loss because the softmax mass concentrates on a
    negative."""
    torch.manual_seed(0)
    D = 16
    protos = torch.eye(3, D)
    proto_labels = torch.tensor([0, 1, 2])
    # Query exactly matches class 0's prototype, but the true
    # label is 2.
    feats = protos[0:1].clone()
    labels = torch.tensor([2])
    loss = nt_xent_multi_prototype_loss(
        feats, labels, protos, proto_labels, temperature=0.5,
    )
    # Symmetric to above: the positive (class 2) has sim 0, the
    # negative-but-attended class 0 has sim 1. At temp 0.5: logits
    # = [2, 0, 0]; positive class 2's log_softmax = -2 - log(...)
    # so -log_prob ≈ 2 + log(9.39) ≈ 2 + 2.24 = 4.24 → > 2.0. Need
    # to mean over positives (n=1) so loss ≈ 2.24. Assert > 2.0.
    assert float(loss) > 2.0, f"expected high loss for wrong-class match, got {float(loss):.3f}"


def test_nt_xent_loss_zero_when_no_prototypes():
    """Edge case: empty prototype memory must yield a zero loss
    (not NaN, not error)."""
    feats = torch.randn(4, 16)
    labels = torch.tensor([0, 1, 2, 3])
    protos = torch.empty(0, 16)
    proto_labels = torch.empty(0, dtype=torch.long)
    loss = nt_xent_multi_prototype_loss(
        feats, labels, protos, proto_labels, temperature=0.5,
    )
    assert float(loss) == 0.0


def test_nt_xent_loss_zero_when_no_positives_in_memory():
    """If none of the query labels have a matching prototype in
    memory, the loss must be 0 (skipped, not NaN)."""
    protos = torch.eye(2, 16)
    proto_labels = torch.tensor([0, 1])
    feats = torch.randn(3, 16)
    # All query labels are in {2, 3, 4}; no positives in memory.
    labels = torch.tensor([2, 3, 4])
    loss = nt_xent_multi_prototype_loss(
        feats, labels, protos, proto_labels, temperature=0.5,
    )
    assert float(loss) == 0.0
