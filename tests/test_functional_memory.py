"""Tests for FunctionalMemory + Hinton distillation loss."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from continual_synapse.functional.functional_memory import (
    FunctionalMemory,
    distillation_loss,
)


def _dummy_forward(input_dim: int = 8, num_classes: int = 4):
    """Build a tiny deterministic forward closure for the snapshot
    tests. Uses a fixed seed so the soft targets are reproducible
    across the test file."""
    torch.manual_seed(0)
    W = torch.randn(input_dim, num_classes)

    def fwd(x: torch.Tensor) -> torch.Tensor:
        return x @ W

    return fwd, num_classes


# ---- 1. empty memory returns None ----


def test_empty_memory_returns_none() -> None:
    """sample_batch on an empty store must return None — the train
    step skips the distillation branch entirely on that signal."""
    mem = FunctionalMemory(samples_per_task=10)
    assert mem.sample_batch(batch_size=8) is None
    assert len(mem) == 0


# ---- 2. record adds entries ----


def test_record_task_end_adds_entries() -> None:
    """Recording from a 200-row pool with samples_per_task=100 must
    add exactly 100 entries, all carrying the supplied task_id."""
    fwd, _ = _dummy_forward(input_dim=8, num_classes=4)
    mem = FunctionalMemory(samples_per_task=100)
    pool = torch.randn(200, 8)
    n_added = mem.record_task_end(
        model_forward=fwd, task_inputs=pool, task_id=3,
    )
    assert n_added == 100
    assert len(mem) == 100
    assert all(t == 3 for t in mem.task_ids)
    # The stored inputs are CPU tensors of the right shape.
    assert mem.inputs[0].shape == (8,)
    assert mem.soft_targets[0].shape == (4,)


# ---- 3. max_total cap ----


def test_max_total_enforces_cap() -> None:
    """With max_total=50 and 100 inputs to record, the store ends
    up at exactly 50 entries — evictions happen as new ones are
    added past the cap."""
    fwd, _ = _dummy_forward(input_dim=8, num_classes=4)
    mem = FunctionalMemory(samples_per_task=100, max_total=50)
    pool = torch.randn(100, 8)
    n_added = mem.record_task_end(
        model_forward=fwd, task_inputs=pool, task_id=0,
    )
    assert n_added == 100  # the function still processed 100 inputs
    assert len(mem) == 50  # but the store caps at 50 after evictions


# ---- 4. soft targets are valid probability distributions ----


def test_soft_targets_are_probabilities() -> None:
    """Every stored soft_target must sum to 1 along the class axis
    (it's a softmax output) and be non-negative."""
    fwd, num_classes = _dummy_forward(input_dim=6, num_classes=5)
    mem = FunctionalMemory(samples_per_task=20)
    pool = torch.randn(20, 6)
    mem.record_task_end(model_forward=fwd, task_inputs=pool, task_id=1)
    for soft in mem.soft_targets:
        assert torch.all(soft >= 0), "soft targets must be non-negative"
        s = float(soft.sum())
        assert math.isclose(s, 1.0, abs_tol=1e-5), (
            f"soft target should sum to 1, got {s}"
        )


# ---- 5. sample_batch returns correctly-shaped tensors ----


def test_sample_batch_returns_correct_shapes() -> None:
    """sample_batch(32) on a store with at least 32 entries returns
    (inputs: (32, input_dim), soft_targets: (32, n_classes))."""
    fwd, num_classes = _dummy_forward(input_dim=6, num_classes=5)
    mem = FunctionalMemory(samples_per_task=100)
    pool = torch.randn(100, 6)
    mem.record_task_end(model_forward=fwd, task_inputs=pool, task_id=0)

    out = mem.sample_batch(batch_size=32)
    assert out is not None
    inputs, targets = out
    assert inputs.shape == (32, 6)
    assert targets.shape == (32, 5)
    # Asking for more than we have just gives us all available.
    out_big = mem.sample_batch(batch_size=500)
    assert out_big is not None
    assert out_big[0].shape == (100, 6)


# ---- 6. distillation_loss is ~0 on perfectly aligned student/teacher ----


def test_distillation_loss_zero_on_identical() -> None:
    """If the current model's logits, when softened by T, exactly
    match the stored soft target's T-softened distribution, the KL
    must be ~0. We construct this by setting current_logits =
    log(stored_soft) — then softmax(log(p)/T) on both sides
    produces identical distributions."""
    torch.manual_seed(0)
    soft = F.softmax(torch.randn(4, 6), dim=-1)
    # Reverse-engineer logits that produce exactly this soft target:
    # logit = log(p) works because softmax(log(p)) = p.
    logits = soft.clamp(min=1e-8).log()
    loss = float(distillation_loss(logits, soft, temperature=2.0))
    assert loss < 1e-5, (
        f"identical student/teacher should give ~0 loss, got {loss:.6f}"
    )


# ---- 7. distillation_loss > 0 on divergent distributions ----


def test_distillation_loss_positive_on_divergent() -> None:
    """Two unrelated distributions should give a strictly positive
    KL. Also verifies the T² rebalancing: doubling the temperature
    should change the loss (the gradient-scale rebalancing depends
    on it)."""
    torch.manual_seed(1)
    soft = F.softmax(torch.randn(8, 5), dim=-1)
    logits = torch.randn(8, 5)  # uncorrelated with soft
    loss_T1 = float(distillation_loss(logits, soft, temperature=1.0))
    loss_T4 = float(distillation_loss(logits, soft, temperature=4.0))
    assert loss_T1 > 0
    assert loss_T4 > 0
    # The losses should differ because T² rebalancing and the
    # different softening change the comparison.
    assert not math.isclose(loss_T1, loss_T4, abs_tol=1e-3), (
        f"T=1 ({loss_T1}) and T=4 ({loss_T4}) shouldn't coincide for "
        f"uncorrelated student/teacher distributions"
    )
