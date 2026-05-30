"""Tests for the sparse top-k helper module."""

from __future__ import annotations

import pytest
import torch

from continual_synapse.synapse_layer.topk import (
    apply_topk_mask_inplace,
    compute_topk_mask,
)


def test_topk_keeps_only_k_per_row() -> None:
    strengths = torch.tensor(
        [
            [0.1, 0.5, -0.4, 0.2, 0.0],
            [0.0, 0.0, 0.7, -0.8, 0.3],
        ]
    )
    mask = compute_topk_mask(strengths, k=2)
    assert mask.dtype == torch.bool
    assert mask.shape == (2, 5)
    # Row 0: keep entries with |s| ∈ {0.5, 0.4}.
    assert mask[0].tolist() == [False, True, True, False, False]
    # Row 1: keep entries with |s| ∈ {0.8, 0.7}.
    assert mask[1].tolist() == [False, False, True, True, False]


def test_topk_with_k_at_full_width_is_all_true() -> None:
    s = torch.randn(3, 5)
    mask = compute_topk_mask(s, k=5)
    assert mask.all().item()


def test_topk_with_k_greater_than_width_is_all_true() -> None:
    s = torch.randn(3, 5)
    mask = compute_topk_mask(s, k=100)
    assert mask.all().item()


def test_topk_with_zero_k_is_all_false() -> None:
    s = torch.randn(3, 5)
    mask = compute_topk_mask(s, k=0)
    assert not mask.any().item()


def test_topk_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        compute_topk_mask(torch.zeros(5), k=2)


def test_topk_uses_absolute_values_not_signed() -> None:
    """A strongly negative entry should beat a weakly positive one."""
    strengths = torch.tensor([[-5.0, 0.1, 0.2]])
    mask = compute_topk_mask(strengths, k=1)
    assert mask.tolist() == [[True, False, False]]


def test_apply_topk_mask_zeros_out_excluded_entries() -> None:
    buf = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[True, False, True]])
    apply_topk_mask_inplace([buf], mask)
    assert buf.tolist() == [[1.0, 0.0, 3.0]]


def test_apply_topk_mask_handles_int_buffers() -> None:
    buf = torch.tensor([[5, 6, 7]], dtype=torch.int64)
    mask = torch.tensor([[True, False, False]])
    apply_topk_mask_inplace([buf], mask)
    assert buf.tolist() == [[5, 0, 0]]
    assert buf.dtype == torch.int64


def test_apply_topk_mask_applies_to_all_buffers() -> None:
    a = torch.tensor([[1.0, 2.0]])
    b = torch.tensor([[10, 20]], dtype=torch.int64)
    c = torch.tensor([[0.5, 0.5]])
    mask = torch.tensor([[True, False]])
    apply_topk_mask_inplace([a, b, c], mask)
    assert a.tolist() == [[1.0, 0.0]]
    assert b.tolist() == [[10, 0]]
    assert c.tolist() == [[0.5, 0.0]]


def test_apply_topk_mask_rejects_shape_mismatch() -> None:
    buf = torch.zeros(2, 3)
    mask = torch.zeros(3, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="match"):
        apply_topk_mask_inplace([buf], mask)
