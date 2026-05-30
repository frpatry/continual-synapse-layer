"""Tests for the precision module (quantisation + serialisation)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from agi.memory.precision import (
    PRECISION_MODIFIER,
    PrecisionLevel,
    dequantize_to_float32,
    estimate_storage_bytes,
    quantize_to_level,
    serialize_facts,
)


# ---------- helpers ----------

def _random_unit_vec(dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm()


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


# ---------- L0 identity ----------

def test_quantize_dequantize_L0_identity():
    """L0 is the identity path — round-trip preserves the
    tensor element-for-element."""
    x = _random_unit_vec(64, 0)
    stored = quantize_to_level(x, PrecisionLevel.L0)
    out = dequantize_to_float32(stored, PrecisionLevel.L0, target_dim=64)
    assert torch.allclose(out, x, atol=1e-6)


# ---------- L1 (float16) ----------

def test_quantize_dequantize_L1_preserves_cosine():
    """Float16 round-trip should keep cosine sim > 0.99 on any
    moderate-magnitude embedding."""
    x = _random_unit_vec(256, 1)
    stored = quantize_to_level(x, PrecisionLevel.L1)
    out = dequantize_to_float32(stored, PrecisionLevel.L1, target_dim=256)
    assert _cos(x, out) > 0.99


# ---------- L2 (int8 sym) ----------

def test_quantize_dequantize_L2_preserves_cosine():
    """Int8 symmetric quantisation should keep cosine sim > 0.95."""
    x = _random_unit_vec(256, 2)
    stored = quantize_to_level(x, PrecisionLevel.L2)
    out = dequantize_to_float32(stored, PrecisionLevel.L2, target_dim=256)
    assert _cos(x, out) > 0.95


# ---------- L3 (int4 packed) ----------

def test_quantize_dequantize_L3_preserves_cosine():
    """Int4 packed quantisation should still keep cosine > 0.85."""
    x = _random_unit_vec(256, 3)
    stored = quantize_to_level(x, PrecisionLevel.L3)
    out = dequantize_to_float32(stored, PrecisionLevel.L3, target_dim=256)
    assert _cos(x, out) > 0.85


def test_quantize_dequantize_L3_odd_dim():
    """Odd dim must round-trip cleanly (Int4 packing pads the
    last nibble; dequantisation trims back to ``orig_dim``)."""
    x = _random_unit_vec(33, 4)
    stored = quantize_to_level(x, PrecisionLevel.L3)
    out = dequantize_to_float32(stored, PrecisionLevel.L3, target_dim=33)
    assert out.shape == (33,)
    assert _cos(x, out) > 0.85


# ---------- L4 (binary) ----------

def test_quantize_dequantize_L4_sign_only():
    """Binary L4 keeps only the sign of each component."""
    x = torch.tensor([0.9, -0.2, 0.0, -0.7, 0.3], dtype=torch.float32)
    stored = quantize_to_level(x, PrecisionLevel.L4)
    out = dequantize_to_float32(stored, PrecisionLevel.L4, target_dim=5)
    # x[2] == 0 is treated as "non-negative" → +1.
    expected = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0])
    assert torch.equal(out, expected)


def test_quantize_dequantize_L4_dim_not_multiple_of_8():
    """Non-multiple-of-8 dims must round-trip cleanly (pad bits
    are dropped via ``orig_dim``)."""
    x = _random_unit_vec(13, 5)
    stored = quantize_to_level(x, PrecisionLevel.L4)
    out = dequantize_to_float32(stored, PrecisionLevel.L4, target_dim=13)
    assert out.shape == (13,)
    # Sign agreement on every coord.
    assert torch.all((out > 0) == (x >= 0))


# ---------- L5 (existence trace) ----------

def test_L5_returns_zeros():
    """L5 dequantises to a zero vector of the target dim — these
    entries are skipped by the retriever in normal usage but the
    fallback shape matters for shape-invariant downstream code."""
    stored = quantize_to_level(torch.randn(16), PrecisionLevel.L5)
    assert stored is None
    out = dequantize_to_float32(None, PrecisionLevel.L5, target_dim=16)
    assert out.shape == (16,)
    assert torch.all(out == 0.0)


# ---------- storage_bytes ----------

def test_storage_bytes_monotonic():
    """Storage shrinks (or stays flat at L5=0) as the level
    increases. Catches a swapped table or a bookkeeping typo."""
    dim = 128
    sizes = [
        estimate_storage_bytes(lvl, dim) for lvl in PrecisionLevel
    ]
    # L0 ≥ L1 ≥ L2 ≥ L3 ≥ L4 ≥ L5.
    for prev, cur in zip(sizes, sizes[1:]):
        assert prev >= cur


def test_storage_bytes_specific_values():
    """Pin the exact-byte values per spec — protects against
    drift if the quantisation representation changes."""
    dim = 16
    assert estimate_storage_bytes(PrecisionLevel.L0, dim) == 16 * 4
    assert estimate_storage_bytes(PrecisionLevel.L1, dim) == 16 * 2
    assert estimate_storage_bytes(PrecisionLevel.L2, dim) == 16 + 4
    assert estimate_storage_bytes(PrecisionLevel.L3, dim) == 16 // 2 + 4
    assert estimate_storage_bytes(PrecisionLevel.L4, dim) == 16 // 8
    assert estimate_storage_bytes(PrecisionLevel.L5, dim) == 0


# ---------- precision modifier ----------

def test_precision_modifier_monotonic_decreasing():
    """Higher precision → larger modifier. L5 zeroes it out
    (effectively un-retrievable)."""
    vals = [PRECISION_MODIFIER[lvl] for lvl in PrecisionLevel]
    for prev, cur in zip(vals, vals[1:]):
        assert prev >= cur
    assert PRECISION_MODIFIER[PrecisionLevel.L0] == 1.0
    assert PRECISION_MODIFIER[PrecisionLevel.L5] == 0.0


# ---------- serialize_facts ----------

def test_serialize_facts_deterministic():
    """Identical fact dicts (regardless of insertion order) must
    serialise to identical strings — reconsolidation depends on
    this for stable re-encoding."""
    a = {"name": "Francois", "location": "Montreal", "age": 30}
    b = {"age": 30, "location": "Montreal", "name": "Francois"}
    assert serialize_facts(a) == serialize_facts(b)


def test_serialize_facts_handles_lists():
    """List / tuple values get joined with ", " in insertion
    order (so the serialiser is fully deterministic)."""
    facts = {"preferences": ["coffee", "short answers"]}
    out = serialize_facts(facts)
    assert out == "preferences: coffee, short answers"


def test_serialize_facts_handles_empty_dict():
    """Empty dict → empty string. Not used in practice (empty
    fact dicts aren't stored) but the serialiser shouldn't
    crash on the edge case."""
    assert serialize_facts({}) == ""


def test_serialize_facts_handles_scalar_values():
    facts = {"age": 30, "name": "Francois"}
    out = serialize_facts(facts)
    assert "age: 30" in out
    assert "name: Francois" in out
    # Keys are sorted alphabetically.
    assert out == "age: 30; name: Francois"


# ---------- error handling ----------

def test_unknown_level_raises_in_quantize():
    """Defensive: a bogus IntEnum-shaped value must raise rather
    than silently dropping data."""
    with pytest.raises(ValueError):
        quantize_to_level(torch.zeros(8), 99)  # type: ignore[arg-type]


def test_unknown_level_raises_in_dequantize():
    with pytest.raises(ValueError):
        dequantize_to_float32(None, 99, target_dim=8)  # type: ignore[arg-type]
