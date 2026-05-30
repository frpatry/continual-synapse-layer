"""Tests for the cold-storage compression pipeline."""

from __future__ import annotations

import pytest
import torch

from continual_synapse.cold_storage.compression import (
    CompressionSchedule,
    byte_size,
    dequantize,
    quantize,
)


def test_32bit_round_trip_is_exact() -> None:
    x = torch.tensor([1.5, -2.0, 3.25, 0.0, 100.0])
    data = quantize(x, precision=32)
    restored = dequantize(data, precision=32, shape=x.shape)
    torch.testing.assert_close(restored, x)


def test_16bit_round_trip_is_close() -> None:
    x = torch.tensor([1.0, -2.5, 3.14159, 0.0])
    data = quantize(x, precision=16)
    restored = dequantize(data, precision=16, shape=x.shape)
    # fp16 has limited precision; tolerate the conversion error.
    torch.testing.assert_close(restored, x, atol=1e-3, rtol=1e-3)


def test_8bit_round_trip_is_approximate() -> None:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, generator=g) * 2.0
    data = quantize(x, precision=8)
    restored = dequantize(data, precision=8, shape=x.shape)
    # Symmetric int8 with max-abs ~6 gives step size ~6/127 ≈ 0.047
    # and worst-case half-step error ≈ 0.024. Allow a small margin.
    max_abs = x.abs().max().item()
    step = max_abs / 127.0
    torch.testing.assert_close(restored, x, atol=step, rtol=0.1)


def test_4bit_round_trip_is_approximate() -> None:
    g = torch.Generator().manual_seed(1)
    x = torch.randn(32, generator=g)
    data = quantize(x, precision=4)
    restored = dequantize(data, precision=4, shape=x.shape)
    # 4-bit signed quantization is coarse — the test only needs
    # the result to be roughly the same as the original, with
    # mean error well under 0.5 (step size ≈ max_abs / 7).
    err = (restored - x).abs().mean().item()
    assert err < 0.5


def test_4bit_round_trip_handles_odd_size() -> None:
    x = torch.tensor([1.0, -1.0, 0.5, 0.25, -0.5])  # length 5
    data = quantize(x, precision=4)
    restored = dequantize(data, precision=4, shape=x.shape)
    assert restored.shape == x.shape


def test_byte_size_matches_actual_bytes() -> None:
    g = torch.Generator().manual_seed(2)
    x = torch.randn(20, generator=g)
    for p in (32, 16, 8, 4):
        data = quantize(x, precision=p)
        assert len(data) == byte_size(num_elements=20, precision=p)


def test_compression_ratios_are_decreasing() -> None:
    """Lower precision should produce smaller byte strings."""
    g = torch.Generator().manual_seed(3)
    x = torch.randn(64, generator=g)
    sizes = [len(quantize(x, precision=p)) for p in (32, 16, 8, 4)]
    assert sizes == sorted(sizes, reverse=True)


def test_zero_tensor_quantizes_without_crash() -> None:
    """Max-abs = 0 needs a guarded scale; rolling back must give zeros."""
    x = torch.zeros(10)
    for p in (8, 4):
        data = quantize(x, precision=p)
        restored = dequantize(data, precision=p, shape=x.shape)
        torch.testing.assert_close(restored, x)


def test_shape_round_trip_for_multi_d() -> None:
    g = torch.Generator().manual_seed(4)
    x = torch.randn(3, 5, 7, generator=g)
    data = quantize(x, precision=16)
    restored = dequantize(data, precision=16, shape=x.shape)
    assert restored.shape == x.shape


def test_quantize_rejects_invalid_precision() -> None:
    with pytest.raises(ValueError, match="precision"):
        quantize(torch.zeros(3), precision=12)
    with pytest.raises(ValueError, match="precision"):
        dequantize(b"", precision=2, shape=(3,))


# ---- schedule ----


def test_schedule_returns_high_precision_for_low_age() -> None:
    sched = CompressionSchedule()
    assert sched.precision_for(age=0, access_count=0) == 32
    assert sched.precision_for(age=99, access_count=0) == 32


def test_schedule_decreases_with_age() -> None:
    sched = CompressionSchedule()
    assert sched.precision_for(age=200, access_count=0) == 16
    assert sched.precision_for(age=800, access_count=0) == 8
    assert sched.precision_for(age=5000, access_count=0) == 4


def test_access_count_bumps_precision_up_one_tier() -> None:
    sched = CompressionSchedule()
    # Tier for age=200 is 16. With many accesses, bump to 32.
    assert sched.precision_for(age=200, access_count=20) == 32
    # Tier for age=5000 is 4. With many accesses, bump to 8.
    assert sched.precision_for(age=5000, access_count=20) == 8


def test_access_count_cannot_bump_past_highest_tier() -> None:
    sched = CompressionSchedule()
    # Already at 32 — no further bump.
    assert sched.precision_for(age=0, access_count=10_000) == 32


def test_schedule_rejects_inconsistent_args() -> None:
    with pytest.raises(ValueError, match="tier_precisions"):
        CompressionSchedule(
            age_thresholds=(100, 500),
            tier_precisions=(32, 16),  # should have 3 entries
        )
    with pytest.raises(ValueError, match="increasing"):
        CompressionSchedule(
            age_thresholds=(500, 100, 200),
            tier_precisions=(32, 16, 8, 4),
        )
    with pytest.raises(ValueError, match="unknown precision"):
        CompressionSchedule(
            age_thresholds=(100,),
            tier_precisions=(32, 12),
        )
