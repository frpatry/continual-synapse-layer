"""Compression pipeline for cold-storage entries.

DESIGN.md §3.3 calls for progressive quantization — recently
archived clusters stay at full precision; older and rarely-accessed
clusters get squeezed down to 4-bit. This module implements the
quantize/dequantize primitives and a schedule that picks the
precision from ``age`` and ``access_count``.

Supported precisions and their byte layout (little-endian):

| precision | size for ``N`` floats | layout                              |
|-----------|-----------------------|-------------------------------------|
| 32        | ``4 * N``             | raw float32                         |
| 16        | ``2 * N``             | raw float16 (cast from float32)     |
| 8         | ``4 + N``             | ``float32 scale`` + int8 values     |
| 4         | ``4 + ceil(N / 2)``   | ``float32 scale`` + packed int4     |

Quantization at 8 and 4 bits is symmetric per-tensor: every value
is clipped to a signed integer range and divided by a scale that
maps the tensor's max-abs to the integer max. This is the
standard "max-abs quantization" used in most neural-network
post-training quantization recipes.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


VALID_PRECISIONS = (32, 16, 8, 4)
_INT_RANGES = {8: 127, 4: 7}


def quantize(x: torch.Tensor, precision: int) -> bytes:
    """Serialise ``x`` (any shape) into the byte layout for ``precision``.

    The caller is responsible for tracking the original tensor
    shape — ``dequantize`` needs it.

    Raises ``ValueError`` for unsupported precisions.
    """
    if precision not in VALID_PRECISIONS:
        raise ValueError(
            f"precision must be one of {VALID_PRECISIONS}, got {precision}"
        )
    arr = x.detach().to(torch.float32).contiguous().cpu().numpy()

    if precision == 32:
        return arr.tobytes()
    if precision == 16:
        return arr.astype(np.float16).tobytes()

    # 8-bit and 4-bit: symmetric max-abs quantization.
    int_max = _INT_RANGES[precision]
    max_abs = float(np.abs(arr).max()) if arr.size else 0.0
    scale = max_abs / int_max if max_abs > 0.0 else 1.0
    q = np.clip(np.round(arr / scale), -int_max, int_max).astype(np.int8)

    header = struct.pack("<f", scale)
    if precision == 8:
        return header + q.tobytes()

    # 4-bit: pack two signed 4-bit values per byte. We map signed
    # int4 [-7, 7] into the low nibble via two's-complement masking.
    flat = q.flatten()
    if flat.size % 2 == 1:
        flat = np.concatenate([flat, np.zeros(1, dtype=np.int8)])
    low = (flat[0::2] & 0x0F).astype(np.uint8)
    high = (flat[1::2] & 0x0F).astype(np.uint8) << 4
    packed = (low | high).tobytes()
    return header + packed


def dequantize(
    data: bytes, precision: int, shape: Iterable[int]
) -> torch.Tensor:
    """Inverse of :func:`quantize`.

    Args:
        data: Byte string produced by ``quantize``.
        precision: Same precision used for ``quantize``.
        shape: Original tensor shape (the bytes contain no shape
            information; callers persist this separately).
    """
    if precision not in VALID_PRECISIONS:
        raise ValueError(
            f"precision must be one of {VALID_PRECISIONS}, got {precision}"
        )
    shape_tuple = tuple(int(s) for s in shape)
    n = int(np.prod(shape_tuple)) if shape_tuple else 0

    if precision == 32:
        arr = np.frombuffer(data, dtype=np.float32).copy()
        return torch.from_numpy(arr).reshape(shape_tuple)
    if precision == 16:
        arr = np.frombuffer(data, dtype=np.float16).astype(np.float32).copy()
        return torch.from_numpy(arr).reshape(shape_tuple)

    scale = struct.unpack("<f", data[:4])[0]
    body = data[4:]

    if precision == 8:
        q = np.frombuffer(body, dtype=np.int8).copy()
        return torch.from_numpy((q.astype(np.float32) * scale)).reshape(
            shape_tuple
        )

    # 4-bit unpacking. Each byte holds two values; the low nibble of
    # the byte is the even-index value, the high nibble is the
    # odd-index value. Sign-extend the 4-bit value to int8.
    packed = np.frombuffer(body, dtype=np.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    # Sign-extend 4-bit to int8: values >= 8 are negative.
    low_signed = np.where(low >= 8, low.astype(np.int16) - 16, low.astype(np.int16))
    high_signed = np.where(high >= 8, high.astype(np.int16) - 16, high.astype(np.int16))
    interleaved = np.empty(packed.size * 2, dtype=np.int16)
    interleaved[0::2] = low_signed
    interleaved[1::2] = high_signed
    flat = interleaved[:n].astype(np.float32) * scale
    return torch.from_numpy(flat.copy()).reshape(shape_tuple)


def byte_size(num_elements: int, precision: int) -> int:
    """Return the byte size for ``num_elements`` at ``precision``."""
    if precision == 32:
        return 4 * num_elements
    if precision == 16:
        return 2 * num_elements
    if precision == 8:
        return 4 + num_elements
    if precision == 4:
        return 4 + (num_elements + 1) // 2
    raise ValueError(precision)


@dataclass
class CompressionSchedule:
    """Pick a precision from ``(age, access_count)``.

    The default schedule mirrors the qualitative description in
    DESIGN.md §3.3: recent or recently-accessed entries stay at
    higher precision; old and rarely-accessed entries are squeezed
    down progressively.

    Attributes:
        age_thresholds: Boundaries between precision tiers, in
            *ascending* age order, paired with ``tier_precisions``.
            ``age < age_thresholds[i]`` maps to
            ``tier_precisions[i]``.
        tier_precisions: One precision per tier, plus one final
            value for ages beyond the last threshold.
        access_count_floor: If ``access_count`` exceeds this, the
            schedule bumps precision up one tier (to a maximum of
            32) — frequently-accessed entries stay sharper for
            longer.
    """

    age_thresholds: tuple[int, ...] = (100, 500, 2000)
    tier_precisions: tuple[int, ...] = (32, 16, 8, 4)
    access_count_floor: int = 5

    def __post_init__(self) -> None:
        if len(self.tier_precisions) != len(self.age_thresholds) + 1:
            raise ValueError(
                "tier_precisions must have len(age_thresholds) + 1 entries"
            )
        for p in self.tier_precisions:
            if p not in VALID_PRECISIONS:
                raise ValueError(f"unknown precision in schedule: {p}")
        for a, b in zip(self.age_thresholds, self.age_thresholds[1:]):
            if a >= b:
                raise ValueError("age_thresholds must be strictly increasing")

    def precision_for(self, age: int, access_count: int) -> int:
        """Look up the precision tier and apply the access-count bump."""
        tier = len(self.tier_precisions) - 1
        for i, threshold in enumerate(self.age_thresholds):
            if age < threshold:
                tier = i
                break
        if access_count >= self.access_count_floor and tier > 0:
            tier -= 1
        return self.tier_precisions[tier]
