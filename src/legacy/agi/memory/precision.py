"""Precision levels + quantisation for the X-Ray episodic memory.

Phase 2c bis adds *graded* precision to memory storage. Fresh
entries live at ``L0`` (Float32) and are downgraded one level at
a time (``L0 → L1 → … → L4 → L5``) as they idle, then promoted
back up by one level whenever they're successfully retrieved.
The result is a self-organising store where frequently-accessed
facts live at high precision and rarely-accessed ones degrade
toward an existence trace.

This module is intentionally pure-numerical — no datetime / no
side effects beyond the quantisation maths and a string
serialiser used by the reconsolidation path. The lifecycle
policy (when to decay, when to promote, when to consolidate)
lives in :mod:`agi.memory.consolidation` and
:mod:`agi.memory.xray_episodic`.

Numerical-stability convention: a single ``_EPS = 1e-8`` is used
everywhere a divide-by-zero could otherwise occur.
"""

from __future__ import annotations

from datetime import timedelta
from enum import IntEnum
from typing import Any

import torch


_EPS: float = 1e-8


class PrecisionLevel(IntEnum):
    """Six discrete precision tiers for memory storage.

    The integer values are ordinal: lower number = higher
    precision. The lifecycle code uses ``int(level) ± 1`` to
    walk the ladder.
    """

    L0 = 0  # Float32 — full precision
    L1 = 1  # Float16 — ~99% semantic preserved
    L2 = 2  # Int8    — ~95% semantic preserved
    L3 = 3  # Int4    — ~85% semantic preserved
    L4 = 4  # Binary  — sign only (1 bit per dim)
    L5 = 5  # Existence trace — not retrievable by similarity


DECAY_SCHEDULE: dict[PrecisionLevel, timedelta] = {
    PrecisionLevel.L0: timedelta(days=1),
    PrecisionLevel.L1: timedelta(days=7),
    PrecisionLevel.L2: timedelta(days=30),
    PrecisionLevel.L3: timedelta(days=180),
    PrecisionLevel.L4: timedelta(days=730),
}
"""Idle-time threshold per level. An entry at level ``L`` that
hasn't been accessed for longer than ``DECAY_SCHEDULE[L]`` (after
importance weighting) is compressed to ``L+1`` on the next
consolidation pass. ``L5`` has no entry — already at the bottom.
"""

PRECISION_MODIFIER: dict[PrecisionLevel, float] = {
    PrecisionLevel.L0: 1.0,
    PrecisionLevel.L1: 0.9,
    PrecisionLevel.L2: 0.8,
    PrecisionLevel.L3: 0.7,
    PrecisionLevel.L4: 0.6,
    PrecisionLevel.L5: 0.0,
}
"""Multiplicative penalty applied to raw cosine similarity at
retrieval time. An entry at ``L3`` matches with only 70% of its
true similarity score — encouraging the retriever to prefer
high-precision entries when both are present.
"""

RECONSOLIDATION_BLEND_RATIO: float = 0.1
"""Fraction of the current query embedding blended into a fact's
refreshed key during reconsolidation. The rest (``0.9``) comes
from re-encoding the entry's stable ``facts`` dict via the
foundation. The blend is what lets repeated retrievals
progressively enrich a memory's contextual associations — see
the *mustache analogy* in
:meth:`XRayEpisodicMemory._reconsolidate`.
"""


# ----------------------------------------------------------------------
# Quantisation primitives
# ----------------------------------------------------------------------

def quantize_to_level(embedding: torch.Tensor, target_level: PrecisionLevel) -> Any:
    """Quantise a Float32 1-D embedding to the target level.

    Returns the storage representation appropriate for the level:

    - ``L0`` — the input tensor (caller is responsible for clone).
    - ``L1`` — a Float16 tensor.
    - ``L2`` — ``(int8_tensor, scale_float)``.
    - ``L3`` — ``(packed_uint8_tensor, scale_float, orig_dim_int)``.
    - ``L4`` — ``(packed_uint8_tensor, orig_dim_int)``.
    - ``L5`` — ``None`` (existence-only).

    The quantisation is per-tensor symmetric for L2 / L3, sign
    only for L4. Bit-packing for L3 / L4 rounds the storage size
    up to the nearest byte when the dim isn't a multiple of 2 / 8;
    ``orig_dim`` is kept so dequantisation can trim the trailing
    pad bits cleanly.
    """
    x = embedding.detach().to(torch.float32).flatten().cpu()
    dim = int(x.numel())

    if target_level == PrecisionLevel.L0:
        return x.clone()

    if target_level == PrecisionLevel.L1:
        return x.to(torch.float16)

    if target_level == PrecisionLevel.L2:
        max_abs = float(x.abs().max().item())
        scale = max(max_abs / 127.0, _EPS)
        ints = torch.round(x / scale).clamp(-128, 127).to(torch.int8)
        return (ints, scale)

    if target_level == PrecisionLevel.L3:
        max_abs = float(x.abs().max().item())
        scale = max(max_abs / 7.0, _EPS)
        ints = torch.round(x / scale).clamp(-8, 7).to(torch.int8)
        # Shift to 0..15, pack pairs into uint8 nibbles.
        nibbles = (ints + 8).to(torch.uint8)
        if nibbles.numel() % 2 == 1:
            nibbles = torch.cat(
                [nibbles, torch.zeros(1, dtype=torch.uint8)],
            )
        low = nibbles[0::2]
        high = nibbles[1::2]
        packed = (low | (high << 4)).to(torch.uint8)
        return (packed, scale, dim)

    if target_level == PrecisionLevel.L4:
        # Sign bit only: 1 → +1, 0 → -1.
        bits = (x >= 0).to(torch.uint8)
        n_pad = (8 - (dim % 8)) % 8
        if n_pad:
            bits = torch.cat([bits, torch.zeros(n_pad, dtype=torch.uint8)])
        # Pack 8 bits per byte (bit i → position i within the byte).
        bits_2d = bits.view(-1, 8)
        weights = (1 << torch.arange(8, dtype=torch.uint8))
        packed = (bits_2d * weights).sum(dim=1).to(torch.uint8)
        return (packed, dim)

    if target_level == PrecisionLevel.L5:
        return None

    raise ValueError(f"unknown precision level: {target_level!r}")


def dequantize_to_float32(
    stored: Any,
    level: PrecisionLevel,
    target_dim: int,
) -> torch.Tensor:
    """Reinflate quantised storage back to a Float32 1-D tensor.

    Information lost during quantisation is NOT recovered; this
    routine just gets the storage back into a form the cosine
    retriever can consume. For ``L5`` the function returns a
    zero vector of length ``target_dim`` — these entries are
    skipped by the retriever anyway.
    """
    if level == PrecisionLevel.L0:
        return stored.to(torch.float32) if isinstance(stored, torch.Tensor) \
            else torch.zeros(target_dim, dtype=torch.float32)

    if level == PrecisionLevel.L1:
        return stored.to(torch.float32)

    if level == PrecisionLevel.L2:
        ints, scale = stored
        return ints.to(torch.float32) * float(scale)

    if level == PrecisionLevel.L3:
        packed, scale, orig_dim = stored
        low = (packed & 0x0F).to(torch.int16)
        high = ((packed >> 4) & 0x0F).to(torch.int16)
        pairs = torch.stack([low, high], dim=1).flatten()
        # Undo the +8 shift, trim trailing pad to orig_dim.
        ints = pairs[:orig_dim] - 8
        return ints.to(torch.float32) * float(scale)

    if level == PrecisionLevel.L4:
        packed, orig_dim = stored
        bits_2d = (
            packed.unsqueeze(1)
            >> torch.arange(8, dtype=torch.uint8).unsqueeze(0)
        ) & 1
        bits = bits_2d.flatten()[:orig_dim].to(torch.float32)
        # 0 → -1, 1 → +1.
        return bits * 2.0 - 1.0

    if level == PrecisionLevel.L5:
        return torch.zeros(target_dim, dtype=torch.float32)

    raise ValueError(f"unknown precision level: {level!r}")


def estimate_storage_bytes(level: PrecisionLevel, embedding_dim: int) -> int:
    """Return the storage footprint at ``level`` for an embedding
    of dimensionality ``embedding_dim``.

    Used by tests + diagnostics; not a strict on-disk bound
    (Python object overhead is excluded — these are the *payload*
    sizes that motivated the precision ladder in the first place).
    """
    sizes = {
        PrecisionLevel.L0: embedding_dim * 4,
        PrecisionLevel.L1: embedding_dim * 2,
        PrecisionLevel.L2: embedding_dim * 1 + 4,        # int8 + float32 scale
        PrecisionLevel.L3: (embedding_dim + 1) // 2 + 4,  # nibble-packed + scale
        PrecisionLevel.L4: (embedding_dim + 7) // 8,      # bit-packed
        PrecisionLevel.L5: 0,
    }
    return int(sizes[level])


# ----------------------------------------------------------------------
# Fact-dict → text helper (used by reconsolidation)
# ----------------------------------------------------------------------

def serialize_facts(facts: dict) -> str:
    """Stable, deterministic text rendering of a fact dict.

    Used during reconsolidation: the foundation re-encodes this
    string to produce a "fresh" embedding that the entry's new
    key is blended from. Determinism matters because the same
    fact dict at two different reconsolidation events must
    produce *the same* embedding — otherwise the entry's key
    would drift even without contextual updates.

    Format: ``key1: val1; key2: val2``. Keys are sorted; list /
    tuple values are joined with ``", "``.
    """
    parts: list[str] = []
    for key in sorted(facts.keys()):
        value = facts[key]
        if isinstance(value, (list, tuple)):
            value_str = ", ".join(str(v) for v in value)
        else:
            value_str = str(value)
        parts.append(f"{key}: {value_str}")
    return "; ".join(parts)
