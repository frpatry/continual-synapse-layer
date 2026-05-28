"""Tests for active reconsolidation on retrieval (Phase 2c bis)."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import torch

from agi.memory.consolidation import consolidate
from agi.memory.precision import (
    DECAY_SCHEDULE,
    PrecisionLevel,
    RECONSOLIDATION_BLEND_RATIO,
    serialize_facts,
)
from agi.memory.xray_episodic import EpisodicEntry, XRayEpisodicMemory


# ---------- Mock foundation ----------

class _MockFoundation:
    """Deterministic foundation stub: ``get_key(text)`` returns a
    unit vector seeded from the text hash. Tests can wrap to
    observe which texts were requested."""

    def __init__(self, key_dim: int = 16):
        self.key_dim = key_dim
        self.get_key_calls: list[str] = []

    def get_key(self, text: str) -> torch.Tensor:
        self.get_key_calls.append(text)
        h = abs(hash(text)) & 0xFFFFFFFF
        g = torch.Generator()
        g.manual_seed(h)
        v = torch.randn(self.key_dim, generator=g)
        return v / v.norm()


def _seeded_key(dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm()


def _make_mem(
    foundation: Optional[_MockFoundation] = None,
    threshold: float = -1.0,  # always retrieve
) -> XRayEpisodicMemory:
    return XRayEpisodicMemory(
        key_dim=16, retrieval_threshold=threshold, foundation=foundation,
    )


# ---------- Freshness-only at L0 ----------

def test_reconsolidation_at_L0_only_resets_freshness():
    """L0 entry can't be promoted higher — retrieval only
    refreshes access_count and last_accessed, the precision
    stays at L0."""
    fnd = _MockFoundation()
    mem = _make_mem(fnd)
    key = _seeded_key(16, 0)
    mem.add_entry(key, {"name": "Francois"})
    entry = mem.entries[0]
    before_count = entry.access_count
    fnd.get_key_calls.clear()

    out = mem.retrieve(key, top_k=1)
    assert len(out) == 1
    assert entry.precision_level == PrecisionLevel.L0
    assert entry.access_count == before_count + 1
    # No re-encoding call when already at L0.
    assert fnd.get_key_calls == []


# ---------- Promotion on access ----------

def test_reconsolidation_promotes_one_level():
    """An entry at L3 is promoted to L2 on a successful match."""
    fnd = _MockFoundation()
    mem = _make_mem(fnd)
    key = _seeded_key(16, 1)
    mem.add_entry(key, {"name": "Francois"})
    entry = mem.entries[0]
    # Hand-degrade to L3.
    entry.compress_to(PrecisionLevel.L3)
    assert entry.precision_level == PrecisionLevel.L3

    mem.retrieve(key, top_k=1)
    assert entry.precision_level == PrecisionLevel.L2


# ---------- Re-encoding source ----------

def test_reconsolidation_uses_fact_reencoding():
    """The refreshed key is computed from
    ``foundation.get_key(serialize_facts(entry.facts))``.
    Wrap the foundation to assert the exact text passed."""
    fnd = _MockFoundation()
    mem = _make_mem(fnd)
    key = _seeded_key(16, 2)
    mem.add_entry(key, {"name": "Francois", "location": "Montreal"})
    mem.entries[0].compress_to(PrecisionLevel.L2)
    fnd.get_key_calls.clear()

    mem.retrieve(key, top_k=1)
    # Exactly one re-encoding call with the serialised facts.
    expected_text = serialize_facts({
        "name": "Francois", "location": "Montreal",
    })
    assert fnd.get_key_calls == [expected_text]


# ---------- Blend ratio mixing ----------

def test_reconsolidation_blends_query_with_facts():
    """The refreshed key must be a 0.9 / 0.1 blend of
    ``foundation.get_key(facts_text)`` and the query."""
    fnd = _MockFoundation()
    mem = _make_mem(fnd)
    key = _seeded_key(16, 3)
    mem.add_entry(key, {"name": "Francois"})
    entry = mem.entries[0]

    # Force entry to L1 so reconsolidation promotes it to L0
    # (storing the blended key as ``entry.key``).
    entry.compress_to(PrecisionLevel.L1)

    # Use a query orthogonal to the fact embedding.
    query = _seeded_key(16, 99)
    mem.retrieve(query, top_k=1)
    assert entry.precision_level == PrecisionLevel.L0

    # Reconstruct what the blend SHOULD be.
    fact_emb = fnd.get_key(serialize_facts({"name": "Francois"}))
    # The retrieve path normalises the query before passing it
    # to _reconsolidate (we explicitly read query_n at L173 of
    # xray_episodic.py), so reproduce that here.
    q_norm = query / query.norm()
    expected = (
        (1.0 - RECONSOLIDATION_BLEND_RATIO) * fact_emb
        + RECONSOLIDATION_BLEND_RATIO * q_norm
    )
    assert torch.allclose(entry.key, expected, atol=1e-5)


# ---------- Progressive promotion ----------

def test_progressive_promotion_through_levels():
    """A fact starting at L4, accessed 4 times, lands at L0."""
    fnd = _MockFoundation()
    mem = _make_mem(fnd)
    key = _seeded_key(16, 4)
    mem.add_entry(key, {"name": "Francois"})
    entry = mem.entries[0]
    entry.compress_to(PrecisionLevel.L4)
    assert entry.precision_level == PrecisionLevel.L4

    for _ in range(4):
        mem.retrieve(key, top_k=1)
    assert entry.precision_level == PrecisionLevel.L0


# ---------- Decay-then-reconsolidation cycle ----------

def test_decay_then_reconsolidation_recovers_some_precision():
    """A fresh L0 entry can decay to L2 via consolidation,
    then climb back to L1 on a successful retrieval — showing
    the full lifecycle."""
    fnd = _MockFoundation()
    mem = _make_mem(fnd)
    key = _seeded_key(16, 5)
    mem.add_entry(key, {"name": "Francois"})
    entry = mem.entries[0]

    # Simulate idle long enough to decay L0→L1, then L1→L2.
    # Two manual passes (the decay function demotes one level
    # at a time).
    far_past = datetime.now() - DECAY_SCHEDULE[PrecisionLevel.L1] * 5
    entry.last_accessed = far_past
    consolidate(mem, scope="light")  # → L1
    entry.last_accessed = far_past
    consolidate(mem, scope="light")  # → L2
    assert entry.precision_level == PrecisionLevel.L2

    # Retrieval reconsolidates → L1.
    mem.retrieve(key, top_k=1)
    assert entry.precision_level == PrecisionLevel.L1


# ---------- Self-organising distribution ----------

def test_auto_organization_pattern():
    """The whole point of precision-decay + reconsolidation:
    frequently-accessed facts cluster at high precision,
    untouched facts decay toward L4+.

    100 entries, half accessed many times (with backdated
    last_accessed kept fresh-ish via reconsolidation), half
    completely stale. After one consolidation cycle the
    distributions should diverge cleanly."""
    fnd = _MockFoundation()
    mem = _make_mem(fnd, threshold=-1.0)
    dim = 16

    # 50 popular facts.
    popular_keys = [_seeded_key(dim, 100 + i) for i in range(50)]
    for i, k in enumerate(popular_keys):
        mem.add_entry(k, {"popular": True, "id": i})

    # 50 untouched facts — backdate their last_accessed deep
    # into the past so consolidation cascades them through
    # multiple levels.
    untouched_keys = [_seeded_key(dim, 200 + i) for i in range(50)]
    for i, k in enumerate(untouched_keys):
        mem.add_entry(k, {"untouched": True, "id": i})
    very_old = datetime.now() - timedelta(days=10_000)
    for entry in mem.entries[50:]:
        entry.last_accessed = very_old

    # "Use" the popular facts. Each retrieval bumps
    # access_count and refreshes last_accessed.
    for _ in range(5):
        for k in popular_keys:
            mem.retrieve(k, top_k=1)

    # Run multiple consolidation passes (each demotes one level
    # for the untouched batch).
    for _ in range(5):
        for entry in mem.entries[50:]:
            entry.last_accessed = very_old
        consolidate(mem, scope="light")

    dist = mem.precision_distribution()
    # Popular entries should be at L0 (they were never idle long
    # enough to decay AND each retrieval keeps them fresh).
    popular_at_l0 = sum(
        1 for e in mem.entries[:50] if e.precision_level == PrecisionLevel.L0
    )
    assert popular_at_l0 >= 45, (
        f"expected ≥45 popular entries at L0, got {popular_at_l0}"
    )

    # Untouched entries should have decayed to at least L3.
    untouched_at_l3_or_worse = sum(
        1 for e in mem.entries[50:]
        if e.precision_level >= PrecisionLevel.L3
    )
    assert untouched_at_l3_or_worse >= 45, (
        f"expected ≥45 untouched entries at L3+, got "
        f"{untouched_at_l3_or_worse}. distribution: {dist}"
    )
