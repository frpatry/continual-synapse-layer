"""Tests for the consolidation pass (decay + merge + stub)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest
import torch

from agi.memory.consolidation import (
    apply_precision_decay,
    consolidate,
    detect_contradictions,
    merge_similar_entries,
)
from agi.memory.precision import (
    DECAY_SCHEDULE,
    PrecisionLevel,
)
from agi.memory.xray_episodic import EpisodicEntry, XRayEpisodicMemory


# ---------- helpers ----------

def _seeded_key(dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm()


def _make_entry(
    dim: int = 16,
    seed: int = 0,
    last_accessed: datetime | None = None,
    access_count: int = 0,
    facts: dict | None = None,
    precision_level: PrecisionLevel = PrecisionLevel.L0,
) -> EpisodicEntry:
    key = _seeded_key(dim, seed)
    ts = datetime.now()
    return EpisodicEntry(
        key=key,
        facts=facts or {"name": "X"},
        timestamp=ts,
        last_accessed=last_accessed or ts,
        access_count=access_count,
        precision_level=precision_level,
        embedding_dim=dim,
    )


# ---------- apply_precision_decay ----------

def test_decay_L0_to_L1_after_threshold():
    """An L0 entry idle past the L0 threshold (1 day) decays."""
    mem = XRayEpisodicMemory(key_dim=8)
    stale = _make_entry(
        dim=8,
        last_accessed=datetime.now() - DECAY_SCHEDULE[PrecisionLevel.L0] * 2,
    )
    mem.entries.append(stale)
    apply_precision_decay(mem)
    assert stale.precision_level == PrecisionLevel.L1


def test_decay_does_not_demote_fresh_entries():
    """A just-accessed entry stays put."""
    mem = XRayEpisodicMemory(key_dim=8)
    fresh = _make_entry(dim=8, last_accessed=datetime.now())
    mem.entries.append(fresh)
    apply_precision_decay(mem)
    assert fresh.precision_level == PrecisionLevel.L0


def test_decay_skips_L5_terminal_level():
    """Already at L5 — no further demotion possible."""
    mem = XRayEpisodicMemory(key_dim=8)
    bottomed_out = _make_entry(
        dim=8,
        last_accessed=datetime.now() - timedelta(days=10_000),
        precision_level=PrecisionLevel.L5,
    )
    mem.entries.append(bottomed_out)
    apply_precision_decay(mem)
    assert bottomed_out.precision_level == PrecisionLevel.L5


def test_decay_respects_importance_weighting():
    """An entry with a large access_count gets a softened
    threshold and survives an idle period that would have
    demoted an unused entry."""
    base = DECAY_SCHEDULE[PrecisionLevel.L0]
    idle = base * 2  # would decay an access_count=0 entry

    mem = XRayEpisodicMemory(key_dim=8)
    untouched = _make_entry(
        dim=8, seed=0,
        last_accessed=datetime.now() - idle,
        access_count=0,
    )
    popular = _make_entry(
        dim=8, seed=1,
        last_accessed=datetime.now() - idle,
        access_count=10_000,  # importance ~ 1 + log(10001) ≈ 10
    )
    mem.entries.extend([untouched, popular])

    apply_precision_decay(mem)
    assert untouched.precision_level == PrecisionLevel.L1
    assert popular.precision_level == PrecisionLevel.L0


def test_decay_uses_log_importance_formula():
    """Spot check the importance formula: idle ~ 1.5x base must
    still demote an access_count=0 entry but should NOT demote
    an entry whose importance factor exceeds 1.5."""
    mem = XRayEpisodicMemory(key_dim=8)
    base = DECAY_SCHEDULE[PrecisionLevel.L0]
    # 1 + log1p(access) > 1.5  →  log1p(access) > 0.5  →
    # access > exp(0.5) - 1 ≈ 0.65 → access_count = 1 already
    # gives importance ≈ 1.69.
    boundary = _make_entry(
        dim=8, seed=2,
        last_accessed=datetime.now() - base * 1.5,
        access_count=1,
    )
    mem.entries.append(boundary)
    apply_precision_decay(mem)
    assert boundary.precision_level == PrecisionLevel.L0


# ---------- consolidate (scope dispatch) ----------

def test_consolidation_light_only_decays(monkeypatch):
    """``scope="light"`` runs decay but skips merge + stub."""
    mem = XRayEpisodicMemory(key_dim=8)
    calls: list[str] = []

    def _track_merge(_m, **__):
        calls.append("merge")
        return 0

    def _track_detect(_m):
        calls.append("detect")
        return 0

    monkeypatch.setattr(
        "agi.memory.consolidation.merge_similar_entries", _track_merge,
    )
    monkeypatch.setattr(
        "agi.memory.consolidation.detect_contradictions", _track_detect,
    )
    consolidate(mem, scope="light")
    assert calls == []


def test_consolidation_full_runs_all_steps(monkeypatch):
    """``scope="full"`` exercises decay + merge + detect."""
    mem = XRayEpisodicMemory(key_dim=8)
    calls: list[str] = []
    monkeypatch.setattr(
        "agi.memory.consolidation.apply_precision_decay",
        lambda m: calls.append("decay"),
    )
    monkeypatch.setattr(
        "agi.memory.consolidation.merge_similar_entries",
        lambda m, **_: calls.append("merge") or 0,
    )
    monkeypatch.setattr(
        "agi.memory.consolidation.detect_contradictions",
        lambda m: calls.append("detect") or 0,
    )
    consolidate(mem, scope="full")
    assert calls == ["decay", "merge", "detect"]


def test_consolidate_rejects_unknown_scope():
    with pytest.raises(ValueError):
        consolidate(XRayEpisodicMemory(key_dim=8), scope="medium")


def test_consolidate_updates_last_consolidation_timestamp():
    mem = XRayEpisodicMemory(key_dim=8)
    before = mem.last_consolidation
    consolidate(mem, scope="light")
    assert mem.last_consolidation > before


# ---------- merge_similar_entries ----------

def test_merge_similar_entries_keeps_higher_precision():
    """When two near-duplicate entries exist at different
    precision levels, the higher-precision (lower-number) one
    survives and absorbs the loser's access_count."""
    mem = XRayEpisodicMemory(key_dim=16)
    k = _seeded_key(16, 42)
    survivor = EpisodicEntry(
        key=k.clone(), facts={"a": 1},
        timestamp=datetime.now(),
        access_count=5,
        precision_level=PrecisionLevel.L0,
        embedding_dim=16,
    )
    # Loser is the SAME key (cosine = 1.0) but at L2.
    loser = EpisodicEntry(
        key=None,
        facts={"b": 2},
        timestamp=datetime.now(),
        access_count=3,
        precision_level=PrecisionLevel.L0,
        embedding_dim=16,
    )
    loser.compress_to(PrecisionLevel.L2, source_embedding=k.clone())
    mem.entries.extend([survivor, loser])

    removed = merge_similar_entries(mem, similarity_threshold=0.95)
    assert removed == 1
    assert len(mem.entries) == 1
    kept = mem.entries[0]
    assert kept.precision_level == PrecisionLevel.L0
    # Loser's access_count rolled in.
    assert kept.access_count == 8
    # Facts merged (non-overlapping scalar keys both preserved).
    assert kept.facts == {"a": 1, "b": 2}


def test_merge_similar_entries_no_op_when_below_threshold():
    """Two orthogonal-ish keys should not merge."""
    mem = XRayEpisodicMemory(key_dim=16)
    e1 = _make_entry(dim=16, seed=10, facts={"name": "A"})
    e2 = _make_entry(dim=16, seed=11, facts={"name": "B"})
    mem.entries.extend([e1, e2])
    removed = merge_similar_entries(mem, similarity_threshold=0.95)
    assert removed == 0
    assert len(mem.entries) == 2


def test_merge_similar_entries_skips_L5():
    """L5 entries (existence-only) are ineligible for merging."""
    mem = XRayEpisodicMemory(key_dim=16)
    k = _seeded_key(16, 0)
    alive = EpisodicEntry(
        key=k, facts={"a": 1}, timestamp=datetime.now(),
        embedding_dim=16,
    )
    ghost = EpisodicEntry(
        key=None, facts={"a": 1}, timestamp=datetime.now(),
        precision_level=PrecisionLevel.L5,
        embedding_dim=16,
    )
    mem.entries.extend([alive, ghost])
    removed = merge_similar_entries(mem, similarity_threshold=0.5)
    assert removed == 0


# ---------- detect_contradictions stub ----------

def test_contradiction_detection_stub_logs():
    """The stub doesn't *resolve* contradictions — it just
    records them in ``memory.contradiction_log``."""
    mem = XRayEpisodicMemory(key_dim=8)
    e1 = _make_entry(dim=8, seed=0, facts={"name": "Francois"})
    e2 = _make_entry(dim=8, seed=1, facts={"name": "Marie"})
    mem.entries.extend([e1, e2])

    n = detect_contradictions(mem)
    assert n == 1
    assert mem.contradiction_log[-1]["key"] == "name"
    assert set(mem.contradiction_log[-1]["values"]) == {"Francois", "Marie"}


def test_contradiction_detection_ignores_list_values():
    """Lists / dicts get skipped — comparing them for
    contradiction requires semantic logic the stub doesn't
    pretend to have."""
    mem = XRayEpisodicMemory(key_dim=8)
    e1 = _make_entry(
        dim=8, seed=0, facts={"preferences": ["coffee"]},
    )
    e2 = _make_entry(
        dim=8, seed=1, facts={"preferences": ["tea"]},
    )
    mem.entries.extend([e1, e2])
    n = detect_contradictions(mem)
    assert n == 0


def test_contradiction_detection_groups_by_key():
    """When the same scalar key appears with three distinct
    values across three entries, one log entry is added (not
    three pair-wise)."""
    mem = XRayEpisodicMemory(key_dim=8)
    mem.entries.extend([
        _make_entry(dim=8, seed=0, facts={"city": "Paris"}),
        _make_entry(dim=8, seed=1, facts={"city": "Lyon"}),
        _make_entry(dim=8, seed=2, facts={"city": "Marseille"}),
    ])
    n = detect_contradictions(mem)
    assert n == 1
    assert set(mem.contradiction_log[-1]["values"]) == {
        "Paris", "Lyon", "Marseille",
    }
