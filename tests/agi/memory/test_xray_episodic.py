"""Tests for XRayEpisodicMemory (privacy + retrieval mechanics +
Phase 2c bis precision lifecycle hooks)."""

from __future__ import annotations

from datetime import datetime, timedelta

import torch

from agi.memory import EpisodicEntry, PrecisionLevel, XRayEpisodicMemory


def _rand_key(dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(dim, generator=g)


def test_empty_memory_returns_no_results():
    mem = XRayEpisodicMemory(key_dim=32)
    assert len(mem) == 0
    out = mem.retrieve(_rand_key(32, 0))
    assert out == []


def test_add_and_retrieve_exact_match():
    """Storing a key and retrieving with the same key should
    return that entry with cosine sim ≈ 1.0."""
    mem = XRayEpisodicMemory(key_dim=32, retrieval_threshold=0.7)
    k = _rand_key(32, 0)
    mem.add_entry(k, {"name": "Francois"})
    out = mem.retrieve(k)
    assert len(out) == 1
    entry, sim = out[0]
    assert entry.facts == {"name": "Francois"}
    assert sim > 0.99  # same key → cosine ≈ 1


def test_retrieval_threshold_filters_dissimilar():
    """A query that's nearly orthogonal to all stored keys should
    yield no results when above the configured threshold."""
    mem = XRayEpisodicMemory(key_dim=32, retrieval_threshold=0.9)
    stored = torch.zeros(32); stored[0] = 1.0
    query  = torch.zeros(32); query[1]  = 1.0
    mem.add_entry(stored, {"name": "X"})
    assert mem.retrieve(query) == []


def test_empty_facts_does_not_store():
    """add_entry with an empty facts dict must be a no-op —
    there's nothing to index against."""
    mem = XRayEpisodicMemory(key_dim=32)
    mem.add_entry(_rand_key(32, 0), {})
    assert len(mem) == 0


def test_no_raw_text_field_on_entry():
    """Privacy contract: EpisodicEntry must not expose any field
    that holds the original utterance."""
    entry = EpisodicEntry(
        key=torch.zeros(4), facts={"name": "X"},
        timestamp=__import__("datetime").datetime.now(),
    )
    for forbidden in ("raw_text", "original_input", "raw", "samples", "utterance"):
        assert not hasattr(entry, forbidden), (
            f"Privacy violation — EpisodicEntry exposes {forbidden!r}"
        )


def test_access_count_increments_on_retrieval():
    mem = XRayEpisodicMemory(key_dim=8, retrieval_threshold=0.7)
    k = _rand_key(8, 0)
    mem.add_entry(k, {"name": "X"})
    mem.retrieve(k)
    mem.retrieve(k)
    mem.retrieve(k)
    assert mem.entries[0].access_count == 3


def test_merge_facts_higher_sim_wins_on_conflict():
    """When two entries have the same scalar key (e.g., name) but
    different values, the higher-similarity entry wins."""
    mem = XRayEpisodicMemory(key_dim=4, retrieval_threshold=0.0)
    high = EpisodicEntry(
        key=torch.tensor([1.0, 0, 0, 0]),
        facts={"name": "Marie"},
        timestamp=__import__("datetime").datetime.now(),
    )
    low = EpisodicEntry(
        key=torch.tensor([0.5, 0.5, 0, 0]),
        facts={"name": "Other"},
        timestamp=__import__("datetime").datetime.now(),
    )
    merged = mem.merge_facts([(high, 0.99), (low, 0.50)])
    assert merged["name"] == "Marie"


def test_merge_preferences_handles_str_or_list():
    """``preferences`` may arrive as a ``str`` (LLM extractor) or a
    ``list[str]`` (regex extractor). The merge must treat both as
    single units rather than iterating characters of the string.

    This guards against the Phase 1.1 bug where the demo's
    on-disk record showed
    ``preferences: ['T', 'e', 'c', 'h', 'n', 'o', 'l', ...]``
    instead of ``['Technology']`` — the merge loop ate the
    string letter by letter.
    """
    now = __import__("datetime").datetime.now()
    # LLM-style: one entry's prefs as a string, another's as a string.
    e_llm_a = EpisodicEntry(
        key=torch.zeros(4),
        facts={"preferences": "coffee in the morning"},
        timestamp=now,
    )
    e_llm_b = EpisodicEntry(
        key=torch.zeros(4),
        facts={"preferences": "short answers"},
        timestamp=now,
    )
    merged = XRayEpisodicMemory(key_dim=4).merge_facts(
        [(e_llm_a, 0.95), (e_llm_b, 0.90)]
    )
    assert merged["preferences"] == ["coffee in the morning", "short answers"]

    # Regex-style: both entries with lists.
    e_rx_a = EpisodicEntry(
        key=torch.zeros(4),
        facts={"preferences": ["coffee", "tea"]},
        timestamp=now,
    )
    e_rx_b = EpisodicEntry(
        key=torch.zeros(4),
        facts={"preferences": ["short answers", "coffee"]},
        timestamp=now,
    )
    merged = XRayEpisodicMemory(key_dim=4).merge_facts(
        [(e_rx_a, 0.95), (e_rx_b, 0.90)]
    )
    assert merged["preferences"] == ["coffee", "tea", "short answers"]

    # Mixed: one entry str, one list — should still concatenate cleanly.
    e_mix_str = EpisodicEntry(
        key=torch.zeros(4),
        facts={"preferences": "coffee"},
        timestamp=now,
    )
    e_mix_list = EpisodicEntry(
        key=torch.zeros(4),
        facts={"preferences": ["tea"]},
        timestamp=now,
    )
    merged = XRayEpisodicMemory(key_dim=4).merge_facts(
        [(e_mix_str, 0.95), (e_mix_list, 0.90)]
    )
    assert merged["preferences"] == ["coffee", "tea"]


def test_merge_facts_unions_preferences():
    """Preference lists across entries should union (preserving
    first-appearance order), not overwrite."""
    mem = XRayEpisodicMemory(key_dim=4, retrieval_threshold=0.0)
    now = __import__("datetime").datetime.now()
    e1 = EpisodicEntry(
        key=torch.zeros(4), facts={"preferences": ["coffee"]}, timestamp=now,
    )
    e2 = EpisodicEntry(
        key=torch.zeros(4), facts={"preferences": ["short answers"]}, timestamp=now,
    )
    merged = mem.merge_facts([(e1, 0.95), (e2, 0.90)])
    assert merged["preferences"] == ["coffee", "short answers"]


def test_new_session_increments_counter():
    mem = XRayEpisodicMemory(key_dim=4)
    assert mem.current_session == 0
    assert mem.new_session() == 1
    assert mem.new_session() == 2


def test_creation_session_recorded():
    mem = XRayEpisodicMemory(key_dim=4)
    mem.new_session()  # → 1
    e = mem.add_entry(_rand_key(4, 0), {"name": "X"})
    assert e is not None
    assert e.creation_session == 1


# ----------------------------------------------------------------------
# Phase 2c bis — precision lifecycle integration tests
# ----------------------------------------------------------------------


class _MockFoundationForReconsolidation:
    """Minimal foundation stub for the reconsolidation path —
    only ``get_key`` is exercised."""

    def __init__(self, key_dim: int):
        self.key_dim = key_dim
        self.calls: list[str] = []

    def get_key(self, text: str) -> torch.Tensor:
        self.calls.append(text)
        h = abs(hash(text)) & 0xFFFFFFFF
        g = torch.Generator()
        g.manual_seed(h)
        v = torch.randn(self.key_dim, generator=g)
        return v / v.norm()


def test_retrieve_applies_precision_modifier():
    """Two entries with identical keys but different precisions:
    the L0 entry must rank ABOVE the L3 entry because the
    precision modifier (1.0 vs 0.7) tips the score."""
    mem = XRayEpisodicMemory(key_dim=16, retrieval_threshold=-1.0)
    key = _rand_key(16, 7)
    mem.add_entry(key, {"name": "A"})
    mem.add_entry(key, {"name": "B"})
    # Demote the second entry to L3.
    mem.entries[1].compress_to(PrecisionLevel.L3)
    out = mem.retrieve(key, top_k=2)
    assert len(out) == 2
    # Highest-scoring entry is the L0 one ("A").
    top_entry, top_score = out[0]
    assert top_entry.facts == {"name": "A"}
    assert top_score > out[1][1]


def test_retrieve_skips_L5_entries():
    """L5 entries are existence traces — the retriever ignores
    them entirely (they have no usable key)."""
    mem = XRayEpisodicMemory(key_dim=16, retrieval_threshold=-1.0)
    key = _rand_key(16, 8)
    e_alive = mem.add_entry(key, {"name": "alive"})
    e_ghost = mem.add_entry(key, {"name": "ghost"})
    assert e_alive is not None and e_ghost is not None
    e_ghost.compress_to(PrecisionLevel.L5)

    out = mem.retrieve(key, top_k=5)
    assert len(out) == 1
    assert out[0][0].facts == {"name": "alive"}


def test_retrieve_triggers_reconsolidation_on_matches():
    """With a foundation wired in, a successful retrieve must
    invoke reconsolidation: the demoted entry climbs one level."""
    fnd = _MockFoundationForReconsolidation(key_dim=16)
    mem = XRayEpisodicMemory(
        key_dim=16, retrieval_threshold=-1.0, foundation=fnd,
    )
    key = _rand_key(16, 9)
    mem.add_entry(key, {"name": "Francois"})
    mem.entries[0].compress_to(PrecisionLevel.L2)

    mem.retrieve(key, top_k=1)
    assert mem.entries[0].precision_level == PrecisionLevel.L1
    # The foundation was consulted to re-encode the facts.
    assert fnd.calls, "foundation.get_key should be called during reconsolidation"


def test_retrieve_without_foundation_does_not_reconsolidate():
    """Phase 1 compat: a memory built without a foundation
    (the default) still works; reconsolidation is just a no-op."""
    mem = XRayEpisodicMemory(key_dim=16, retrieval_threshold=-1.0)
    key = _rand_key(16, 10)
    mem.add_entry(key, {"name": "X"})
    mem.entries[0].compress_to(PrecisionLevel.L2)
    mem.retrieve(key, top_k=1)
    # Still at L2 — no promotion because there's no foundation
    # to re-encode the facts.
    assert mem.entries[0].precision_level == PrecisionLevel.L2
    # But the freshness bookkeeping always runs.
    assert mem.entries[0].access_count == 1


# ---------- maybe_consolidate / force_consolidate ----------

def test_maybe_consolidate_triggers_on_idle():
    """Past the idle threshold, ``maybe_consolidate`` returns
    True and runs a light pass (decay only)."""
    mem = XRayEpisodicMemory(key_dim=8)
    mem.add_entry(_rand_key(8, 11), {"x": 1})
    # Backdate both ``last_activity`` and the entry's
    # last_accessed so the trigger fires AND the entry actually
    # decays.
    far_past = datetime.now() - timedelta(days=5)
    mem.last_activity = far_past
    mem.entries[0].last_accessed = far_past

    fired = mem.maybe_consolidate()
    assert fired is True
    assert mem.entries[0].precision_level >= PrecisionLevel.L1


def test_maybe_consolidate_does_nothing_when_active():
    """Memory just touched → no trigger fires."""
    mem = XRayEpisodicMemory(key_dim=8)
    mem.add_entry(_rand_key(8, 12), {"x": 1})
    mem.last_activity = datetime.now()
    fired = mem.maybe_consolidate()
    assert fired is False


def test_maybe_consolidate_triggers_on_size():
    """Past the SIZE threshold, the trigger always runs (full
    scope)."""
    mem = XRayEpisodicMemory(key_dim=4)
    # Drop the size cap to make the test cheap.
    mem.SIZE_CONSOLIDATION_THRESHOLD = 1
    mem.add_entry(_rand_key(4, 13), {"x": 1})
    mem.add_entry(_rand_key(4, 14), {"y": 2})
    fired = mem.maybe_consolidate()
    assert fired is True


def test_force_consolidate_runs_unconditionally():
    """No trigger needed — ``force_consolidate`` always runs."""
    mem = XRayEpisodicMemory(key_dim=8)
    mem.add_entry(_rand_key(8, 15), {"x": 1})
    mem.last_activity = datetime.now()  # not idle
    before = mem.last_consolidation
    mem.force_consolidate(scope="light")
    assert mem.last_consolidation > before


# ---------- effective_key + persistence-friendly access ----------

def test_effective_key_returns_float32_after_compression():
    """A demoted entry returns a fresh Float32 vector each time
    ``effective_key`` is read."""
    mem = XRayEpisodicMemory(key_dim=16)
    mem.add_entry(_rand_key(16, 16), {"x": 1})
    e = mem.entries[0]
    e.compress_to(PrecisionLevel.L2)
    assert e.key is None  # raw key freed
    out = e.effective_key
    assert out.dtype == torch.float32
    assert out.shape == (16,)


def test_total_storage_bytes_drops_after_decay():
    """Compressing entries should reduce the total payload
    footprint reported by ``total_storage_bytes``."""
    mem = XRayEpisodicMemory(key_dim=128)
    for i in range(10):
        mem.add_entry(_rand_key(128, 100 + i), {"id": i})
    before = mem.total_storage_bytes()
    for e in mem.entries:
        e.compress_to(PrecisionLevel.L4)
    after = mem.total_storage_bytes()
    assert after < before
