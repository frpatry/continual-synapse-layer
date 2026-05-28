"""Tests for XRayEpisodicMemory (privacy + retrieval mechanics)."""

from __future__ import annotations

import torch

from agi.memory import EpisodicEntry, XRayEpisodicMemory


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
