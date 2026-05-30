"""Tests for AGISystem.save() / load() — privacy + round-trip."""

from __future__ import annotations

import json

import torch

from agi.integration import AGISystem


class _MockFoundation:
    """Same minimal mock used elsewhere — fixed key_dim, mock
    generate() that returns a non-JSON string so LLM extraction
    falls back to regex (the path we want to exercise here)."""

    def __init__(self, key_dim: int = 16):
        self.key_dim = key_dim
        self.last_prompt: str | None = None

    def get_key(self, text: str) -> torch.Tensor:
        h = abs(hash(text)) & 0xFFFFFFFF
        g = torch.Generator()
        g.manual_seed(h)
        return torch.randn(self.key_dim, generator=g)

    def generate(self, prompt: str, max_new_tokens: int = 80, temperature: float = 0.7) -> str:
        self.last_prompt = prompt
        return f"MOCK_RESPONSE prompt_len={len(prompt)}"


def test_save_load_round_trip_preserves_entries(tmp_path):
    """Save, reload into a fresh AGISystem, and verify the entries
    match on the fields we persist."""
    f1 = _MockFoundation()
    a = AGISystem(f1, retrieval_threshold=-1.0)
    a.new_session()
    a.observe("Mon nom est Francois et je vis à Montréal.")
    a.observe("J'aime le café.")
    assert len(a.memory) == 2

    save_path = tmp_path / "mem.json"
    a.save(str(save_path))

    f2 = _MockFoundation()
    b = AGISystem(f2, retrieval_threshold=-1.0)
    b.load(str(save_path))

    assert len(b.memory) == 2
    # Facts survive verbatim.
    a_facts = sorted([sorted(e.facts.items()) for e in a.memory.entries])
    b_facts = sorted([sorted(e.facts.items()) for e in b.memory.entries])
    assert a_facts == b_facts
    # Keys round-trip element-wise.
    for ea, eb in zip(a.memory.entries, b.memory.entries):
        assert torch.allclose(ea.key, eb.key, atol=1e-6)
    # Session counter survives.
    assert b.memory.current_session == a.memory.current_session


def test_persisted_file_contains_no_raw_text(tmp_path):
    """The JSON written to disk must not contain a raw_text-shaped
    field on any entry. This is the on-disk extension of the
    Phase 1.0 privacy contract."""
    a = AGISystem(_MockFoundation(), retrieval_threshold=-1.0)
    a.observe("Mon nom est Francois et je vis à Montréal.")
    a.observe("Je m'appelle Marie, j'ai 30 ans, je vis à Paris.")

    save_path = tmp_path / "mem.json"
    a.save(str(save_path))

    with open(save_path) as f:
        state = json.load(f)

    assert isinstance(state["entries"], list)
    assert len(state["entries"]) >= 2
    forbidden = ("raw_text", "original_input", "raw", "samples", "utterance")
    for entry in state["entries"]:
        for k in forbidden:
            assert k not in entry, f"Privacy violation on disk: {k!r} written"
        # Allowed fields only.
        assert set(entry.keys()) <= {
            "key", "facts", "timestamp", "access_count", "creation_session",
        }


def test_load_replaces_existing_entries(tmp_path):
    """Loading from disk should not append to the current memory —
    it should replace it. Otherwise restart + load would double-
    count entries."""
    a = AGISystem(_MockFoundation(), retrieval_threshold=-1.0)
    a.observe("Mon nom est Francois.")
    save_path = tmp_path / "mem.json"
    a.save(str(save_path))

    b = AGISystem(_MockFoundation(), retrieval_threshold=-1.0)
    b.observe("Je m'appelle Marie.")
    assert len(b.memory) == 1
    b.load(str(save_path))
    assert len(b.memory) == 1  # not 2 — load replaced, not merged
    assert b.memory.entries[0].facts.get("name") == "Francois"


def test_loaded_memory_supports_retrieval(tmp_path):
    """After save + load, retrieval should still work using the
    deserialised keys."""
    f1 = _MockFoundation()
    a = AGISystem(f1, retrieval_threshold=-1.0)
    a.observe("Mon nom est Francois et je vis à Montréal.")
    save_path = tmp_path / "mem.json"
    a.save(str(save_path))

    f2 = _MockFoundation()
    b = AGISystem(f2, retrieval_threshold=-1.0)
    b.load(str(save_path))

    # MockFoundation derives keys deterministically from text hash,
    # so the same query text produces the same key across instances.
    q_key = f2.get_key("Mon nom est Francois et je vis à Montréal.")
    hits = b.memory.retrieve(q_key, top_k=1)
    assert len(hits) == 1
    entry, sim = hits[0]
    assert entry.facts.get("name") == "Francois"
    assert sim > 0.99


def test_conversation_history_not_persisted(tmp_path):
    """Conversation turns are session-scoped and must not be on
    disk — the saved JSON should only describe the memory."""
    a = AGISystem(_MockFoundation(), retrieval_threshold=-1.0)
    a.conversation.add_turn("user", "Mon nom est Francois.")
    a.conversation.add_turn("assistant", "Bonjour Francois!")
    a.observe("Mon nom est Francois.")

    save_path = tmp_path / "mem.json"
    a.save(str(save_path))

    with open(save_path) as f:
        state = json.load(f)

    assert "turns" not in state
    assert "conversation" not in state
    assert "history" not in state
    blob = json.dumps(state)
    assert "Bonjour Francois!" not in blob


def test_new_session_clears_conversation_but_keeps_memory():
    a = AGISystem(_MockFoundation())
    a.observe("Mon nom est Francois.")
    a.conversation.add_turn("user", "hi")
    a.conversation.add_turn("assistant", "hello")
    assert len(a.memory) == 1
    assert len(a.conversation) == 2

    sid = a.new_session()
    assert sid >= 1
    assert len(a.conversation) == 0  # cleared
    assert len(a.memory) == 1        # preserved
