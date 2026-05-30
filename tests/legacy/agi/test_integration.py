"""Tests for AGISystem with a MockFoundation (no Qwen download)."""

from __future__ import annotations

from typing import Any

import torch

from agi.integration import AGISystem


class _MockFoundation:
    """Minimal foundation interface for testing AGISystem.

    - Stable keys: derived deterministically from the text content
      (hash → seeded torch.Generator → random key). Same text →
      same key, different text → different key (with very high
      probability).
    - Generation: echoes the system context if present so we can
      assert the augmented prompt contained the expected facts.
    """

    def __init__(self, key_dim: int = 32):
        self.key_dim = key_dim
        self.last_prompt: str | None = None

    def get_key(self, text: str) -> torch.Tensor:
        h = abs(hash(text)) & 0xFFFFFFFF
        g = torch.Generator()
        g.manual_seed(h)
        return torch.randn(self.key_dim, generator=g)

    def generate(
        self, prompt: str, max_new_tokens: int = 80, temperature: float = 0.7,
    ) -> str:
        self.last_prompt = prompt
        # Echo a stable marker the caller can grep for.
        return f"MOCK_RESPONSE prompt_len={len(prompt)}"


def test_observe_extracts_and_stores_one_fact():
    sys = AGISystem(_MockFoundation())
    facts = sys.observe("Mon nom est Francois.")
    assert facts.get("name") == "Francois"
    assert len(sys.memory) == 1


def test_observe_with_no_facts_does_not_store():
    sys = AGISystem(_MockFoundation())
    facts = sys.observe("The weather is nice today.")
    assert facts == {}
    assert len(sys.memory) == 0


def test_respond_retrieves_stored_fact():
    """After observing the user's name, the augmented prompt the
    foundation sees on a follow-up query must include the name."""
    foundation = _MockFoundation()
    sys = AGISystem(foundation, retrieval_threshold=-1.0)  # always retrieve
    sys.observe("Mon nom est Francois et je vis à Montréal.")
    result = sys.respond("Comment je m'appelle?")
    assert result["memory_used"] is True
    assert any(
        "name" in facts and facts["name"] == "Francois"
        for facts, _ in result["retrieved_facts"]
    )
    assert "Francois" in foundation.last_prompt
    assert "Montréal" in foundation.last_prompt


def test_respond_without_memory_does_not_augment():
    """With an empty memory, the prompt must not contain a system
    context block."""
    foundation = _MockFoundation()
    sys = AGISystem(foundation)
    result = sys.respond("Comment je m'appelle?")
    assert result["memory_used"] is False
    assert "system" not in foundation.last_prompt.lower() or (
        "user" in foundation.last_prompt
        and "system" not in foundation.last_prompt.split("user")[0].lower()
    )


def test_observe_then_respond_pipeline_end_to_end():
    """Smoke through the full observe → respond pipeline; assert
    that the retrieved facts dict contains the introduced fields."""
    sys = AGISystem(_MockFoundation(), retrieval_threshold=-1.0)
    sys.observe("Je m'appelle Marie, j'ai 30 ans, je vis à Paris.")
    result = sys.respond("Que sais-tu sur moi?")
    assert result["memory_used"] is True
    merged = result["merged_facts"]
    assert merged.get("name") == "Marie"
    assert merged.get("age") == 30
    assert merged.get("location") == "Paris"


def test_no_raw_text_in_memory_entries_post_observe():
    """Even after several observations, no entry should hold a
    field that smells like the original utterance."""
    sys = AGISystem(_MockFoundation())
    sys.observe("Mon nom est Francois.")
    sys.observe("J'aime le café.")
    for e in sys.memory.entries:
        for forbidden in ("raw_text", "original_input", "raw", "utterance"):
            assert not hasattr(e, forbidden), (
                f"Privacy violation — entry exposes {forbidden!r}"
            )
