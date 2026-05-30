"""Tests for LLMFactExtractor (JSON parsing + fallback semantics).

These tests use a ``_ScriptedFoundation`` that returns a fixed
generation string regardless of the prompt. That lets us
exercise the JSON-parsing path without loading Qwen (and keeps
the test suite offline).
"""

from __future__ import annotations

import torch

from agi.integration import AGISystem
from agi.llm_extraction import LLMFactExtractor, _find_first_json_object


class _ScriptedFoundation:
    """Foundation stub whose ``generate`` returns a pre-set
    string. ``key_dim`` and ``get_key`` are provided so the same
    stub can also stand in for AGISystem fallback tests."""

    def __init__(self, scripted_response: str, key_dim: int = 16):
        self.scripted_response = scripted_response
        self.key_dim = key_dim
        self.last_prompt: str | None = None

    def get_key(self, text: str) -> torch.Tensor:
        h = abs(hash(text)) & 0xFFFFFFFF
        g = torch.Generator()
        g.manual_seed(h)
        return torch.randn(self.key_dim, generator=g)

    def generate(self, prompt: str, max_new_tokens: int = 200, temperature: float = 0.0) -> str:
        self.last_prompt = prompt
        return self.scripted_response


# ---------- _find_first_json_object ----------

def test_find_json_handles_flat_object():
    assert _find_first_json_object('foo {"a": 1} bar') == '{"a": 1}'


def test_find_json_handles_nested_object():
    text = 'pre {"outer": {"inner": 2}} post'
    assert _find_first_json_object(text) == '{"outer": {"inner": 2}}'


def test_find_json_handles_list_value():
    """The naive ``\\{[^{}]*\\}`` approach would refuse to match
    objects whose values contain braces. The brace-balanced
    scanner should handle nested structure cleanly."""
    text = 'noise {"prefs": ["a", "b"], "name": "X"} tail'
    assert _find_first_json_object(text) == '{"prefs": ["a", "b"], "name": "X"}'


def test_find_json_returns_none_on_no_object():
    assert _find_first_json_object("plain text, no json") is None


def test_find_json_handles_braces_in_strings():
    """A ``}`` inside a JSON string literal must not close the
    object early."""
    text = '{"k": "} not closed"}'
    assert _find_first_json_object(text) == '{"k": "} not closed"}'


# ---------- LLMFactExtractor.extract ----------

# Inputs below are deliberately ≥3 words and not questions —
# Phase 1.2 added a heuristic gate that returns ``{}`` early for
# short or question-shaped inputs WITHOUT calling the LLM. These
# tests exercise the JSON-parsing path, so we need to slip past
# the gate.
_NON_GATED_INPUT = "Je m'appelle Sarah et je vis à Paris."


def test_extract_parses_clean_json():
    f = _ScriptedFoundation('{"name": "Francois", "location": "Montreal"}')
    ex = LLMFactExtractor(f)
    facts = ex.extract(_NON_GATED_INPUT)
    assert facts == {"name": "Francois", "location": "Montreal"}


def test_extract_extracts_varied_fact_types():
    """LLM extraction should handle fact types beyond what regex
    covers (profession, opinions, dates, etc.)."""
    response = (
        'Here are the facts: '
        '{"name": "Alice", "profession": "data scientist", '
        '"opinion": "loves Python", "birthday": "1990-05-12"}'
    )
    f = _ScriptedFoundation(response)
    ex = LLMFactExtractor(f)
    facts = ex.extract(_NON_GATED_INPUT)
    assert facts["name"] == "Alice"
    assert facts["profession"] == "data scientist"
    assert facts["opinion"] == "loves Python"
    assert facts["birthday"] == "1990-05-12"


def test_extract_drops_null_and_empty_values():
    response = '{"name": "Bob", "age": null, "location": "", "prefs": []}'
    facts = LLMFactExtractor(_ScriptedFoundation(response)).extract(_NON_GATED_INPUT)
    assert facts == {"name": "Bob"}


def test_extract_returns_empty_on_no_json():
    facts = LLMFactExtractor(_ScriptedFoundation("just prose")).extract(_NON_GATED_INPUT)
    assert facts == {}


def test_extract_returns_empty_on_malformed_json():
    facts = LLMFactExtractor(_ScriptedFoundation('{"name": Francois}')).extract(_NON_GATED_INPUT)
    assert facts == {}


def test_extract_returns_empty_on_non_dict_json():
    """The LLM might emit a JSON array; the extractor should
    refuse to misinterpret it as a fact dict."""
    facts = LLMFactExtractor(_ScriptedFoundation('[1, 2, 3]')).extract(_NON_GATED_INPUT)
    assert facts == {}


# ---------- Fallback semantics inside AGISystem.observe ----------

def test_agisystem_falls_back_to_regex_when_llm_returns_empty():
    """When LLM extraction yields nothing parseable, AGISystem
    must run the regex extractor as a backup so the Phase 1.0
    deterministic path still works."""
    f = _ScriptedFoundation("MOCK_RESPONSE no json here")
    a = AGISystem(f, use_llm_extraction=True)
    facts = a.observe("Mon nom est Francois.")
    assert facts.get("name") == "Francois"
    assert len(a.memory) == 1


def test_agisystem_prefers_llm_when_both_would_succeed():
    """If the LLM extractor returns a non-empty dict, regex
    should NOT also run — the LLM result is authoritative."""
    f = _ScriptedFoundation('{"name": "Alice", "color": "blue"}')
    a = AGISystem(f, use_llm_extraction=True)
    # The user text would also match the regex name pattern
    # (Francois), but the scripted LLM response wins.
    facts = a.observe("Mon nom est Francois.")
    assert facts == {"name": "Alice", "color": "blue"}


def test_agisystem_regex_only_when_llm_disabled():
    f = _ScriptedFoundation('{"name": "ShouldBeIgnored"}')
    a = AGISystem(f, use_llm_extraction=False)
    facts = a.observe("Mon nom est Francois.")
    # LLM disabled → regex result wins, never even queries the LLM.
    assert facts.get("name") == "Francois"
    assert a.llm_extractor is None


# ---------- Heuristic gating (Phase 1.2) ----------

class _ExplodingFoundation:
    """Foundation stub that raises if ``generate`` is called.

    Used to assert the gating short-circuits BEFORE any LLM call.
    If the test reaches the foundation, the AssertionError fires
    and the test fails loudly — confirming the gate is the only
    thing that could have returned an empty dict.
    """

    def __init__(self, key_dim: int = 16):
        self.key_dim = key_dim

    def get_key(self, text):  # pragma: no cover — gating tests don't retrieve
        raise AssertionError("get_key should not be called by extraction-gating tests")

    def generate(self, prompt, max_new_tokens=150, temperature=0.0):
        raise AssertionError(
            "LLM was called for a gated input — heuristic gate failed"
        )


def test_extraction_skips_questions():
    """A question should short-circuit before any foundation call.
    Using ``_ExplodingFoundation`` guarantees the gate fires."""
    ex = LLMFactExtractor(_ExplodingFoundation())
    assert ex.extract("Quel est mon nom?") == {}
    assert ex.last_gated is True


def test_extraction_skips_short_input():
    """Two-word input is below the length threshold; gate fires."""
    ex = LLMFactExtractor(_ExplodingFoundation())
    assert ex.extract("Bonjour") == {}
    assert ex.last_gated is True


def test_extraction_gates_english_questions_too():
    ex = LLMFactExtractor(_ExplodingFoundation())
    assert ex.extract("What is my name?") == {}
    assert ex.last_gated is True
    assert ex.extract("Where do I live?") == {}
    assert ex.last_gated is True


def test_extraction_does_not_gate_real_statements():
    """A clear self-statement should reach the LLM. The exploding
    foundation would make this a hard fail if gating misfires."""
    f = _ScriptedFoundation('{"name": "Sarah", "location": "Paris"}')
    ex = LLMFactExtractor(f)
    facts = ex.extract("Bonjour, je m'appelle Sarah et je vis à Paris.")
    assert ex.last_gated is False
    assert facts == {"name": "Sarah", "location": "Paris"}


# ---------- No-hallucination contract ----------

def test_extraction_no_hallucination():
    """When the model returns a single-fact JSON, the extractor
    must return ONLY that fact — not pad it with extra fields."""
    f = _ScriptedFoundation('{"name": "Sarah"}')
    ex = LLMFactExtractor(f)
    facts = ex.extract("Je m'appelle Sarah.")
    assert facts == {"name": "Sarah"}
    assert "age" not in facts
    assert "location" not in facts
    assert "profession" not in facts


def test_extraction_handles_multiple_facts():
    """Real multi-fact LLM responses should round-trip with all
    keys preserved."""
    f = _ScriptedFoundation(
        '{"profession": "médecin", "location": "Lyon", "years_experience": 5}'
    )
    ex = LLMFactExtractor(f)
    facts = ex.extract(
        "Je travaille comme médecin à Lyon depuis 5 ans."
    )
    assert facts == {
        "profession": "médecin", "location": "Lyon", "years_experience": 5,
    }


def test_extraction_returns_empty_on_no_facts():
    """Ambiguous self-statement ("J'aime le beau temps"). The
    extractor's behaviour depends on what the LLM emits — both
    ``{}`` (model treats weather as not-about-the-user) and
    ``{"preferences": ["beau temps"]}`` (model treats it as a
    preference) are acceptable. The contract is: parse what the
    LLM said cleanly; don't drop / pad."""
    # Case A: LLM says no fact.
    f_a = _ScriptedFoundation('{}')
    assert LLMFactExtractor(f_a).extract("J'aime le beau temps.") == {}

    # Case B: LLM treats it as a preference.
    f_b = _ScriptedFoundation('{"preferences": ["beau temps"]}')
    assert LLMFactExtractor(f_b).extract("J'aime le beau temps.") == {
        "preferences": ["beau temps"]
    }


# ---------- last_gated stays accurate across consecutive calls ----------

def test_last_gated_resets_between_calls():
    f = _ScriptedFoundation('{"name": "Sarah"}')
    ex = LLMFactExtractor(f)
    # First call: gated.
    ex.extract("Bonjour")
    assert ex.last_gated is True
    # Second call: not gated — must reset.
    ex.extract("Je m'appelle Sarah et j'habite à Paris.")
    assert ex.last_gated is False
