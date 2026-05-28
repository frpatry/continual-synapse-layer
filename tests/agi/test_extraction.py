"""Tests for the pattern-based FactExtractor."""

from __future__ import annotations

from agi.extraction import FactExtractor


def test_extract_name_french():
    facts = FactExtractor().extract(
        "Bonjour, mon nom est Francois."
    )
    assert facts.get("name") == "Francois"


def test_extract_name_je_mappelle():
    facts = FactExtractor().extract(
        "Je m'appelle Marie."
    )
    assert facts.get("name") == "Marie"


def test_extract_name_english_my_name():
    facts = FactExtractor().extract(
        "My name is Alice."
    )
    assert facts.get("name") == "Alice"


def test_extract_age_french():
    facts = FactExtractor().extract(
        "j'ai 30 ans"
    )
    assert facts.get("age") == 30


def test_extract_age_english():
    facts = FactExtractor().extract(
        "I am 42 years old."
    )
    assert facts.get("age") == 42


def test_extract_location_french():
    facts = FactExtractor().extract(
        "Je vis à Montréal."
    )
    assert facts.get("location") == "Montréal"


def test_extract_location_english():
    facts = FactExtractor().extract(
        "I live in New York."
    )
    assert facts.get("location") == "New York"


def test_extract_preferences_collected():
    facts = FactExtractor().extract(
        "I like coffee. I prefer short answers."
    )
    prefs = facts.get("preferences")
    assert prefs is not None
    # Each pattern fires independently; both should appear.
    assert any("coffee" in p for p in prefs)
    assert any("short" in p for p in prefs)


def test_extract_multiple_facts_one_utterance():
    facts = FactExtractor().extract(
        "Je m'appelle Marie, j'ai 30 ans et je vis à Paris. "
        "Je préfère les réponses courtes."
    )
    assert facts.get("name") == "Marie"
    assert facts.get("age") == 30
    assert facts.get("location") == "Paris"
    assert "preferences" in facts
    assert any("courtes" in p for p in facts["preferences"])


def test_no_extraction_returns_empty_dict():
    facts = FactExtractor().extract(
        "The weather is nice today."
    )
    assert facts == {}
