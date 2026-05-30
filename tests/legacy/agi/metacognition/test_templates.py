"""Tests for ResponseTemplates."""

from __future__ import annotations

import pytest

from agi.metacognition.templates import ResponseTemplates


def test_default_templates_present_for_both_languages():
    t = ResponseTemplates()
    keys = set(t.list_templates())
    for base in (
        "ignorance_polite", "ignorance_curious",
        "uncertainty", "asking_clarification",
    ):
        assert f"{base}_fr" in keys, f"missing {base}_fr"
        assert f"{base}_en" in keys, f"missing {base}_en"


def test_retrieve_returns_string_for_known_key():
    t = ResponseTemplates()
    out = t.retrieve("ignorance_polite_fr")
    assert isinstance(out, str)
    assert out.strip() != ""


def test_retrieve_unformatted_when_no_kwargs():
    """A template with placeholders should still return the raw
    string (placeholders intact) when no kwargs are passed —
    avoids accidental KeyError when the caller didn't intend to
    fill anything in."""
    t = ResponseTemplates()
    raw = t.retrieve("uncertainty_fr")
    assert "{guess}" in raw


def test_retrieve_formats_with_kwargs():
    t = ResponseTemplates()
    out = t.retrieve("uncertainty_fr", guess="il s'agit de Paris")
    assert "Paris" in out
    assert "{guess}" not in out


def test_retrieve_formats_clarification_topic():
    t = ResponseTemplates()
    out = t.retrieve("asking_clarification_en", topic="the location")
    assert "the location" in out


def test_retrieve_unknown_key_raises():
    t = ResponseTemplates()
    with pytest.raises(KeyError):
        t.retrieve("does_not_exist")


def test_add_template_registers_new_key():
    t = ResponseTemplates()
    t.add_template("custom_apology_en", "Sorry, I am still learning {topic}.")
    assert "custom_apology_en" in t.list_templates()
    out = t.retrieve("custom_apology_en", topic="French")
    assert out == "Sorry, I am still learning French."


def test_add_template_overwrites_existing_key():
    t = ResponseTemplates()
    original = t.retrieve("ignorance_polite_en")
    t.add_template("ignorance_polite_en", "Hard pass.")
    assert t.retrieve("ignorance_polite_en") == "Hard pass."
    assert t.retrieve("ignorance_polite_en") != original


def test_list_templates_returns_sorted_list():
    t = ResponseTemplates()
    keys = t.list_templates()
    assert keys == sorted(keys)
