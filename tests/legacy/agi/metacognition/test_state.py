"""Tests for MetacognitiveState — fields, serialisation, safety check."""

from __future__ import annotations

import json

import pytest

from agi.metacognition.state import MetacognitiveState


def _make(
    *,
    epistemic_status="known",
    confidence=0.9,
    memory_coverage=0.8,
    memory_quality=0.85,
    generation_alignment=None,
    recommended_action="answer",
    raw_features=None,
) -> MetacognitiveState:
    return MetacognitiveState(
        epistemic_status=epistemic_status,
        confidence=confidence,
        memory_coverage=memory_coverage,
        memory_quality=memory_quality,
        generation_alignment=generation_alignment,
        recommended_action=recommended_action,
        raw_features=raw_features or {},
    )


def test_state_constructs_with_each_status():
    for s in ("known", "unknown", "uncertain", "hallucinated"):
        st = _make(epistemic_status=s)
        assert st.epistemic_status == s


@pytest.mark.parametrize(
    "status,expected",
    [
        ("known", True),
        ("uncertain", True),
        ("unknown", False),
        ("hallucinated", False),
    ],
)
def test_is_safe_to_answer(status, expected):
    st = _make(epistemic_status=status)
    assert st.is_safe_to_answer() is expected


def test_to_dict_keys_and_types():
    st = _make(
        epistemic_status="uncertain",
        confidence=0.42,
        memory_coverage=0.5,
        memory_quality=0.6,
        generation_alignment=0.7,
        recommended_action="answer_with_caveat",
        raw_features={"a": 1, "b": [0.1, 0.2]},
    )
    d = st.to_dict()
    assert set(d.keys()) == {
        "epistemic_status",
        "confidence",
        "memory_coverage",
        "memory_quality",
        "generation_alignment",
        "recommended_action",
        "raw_features",
    }
    assert d["epistemic_status"] == "uncertain"
    assert d["confidence"] == 0.42
    assert d["generation_alignment"] == 0.7
    assert d["raw_features"] == {"a": 1, "b": [0.1, 0.2]}


def test_to_dict_is_json_serialisable():
    st = _make(generation_alignment=0.5, raw_features={"x": 1.5, "y": [1, 2]})
    blob = json.dumps(st.to_dict())
    round_tripped = json.loads(blob)
    assert round_tripped["epistemic_status"] == st.epistemic_status
    assert round_tripped["generation_alignment"] == 0.5


def test_to_dict_handles_none_generation_alignment():
    st = _make(generation_alignment=None)
    d = st.to_dict()
    assert d["generation_alignment"] is None
    json.dumps(d)  # would raise if not serialisable
