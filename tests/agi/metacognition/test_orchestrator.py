"""Tests for MetacognitiveOrchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pytest
import torch

from agi.metacognition.layer import MetacognitiveLayer
from agi.metacognition.orchestrator import MetacognitiveOrchestrator
from agi.metacognition.state import MetacognitiveState
from agi.metacognition.templates import ResponseTemplates


@dataclass
class _MockEntry:
    timestamp: datetime
    access_count: int = 0
    facts: dict = field(default_factory=dict)


def _pin_layer_to_status(layer: MetacognitiveLayer, status_idx: int) -> None:
    """Force the layer's argmax to a chosen status index by
    rewriting the status-head bias. Lets the orchestrator tests
    drive the decision deterministically without training."""
    with torch.no_grad():
        layer.status_head.weight.zero_()
        layer.status_head.bias.zero_()
        layer.status_head.bias[status_idx] = 10.0


def _build_orch() -> MetacognitiveOrchestrator:
    return MetacognitiveOrchestrator(
        pre_layer=MetacognitiveLayer(mode="pre"),
        post_layer=MetacognitiveLayer(mode="post"),
        templates=ResponseTemplates(),
    )


# ---------- Constructor validation ----------

def test_constructor_rejects_swapped_layers():
    pre = MetacognitiveLayer(mode="pre")
    post = MetacognitiveLayer(mode="post")
    with pytest.raises(ValueError):
        MetacognitiveOrchestrator(
            pre_layer=post,  # wrong mode
            post_layer=pre,
            templates=ResponseTemplates(),
        )


def test_constructor_accepts_optional_memory_arg():
    orch = MetacognitiveOrchestrator(
        pre_layer=MetacognitiveLayer(mode="pre"),
        post_layer=MetacognitiveLayer(mode="post"),
        templates=ResponseTemplates(),
        memory="sentinel",
    )
    assert orch.memory == "sentinel"


# ---------- maybe_consolidate hook (Phase 2c bis) ----------

class _MemorySpy:
    """Minimal stand-in: tracks ``maybe_consolidate`` calls so the
    orchestrator-side hook can be observed without spinning up a
    real XRayEpisodicMemory."""

    def __init__(self) -> None:
        self.calls: int = 0

    def maybe_consolidate(self) -> bool:
        self.calls += 1
        return False


def test_pre_evaluate_calls_maybe_consolidate_when_memory_present():
    """If a memory is wired in, every pre_evaluate gets one
    cheap ``maybe_consolidate`` poke before running."""
    spy = _MemorySpy()
    orch = MetacognitiveOrchestrator(
        pre_layer=MetacognitiveLayer(mode="pre"),
        post_layer=MetacognitiveLayer(mode="post"),
        templates=ResponseTemplates(),
        memory=spy,
    )
    orch.pre_evaluate("Quel est mon nom?", retrieval=[])
    assert spy.calls == 1
    orch.pre_evaluate("encore une question?", retrieval=[])
    assert spy.calls == 2


def test_pre_evaluate_does_not_crash_without_memory():
    """The default constructor (no memory) must still be safe to
    use — the hook is opt-in."""
    orch = MetacognitiveOrchestrator(
        pre_layer=MetacognitiveLayer(mode="pre"),
        post_layer=MetacognitiveLayer(mode="post"),
        templates=ResponseTemplates(),
    )
    # No exception.
    orch.pre_evaluate("Quel est mon nom?", retrieval=[])


def test_pre_evaluate_skips_hook_when_memory_lacks_method():
    """A memory object without ``maybe_consolidate`` (e.g. an
    old mock) shouldn't blow up the orchestrator — the hook
    uses ``hasattr`` to feature-detect."""
    orch = MetacognitiveOrchestrator(
        pre_layer=MetacognitiveLayer(mode="pre"),
        post_layer=MetacognitiveLayer(mode="post"),
        templates=ResponseTemplates(),
        memory="just-a-sentinel-string",  # no methods
    )
    orch.pre_evaluate("Quel est mon nom?", retrieval=[])


# ---------- pre_evaluate ----------

def test_pre_evaluate_returns_metacognitive_state():
    orch = _build_orch()
    state = orch.pre_evaluate("Quel est mon nom?", retrieval=[])
    assert isinstance(state, MetacognitiveState)
    assert state.generation_alignment is None  # pre-mode


def test_pre_evaluate_empty_retrieval_with_unknown_pin_admits_ignorance():
    """When the pre-layer is forced to output "unknown" on an
    empty-retrieval input, the recommended action must be
    "admit_ignorance"."""
    orch = _build_orch()
    _pin_layer_to_status(orch.pre_layer, status_idx=2)  # "unknown"
    state = orch.pre_evaluate("Quel est mon nom?", retrieval=[])
    assert state.epistemic_status == "unknown"
    assert state.recommended_action == "admit_ignorance"
    assert state.memory_coverage == 0.0  # no facts retrieved
    assert state.memory_quality == 0.0


def test_pre_evaluate_with_retrieval_sets_memory_coverage():
    """With three retrieved facts and reasonable similarities,
    memory_coverage clamps to 1.0 and memory_quality picks up the
    max similarity."""
    orch = _build_orch()
    now = datetime.now()
    retrieval = [
        (_MockEntry(timestamp=now, access_count=2), 0.95),
        (_MockEntry(timestamp=now, access_count=1), 0.80),
        (_MockEntry(timestamp=now, access_count=0), 0.72),
    ]
    state = orch.pre_evaluate("Bonjour, je m'appelle Francois.", retrieval)
    assert state.memory_coverage == 1.0
    assert state.memory_quality == pytest.approx(0.95)
    # raw_features should carry the per-name values.
    assert state.raw_features["n_facts_retrieved"] == 3.0
    assert state.raw_features["query_length_tokens"] > 0


# ---------- post_evaluate ----------

def test_post_evaluate_returns_state_with_generation_alignment_field():
    orch = _build_orch()
    state = orch.post_evaluate(
        query="Quel est mon nom?",
        retrieval=[],
        response="Je n'ai pas cette information.",
        gen_info=None,
    )
    assert isinstance(state, MetacognitiveState)
    # generation_alignment is the mean of the 3 alignment slots.
    # Phase 2b refactor: when ``retrieval`` is empty, ALL three
    # alignment features fold to 0.0 (including
    # ``alignment_novel_token_ratio``) — see the docstring on
    # ``extract_alignment_features`` for the architectural
    # rationale. Mean of (0, 0, 0) → 0.0.
    assert state.generation_alignment == 0.0


def test_post_evaluate_carries_full_18_features_in_raw():
    orch = _build_orch()
    state = orch.post_evaluate(
        query="Tell me about yourself.",
        retrieval=[],
        response="anything",
    )
    # Memory + query + generation + alignment names must all
    # appear in raw_features. Phase 2b renamed both the
    # generation slots (to match GenerationInfo field names) and
    # the alignment slots (to describe what they actually
    # compute).
    for name in (
        "n_facts_retrieved", "query_length_tokens",
        "mean_token_entropy", "alignment_max_cosine",
    ):
        assert name in state.raw_features


# ---------- get_template_response ----------

def test_get_template_response_returns_template_for_admit_ignorance():
    orch = _build_orch()
    _pin_layer_to_status(orch.pre_layer, status_idx=2)  # "unknown"
    state = orch.pre_evaluate("Quel est mon nom?", retrieval=[])
    out = orch.get_template_response(state, query="Quel est mon nom?")
    assert isinstance(out, str)
    assert out.strip() != ""


def test_get_template_response_returns_none_for_known():
    orch = _build_orch()
    _pin_layer_to_status(orch.pre_layer, status_idx=0)  # "known"
    state = orch.pre_evaluate("Bonjour je m'appelle Francois.", retrieval=[])
    out = orch.get_template_response(state, query="anything")
    assert out is None


def test_get_template_response_lang_selects_english_variant():
    orch = _build_orch()
    _pin_layer_to_status(orch.pre_layer, status_idx=2)  # "unknown"
    state = orch.pre_evaluate("What is my name?", retrieval=[])
    out_en = orch.get_template_response(state, query="anything", lang="en")
    out_fr = orch.get_template_response(state, query="anything", lang="fr")
    assert out_en != out_fr  # different templates per language


def test_get_template_response_template_key_override():
    orch = _build_orch()
    _pin_layer_to_status(orch.pre_layer, status_idx=2)  # "unknown"
    state = orch.pre_evaluate("Quel est mon nom?", retrieval=[])
    out = orch.get_template_response(
        state, query="anything", template_key="ignorance_curious_fr",
    )
    # The curious variant ends with a "?" — handy assertion.
    assert out.rstrip().endswith("?")
