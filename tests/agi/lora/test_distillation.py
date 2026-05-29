"""Tests for the LoRA distillation teacher pipeline + helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pytest
import torch

from agi.lora.distillation import (
    StudentInput,
    TeacherOutput,
    TeacherPipeline,
    build_caveated_prompt,
    build_normal_prompt,
    build_student_input,
)
from agi.memory.xray_episodic import EpisodicEntry, XRayEpisodicMemory
from agi.metacognition.layer import MetacognitiveLayer
from agi.metacognition.orchestrator import MetacognitiveOrchestrator
from agi.metacognition.state import MetacognitiveState
from agi.metacognition.templates import ResponseTemplates


# ---------- Mock foundation that doesn't load Qwen ----------

class _MockFoundation:
    """Stable-key + scripted-generation stub. ``get_key`` is
    deterministic per text. ``generate_with_signals`` returns a
    scripted response so tests can drive teacher branches."""

    def __init__(self, key_dim: int = 16, scripted_response: str = "scripted answer"):
        self.key_dim = key_dim
        self.scripted_response = scripted_response
        self.last_prompt: str | None = None

    def get_key(self, text: str) -> torch.Tensor:
        h = abs(hash(text)) & 0xFFFFFFFF
        g = torch.Generator()
        g.manual_seed(h)
        v = torch.randn(self.key_dim, generator=g)
        return v / v.norm()

    def generate_with_signals(
        self, prompt: str, max_new_tokens: int = 96,
        temperature: float = 0.0, fact_token_ranges=None,
    ):
        self.last_prompt = prompt
        from agi.foundation import GenerationInfo
        return GenerationInfo(
            response_text=self.scripted_response,
            generated_token_ids=[1, 2, 3],
            response_length_tokens=3,
            mean_token_entropy=1.0,
            max_token_entropy=2.0,
            attention_to_facts_mean=0.5,
            attention_to_facts_max=0.7,
            generation_time_seconds=0.01,
        )


def _pin_action(layer: MetacognitiveLayer, status_idx: int) -> None:
    """Force the layer's argmax to a chosen status to drive a
    deterministic branch of the teacher pipeline."""
    with torch.no_grad():
        layer.status_head.weight.zero_()
        layer.status_head.bias.zero_()
        layer.status_head.bias[status_idx] = 10.0


def _build_orchestrator(memory_obj=None) -> MetacognitiveOrchestrator:
    return MetacognitiveOrchestrator(
        pre_layer=MetacognitiveLayer(mode="pre"),
        post_layer=MetacognitiveLayer(mode="post"),
        templates=ResponseTemplates(),
        memory=memory_obj,
    )


# ---------- Teacher pipeline ----------

def test_teacher_defers_on_empty_memory():
    """Phase 2e (revised): the teacher defers immediately when
    retrieval is empty — bypasses metacog pre_evaluate entirely
    on this path. The foundation is NOT called."""
    foundation = _MockFoundation(scripted_response="should-not-be-used")
    orch = _build_orchestrator()
    teacher = TeacherPipeline(foundation, orch, ResponseTemplates())

    out = teacher.respond("Quel est mon code postal?", memory=None)
    assert isinstance(out, TeacherOutput)
    assert out.used_template is True
    assert out.action_taken == "admit_ignorance"
    assert "should-not-be-used" not in out.response
    # Foundation was not called for generation.
    assert foundation.last_prompt is None


def test_teacher_does_not_defer_on_unknown_pre_when_memory_has_facts():
    """Phase 2e (revised): even if PRE says ``unknown`` /
    ``admit_ignorance``, the teacher STILL generates via Qwen
    when memory has facts (the previous behaviour caused 100% of
    training rows to be templates because PRE's real-Qwen recall
    on ``known`` is only 0.24). PRE's decision now only picks the
    posture (answer vs answer_with_caveat) on the generate path,
    not whether to defer."""
    foundation = _MockFoundation(scripted_response="Vous habitez à Lyon.")
    orch = _build_orchestrator()
    # Pin PRE to ``unknown`` (status_idx=2) — would previously have
    # caused immediate template defer; now the teacher must still
    # generate because memory has facts.
    _pin_action(orch.pre_layer, status_idx=2)
    _pin_action(orch.post_layer, status_idx=0)  # known → keep
    teacher = TeacherPipeline(foundation, orch, ResponseTemplates())

    memory = XRayEpisodicMemory(key_dim=16, retrieval_threshold=-1.0)
    memory.add_entry(foundation.get_key("city=Lyon"), {"city": "Lyon"})

    out = teacher.respond("Où je vis?", memory)
    assert out.used_template is False
    assert out.response == "Vous habitez à Lyon."
    assert foundation.last_prompt is not None


def test_teacher_uses_foundation_for_known_branch():
    """When pre_evaluate says known, teacher calls Qwen and
    returns the generated text."""
    foundation = _MockFoundation(scripted_response="Vous vous appelez François.")
    orch = _build_orchestrator()
    _pin_action(orch.pre_layer, status_idx=0)   # known → answer
    _pin_action(orch.post_layer, status_idx=0)  # confirms not hallucinated
    teacher = TeacherPipeline(foundation, orch, ResponseTemplates())

    memory = XRayEpisodicMemory(key_dim=16, retrieval_threshold=-1.0)
    key = foundation.get_key("name=François")
    memory.add_entry(key, {"name": "François"})

    out = teacher.respond("Comment je m'appelle?", memory)
    assert out.used_template is False
    assert out.action_taken == "answer"
    assert out.response == "Vous vous appelez François."
    assert foundation.last_prompt is not None
    assert "François" in foundation.last_prompt  # facts injected


def test_teacher_does_not_override_on_post_hallucination():
    """Phase 2e (revised again): POST hallucination is recorded
    as telemetry on the TeacherOutput but does NOT override the
    response. The earlier override behaviour caused a 100 %
    monoculture training set (POST trained on synthetic data
    flagged 100 % of real Qwen outputs on the simpler
    ``Contexte connu`` prompt as hallucinated). POST stays
    useful at inference time, just not as a distillation-data
    filter."""
    foundation = _MockFoundation(scripted_response="I made this up")
    orch = _build_orchestrator()
    _pin_action(orch.pre_layer, status_idx=0)   # known → call foundation
    _pin_action(orch.post_layer, status_idx=3)  # hallucinated (telemetry only)
    teacher = TeacherPipeline(foundation, orch, ResponseTemplates())

    memory = XRayEpisodicMemory(key_dim=16, retrieval_threshold=-1.0)
    memory.add_entry(foundation.get_key("x"), {"x": "y"})

    out = teacher.respond("anything", memory)
    # Response is preserved despite POST flagging it as hallucinated.
    assert out.used_template is False
    assert out.response == "I made this up"
    # Telemetry still reports what POST thought.
    assert out.epistemic_status == "hallucinated"
    assert out.action_taken != "admit_ignorance_override"


def test_teacher_caveated_branch_uses_caveat_prompt():
    """uncertain → answer_with_caveat path builds the caveated
    prompt (verified by inspecting what reached the foundation)."""
    foundation = _MockFoundation(scripted_response="Je crois que oui.")
    orch = _build_orchestrator()
    _pin_action(orch.pre_layer, status_idx=1)   # uncertain
    _pin_action(orch.post_layer, status_idx=1)
    teacher = TeacherPipeline(foundation, orch, ResponseTemplates())

    memory = XRayEpisodicMemory(key_dim=16, retrieval_threshold=-1.0)
    memory.add_entry(foundation.get_key("sport=natation"), {"sport": "natation"})
    memory.add_entry(foundation.get_key("sport=vélo"), {"sport": "vélo"})

    out = teacher.respond("Quel sport je pratique?", memory)
    assert out.action_taken == "answer_with_caveat"
    assert "incertain" in (foundation.last_prompt or "").lower()


# ---------- Prompt builders ----------

@dataclass
class _Entry:
    facts: dict


def test_build_normal_prompt_with_facts():
    prompt = build_normal_prompt(
        "Comment je m'appelle?",
        [(_Entry(facts={"name": "François"}), 0.9)],
    )
    assert "Contexte connu:" in prompt
    assert "François" in prompt
    assert prompt.endswith("Réponse:")


def test_build_normal_prompt_without_facts():
    prompt = build_normal_prompt("Quel est mon code postal?", [])
    assert "Contexte" not in prompt
    assert prompt.endswith("Réponse:")


def test_build_caveated_prompt_marks_uncertainty():
    prompt = build_caveated_prompt(
        "Quel sport je pratique?",
        [
            (_Entry(facts={"sport": "natation"}), 0.7),
            (_Entry(facts={"sport": "vélo"}), 0.6),
        ],
    )
    assert "incertain" in prompt.lower()
    assert "natation" in prompt
    assert "vélo" in prompt


# ---------- build_student_input ----------

def test_build_student_input_includes_facts_in_prompt():
    out = TeacherOutput(
        query="Comment je m'appelle?",
        facts_in_context=[{"name": "François"}],
        response="Vous vous appelez François.",
        epistemic_status="known",
        action_taken="answer",
        used_template=False,
        metacog_confidence=0.95,
    )
    si = build_student_input(out)
    assert isinstance(si, StudentInput)
    assert "Contexte connu:" in si.prompt
    assert "François" in si.prompt
    assert si.target == "Vous vous appelez François."


def test_build_student_input_no_facts_uses_simple_prompt():
    out = TeacherOutput(
        query="Quel est mon code postal?",
        facts_in_context=[],
        response="Je n'ai pas cette information dans ma mémoire.",
        epistemic_status="unknown",
        action_taken="admit_ignorance",
        used_template=True,
        metacog_confidence=0.99,
    )
    si = build_student_input(out)
    assert "Contexte" not in si.prompt
    assert si.target.startswith("Je n'ai pas")
