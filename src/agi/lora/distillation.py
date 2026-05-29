"""Teacher/student pipeline for Phase 2e LoRA distillation.

The teacher runs Qwen2.5-1.5B + the full metacog scaffolding
(``pre_evaluate`` decides whether to generate, ``post_evaluate``
overrides on hallucination, templates handle deferrals). The
student is the same Qwen — with a small LoRA adapter — that
must learn to *reproduce the teacher's response* without the
scaffolding loop at inference time.

This module owns the teacher's behaviour + the
``(prompt, target)`` formatting that the LoRA trainer will
consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, TYPE_CHECKING

from agi.memory.precision import serialize_facts

if TYPE_CHECKING:
    from agi.foundation import FrozenFoundation
    from agi.memory.xray_episodic import XRayEpisodicMemory
    from agi.metacognition.orchestrator import MetacognitiveOrchestrator
    from agi.metacognition.templates import ResponseTemplates


@dataclass
class TeacherOutput:
    """Bundle of ``(query, facts, response, decision metadata)`` that
    the teacher pipeline produces for one query."""

    query: str
    facts_in_context: List[dict]
    response: str
    epistemic_status: str
    action_taken: str
    used_template: bool
    metacog_confidence: float


@dataclass
class StudentInput:
    """A single distillation example: ``(prompt, target)`` plus
    optional metadata. The LoRA trainer tokenises ``prompt`` +
    ``target``, masks the prompt portion in the labels, and
    trains the model to produce ``target`` autoregressively."""

    prompt: str
    target: str
    fact_token_ranges: List[tuple] = field(default_factory=list)


# ----------------------------------------------------------------------
# Prompt builders
# ----------------------------------------------------------------------

def build_normal_prompt(
    query: str,
    retrieval: List[tuple],
) -> str:
    """Memory-aware prompt for the answer / answer_with_caveat path."""
    if not retrieval:
        return f"Question: {query}\n\nRéponse:"
    facts_str = "\n".join(
        f"- {serialize_facts(entry.facts)}" for entry, _ in retrieval
    )
    return (
        f"Contexte connu:\n{facts_str}\n\n"
        f"Question: {query}\n\nRéponse:"
    )


def build_caveated_prompt(
    query: str,
    retrieval: List[tuple],
) -> str:
    """Prompt variant when the pre-layer recommends
    ``answer_with_caveat`` — explicitly tells the model to hedge."""
    if not retrieval:
        return (
            f"Question: {query}\n\n"
            f"Réponse (avec nuance d'incertitude):"
        )
    facts_str = "\n".join(
        f"- {serialize_facts(entry.facts)}" for entry, _ in retrieval
    )
    return (
        f"Contexte (incertain):\n{facts_str}\n\n"
        f"Question: {query}\n\n"
        f"Réponse (avec nuance d'incertitude):"
    )


def build_student_input(teacher_output: TeacherOutput) -> StudentInput:
    """Format a ``TeacherOutput`` as a ``(prompt, target)``
    distillation example.

    The student sees the same input shape as the teacher does
    on the answer path — fact bullets + query + ``Réponse:``
    prompt — and learns to produce the teacher's response
    autoregressively. When the teacher used a template (deferral),
    the student still learns to produce that template text from
    the same (query, facts) input — so at inference time it
    learns *when to defer* implicitly.
    """
    facts = teacher_output.facts_in_context
    if facts:
        facts_str = "\n".join(
            f"- {serialize_facts(f)}" for f in facts
        )
        prompt = (
            f"Contexte connu:\n{facts_str}\n\n"
            f"Question: {teacher_output.query}\n\nRéponse:"
        )
    else:
        prompt = f"Question: {teacher_output.query}\n\nRéponse:"
    return StudentInput(
        prompt=prompt,
        target=teacher_output.response,
        fact_token_ranges=[],
    )


# ----------------------------------------------------------------------
# Teacher pipeline
# ----------------------------------------------------------------------

class TeacherPipeline:
    """The metacog-protected pipeline that produces honest
    responses for distillation targets.

    The pipeline is intentionally *deterministic* given the
    (query, memory_state) — pre-evaluation gates the model
    output, the template-deferral path is rule-based, and the
    LLM call uses low temperature. This determinism is what
    lets distillation (SFT) work cleanly without needing
    reference-model KL constraints.
    """

    def __init__(
        self,
        foundation: "FrozenFoundation",
        orchestrator: "MetacognitiveOrchestrator",
        templates: "ResponseTemplates",
        *,
        template_key: str = "ignorance_polite_fr",
        max_new_tokens: int = 96,
        temperature: float = 0.0,
    ) -> None:
        self.foundation = foundation
        self.orchestrator = orchestrator
        self.templates = templates
        self.template_key = template_key
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def respond(
        self,
        query: str,
        memory: Optional["XRayEpisodicMemory"] = None,
    ) -> TeacherOutput:
        """Run one query through the protected pipeline."""
        # 1. Retrieve from memory (or empty if no memory).
        retrieval = (
            memory.retrieve(self.foundation.get_key(query), top_k=5)
            if memory is not None
            else []
        )
        facts: List[dict] = [entry.facts for entry, _sim in retrieval]

        # 2. Pre-evaluate.
        pre_state = self.orchestrator.pre_evaluate(query, retrieval)

        # 3. Pre-deferral path: use template instead of calling Qwen.
        if pre_state.recommended_action == "admit_ignorance":
            response_text = self.templates.retrieve(self.template_key)
            return TeacherOutput(
                query=query,
                facts_in_context=facts,
                response=response_text,
                epistemic_status=pre_state.epistemic_status,
                action_taken="admit_ignorance",
                used_template=True,
                metacog_confidence=float(pre_state.confidence),
            )

        # 4. Answer path: generate via Qwen with facts in context.
        if pre_state.recommended_action == "answer_with_caveat":
            prompt = build_caveated_prompt(query, retrieval)
        else:
            prompt = build_normal_prompt(query, retrieval)

        gen_info = self.foundation.generate_with_signals(
            prompt=prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        response_text = gen_info.response_text

        # 5. Post-evaluate safety check.
        post_state = self.orchestrator.post_evaluate(
            query=query,
            retrieval=retrieval,
            response=response_text,
            gen_info=gen_info,
            foundation=self.foundation,
        )

        # 6. Override if post-eval detected hallucination.
        if post_state.epistemic_status == "hallucinated":
            response_text = self.templates.retrieve(self.template_key)
            return TeacherOutput(
                query=query,
                facts_in_context=facts,
                response=response_text,
                epistemic_status="hallucinated",
                action_taken="admit_ignorance_override",
                used_template=True,
                metacog_confidence=float(post_state.confidence),
            )

        return TeacherOutput(
            query=query,
            facts_in_context=facts,
            response=response_text,
            epistemic_status=post_state.epistemic_status,
            action_taken=pre_state.recommended_action,
            used_template=False,
            metacog_confidence=float(post_state.confidence),
        )


__all__ = [
    "StudentInput",
    "TeacherOutput",
    "TeacherPipeline",
    "build_caveated_prompt",
    "build_normal_prompt",
    "build_student_input",
]
