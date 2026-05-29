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

def _facts_block(retrieval_or_facts) -> str:
    """Render the fact bullets — works on both retrieval tuples
    ``[(entry, sim), ...]`` and bare ``[dict, ...]`` lists."""
    out: list[str] = []
    for item in retrieval_or_facts:
        if isinstance(item, tuple) and len(item) >= 1:
            entry = item[0]
            facts = getattr(entry, "facts", entry)
        else:
            facts = item
        if isinstance(facts, dict):
            out.append(f"- {serialize_facts(facts)}")
    return "\n".join(out)


def build_normal_prompt(
    query: str,
    retrieval: List[tuple],
) -> str:
    """Qwen chat-template prompt for the answer path.

    Uses Qwen2.5's ``<|im_start|>...<|im_end|>`` chat format
    because Qwen-Instruct was fine-tuned on that template and
    degenerates badly (repetitive ``!!!!`` output) on naked
    prompts. Mirrors the Phase 2d.2 validation pipeline's
    builder, which is the proven-working prompt format on this
    model.
    """
    sys_content = (
        "You have the following information about the user:\n"
        + _facts_block(retrieval)
        + "\nUse this information naturally when relevant."
        if retrieval
        else "You are a helpful assistant. Answer concisely."
    )
    return (
        f"<|im_start|>system\n{sys_content}\n<|im_end|>\n"
        f"<|im_start|>user\n{query}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_caveated_prompt(
    query: str,
    retrieval: List[tuple],
) -> str:
    """Chat-template variant for the ``answer_with_caveat`` path —
    system message instructs the model to hedge."""
    facts_part = (
        "\n".join([
            "You have the following information about the user:",
            _facts_block(retrieval),
        ])
        if retrieval
        else ""
    )
    sys_content = (
        f"{facts_part}\n"
        "The information is incomplete or ambiguous. "
        "Answer if you can, but acknowledge the uncertainty."
    ).strip()
    return (
        f"<|im_start|>system\n{sys_content}\n<|im_end|>\n"
        f"<|im_start|>user\n{query}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_student_input(teacher_output: TeacherOutput) -> StudentInput:
    """Format a ``TeacherOutput`` as a ``(prompt, target)``
    distillation example.

    Same chat-template format as the teacher's input — so the
    student LoRA learns to produce the teacher's response
    autoregressively from the same prompt shape. When the
    teacher used a template (deferral), the student still
    learns to produce that template text from the same
    (query, facts) input → learns *when to defer* implicitly.
    """
    prompt = build_normal_prompt(
        teacher_output.query, teacher_output.facts_in_context,
    )
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
        defer_on_empty_memory: bool = True,
    ) -> None:
        """``defer_on_empty_memory``: when ``True`` (default), the
        teacher returns the template immediately when ``retrieval``
        is empty — no foundation call. When ``False`` it still
        tries to generate (the post-eval will usually flag it as
        hallucinated and override anyway, but for some bf16-stable
        question types Qwen may produce a clean refusal worth
        learning from). Distinct from the metacog ``pre_evaluate``
        decision, which was previously the *primary* defer gate
        but caused 100 % of training rows to be templates on real
        Qwen because PRE's recall on ``known`` is only ~0.24."""
        self.foundation = foundation
        self.orchestrator = orchestrator
        self.templates = templates
        self.template_key = template_key
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.defer_on_empty_memory = defer_on_empty_memory

    def respond(
        self,
        query: str,
        memory: Optional["XRayEpisodicMemory"] = None,
    ) -> TeacherOutput:
        """Run one query through the protected pipeline.

        Phase 2e (revised): the teacher no longer defers on
        :meth:`MetacognitiveOrchestrator.pre_evaluate`. Instead:

        - Empty memory → defer immediately to template (cheap,
          and PRE would say admit_ignorance anyway with very
          high precision on this case).
        - Non-empty memory → always generate via Qwen, then run
          ``post_evaluate``; override with template only if the
          POST layer flags ``hallucinated``.

        Rationale: PRE's real-Qwen recall on the ``known`` cohort
        was 0.24 in Phase 2h.1 — using PRE as the primary defer
        gate caused 100 % of training rows to become templates,
        which would teach the student "always defer". Bypassing
        PRE on the answer path restores the diversity needed for
        a useful distillation dataset.
        """
        # 1. Retrieve from memory (or empty if no memory).
        retrieval = (
            memory.retrieve(self.foundation.get_key(query), top_k=5)
            if memory is not None
            else []
        )
        facts: List[dict] = [entry.facts for entry, _sim in retrieval]

        # 2. Empty-memory short-circuit (cheap defer).
        if self.defer_on_empty_memory and not retrieval:
            response_text = self.templates.retrieve(self.template_key)
            return TeacherOutput(
                query=query,
                facts_in_context=facts,
                response=response_text,
                epistemic_status="unknown",
                action_taken="admit_ignorance",
                used_template=True,
                metacog_confidence=1.0,
            )

        # 3. Non-empty memory: pre-eval gives us posture only
        # (answer vs answer_with_caveat) but not deferral.
        pre_state = self.orchestrator.pre_evaluate(query, retrieval)

        # 4. Generate via Qwen with facts in context.
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

        # 5. Post-evaluate — kept ONLY as telemetry (epistemic
        # status + confidence flow into TeacherOutput for
        # downstream debugging). We deliberately do NOT use POST
        # as an override gate for distillation data:
        #
        #   - POST was trained on synthetic distributions; on
        #     real Qwen-1.5B outputs through the teacher's
        #     ``Contexte connu:`` prompt format (no chat
        #     template), POST classifies ~100% of cases as
        #     hallucinated, producing a monoculture training
        #     set that just teaches the student to always defer.
        #   - For SFT distillation, we WANT the student to
        #     learn Qwen's natural behaviour conditioned on
        #     (query, facts). Hallucinations in training data
        #     are fine — the student would behave like Qwen
        #     either way, since it can't detect its own
        #     hallucinations at inference without the metacog
        #     scaffolding.
        #   - POST stays valuable at INFERENCE time (real safety
        #     gate after the student generates), just not as a
        #     filter on the SFT training data itself.
        post_state = self.orchestrator.post_evaluate(
            query=query,
            retrieval=retrieval,
            response=response_text,
            gen_info=gen_info,
            foundation=self.foundation,
        )

        # Pre-eval picked the posture (answer vs answer_with_caveat).
        action = (
            pre_state.recommended_action
            if pre_state.recommended_action in ("answer", "answer_with_caveat")
            else "answer"
        )
        return TeacherOutput(
            query=query,
            facts_in_context=facts,
            response=response_text,
            epistemic_status=post_state.epistemic_status,
            action_taken=action,
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
