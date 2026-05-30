"""Metacognitive state — what the layer thinks about the system's own knowledge.

A :class:`MetacognitiveState` is the output of a metacognitive
layer (:mod:`agi.metacognition.layer`) for a given query. It
summarises:

- **What** the system thinks it knows (``epistemic_status``).
- **How confident** that judgement is (``confidence``).
- **Whether** to actually answer or to defer / ask for help
  (``recommended_action``).

The class is intentionally a thin dataclass — the metacognitive
*decisions* live here, but the metacognitive *features* that
backed them live in ``raw_features``. Phase 2a does not couple
this to the foundation or the rest of AGISystem; later phases
will plug it into the chat loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


EpistemicStatus = Literal["known", "unknown", "uncertain", "hallucinated"]
"""The four mutually-exclusive epistemic categories the layer can output.

``known``       — the system has reliable information for the query.
``uncertain``   — partial / weakly-retrieved information; answer with caveat.
``unknown``     — no useful retrieval; should defer.
``hallucinated``— the system caught itself making things up (post-only).
"""

RecommendedAction = Literal["answer", "answer_with_caveat", "admit_ignorance"]
"""The three response strategies the orchestrator can take."""


@dataclass
class MetacognitiveState:
    """A single metacognitive judgement for a single query.

    Attributes:
        epistemic_status: One of :data:`EpistemicStatus`.
        confidence: ``[0, 1]`` calibrated confidence in the
            epistemic-status decision (NOT the answer itself).
        memory_coverage: ``[0, 1]`` scalar summary of how well
            the retrieved memory covers the query.
        memory_quality: ``[0, 1]`` scalar summary of retrieval
            quality (e.g. max cosine similarity).
        generation_alignment: Only meaningful for the post-layer
            (``None`` for the pre-layer); ``[0, 1]`` summary of
            how well the generated response aligns with the
            retrieved facts.
        recommended_action: One of :data:`RecommendedAction`,
            derived from ``epistemic_status``.
        raw_features: The full feature dict that produced this
            judgement, kept for logging / debugging. Never read
            by downstream code in production paths.
    """

    epistemic_status: EpistemicStatus
    confidence: float
    memory_coverage: float
    memory_quality: float
    generation_alignment: Optional[float]
    recommended_action: RecommendedAction
    raw_features: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-serialisable rendering for logging.

        ``raw_features`` is forwarded as-is — callers should ensure
        the values are themselves JSON-friendly (floats / ints /
        strings / lists thereof). Tensors are NOT auto-converted.
        """
        return {
            "epistemic_status": self.epistemic_status,
            "confidence": float(self.confidence),
            "memory_coverage": float(self.memory_coverage),
            "memory_quality": float(self.memory_quality),
            "generation_alignment": (
                None if self.generation_alignment is None
                else float(self.generation_alignment)
            ),
            "recommended_action": self.recommended_action,
            "raw_features": dict(self.raw_features),
        }

    def is_safe_to_answer(self) -> bool:
        """True when the orchestrator should generate a response
        (with or without a caveat). False when it should defer
        via a template / clarification request.

        ``known`` and ``uncertain`` are safe — uncertain answers
        get a caveat from the orchestrator. ``unknown`` and
        ``hallucinated`` are not safe; the orchestrator should
        admit ignorance instead.
        """
        return self.epistemic_status in ("known", "uncertain")
