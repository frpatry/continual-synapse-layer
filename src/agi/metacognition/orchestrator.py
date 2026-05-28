"""High-level orchestrator that wires features → layer → templates.

The :class:`MetacognitiveOrchestrator` is the *only* piece of
this subpackage that the rest of AGISystem will eventually
import from. It hides:

- Which features each layer expects.
- The pre / post layer split.
- The template-fallback decision.

Phase 2a defines the API + the per-call evaluation routines but
does NOT yet stitch the orchestrator into ``AGISystem.chat`` —
that's a Phase 2b task. The orchestrator works without a
foundation in the loop; tests construct it with mock layers.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from .features import (
    assemble_feature_vector,
    extract_alignment_features,
    extract_generation_features,
    extract_memory_features,
    extract_query_features,
)
from .layer import MetacognitiveLayer
from .state import MetacognitiveState
from .templates import ResponseTemplates


class MetacognitiveOrchestrator:
    """Wire feature extractors + pre/post layers + templates.

    Parameters:
        pre_layer:  the 9-input layer judged before generation.
        post_layer: the 18-input layer judged after generation.
        templates:  the canned-reply library used when an
            epistemic decision is "admit_ignorance".
        memory:     reserved for later integration — accepted but
            unused in Phase 2a so tests can still construct the
            orchestrator without a real memory instance.
    """

    def __init__(
        self,
        pre_layer: MetacognitiveLayer,
        post_layer: MetacognitiveLayer,
        templates: ResponseTemplates,
        memory: Any | None = None,
    ) -> None:
        if pre_layer.mode != "pre":
            raise ValueError(
                f"pre_layer must be mode='pre', got {pre_layer.mode!r}"
            )
        if post_layer.mode != "post":
            raise ValueError(
                f"post_layer must be mode='post', got {post_layer.mode!r}"
            )
        self.pre_layer = pre_layer
        self.post_layer = post_layer
        self.templates = templates
        self.memory = memory

    # ---------- evaluation ----------

    def pre_evaluate(
        self,
        query: str,
        retrieval: Iterable[tuple[Any, float]],
        foundation: Optional[Any] = None,
    ) -> MetacognitiveState:
        """Before-generation epistemic check.

        Reads memory + query features only. The result tells the
        chat loop whether to bother generating at all (deferral
        avoids a wasted foundation call when memory is empty).
        """
        memory_feats = extract_memory_features(retrieval)
        query_feats = extract_query_features(query, foundation)
        combined = {**memory_feats, **query_feats}
        tensor = assemble_feature_vector(combined, mode="pre")
        return self.pre_layer.predict(tensor, raw_features=combined)

    def post_evaluate(
        self,
        query: str,
        retrieval: Iterable[tuple[Any, float]],
        response: str,
        gen_info: Any | None = None,
        foundation: Optional[Any] = None,
    ) -> MetacognitiveState:
        """After-generation epistemic check.

        ``gen_info`` is whatever the LLM exposes about its
        generation (token probabilities, etc.). Phase 2a accepts
        it as opaque and ignores its contents — the generation
        and alignment feature extractors return zeros.
        """
        memory_feats = extract_memory_features(retrieval)
        query_feats = extract_query_features(query, foundation)
        gen_feats = extract_generation_features(gen_info)
        align_feats = extract_alignment_features(
            response=response, facts=retrieval, foundation=foundation,
        )
        combined = {**memory_feats, **query_feats, **gen_feats, **align_feats}
        tensor = assemble_feature_vector(combined, mode="post")
        return self.post_layer.predict(tensor, raw_features=combined)

    # ---------- response selection ----------

    def get_template_response(
        self,
        state: MetacognitiveState,
        query: str,
        *,
        lang: str = "fr",
        template_key: Optional[str] = None,
    ) -> str | None:
        """Return a canned response when the state says to defer.

        ``template_key`` overrides the default selection; when
        unset, the orchestrator picks ``ignorance_polite_{lang}``
        for ``admit_ignorance`` states.

        Returns ``None`` for any state that is safe to answer —
        the caller should then generate via the foundation.
        """
        if state.recommended_action != "admit_ignorance":
            return None
        key = template_key or f"ignorance_polite_{lang}"
        return self.templates.retrieve(key)
