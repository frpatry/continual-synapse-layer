"""Metacognitive layer for the AGI architecture (Phase 2).

A small subsystem that looks at memory-retrieval signals,
query features, and (post-generation) response signals to
decide whether the system should:

- answer directly,
- answer with a caveat, or
- admit ignorance via a canned template.

Phase 2a (this commit) ships the structural skeleton:
:class:`MetacognitiveState`, the feature extractors, two
``MetacognitiveLayer`` MLPs (pre / post), a small
:class:`ResponseTemplates` library, and a
:class:`MetacognitiveOrchestrator` that wires them together.
No integration with :class:`AGISystem` and no training data —
those come in Phases 2b / 2c.

This package depends only on ``torch`` (and the Python stdlib);
in particular it does NOT import ``transformers``, so it stays
cheap to import in pure-feature / pure-state tests.
"""

from .features import (
    POST_FEATURE_DIM,
    PRE_FEATURE_DIM,
    assemble_feature_vector,
    extract_alignment_features,
    extract_generation_features,
    extract_memory_features,
    extract_query_features,
)
from .layer import MetacognitiveLayer
from .orchestrator import MetacognitiveOrchestrator
from .state import (
    EpistemicStatus,
    MetacognitiveState,
    RecommendedAction,
)
from .templates import ResponseTemplates

__all__ = [
    "EpistemicStatus",
    "MetacognitiveLayer",
    "MetacognitiveOrchestrator",
    "MetacognitiveState",
    "POST_FEATURE_DIM",
    "PRE_FEATURE_DIM",
    "RecommendedAction",
    "ResponseTemplates",
    "assemble_feature_vector",
    "extract_alignment_features",
    "extract_generation_features",
    "extract_memory_features",
    "extract_query_features",
]
