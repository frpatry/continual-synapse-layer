"""Feature extraction for the metacognitive layer.

The metacognitive layer reads a small vector of hand-engineered
features summarising:

- **Memory**  — 6 features describing the retrieval result
  (count, similarity distribution, recency, access patterns).
- **Query**   — 3 features describing the query itself (length,
  entity presence, lexical specificity).
- **Generation** — 4 features describing the LLM's generation
  process (placeholder in Phase 2a; zero-filled).
- **Alignment**  — 3 features comparing the generated response
  to retrieved facts (placeholder in Phase 2a; zero-filled).
- **Reserved** — 2 padding slots so the post-layer input vector
  is a clean ``18`` rather than ``16``; reserved for features
  added in subsequent phases.

The pre-layer reads the first ``9`` (memory + query); the
post-layer reads all ``18``. Both feature orderings are fixed
constants in this module so a trained network's input mapping
stays stable across releases.
"""

from __future__ import annotations

import math
import re
import statistics
from datetime import datetime
from typing import Any, Iterable, Optional

import torch


# Fixed feature names + ordering. Changing these breaks any
# trained checkpoint of the metacognitive layers.

MEMORY_FEATURE_NAMES: tuple[str, ...] = (
    "n_facts_retrieved",
    "max_similarity",
    "mean_similarity",
    "similarity_variance",
    "max_recency_days",
    "mean_access_count",
)

QUERY_FEATURE_NAMES: tuple[str, ...] = (
    "query_length_tokens",
    "has_named_entity",
    "query_specificity",
)

GENERATION_FEATURE_NAMES: tuple[str, ...] = (
    "gen_perplexity",
    "gen_token_entropy",
    "gen_max_token_prob",
    "gen_repetition_score",
)

ALIGNMENT_FEATURE_NAMES: tuple[str, ...] = (
    "align_semantic_similarity",
    "align_entity_overlap",
    "align_contradiction_score",
)

# 2 padding slots so post-mode totals an even 18 — reserved for
# features that will be added in a later phase (e.g. session-level
# confidence drift). Currently zero-filled.
RESERVED_FEATURE_NAMES: tuple[str, ...] = (
    "reserved_0",
    "reserved_1",
)

PRE_FEATURE_ORDER: tuple[str, ...] = MEMORY_FEATURE_NAMES + QUERY_FEATURE_NAMES
POST_FEATURE_ORDER: tuple[str, ...] = (
    MEMORY_FEATURE_NAMES
    + QUERY_FEATURE_NAMES
    + GENERATION_FEATURE_NAMES
    + ALIGNMENT_FEATURE_NAMES
    + RESERVED_FEATURE_NAMES
)

PRE_FEATURE_DIM: int = len(PRE_FEATURE_ORDER)   # 9
POST_FEATURE_DIM: int = len(POST_FEATURE_ORDER)  # 18


# ----------------------------------------------------------------------
# Memory features
# ----------------------------------------------------------------------

def extract_memory_features(retrieval_result: Iterable[tuple[Any, float]]) -> dict:
    """Summarise a memory-retrieval result as 6 scalar features.

    ``retrieval_result`` is the list returned by
    :meth:`XRayEpisodicMemory.retrieve` —
    ``[(entry, cosine_similarity), ...]`` sorted by similarity
    descending. Empty retrieval yields all zeros (the layer can
    learn that ``n_facts_retrieved == 0`` is the unambiguous
    "no memory" signal).

    Recency is measured in **days since now**; the older the
    entry, the larger the value. ``max_recency_days`` is the age
    of the OLDEST returned entry — useful as a "how stale is the
    backup retrieval" signal. ``mean_access_count`` is the mean
    of the entries' ``access_count`` attribute, a proxy for
    how-often-this-memory-has-been-useful.
    """
    items = list(retrieval_result)
    if not items:
        return {name: 0.0 for name in MEMORY_FEATURE_NAMES}

    sims: list[float] = [float(sim) for _entry, sim in items]
    now = datetime.now()
    ages_days: list[float] = []
    access_counts: list[float] = []
    for entry, _sim in items:
        ts = getattr(entry, "timestamp", None)
        if isinstance(ts, datetime):
            ages_days.append(max(0.0, (now - ts).total_seconds() / 86_400.0))
        else:
            ages_days.append(0.0)
        access_counts.append(float(getattr(entry, "access_count", 0)))

    sim_var = (
        statistics.pvariance(sims) if len(sims) > 1 else 0.0
    )

    return {
        "n_facts_retrieved": float(len(items)),
        "max_similarity": max(sims),
        "mean_similarity": statistics.fmean(sims),
        "similarity_variance": float(sim_var),
        "max_recency_days": max(ages_days),
        "mean_access_count": statistics.fmean(access_counts),
    }


# ----------------------------------------------------------------------
# Query features
# ----------------------------------------------------------------------

# A "named entity" here is any token that *starts* with an
# upper-case letter and is not at the start of the sentence —
# a cheap proper-noun heuristic. The first token gets stripped
# from consideration so a sentence-initial capital doesn't
# trigger a false positive.
_PROPER_NOUN_RE = re.compile(
    r"\b[A-ZÀ-Ö][A-Za-zÀ-ÖØ-öø-ÿ\-']+\b"
)


def extract_query_features(query: str, foundation: Optional[Any] = None) -> dict:
    """Summarise the query as 3 scalar features.

    ``foundation`` is accepted but not used in Phase 2a — Phase 2b
    may use it for an embedding-based specificity score. For now
    specificity is the simple type-token ratio (TTR) of the
    lowercased tokens, clamped to ``[0, 1]``.

    ``has_named_entity`` returns ``1.0`` / ``0.0`` (not bool) so
    the value can flow into a torch tensor uniformly with the
    other features.
    """
    tokens = query.split()
    n_tokens = len(tokens)

    # Drop the first token before scanning for proper nouns so
    # "Quel est mon nom?" doesn't flag "Quel".
    rest = " ".join(tokens[1:]) if n_tokens > 1 else ""
    has_entity = 1.0 if _PROPER_NOUN_RE.search(rest) else 0.0

    if n_tokens == 0:
        specificity = 0.0
    else:
        lowered = [t.lower() for t in tokens]
        specificity = len(set(lowered)) / float(n_tokens)

    return {
        "query_length_tokens": float(n_tokens),
        "has_named_entity": has_entity,
        "query_specificity": float(min(1.0, max(0.0, specificity))),
    }


# ----------------------------------------------------------------------
# Generation features (placeholder — zero-filled)
# ----------------------------------------------------------------------

def extract_generation_features(generation_output: Any = None) -> dict:
    """Generation-time signals — placeholder.

    Will eventually surface: average per-token perplexity,
    per-token entropy, max softmax probability, simple
    repetition score. Phase 2a returns zeros for all four so the
    feature vector has a stable shape across the placeholder /
    real-implementation transition.
    """
    return {name: 0.0 for name in GENERATION_FEATURE_NAMES}


# ----------------------------------------------------------------------
# Alignment features (placeholder — zero-filled)
# ----------------------------------------------------------------------

def extract_alignment_features(
    response: Optional[str] = None,
    facts: Any = None,
    foundation: Optional[Any] = None,
) -> dict:
    """Response-vs-facts alignment signals — placeholder.

    Will eventually surface: semantic similarity (embedding
    cosine) between the response and the retrieved fact summary,
    entity overlap, a contradiction score. Phase 2a returns
    zeros so post-layer training data can be generated without
    a real foundation in the loop yet.
    """
    return {name: 0.0 for name in ALIGNMENT_FEATURE_NAMES}


# ----------------------------------------------------------------------
# Vector assembly
# ----------------------------------------------------------------------

def assemble_feature_vector(features_dict: dict, mode: str = "pre") -> torch.Tensor:
    """Build the float32 input tensor for a metacognitive layer.

    Missing features default to ``0.0`` — this is what lets
    pre-mode callers pass only the 9 memory+query features and
    post-mode callers pass everything they have without having to
    pre-fill the placeholder slots.

    The returned tensor is 1-D and has shape ``(9,)`` for
    ``mode="pre"`` or ``(18,)`` for ``mode="post"``.
    """
    if mode == "pre":
        order = PRE_FEATURE_ORDER
    elif mode == "post":
        order = POST_FEATURE_ORDER
    else:
        raise ValueError(f"mode must be 'pre' or 'post', got {mode!r}")
    values = [_coerce_to_float(features_dict.get(name, 0.0)) for name in order]
    return torch.tensor(values, dtype=torch.float32)


def _coerce_to_float(value: Any) -> float:
    """Lenient float-coercion for feature values.

    Bools become ``1.0`` / ``0.0``; NaN / inf are folded to
    ``0.0`` so a downstream linear layer never sees a pathological
    input from a misbehaving placeholder.
    """
    if isinstance(value, bool):
        return float(value)
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(x) or math.isinf(x):
        return 0.0
    return x
