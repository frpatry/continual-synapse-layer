"""Feature extraction for the metacognitive layer.

The metacognitive layer reads a small vector of hand-engineered
features summarising:

- **Memory**  — 7 features describing the retrieval result
  (count, similarity distribution, recency, access patterns,
  plus a precision-quality scalar derived from the precision
  levels of the retrieved entries — see Phase 2c bis).
- **Query**   — 3 features describing the query itself (length,
  entity presence, lexical specificity).
- **Generation** — 4 features describing the LLM's generation
  process. Phase 2b populates these from
  :class:`agi.foundation.GenerationInfo`; the pre-layer path
  still passes ``None`` and gets zeros.
- **Alignment**  — 3 features comparing the generated response
  to retrieved facts. Phase 2b populates these from cosine
  similarity (response vs each fact's stringified form) plus a
  novelty signal.
- **Reserved** — 1 padding slot (``reserved_1`` — Phase 2c bis
  consumed ``reserved_0`` for ``precision_quality``). Reserved
  for a future internal_consistency_score feature.

The pre-layer reads the first ``10`` (memory + query); the
post-layer reads all ``18``. Both feature orderings are fixed
constants in this module so a trained network's input mapping
stays stable across releases.

The three ``ALIGNMENT_FEATURE_NAMES`` slot labels describe
exactly what they compute:

    alignment_max_cosine         ← max  cosine(response, fact)
    alignment_mean_cosine        ← mean cosine(response, fact)
    alignment_novel_token_ratio  ← novel-token ratio
                                   (response tokens absent
                                    from any fact)
"""

from __future__ import annotations

import math
import re
import statistics
from datetime import datetime
from typing import Any, Iterable, List, Optional, TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:  # avoid a hard transformers dependency at import time
    from agi.foundation import FrozenFoundation, GenerationInfo


# Fixed feature names + ordering. Changing these breaks any
# trained checkpoint of the metacognitive layers.

MEMORY_FEATURE_NAMES: tuple[str, ...] = (
    "n_facts_retrieved",
    "max_similarity",
    "mean_similarity",
    "similarity_variance",
    "max_recency_days",
    "mean_access_count",
    "precision_quality",
)

QUERY_FEATURE_NAMES: tuple[str, ...] = (
    "query_length_tokens",
    "has_named_entity",
    "query_specificity",
)

GENERATION_FEATURE_NAMES: tuple[str, ...] = (
    "mean_token_entropy",
    "max_token_entropy",
    "response_length_tokens",
    "attention_to_facts_mean",
)

ALIGNMENT_FEATURE_NAMES: tuple[str, ...] = (
    "alignment_max_cosine",
    "alignment_mean_cosine",
    "alignment_novel_token_ratio",
)

# 1 padding slot (was 2 in Phase 2b — ``precision_quality``
# consumed ``reserved_0`` in Phase 2c bis). Currently zero-filled;
# slated to hold an internal-consistency / self-coherence signal
# in a later phase.
RESERVED_FEATURE_NAMES: tuple[str, ...] = (
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
    """Summarise a memory-retrieval result as 7 scalar features.

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

    Phase 2c bis adds ``precision_quality`` — a ``[0, 1]`` scalar
    summarising the precision of the retrieved entries.
    Computation: ``1 - mean(precision_level) / 5``. All-L0
    retrieval scores 1.0 (best); all-L5 would be 0.0 (worst, but
    L5 entries are skipped by ``retrieve`` so this is degenerate
    in practice). Entries without a ``precision_level`` attribute
    are treated as L0 — keeps the function dual-compatible with
    mocks that don't include the field.
    """
    items = list(retrieval_result)
    if not items:
        return {name: 0.0 for name in MEMORY_FEATURE_NAMES}

    sims: list[float] = [float(sim) for _entry, sim in items]
    now = datetime.now()
    ages_days: list[float] = []
    access_counts: list[float] = []
    precision_levels_int: list[int] = []
    for entry, _sim in items:
        ts = getattr(entry, "timestamp", None)
        if isinstance(ts, datetime):
            ages_days.append(max(0.0, (now - ts).total_seconds() / 86_400.0))
        else:
            ages_days.append(0.0)
        access_counts.append(float(getattr(entry, "access_count", 0)))
        lvl = getattr(entry, "precision_level", 0)
        precision_levels_int.append(int(lvl))

    sim_var = (
        statistics.pvariance(sims) if len(sims) > 1 else 0.0
    )
    mean_level = sum(precision_levels_int) / len(precision_levels_int)
    precision_quality = max(0.0, min(1.0, 1.0 - mean_level / 5.0))

    return {
        "n_facts_retrieved": float(len(items)),
        "max_similarity": max(sims),
        "mean_similarity": statistics.fmean(sims),
        "similarity_variance": float(sim_var),
        "max_recency_days": max(ages_days),
        "mean_access_count": statistics.fmean(access_counts),
        "precision_quality": float(precision_quality),
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
# Generation features
# ----------------------------------------------------------------------

def extract_generation_features(
    gen_info: Optional["GenerationInfo"] = None,
) -> dict:
    """Surface the metacognition-relevant fields of a
    :class:`agi.foundation.GenerationInfo`.

    ``gen_info`` is ``None`` in the pre-evaluation path (the
    response hasn't been generated yet) and on any caller that
    used the legacy plain ``generate()`` API — in both cases we
    return zeros so the feature vector keeps a stable shape.

    No ``transformers`` import happens here — the function only
    reads attribute values; we duck-type so a dict-shaped stub
    works too (useful for tests that don't want to import
    ``GenerationInfo`` to build a mock).
    """
    if gen_info is None:
        return {name: 0.0 for name in GENERATION_FEATURE_NAMES}

    def _read(name: str) -> float:
        if isinstance(gen_info, dict):
            value = gen_info.get(name, 0.0)
        else:
            value = getattr(gen_info, name, 0.0)
        return _coerce_to_float(value)

    return {
        "mean_token_entropy": _read("mean_token_entropy"),
        "max_token_entropy": _read("max_token_entropy"),
        "response_length_tokens": _read("response_length_tokens"),
        "attention_to_facts_mean": _read("attention_to_facts_mean"),
    }


# ----------------------------------------------------------------------
# Alignment features
# ----------------------------------------------------------------------

def extract_alignment_features(
    response: Optional[str] = None,
    facts: Any = None,
    foundation: Optional["FrozenFoundation"] = None,
) -> dict:
    """Compute response-vs-facts alignment from foundation
    embeddings + a lexical novelty signal.

    Returns the three ``ALIGNMENT_FEATURE_NAMES`` slots:

    - ``alignment_max_cosine``  — best cosine similarity between
      the response embedding and each fact's stringified
      embedding.
    - ``alignment_mean_cosine`` — mean cosine similarity over
      the same set.
    - ``alignment_novel_token_ratio`` — fraction of response
      tokens that don't appear (case-insensitively) in any
      fact.

    **Empty-facts behaviour** (architectural decision): when
    ``facts`` is empty / ``None``, **all three** alignment
    features fold to ``0.0`` — including the novel-token ratio,
    even though the lexical formula would otherwise give 1.0.

    Rationale: alignment features measure "does the generated
    response stay consistent with the information we retrieved".
    When no information was retrieved, the question is
    *undefined* rather than "maximally violated". The
    hallucination case — "the model spoke confidently with no
    supporting memory" — is the orchestrator's job to detect
    via the conjunction ``memory_coverage == 0`` AND
    ``response_length_tokens > 0``. Keeping memory features and
    alignment features architecturally orthogonal lets the
    metacognitive layer learn each signal cleanly rather than
    having a derived novelty score double-count the
    "memory-empty" signal.

    Missing ``foundation`` keeps the same shape but the cosine
    pair stay at ``0.0`` (we can't embed without it) — the
    lexical novelty signal still flows because it doesn't need
    embeddings.

    ``facts`` is intentionally lenient: it may be a list of
    fact dicts (the spec's primary type), the orchestrator's
    raw retrieval list of ``(entry, sim)`` tuples, or any
    iterable of strings. Each item is normalised to a string
    before embedding lookup.
    """
    normalised_facts = _normalise_facts(facts)

    # Empty facts → no alignment signal is defined. See the
    # architectural-decision note in the docstring above.
    if not normalised_facts or not response or not response.strip():
        return {
            "alignment_max_cosine": 0.0,
            "alignment_mean_cosine": 0.0,
            "alignment_novel_token_ratio": 0.0,
        }

    if foundation is None:
        return {
            "alignment_max_cosine": 0.0,
            "alignment_mean_cosine": 0.0,
            "alignment_novel_token_ratio": _novel_token_ratio(
                response, normalised_facts,
            ),
        }

    response_emb = foundation.get_key(response).unsqueeze(0)
    fact_embs = [foundation.get_key(f).unsqueeze(0) for f in normalised_facts]
    alignments: List[float] = []
    for fe in fact_embs:
        sim = F.cosine_similarity(response_emb, fe, dim=-1)
        alignments.append(float(sim.item()))

    novel_ratio = _novel_token_ratio(response, normalised_facts)
    return {
        "alignment_max_cosine": float(max(alignments)),
        "alignment_mean_cosine": float(sum(alignments) / len(alignments)),
        "alignment_novel_token_ratio": novel_ratio,
    }


def _normalise_facts(facts: Any) -> List[str]:
    """Coerce a heterogeneous facts argument into a list of
    non-empty strings.

    Accepts (in priority order):

    - ``None`` / falsy → ``[]``
    - iterables of ``(entry, sim)`` tuples — pull ``entry.facts``
      (or ``entry`` itself if no such attribute) and stringify.
    - iterables of dicts — stringify each via ``str(fact)``.
    - iterables of plain strings — pass through after a strip.
    - anything else iterable — stringify each.
    """
    if not facts:
        return []
    out: List[str] = []
    for item in facts:
        # (entry, sim) tuple from the retrieval list.
        if isinstance(item, tuple) and len(item) == 2:
            entry = item[0]
            payload = getattr(entry, "facts", entry)
            text = str(payload).strip()
        elif isinstance(item, str):
            text = item.strip()
        else:
            text = str(item).strip()
        if text:
            out.append(text)
    return out


_TOKEN_RE = re.compile(r"[\wÀ-ÖØ-öø-ÿ]+")


def _novel_token_ratio(response: str, facts: Iterable[str]) -> float:
    """Fraction of response tokens that don't appear in any
    fact (case-insensitive).

    Empty response yields ``1.0`` — a response that doesn't
    exist can't recycle any fact tokens, but the metacognitive
    layer treats max-novelty as a "fully novel / nothing to
    align with" signal, which matches the empty-response case.
    """
    response_tokens = set(_TOKEN_RE.findall(response.lower()))
    if not response_tokens:
        return 1.0
    fact_tokens: set[str] = set()
    for f in facts:
        fact_tokens.update(_TOKEN_RE.findall(str(f).lower()))
    novel = response_tokens - fact_tokens
    return len(novel) / float(len(response_tokens))


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
