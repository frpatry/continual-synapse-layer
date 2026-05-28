"""Tests for the metacognitive feature extractors + vector assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pytest
import torch

from agi.metacognition.features import (
    ALIGNMENT_FEATURE_NAMES,
    GENERATION_FEATURE_NAMES,
    MEMORY_FEATURE_NAMES,
    POST_FEATURE_DIM,
    POST_FEATURE_ORDER,
    PRE_FEATURE_DIM,
    PRE_FEATURE_ORDER,
    QUERY_FEATURE_NAMES,
    assemble_feature_vector,
    extract_alignment_features,
    extract_generation_features,
    extract_memory_features,
    extract_query_features,
)


# ---------- Mock retrieval entries ----------

@dataclass
class _MockEntry:
    timestamp: datetime
    access_count: int = 0
    facts: dict = field(default_factory=dict)


# ---------- Memory features ----------

def test_memory_features_empty_retrieval_returns_zeros():
    feats = extract_memory_features([])
    assert set(feats) == set(MEMORY_FEATURE_NAMES)
    assert all(v == 0.0 for v in feats.values())


def test_memory_features_single_entry():
    e = _MockEntry(timestamp=datetime.now(), access_count=2)
    feats = extract_memory_features([(e, 0.85)])
    assert feats["n_facts_retrieved"] == 1.0
    assert feats["max_similarity"] == 0.85
    assert feats["mean_similarity"] == 0.85
    # single-sample variance is 0.
    assert feats["similarity_variance"] == 0.0
    assert feats["mean_access_count"] == 2.0
    # max_recency_days should be ~0 (entry is freshly created).
    assert feats["max_recency_days"] < 0.001


def test_memory_features_multiple_entries_distributional():
    now = datetime.now()
    e_old = _MockEntry(timestamp=now - timedelta(days=10), access_count=5)
    e_mid = _MockEntry(timestamp=now - timedelta(days=2), access_count=1)
    e_new = _MockEntry(timestamp=now, access_count=0)
    sims = [0.9, 0.6, 0.3]
    retrieval = [
        (e_new, sims[0]),
        (e_mid, sims[1]),
        (e_old, sims[2]),
    ]
    feats = extract_memory_features(retrieval)
    assert feats["n_facts_retrieved"] == 3.0
    assert feats["max_similarity"] == 0.9
    assert feats["mean_similarity"] == sum(sims) / 3
    assert feats["similarity_variance"] > 0.0
    assert feats["max_recency_days"] >= 9.99  # 10-day-old entry
    assert feats["mean_access_count"] == 2.0  # (5+1+0)/3


# ---------- Query features ----------

def test_query_features_empty_string():
    feats = extract_query_features("")
    assert feats["query_length_tokens"] == 0.0
    assert feats["has_named_entity"] == 0.0
    assert feats["query_specificity"] == 0.0


def test_query_features_token_count_and_specificity():
    feats = extract_query_features("le chat est sur le tapis")
    assert feats["query_length_tokens"] == 6.0
    # 5 unique tokens / 6 total → 5/6 ≈ 0.833.
    assert abs(feats["query_specificity"] - 5 / 6) < 1e-6
    assert feats["has_named_entity"] == 0.0


def test_query_features_detects_named_entity_after_first_word():
    feats = extract_query_features("Bonjour, je m'appelle Francois.")
    assert feats["has_named_entity"] == 1.0


def test_query_features_first_word_capitalisation_does_not_flag_entity():
    """Sentence-initial capitals should not register as proper
    nouns — otherwise "Quel est mon nom?" would falsely flag."""
    feats = extract_query_features("Quel est mon nom?")
    assert feats["has_named_entity"] == 0.0


# ---------- Generation features (Phase 2b) ----------

def test_generation_features_none_returns_zeros():
    """The pre-evaluation path and any legacy caller pass
    ``gen_info=None``; the extractor must return a zero-valued
    dict so the feature vector keeps a stable shape."""
    feats = extract_generation_features(None)
    assert set(feats) == set(GENERATION_FEATURE_NAMES)
    assert all(v == 0.0 for v in feats.values())


def test_generation_features_reads_genination_info_fields():
    """A real-ish GenerationInfo populates each slot from the
    matching attribute on the dataclass."""
    from agi.foundation import GenerationInfo
    info = GenerationInfo(
        response_text="hello world",
        generated_token_ids=[1, 2, 3, 4, 5],
        response_length_tokens=5,
        mean_token_entropy=1.234,
        max_token_entropy=2.5,
        attention_to_facts_mean=0.42,
        attention_to_facts_max=0.7,
        generation_time_seconds=0.01,
    )
    feats = extract_generation_features(info)
    assert feats["mean_token_entropy"] == 1.234
    assert feats["max_token_entropy"] == 2.5
    assert feats["response_length_tokens"] == 5.0
    assert feats["attention_to_facts_mean"] == 0.42


def test_generation_features_accepts_dict_stub():
    """A dict with the right keys also works — handy for tests
    that don't want to import GenerationInfo."""
    feats = extract_generation_features({
        "mean_token_entropy": 0.5,
        "max_token_entropy": 1.0,
        "response_length_tokens": 8,
        "attention_to_facts_mean": 0.1,
    })
    assert feats["mean_token_entropy"] == 0.5
    assert feats["response_length_tokens"] == 8.0


# ---------- Alignment features (Phase 2b) ----------

class _StubFoundationDirected:
    """Foundation stub whose embeddings are *directed* — each
    distinct text gets a deterministic unit vector, so a self-
    match gives cosine ≈ 1 and a different text gives cosine
    close to 0. Lets us assert alignment behaviour without
    loading a real model.
    """

    def __init__(self, dim: int = 32):
        self._dim = dim

    def get_key(self, text: str) -> torch.Tensor:
        # Seed from the text's hash so identical strings map to
        # identical vectors and different strings map to nearly
        # orthogonal ones (with high probability at dim=32).
        h = abs(hash(text)) & 0xFFFFFFFF
        g = torch.Generator()
        g.manual_seed(h)
        v = torch.randn(self._dim, generator=g)
        return v / v.norm()


def test_alignment_features_empty_facts_returns_all_zeros():
    """Empty facts → ALL three alignment slots fold to 0.0,
    including ``alignment_novel_token_ratio``.

    The hallucination case ("response generated with no
    supporting memory") is the orchestrator's job to detect via
    ``memory_coverage == 0`` AND ``response_length_tokens > 0``,
    NOT via a derived novelty score. Returning 0.0 across the
    alignment slots keeps memory features and alignment features
    architecturally orthogonal — see the module docstring on
    ``extract_alignment_features``."""
    feats = extract_alignment_features(
        response="the cat sat on the mat",
        facts=[],
        foundation=_StubFoundationDirected(),
    )
    assert feats["alignment_max_cosine"] == 0.0
    assert feats["alignment_mean_cosine"] == 0.0
    assert feats["alignment_novel_token_ratio"] == 0.0


def test_alignment_features_perfect_match_high_similarity():
    """Response identical to the only fact → max + mean cosine
    ≈ 1. Novelty ≈ 0 (every response token appears in the fact)."""
    f = _StubFoundationDirected()
    fact_text = "name=Francois location=Montreal"
    feats = extract_alignment_features(
        response=fact_text,
        facts=[fact_text],
        foundation=f,
    )
    assert feats["alignment_max_cosine"] == pytest.approx(1.0, abs=1e-5)
    assert feats["alignment_mean_cosine"] == pytest.approx(1.0, abs=1e-5)
    assert feats["alignment_novel_token_ratio"] == 0.0


def test_alignment_features_novel_token_ratio_simple_case():
    """3 of 5 response tokens overlap with facts → novelty = 2/5.

    Novelty arithmetic on the non-empty path is unchanged from
    the pre-refactor behaviour — only the empty-facts edge case
    was modified."""
    f = _StubFoundationDirected()
    feats = extract_alignment_features(
        response="alpha beta gamma delta epsilon",
        facts=["alpha beta gamma"],
        foundation=f,
    )
    # 2 novel (delta, epsilon) out of 5 → 0.4.
    assert feats["alignment_novel_token_ratio"] == pytest.approx(0.4)


def test_alignment_features_accepts_retrieval_tuples():
    """The orchestrator passes ``[(entry, sim), ...]`` rather than
    a list of dicts. The extractor must transparently pull the
    facts out via the ``entry.facts`` attribute (or fall back to
    str() of the entry)."""
    from dataclasses import dataclass

    @dataclass
    class _Entry:
        facts: dict

    f = _StubFoundationDirected()
    retrieval = [
        (_Entry(facts={"name": "Francois"}), 0.95),
        (_Entry(facts={"location": "Montreal"}), 0.80),
    ]
    feats = extract_alignment_features(
        response="Francois Montreal",
        facts=retrieval,
        foundation=f,
    )
    # At least one alignment value should be finite / non-NaN.
    assert -1.0 <= feats["alignment_max_cosine"] <= 1.0
    assert -1.0 <= feats["alignment_mean_cosine"] <= 1.0
    # Novelty: 0 out of 2 unique response tokens are novel
    # (both Francois and Montreal appear in the facts).
    assert feats["alignment_novel_token_ratio"] == 0.0


def test_alignment_features_missing_foundation_returns_only_novelty():
    """Without a foundation we can't compute cosine, but the
    novelty signal is purely lexical so it stays meaningful —
    as long as facts is non-empty (the empty-facts gate fires
    BEFORE the foundation check)."""
    feats = extract_alignment_features(
        response="alpha beta gamma",
        facts=["alpha"],
        foundation=None,
    )
    assert feats["alignment_max_cosine"] == 0.0
    assert feats["alignment_mean_cosine"] == 0.0
    assert feats["alignment_novel_token_ratio"] == pytest.approx(2.0 / 3.0)


def test_alignment_features_empty_response_returns_all_zeros():
    """An empty response also routes through the empty-input
    gate (no signal to extract → all zeros)."""
    feats = extract_alignment_features(
        response="",
        facts=["something"],
        foundation=_StubFoundationDirected(),
    )
    assert feats["alignment_max_cosine"] == 0.0
    assert feats["alignment_mean_cosine"] == 0.0
    assert feats["alignment_novel_token_ratio"] == 0.0


# ---------- Vector assembly ----------

def test_assemble_pre_mode_shape_and_order():
    feats = {name: float(i) for i, name in enumerate(PRE_FEATURE_ORDER)}
    vec = assemble_feature_vector(feats, mode="pre")
    assert vec.shape == (PRE_FEATURE_DIM,)
    assert vec.dtype == torch.float32
    for i, _name in enumerate(PRE_FEATURE_ORDER):
        assert float(vec[i].item()) == float(i)


def test_assemble_post_mode_shape_is_eighteen():
    feats = {name: float(i) for i, name in enumerate(POST_FEATURE_ORDER)}
    vec = assemble_feature_vector(feats, mode="post")
    assert vec.shape == (POST_FEATURE_DIM,)
    assert POST_FEATURE_DIM == 18


def test_assemble_missing_features_zero_filled():
    """Pre-mode features in a post-mode assembly → trailing
    slots are zero-padded."""
    feats = {name: 1.0 for name in PRE_FEATURE_ORDER}
    vec = assemble_feature_vector(feats, mode="post")
    # First 9 slots = 1.0 (memory + query), remaining 9 = 0.
    assert torch.allclose(vec[:PRE_FEATURE_DIM], torch.ones(PRE_FEATURE_DIM))
    assert torch.allclose(vec[PRE_FEATURE_DIM:], torch.zeros(POST_FEATURE_DIM - PRE_FEATURE_DIM))


def test_assemble_bad_mode_raises():
    import pytest
    with pytest.raises(ValueError):
        assemble_feature_vector({}, mode="middle")


def test_assemble_drops_nans_and_infs():
    feats = {
        "n_facts_retrieved": float("nan"),
        "max_similarity": float("inf"),
        "mean_similarity": 0.5,
    }
    vec = assemble_feature_vector(feats, mode="pre")
    # NaN / inf folded to 0; valid stays.
    assert float(vec[0].item()) == 0.0
    assert float(vec[1].item()) == 0.0
    assert float(vec[2].item()) == 0.5


# ---------- Cardinality sanity ----------

def test_feature_dimensions_match_spec():
    assert PRE_FEATURE_DIM == 9
    assert POST_FEATURE_DIM == 18
    assert len(MEMORY_FEATURE_NAMES) == 6
    assert len(QUERY_FEATURE_NAMES) == 3
    assert len(GENERATION_FEATURE_NAMES) == 4
    assert len(ALIGNMENT_FEATURE_NAMES) == 3
