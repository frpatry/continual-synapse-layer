"""Tests for the metacognitive feature extractors + vector assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

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


# ---------- Generation + alignment placeholders ----------

def test_generation_features_placeholder_zero_filled():
    feats = extract_generation_features(None)
    assert set(feats) == set(GENERATION_FEATURE_NAMES)
    assert all(v == 0.0 for v in feats.values())


def test_alignment_features_placeholder_zero_filled():
    feats = extract_alignment_features(
        response="anything", facts=[], foundation=None,
    )
    assert set(feats) == set(ALIGNMENT_FEATURE_NAMES)
    assert all(v == 0.0 for v in feats.values())


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
