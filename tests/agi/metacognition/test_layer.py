"""Tests for MetacognitiveLayer (forward shapes, predict path)."""

from __future__ import annotations

import pytest
import torch

from agi.metacognition.features import (
    POST_FEATURE_DIM,
    PRE_FEATURE_DIM,
)
from agi.metacognition.layer import MetacognitiveLayer
from agi.metacognition.state import MetacognitiveState


# ---------- Instantiation ----------

def test_instantiate_pre_layer():
    layer = MetacognitiveLayer(mode="pre")
    assert layer.mode == "pre"
    assert layer.in_dim == PRE_FEATURE_DIM


def test_instantiate_post_layer():
    layer = MetacognitiveLayer(mode="post")
    assert layer.mode == "post"
    assert layer.in_dim == POST_FEATURE_DIM


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        MetacognitiveLayer(mode="middle")


# ---------- Forward shapes ----------

def test_forward_single_pre_returns_4_logits_and_scalar_confidence():
    layer = MetacognitiveLayer(mode="pre")
    feats = torch.zeros(PRE_FEATURE_DIM)
    logits, conf = layer(feats)
    assert logits.shape == (4,)
    assert conf.dim() == 0  # scalar
    assert 0.0 <= float(conf.item()) <= 1.0


def test_forward_batch_post_returns_batched_shapes():
    layer = MetacognitiveLayer(mode="post")
    batch = torch.zeros(5, POST_FEATURE_DIM)
    logits, conf = layer(batch)
    assert logits.shape == (5, 4)
    assert conf.shape == (5,)
    assert torch.all((conf >= 0) & (conf <= 1))


def test_forward_wrong_dim_raises():
    layer = MetacognitiveLayer(mode="pre")
    with pytest.raises(ValueError):
        layer(torch.zeros(POST_FEATURE_DIM))  # wrong size for pre


# ---------- predict() ----------

def test_predict_returns_metacognitive_state_from_tensor():
    torch.manual_seed(0)
    layer = MetacognitiveLayer(mode="pre")
    feats = torch.randn(PRE_FEATURE_DIM)
    state = layer.predict(feats)
    assert isinstance(state, MetacognitiveState)
    assert state.epistemic_status in (
        "known", "unknown", "uncertain", "hallucinated",
    )
    assert state.recommended_action in (
        "answer", "answer_with_caveat", "admit_ignorance",
    )
    assert 0.0 <= state.confidence <= 1.0
    assert state.generation_alignment is None  # pre-mode


def test_predict_post_mode_populates_generation_alignment():
    torch.manual_seed(0)
    layer = MetacognitiveLayer(mode="post")
    feats = torch.zeros(POST_FEATURE_DIM)
    state = layer.predict(feats)
    assert state.generation_alignment is not None
    # All-zero post input → alignment is the mean of zero slots = 0.
    assert state.generation_alignment == 0.0


def test_predict_accepts_feature_dict_and_populates_raw_features():
    layer = MetacognitiveLayer(mode="pre")
    feats_dict = {
        "n_facts_retrieved": 2.0,
        "max_similarity": 0.8,
        "mean_similarity": 0.6,
        "similarity_variance": 0.05,
        "max_recency_days": 1.0,
        "mean_access_count": 1.0,
        "query_length_tokens": 5.0,
        "has_named_entity": 1.0,
        "query_specificity": 0.9,
    }
    state = layer.predict(feats_dict)
    # raw_features should mirror the input dict.
    for k, v in feats_dict.items():
        assert state.raw_features[k] == v
    # memory_coverage / quality should reflect the inputs.
    # memory_quality is read back from a float32 tensor → use approx
    # to absorb the fp32 round-trip.
    assert state.memory_coverage == pytest.approx(min(1.0, 2.0 / 3.0))
    assert state.memory_quality == pytest.approx(0.8, abs=1e-5)


def test_predict_status_to_action_mapping_is_consistent():
    """Every status the layer can emit must map to the right
    action — verified by exhausting the 4-way logit head."""
    layer = MetacognitiveLayer(mode="pre")
    # Synthesize each one-hot logit pattern by overwriting the
    # status_head bias and zeroing the weight, so argmax pins
    # to a chosen index regardless of input.
    expected = [
        ("known", "answer"),
        ("uncertain", "answer_with_caveat"),
        ("unknown", "admit_ignorance"),
        ("hallucinated", "admit_ignorance"),
    ]
    with torch.no_grad():
        layer.status_head.weight.zero_()
        for idx, (status, action) in enumerate(expected):
            layer.status_head.bias.zero_()
            layer.status_head.bias[idx] = 1.0
            state = layer.predict(torch.zeros(PRE_FEATURE_DIM))
            assert state.epistemic_status == status
            assert state.recommended_action == action


# ---------- Heuristic / smoke ----------

def test_predict_with_zeros_returns_valid_state_at_random_init():
    """A zero-feature input must produce a structurally valid
    state at random init — no crash, no nan, valid status /
    action. Doesn't assert WHICH status (network isn't trained)."""
    torch.manual_seed(42)
    layer = MetacognitiveLayer(mode="pre")
    state = layer.predict(torch.zeros(PRE_FEATURE_DIM))
    assert isinstance(state, MetacognitiveState)
    import math
    assert not math.isnan(state.confidence)
    assert 0.0 <= state.confidence <= 1.0


@pytest.mark.xfail(
    reason=(
        "Random-init network has no reason to pick admit_ignorance "
        "for zero input — this becomes meaningful only after the "
        "Phase 2b training pass."
    ),
    strict=False,
)
def test_predict_zeros_tends_toward_admit_ignorance_after_training():
    """Heuristic: an empty-memory input should be classified as
    ``unknown`` (→ admit_ignorance). Marked xfail because the
    random-init layer hasn't learned this yet — flips to a real
    assertion in Phase 2b."""
    torch.manual_seed(0)
    layer = MetacognitiveLayer(mode="pre")
    hits = 0
    trials = 16
    for seed in range(trials):
        torch.manual_seed(seed)
        # Re-init each iteration to sample over the random-init
        # distribution, not a single fixed network.
        layer = MetacognitiveLayer(mode="pre")
        state = layer.predict(torch.zeros(PRE_FEATURE_DIM))
        if state.recommended_action == "admit_ignorance":
            hits += 1
    assert hits > trials // 2
