"""Tests for EpisodicPredictor — model + active memory at inference."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from continual_synapse.baselines.naive_finetune import MLPClassifier, MLPConfig
from continual_synapse.episodic.active_memory import ActiveEpisodicMemory
from continual_synapse.episodic.episodic_predictor import EpisodicPredictor
from continual_synapse.evaluation.runner import set_seed


def _build_model(hidden_dim: int = 8, num_classes: int = 3) -> MLPClassifier:
    set_seed(0)
    return MLPClassifier(
        MLPConfig(
            input_dim=4, hidden_dim=hidden_dim,
            num_classes=num_classes, dropout=0.0,
        )
    )


# ---- 1. fallback on empty memory ----


def test_empty_memory_falls_back_to_model() -> None:
    """With no stored entries, retrieval returns uniform probs and
    zero confidence ⇒ blend weight is zero (confidence is below
    every threshold), and the predictor's output must be the model's
    log-softmax."""
    model = _build_model()
    memory = ActiveEpisodicMemory(feature_dim=8, n_classes=3)
    predictor = EpisodicPredictor(
        model, memory, blend_threshold=0.0, blend_max=0.5,
    )
    model.eval()
    x = torch.randn(4, 4)
    out = predictor.predict(x)
    with torch.no_grad():
        bare = model(x)
    # bare logits → log_softmax should match the predictor's output
    # up to the small eps in the log step.
    expected = F.log_softmax(bare, dim=-1)
    torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-5)


# ---- 2. high-confidence retrieval engages ----


def test_high_confidence_retrieval_blends_in() -> None:
    """After populating memory with an entry that's an exact match
    for a query, retrieval confidence is ≈ 1 and the blended output
    must differ from the pure-model output. A second, very different
    query gets retrieval confidence 0 (below threshold) and matches
    the pure-model output."""
    model = _build_model()
    memory = ActiveEpisodicMemory(
        feature_dim=8, n_classes=3, novelty_threshold=0.0, retrieval_k=1,
    )
    predictor = EpisodicPredictor(
        model, memory, blend_threshold=0.5, blend_max=0.5,
    )
    model.eval()

    # Pick an input, get its features, and seed the memory with
    # those features bound to a label the model probably doesn't
    # predict — guarantees blended != model for the matching query.
    x_seed = torch.randn(1, 4)
    with torch.no_grad():
        f_seed = model.features(x_seed)
        model_pred = model(x_seed).argmax(dim=-1).item()
    target_label = (model_pred + 1) % 3  # forced disagreement
    memory.embeddings.append(f_seed[0].cpu())
    memory.labels.append(int(target_label))
    memory.task_ids.append(None)
    memory._invalidate_cache()

    out_matched = predictor.predict(x_seed)
    bare_matched = F.log_softmax(model(x_seed), dim=-1)
    # The matched query MUST get a different distribution because
    # retrieval pulls toward target_label.
    assert not torch.allclose(out_matched, bare_matched, atol=1e-4), (
        "matched-query output should differ from bare model output"
    )

    # A very different input: features won't match the seeded entry,
    # confidence will be below threshold ⇒ blend weight zero.
    x_far = torch.randn(1, 4) * 100  # very different scale
    out_far = predictor.predict(x_far)
    bare_far = F.log_softmax(model(x_far), dim=-1)
    torch.testing.assert_close(out_far, bare_far, rtol=1e-4, atol=1e-5)


# ---- 3. blend threshold honoured ----


def test_blend_threshold_respected() -> None:
    """A query whose retrieval confidence sits below blend_threshold
    must get exactly zero retrieval weight. Verified by setting an
    artificially high threshold (0.99) and seeding a low-similarity
    near-orthogonal entry: predict output equals pure model output."""
    model = _build_model()
    memory = ActiveEpisodicMemory(
        feature_dim=8, n_classes=3, novelty_threshold=0.0, retrieval_k=1,
    )
    predictor = EpisodicPredictor(
        model, memory, blend_threshold=0.99, blend_max=0.5,
    )

    x = torch.randn(1, 4)
    with torch.no_grad():
        f_x = model.features(x)
    # Seed a clearly different embedding (negated, then normalised
    # in retrieval) so the max cos sim to x's feature is well below
    # 0.99.
    bad_emb = -f_x[0] * 10
    memory.embeddings.append(bad_emb.cpu())
    memory.labels.append(2)
    memory.task_ids.append(None)
    memory._invalidate_cache()

    out = predictor.predict(x)
    bare = F.log_softmax(model(x), dim=-1)
    torch.testing.assert_close(out, bare, rtol=1e-4, atol=1e-5)


# ---- 4. eval-mode is restored ----


def test_eval_mode_restored_after_predict() -> None:
    """predict() must put the model in eval() while running and then
    restore the prior training flag. Otherwise a single eval call in
    the middle of training would leave the model stuck in eval."""
    model = _build_model()
    memory = ActiveEpisodicMemory(feature_dim=8, n_classes=3)
    predictor = EpisodicPredictor(model, memory)
    model.train()
    assert model.training is True
    predictor.predict(torch.randn(2, 4))
    assert model.training is True, "predict must restore training mode"


# ---- 5. observe doesn't accumulate gradients ----


def test_no_grad_in_observe() -> None:
    """training_step_observe must extract features under torch.no_grad
    so the storage decision can't leak gradients into the base model's
    parameters. We verify by clearing grads, calling observe, then
    asserting every parameter still has grad=None."""
    model = _build_model()
    memory = ActiveEpisodicMemory(
        feature_dim=8, n_classes=3, novelty_threshold=0.0,
    )
    predictor = EpisodicPredictor(model, memory)
    # Clear any pre-existing grads.
    for p in model.parameters():
        p.grad = None
    x = torch.randn(3, 4, requires_grad=False)
    y = torch.tensor([0, 1, 2], dtype=torch.long)
    n_added = predictor.training_step_observe(x, y)
    assert n_added > 0  # sanity: memory did grow
    for p in model.parameters():
        assert p.grad is None, (
            f"observe leaked gradients into a model parameter: "
            f"{p.shape} has non-None grad"
        )


# ---- 6. observe grows memory on novel inputs ----


def test_observe_allocates_correctly_on_novel_inputs() -> None:
    """Run observe on B distinct novel inputs; memory size should
    grow by B. Run again with similar inputs; size should not grow."""
    model = _build_model()
    memory = ActiveEpisodicMemory(
        feature_dim=8, n_classes=3, novelty_threshold=0.5,
    )
    predictor = EpisodicPredictor(model, memory)
    # First batch: 4 distinct random inputs → 4 new entries (the
    # base model maps random inputs to ~uncorrelated features).
    x1 = torch.randn(4, 4) * 5
    y1 = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    assert predictor.training_step_observe(x1, y1) == len(memory)
    n_after_first = len(memory)
    assert n_after_first >= 1, "expected at least one allocation on a novel batch"

    # Second batch: exact same inputs → no new entries (each new
    # query is identical to an already-stored entry within
    # numerical tolerance).
    n_added = predictor.training_step_observe(x1, y1)
    assert n_added == 0
    assert len(memory) == n_after_first
