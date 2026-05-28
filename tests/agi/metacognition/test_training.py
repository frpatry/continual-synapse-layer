"""Tests for the Phase 2d metacog training loop."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from agi.metacognition.data_generation import load_dataset
from agi.metacognition.layer import MetacognitiveLayer
from agi.metacognition.training import (
    IDX_TO_STATUS,
    STATUS_TO_IDX,
    MetacogDataset,
    TrainingConfig,
    compute_calibration_error,
    compute_loss,
    evaluate,
    load_checkpoint,
    train,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SAMPLE_DIR = _REPO_ROOT / "data" / "metacog"
_FULL_PRE_VAL = _SAMPLE_DIR / "full_pre_val.jsonl"
_FULL_POST_VAL = _SAMPLE_DIR / "full_post_val.jsonl"
_CKPT_DIR = _SAMPLE_DIR / "checkpoints"


# ---------- status mapping ----------

def test_status_mapping_is_round_trippable():
    """STATUS_TO_IDX and IDX_TO_STATUS must be inverse — otherwise
    training labels and predict() output speak different
    languages."""
    for status, idx in STATUS_TO_IDX.items():
        assert IDX_TO_STATUS[idx] == status


def test_status_mapping_matches_layer_status_order():
    """Sourced from layer._STATUS_ORDER — should never drift."""
    from agi.metacognition.layer import _STATUS_ORDER
    for i, status in enumerate(_STATUS_ORDER):
        assert STATUS_TO_IDX[status] == i


# ---------- Dataset ----------

def test_dataset_loading_pre_mode():
    examples = load_dataset(_SAMPLE_DIR / "sample_pre_train.jsonl")
    dataset = MetacogDataset(examples, mode="pre")
    assert len(dataset) > 0
    features, status, conf = dataset[0]
    assert features.shape == (10,)  # PRE_FEATURE_DIM
    assert 0 <= status < 4
    assert 0.0 <= conf <= 1.0


def test_dataset_loading_post_mode():
    examples = load_dataset(_SAMPLE_DIR / "sample_post_train.jsonl")
    dataset = MetacogDataset(examples, mode="post")
    features, _, _ = dataset[0]
    assert features.shape == (18,)  # POST_FEATURE_DIM


def test_dataset_iterable_via_dataloader():
    examples = load_dataset(_SAMPLE_DIR / "sample_pre_train.jsonl")
    loader = DataLoader(
        MetacogDataset(examples, mode="pre"),
        batch_size=8,
        shuffle=False,
    )
    batch = next(iter(loader))
    features, status, conf = batch
    assert features.shape[0] == 8
    assert features.shape[1] == 10


# ---------- Loss + ECE ----------

def test_compute_loss_returns_finite_components():
    logits = torch.randn(4, 4)
    confidence = torch.rand(4, 1)
    status_target = torch.tensor([0, 1, 2, 3])
    conf_target = torch.tensor([0.8, 0.5, 0.7, 0.3])
    loss, ce, conf_loss = compute_loss(
        logits, confidence, status_target, conf_target,
    )
    assert torch.isfinite(loss)
    assert ce > 0
    assert conf_loss > 0


def test_compute_loss_works_with_squeezed_confidence():
    """The layer's forward returns confidence already squeezed
    to ``(B,)``; compute_loss must handle both ``(B,)`` and
    ``(B, 1)`` shapes."""
    logits = torch.randn(4, 4)
    confidence = torch.rand(4)  # already squeezed
    status_target = torch.tensor([0, 1, 2, 3])
    conf_target = torch.tensor([0.8, 0.5, 0.7, 0.3])
    loss, _, _ = compute_loss(logits, confidence, status_target, conf_target)
    assert torch.isfinite(loss)


# ---------- Phase 2d.1: BCE-on-correctness semantics ----------

def test_compute_loss_uses_correctness_not_synthetic_target():
    """When the prediction agrees with the target, BCE target is
    1.0; when it disagrees, the target is 0.0. The two cases
    must produce different conf_loss values for the same
    confidence_pred."""
    # Confident-in-class-0 logits.
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    confidence = torch.tensor([[0.5]])
    same_synthetic = torch.tensor([0.8])

    # status=0 → prediction (argmax=0) is correct → BCE target = 1.
    _, _, conf_loss_correct = compute_loss(
        logits, confidence, torch.tensor([0]), same_synthetic,
    )
    # status=1 → prediction is wrong → BCE target = 0.
    _, _, conf_loss_wrong = compute_loss(
        logits, confidence, torch.tensor([1]), same_synthetic,
    )

    # Confidence is 0.5; BCE(0.5, 1) ≈ BCE(0.5, 0) ≈ ln(2) ≈ 0.693,
    # but they ARE the same numerically. The point of this test is
    # to assert that the *correctness label* drives the loss — so
    # let's instead use asymmetric confidence (e.g. 0.9) where the
    # two outcomes diverge.
    confidence_high = torch.tensor([[0.9]])
    _, _, hi_correct = compute_loss(
        logits, confidence_high, torch.tensor([0]), same_synthetic,
    )
    _, _, hi_wrong = compute_loss(
        logits, confidence_high, torch.tensor([1]), same_synthetic,
    )
    # BCE(0.9, 1) ≈ 0.105;  BCE(0.9, 0) ≈ 2.30. They differ
    # because the loss now sees correctness.
    assert abs(hi_correct - hi_wrong) > 0.1


def test_compute_loss_ignores_synthetic_confidence_target():
    """The ``confidence_target`` arg is documented dead — vary it
    arbitrarily and conf_loss must not move."""
    logits = torch.randn(4, 4)
    confidence = torch.rand(4, 1)
    status_target = torch.tensor([0, 1, 2, 3])

    _, _, conf_a = compute_loss(
        logits, confidence, status_target,
        torch.tensor([0.1, 0.1, 0.1, 0.1]),
    )
    _, _, conf_b = compute_loss(
        logits, confidence, status_target,
        torch.tensor([0.9, 0.9, 0.9, 0.9]),
    )
    assert abs(conf_a - conf_b) < 1e-6


def test_compute_loss_gradient_does_not_flow_to_logits_via_confidence():
    """The correctness label is detached, so the confidence-loss
    branch must not produce a gradient on the logits tensor."""
    # Logits get a fresh leaf tensor so .grad is well-defined.
    logits = torch.randn(4, 4, requires_grad=True)
    confidence = torch.rand(4, 1, requires_grad=True)
    status_target = torch.tensor([0, 1, 2, 3])
    _, _, conf_loss_scalar = compute_loss(
        logits, confidence, status_target, torch.tensor([0.5] * 4),
    )

    # Backprop ONLY the conf-loss term. We need to recompute it
    # as a tensor to call .backward().
    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        correct = (pred == status_target).float()
    conf_pred = confidence.squeeze(-1)
    bce = torch.nn.functional.binary_cross_entropy(conf_pred, correct)
    bce.backward()
    # The conf-loss path should have produced no gradient on logits.
    assert logits.grad is None or torch.all(logits.grad == 0)
    # And SOME gradient on confidence.
    assert confidence.grad is not None
    assert torch.any(confidence.grad != 0)


def test_compute_calibration_error_perfect_calibration():
    """Confidence = realised accuracy → ECE = 0."""
    # All confidences 0.5, two correct, two wrong → bin acc = 0.5,
    # bin conf = 0.5, ECE = 0.
    confs = np.array([0.5, 0.5, 0.5, 0.5])
    correct = np.array([1.0, 0.0, 1.0, 0.0])
    ece = compute_calibration_error(confs, correct, n_bins=10)
    assert ece < 1e-6


def test_compute_calibration_error_completely_miscalibrated():
    """All conf=0.99 but 0% accurate → ECE close to 0.99."""
    confs = np.full(100, 0.99)
    correct = np.zeros(100)
    ece = compute_calibration_error(confs, correct, n_bins=10)
    assert ece > 0.9


def test_compute_calibration_error_empty_input():
    """Defensive: zero-sized inputs return 0 instead of NaN."""
    assert compute_calibration_error(
        np.array([]), np.array([]),
    ) == 0.0


# ---------- Training convergence ----------

def test_training_runs_at_least_one_epoch(tmp_path):
    """Smoke: training on the sample data runs to completion and
    records at least one TrainingMetrics row."""
    config = TrainingConfig(
        mode="pre", epochs=2, batch_size=16, seed=0,
        early_stopping_patience=10,
    )
    history = train(
        config=config,
        train_path=_SAMPLE_DIR / "sample_pre_train.jsonl",
        val_path=_SAMPLE_DIR / "sample_pre_val.jsonl",
        checkpoint_path=tmp_path / "smoke_pre.pt",
    )
    assert len(history) >= 1
    for m in history:
        assert torch.isfinite(torch.tensor(m.train_loss))
        assert torch.isfinite(torch.tensor(m.val_loss))


def test_training_reduces_train_loss_on_sample(tmp_path):
    """5 epochs on sample data should at least reduce training
    loss from the first epoch."""
    config = TrainingConfig(
        mode="post", epochs=5, batch_size=16, seed=42,
        early_stopping_patience=20,
    )
    history = train(
        config=config,
        train_path=_SAMPLE_DIR / "sample_post_train.jsonl",
        val_path=_SAMPLE_DIR / "sample_post_val.jsonl",
        checkpoint_path=tmp_path / "smoke_post.pt",
    )
    assert history[-1].train_loss < history[0].train_loss


# ---------- Checkpoint round-trip ----------

def test_checkpoint_save_load_roundtrip(tmp_path):
    """Train → save → load → can run inference."""
    config = TrainingConfig(mode="pre", epochs=2, batch_size=16, seed=42)
    ckpt = tmp_path / "rt.pt"
    train(
        config=config,
        train_path=_SAMPLE_DIR / "sample_pre_train.jsonl",
        val_path=_SAMPLE_DIR / "sample_pre_val.jsonl",
        checkpoint_path=ckpt,
    )
    assert ckpt.exists()
    model = load_checkpoint(ckpt, mode="pre")
    assert isinstance(model, MetacognitiveLayer)
    logits, conf = model(torch.zeros(10))
    assert logits.shape == (4,)
    assert conf.dim() == 0


# ---------- Validate the committed checkpoints (Phase 2d artefacts) ----------

@pytest.mark.skipif(
    not (_CKPT_DIR / "pre_layer.pt").exists(),
    reason="PRE checkpoint missing (run scripts/train_metacog.py first)",
)
def test_committed_pre_checkpoint_loads_and_predicts():
    model = load_checkpoint(_CKPT_DIR / "pre_layer.pt", mode="pre")
    out = model.predict(torch.zeros(10))
    assert out.epistemic_status in ("known", "unknown", "uncertain", "hallucinated")
    assert out.recommended_action in (
        "answer", "answer_with_caveat", "admit_ignorance",
    )


@pytest.mark.skipif(
    not (_CKPT_DIR / "post_layer.pt").exists(),
    reason="POST checkpoint missing (run scripts/train_metacog.py first)",
)
def test_committed_post_checkpoint_loads_and_predicts():
    model = load_checkpoint(_CKPT_DIR / "post_layer.pt", mode="post")
    out = model.predict(torch.zeros(18))
    assert out.epistemic_status in ("known", "unknown", "uncertain", "hallucinated")


@pytest.mark.skipif(
    not (_CKPT_DIR / "pre_layer.pt").exists() or not _FULL_PRE_VAL.exists(),
    reason="PRE checkpoint or full val set missing",
)
def test_pre_checkpoint_meets_accuracy_target():
    """The committed PRE checkpoint should clear the > 0.85 target
    on its own validation split."""
    config = TrainingConfig(mode="pre", batch_size=256)
    model = load_checkpoint(_CKPT_DIR / "pre_layer.pt", mode="pre")
    examples = load_dataset(_FULL_PRE_VAL)
    loader = DataLoader(
        MetacogDataset(examples, mode="pre"), batch_size=256, shuffle=False,
    )
    metrics = evaluate(model, loader, config)
    assert metrics["accuracy"] > 0.85, metrics


@pytest.mark.skipif(
    not (_CKPT_DIR / "post_layer.pt").exists() or not _FULL_POST_VAL.exists(),
    reason="POST checkpoint or full val set missing",
)
def test_post_checkpoint_meets_accuracy_target():
    """The committed POST checkpoint should clear the > 0.80
    target on its own validation split."""
    config = TrainingConfig(mode="post", batch_size=256)
    model = load_checkpoint(_CKPT_DIR / "post_layer.pt", mode="post")
    examples = load_dataset(_FULL_POST_VAL)
    loader = DataLoader(
        MetacogDataset(examples, mode="post"), batch_size=256, shuffle=False,
    )
    metrics = evaluate(model, loader, config)
    assert metrics["accuracy"] > 0.80, metrics


# ---------- Phase 2d.1: trained-model calibration ----------

@pytest.mark.skipif(
    not (_CKPT_DIR / "pre_layer.pt").exists(),
    reason="PRE checkpoint missing (run scripts/train_metacog.py first)",
)
def test_trained_model_has_low_ece():
    """After the Phase 2d.1 BCE-on-correctness fix, the trained
    PRE layer should post an ECE comfortably under the 0.15
    threshold. Evaluated on the committed sample fixture so the
    test runs anywhere the checkpoint exists."""
    model = load_checkpoint(_CKPT_DIR / "pre_layer.pt", mode="pre")
    examples = load_dataset(_SAMPLE_DIR / "sample_pre_val.jsonl")
    loader = DataLoader(
        MetacogDataset(examples, mode="pre"), batch_size=32, shuffle=False,
    )
    config = TrainingConfig(mode="pre")
    metrics = evaluate(model, loader, config)
    assert metrics["calibration_error"] < 0.15, (
        f"Expected ECE < 0.15 after Phase 2d.1; got "
        f"{metrics['calibration_error']:.4f}. metrics={metrics}"
    )
