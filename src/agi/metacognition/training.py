"""Training loop for the metacognitive layers (Phase 2d).

Trains the two :class:`agi.metacognition.layer.MetacognitiveLayer`
instances on the synthetic data produced in Phase 2c:

- **PRE**  (10 features → 3 in-use classes): tells the chat loop
  *before* generation whether to bother calling the LLM.
- **POST** (18 features → 4 classes): tells the chat loop *after*
  generation whether to trust the response or fall back to a
  template.

Loss is a weighted sum of cross-entropy (status classification)
and BCE (confidence vs on-the-fly correctness) with the
confidence term scaled by ``confidence_loss_weight`` (default
0.2 — status is the primary signal, confidence calibration is
secondary).

**Phase 2d.1 calibration fix.** The confidence head is now
trained to predict ``P(classification correct)`` via BCE against
``(argmax(logits) == status_target).float()`` — *not* to mimic
the per-example synthetic ``Beta``-distributed targets in
:attr:`agi.metacognition.data_generation.TrainingExample.confidence`.
That field is preserved on the dataclass for backward
compatibility but is **ignored during training**. The result is
a confidence score that means "probability this classification
is right" — directly usable by the Phase 2e reward-gating loop.

.. warning::
   Feature dimensions are **locked** by these checkpoints:

   - ``PRE_FEATURE_DIM = 10``
   - ``POST_FEATURE_DIM = 18``

   Adding or reordering features in any later phase invalidates
   the committed ``data/metacog/checkpoints/*.pt`` files. The
   single source of truth for ordering lives in
   :mod:`agi.metacognition.features` constants
   (``PRE_FEATURE_ORDER`` / ``POST_FEATURE_ORDER``); the single
   source of truth for the status-index mapping lives in
   :data:`agi.metacognition.layer._STATUS_ORDER` and is mirrored
   here as :data:`STATUS_TO_IDX`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data_generation import (
    POST_CLASSES,
    PRE_CLASSES,
    TrainingExample,
    load_dataset,
)
from .features import assemble_feature_vector
from .layer import MetacognitiveLayer, _STATUS_ORDER


# Derive the status ↔ index mapping from the layer's canonical
# ordering. This way trained logits at index ``i`` are interpreted
# the same way by training (the loss target) and by
# :meth:`MetacognitiveLayer.predict` (the inference path). Spec
# wrote out the literal mapping; we preserve operational
# equivalence by sourcing from layer.py.
STATUS_TO_IDX: dict[str, int] = {s: i for i, s in enumerate(_STATUS_ORDER)}
IDX_TO_STATUS: dict[int, str] = {i: s for i, s in enumerate(_STATUS_ORDER)}


@dataclass
class TrainingConfig:
    """Hyperparameters for a single training run."""

    mode: Literal["pre", "post"]
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    confidence_loss_weight: float = 0.2
    early_stopping_patience: int = 5
    lr_scheduler_patience: int = 3
    lr_scheduler_factor: float = 0.5
    seed: int = 42
    # Phase 2d is CPU-only by spec. The field is kept for forward
    # compat but no device juggling is done in this file.
    device: str = "cpu"


@dataclass
class TrainingMetrics:
    """One epoch's training + validation snapshot."""

    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    per_class_f1: dict[str, float] = field(default_factory=dict)
    calibration_error: float = 0.0


class MetacogDataset(Dataset):
    """Wrap a list of :class:`TrainingExample` into a PyTorch
    Dataset, projecting each example's feature dict into the right-
    sized tensor via :func:`assemble_feature_vector`.

    ``__getitem__`` returns ``(features_tensor, status_idx,
    confidence)`` so the default ``DataLoader`` collate stacks
    cleanly into ``(features_batch, status_idx_batch,
    confidence_batch)``.
    """

    def __init__(
        self,
        examples: list[TrainingExample],
        mode: Literal["pre", "post"],
    ) -> None:
        self.examples = examples
        self.mode = mode

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        features = assemble_feature_vector(ex.features, mode=self.mode)
        status_idx = STATUS_TO_IDX[ex.status]
        confidence = float(ex.confidence)
        return features, status_idx, confidence


# ----------------------------------------------------------------------
# Loss + calibration
# ----------------------------------------------------------------------

def compute_loss(
    logits: torch.Tensor,
    confidence_pred: torch.Tensor,
    status_target: torch.Tensor,
    confidence_target: torch.Tensor,
    confidence_weight: float = 0.2,
) -> tuple[torch.Tensor, float, float]:
    """Combined CE-over-status + BCE-over-correctness loss.

    Phase 2d.1 semantics:

    - ``ce_loss = CrossEntropy(logits, status_target)`` — unchanged.
    - ``conf_loss = BCE(confidence_pred,
       (argmax(logits) == status_target).float())`` — confidence is
       trained against on-the-fly correctness, NOT against the
       per-example synthetic ``confidence_target``.

    ``confidence_target`` is kept in the signature for backward
    compatibility with existing call sites + tests but is
    **ignored** by the computation.

    Gradient flow: the correctness label is computed under
    :func:`torch.no_grad`, so the confidence-loss gradient
    backprops only through the ``confidence_pred`` path. This
    prevents a pathological optimum where the model would
    *deliberately misclassify* in order to make confidence
    cheaper to predict.
    """
    ce_loss = F.cross_entropy(logits, status_target)

    # confidence_pred may arrive as ``(B,)`` (the layer's already-
    # squeezed output) or ``(B, 1)`` (raw test wiring); the
    # squeeze handles both shapes.
    conf_pred = (
        confidence_pred.squeeze(-1)
        if confidence_pred.dim() > 1
        else confidence_pred
    )

    # On-the-fly correctness. ``argmax`` is non-differentiable
    # already, but the explicit ``no_grad`` makes the intent
    # obvious + defends against a future refactor that swaps in
    # a differentiable proxy.
    with torch.no_grad():
        predicted = logits.argmax(dim=-1)
        correct = (predicted == status_target).float()

    conf_loss = F.binary_cross_entropy(conf_pred, correct)

    total = ce_loss + confidence_weight * conf_loss
    return total, float(ce_loss.item()), float(conf_loss.item())


def compute_calibration_error(
    confidence_preds: np.ndarray,
    correct_mask: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    Bins confidences into ``n_bins`` equal-width buckets in
    ``[0, 1]``; for each non-empty bin, accumulates
    ``(bin_size / N) * |mean_accuracy - mean_confidence|``.
    A perfectly calibrated model returns 0; a model that's
    over-confident or under-confident returns > 0.
    """
    confidence_preds = np.asarray(confidence_preds, dtype=float)
    correct_mask = np.asarray(correct_mask, dtype=float)
    if confidence_preds.size == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n_total = float(confidence_preds.size)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Include upper edge in the last bin so confidence == 1.0
        # doesn't fall off the edge.
        if i == n_bins - 1:
            mask = (confidence_preds >= lo) & (confidence_preds <= hi)
        else:
            mask = (confidence_preds >= lo) & (confidence_preds < hi)
        if not mask.any():
            continue
        bin_size = float(mask.sum())
        bin_acc = float(correct_mask[mask].mean())
        bin_conf = float(confidence_preds[mask].mean())
        ece += (bin_size / n_total) * abs(bin_acc - bin_conf)
    return float(ece)


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def evaluate(
    model: MetacognitiveLayer,
    loader: DataLoader,
    config: TrainingConfig,
) -> dict:
    """Run validation, return loss / accuracy / per-class F1 / ECE."""
    model.eval()
    total_loss = 0.0
    n_seen = 0
    all_preds: list[int] = []
    all_targets: list[int] = []
    all_confs: list[float] = []

    with torch.no_grad():
        for features, status_target, conf_target in loader:
            # PyTorch's default collate gives a long tensor for
            # the status indices and a float64 tensor for the
            # confidence; CrossEntropy needs int64, MSE needs
            # float32.
            status_target = status_target.to(torch.long)
            conf_target = conf_target.to(torch.float32)
            logits, confidence = model(features)
            loss, _, _ = compute_loss(
                logits, confidence, status_target, conf_target,
                confidence_weight=config.confidence_loss_weight,
            )
            bs = features.shape[0]
            total_loss += float(loss.item()) * bs
            n_seen += bs
            preds = logits.argmax(dim=-1).tolist()
            all_preds.extend(preds)
            all_targets.extend(status_target.tolist())
            conf_flat = (
                confidence.squeeze(-1) if confidence.dim() > 1 else confidence
            )
            all_confs.extend(conf_flat.tolist())

    preds_arr = np.array(all_preds)
    targets_arr = np.array(all_targets)
    confs_arr = np.array(all_confs, dtype=float)

    accuracy = float((preds_arr == targets_arr).mean()) if n_seen else 0.0

    # Per-class F1: PRE has 3 in-use classes, POST has 4.
    classes = PRE_CLASSES if config.mode == "pre" else POST_CLASSES
    per_class_f1: dict[str, float] = {}
    for cls_name in classes:
        cls_idx = STATUS_TO_IDX[cls_name]
        tp = int(((preds_arr == cls_idx) & (targets_arr == cls_idx)).sum())
        fp = int(((preds_arr == cls_idx) & (targets_arr != cls_idx)).sum())
        fn = int(((preds_arr != cls_idx) & (targets_arr == cls_idx)).sum())
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        per_class_f1[cls_name] = float(f1)

    correct_mask = (preds_arr == targets_arr).astype(float)
    calibration = compute_calibration_error(confs_arr, correct_mask)

    return {
        "loss": total_loss / n_seen if n_seen else 0.0,
        "accuracy": accuracy,
        "per_class_f1": per_class_f1,
        "calibration_error": calibration,
    }


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------

def train(
    config: TrainingConfig,
    train_path: Path,
    val_path: Path,
    checkpoint_path: Path,
) -> list[TrainingMetrics]:
    """Train one layer, write the best checkpoint, return history.

    Best-checkpoint policy: tracks ``val_accuracy``; only writes
    when a new maximum is reached. Early stopping triggers when
    ``early_stopping_patience`` consecutive epochs see no
    improvement.
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    train_examples = load_dataset(train_path)
    val_examples = load_dataset(val_path)
    train_ds = MetacogDataset(train_examples, mode=config.mode)
    val_ds = MetacogDataset(val_examples, mode=config.mode)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    model = MetacognitiveLayer(mode=config.mode)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience,
    )

    history: list[TrainingMetrics] = []
    best_val_acc = -math.inf
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        total_train_loss = 0.0
        n_train = 0
        train_correct = 0
        for features, status_target, conf_target in train_loader:
            status_target = status_target.to(torch.long)
            conf_target = conf_target.to(torch.float32)
            optimizer.zero_grad()
            logits, confidence = model(features)
            loss, _, _ = compute_loss(
                logits, confidence, status_target, conf_target,
                confidence_weight=config.confidence_loss_weight,
            )
            loss.backward()
            optimizer.step()
            bs = features.shape[0]
            total_train_loss += float(loss.item()) * bs
            n_train += bs
            train_correct += int((logits.argmax(dim=-1) == status_target).sum().item())

        train_loss = total_train_loss / n_train
        train_acc = train_correct / n_train
        val = evaluate(model, val_loader, config)
        scheduler.step(val["accuracy"])

        history.append(
            TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val["loss"],
                train_accuracy=train_acc,
                val_accuracy=val["accuracy"],
                per_class_f1=val["per_class_f1"],
                calibration_error=val["calibration_error"],
            )
        )

        if val["accuracy"] > best_val_acc:
            best_val_acc = val["accuracy"]
            patience_counter = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "epoch": epoch,
                    "best_val_accuracy": float(best_val_acc),
                    "per_class_f1": val["per_class_f1"],
                    "calibration_error": val["calibration_error"],
                    "feature_dim": 10 if config.mode == "pre" else 18,
                    "n_classes": 3 if config.mode == "pre" else 4,
                    "status_order": list(_STATUS_ORDER),
                    "trained_at": datetime.now().isoformat(),
                },
                checkpoint_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                break

    return history


def load_checkpoint(
    checkpoint_path: Path,
    mode: Literal["pre", "post"],
) -> MetacognitiveLayer:
    """Load a checkpoint into a fresh ``MetacognitiveLayer``.

    Returns the model in eval mode. The caller can switch to
    train mode if they want to fine-tune; for inference the eval
    setting matches the predict path's no-grad expectation.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = MetacognitiveLayer(mode=mode)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def save_training_metadata(
    out_path: Path,
    per_layer: dict[str, dict],
) -> None:
    """Write the small JSON metadata blob that summarises a
    train_metacog.py run. Schema: ``{ "pre": {config, metrics},
    "post": {config, metrics} }``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(per_layer, f, indent=2)


__all__ = [
    "IDX_TO_STATUS",
    "MetacogDataset",
    "STATUS_TO_IDX",
    "TrainingConfig",
    "TrainingMetrics",
    "compute_calibration_error",
    "compute_loss",
    "evaluate",
    "load_checkpoint",
    "save_training_metadata",
    "train",
]
