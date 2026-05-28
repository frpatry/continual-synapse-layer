"""Experiment 48 — Phase 5.7.1: XRayMemory integration smoke.

End-to-end mechanical verification of the X-Ray training pipeline
on a synthetic 2-task class-incremental setup before scaling to
real benchmarks (Split-MNIST CI in Phase 5.7.2, CIFAR-100 later).

What this script integrates from Phase 5.7.0:

- :class:`XRayMemory` (per-class prototype storage with EMA
  refinement, sparsification, temperature schedule).
- :func:`nt_xent_multi_prototype_loss` (multi-positive supervised
  contrastive against the prototype set).

Dual-substrate architecture (Option A from the spec):

- A small hippocampe and a small neocortex share inputs.
- A SINGLE shared XRayMemory stores prototypes built from the
  neocortex's feature space.
- Per-batch the neocortex sees (CE on current labels) + λ·(NT-Xent
  vs the entire prototype set). Memory is then updated using the
  neocortex's features only for entries that were classified
  correctly.
- The hippocampe trains independently on CE (no contrastive). It
  serves as a "naive control" so we can see catastrophic forgetting
  in the same run.

Consolidation phase (after each task):

- A short fine-tuning loop where the neocortex's CLASSIFIER HEAD
  ONLY is updated against (prototype, prototype_label) pairs via
  cross-entropy. The encoder is frozen for this step — the
  prototypes are the encoder's "ground truth" snapshot of past
  classes and we don't want to perturb them.

Synthetic setup:

- ``num_classes=10``, two tasks of 5 classes each (CI: only
  current task's classes seen during a given task's training).
- Per-class isotropic Gaussian clusters in a 32-D input space.
- TinyEncoder maps 32-D → 64-D features; TinyClassifier maps
  64-D → 10-D logits.

Verdict block at the end checks:

- No NaN/Inf in any per-batch loss or eval ACC.
- Memory size ends in a sensible range (≥ 5 prototypes, ≤
  num_classes × prototypes_per_class).
- Task-0 retention after task 1 (neocortex ACC on classes 0..4)
  > 0.30 — the spec's threshold for "beats naive forgetting".
- Hippocampe shows catastrophic forgetting (task-0 retention
  near chance after task 1) — the negative control.

Run from the repo root::

    python experiments/48_xray_integration_smoke.py
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from continual_synapse.evaluation.runner import set_seed  # noqa: E402
from continual_synapse.memory import (  # noqa: E402
    XRayMemory, nt_xent_multi_prototype_loss,
)


# ---------- tiny models ----------


class TinyEncoderClassifier(nn.Module):
    """Two-layer MLP encoder + linear classifier.

    Exposes ``features(x)`` returning a (B, feature_dim) tensor and
    ``classifier`` as a separate ``nn.Linear`` so the consolidation
    step can freeze the encoder and update only the head.
    """

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 64,
        feature_dim: int = 64,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def features(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.encoder(x))


# ---------- synthetic data ----------


def _make_synthetic_ci_data(
    num_classes: int = 10, samples_per_class: int = 400,
    input_dim: int = 32, cluster_std: float = 0.4,
    seed: int = 0,
) -> tuple[list[tuple[Tensor, Tensor]], list[tuple[Tensor, Tensor]]]:
    """Return ``[(x_class0, y_class0), …, (x_classN-1, y_classN-1)]`` for
    both train and test splits.

    Each class is a Gaussian cluster around a well-separated mean.
    Test set uses the same per-class means with fresh draws.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    means = torch.randn(num_classes, input_dim, generator=g) * 3.0
    train, test = [], []
    n_test = samples_per_class // 4
    for c in range(num_classes):
        x_tr = means[c] + cluster_std * torch.randn(
            samples_per_class, input_dim, generator=g,
        )
        y_tr = torch.full((samples_per_class,), c, dtype=torch.long)
        x_te = means[c] + cluster_std * torch.randn(
            n_test, input_dim, generator=g,
        )
        y_te = torch.full((n_test,), c, dtype=torch.long)
        train.append((x_tr, y_tr))
        test.append((x_te, y_te))
    return train, test


def _task_loader(
    per_class_data: list[tuple[Tensor, Tensor]],
    task_classes: list[int], batch_size: int, shuffle: bool,
) -> torch.utils.data.DataLoader:
    xs = torch.cat([per_class_data[c][0] for c in task_classes], dim=0)
    ys = torch.cat([per_class_data[c][1] for c in task_classes], dim=0)
    ds = torch.utils.data.TensorDataset(xs, ys)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ---------- per-batch training step ----------


def train_step_dual_xray(
    hipp: TinyEncoderClassifier,
    neo: TinyEncoderClassifier,
    memory: XRayMemory,
    hipp_optimizer: torch.optim.Optimizer,
    neo_optimizer: torch.optim.Optimizer,
    x_batch: Tensor, y_batch: Tensor,
    *,
    lambda_contrast: float,
    grad_clip: float | None = 1.0,
) -> dict[str, float]:
    """Hipp: CE only (naive control). Neo: CE + λ·NT-Xent vs all
    prototypes. Memory is updated using neo features for
    correctly-classified samples."""
    # ----- Hippocampe (naive control) -----
    hipp_optimizer.zero_grad()
    hipp_logits = hipp(x_batch)
    hipp_loss = F.cross_entropy(hipp_logits, y_batch)
    hipp_loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(hipp.parameters(), max_norm=grad_clip)
    hipp_optimizer.step()

    # ----- Neocortex (CE + contrastive vs prototypes) -----
    neo_optimizer.zero_grad()
    neo_features = neo.features(x_batch)
    neo_logits = neo.classifier(neo_features)
    ce_loss = F.cross_entropy(neo_logits, y_batch)

    contrast_loss = torch.zeros((), device=x_batch.device)
    if memory.num_occupied() > 0:
        prototypes, proto_labels = memory.get_all_prototypes()
        # Memory buffers live on memory's device; pull to the
        # working device for this forward.
        prototypes = prototypes.to(x_batch.device)
        proto_labels = proto_labels.to(x_batch.device)
        contrast_loss = nt_xent_multi_prototype_loss(
            neo_features, y_batch, prototypes, proto_labels,
            temperature=memory.temperature(),
        )

    total_neo_loss = ce_loss + lambda_contrast * contrast_loss
    total_neo_loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(neo.parameters(), max_norm=grad_clip)
    neo_optimizer.step()

    # ----- Memory update (no grad, after the step) -----
    with torch.no_grad():
        pred = neo_logits.argmax(dim=-1)
        correct_mask = (pred == y_batch)
        # Re-detach features to be defensive; they were already
        # detached by .item() / forward semantics but explicit is
        # better than implicit when crossing the autograd boundary.
        memory.update(neo_features.detach(), y_batch, correct_mask)

    return {
        "hipp_loss":     float(hipp_loss.item()),
        "ce_loss":       float(ce_loss.item()),
        "contrast_loss": float(contrast_loss.item()),
        "total_loss":    float(total_neo_loss.item()),
    }


# ---------- consolidation (classifier-head-only) ----------


def consolidate_with_xray(
    neo: TinyEncoderClassifier,
    memory: XRayMemory,
    classifier_optimizer: torch.optim.Optimizer,
    *,
    cons_epochs: int = 2,
    grad_clip: float | None = 1.0,
) -> dict[str, float]:
    """Fine-tune the neocortex classifier head on the prototype set.

    Encoder is frozen (its parameters were not added to
    ``classifier_optimizer``). Each "epoch" is one pass over the
    entire prototype set treated as a batch. This is intended to
    re-align the classifier with the prototype geometry without
    perturbing the encoder.
    """
    prototypes, proto_labels = memory.get_all_prototypes()
    if prototypes.shape[0] == 0:
        return {"cons_loss_mean": float("nan"), "cons_n_prototypes": 0}
    device = next(neo.parameters()).device
    prototypes = prototypes.to(device)
    proto_labels = proto_labels.to(device)

    losses: list[float] = []
    for _ in range(cons_epochs):
        classifier_optimizer.zero_grad()
        logits = neo.classifier(prototypes)
        loss = F.cross_entropy(logits, proto_labels)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                neo.classifier.parameters(), max_norm=grad_clip,
            )
        classifier_optimizer.step()
        losses.append(float(loss.item()))
    return {
        "cons_loss_mean": float(statistics.fmean(losses)),
        "cons_n_prototypes": int(prototypes.shape[0]),
    }


# ---------- eval ----------


@torch.no_grad()
def evaluate(
    model: nn.Module, x: Tensor, y: Tensor, device: torch.device,
) -> tuple[float, list[float]]:
    model.eval()
    x = x.to(device); y = y.to(device)
    preds = model(x).argmax(dim=-1)
    acc = float((preds == y).float().mean().item())
    per_class: list[float] = []
    for c in range(int(y.max().item()) + 1):
        mask = (y == c)
        if mask.any():
            per_class.append(
                float((preds[mask] == c).float().mean().item())
            )
        else:
            per_class.append(float("nan"))
    return acc, per_class


def _eval_on_classes(
    model: nn.Module, test_data: list[tuple[Tensor, Tensor]],
    classes: list[int], device: torch.device,
) -> float:
    xs = torch.cat([test_data[c][0] for c in classes], dim=0)
    ys = torch.cat([test_data[c][1] for c in classes], dim=0)
    acc, _ = evaluate(model, xs, ys, device)
    return acc


# ---------- main ----------


def _ok(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--task_size", type=int, default=5)
    p.add_argument("--input_dim", type=int, default=32)
    p.add_argument("--feature_dim", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--samples_per_class", type=int, default=400)
    p.add_argument("--cluster_std", type=float, default=0.4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs_per_task", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--lambda_contrast", type=float, default=1.0)
    p.add_argument(
        "--cons_epochs", type=int, default=5,
        help="Classifier-head-only consolidation steps after each task.",
    )
    p.add_argument(
        "--cons_lr", type=float, default=0.05,
        help="Learning rate for the classifier-only consolidation optimizer.",
    )
    p.add_argument("--prototypes_per_class", type=int, default=3)
    p.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    print("=== Phase 5.7.1 — XRay integration synthetic smoke ===\n")
    print(
        f"Setup: num_classes={args.num_classes}, task_size="
        f"{args.task_size}, input_dim={args.input_dim}, "
        f"feature_dim={args.feature_dim}\n"
        f"Training: epochs_per_task={args.epochs_per_task}, "
        f"batch={args.batch_size}, lr={args.lr}, "
        f"λ_contrast={args.lambda_contrast}\n"
        f"Memory: prototypes_per_class="
        f"{args.prototypes_per_class}, cons_epochs={args.cons_epochs}\n"
        f"Device: {device}\n"
    )

    # ----- build models, memory, optimizers -----
    hipp = TinyEncoderClassifier(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        feature_dim=args.feature_dim,
        num_classes=args.num_classes,
    ).to(device)
    neo = TinyEncoderClassifier(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        feature_dim=args.feature_dim,
        num_classes=args.num_classes,
    ).to(device)
    memory = XRayMemory(
        num_classes=args.num_classes,
        feature_dim=args.feature_dim,
        prototypes_per_class=args.prototypes_per_class,
        # Bring the schedules into a range the smoke can actually
        # exercise — defaults assume long training runs.
        sparsity_start_refinements=10,
        sparsity_end_refinements=80,
        temp_start_refinements=10,
        temp_end_refinements=80,
    ).to(device)

    hipp_opt = torch.optim.SGD(hipp.parameters(), lr=args.lr, momentum=0.9)
    neo_opt  = torch.optim.SGD(neo.parameters(),  lr=args.lr, momentum=0.9)
    # Separate optimizer holding ONLY the classifier head's params,
    # used during consolidation so the encoder stays frozen.
    neo_cls_opt = torch.optim.SGD(
        neo.classifier.parameters(), lr=args.cons_lr, momentum=0.9,
    )

    # ----- synthetic data -----
    train, test = _make_synthetic_ci_data(
        num_classes=args.num_classes,
        samples_per_class=args.samples_per_class,
        input_dim=args.input_dim,
        cluster_std=args.cluster_std,
        seed=args.seed,
    )

    n_tasks = args.num_classes // args.task_size
    classes_seen_so_far: list[int] = []
    per_task_log: list[dict[str, Any]] = []

    any_nan = False
    t_start = time.time()
    for task_id in range(n_tasks):
        task_classes = list(range(
            task_id * args.task_size,
            (task_id + 1) * args.task_size,
        ))
        classes_seen_so_far.extend(task_classes)
        loader = _task_loader(train, task_classes, args.batch_size, shuffle=True)

        last_diag: dict[str, float] = {}
        for epoch in range(args.epochs_per_task):
            per_batch: dict[str, list[float]] = {
                "hipp_loss":     [], "ce_loss":       [],
                "contrast_loss": [], "total_loss":    [],
            }
            hipp.train(); neo.train()
            for xb, yb in loader:
                xb = xb.to(device); yb = yb.to(device)
                step = train_step_dual_xray(
                    hipp, neo, memory,
                    hipp_opt, neo_opt,
                    xb, yb,
                    lambda_contrast=args.lambda_contrast,
                )
                for k, v in step.items():
                    per_batch[k].append(v)
                    if not math.isfinite(v):
                        any_nan = True
            last_diag = {
                k: float(statistics.fmean(vs)) if vs else float("nan")
                for k, vs in per_batch.items()
            }

        # Classifier-head-only consolidation on the prototype set.
        cons_diag = consolidate_with_xray(
            neo, memory, neo_cls_opt, cons_epochs=args.cons_epochs,
        )
        if not math.isfinite(cons_diag["cons_loss_mean"]):
            # NaN sentinel allowed when memory empty (cons_n=0);
            # only flag if memory had prototypes.
            if cons_diag["cons_n_prototypes"] > 0:
                any_nan = True

        # ----- eval -----
        hipp_acc_full   = _eval_on_classes(
            hipp, test, classes_seen_so_far, device,
        )
        neo_acc_full    = _eval_on_classes(
            neo,  test, classes_seen_so_far, device,
        )
        # Retention metric: ACC on TASK 0's classes only.
        task0_classes = list(range(args.task_size))
        hipp_acc_task0 = _eval_on_classes(hipp, test, task0_classes, device)
        neo_acc_task0  = _eval_on_classes(neo,  test, task0_classes, device)

        task_record = {
            "task_id": task_id,
            "classes": task_classes,
            "last_epoch_losses": last_diag,
            "cons": cons_diag,
            "memory_size": memory.num_occupied(),
            "per_class_counts": memory.per_class_counts(),
            "mean_refinement": float(
                memory.refinement_counts[memory.is_occupied].float().mean().item()
                if memory.num_occupied() > 0 else 0.0
            ),
            "temperature": memory.temperature(),
            "hipp_acc_full":   hipp_acc_full,
            "neo_acc_full":    neo_acc_full,
            "hipp_acc_task0":  hipp_acc_task0,
            "neo_acc_task0":   neo_acc_task0,
        }
        per_task_log.append(task_record)

        print(
            f"  task {task_id} (classes {task_classes[0]}-{task_classes[-1]}): "
            f"last_epoch CE={last_diag['ce_loss']:.3f} "
            f"contrast={last_diag['contrast_loss']:.3f}  "
            f"|cons| loss={cons_diag['cons_loss_mean']:.3f} "
            f"on {cons_diag['cons_n_prototypes']} protos  "
            f"|mem|={memory.num_occupied()}  "
            f"NEO ACC={neo_acc_full:.3f} (task0={neo_acc_task0:.3f})  "
            f"HIPP ACC={hipp_acc_full:.3f} (task0={hipp_acc_task0:.3f})",
            flush=True,
        )

    wall = time.time() - t_start

    # ----- verdict -----
    final_record = per_task_log[-1]
    neo_task0_retention = final_record["neo_acc_task0"]
    hipp_task0_retention = final_record["hipp_acc_task0"]
    mem_total = final_record["memory_size"]
    mem_min = max(args.num_classes, 5)
    mem_max = args.num_classes * args.prototypes_per_class
    mem_in_range = mem_min <= mem_total <= mem_max

    print()
    print("=== Verdict ===")
    print(f"Wall time: {wall:.1f}s")
    print(
        f"NEO retention on task-0 classes (0-{args.task_size-1}) "
        f"after task {n_tasks-1}: {neo_task0_retention:.3f}"
    )
    print(
        f"HIPP (naive control) retention on task-0 classes after "
        f"task {n_tasks-1}: {hipp_task0_retention:.3f}"
    )
    print(
        f"Final memory size: {mem_total} prototypes "
        f"(per-class counts: {final_record['per_class_counts']})"
    )

    checks = [
        ("No NaN/Inf during training:", not any_nan),
        (
            f"Memory size in [{mem_min}, {mem_max}]:",
            mem_in_range,
        ),
        ("NEO task-0 retention > 0.30:", neo_task0_retention > 0.30),
        (
            "HIPP task-0 retention < 0.30 (naive control forgets):",
            hipp_task0_retention < 0.30,
        ),
    ]
    all_pass = True
    print()
    for label, passed in checks:
        all_pass = all_pass and passed
        print(f"  {label:<55} {_ok(passed)}")

    print()
    if all_pass:
        print("All mechanics PASS — XRay integration ready for Phase 5.7.2.")
        return 0
    print("Some checks failed — investigate before scaling to real benchmarks.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
