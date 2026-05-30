"""Experiment 39 — Phase 5.5.4: CLS Variant C adapted to Split-MNIST CI.

Cross-paradigm port of the CLS dual-system architecture
(Phase 4 Variant C) to class-incremental Split-MNIST. The core
mechanism is unchanged: a small hippocampe with strong learning
rate and a larger slow neocortex, connected by a consolidation
step that anchors hippocampe features at three levels and
distills its soft predictions into the neocortex.

Three adjustments versus the Permuted-MNIST version:

1. **10-class head everywhere.** Both hippocampe and neocortex
   output 10 logits, not one head per task.

2. **classes_seen_at_storage stored per entry.** The hippocampe's
   soft target at storage time was produced by a model that had
   only seen the current task's 2 classes. The 8 non-task classes
   sit at unreliable values (small, noisy, biased), so the
   distillation step needs to KNOW which classes the teacher was
   qualified to predict about.

3. **Masked KL distillation.** The KL between current neo and
   stored soft is computed only over the ``classes_seen_at_storage``
   subset, with the stored soft renormalised within that subset.
   This prevents the stored teacher from "voting against" classes
   it never saw.

Variant C lambdas: anchor_low=1.0, anchor_mid=0.5, anchor_high=0.1,
distill=1.0, cons_epochs=2. cons_epochs=2 is what closed the gap
to DER at T=15 on Permuted-MNIST.

Run from the repo root::

    python experiments/39_split_mnist_ci_cls.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from continual_synapse.evaluation.benchmarks import (  # noqa: E402
    SplitMNISTClassIncremental,
)
from continual_synapse.evaluation.runner import set_seed  # noqa: E402


# ---------- models (matching Phase 4 spec) ----------


class Hippocampus(nn.Module):
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: tuple = (128, 64),
        n_classes: int = 10,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.encoder(x))


class Neocortex(nn.Module):
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: tuple = (256, 256, 128),
        n_classes: int = 10,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev, n_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.encoder(x))


# ---------- memory (CI adaptation: classes_seen + labels) ----------


class CIMultiLevelMemory:
    """Phase 3's MultiLevelMemory + labels + classes_seen_at_storage
    for class-incremental consolidation. Storage policy is random
    sampling, ``samples_per_task`` per task, no eviction (memory
    grows to ``samples_per_task * num_tasks``).
    """

    def __init__(
        self, samples_per_task: int = 100, n_classes: int = 10,
    ) -> None:
        self.samples_per_task = int(samples_per_task)
        self.n_classes = int(n_classes)
        self.inputs: list[Tensor] = []
        self.low_features: list[Tensor] = []
        self.mid_features: list[Tensor] = []
        self.high_features: list[Tensor] = []
        self.soft_targets: list[Tensor] = []
        self.labels: list[int] = []
        self.task_ids: list[int] = []
        # NEW vs Phase 4: per-entry set of class indices the
        # hippocampe had seen by storage time. Stored as a 10-dim
        # boolean tensor for fast batched masking during the
        # consolidation step.
        self.classes_seen_mask: list[Tensor] = []

    @torch.no_grad()
    def record_task_end(
        self,
        hippocampus: Hippocampus,
        task_inputs: Tensor, task_labels: Tensor,
        task_id: int, device: torch.device,
        classes_seen: set[int],
        generator: torch.Generator | None = None,
    ) -> int:
        n_pool = len(task_inputs)
        n = min(self.samples_per_task, n_pool)
        if generator is None:
            idx = torch.randperm(n_pool)[:n]
        else:
            idx = torch.randperm(n_pool, generator=generator)[:n]
        sampled = task_inputs[idx].to(device)
        sampled_labels = task_labels[idx]

        h = sampled
        layer_outputs: list[Tensor] = []
        for layer in hippocampus.encoder:
            h = layer(h)
            if isinstance(layer, nn.ReLU):
                layer_outputs.append(h.detach().cpu())
        if len(layer_outputs) < 2:
            raise RuntimeError(
                "Hippocampe encoder has fewer than 2 ReLU layers; "
                "architecture has drifted."
            )

        soft = F.softmax(
            hippocampus.classifier(h), dim=-1,
        ).detach().cpu()
        mask = torch.zeros(self.n_classes, dtype=torch.bool)
        for c in classes_seen:
            mask[c] = True

        for i in range(n):
            self.inputs.append(sampled[i].detach().cpu())
            self.low_features.append(layer_outputs[0][i])
            self.mid_features.append(layer_outputs[1][i])
            self.high_features.append(h[i].detach().cpu())
            self.soft_targets.append(soft[i])
            self.labels.append(int(sampled_labels[i].item()))
            self.task_ids.append(int(task_id))
            self.classes_seen_mask.append(mask.clone())
        return n

    def __len__(self) -> int:
        return len(self.inputs)

    def per_task_counts(self) -> dict[int, int]:
        return dict(Counter(self.task_ids))


# ---------- masked KL distillation ----------


def masked_kl(
    neo_logits: Tensor,
    stored_soft: Tensor,
    classes_seen_mask: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """KL divergence computed only on classes the hippocampe had
    seen at storage time. The stored soft target is renormalised
    over the seen-class subset, then KL(masked_teacher || masked_neo)
    is computed via batchmean.

    Args:
        neo_logits: (B, C) raw logits from the current neocortex.
        stored_soft: (B, C) hippocampe's softmax at storage time.
        classes_seen_mask: (B, C) bool — True where the teacher
            had seen that class.
        eps: small constant for normalisation stability.
    """
    # Mask + renormalise the teacher.
    mask_f = classes_seen_mask.to(stored_soft.dtype)
    masked = stored_soft * mask_f
    masked = masked / (masked.sum(dim=-1, keepdim=True) + eps)

    # Neo log-probs over the full 10-class space, but the loss
    # only reads the masked entries. We do this with a flat
    # batchmean over the masked indices to keep gradient scale
    # consistent across samples that have different numbers of
    # seen classes (in this paradigm every sample has the same
    # 2 classes seen, so this is mostly defensive).
    log_neo = F.log_softmax(neo_logits, dim=-1)
    # KL(teacher || student) elementwise, masked to seen classes,
    # summed per-sample, then averaged across the batch.
    elem = masked * (masked.clamp(min=eps).log() - log_neo)
    elem = elem * mask_f  # zero out unseen entries safely
    return (elem.sum(dim=-1)).mean()


# ---------- consolidation ----------


def consolidate(
    hippocampus: Hippocampus, neocortex: Neocortex,
    memory: CIMultiLevelMemory,
    hipp_optimizer: torch.optim.Optimizer,
    neo_optimizer: torch.optim.Optimizer,
    *,
    batch_size: int, n_epochs: int,
    lambda_distill: float, lambda_anchor_low: float,
    lambda_anchor_mid: float, lambda_anchor_high: float,
    device: torch.device,
) -> dict[str, float]:
    n = len(memory)
    indices = torch.randperm(n)
    metrics: dict[str, list[float]] = {
        "task_losses":       [],
        "distill_losses":    [],
        "anchor_low_losses": [],
        "anchor_mid_losses": [],
        "anchor_high_losses": [],
        "drift_low_corr":    [],
        "drift_mid_corr":    [],
        "drift_high_corr":   [],
    }
    hippocampus.train()
    neocortex.train()

    for _ in range(n_epochs):
        for i in range(0, n, batch_size):
            batch_idx = indices[i : i + batch_size]
            x = torch.stack(
                [memory.inputs[int(j)] for j in batch_idx]
            ).to(device)
            stored_low = torch.stack(
                [memory.low_features[int(j)] for j in batch_idx]
            ).to(device)
            stored_mid = torch.stack(
                [memory.mid_features[int(j)] for j in batch_idx]
            ).to(device)
            stored_high = torch.stack(
                [memory.high_features[int(j)] for j in batch_idx]
            ).to(device)
            stored_soft = torch.stack(
                [memory.soft_targets[int(j)] for j in batch_idx]
            ).to(device)
            classes_seen = torch.stack(
                [memory.classes_seen_mask[int(j)] for j in batch_idx]
            ).to(device)
            y = torch.tensor(
                [memory.labels[int(j)] for j in batch_idx],
                dtype=torch.long, device=device,
            )

            # ----- Hippocampe: anchor loss -----
            hipp_optimizer.zero_grad()
            h = x
            new_outputs: list[Tensor] = []
            for layer in hippocampus.encoder:
                h = layer(h)
                if isinstance(layer, nn.ReLU):
                    new_outputs.append(h)
            new_low, new_mid = new_outputs[0], new_outputs[1]
            new_high = h
            anchor_low  = F.mse_loss(new_low,  stored_low)
            anchor_mid  = F.mse_loss(new_mid,  stored_mid)
            anchor_high = F.mse_loss(new_high, stored_high)
            anchor_total = (
                lambda_anchor_low  * anchor_low
                + lambda_anchor_mid  * anchor_mid
                + lambda_anchor_high * anchor_high
            )
            anchor_total.backward()
            hipp_optimizer.step()

            with torch.no_grad():
                metrics["drift_low_corr"].append(float(F.cosine_similarity(
                    new_low.detach(),  stored_low,  dim=-1).mean().item()))
                metrics["drift_mid_corr"].append(float(F.cosine_similarity(
                    new_mid.detach(),  stored_mid,  dim=-1).mean().item()))
                metrics["drift_high_corr"].append(float(F.cosine_similarity(
                    new_high.detach(), stored_high, dim=-1).mean().item()))

            # ----- Neocortex: task + masked distill -----
            neo_optimizer.zero_grad()
            neo_logits = neocortex(x)
            task_loss = F.cross_entropy(neo_logits, y)
            distill = masked_kl(neo_logits, stored_soft, classes_seen)
            (task_loss + lambda_distill * distill).backward()
            neo_optimizer.step()

            metrics["task_losses"].append(float(task_loss.item()))
            metrics["distill_losses"].append(float(distill.item()))
            metrics["anchor_low_losses"].append(float(anchor_low.item()))
            metrics["anchor_mid_losses"].append(float(anchor_mid.item()))
            metrics["anchor_high_losses"].append(float(anchor_high.item()))

    return {
        k: float(statistics.fmean(v)) if v else float("nan")
        for k, v in metrics.items()
    }


# ---------- training helpers ----------


def _train_one_task(
    model: nn.Module, optimizer: torch.optim.Optimizer,
    loader: DataLoader, epochs: int, device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    for _ in range(epochs):
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
    return statistics.fmean(losses) if losses else float("nan")


def _eval_class_incremental(
    model: nn.Module, test_ds: TensorDataset, device: torch.device,
    n_classes: int = 10,
) -> dict[str, Any]:
    model.eval()
    x = test_ds.tensors[0].to(device)
    y = test_ds.tensors[1].to(device)
    with torch.no_grad():
        preds = model(x).argmax(dim=-1)
    overall_acc = float((preds == y).float().mean().item())
    per_class: list[float] = []
    for c in range(n_classes):
        mask = (y == c)
        per_class.append(
            float((preds[mask] == y[mask]).float().mean().item())
            if mask.any() else float("nan")
        )
    return {"acc": overall_acc, "per_class": per_class}


# ---------- per-seed driver ----------


def _run_one_seed(
    bench: SplitMNISTClassIncremental,
    args: argparse.Namespace, seed: int,
) -> dict[str, Any]:
    set_seed(seed)
    device = torch.device(args.device)
    hipp = Hippocampus(
        input_dim=args.input_dim,
        hidden_dims=tuple(args.hipp_hidden_dims),
        n_classes=args.n_classes,
    ).to(device)
    neo  = Neocortex(
        input_dim=args.input_dim,
        hidden_dims=tuple(args.neo_hidden_dims),
        n_classes=args.n_classes,
    ).to(device)
    hipp_optimizer = torch.optim.SGD(
        hipp.parameters(), lr=args.hipp_lr, momentum=args.momentum,
    )
    neo_optimizer = torch.optim.SGD(
        neo.parameters(),  lr=args.neo_lr,  momentum=args.momentum,
    )
    memory = CIMultiLevelMemory(
        samples_per_task=args.samples_per_task,
        n_classes=args.n_classes,
    )
    classes_seen: set[int] = set()

    consolidation_diag: list[dict[str, Any]] = []
    per_task_full_acc_neo:  list[float] = []
    per_task_full_acc_hipp: list[float] = []
    full_test = bench.all_test_dataset()
    t_start = time.time()

    for task_idx, task in enumerate(bench.tasks()):
        # Update the "seen so far" set BEFORE storage so the
        # masked-KL teacher knows the current task's classes were
        # in scope.
        for c in task.classes:
            classes_seen.add(int(c))

        loader = DataLoader(
            task.train, batch_size=args.batch_size, shuffle=True,
        )
        hipp_loss = _train_one_task(
            hipp, hipp_optimizer, loader,
            epochs=args.epochs_per_task, device=device,
        )
        neo_loss = _train_one_task(
            neo,  neo_optimizer,  loader,
            epochs=args.epochs_per_task, device=device,
        )

        gen = torch.Generator()
        gen.manual_seed(int(seed * 1009 + task_idx))
        n_stored = memory.record_task_end(
            hipp,
            task_inputs=task.train.tensors[0],
            task_labels=task.train.tensors[1],
            task_id=task_idx, device=device,
            classes_seen=set(classes_seen),
            generator=gen,
        )

        cons = consolidate(
            hipp, neo, memory,
            hipp_optimizer, neo_optimizer,
            batch_size=args.cons_batch_size,
            n_epochs=args.cons_epochs,
            lambda_distill=args.lambda_distill,
            lambda_anchor_low=args.lambda_anchor_low,
            lambda_anchor_mid=args.lambda_anchor_mid,
            lambda_anchor_high=args.lambda_anchor_high,
            device=device,
        )
        cons["task_idx"] = int(task_idx)
        cons["memory_size_after"] = int(len(memory))
        consolidation_diag.append(cons)

        per_task_full_acc_neo.append(
            _eval_class_incremental(
                neo, full_test, device, n_classes=args.n_classes,
            )["acc"]
        )
        per_task_full_acc_hipp.append(
            _eval_class_incremental(
                hipp, full_test, device, n_classes=args.n_classes,
            )["acc"]
        )
        print(
            f"  seed={seed}  task={task_idx} ({task.classes})  "
            f"hipp_loss={hipp_loss:.3f}  neo_loss={neo_loss:.3f}  "
            f"|cons| task={cons['task_losses']:.3f} "
            f"distill={cons['distill_losses']:.3f} "
            f"drift_low={cons['drift_low_corr']:.3f}  "
            f"|mem|={len(memory)}  "
            f"NEO 10-class ACC={per_task_full_acc_neo[-1]:.3f}  "
            f"HIPP 10-class ACC={per_task_full_acc_hipp[-1]:.3f}",
            flush=True,
        )

    final_neo  = _eval_class_incremental(
        neo,  full_test, device, n_classes=args.n_classes,
    )
    final_hipp = _eval_class_incremental(
        hipp, full_test, device, n_classes=args.n_classes,
    )
    return {
        "seed": int(seed),
        "consolidation_diagnostics": consolidation_diag,
        "per_task_full_acc_neo":  per_task_full_acc_neo,
        "per_task_full_acc_hipp": per_task_full_acc_hipp,
        "neo_final_acc":  final_neo["acc"],
        "hipp_final_acc": final_hipp["acc"],
        "neo_per_class_final":  final_neo["per_class"],
        "hipp_per_class_final": final_hipp["per_class"],
        # FGT proxy = ACC after task 0 minus ACC after task N.
        "fgt_proxy": float(
            per_task_full_acc_neo[0] - per_task_full_acc_neo[-1]
        ),
        "wall_time_s": float(time.time() - t_start),
    }


# ---------- main ----------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--seed-base", type=int, default=0)

    p.add_argument(
        "--epochs-per-task", "--epochs_per_task",
        dest="epochs_per_task", type=int, default=1,
    )
    p.add_argument("--batch-size", type=int, default=64)

    p.add_argument(
        "--hipp-hidden-dims", type=int, nargs="+", default=[128, 64],
    )
    p.add_argument("--hipp-lr", type=float, default=0.05)
    p.add_argument(
        "--neo-hidden-dims", type=int, nargs="+", default=[256, 256, 128],
    )
    p.add_argument("--neo-lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--input-dim", type=int, default=784)
    p.add_argument("--n-classes", type=int, default=10)
    p.add_argument("--samples-per-task", type=int, default=100)

    p.add_argument("--cons-batch-size", type=int, default=64)
    # Variant C: cons_epochs = 2.
    p.add_argument(
        "--cons-epochs", "--cons_epochs",
        dest="cons_epochs", type=int, default=2,
    )
    p.add_argument(
        "--lambda-distill", "--lambda_distill",
        dest="lambda_distill", type=float, default=1.0,
    )
    p.add_argument(
        "--lambda-anchor-low", "--lambda_anchor_low",
        dest="lambda_anchor_low", type=float, default=1.0,
    )
    p.add_argument(
        "--lambda-anchor-mid", "--lambda_anchor_mid",
        dest="lambda_anchor_mid", type=float, default=0.5,
    )
    p.add_argument(
        "--lambda-anchor-high", "--lambda_anchor_high",
        dest="lambda_anchor_high", type=float, default=0.1,
    )

    p.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=_REPO_ROOT / "results" / "logs" / "split_mnist_ci",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.hipp_hidden_dims = [int(h) for h in args.hipp_hidden_dims]
    args.neo_hidden_dims  = [int(h) for h in args.neo_hidden_dims]

    print(
        f"Phase 5.5.4 — CLS Variant C on Split-MNIST class-incremental\n"
        f"  n_seeds={args.n_seeds}  epochs_per_task={args.epochs_per_task}\n"
        f"  hipp: dims={tuple(args.hipp_hidden_dims)} lr={args.hipp_lr}\n"
        f"  neo:  dims={tuple(args.neo_hidden_dims)} lr={args.neo_lr}\n"
        f"  consolidation: batch={args.cons_batch_size} "
        f"n_epochs={args.cons_epochs}  "
        f"distill={args.lambda_distill}  "
        f"anchor=[{args.lambda_anchor_low}, "
        f"{args.lambda_anchor_mid}, {args.lambda_anchor_high}]",
        flush=True,
    )

    bench = SplitMNISTClassIncremental.from_huggingface()
    per_seed: list[dict[str, Any]] = []
    for seed in range(args.seed_base, args.seed_base + args.n_seeds):
        print(f"\n--- seed {seed} ---", flush=True)
        per_seed.append(_run_one_seed(bench, args, seed=seed))

    neo_accs  = [s["neo_final_acc"]  for s in per_seed]
    hipp_accs = [s["hipp_final_acc"] for s in per_seed]
    fgts      = [s["fgt_proxy"]      for s in per_seed]

    per_class_means_neo: list[float] = []
    per_class_means_hipp: list[float] = []
    for c in range(args.n_classes):
        vals_n = [
            s["neo_per_class_final"][c] for s in per_seed
            if not (s["neo_per_class_final"][c] != s["neo_per_class_final"][c])
        ]
        vals_h = [
            s["hipp_per_class_final"][c] for s in per_seed
            if not (s["hipp_per_class_final"][c] != s["hipp_per_class_final"][c])
        ]
        per_class_means_neo.append(
            statistics.fmean(vals_n) if vals_n else float("nan")
        )
        per_class_means_hipp.append(
            statistics.fmean(vals_h) if vals_h else float("nan")
        )

    print()
    print(f"=== CLS Variant C (n={len(per_seed)}) ===")
    print(
        f"  NEO  Final ACC: mean={statistics.fmean(neo_accs):.3f}  "
        f"std={statistics.stdev(neo_accs) if len(neo_accs)>1 else 0:.3f}"
    )
    print(
        f"  HIPP Final ACC: mean={statistics.fmean(hipp_accs):.3f}  "
        f"std={statistics.stdev(hipp_accs) if len(hipp_accs)>1 else 0:.3f}"
    )
    print(
        f"  FGT:            mean={statistics.fmean(fgts):.3f}"
    )
    print(
        "  NEO  per-class final acc: [" +
        ", ".join(f"{c}:{per_class_means_neo[c]:.2f}" for c in range(args.n_classes))
        + "]"
    )
    print(
        "  HIPP per-class final acc: [" +
        ", ".join(f"{c}:{per_class_means_hipp[c]:.2f}" for c in range(args.n_classes))
        + "]"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_path = args.output_dir / f"{ts}_39_split_mnist_ci_cls.json"
    with out_path.open("w") as f:
        json.dump({
            "experiment": "39_split_mnist_ci_cls",
            "method": "cls_variant_c",
            "phase": "5.5.4",
            "timestamp": ts,
            "config": {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in vars(args).items()
            },
            "per_seed": per_seed,
            "summary": {
                "neo_final_acc_mean":  statistics.fmean(neo_accs),
                "neo_final_acc_std":   (
                    statistics.stdev(neo_accs) if len(neo_accs) > 1 else 0.0
                ),
                "hipp_final_acc_mean": statistics.fmean(hipp_accs),
                "fgt_mean":            statistics.fmean(fgts),
                "per_class_means_neo":  per_class_means_neo,
                "per_class_means_hipp": per_class_means_hipp,
            },
        }, f, indent=2)
    print(f"\nWrote results JSON to {out_path}")


if __name__ == "__main__":
    main()
