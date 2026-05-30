"""Experiment 36 — Phase 5.5.1 of Phase 5.5: naive baseline on Split-MNIST CI.

First piece of the cross-paradigm validation: train a vanilla MLP
naively across the 5 binary tasks of class-incremental Split-MNIST,
without any continual-learning machinery. Expected to produce
catastrophic forgetting (final ACC near chance ~ 0.20).

Setup:
- Benchmark: SplitMNISTClassIncremental ((0,1)→(2,3)→...→(8,9))
- Model: same Neocortex MLP we use elsewhere (256, 256, 128) → 10
- Optimizer: SGD lr=0.01 mom=0.9, batch 64, 1 epoch/task
- Eval: after all 5 tasks, predict over all 10 classes on the
  full Split-MNIST test set (no task ID), report ACC + per-class
  accuracy + FGT.

Run from the repo root::

    python experiments/36_split_mnist_ci_naive.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from continual_synapse.evaluation.benchmarks import (  # noqa: E402
    SplitMNISTClassIncremental,
)
from continual_synapse.evaluation.runner import set_seed  # noqa: E402


# ---------- model (same Neocortex shape used in Phase 2 & 4) ----------


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))


# ---------- training / eval ----------


def _train_one_task(
    model: Neocortex, optimizer: torch.optim.Optimizer,
    loader: DataLoader, epochs: int, device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    for _ in range(epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
    return statistics.fmean(losses) if losses else float("nan")


def _eval_class_incremental(
    model: Neocortex, test_ds: TensorDataset, device: torch.device,
    n_classes: int = 10,
) -> dict[str, Any]:
    """Predict over the full ``n_classes`` label space on the union
    of all test samples. Return aggregate ACC + per-class accuracy."""
    model.eval()
    x = test_ds.tensors[0].to(device)
    y = test_ds.tensors[1].to(device)
    with torch.no_grad():
        preds = model(x).argmax(dim=-1)
    overall_acc = float((preds == y).float().mean().item())
    per_class: list[float] = []
    for c in range(n_classes):
        mask = (y == c)
        if mask.any():
            per_class.append(
                float((preds[mask] == y[mask]).float().mean().item())
            )
        else:
            per_class.append(float("nan"))
    return {"acc": overall_acc, "per_class": per_class}


def _run_one_seed(
    bench: SplitMNISTClassIncremental,
    args: argparse.Namespace, seed: int,
) -> dict[str, Any]:
    set_seed(seed)
    device = torch.device(args.device)
    model = Neocortex(
        input_dim=args.input_dim,
        hidden_dims=tuple(args.hidden_dims),
        n_classes=args.n_classes,
    ).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
    )

    per_task_train_loss: list[float] = []
    per_task_eval_acc: list[float] = []
    full_test = bench.all_test_dataset()
    t0 = time.time()

    for task_idx, task in enumerate(bench.tasks()):
        loader = DataLoader(
            task.train, batch_size=args.batch_size, shuffle=True,
        )
        avg_loss = _train_one_task(
            model, optimizer, loader,
            epochs=args.epochs_per_task, device=device,
        )
        per_task_train_loss.append(avg_loss)

        # Track 10-class eval after each task so we can show the
        # forgetting curve, not just the final number.
        per_task_eval_acc.append(
            _eval_class_incremental(
                model, full_test, device,
                n_classes=args.n_classes,
            )["acc"]
        )
        print(
            f"  seed={seed}  task={task_idx} ({task.classes})  "
            f"train_loss={avg_loss:.3f}  "
            f"full-10-class ACC={per_task_eval_acc[-1]:.3f}",
            flush=True,
        )

    final = _eval_class_incremental(
        model, full_test, device, n_classes=args.n_classes,
    )
    # Forgetting proxy: how much aggregate ACC fell off after
    # learning the last task vs after the first. For naive this
    # should be a big drop (catastrophic).
    fgt = per_task_eval_acc[0] - per_task_eval_acc[-1]
    return {
        "seed": int(seed),
        "per_task_train_loss": per_task_train_loss,
        "per_task_full_acc":   per_task_eval_acc,
        "final_acc":           final["acc"],
        "per_class_final":     final["per_class"],
        "fgt_proxy":           float(fgt),
        "wall_time_s":         float(time.time() - t0),
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
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument(
        "--hidden-dims", type=int, nargs="+", default=[256, 256, 128],
    )
    p.add_argument("--input-dim", type=int, default=784)
    p.add_argument("--n-classes", type=int, default=10)
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
    args.hidden_dims = [int(h) for h in args.hidden_dims]

    print(
        f"Phase 5.5.1 — naive baseline on Split-MNIST class-incremental\n"
        f"  n_seeds={args.n_seeds}  epochs_per_task={args.epochs_per_task}\n"
        f"  SGD lr={args.lr} mom={args.momentum} batch={args.batch_size}\n"
        f"  model: {tuple(args.hidden_dims)} -> {args.n_classes}",
        flush=True,
    )

    bench = SplitMNISTClassIncremental.from_huggingface()

    per_seed: list[dict[str, Any]] = []
    for seed in range(args.seed_base, args.seed_base + args.n_seeds):
        print(f"\n--- seed {seed} ---", flush=True)
        per_seed.append(_run_one_seed(bench, args, seed=seed))

    final_accs = [s["final_acc"] for s in per_seed]
    fgts       = [s["fgt_proxy"] for s in per_seed]
    print()
    print(f"=== Naive (n={len(per_seed)}) ===")
    print(
        f"  Final ACC: mean={statistics.fmean(final_accs):.3f}  "
        f"std={statistics.stdev(final_accs) if len(final_accs)>1 else 0:.3f}"
    )
    print(
        f"  FGT:       mean={statistics.fmean(fgts):.3f}  "
        f"std={statistics.stdev(fgts) if len(fgts)>1 else 0:.3f}"
    )

    # Per-class final accuracy averaged across seeds.
    per_class_means: list[float] = []
    for c in range(args.n_classes):
        vals = [
            s["per_class_final"][c] for s in per_seed
            if not (s["per_class_final"][c] != s["per_class_final"][c])
        ]
        per_class_means.append(
            statistics.fmean(vals) if vals else float("nan")
        )
    print(
        "  Per-class final acc: [" +
        ", ".join(f"{c}:{per_class_means[c]:.2f}" for c in range(args.n_classes))
        + "]"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_path = (
        args.output_dir / f"{ts}_36_split_mnist_ci_naive.json"
    )
    with out_path.open("w") as f:
        json.dump({
            "experiment": "36_split_mnist_ci_naive",
            "method": "naive",
            "phase": "5.5.1",
            "timestamp": ts,
            "config": {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in vars(args).items()
            },
            "per_seed": per_seed,
            "summary": {
                "final_acc_mean": statistics.fmean(final_accs),
                "final_acc_std":  (
                    statistics.stdev(final_accs) if len(final_accs) > 1 else 0.0
                ),
                "fgt_mean":       statistics.fmean(fgts),
                "per_class_means": per_class_means,
            },
        }, f, indent=2)
    print(f"\nWrote results JSON to {out_path}")


if __name__ == "__main__":
    main()
