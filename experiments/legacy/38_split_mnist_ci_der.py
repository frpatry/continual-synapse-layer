"""Experiment 38 — Phase 5.5.3: DER baseline on Split-MNIST CI.

Pure DER (Dark Experience Replay, Buzzega et al., 2020) adapted
for class-incremental Split-MNIST. Same Neocortex MLP as exp 36
/ 37. The only addition is a functional-memory replay channel:
after each task we snapshot 100 (input, soft_target) pairs into
a reservoir-style buffer (max_total = 500 across all 5 tasks);
during training of each subsequent task we sample a batch from
the buffer and add a distillation loss against the stored soft
targets.

This is the project's existing ``cs_functional_only`` recipe
ported to the class-incremental setup, and serves as the
"memory-based" baseline against which Phase 5.5.4's CLS Variant C
is compared.

Run from the repo root::

    python experiments/38_split_mnist_ci_der.py
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
from continual_synapse.functional import (  # noqa: E402
    FunctionalMemory, distillation_loss,
)


# ---------- model (matches exp 36 / 37) ----------


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
    memory: FunctionalMemory, loader: DataLoader,
    epochs: int, device: torch.device,
    lambda_distill: float, temperature: float, reg_batch_size: int,
) -> tuple[float, float]:
    """Standard cross-entropy + replay distillation. Returns the
    mean (task_loss, distill_loss) over batches in this task."""
    model.train()
    task_losses: list[float] = []
    distill_losses: list[float] = []
    for _ in range(epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            task_loss = F.cross_entropy(logits, y)
            distill = torch.zeros((), device=device)
            replay = memory.sample_batch(reg_batch_size, device=device)
            if replay is not None:
                rx, rs = replay
                distill = distillation_loss(
                    model(rx), rs, temperature=temperature,
                )
            (task_loss + lambda_distill * distill).backward()
            optimizer.step()
            task_losses.append(float(task_loss.item()))
            distill_losses.append(float(distill.item()))
    return (
        statistics.fmean(task_losses) if task_losses else float("nan"),
        statistics.fmean(distill_losses) if distill_losses else float("nan"),
    )


def _eval_class_incremental(
    model: Neocortex, test_ds: TensorDataset, device: torch.device,
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
    memory = FunctionalMemory(
        samples_per_task=args.samples_per_task,
        max_total=args.max_memory,
        rng_seed=seed,
    )

    per_task_eval_acc: list[float] = []
    per_task_losses: list[dict[str, float]] = []
    full_test = bench.all_test_dataset()
    t0 = time.time()

    for task_idx, task in enumerate(bench.tasks()):
        loader = DataLoader(
            task.train, batch_size=args.batch_size, shuffle=True,
        )
        task_loss, distill_loss_v = _train_one_task(
            model, optimizer, memory, loader,
            epochs=args.epochs_per_task, device=device,
            lambda_distill=args.lambda_distill,
            temperature=args.temperature,
            reg_batch_size=args.reg_batch_size,
        )
        per_task_losses.append({
            "task_loss": task_loss, "distill_loss": distill_loss_v,
        })

        # End-of-task snapshot: store inputs + the model's soft
        # predictions on them. memory.record_task_end calls the
        # supplied forward, so we just hand it model itself.
        n_stored = memory.record_task_end(
            model_forward=lambda x: model(x),
            task_inputs=task.train.tensors[0],
            task_id=task_idx,
            device=device,
        )

        per_task_eval_acc.append(
            _eval_class_incremental(
                model, full_test, device,
                n_classes=args.n_classes,
            )["acc"]
        )
        print(
            f"  seed={seed}  task={task_idx} ({task.classes})  "
            f"task_loss={task_loss:.3f}  "
            f"distill={distill_loss_v:.3f}  "
            f"|mem|={len(memory)} (+{n_stored})  "
            f"full-10-class ACC={per_task_eval_acc[-1]:.3f}",
            flush=True,
        )

    final = _eval_class_incremental(
        model, full_test, device, n_classes=args.n_classes,
    )
    fgt = per_task_eval_acc[0] - per_task_eval_acc[-1]
    return {
        "seed": int(seed),
        "per_task_losses": per_task_losses,
        "per_task_full_acc": per_task_eval_acc,
        "final_acc": final["acc"],
        "per_class_final": final["per_class"],
        "fgt_proxy": float(fgt),
        "final_memory_size": int(len(memory)),
        "wall_time_s": float(time.time() - t0),
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
    # DER knobs — match cs_functional_only defaults from exp 30.
    p.add_argument("--samples-per-task", type=int, default=100)
    p.add_argument(
        "--max-memory", type=int, default=500,
        help="Reservoir cap on buffer size (100 per task × 5 tasks "
             "= 500 by default, matching the spec).",
    )
    p.add_argument("--lambda-distill", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument(
        "--reg-batch-size", type=int, default=64,
        help="How many memory entries to draw per training batch "
             "for the distillation replay channel.",
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
    args.hidden_dims = [int(h) for h in args.hidden_dims]

    print(
        f"Phase 5.5.3 — DER on Split-MNIST class-incremental\n"
        f"  n_seeds={args.n_seeds}  epochs_per_task={args.epochs_per_task}\n"
        f"  SGD lr={args.lr} mom={args.momentum} batch={args.batch_size}\n"
        f"  memory: samples_per_task={args.samples_per_task} "
        f"max={args.max_memory}  λ_distill={args.lambda_distill} "
        f"T={args.temperature}",
        flush=True,
    )

    bench = SplitMNISTClassIncremental.from_huggingface()

    per_seed: list[dict[str, Any]] = []
    for seed in range(args.seed_base, args.seed_base + args.n_seeds):
        print(f"\n--- seed {seed} ---", flush=True)
        per_seed.append(_run_one_seed(bench, args, seed=seed))

    final_accs = [s["final_acc"] for s in per_seed]
    fgts       = [s["fgt_proxy"] for s in per_seed]
    per_class_means: list[float] = []
    for c in range(args.n_classes):
        vals = [
            s["per_class_final"][c] for s in per_seed
            if not (s["per_class_final"][c] != s["per_class_final"][c])
        ]
        per_class_means.append(
            statistics.fmean(vals) if vals else float("nan")
        )

    print()
    print(f"=== DER (n={len(per_seed)}) ===")
    print(
        f"  Final ACC: mean={statistics.fmean(final_accs):.3f}  "
        f"std={statistics.stdev(final_accs) if len(final_accs)>1 else 0:.3f}"
    )
    print(
        f"  FGT:       mean={statistics.fmean(fgts):.3f}  "
        f"std={statistics.stdev(fgts) if len(fgts)>1 else 0:.3f}"
    )
    print(
        "  Per-class final acc: [" +
        ", ".join(f"{c}:{per_class_means[c]:.2f}" for c in range(args.n_classes))
        + "]"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_path = args.output_dir / f"{ts}_38_split_mnist_ci_der.json"
    with out_path.open("w") as f:
        json.dump({
            "experiment": "38_split_mnist_ci_der",
            "method": "der",
            "phase": "5.5.3",
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
