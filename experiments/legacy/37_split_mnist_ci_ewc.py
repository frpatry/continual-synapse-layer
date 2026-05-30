"""Experiment 37 — Phase 5.5.2: EWC baseline on Split-MNIST CI.

Elastic Weight Consolidation (Kirkpatrick et al., 2017) on the
class-incremental Split-MNIST setup. Same Neocortex MLP and
optimizer as the naive baseline (exp 36) — the only addition
is the EWC quadratic penalty applied during training and a
Fisher diagonal estimated at the end of each task.

Spec sweeps λ ∈ {10, 100, 1000} at n=2 to pick the best, then
reports that λ at n=3 alongside per-class accuracies.

Run from the repo root::

    python experiments/37_split_mnist_ci_ewc.py
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

from continual_synapse.baselines.ewc import EWC  # noqa: E402
from continual_synapse.evaluation.benchmarks import (  # noqa: E402
    SplitMNISTClassIncremental,
)
from continual_synapse.evaluation.runner import set_seed  # noqa: E402


# ---------- model (matches exp 36 for apples-to-apples) ----------


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
    ewc: EWC, loader: DataLoader, epochs: int,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    for _ in range(epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y) + ewc.penalty(model)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
    return statistics.fmean(losses) if losses else float("nan")


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
    args: argparse.Namespace, seed: int, lam: float,
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
    ewc = EWC(
        lam=lam,
        fisher_sample_size=args.fisher_sample_size,
        device=str(device),
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
            model, optimizer, ewc, loader,
            epochs=args.epochs_per_task, device=device,
        )
        per_task_train_loss.append(avg_loss)

        # Consolidate Fisher for this task before moving to the
        # next one. The EWC penalty in the next task's training
        # loop will then include this task's quadratic term.
        ewc.consolidate(model, task.train)

        per_task_eval_acc.append(
            _eval_class_incremental(
                model, full_test, device,
                n_classes=args.n_classes,
            )["acc"]
        )
        print(
            f"  seed={seed} lam={lam}  task={task_idx} ({task.classes})  "
            f"train_loss={avg_loss:.3f}  "
            f"full-10-class ACC={per_task_eval_acc[-1]:.3f}",
            flush=True,
        )

    final = _eval_class_incremental(
        model, full_test, device, n_classes=args.n_classes,
    )
    fgt = per_task_eval_acc[0] - per_task_eval_acc[-1]
    return {
        "seed": int(seed),
        "lam": float(lam),
        "per_task_train_loss": per_task_train_loss,
        "per_task_full_acc": per_task_eval_acc,
        "final_acc": final["acc"],
        "per_class_final": final["per_class"],
        "fgt_proxy": float(fgt),
        "wall_time_s": float(time.time() - t0),
    }


def _summarise(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    accs = [s["final_acc"] for s in per_seed]
    fgts = [s["fgt_proxy"] for s in per_seed]
    n_classes = len(per_seed[0]["per_class_final"])
    per_class_means: list[float] = []
    for c in range(n_classes):
        vals = [
            s["per_class_final"][c] for s in per_seed
            if not (s["per_class_final"][c] != s["per_class_final"][c])
        ]
        per_class_means.append(
            statistics.fmean(vals) if vals else float("nan")
        )
    return {
        "final_acc_mean": float(statistics.fmean(accs)),
        "final_acc_std":  float(
            statistics.stdev(accs) if len(accs) > 1 else 0.0
        ),
        "fgt_mean":        float(statistics.fmean(fgts)),
        "per_class_means": per_class_means,
    }


# ---------- main ----------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument(
        "--lambdas", type=float, nargs="+",
        default=[10.0, 100.0, 1000.0],
        help="EWC λ values to sweep; best one is then re-run at n_seeds.",
    )
    p.add_argument(
        "--sweep-seeds", type=int, default=2,
        help="Number of seeds used during the λ sweep; usually small.",
    )
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
        "--fisher-sample-size", type=int, default=500,
        help="Cap on samples used per task for Fisher estimation. "
             "EWC's Fisher loop is sample-by-sample (batch=1) so "
             "this caps the cost; 500 is plenty for MNIST.",
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
    args.lambdas = [float(x) for x in args.lambdas]

    print(
        f"Phase 5.5.2 — EWC on Split-MNIST class-incremental\n"
        f"  lambda sweep {args.lambdas} @ n={args.sweep_seeds}, "
        f"then best λ re-run @ n={args.n_seeds}\n"
        f"  epochs_per_task={args.epochs_per_task}  lr={args.lr}",
        flush=True,
    )

    bench = SplitMNISTClassIncremental.from_huggingface()

    # ----- λ sweep at small n -----
    sweep_results: dict[float, list[dict[str, Any]]] = {
        lam: [] for lam in args.lambdas
    }
    for lam in args.lambdas:
        print(f"\n=== Sweep — λ = {lam} ===", flush=True)
        for seed in range(args.seed_base, args.seed_base + args.sweep_seeds):
            sweep_results[lam].append(
                _run_one_seed(bench, args, seed=seed, lam=lam)
            )

    print("\n=== Sweep summary (per λ) ===")
    sweep_summary: dict[float, dict[str, Any]] = {}
    for lam, rs in sweep_results.items():
        s = _summarise(rs)
        sweep_summary[lam] = s
        print(
            f"  λ={lam:>6.1f}: ACC={s['final_acc_mean']:.3f} ± "
            f"{s['final_acc_std']:.3f}  FGT={s['fgt_mean']:.3f}"
        )
    best_lam = max(args.lambdas, key=lambda l: sweep_summary[l]["final_acc_mean"])
    print(f"\n  Best λ = {best_lam} "
          f"(ACC={sweep_summary[best_lam]['final_acc_mean']:.3f})")

    # ----- Re-run best λ at n=n_seeds (full set, including sweep seeds) -----
    print(f"\n=== Re-run @ λ={best_lam}, n={args.n_seeds} ===", flush=True)
    best_runs: list[dict[str, Any]] = []
    for seed in range(args.seed_base, args.seed_base + args.n_seeds):
        # Reuse the sweep run if it's the same (seed, λ) pair to save
        # a few seconds — the sweep already covered seed_base..seed_base+sweep_seeds.
        if (
            seed < args.seed_base + args.sweep_seeds
            and best_lam in sweep_results
        ):
            existing = next(
                (r for r in sweep_results[best_lam] if r["seed"] == seed),
                None,
            )
            if existing is not None:
                best_runs.append(existing)
                continue
        best_runs.append(
            _run_one_seed(bench, args, seed=seed, lam=best_lam)
        )

    summary = _summarise(best_runs)
    print()
    print(f"=== EWC (best λ={best_lam}) (n={len(best_runs)}) ===")
    print(
        f"  Final ACC: mean={summary['final_acc_mean']:.3f}  "
        f"std={summary['final_acc_std']:.3f}"
    )
    print(f"  FGT:       mean={summary['fgt_mean']:.3f}")
    print(
        "  Per-class final acc: [" +
        ", ".join(
            f"{c}:{summary['per_class_means'][c]:.2f}"
            for c in range(args.n_classes)
        )
        + "]"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_path = args.output_dir / f"{ts}_37_split_mnist_ci_ewc.json"
    with out_path.open("w") as f:
        json.dump({
            "experiment": "37_split_mnist_ci_ewc",
            "method": "ewc",
            "phase": "5.5.2",
            "timestamp": ts,
            "config": {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in vars(args).items()
            },
            "best_lambda": best_lam,
            "sweep_summary": {
                str(lam): s for lam, s in sweep_summary.items()
            },
            "per_seed_best": best_runs,
            "summary": summary,
        }, f, indent=2)
    print(f"\nWrote results JSON to {out_path}")


if __name__ == "__main__":
    main()
