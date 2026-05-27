"""Experiment 33 — Phase 2 of the CLS rebuild: neocortex alone.

This is component 2 of 6 in the incremental Complementary Learning
Systems (CLS) build. Here we ONLY construct the slow learner
(neocortex) and verify that — trained in isolation, no hippocampe,
no consolidation, no protection — it produces the standard naive-
continual baseline signature on Permuted-MNIST.

This is the *reference point*. It is NOT supposed to be good. It
is supposed to match published naive-finetune numbers so that any
later improvement attributable to consolidation has a clean,
honest comparison.

Expected naive-baseline behavior at T=15:
- ACC ~0.66
- Task-0 ~0.20
- Task-N ~0.91
- Retention curve R[T-1, :] decays smoothly from Task-0 to Task-N
  (gradient of forgetting), NOT the sharp endpoint ramp the
  hippocampe produces.

Architecture:
- Neocortex MLP: 784 -> 256 -> 256 -> 128 -> 10
- SGD, lr=0.01 (5× slower than the hippocampe's 0.05),
  momentum=0.9
- Batch size 64, 1 epoch/task (matches the training budget used by
  every other CL experiment in this project, including the
  cs_gated_cosine_functional DER-equivalent baseline at ACC=0.904)
- Permuted-MNIST T=15, n_seeds=2

Run from the repo root::

    python experiments/33_cls_phase2_neocortex.py
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
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from continual_synapse.evaluation.benchmarks import PermutedMNIST  # noqa: E402
from continual_synapse.evaluation.runner import set_seed  # noqa: E402


# ---------- Phase 1 reference numbers ----------
#
# Captured from the Phase 1 hippocampe run (exp 32, T=15 n=2, lr=0.05,
# 3 epochs/task) committed in f18642d. Kept as constants so the
# Phase 2 report can print a side-by-side comparison without
# re-reading the prior log.
_PHASE1_HIPPO_ACC = 0.278
_PHASE1_HIPPO_TASK0 = 0.126
_PHASE1_HIPPO_TASKN = 0.955
_PHASE1_HIPPO_FGT = 0.829


# ---------- model ----------


class Neocortex(nn.Module):
    """Slow learner component of CLS architecture.

    Standard MLP with our usual hyperparameters — establishes the
    baseline performance against which consolidation improvements
    will be measured in later phases. Deliberately UNDEFENDED:
    no EWC, no replay, no cosine gating, no functional reg. The
    forgetting we measure here is exactly what consolidation has
    to push back on.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: tuple = (256, 256, 128),  # larger than hippocampus
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

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))


# ---------- training / eval ----------


def _train_one_task(
    model: Neocortex,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    epochs: int,
    device: str,
) -> float:
    """Standard cross-entropy SGD on one task. Returns mean batch loss."""
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


def _eval_all_tasks(
    model: Neocortex, bench: PermutedMNIST, device: str,
) -> list[float]:
    """Return per-task test accuracy across every task in ``bench``."""
    model.eval()
    accs: list[float] = []
    with torch.no_grad():
        for task in bench.tasks():
            x = task.test.tensors[0].to(device)
            y = task.test.tensors[1].to(device)
            preds = model(x).argmax(dim=-1)
            accs.append(float((preds == y).float().mean().item()))
    return accs


def _train_one_seed(
    bench: PermutedMNIST, args: argparse.Namespace, seed: int,
) -> dict[str, Any]:
    """Train one neocortex seed across all tasks sequentially.

    Returns a dict with the final eval row ``R[T-1, :]`` plus
    summary scalars (ACC, Task-0, Task-N, FGT).
    """
    set_seed(seed)
    model = Neocortex(
        input_dim=args.input_dim,
        hidden_dims=tuple(args.hidden_dims),
        n_classes=args.n_classes,
    ).to(args.device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
    )

    t0 = time.time()
    per_task_train_loss: list[float] = []
    for task_idx, task in enumerate(bench.tasks()):
        loader = DataLoader(
            task.train, batch_size=args.batch_size, shuffle=True,
        )
        avg_loss = _train_one_task(
            model, optimizer, loader, epochs=args.epochs_per_task,
            device=args.device,
        )
        per_task_train_loss.append(avg_loss)
        print(
            f"    seed={seed}  task={task_idx:2d}  "
            f"train_loss={avg_loss:.4f}",
            flush=True,
        )

    final_row = _eval_all_tasks(model, bench, device=args.device)
    wall = time.time() - t0

    acc = statistics.fmean(final_row)
    task0 = final_row[0]
    taskN = final_row[-1]
    # Signed gap (Task-N − Task-0). Naive finetune produces a
    # positive but smoothly-graded curve, so the gap is moderate
    # — not the catastrophic ~0.83 we saw on the hippocampe.
    fgt = taskN - task0

    return {
        "seed": int(seed),
        "final_row": final_row,
        "per_task_train_loss": per_task_train_loss,
        "metrics": {
            "average_accuracy": acc,
            "task0_retention": task0,
            "taskN_final": taskN,
            "forgetting_proxy": fgt,
        },
        "wall_time_s": wall,
    }


# ---------- reporting ----------


def _signature_check(
    value: float, low: float, high: float,
) -> str:
    """Render the ``PASS/FAIL (got X.XXX)`` tail of an interval check."""
    passed = low <= value <= high
    status = "PASS" if passed else "FAIL"
    return f"{status} (got {value:.3f})"


def _print_report(
    args: argparse.Namespace,
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    accs = [s["metrics"]["average_accuracy"] for s in per_seed]
    t0s = [s["metrics"]["task0_retention"] for s in per_seed]
    tNs = [s["metrics"]["taskN_final"] for s in per_seed]
    fgts = [s["metrics"]["forgetting_proxy"] for s in per_seed]
    mean_acc = statistics.fmean(accs)
    mean_t0 = statistics.fmean(t0s)
    mean_tN = statistics.fmean(tNs)
    mean_fgt = statistics.fmean(fgts)

    print()
    print("=== Phase 2: Néocortex standalone behavior test ===")
    print(
        f"T={args.T}, n={args.n_seeds}, lr={args.lr} "
        f"(standard baseline)"
    )
    print()
    print("Per-seed results:")
    for s in per_seed:
        m = s["metrics"]
        print(
            f"  seed {s['seed']}: ACC={m['average_accuracy']:.3f}  "
            f"Task-0={m['task0_retention']:.3f}  "
            f"Task-N={m['taskN_final']:.3f}  "
            f"FGT={m['forgetting_proxy']:.3f}"
        )

    print()
    print("Mean across seeds:")
    print(
        f"  ACC: {mean_acc:.3f}  Task-0: {mean_t0:.3f}  "
        f"Task-N: {mean_tN:.3f}  FGT: {mean_fgt:.3f}"
    )

    # Per-task retention curve (averaged across seeds).
    T = args.T
    per_task_means: list[float] = []
    for k in range(T):
        per_task_means.append(
            statistics.fmean(s["final_row"][k] for s in per_seed)
        )
    print()
    print("Per-task final accuracy (R[T-1, k]):")
    chunk = "  ".join(
        f"t={k}: {v:.2f}" for k, v in enumerate(per_task_means)
    )
    print(f"  {chunk}")

    print()
    print("=== Behavioral signature check (vs naive baseline) ===")
    print(
        f"[expected] ACC ~0.66 (naive level):  "
        f"{_signature_check(mean_acc, 0.60, 0.72)}"
    )
    print(
        f"[expected] Task-0 ~0.20:               "
        f"{_signature_check(mean_t0, 0.15, 0.30)}"
    )
    print(
        f"[expected] Task-N ~0.91:               "
        f"{_signature_check(mean_tN, 0.85, 0.95)}"
    )

    print()
    print("Comparison to Phase 1 hippocampus:")
    print("                 hippocampus    neocortex")
    print(
        f"  ACC:           {_PHASE1_HIPPO_ACC:.3f}          "
        f"{mean_acc:.3f}"
    )
    print(
        f"  Task-0:        {_PHASE1_HIPPO_TASK0:.3f}          "
        f"{mean_t0:.3f}"
    )
    print(
        f"  Task-N:        {_PHASE1_HIPPO_TASKN:.3f}          "
        f"{mean_tN:.3f}"
    )
    print(
        f"  FGT:           {_PHASE1_HIPPO_FGT:.3f}          "
        f"{mean_fgt:.3f}"
    )

    # Verdict logic. PASS = all three metrics inside their expected
    # naive-baseline intervals. WARN paths are split so the user
    # can tell whether the model is suspiciously stable (no
    # protection but somehow retaining old tasks) or suspiciously
    # weak (can't even learn the current task).
    pass_acc = 0.60 <= mean_acc <= 0.72
    pass_t0 = 0.15 <= mean_t0 <= 0.30
    pass_tN = 0.85 <= mean_tN <= 0.95
    verdict_pass = pass_acc and pass_t0 and pass_tN

    print()
    if verdict_pass:
        print("Verdict: NÉOCORTEX BEHAVES AS BASELINE")
    else:
        # Two distinct WARN paths from the spec.
        if mean_acc > 0.75 or mean_t0 > 0.35:
            print("Verdict: NEEDS ADJUSTMENT — neocortex too good for naive")
            print(
                f"  ACC={mean_acc:.3f} (>0.75?) or "
                f"Task-0={mean_t0:.3f} (>0.35?): suspicious. With no "
                f"protection mechanisms (no EWC, no replay, no "
                f"gating) the network shouldn't be holding onto old "
                f"tasks this well. Likely causes: bug in the "
                f"continual loop (e.g., re-shuffling tasks instead "
                f"of sequencing them), or lr too small relative to "
                f"the model size."
            )
        elif mean_acc < 0.55:
            print("Verdict: NEEDS ADJUSTMENT — neocortex too bad")
            print(
                f"  ACC={mean_acc:.3f} (<0.55): the network isn't "
                f"even learning the current task well. Likely "
                f"causes: lr too small, too few epochs/task, or an "
                f"architecture bug. Sanity-check Task-N "
                f"({mean_tN:.3f}) — if it's also below ~0.85 the "
                f"model itself is under-fitting; if Task-N is fine "
                f"but ACC is low, the forgetting is exaggerated "
                f"(check for accidental high effective lr)."
            )
        else:
            # In-range on the WARN axes but at least one of the three
            # interval checks failed — usually a near-miss.
            print("Verdict: NEEDS ADJUSTMENT — near-miss on baseline intervals")
            failing = []
            if not pass_acc:
                failing.append(f"ACC={mean_acc:.3f} ∉ [0.60, 0.72]")
            if not pass_t0:
                failing.append(f"Task-0={mean_t0:.3f} ∉ [0.15, 0.30]")
            if not pass_tN:
                failing.append(f"Task-N={mean_tN:.3f} ∉ [0.85, 0.95]")
            for f_ in failing:
                print(f"  - {f_}")
            print(
                "  Decide whether the bracket needs widening or "
                "whether the training schedule needs a small "
                "adjustment (epochs/task, lr) before Phase 3."
            )

    return {
        "n_seeds": len(per_seed),
        "metric_means": {
            "average_accuracy": mean_acc,
            "task0_retention": mean_t0,
            "taskN_final": mean_tN,
            "forgetting_proxy": mean_fgt,
        },
        "per_task_means": per_task_means,
        "verdict": "PASS" if verdict_pass else "NEEDS_ADJUSTMENT",
        "phase1_hippocampus_reference": {
            "average_accuracy": _PHASE1_HIPPO_ACC,
            "task0_retention": _PHASE1_HIPPO_TASK0,
            "taskN_final": _PHASE1_HIPPO_TASKN,
            "forgetting_proxy": _PHASE1_HIPPO_FGT,
        },
    }


# ---------- main ----------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--T", type=int, default=15)
    p.add_argument("--n_seeds", type=int, default=2)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument(
        "--epochs-per-task", "--epochs_per_task",
        dest="epochs_per_task", type=int, default=1,
        help=(
            "Training budget per task. Default 1 to match every "
            "other CL experiment in this project; later phases "
            "should keep this fixed so Phase 6 eval is comparable "
            "to the audited 0.904/0.908 DER-equivalent baseline."
        ),
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument(
        "--hidden-dims", type=int, nargs="+", default=[256, 256, 128],
        help="Hidden layer sizes for the neocortex encoder.",
    )
    p.add_argument("--input-dim", type=int, default=784)
    p.add_argument("--n-classes", type=int, default=10)
    p.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--bench-seed", type=int, default=42,
        help="Seed for the permutation set (same across model seeds).",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=_REPO_ROOT / "results" / "logs" / "cls_phase2",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.epochs_per_task = int(args.epochs_per_task)
    args.hidden_dims = [int(h) for h in args.hidden_dims]

    _stub = Neocortex(
        input_dim=args.input_dim,
        hidden_dims=tuple(args.hidden_dims),
        n_classes=args.n_classes,
    )
    total_params = sum(p.numel() for p in _stub.parameters())
    print(
        f"Néocortex: input_dim={args.input_dim} "
        f"hidden_dims={tuple(args.hidden_dims)} "
        f"n_classes={args.n_classes}  =>  {total_params:,} params",
        flush=True,
    )
    del _stub

    print(
        f"CLS Phase 2 — neocortex standalone\n"
        f"  T={args.T}  n_seeds={args.n_seeds}  "
        f"epochs_per_task={args.epochs_per_task}\n"
        f"  optimizer=SGD lr={args.lr} momentum={args.momentum} "
        f"batch_size={args.batch_size}\n"
        f"  device={args.device}",
        flush=True,
    )

    bench = PermutedMNIST.from_huggingface(
        num_tasks=args.T, seed=args.bench_seed,
    )

    per_seed: list[dict[str, Any]] = []
    seeds = list(range(args.seed_base, args.seed_base + args.n_seeds))
    for seed in seeds:
        print(f"\n  --- seed {seed} ---", flush=True)
        result = _train_one_seed(bench, args, seed=seed)
        per_seed.append(result)
        m = result["metrics"]
        print(
            f"  seed={seed} done in {result['wall_time_s']:.1f}s   "
            f"ACC={m['average_accuracy']:.3f}  "
            f"Task-0={m['task0_retention']:.3f}  "
            f"Task-N={m['taskN_final']:.3f}  "
            f"FGT={m['forgetting_proxy']:.3f}",
            flush=True,
        )

    summary = _print_report(args, per_seed)

    # Persist JSON so the result is reproducible and lives next to
    # Phase 1's log under the same naming convention.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_path = args.output_dir / f"{ts}_33_T{args.T}_cls_phase2.json"
    payload = {
        "experiment": "33_cls_phase2_neocortex",
        "phase": 2,
        "component": "neocortex_standalone",
        "num_tasks": args.T,
        "timestamp": ts,
        "config": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
        "param_count": int(total_params),
        "per_seed": per_seed,
        "summary": summary,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote results JSON to {out_path}", flush=True)


if __name__ == "__main__":
    main()
