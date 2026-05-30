"""Experiment 49 — Phase 5.7.2: XRay on Split-MNIST class-incremental.

Real-data validation of the X-Ray training pipeline on the
familiar Split-MNIST CI benchmark where the project's current
recipes have known numbers:

- CLS-CI v2 (exp 41, n=10):  NEO ACC = 0.861 ± 0.020
- DER (exp 38, n=3):         ACC = 0.879 ± 0.005
- naive (exp 36):            ACC ≈ 0.191
- EWC (exp 37):              ACC ≈ 0.193

If XRay matches/beats the current CLS-CI v2 on this paradigm,
the architecture is real — proceed to harder CIFAR. If it
underperforms, fix the integration before scaling.

Architecture (locked from Phase 5.7.1, MLP for Split-MNIST):

- Hippocampe MLP (128, 64): trains naive (CE only); serves as
  the negative control inside each run so catastrophic
  forgetting is visible on the same data the neo sees.
- Neocortex MLP (256, 256, 128): trains on CE + λ·NT-Xent
  against the *whole* prototype set. Memory is updated with
  the neo's 128-d features on correctly-classified samples.
- XRayMemory(feature_dim=128, prototypes_per_class=3): the
  single shared prototype store.
- Consolidation phase after each task: classifier-head-only
  fine-tune on the prototype set; the encoder is frozen by
  virtue of a separate optimizer that holds only
  ``neo.classifier.parameters()``.

Training schedule matches every other Split-MNIST CI experiment
in this repo:  epochs_per_task=1, batch=64, SGD momentum=0.9,
hipp_lr=0.05, neo_lr=0.01. The variable being swept is
``--lambda_contrast`` ∈ {1.0, 2.0, 5.0} per the Phase 5.7.2
spec.

Run from the repo root::

    python experiments/49_xray_split_mnist_ci.py
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
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from continual_synapse.evaluation.benchmarks import (  # noqa: E402
    SplitMNISTClassIncremental,
)
from continual_synapse.evaluation.runner import set_seed  # noqa: E402
from continual_synapse.memory import (  # noqa: E402
    XRayMemory, nt_xent_multi_prototype_loss,
)


# ---------- baselines from prior phases (for comparison table) ----------


_BASELINES: dict[str, float] = {
    "CLS-CI v2": 0.861,   # Phase 5.5.6 n=10
    "DER":        0.879,   # exp 38 n=3
    "naive":      0.191,   # exp 36 n=3
    "EWC":        0.193,   # exp 37 n=3
}


# ---------- MLP models (mirror exp 41) ----------


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
        self.feature_dim = prev

    def features(self, x: Tensor) -> Tensor:
        return self.encoder(x)

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
        self.feature_dim = prev

    def features(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.encoder(x))


# ---------- training step (mirrors exp 48) ----------


def train_step_dual_xray(
    hipp: Hippocampus, neo: Neocortex,
    memory: XRayMemory,
    hipp_optimizer: torch.optim.Optimizer,
    neo_optimizer: torch.optim.Optimizer,
    x_batch: Tensor, y_batch: Tensor,
    *,
    lambda_contrast: float,
    grad_clip: float | None = 1.0,
) -> dict[str, float]:
    """Hipp CE only (control). Neo CE + λ·NT-Xent vs prototypes.
    Memory updated under no_grad with neo features for samples
    the neo classified correctly this step."""
    # Hippocampe.
    hipp_optimizer.zero_grad()
    hipp_logits = hipp(x_batch)
    hipp_loss = F.cross_entropy(hipp_logits, y_batch)
    hipp_loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(hipp.parameters(), max_norm=grad_clip)
    hipp_optimizer.step()

    # Neocortex.
    neo_optimizer.zero_grad()
    neo_features = neo.features(x_batch)
    neo_logits = neo.classifier(neo_features)
    ce_loss = F.cross_entropy(neo_logits, y_batch)

    contrast_loss = torch.zeros((), device=x_batch.device)
    if memory.num_occupied() > 0:
        prototypes, proto_labels = memory.get_all_prototypes()
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

    # Memory update under no_grad: reward signal = was-classified-correctly.
    with torch.no_grad():
        pred = neo_logits.argmax(dim=-1)
        correct_mask = (pred == y_batch)
        memory.update(neo_features.detach(), y_batch, correct_mask)

    return {
        "hipp_loss":     float(hipp_loss.item()),
        "ce_loss":       float(ce_loss.item()),
        "contrast_loss": float(contrast_loss.item()),
    }


# ---------- consolidation (classifier-head-only fine-tune) ----------


def consolidate_with_xray(
    neo: Neocortex,
    memory: XRayMemory,
    classifier_optimizer: torch.optim.Optimizer,
    *,
    cons_epochs: int = 2,
    grad_clip: float | None = 1.0,
) -> dict[str, float]:
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
    model: nn.Module, ds: TensorDataset, device: torch.device,
    n_classes: int = 10,
) -> dict[str, Any]:
    model.eval()
    x = ds.tensors[0].to(device)
    y = ds.tensors[1].to(device)
    preds = model(x).argmax(dim=-1)
    acc = float((preds == y).float().mean().item())
    per_class: list[float] = []
    for c in range(n_classes):
        mask = (y == c)
        if mask.any():
            per_class.append(
                float((preds[mask] == c).float().mean().item())
            )
        else:
            per_class.append(float("nan"))
    return {"acc": acc, "per_class": per_class}


# ---------- per-seed driver ----------


def _run_one_seed(
    bench: SplitMNISTClassIncremental, args: argparse.Namespace,
    lambda_contrast: float, seed: int,
) -> dict[str, Any]:
    set_seed(seed)
    device = torch.device(args.device)
    hipp = Hippocampus(
        input_dim=args.input_dim,
        hidden_dims=tuple(args.hipp_hidden_dims),
        n_classes=args.n_classes,
    ).to(device)
    neo = Neocortex(
        input_dim=args.input_dim,
        hidden_dims=tuple(args.neo_hidden_dims),
        n_classes=args.n_classes,
    ).to(device)
    memory = XRayMemory(
        num_classes=args.n_classes,
        feature_dim=neo.feature_dim,
        prototypes_per_class=args.prototypes_per_class,
        # Schedules tuned for the 1-epoch-per-task Split-MNIST
        # regime — defaults are calibrated for long CIFAR runs
        # and won't fire here without these overrides.
        sparsity_start_refinements=10,
        sparsity_end_refinements=80,
        temp_start_refinements=10,
        temp_end_refinements=80,
    ).to(device)

    hipp_opt = torch.optim.SGD(
        hipp.parameters(), lr=args.hipp_lr,
        momentum=args.momentum, weight_decay=args.weight_decay,
    )
    neo_opt = torch.optim.SGD(
        neo.parameters(), lr=args.neo_lr,
        momentum=args.momentum, weight_decay=args.weight_decay,
    )
    neo_cls_opt = torch.optim.SGD(
        neo.classifier.parameters(), lr=args.cons_lr,
        momentum=args.momentum, weight_decay=args.weight_decay,
    )

    full_test = bench.all_test_dataset()
    per_task_acc_neo:  list[float] = []
    per_task_acc_hipp: list[float] = []
    t0 = time.time()

    for task_id, task in enumerate(bench.tasks()):
        loader = DataLoader(
            task.train, batch_size=args.batch_size, shuffle=True,
        )
        hipp.train(); neo.train()
        for _ in range(args.epochs_per_task):
            for x, y in loader:
                x = x.to(device); y = y.to(device)
                train_step_dual_xray(
                    hipp, neo, memory,
                    hipp_opt, neo_opt, x, y,
                    lambda_contrast=lambda_contrast,
                    grad_clip=args.grad_clip,
                )

        cons_diag = consolidate_with_xray(
            neo, memory, neo_cls_opt, cons_epochs=args.cons_epochs,
            grad_clip=args.grad_clip,
        )

        neo_eval = evaluate(neo, full_test, device, n_classes=args.n_classes)
        hipp_eval = evaluate(hipp, full_test, device, n_classes=args.n_classes)
        per_task_acc_neo.append(neo_eval["acc"])
        per_task_acc_hipp.append(hipp_eval["acc"])

    wall = time.time() - t0
    # Diagnostic snapshot of memory at end of run.
    occupied_counts = memory.refinement_counts[memory.is_occupied]
    mean_refinement = float(occupied_counts.float().mean().item()) if memory.num_occupied() > 0 else 0.0
    # Mean sparsification across occupied prototypes.
    prototypes_t, _ = memory.get_all_prototypes()
    if prototypes_t.shape[0] > 0:
        zero_frac = float((prototypes_t == 0).float().mean().item())
    else:
        zero_frac = float("nan")

    final = {
        "seed": seed,
        "lambda_contrast": lambda_contrast,
        "per_task_acc_neo":  per_task_acc_neo,
        "per_task_acc_hipp": per_task_acc_hipp,
        "neo_per_class_final":  evaluate(
            neo, full_test, device, n_classes=args.n_classes,
        )["per_class"],
        "hipp_per_class_final": evaluate(
            hipp, full_test, device, n_classes=args.n_classes,
        )["per_class"],
        "final_neo_acc":  per_task_acc_neo[-1],
        "final_hipp_acc": per_task_acc_hipp[-1],
        "memory_size":    memory.num_occupied(),
        "per_class_counts": memory.per_class_counts(),
        "mean_refinement": mean_refinement,
        "mean_zero_fraction": zero_frac,
        "temperature_end": memory.temperature(),
        "wall_time_s": wall,
    }
    return final


# ---------- reporting ----------


def _print_sweep_table(
    sweep: dict[float, list[dict[str, Any]]],
) -> None:
    print()
    print("Reference baselines:")
    for name, v in _BASELINES.items():
        print(f"  {name:<12}: {v:.3f}")

    print()
    hdr = (
        f"{'Config':<14}{'lambda':>8}  {'ACC mean ± std':<18} "
        f"{'vs CLS-CI v2':>14} {'vs DER':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for lam, runs in sweep.items():
        accs = [r["final_neo_acc"] for r in runs]
        m = statistics.fmean(accs)
        s = statistics.stdev(accs) if len(accs) > 1 else 0.0
        delta_v2  = (m - _BASELINES["CLS-CI v2"]) * 100  # pp
        delta_der = (m - _BASELINES["DER"]) * 100
        cfg = f"xray_l{lam:g}"
        sign_v2 = "+" if delta_v2 >= 0 else ""
        sign_der = "+" if delta_der >= 0 else ""
        print(
            f"{cfg:<14}{lam:>8.1f}  {m:.3f} ± {s:.3f}     "
            f"{sign_v2}{delta_v2:>+7.3f} pp  "
            f"{sign_der}{delta_der:>+7.3f} pp"
        )


def _print_best_diagnostics(
    sweep: dict[float, list[dict[str, Any]]],
) -> tuple[float, list[dict[str, Any]]]:
    """Pick best lambda by mean final NEO ACC; print its
    per-task retention curve + memory diagnostics."""
    best_lam = max(
        sweep.keys(),
        key=lambda l: statistics.fmean(r["final_neo_acc"] for r in sweep[l]),
    )
    runs = sweep[best_lam]
    n_tasks = len(runs[0]["per_task_acc_neo"])
    per_task_means = [
        statistics.fmean(r["per_task_acc_neo"][t] for r in runs)
        for t in range(n_tasks)
    ]
    mem_size_mean = statistics.fmean(r["memory_size"] for r in runs)
    per_class_counts_mean = [
        statistics.fmean(r["per_class_counts"][c] for r in runs)
        for c in range(len(runs[0]["per_class_counts"]))
    ]
    mean_refinement_mean = statistics.fmean(r["mean_refinement"] for r in runs)
    zero_frac_mean = statistics.fmean(r["mean_zero_fraction"] for r in runs)
    temp_end_mean = statistics.fmean(r["temperature_end"] for r in runs)

    print()
    print(f"Per-config diagnostics (best: λ={best_lam}, n={len(runs)}):")
    print(f"  Memory final size: {mem_size_mean:.1f} prototypes (mean across seeds)")
    print(
        "  Per-class prototype counts: ["
        + ", ".join(f"{c:.1f}" for c in per_class_counts_mean)
        + "]"
    )
    print(f"  Mean prototype refinement count: {mean_refinement_mean:.1f}")
    print(f"  Mean sparsification (% zeros): {zero_frac_mean*100:.1f}%")
    print(f"  Temperature at end: {temp_end_mean:.3f}")
    print()
    print(f"Per-task retention curve (best λ={best_lam}, mean across seeds):")
    for t, acc in enumerate(per_task_means):
        cls_lo = 2 * t
        cls_hi = 2 * t + 1
        print(f"  task {t} (classes {cls_lo}-{cls_hi}):  {acc:.3f}")
    return best_lam, runs


def _print_verdict(
    sweep: dict[float, list[dict[str, Any]]],
) -> str:
    best_lam = max(
        sweep.keys(),
        key=lambda l: statistics.fmean(r["final_neo_acc"] for r in sweep[l]),
    )
    best_mean = statistics.fmean(
        r["final_neo_acc"] for r in sweep[best_lam]
    )
    print()
    print("=== Verdict ===")
    if best_mean >= 0.85:
        verdict = "STRONG"
        msg = (
            "STRONG — matches/beats current CLS-CI v2 (0.861) and "
            "approaches DER (0.879). XRay works on Split-MNIST CI; "
            "proceed to Phase 5.7.3."
        )
    elif best_mean >= 0.75:
        verdict = "MODERATE"
        msg = (
            "MODERATE — competitive but slightly below CLS-CI v2. "
            "Refine λ / prototype balance before scaling."
        )
    elif best_mean >= 0.60:
        verdict = "WEAK"
        msg = (
            "WEAK — XRay underperforms the established baselines. "
            "Investigate whether memory is engaging (per-class counts), "
            "contrastive too weak, or classifier-head consolidation "
            "is overwhelming the encoder's representation."
        )
    else:
        verdict = "FAILURE"
        msg = (
            "FAILURE — fundamental issue with integration on this "
            "paradigm. Diagnose before scaling."
        )
    print(f"  Best config: λ={best_lam}, ACC={best_mean:.3f}")
    print(f"  {msg}")
    return verdict


# ---------- main ----------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--lambdas", type=float, nargs="+", default=[1.0, 2.0, 5.0],
        help="Sweep these λ_contrast values, n_seeds per value.",
    )
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--seed_base", type=int, default=0)
    p.add_argument("--epochs_per_task", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--input_dim", type=int, default=784)
    p.add_argument("--n_classes", type=int, default=10)
    p.add_argument(
        "--hipp_hidden_dims", type=int, nargs="+", default=[128, 64],
    )
    p.add_argument(
        "--neo_hidden_dims", type=int, nargs="+", default=[256, 256, 128],
    )
    p.add_argument("--hipp_lr", type=float, default=0.05)
    p.add_argument("--neo_lr", type=float, default=0.01)
    p.add_argument("--cons_lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--cons_epochs", type=int, default=2)
    p.add_argument("--prototypes_per_class", type=int, default=3)
    p.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=_REPO_ROOT / "results" / "logs" / "split_mnist_ci",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print(
        f"Phase 5.7.2 — XRay on Split-MNIST CI\n"
        f"  λ_contrast sweep: {args.lambdas}\n"
        f"  n_seeds={args.n_seeds}  epochs_per_task={args.epochs_per_task}\n"
        f"  hipp: dims={tuple(args.hipp_hidden_dims)} lr={args.hipp_lr}\n"
        f"  neo:  dims={tuple(args.neo_hidden_dims)} lr={args.neo_lr}\n"
        f"  consolidation: cons_epochs={args.cons_epochs} "
        f"cons_lr={args.cons_lr}\n"
        f"  XRay: prototypes_per_class={args.prototypes_per_class}\n"
        f"  device={args.device}",
        flush=True,
    )

    t_load = time.time()
    bench = SplitMNISTClassIncremental.from_huggingface()
    print(f"Loaded benchmark in {time.time() - t_load:.1f}s.\n")

    sweep: dict[float, list[dict[str, Any]]] = {}
    for lam in args.lambdas:
        print(f"=== λ_contrast = {lam} ===", flush=True)
        runs: list[dict[str, Any]] = []
        for seed in range(args.seed_base, args.seed_base + args.n_seeds):
            t_seed = time.time()
            r = _run_one_seed(bench, args, lambda_contrast=lam, seed=seed)
            runs.append(r)
            print(
                f"  seed={seed:>2}  final NEO ACC={r['final_neo_acc']:.3f}  "
                f"|mem|={r['memory_size']}  "
                f"in {time.time() - t_seed:.1f}s",
                flush=True,
            )
        sweep[lam] = runs

    print()
    print("=== Phase 5.7.2 — XRay on Split-MNIST CI sweep results ===")
    _print_sweep_table(sweep)
    best_lam, _ = _print_best_diagnostics(sweep)
    verdict = _print_verdict(sweep)

    # Persist JSON.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_path = (
        args.output_dir / f"{ts}_49_xray_split_mnist_ci.json"
    )
    payload = {
        "experiment": "49_xray_split_mnist_ci",
        "phase": "5.7.2",
        "timestamp": ts,
        "config": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
        "baselines": _BASELINES,
        "sweep": {str(lam): runs for lam, runs in sweep.items()},
        "best_lambda": best_lam,
        "verdict": verdict,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote results JSON to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
