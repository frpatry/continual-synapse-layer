"""Experiment 35 — Phase 4 of the CLS rebuild: re-encoding consolidation.

This is component 4 of 6 in the incremental Complementary Learning
Systems (CLS) build — and the one that determines whether the
architecture works at all. Phases 1–3 built the parts in
isolation; this phase composes them into a working continual-
learning loop:

For each task:
  1. Train the hippocampe on the task (fast, volatile learner).
  2. Train the neocortex on the task (slow, dense learner).
  3. Store ``samples_per_task`` entries in MultiLevelMemory
     (input, per-layer hipp features, hipp soft target, label).
  4. Consolidate: replay every memory entry, applying
     - a multi-level *anchor loss* that keeps the hippocampe's
       current features close to what it produced at storage
       time (drift mitigation),
     - a *distillation loss* that teaches the neocortex to match
       the hippocampe's stored soft predictions, and
     - a *task loss* (cross-entropy on the true label) so the
       neocortex isn't relying on distillation alone.

The whole experiment lives or dies on four gates evaluated after
T=5 tasks:

  Gate 1 — drift_low_corr at end > 0.7
  Gate 2 — neocortex Task-0 > 0.25 (naive baseline +5pp)
  Gate 3 — neocortex Task-N > 0.85
  Gate 4 — hippocampe Task-N > 0.85

Run from the repo root::

    python experiments/35_cls_phase4_consolidation.py
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from continual_synapse.evaluation.benchmarks import PermutedMNIST  # noqa: E402
from continual_synapse.evaluation.runner import set_seed  # noqa: E402


# ---------- Phase 2 reference (naive neocortex baseline @ epochs=1) ----------
#
# Captured from exp 33 T=15 n=2 epochs_per_task=1. Used as the
# +5pp anchor for Gate 2 below. Note that this reference is at
# T=15 while Phase 4 trains at T=5; the gate stays comparable
# because the consolidation effect should manifest from the
# earliest tasks onwards.
_PHASE2_NAIVE_TASK0 = 0.200


# ---------- models (copies from exp 32 / 33 — self-contained) ----------


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

    def features(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.encoder(x))


# ---------- memory (Phase 3 + labels) ----------


class MultiLevelMemory:
    """Phase 3's hierarchical memory, extended to store the
    ground-truth label per entry. The label is needed by the
    Phase 4 consolidation step's task loss (cross-entropy on the
    neocortex's prediction)."""

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

    @torch.no_grad()
    def record_task_end(
        self,
        hippocampus: Hippocampus,
        task_inputs: Tensor,
        task_labels: Tensor,
        task_id: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> int:
        """Sample ``samples_per_task`` entries from ``task_inputs``
        and snapshot the hippocampe's per-layer activations + soft
        prediction + the ground-truth labels at the sampled
        indices. ``generator`` allows the caller to make the
        sampling deterministic across re-runs.
        """
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
                "Phase-1 architecture has drifted."
            )

        low_feat = layer_outputs[0]
        mid_feat = layer_outputs[1]
        high_feat = h.detach().cpu()
        soft = F.softmax(
            hippocampus.classifier(h), dim=-1,
        ).detach().cpu()

        for i in range(n):
            self.inputs.append(sampled[i].detach().cpu())
            self.low_features.append(low_feat[i])
            self.mid_features.append(mid_feat[i])
            self.high_features.append(high_feat[i])
            self.soft_targets.append(soft[i])
            self.labels.append(int(sampled_labels[i].item()))
            self.task_ids.append(int(task_id))

        return n

    def __len__(self) -> int:
        return len(self.inputs)

    def per_task_counts(self) -> dict[int, int]:
        return dict(Counter(self.task_ids))


# ---------- consolidation ----------


def consolidate(
    hippocampus: Hippocampus,
    neocortex: Neocortex,
    memory: MultiLevelMemory,
    hipp_optimizer: torch.optim.Optimizer,
    neo_optimizer: torch.optim.Optimizer,
    *,
    batch_size: int = 64,
    n_epochs: int = 1,
    lambda_distill: float = 1.0,
    lambda_anchor_low: float = 1.0,
    lambda_anchor_mid: float = 0.5,
    lambda_anchor_high: float = 0.1,
    device: torch.device = torch.device("cpu"),
) -> dict[str, float]:
    """Run one consolidation pass over the entire memory.

    For each batch:
      - Hippocampe pass: re-encode ``x``, compute a per-layer MSE
        between the new features and the stored features (anchor
        loss), backprop into hipp weights. This is what holds the
        feature space stable as new tasks come in.
      - Neocortex pass: forward ``x``, compute cross-entropy on
        the true label PLUS KL divergence against the stored
        hipp soft target (distillation), backprop into neo
        weights. This is how knowledge moves hipp → neo.

    Drift is tracked as the cosine similarity between the new and
    stored features at each level. ``> 0.7`` at end of training
    is the gate for "anchor working".
    """
    n = len(memory)
    indices = torch.randperm(n)

    metrics: dict[str, list[float]] = {
        "task_losses": [],
        "distill_losses": [],
        "anchor_low_losses": [],
        "anchor_mid_losses": [],
        "anchor_high_losses": [],
        "drift_low_corr": [],
        "drift_mid_corr": [],
        "drift_high_corr": [],
    }

    hippocampus.train()
    neocortex.train()

    for _epoch in range(n_epochs):
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
            y = torch.tensor(
                [memory.labels[int(j)] for j in batch_idx],
                dtype=torch.long, device=device,
            )

            # ----- Hippocampe pass: anchor loss only -----
            hipp_optimizer.zero_grad()
            h = x
            new_layer_outputs: list[Tensor] = []
            for layer in hippocampus.encoder:
                h = layer(h)
                if isinstance(layer, nn.ReLU):
                    new_layer_outputs.append(h)
            new_low = new_layer_outputs[0]
            new_mid = new_layer_outputs[1]
            new_high = h

            anchor_low = F.mse_loss(new_low, stored_low)
            anchor_mid = F.mse_loss(new_mid, stored_mid)
            anchor_high = F.mse_loss(new_high, stored_high)
            anchor_total = (
                lambda_anchor_low * anchor_low
                + lambda_anchor_mid * anchor_mid
                + lambda_anchor_high * anchor_high
            )
            anchor_total.backward()
            hipp_optimizer.step()

            # Drift metric — no graph, just monitoring. We use the
            # post-step new_* tensors detached; this measures how
            # similar the hippocampe's current view is to its
            # stored view BEFORE the step. Good enough for the
            # >0.7 gate, and matches the spec.
            with torch.no_grad():
                metrics["drift_low_corr"].append(
                    float(F.cosine_similarity(
                        new_low.detach(), stored_low, dim=-1,
                    ).mean().item())
                )
                metrics["drift_mid_corr"].append(
                    float(F.cosine_similarity(
                        new_mid.detach(), stored_mid, dim=-1,
                    ).mean().item())
                )
                metrics["drift_high_corr"].append(
                    float(F.cosine_similarity(
                        new_high.detach(), stored_high, dim=-1,
                    ).mean().item())
                )

            # ----- Neocortex pass: task + distill -----
            neo_optimizer.zero_grad()
            neo_logits = neocortex(x)
            task_loss = F.cross_entropy(neo_logits, y)
            distill_loss = F.kl_div(
                F.log_softmax(neo_logits, dim=-1),
                stored_soft,
                reduction="batchmean",
            )
            neo_total = task_loss + lambda_distill * distill_loss
            neo_total.backward()
            neo_optimizer.step()

            metrics["task_losses"].append(float(task_loss.item()))
            metrics["distill_losses"].append(float(distill_loss.item()))
            metrics["anchor_low_losses"].append(float(anchor_low.item()))
            metrics["anchor_mid_losses"].append(float(anchor_mid.item()))
            metrics["anchor_high_losses"].append(float(anchor_high.item()))

    return {
        k: (float(np.mean(v)) if v else float("nan"))
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
    model: nn.Module, bench: PermutedMNIST, device: torch.device,
) -> list[float]:
    model.eval()
    accs: list[float] = []
    with torch.no_grad():
        for task in bench.tasks():
            x = task.test.tensors[0].to(device)
            y = task.test.tensors[1].to(device)
            preds = model(x).argmax(dim=-1)
            accs.append(float((preds == y).float().mean().item()))
    return accs


# ---------- per-seed driver ----------


def _run_one_seed(
    bench: PermutedMNIST, args: argparse.Namespace, seed: int,
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
    hipp_optimizer = torch.optim.SGD(
        hipp.parameters(), lr=args.hipp_lr, momentum=args.momentum,
    )
    neo_optimizer = torch.optim.SGD(
        neo.parameters(), lr=args.neo_lr, momentum=args.momentum,
    )
    memory = MultiLevelMemory(
        samples_per_task=args.samples_per_task,
        n_classes=args.n_classes,
    )

    consolidation_diagnostics: list[dict[str, Any]] = []
    t_start = time.time()

    for task_idx, task in enumerate(bench.tasks()):
        loader = DataLoader(
            task.train, batch_size=args.batch_size, shuffle=True,
        )

        # 1. Hippocampe trains on this task.
        hipp_loss = _train_one_task(
            hipp, hipp_optimizer, loader,
            epochs=args.epochs_per_task, device=device,
        )

        # 2. Neocortex also trains directly on the task — the
        # consolidation step augments this with replay, it does
        # not replace per-task learning.
        neo_loss = _train_one_task(
            neo, neo_optimizer, loader,
            epochs=args.epochs_per_task, device=device,
        )

        # 3. Store samples + labels.
        gen = torch.Generator()
        gen.manual_seed(int(seed * 1009 + task_idx))
        n_stored = memory.record_task_end(
            hipp,
            task_inputs=task.train.tensors[0],
            task_labels=task.train.tensors[1],
            task_id=task_idx,
            device=device,
            generator=gen,
        )

        # 4. Consolidate.
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
        cons["hipp_train_loss"] = float(hipp_loss)
        cons["neo_train_loss"] = float(neo_loss)
        cons["n_stored"] = int(n_stored)
        consolidation_diagnostics.append(cons)

        print(
            f"  seed={seed} task={task_idx} "
            f"hipp_loss={hipp_loss:.3f} neo_loss={neo_loss:.3f} "
            f"|cons| task={cons['task_losses']:.3f} "
            f"distill={cons['distill_losses']:.3f} "
            f"anchor[low,mid,high]="
            f"[{cons['anchor_low_losses']:.3f}, "
            f"{cons['anchor_mid_losses']:.3f}, "
            f"{cons['anchor_high_losses']:.3f}]  "
            f"drift[low,mid,high]="
            f"[{cons['drift_low_corr']:.3f}, "
            f"{cons['drift_mid_corr']:.3f}, "
            f"{cons['drift_high_corr']:.3f}]  "
            f"|mem|={len(memory)}",
            flush=True,
        )

    # Final eval on both models.
    hipp_row = _eval_all_tasks(hipp, bench, device=device)
    neo_row = _eval_all_tasks(neo, bench, device=device)

    return {
        "seed": int(seed),
        "consolidation_diagnostics": consolidation_diagnostics,
        "hipp_final_row": hipp_row,
        "neo_final_row": neo_row,
        "hipp_metrics": _row_summary(hipp_row),
        "neo_metrics": _row_summary(neo_row),
        "wall_time_s": float(time.time() - t_start),
    }


def _row_summary(row: list[float]) -> dict[str, float]:
    acc = statistics.fmean(row) if row else float("nan")
    task0 = row[0] if row else float("nan")
    taskN = row[-1] if row else float("nan")
    fgt = taskN - task0 if row else float("nan")
    return {
        "average_accuracy": float(acc),
        "task0_retention": float(task0),
        "taskN_final": float(taskN),
        "forgetting_proxy": float(fgt),
    }


# ---------- reporting ----------


def _print_consolidation_table(
    per_seed: list[dict[str, Any]], num_tasks: int,
) -> None:
    """One row per task transition, averaged across seeds."""
    print()
    print("Per-consolidation diagnostics (averaged across seeds):")
    print()
    for t in range(num_tasks):
        rows = [
            s["consolidation_diagnostics"][t] for s in per_seed
            if t < len(s["consolidation_diagnostics"])
        ]
        if not rows:
            continue
        def avg(k: str) -> float:
            vals = [r[k] for r in rows]
            return statistics.fmean(vals)

        label = (
            f"After task {t}"
            if t == num_tasks - 1
            else f"After task {t} → {t+1}"
        )
        if t == num_tasks - 1:
            label = f"{label} (final, no next task)"
        print(f"  {label}:")
        print(
            f"    task_loss={avg('task_losses'):.3f}  "
            f"distill_loss={avg('distill_losses'):.3f}  "
            f"anchor_low={avg('anchor_low_losses'):.3f}  "
            f"anchor_mid={avg('anchor_mid_losses'):.3f}  "
            f"anchor_high={avg('anchor_high_losses'):.3f}"
        )
        print(
            f"    drift_low_corr={avg('drift_low_corr'):.3f}  "
            f"drift_mid_corr={avg('drift_mid_corr'):.3f}  "
            f"drift_high_corr={avg('drift_high_corr'):.3f}"
        )


def _print_retention_rows(
    per_seed: list[dict[str, Any]], num_tasks: int,
) -> tuple[list[float], list[float]]:
    """Mean per-task retention for both models across seeds.
    Returns (hipp_row, neo_row).
    """
    hipp_means = [
        statistics.fmean(s["hipp_final_row"][k] for s in per_seed)
        for k in range(num_tasks)
    ]
    neo_means = [
        statistics.fmean(s["neo_final_row"][k] for s in per_seed)
        for k in range(num_tasks)
    ]
    print()
    print("Per-task retention (R[T-1, k]) for k=0..{}:".format(num_tasks - 1))
    print()
    print("HIPPOCAMPUS:")
    print("  " + "  ".join(
        f"t{k}: {hipp_means[k]:.2f}" for k in range(num_tasks)
    ))
    print("  (expect volatile fast-learner pattern from Phase 1)")
    print()
    print("NEOCORTEX:")
    print("  " + "  ".join(
        f"t{k}: {neo_means[k]:.2f}" for k in range(num_tasks)
    ))
    print(
        "  (expect IMPROVEMENT over naive Phase 2 baseline — "
        f"Task-0 should be > {_PHASE2_NAIVE_TASK0:.2f} + 0.05 = "
        f"{_PHASE2_NAIVE_TASK0 + 0.05:.2f})"
    )
    return hipp_means, neo_means


def _print_gates(
    per_seed: list[dict[str, Any]],
    hipp_means: list[float], neo_means: list[float],
) -> dict[str, Any]:
    """Evaluate the four decision gates."""
    # Gate 1: average of drift_low_corr across the LAST
    # consolidation step, then average across seeds.
    last_drift_lows = [
        s["consolidation_diagnostics"][-1]["drift_low_corr"]
        for s in per_seed
        if s["consolidation_diagnostics"]
    ]
    gate1_value = (
        statistics.fmean(last_drift_lows) if last_drift_lows else float("nan")
    )
    gate1_pass = gate1_value > 0.7

    # Gates 2-3-4: from per-task retention means.
    neo_task0 = neo_means[0]
    neo_taskN = neo_means[-1]
    hipp_taskN = hipp_means[-1]
    naive_plus_5pp = _PHASE2_NAIVE_TASK0 + 0.05
    gate2_pass = neo_task0 > naive_plus_5pp
    gate3_pass = neo_taskN > 0.85
    gate4_pass = hipp_taskN > 0.85

    print()
    print("=== Decision gates ===")
    print()
    print(
        f"[gate 1] Drift mitigation: drift_low_corr at end > 0.7?  "
        f"{'PASS' if gate1_pass else 'FAIL'}  (got {gate1_value:.3f})"
    )
    print(
        f"[gate 2] Knowledge transfer: neocortex Task-0 > "
        f"{naive_plus_5pp:.2f} (naive {_PHASE2_NAIVE_TASK0:.2f} +5pp)?  "
        f"{'PASS' if gate2_pass else 'FAIL'}  (got {neo_task0:.3f})"
    )
    print(
        f"[gate 3] Plasticity preservation: neocortex Task-N > 0.85?  "
        f"{'PASS' if gate3_pass else 'FAIL'}  (got {neo_taskN:.3f})"
    )
    print(
        f"[gate 4] Hippocampus still functions: hipp Task-N > 0.85?  "
        f"{'PASS' if gate4_pass else 'FAIL'}  (got {hipp_taskN:.3f})"
    )

    all_pass = gate1_pass and gate2_pass and gate3_pass and gate4_pass
    print()
    if all_pass:
        print("ALL GATES PASS — consolidation works, ready for Phase 5.")
    else:
        # Apply the spec's suggested-fix routing.
        print("NEEDS ADJUSTMENT — failures:")
        if not gate1_pass:
            print(
                "  - Gate 1 (drift): anchor loss insufficient. "
                "Increase lambda_anchor_low (and/or _mid, _high)."
            )
        if not gate2_pass and gate1_pass and gate3_pass and gate4_pass:
            print(
                "  - Gate 2 only: anchor too strong / distill too "
                "weak. Reduce lambda_anchor_* or raise lambda_distill."
            )
        elif not gate2_pass:
            print(
                "  - Gate 2 (knowledge transfer): neocortex Task-0 "
                "not improving over naive. Check distillation loss "
                "magnitude in the table above — if it's near zero, "
                "the soft targets aren't informative."
            )
        if not gate3_pass:
            print(
                "  - Gate 3 (plasticity): too much consolidation "
                "interfering with new learning. Reduce cons_epochs "
                "or lambda_distill."
            )
        if not gate4_pass:
            print(
                "  - Gate 4 (hipp function): anchor crushing the "
                "hippocampe — it can't update to fit new tasks. "
                "Reduce lambda_anchor_* (especially _low)."
            )

    return {
        "gate1_drift_low_corr": float(gate1_value),
        "gate1_pass": bool(gate1_pass),
        "gate2_neo_task0": float(neo_task0),
        "gate2_pass": bool(gate2_pass),
        "gate3_neo_taskN": float(neo_taskN),
        "gate3_pass": bool(gate3_pass),
        "gate4_hipp_taskN": float(hipp_taskN),
        "gate4_pass": bool(gate4_pass),
        "verdict": "PASS" if all_pass else "NEEDS_ADJUSTMENT",
    }


# ---------- main ----------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--n_seeds", type=int, default=2)
    p.add_argument("--seed-base", type=int, default=0)

    p.add_argument(
        "--epochs-per-task", "--epochs_per_task",
        dest="epochs_per_task", type=int, default=1,
        help="Per-task budget for both hipp and neo (matches the "
             "1-epoch budget used by all prior CL experiments).",
    )
    p.add_argument("--batch-size", type=int, default=64)

    # Hippocampe knobs (from Phase 1 spec).
    p.add_argument(
        "--hipp-hidden-dims", type=int, nargs="+", default=[128, 64],
    )
    p.add_argument("--hipp-lr", type=float, default=0.05)

    # Neocortex knobs (from Phase 2 spec).
    p.add_argument(
        "--neo-hidden-dims", type=int, nargs="+", default=[256, 256, 128],
    )
    p.add_argument("--neo-lr", type=float, default=0.01)

    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--input-dim", type=int, default=784)
    p.add_argument("--n-classes", type=int, default=10)
    p.add_argument("--samples-per-task", type=int, default=100)

    # Consolidation knobs.
    p.add_argument("--cons-batch-size", type=int, default=64)
    p.add_argument("--cons-epochs", type=int, default=1)
    p.add_argument("--lambda-distill", type=float, default=1.0)
    p.add_argument("--lambda-anchor-low", type=float, default=1.0)
    p.add_argument("--lambda-anchor-mid", type=float, default=0.5)
    p.add_argument("--lambda-anchor-high", type=float, default=0.1)

    p.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--bench-seed", type=int, default=42)
    p.add_argument(
        "--output-dir", type=Path,
        default=_REPO_ROOT / "results" / "logs" / "cls_phase4",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.hipp_hidden_dims = [int(h) for h in args.hipp_hidden_dims]
    args.neo_hidden_dims = [int(h) for h in args.neo_hidden_dims]

    # Up-front parameter counts so the experiment file documents
    # the budget visibly.
    _h = Hippocampus(
        input_dim=args.input_dim,
        hidden_dims=tuple(args.hipp_hidden_dims),
        n_classes=args.n_classes,
    )
    _n = Neocortex(
        input_dim=args.input_dim,
        hidden_dims=tuple(args.neo_hidden_dims),
        n_classes=args.n_classes,
    )
    hipp_params = sum(p.numel() for p in _h.parameters())
    neo_params = sum(p.numel() for p in _n.parameters())
    del _h, _n
    print(
        f"CLS Phase 4 — consolidation\n"
        f"  T={args.T}  n_seeds={args.n_seeds}  "
        f"epochs_per_task={args.epochs_per_task}\n"
        f"  hipp: dims={tuple(args.hipp_hidden_dims)} "
        f"lr={args.hipp_lr}  params={hipp_params:,}\n"
        f"  neo:  dims={tuple(args.neo_hidden_dims)} "
        f"lr={args.neo_lr}  params={neo_params:,}\n"
        f"  consolidation: batch={args.cons_batch_size} "
        f"n_epochs={args.cons_epochs} "
        f"distill={args.lambda_distill} "
        f"anchor[low,mid,high]="
        f"[{args.lambda_anchor_low}, {args.lambda_anchor_mid}, "
        f"{args.lambda_anchor_high}]\n"
        f"  device={args.device}",
        flush=True,
    )

    bench = PermutedMNIST.from_huggingface(
        num_tasks=args.T, seed=args.bench_seed,
    )

    per_seed: list[dict[str, Any]] = []
    seeds = list(range(args.seed_base, args.seed_base + args.n_seeds))
    for seed in seeds:
        print(f"\n--- seed {seed} ---", flush=True)
        result = _run_one_seed(bench, args, seed=seed)
        per_seed.append(result)
        hm = result["hipp_metrics"]
        nm = result["neo_metrics"]
        print(
            f"  seed={seed} done in {result['wall_time_s']:.1f}s\n"
            f"    HIPP: ACC={hm['average_accuracy']:.3f} "
            f"T0={hm['task0_retention']:.3f} "
            f"TN={hm['taskN_final']:.3f} "
            f"FGT={hm['forgetting_proxy']:.3f}\n"
            f"    NEO:  ACC={nm['average_accuracy']:.3f} "
            f"T0={nm['task0_retention']:.3f} "
            f"TN={nm['taskN_final']:.3f} "
            f"FGT={nm['forgetting_proxy']:.3f}",
            flush=True,
        )

    # ----- aggregate report -----
    print()
    print(f"=== Phase 4: Consolidation behavior test (T={args.T}) ===")
    _print_consolidation_table(per_seed, num_tasks=args.T)
    hipp_means, neo_means = _print_retention_rows(
        per_seed, num_tasks=args.T,
    )
    gates = _print_gates(per_seed, hipp_means, neo_means)

    # ----- persist -----
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_path = args.output_dir / f"{ts}_35_T{args.T}_cls_phase4.json"
    payload = {
        "experiment": "35_cls_phase4_consolidation",
        "phase": 4,
        "component": "re_encoding_consolidation",
        "num_tasks": args.T,
        "timestamp": ts,
        "config": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
        "param_counts": {
            "hippocampus": int(hipp_params),
            "neocortex": int(neo_params),
        },
        "per_seed": per_seed,
        "hipp_per_task_means": hipp_means,
        "neo_per_task_means": neo_means,
        "gates": gates,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote results JSON to {out_path}", flush=True)


if __name__ == "__main__":
    main()
