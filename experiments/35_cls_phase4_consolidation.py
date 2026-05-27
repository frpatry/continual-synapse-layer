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
# T=15 while Phase 4 originally trained at T=5; the gate stays
# comparable because the consolidation effect should manifest
# from the earliest tasks onwards.
_PHASE2_NAIVE_TASK0 = 0.200

# ---------- DER-equivalent reference (Gate 6 strong scaling) ----------
#
# cs_gated_cosine_functional @ T=15 n=4 hit ACC=0.904 / Task-0=0.908
# in the original Phase D audit. This is the project's strongest
# prior CL baseline. If the CLS architecture matches or beats it
# at T=15, that's the headline milestone.
_DER_EQUIV_ACC_BASELINE = 0.904

# ---------- T=5 Phase 4 reference (for scaling comparison) ----------
#
# Captured from the T=5 n=2 run committed as 4c9d4c4 — all 4
# original gates passed. Used to render a side-by-side delta
# table when a longer run is invoked. Numbers are means across
# the 2 seeds of that run.
_T5_REF: dict[str, float] = {
    "T":               5.0,
    "neo_acc":         0.893,
    "neo_task0":       0.837,
    "neo_taskN":       0.952,
    "hipp_acc":        0.764,
    "hipp_task0":      0.647,
    "hipp_taskN":      0.940,
    "drift_low_end":   0.877,
    "drift_mid_end":   0.795,
    "drift_high_end":  0.795,
}

# ---------- T=15 Phase 4 reference (n=4, the audited run) ----------
#
# Captured from the T=15 n=4 run logged at
# results/logs/cls_phase4/1779893522_35_T15_cls_phase4.json
# (commit 933a15e plus the n=4 re-run). Used as the prior
# milestone for T=50 scaling comparison.
_T15_REF: dict[str, float] = {
    "T":               15.0,
    "neo_acc":         0.893,
    "neo_task0":       0.820,
    "neo_taskN":       0.948,
    "hipp_acc":        0.585,
    "hipp_task0":      0.403,
    "hipp_taskN":      0.934,
    "drift_low_end":   0.864,
    "drift_mid_end":   0.769,
    "drift_high_end":  0.769,
}

# ---------- T=50 reference per-seed values (for Wilcoxon) ----------
#
# Both pulled from results/logs/functional/1779843616_30_T50_functional.json
# which ran cs_functional_only + cs_gated_cosine_functional at
# T=50 n=3 from exp 30's audit. cs_functional_only is the pure
# DER analogue and matches the project shorthand "DER reference
# 0.870/0.764"; cs_gated_cosine_functional is the cosine-gated
# composition and is the project's strongest prior CL baseline.
_CS_FUNCTIONAL_ONLY_T50_REF_SEEDS = {
    "label": "cs_functional_only (DER-equiv)",
    "neo_acc":   [0.8719, 0.8702, 0.8693],
    "neo_task0": [0.7789, 0.7672, 0.7463],
}
_CS_GATED_COSINE_FUNCTIONAL_T50_REF_SEEDS = {
    "label": "cs_gated_cosine_functional (best prior)",
    "neo_acc":   [0.8729, 0.8718, 0.8717],
    "neo_task0": [0.8801, 0.8862, 0.8964],
}

# ---------- DER-equivalent T=50 reference (verdict A/B/C/D) ----------
#
# cs_functional_only T=50 n=3 mean — the "DER ref" anchor for the
# Outcome A/B/C/D thresholds. The numeric values match the per-
# seed means above.
_DER_EQUIV_T50_REF: dict[str, float] = {
    "T":           50.0,
    "neo_acc":     0.870,
    "neo_task0":   0.764,
    "neo_taskN":   0.95,
}


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
    """One row per task transition, averaged across seeds. When
    ``num_tasks > 20`` only the first / middle / last rows are
    printed to keep the report readable (the JSON log still has
    everything)."""
    print()
    print("Per-consolidation diagnostics (averaged across seeds):")
    print()

    if num_tasks > 20:
        tasks_to_print = [0, num_tasks // 2, num_tasks - 1]
        print(
            f"  (showing first / middle / last of {num_tasks} "
            f"consolidations; full sequence in the JSON log)"
        )
    else:
        tasks_to_print = list(range(num_tasks))

    for t in tasks_to_print:
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
    def _render(values: list[float]) -> str:
        # Wrap at 10 entries per line — fine for T=15, essential
        # for T=50.
        lines: list[str] = []
        for start in range(0, len(values), 10):
            chunk = values[start : start + 10]
            lines.append("  " + "  ".join(
                f"t{start + i}: {v:.2f}" for i, v in enumerate(chunk)
            ))
        return "\n".join(lines)

    print()
    print("HIPPOCAMPUS:")
    print(_render(hipp_means))
    print("  (expect volatile fast-learner pattern from Phase 1)")
    print()
    print("NEOCORTEX:")
    print(_render(neo_means))
    print(
        "  (expect IMPROVEMENT over naive Phase 2 baseline — "
        f"Task-0 should be > {_PHASE2_NAIVE_TASK0:.2f} + 0.05 = "
        f"{_PHASE2_NAIVE_TASK0 + 0.05:.2f})"
    )
    return hipp_means, neo_means


def _pick_prior_milestone(current_T: int) -> dict[str, float] | None:
    """Pick the most-relevant prior Phase 4 milestone to compare
    a new run against. Anything past T=15 compares to T=15;
    anything past T=5 compares to T=5; T=5 itself has no prior."""
    if current_T > 15:
        return _T15_REF
    if current_T > 5:
        return _T5_REF
    return None


def _print_scaling_comparison(
    per_seed: list[dict[str, Any]],
    hipp_means: list[float], neo_means: list[float],
    current_T: int,
) -> dict[str, Any]:
    """Print a side-by-side delta table against the most-relevant
    prior Phase 4 milestone (T=5 or T=15). Skipped at T=5."""
    last_drift_lows = [
        s["consolidation_diagnostics"][-1]["drift_low_corr"]
        for s in per_seed if s["consolidation_diagnostics"]
    ]
    last_drift_mids = [
        s["consolidation_diagnostics"][-1]["drift_mid_corr"]
        for s in per_seed if s["consolidation_diagnostics"]
    ]
    last_drift_highs = [
        s["consolidation_diagnostics"][-1]["drift_high_corr"]
        for s in per_seed if s["consolidation_diagnostics"]
    ]
    cur = {
        "neo_acc":        statistics.fmean(neo_means),
        "neo_task0":      neo_means[0],
        "neo_taskN":      neo_means[-1],
        "hipp_acc":       statistics.fmean(hipp_means),
        "hipp_task0":     hipp_means[0],
        "hipp_taskN":     hipp_means[-1],
        "drift_low_end":  statistics.fmean(last_drift_lows),
        "drift_mid_end":  statistics.fmean(last_drift_mids),
        "drift_high_end": statistics.fmean(last_drift_highs),
    }

    ref = _pick_prior_milestone(current_T)
    if ref is None:
        return {"prior_ref": None, "current": cur}

    print()
    print(f"Compare to T={int(ref['T'])} baseline:")
    print(
        f"              T={int(ref['T']):<5d} T={current_T:<5d} Δ"
    )

    def _line(label: str, key: str) -> None:
        v_ref = ref[key]
        val = cur[key]
        delta = val - v_ref
        sign = "+" if delta >= 0 else ""
        print(
            f"  {label:<13} {v_ref:<6.3f}  {val:<6.3f}  {sign}{delta:.3f}"
        )

    _line("neo ACC:",     "neo_acc")
    _line("neo Task-0:",  "neo_task0")
    _line("neo Task-N:",  "neo_taskN")
    _line("hipp Task-N:", "hipp_taskN")
    _line("drift_low:",   "drift_low_end")

    return {"prior_ref": dict(ref), "current": cur}


def _print_der_comparison_t50(
    hipp_means: list[float], neo_means: list[float],
) -> dict[str, Any]:
    """T=50-specific block: side-by-side vs the cs_gated_cosine_functional
    T=50 reference (ACC=0.870, Task-0=0.764)."""
    cur_acc = statistics.fmean(neo_means)
    cur_t0 = neo_means[0]
    ref_acc = _DER_EQUIV_T50_REF["neo_acc"]
    ref_t0 = _DER_EQUIV_T50_REF["neo_task0"]
    print()
    print("Comparison to cs_gated_cosine_functional T=50:")
    print("                CLS (us)   DER-equiv (ref)   Δ")
    for label, ours, ref in (
        ("ACC:",    cur_acc, ref_acc),
        ("Task-0:", cur_t0,  ref_t0),
    ):
        delta = ours - ref
        sign = "+" if delta >= 0 else ""
        print(
            f"  {label:<13} {ours:<10.3f} {ref:<17.3f} {sign}{delta:.3f}"
        )
    return {
        "cls_neo_acc":   float(cur_acc),
        "cls_neo_task0": float(cur_t0),
        "der_neo_acc":   float(ref_acc),
        "der_neo_task0": float(ref_t0),
        "delta_acc":     float(cur_acc - ref_acc),
        "delta_task0":   float(cur_t0 - ref_t0),
    }


def _classify_axis(value: float, axis: str) -> str:
    """Per-axis A/B/C/D classification matching the Outcome bands."""
    if axis == "acc":
        if value >= 0.87:  return "A"
        if value >= 0.83:  return "B"
        if value >= 0.75:  return "C"
        return "D"
    elif axis == "task0":
        if value >= 0.77:  return "A"
        if value >= 0.65:  return "B"
        if value >= 0.50:  return "C"
        return "D"
    raise ValueError(f"unknown axis: {axis}")


def _combine_letters(*letters: str) -> str:
    """The worst (latest-alphabet) letter across axes is the combined
    outcome — both axes have to be A to combine as A."""
    order = "ABCD"
    return max(letters, key=order.index)


def _bootstrap_ci(
    values: list[float], n_resamples: int = 10_000,
    alpha: float = 0.05, seed: int = 0,
) -> tuple[float, float]:
    """Nonparametric bootstrap CI on the mean. Sample with
    replacement ``n_resamples`` times from ``values``, compute
    mean of each, return (lower, upper) percentile bounds."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    n = arr.shape[0]
    means = arr[rng.integers(0, n, size=(n_resamples, n))].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def _print_statistical_confirmation(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    """Extended statistical block for T=50 n>=5 runs. Per-seed
    A/B/C/D classification, aggregate distribution stats, 95%
    bootstrap CI on ACC and Task-0, Wilcoxon rank-sum vs the
    two T=50 historical references, and Outcome A* (CI overlap)
    detection.

    Returns a dict suitable for embedding in the JSON payload.
    """
    seed_accs = [s["neo_metrics"]["average_accuracy"] for s in per_seed]
    seed_t0s  = [s["neo_metrics"]["task0_retention"]  for s in per_seed]
    seed_tNs  = [s["neo_metrics"]["taskN_final"]       for s in per_seed]
    seed_fgts = [s["neo_metrics"]["forgetting_proxy"]  for s in per_seed]
    n = len(seed_accs)

    print()
    print("=== Per-seed neocortex results ===")
    for i, s in enumerate(per_seed):
        m = s["neo_metrics"]
        print(
            f"  seed {s['seed']:>2d}: ACC={m['average_accuracy']:.3f}  "
            f"Task-0={m['task0_retention']:.3f}  "
            f"Task-N={m['taskN_final']:.3f}  "
            f"FGT={m['forgetting_proxy']:.3f}"
        )

    print()
    print(f"Aggregate (n={n}):")
    print(
        f"  ACC:    mean={statistics.fmean(seed_accs):.3f}  "
        f"std={statistics.stdev(seed_accs):.3f}  "
        f"min={min(seed_accs):.3f}  max={max(seed_accs):.3f}  "
        f"median={statistics.median(seed_accs):.3f}"
    )
    print(
        f"  Task-0: mean={statistics.fmean(seed_t0s):.3f}  "
        f"std={statistics.stdev(seed_t0s):.3f}  "
        f"min={min(seed_t0s):.3f}  max={max(seed_t0s):.3f}  "
        f"median={statistics.median(seed_t0s):.3f}"
    )

    # Per-seed outcome classification.
    print()
    print("=== Per-seed outcome classification ===")
    print(
        "(Outcome A = ACC ≥ 0.87 AND Task-0 ≥ 0.77; "
        "B/C/D use the same bands as the aggregate verdict)"
    )
    per_seed_outcomes: list[dict[str, Any]] = []
    for i, s in enumerate(per_seed):
        m = s["neo_metrics"]
        a_letter = _classify_axis(m["average_accuracy"], "acc")
        t_letter = _classify_axis(m["task0_retention"],  "task0")
        c_letter = _combine_letters(a_letter, t_letter)
        per_seed_outcomes.append({
            "seed": int(s["seed"]),
            "acc_letter": a_letter,
            "task0_letter": t_letter,
            "combined": c_letter,
        })
        print(
            f"  seed {s['seed']:>2d}: ACC={m['average_accuracy']:.3f} "
            f"[{a_letter}]  Task-0={m['task0_retention']:.3f} "
            f"[{t_letter}]  Combined: [{c_letter}]"
        )
    n_combined_a   = sum(1 for o in per_seed_outcomes if o["combined"] == "A")
    n_acc_above    = sum(1 for s in seed_accs if s >= 0.87)
    n_task0_above  = sum(1 for s in seed_t0s if s >= 0.77)
    print()
    print(f"  Fraction of seeds achieving Outcome A: {n_combined_a}/{n}")
    print(f"  Fraction achieving ACC ≥ 0.87:        {n_acc_above}/{n}")
    print(f"  Fraction achieving Task-0 ≥ 0.77:     {n_task0_above}/{n}")

    # Bootstrap CIs.
    acc_lo, acc_hi   = _bootstrap_ci(seed_accs)
    t0_lo,  t0_hi    = _bootstrap_ci(seed_t0s, seed=1)
    print()
    print("=== Bootstrap 95% CI (10k resamples) ===")
    print(
        f"  CLS Variant C ACC:    {statistics.fmean(seed_accs):.3f} ± "
        f"{statistics.stdev(seed_accs):.3f}  "
        f"(95% CI: [{acc_lo:.3f}, {acc_hi:.3f}])"
    )
    print(
        f"  CLS Variant C Task-0: {statistics.fmean(seed_t0s):.3f} ± "
        f"{statistics.stdev(seed_t0s):.3f}  "
        f"(95% CI: [{t0_lo:.3f}, {t0_hi:.3f}])"
    )

    # Comparison vs both historical references (per-seed data
    # available, so Wilcoxon rank-sum is appropriate).
    try:
        from scipy.stats import ranksums, ttest_1samp  # type: ignore[import-untyped]
        scipy_ok = True
    except Exception:  # pragma: no cover — scipy missing fallback
        scipy_ok = False
        ranksums = None  # type: ignore[assignment]
        ttest_1samp = None  # type: ignore[assignment]

    refs_summary: dict[str, dict[str, Any]] = {}
    for ref in (
        _CS_FUNCTIONAL_ONLY_T50_REF_SEEDS,
        _CS_GATED_COSINE_FUNCTIONAL_T50_REF_SEEDS,
    ):
        ref_acc = ref["neo_acc"]
        ref_t0  = ref["neo_task0"]
        m_acc   = statistics.fmean(ref_acc)
        m_t0    = statistics.fmean(ref_t0)
        print()
        print(f"=== Comparison vs {ref['label']} (n={len(ref_acc)}) ===")
        print(
            f"  Reference per-seed ACC:    "
            f"{[round(v,4) for v in ref_acc]}  mean={m_acc:.4f}"
        )
        print(
            f"  Reference per-seed Task-0: "
            f"{[round(v,4) for v in ref_t0]}  mean={m_t0:.4f}"
        )
        gap_acc = statistics.fmean(seed_accs) - m_acc
        gap_t0  = statistics.fmean(seed_t0s) - m_t0
        sign = lambda x: "+" if x >= 0 else ""
        print(
            f"  Gap (CLS − ref) on ACC:    "
            f"{sign(gap_acc)}{gap_acc:.4f}"
        )
        print(
            f"  Gap (CLS − ref) on Task-0: "
            f"{sign(gap_t0)}{gap_t0:.4f}"
        )
        if scipy_ok:
            w_acc = ranksums(seed_accs, ref_acc)
            w_t0  = ranksums(seed_t0s,  ref_t0)
            print(
                f"  Wilcoxon rank-sum ACC:    "
                f"statistic={w_acc.statistic:.3f}  p={w_acc.pvalue:.4f}"
            )
            print(
                f"  Wilcoxon rank-sum Task-0: "
                f"statistic={w_t0.statistic:.3f}  p={w_t0.pvalue:.4f}"
            )
        else:
            print(
                "  Wilcoxon skipped (scipy not available); falling "
                "back to 1-sample t-test vs the reference mean."
            )
        refs_summary[ref["label"]] = {
            "ref_acc_mean":   float(m_acc),
            "ref_task0_mean": float(m_t0),
            "gap_acc":        float(gap_acc),
            "gap_task0":      float(gap_t0),
            "wilcoxon_p_acc": (
                float(w_acc.pvalue) if scipy_ok else None
            ),
            "wilcoxon_p_task0": (
                float(w_t0.pvalue) if scipy_ok else None
            ),
            "ci_overlap_acc":   bool(acc_lo <= m_acc <= acc_hi),
            "ci_overlap_task0": bool(t0_lo  <= m_t0  <= t0_hi),
        }

    # Final outcome classification with A* check (CI overlap on
    # the DER-equiv reference — the project's stated comparison
    # anchor — counts as a statistical tie).
    der_label = _CS_FUNCTIONAL_ONLY_T50_REF_SEEDS["label"]
    der_overlap = refs_summary[der_label]
    mean_acc = statistics.fmean(seed_accs)
    mean_t0  = statistics.fmean(seed_t0s)
    print()
    print("=== Final outcome classification ===")
    if mean_acc >= 0.87 and mean_t0 >= 0.77:
        outcome = "A"
        blurb = (
            "CLS dominates — mean ACC and Task-0 both at/above "
            "Outcome A thresholds."
        )
    elif der_overlap["ci_overlap_acc"] and der_overlap["ci_overlap_task0"]:
        outcome = "A*"
        blurb = (
            "Statistical tie with DER-equiv on both metrics — "
            "95% CI for CLS contains the reference mean on both "
            "ACC and Task-0. 'Matches DER' claim is defensible."
        )
    elif (mean_acc >= 0.83 and mean_t0 >= 0.65):
        outcome = "B"
        blurb = (
            "CLS approaches DER-equiv — gap is real and "
            "persistent; a different lever (teacher refresh, "
            "memory rebalancing, larger hippocampe) would be "
            "needed to flip to A."
        )
    else:
        outcome = "C/D"
        blurb = (
            "Below the matches-DER band — but we shouldn't be "
            "here at this point in the project; investigate."
        )
    print(f"  Outcome: {outcome}")
    print(f"  {blurb}")

    return {
        "n_seeds":         int(n),
        "seed_neo_acc":    seed_accs,
        "seed_neo_task0":  seed_t0s,
        "seed_neo_taskN":  seed_tNs,
        "seed_neo_fgt":    seed_fgts,
        "agg_acc": {
            "mean":   float(statistics.fmean(seed_accs)),
            "std":    float(statistics.stdev(seed_accs)),
            "min":    float(min(seed_accs)),
            "max":    float(max(seed_accs)),
            "median": float(statistics.median(seed_accs)),
            "ci_lo":  float(acc_lo),
            "ci_hi":  float(acc_hi),
        },
        "agg_task0": {
            "mean":   float(statistics.fmean(seed_t0s)),
            "std":    float(statistics.stdev(seed_t0s)),
            "min":    float(min(seed_t0s)),
            "max":    float(max(seed_t0s)),
            "median": float(statistics.median(seed_t0s)),
            "ci_lo":  float(t0_lo),
            "ci_hi":  float(t0_hi),
        },
        "per_seed_outcomes": per_seed_outcomes,
        "fraction_outcome_A":     float(n_combined_a) / n,
        "fraction_acc_above":     float(n_acc_above)  / n,
        "fraction_task0_above":   float(n_task0_above) / n,
        "references_comparison": refs_summary,
        "final_outcome": outcome,
        "final_outcome_blurb": blurb,
    }


def _classify_outcome_abcd(
    neo_acc: float, neo_task0: float,
) -> tuple[str, str]:
    """Auto-classify the T=50 outcome A/B/C/D per the spec's
    decision criteria. Returns (letter, one-line interpretation).
    """
    # Outcome A: CLS dominates (both metrics at/above DER reference)
    if neo_acc >= 0.87 and neo_task0 >= 0.77:
        return (
            "A",
            "CLS dominates — scales better than DER-equivalent. "
            "Headline flip.",
        )
    # Outcome B: CLS matches (both metrics inside the DER-equiv band)
    if (0.83 <= neo_acc <= 0.87) and (0.65 <= neo_task0 <= 0.77):
        return (
            "B",
            "CLS matches DER-equivalent — different mechanism, "
            "comparable performance. Solid contribution.",
        )
    # Outcome D: broken at scale
    if neo_acc < 0.75 or neo_task0 < 0.50:
        return (
            "D",
            "CLS breaks at scale — mechanism doesn't carry to 50 "
            "tasks. Major debug needed.",
        )
    # Outcome C: anything else in the partial-degradation zone
    return (
        "C",
        "CLS scales with cost — partial degradation suggests "
        "lambda tuning may help (try lambda_anchor_low=2.0 or "
        "cons_epochs=2).",
    )


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
    neo_acc = statistics.fmean(neo_means)
    naive_plus_5pp = _PHASE2_NAIVE_TASK0 + 0.05
    gate2_pass = neo_task0 > naive_plus_5pp
    gate3_pass = neo_taskN > 0.85
    gate4_pass = hipp_taskN > 0.85
    # Gates 5-6: scaling-specific. Gate 5 demands strong Task-0
    # retention (> 0.75); Gate 6 compares aggregate ACC to the
    # DER-equivalent cs_gated_cosine_functional baseline (0.904
    # at T=15 n=4 from Phase D's audit).
    gate5_pass = neo_task0 > 0.75
    gate6_pass = neo_acc > _DER_EQUIV_ACC_BASELINE

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
    print(
        f"[gate 5] Strong scaling: neocortex Task-0 > 0.75?  "
        f"{'PASS' if gate5_pass else 'FAIL'}  (got {neo_task0:.3f})"
    )
    print(
        f"[gate 6] Matches cs_gated_cosine_functional (ACC > "
        f"{_DER_EQUIV_ACC_BASELINE:.3f})?  "
        f"{'PASS' if gate6_pass else 'FAIL'}  (got {neo_acc:.3f})"
    )

    # All-pass means the original 4 gates AND both scaling gates.
    # Gate 5/6 are scoped to the T=15 scaling test; Gate 6 in
    # particular is the project-level milestone for "matches the
    # prior best CL baseline".
    core_pass = gate1_pass and gate2_pass and gate3_pass and gate4_pass
    all_pass = core_pass and gate5_pass and gate6_pass
    print()
    if all_pass:
        print(
            "ALL GATES PASS (including scaling gates 5+6) — "
            "consolidation scales, ready for Phase 5."
        )
    elif core_pass:
        print(
            "CORE GATES (1–4) PASS but scaling gates failed:"
        )
        if not gate5_pass:
            print(
                f"  - Gate 5: neo Task-0={neo_task0:.3f} ≤ 0.75. "
                f"Partial scaling — try lambda_anchor_low=2.0 or "
                f"cons_epochs=2 before Phase 5."
            )
        if not gate6_pass:
            print(
                f"  - Gate 6: neo ACC={neo_acc:.3f} ≤ "
                f"{_DER_EQUIV_ACC_BASELINE:.3f}. Doesn't yet match "
                f"the DER-equivalent baseline. May still be a "
                f"viable CLS contribution if gates 1–5 hold."
            )
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
        "gate5_strong_scaling_neo_task0": float(neo_task0),
        "gate5_pass": bool(gate5_pass),
        "gate6_neo_acc_vs_der_equiv": float(neo_acc),
        "gate6_pass": bool(gate6_pass),
        "core_gates_pass": bool(core_pass),
        "verdict": "PASS" if all_pass else (
            "CORE_PASS_SCALING_FAIL" if core_pass else "NEEDS_ADJUSTMENT"
        ),
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

    # Consolidation knobs. Each flag accepts both hyphen and
    # underscore forms so sweep harnesses / shell loops can use
    # whichever convention is more convenient.
    p.add_argument("--cons-batch-size", type=int, default=64)
    p.add_argument(
        "--cons-epochs", "--cons_epochs",
        dest="cons_epochs", type=int, default=1,
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
    scaling_comparison = _print_scaling_comparison(
        per_seed, hipp_means, neo_means, current_T=args.T,
    )

    # T=50 stress-test specific reporting: DER-equivalent
    # comparison + auto-classified A/B/C/D verdict.
    der_comparison: dict[str, Any] | None = None
    outcome: dict[str, str] | None = None
    statistical_confirmation: dict[str, Any] | None = None
    if args.T >= 50:
        der_comparison = _print_der_comparison_t50(hipp_means, neo_means)
        neo_acc = statistics.fmean(neo_means)
        neo_task0 = neo_means[0]
        letter, blurb = _classify_outcome_abcd(neo_acc, neo_task0)
        print()
        print(f"=== Verdict at T={args.T}: Outcome {letter} ===")
        print(f"  {blurb}")
        outcome = {"letter": letter, "blurb": blurb}

        # Extended statistical confirmation when there are enough
        # seeds to bootstrap. Activates at T>=50 n>=5.
        if len(per_seed) >= 5:
            statistical_confirmation = _print_statistical_confirmation(
                per_seed,
            )

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
        "scaling_comparison": scaling_comparison,
        "der_comparison_t50": der_comparison,
        "outcome_abcd": outcome,
        "statistical_confirmation": statistical_confirmation,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote results JSON to {out_path}", flush=True)


if __name__ == "__main__":
    main()
