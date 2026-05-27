"""Experiment 41 — Phase 5.5.6: CLS-CI v2 with interleaved replay.

Fix for the Phase 5.5.4 failure (NEO ACC=0.110, worse than naive)
on Split-MNIST class-incremental. Diagnosis from exp 39's
per-task progression: per-task training on a disjoint class set
rewrites class representations BEFORE the separate consolidation
phase can intervene, producing drift_low_corr collapse from 1.000
→ 0.508 in one task transition.

Fix (biologically motivated): instead of training the neocortex
on just the current task's classes and waiting for a separate
consolidation phase, interleave a memory-replay loss INSIDE every
per-task training batch. Continuous hippocampe→neocortex
consolidation during active experience, with the separate Phase-4
consolidation phase preserved as "deep consolidation" (analogous
to sleep amplifying — not replacing — waking-state interleaving).

What's new vs exp 39:
- Neocortex per-task training step now computes
  L = CE(neo(x), y) + λ_replay_inline * masked_kl(neo(replay_x),
                                                  replay_soft, replay_mask)
  when memory is non-empty.
- CIMultiLevelMemory.sample_batch returns the triple
  (inputs, soft_targets, classes_seen_mask) drawn uniformly at
  random.

What's unchanged:
- Hippocampe per-task training (no replay there — hipp is still
  the volatile fast learner).
- Multi-level features stored at task end.
- Separate consolidation phase (anchor on hipp + task + masked
  KL on neo), cons_epochs=2, all Variant C lambdas.

Run from the repo root::

    python experiments/41_split_mnist_ci_cls_v2.py
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


# ---------- models (matches exp 39) ----------


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


# ---------- memory ----------


class CIMultiLevelMemory:
    """Phase 5.5.4's memory + a ``sample_batch`` method that returns
    the triple needed for interleaved replay
    ``(inputs, soft_targets, classes_seen_mask)``.
    """

    def __init__(
        self, samples_per_task: int = 100, n_classes: int = 10,
        rng_seed: int | None = None,
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
        self.classes_seen_mask: list[Tensor] = []
        import random as _random
        self._rng = (
            _random.Random(rng_seed) if rng_seed is not None else _random
        )

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
                "Hippocampe encoder has fewer than 2 ReLU layers."
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

    def sample_batch(
        self, batch_size: int, device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor] | None:
        """Uniform random sample for the interleaved-replay channel.

        Returns ``(inputs, soft_targets, classes_seen_mask)`` stacked
        and moved to ``device``, or ``None`` when memory is empty
        (the caller's contract: skip the replay loss in that case)."""
        if not self.inputs:
            return None
        n = min(batch_size, len(self.inputs))
        idx = self._rng.sample(range(len(self.inputs)), n)
        x   = torch.stack([self.inputs[i]            for i in idx]).to(device)
        s   = torch.stack([self.soft_targets[i]      for i in idx]).to(device)
        m   = torch.stack([self.classes_seen_mask[i] for i in idx]).to(device)
        return x, s, m

    def __len__(self) -> int:
        return len(self.inputs)

    def per_task_counts(self) -> dict[int, int]:
        return dict(Counter(self.task_ids))


# ---------- masked KL distillation (unchanged from exp 39) ----------


def masked_kl(
    neo_logits: Tensor,
    stored_soft: Tensor,
    classes_seen_mask: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    mask_f = classes_seen_mask.to(stored_soft.dtype)
    masked = stored_soft * mask_f
    masked = masked / (masked.sum(dim=-1, keepdim=True) + eps)
    log_neo = F.log_softmax(neo_logits, dim=-1)
    elem = masked * (masked.clamp(min=eps).log() - log_neo)
    elem = elem * mask_f
    return (elem.sum(dim=-1)).mean()


# ---------- consolidation (separate, "deep" phase — unchanged) ----------


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
            stored_low  = torch.stack(
                [memory.low_features[int(j)] for j in batch_idx]
            ).to(device)
            stored_mid  = torch.stack(
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


def _train_hipp_one_task(
    hipp: Hippocampus, optimizer: torch.optim.Optimizer,
    loader: DataLoader, epochs: int, device: torch.device,
) -> float:
    """Hippocampe per-task training is unchanged: pure CE on the
    current task's 2 classes. The hipp is still the volatile
    fast learner; its drift will be cleaned up in the separate
    consolidation phase by the multi-level anchor."""
    hipp.train()
    losses: list[float] = []
    for _ in range(epochs):
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(hipp(x), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
    return statistics.fmean(losses) if losses else float("nan")


def _train_neo_one_task_with_replay(
    neo: Neocortex, optimizer: torch.optim.Optimizer,
    loader: DataLoader, epochs: int, device: torch.device,
    memory: CIMultiLevelMemory, replay_batch_size: int,
    lambda_replay_inline: float,
) -> tuple[float, float]:
    """Neocortex per-task training with interleaved replay.

    For every batch of current-task data we additionally sample a
    replay batch from memory and add ``lambda_replay_inline *
    masked_kl(neo(replay_x), replay_soft, replay_mask)`` to the
    task loss. This is the Phase-5.5.6 fix: continuous replay
    during active experience prevents the per-task gradient from
    catastrophically rewriting class representations.
    Returns mean (task_loss, replay_loss) over batches."""
    neo.train()
    task_losses: list[float] = []
    replay_losses: list[float] = []
    for _ in range(epochs):
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            optimizer.zero_grad()
            current_logits = neo(x)
            loss_task = F.cross_entropy(current_logits, y)
            loss_replay = torch.zeros((), device=device)
            replay = memory.sample_batch(replay_batch_size, device=device)
            if replay is not None:
                rx, rs, rm = replay
                loss_replay = masked_kl(neo(rx), rs, rm)
            (loss_task + lambda_replay_inline * loss_replay).backward()
            optimizer.step()
            task_losses.append(float(loss_task.item()))
            replay_losses.append(float(loss_replay.item()))
    return (
        statistics.fmean(task_losses) if task_losses else float("nan"),
        statistics.fmean(replay_losses) if replay_losses else float("nan"),
    )


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
    neo = Neocortex(
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
        rng_seed=seed,
    )
    classes_seen: set[int] = set()

    consolidation_diag: list[dict[str, Any]] = []
    per_task_full_acc_neo:  list[float] = []
    per_task_full_acc_hipp: list[float] = []
    per_task_train_diag:   list[dict[str, float]] = []
    full_test = bench.all_test_dataset()
    t_start = time.time()

    for task_idx, task in enumerate(bench.tasks()):
        for c in task.classes:
            classes_seen.add(int(c))

        loader = DataLoader(
            task.train, batch_size=args.batch_size, shuffle=True,
        )
        hipp_loss = _train_hipp_one_task(
            hipp, hipp_optimizer, loader,
            epochs=args.epochs_per_task, device=device,
        )
        neo_task_loss, neo_replay_loss = _train_neo_one_task_with_replay(
            neo, neo_optimizer, loader,
            epochs=args.epochs_per_task, device=device,
            memory=memory,
            replay_batch_size=args.replay_batch_size,
            lambda_replay_inline=args.lambda_replay_inline,
        )
        per_task_train_diag.append({
            "hipp_loss": float(hipp_loss),
            "neo_task_loss": float(neo_task_loss),
            "neo_replay_loss": float(neo_replay_loss),
        })

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
            f"hipp={hipp_loss:.3f}  "
            f"neo[task,replay]=[{neo_task_loss:.3f},{neo_replay_loss:.3f}]  "
            f"|cons| task={cons['task_losses']:.3f} "
            f"distill={cons['distill_losses']:.3f} "
            f"drift_low={cons['drift_low_corr']:.3f}  "
            f"|mem|={len(memory)}  "
            f"NEO 10cls={per_task_full_acc_neo[-1]:.3f}  "
            f"HIPP 10cls={per_task_full_acc_hipp[-1]:.3f}",
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
        "per_task_train_diagnostics": per_task_train_diag,
        "per_task_full_acc_neo":  per_task_full_acc_neo,
        "per_task_full_acc_hipp": per_task_full_acc_hipp,
        "neo_final_acc":  final_neo["acc"],
        "hipp_final_acc": final_hipp["acc"],
        "neo_per_class_final":  final_neo["per_class"],
        "hipp_per_class_final": final_hipp["per_class"],
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

    # Interleaved replay knobs.
    p.add_argument(
        "--replay-batch-size", "--replay_batch_size",
        dest="replay_batch_size", type=int, default=64,
        help="How many memory entries to draw per neo training "
             "batch for the interleaved-replay channel.",
    )
    p.add_argument(
        "--lambda-replay-inline", "--lambda_replay_inline",
        dest="lambda_replay_inline", type=float, default=1.0,
        help="Weight on the interleaved masked-KL replay loss "
             "during per-task neocortex training. Start at 1.0 "
             "per the Phase 5.5.6 spec.",
    )

    # Separate-phase consolidation knobs (unchanged Variant C).
    p.add_argument("--cons-batch-size", type=int, default=64)
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
        f"Phase 5.5.6 — CLS-CI v2 (interleaved replay) "
        f"on Split-MNIST class-incremental\n"
        f"  n_seeds={args.n_seeds}  epochs_per_task={args.epochs_per_task}\n"
        f"  hipp: dims={tuple(args.hipp_hidden_dims)} lr={args.hipp_lr}\n"
        f"  neo:  dims={tuple(args.neo_hidden_dims)} lr={args.neo_lr}\n"
        f"  interleaved replay: batch={args.replay_batch_size} "
        f"λ_replay_inline={args.lambda_replay_inline}\n"
        f"  separate consolidation: batch={args.cons_batch_size} "
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
    print(f"=== CLS-CI v2 (n={len(per_seed)}) ===")
    print(
        f"  NEO  Final ACC: mean={statistics.fmean(neo_accs):.3f}  "
        f"std={statistics.stdev(neo_accs) if len(neo_accs)>1 else 0:.3f}"
    )
    print(
        f"  HIPP Final ACC: mean={statistics.fmean(hipp_accs):.3f}  "
        f"std={statistics.stdev(hipp_accs) if len(hipp_accs)>1 else 0:.3f}"
    )
    print(f"  FGT:            mean={statistics.fmean(fgts):.3f}")
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

    # Compare against the DER baseline auto-loaded from disk.
    der_path = sorted(
        args.output_dir.glob("*38_split_mnist_ci_der.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if der_path:
        with der_path[-1].open() as f:
            der = json.load(f)
        der_acc = der["summary"]["final_acc_mean"]
        print(
            f"\n  Reference: DER baseline ACC={der_acc:.3f} "
            f"(exp 38, T=5 n=3)."
        )
        gap = statistics.fmean(neo_accs) - der_acc
        sign = "+" if gap >= 0 else ""
        print(f"  CLS-CI v2 − DER = {sign}{gap:.3f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_path = args.output_dir / f"{ts}_41_split_mnist_ci_cls_v2.json"
    with out_path.open("w") as f:
        json.dump({
            "experiment": "41_split_mnist_ci_cls_v2",
            "method": "cls_variant_c_interleaved_replay",
            "phase": "5.5.6",
            "timestamp": ts,
            "config": {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in vars(args).items()
            },
            "per_seed": per_seed,
            "summary": {
                "neo_final_acc_mean":   statistics.fmean(neo_accs),
                "neo_final_acc_std":    (
                    statistics.stdev(neo_accs) if len(neo_accs) > 1 else 0.0
                ),
                "hipp_final_acc_mean":  statistics.fmean(hipp_accs),
                "fgt_mean":             statistics.fmean(fgts),
                "per_class_means_neo":  per_class_means_neo,
                "per_class_means_hipp": per_class_means_hipp,
            },
        }, f, indent=2)
    print(f"\nWrote results JSON to {out_path}")


if __name__ == "__main__":
    main()
