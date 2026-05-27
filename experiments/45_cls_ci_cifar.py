"""Experiment 45 — Phase 5.6.3: CIFAR CLS-CI v2 training mechanism.

Adapts the Phase 5.5.6 (Split-MNIST CI v2, exp 41) training recipe
to the CIFAR CNN dual system from Phases 5.6.0–5.6.2. Same two
mechanisms:

1. **Per-task training with interleaved replay.** Every training
   batch updates the hippocampe on the current task's CE loss
   (standalone — hipp remains the volatile fast learner), and
   updates the neocortex on (current-task CE) +
   ``λ_replay_inline * masked_kl(replay)``. The replay batch is
   sampled from CIFARMultiLevelMemory; the KL is masked to the
   class set the hippocampe had seen at storage time.

2. **Separate "deep" consolidation phase after each task.** Two
   full passes over memory: hippocampe gets a multi-level anchor
   loss (MSE between current GAP features and stored GAP
   features); neocortex gets task CE + masked KL distillation.
   This is the "sleep amplifies wake-time replay" phase.

This experiment runs ONLY the smoke test (``--smoke``) by default
— T=2, 5 epochs/task, n=1 — to validate mechanics before the full
T=10 pilot. The full pilot is deliberately *not* run here.

Run from the repo root::

    # Smoke (mandatory before full pilot):
    python experiments/45_cls_ci_cifar.py --smoke

    # Full pilot (DO NOT run without user go-ahead):
    python experiments/45_cls_ci_cifar.py --num_tasks 10 \\
        --epochs_per_task 30 --n_seeds 3
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
from torch.utils.data import DataLoader, Subset

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from continual_synapse.architectures import (  # noqa: E402
    CIFARHippocampus, CIFARNeocortex,
)
from continual_synapse.benchmarks import (  # noqa: E402
    SplitCIFAR100ClassIncremental,
)
from continual_synapse.evaluation.runner import set_seed  # noqa: E402
from continual_synapse.memory import CIFARMultiLevelMemory  # noqa: E402


# ---------- masked KL ----------


def masked_kl_cifar(
    neo_logits: torch.Tensor,
    stored_soft: torch.Tensor,
    classes_seen_per_entry: list[list[int]],
    num_classes: int = 100,
) -> torch.Tensor:
    """KL divergence over the per-entry seen-class subset.

    For each entry, the stored hippocampe soft target is masked
    and renormalised over the class set that hipp had seen at
    storage time, then KL(teacher || student) is computed against
    the neocortex's log-softmax. Returns the batchmean.
    """
    B = neo_logits.shape[0]
    device = neo_logits.device
    mask = torch.zeros(B, num_classes, dtype=torch.bool, device=device)
    for i, classes in enumerate(classes_seen_per_entry):
        if classes:
            idx = torch.as_tensor(classes, dtype=torch.long, device=device)
            mask[i, idx] = True
    mask_f = mask.to(stored_soft.dtype)

    masked_soft = stored_soft * mask_f
    masked_soft = masked_soft / (
        masked_soft.sum(dim=-1, keepdim=True) + 1e-8
    )
    log_neo = F.log_softmax(neo_logits, dim=-1)
    # KL(teacher || student) per entry, masked to seen classes.
    eps = 1e-8
    elem = masked_soft * (masked_soft.clamp(min=eps).log() - log_neo)
    elem = elem * mask_f
    return elem.sum(dim=-1).mean()


# ---------- per-batch interleaved-replay step ----------


def train_step_with_interleaved_replay(
    hippocampus: CIFARHippocampus,
    neocortex: CIFARNeocortex,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    memory: CIFARMultiLevelMemory,
    hipp_optimizer: torch.optim.Optimizer,
    neo_optimizer: torch.optim.Optimizer,
    *,
    lambda_replay_inline: float = 1.0,
    replay_batch_size: int = 32,
    num_classes: int = 100,
) -> dict[str, float]:
    """Single SGD step: hipp = CE on current task; neo = CE on
    current task + interleaved masked KL on a replay batch."""
    # ----- Hippocampe step (CE on current task) -----
    hipp_optimizer.zero_grad()
    hipp_logits = hippocampus(x_batch)
    hipp_loss = F.cross_entropy(hipp_logits, y_batch)
    hipp_loss.backward()
    hipp_optimizer.step()

    # ----- Neocortex step (CE on current + interleaved replay) -----
    neo_optimizer.zero_grad()
    neo_logits = neocortex(x_batch)
    loss_current = F.cross_entropy(neo_logits, y_batch)

    loss_replay = torch.zeros((), device=x_batch.device)
    if len(memory) > 0:
        replay = memory.sample_batch(replay_batch_size, device=x_batch.device)
        if replay is not None:
            replay_x = replay["inputs"]
            replay_soft = replay["soft_targets"]
            replay_classes = replay["classes_seen"]
            replay_neo_logits = neocortex(replay_x)
            loss_replay = masked_kl_cifar(
                replay_neo_logits, replay_soft, replay_classes,
                num_classes=num_classes,
            )

    total_neo_loss = loss_current + lambda_replay_inline * loss_replay
    total_neo_loss.backward()
    neo_optimizer.step()

    return {
        "hipp_loss":        float(hipp_loss.item()),
        "neo_current_loss": float(loss_current.item()),
        "neo_replay_loss":  float(loss_replay.item()),
    }


# ---------- separate consolidation phase ----------


def consolidate_cifar(
    hippocampus: CIFARHippocampus,
    neocortex: CIFARNeocortex,
    memory: CIFARMultiLevelMemory,
    hipp_optimizer: torch.optim.Optimizer,
    neo_optimizer: torch.optim.Optimizer,
    *,
    cons_epochs: int = 2,
    batch_size: int = 64,
    lambda_distill: float = 1.0,
    lambda_anchor_low: float = 1.0,
    lambda_anchor_mid: float = 0.5,
    lambda_anchor_high: float = 0.1,
    num_classes: int = 100,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Deep consolidation phase: ``cons_epochs`` full passes over
    memory.

    Hippocampe: multi-level GAP anchor loss (MSE between fresh GAP
    features and stored GAP features). Drift_*_corr tracks cosine
    similarity per level.

    Neocortex: task CE on the stored labels + masked KL
    distillation against the stored hipp soft target.
    """
    if len(memory) == 0:
        return {}
    if device is None:
        device = next(hippocampus.parameters()).device

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

    n = len(memory)
    for _ in range(cons_epochs):
        indices = torch.randperm(n).tolist()
        for i in range(0, n, batch_size):
            batch_idx = indices[i : i + batch_size]
            batch = memory.sample_batch_by_indices(batch_idx, device=device)

            x = batch["inputs"]
            y = batch["labels"]
            stored_h_low  = batch["hipp_low_gap"]
            stored_h_mid  = batch["hipp_mid_gap"]
            stored_h_high = batch["hipp_high_gap"]
            stored_soft   = batch["soft_targets"]
            classes_seen  = batch["classes_seen"]

            # ----- Hippocampe: anchor loss -----
            hipp_optimizer.zero_grad()
            hipp_feats = hippocampus.features(x)
            new_h_low  = F.adaptive_avg_pool2d(hipp_feats["low"],  1).flatten(1)
            new_h_mid  = F.adaptive_avg_pool2d(hipp_feats["mid"],  1).flatten(1)
            new_h_high = F.adaptive_avg_pool2d(hipp_feats["high"], 1).flatten(1)
            anchor_low  = F.mse_loss(new_h_low,  stored_h_low)
            anchor_mid  = F.mse_loss(new_h_mid,  stored_h_mid)
            anchor_high = F.mse_loss(new_h_high, stored_h_high)
            anchor_total = (
                lambda_anchor_low  * anchor_low
                + lambda_anchor_mid  * anchor_mid
                + lambda_anchor_high * anchor_high
            )
            anchor_total.backward()
            hipp_optimizer.step()

            with torch.no_grad():
                metrics["drift_low_corr"].append(float(
                    F.cosine_similarity(
                        new_h_low.detach(), stored_h_low, dim=-1,
                    ).mean().item()
                ))
                metrics["drift_mid_corr"].append(float(
                    F.cosine_similarity(
                        new_h_mid.detach(), stored_h_mid, dim=-1,
                    ).mean().item()
                ))
                metrics["drift_high_corr"].append(float(
                    F.cosine_similarity(
                        new_h_high.detach(), stored_h_high, dim=-1,
                    ).mean().item()
                ))

            # ----- Neocortex: task + masked KL distillation -----
            neo_optimizer.zero_grad()
            neo_logits = neocortex(x)
            task_loss = F.cross_entropy(neo_logits, y)
            distill = masked_kl_cifar(
                neo_logits, stored_soft, classes_seen,
                num_classes=num_classes,
            )
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


# ---------- eval helpers ----------


@torch.no_grad()
def evaluate_model(
    model: nn.Module, loader: DataLoader, device: torch.device,
    num_classes: int = 100,
) -> dict[str, Any]:
    """Top-1 accuracy on ``loader`` (eval mode). Returns aggregate
    ACC and per-class accuracy."""
    model.eval()
    n_total = 0
    n_correct = 0
    per_class_correct = torch.zeros(num_classes, dtype=torch.long)
    per_class_total   = torch.zeros(num_classes, dtype=torch.long)
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        preds = model(x).argmax(dim=-1)
        n_correct += int((preds == y).sum().item())
        n_total   += int(y.numel())
        for c in range(num_classes):
            mask = (y == c)
            if mask.any():
                per_class_total[c] += int(mask.sum().item())
                per_class_correct[c] += int((preds[mask] == c).sum().item())
    per_class_acc = [
        float(per_class_correct[c].item() / per_class_total[c].item())
        if per_class_total[c] > 0 else float("nan")
        for c in range(num_classes)
    ]
    return {
        "acc": float(n_correct / max(n_total, 1)),
        "per_class": per_class_acc,
        "n_total": n_total,
    }


def _sample_from_task_for_storage(
    bench: SplitCIFAR100ClassIncremental,
    task_id: int, samples_per_task: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pull ``samples_per_task`` random (augmented) samples from the
    task's training set for memory storage. Augmentation is on so
    the stored inputs represent the diverse training distribution."""
    ds = bench.task_train_dataset(task_id, augment=True)
    n_pool = len(ds)
    n = min(samples_per_task, n_pool)
    idx = torch.randperm(n_pool, generator=generator)[:n].tolist()
    xs = torch.stack([ds[i][0] for i in idx])
    ys = torch.tensor([ds[i][1] for i in idx], dtype=torch.long)
    return xs, ys


# ---------- per-seed driver ----------


def _run_one_seed(
    bench: SplitCIFAR100ClassIncremental,
    args: argparse.Namespace, seed: int,
) -> dict[str, Any]:
    set_seed(seed)
    device = torch.device(args.device)
    hipp = CIFARHippocampus(num_classes=args.num_classes).to(device)
    neo  = CIFARNeocortex(num_classes=args.num_classes).to(device)
    hipp_optimizer = torch.optim.SGD(
        hipp.parameters(), lr=args.hipp_lr,
        momentum=args.momentum, weight_decay=args.weight_decay,
    )
    neo_optimizer = torch.optim.SGD(
        neo.parameters(), lr=args.neo_lr,
        momentum=args.momentum, weight_decay=args.weight_decay,
    )
    memory = CIFARMultiLevelMemory(
        max_total=args.max_memory,
        num_classes=args.num_classes,
        rng_seed=seed,
    )

    classes_seen_so_far: list[int] = []
    results_per_task: list[dict[str, Any]] = []
    t_seed = time.time()

    for task_id in range(args.num_tasks):
        task_classes = bench.task_classes(task_id)
        classes_seen_so_far.extend(int(c) for c in task_classes)
        train_loader = bench.get_task_train_loader(
            task_id, batch_size=args.batch_size, shuffle=True,
        )

        # ---- per-task training with interleaved replay ----
        t_task = time.time()
        last_epoch_diag: dict[str, float] = {}
        for epoch in range(args.epochs_per_task):
            per_batch_diag: dict[str, list[float]] = {
                "hipp_loss": [], "neo_current_loss": [], "neo_replay_loss": [],
            }
            for x, y in train_loader:
                x = x.to(device); y = y.to(device)
                step = train_step_with_interleaved_replay(
                    hipp, neo, x, y, memory,
                    hipp_optimizer, neo_optimizer,
                    lambda_replay_inline=args.lambda_replay_inline,
                    replay_batch_size=args.replay_batch_size,
                    num_classes=args.num_classes,
                )
                for k, v in step.items():
                    per_batch_diag[k].append(v)
            last_epoch_diag = {
                k: float(statistics.fmean(vs)) if vs else float("nan")
                for k, vs in per_batch_diag.items()
            }
            if args.verbose:
                print(
                    f"    task={task_id} epoch={epoch}  "
                    f"hipp={last_epoch_diag['hipp_loss']:.3f}  "
                    f"neo_current={last_epoch_diag['neo_current_loss']:.3f}  "
                    f"neo_replay={last_epoch_diag['neo_replay_loss']:.3f}",
                    flush=True,
                )

        # ---- end of task: store samples ----
        gen = torch.Generator()
        gen.manual_seed(int(seed * 1009 + task_id))
        sample_x, sample_y = _sample_from_task_for_storage(
            bench, task_id, args.samples_per_task, generator=gen,
        )
        n_added = memory.record_batch(
            sample_x, sample_y, hipp, neo,
            classes_seen_so_far=list(classes_seen_so_far),
        )

        # ---- separate consolidation phase ----
        cons_metrics = consolidate_cifar(
            hipp, neo, memory,
            hipp_optimizer, neo_optimizer,
            cons_epochs=args.cons_epochs,
            batch_size=args.cons_batch_size,
            lambda_distill=args.lambda_distill,
            lambda_anchor_low=args.lambda_anchor_low,
            lambda_anchor_mid=args.lambda_anchor_mid,
            lambda_anchor_high=args.lambda_anchor_high,
            num_classes=args.num_classes,
            device=device,
        )

        # ---- eval on all classes seen so far ----
        eval_loader = bench.get_eval_loader(
            up_to_task=task_id, batch_size=args.eval_batch_size,
        )
        neo_eval  = evaluate_model(neo,  eval_loader, device, args.num_classes)
        hipp_eval = evaluate_model(hipp, eval_loader, device, args.num_classes)

        task_result = {
            "task_id": int(task_id),
            "wall_time_s": float(time.time() - t_task),
            "train_loss_last_epoch": last_epoch_diag,
            "memory_size": int(len(memory)),
            "n_added_at_end": int(n_added),
            "cons_metrics": cons_metrics,
            "neo_eval_acc": float(neo_eval["acc"]),
            "hipp_eval_acc": float(hipp_eval["acc"]),
            "neo_per_class_eval": neo_eval["per_class"],
            "hipp_per_class_eval": hipp_eval["per_class"],
        }
        results_per_task.append(task_result)
        print(
            f"  seed={seed} task={task_id} "
            f"({task_classes[0]}-{task_classes[-1]})  "
            f"in {task_result['wall_time_s']:.1f}s | "
            f"|mem|={len(memory)}  "
            f"|cons| anchor_low={cons_metrics.get('anchor_low_losses', float('nan')):.3f} "
            f"distill={cons_metrics.get('distill_losses', float('nan')):.3f} "
            f"drift_low={cons_metrics.get('drift_low_corr', float('nan')):.3f}  "
            f"NEO ACC={neo_eval['acc']:.3f}  "
            f"HIPP ACC={hipp_eval['acc']:.3f}",
            flush=True,
        )

    return {
        "seed": int(seed),
        "per_task": results_per_task,
        "final_neo_acc":  results_per_task[-1]["neo_eval_acc"],
        "final_hipp_acc": results_per_task[-1]["hipp_eval_acc"],
        "wall_time_s": float(time.time() - t_seed),
    }


# ---------- main ----------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    # Smoke flag — sets a small-T / few-epoch / single-seed config.
    p.add_argument(
        "--smoke", action="store_true",
        help="Smoke mode: --num_tasks 2 --epochs_per_task 5 --n_seeds 1.",
    )
    p.add_argument("--num_tasks", type=int, default=10)
    p.add_argument("--epochs_per_task", type=int, default=30)
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--seed_base", type=int, default=0)

    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--eval_batch_size", type=int, default=256)
    p.add_argument("--num_classes", type=int, default=100)

    # Optimizer.
    p.add_argument("--hipp_lr", type=float, default=0.1)
    p.add_argument("--neo_lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)

    # Memory.
    p.add_argument("--samples_per_task", type=int, default=100)
    p.add_argument("--max_memory", type=int, default=5000)

    # Interleaved replay.
    p.add_argument("--lambda_replay_inline", type=float, default=1.0)
    p.add_argument("--replay_batch_size", type=int, default=32)

    # Separate consolidation (Variant C lambdas).
    p.add_argument("--cons_epochs", type=int, default=2)
    p.add_argument("--cons_batch_size", type=int, default=64)
    p.add_argument("--lambda_distill", type=float, default=1.0)
    p.add_argument("--lambda_anchor_low", type=float, default=1.0)
    p.add_argument("--lambda_anchor_mid", type=float, default=0.5)
    p.add_argument("--lambda_anchor_high", type=float, default=0.1)

    p.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--output-dir", type=Path,
        default=_REPO_ROOT / "results" / "logs" / "split_cifar100_ci",
    )
    return p.parse_args()


def _user_passed(*names: str) -> bool:
    """Return True if any of these argument names appears in
    ``sys.argv`` (either ``--foo`` or ``--foo=bar`` form, with
    hyphen or underscore). Used to detect when the user has
    explicitly overridden an arg so ``--smoke``'s defaults
    don't silently clobber them."""
    flat: set[str] = set()
    for n in names:
        for form in (n, n.replace("_", "-")):
            flat.add(f"--{form}")
    for arg in sys.argv[1:]:
        head = arg.split("=", 1)[0]
        if head in flat:
            return True
    return False


def main() -> int:
    args = parse_args()
    if args.smoke:
        # Smoke fills in *defaults* for un-passed args only — if
        # the user explicitly passed e.g. --epochs_per_task 2,
        # respect that rather than overriding to 5.
        if not _user_passed("num_tasks"):
            args.num_tasks = 2
        if not _user_passed("epochs_per_task"):
            args.epochs_per_task = 5
        if not _user_passed("n_seeds"):
            args.n_seeds = 1
        args.verbose = True

    mode = "SMOKE" if args.smoke else "PILOT"
    print(
        f"Phase 5.6.3 — CIFAR CLS-CI v2 [{mode}]\n"
        f"  T={args.num_tasks}  epochs_per_task={args.epochs_per_task}  "
        f"n_seeds={args.n_seeds}\n"
        f"  hipp_lr={args.hipp_lr}  neo_lr={args.neo_lr}  "
        f"momentum={args.momentum}  wd={args.weight_decay}\n"
        f"  memory: samples_per_task={args.samples_per_task}  "
        f"max={args.max_memory}\n"
        f"  interleaved: λ_replay_inline={args.lambda_replay_inline}  "
        f"replay_batch={args.replay_batch_size}\n"
        f"  consolidation: cons_epochs={args.cons_epochs}  "
        f"batch={args.cons_batch_size}\n"
        f"    λ_distill={args.lambda_distill}  "
        f"λ_anchor=[{args.lambda_anchor_low}, "
        f"{args.lambda_anchor_mid}, {args.lambda_anchor_high}]\n"
        f"  device={args.device}",
        flush=True,
    )

    t_load = time.time()
    bench = SplitCIFAR100ClassIncremental.from_huggingface(
        num_tasks=args.num_tasks,
    )
    print(f"Loaded benchmark in {time.time() - t_load:.1f}s.\n")

    per_seed: list[dict[str, Any]] = []
    for seed in range(args.seed_base, args.seed_base + args.n_seeds):
        print(f"--- seed {seed} ---", flush=True)
        per_seed.append(_run_one_seed(bench, args, seed=seed))

    # Aggregate summary.
    final_neo  = [s["final_neo_acc"]  for s in per_seed]
    final_hipp = [s["final_hipp_acc"] for s in per_seed]
    print()
    print(f"=== Aggregate (n={len(per_seed)}) ===")
    print(
        f"  NEO  final ACC: mean={statistics.fmean(final_neo):.3f}  "
        f"std={statistics.stdev(final_neo) if len(final_neo)>1 else 0:.3f}"
    )
    print(
        f"  HIPP final ACC: mean={statistics.fmean(final_hipp):.3f}  "
        f"std={statistics.stdev(final_hipp) if len(final_hipp)>1 else 0:.3f}"
    )

    # Smoke-specific verdict (mechanical health, not absolute performance).
    if args.smoke:
        finite = all(
            (v == v)  # NaN check
            for s in per_seed
            for r in s["per_task"]
            for v in (r["neo_eval_acc"], r["hipp_eval_acc"])
        )
        mem_grew = per_seed[0]["per_task"][-1]["memory_size"] >= args.samples_per_task
        random_chance = 1.0 / (args.num_tasks * 10)  # 10 classes per task
        above_random = per_seed[0]["final_neo_acc"] > random_chance * 2
        print()
        print("Smoke verdict:")
        print(f"  No NaN/Inf in eval metrics: {'PASS' if finite else 'FAIL'}")
        print(
            f"  Memory grew (>= {args.samples_per_task} entries): "
            f"{'PASS' if mem_grew else 'FAIL'}"
        )
        print(
            f"  NEO ACC > 2 * chance ({random_chance*2:.3f}): "
            f"{'PASS' if above_random else 'FAIL'}"
        )
        print(
            "  Overall: " +
            ("PASS — ready for full T=10 pilot."
             if (finite and mem_grew and above_random)
             else "FAIL — debug before scaling.")
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    suffix = "_smoke" if args.smoke else ""
    out_path = (
        args.output_dir / f"{ts}_45_cifar_cls_ci{suffix}.json"
    )
    with out_path.open("w") as f:
        json.dump({
            "experiment": "45_cls_ci_cifar",
            "phase": "5.6.3",
            "mode": mode,
            "timestamp": ts,
            "config": {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in vars(args).items()
            },
            "per_seed": per_seed,
            "summary": {
                "final_neo_acc_mean":  statistics.fmean(final_neo),
                "final_neo_acc_std":   (
                    statistics.stdev(final_neo) if len(final_neo) > 1 else 0.0
                ),
                "final_hipp_acc_mean": statistics.fmean(final_hipp),
            },
        }, f, indent=2)
    print(f"\nWrote results JSON to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
