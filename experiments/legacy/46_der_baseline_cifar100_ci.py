"""Experiment 46 — Pure DER baseline on Split-CIFAR-100 CI.

Apples-to-apples comparator for exp 45 (CLS-CI v2 on CIFAR). The
only architectural difference is the absence of the dual-system /
multi-level anchor / consolidation phase: this script trains a
single :class:`CIFARNeocortex` (Reduced ResNet-18) with pure DER
(Buzzega et al., 2020) — at every training step, in addition to
the current-task cross-entropy, MSE on the stored logits of a
replayed batch is added to the loss.

Pure DER, not DER++: only the MSE-on-logits replay term is used;
the optional hard-label cross-entropy on the replay batch (which
turns this into DER++) is deliberately omitted.

All other knobs (optimizer, lr, momentum, weight decay, grad
clip, batch size, num_tasks, samples_per_task, max_memory,
replay_batch_size, augmentation pipeline, eval cadence, output
format) are intentionally identical to exp 45 so the comparison
isolates the architectural difference.

Run from the repo root::

    # Apples-to-apples pilot:
    python experiments/46_der_baseline_cifar100_ci.py \\
        --num_tasks 10 --epochs_per_task 30 \\
        --batch_size 128 --n_seeds 3

    # Smoke (train first 2 tasks, ~1-2 min on L4):
    python experiments/46_der_baseline_cifar100_ci.py --smoke
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from continual_synapse.architectures import CIFARNeocortex  # noqa: E402
from continual_synapse.benchmarks import (  # noqa: E402
    SplitCIFAR100ClassIncremental,
)
from continual_synapse.evaluation.runner import set_seed  # noqa: E402


# ---------- device + loader helpers (copy from exp 45 for self-contained) ----------


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _user_passed(*names: str) -> bool:
    flat: set[str] = set()
    for n in names:
        for form in (n, n.replace("_", "-")):
            flat.add(f"--{form}")
    for arg in sys.argv[1:]:
        head = arg.split("=", 1)[0]
        if head in flat:
            return True
    return False


def _loader_kwargs(args: argparse.Namespace) -> dict:
    pin = (
        args.pin_memory
        if args.pin_memory is not None
        else (args.device == "cuda")
    )
    return {
        "num_workers": int(args.num_workers),
        "pin_memory":  bool(pin),
        "persistent_workers": args.num_workers > 0,
    }


# ---------- DER memory ----------


class _DERMemoryCIFAR:
    """Pure DER buffer: stores ``(input, raw_logits, label)`` per
    entry. Logits are kept raw (pre-softmax) — that's what the
    paper's MSE consistency loss operates on. Labels are kept for
    diagnostics only (this is pure DER, no hard-label replay).

    Reservoir sampling caps the buffer at ``max_total``. Records
    are written at task end (matching the storage policy used by
    exp 45 for apples-to-apples comparison — not the running
    reservoir over the whole training stream that the DER paper
    uses; with ``samples_per_task=100`` × ``num_tasks=10`` = 1000
    entries we never hit the 5000 cap, so the two policies
    coincide here).
    """

    def __init__(
        self,
        max_total: int = 5000,
        num_classes: int = 100,
        rng_seed: int | None = None,
    ) -> None:
        if max_total <= 0:
            raise ValueError(f"max_total must be positive, got {max_total}")
        self.max_total = int(max_total)
        self.num_classes = int(num_classes)
        self.n_seen = 0
        self.inputs: list[Tensor] = []
        self.logits: list[Tensor] = []
        self.labels: list[int]    = []
        self._rng = random.Random(rng_seed) if rng_seed is not None else random

    @torch.no_grad()
    def record_batch(
        self, inputs: Tensor, labels: Tensor, model: nn.Module,
    ) -> int:
        if inputs.ndim != 4 or inputs.shape[1:] != (3, 32, 32):
            raise ValueError(
                f"inputs must be (B, 3, 32, 32); got {tuple(inputs.shape)}"
            )
        if labels.shape[0] != inputs.shape[0]:
            raise ValueError("labels must match inputs on N")

        device = next(model.parameters()).device
        x = inputs.to(device)
        if x.dtype == torch.uint8:
            x = x.to(torch.float32) / 255.0

        # Switch to eval to avoid drifting BN running stats during
        # the snapshot forward.
        was_training = model.training
        model.eval()
        try:
            logits = model(x).detach().cpu()
        finally:
            model.train(was_training)

        inputs_cpu = inputs.detach().cpu()
        labels_cpu = labels.detach().cpu()

        B = x.shape[0]
        n_added = 0
        for i in range(B):
            self.n_seen += 1
            entry = {
                "input":  inputs_cpu[i].clone(),
                "logits": logits[i].clone(),
                "label":  int(labels_cpu[i].item()),
            }
            if len(self.inputs) < self.max_total:
                self._append(entry)
                n_added += 1
            else:
                j = self._rng.randrange(self.n_seen)
                if j < self.max_total:
                    self._replace(j, entry)
                    n_added += 1
        return n_added

    def _append(self, entry: dict) -> None:
        self.inputs.append(entry["input"])
        self.logits.append(entry["logits"])
        self.labels.append(entry["label"])

    def _replace(self, idx: int, entry: dict) -> None:
        self.inputs[idx] = entry["input"]
        self.logits[idx] = entry["logits"]
        self.labels[idx] = entry["label"]

    def sample_batch(
        self, batch_size: int, device: torch.device | str = "cpu",
    ) -> dict | None:
        if not self.inputs:
            return None
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        n = min(batch_size, len(self.inputs))
        idx = self._rng.sample(range(len(self.inputs)), n)
        device_t = torch.device(device)
        return {
            "inputs": torch.stack(
                [self.inputs[i] for i in idx]
            ).to(device_t),
            "logits": torch.stack(
                [self.logits[i] for i in idx]
            ).to(device_t),
            "labels": torch.tensor(
                [self.labels[i] for i in idx],
                dtype=torch.long, device=device_t,
            ),
        }

    def __len__(self) -> int:
        return len(self.inputs)


# ---------- training step ----------


def train_step_der(
    model: nn.Module,
    x_batch: Tensor, y_batch: Tensor,
    memory: _DERMemoryCIFAR,
    optimizer: torch.optim.Optimizer,
    *,
    lambda_replay: float,
    replay_batch_size: int,
    grad_clip: float | None,
) -> dict[str, float]:
    """Pure DER step: CE on current task + λ · MSE(current logits,
    stored logits) on a replayed batch. No hard-label loss on
    replay (that would be DER++)."""
    optimizer.zero_grad()
    logits = model(x_batch)
    loss_current = F.cross_entropy(logits, y_batch)

    loss_replay = torch.zeros((), device=x_batch.device)
    if len(memory) > 0:
        replay = memory.sample_batch(replay_batch_size, device=x_batch.device)
        if replay is not None:
            current_replay_logits = model(replay["inputs"])
            loss_replay = F.mse_loss(current_replay_logits, replay["logits"])

    total = loss_current + lambda_replay * loss_replay
    total.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
    optimizer.step()
    return {
        "current_loss": float(loss_current.item()),
        "replay_loss":  float(loss_replay.item()),
    }


# ---------- eval ----------


@torch.no_grad()
def evaluate_model(
    model: nn.Module, loader: DataLoader, device: torch.device,
    num_classes: int = 100,
) -> dict[str, Any]:
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
) -> tuple[Tensor, Tensor]:
    ds = bench.task_train_dataset(task_id, augment=True)
    n_pool = len(ds)
    n = min(samples_per_task, n_pool)
    idx = torch.randperm(n_pool, generator=generator)[:n].tolist()
    xs = torch.stack([ds[i][0] for i in idx])
    ys = torch.tensor([ds[i][1] for i in idx], dtype=torch.long)
    return xs, ys


# ---------- checkpointing (mirrors exp 45) ----------


def _checkpoint_path(
    args: argparse.Namespace, seed: int, task_id: int,
) -> Path:
    return Path(args.checkpoint_dir) / (
        f"{args.config_name}_seed{seed}_task{task_id}.pt"
    )


def _save_task_checkpoint(
    args: argparse.Namespace, *,
    seed: int, task_id: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    memory: _DERMemoryCIFAR,
    classes_seen_so_far: list[int],
    results_per_task: list[dict[str, Any]],
) -> Path:
    path = _checkpoint_path(args, seed=seed, task_id=task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "config_name": args.config_name,
        "seed": int(seed),
        "task_id": int(task_id),
        "classes_seen_so_far": list(classes_seen_so_far),
        "results_per_task": results_per_task,
        "model_state_dict": model.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "memory": {
            "max_total":   memory.max_total,
            "num_classes": memory.num_classes,
            "n_seen":      memory.n_seen,
            "inputs":      memory.inputs,
            "logits":      memory.logits,
            "labels":      memory.labels,
        },
        "config": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def _maybe_resume(
    args: argparse.Namespace, *, seed: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    memory: _DERMemoryCIFAR,
    classes_seen_so_far: list[int],
    results_per_task: list[dict[str, Any]],
) -> int:
    if args.no_checkpoint or not args.resume:
        return -1
    latest_task = -1
    latest_path: Path | None = None
    for t in range(args.num_tasks - 1, -1, -1):
        p = _checkpoint_path(args, seed=seed, task_id=t)
        if p.exists():
            latest_task = t
            latest_path = p
            break
    if latest_path is None:
        return -1

    ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
    if ckpt.get("config_name") != args.config_name:
        raise RuntimeError(
            f"Checkpoint at {latest_path} was written by "
            f"config_name={ckpt.get('config_name')!r}, but the "
            f"current run uses config_name={args.config_name!r}."
        )
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    ms = ckpt["memory"]
    memory.n_seen = int(ms["n_seen"])
    memory.inputs = list(ms["inputs"])
    memory.logits = list(ms["logits"])
    memory.labels = list(ms["labels"])
    classes_seen_so_far.clear()
    classes_seen_so_far.extend(ckpt["classes_seen_so_far"])
    results_per_task.clear()
    results_per_task.extend(ckpt["results_per_task"])
    return int(latest_task)


# ---------- per-seed driver ----------


def _run_one_seed(
    bench: SplitCIFAR100ClassIncremental,
    args: argparse.Namespace, seed: int,
) -> dict[str, Any]:
    set_seed(seed)
    device = torch.device(args.device)
    model = CIFARNeocortex(num_classes=args.num_classes).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay,
    )
    memory = _DERMemoryCIFAR(
        max_total=args.max_memory,
        num_classes=args.num_classes,
        rng_seed=seed,
    )

    classes_seen_so_far: list[int] = []
    results_per_task: list[dict[str, Any]] = []
    t_seed = time.time()

    resume_task = _maybe_resume(
        args, seed=seed, model=model, optimizer=optimizer,
        memory=memory, classes_seen_so_far=classes_seen_so_far,
        results_per_task=results_per_task,
    )
    start_task = resume_task + 1 if resume_task >= 0 else 0
    if start_task > 0:
        print(
            f"  seed={seed}: resumed from checkpoint at task "
            f"{resume_task}; starting at task {start_task}",
            flush=True,
        )

    loader_kwargs = _loader_kwargs(args)
    for task_id in range(start_task, args.num_tasks):
        task_classes = bench.task_classes(task_id)
        classes_seen_so_far.extend(int(c) for c in task_classes)
        train_loader = bench.get_task_train_loader(
            task_id, batch_size=args.batch_size, shuffle=True,
            **loader_kwargs,
        )

        t_task = time.time()
        last_diag: dict[str, float] = {}
        for epoch in range(args.epochs_per_task):
            per_batch: dict[str, list[float]] = {
                "current_loss": [], "replay_loss": [],
            }
            for x, y in train_loader:
                x = x.to(device); y = y.to(device)
                step = train_step_der(
                    model, x, y, memory, optimizer,
                    lambda_replay=args.lambda_replay,
                    replay_batch_size=args.replay_batch_size,
                    grad_clip=(
                        args.grad_clip if args.grad_clip > 0 else None
                    ),
                )
                for k, v in step.items():
                    per_batch[k].append(v)
            last_diag = {
                k: float(statistics.fmean(vs)) if vs else float("nan")
                for k, vs in per_batch.items()
            }
            if args.verbose:
                print(
                    f"    seed={seed} task={task_id} epoch={epoch}  "
                    f"current={last_diag['current_loss']:.3f}  "
                    f"replay={last_diag['replay_loss']:.3f}",
                    flush=True,
                )

        # End-of-task storage (same policy as exp 45 for apples-to-
        # apples: sample 100 random task inputs, snapshot their
        # logits at task end, add to memory).
        gen = torch.Generator()
        gen.manual_seed(int(seed * 1009 + task_id))
        sample_x, sample_y = _sample_from_task_for_storage(
            bench, task_id, args.samples_per_task, generator=gen,
        )
        n_added = memory.record_batch(sample_x, sample_y, model)

        # Eval on every class seen so far.
        eval_loader = bench.get_eval_loader(
            up_to_task=task_id, batch_size=args.eval_batch_size,
            **loader_kwargs,
        )
        eval_out = evaluate_model(
            model, eval_loader, device, args.num_classes,
        )

        task_result = {
            "task_id": int(task_id),
            "wall_time_s": float(time.time() - t_task),
            "train_loss_last_epoch": last_diag,
            "memory_size": int(len(memory)),
            "n_added_at_end": int(n_added),
            "eval_acc": float(eval_out["acc"]),
            "per_class_eval": eval_out["per_class"],
        }
        results_per_task.append(task_result)
        print(
            f"  seed={seed} task={task_id} "
            f"({task_classes[0]}-{task_classes[-1]})  "
            f"in {task_result['wall_time_s']:.1f}s | "
            f"|mem|={len(memory)}  "
            f"current={last_diag['current_loss']:.3f} "
            f"replay={last_diag['replay_loss']:.3f}  "
            f"ACC={eval_out['acc']:.3f}",
            flush=True,
        )

        if not args.no_checkpoint:
            _save_task_checkpoint(
                args, seed=seed, task_id=task_id,
                model=model, optimizer=optimizer, memory=memory,
                classes_seen_so_far=classes_seen_so_far,
                results_per_task=results_per_task,
            )

    return {
        "seed": int(seed),
        "per_task": results_per_task,
        "final_acc": results_per_task[-1]["eval_acc"],
        "wall_time_s": float(time.time() - t_seed),
    }


# ---------- main ----------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    p.add_argument("--smoke", action="store_true")
    p.add_argument("--num_tasks", type=int, default=10)
    p.add_argument("--bench_num_tasks", type=int, default=10)
    p.add_argument("--epochs_per_task", type=int, default=30)
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--seed_base", type=int, default=0)

    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--eval_batch_size", type=int, default=256)
    p.add_argument("--num_classes", type=int, default=100)

    # Optimizer — match exp 45 / CLS-CI v2 defaults for apples-to-
    # apples comparison.
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Memory + replay (same numbers as exp 45's CLS-CI v2 config).
    p.add_argument("--samples_per_task", type=int, default=100)
    p.add_argument("--max_memory", type=int, default=5000)
    p.add_argument("--replay_batch_size", type=int, default=32)
    p.add_argument(
        "--lambda_replay", type=float, default=1.0,
        help="Weight on the MSE-on-logits replay loss. Matches "
             "the exp 45 --lambda_replay_inline=1.0 default for "
             "fair comparison.",
    )

    p.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument(
        "--pin_memory", type=lambda s: s.lower() in ("1", "true", "yes"),
        default=None,
    )
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--output-dir", type=Path,
        default=_REPO_ROOT / "results" / "logs" / "split_cifar100_ci",
    )
    p.add_argument(
        "--config_name", type=str, default="der_baseline_cifar",
    )
    p.add_argument(
        "--checkpoint_dir", type=Path,
        default=_REPO_ROOT / "results" / "checkpoints" / "cifar100_ci",
    )
    p.add_argument("--no_checkpoint", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke:
        if not _user_passed("num_tasks"):
            args.num_tasks = 2
        if not _user_passed("epochs_per_task"):
            args.epochs_per_task = 2
        if not _user_passed("n_seeds"):
            args.n_seeds = 1
        args.verbose = True
    if not _user_passed("device"):
        args.device = str(get_device())

    if args.num_tasks > args.bench_num_tasks:
        raise ValueError(
            f"--num_tasks ({args.num_tasks}) cannot exceed "
            f"--bench_num_tasks ({args.bench_num_tasks})."
        )

    mode = "SMOKE" if args.smoke else "PILOT"
    print(
        f"Phase 5.6 — DER baseline on Split-CIFAR-100 CI [{mode}]\n"
        f"  T={args.num_tasks}/{args.bench_num_tasks}  "
        f"epochs_per_task={args.epochs_per_task}  "
        f"n_seeds={args.n_seeds}\n"
        f"  optimizer: SGD lr={args.lr}  momentum={args.momentum}  "
        f"wd={args.weight_decay}  grad_clip={args.grad_clip}\n"
        f"  memory: samples_per_task={args.samples_per_task}  "
        f"max={args.max_memory}\n"
        f"  replay: batch={args.replay_batch_size}  "
        f"λ_replay={args.lambda_replay}  (pure DER: MSE on logits "
        f"only, no hard-label CE on replay)\n"
        f"  device={args.device}",
        flush=True,
    )

    t_load = time.time()
    bench = SplitCIFAR100ClassIncremental.from_huggingface(
        num_tasks=args.bench_num_tasks,
    )
    print(f"Loaded benchmark in {time.time() - t_load:.1f}s.\n")

    per_seed: list[dict[str, Any]] = []
    for seed in range(args.seed_base, args.seed_base + args.n_seeds):
        print(f"--- seed {seed} ---", flush=True)
        per_seed.append(_run_one_seed(bench, args, seed=seed))

    final_acc = [s["final_acc"] for s in per_seed]
    print()
    print(f"=== DER baseline aggregate (n={len(per_seed)}) ===")
    print(
        f"  Final ACC: mean={statistics.fmean(final_acc):.3f}  "
        f"std={statistics.stdev(final_acc) if len(final_acc)>1 else 0:.3f}"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    suffix = "_smoke" if args.smoke else ""
    out_path = (
        args.output_dir / f"{ts}_46_der_baseline_cifar{suffix}.json"
    )
    with out_path.open("w") as f:
        json.dump({
            "experiment": "46_der_baseline_cifar100_ci",
            "phase": "5.6",
            "mode": mode,
            "method": "der_pure",
            "timestamp": ts,
            "config": {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in vars(args).items()
            },
            "per_seed": per_seed,
            "summary": {
                "final_acc_mean": statistics.fmean(final_acc),
                "final_acc_std":  (
                    statistics.stdev(final_acc) if len(final_acc) > 1 else 0.0
                ),
            },
        }, f, indent=2)
    print(f"\nWrote results JSON to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
