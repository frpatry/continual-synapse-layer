"""Experiment 34 — Phase 3 of the CLS rebuild: multi-level memory.

This is component 3 of 6 in the incremental Complementary Learning
Systems (CLS) build. Here we add the storage infrastructure that
will later feed consolidation, but **NO consolidation yet** — we
just verify the storage mechanism is mechanically correct.

What this phase tests:
- ``MultiLevelMemory`` stores ``(input, low_features, mid_features,
  high_features, soft_target, task_id)`` correctly.
- Soft targets are valid probability distributions (sum to 1, in
  [0,1], no NaN/Inf).
- Feature tensors have the expected shapes coming off the
  hippocampe's encoder.
- Stored features carry *some* class structure — measured by
  silhouette score in the original 64-dim ``high_features`` space
  using ground-truth labels, plus a t-SNE 2D plot for visual
  inspection.

This is intentionally short: ``T=5`` is enough to verify the
mechanics; the consolidation experiments later will use the
full ``T=15`` budget.

Run from the repo root::

    python experiments/34_cls_phase3_storage.py
"""

from __future__ import annotations

import argparse
import json
import math
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


# ---------- model (copied verbatim from exp 32 — keep each phase
# self-contained so the experiment file documents what it ran) ----------


class Hippocampus(nn.Module):
    """Fast learner component of CLS architecture (see exp 32)."""

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


# ---------- memory ----------


class MultiLevelMemory:
    """Hierarchical memory storage for CLS consolidation.

    Stores ``(input, features_at_each_layer, soft_target)`` tuples
    captured from the hippocampe at task-end. The intermediate
    features will later serve as anchors for the consolidation
    consistency loss in Phase 4. Phase 3 only exercises the
    storage path — no read-back, no replay, no gradients flowing
    through stored tensors.

    Storage policy (deliberately simple): random ``samples_per_task``
    inputs per task, no class balancing yet, no compression. Future
    phases will revisit policy choices.
    """

    def __init__(
        self, samples_per_task: int = 100, n_classes: int = 10,
    ) -> None:
        self.samples_per_task = int(samples_per_task)
        self.n_classes = int(n_classes)
        self.inputs: list[Tensor] = []
        # low = output of first ReLU (128-dim for the default
        # hippocampe), mid = output of second ReLU (64-dim),
        # high = final encoder output. For the current 2-hidden-layer
        # hippocampe high == mid by construction; the field exists so
        # later phases can plug in a separate projection head without
        # changing the storage shape.
        self.low_features: list[Tensor] = []
        self.mid_features: list[Tensor] = []
        self.high_features: list[Tensor] = []
        self.soft_targets: list[Tensor] = []
        self.task_ids: list[int] = []

    @torch.no_grad()
    def record_task_end(
        self,
        hippocampus: Hippocampus,
        task_inputs: Tensor,
        task_id: int,
        device: torch.device,
    ) -> int:
        """Sample ``samples_per_task`` inputs from ``task_inputs`` and
        store the hippocampe's per-layer features + softmax output.
        Returns the number of samples stored."""
        n = min(self.samples_per_task, len(task_inputs))
        idx = torch.randperm(len(task_inputs))[:n]
        sampled = task_inputs[idx].to(device)

        # Walk the encoder Sequential layer-by-layer and snapshot
        # the post-ReLU activations. Matches the structure laid out
        # in exp 32: [Linear -> ReLU -> Linear -> ReLU], so we end
        # up with two snapshots: low (after first ReLU, 128-dim)
        # and mid (after second ReLU, 64-dim).
        h = sampled
        layer_outputs: list[Tensor] = []
        for layer in hippocampus.encoder:
            h = layer(h)
            if isinstance(layer, nn.ReLU):
                layer_outputs.append(h.detach().cpu())

        if len(layer_outputs) < 2:
            raise RuntimeError(
                "Expected at least 2 ReLU outputs in the hippocampe "
                f"encoder, got {len(layer_outputs)}. The encoder "
                f"architecture has drifted from the Phase-1 spec."
            )

        low_feat = layer_outputs[0]
        mid_feat = layer_outputs[1]
        # ``h`` after the loop is the final encoder output — same
        # tensor as the last ReLU snapshot for the current
        # architecture, kept separate so future encoders with a
        # post-ReLU projection still feed a well-defined "high"
        # tensor.
        high_feat = h.detach().cpu()

        logits = hippocampus.classifier(h)
        soft = F.softmax(logits, dim=-1).detach().cpu()

        for i in range(n):
            self.inputs.append(sampled[i].detach().cpu())
            self.low_features.append(low_feat[i])
            self.mid_features.append(mid_feat[i])
            self.high_features.append(high_feat[i])
            self.soft_targets.append(soft[i])
            self.task_ids.append(int(task_id))

        return n

    def __len__(self) -> int:
        return len(self.inputs)

    def per_task_counts(self) -> dict[int, int]:
        return dict(Counter(self.task_ids))

    def task_indices(self, task_id: int) -> list[int]:
        return [i for i, t in enumerate(self.task_ids) if t == task_id]


# ---------- training ----------


def _train_hippocampus(
    bench: PermutedMNIST,
    memory: MultiLevelMemory,
    task_labels_storage: dict[int, Tensor],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Train the hippocampe sequentially across ``bench.tasks()``.

    At the end of each task, record samples into ``memory`` and
    stash their ground-truth labels in ``task_labels_storage`` so
    the t-SNE check can compute silhouette by true class. Labels
    aren't stored in ``MultiLevelMemory`` itself because the spec
    deliberately keeps the memory class label-free (the soft
    target is the hippocampe's prediction, not the ground truth).
    """
    set_seed(args.seed)
    model = Hippocampus(
        input_dim=args.input_dim,
        hidden_dims=tuple(args.hidden_dims),
        n_classes=args.n_classes,
    ).to(args.device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
    )

    per_task_train_loss: list[float] = []
    t0 = time.time()
    for task_idx, task in enumerate(bench.tasks()):
        loader = DataLoader(
            task.train, batch_size=args.batch_size, shuffle=True,
        )
        model.train()
        task_losses: list[float] = []
        for _ in range(args.epochs_per_task):
            for x, y in loader:
                x = x.to(args.device)
                y = y.to(args.device)
                optimizer.zero_grad()
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
                task_losses.append(float(loss.item()))
        avg = statistics.fmean(task_losses) if task_losses else float("nan")
        per_task_train_loss.append(avg)

        # End of task: store. We mirror the index-selection so we
        # can pair the stored entries with their ground-truth
        # labels without duplicating the sampling logic.
        pool = task.train.tensors[0]
        labels_pool = task.train.tensors[1]
        n_pool = pool.shape[0]
        gen = torch.Generator()
        gen.manual_seed(int(args.seed * 1009 + task_idx))
        n = min(memory.samples_per_task, n_pool)
        idx = torch.randperm(n_pool, generator=gen)[:n]

        # Inline the storage walk so we use the same indices we'll
        # record for the labels. (Calling record_task_end would
        # re-randomise without exposing the indices.)
        sampled = pool[idx].to(args.device)
        h = sampled
        layer_outputs: list[Tensor] = []
        with torch.no_grad():
            for layer in model.encoder:
                h = layer(h)
                if isinstance(layer, nn.ReLU):
                    layer_outputs.append(h.detach().cpu())
            soft = F.softmax(
                model.classifier(h), dim=-1
            ).detach().cpu()
        low_feat, mid_feat = layer_outputs[0], layer_outputs[1]
        high_feat = h.detach().cpu()
        for i in range(n):
            memory.inputs.append(sampled[i].detach().cpu())
            memory.low_features.append(low_feat[i])
            memory.mid_features.append(mid_feat[i])
            memory.high_features.append(high_feat[i])
            memory.soft_targets.append(soft[i])
            memory.task_ids.append(int(task_idx))
        task_labels_storage[int(task_idx)] = labels_pool[idx].clone()

        print(
            f"    task={task_idx}  train_loss={avg:.4f}  "
            f"stored={n}  memory_size={len(memory)}",
            flush=True,
        )

    return {
        "per_task_train_loss": per_task_train_loss,
        "wall_time_s": time.time() - t0,
        "final_memory_size": len(memory),
    }


# ---------- mechanical checks ----------


def _check(label: str, condition: bool, detail: str = "") -> dict[str, Any]:
    status = "PASS" if condition else "FAIL"
    line = f"  {label}  {status}"
    if detail:
        line = f"{line}  ({detail})"
    print(line)
    return {"label": label, "status": status, "detail": detail}


def _shape_check(
    label: str, t: Tensor, expected: tuple[int, ...],
) -> dict[str, Any]:
    actual = tuple(t.shape)
    ok = actual == expected
    detail = f"got {actual}, expected {expected}"
    return _check(label, ok, detail)


def _soft_target_validity(
    memory: MultiLevelMemory, n_check: int = 10,
) -> list[dict[str, Any]]:
    """Validate that ``n_check`` random soft targets are real
    probability distributions: sum to 1 within 1e-5, all values
    inside [0, 1], no NaN / Inf."""
    n = len(memory)
    gen = torch.Generator()
    gen.manual_seed(0)
    idx = torch.randperm(n, generator=gen)[: min(n_check, n)]
    sample = torch.stack([memory.soft_targets[int(i)] for i in idx])

    sums = sample.sum(dim=-1)
    sum_ok = bool(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))
    range_ok = bool(((sample >= 0.0) & (sample <= 1.0)).all().item())
    finite_ok = bool(torch.isfinite(sample).all().item())

    results = [
        _check(
            f"All {sample.shape[0]} soft targets sum to ~1.0 (atol=1e-5):",
            sum_ok,
            f"max |sum-1| = {float((sums - 1.0).abs().max()):.2e}",
        ),
        _check(
            "All values in [0, 1]:",
            range_ok,
            f"min={float(sample.min()):.3e}, "
            f"max={float(sample.max()):.3e}",
        ),
        _check(
            "All non-NaN, non-Inf:",
            finite_ok,
        ),
    ]
    return results


def _feature_distribution_sanity(
    memory: MultiLevelMemory, num_tasks: int,
) -> dict[str, Any]:
    """Print per-task mean/std for the low/mid/high feature tensors
    and check for the two pathologies: features all near zero
    (dead ReLUs) or features saturated huge (scale issues)."""
    print("\nFeature distribution sanity (per task):")
    flags: dict[str, Any] = {
        "dead_relu_tasks": [],
        "saturated_tasks": [],
        "stats": {},
    }
    for task_id in range(num_tasks):
        ids = memory.task_indices(task_id)
        if not ids:
            print(f"  task {task_id}: no stored entries")
            continue
        low = torch.stack([memory.low_features[i] for i in ids])
        mid = torch.stack([memory.mid_features[i] for i in ids])
        high = torch.stack([memory.high_features[i] for i in ids])
        stats = {
            "low_mean":  float(low.mean()),  "low_std":  float(low.std()),
            "mid_mean":  float(mid.mean()),  "mid_std":  float(mid.std()),
            "high_mean": float(high.mean()), "high_std": float(high.std()),
        }
        flags["stats"][int(task_id)] = stats
        print(
            f"  task {task_id}: "
            f"low_mean={stats['low_mean']:.3f}, low_std={stats['low_std']:.3f}  |  "
            f"mid_mean={stats['mid_mean']:.3f}, mid_std={stats['mid_std']:.3f}  |  "
            f"high_mean={stats['high_mean']:.3f}, high_std={stats['high_std']:.3f}"
        )

        # Pathology heuristics: ReLU outputs are non-negative, so
        # a healthy population has positive mean AND non-tiny std.
        # "Dead" = mean ≈ 0 (almost all neurons silent). "Saturated"
        # = mean > ~10 (unusual scale for this architecture).
        if stats["low_mean"] < 1e-3 and stats["mid_mean"] < 1e-3:
            flags["dead_relu_tasks"].append(int(task_id))
        if max(stats["low_mean"], stats["mid_mean"], stats["high_mean"]) > 10.0:
            flags["saturated_tasks"].append(int(task_id))

    print()
    _check(
        "No dead-ReLU tasks (mean > 1e-3):",
        not flags["dead_relu_tasks"],
        f"flagged tasks: {flags['dead_relu_tasks']}"
        if flags["dead_relu_tasks"] else "",
    )
    _check(
        "No saturated-feature tasks (mean ≤ 10):",
        not flags["saturated_tasks"],
        f"flagged tasks: {flags['saturated_tasks']}"
        if flags["saturated_tasks"] else "",
    )
    return flags


def _tsne_class_clustering(
    memory: MultiLevelMemory,
    task_labels: dict[int, Tensor],
    task_id: int,
    output_dir: Path,
    figure_path: Path,
) -> dict[str, Any]:
    """Compute t-SNE 2D projection of ``high_features`` for one task,
    save the colored scatter, and report silhouette score in the
    original feature space (more honest than scoring the 2D
    projection since t-SNE distorts distances)."""
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ids = memory.task_indices(task_id)
    if not ids:
        raise RuntimeError(
            f"No stored entries for task {task_id} — t-SNE check "
            f"cannot run."
        )

    feats = torch.stack([memory.high_features[i] for i in ids]).numpy()
    labels = task_labels[task_id].numpy()

    # Silhouette in the 64-dim feature space — the t-SNE plot
    # below is just for the eye. perplexity must be < n_samples; for
    # 100 samples sklearn's default of 30 is fine.
    if len(np.unique(labels)) < 2:
        sil_high = float("nan")
    else:
        sil_high = float(silhouette_score(feats, labels))

    perplexity = min(30, max(5, feats.shape[0] // 4))
    tsne = TSNE(
        n_components=2, perplexity=perplexity, random_state=0,
        init="pca",
    )
    proj = tsne.fit_transform(feats)
    if len(np.unique(labels)) < 2:
        sil_2d = float("nan")
    else:
        sil_2d = float(silhouette_score(proj, labels))

    print(
        f"\nt-SNE class clustering on task {task_id} "
        f"high_features (n={feats.shape[0]}, dim={feats.shape[1]}):"
    )
    print(f"  silhouette (64-dim feature space): {sil_high:.3f}")
    print(f"  silhouette (2-D t-SNE projection): {sil_2d:.3f}")
    _check(
        "Silhouette > 0.1 in feature space:",
        sil_high > 0.1,
        f"sil={sil_high:.3f}",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        proj[:, 0], proj[:, 1], c=labels, cmap="tab10",
        s=24, alpha=0.85, edgecolors="black", linewidths=0.3,
    )
    legend = ax.legend(
        *scatter.legend_elements(),
        title="class", loc="best", fontsize=8,
    )
    ax.add_artist(legend)
    ax.set_title(
        f"Phase 3 — t-SNE of hippocampe high_features "
        f"(task {task_id}, n={feats.shape[0]})\n"
        f"silhouette (64-dim)={sil_high:.3f}   "
        f"silhouette (2-D)={sil_2d:.3f}"
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    print(f"  figure saved to {figure_path}")

    return {
        "task_id": int(task_id),
        "n_samples": int(feats.shape[0]),
        "feature_dim": int(feats.shape[1]),
        "silhouette_feature_space": sil_high,
        "silhouette_2d": sil_2d,
        "figure_path": str(figure_path),
        "perplexity": int(perplexity),
    }


# ---------- main ----------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--epochs-per-task", "--epochs_per_task",
        dest="epochs_per_task", type=int, default=1,
        help="Default 1 to match other CL experiments in this repo.",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument(
        "--hidden-dims", type=int, nargs="+", default=[128, 64],
    )
    p.add_argument("--input-dim", type=int, default=784)
    p.add_argument("--n-classes", type=int, default=10)
    p.add_argument("--samples-per-task", type=int, default=100)
    p.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--bench-seed", type=int, default=42)
    p.add_argument(
        "--output-dir", type=Path,
        default=_REPO_ROOT / "results" / "logs" / "cls_phase3",
    )
    p.add_argument(
        "--figure-dir", type=Path,
        default=_REPO_ROOT / "results" / "figures" / "cls_phase3",
    )
    p.add_argument(
        "--tsne-task", type=int, default=0,
        help="Which task's high_features to project + plot.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.hidden_dims = [int(h) for h in args.hidden_dims]

    print(
        f"CLS Phase 3 — multi-level memory storage\n"
        f"  T={args.T}  samples_per_task={args.samples_per_task}\n"
        f"  hippocampus: hidden_dims={tuple(args.hidden_dims)} "
        f"lr={args.lr} epochs_per_task={args.epochs_per_task}\n"
        f"  device={args.device}",
        flush=True,
    )

    bench = PermutedMNIST.from_huggingface(
        num_tasks=args.T, seed=args.bench_seed,
    )
    memory = MultiLevelMemory(
        samples_per_task=args.samples_per_task,
        n_classes=args.n_classes,
    )
    task_labels: dict[int, Tensor] = {}

    train_diag = _train_hippocampus(bench, memory, task_labels, args)

    print(
        f"\nTraining done in {train_diag['wall_time_s']:.1f}s; "
        f"final memory size = {len(memory)}",
        flush=True,
    )

    # ----- mechanical checks -----
    print("\n=== Phase 3: Storage mechanical verification ===\n")
    print("Storage stats:")
    expected_total = args.samples_per_task * args.T
    counts = memory.per_task_counts()
    total_ok = len(memory) == expected_total
    per_task_ok = all(
        counts.get(t, 0) == args.samples_per_task for t in range(args.T)
    )
    print(
        f"  Total entries: {len(memory)} "
        f"(expected {expected_total})"
    )
    counts_str = ", ".join(
        f"t{t}={counts.get(t, 0)}" for t in range(args.T)
    )
    _check(f"Per-task counts: {counts_str}", total_ok and per_task_ok)

    print("\nTensor shape checks (entry 0 from each list):")
    shape_results = [
        _shape_check("inputs[0].shape:        ",
                     memory.inputs[0],         (args.input_dim,)),
        _shape_check("low_features[0].shape:  ",
                     memory.low_features[0],   (args.hidden_dims[0],)),
        _shape_check("mid_features[0].shape:  ",
                     memory.mid_features[0],   (args.hidden_dims[1],)),
        _shape_check("high_features[0].shape: ",
                     memory.high_features[0],  (args.hidden_dims[1],)),
        _shape_check("soft_targets[0].shape:  ",
                     memory.soft_targets[0],   (args.n_classes,)),
    ]

    print("\nSoft target validity (10 random entries):")
    soft_results = _soft_target_validity(memory, n_check=10)

    feature_flags = _feature_distribution_sanity(memory, num_tasks=args.T)

    figure_path = (
        args.figure_dir
        / f"phase3_tsne_task{args.tsne_task}_T{args.T}.png"
    )
    tsne_result = _tsne_class_clustering(
        memory, task_labels, task_id=args.tsne_task,
        output_dir=args.figure_dir,
        figure_path=figure_path,
    )

    # ----- verdict -----
    print("\n=== Verdict ===")
    mechanical_pass = (
        total_ok and per_task_ok
        and all(r["status"] == "PASS" for r in shape_results)
        and all(r["status"] == "PASS" for r in soft_results)
        and not feature_flags["dead_relu_tasks"]
        and not feature_flags["saturated_tasks"]
    )
    silhouette_pass = tsne_result["silhouette_feature_space"] > 0.1
    overall_pass = mechanical_pass and silhouette_pass

    if overall_pass:
        print("STORAGE BEHAVES AS EXPECTED — ready for Phase 4 (consolidation).")
    else:
        print("STORAGE NEEDS DEBUGGING:")
        if not mechanical_pass:
            print("  - One or more mechanical checks failed (see PASS/FAIL above).")
        if not silhouette_pass:
            print(
                f"  - Silhouette in feature space "
                f"({tsne_result['silhouette_feature_space']:.3f}) ≤ 0.1; "
                f"the hippocampe's high_features don't carry enough "
                f"class structure for the consolidation consistency "
                f"loss to anchor against. Check the t-SNE plot at "
                f"{figure_path} for a visual."
            )

    # Persist a JSON snapshot so the result is reproducible.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_path = args.output_dir / f"{ts}_34_T{args.T}_cls_phase3.json"
    payload = {
        "experiment": "34_cls_phase3_storage",
        "phase": 3,
        "component": "multi_level_memory_storage",
        "num_tasks": args.T,
        "timestamp": ts,
        "config": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
        "train_diagnostics": train_diag,
        "storage": {
            "total_entries": len(memory),
            "per_task_counts": counts,
        },
        "shape_checks": shape_results,
        "soft_target_checks": soft_results,
        "feature_distribution": feature_flags,
        "tsne": tsne_result,
        "verdict": "PASS" if overall_pass else "NEEDS_DEBUG",
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote results JSON to {out_path}")


if __name__ == "__main__":
    main()
