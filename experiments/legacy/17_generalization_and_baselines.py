"""Experiment 17 — generalisation and stronger-baseline validation.

After experiment 16 confirmed cs_gated_cosine beats naive on
Permuted-MNIST at n=5, this experiment locks in the statistical
foundation (n=10 seeds for clean Wilcoxon Bonferroni) and
addresses two questions experiment 16 did not:

1. Does cs_gated_cosine beat **EWC** (a real continual-learning
   baseline) and not just naive sequential fine-tuning?
2. Does the effect **generalise** to a second benchmark? Permuted-
   MNIST is a single benchmark — we need at least one more before
   making any architectural claim.

Methods (n=10 per benchmark):
- naive: sequential fine-tuning, no continual-learning mechanism.
- ewc: Elastic Weight Consolidation (Kirkpatrick et al. 2017),
  λ=1000, Fisher estimated on 500 samples per task.
- cs_gated_cosine: SynapseLayer + ColdStorage + gradient gating
  with familiarity_mode="cosine" (the exp 16 method).

Benchmarks:
- Permuted-MNIST 15 tasks, single shared 10-class head, dropout=0.5.
  Matches exp 16's protocol so cs_gated_cosine numbers are directly
  comparable.
- Split-MNIST 5 tasks, multi-head (one binary head per task),
  dropout=0.5 (to keep cs_gated_cosine's multi-pass denoising
  meaningful), 2 epochs per task, no zero-shot evaluation
  (untrained heads on unseen tasks would return random outputs).

Outputs two separate JSONs under
``results/logs/generalization/``:
- ``<ts>_17_permuted_mnist_T15.json``
- ``<ts>_17_split_mnist_multihead_T5.json``

Per-method checkpointing within each benchmark (atomic .tmp +
os.replace). The script proceeds Permuted → Split, so if killed
mid-Split the Permuted file is final and the Split file is
either absent or partial.

Run from the repo root::

    python experiments/17_generalization_and_baselines.py --seeds 0 1 2 3 4 5 6 7 8 9
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import chromadb
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from continual_synapse.baselines.ewc import EWC  # noqa: E402
from continual_synapse.baselines.multi_head import MultiHeadMLPClassifier  # noqa: E402
from continual_synapse.baselines.naive_finetune import MLPClassifier, MLPConfig  # noqa: E402
from continual_synapse.baselines.synapse_finetune import SynapseAugmentedMLP  # noqa: E402
from continual_synapse.cold_storage.compression import CompressionSchedule  # noqa: E402
from continual_synapse.cold_storage.store import ColdStorage  # noqa: E402
from continual_synapse.consolidation.trigger import ConsolidationTrigger  # noqa: E402
from continual_synapse.evaluation.benchmarks import (  # noqa: E402
    PermutedMNIST,
    SplitMNIST,
)
from continual_synapse.evaluation.multi_seed import MultiSeedRun, run_multi_seed  # noqa: E402
from continual_synapse.evaluation.reporting import compute_metrics  # noqa: E402
from continual_synapse.evaluation.runner import ContinualRunner, set_seed  # noqa: E402
from continual_synapse.evaluation.statistics import (  # noqa: E402
    format_pairwise_table,
    format_summary_table,
    pairwise_wilcoxon,
    summarise_method,
)
from continual_synapse.reward.consistency import ConsistencyReward  # noqa: E402
from continual_synapse.reward.external import ExternalReward  # noqa: E402
from continual_synapse.reward.mixer import RewardMixer  # noqa: E402
from continual_synapse.reward.surprise import SurpriseReward  # noqa: E402
from continual_synapse.synapse_layer.layer import SynapseLayer  # noqa: E402
from continual_synapse.synapse_layer.modulation import SynapseModulation  # noqa: E402


_METHODS = ("naive", "ewc", "cs_gated_cosine")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    p.add_argument("--methods", nargs="+", default=list(_METHODS))
    p.add_argument("--benchmarks", nargs="+",
                   default=["permuted_mnist", "split_mnist"],
                   choices=["permuted_mnist", "split_mnist"])
    # ---- Shared model / training hyperparameters ----
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-hidden-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.5,
                   help="Kept at 0.5 on both benchmarks so cs_gated_cosine's "
                        "multi-pass denoising has stochastic forwards to average "
                        "over (n_passes=5).")
    # ---- Per-benchmark task / epoch counts ----
    p.add_argument("--permuted-num-tasks", type=int, default=15)
    p.add_argument("--permuted-epochs", type=int, default=1)
    p.add_argument("--split-num-tasks", type=int, default=5)
    p.add_argument("--split-epochs", type=int, default=2)
    p.add_argument("--permutation-seed", type=int, default=42)
    # ---- EWC ----
    p.add_argument("--ewc-lam", type=float, default=1000.0)
    p.add_argument("--ewc-fisher-samples", type=int, default=500)
    # ---- Synapse layer + cold storage (shared with exp 16) ----
    p.add_argument("--synapse-lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=10.0)
    p.add_argument("--gamma", type=float, default=1e-3)
    p.add_argument("--w-consistency", type=float, default=1.0)
    p.add_argument("--w-surprise", type=float, default=0.5)
    p.add_argument("--pressure-threshold", type=float, default=0.005)
    p.add_argument("--min-steps-between-consolidations", type=int, default=60)
    p.add_argument("--candidate-quantile", type=float, default=0.05)
    p.add_argument("--retrieval-k", type=int, default=4)
    p.add_argument("--retrieval-refresh-interval", type=int, default=20)
    p.add_argument("--n-passes", type=int, default=5)
    p.add_argument("--compression-sweep-interval", type=int, default=100)
    p.add_argument("--gradient-gating-alpha", type=float, default=0.9)
    p.add_argument(
        "--age-thresholds", type=int, nargs="+", default=[100, 500, 2000]
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda", "mps"],
    )
    p.add_argument(
        "--cache-dir", default=str(_REPO_ROOT / "data" / "hf_cache")
    )
    p.add_argument(
        "--output-dir",
        default=str(_REPO_ROOT / "results" / "logs" / "generalization"),
    )
    return p.parse_args()


def _make_compression_schedule(args) -> CompressionSchedule:
    n_thresholds = len(args.age_thresholds)
    all_tiers = (32, 16, 8, 4)
    tiers = all_tiers[: n_thresholds + 1]
    return CompressionSchedule(
        age_thresholds=tuple(args.age_thresholds),
        tier_precisions=tiers,
    )


def _store_byte_size(store: ColdStorage) -> int:
    return sum(
        len(base64.b64decode(e.document)) for e in store.all_entries()
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _multi_seed_to_jsonable(run: MultiSeedRun) -> dict:
    out: dict = {"method": run.method, "seeds": run.seeds, "results": []}
    for r in run.results:
        summary = compute_metrics(r)
        out["results"].append(
            {
                "benchmark": r.benchmark,
                "task_names": r.task_names,
                "accuracy_matrix": [
                    [None if math.isnan(v) else float(v) for v in row]
                    for row in r.accuracy_matrix
                ],
                "random_baseline": r.random_baseline.tolist(),
                "metrics": asdict(summary),
            }
        )
    return out


def _build_payload(
    benchmark_name: str,
    args: argparse.Namespace,
    ts: int,
    runs: list[MultiSeedRun],
    summaries: list,
    diagnostics: dict[str, list[dict]],
    method_times: dict[str, float],
    *,
    is_partial: bool,
    methods_completed: list[str],
    methods_requested: list[str],
) -> dict[str, Any]:
    pairwise_acc = (
        [asdict(c) for c in pairwise_wilcoxon(summaries, metric="average_accuracy")]
        if len(summaries) >= 2 else []
    )
    pairwise_fgt = (
        [asdict(c) for c in pairwise_wilcoxon(summaries, metric="average_forgetting")]
        if len(summaries) >= 2 else []
    )
    return {
        "experiment": "17_generalization_and_baselines",
        "benchmark": benchmark_name,
        "timestamp": ts,
        "config": vars(args),
        "is_partial": is_partial,
        "methods_completed": list(methods_completed),
        "methods_requested": list(methods_requested),
        "methods": [_multi_seed_to_jsonable(r) for r in runs],
        "summaries": [
            {
                "method": s.method,
                "n_seeds": s.n_seeds,
                "metric_means": s.metric_means,
                "metric_stds": s.metric_stds,
                "per_seed_metrics": s.per_seed_metrics,
            }
            for s in summaries
        ],
        "pairwise_accuracy": pairwise_acc,
        "pairwise_forgetting": pairwise_fgt,
        "diagnostics": diagnostics,
        "method_times_seconds": method_times,
    }


# ---------- benchmark-specific factory builders ----------


def _build_permuted_factories(
    args, num_classes: int
) -> tuple[dict[str, Callable], dict[str, list[dict]]]:
    """Single-head MLP factories for Permuted-MNIST. Matches exp 16."""
    chroma_client = chromadb.Client()
    diagnostics: dict[str, list[dict]] = {m: [] for m in args.methods}

    runner_kwargs = dict(
        optimizer_factory=lambda p: torch.optim.SGD(
            p, lr=args.lr, momentum=args.momentum
        ),
        epochs_per_task=args.permuted_epochs,
        batch_size=args.batch_size,
        eval_batch_size=256,
        device=args.device,
        record_zero_shot=True,
    )

    def _build_mlp() -> MLPClassifier:
        return MLPClassifier(
            MLPConfig(
                input_dim=784,
                hidden_dim=args.hidden_dim,
                num_classes=num_classes,
                num_hidden_layers=args.num_hidden_layers,
                dropout=args.dropout,
            )
        )

    def naive(seed: int):
        set_seed(seed)
        model = _build_mlp()
        runner = ContinualRunner(seed=seed, **runner_kwargs)
        diagnostics["naive"].append({"seed": seed})
        return model, runner

    def ewc(seed: int):
        set_seed(seed)
        model = _build_mlp()
        e = EWC(
            lam=args.ewc_lam,
            fisher_sample_size=args.ewc_fisher_samples,
            device=args.device,
        )
        per_task_diag: list[dict] = []

        def on_task_end(i, task, m):
            e.consolidate(m, task.train)
            per_task_diag.append(
                {
                    "task_index": int(i),
                    "num_consolidated_tasks": int(e.num_consolidated_tasks),
                }
            )

        runner = ContinualRunner(
            seed=seed,
            regulariser=e.penalty,
            on_task_end=on_task_end,
            **runner_kwargs,
        )
        diagnostics["ewc"].append({"seed": seed, "per_task": per_task_diag})
        return model, runner

    def cs_gated_cosine(seed: int):
        set_seed(seed)
        base = _build_mlp()
        synapse = SynapseLayer(
            n_neurons=args.hidden_dim,
            learning_rate=args.synapse_lr,
            resistance_beta=args.beta,
            sparse=False,
            n_passes=args.n_passes,
        )
        modulator = SynapseModulation(init_gate=0.0)
        reward_computer = RewardMixer(
            external=ExternalReward(default=1.0),
            consistency=ConsistencyReward(
                n_neurons=args.hidden_dim, decay=0.99
            ),
            surprise=SurpriseReward(n_neurons=args.hidden_dim),
            gamma=args.gamma,
            w_consistency=args.w_consistency,
            w_surprise=args.w_surprise,
        )
        cold_storage = ColdStorage(
            collection_name=f"exp17_permuted_csgc_seed_{seed}_{time.time_ns()}",
            client=chroma_client,
        )
        trigger = ConsolidationTrigger(
            avg_pressure_threshold=args.pressure_threshold,
            min_steps_between=args.min_steps_between_consolidations,
            candidate_quantile=args.candidate_quantile,
        )
        model = SynapseAugmentedMLP(
            base, synapse, modulator,
            reward_computer=reward_computer,
            cold_storage=cold_storage,
            consolidation_trigger=trigger,
            retrieval_k=args.retrieval_k,
            retrieval_refresh_interval=args.retrieval_refresh_interval,
            n_passes=args.n_passes,
            compression_sweep_interval=args.compression_sweep_interval,
            compression_schedule=_make_compression_schedule(args),
            gate_modulation_enabled=False,
            gradient_gating_enabled=True,
            gradient_gating_alpha=args.gradient_gating_alpha,
            familiarity_mode="cosine",
        )
        per_task_diag, task_state = _gated_cosine_diag_hooks(model)

        def on_pre_step(i, task, m):
            scale = m.apply_gradient_gating()
            fam = float(m.last_familiarity)
            task_state["familiarity_sum"] += fam
            task_state["familiarity_count"] += 1
            task_state["gradient_scale_sum"] += float(scale)
            task_state["gradient_scale_count"] += 1
            if fam > task_state["max_similarity_this_task"]:
                task_state["max_similarity_this_task"] = fam

        def on_after_batch(i, task, m, x, y):
            m.apply_hebbian_update()

        def on_task_end(i, task, m):
            _snapshot_gated_per_task(per_task_diag, task_state, i, m)

        runner = ContinualRunner(
            seed=seed,
            on_after_batch=on_after_batch,
            on_task_end=on_task_end,
            on_pre_optimizer_step=on_pre_step,
            **runner_kwargs,
        )
        diagnostics["cs_gated_cosine"].append(
            {"seed": seed, "per_task": per_task_diag}
        )
        return model, runner

    return {
        "naive": naive,
        "ewc": ewc,
        "cs_gated_cosine": cs_gated_cosine,
    }, diagnostics


def _build_split_factories(
    args, num_classes: int, num_tasks: int
) -> tuple[dict[str, Callable], dict[str, list[dict]]]:
    """Multi-head factories for Split-MNIST. on_task_change selects head."""
    chroma_client = chromadb.Client()
    diagnostics: dict[str, list[dict]] = {m: [] for m in args.methods}

    def _select_head(i, task, m):
        m.set_active_head(int(i))

    runner_kwargs = dict(
        optimizer_factory=lambda p: torch.optim.SGD(
            p, lr=args.lr, momentum=args.momentum
        ),
        epochs_per_task=args.split_epochs,
        batch_size=args.batch_size,
        eval_batch_size=256,
        device=args.device,
        record_zero_shot=False,  # untrained heads on unseen tasks ⇒ random
        on_task_change=_select_head,
    )

    def _build_multihead() -> MultiHeadMLPClassifier:
        return MultiHeadMLPClassifier(
            num_tasks=num_tasks,
            config=MLPConfig(
                input_dim=784,
                hidden_dim=args.hidden_dim,
                num_classes=num_classes,
                num_hidden_layers=args.num_hidden_layers,
                dropout=args.dropout,
            ),
        )

    def naive(seed: int):
        set_seed(seed)
        model = _build_multihead()
        runner = ContinualRunner(seed=seed, **runner_kwargs)
        diagnostics["naive"].append({"seed": seed})
        return model, runner

    def ewc(seed: int):
        set_seed(seed)
        model = _build_multihead()
        e = EWC(
            lam=args.ewc_lam,
            fisher_sample_size=args.ewc_fisher_samples,
            device=args.device,
        )
        per_task_diag: list[dict] = []

        def on_task_end(i, task, m):
            e.consolidate(m, task.train)
            per_task_diag.append(
                {
                    "task_index": int(i),
                    "num_consolidated_tasks": int(e.num_consolidated_tasks),
                }
            )

        runner = ContinualRunner(
            seed=seed,
            regulariser=e.penalty,
            on_task_end=on_task_end,
            **{k: v for k, v in runner_kwargs.items() if k != "on_task_end"},
        )
        diagnostics["ewc"].append({"seed": seed, "per_task": per_task_diag})
        return model, runner

    def cs_gated_cosine(seed: int):
        set_seed(seed)
        base = _build_multihead()
        synapse = SynapseLayer(
            n_neurons=args.hidden_dim,
            learning_rate=args.synapse_lr,
            resistance_beta=args.beta,
            sparse=False,
            n_passes=args.n_passes,
        )
        modulator = SynapseModulation(init_gate=0.0)
        reward_computer = RewardMixer(
            external=ExternalReward(default=1.0),
            consistency=ConsistencyReward(
                n_neurons=args.hidden_dim, decay=0.99
            ),
            surprise=SurpriseReward(n_neurons=args.hidden_dim),
            gamma=args.gamma,
            w_consistency=args.w_consistency,
            w_surprise=args.w_surprise,
        )
        cold_storage = ColdStorage(
            collection_name=f"exp17_split_csgc_seed_{seed}_{time.time_ns()}",
            client=chroma_client,
        )
        trigger = ConsolidationTrigger(
            avg_pressure_threshold=args.pressure_threshold,
            min_steps_between=args.min_steps_between_consolidations,
            candidate_quantile=args.candidate_quantile,
        )
        model = SynapseAugmentedMLP(
            base, synapse, modulator,
            reward_computer=reward_computer,
            cold_storage=cold_storage,
            consolidation_trigger=trigger,
            retrieval_k=args.retrieval_k,
            retrieval_refresh_interval=args.retrieval_refresh_interval,
            n_passes=args.n_passes,
            compression_sweep_interval=args.compression_sweep_interval,
            compression_schedule=_make_compression_schedule(args),
            gate_modulation_enabled=False,
            gradient_gating_enabled=True,
            gradient_gating_alpha=args.gradient_gating_alpha,
            familiarity_mode="cosine",
        )
        per_task_diag, task_state = _gated_cosine_diag_hooks(model)

        def on_pre_step(i, task, m):
            scale = m.apply_gradient_gating()
            fam = float(m.last_familiarity)
            task_state["familiarity_sum"] += fam
            task_state["familiarity_count"] += 1
            task_state["gradient_scale_sum"] += float(scale)
            task_state["gradient_scale_count"] += 1
            if fam > task_state["max_similarity_this_task"]:
                task_state["max_similarity_this_task"] = fam

        def on_after_batch(i, task, m, x, y):
            m.apply_hebbian_update()

        def on_task_end(i, task, m):
            _snapshot_gated_per_task(per_task_diag, task_state, i, m)

        runner = ContinualRunner(
            seed=seed,
            on_after_batch=on_after_batch,
            on_task_end=on_task_end,
            on_pre_optimizer_step=on_pre_step,
            **runner_kwargs,
        )
        diagnostics["cs_gated_cosine"].append(
            {"seed": seed, "per_task": per_task_diag}
        )
        return model, runner

    return {
        "naive": naive,
        "ewc": ewc,
        "cs_gated_cosine": cs_gated_cosine,
    }, diagnostics


# ---------- shared diagnostics helpers ----------


def _gated_cosine_diag_hooks(model) -> tuple[list[dict], dict[str, Any]]:
    per_task_diag: list[dict] = []
    task_state: dict[str, Any] = {
        "familiarity_sum": 0.0,
        "familiarity_count": 0,
        "gradient_scale_sum": 0.0,
        "gradient_scale_count": 0,
        "max_similarity_this_task": 0.0,
    }
    return per_task_diag, task_state


def _snapshot_gated_per_task(
    per_task_diag: list[dict],
    task_state: dict[str, Any],
    task_index: int,
    m,
) -> None:
    fc = task_state["familiarity_count"]
    gc = task_state["gradient_scale_count"]
    sims_snapshot = sorted(m.last_similarities, reverse=True)
    per_task_diag.append(
        {
            "task_index": int(task_index),
            "consolidation_count": int(m.consolidation_count),
            "store_count": int(m.cold_storage.count()),
            "compression_sweep_count": int(m.compression_sweep_count),
            "store_byte_size": int(_store_byte_size(m.cold_storage)),
            "last_compression_counts": {
                int(k): int(v)
                for k, v in m.last_compression_counts.items()
            },
            "avg_familiarity": (
                task_state["familiarity_sum"] / fc if fc > 0 else None
            ),
            "avg_gradient_scale": (
                task_state["gradient_scale_sum"] / gc if gc > 0 else None
            ),
            "familiarity_max": float(m.familiarity_max),
            "max_similarity_this_task": float(
                task_state["max_similarity_this_task"]
            ),
            "last_similarities_distribution": [
                float(v) for v in sims_snapshot
            ],
            "modulator_gate": float(m.modulator.gate.item()),
        }
    )
    task_state["familiarity_sum"] = 0.0
    task_state["familiarity_count"] = 0
    task_state["gradient_scale_sum"] = 0.0
    task_state["gradient_scale_count"] = 0
    task_state["max_similarity_this_task"] = 0.0


# ---------- per-benchmark driver ----------


def _run_benchmark(
    benchmark_name: str,
    bench,
    factories: dict[str, Callable],
    diagnostics: dict[str, list[dict]],
    args: argparse.Namespace,
    output_path: Path,
) -> tuple[list[MultiSeedRun], list]:
    ts = int(time.time())
    print(f"\n=== Running benchmark: {benchmark_name} ===", flush=True)
    print(f"Checkpoint path: {output_path}", flush=True)

    runs: list[MultiSeedRun] = []
    summaries: list = []
    method_times: dict[str, float] = {}
    methods_completed: list[str] = []

    for method in args.methods:
        if method not in factories:
            raise SystemExit(
                f"unknown method {method!r}; known: {list(factories)}"
            )
        t0 = time.time()

        def _seed_done(m, i, n, seed, result):
            s = compute_metrics(result)
            elapsed = time.time() - t0
            print(
                f"    {m} seed {i + 1}/{n} (seed={seed}, "
                f"{elapsed:.0f}s into method): "
                f"acc={s.average_accuracy:.3f}, "
                f"fgt={s.average_forgetting:+.3f}",
                flush=True,
            )

        run = run_multi_seed(
            method,
            factories[method],
            bench,
            seeds=args.seeds,
            progress=lambda m, i, n: print(
                f"  {m}: seed {i + 1}/{n}", flush=True
            ),
            on_seed_complete=_seed_done,
        )
        elapsed = time.time() - t0
        method_times[method] = elapsed
        runs.append(run)
        summaries.append(summarise_method(run))
        methods_completed.append(method)
        print(f"{method} finished in {elapsed:.1f}s", flush=True)

        is_partial = len(methods_completed) < len(args.methods)
        payload = _build_payload(
            benchmark_name, args, ts, runs, summaries, diagnostics,
            method_times, is_partial=is_partial,
            methods_completed=methods_completed,
            methods_requested=list(args.methods),
        )
        _atomic_write_json(output_path, payload)
        tag = "partial" if is_partial else "final"
        print(
            f"  Checkpoint written ({tag}, "
            f"{len(methods_completed)}/{len(args.methods)} methods): "
            f"{output_path}",
            flush=True,
        )

    print()
    print(
        f"[{benchmark_name}] Per-method mean ± std (n={len(args.seeds)} seeds):"
    )
    print(format_summary_table(summaries))

    for metric in ("average_accuracy", "average_forgetting"):
        print(
            f"[{benchmark_name}] Pairwise Wilcoxon signed-rank on {metric} "
            f"(Bonferroni-corrected):"
        )
        print(format_pairwise_table(pairwise_wilcoxon(summaries, metric=metric)))

    return runs, summaries


def _print_task0_trajectory(
    benchmark_name: str,
    runs: list[MultiSeedRun],
    num_tasks: int,
) -> None:
    print()
    print(f"[{benchmark_name}] Task-0 accuracy after each subsequent task "
          f"(mean ± std):")
    print(f"  {'after_task':<10s} ", end="")
    for run in runs:
        print(f"{run.method:>18s}", end=" ")
    print()
    for i in range(num_tasks):
        print(f"  {i:<10d} ", end="")
        for run in runs:
            vals = []
            for r in run.results:
                v = r.accuracy_matrix[i, 0]
                if not math.isnan(v):
                    vals.append(float(v))
            if vals:
                mean = sum(vals) / len(vals)
                std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
                print(f"{mean:>9.3f}±{std:<7.3f}", end=" ")
            else:
                print(f"{'—':>18s}", end=" ")
        print()


def main() -> None:
    args = parse_args()
    if len(args.seeds) < 2:
        raise SystemExit("multi-seed experiment needs at least 2 seeds")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts_root = int(time.time())

    benchmark_results: dict[str, tuple[list[MultiSeedRun], list, int]] = {}

    # ---------- Permuted-MNIST 15 tasks (single head) ----------
    if "permuted_mnist" in args.benchmarks:
        print(
            f"Loading PermutedMNIST with {args.permuted_num_tasks} tasks "
            f"(permutation seed={args.permutation_seed}, "
            f"dropout={args.dropout})..."
        )
        perm_bench = PermutedMNIST.from_huggingface(
            num_tasks=args.permuted_num_tasks,
            seed=args.permutation_seed,
            cache_dir=args.cache_dir,
        )
        perm_factories, perm_diag = _build_permuted_factories(
            args, num_classes=perm_bench.num_classes_per_task
        )
        perm_path = (
            output_dir
            / f"{ts_root}_17_permuted_mnist_T{args.permuted_num_tasks}.json"
        )
        perm_runs, perm_summaries = _run_benchmark(
            "permuted_mnist", perm_bench, perm_factories, perm_diag, args,
            perm_path,
        )
        benchmark_results["permuted_mnist"] = (
            perm_runs, perm_summaries, args.permuted_num_tasks
        )

    # ---------- Split-MNIST 5 tasks (multi-head) ----------
    if "split_mnist" in args.benchmarks:
        print(
            f"\nLoading SplitMNIST (5 binary tasks, multi-head, "
            f"dropout={args.dropout})..."
        )
        split_bench = SplitMNIST.from_huggingface(cache_dir=args.cache_dir)
        actual_split_tasks = len(split_bench.tasks())
        if actual_split_tasks != args.split_num_tasks:
            print(
                f"Warning: --split-num-tasks={args.split_num_tasks} but "
                f"SplitMNIST produced {actual_split_tasks} tasks; "
                f"using actual count.",
                flush=True,
            )
        split_factories, split_diag = _build_split_factories(
            args,
            num_classes=split_bench.num_classes_per_task,
            num_tasks=actual_split_tasks,
        )
        split_path = (
            output_dir
            / f"{ts_root}_17_split_mnist_multihead_T{actual_split_tasks}.json"
        )
        split_runs, split_summaries = _run_benchmark(
            "split_mnist_multihead", split_bench, split_factories, split_diag,
            args, split_path,
        )
        benchmark_results["split_mnist"] = (
            split_runs, split_summaries, actual_split_tasks
        )

    # ---------- Task-0 trajectories side-by-side per benchmark ----------
    for name, (runs, _, T) in benchmark_results.items():
        _print_task0_trajectory(name, runs, T)

    print()
    print("All benchmarks done.")
    if "permuted_mnist" in benchmark_results:
        print(f"  permuted_mnist log:        {perm_path}")
    if "split_mnist" in benchmark_results:
        print(f"  split_mnist_multihead log: {split_path}")


if __name__ == "__main__":
    main()
