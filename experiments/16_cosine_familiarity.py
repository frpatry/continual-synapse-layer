"""Experiment 16 — cosine vs magnitude familiarity for gradient gating.

Tests whether pattern-specific recognition (cosine similarity to
stored cold-storage embeddings) works as the familiarity signal
where the aggregate magnitude measure used by experiment 15's
cs_gated did not. Same gradient-gating mechanism otherwise:
between ``loss.backward()`` and ``optimizer.step()``, scale
``base.parameters()`` gradients by ``1 - alpha * familiarity``.

Methods (intentionally minimal — three-way comparison for direct
diagnostic of the familiarity-mode change):

- naive: control.
- cs_gated: SynapseLayer + ColdStorage + gradient gating in
  magnitude mode (matches exp 15 cs_gated bit-exact after the
  multi-pass fix in commit ``3f74238``). The baseline whose
  per-batch familiarity saturates as the synapse layer
  accumulates patterns.
- cs_gated_cosine: same gating mechanism but
  ``familiarity_mode="cosine"``. Per-batch familiarity is
  ``max(0, max(cosine_sim(activation, every stored embedding)))``,
  so high familiarity means "I've seen this specific pattern
  before" rather than "the synapse layer has high total
  magnitude".

Per-task diagnostics extend exp 15 with two cosine-specific fields:
- ``per_task_max_similarity``: max of per-batch raw familiarity
  values observed during this task. In cosine mode that is the
  highest cosine sim to any stored pattern over the task.
- ``last_similarities_distribution``: snapshot of
  ``model.last_similarities`` at task end. Lets you inspect the
  per-entry similarity vector and tell whether high familiarity
  is concentrated on a few entries or spread.

Standard ACC / FGT / BWT / FWT plus Wilcoxon pairwise comparisons
on average_accuracy and average_forgetting (Bonferroni-corrected,
limited power at n=5 but informative). Full ``(T, T)`` accuracy
matrix per (method, seed) preserved so the Task-0 retention
trajectory can be inspected.

Run from the repo root::

    python experiments/16_cosine_familiarity.py --num-tasks 15
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

from continual_synapse.baselines.naive_finetune import MLPClassifier, MLPConfig  # noqa: E402
from continual_synapse.baselines.synapse_finetune import SynapseAugmentedMLP  # noqa: E402
from continual_synapse.cold_storage.compression import CompressionSchedule  # noqa: E402
from continual_synapse.cold_storage.store import ColdStorage  # noqa: E402
from continual_synapse.consolidation.trigger import ConsolidationTrigger  # noqa: E402
from continual_synapse.evaluation.benchmarks import PermutedMNIST  # noqa: E402
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


_DEFAULT_METHODS = ("naive", "cs_gated", "cs_gated_cosine")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--num-tasks", type=int, default=15)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs-per-task", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-hidden-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.5)
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
    p.add_argument("--permutation-seed", type=int, default=42)
    p.add_argument("--methods", nargs="+", default=list(_DEFAULT_METHODS))
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
        default=str(_REPO_ROOT / "results" / "logs" / "cosine_familiarity"),
    )
    return p.parse_args()


def _build_mlp(args, num_classes: int) -> MLPClassifier:
    return MLPClassifier(
        MLPConfig(
            input_dim=784,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            num_hidden_layers=args.num_hidden_layers,
            dropout=args.dropout,
        )
    )


def _make_schedule(args) -> CompressionSchedule:
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


def _build_factories(
    args, num_classes: int
) -> tuple[
    dict[str, Callable],
    dict[str, list[dict]],
]:
    chroma_client = chromadb.Client()
    diagnostics: dict[str, list[dict]] = {m: [] for m in args.methods}

    runner_kwargs = dict(
        optimizer_factory=lambda p: torch.optim.SGD(
            p, lr=args.lr, momentum=args.momentum
        ),
        epochs_per_task=args.epochs_per_task,
        batch_size=args.batch_size,
        eval_batch_size=256,
        device=args.device,
        record_zero_shot=True,
    )

    def naive(seed: int):
        set_seed(seed)
        model = _build_mlp(args, num_classes)
        runner = ContinualRunner(seed=seed, **runner_kwargs)
        diagnostics["naive"].append({"seed": seed})
        return model, runner

    def _gated(seed: int, method_key: str, *, familiarity_mode: str):
        set_seed(seed)
        base = _build_mlp(args, num_classes)
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
            collection_name=f"exp16_{method_key}_seed_{seed}_{time.time_ns()}",
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
            compression_schedule=_make_schedule(args),
            gate_modulation_enabled=False,
            gradient_gating_enabled=True,
            gradient_gating_alpha=args.gradient_gating_alpha,
            familiarity_mode=familiarity_mode,
        )

        per_task_diag: list[dict] = []
        task_state: dict[str, Any] = {
            "familiarity_sum": 0.0,
            "familiarity_count": 0,
            "gradient_scale_sum": 0.0,
            "gradient_scale_count": 0,
            "max_similarity_this_task": 0.0,
        }

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

        def on_task_end_diag(i, task, m):
            fc = task_state["familiarity_count"]
            gc = task_state["gradient_scale_count"]
            # Snapshot last_similarities at task end. Sorted descending
            # so the head of the list is "most-similar stored entry to
            # the last batch's activation".
            sims_snapshot = sorted(m.last_similarities, reverse=True)
            per_task_diag.append(
                {
                    "task_index": int(i),
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

        runner = ContinualRunner(
            seed=seed,
            on_after_batch=on_after_batch,
            on_task_end=on_task_end_diag,
            on_pre_optimizer_step=on_pre_step,
            **runner_kwargs,
        )
        diagnostics[method_key].append(
            {"seed": seed, "per_task": per_task_diag}
        )
        return model, runner

    factories: dict[str, Callable] = {
        "naive": naive,
        "cs_gated": lambda s: _gated(
            s, "cs_gated", familiarity_mode="magnitude"
        ),
        "cs_gated_cosine": lambda s: _gated(
            s, "cs_gated_cosine", familiarity_mode="cosine"
        ),
    }
    return factories, diagnostics


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
    args: argparse.Namespace,
    ts: int,
    runs: list[MultiSeedRun],
    summaries: list,
    diagnostics: dict[str, list[dict]],
    method_times: dict[str, float],
    *,
    is_partial: bool,
    methods_completed: list[str],
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
        "experiment": "16_cosine_familiarity",
        "timestamp": ts,
        "config": vars(args),
        "is_partial": is_partial,
        "methods_completed": list(methods_completed),
        "methods_requested": list(args.methods),
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


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def main() -> None:
    args = parse_args()
    if len(args.seeds) < 2:
        raise SystemExit("multi-seed experiment needs at least 2 seeds")

    print(
        f"Loading PermutedMNIST with {args.num_tasks} tasks "
        f"(permutation seed={args.permutation_seed}, dropout={args.dropout})..."
    )
    bench = PermutedMNIST.from_huggingface(
        num_tasks=args.num_tasks,
        seed=args.permutation_seed,
        cache_dir=args.cache_dir,
    )
    factories, diagnostics = _build_factories(
        args, num_classes=bench.num_classes_per_task
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    path = output_dir / f"{ts}_16_cosine_familiarity_T{args.num_tasks}.json"
    print(f"Checkpoint path: {path}", flush=True)

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
            args, ts, runs, summaries, diagnostics, method_times,
            is_partial=is_partial, methods_completed=methods_completed,
        )
        _atomic_write_json(path, payload)
        tag = "partial" if is_partial else "final"
        print(
            f"  Checkpoint written ({tag}, "
            f"{len(methods_completed)}/{len(args.methods)} methods): {path}",
            flush=True,
        )

    print()
    print(
        f"Per-method mean ± std (n={len(args.seeds)} seeds, "
        f"PERMUTED-MNIST {args.num_tasks} tasks, dropout={args.dropout}):"
    )
    print(format_summary_table(summaries))

    for metric in ("average_accuracy", "average_forgetting"):
        print(
            f"Pairwise Wilcoxon signed-rank on {metric} "
            f"(Bonferroni-corrected; n={len(args.seeds)} so power is limited):"
        )
        print(format_pairwise_table(pairwise_wilcoxon(summaries, metric=metric)))

    # Task-0 retention trajectory.
    print()
    print("Task-0 accuracy after each subsequent training task (mean ± std):")
    print(f"  {'after_task':<10s} ", end="")
    for run in runs:
        print(f"{run.method:>18s}", end=" ")
    print()
    T = args.num_tasks
    for i in range(T):
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

    # End-of-run gating diagnostics, per gated method.
    print()
    print("End-of-run gating diagnostics (mean across seeds):")
    for method in ("cs_gated", "cs_gated_cosine"):
        if method not in args.methods:
            continue
        seed_diags = diagnostics.get(method, [])
        if not seed_diags or not any(d.get("per_task") for d in seed_diags):
            continue
        finals = [d["per_task"][-1] for d in seed_diags if d.get("per_task")]
        if not finals:
            continue

        def _mean(key: str) -> float:
            vals = [
                float(f[key]) for f in finals
                if f.get(key) is not None
            ]
            return sum(vals) / len(vals) if vals else float("nan")

        print(f"  --- {method} ---")
        print(f"    avg_familiarity (last task):     {_mean('avg_familiarity'):.3f}")
        print(f"    avg_gradient_scale (last task):  {_mean('avg_gradient_scale'):.3f}")
        print(f"    familiarity_max (end of run):    {_mean('familiarity_max'):.3f}")
        print(f"    max_similarity_this_task (last): {_mean('max_similarity_this_task'):.3f}")
        print(f"    consolidation_count (end):       {_mean('consolidation_count'):.1f}")
        print(f"    store_count (end):               {_mean('store_count'):.1f}")
        print(f"    modulator_gate (should be 0):    {_mean('modulator_gate'):.4f}")

    print(f"Saved run log to {path}")


if __name__ == "__main__":
    main()
