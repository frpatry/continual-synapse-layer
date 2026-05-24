"""Experiment 12 — audit-driven architectural fixes.

The 2026-05-23 audit found three gaps in the architecture that
experiment 11 was already supposed to address:

1. Sparse top-k partner selection (DESIGN.md §3.2) was
   implemented but never activated — every experiment ran with
   ``sparse=False``.
2. The "full reward system" in cs_full was architecturally
   present but functionally inactive — external reward stayed at
   1.0, consistency saturated at ~0.97, α(t) blended two
   near-constant signals.
3. Multi-pass retrieval query used the noisy first-forward
   activation rather than the denoised buffer average.

This experiment isolates each fix and a stacked variant, against
the unchanged cs_full from experiment 11. Same 15-task PermutedMNIST
benchmark, same hyperparameters, dropout=0.5, 5 seeds.

Methods:
- naive: control.
- cs_full: as in experiment 11 — for direct comparison; carries
  every audit gap.
- cs_full_sparse: + sparse top-k, k=64 on the 256-d hidden state.
- cs_full_real_reward: + recentered consistency (center=0.95,
  scale=0.05, clip ±1) + external driven by per-batch accuracy.
- cs_full_complete: + sparse + real reward + multi-pass query fix.

Per-method diagnostics extended over experiment 11:
- All standard ACC/FGT/BWT/FWT metrics.
- Per-batch reward summary (mean, std, min, max) so we can verify
  the reward signal is actually varying now.
- Sparse mask density per task (rows × non-zero fraction) for the
  sparse variants.
- Existing consolidation count, store count, sweep count.

Run from the repo root::

    python experiments/12_audit_fixes.py --num-tasks 15

Output: ``results/logs/audit_fixes/<ts>_12_audit_fixes_T15.json``.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
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


_DEFAULT_METHODS = (
    "naive",
    "cs_full",
    "cs_full_sparse",
    "cs_full_real_reward",
    "cs_full_complete",
)


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
    p.add_argument(
        "--top-k",
        type=int,
        default=64,
        help="Sparse top-k partners per source neuron in sparse variants.",
    )
    p.add_argument(
        "--consistency-center", type=float, default=0.95,
        help="Center for recentered consistency reward.",
    )
    p.add_argument(
        "--consistency-scale", type=float, default=0.05,
        help="Scale for recentered consistency reward.",
    )
    p.add_argument(
        "--consistency-clip-min", type=float, default=-1.0,
    )
    p.add_argument(
        "--consistency-clip-max", type=float, default=1.0,
    )
    p.add_argument("--age-thresholds", type=int, nargs="+", default=[100, 500, 2000])
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
        default=str(_REPO_ROOT / "results" / "logs" / "audit_fixes"),
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


def _sparse_density(synapse: SynapseLayer) -> float:
    """Fraction of strength entries that are non-zero."""
    return float((synapse.strengths != 0).float().mean().item())


def _build_factories(
    args, num_classes: int
) -> tuple[dict[str, Callable], dict[str, list[dict]], dict[str, list[list[float]]]]:
    chroma_client = chromadb.Client()
    diagnostics: dict[str, list[dict]] = {m: [] for m in args.methods}
    # Per-method per-seed per-batch reward sample (capped so memory stays sane).
    reward_samples: dict[str, list[list[float]]] = {m: [] for m in args.methods}

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

    def _build_reward(use_real_reward: bool):
        consistency_kwargs: dict[str, Any] = dict(
            n_neurons=args.hidden_dim, decay=0.99
        )
        if use_real_reward:
            consistency_kwargs.update(
                center=args.consistency_center,
                scale=args.consistency_scale,
                clip_min=args.consistency_clip_min,
                clip_max=args.consistency_clip_max,
            )
        return RewardMixer(
            external=ExternalReward(default=1.0),
            consistency=ConsistencyReward(**consistency_kwargs),
            surprise=SurpriseReward(n_neurons=args.hidden_dim),
            gamma=args.gamma,
            w_consistency=args.w_consistency,
            w_surprise=args.w_surprise,
        )

    def _synapse(
        seed: int,
        method_key: str,
        *,
        sparse: bool,
        use_real_reward: bool,
    ):
        set_seed(seed)
        base = _build_mlp(args, num_classes)
        synapse = SynapseLayer(
            n_neurons=args.hidden_dim,
            learning_rate=args.synapse_lr,
            resistance_beta=args.beta,
            sparse=sparse,
            top_k=args.top_k,
            n_passes=args.n_passes,
        )
        modulator = SynapseModulation(init_gate=0.0)
        reward_computer = _build_reward(use_real_reward)
        cold_storage = ColdStorage(
            collection_name=f"exp12_{method_key}_seed_{seed}_{time.time_ns()}",
            client=chroma_client,
        )
        trigger = ConsolidationTrigger(
            avg_pressure_threshold=args.pressure_threshold,
            min_steps_between=args.min_steps_between_consolidations,
            candidate_quantile=args.candidate_quantile,
        )
        model = SynapseAugmentedMLP(
            base,
            synapse,
            modulator,
            reward_computer=reward_computer,
            cold_storage=cold_storage,
            consolidation_trigger=trigger,
            retrieval_k=args.retrieval_k,
            retrieval_refresh_interval=args.retrieval_refresh_interval,
            n_passes=args.n_passes,
            compression_sweep_interval=args.compression_sweep_interval,
            compression_schedule=_make_schedule(args),
        )

        per_task_diag: list[dict] = []
        seed_rewards: list[float] = []
        reward_samples[method_key].append(seed_rewards)

        def on_after_batch(i, task, m, x, y):
            if use_real_reward:
                applied = m.apply_hebbian_update(training_target=y)
            else:
                applied = m.apply_hebbian_update()
            # Capture every reward for distribution analysis.
            seed_rewards.append(float(applied))

        def on_task_end_diag(i, task, m):
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
                    "sparse_density": _sparse_density(m.synapse),
                    "modulator_gate": float(m.modulator.gate.item()),
                }
            )

        runner = ContinualRunner(
            seed=seed,
            on_after_batch=on_after_batch,
            on_task_end=on_task_end_diag,
            **runner_kwargs,
        )
        diagnostics[method_key].append(
            {"seed": seed, "per_task": per_task_diag}
        )
        return model, runner

    factories: dict[str, Callable] = {
        "naive": naive,
        # cs_full: every audit gap present (sparse off, fake reward, first-pass query).
        "cs_full": lambda s: _synapse(
            s, "cs_full", sparse=False, use_real_reward=False
        ),
        # +sparse only.
        "cs_full_sparse": lambda s: _synapse(
            s, "cs_full_sparse", sparse=True, use_real_reward=False
        ),
        # +real reward only (consistency recentered + external from acc).
        "cs_full_real_reward": lambda s: _synapse(
            s, "cs_full_real_reward", sparse=False, use_real_reward=True
        ),
        # +sparse +real reward (multi-pass query fix is unconditional in the code).
        "cs_full_complete": lambda s: _synapse(
            s, "cs_full_complete", sparse=True, use_real_reward=True
        ),
    }
    return factories, diagnostics, reward_samples


def _reward_histogram(samples: list[float]) -> dict[str, Any]:
    """Summary statistics for the per-batch reward stream."""
    if not samples:
        return {"n": 0}
    arr = sorted(samples)
    n = len(arr)
    return {
        "n": n,
        "mean": sum(arr) / n,
        "min": arr[0],
        "max": arr[-1],
        "p10": arr[int(n * 0.10)],
        "p50": arr[int(n * 0.50)],
        "p90": arr[int(n * 0.90)],
        "std": (sum((x - sum(arr) / n) ** 2 for x in arr) / n) ** 0.5,
    }


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
    factories, diagnostics, reward_samples = _build_factories(
        args, num_classes=bench.num_classes_per_task
    )

    runs: list[MultiSeedRun] = []
    method_times: dict[str, float] = {}
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
        print(f"{method} finished in {elapsed:.1f}s", flush=True)

    summaries = [summarise_method(r) for r in runs]
    print()
    print(
        f"Per-method mean ± std (n={len(args.seeds)} seeds, "
        f"PERMUTED-MNIST {args.num_tasks} tasks, dropout={args.dropout}):"
    )
    print(format_summary_table(summaries))

    for metric in ("average_accuracy", "average_forgetting"):
        print(f"Pairwise Wilcoxon signed-rank on {metric} (Bonferroni-corrected):")
        print(format_pairwise_table(pairwise_wilcoxon(summaries, metric=metric)))

    # Reward distribution per method (across all seeds, flattened).
    print("Per-batch reward distribution (flattened across seeds):")
    print(
        f"  {'method':<24s} {'n':>7s} {'min':>7s} {'p10':>7s} "
        f"{'p50':>7s} {'p90':>7s} {'max':>7s} {'mean':>7s} {'std':>7s}"
    )
    reward_summary: dict[str, dict] = {}
    for method in args.methods:
        flat = [r for seed_list in reward_samples[method] for r in seed_list]
        h = _reward_histogram(flat)
        reward_summary[method] = h
        if h["n"] == 0:
            print(f"  {method:<24s}  (no reward samples — naive control)")
            continue
        print(
            f"  {method:<24s} {h['n']:>7d} "
            f"{h['min']:>7.3f} {h['p10']:>7.3f} {h['p50']:>7.3f} "
            f"{h['p90']:>7.3f} {h['max']:>7.3f} "
            f"{h['mean']:>7.3f} {h['std']:>7.3f}"
        )

    # Sparse density at end of run (avg across seeds).
    print()
    print("End-of-run sparse density (fraction of non-zero strengths):")
    for method in args.methods:
        seed_diags = diagnostics.get(method, [])
        last_densities = []
        for d in seed_diags:
            per_task = d.get("per_task", [])
            if per_task and "sparse_density" in per_task[-1]:
                last_densities.append(per_task[-1]["sparse_density"])
        if last_densities:
            mean_density = sum(last_densities) / len(last_densities)
            print(f"  {method:<24s} {mean_density:.3f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    path = output_dir / (
        f"{ts}_12_audit_fixes_T{args.num_tasks}.json"
    )
    payload: dict[str, Any] = {
        "experiment": "12_audit_fixes",
        "timestamp": ts,
        "config": vars(args),
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
        "pairwise_accuracy": [
            asdict(c)
            for c in pairwise_wilcoxon(summaries, metric="average_accuracy")
        ],
        "pairwise_forgetting": [
            asdict(c)
            for c in pairwise_wilcoxon(
                summaries, metric="average_forgetting"
            )
        ],
        "diagnostics": diagnostics,
        "reward_summary": reward_summary,
        "method_times_seconds": method_times,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Saved run log to {path}")


if __name__ == "__main__":
    main()
