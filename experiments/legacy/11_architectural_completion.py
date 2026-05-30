"""Experiment 11 — architectural completion.

The 2026-05-23 audit surfaced two silent simplifications in the
synapse + cold-storage architecture relative to DESIGN.md /
PROJECT_PLAN.md:

1. Multi-pass averaging (PROJECT_PLAN.md §4.2.1) was never built.
   Phase 2 v1 simplified to single-pass batch-mean and the
   simplification was never logged.
2. Compression schedule re-evaluation (DESIGN.md §3.3) is a lookup
   function called once at insertion time with hardcoded
   ``age=0, access_count=0``. Every stored entry stays at 32-bit
   forever; the precision tiers never activate.

Both gaps are now closed in the codebase (commits 22cde31 +
e2d5884 + 692a604). This experiment re-runs the Phase 4b
comparison with the spec-complete architecture to quantify how
much of the prior "no benefit" finding was an artifact of those
simplifications versus the underlying mechanism.

Methods compared (5 seeds, configurable task count):

- naive: Phase-1 baseline.
- cs_current: cold-storage variant matching Phase 4b — single-pass,
  no compression sweep.
- cs_multi_pass: cold-storage variant + n_passes=5 (multi-pass on,
  sweep off).
- cs_sweep: cold-storage variant + compression sweep on (multi-pass
  off).
- cs_full: spec-complete — multi-pass + sweep both on.

Per-method diagnostics:
- Per-task accuracy matrix (already standard).
- Consolidation cycle counts (already standard).
- Compression sweep counts and final-sweep precision distribution
  (new; only meaningful when sweep is enabled).
- Total stored-document byte size per task (new; reveals whether
  compression actually bounds memory growth).

Run from the repo root, e.g.::

    python experiments/11_architectural_completion.py --num-tasks 15

Output JSON in ``results/logs/architectural_completion/``.
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
    "cs_current",
    "cs_multi_pass",
    "cs_sweep",
    "cs_full",
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
    p.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout in the base MLP. Must be >0 for multi-pass to do "
        "anything (deterministic forwards average to themselves).",
    )
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
    p.add_argument(
        "--n-passes",
        type=int,
        default=5,
        help="Multi-pass observation count for cs_multi_pass and cs_full.",
    )
    p.add_argument(
        "--compression-sweep-interval",
        type=int,
        default=100,
        help="Apply the compression schedule sweep every N apply_hebbian_update "
        "calls (for cs_sweep and cs_full). 0 disables.",
    )
    p.add_argument(
        "--age-thresholds",
        type=int,
        nargs="+",
        default=[100, 500, 2000],
        help="Age thresholds for the compression schedule (used by sweep).",
    )
    p.add_argument(
        "--permutation-seed",
        type=int,
        default=42,
        help="Seed for generating PermutedMNIST permutations. Shared across "
        "all methods and seeds so every run sees the same tasks.",
    )
    p.add_argument(
        "--methods", nargs="+", default=list(_DEFAULT_METHODS)
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
        default=str(_REPO_ROOT / "results" / "logs" / "architectural_completion"),
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
    """Build the schedule used by the sweep variants."""
    n_thresholds = len(args.age_thresholds)
    if n_thresholds == 3:
        tiers = (32, 16, 8, 4)
    else:
        # Map any threshold count by taking the first n+1 tiers in order.
        all_tiers = (32, 16, 8, 4)
        tiers = all_tiers[: n_thresholds + 1]
    return CompressionSchedule(
        age_thresholds=tuple(args.age_thresholds),
        tier_precisions=tiers,
    )


def _store_byte_size(store: ColdStorage) -> int:
    """Total bytes of decompressed document blobs (proxy for memory)."""
    return sum(
        len(base64.b64decode(e.document)) for e in store.all_entries()
    )


def _build_factories(
    args, num_classes: int
) -> tuple[dict[str, Callable], dict[str, list[dict]]]:
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

    def _synapse(
        seed: int,
        method_key: str,
        *,
        n_passes: int,
        sweep_interval: int,
    ):
        set_seed(seed)
        base = _build_mlp(args, num_classes)
        synapse = SynapseLayer(
            n_neurons=args.hidden_dim,
            learning_rate=args.synapse_lr,
            resistance_beta=args.beta,
            sparse=False,
            n_passes=n_passes,
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
            collection_name=f"exp11_{method_key}_seed_{seed}_{time.time_ns()}",
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
            n_passes=n_passes,
            compression_sweep_interval=sweep_interval,
            compression_schedule=_make_schedule(args),
        )

        per_task_diag: list[dict] = []

        def on_task_end_diag(i, task, m):
            if hasattr(m, "consolidation_count"):
                per_task_diag.append(
                    {
                        "task_index": int(i),
                        "consolidation_count": int(m.consolidation_count),
                        "store_count": int(m.cold_storage.count()),
                        "compression_sweep_count": int(
                            m.compression_sweep_count
                        ),
                        "store_byte_size": int(
                            _store_byte_size(m.cold_storage)
                        ),
                        "last_compression_counts": {
                            int(k): int(v)
                            for k, v in m.last_compression_counts.items()
                        },
                    }
                )

        runner = ContinualRunner(
            seed=seed,
            on_after_batch=lambda i, t, m, x, y: m.apply_hebbian_update(),
            on_task_end=on_task_end_diag,
            **runner_kwargs,
        )
        diagnostics[method_key].append(
            {"seed": seed, "per_task": per_task_diag}
        )
        return model, runner

    factories: dict[str, Callable] = {
        "naive": naive,
        "cs_current": lambda s: _synapse(
            s, "cs_current", n_passes=1, sweep_interval=0
        ),
        "cs_multi_pass": lambda s: _synapse(
            s, "cs_multi_pass", n_passes=args.n_passes, sweep_interval=0
        ),
        "cs_sweep": lambda s: _synapse(
            s,
            "cs_sweep",
            n_passes=1,
            sweep_interval=args.compression_sweep_interval,
        ),
        "cs_full": lambda s: _synapse(
            s,
            "cs_full",
            n_passes=args.n_passes,
            sweep_interval=args.compression_sweep_interval,
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
            fwt = (
                f", fwt={s.forward_transfer:+.3f}"
                if s.forward_transfer is not None
                else ""
            )
            print(
                f"    {m} seed {i + 1}/{n} (seed={seed}, "
                f"{elapsed:.0f}s into method): "
                f"acc={s.average_accuracy:.3f}, "
                f"fgt={s.average_forgetting:+.3f}, "
                f"bwt={s.backward_transfer:+.3f}{fwt}",
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
        comparisons = pairwise_wilcoxon(summaries, metric=metric)
        print(format_pairwise_table(comparisons))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    path = output_dir / (
        f"{ts}_11_architectural_completion_T{args.num_tasks}.json"
    )
    payload: dict[str, Any] = {
        "experiment": "11_architectural_completion",
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
        "method_times_seconds": method_times,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Saved run log to {path}")


if __name__ == "__main__":
    main()
