"""Experiment 08 — long-sequence decisive test for the synapse layer.

15-task Permuted-MNIST with a shared 10-class head — the regime
where the synapse layer's working memory should genuinely
saturate and cold-storage retrieval has the opportunity to make
a measurable difference. This experiment is the test the
project's architectural call hinges on:

- Beating EWC absolutely on this benchmark is unlikely; that
  outcome would be a major finding.
- Cold storage preserving long-term accuracy noticeably better
  than the bare synapse layer (even if both trail EWC) is the
  defensible positive result.
- All methods degrading similarly is the negative-results
  trigger.

Runtime expectation: ~20 minutes on CPU for 4 methods × 5 seeds.

Run from the repo root::

    python experiments/08_long_sequence_decisive.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import chromadb
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from continual_synapse.baselines.ewc import EWC  # noqa: E402
from continual_synapse.baselines.naive_finetune import MLPClassifier, MLPConfig  # noqa: E402
from continual_synapse.baselines.synapse_finetune import SynapseAugmentedMLP  # noqa: E402
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
    p.add_argument("--ewc-lam", type=float, default=1000.0)
    p.add_argument("--ewc-fisher-samples", type=int, default=500)
    p.add_argument("--synapse-lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=10.0)
    p.add_argument("--gamma", type=float, default=1e-3)
    p.add_argument("--w-consistency", type=float, default=1.0)
    p.add_argument("--w-surprise", type=float, default=0.5)
    p.add_argument(
        "--pressure-threshold",
        type=float,
        default=0.005,
        help="Phase-4 found 0.005 fires reasonably on Split-MNIST; "
        "long-sequence may need adjustment.",
    )
    p.add_argument(
        "--min-steps-between-consolidations", type=int, default=30
    )
    p.add_argument("--candidate-quantile", type=float, default=0.05)
    p.add_argument("--retrieval-k", type=int, default=4)
    p.add_argument(
        "--retrieval-refresh-interval",
        type=int,
        default=20,
        help="Refresh retrieval cache every N forwards. Larger = less "
        "Chroma overhead at the cost of staler retrievals.",
    )
    p.add_argument(
        "--permutation-seed",
        type=int,
        default=42,
        help="Seed for generating PermutedMNIST permutations. Shared "
        "across all methods and seeds so every run sees the same tasks.",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=[
            "naive",
            "ewc",
            "synapse_full",
            "synapse_full_cold_storage",
        ],
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
        default=str(_REPO_ROOT / "results" / "logs" / "long_sequence"),
    )
    return p.parse_args()


def _build_base(args, num_classes: int) -> MLPClassifier:
    """Shared-head MLP for PermutedMNIST."""
    return MLPClassifier(
        MLPConfig(
            input_dim=784,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            num_hidden_layers=args.num_hidden_layers,
        )
    )


def _build_factories(
    args, num_classes: int
) -> tuple[dict[str, Callable], dict[str, list[dict]]]:
    """Return (factories, diagnostics_per_method).

    The diagnostics dict gains one list per method, with one dict per
    seed capturing model-side bookkeeping (consolidation count,
    cold-storage entry count).
    """
    chroma_client = chromadb.Client()
    diagnostics: dict[str, list[dict]] = {m: [] for m in args.methods}

    def naive(seed: int):
        set_seed(seed)
        model = _build_base(args, num_classes)
        runner = ContinualRunner(
            optimizer_factory=lambda p: torch.optim.SGD(
                p, lr=args.lr, momentum=args.momentum
            ),
            epochs_per_task=args.epochs_per_task,
            batch_size=args.batch_size,
            eval_batch_size=256,
            device=args.device,
            seed=seed,
            record_zero_shot=True,
        )
        diagnostics["naive"].append({"seed": seed})
        return model, runner

    def ewc(seed: int):
        set_seed(seed)
        model = _build_base(args, num_classes)
        e = EWC(
            lam=args.ewc_lam,
            fisher_sample_size=args.ewc_fisher_samples,
            device=args.device,
        )
        runner = ContinualRunner(
            optimizer_factory=lambda p: torch.optim.SGD(
                p, lr=args.lr, momentum=args.momentum
            ),
            epochs_per_task=args.epochs_per_task,
            batch_size=args.batch_size,
            eval_batch_size=256,
            device=args.device,
            seed=seed,
            record_zero_shot=True,
            regulariser=e.penalty,
            on_task_end=lambda i, task, m: e.consolidate(m, task.train),
        )
        diagnostics["ewc"].append({"seed": seed})
        return model, runner

    def _synapse(seed: int, *, with_cold_storage: bool):
        set_seed(seed)
        base = _build_base(args, num_classes)
        synapse = SynapseLayer(
            n_neurons=args.hidden_dim,
            learning_rate=args.synapse_lr,
            resistance_beta=args.beta,
            sparse=False,
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

        cold_storage = None
        trigger = None
        if with_cold_storage:
            cold_storage = ColdStorage(
                collection_name=f"exp08_cs_seed_{seed}_{time.time_ns()}",
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
        )

        # Track per-task consolidation counts and storage size via on_task_end.
        per_task_diag: list[dict] = []

        def on_task_end_diag(i, task, m):
            if hasattr(m, "consolidation_count"):
                per_task_diag.append(
                    {
                        "task_index": int(i),
                        "consolidation_count": int(m.consolidation_count),
                        "store_count": int(
                            m.cold_storage.count()
                            if m.cold_storage is not None
                            else 0
                        ),
                    }
                )

        runner = ContinualRunner(
            optimizer_factory=lambda p: torch.optim.SGD(
                p, lr=args.lr, momentum=args.momentum
            ),
            epochs_per_task=args.epochs_per_task,
            batch_size=args.batch_size,
            eval_batch_size=256,
            device=args.device,
            seed=seed,
            record_zero_shot=True,
            on_after_batch=lambda i, t, m, x, y: m.apply_hebbian_update(),
            on_task_end=on_task_end_diag,
        )
        method_key = (
            "synapse_full_cold_storage" if with_cold_storage else "synapse_full"
        )
        diagnostics[method_key].append(
            {"seed": seed, "per_task": per_task_diag}
        )
        return model, runner

    return {
        "naive": naive,
        "ewc": ewc,
        "synapse_full": lambda s: _synapse(s, with_cold_storage=False),
        "synapse_full_cold_storage": lambda s: _synapse(
            s, with_cold_storage=True
        ),
    }, diagnostics


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
        f"(permutation seed={args.permutation_seed})..."
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
        run = run_multi_seed(
            method,
            factories[method],
            bench,
            seeds=args.seeds,
            progress=lambda m, i, n: print(
                f"  {m}: seed {i + 1}/{n}", flush=True
            ),
        )
        elapsed = time.time() - t0
        method_times[method] = elapsed
        runs.append(run)
        print(f"{method} finished in {elapsed:.1f}s")

    summaries = [summarise_method(r) for r in runs]
    print()
    print(
        f"Per-method mean ± std (n={len(args.seeds)} seeds, "
        f"PERMUTED-MNIST {args.num_tasks} tasks):"
    )
    print(format_summary_table(summaries))

    for metric in ("average_accuracy", "average_forgetting"):
        print(f"Pairwise Wilcoxon signed-rank on {metric} (Bonferroni-corrected):")
        comparisons = pairwise_wilcoxon(summaries, metric=metric)
        print(format_pairwise_table(comparisons))

    # Cold-storage diagnostics summary.
    cs_diag = diagnostics.get("synapse_full_cold_storage", [])
    if cs_diag and any(d.get("per_task") for d in cs_diag):
        print("Cold-storage cycle counts per seed (final / total entries):")
        for d in cs_diag:
            per_task = d.get("per_task", [])
            if not per_task:
                continue
            final = per_task[-1]
            print(
                f"  seed {d['seed']}: "
                f"consolidations={final['consolidation_count']}, "
                f"store_count={final['store_count']}"
            )
        print()

    print("Method run-time (seconds):")
    for m, t in method_times.items():
        print(f"  {m}: {t:.1f}")
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    path = output_dir / f"{ts}_08_long_sequence_decisive.json"
    payload = {
        "experiment": "08_long_sequence_decisive",
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
