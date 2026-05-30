"""Experiment 05 — multi-seed comparison of all Phase-1/2/3 methods.

Implements the protocol from PROJECT_PLAN.md §8.2: ≥5 seeds per
method, mean ± std, pairwise Wilcoxon signed-rank with Bonferroni
correction. The headline question is whether the Phase-3 synapse
system *statistically significantly* beats the naive baseline and
EWC on Split-MNIST.

Methods compared:
- naive: Phase-1 baseline.
- ewc: Phase-1 EWC (λ=1000, fisher_sample_size=500).
- synapse_resistance: Phase-3 with β=10, no reward mixer.
- synapse_full: Phase-3 with β=10 + RewardMixer
  (external + consistency + surprise).
- synapse_full_sparse: synapse_full with sparse top-k partner
  selection (k=64).

Run from the repo root::

    python experiments/05_multi_seed_comparison.py --seeds 0 1 2 3 4

A combined JSON log is written to ``results/logs/`` capturing
every per-seed RunResult and the statistical summary.
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

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from continual_synapse.baselines.ewc import EWC  # noqa: E402
from continual_synapse.baselines.naive_finetune import MLPClassifier, MLPConfig  # noqa: E402
from continual_synapse.baselines.synapse_finetune import SynapseAugmentedMLP  # noqa: E402
from continual_synapse.evaluation.benchmarks import SplitMNIST  # noqa: E402
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
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs-per-task", type=int, default=2)
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
    p.add_argument("--top-k", type=int, default=64)
    p.add_argument(
        "--methods",
        nargs="+",
        default=[
            "naive",
            "ewc",
            "synapse_resistance",
            "synapse_full",
            "synapse_full_sparse",
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
        "--output-dir", default=str(_REPO_ROOT / "results" / "logs")
    )
    return p.parse_args()


def _build_mlp(args, num_classes: int) -> MLPClassifier:
    return MLPClassifier(
        MLPConfig(
            input_dim=784,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            num_hidden_layers=args.num_hidden_layers,
        )
    )


def _build_factories(args, num_classes: int) -> dict[str, Callable]:
    """Return a dict of method-name -> factory(seed) -> (model, runner)."""

    def naive(seed: int):
        set_seed(seed)
        model = _build_mlp(args, num_classes)
        runner = ContinualRunner(
            optimizer_factory=lambda p: torch.optim.SGD(
                p, lr=args.lr, momentum=args.momentum
            ),
            epochs_per_task=args.epochs_per_task,
            batch_size=args.batch_size,
            device=args.device,
            seed=seed,
        )
        return model, runner

    def ewc(seed: int):
        set_seed(seed)
        model = _build_mlp(args, num_classes)
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
            device=args.device,
            seed=seed,
            regulariser=e.penalty,
            on_task_end=lambda i, task, m: e.consolidate(m, task.train),
        )
        return model, runner

    def _synapse(
        seed: int,
        *,
        with_reward: bool,
        sparse: bool,
    ):
        set_seed(seed)
        base = _build_mlp(args, num_classes)
        synapse = SynapseLayer(
            n_neurons=args.hidden_dim,
            learning_rate=args.synapse_lr,
            resistance_beta=args.beta,
            sparse=sparse,
            top_k=args.top_k,
        )
        modulator = SynapseModulation(init_gate=0.0)
        reward_computer = None
        if with_reward:
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
        model = SynapseAugmentedMLP(
            base, synapse, modulator, reward_computer=reward_computer
        )
        runner = ContinualRunner(
            optimizer_factory=lambda p: torch.optim.SGD(
                p, lr=args.lr, momentum=args.momentum
            ),
            epochs_per_task=args.epochs_per_task,
            batch_size=args.batch_size,
            device=args.device,
            seed=seed,
            on_after_batch=lambda i, t, m, x, y: m.apply_hebbian_update(),
        )
        return model, runner

    return {
        "naive": naive,
        "ewc": ewc,
        "synapse_resistance": lambda s: _synapse(
            s, with_reward=False, sparse=False
        ),
        "synapse_full": lambda s: _synapse(
            s, with_reward=True, sparse=False
        ),
        "synapse_full_sparse": lambda s: _synapse(
            s, with_reward=True, sparse=True
        ),
    }


def _multi_seed_to_jsonable(run: MultiSeedRun) -> dict:
    """JSON-ready form of a MultiSeedRun (no NaN, no numpy types)."""
    out: dict = {
        "method": run.method,
        "seeds": run.seeds,
        "results": [],
    }
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

    bench = SplitMNIST.from_huggingface(cache_dir=args.cache_dir)
    factories = _build_factories(args, num_classes=bench.num_classes_per_task)

    runs: list[MultiSeedRun] = []
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
        runs.append(run)
        print(f"{method} finished in {time.time() - t0:.1f}s")

    summaries = [summarise_method(r) for r in runs]
    print()
    print("Per-method mean ± std (n={} seeds each):".format(len(args.seeds)))
    print(format_summary_table(summaries))

    for metric in ("average_accuracy", "average_forgetting"):
        print(f"Pairwise Wilcoxon signed-rank on {metric} (Bonferroni-corrected):")
        comparisons = pairwise_wilcoxon(summaries, metric=metric)
        print(format_pairwise_table(comparisons))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    path = output_dir / f"{ts}_05_multi_seed_comparison.json"
    payload = {
        "experiment": "05_multi_seed_comparison",
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
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Saved run log to {path}")


if __name__ == "__main__":
    main()
