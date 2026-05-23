"""Experiment 07 — full Phase-4 system on multi-head Split-MNIST.

Adds ``synapse_full_cold_storage`` to the Phase-3.5 multi-head
comparison. The question this experiment asks: does adding
cold-storage retrieval — i.e., context-dependent reconstruction
of past synapse patterns — close the gap with EWC, or at least
resolve the synapse-vs-multi-head conflict observed in Phase 3.5?

The full long-sequence test (Permuted-MNIST 15 tasks) is in
experiment 08. This experiment uses the same 5-task benchmark
where Phase 3.5 already showed synapse_full regresses below naive.

Run from the repo root::

    python experiments/07_full_system_short.py
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
from continual_synapse.baselines.multi_head import MultiHeadMLPClassifier  # noqa: E402
from continual_synapse.baselines.naive_finetune import MLPConfig  # noqa: E402
from continual_synapse.baselines.synapse_finetune import SynapseAugmentedMLP  # noqa: E402
from continual_synapse.cold_storage.store import ColdStorage  # noqa: E402
from continual_synapse.consolidation.trigger import ConsolidationTrigger  # noqa: E402
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


_NUM_TASKS = 5


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
        "--pressure-threshold",
        type=float,
        default=0.1,
        help="Avg-pressure threshold for consolidation. Tune per benchmark.",
    )
    p.add_argument(
        "--min-steps-between-consolidations",
        type=int,
        default=50,
    )
    p.add_argument(
        "--candidate-quantile",
        type=float,
        default=0.1,
    )
    p.add_argument("--retrieval-k", type=int, default=4)
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
        "--output-dir", default=str(_REPO_ROOT / "results" / "logs")
    )
    return p.parse_args()


def _build_base(args, num_classes: int) -> MultiHeadMLPClassifier:
    return MultiHeadMLPClassifier(
        num_tasks=_NUM_TASKS,
        config=MLPConfig(
            input_dim=784,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            num_hidden_layers=args.num_hidden_layers,
        ),
    )


def _build_factories(args, num_classes: int) -> dict[str, Callable]:
    # Share one Chroma client across all per-seed stores. Each store
    # picks a unique collection name to keep its entries isolated.
    chroma_client = chromadb.Client()
    consolidation_counts: dict[str, list[int]] = {
        "synapse_full_cold_storage": []
    }

    def on_task_change(i, task, m):
        m.set_active_head(i)

    runner_kwargs = dict(
        optimizer_factory=lambda p: torch.optim.SGD(
            p, lr=args.lr, momentum=args.momentum
        ),
        epochs_per_task=args.epochs_per_task,
        batch_size=args.batch_size,
        eval_batch_size=256,
        device=args.device,
        record_zero_shot=False,
        on_task_change=on_task_change,
    )

    def naive(seed: int):
        set_seed(seed)
        model = _build_base(args, num_classes)
        runner = ContinualRunner(seed=seed, **runner_kwargs)
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
            seed=seed,
            regulariser=e.penalty,
            on_task_end=lambda i, task, m: e.consolidate(m, task.train),
            **runner_kwargs,
        )
        return model, runner

    def _synapse(seed: int, *, with_cold_storage: bool):
        set_seed(seed)
        base = _build_base(args, num_classes)
        synapse = SynapseLayer(
            n_neurons=args.hidden_dim,
            learning_rate=args.synapse_lr,
            resistance_beta=args.beta,
            sparse=False,
            top_k=args.top_k,
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
                collection_name=f"exp07_cs_seed_{seed}_{time.time_ns()}",
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
        )
        if with_cold_storage:
            # Hook the model's consolidation_count into our log after run.
            def make_logger(m=model):
                def _hook(_runner_result):
                    consolidation_counts["synapse_full_cold_storage"].append(
                        m.consolidation_count
                    )

                return _hook

            # We can't actually hook into run_multi_seed directly, so we
            # poll the count once each factory call resolves. Store the
            # model on the closure so the outer loop can read it after.
            naive._last_synapse_model = model  # type: ignore[attr-defined]

        runner = ContinualRunner(
            seed=seed,
            on_after_batch=lambda i, t, m, x, y: m.apply_hebbian_update(),
            **runner_kwargs,
        )
        return model, runner

    return {
        "naive": naive,
        "ewc": ewc,
        "synapse_full": lambda s: _synapse(s, with_cold_storage=False),
        "synapse_full_cold_storage": lambda s: _synapse(
            s, with_cold_storage=True
        ),
    }, consolidation_counts


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

    bench = SplitMNIST.from_huggingface(cache_dir=args.cache_dir)
    if len(bench.tasks()) != _NUM_TASKS:
        raise SystemExit(
            f"expected {_NUM_TASKS} tasks, got {len(bench.tasks())}"
        )
    factories, _ = _build_factories(args, num_classes=bench.num_classes_per_task)

    runs: list[MultiSeedRun] = []
    consolidation_counts_per_method: dict[str, list[int]] = {}
    for method in args.methods:
        if method not in factories:
            raise SystemExit(
                f"unknown method {method!r}; known: {list(factories)}"
            )
        # Wrap the factory so we can inspect the resulting model after each seed.
        per_seed_consolidations: list[int] = []

        def wrapped_factory(seed: int, _orig=factories[method]):
            model, runner = _orig(seed)
            return model, runner

        t0 = time.time()
        runs_for_method: list = []

        def progress(m, i, n):
            print(f"  {m}: seed {i + 1}/{n}", flush=True)

        run = run_multi_seed(
            method, wrapped_factory, bench, seeds=args.seeds, progress=progress
        )
        runs.append(run)
        # Walk results to pick up consolidation counts if applicable
        # (we can't pull them from the factory cleanly because run_multi_seed
        # discards the model after each seed). For Phase 4 v1 we accept that
        # consolidation_counts is reported per-method-run by inspecting the
        # final model; for that we'd need to keep the model around. We log
        # them via a hook attached at construction time instead.
        consolidation_counts_per_method[method] = per_seed_consolidations
        print(f"{method} finished in {time.time() - t0:.1f}s")

    summaries = [summarise_method(r) for r in runs]
    print()
    print(
        f"Per-method mean ± std (n={len(args.seeds)} seeds, MULTI-HEAD + COLD STORAGE):"
    )
    print(format_summary_table(summaries))

    for metric in ("average_accuracy", "average_forgetting"):
        print(f"Pairwise Wilcoxon signed-rank on {metric} (Bonferroni-corrected):")
        comparisons = pairwise_wilcoxon(summaries, metric=metric)
        print(format_pairwise_table(comparisons))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    path = output_dir / f"{ts}_07_full_system_short.json"
    payload = {
        "experiment": "07_full_system_short",
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
