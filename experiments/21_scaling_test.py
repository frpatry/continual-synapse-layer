"""Experiment 21 — task-length scaling test.

Characterises how cs_gated_cosine_developmental and EWC scale to
longer Permuted-MNIST task sequences. Three task lengths
(T=15, 30, 50) × three methods × five seeds = 45 runs total.

Methods:
- naive: sequential fine-tuning, no continual-learning mechanism.
- ewc_lam_10: Elastic Weight Consolidation at λ=10 (from the
  exp 18 sweep). Fisher estimated on 500 samples per task.
- cs_gated_cosine_developmental: cosine familiarity gating with
  developmental maturity (target=50). Same flags as exp 19.

Benchmark: Permuted-MNIST at T ∈ {15, 30, 50}. Same model
hyperparameters as exp 17 / 19 (256-d hidden, 3 hidden layers,
dropout=0.5, lr=0.01, momentum=0.9, batch=64, 1 epoch/task).
Fresh model per seed via the per-seed factory pattern; the
``set_seed`` contract ensures isolation.

Per-task-length JSON outputs under
``results/logs/scaling/<ts>_21_scaling_T{T}.json`` with the same
schema as exp 17 (methods list + summaries + diagnostics +
pairwise Wilcoxon Bonferroni). Per-method checkpointing inside
each task length: a kill mid-sweep preserves each completed
method's data.

Reported per (method, length):
- ACC / FGT / BWT / FWT mean ± std across seeds.
- Full ``(T, T)`` per-task accuracy matrix per seed (so the
  Task-0 retention trajectory and any per-task analysis can be
  reconstructed post-hoc).
- Wall-clock per method per length.

The headline plot (per-task-length ACC mean ± std for each
method) is produced by ``experiments/21b_plot_scaling.py``,
which reads all three JSONs once they exist.

Run from the repo root::

    python experiments/21_scaling_test.py --seeds 0 1 2 3 4

Estimated total runtime ≈ 9-10 hours wall-clock on a MacBook
CPU. Use nohup + caffeinate to survive sleep / SIGHUP.
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


_METHODS = ("naive", "ewc_lam_10", "cs_gated_cosine_developmental")
_TASK_LENGTHS = (15, 30, 50)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--methods", nargs="+", default=list(_METHODS))
    p.add_argument(
        "--task-lengths", type=int, nargs="+", default=list(_TASK_LENGTHS),
    )
    # ---- Training hyperparameters (mirror exp 17 / 19) ----
    p.add_argument("--epochs-per-task", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-hidden-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--permutation-seed", type=int, default=42)
    # ---- EWC ----
    p.add_argument("--ewc-lam", type=float, default=10.0)
    p.add_argument("--ewc-fisher-samples", type=int, default=500)
    # ---- Synapse + cold storage (cs_gated_cosine_developmental) ----
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
    p.add_argument("--maturity-target-consolidations", type=int, default=50)
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
        default=str(_REPO_ROOT / "results" / "logs" / "scaling"),
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
    # default=str so PosixPath / other non-serialisables in args don't
    # crash the write (lesson from exp 20).
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
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
    args: argparse.Namespace,
    ts: int,
    num_tasks: int,
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
        "experiment": "21_scaling_test",
        "num_tasks": int(num_tasks),
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


# ---------- factory builder ----------


def _build_factories(
    args, num_classes: int
) -> tuple[dict[str, Callable], dict[str, list[dict]]]:
    """One factory per requested method. Self-contained chromadb client
    per task-length run so prior task lengths don't leave stale
    collections lingering across the long sweep."""
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

    def ewc_lam_10(seed: int):
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
        diagnostics["ewc_lam_10"].append(
            {"seed": seed, "per_task": per_task_diag}
        )
        return model, runner

    def cs_gated_cosine_developmental(seed: int):
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
            collection_name=f"exp21_dev_seed_{seed}_{time.time_ns()}",
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
            maturity_target_consolidations=args.maturity_target_consolidations,
        )

        per_task_diag: list[dict] = []
        task_state: dict[str, Any] = {
            "familiarity_sum": 0.0, "familiarity_count": 0,
            "gradient_scale_sum": 0.0, "gradient_scale_count": 0,
            "maturity_sum": 0.0, "maturity_count": 0,
            "max_similarity_this_task": 0.0,
        }

        def on_pre_step(i, task, m):
            scale = m.apply_gradient_gating()
            fam = float(m.last_familiarity)
            task_state["familiarity_sum"] += fam
            task_state["familiarity_count"] += 1
            task_state["gradient_scale_sum"] += float(scale)
            task_state["gradient_scale_count"] += 1
            task_state["maturity_sum"] += float(m.last_maturity)
            task_state["maturity_count"] += 1
            if fam > task_state["max_similarity_this_task"]:
                task_state["max_similarity_this_task"] = fam

        def on_after_batch(i, task, m, x, y):
            m.apply_hebbian_update()

        def on_task_end(i, task, m):
            fc = task_state["familiarity_count"]
            gc = task_state["gradient_scale_count"]
            mc = task_state["maturity_count"]
            per_task_diag.append(
                {
                    "task_index": int(i),
                    "consolidation_count": int(m.consolidation_count),
                    "store_count": int(m.cold_storage.count()),
                    "store_byte_size": int(_store_byte_size(m.cold_storage)),
                    "avg_familiarity": (
                        task_state["familiarity_sum"] / fc if fc > 0 else None
                    ),
                    "avg_gradient_scale": (
                        task_state["gradient_scale_sum"] / gc if gc > 0 else None
                    ),
                    "avg_maturity": (
                        task_state["maturity_sum"] / mc if mc > 0 else None
                    ),
                    "familiarity_max": float(m.familiarity_max),
                    "max_similarity_this_task": float(
                        task_state["max_similarity_this_task"]
                    ),
                    "modulator_gate": float(m.modulator.gate.item()),
                }
            )
            for k in (
                "familiarity_sum", "familiarity_count",
                "gradient_scale_sum", "gradient_scale_count",
                "maturity_sum", "maturity_count",
            ):
                task_state[k] = 0 if "count" in k else 0.0
            task_state["max_similarity_this_task"] = 0.0

        runner = ContinualRunner(
            seed=seed,
            on_after_batch=on_after_batch,
            on_task_end=on_task_end,
            on_pre_optimizer_step=on_pre_step,
            **runner_kwargs,
        )
        diagnostics["cs_gated_cosine_developmental"].append(
            {"seed": seed, "per_task": per_task_diag}
        )
        return model, runner

    return {
        "naive": naive,
        "ewc_lam_10": ewc_lam_10,
        "cs_gated_cosine_developmental": cs_gated_cosine_developmental,
    }, diagnostics


# ---------- per-task-length driver ----------


def _run_task_length(
    num_tasks: int,
    args: argparse.Namespace,
    output_path: Path,
) -> tuple[list[MultiSeedRun], list]:
    ts = int(time.time())
    print(f"\n=== Task length T={num_tasks} ===", flush=True)
    print(f"Checkpoint path: {output_path}", flush=True)

    print(
        f"Loading PermutedMNIST with {num_tasks} tasks "
        f"(permutation seed={args.permutation_seed}, dropout={args.dropout})..."
    )
    bench = PermutedMNIST.from_huggingface(
        num_tasks=num_tasks,
        seed=args.permutation_seed,
        cache_dir=args.cache_dir,
    )
    factories, diagnostics = _build_factories(
        args, num_classes=bench.num_classes_per_task
    )

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
                f"  T={num_tasks} {m}: seed {i + 1}/{n}", flush=True
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
            args, ts, num_tasks, runs, summaries, diagnostics, method_times,
            is_partial=is_partial,
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
        f"[T={num_tasks}] Per-method mean ± std (n={len(args.seeds)} seeds):"
    )
    print(format_summary_table(summaries))

    for metric in ("average_accuracy", "average_forgetting"):
        print(
            f"[T={num_tasks}] Pairwise Wilcoxon signed-rank on {metric} "
            f"(Bonferroni-corrected):"
        )
        print(format_pairwise_table(pairwise_wilcoxon(summaries, metric=metric)))

    # ---- Task-0 retention at end of training (the headline scaling stat) ----
    print()
    print(f"[T={num_tasks}] Task-0 ACC at end of training (R[T-1, 0]):")
    print(f"  {'method':<34s} {'mean':>8s} {'std':>8s}  per-seed")
    print("  " + "-" * 70)
    for run in runs:
        vals = [
            r.accuracy_matrix[num_tasks - 1, 0]
            for r in run.results
            if not math.isnan(r.accuracy_matrix[num_tasks - 1, 0])
        ]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        per_seed = "  ".join(f"{v:.3f}" for v in vals)
        print(f"  {run.method:<34s} {mean:>8.3f} {std:>8.3f}  {per_seed}")

    return runs, summaries


def main() -> None:
    args = parse_args()
    if len(args.seeds) < 2:
        raise SystemExit("multi-seed experiment needs at least 2 seeds")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts_root = int(time.time())

    all_results: dict[int, dict[str, Any]] = {}

    for num_tasks in args.task_lengths:
        out_path = output_dir / f"{ts_root}_21_scaling_T{num_tasks}.json"
        runs, summaries = _run_task_length(num_tasks, args, out_path)
        all_results[num_tasks] = {
            "path": out_path,
            "summaries": summaries,
        }

    # ---- Final cross-length summary (the scaling story in one table) ----
    print()
    print("=" * 90)
    print("CROSS-LENGTH ACC SUMMARY (mean ± std):")
    print(
        f"  {'method':<34s} " + "  ".join(f"T={T:>3d}" for T in args.task_lengths)
    )
    print("  " + "-" * 80)
    for method in args.methods:
        cells = [f"  {method:<34s} "]
        for T in args.task_lengths:
            res = all_results.get(T)
            if res is None:
                cells.append(f"{'—':>16s}")
                continue
            summary = next(
                (s for s in res["summaries"] if s.method == method), None
            )
            if summary is None:
                cells.append(f"{'—':>16s}")
                continue
            m = summary.metric_means.get("average_accuracy", float("nan"))
            std = summary.metric_stds.get("average_accuracy", float("nan"))
            cells.append(f"{m:.3f}±{std:.3f}    ")
        print("".join(cells))

    print()
    print("Output files:")
    for T, res in all_results.items():
        print(f"  T={T:>3d}: {res['path']}")


if __name__ == "__main__":
    main()
