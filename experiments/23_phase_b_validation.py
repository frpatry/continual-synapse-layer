"""Experiment 23 — Phase B hyperparameter validation.

Targeted 5-seed validation of three ``cs_gated_cosine_developmental``
configurations identified by the Phase A scout (exp 22):

- ``scout_mat100_validated``: maturity_target=100, alpha=0.9
- ``scout_a095_validated``:   maturity_target=50,  alpha=0.95
- ``scout_combined``:         maturity_target=100, alpha=0.95   (new combination)

All other hyperparameters held at the exp 19 / exp 21 baseline.
The Phase A scout established that compression-related
hyperparameters (``age_thresholds`` and
``compression_sweep_interval``) have no measurable effect on ACC,
FGT, or Task-0 retention at T=15 on Permuted-MNIST — documented
in the 2026-05-25 decisions_log entry. Phase B therefore holds
them at defaults.

Benchmarks: Permuted-MNIST at T=15 (short, baseline regime) and
T=50 (long, where catastrophic forgetting becomes brutal — the
regime exp 21 showed is where the developmental variant earns
its keep).

Baseline reuse: the script does NOT re-run naive / ewc /
cs_gated_cosine_developmental — those are reused from exp 21's
T=15 and T=50 JSONs via ``--baseline-log-t15`` and
``--baseline-log-t50``. When supplied, the script computes
Wilcoxon pairwise (each Phase-B config vs baseline
cs_gated_cosine_developmental, Bonferroni × 3 pairs per task
length) and prints a combined comparison table.

Output: two JSONs under ``results/logs/phase_b_validation/``,
one per task length, with the same schema as exp 21
(per-method/seed accuracy matrices, summaries, diagnostics,
pairwise Wilcoxon among the 3 new configs). Per-config
checkpointing inside each length sweep so a kill mid-run
preserves completed configs.

Estimated runtime:
- T=15: 3 configs × 5 seeds × ~3 min  = ~45 min
- T=50: 3 configs × 5 seeds × ~20 min = ~5 hours
- Total: ~6 hours wall-clock.

Run from the repo root::

    python experiments/23_phase_b_validation.py --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
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
    MethodSummary,
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


@dataclass
class PhaseBConfig:
    name: str
    maturity_target_consolidations: int
    gradient_gating_alpha: float


_CONFIGS: tuple[PhaseBConfig, ...] = (
    PhaseBConfig("scout_mat100_validated",
                 maturity_target_consolidations=100,
                 gradient_gating_alpha=0.9),
    PhaseBConfig("scout_a095_validated",
                 maturity_target_consolidations=50,
                 gradient_gating_alpha=0.95),
    PhaseBConfig("scout_combined",
                 maturity_target_consolidations=100,
                 gradient_gating_alpha=0.95),
)
_TASK_LENGTHS = (15, 50)
_BASELINE_METHOD = "cs_gated_cosine_developmental"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument(
        "--configs", nargs="+", default=[c.name for c in _CONFIGS],
        help="Subset of Phase-B configurations to run.",
    )
    p.add_argument(
        "--task-lengths", type=int, nargs="+", default=list(_TASK_LENGTHS),
    )
    p.add_argument(
        "--baseline-log-t15", type=Path, default=None,
        help="Path to exp 21's T=15 scaling JSON for Wilcoxon vs baseline "
             "cs_gated_cosine_developmental at T=15.",
    )
    p.add_argument(
        "--baseline-log-t50", type=Path, default=None,
        help="Path to exp 21's T=50 scaling JSON for Wilcoxon vs baseline "
             "at T=50.",
    )
    # ---- Training hyperparameters (mirror exp 21 / exp 19) ----
    p.add_argument("--epochs-per-task", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-hidden-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--permutation-seed", type=int, default=42)
    # ---- Synapse + cold storage (held at defaults — see decisions_log
    # ---- 2026-05-25 Phase A null finding) ----
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
        default=str(_REPO_ROOT / "results" / "logs" / "phase_b_validation"),
    )
    return p.parse_args()


# ---------- shared helpers ----------


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


# ---------- factory builder for one Phase-B config ----------


def _build_factory(
    cfg: PhaseBConfig, args, num_classes: int, chroma_client
) -> tuple[Callable[[int], tuple[Any, ContinualRunner]], list[dict]]:
    diagnostics: list[dict] = []

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

    def factory(seed: int):
        set_seed(seed)
        base = MLPClassifier(
            MLPConfig(
                input_dim=784,
                hidden_dim=args.hidden_dim,
                num_classes=num_classes,
                num_hidden_layers=args.num_hidden_layers,
                dropout=args.dropout,
            )
        )
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
            collection_name=f"exp23_{cfg.name}_seed_{seed}_{time.time_ns()}",
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
            gradient_gating_alpha=cfg.gradient_gating_alpha,
            familiarity_mode="cosine",
            maturity_target_consolidations=cfg.maturity_target_consolidations,
        )

        per_task_diag: list[dict] = []
        task_state: dict[str, Any] = {
            "familiarity_sum": 0.0, "familiarity_count": 0,
            "gradient_scale_sum": 0.0, "gradient_scale_count": 0,
            "maturity_sum": 0.0, "maturity_count": 0,
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
                    "modulator_gate": float(m.modulator.gate.item()),
                }
            )
            for k in (
                "familiarity_sum", "familiarity_count",
                "gradient_scale_sum", "gradient_scale_count",
                "maturity_sum", "maturity_count",
            ):
                task_state[k] = 0 if "count" in k else 0.0

        runner = ContinualRunner(
            seed=seed,
            on_after_batch=on_after_batch,
            on_task_end=on_task_end,
            on_pre_optimizer_step=on_pre_step,
            **runner_kwargs,
        )
        diagnostics.append({"seed": seed, "per_task": per_task_diag})
        return model, runner

    return factory, diagnostics


# ---------- per-task-length driver ----------


def _build_payload(
    args: argparse.Namespace,
    ts: int,
    num_tasks: int,
    runs: list[MultiSeedRun],
    summaries: list,
    diagnostics_by_config: dict[str, list[dict]],
    method_times: dict[str, float],
    *,
    is_partial: bool,
    configs_completed: list[str],
    configs_requested: list[str],
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
        "experiment": "23_phase_b_validation",
        "num_tasks": int(num_tasks),
        "timestamp": ts,
        "config": vars(args),
        "is_partial": is_partial,
        "configs_completed": list(configs_completed),
        "configs_requested": list(configs_requested),
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
        "diagnostics": diagnostics_by_config,
        "method_times_seconds": method_times,
    }


def _run_task_length(
    num_tasks: int,
    args: argparse.Namespace,
    configs: list[PhaseBConfig],
    output_path: Path,
) -> tuple[list[MultiSeedRun], list, dict[str, list[dict]]]:
    ts = int(time.time())
    print(f"\n=== Phase B  T={num_tasks} ===", flush=True)
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
    # Per-task-length Chroma client so collections from prior task lengths
    # don't pile up across the long sweep.
    chroma_client = chromadb.Client()

    runs: list[MultiSeedRun] = []
    summaries: list = []
    method_times: dict[str, float] = {}
    diagnostics_by_config: dict[str, list[dict]] = {}
    configs_completed: list[str] = []
    configs_requested = [c.name for c in configs]

    for cfg in configs:
        print(f"\n  --- {cfg.name} (target={cfg.maturity_target_consolidations}, "
              f"alpha={cfg.gradient_gating_alpha}) ---", flush=True)
        factory, diagnostics = _build_factory(
            cfg, args, num_classes=bench.num_classes_per_task,
            chroma_client=chroma_client,
        )
        t0 = time.time()

        def _seed_done(m, i, n, seed, result):
            s = compute_metrics(result)
            elapsed = time.time() - t0
            print(
                f"    {m} seed {i + 1}/{n} (seed={seed}, "
                f"{elapsed:.0f}s into config): "
                f"acc={s.average_accuracy:.3f}, "
                f"fgt={s.average_forgetting:+.3f}",
                flush=True,
            )

        run = run_multi_seed(
            cfg.name, factory, bench, seeds=args.seeds,
            progress=lambda m, i, n: print(
                f"  T={num_tasks} {m}: seed {i + 1}/{n}", flush=True
            ),
            on_seed_complete=_seed_done,
        )
        elapsed = time.time() - t0
        method_times[cfg.name] = elapsed
        runs.append(run)
        summaries.append(summarise_method(run))
        diagnostics_by_config[cfg.name] = diagnostics
        configs_completed.append(cfg.name)
        print(f"  {cfg.name} finished in {elapsed:.1f}s", flush=True)

        is_partial = len(configs_completed) < len(configs)
        payload = _build_payload(
            args, ts, num_tasks, runs, summaries, diagnostics_by_config,
            method_times, is_partial=is_partial,
            configs_completed=configs_completed,
            configs_requested=configs_requested,
        )
        _atomic_write_json(output_path, payload)
        tag = "partial" if is_partial else "final"
        print(
            f"  Checkpoint written ({tag}, "
            f"{len(configs_completed)}/{len(configs)} configs): {output_path}",
            flush=True,
        )

    print()
    print(f"[T={num_tasks}] Per-config mean ± std (n={len(args.seeds)} seeds):")
    print(format_summary_table(summaries))

    for metric in ("average_accuracy", "average_forgetting"):
        print(
            f"[T={num_tasks}] Pairwise Wilcoxon (Phase-B configs only, "
            f"Bonferroni × 3) on {metric}:"
        )
        print(format_pairwise_table(pairwise_wilcoxon(summaries, metric=metric)))

    # Task-0 retention summary
    print()
    print(f"[T={num_tasks}] Task-0 ACC at end of training (R[T-1, 0]):")
    print(f"  {'config':<32s} {'mean':>8s} {'std':>8s}  per-seed")
    print("  " + "-" * 78)
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
        print(f"  {run.method:<32s} {mean:>8.3f} {std:>8.3f}  {per_seed}")

    return runs, summaries, diagnostics_by_config


# ---------- Wilcoxon vs baseline ----------


def _extract_method_summary(
    log_payload: dict, method_name: str
) -> MethodSummary | None:
    """Build a MethodSummary for ``method_name`` from a prior log file."""
    summary_block = None
    for s in log_payload["summaries"]:
        if s["method"] == method_name:
            summary_block = s
            break
    if summary_block is None:
        return None
    return MethodSummary(
        method=method_name,
        n_seeds=int(summary_block["n_seeds"]),
        metric_means=dict(summary_block["metric_means"]),
        metric_stds=dict(summary_block["metric_stds"]),
        per_seed_metrics={
            k: list(v) for k, v in summary_block["per_seed_metrics"].items()
        },
    )


def _wilcoxon_vs_baseline(
    summaries: list,
    baseline_log_path: Path,
    num_tasks: int,
) -> None:
    print()
    print("=" * 78)
    print(
        f"[T={num_tasks}] Wilcoxon vs baseline "
        f"{_BASELINE_METHOD!r} (from {baseline_log_path}):"
    )
    print("=" * 78)
    payload = json.loads(baseline_log_path.read_text())
    baseline = _extract_method_summary(payload, _BASELINE_METHOD)
    if baseline is None:
        print(f"  ERROR: {_BASELINE_METHOD!r} not in baseline log; skipping.")
        return
    print(f"  baseline n={baseline.n_seeds}, "
          f"ACC={baseline.metric_means.get('average_accuracy', float('nan')):.4f} "
          f"± {baseline.metric_stds.get('average_accuracy', float('nan')):.4f}")
    print()
    # One Wilcoxon comparison per (new config, baseline). Bonferroni × 3.
    n_pairs = len(summaries)
    print(f"  Pairwise (Bonferroni × {n_pairs}):")
    print(
        f"    {'config':<32s} {'metric':<22s} {'n':>3s}  "
        f"{'p_raw':>9s}  {'p_bonf':>9s}  {'sig@0.05':>9s}"
    )
    print("    " + "-" * 90)
    for metric in ("average_accuracy", "average_forgetting"):
        results_for_metric = pairwise_wilcoxon(
            [baseline] + summaries, metric=metric
        )
        # We only want comparisons that involve the baseline.
        for c in results_for_metric:
            if c.method_a != _BASELINE_METHOD and c.method_b != _BASELINE_METHOD:
                continue
            new_config = (
                c.method_b if c.method_a == _BASELINE_METHOD else c.method_a
            )
            sig = "SIG" if c.significant_05 else "n.s."
            print(
                f"    {new_config:<32s} {metric:<22s} {c.n:>3d}  "
                f"{c.p_value:>9.5f}  {c.p_value_bonferroni:>9.5f}  {sig:>9s}"
            )


def _ranking_report(
    summaries_by_T: dict[int, list],
    baseline_per_seed_by_T: dict[int, dict[str, list[float]] | None],
) -> None:
    """Print delta-vs-baseline summary for each (T, config) pair."""
    print()
    print("=" * 78)
    print("PHASE B SUMMARY: delta-vs-baseline per (T, config)")
    print("=" * 78)
    for T, summaries in summaries_by_T.items():
        baseline_per_seed = baseline_per_seed_by_T.get(T)
        baseline_acc = (
            sum(baseline_per_seed["average_accuracy"]) / len(baseline_per_seed["average_accuracy"])
            if baseline_per_seed else None
        )
        baseline_fgt = (
            sum(baseline_per_seed["average_forgetting"]) / len(baseline_per_seed["average_forgetting"])
            if baseline_per_seed else None
        )
        print(f"\n  T={T}:  baseline cs_gated_cosine_developmental "
              f"ACC={baseline_acc:.4f}, FGT={baseline_fgt:+.4f}"
              if baseline_acc is not None
              else f"\n  T={T}:  no baseline loaded")
        print(
            f"  {'config':<32s} {'ACC':>8s} {'ΔACC':>8s} "
            f"{'FGT':>8s} {'ΔFGT':>8s}"
        )
        print("  " + "-" * 72)
        for s in summaries:
            acc = s.metric_means.get("average_accuracy", float("nan"))
            fgt = s.metric_means.get("average_forgetting", float("nan"))
            dacc = acc - baseline_acc if baseline_acc is not None else float("nan")
            dfgt = fgt - baseline_fgt if baseline_fgt is not None else float("nan")
            print(
                f"  {s.method:<32s} {acc:>8.4f} {dacc:>+8.4f} "
                f"{fgt:>+8.4f} {dfgt:>+8.4f}"
            )


def main() -> None:
    args = parse_args()
    if len(args.seeds) < 2:
        raise SystemExit("multi-seed experiment needs at least 2 seeds")
    by_name = {c.name: c for c in _CONFIGS}
    configs = [by_name[name] for name in args.configs if name in by_name]
    if not configs:
        raise SystemExit(
            f"none of --configs {args.configs} are known. "
            f"Known: {[c.name for c in _CONFIGS]}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts_root = int(time.time())

    summaries_by_T: dict[int, list] = {}
    baseline_per_seed_by_T: dict[int, dict[str, list[float]] | None] = {}

    for T in args.task_lengths:
        out_path = output_dir / f"{ts_root}_23_phase_b_T{T}.json"
        runs, summaries, _ = _run_task_length(T, args, configs, out_path)
        summaries_by_T[T] = summaries

    # ---- Optional Wilcoxon vs baseline from exp 21 logs ----
    baseline_paths = {
        15: args.baseline_log_t15,
        50: args.baseline_log_t50,
    }
    for T, summaries in summaries_by_T.items():
        baseline_path = baseline_paths.get(T)
        if baseline_path is None:
            print(
                f"\n[T={T}] No --baseline-log-t{T} supplied; skipping Wilcoxon "
                f"vs baseline. Run the comparison externally if needed."
            )
            baseline_per_seed_by_T[T] = None
            continue
        if not baseline_path.exists():
            print(f"\n[T={T}] baseline log not found at {baseline_path}; skipping.")
            baseline_per_seed_by_T[T] = None
            continue
        baseline_payload = json.loads(baseline_path.read_text())
        baseline = _extract_method_summary(baseline_payload, _BASELINE_METHOD)
        baseline_per_seed_by_T[T] = (
            baseline.per_seed_metrics if baseline else None
        )
        _wilcoxon_vs_baseline(summaries, baseline_path, T)

    _ranking_report(summaries_by_T, baseline_per_seed_by_T)
    print()
    print("Output files:")
    for T in args.task_lengths:
        path = output_dir / f"{ts_root}_23_phase_b_T{T}.json"
        print(f"  T={T:>3d}: {path}")


if __name__ == "__main__":
    main()
