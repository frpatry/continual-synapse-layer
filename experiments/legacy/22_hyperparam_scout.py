"""Experiment 22 — single-seed hyperparameter scout for the
developmental cs_gated_cosine variant.

Cross-design over four hyperparameter dimensions, each varied
independently from the exp 19 / 21 baseline
(maturity_target=50, gradient_gating_alpha=0.9, default age
thresholds (100, 500, 2000), compression_sweep_interval=100).
11 configurations total; 1 seed each for speed (~3 min per
config ⇒ ~35 min total).

Configurations:

Maturity sweep (alpha=0.9, default age thresholds, sweep=100):
  scout_mat20   target=20
  scout_mat50   target=50    (BASELINE — matches exp 19 dev)
  scout_mat100  target=100
  scout_mat200  target=200

Alpha sweep (target=50, default age thresholds, sweep=100):
  scout_a050    alpha=0.50
  scout_a070    alpha=0.70
  scout_a095    alpha=0.95
  (alpha=0.90 = scout_mat50 baseline)

Decay-schedule sweep (target=50, alpha=0.9, sweep=100):
  scout_decay_cons   age_thresholds=(500, 2000, 10000) — conservative
  scout_decay_agg    age_thresholds=(50, 200, 1000)    — aggressive
  (default (100, 500, 2000) = scout_mat50 baseline)

Compression-sweep interval (target=50, alpha=0.9, default thresholds):
  scout_sweep50    compression_sweep_interval=50
  scout_sweep200   compression_sweep_interval=200
  (default 100 = scout_mat50 baseline)

Each configuration is a separate "method" in the output JSON so
the per-method checkpointing pattern applies — partial results
survive any kill. The scout_mat50 baseline is run exactly once
(as the maturity-sweep entry) and reused as the reference point
in the rankings for the other three dimensions, which keeps the
total at 11 runs rather than 14.

Per-config metrics collected:
- ACC, FGT, BWT, FWT (standard).
- Task-0 retention R[T-1, 0] (the long-task-protection signal).
- Final consolidation_count, store_count, store_byte_size_mb.
- avg_familiarity, avg_gradient_scale across all training batches.

The end-of-run stdout produces three ranked top-3 tables:
1. Top 3 by ACC.
2. Top 3 by Task-0 retention.
3. Top 3 by combined score = ACC + 0.5 × Task-0 retention.

Output: ``results/logs/hyperparam_scout/<ts>_22_hyperparam_scout.json``
(atomic .tmp + os.replace, default=str so Path-valued args don't
crash the write).

Run from the repo root::

    python experiments/22_hyperparam_scout.py
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
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
from continual_synapse.reward.consistency import ConsistencyReward  # noqa: E402
from continual_synapse.reward.external import ExternalReward  # noqa: E402
from continual_synapse.reward.mixer import RewardMixer  # noqa: E402
from continual_synapse.reward.surprise import SurpriseReward  # noqa: E402
from continual_synapse.synapse_layer.layer import SynapseLayer  # noqa: E402
from continual_synapse.synapse_layer.modulation import SynapseModulation  # noqa: E402


# ---------- scout configurations ----------


@dataclass
class ScoutConfig:
    name: str
    maturity_target_consolidations: int = 50
    gradient_gating_alpha: float = 0.9
    age_thresholds: tuple[int, ...] = (100, 500, 2000)
    compression_sweep_interval: int = 100


# 11 configurations. scout_mat50 is the explicit baseline; the
# alpha/decay/sweep dimensions are NOT re-running mat50 as their
# baseline — they reuse it from the maturity row when ranking.
_CONFIGS: tuple[ScoutConfig, ...] = (
    # ---- Maturity sweep ----
    ScoutConfig("scout_mat20",  maturity_target_consolidations=20),
    ScoutConfig("scout_mat50",  maturity_target_consolidations=50),     # baseline
    ScoutConfig("scout_mat100", maturity_target_consolidations=100),
    ScoutConfig("scout_mat200", maturity_target_consolidations=200),
    # ---- Alpha sweep (target=50, default decay, sweep=100) ----
    ScoutConfig("scout_a050", gradient_gating_alpha=0.50),
    ScoutConfig("scout_a070", gradient_gating_alpha=0.70),
    ScoutConfig("scout_a095", gradient_gating_alpha=0.95),
    # ---- Decay-schedule sweep (target=50, alpha=0.9, sweep=100) ----
    ScoutConfig("scout_decay_cons", age_thresholds=(500, 2000, 10000)),
    ScoutConfig("scout_decay_agg",  age_thresholds=(50, 200, 1000)),
    # ---- Compression-sweep-interval sweep (target=50, alpha=0.9, default decay) ----
    ScoutConfig("scout_sweep50",  compression_sweep_interval=50),
    ScoutConfig("scout_sweep200", compression_sweep_interval=200),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scout-seed", type=int, default=0)
    p.add_argument("--num-tasks", type=int, default=15)
    p.add_argument(
        "--configs", nargs="+", default=[c.name for c in _CONFIGS],
        help="Subset of configurations to run (defaults to all 11).",
    )
    # ---- Training hyperparameters (mirror exp 19 / 21) ----
    p.add_argument("--epochs-per-task", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-hidden-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--permutation-seed", type=int, default=42)
    # ---- Synapse + cold storage (cs_gated_cosine_developmental shared) ----
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
        default=str(_REPO_ROOT / "results" / "logs" / "hyperparam_scout"),
    )
    return p.parse_args()


# ---------- helpers ----------


def _make_compression_schedule(age_thresholds: tuple[int, ...]) -> CompressionSchedule:
    n_thresholds = len(age_thresholds)
    all_tiers = (32, 16, 8, 4)
    tiers = all_tiers[: n_thresholds + 1]
    return CompressionSchedule(
        age_thresholds=tuple(age_thresholds),
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


# ---------- factory builder ----------


def _build_factory(
    cfg: ScoutConfig, args, num_classes: int, chroma_client
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
            collection_name=f"exp22_{cfg.name}_seed_{seed}_{time.time_ns()}",
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
            compression_sweep_interval=cfg.compression_sweep_interval,
            compression_schedule=_make_compression_schedule(cfg.age_thresholds),
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
        # Aggregate counters across the WHOLE run for the rankings.
        run_state: dict[str, Any] = {
            "total_familiarity_sum": 0.0, "total_familiarity_count": 0,
            "total_gradient_scale_sum": 0.0, "total_gradient_scale_count": 0,
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
            run_state["total_familiarity_sum"] += fam
            run_state["total_familiarity_count"] += 1
            run_state["total_gradient_scale_sum"] += float(scale)
            run_state["total_gradient_scale_count"] += 1

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
        diagnostics.append(
            {
                "seed": seed,
                "per_task": per_task_diag,
                "run_state": run_state,  # exposed for end-of-config aggregation
            }
        )
        return model, runner

    return factory, diagnostics


# ---------- per-config diagnostics aggregator ----------


@dataclass
class ScoutResult:
    name: str
    config: dict[str, Any] = field(default_factory=dict)
    acc: float = float("nan")
    fgt: float = float("nan")
    bwt: float = float("nan")
    fwt: float | None = None
    task0_retention: float = float("nan")
    consolidation_count: int = 0
    store_count: int = 0
    store_byte_size_mb: float = 0.0
    avg_familiarity: float = float("nan")
    avg_gradient_scale: float = float("nan")
    elapsed_seconds: float = 0.0


def _collect_scout_result(
    cfg: ScoutConfig,
    run: MultiSeedRun,
    diagnostics: list[dict],
    elapsed: float,
) -> ScoutResult:
    """Collapse the single-seed run + diagnostics into a flat result row."""
    r = run.results[0]
    summary = compute_metrics(r)
    T = r.accuracy_matrix.shape[0]
    task0_v = r.accuracy_matrix[T - 1, 0]
    sd = diagnostics[0]
    pt = sd["per_task"]
    final_pt = pt[-1] if pt else {}
    rs = sd.get("run_state", {})
    fc = rs.get("total_familiarity_count", 0)
    gc = rs.get("total_gradient_scale_count", 0)
    return ScoutResult(
        name=cfg.name,
        config={
            "maturity_target_consolidations": cfg.maturity_target_consolidations,
            "gradient_gating_alpha": cfg.gradient_gating_alpha,
            "age_thresholds": list(cfg.age_thresholds),
            "compression_sweep_interval": cfg.compression_sweep_interval,
        },
        acc=float(summary.average_accuracy),
        fgt=float(summary.average_forgetting),
        bwt=float(summary.backward_transfer),
        fwt=(
            None if summary.forward_transfer is None
            else float(summary.forward_transfer)
        ),
        task0_retention=float(task0_v) if not math.isnan(task0_v) else float("nan"),
        consolidation_count=int(final_pt.get("consolidation_count", 0)),
        store_count=int(final_pt.get("store_count", 0)),
        store_byte_size_mb=(
            float(final_pt.get("store_byte_size", 0)) / (1024 * 1024)
        ),
        avg_familiarity=(
            rs["total_familiarity_sum"] / fc if fc > 0 else float("nan")
        ),
        avg_gradient_scale=(
            rs["total_gradient_scale_sum"] / gc if gc > 0 else float("nan")
        ),
        elapsed_seconds=float(elapsed),
    )


# ---------- payload + ranking ----------


def _build_payload(
    args: argparse.Namespace,
    ts: int,
    runs: list[MultiSeedRun],
    diagnostics_by_config: dict[str, list[dict]],
    scout_results: list[ScoutResult],
    *,
    is_partial: bool,
    configs_completed: list[str],
    configs_requested: list[str],
) -> dict[str, Any]:
    return {
        "experiment": "22_hyperparam_scout",
        "timestamp": ts,
        "scout_seed": args.scout_seed,
        "num_tasks": args.num_tasks,
        "config": vars(args),
        "is_partial": is_partial,
        "configs_completed": list(configs_completed),
        "configs_requested": list(configs_requested),
        "methods": [_multi_seed_to_jsonable(r) for r in runs],
        "diagnostics": diagnostics_by_config,
        "scout_results": [asdict(r) for r in scout_results],
    }


def _print_ranking(
    scout_results: list[ScoutResult],
    key: Callable[[ScoutResult], float],
    label: str,
    n: int = 3,
) -> None:
    ranked = sorted(scout_results, key=key, reverse=True)
    print(f"\n  Top {n} by {label}:")
    print(
        f"    {'rank':>4s}  {'config':<22s}  {'ACC':>7s}  {'FGT':>7s}  "
        f"{'Task-0':>8s}  {'consol':>7s}  {'store_MB':>9s}  "
        f"{'avg_fam':>9s}  {'avg_scale':>10s}"
    )
    print("    " + "-" * 102)
    for i, r in enumerate(ranked[:n], start=1):
        print(
            f"    {i:>4d}  {r.name:<22s}  "
            f"{r.acc:>7.3f}  {r.fgt:>+7.3f}  "
            f"{r.task0_retention:>8.3f}  "
            f"{r.consolidation_count:>7d}  {r.store_byte_size_mb:>9.2f}  "
            f"{r.avg_familiarity:>9.3f}  {r.avg_gradient_scale:>10.3f}"
        )


def _print_full_table(scout_results: list[ScoutResult]) -> None:
    print("\nAll configurations (in run order):")
    print(
        f"  {'config':<22s}  {'ACC':>7s}  {'FGT':>7s}  {'Task-0':>8s}  "
        f"{'consol':>7s}  {'store_MB':>9s}  {'avg_fam':>9s}  {'avg_scale':>10s}  "
        f"{'elapsed_s':>10s}"
    )
    print("  " + "-" * 110)
    for r in scout_results:
        print(
            f"  {r.name:<22s}  "
            f"{r.acc:>7.3f}  {r.fgt:>+7.3f}  "
            f"{r.task0_retention:>8.3f}  "
            f"{r.consolidation_count:>7d}  {r.store_byte_size_mb:>9.2f}  "
            f"{r.avg_familiarity:>9.3f}  {r.avg_gradient_scale:>10.3f}  "
            f"{r.elapsed_seconds:>10.1f}"
        )


def main() -> None:
    args = parse_args()
    by_name = {c.name: c for c in _CONFIGS}
    selected = [by_name[name] for name in args.configs if name in by_name]
    if not selected:
        raise SystemExit(
            f"none of --configs {args.configs} are known. "
            f"Known: {[c.name for c in _CONFIGS]}"
        )

    print(
        f"Loading PermutedMNIST with {args.num_tasks} tasks "
        f"(permutation seed={args.permutation_seed}, dropout={args.dropout})..."
    )
    bench = PermutedMNIST.from_huggingface(
        num_tasks=args.num_tasks,
        seed=args.permutation_seed,
        cache_dir=args.cache_dir,
    )
    num_classes = bench.num_classes_per_task

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    path = output_dir / f"{ts}_22_hyperparam_scout.json"
    print(f"Checkpoint path: {path}", flush=True)
    print(f"Running {len(selected)} configurations at seed={args.scout_seed}.")

    # One Chroma client shared across all configs (matches prior scripts).
    # Per-config collection names + per-config ColdStorage instances guarantee
    # isolation of stored entries.
    chroma_client = chromadb.Client()

    runs: list[MultiSeedRun] = []
    diagnostics_by_config: dict[str, list[dict]] = {}
    scout_results: list[ScoutResult] = []
    configs_completed: list[str] = []
    configs_requested = [c.name for c in selected]

    for cfg in selected:
        print(f"\n=== {cfg.name} ===", flush=True)
        print(
            f"  maturity_target={cfg.maturity_target_consolidations}  "
            f"alpha={cfg.gradient_gating_alpha}  "
            f"age_thresholds={cfg.age_thresholds}  "
            f"sweep_interval={cfg.compression_sweep_interval}",
            flush=True,
        )
        factory, diagnostics = _build_factory(cfg, args, num_classes, chroma_client)
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
            cfg.name, factory, bench, seeds=[args.scout_seed],
            progress=lambda m, i, n: print(
                f"  {m}: seed {i + 1}/{n}", flush=True
            ),
            on_seed_complete=_seed_done,
        )
        elapsed = time.time() - t0
        runs.append(run)
        diagnostics_by_config[cfg.name] = diagnostics
        scout_results.append(_collect_scout_result(cfg, run, diagnostics, elapsed))
        configs_completed.append(cfg.name)
        print(f"  {cfg.name} finished in {elapsed:.1f}s", flush=True)

        is_partial = len(configs_completed) < len(selected)
        payload = _build_payload(
            args, ts, runs, diagnostics_by_config, scout_results,
            is_partial=is_partial,
            configs_completed=configs_completed,
            configs_requested=configs_requested,
        )
        _atomic_write_json(path, payload)
        tag = "partial" if is_partial else "final"
        print(
            f"  Checkpoint written ({tag}, "
            f"{len(configs_completed)}/{len(selected)} configs): {path}",
            flush=True,
        )

    # ---- End-of-run ranked output ----
    print()
    print("=" * 110)
    print(f"SCOUT RANKINGS  ({len(scout_results)} configurations, "
          f"Permuted-MNIST T={args.num_tasks}, seed={args.scout_seed})")
    print("=" * 110)
    _print_full_table(scout_results)
    _print_ranking(
        scout_results, key=lambda r: r.acc, label="ACC"
    )
    _print_ranking(
        scout_results, key=lambda r: r.task0_retention,
        label="Task-0 retention (R[T-1, 0])",
    )
    _print_ranking(
        scout_results,
        key=lambda r: r.acc + 0.5 * r.task0_retention,
        label="combined score (ACC + 0.5 × Task-0)",
    )
    print()
    print(f"Saved run log to {path}")


if __name__ == "__main__":
    main()
