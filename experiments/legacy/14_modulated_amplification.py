"""Experiment 14 — reward-modulated amplification (gentle, task-aware).

Tests the combined hypothesis from prior failure analysis:
gentle amplification (alpha = 0.1, ten times smaller than the
exp 13 default) + reward modulation (alpha scaled by the per-batch
reward signal) + task awareness (recency-weighted retrieval +
warmup downweight) + real reward + no-drain consolidation +
repeat-consolidation merging + retrieval-success feedback. The
hypothesis: each prior amplified variant introduced an
unbounded-strengthening pathology — strong alpha plus always-on
amplification plus retained-after-drain candidates produced
runaway corrections. The fix is to make the amplification factor
itself respect the model's per-batch confidence (the reward
signal), so high-reward batches amplify and low/negative-reward
batches damp.

Methods:
- naive: control.
- cs_full: cold storage with every audit-fix gap (matches exp 12 /
  exp 13 cs_full — fake reward, additive composition).
- cs_full_real_reward: cs_full + recentered consistency + external
  reward from per-batch accuracy.
- cs_full_amplified: cs_full + the five amplification flags from
  exp 13 at full strength (alpha=1.0, fake reward). The "baseline
  crash" reference — the variant whose pathology this experiment
  is trying to fix.
- cs_full_amplified_modulated: cs_full + gentle amplification
  (alpha=0.1) + all five amplification flags + task-aware
  retrieval + real reward + reward-modulated composition. Where
  the hypothesis lives.

Same 15-task Permuted-MNIST, dropout=0.5, 5-seed protocol as
exp 12 and exp 13. Per-method JSON checkpointing
(``results/logs/modulated_amplification/<ts>_14_modulated_T15.json``)
with atomic .tmp + os.replace and ``is_partial`` /
``methods_completed`` / ``methods_requested`` flags so the last
completed method's data survives any kill.

Reports per-method ACC / FGT / BWT / FWT, the full ``(T, T)``
per-task accuracy trajectory matrix per (method, seed), Wilcoxon
signed-rank pairwise tests with Bonferroni correction, per-batch
reward distribution, and the amplification-variant diagnostics
introduced in exp 13 (merge counts, retrieval feedback events,
avg retrieved access_count per task, modulator gate, loss EMA).
Plus a new per-task field — ``last_reward`` — that lets us see
the reward-modulation signal at task boundaries.

Run from the repo root::

    python experiments/14_modulated_amplification.py --num-tasks 15
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


_DEFAULT_METHODS = (
    "naive",
    "cs_full",
    "cs_full_real_reward",
    "cs_full_amplified",
    "cs_full_amplified_modulated",
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
    p.add_argument("--consistency-center", type=float, default=0.95)
    p.add_argument("--consistency-scale", type=float, default=0.05)
    p.add_argument("--consistency-clip-min", type=float, default=-1.0)
    p.add_argument("--consistency-clip-max", type=float, default=1.0)
    # ---- Amplification flags shared by cs_full_amplified and
    # ---- cs_full_amplified_modulated (except amplification_alpha,
    # ---- which differs between them). Defaults match the exp 13
    # ---- baseline "full strength" amplification.
    p.add_argument("--amplification-alpha", type=float, default=1.0,
                   help="alpha for cs_full_amplified (full-strength baseline).")
    p.add_argument("--modulated-amplification-alpha", type=float, default=0.1,
                   help="alpha for cs_full_amplified_modulated (gentle, "
                        "reward-modulated; 10x smaller than the baseline).")
    p.add_argument("--confidence-exponent", type=float, default=0.5)
    p.add_argument("--repeat-consolidation-threshold", type=float, default=0.85)
    p.add_argument("--retrieval-feedback-threshold", type=float, default=0.9)
    p.add_argument("--retrieval-feedback-decay", type=float, default=0.95)
    p.add_argument("--retrieval-feedback-bump", type=float, default=0.5)
    # ---- Task-aware variant flags (used by cs_full_amplified_modulated only)
    p.add_argument("--task-aware-decay", type=float, default=0.5)
    p.add_argument("--task-warmup-batches", type=int, default=100)
    p.add_argument("--task-warmup-downweight", type=float, default=0.1)
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
        default=str(_REPO_ROOT / "results" / "logs" / "modulated_amplification"),
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
    dict[str, list[list[float]]],
]:
    chroma_client = chromadb.Client()
    diagnostics: dict[str, list[dict]] = {m: [] for m in args.methods}
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
        use_real_reward: bool,
        amplification_alpha: float,
        modulated: bool,
        task_aware: bool,
    ):
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
        reward_computer = _build_reward(use_real_reward)
        cold_storage = ColdStorage(
            collection_name=f"exp14_{method_key}_seed_{seed}_{time.time_ns()}",
            client=chroma_client,
        )
        trigger = ConsolidationTrigger(
            avg_pressure_threshold=args.pressure_threshold,
            min_steps_between=args.min_steps_between_consolidations,
            candidate_quantile=args.candidate_quantile,
        )
        amp_kwargs: dict[str, Any] = {}
        if amplification_alpha > 0.0:
            amp_kwargs = dict(
                amplification_alpha=amplification_alpha,
                confidence_exponent=args.confidence_exponent,
                no_drain_on_consolidate=True,
                repeat_consolidation_threshold=args.repeat_consolidation_threshold,
                retrieval_feedback_threshold=args.retrieval_feedback_threshold,
                retrieval_feedback_decay=args.retrieval_feedback_decay,
                retrieval_feedback_bump=args.retrieval_feedback_bump,
            )
        if modulated:
            amp_kwargs["reward_modulated_amplification"] = True
        if task_aware:
            amp_kwargs.update(
                task_aware_decay=args.task_aware_decay,
                task_warmup_batches=args.task_warmup_batches,
                task_warmup_downweight=args.task_warmup_downweight,
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
            **amp_kwargs,
        )

        per_task_diag: list[dict] = []
        seed_rewards: list[float] = []
        reward_samples[method_key].append(seed_rewards)

        task_state: dict[str, Any] = {
            "task_retrieval_access_sum": 0.0,
            "task_retrieval_access_batches": 0,
            "prev_consolidation_count": 0,
            "prev_merge_count": 0,
            "prev_feedback_count": 0,
        }

        needs_training_target = use_real_reward or modulated

        def on_after_batch(i, task, m, x, y):
            if needs_training_target:
                applied = m.apply_hebbian_update(training_target=y)
            else:
                applied = m.apply_hebbian_update()
            seed_rewards.append(float(applied))
            meta = m.last_retrieved_meta
            if meta:
                avg = sum(c for _, c in meta) / len(meta)
                task_state["task_retrieval_access_sum"] += avg
                task_state["task_retrieval_access_batches"] += 1

        def on_task_end_diag(i, task, m):
            n_batches = task_state["task_retrieval_access_batches"]
            avg_access = (
                task_state["task_retrieval_access_sum"] / n_batches
                if n_batches > 0
                else None
            )
            per_task_diag.append(
                {
                    "task_index": int(i),
                    "consolidation_count": int(m.consolidation_count),
                    "merge_count": int(m.merge_count),
                    "new_entry_count_this_task": int(
                        (m.consolidation_count - task_state["prev_consolidation_count"])
                        - (m.merge_count - task_state["prev_merge_count"])
                    ),
                    "merge_count_this_task": int(
                        m.merge_count - task_state["prev_merge_count"]
                    ),
                    "consolidation_count_this_task": int(
                        m.consolidation_count - task_state["prev_consolidation_count"]
                    ),
                    "retrieval_feedback_events": int(
                        m.retrieval_feedback_event_count
                    ),
                    "retrieval_feedback_events_this_task": int(
                        m.retrieval_feedback_event_count
                        - task_state["prev_feedback_count"]
                    ),
                    "store_count": int(m.cold_storage.count()),
                    "compression_sweep_count": int(m.compression_sweep_count),
                    "store_byte_size": int(_store_byte_size(m.cold_storage)),
                    "last_compression_counts": {
                        int(k): int(v)
                        for k, v in m.last_compression_counts.items()
                    },
                    "avg_retrieved_access_count": (
                        None if avg_access is None else float(avg_access)
                    ),
                    "modulator_gate": float(m.modulator.gate.item()),
                    "loss_ema": (
                        None if m.loss_ema is None else float(m.loss_ema)
                    ),
                    "current_task_id": int(m.current_task_id),
                    "batches_since_task_change": int(m.batches_since_task_change),
                    "last_reward": (
                        None if m.last_reward is None else float(m.last_reward)
                    ),
                }
            )
            task_state["task_retrieval_access_sum"] = 0.0
            task_state["task_retrieval_access_batches"] = 0
            task_state["prev_consolidation_count"] = int(m.consolidation_count)
            task_state["prev_merge_count"] = int(m.merge_count)
            task_state["prev_feedback_count"] = int(
                m.retrieval_feedback_event_count
            )

        def on_task_change_notify(j, task, m):
            m.notify_task_change(int(j))

        runner = ContinualRunner(
            seed=seed,
            on_after_batch=on_after_batch,
            on_task_end=on_task_end_diag,
            on_task_change=on_task_change_notify,
            **runner_kwargs,
        )
        diagnostics[method_key].append(
            {"seed": seed, "per_task": per_task_diag}
        )
        return model, runner

    factories: dict[str, Callable] = {
        "naive": naive,
        "cs_full": lambda s: _synapse(
            s, "cs_full",
            use_real_reward=False, amplification_alpha=0.0,
            modulated=False, task_aware=False,
        ),
        "cs_full_real_reward": lambda s: _synapse(
            s, "cs_full_real_reward",
            use_real_reward=True, amplification_alpha=0.0,
            modulated=False, task_aware=False,
        ),
        # Baseline crash reference: full-strength amplification, fake
        # reward, no task awareness. Matches exp 13 cs_full_amplified.
        "cs_full_amplified": lambda s: _synapse(
            s, "cs_full_amplified",
            use_real_reward=False,
            amplification_alpha=args.amplification_alpha,
            modulated=False, task_aware=False,
        ),
        # The hypothesis: gentle alpha (0.1) + real reward + reward
        # modulation + task awareness. Synthesis of every lesson.
        "cs_full_amplified_modulated": lambda s: _synapse(
            s, "cs_full_amplified_modulated",
            use_real_reward=True,
            amplification_alpha=args.modulated_amplification_alpha,
            modulated=True, task_aware=True,
        ),
    }
    return factories, diagnostics, reward_samples


def _reward_histogram(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return {"n": 0}
    arr = sorted(samples)
    n = len(arr)
    mean = sum(arr) / n
    return {
        "n": n,
        "mean": mean,
        "min": arr[0],
        "max": arr[-1],
        "p10": arr[int(n * 0.10)],
        "p50": arr[int(n * 0.50)],
        "p90": arr[int(n * 0.90)],
        "std": (sum((x - mean) ** 2 for x in arr) / n) ** 0.5,
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


def _build_payload(
    args: argparse.Namespace,
    ts: int,
    runs: list[MultiSeedRun],
    summaries: list,
    diagnostics: dict[str, list[dict]],
    reward_summary: dict[str, dict],
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
        "experiment": "14_modulated_amplification",
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
        "reward_summary": reward_summary,
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
    factories, diagnostics, reward_samples = _build_factories(
        args, num_classes=bench.num_classes_per_task
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    path = output_dir / f"{ts}_14_modulated_T{args.num_tasks}.json"
    print(f"Checkpoint path: {path}", flush=True)

    runs: list[MultiSeedRun] = []
    summaries: list = []
    reward_summary: dict[str, dict] = {}
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
        flat = [r for seed_list in reward_samples[method] for r in seed_list]
        reward_summary[method] = _reward_histogram(flat)
        methods_completed.append(method)
        print(f"{method} finished in {elapsed:.1f}s", flush=True)

        is_partial = len(methods_completed) < len(args.methods)
        payload = _build_payload(
            args, ts, runs, summaries, diagnostics, reward_summary,
            method_times,
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
        print(f"Pairwise Wilcoxon signed-rank on {metric} (Bonferroni-corrected):")
        print(format_pairwise_table(pairwise_wilcoxon(summaries, metric=metric)))

    print("Per-batch reward distribution (flattened across seeds):")
    print(
        f"  {'method':<32s} {'n':>7s} {'min':>7s} {'p10':>7s} "
        f"{'p50':>7s} {'p90':>7s} {'max':>7s} {'mean':>7s} {'std':>7s}"
    )
    for method in args.methods:
        h = reward_summary.get(method, {"n": 0})
        if h["n"] == 0:
            print(f"  {method:<32s}  (no reward samples — naive control)")
            continue
        print(
            f"  {method:<32s} {h['n']:>7d} "
            f"{h['min']:>7.3f} {h['p10']:>7.3f} {h['p50']:>7.3f} "
            f"{h['p90']:>7.3f} {h['max']:>7.3f} "
            f"{h['mean']:>7.3f} {h['std']:>7.3f}"
        )

    print()
    print("End-of-run modulation diagnostics (mean across seeds):")
    print(
        f"  {'method':<32s} {'consol':>8s} {'merge':>8s} "
        f"{'fb_evts':>8s} {'store':>8s} {'avg_access':>12s} "
        f"{'gate':>9s} {'last_R':>9s}"
    )
    for method in args.methods:
        seed_diags = diagnostics.get(method, [])
        if not seed_diags or not any(d.get("per_task") for d in seed_diags):
            continue
        finals = [d["per_task"][-1] for d in seed_diags if d.get("per_task")]
        if not finals:
            continue

        def _mean(key: str, default: float = 0.0) -> float:
            vals = [
                float(f.get(key) if f.get(key) is not None else default)
                for f in finals
            ]
            return sum(vals) / len(vals) if vals else default

        avg_access_vals = [f.get("avg_retrieved_access_count") for f in finals]
        avg_access_vals = [v for v in avg_access_vals if v is not None]
        avg_access = (
            sum(avg_access_vals) / len(avg_access_vals)
            if avg_access_vals else float("nan")
        )
        last_R_vals = [f.get("last_reward") for f in finals]
        last_R_vals = [v for v in last_R_vals if v is not None]
        last_R = (
            sum(last_R_vals) / len(last_R_vals)
            if last_R_vals else float("nan")
        )
        print(
            f"  {method:<32s} "
            f"{_mean('consolidation_count'):>8.1f} "
            f"{_mean('merge_count'):>8.1f} "
            f"{_mean('retrieval_feedback_events'):>8.1f} "
            f"{_mean('store_count'):>8.1f} "
            f"{avg_access:>12.3f} "
            f"{_mean('modulator_gate'):>+9.4f} "
            f"{last_R:>+9.4f}"
        )

    print(f"Saved run log to {path}")


if __name__ == "__main__":
    main()
