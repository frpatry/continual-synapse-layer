"""Experiment 20 — n=10 statistical validation of cs_gated_cosine_developmental.

Follow-up to exp 19's n=5 exploratory finding (+6.5 pp ACC over
cs_gated_cosine baseline, unanimous across all 5 seeds). At n=5
the Wilcoxon Bonferroni-corrected p floor is 0.1875 — the effect
direction is unambiguous but the statistical test can never
clear p < 0.05 by construction. This experiment lifts the
developmental variant to n=10 by running 5 ADDITIONAL seeds
(5-9 by default) and combining them with exp 19's seeds (0-4).
The result is a head-to-head against naive at matching n=10
(from exp 17), with Wilcoxon × 1 comparison (no Bonferroni
inflation since there's only one pair).

Why not re-run all 10 fresh: the per-seed factory + set_seed
contract guarantees that seeds 5-9 in isolation produce the
same trajectories as seeds 5-9 in a full 0-9 run (the prior
seed-isolation investigation confirmed no cross-seed leakage
through chromadb client state for non-sparse methods). Saves
~25 minutes of compute and the combined n=10 sample is
identical to a single-run n=10 result.

What this script does:
1. Runs cs_gated_cosine_developmental on --seeds (default 5..9)
   on Permuted-MNIST 15 tasks. Same protocol as exp 19.
2. Writes its own self-contained JSON.
3. If --prior-log AND --naive-log are supplied: reads the
   developmental seeds from --prior-log, the naive seeds from
   --naive-log, combines, and prints the combined-n=10
   ACC ± std plus the single-pair Wilcoxon.

Defaults are wired so that the user can pass the canonical paths
without thinking::

    python experiments/20_developmental_n10_validation.py \\
        --seeds 5 6 7 8 9 \\
        --prior-log results/logs/developmental_maturity/<ts>_19_permuted_mnist_T15.json \\
        --naive-log results/logs/generalization/<ts>_17_permuted_mnist_T15.json
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
    MethodSummary,
    format_pairwise_table,
    pairwise_wilcoxon,
)
from continual_synapse.reward.consistency import ConsistencyReward  # noqa: E402
from continual_synapse.reward.external import ExternalReward  # noqa: E402
from continual_synapse.reward.mixer import RewardMixer  # noqa: E402
from continual_synapse.reward.surprise import SurpriseReward  # noqa: E402
from continual_synapse.synapse_layer.layer import SynapseLayer  # noqa: E402
from continual_synapse.synapse_layer.modulation import SynapseModulation  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, nargs="+", default=[5, 6, 7, 8, 9],
                   help="Seeds to RUN this time. Defaults to 5-9 so the "
                        "result can be combined with exp 19's 0-4.")
    p.add_argument("--num-tasks", type=int, default=15)
    p.add_argument(
        "--prior-log", type=Path, default=None,
        help="Path to exp 19's Permuted-MNIST JSON. If supplied alongside "
             "--naive-log, the script combines developmental seeds from "
             "this log with the newly-run ones and prints the combined "
             "n=10 Wilcoxon vs naive.",
    )
    p.add_argument(
        "--naive-log", type=Path, default=None,
        help="Path to exp 17's Permuted-MNIST JSON (provides the n=10 "
             "naive data for the combined Wilcoxon).",
    )
    # ---- Training hyperparameters (mirror exp 19 / exp 17 Permuted) ----
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
    p.add_argument("--maturity-target-consolidations", type=int, default=50)
    p.add_argument(
        "--age-thresholds", type=int, nargs="+", default=[100, 500, 2000]
    )
    p.add_argument("--permutation-seed", type=int, default=42)
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
        default=str(_REPO_ROOT / "results" / "logs" / "developmental_n10"),
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


def _make_developmental_factory(
    args, num_classes: int
) -> tuple[Callable[[int], tuple[Any, ContinualRunner]], list[dict]]:
    chroma_client = chromadb.Client()
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
            collection_name=f"exp20_dev_seed_{seed}_{time.time_ns()}",
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
        diagnostics.append({"seed": seed, "per_task": per_task_diag})
        return model, runner

    return factory, diagnostics


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


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _extract_per_seed_metrics(
    log_payload: dict, method_name: str
) -> dict[str, tuple[list[int], list[float], list[float]]]:
    """Return (seeds, acc_values, fgt_values) for ``method_name`` in
    a prior log."""
    method_block = None
    for m in log_payload["methods"]:
        if m["method"] == method_name:
            method_block = m
            break
    if method_block is None:
        raise SystemExit(
            f"method {method_name!r} not in log "
            f"(found: {[m['method'] for m in log_payload['methods']]})"
        )
    summary_block = None
    for s in log_payload["summaries"]:
        if s["method"] == method_name:
            summary_block = s
            break
    if summary_block is None:
        raise SystemExit(f"summary for {method_name!r} missing from log")
    return {
        "seeds": list(method_block["seeds"]),
        "acc": list(summary_block["per_seed_metrics"]["average_accuracy"]),
        "fgt": list(summary_block["per_seed_metrics"]["average_forgetting"]),
        "bwt": list(summary_block["per_seed_metrics"]["backward_transfer"]),
        "fwt": list(summary_block["per_seed_metrics"]["forward_transfer"]),
    }


def _make_combined_summary(
    method_name: str,
    prior: dict[str, list],
    new: dict[str, list],
) -> MethodSummary:
    """Concatenate per-seed metrics from a prior log and a fresh run
    into a single MethodSummary that pairwise_wilcoxon can consume."""
    import numpy as np

    combined: dict[str, list[float]] = {}
    for key in ("acc", "fgt", "bwt", "fwt"):
        combined[key] = list(prior[key]) + list(new[key])
    metric_means: dict[str, float] = {}
    metric_stds: dict[str, float] = {}
    per_seed: dict[str, list[float]] = {}
    name_map = {
        "acc": "average_accuracy",
        "fgt": "average_forgetting",
        "bwt": "backward_transfer",
        "fwt": "forward_transfer",
    }
    for short, full in name_map.items():
        arr = np.asarray(combined[short], dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        metric_means[full] = float(finite.mean()) if finite.size else float("nan")
        metric_stds[full] = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
        per_seed[full] = combined[short]
    return MethodSummary(
        method=method_name,
        n_seeds=len(combined["acc"]),
        metric_means=metric_means,
        metric_stds=metric_stds,
        per_seed_metrics=per_seed,
    )


def main() -> None:
    args = parse_args()
    if len(args.seeds) < 1:
        raise SystemExit("need at least one seed to run")

    print(
        f"Loading PermutedMNIST with {args.num_tasks} tasks "
        f"(permutation seed={args.permutation_seed}, dropout={args.dropout})..."
    )
    bench = PermutedMNIST.from_huggingface(
        num_tasks=args.num_tasks,
        seed=args.permutation_seed,
        cache_dir=args.cache_dir,
    )
    factory, diagnostics = _make_developmental_factory(
        args, num_classes=bench.num_classes_per_task
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    output_path = (
        output_dir
        / f"{ts}_20_developmental_n10_validation_T{args.num_tasks}.json"
    )
    print(f"Checkpoint path: {output_path}", flush=True)
    print(f"Running cs_gated_cosine_developmental on seeds {args.seeds}...")

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
        "cs_gated_cosine_developmental",
        factory, bench, seeds=args.seeds,
        progress=lambda m, i, n: print(
            f"  {m}: seed {i + 1}/{n}", flush=True
        ),
        on_seed_complete=_seed_done,
    )
    elapsed = time.time() - t0
    print(f"\ncs_gated_cosine_developmental finished in {elapsed:.1f}s")

    # ---- Write self-contained JSON for these new seeds ----
    payload = {
        "experiment": "20_developmental_n10_validation",
        "timestamp": ts,
        "config": vars(args),
        "methods": [_multi_seed_to_jsonable(run)],
        "diagnostics": {"cs_gated_cosine_developmental": diagnostics},
        "method_times_seconds": {"cs_gated_cosine_developmental": elapsed},
    }
    _atomic_write_json(output_path, payload)
    print(f"Saved new-seeds log to {output_path}")

    new_per_seed = {
        "seeds": list(args.seeds),
        "acc": [compute_metrics(r).average_accuracy for r in run.results],
        "fgt": [compute_metrics(r).average_forgetting for r in run.results],
        "bwt": [compute_metrics(r).backward_transfer for r in run.results],
        "fwt": [
            float("nan") if compute_metrics(r).forward_transfer is None
            else compute_metrics(r).forward_transfer
            for r in run.results
        ],
    }

    # ---- Combined n=10 analysis if both prior logs supplied ----
    if args.prior_log is None or args.naive_log is None:
        print()
        print(
            "No --prior-log / --naive-log supplied; skipping combined "
            "Wilcoxon. The new-seeds JSON above is the data you need to "
            "combine externally with exp 19's 0-4 seeds for cs_gated_cosine"
            "_developmental and exp 17's naive (n=10)."
        )
        return

    print()
    print("=" * 78)
    print(f"COMBINED n=10 ANALYSIS")
    print(f"  prior-log (developmental 0-4): {args.prior_log}")
    print(f"  naive-log (naive 0-9):         {args.naive_log}")
    print("=" * 78)

    prior = json.loads(args.prior_log.read_text())
    naive_log = json.loads(args.naive_log.read_text())

    prior_dev = _extract_per_seed_metrics(
        prior, "cs_gated_cosine_developmental"
    )
    naive_data = _extract_per_seed_metrics(naive_log, "naive")

    dev_combined = _make_combined_summary(
        "cs_gated_cosine_developmental", prior_dev, new_per_seed
    )
    # Naive is already n=10 from exp 17. Build a MethodSummary with the
    # full per-seed arrays (don't combine — there's nothing new to add).
    import numpy as np
    name_map = {
        "acc": "average_accuracy", "fgt": "average_forgetting",
        "bwt": "backward_transfer", "fwt": "forward_transfer",
    }
    n_means: dict[str, float] = {}
    n_stds: dict[str, float] = {}
    n_seed: dict[str, list[float]] = {}
    for short, full in name_map.items():
        arr = np.asarray(naive_data[short], dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        n_means[full] = float(finite.mean()) if finite.size else float("nan")
        n_stds[full] = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
        n_seed[full] = naive_data[short]
    naive_summary = MethodSummary(
        method="naive",
        n_seeds=len(naive_data["acc"]),
        metric_means=n_means, metric_stds=n_stds, per_seed_metrics=n_seed,
    )

    # Sanity: align seed counts and identify any non-overlap.
    print()
    print(
        f"Sample sizes:  developmental n={dev_combined.n_seeds}, "
        f"naive n={naive_summary.n_seeds}"
    )
    print(
        f"Developmental seeds: prior={prior_dev['seeds']}  "
        f"new={new_per_seed['seeds']}"
    )
    print(f"Naive seeds:         {naive_data['seeds']}")
    if dev_combined.n_seeds != naive_summary.n_seeds:
        print(
            "  WARNING: sample sizes differ. Wilcoxon will use the smaller "
            "of the two but the pairing semantics may not be meaningful."
        )

    print()
    print("Per-method ACC and FGT:")
    print(f"  {'method':<36s} {'n':>3s}  {'ACC mean':>10s} {'ACC std':>10s}  "
          f"{'FGT mean':>10s} {'FGT std':>10s}")
    print("  " + "-" * 86)
    for s in (naive_summary, dev_combined):
        print(
            f"  {s.method:<36s} {s.n_seeds:>3d}  "
            f"{s.metric_means['average_accuracy']:>10.4f} "
            f"{s.metric_stds['average_accuracy']:>10.4f}  "
            f"{s.metric_means['average_forgetting']:>+10.4f} "
            f"{s.metric_stds['average_forgetting']:>10.4f}"
        )

    print()
    print("Per-seed pairing (developmental − naive ACC delta):")
    nv_acc = naive_summary.per_seed_metrics["average_accuracy"]
    dv_acc = dev_combined.per_seed_metrics["average_accuracy"]
    nseed = naive_data["seeds"]
    # Combined developmental seeds = prior + new, in that order.
    dseed = list(prior_dev["seeds"]) + list(new_per_seed["seeds"])
    n_pairs = min(len(nv_acc), len(dv_acc))
    for i in range(n_pairs):
        d = dv_acc[i] - nv_acc[i]
        marker = " ✓" if d > 0 else (" ✗" if d < 0 else "  ")
        nv_s = nseed[i] if i < len(nseed) else "?"
        dv_s = dseed[i] if i < len(dseed) else "?"
        print(
            f"  seed pair {i}: naive[{nv_s}]={nv_acc[i]:.4f}  "
            f"dev[{dv_s}]={dv_acc[i]:.4f}  Δ={d:+.4f}{marker}"
        )

    print()
    print("Pairwise Wilcoxon (1 comparison ⇒ Bonferroni × 1 = no inflation):")
    for metric in ("average_accuracy", "average_forgetting"):
        print(f"  {metric}:")
        for c in pairwise_wilcoxon([naive_summary, dev_combined], metric=metric):
            sig = "SIG @ 0.05" if c.significant_05 else "n.s."
            print(
                f"    {c.method_a:<32s} vs {c.method_b:<32s}  "
                f"n={c.n}  p_raw={c.p_value:.5f}  "
                f"p_bonf={c.p_value_bonferroni:.5f}  {sig}"
            )

    print()
    print("=" * 78)
    delta_acc = (
        dev_combined.metric_means["average_accuracy"]
        - naive_summary.metric_means["average_accuracy"]
    )
    delta_fgt = (
        dev_combined.metric_means["average_forgetting"]
        - naive_summary.metric_means["average_forgetting"]
    )
    print(f"  ACC improvement: {delta_acc:+.4f} ({delta_acc * 100:+.2f} pp)")
    print(f"  FGT change:      {delta_fgt:+.4f} ({delta_fgt * 100:+.2f} pp; "
          f"negative = less forgetting)")
    print("=" * 78)


if __name__ == "__main__":
    main()
