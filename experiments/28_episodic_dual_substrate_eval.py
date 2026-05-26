"""Experiment 28 — Dual-substrate episodic memory pilot.

Compares the new ``cs_episodic_dual_substrate`` config to the
existing ``cs_gated_cosine_developmental`` baseline at the same T.

The dual-substrate config trains a plain
:class:`MLPClassifier` with standard backprop on the task loss —
no synapse layer, no cosine gating, no Hebbian state, no EWC. An
:class:`ActiveEpisodicMemory` grows alongside via gradient-free
novelty-thresholded allocation, and predictions at inference use
an :class:`EpisodicPredictor` to blend the bare-model softmax with
a retrieval-based label distribution.

The baseline is re-evaluated from checkpoints saved by exp 27
(``results/checkpoints/phase_d/cs_gated_cosine_developmental_T*_seed*.pt``)
if available. If the baseline checkpoint is missing, the script
either trains it fresh (matches exp 27's scout_a095_validated
defaults) or skips that seed depending on
``--skip-missing-baseline``.

Output JSON follows the exp-23 schema so
``experiments/24_retention_analysis.py`` ingests it directly.
Storage diagnostics (per-task memory growth, final memory size,
total allocations) land alongside the methods/summaries blocks.

Run from the repo root::

    python experiments/28_episodic_dual_substrate_eval.py \\
        --T 15 --n_seeds 2

This script is meant for **manual** execution — the prior session
shipped the infrastructure and decision criteria; the operator
triggers this when ready to test the dual-substrate hypothesis.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from continual_synapse.baselines.naive_finetune import MLPClassifier, MLPConfig  # noqa: E402
from continual_synapse.baselines.synapse_finetune import SynapseAugmentedMLP  # noqa: E402
from continual_synapse.cold_storage.compression import CompressionSchedule  # noqa: E402
from continual_synapse.cold_storage.store import ColdStorage  # noqa: E402
from continual_synapse.consolidation.trigger import ConsolidationTrigger  # noqa: E402
from continual_synapse.episodic import (  # noqa: E402
    EPISODIC_CONFIGS,
    ActiveEpisodicMemory,
    EpisodicConfig,
    EpisodicPredictor,
)
from continual_synapse.evaluation.benchmarks import PermutedMNIST  # noqa: E402
from continual_synapse.evaluation.runner import ContinualRunner, set_seed  # noqa: E402
from continual_synapse.reward.consistency import ConsistencyReward  # noqa: E402
from continual_synapse.reward.external import ExternalReward  # noqa: E402
from continual_synapse.reward.mixer import RewardMixer  # noqa: E402
from continual_synapse.reward.surprise import SurpriseReward  # noqa: E402
from continual_synapse.synapse_layer.layer import SynapseLayer  # noqa: E402
from continual_synapse.synapse_layer.modulation import SynapseModulation  # noqa: E402


_BASELINE_METHOD = "cs_gated_cosine_developmental"
_EPISODIC_METHOD = "cs_episodic_dual_substrate"
# scout_a095 constants — needed when rebuilding the baseline model
# architecturally identical to exp 27 / 23.
_SCOUT_A095_ALPHA = 0.95
_SCOUT_A095_TARGET = 50


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--T", type=int, default=15)
    p.add_argument("--n_seeds", type=int, default=2)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument(
        "--novelty-threshold", type=float, default=0.7,
        help="Override the default novelty threshold on the episodic "
             "config. Sweepable.",
    )
    p.add_argument(
        "--blend-max", type=float, default=0.5,
        help="Override the default max retrieval-blend weight.",
    )
    p.add_argument(
        "--retrieval-k", type=int, default=5,
        help="Top-k entries consulted at retrieval time.",
    )
    p.add_argument(
        "--blend-threshold", type=float, default=0.5,
        help="Retrieval-confidence threshold below which the memory "
             "contributes nothing.",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=_REPO_ROOT / "results" / "logs" / "episodic",
    )
    p.add_argument(
        "--baseline-checkpoint-dir", type=Path,
        default=_REPO_ROOT / "results" / "checkpoints" / "phase_d",
        help="Where to look for baseline cs_gated_cosine_developmental "
             "checkpoints (saved by exp 27).",
    )
    p.add_argument(
        "--episodic-checkpoint-dir", type=Path,
        default=_REPO_ROOT / "results" / "checkpoints" / "phase_e",
        help="Where to save episodic-config checkpoints.",
    )
    p.add_argument(
        "--skip-missing-baseline", action="store_true",
        help="If a baseline checkpoint is missing for a seed, skip "
             "that seed's baseline comparison instead of training "
             "the baseline from scratch.",
    )
    p.add_argument(
        "--skip-baseline", action="store_true",
        help="Skip the baseline comparison entirely. Useful when "
             "iterating on the episodic config alone.",
    )
    # ---- Training hyperparameters (mirror scout_a095 / exp 27) ----
    p.add_argument("--epochs-per-task", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-hidden-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--permutation-seed", type=int, default=42)
    # ---- Baseline-only hyperparameters (used when retraining
    # baseline if a checkpoint is missing) ----
    p.add_argument("--synapse-lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=10.0)
    p.add_argument("--reward-mixer-gamma", type=float, default=1e-3)
    p.add_argument("--w-consistency", type=float, default=1.0)
    p.add_argument("--w-surprise", type=float, default=0.5)
    p.add_argument("--pressure-threshold", type=float, default=0.005)
    p.add_argument("--min-steps-between-consolidations", type=int, default=60)
    p.add_argument("--candidate-quantile", type=float, default=0.05)
    p.add_argument("--retrieval-refresh-interval", type=int, default=20)
    p.add_argument("--n-passes", type=int, default=5)
    p.add_argument("--compression-sweep-interval", type=int, default=100)
    p.add_argument("--age-thresholds", type=int, nargs="+", default=[100, 500, 2000])
    p.add_argument("--maturity-target", type=int, default=50)
    p.add_argument("--gating-alpha", type=float, default=0.95)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda", "mps"],
    )
    p.add_argument(
        "--cache-dir", default=str(_REPO_ROOT / "data" / "hf_cache")
    )
    return p.parse_args()


# ---------- episodic model builder ----------


def _build_episodic_predictor(
    cfg: EpisodicConfig, args: argparse.Namespace, *,
    num_classes: int, seed: int,
) -> tuple[MLPClassifier, EpisodicPredictor]:
    """Build (bare_model, predictor) for the dual-substrate config.

    The bare model is a plain :class:`MLPClassifier` — NO synapse
    layer, NO modulator, NO reward computer. That's the architectural
    bet: free-running weights paired with an external memory
    substrate.
    """
    set_seed(seed)
    base = MLPClassifier(
        MLPConfig(
            input_dim=784, hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            num_hidden_layers=args.num_hidden_layers,
            dropout=args.dropout,
        )
    )
    memory = cfg.build_memory(
        feature_dim=args.hidden_dim, n_classes=num_classes,
    )
    predictor = cfg.build_predictor(base, memory)
    return base, predictor


# ---------- baseline model builder (mirrors exp 27) ----------


def _build_compression_schedule(args: argparse.Namespace) -> CompressionSchedule:
    n_thresholds = len(args.age_thresholds)
    all_tiers = (32, 16, 8, 4)
    tiers = all_tiers[: n_thresholds + 1]
    return CompressionSchedule(
        age_thresholds=tuple(args.age_thresholds),
        tier_precisions=tiers,
    )


def _build_baseline_model(
    args: argparse.Namespace, *,
    num_classes: int, seed: int, chroma_client, T: int,
) -> SynapseAugmentedMLP:
    """Construct the cs_gated_cosine_developmental baseline.

    Architecturally identical to exp 27's
    _build_model_for_config(REWARD_CONFIGS['cs_gated_cosine_developmental'])
    — same gradient_gating_alpha, same maturity_target, same
    pressure-mode trigger. Used here only when a baseline checkpoint
    is missing AND --skip-missing-baseline is not set.
    """
    set_seed(seed)
    base = MLPClassifier(
        MLPConfig(
            input_dim=784, hidden_dim=args.hidden_dim,
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
        gamma=args.reward_mixer_gamma,
        w_consistency=args.w_consistency,
        w_surprise=args.w_surprise,
    )
    cold_storage = ColdStorage(
        collection_name=(
            f"exp28_baseline_T{T}_seed_{seed}_{time.time_ns()}"
        ),
        client=chroma_client,
    )
    trigger = ConsolidationTrigger(
        avg_pressure_threshold=args.pressure_threshold,
        min_steps_between=args.min_steps_between_consolidations,
        candidate_quantile=args.candidate_quantile,
    )
    return SynapseAugmentedMLP(
        base, synapse, modulator,
        reward_computer=reward_computer,
        cold_storage=cold_storage,
        consolidation_trigger=trigger,
        retrieval_k=args.retrieval_refresh_interval,  # match exp 27 wiring
        retrieval_refresh_interval=args.retrieval_refresh_interval,
        n_passes=args.n_passes,
        compression_sweep_interval=args.compression_sweep_interval,
        compression_schedule=_build_compression_schedule(args),
        gate_modulation_enabled=False,
        gradient_gating_enabled=True,
        gradient_gating_alpha=_SCOUT_A095_ALPHA,
        familiarity_mode="cosine",
        maturity_target_consolidations=_SCOUT_A095_TARGET,
    )


def _load_baseline_checkpoint(
    path: Path, args: argparse.Namespace, *,
    num_classes: int, T: int, seed: int, chroma_client,
) -> SynapseAugmentedMLP:
    ckpt = torch.load(path, map_location=args.device, weights_only=False)
    model = _build_baseline_model(
        args, num_classes=num_classes, seed=seed,
        chroma_client=chroma_client, T=T,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for e in ckpt.get("cold_storage_entries", []):
        model.cold_storage.store_cluster(
            embedding=e["embedding"],
            metadata=e["metadata"],
            document=e["document"],
            entry_id=e["id"],
        )
    return model


# ---------- training + eval ----------


def _episodic_checkpoint_path(
    ckpt_dir: Path, T: int, seed: int
) -> Path:
    return ckpt_dir / f"cs_episodic_dual_substrate_T{T}_seed{seed}.pt"


def _save_episodic_checkpoint(
    path: Path, base: MLPClassifier, memory: ActiveEpisodicMemory,
    diagnostics: dict[str, Any], config_dict: dict[str, Any],
) -> None:
    payload = {
        "config_name": _EPISODIC_METHOD,
        "config": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in config_dict.items()
        },
        "model_state_dict": base.state_dict(),
        "memory": {
            "embeddings": [e.cpu().tolist() for e in memory.embeddings],
            "labels": list(memory.labels),
            "task_ids": list(memory.task_ids),
            "feature_dim": memory.feature_dim,
            "n_classes": memory.n_classes,
            "novelty_threshold": memory.novelty_threshold,
            "retrieval_k": memory.retrieval_k,
            "max_entries": memory.max_entries,
        },
        "diagnostics": diagnostics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _load_episodic_checkpoint(
    path: Path, cfg: EpisodicConfig, args: argparse.Namespace, *,
    num_classes: int, seed: int,
) -> tuple[MLPClassifier, EpisodicPredictor, dict[str, Any]]:
    ckpt = torch.load(path, map_location=args.device, weights_only=False)
    base, predictor = _build_episodic_predictor(
        cfg, args, num_classes=num_classes, seed=seed,
    )
    base.load_state_dict(ckpt["model_state_dict"])
    base.eval()
    mem_state = ckpt["memory"]
    for emb, lbl, tid in zip(
        mem_state["embeddings"], mem_state["labels"], mem_state["task_ids"]
    ):
        predictor.memory.embeddings.append(torch.tensor(emb, dtype=torch.float32))
        predictor.memory.labels.append(int(lbl))
        predictor.memory.task_ids.append(tid)
    predictor.memory._invalidate_cache()
    return base, predictor, ckpt.get("diagnostics", {})


def _train_episodic_one_seed(
    cfg: EpisodicConfig, args: argparse.Namespace, bench, T: int,
    seed: int,
) -> tuple[MLPClassifier, EpisodicPredictor, dict[str, Any]]:
    """Train the dual-substrate config for one seed. Returns the
    bare model, the predictor (with populated memory), and a
    diagnostics dict capturing per-task memory growth."""
    print(
        f"    training {cfg.name}  T={T}  seed={seed}  "
        f"(novelty_thr={cfg.novelty_threshold}, "
        f"blend_max={cfg.blend_max})...",
        flush=True,
    )
    t0 = time.time()
    base, predictor = _build_episodic_predictor(
        cfg, args, num_classes=bench.num_classes_per_task, seed=seed,
    )

    diagnostics: dict[str, Any] = {
        "per_task_memory_size": [],
        "per_batch_novelty_mean": [],
        "per_batch_n_allocated": [],
        "final_memory_size": 0,
        "wall_time_s": 0.0,
    }

    def on_after_batch(task_index, task, model, x, y):
        # Compute features once for both the diagnostic and the
        # allocation call. The duplicate compute_novelty inside
        # maybe_allocate is cheap at the memory sizes we expect
        # (hundreds, not millions); not worth the API contortion.
        with torch.no_grad():
            features = predictor.feature_extract(x)
            novelty = predictor.memory.compute_novelty(features)
        n_added = predictor.memory.maybe_allocate(
            features, y, task_id=int(task_index),
        )
        diagnostics["per_batch_novelty_mean"].append(
            float(novelty.mean().item())
        )
        diagnostics["per_batch_n_allocated"].append(int(n_added))

    def on_task_end(task_index, task, model):
        diagnostics["per_task_memory_size"].append({
            "task_index": int(task_index),
            "memory_size": int(len(predictor.memory)),
        })

    runner = ContinualRunner(
        seed=seed,
        optimizer_factory=lambda p: torch.optim.SGD(
            p, lr=args.lr, momentum=args.momentum
        ),
        epochs_per_task=args.epochs_per_task,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        device=args.device,
        record_zero_shot=False,
        on_after_batch=on_after_batch,
        on_task_end=on_task_end,
    )
    runner.run(base, bench)
    diagnostics["final_memory_size"] = int(len(predictor.memory))
    diagnostics["wall_time_s"] = float(time.time() - t0)

    print(
        f"      trained in {diagnostics['wall_time_s']:.1f}s; "
        f"final memory size = {diagnostics['final_memory_size']}",
        flush=True,
    )
    return base, predictor, diagnostics


def _eval_with_predictor(
    predictor: EpisodicPredictor, bench, T: int, args: argparse.Namespace,
) -> list[float | None]:
    """Run predictor.predict on every task's test set. Returns the
    final row R[T-1, :]."""
    final_row: list[float | None] = []
    with torch.no_grad():
        for task in bench.tasks():
            x = task.test.tensors[0].to(args.device)
            y = task.test.tensors[1].to(args.device)
            logits = predictor.predict(x)
            preds = logits.argmax(dim=-1)
            acc = float((preds == y).float().mean().item())
            final_row.append(acc)
    while len(final_row) < T:
        final_row.append(None)
    return final_row[:T]


def _eval_baseline_model(
    model: SynapseAugmentedMLP, bench, T: int, args: argparse.Namespace,
) -> list[float | None]:
    """Run bare-model forward on every task's test set."""
    model.eval()
    final_row: list[float | None] = []
    with torch.no_grad():
        for task in bench.tasks():
            x = task.test.tensors[0].to(args.device)
            y = task.test.tensors[1].to(args.device)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            acc = float((preds == y).float().mean().item())
            final_row.append(acc)
    while len(final_row) < T:
        final_row.append(None)
    return final_row[:T]


def _final_row_to_matrix(
    final_row: list[float | None], T: int
) -> list[list[float | None]]:
    """Wrap a single final row into a (T, T) matrix in exp-23 schema:
    only the last row populated, all earlier rows ``None``."""
    am: list[list[float | None]] = []
    for i in range(T):
        if i < T - 1:
            am.append([None] * T)
        else:
            am.append(list(final_row))
    return am


def _summarise_final_rows(
    rows: list[list[float | None]]
) -> dict[str, Any]:
    """Aggregate per-seed final-row accuracies into the standard
    metric dict — average ACC, Task-0, Task-N, plus per-seed
    sequences and stds."""
    avg_accs: list[float] = []
    task0_accs: list[float] = []
    taskN_accs: list[float] = []
    for row in rows:
        defined = [v for v in row if v is not None]
        if defined:
            avg_accs.append(statistics.fmean(defined))
        if row and row[0] is not None:
            task0_accs.append(float(row[0]))
        if row and row[-1] is not None:
            taskN_accs.append(float(row[-1]))

    def _mean(xs):
        return statistics.fmean(xs) if xs else float("nan")

    def _std(xs):
        return statistics.stdev(xs) if len(xs) > 1 else 0.0

    return {
        "n_seeds": len(rows),
        "metric_means": {
            "average_accuracy": _mean(avg_accs),
            "task0_retention": _mean(task0_accs),
            "taskN_final": _mean(taskN_accs),
        },
        "metric_stds": {
            "average_accuracy": _std(avg_accs),
            "task0_retention": _std(task0_accs),
            "taskN_final": _std(taskN_accs),
        },
        "per_seed_metrics": {
            "average_accuracy": avg_accs,
            "task0_retention": task0_accs,
            "taskN_final": taskN_accs,
        },
    }


# ---------- main ----------


def main() -> None:
    args = parse_args()
    args.output_dir = Path(args.output_dir)
    args.baseline_checkpoint_dir = Path(args.baseline_checkpoint_dir)
    args.episodic_checkpoint_dir = Path(args.episodic_checkpoint_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.episodic_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seed_base, args.seed_base + args.n_seeds))
    print(
        f"Dual-substrate episodic pilot:\n"
        f"  T={args.T}\n"
        f"  seeds={seeds}\n"
        f"  novelty_threshold={args.novelty_threshold}\n"
        f"  blend_threshold={args.blend_threshold}, "
        f"blend_max={args.blend_max}\n"
        f"  baseline ckpt dir={args.baseline_checkpoint_dir}\n"
        f"  episodic ckpt dir={args.episodic_checkpoint_dir}\n"
        f"  output dir={args.output_dir}",
        flush=True,
    )

    # Override config defaults from CLI.
    base_cfg = EPISODIC_CONFIGS[_EPISODIC_METHOD]
    cfg = EpisodicConfig(
        name=base_cfg.name,
        novelty_threshold=args.novelty_threshold,
        retrieval_k=args.retrieval_k,
        max_entries=base_cfg.max_entries,
        blend_threshold=args.blend_threshold,
        blend_max=args.blend_max,
    )

    bench = PermutedMNIST.from_huggingface(
        num_tasks=args.T,
        seed=args.permutation_seed,
        cache_dir=args.cache_dir,
    )
    chroma_client = chromadb.Client()
    T = args.T

    method_blocks: list[dict[str, Any]] = []
    summary_blocks: list[dict[str, Any]] = []
    storage_diagnostics: list[dict[str, Any]] = []

    # ---- Episodic ----
    print(f"\n=== {_EPISODIC_METHOD} ===", flush=True)
    episodic_rows: list[list[float | None]] = []
    for seed in seeds:
        print(f"\n  --- seed {seed} ---", flush=True)
        ckpt = _episodic_checkpoint_path(
            args.episodic_checkpoint_dir, T, seed
        )
        if ckpt.exists():
            print(f"    loading checkpoint {ckpt}", flush=True)
            base, predictor, diagnostics = _load_episodic_checkpoint(
                ckpt, cfg, args,
                num_classes=bench.num_classes_per_task, seed=seed,
            )
        else:
            base, predictor, diagnostics = _train_episodic_one_seed(
                cfg, args, bench, T=T, seed=seed,
            )
            _save_episodic_checkpoint(
                ckpt, base, predictor.memory, diagnostics,
                config_dict=vars(args),
            )
            print(f"    checkpoint saved to {ckpt}", flush=True)

        t_eval = time.time()
        row = _eval_with_predictor(predictor, bench, T=T, args=args)
        defined = [v for v in row if v is not None]
        avg = statistics.fmean(defined) if defined else float("nan")
        t0 = row[0] if row else float("nan")
        tN = row[-1] if row else float("nan")
        print(
            f"    eval done in {time.time() - t_eval:.1f}s   "
            f"ACC={avg:.3f}  Task-0={t0:.3f}  Task-N={tN:.3f}  "
            f"memory={len(predictor.memory)} entries",
            flush=True,
        )
        # Per-task allocation summary.
        per_task_growth = diagnostics.get("per_task_memory_size", [])
        if per_task_growth:
            growth_str = ", ".join(
                f"t{e['task_index']}={e['memory_size']}"
                for e in per_task_growth
            )
            print(f"    per-task memory size: {growth_str}", flush=True)
        # Average novelty across training batches (decreasing curve
        # = memory is starting to cover the space).
        nov_means = diagnostics.get("per_batch_novelty_mean", [])
        if nov_means:
            print(
                f"    novelty mean (first 100 batches): "
                f"{statistics.fmean(nov_means[:100]):.3f}; "
                f"last 100: "
                f"{statistics.fmean(nov_means[-100:]):.3f}",
                flush=True,
            )
        episodic_rows.append(row)
        storage_diagnostics.append({
            "seed": int(seed),
            "final_memory_size": int(len(predictor.memory)),
            "per_task_memory_size": diagnostics.get("per_task_memory_size", []),
            "novelty_first_100_mean": (
                statistics.fmean(nov_means[:100])
                if len(nov_means) >= 1 else float("nan")
            ),
            "novelty_last_100_mean": (
                statistics.fmean(nov_means[-100:])
                if len(nov_means) >= 1 else float("nan")
            ),
        })

    method_blocks.append({
        "method": _EPISODIC_METHOD,
        "seeds": seeds,
        "results": [
            {"seed": int(s), "accuracy_matrix": _final_row_to_matrix(r, T)}
            for s, r in zip(seeds, episodic_rows)
        ],
    })
    summary_blocks.append({
        "method": _EPISODIC_METHOD,
        **_summarise_final_rows(episodic_rows),
    })

    # ---- Baseline (optional) ----
    if not args.skip_baseline:
        print(f"\n=== {_BASELINE_METHOD} (reference) ===", flush=True)
        baseline_rows: list[list[float | None]] = []
        baseline_seeds_used: list[int] = []
        for seed in seeds:
            print(f"\n  --- seed {seed} ---", flush=True)
            bpath = (
                args.baseline_checkpoint_dir
                / f"{_BASELINE_METHOD}_T{T}_seed{seed}.pt"
            )
            if not bpath.exists():
                if args.skip_missing_baseline:
                    print(
                        f"    baseline checkpoint missing at {bpath}; "
                        f"skipping (--skip-missing-baseline).",
                        flush=True,
                    )
                    continue
                print(
                    f"    baseline checkpoint missing at {bpath}; "
                    f"would need to retrain — not supported in this "
                    f"script. Pass --skip-missing-baseline to skip "
                    f"this seed, or run exp 27 first to populate "
                    f"the baseline checkpoints.",
                    flush=True,
                )
                continue
            print(f"    loading baseline {bpath}", flush=True)
            model = _load_baseline_checkpoint(
                bpath, args,
                num_classes=bench.num_classes_per_task, T=T,
                seed=seed, chroma_client=chroma_client,
            )
            t_eval = time.time()
            row = _eval_baseline_model(model, bench, T=T, args=args)
            defined = [v for v in row if v is not None]
            avg = statistics.fmean(defined) if defined else float("nan")
            t0 = row[0] if row else float("nan")
            tN = row[-1] if row else float("nan")
            print(
                f"    eval done in {time.time() - t_eval:.1f}s   "
                f"ACC={avg:.3f}  Task-0={t0:.3f}  Task-N={tN:.3f}",
                flush=True,
            )
            baseline_rows.append(row)
            baseline_seeds_used.append(seed)

        if baseline_rows:
            method_blocks.append({
                "method": _BASELINE_METHOD,
                "seeds": baseline_seeds_used,
                "results": [
                    {"seed": int(s), "accuracy_matrix": _final_row_to_matrix(r, T)}
                    for s, r in zip(baseline_seeds_used, baseline_rows)
                ],
            })
            summary_blocks.append({
                "method": _BASELINE_METHOD,
                **_summarise_final_rows(baseline_rows),
            })

    # ---- Write outputs ----
    ts = int(time.time())
    out_path = args.output_dir / f"{ts}_28_T{T}_dual_substrate.json"
    payload = {
        "experiment": "28_episodic_dual_substrate_eval",
        "num_tasks": int(T),
        "timestamp": ts,
        "config": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
        "is_partial": False,
        "methods_completed": [m["method"] for m in method_blocks],
        "methods_requested": [m["method"] for m in method_blocks],
        "configs_completed": [m["method"] for m in method_blocks],
        "configs_requested": [m["method"] for m in method_blocks],
        "methods": method_blocks,
        "summaries": summary_blocks,
        "storage_diagnostics": storage_diagnostics,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nWrote results JSON to {out_path}")

    # ---- Headline summary ----
    print()
    print("=" * 82)
    print(f"=== Dual-substrate episodic — T={T}, n={len(seeds)} ===")
    print("=" * 82)
    print(
        f"{'config':<40s} {'ACC':>10s} {'Task-0':>10s} "
        f"{'Task-N':>10s} {'memory':>10s}"
    )
    print("-" * 82)
    baseline_block = next(
        (s for s in summary_blocks if s["method"] == _BASELINE_METHOD),
        None,
    )
    bm = baseline_block["metric_means"] if baseline_block else None
    for s in summary_blocks:
        means = s["metric_means"]
        if s["method"] == _BASELINE_METHOD:
            print(
                f"{s['method'] + ' (ref)':<40s} "
                f"{means['average_accuracy']:>10.3f} "
                f"{means['task0_retention']:>10.3f} "
                f"{means['taskN_final']:>10.3f} "
                f"{'N/A':>10s}"
            )
        else:
            if storage_diagnostics:
                avg_mem = statistics.fmean(
                    d["final_memory_size"] for d in storage_diagnostics
                )
                mem_str = f"{avg_mem:.0f} avg"
            else:
                mem_str = "n/a"
            print(
                f"{s['method']:<40s} "
                f"{means['average_accuracy']:>10.3f} "
                f"{means['task0_retention']:>10.3f} "
                f"{means['taskN_final']:>10.3f} "
                f"{mem_str:>10s}"
            )
            if bm is not None:
                d_acc = (means["average_accuracy"] - bm["average_accuracy"]) * 100
                d_t0 = (means["task0_retention"] - bm["task0_retention"]) * 100
                d_tn = (means["taskN_final"] - bm["taskN_final"]) * 100
                print(
                    f"{'  (Δ vs baseline, pp):':<40s} "
                    f"{d_acc:>+10.2f} "
                    f"{d_t0:>+10.2f} "
                    f"{d_tn:>+10.2f}"
                )

    print()
    print(f"Output JSON: {out_path}")


if __name__ == "__main__":
    main()
