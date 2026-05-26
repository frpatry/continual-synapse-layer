"""Experiment 25 — Cold Storage v2 retrieval-ensemble pilot.

Combined train-and-evaluate script:

1. For each (task length T, seed) checks whether
   ``results/checkpoints/phase_b/scout_a095_T{T}_seed{s}.pt`` exists.
   If not, trains ``scout_a095_validated`` from scratch on
   Permuted-MNIST at the requested T (same training pipeline as
   exp 23's scout_a095_validated, no other configs) and saves a
   checkpoint containing the trained model state_dict plus every
   cold-storage entry (embedding + document + metadata).
2. Loads each checkpoint, reconstructs the model + cold storage,
   and evaluates four predictors on the full test suite:
   - baseline ``scout_a095_baseline`` (bare model, no retrieval)
   - ``v2_mild``       k=5, tau=0.70, lambda=0.30
   - ``v2_moderate``   k=5, tau=0.80, lambda=0.50
   - ``v2_aggressive`` k=5, tau=0.50, lambda=0.50
3. Saves results to
   ``results/logs/retrieval_ensemble/<ts>_25_T{T}.json``
   in the same schema as exp 23 (per-method/seed accuracy matrices
   with NaN-filled upper rows and the real ``R[T-1, :]`` final
   row) so ``experiments/24_retention_analysis.py`` can ingest the
   files via ``--log-paths``.
4. Prints a clean pilot summary at the end:

::

    === Retrieval Ensemble v2 — Pilot Results ===
    At T=50:
      baseline scout_a095 (no retrieval):  ACC=X.XXX  Task-0=X.XXX
      v2_mild      (k=5, τ=0.70, λ=0.30):  ACC=X.XXX  Task-0=X.XXX  (Δ Task-0: +X.X pp)
      ...

Success criteria for the pilot (also printed at the end):

- At least one retrieval config improves Task-0 by >+5 pp at T=50
  vs the scout_a095 baseline.
- Aggregate ACC drops by no more than -2 pp.

If neither holds, the store-label derivation is probably too
noisy and the approach needs path-A retraining (see source
comment near label derivation).

Estimated runtime:
- Training (if checkpoints missing): T=15 ≈ 3 min/seed, T=50 ≈
  20 min/seed. With default n=3 seeds × 2 lengths = ~70 min.
- Evaluation: 4 predictors × 3 seeds × (T tasks × ~1 s/task per
  config). T=15 ≈ 1 min/seed/config; T=50 ≈ 4 min/seed/config.
  Total eval ≈ 75 min.
- Combined first run: ~2.5 hours. Re-runs with checkpoints in
  place: only the eval portion (~75 min).

Run from the repo root::

    python experiments/25_retrieval_ensemble_eval.py
"""

from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from continual_synapse.baselines.naive_finetune import MLPClassifier, MLPConfig  # noqa: E402
from continual_synapse.baselines.synapse_finetune import SynapseAugmentedMLP  # noqa: E402
from continual_synapse.cold_storage.compression import CompressionSchedule  # noqa: E402
from continual_synapse.cold_storage.store import ColdStorage  # noqa: E402
from continual_synapse.consolidation.trigger import ConsolidationTrigger  # noqa: E402
from continual_synapse.evaluation.benchmarks import PermutedMNIST  # noqa: E402
from continual_synapse.evaluation.runner import ContinualRunner, set_seed  # noqa: E402
from continual_synapse.inference.retrieval_ensemble import RetrievalEnsemble  # noqa: E402
from continual_synapse.reward.consistency import ConsistencyReward  # noqa: E402
from continual_synapse.reward.external import ExternalReward  # noqa: E402
from continual_synapse.reward.mixer import RewardMixer  # noqa: E402
from continual_synapse.reward.surprise import SurpriseReward  # noqa: E402
from continual_synapse.synapse_layer.layer import SynapseLayer  # noqa: E402
from continual_synapse.synapse_layer.modulation import SynapseModulation  # noqa: E402


# ---------- experiment configuration ----------


@dataclass
class RetrievalConfig:
    name: str
    k: int
    tau: float
    lambda_blend: float


_RETRIEVAL_CONFIGS: tuple[RetrievalConfig, ...] = (
    RetrievalConfig("v2_mild",       k=5, tau=0.70, lambda_blend=0.30),
    RetrievalConfig("v2_moderate",   k=5, tau=0.80, lambda_blend=0.50),
    RetrievalConfig("v2_aggressive", k=5, tau=0.50, lambda_blend=0.50),
)
_BASELINE_NAME = "scout_a095_baseline"
_SCOUT_A095_TARGET = 50
_SCOUT_A095_ALPHA = 0.95


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument(
        "--task-lengths", type=int, nargs="+", default=[15, 50],
    )
    p.add_argument(
        "--checkpoint-dir", type=Path,
        default=_REPO_ROOT / "results" / "checkpoints" / "phase_b",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=_REPO_ROOT / "results" / "logs" / "retrieval_ensemble",
    )
    p.add_argument(
        "--skip-training", action="store_true",
        help="Hard error if any required checkpoint is missing "
             "(instead of training to fill the gap).",
    )
    # ---- Training hyperparameters (mirror exp 23 scout_a095_validated) ----
    p.add_argument("--epochs-per-task", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-hidden-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--permutation-seed", type=int, default=42)
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
    return p.parse_args()


# ---------- model construction (mirrors exp 23 scout_a095_validated) ----------


def _build_compression_schedule(args) -> CompressionSchedule:
    n_thresholds = len(args.age_thresholds)
    all_tiers = (32, 16, 8, 4)
    tiers = all_tiers[: n_thresholds + 1]
    return CompressionSchedule(
        age_thresholds=tuple(args.age_thresholds),
        tier_precisions=tiers,
    )


def _build_scout_a095_model(
    args: argparse.Namespace,
    num_classes: int,
    seed: int,
    chroma_client,
    T: int,
) -> SynapseAugmentedMLP:
    """Construct a fresh scout_a095_validated model — the same path
    exp 23 uses, with the same hyperparameter wiring."""
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
        collection_name=f"exp25_scout_a095_T{T}_seed_{seed}_{time.time_ns()}",
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
        retrieval_k=args.retrieval_k,
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


# ---------- checkpoint save / load ----------


def _checkpoint_path(checkpoint_dir: Path, T: int, seed: int) -> Path:
    return checkpoint_dir / f"scout_a095_T{T}_seed{seed}.pt"


def _save_checkpoint(
    model: SynapseAugmentedMLP, path: Path, T: int, seed: int,
    config_dict: dict[str, Any],
) -> None:
    """Persist model state + all cold-storage entries to ``path``.

    ``config_dict`` is the args namespace as a dict (Path values
    coerced to str) — enough to rebuild the model architecture
    exactly at load time.
    """
    entries = (
        model.cold_storage.all_entries() if model.cold_storage is not None
        else []
    )
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "cold_storage_entries": [
            {
                "id": e.id,
                "embedding": list(e.embedding),
                "document": e.document,
                "metadata": dict(e.metadata),
            }
            for e in entries
        ],
        "config": {**config_dict, "T": int(T), "seed": int(seed)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, tmp)
    os.replace(tmp, path)


def _load_checkpoint(
    path: Path, args: argparse.Namespace, num_classes: int, T: int,
    seed: int, chroma_client,
) -> tuple[SynapseAugmentedMLP, ColdStorage]:
    """Rebuild model + cold storage from disk."""
    ckpt = torch.load(path, map_location=args.device, weights_only=False)
    model = _build_scout_a095_model(
        args, num_classes=num_classes, seed=seed,
        chroma_client=chroma_client, T=T,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    # Re-populate cold storage by inserting every entry. The
    # collection_name in _build_scout_a095_model is fresh per call
    # (timestamped), so this is a clean empty store we fill.
    for e_data in ckpt["cold_storage_entries"]:
        model.cold_storage.store_cluster(
            embedding=e_data["embedding"],
            metadata=e_data["metadata"],
            document=e_data["document"],
            entry_id=e_data["id"],
        )
    return model, model.cold_storage


# ---------- training (when checkpoint missing) ----------


def _train_scout_a095_and_save(
    args: argparse.Namespace, bench, T: int, seed: int,
    chroma_client, ckpt_path: Path,
) -> SynapseAugmentedMLP:
    """Train one scout_a095_validated model on ``bench`` at the given
    seed, save its state + cold storage to ``ckpt_path``, and return
    the trained model.

    The training loop mirrors exp 23's scout_a095_validated branch
    exactly (callbacks for gradient gating + Hebbian update; no
    other diagnostics needed for the checkpoint).
    """
    model = _build_scout_a095_model(
        args, num_classes=bench.num_classes_per_task,
        seed=seed, chroma_client=chroma_client, T=T,
    )

    def on_pre_step(i, task, m):
        m.apply_gradient_gating()

    def on_after_batch(i, task, m, x, y):
        m.apply_hebbian_update()

    runner = ContinualRunner(
        seed=seed,
        optimizer_factory=lambda p: torch.optim.SGD(
            p, lr=args.lr, momentum=args.momentum
        ),
        epochs_per_task=args.epochs_per_task,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        device=args.device,
        record_zero_shot=False,  # we don't need the full upper-triangular eval
        on_after_batch=on_after_batch,
        on_pre_optimizer_step=on_pre_step,
    )
    print(
        f"    training scout_a095_validated  T={T}  seed={seed}  "
        f"(target={_SCOUT_A095_TARGET}, alpha={_SCOUT_A095_ALPHA})...",
        flush=True,
    )
    t0 = time.time()
    runner.run(model, bench)
    elapsed = time.time() - t0
    print(
        f"      trained in {elapsed:.1f}s; "
        f"{model.consolidation_count} consolidations, "
        f"{model.cold_storage.count()} stored entries.",
        flush=True,
    )
    # Coerce Path values to str for the saved config so torch.save
    # doesn't choke on non-serialisable types (lesson from exp 20).
    config_dict = {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in vars(args).items()
    }
    _save_checkpoint(model, ckpt_path, T=T, seed=seed, config_dict=config_dict)
    print(f"      checkpoint saved to {ckpt_path}", flush=True)
    return model


# ---------- evaluation ----------


def _eval_predictor_on_all_tasks(
    predictor_call: Callable[[torch.Tensor], torch.Tensor],
    bench, T: int, batch_size: int, device: str,
) -> list[float]:
    """Run ``predictor_call`` on every task's test set; return a
    length-T list of accuracies."""
    accuracies: list[float] = []
    for j, task in enumerate(bench.tasks()):
        if j >= T:
            break
        loader = DataLoader(
            task.test, batch_size=batch_size, shuffle=False, drop_last=False,
        )
        correct = 0
        total = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = predictor_call(x)
            preds = logits.argmax(dim=-1)
            correct += int((preds == y).sum().item())
            total += int(y.numel())
        accuracies.append(correct / total if total > 0 else float("nan"))
    return accuracies


def _bare_model_predict(model: SynapseAugmentedMLP, x: torch.Tensor) -> torch.Tensor:
    """Eval-mode forward through the trained model with NO retrieval,
    NO modulator output (cs_gated has gate frozen at 0 anyway), NO
    multi-pass observation: just base.features → base.classify.
    Equivalent to what the runner computed during training-time
    evaluation for cs_gated."""
    model.eval()
    with torch.no_grad():
        h = model.base.features(x)
        return model.base.classify(h)


# ---------- JSON output (exp 23 schema) ----------


def _accuracy_matrix_with_only_final_row(
    final_row: list[float], T: int,
) -> list[list[float | None]]:
    """Return a T×T matrix as nested lists with NaN→None everywhere
    except the final row. exp 24 reads only ``am[T-1]`` so this
    matches the format without computing rows we don't have."""
    matrix: list[list[float | None]] = [
        [None] * T for _ in range(T - 1)
    ]
    matrix.append([float(v) if not math.isnan(v) else None for v in final_row])
    return matrix


def _safe_average_accuracy(final_row: list[float]) -> float:
    """Mean of finite entries in the final row."""
    finite = [v for v in final_row if not math.isnan(v)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def _build_method_block(
    name: str, seeds: list[int],
    final_rows_per_seed: list[list[float]], T: int,
) -> dict[str, Any]:
    """Match exp 23 schema for a single method entry."""
    results: list[dict[str, Any]] = []
    for seed, row in zip(seeds, final_rows_per_seed):
        avg_acc = _safe_average_accuracy(row)
        results.append(
            {
                "benchmark": "PermutedMNIST",
                "task_names": [f"task_{j}" for j in range(T)],
                "accuracy_matrix": _accuracy_matrix_with_only_final_row(row, T),
                "random_baseline": [0.1] * T,
                # Only ACC is well-defined from the final row alone; other
                # metrics need rows we don't have. exp 24 doesn't read
                # this block, so we keep it minimal but well-typed.
                "metrics": {
                    "average_accuracy": avg_acc,
                    "average_forgetting": float("nan"),
                    "backward_transfer": float("nan"),
                    "forward_transfer": None,
                    "per_task_final": {
                        f"task_{j}": float(v) if not math.isnan(v) else float("nan")
                        for j, v in enumerate(row)
                    },
                },
            }
        )
    return {"method": name, "seeds": list(seeds), "results": results}


def _build_summary_block(
    name: str, seeds: list[int], final_rows_per_seed: list[list[float]],
) -> dict[str, Any]:
    """Compute per-method aggregate ACC across seeds. Other metrics
    are NaN since the sparse matrices can't supply them."""
    accs = [_safe_average_accuracy(r) for r in final_rows_per_seed]
    finite = [a for a in accs if not math.isnan(a)]
    if not finite:
        mean_acc = float("nan")
        std_acc = 0.0
    else:
        n = len(finite)
        mean_acc = sum(finite) / n
        std_acc = (
            (sum((x - mean_acc) ** 2 for x in finite) / (n - 1)) ** 0.5
            if n > 1 else 0.0
        )
    return {
        "method": name,
        "n_seeds": len(seeds),
        "metric_means": {
            "average_accuracy": mean_acc,
            "average_forgetting": float("nan"),
            "backward_transfer": float("nan"),
            "forward_transfer": float("nan"),
        },
        "metric_stds": {
            "average_accuracy": std_acc,
            "average_forgetting": 0.0,
            "backward_transfer": 0.0,
            "forward_transfer": 0.0,
        },
        "per_seed_metrics": {
            "average_accuracy": accs,
            "average_forgetting": [float("nan")] * len(seeds),
            "backward_transfer": [float("nan")] * len(seeds),
            "forward_transfer": [float("nan")] * len(seeds),
        },
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    os.replace(tmp, path)


# ---------- pilot-summary printing ----------


@dataclass
class PilotSummary:
    T: int
    baseline_acc: float
    baseline_task0: float
    rows: list[dict[str, Any]] = field(default_factory=list)


def _format_pilot_summary(summary: PilotSummary) -> str:
    lines: list[str] = []
    lines.append(f"At T={summary.T}:")
    base_label = f"baseline scout_a095 (no retrieval):"
    lines.append(
        f"  {base_label:<48s}  "
        f"ACC={summary.baseline_acc:.3f}  Task-0={summary.baseline_task0:.3f}"
    )
    for row in summary.rows:
        label = (
            f"{row['name']:<14s} "
            f"(k={row['k']}, τ={row['tau']:.2f}, λ={row['lambda_blend']:.2f}):"
        )
        d_acc = (row["acc"] - summary.baseline_acc) * 100
        d_t0 = (row["task0"] - summary.baseline_task0) * 100
        lines.append(
            f"  {label:<48s}  "
            f"ACC={row['acc']:.3f}  Task-0={row['task0']:.3f}  "
            f"(Δ ACC: {d_acc:+.1f} pp, Δ Task-0: {d_t0:+.1f} pp)"
        )
    return "\n".join(lines)


def _check_success_criteria(
    summary: PilotSummary,
) -> tuple[bool, list[str]]:
    """Return (any_config_meets_both, per-criterion verdicts).

    Per spec: at least one config must improve Task-0 by >+5 pp AND
    must not drop aggregate ACC by more than -2 pp.
    """
    notes: list[str] = []
    any_hit = False
    for row in summary.rows:
        d_acc = (row["acc"] - summary.baseline_acc) * 100
        d_t0 = (row["task0"] - summary.baseline_task0) * 100
        hit_task0 = d_t0 > 5.0
        hit_acc = d_acc >= -2.0
        if hit_task0 and hit_acc:
            any_hit = True
            notes.append(
                f"  ✓ {row['name']}: Δ Task-0={d_t0:+.1f} pp, "
                f"Δ ACC={d_acc:+.1f} pp — passes both criteria"
            )
    if not any_hit:
        notes.append(
            "  ✗ no config passes both criteria — store-label noise "
            "is probably too high and the approach needs path-A "
            "retraining (see exp 25 docstring)."
        )
    return any_hit, notes


# ---------- main driver ----------


def _run_one_T(
    T: int, args: argparse.Namespace,
) -> tuple[PilotSummary, Path]:
    print(f"\n=== T={T} ===", flush=True)
    bench = PermutedMNIST.from_huggingface(
        num_tasks=T, seed=args.permutation_seed, cache_dir=args.cache_dir,
    )
    # One chroma client for this entire T sweep (per-seed collections
    # are timestamped + isolated; matches the pattern in exp 21/23).
    chroma_client = chromadb.Client()

    final_rows_baseline: list[list[float]] = []
    final_rows_per_config: dict[str, list[list[float]]] = {
        cfg.name: [] for cfg in _RETRIEVAL_CONFIGS
    }
    successfully_evaluated_seeds: list[int] = []

    for seed in args.seeds:
        print(f"\n  --- seed {seed} ---", flush=True)
        ckpt_path = _checkpoint_path(args.checkpoint_dir, T, seed)
        # Resolve checkpoint: load if present, train+save if missing.
        if ckpt_path.exists():
            print(f"    loading checkpoint {ckpt_path}", flush=True)
            model, _ = _load_checkpoint(
                ckpt_path, args, num_classes=bench.num_classes_per_task,
                T=T, seed=seed, chroma_client=chroma_client,
            )
        else:
            if args.skip_training:
                raise SystemExit(
                    f"checkpoint missing at {ckpt_path} and --skip-training "
                    f"is set."
                )
            model = _train_scout_a095_and_save(
                args, bench, T=T, seed=seed,
                chroma_client=chroma_client, ckpt_path=ckpt_path,
            )

        # ---- Baseline (no retrieval) ----
        t0 = time.time()
        baseline_row = _eval_predictor_on_all_tasks(
            lambda x: _bare_model_predict(model, x),
            bench, T=T, batch_size=args.eval_batch_size, device=args.device,
        )
        final_rows_baseline.append(baseline_row)
        baseline_avg = _safe_average_accuracy(baseline_row)
        print(
            f"    baseline eval done in {time.time() - t0:.1f}s   "
            f"ACC={baseline_avg:.3f}  Task-0={baseline_row[0]:.3f}",
            flush=True,
        )

        # ---- Each retrieval config ----
        for cfg in _RETRIEVAL_CONFIGS:
            ensemble = RetrievalEnsemble.from_model_and_storage(
                model, model.cold_storage,
                k=cfg.k, tau=cfg.tau, lambda_blend=cfg.lambda_blend,
                device=args.device,
            )
            t1 = time.time()
            cfg_row = _eval_predictor_on_all_tasks(
                lambda x: ensemble.predict(x),
                bench, T=T, batch_size=args.eval_batch_size,
                device=args.device,
            )
            final_rows_per_config[cfg.name].append(cfg_row)
            cfg_avg = _safe_average_accuracy(cfg_row)
            print(
                f"    {cfg.name:<14s} (k={cfg.k}, τ={cfg.tau:.2f}, "
                f"λ={cfg.lambda_blend:.2f}) eval done in "
                f"{time.time() - t1:.1f}s   "
                f"ACC={cfg_avg:.3f}  Task-0={cfg_row[0]:.3f}",
                flush=True,
            )
        successfully_evaluated_seeds.append(seed)

    # ---- Aggregate + write JSON ----
    method_blocks: list[dict[str, Any]] = []
    summary_blocks: list[dict[str, Any]] = []
    method_blocks.append(
        _build_method_block(
            _BASELINE_NAME, successfully_evaluated_seeds, final_rows_baseline, T
        )
    )
    summary_blocks.append(
        _build_summary_block(
            _BASELINE_NAME, successfully_evaluated_seeds, final_rows_baseline
        )
    )
    for cfg in _RETRIEVAL_CONFIGS:
        rows = final_rows_per_config[cfg.name]
        method_blocks.append(
            _build_method_block(
                cfg.name, successfully_evaluated_seeds, rows, T
            )
        )
        summary_blocks.append(
            _build_summary_block(
                cfg.name, successfully_evaluated_seeds, rows
            )
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    out_path = output_dir / f"{ts}_25_T{T}.json"
    payload = {
        "experiment": "25_retrieval_ensemble_eval",
        "num_tasks": int(T),
        "timestamp": ts,
        "config": vars(args),
        "is_partial": False,
        "methods_completed": [_BASELINE_NAME]
        + [c.name for c in _RETRIEVAL_CONFIGS],
        "methods_requested": [_BASELINE_NAME]
        + [c.name for c in _RETRIEVAL_CONFIGS],
        "configs_completed": [_BASELINE_NAME]
        + [c.name for c in _RETRIEVAL_CONFIGS],
        "configs_requested": [_BASELINE_NAME]
        + [c.name for c in _RETRIEVAL_CONFIGS],
        "methods": method_blocks,
        "summaries": summary_blocks,
        "retrieval_configs": [asdict(c) for c in _RETRIEVAL_CONFIGS],
    }
    _atomic_write_json(out_path, payload)
    print(f"\n  Wrote results JSON to {out_path}", flush=True)

    # ---- Pilot summary for this T ----
    baseline_acc_mean = _safe_average_accuracy(
        [_safe_average_accuracy(r) for r in final_rows_baseline]
    )
    baseline_task0_mean = _safe_average_accuracy(
        [r[0] for r in final_rows_baseline]
    )
    summary_rows: list[dict[str, Any]] = []
    for cfg in _RETRIEVAL_CONFIGS:
        rows = final_rows_per_config[cfg.name]
        cfg_acc_mean = _safe_average_accuracy(
            [_safe_average_accuracy(r) for r in rows]
        )
        cfg_task0_mean = _safe_average_accuracy([r[0] for r in rows])
        summary_rows.append(
            {
                "name": cfg.name, "k": cfg.k, "tau": cfg.tau,
                "lambda_blend": cfg.lambda_blend,
                "acc": cfg_acc_mean, "task0": cfg_task0_mean,
            }
        )
    return PilotSummary(
        T=T,
        baseline_acc=baseline_acc_mean,
        baseline_task0=baseline_task0_mean,
        rows=summary_rows,
    ), out_path


def main() -> None:
    args = parse_args()
    args.checkpoint_dir = Path(args.checkpoint_dir)
    args.output_dir = Path(args.output_dir)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Cold Storage v2 retrieval-ensemble pilot:\n"
        f"  task lengths: {args.task_lengths}\n"
        f"  seeds:        {args.seeds}\n"
        f"  checkpoints:  {args.checkpoint_dir}\n"
        f"  outputs:      {args.output_dir}",
        flush=True,
    )

    summaries: list[PilotSummary] = []
    output_paths: list[Path] = []
    for T in args.task_lengths:
        summary, out_path = _run_one_T(T, args)
        summaries.append(summary)
        output_paths.append(out_path)

    # ---- Headline pilot summary ----
    print()
    print("=" * 78)
    print("=== Retrieval Ensemble v2 — Pilot Results ===")
    print("=" * 78)
    for s in summaries:
        print()
        print(_format_pilot_summary(s))
        print()
        print("  Success-criterion check:")
        any_hit, notes = _check_success_criteria(s)
        for note in notes:
            print(note)
    print()
    print("Output JSONs (consumable by experiments/24_retention_analysis.py "
          "via --log-paths):")
    for p in output_paths:
        print(f"  {p}")
    print()
    print("To produce retention-curve figures from these logs::")
    paths_str = " ".join(str(p) for p in output_paths)
    print(f"  python experiments/24_retention_analysis.py \\")
    print(f"      --log-paths {paths_str} \\")
    print(f"      --fig-dir results/figures/retrieval_ensemble \\")
    print(f"      --analysis-path results/analysis/retrieval_ensemble_retention.json")


if __name__ == "__main__":
    main()
