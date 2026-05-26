"""Experiment 27 — Reward-as-confidence training + evaluation.

Trains four configs on Permuted-MNIST and compares them on the
continual-learning metrics:

- ``cs_gated_cosine_developmental`` (baseline, unchanged):
  cosine gating ON, constant scalar R. The anchor.
- ``cs_reward_developmental``: cosine gating OFF, per-sample R
  with developmental ``α``. Isolates the contribution of the
  reward signal in isolation.
- ``cosine_reward_developmental``: cosine gating ON, per-sample
  R with developmental ``α``. The composition we expect to be
  best — gating handles "what to protect", reward handles "what
  to learn".
- ``reward_only_static``: cosine gating OFF, per-sample R with
  constant ``α = 0.5``. Ablates the developmental component.

The script also tracks ``reward_statistics_per_consolidation``: for
every batch that triggers a consolidation event, it records the
mean and variance of the batch's normalised R. Late-stage variance
collapse (> 50 % drop from early values) is a tell-tale sign that
the developmental ``α`` cap was insufficient and the system has
stopped responding to within-batch informativeness differences.

Run from the repo root::

    python experiments/27_reward_as_confidence_eval.py --T 15 --n_seeds 3

Output: one results JSON under ``--output-dir`` (default
``results/logs/reward_confidence/``) in the exp-23-compatible
schema that ``experiments/24_retention_analysis.py`` reads
unchanged, plus a sidecar ``..._reward_stats.json`` carrying the
per-consolidation R statistics.

Checkpoints (one ``.pt`` per (config, seed)) land under
``--checkpoint-dir`` (default ``results/checkpoints/phase_d/``).
Re-runs skip training when the checkpoint already exists; the
eval phase is fast (~10 s/seed/config) so iterating on the
diagnostics or the summary print doesn't require re-training.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

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
from continual_synapse.evaluation.benchmarks import PermutedMNIST  # noqa: E402
from continual_synapse.evaluation.reporting import compute_metrics  # noqa: E402
from continual_synapse.evaluation.runner import ContinualRunner, set_seed  # noqa: E402
from continual_synapse.reward.confidence_reward import (  # noqa: E402
    compute_reward_signal,
    developmental_alpha,
    normalize_reward_batch,
)
from continual_synapse.reward.consistency import ConsistencyReward  # noqa: E402
from continual_synapse.reward.external import ExternalReward  # noqa: E402
from continual_synapse.reward.mixer import RewardMixer  # noqa: E402
from continual_synapse.reward.surprise import SurpriseReward  # noqa: E402
from continual_synapse.reward.training_configs import (  # noqa: E402
    REWARD_CONFIGS,
    RewardConfig,
)
from continual_synapse.synapse_layer.layer import SynapseLayer  # noqa: E402
from continual_synapse.synapse_layer.modulation import SynapseModulation  # noqa: E402


# Scout-a095 hyperparameters (mirror exp 25 / exp 23 so the baseline
# config behaves identically and the comparison stays fair).
_DEFAULT_CONFIGS: tuple[str, ...] = (
    "cs_gated_cosine_developmental",
    "cs_reward_developmental",
    "cosine_reward_developmental",
    "reward_only_static",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--T", type=int, default=15)
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--seed-base", type=int, default=0, help="First seed; uses seeds [base..base+n_seeds-1].")
    p.add_argument(
        "--configs", nargs="+", default=list(_DEFAULT_CONFIGS),
        help="Which reward configs to train + evaluate.",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=_REPO_ROOT / "results" / "logs" / "reward_confidence",
    )
    p.add_argument(
        "--checkpoint-dir", type=Path,
        default=_REPO_ROOT / "results" / "checkpoints" / "phase_d",
    )
    p.add_argument(
        "--skip-training", action="store_true",
        help="Hard error if any required checkpoint is missing.",
    )
    # ---- Training hyperparameters (mirror scout_a095) ----
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
    p.add_argument("--reward-mixer-gamma", type=float, default=1e-3,
                   help="Mixer trajectory decay (different gamma than reward.confidence_reward.gamma).")
    p.add_argument("--w-consistency", type=float, default=1.0)
    p.add_argument("--w-surprise", type=float, default=0.5)
    p.add_argument("--pressure-threshold", type=float, default=0.005)
    p.add_argument("--min-steps-between-consolidations", type=int, default=60)
    p.add_argument("--candidate-quantile", type=float, default=0.05)
    p.add_argument("--retrieval-k", type=int, default=4)
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


# ---------- model construction (config-aware variant of exp 25) ----------


def _build_compression_schedule(args: argparse.Namespace) -> CompressionSchedule:
    n_thresholds = len(args.age_thresholds)
    all_tiers = (32, 16, 8, 4)
    tiers = all_tiers[: n_thresholds + 1]
    return CompressionSchedule(
        age_thresholds=tuple(args.age_thresholds),
        tier_precisions=tiers,
    )


def _build_model_for_config(
    config: RewardConfig,
    args: argparse.Namespace,
    *,
    num_classes: int,
    seed: int,
    chroma_client,
    T: int,
) -> SynapseAugmentedMLP:
    """Build a scout-a095-style model with the gating flag set per
    config. All other hyperparameters mirror scout_a095_validated so
    the baseline cell in this comparison is bit-identical to exp 25's
    baseline."""
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
            f"exp27_{config.name}_T{T}_seed_{seed}_{time.time_ns()}"
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
        retrieval_k=args.retrieval_k,
        retrieval_refresh_interval=args.retrieval_refresh_interval,
        n_passes=args.n_passes,
        compression_sweep_interval=args.compression_sweep_interval,
        compression_schedule=_build_compression_schedule(args),
        gate_modulation_enabled=False,
        gradient_gating_enabled=config.gradient_gating_enabled,
        gradient_gating_alpha=args.gating_alpha,
        familiarity_mode="cosine",
        maturity_target_consolidations=args.maturity_target,
    )


# ---------- R-signal recorder ----------


@dataclass
class RewardEvent:
    """One observation of the per-sample reward at a consolidation."""
    event_index: int
    batch_index: int
    task_index: int
    consolidation_count: int
    R_mean: float
    R_var: float
    R_min: float
    R_max: float
    maturity: float
    alpha_used: float


class RewardStatsRecorder:
    """Buffer the R distribution at each consolidation event.

    The eval driver consults ``events`` after training to compute
    early/mid/late variance summaries and to write the sidecar JSON.
    """

    def __init__(self) -> None:
        self.events: list[RewardEvent] = []
        self._event_index = 0
        self._batch_index = 0
        self._last_consolidation_count = 0

    def make_after_batch(
        self,
        cfg: RewardConfig,
        base_after_batch: Callable,
    ) -> Callable:
        """Wrap ``base_after_batch`` to capture R stats on the batches
        that triggered a consolidation. The wrapper computes R the
        same way the base callback does (to avoid relying on the
        model's normalised internal state) and snapshots
        consolidation_count before/after to detect when an event
        actually fired."""

        def wrapped(task_index, task, model, x, y) -> None:
            # Compute R the same way the base callback will, BEFORE
            # the base callback consumes _last_logits (which gets
            # cleared at the end of apply_hebbian_update). When the
            # config doesn't use a reward signal, skip the snapshot
            # entirely — R is undefined for the baseline.
            R_snapshot = None
            alpha_used = float("nan")
            if (
                cfg.uses_reward_signal()
                and model._last_logits is not None
                and y is not None
                and y.numel() > 0
            ):
                alpha_used = cfg.alpha_for(model)
                R = compute_reward_signal(
                    model._last_logits,
                    y.to(torch.long),
                    alpha=alpha_used,
                    gamma=cfg.gamma,
                )
                R_snapshot = normalize_reward_batch(
                    R.detach().to(model._last_logits.dtype).cpu()
                )

            count_before = model.consolidation_count
            base_after_batch(task_index, task, model, x, y)
            count_after = model.consolidation_count

            if (
                R_snapshot is not None
                and count_after > count_before
            ):
                # A consolidation fired this batch.
                self.events.append(
                    RewardEvent(
                        event_index=self._event_index,
                        batch_index=self._batch_index,
                        task_index=int(task_index),
                        consolidation_count=int(count_after),
                        R_mean=float(R_snapshot.mean()),
                        R_var=float(R_snapshot.var(unbiased=False)),
                        R_min=float(R_snapshot.min()),
                        R_max=float(R_snapshot.max()),
                        maturity=float(model.current_maturity),
                        alpha_used=float(alpha_used),
                    )
                )
                self._event_index += 1
            self._batch_index += 1

        return wrapped

    def summary(self) -> dict[str, Any]:
        """Return early / mid / late R variance summary for the
        printed report. Empty when no events were recorded."""
        if not self.events:
            return {
                "n_events": 0,
                "early_mean_R_var": float("nan"),
                "mid_mean_R_var": float("nan"),
                "late_mean_R_var": float("nan"),
                "collapse_fraction": float("nan"),
            }
        vars_all = [e.R_var for e in self.events]
        n = len(vars_all)
        # Early = first min(5, n) events. Mid = ~middle 5. Late =
        # last min(5, n) events.
        k = max(1, min(5, n))
        mid_start = max(0, n // 2 - k // 2)
        early = vars_all[:k]
        mid = vars_all[mid_start : mid_start + k]
        late = vars_all[-k:]
        early_v = statistics.fmean(early) if early else float("nan")
        mid_v = statistics.fmean(mid) if mid else float("nan")
        late_v = statistics.fmean(late) if late else float("nan")
        collapse_fraction = (
            1.0 - (late_v / early_v) if early_v > 0 else float("nan")
        )
        return {
            "n_events": n,
            "early_mean_R_var": early_v,
            "mid_mean_R_var": mid_v,
            "late_mean_R_var": late_v,
            "collapse_fraction": collapse_fraction,
        }


# ---------- checkpoint save / load ----------


def _checkpoint_path(
    checkpoint_dir: Path, config_name: str, T: int, seed: int
) -> Path:
    return checkpoint_dir / f"{config_name}_T{T}_seed{seed}.pt"


def _save_checkpoint(
    model: SynapseAugmentedMLP, path: Path,
    config_name: str, T: int, seed: int,
    config_dict: dict[str, Any],
    reward_events: list[RewardEvent],
) -> None:
    entries = (
        model.cold_storage.all_entries() if model.cold_storage is not None
        else []
    )
    payload = {
        "config_name": config_name,
        "T": int(T),
        "seed": int(seed),
        "config": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in config_dict.items()
        },
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
        "reward_events": [asdict(e) for e in reward_events],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _load_checkpoint(
    path: Path,
    config: RewardConfig,
    args: argparse.Namespace,
    num_classes: int, T: int, seed: int, chroma_client,
) -> tuple[SynapseAugmentedMLP, list[RewardEvent]]:
    ckpt = torch.load(path, map_location=args.device, weights_only=False)
    model = _build_model_for_config(
        config, args, num_classes=num_classes,
        seed=seed, chroma_client=chroma_client, T=T,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for e_data in ckpt.get("cold_storage_entries", []):
        model.cold_storage.store_cluster(
            embedding=e_data["embedding"],
            metadata=e_data["metadata"],
            document=e_data["document"],
            entry_id=e_data["id"],
        )
    events_raw = ckpt.get("reward_events", [])
    events = [RewardEvent(**e) for e in events_raw]
    return model, events


# ---------- eval ----------


def _eval_one(
    model: SynapseAugmentedMLP, bench: PermutedMNIST, T: int,
    batch_size: int, device: str,
) -> list[list[float | None]]:
    """Return the final accuracy row R[T-1, :] as a list (None where
    a column was never evaluated). Sufficient for exp-24-compatible
    JSON; the upper-triangle of the R matrix is filled with None."""
    model.eval()
    final_row: list[float | None] = []
    with torch.no_grad():
        for task in bench.tasks():
            x_t = task.test.tensors[0].to(device)
            y_t = task.test.tensors[1].to(device)
            logits = model(x_t)
            preds = logits.argmax(dim=-1)
            acc = float((preds == y_t).float().mean().item())
            final_row.append(acc)
    # Pad up to T (defensive — bench should already produce T tasks).
    while len(final_row) < T:
        final_row.append(None)
    return final_row[:T]


def _build_accuracy_matrix(
    final_row: list[float | None], T: int
) -> list[list[float | None]]:
    """Wrap a single final row into a (T, T) matrix in exp-23
    schema: only the last row populated, all other rows ``None``."""
    am: list[list[float | None]] = []
    for i in range(T):
        if i < T - 1:
            am.append([None] * T)
        else:
            am.append(list(final_row))
    return am


# ---------- training driver ----------


def _train_one(
    config: RewardConfig, args: argparse.Namespace, bench, T: int,
    seed: int, chroma_client, ckpt_path: Path,
) -> tuple[SynapseAugmentedMLP, list[RewardEvent]]:
    """Train a single (config, seed) and save the checkpoint."""
    print(
        f"    training {config.name}  T={T}  seed={seed}  "
        f"(gating={config.gradient_gating_enabled}, "
        f"alpha_mode={config.alpha_mode})...",
        flush=True,
    )
    t0 = time.time()
    model = _build_model_for_config(
        config, args, num_classes=bench.num_classes_per_task,
        seed=seed, chroma_client=chroma_client, T=T,
    )

    # Build the config's base callbacks, then wrap on_after_batch
    # with the R-stats recorder.
    callbacks = config.make_callbacks()
    recorder = RewardStatsRecorder()
    callbacks["on_after_batch"] = recorder.make_after_batch(
        config, callbacks["on_after_batch"]
    )

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
        **callbacks,
    )
    runner.run(model, bench)
    elapsed = time.time() - t0
    print(
        f"      trained in {elapsed:.1f}s; "
        f"{model.consolidation_count} consolidations, "
        f"{model.cold_storage.count()} stored entries, "
        f"{len(recorder.events)} reward events.",
        flush=True,
    )
    _save_checkpoint(
        model, ckpt_path,
        config_name=config.name, T=T, seed=seed,
        config_dict=vars(args), reward_events=recorder.events,
    )
    print(f"      checkpoint saved to {ckpt_path}", flush=True)
    return model, recorder.events


# ---------- main ----------


def main() -> None:
    args = parse_args()
    args.checkpoint_dir = Path(args.checkpoint_dir)
    args.output_dir = Path(args.output_dir)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seed_base, args.seed_base + args.n_seeds))
    print(
        f"Reward-as-confidence pilot:\n"
        f"  T={args.T}\n"
        f"  seeds={seeds}\n"
        f"  configs={args.configs}\n"
        f"  checkpoints={args.checkpoint_dir}\n"
        f"  outputs={args.output_dir}",
        flush=True,
    )

    unknown = [c for c in args.configs if c not in REWARD_CONFIGS]
    if unknown:
        raise SystemExit(
            f"unknown config(s): {unknown}; available: "
            f"{sorted(REWARD_CONFIGS)}"
        )

    bench = PermutedMNIST.from_huggingface(
        num_tasks=args.T,
        seed=args.permutation_seed,
        cache_dir=args.cache_dir,
    )
    chroma_client = chromadb.Client()
    T = args.T

    # Per-(config, seed) results: final accuracy row + reward stats.
    method_blocks: list[dict[str, Any]] = []
    summary_blocks: list[dict[str, Any]] = []
    reward_stats_blocks: list[dict[str, Any]] = []

    for cfg_name in args.configs:
        cfg = REWARD_CONFIGS[cfg_name]
        print(f"\n=== {cfg_name} ===", flush=True)

        per_seed_final_rows: list[list[float | None]] = []
        per_seed_events: list[list[RewardEvent]] = []
        for seed in seeds:
            print(f"\n  --- seed {seed} ---", flush=True)
            ckpt_path = _checkpoint_path(
                args.checkpoint_dir, cfg_name, T, seed
            )
            if ckpt_path.exists():
                print(f"    loading checkpoint {ckpt_path}", flush=True)
                model, events = _load_checkpoint(
                    ckpt_path, cfg, args,
                    num_classes=bench.num_classes_per_task,
                    T=T, seed=seed, chroma_client=chroma_client,
                )
            else:
                if args.skip_training:
                    raise SystemExit(
                        f"checkpoint missing at {ckpt_path} and "
                        f"--skip-training is set."
                    )
                model, events = _train_one(
                    cfg, args, bench, T=T, seed=seed,
                    chroma_client=chroma_client, ckpt_path=ckpt_path,
                )
            # Evaluate on all T test sets to fill the final row.
            t_eval = time.time()
            final_row = _eval_one(
                model, bench, T=T,
                batch_size=args.eval_batch_size, device=args.device,
            )
            avg = (
                float(np.nanmean([v for v in final_row if v is not None]))
                if any(v is not None for v in final_row) else float("nan")
            )
            t0 = next((v for v in final_row if v is not None), float("nan"))
            t_n = final_row[-1] if final_row else float("nan")
            print(
                f"    eval done in {time.time() - t_eval:.1f}s   "
                f"ACC={avg:.3f}  Task-0={t0:.3f}  "
                f"Task-N={t_n if t_n is not None else float('nan'):.3f}",
                flush=True,
            )
            per_seed_final_rows.append(final_row)
            per_seed_events.append(events)

        # ---- Aggregate this config ----
        results_block: list[dict[str, Any]] = []
        avg_accs: list[float] = []
        task0_accs: list[float] = []
        taskN_accs: list[float] = []
        forgetting_proxies: list[float] = []
        for seed, final_row in zip(seeds, per_seed_final_rows):
            am = _build_accuracy_matrix(final_row, T)
            results_block.append({"seed": int(seed), "accuracy_matrix": am})
            defined = [v for v in final_row if v is not None]
            if defined:
                avg_accs.append(statistics.fmean(defined))
            if final_row and final_row[0] is not None:
                task0_accs.append(float(final_row[0]))
            if final_row and final_row[-1] is not None:
                taskN_accs.append(float(final_row[-1]))
            # Crude forgetting proxy: drop from final-task ACC to
            # task-0 ACC. Properly defined forgetting needs the full
            # R matrix; this is a comparable diagnostic only.
            if defined and final_row[-1] is not None and final_row[0] is not None:
                forgetting_proxies.append(
                    float(final_row[-1] - final_row[0])
                )

        method_blocks.append({
            "method": cfg_name,
            "seeds": seeds,
            "results": results_block,
        })
        summary_blocks.append({
            "method": cfg_name,
            "n_seeds": len(seeds),
            "metric_means": {
                "average_accuracy": (
                    statistics.fmean(avg_accs) if avg_accs else float("nan")
                ),
                "task0_retention": (
                    statistics.fmean(task0_accs)
                    if task0_accs else float("nan")
                ),
                "taskN_final": (
                    statistics.fmean(taskN_accs)
                    if taskN_accs else float("nan")
                ),
                "forgetting_proxy": (
                    statistics.fmean(forgetting_proxies)
                    if forgetting_proxies else float("nan")
                ),
            },
            "metric_stds": {
                "average_accuracy": (
                    statistics.stdev(avg_accs) if len(avg_accs) > 1 else 0.0
                ),
                "task0_retention": (
                    statistics.stdev(task0_accs)
                    if len(task0_accs) > 1 else 0.0
                ),
                "taskN_final": (
                    statistics.stdev(taskN_accs)
                    if len(taskN_accs) > 1 else 0.0
                ),
            },
            "per_seed_metrics": {
                "average_accuracy": avg_accs,
                "task0_retention": task0_accs,
                "taskN_final": taskN_accs,
            },
        })

        # Per-config reward statistics aggregate.
        per_seed_summaries: list[dict[str, Any]] = []
        for seed, events in zip(seeds, per_seed_events):
            recorder = RewardStatsRecorder()
            recorder.events = events
            recorder._event_index = len(events)
            per_seed_summaries.append({
                "seed": int(seed),
                **recorder.summary(),
            })
        reward_stats_blocks.append({
            "method": cfg_name,
            "per_seed": per_seed_summaries,
        })

    # ---- Write outputs ----
    ts = int(time.time())
    out_path = args.output_dir / f"{ts}_27_T{T}_path_d.json"
    sidecar = args.output_dir / f"{ts}_27_T{T}_path_d_reward_stats.json"
    payload = {
        "experiment": "27_reward_as_confidence_eval",
        "num_tasks": int(T),
        "timestamp": ts,
        "config": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
        "is_partial": False,
        "methods_completed": list(args.configs),
        "methods_requested": list(args.configs),
        "configs_completed": list(args.configs),
        "configs_requested": list(args.configs),
        "methods": method_blocks,
        "summaries": summary_blocks,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nWrote results JSON to {out_path}")

    sidecar_payload = {
        "experiment": "27_reward_as_confidence_eval",
        "timestamp": ts,
        "T": int(T),
        "seeds": seeds,
        "configs": list(args.configs),
        "per_config_summaries": reward_stats_blocks,
        "per_event": {
            cfg_name: [
                [asdict(e) for e in evs]
                for evs in _gather_events_for_config(
                    cfg_name, args.configs, method_blocks=method_blocks,
                    payload=payload, seeds=seeds,
                    checkpoint_dir=args.checkpoint_dir, T=T,
                )
            ]
            for cfg_name in args.configs
        },
    }
    sidecar.write_text(json.dumps(sidecar_payload, indent=2, default=str))
    print(f"Wrote sidecar R-stats JSON to {sidecar}")

    # ---- Headline summary ----
    print()
    print("=" * 82)
    print(f"=== Reward-as-confidence pilot — T={T}, n_seeds={len(seeds)} ===")
    print("=" * 82)
    print(
        f"{'config':<36s} {'ACC':>10s} {'Task-0':>10s} "
        f"{'Task-N':>10s} {'FGT proxy':>11s}"
    )
    print("-" * 82)
    baseline_name = "cs_gated_cosine_developmental"
    baseline_block = next(
        (s for s in summary_blocks if s["method"] == baseline_name), None
    )
    for s in summary_blocks:
        means = s["metric_means"]
        if s["method"] == baseline_name or baseline_block is None:
            print(
                f"{s['method']:<36s} "
                f"{means['average_accuracy']:>10.3f} "
                f"{means['task0_retention']:>10.3f} "
                f"{means['taskN_final']:>10.3f} "
                f"{means['forgetting_proxy']:>+11.3f}   "
                f"{'(baseline)' if s['method'] == baseline_name else ''}"
            )
        else:
            bm = baseline_block["metric_means"]
            d_acc = (means["average_accuracy"] - bm["average_accuracy"]) * 100
            d_t0 = (means["task0_retention"] - bm["task0_retention"]) * 100
            d_tn = (means["taskN_final"] - bm["taskN_final"]) * 100
            d_fgt = (
                means["forgetting_proxy"] - bm["forgetting_proxy"]
            ) * 100
            print(
                f"{s['method']:<36s} "
                f"{means['average_accuracy']:>10.3f} "
                f"{means['task0_retention']:>10.3f} "
                f"{means['taskN_final']:>10.3f} "
                f"{means['forgetting_proxy']:>+11.3f}"
            )
            print(
                f"{'  (Δ vs baseline, pp):':<36s} "
                f"{d_acc:>+10.2f} "
                f"{d_t0:>+10.2f} "
                f"{d_tn:>+10.2f} "
                f"{d_fgt:>+11.2f}"
            )

    print()
    print("R-signal health:")
    for block in reward_stats_blocks:
        cfg_name = block["method"]
        seed_summaries = block["per_seed"]
        n_events_total = sum(s["n_events"] for s in seed_summaries)
        if n_events_total == 0:
            print(
                f"  {cfg_name}: no reward events (constant-R config — "
                f"R signal not computed)"
            )
            continue
        early = statistics.fmean(
            s["early_mean_R_var"] for s in seed_summaries
            if not math.isnan(s["early_mean_R_var"])
        )
        mid = statistics.fmean(
            s["mid_mean_R_var"] for s in seed_summaries
            if not math.isnan(s["mid_mean_R_var"])
        )
        late = statistics.fmean(
            s["late_mean_R_var"] for s in seed_summaries
            if not math.isnan(s["late_mean_R_var"])
        )
        collapse = 1.0 - (late / early) if early > 0 else float("nan")
        print(
            f"  {cfg_name}:  early={early:.4f}  mid={mid:.4f}  "
            f"late={late:.4f}  (collapse fraction: "
            f"{collapse * 100:+.1f}%)"
        )
        if not math.isnan(collapse) and collapse > 0.5:
            print(
                f"    ⚠ late variance dropped > 50% from early — "
                f"R signal may have saturated; consider tightening "
                f"the developmental alpha cap."
            )

    print()
    print(f"Output JSON: {out_path}")
    print(f"R-stats sidecar: {sidecar}")


def _gather_events_for_config(
    cfg_name: str, all_configs: list[str], *,
    method_blocks, payload, seeds, checkpoint_dir, T,
) -> list[list]:
    """Pull the recorded reward events for each seed of a config by
    re-reading the checkpoint files (events were serialised at
    save time). Keeps the sidecar JSON self-contained for
    downstream analysis."""
    out: list[list] = []
    for seed in seeds:
        path = _checkpoint_path(checkpoint_dir, cfg_name, T, seed)
        if not path.exists():
            out.append([])
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        out.append(ckpt.get("reward_events", []))
    return out


if __name__ == "__main__":
    main()
