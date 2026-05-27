"""Experiment 30 — Functional regularization (LwF-style) continual eval.

Tests the new pivot: store ``(input, soft_target)`` pairs at the end
of each task, then during subsequent task training add a knowledge-
distillation loss that penalises deviation from the stored soft
targets. The model's weights are free to move; what's anchored is
its **function on selected past inputs**.

Three configs compared at T=15 n=2:

- ``cs_gated_cosine_developmental`` (baseline, unchanged):
  cosine gating only, no functional regularization.
- ``cs_functional_only``: plain ``MLPClassifier`` (no synapse layer,
  no cosine gating, no Hebbian state) + functional memory + LwF
  distillation loss. Isolates the contribution of functional
  regularization alone.
- ``cs_gated_cosine_functional``: ``SynapseAugmentedMLP`` with the
  scout_a095 cosine-gating stack + functional memory + LwF
  distillation. The composition test — both interventions act
  during training but on different axes (gating scales gradients
  on familiar inputs; LwF anchors the function on stored inputs).

The baseline is re-evaluated from the existing exp-27 checkpoints
under ``results/checkpoints/phase_d/`` when present (skips
training; same pattern exp 28 uses).

Run from the repo root::

    python experiments/30_functional_regularization_eval.py --T 15 --n_seeds 2

Output JSON follows the exp-23 schema so
``experiments/24_retention_analysis.py`` ingests it directly.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import chromadb
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from continual_synapse.baselines.naive_finetune import MLPClassifier, MLPConfig  # noqa: E402
from continual_synapse.baselines.synapse_finetune import SynapseAugmentedMLP  # noqa: E402
from continual_synapse.cold_storage.compression import CompressionSchedule  # noqa: E402
from continual_synapse.cold_storage.store import ColdStorage  # noqa: E402
from continual_synapse.consolidation.trigger import ConsolidationTrigger  # noqa: E402
from continual_synapse.evaluation.benchmarks import PermutedMNIST  # noqa: E402
from continual_synapse.evaluation.runner import set_seed  # noqa: E402
from continual_synapse.functional import (  # noqa: E402
    FunctionalMemory,
    distillation_loss,
)
from continual_synapse.reward.consistency import ConsistencyReward  # noqa: E402
from continual_synapse.reward.external import ExternalReward  # noqa: E402
from continual_synapse.reward.mixer import RewardMixer  # noqa: E402
from continual_synapse.reward.surprise import SurpriseReward  # noqa: E402
from continual_synapse.synapse_layer.layer import SynapseLayer  # noqa: E402
from continual_synapse.synapse_layer.modulation import SynapseModulation  # noqa: E402


_BASELINE_METHOD = "cs_gated_cosine_developmental"
# scout_a095 constants — needed to rebuild the baseline architecture
# in the (rare) case where a checkpoint is missing.
_SCOUT_A095_ALPHA = 0.95
_SCOUT_A095_TARGET = 50


# ---------- config dataclass ----------


@dataclass(frozen=True)
class FunctionalConfig:
    """Named config combining synapse machinery + functional reg flags."""

    name: str
    use_synapse: bool          # SynapseAugmentedMLP vs plain MLPClassifier
    use_cosine_gating: bool    # apply_gradient_gating each step
    use_hebbian: bool          # apply_hebbian_update each batch
    use_functional_reg: bool   # add LwF distillation loss


CONFIGS: dict[str, FunctionalConfig] = {
    "cs_gated_cosine_developmental": FunctionalConfig(
        name="cs_gated_cosine_developmental",
        use_synapse=True,
        use_cosine_gating=True,
        use_hebbian=True,
        use_functional_reg=False,
    ),
    "cs_functional_only": FunctionalConfig(
        name="cs_functional_only",
        use_synapse=False,
        use_cosine_gating=False,
        use_hebbian=False,
        use_functional_reg=True,
    ),
    "cs_gated_cosine_functional": FunctionalConfig(
        name="cs_gated_cosine_functional",
        use_synapse=True,
        use_cosine_gating=True,
        use_hebbian=True,
        use_functional_reg=True,
    ),
}
_DEFAULT_CONFIGS = list(CONFIGS.keys())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--T", type=int, default=15)
    p.add_argument("--n_seeds", type=int, default=2)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument(
        "--configs", nargs="+", default=_DEFAULT_CONFIGS,
        help=(
            "Which configs to evaluate. Default is all three "
            "(baseline + cs_functional_only + cs_gated_cosine_functional)."
        ),
    )
    # ---- Functional regularization hyperparameters ----
    p.add_argument(
        "--samples-per-task", type=int, default=100,
        help="Number of inputs sampled at each task end and stored "
             "with their soft targets. Sweep candidate.",
    )
    p.add_argument(
        "--lambda-reg", type=float, default=1.0,
        help="Weight on the distillation loss term. Sweep candidate.",
    )
    p.add_argument(
        "--temperature", type=float, default=2.0,
        help="Temperature in the distillation softmax. Sweep candidate.",
    )
    p.add_argument(
        "--max-memory", type=int, default=None,
        help="Hard cap on functional memory size. Default None = "
             "unbounded across tasks (T=15 × 100 = 1500 entries).",
    )
    p.add_argument(
        "--reg-batch-size", type=int, default=64,
        help="Batch size for the distillation forward pass each "
             "training step.",
    )
    # ---- I/O ----
    p.add_argument(
        "--output-dir", type=Path,
        default=_REPO_ROOT / "results" / "logs" / "functional",
    )
    p.add_argument(
        "--baseline-checkpoint-dir", type=Path,
        default=_REPO_ROOT / "results" / "checkpoints" / "phase_d",
        help="Where to look for cs_gated_cosine_developmental "
             "checkpoints saved by exp 27 (skips baseline training "
             "when found).",
    )
    p.add_argument(
        "--functional-checkpoint-dir", type=Path,
        default=_REPO_ROOT / "results" / "checkpoints" / "phase_f",
        help="Where to save the functional-reg checkpoints.",
    )
    p.add_argument(
        "--skip-missing-baseline", action="store_true",
        help="If a baseline checkpoint is missing, skip that seed "
             "instead of retraining.",
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
    # ---- Synapse-specific (only used when use_synapse=True) ----
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
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda", "mps"],
    )
    p.add_argument(
        "--cache-dir", default=str(_REPO_ROOT / "data" / "hf_cache")
    )
    return p.parse_args()


# ---------- model construction ----------


def _build_compression_schedule(args: argparse.Namespace) -> CompressionSchedule:
    n_thresholds = len(args.age_thresholds)
    all_tiers = (32, 16, 8, 4)
    tiers = all_tiers[: n_thresholds + 1]
    return CompressionSchedule(
        age_thresholds=tuple(args.age_thresholds),
        tier_precisions=tiers,
    )


def _build_plain_mlp(
    args: argparse.Namespace, num_classes: int, seed: int,
) -> MLPClassifier:
    set_seed(seed)
    return MLPClassifier(
        MLPConfig(
            input_dim=784, hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            num_hidden_layers=args.num_hidden_layers,
            dropout=args.dropout,
        )
    )


def _build_synapse_augmented(
    args: argparse.Namespace, *,
    num_classes: int, seed: int, chroma_client, T: int,
) -> SynapseAugmentedMLP:
    """Scout_a095-style synapse-augmented MLP with cosine gating."""
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
            f"exp30_synapse_T{T}_seed_{seed}_{time.time_ns()}"
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
        retrieval_k=args.retrieval_refresh_interval,
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
    model = _build_synapse_augmented(
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


# ---------- training loop ----------


def _train_one_seed(
    cfg: FunctionalConfig, args: argparse.Namespace, bench, T: int,
    seed: int, chroma_client,
) -> tuple[torch.nn.Module, FunctionalMemory, dict[str, Any]]:
    """Train one (config, seed) with the LwF-style training loop.

    The loop is intentionally inlined (rather than going through
    ContinualRunner) so the per-batch ``task_loss + λ * reg_loss``
    composition is explicit. Diagnostics for memory growth and the
    average reg-loss trajectory are captured per task.
    """
    print(
        f"    training {cfg.name}  T={T}  seed={seed}  "
        f"(synapse={cfg.use_synapse}, gating={cfg.use_cosine_gating}, "
        f"functional={cfg.use_functional_reg})...",
        flush=True,
    )
    t0 = time.time()

    if cfg.use_synapse:
        model = _build_synapse_augmented(
            args, num_classes=bench.num_classes_per_task,
            seed=seed, chroma_client=chroma_client, T=T,
        )
    else:
        model = _build_plain_mlp(
            args, num_classes=bench.num_classes_per_task, seed=seed,
        )
    model = model.to(args.device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
    )
    memory = FunctionalMemory(
        samples_per_task=args.samples_per_task,
        max_total=args.max_memory,
        rng_seed=seed,
    )

    diagnostics: dict[str, Any] = {
        "per_task_memory_added": [],
        "per_task_avg_reg_loss": [],
        "per_task_avg_task_loss": [],
        "final_memory_size": 0,
        "wall_time_s": 0.0,
    }

    # Re-seed the runner-equivalent so DataLoader shuffling is
    # deterministic per seed.
    set_seed(seed)

    for task_idx, task in enumerate(bench.tasks()):
        if cfg.use_synapse:
            model.notify_task_change(int(task_idx))

        model.train()
        loader = DataLoader(
            task.train, batch_size=args.batch_size, shuffle=True,
        )

        task_losses: list[float] = []
        reg_losses: list[float] = []

        for _ in range(args.epochs_per_task):
            for x, y in loader:
                x = x.to(args.device)
                y = y.to(args.device)

                optimizer.zero_grad()
                logits = model(x)
                task_loss = F.cross_entropy(logits, y)

                # Snapshot the SynapseAugmentedMLP's per-batch
                # caches BEFORE the optional memory forward — the
                # memory forward calls model(x_old) which would
                # otherwise overwrite ``_last_features`` and
                # ``_last_logits`` with the memory batch's values.
                # The post-step gating + Hebbian update need to see
                # the current TASK batch's features, not the
                # rehearsal batch's. (Plain MLPClassifier doesn't
                # cache, so this snapshot is a no-op for
                # cs_functional_only.)
                saved_last_features = getattr(model, "_last_features", None)
                saved_last_logits = getattr(model, "_last_logits", None)

                if cfg.use_functional_reg and len(memory) > 0:
                    sample = memory.sample_batch(
                        batch_size=args.reg_batch_size,
                        device=args.device,
                    )
                    if sample is not None:
                        x_old, soft_old = sample
                        # Force eval mode for the memory forward.
                        # Two reasons:
                        # (1) SynapseAugmentedMLP's multi-pass
                        #     observation buffer accumulates
                        #     activations on every training-mode
                        #     forward. With memory batch size
                        #     differing from task batch size, the
                        #     buffer's later stack+mean would fail
                        #     with "all observed activations must
                        #     share the same shape".
                        # (2) Dropout off + batchnorm-stable: the
                        #     distillation target is a fixed
                        #     soft_target snapshotted in eval-like
                        #     conditions; the student forward
                        #     should match that statistical regime.
                        # Gradients still flow through the forward;
                        # only the synapse observation path is
                        # gated on self.training.
                        was_training = model.training
                        model.eval()
                        try:
                            logits_old = model(x_old)
                        finally:
                            model.train(was_training)
                        reg_loss = distillation_loss(
                            logits_old, soft_old,
                            temperature=args.temperature,
                        )
                    else:
                        reg_loss = torch.zeros((), device=args.device)
                else:
                    reg_loss = torch.zeros((), device=args.device)

                # Restore the current-task caches so the synapse
                # machinery sees the right inputs. Only meaningful
                # when use_synapse=True; we restore unconditionally
                # because writing the same value back is cheap and
                # keeps the contract simple.
                if saved_last_features is not None:
                    model._last_features = saved_last_features
                if saved_last_logits is not None:
                    model._last_logits = saved_last_logits

                total = task_loss + args.lambda_reg * reg_loss
                total.backward()

                if cfg.use_cosine_gating:
                    model.apply_gradient_gating()

                optimizer.step()

                if cfg.use_hebbian:
                    model.apply_hebbian_update(training_target=y)

                task_losses.append(float(task_loss.item()))
                reg_losses.append(float(reg_loss.item()))

        diagnostics["per_task_avg_task_loss"].append(
            statistics.fmean(task_losses) if task_losses else 0.0
        )
        diagnostics["per_task_avg_reg_loss"].append(
            statistics.fmean(reg_losses) if reg_losses else 0.0
        )

        # End of task: snapshot soft targets if this config uses LwF.
        # Same eval-mode discipline as the per-batch memory forward:
        # SynapseAugmentedMLP's multi-pass observation buffer
        # accumulates on any training-mode forward, and the
        # 100-sample snapshot would leave 5×(100, hidden_dim)
        # entries in that buffer — which then explode when the next
        # task's 64-sample training forward tries to add (64, hidden)
        # observations to the same buffer.
        if cfg.use_functional_reg:
            task_pool = task.train.tensors[0]
            was_training = model.training
            model.eval()
            try:
                def _snapshot_forward(x: torch.Tensor) -> torch.Tensor:
                    return model(x)

                n_added = memory.record_task_end(
                    model_forward=_snapshot_forward,
                    task_inputs=task_pool,
                    task_id=int(task_idx),
                    device=args.device,
                )
            finally:
                model.train(was_training)
            diagnostics["per_task_memory_added"].append({
                "task_index": int(task_idx),
                "n_added": int(n_added),
                "memory_size_after": int(len(memory)),
            })

    diagnostics["final_memory_size"] = int(len(memory))
    diagnostics["wall_time_s"] = float(time.time() - t0)
    print(
        f"      trained in {diagnostics['wall_time_s']:.1f}s; "
        f"final memory size = {diagnostics['final_memory_size']}",
        flush=True,
    )
    return model, memory, diagnostics


def _save_functional_checkpoint(
    path: Path, model: torch.nn.Module, memory: FunctionalMemory,
    diagnostics: dict[str, Any], config_dict: dict[str, Any],
) -> None:
    payload = {
        "config": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in config_dict.items()
        },
        "model_state_dict": model.state_dict(),
        "memory": {
            "inputs": [t.detach().cpu() for t in memory.inputs],
            "soft_targets": [t.detach().cpu() for t in memory.soft_targets],
            "task_ids": list(memory.task_ids),
            "samples_per_task": memory.samples_per_task,
            "max_total": memory.max_total,
        },
        # Training-budget parameter that changes the weights but not
        # the architecture — stored explicitly so the loader can
        # refuse silent re-eval of a stale checkpoint when a sweep
        # bumps --epochs-per-task. Same fail-loud discipline as the
        # exp 31 maturity_target guard (commit 89380e9).
        "training_meta": {
            "epochs_per_task": int(config_dict.get("epochs_per_task", 1)),
        },
        "diagnostics": diagnostics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _functional_checkpoint_path(
    ckpt_dir: Path, config_name: str, T: int, seed: int,
) -> Path:
    return ckpt_dir / f"{config_name}_T{T}_seed{seed}.pt"


# ---------- eval ----------


def _eval_model(
    model: torch.nn.Module, bench, T: int, args: argparse.Namespace,
) -> list[float | None]:
    """Final row R[T-1, :] from model.forward on each task's test set."""
    model.eval()
    row: list[float | None] = []
    with torch.no_grad():
        for task in bench.tasks():
            x = task.test.tensors[0].to(args.device)
            y = task.test.tensors[1].to(args.device)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            row.append(float((preds == y).float().mean().item()))
    while len(row) < T:
        row.append(None)
    return row[:T]


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
    avg_accs: list[float] = []
    task0_accs: list[float] = []
    taskN_accs: list[float] = []
    fgt_proxies: list[float] = []
    for row in rows:
        defined = [v for v in row if v is not None]
        if defined:
            avg_accs.append(statistics.fmean(defined))
        if row and row[0] is not None:
            task0_accs.append(float(row[0]))
        if row and row[-1] is not None:
            taskN_accs.append(float(row[-1]))
        # Forgetting proxy: best-ever−final on Task-0 isn't available
        # without per-task R rows, so report (Task-N − Task-0) as a
        # signed gap (positive = Task-N higher than Task-0).
        if row and row[0] is not None and row[-1] is not None:
            fgt_proxies.append(float(row[-1] - row[0]))

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
            "forgetting_proxy": _mean(fgt_proxies),
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
    args.functional_checkpoint_dir = Path(args.functional_checkpoint_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.functional_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seed_base, args.seed_base + args.n_seeds))
    print(
        f"Functional regularization pilot (Phase F):\n"
        f"  T={args.T}\n"
        f"  seeds={seeds}\n"
        f"  configs={args.configs}\n"
        f"  samples_per_task={args.samples_per_task}\n"
        f"  lambda_reg={args.lambda_reg}\n"
        f"  temperature={args.temperature}\n"
        f"  max_memory={args.max_memory}\n"
        f"  reg_batch_size={args.reg_batch_size}\n"
        f"  baseline ckpt dir={args.baseline_checkpoint_dir}\n"
        f"  functional ckpt dir={args.functional_checkpoint_dir}\n"
        f"  output dir={args.output_dir}",
        flush=True,
    )

    unknown = [c for c in args.configs if c not in CONFIGS]
    if unknown:
        raise SystemExit(
            f"unknown config(s): {unknown}; available: {sorted(CONFIGS)}"
        )

    bench = PermutedMNIST.from_huggingface(
        num_tasks=args.T, seed=args.permutation_seed,
        cache_dir=args.cache_dir,
    )
    chroma_client = chromadb.Client()
    T = args.T

    method_blocks: list[dict[str, Any]] = []
    summary_blocks: list[dict[str, Any]] = []
    storage_diagnostics: list[dict[str, Any]] = []

    for cfg_name in args.configs:
        cfg = CONFIGS[cfg_name]
        print(f"\n=== {cfg_name} ===", flush=True)
        rows: list[list[float | None]] = []
        seeds_used: list[int] = []
        for seed in seeds:
            print(f"\n  --- seed {seed} ---", flush=True)
            if cfg_name == _BASELINE_METHOD:
                # Try to load from exp 27's checkpoints; fall back
                # to retraining (rare, only when ckpt missing).
                bpath = (
                    args.baseline_checkpoint_dir
                    / f"{cfg_name}_T{T}_seed{seed}.pt"
                )
                if bpath.exists():
                    print(f"    loading baseline {bpath}", flush=True)
                    model = _load_baseline_checkpoint(
                        bpath, args,
                        num_classes=bench.num_classes_per_task,
                        T=T, seed=seed, chroma_client=chroma_client,
                    )
                    memory = FunctionalMemory(
                        samples_per_task=args.samples_per_task,
                    )
                    diagnostics = {
                        "loaded_from": str(bpath),
                        "final_memory_size": 0,
                        "wall_time_s": 0.0,
                        "per_task_memory_added": [],
                        "per_task_avg_reg_loss": [],
                        "per_task_avg_task_loss": [],
                    }
                else:
                    if args.skip_missing_baseline:
                        print(
                            f"    baseline checkpoint missing at "
                            f"{bpath}; skipping seed (--skip-missing-baseline).",
                            flush=True,
                        )
                        continue
                    print(
                        f"    baseline checkpoint missing at {bpath}; "
                        f"retraining from scratch.",
                        flush=True,
                    )
                    model, memory, diagnostics = _train_one_seed(
                        cfg, args, bench, T=T, seed=seed,
                        chroma_client=chroma_client,
                    )
            else:
                # Functional variants — always train.
                ckpt = _functional_checkpoint_path(
                    args.functional_checkpoint_dir, cfg_name, T, seed,
                )
                if ckpt.exists():
                    print(f"    loading functional ckpt {ckpt}", flush=True)
                    payload = torch.load(
                        ckpt, map_location=args.device,
                        weights_only=False,
                    )
                    # Refuse to load a checkpoint trained under a
                    # different epochs_per_task budget — otherwise
                    # the sweep silently re-evaluates the prior
                    # invocation's checkpoint and reports identical
                    # numbers, exactly the failure mode that wasted
                    # the maturity-target sweep before commit
                    # 89380e9.
                    training_meta = payload.get("training_meta", {})
                    saved_epochs = training_meta.get("epochs_per_task")
                    if (
                        saved_epochs is not None
                        and int(saved_epochs) != int(args.epochs_per_task)
                    ):
                        raise RuntimeError(
                            f"Functional checkpoint at {ckpt} was trained "
                            f"with epochs_per_task={saved_epochs}, but the "
                            f"current run is configured for "
                            f"epochs_per_task={args.epochs_per_task}. "
                            f"Loading this checkpoint would silently "
                            f"re-evaluate the old training run and produce "
                            f"numbers identical to the original "
                            f"epochs_per_task={saved_epochs} configuration. "
                            f"Resolutions:\n"
                            f"  - delete the stale checkpoint and let the "
                            f"script retrain at the new epoch budget:\n"
                            f"      rm {ckpt}\n"
                            f"  - or pass --epochs-per-task {saved_epochs} "
                            f"to match the checkpoint's original training "
                            f"config."
                        )
                    if cfg.use_synapse:
                        model = _build_synapse_augmented(
                            args, num_classes=bench.num_classes_per_task,
                            seed=seed, chroma_client=chroma_client, T=T,
                        )
                    else:
                        model = _build_plain_mlp(
                            args, num_classes=bench.num_classes_per_task,
                            seed=seed,
                        )
                    model.load_state_dict(payload["model_state_dict"])
                    model.eval()
                    memory = FunctionalMemory(
                        samples_per_task=args.samples_per_task,
                        max_total=args.max_memory,
                    )
                    for inp, soft, tid in zip(
                        payload["memory"]["inputs"],
                        payload["memory"]["soft_targets"],
                        payload["memory"]["task_ids"],
                    ):
                        memory.inputs.append(inp)
                        memory.soft_targets.append(soft)
                        memory.task_ids.append(int(tid))
                    diagnostics = payload.get("diagnostics", {})
                else:
                    model, memory, diagnostics = _train_one_seed(
                        cfg, args, bench, T=T, seed=seed,
                        chroma_client=chroma_client,
                    )
                    _save_functional_checkpoint(
                        ckpt, model, memory, diagnostics,
                        config_dict=vars(args),
                    )
                    print(f"    checkpoint saved to {ckpt}", flush=True)

            # ---- Eval ----
            t_eval = time.time()
            row = _eval_model(model, bench, T=T, args=args)
            defined = [v for v in row if v is not None]
            avg = statistics.fmean(defined) if defined else float("nan")
            t0_v = row[0] if row else float("nan")
            tN_v = row[-1] if row else float("nan")
            print(
                f"    eval done in {time.time() - t_eval:.1f}s   "
                f"ACC={avg:.3f}  Task-0={t0_v:.3f}  Task-N={tN_v:.3f}  "
                f"memory={len(memory)} entries",
                flush=True,
            )
            per_task = diagnostics.get("per_task_memory_added", [])
            if per_task:
                growth_str = ", ".join(
                    f"t{e['task_index']}=+{e['n_added']}({e['memory_size_after']})"
                    for e in per_task
                )
                print(f"    per-task memory growth: {growth_str}", flush=True)
            reg_means = diagnostics.get("per_task_avg_reg_loss", [])
            if reg_means and any(r > 0 for r in reg_means):
                first = reg_means[0]
                last = reg_means[-1]
                non_zero = [r for r in reg_means if r > 0]
                print(
                    f"    avg reg_loss: first task={first:.4f}  "
                    f"last task={last:.4f}  "
                    f"non-zero tasks={len(non_zero)}/{len(reg_means)}",
                    flush=True,
                )
            rows.append(row)
            seeds_used.append(seed)

            # Storage diagnostics for this seed.
            storage_diagnostics.append({
                "method": cfg_name,
                "seed": int(seed),
                "final_memory_size": int(len(memory)),
                "per_task_counts": memory.per_task_counts(),
                "per_task_avg_reg_loss": reg_means,
                "per_task_avg_task_loss": diagnostics.get(
                    "per_task_avg_task_loss", []
                ),
            })

        if not rows:
            continue
        method_blocks.append({
            "method": cfg_name,
            "seeds": seeds_used,
            "results": [
                {"seed": int(s), "accuracy_matrix": _final_row_to_matrix(r, T)}
                for s, r in zip(seeds_used, rows)
            ],
        })
        summary_blocks.append({
            "method": cfg_name,
            **_summarise_final_rows(rows),
        })

    # ---- Write outputs ----
    ts = int(time.time())
    out_path = args.output_dir / f"{ts}_30_T{T}_functional.json"
    payload = {
        "experiment": "30_functional_regularization_eval",
        "num_tasks": int(T),
        "timestamp": ts,
        "config": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
        "is_partial": False,
        "methods_completed": [m["method"] for m in method_blocks],
        "methods_requested": list(args.configs),
        "configs_completed": [m["method"] for m in method_blocks],
        "configs_requested": list(args.configs),
        "methods": method_blocks,
        "summaries": summary_blocks,
        "storage_diagnostics": storage_diagnostics,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nWrote results JSON to {out_path}")

    # ---- Headline summary ----
    print()
    print("=" * 86)
    print(f"=== Functional regularization pilot — T={T}, n={len(seeds)} ===")
    print("=" * 86)
    print(
        f"{'config':<36s} {'ACC':>10s} {'Task-0':>10s} "
        f"{'Task-N':>10s} {'FGT':>10s} {'memory':>10s}"
    )
    print("-" * 86)
    baseline_block = next(
        (s for s in summary_blocks if s["method"] == _BASELINE_METHOD),
        None,
    )
    bm = baseline_block["metric_means"] if baseline_block else None
    for s in summary_blocks:
        means = s["metric_means"]
        # Compute average memory size for this method across seeds.
        mem_sizes = [
            d["final_memory_size"] for d in storage_diagnostics
            if d["method"] == s["method"]
        ]
        avg_mem = (
            statistics.fmean(mem_sizes) if mem_sizes else float("nan")
        )
        mem_str = (
            f"{avg_mem:.0f} avg" if avg_mem and avg_mem > 0 else "N/A"
        )
        tag = "(ref)" if s["method"] == _BASELINE_METHOD else ""
        print(
            f"{(s['method'] + ' ' + tag).strip():<36s} "
            f"{means['average_accuracy']:>10.3f} "
            f"{means['task0_retention']:>10.3f} "
            f"{means['taskN_final']:>10.3f} "
            f"{means['forgetting_proxy']:>+10.3f} "
            f"{mem_str:>10s}"
        )
        if bm is not None and s["method"] != _BASELINE_METHOD:
            d_acc = (means["average_accuracy"] - bm["average_accuracy"]) * 100
            d_t0 = (means["task0_retention"] - bm["task0_retention"]) * 100
            d_tn = (means["taskN_final"] - bm["taskN_final"]) * 100
            print(
                f"{'  (Δ vs baseline, pp):':<36s} "
                f"{d_acc:>+10.2f} "
                f"{d_t0:>+10.2f} "
                f"{d_tn:>+10.2f}"
            )

    print()
    print(f"Output JSON: {out_path}")


if __name__ == "__main__":
    main()
