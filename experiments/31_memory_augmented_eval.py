"""Experiment 31 — Memory-augmented native architecture pilot.

Tests the architectural pivot: a model where external-memory access
is part of the forward pass during training. The model learns to
query memory, learns when to trust it via a sigmoid gate, and
learns the value representation it writes — all from the task loss,
end-to-end from batch 0.

Three configs at T=15 n=3:

- ``memory_augmented_native`` (the proposal):
  ``MemoryAugmentedMLP`` writes 100 samples per task into external
  memory at task end; the forward pass attends over those entries
  through learnable query / value projections.
- ``memory_augmented_no_memory`` (architectural control): the
  identical model never writes to memory. Forward always falls
  through the empty-memory branch, so the output equals
  ``classifier(encoder(x))``. Verifies that any wins on
  ``memory_augmented_native`` come from the *memory*, not just the
  architecture's parameter count or shape.
- ``cs_gated_cosine_functional`` (reference): the prior pilot's
  DER-equivalent baseline (ACC=0.904, Task-0=0.908 at T=15 n=4),
  reloaded from ``results/checkpoints/phase_f/`` when present.

Run from the repo root::

    python experiments/31_memory_augmented_eval.py --T 15 --n_seeds 3

Output JSON follows the exp-23 schema so
``experiments/24_retention_analysis.py`` reads it directly.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from continual_synapse.memory_augmented import MemoryAugmentedMLP  # noqa: E402
from continual_synapse.reward.consistency import ConsistencyReward  # noqa: E402
from continual_synapse.reward.external import ExternalReward  # noqa: E402
from continual_synapse.reward.mixer import RewardMixer  # noqa: E402
from continual_synapse.reward.surprise import SurpriseReward  # noqa: E402
from continual_synapse.synapse_layer.layer import SynapseLayer  # noqa: E402
from continual_synapse.synapse_layer.modulation import SynapseModulation  # noqa: E402


_NATIVE_METHOD = "memory_augmented_native"
_NO_MEM_METHOD = "memory_augmented_no_memory"
_REF_METHOD = "cs_gated_cosine_functional"
_SCOUT_A095_ALPHA = 0.95
_SCOUT_A095_TARGET = 50


# ---------- configs ----------


@dataclass(frozen=True)
class MemAugConfig:
    """Named training config for the memory-augmented pilot."""

    name: str
    use_memory_augmented: bool      # True ⇒ MemoryAugmentedMLP, False ⇒ reference path
    enable_memory_writes: bool      # only relevant when use_memory_augmented
    use_functional_reg: bool        # only for the reference cs_gated_cosine_functional path
    use_synapse: bool               # only for the reference path
    use_cosine_gating: bool
    use_hebbian: bool


CONFIGS: dict[str, MemAugConfig] = {
    _NATIVE_METHOD: MemAugConfig(
        name=_NATIVE_METHOD,
        use_memory_augmented=True, enable_memory_writes=True,
        use_functional_reg=False, use_synapse=False,
        use_cosine_gating=False, use_hebbian=False,
    ),
    _NO_MEM_METHOD: MemAugConfig(
        name=_NO_MEM_METHOD,
        use_memory_augmented=True, enable_memory_writes=False,
        use_functional_reg=False, use_synapse=False,
        use_cosine_gating=False, use_hebbian=False,
    ),
    _REF_METHOD: MemAugConfig(
        name=_REF_METHOD,
        use_memory_augmented=False, enable_memory_writes=False,
        use_functional_reg=True, use_synapse=True,
        use_cosine_gating=True, use_hebbian=True,
    ),
}
_DEFAULT_CONFIGS = list(CONFIGS.keys())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--T", type=int, default=15)
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument(
        "--configs", nargs="+", default=_DEFAULT_CONFIGS,
        help=f"Subset of {sorted(CONFIGS)}; default runs all three.",
    )
    # ---- Memory-augmented hyperparameters ----
    p.add_argument("--samples-per-task", type=int, default=100)
    p.add_argument("--key-dim", type=int, default=64)
    p.add_argument("--value-dim", type=int, default=64)
    p.add_argument(
        "--gate-init", type=float, default=0.0,
        help="Initial bias on the memory_gate logit. 0 → "
             "sigmoid(0)=0.5 (model starts undecided). Higher → "
             "biases initial gate open (model uses memory more "
             "from the start).",
    )
    p.add_argument(
        "--maturity-target", type=int, default=750,
        help=(
            "Memory size at which the developmental maturity floor "
            "crosses 0.5. The floor is a sigmoid of "
            "5 * (len(memory) / target - 1), so the gate is "
            "structurally forced toward 'use memory' as memory "
            "fills. Default 750 is calibrated for T=15 × "
            "samples_per_task=100 = 1500 entries: crossover lands "
            "at half-full, so tasks 1–7 are largely free of floor "
            "pressure and tasks 8–14 see the floor dominate. "
            "Try 300 (early crossover, aggressive memory pressure) "
            "or 1200 (late crossover) for sensitivity."
        ),
    )
    # ---- Reference-config (cs_gated_cosine_functional) hyperparameters ----
    p.add_argument("--lambda-reg", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--max-memory", type=int, default=None)
    p.add_argument("--reg-batch-size", type=int, default=64)
    # ---- I/O ----
    p.add_argument(
        "--output-dir", type=Path,
        default=_REPO_ROOT / "results" / "logs" / "memory_augmented",
    )
    p.add_argument(
        "--reference-checkpoint-dir", type=Path,
        default=_REPO_ROOT / "results" / "checkpoints" / "phase_f",
        help="Where to look for cs_gated_cosine_functional "
             "checkpoints (skips ref training when found).",
    )
    p.add_argument(
        "--memaug-checkpoint-dir", type=Path,
        default=_REPO_ROOT / "results" / "checkpoints" / "phase_g",
        help="Where to save MemoryAugmentedMLP checkpoints.",
    )
    p.add_argument(
        "--skip-missing-reference", action="store_true",
        help="Skip cs_gated_cosine_functional seeds whose checkpoints "
             "are missing rather than retraining.",
    )
    # ---- Training (shared) ----
    p.add_argument("--epochs-per-task", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-hidden-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--permutation-seed", type=int, default=42)
    # ---- Reference-only synapse stuff (mirrors scout_a095) ----
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


def _build_memory_augmented(
    args: argparse.Namespace, *, num_classes: int, seed: int,
) -> MemoryAugmentedMLP:
    set_seed(seed)
    return MemoryAugmentedMLP(
        input_dim=784,
        hidden_dim=args.hidden_dim,
        n_classes=num_classes,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
        n_encoder_layers=args.num_hidden_layers,
        gate_init=args.gate_init,
        maturity_target=args.maturity_target,
    )


def _build_compression_schedule(args: argparse.Namespace) -> CompressionSchedule:
    n_thresholds = len(args.age_thresholds)
    all_tiers = (32, 16, 8, 4)
    tiers = all_tiers[: n_thresholds + 1]
    return CompressionSchedule(
        age_thresholds=tuple(args.age_thresholds),
        tier_precisions=tiers,
    )


def _build_synapse_augmented(
    args: argparse.Namespace, *,
    num_classes: int, seed: int, chroma_client, T: int,
) -> SynapseAugmentedMLP:
    """Reference cs_gated_cosine_functional model (mirrors exp 30)."""
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
            f"exp31_ref_T{T}_seed_{seed}_{time.time_ns()}"
        ),
        client=chroma_client,
    )
    trigger = ConsolidationTrigger(
        avg_pressure_threshold=args.pressure_threshold,
        min_steps_between=args.min_steps_between_consolidations,
        candidate_quantile=args.candidate_quantile,
    )
    return SynapseAugmentedMLP(
        base, synapse, SynapseModulation(init_gate=0.0),
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


def _load_reference_checkpoint(
    path: Path, args: argparse.Namespace, *,
    num_classes: int, T: int, seed: int, chroma_client,
) -> tuple[SynapseAugmentedMLP, FunctionalMemory]:
    """Load a cs_gated_cosine_functional checkpoint produced by exp 30."""
    ckpt = torch.load(path, map_location=args.device, weights_only=False)
    model = _build_synapse_augmented(
        args, num_classes=num_classes, seed=seed,
        chroma_client=chroma_client, T=T,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    memory = FunctionalMemory(
        samples_per_task=args.samples_per_task,
        max_total=args.max_memory,
    )
    mem_state = ckpt.get("memory", {})
    for inp, soft, tid in zip(
        mem_state.get("inputs", []),
        mem_state.get("soft_targets", []),
        mem_state.get("task_ids", []),
    ):
        memory.inputs.append(inp)
        memory.soft_targets.append(soft)
        memory.task_ids.append(int(tid))
    return model, memory


# ---------- training loops ----------


def _train_memaug_one_seed(
    cfg: MemAugConfig, args: argparse.Namespace, bench, T: int, seed: int,
) -> tuple[MemoryAugmentedMLP, dict[str, Any]]:
    """Train the MemoryAugmentedMLP for one seed. Standard backprop
    on cross-entropy. The memory access path is in the forward from
    batch 0. At task end (when enabled), write 100 samples into
    memory."""
    print(
        f"    training {cfg.name}  T={T}  seed={seed}  "
        f"(memaug=True, writes={cfg.enable_memory_writes})...",
        flush=True,
    )
    t0 = time.time()
    model = _build_memory_augmented(
        args, num_classes=bench.num_classes_per_task, seed=seed,
    ).to(args.device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
    )

    diagnostics: dict[str, Any] = {
        "per_task_memory_added": [],
        "per_task_learned_gate_mean": [],
        "per_task_effective_gate_mean": [],
        "per_task_maturity_floor": [],
        "per_task_attention_entropy": [],
        "per_task_avg_task_loss": [],
        "final_memory_size": 0,
        "wall_time_s": 0.0,
        "maturity_target": int(args.maturity_target),
    }

    set_seed(seed)
    for task_idx, task in enumerate(bench.tasks()):
        model.train()
        loader = DataLoader(
            task.train, batch_size=args.batch_size, shuffle=True,
        )
        task_losses: list[float] = []
        learned_gates: list[float] = []
        effective_gates: list[float] = []
        floors: list[float] = []
        attn_entropies: list[float] = []

        for _ in range(args.epochs_per_task):
            for x, y in loader:
                x = x.to(args.device)
                y = y.to(args.device)
                optimizer.zero_grad()
                logits, diag = model(x, return_diagnostics=True)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
                task_losses.append(float(loss.item()))
                learned_gates.append(diag["learned_gate_mean"])
                effective_gates.append(diag["effective_gate_mean"])
                floors.append(diag["maturity_floor"])
                attn_entropies.append(diag["attention_entropy"])

        diagnostics["per_task_avg_task_loss"].append(
            statistics.fmean(task_losses) if task_losses else 0.0
        )
        diagnostics["per_task_learned_gate_mean"].append(
            statistics.fmean(learned_gates) if learned_gates else 0.0
        )
        diagnostics["per_task_effective_gate_mean"].append(
            statistics.fmean(effective_gates) if effective_gates else 0.0
        )
        diagnostics["per_task_maturity_floor"].append(
            statistics.fmean(floors) if floors else 0.0
        )
        diagnostics["per_task_attention_entropy"].append(
            statistics.fmean(attn_entropies) if attn_entropies else 0.0
        )

        # End of task: write samples into memory (config-gated).
        if cfg.enable_memory_writes:
            n_pool = task.train.tensors[0].shape[0]
            n = min(args.samples_per_task, n_pool)
            idx = torch.randperm(n_pool)[:n]
            sampled = task.train.tensors[0][idx].to(args.device)
            model.write_batch_to_memory(sampled, task_id=int(task_idx))
            diagnostics["per_task_memory_added"].append({
                "task_index": int(task_idx),
                "n_added": int(n),
                "memory_size_after": int(len(model.memory)),
            })

    diagnostics["final_memory_size"] = int(len(model.memory))
    diagnostics["wall_time_s"] = float(time.time() - t0)
    print(
        f"      trained in {diagnostics['wall_time_s']:.1f}s; "
        f"final memory size = {diagnostics['final_memory_size']}",
        flush=True,
    )
    return model, diagnostics


def _train_reference_one_seed(
    cfg: MemAugConfig, args: argparse.Namespace, bench, T: int,
    seed: int, chroma_client,
) -> tuple[SynapseAugmentedMLP, FunctionalMemory, dict[str, Any]]:
    """cs_gated_cosine_functional retraining (mirrors exp 30's
    composition loop). Only used when no exp-30 checkpoint exists
    for this (T, seed)."""
    print(
        f"    training {cfg.name}  T={T}  seed={seed}  "
        f"(synapse=True, gating=True, functional=True)...",
        flush=True,
    )
    t0 = time.time()
    model = _build_synapse_augmented(
        args, num_classes=bench.num_classes_per_task,
        seed=seed, chroma_client=chroma_client, T=T,
    )
    model = model.to(args.device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum,
    )
    memory = FunctionalMemory(
        samples_per_task=args.samples_per_task,
        max_total=args.max_memory, rng_seed=seed,
    )
    diagnostics: dict[str, Any] = {
        "wall_time_s": 0.0,
        "final_memory_size": 0,
    }

    set_seed(seed)
    for task_idx, task in enumerate(bench.tasks()):
        model.notify_task_change(int(task_idx))
        model.train()
        loader = DataLoader(
            task.train, batch_size=args.batch_size, shuffle=True,
        )
        for _ in range(args.epochs_per_task):
            for x, y in loader:
                x = x.to(args.device)
                y = y.to(args.device)
                optimizer.zero_grad()
                logits = model(x)
                task_loss = F.cross_entropy(logits, y)
                saved_lf = getattr(model, "_last_features", None)
                saved_ll = getattr(model, "_last_logits", None)
                if len(memory) > 0:
                    sample = memory.sample_batch(
                        batch_size=args.reg_batch_size,
                        device=args.device,
                    )
                    if sample is not None:
                        x_old, soft_old = sample
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
                if saved_lf is not None:
                    model._last_features = saved_lf
                if saved_ll is not None:
                    model._last_logits = saved_ll
                total = task_loss + args.lambda_reg * reg_loss
                total.backward()
                model.apply_gradient_gating()
                optimizer.step()
                model.apply_hebbian_update(training_target=y)
        # End-of-task snapshot
        task_pool = task.train.tensors[0]
        was_training = model.training
        model.eval()
        try:
            memory.record_task_end(
                model_forward=lambda v: model(v),
                task_inputs=task_pool,
                task_id=int(task_idx),
                device=args.device,
            )
        finally:
            model.train(was_training)

    diagnostics["final_memory_size"] = int(len(memory))
    diagnostics["wall_time_s"] = float(time.time() - t0)
    print(
        f"      trained in {diagnostics['wall_time_s']:.1f}s; "
        f"final memory size = {diagnostics['final_memory_size']}",
        flush=True,
    )
    return model, memory, diagnostics


# ---------- checkpoint save/load (memaug only) ----------


def _memaug_checkpoint_path(
    ckpt_dir: Path, config_name: str, T: int, seed: int,
) -> Path:
    return ckpt_dir / f"{config_name}_T{T}_seed{seed}.pt"


def _save_memaug_checkpoint(
    path: Path, model: MemoryAugmentedMLP,
    diagnostics: dict[str, Any], config_dict: dict[str, Any],
) -> None:
    payload = {
        "config": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in config_dict.items()
        },
        # state_dict includes the memory buffers (keys, values,
        # task_ids), so reload reconstitutes the full memory.
        "model_state_dict": model.state_dict(),
        "memory_meta": {
            "key_dim": model.memory.key_dim,
            "value_dim": model.memory.value_dim,
            "final_size": int(len(model.memory)),
        },
        "diagnostics": diagnostics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _load_memaug_checkpoint(
    path: Path, args: argparse.Namespace, *,
    num_classes: int, seed: int,
) -> tuple[MemoryAugmentedMLP, dict[str, Any]]:
    ckpt = torch.load(path, map_location=args.device, weights_only=False)
    model = _build_memory_augmented(
        args, num_classes=num_classes, seed=seed,
    ).to(args.device)
    # Need to make the buffer shapes match the saved memory size
    # before load_state_dict can replace them.
    meta = ckpt.get("memory_meta", {})
    final_size = int(meta.get("final_size", 0))
    if final_size > 0:
        with torch.no_grad():
            model.memory.keys = torch.empty(
                final_size, model.memory.key_dim, device=args.device,
            )
            model.memory.values = torch.empty(
                final_size, model.memory.value_dim, device=args.device,
            )
            model.memory.task_ids = torch.empty(
                final_size, dtype=torch.long, device=args.device,
            )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt.get("diagnostics", {})


# ---------- eval ----------


def _eval_model(
    model: torch.nn.Module, bench, T: int, args: argparse.Namespace,
) -> list[float | None]:
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
    final_row: list[float | None], T: int,
) -> list[list[float | None]]:
    am: list[list[float | None]] = []
    for i in range(T):
        if i < T - 1:
            am.append([None] * T)
        else:
            am.append(list(final_row))
    return am


def _summarise_final_rows(rows: list[list[float | None]]) -> dict[str, Any]:
    avg_accs: list[float] = []
    task0: list[float] = []
    taskN: list[float] = []
    fgt: list[float] = []
    for row in rows:
        defined = [v for v in row if v is not None]
        if defined:
            avg_accs.append(statistics.fmean(defined))
        if row and row[0] is not None:
            task0.append(float(row[0]))
        if row and row[-1] is not None:
            taskN.append(float(row[-1]))
        if row and row[0] is not None and row[-1] is not None:
            fgt.append(float(row[-1] - row[0]))

    def _m(xs): return statistics.fmean(xs) if xs else float("nan")
    def _s(xs): return statistics.stdev(xs) if len(xs) > 1 else 0.0

    return {
        "n_seeds": len(rows),
        "metric_means": {
            "average_accuracy": _m(avg_accs),
            "task0_retention": _m(task0),
            "taskN_final": _m(taskN),
            "forgetting_proxy": _m(fgt),
        },
        "metric_stds": {
            "average_accuracy": _s(avg_accs),
            "task0_retention": _s(task0),
            "taskN_final": _s(taskN),
        },
        "per_seed_metrics": {
            "average_accuracy": avg_accs,
            "task0_retention": task0,
            "taskN_final": taskN,
        },
    }


# ---------- main ----------


def main() -> None:
    args = parse_args()
    args.output_dir = Path(args.output_dir)
    args.reference_checkpoint_dir = Path(args.reference_checkpoint_dir)
    args.memaug_checkpoint_dir = Path(args.memaug_checkpoint_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.memaug_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seed_base, args.seed_base + args.n_seeds))
    print(
        f"Memory-augmented native pilot (Phase G):\n"
        f"  T={args.T}\n"
        f"  seeds={seeds}\n"
        f"  configs={args.configs}\n"
        f"  samples_per_task={args.samples_per_task}\n"
        f"  key_dim={args.key_dim}  value_dim={args.value_dim}  "
        f"gate_init={args.gate_init}\n"
        f"  memaug ckpt dir={args.memaug_checkpoint_dir}\n"
        f"  reference ckpt dir={args.reference_checkpoint_dir}\n"
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
            if cfg.use_memory_augmented:
                ckpt = _memaug_checkpoint_path(
                    args.memaug_checkpoint_dir, cfg_name, T, seed,
                )
                if ckpt.exists():
                    print(f"    loading {ckpt}", flush=True)
                    model, diagnostics = _load_memaug_checkpoint(
                        ckpt, args,
                        num_classes=bench.num_classes_per_task,
                        seed=seed,
                    )
                else:
                    model, diagnostics = _train_memaug_one_seed(
                        cfg, args, bench, T=T, seed=seed,
                    )
                    _save_memaug_checkpoint(
                        ckpt, model, diagnostics, config_dict=vars(args),
                    )
                    print(f"    checkpoint saved to {ckpt}", flush=True)
            else:
                # cs_gated_cosine_functional reference path
                ref_ckpt = (
                    args.reference_checkpoint_dir
                    / f"{cfg_name}_T{T}_seed{seed}.pt"
                )
                if ref_ckpt.exists():
                    print(f"    loading reference {ref_ckpt}", flush=True)
                    model, _memory = _load_reference_checkpoint(
                        ref_ckpt, args,
                        num_classes=bench.num_classes_per_task,
                        T=T, seed=seed, chroma_client=chroma_client,
                    )
                    diagnostics = {
                        "loaded_from": str(ref_ckpt),
                        "final_memory_size": int(len(_memory)),
                    }
                else:
                    if args.skip_missing_reference:
                        print(
                            f"    reference checkpoint missing at "
                            f"{ref_ckpt}; skipping (--skip-missing-reference).",
                            flush=True,
                        )
                        continue
                    print(
                        f"    reference checkpoint missing at {ref_ckpt}; "
                        f"retraining from scratch.",
                        flush=True,
                    )
                    model, _memory, diagnostics = _train_reference_one_seed(
                        cfg, args, bench, T=T, seed=seed,
                        chroma_client=chroma_client,
                    )

            t_eval = time.time()
            row = _eval_model(model, bench, T=T, args=args)
            defined = [v for v in row if v is not None]
            avg = statistics.fmean(defined) if defined else float("nan")
            t0_v = row[0] if row else float("nan")
            tN_v = row[-1] if row else float("nan")
            print(
                f"    eval done in {time.time() - t_eval:.1f}s   "
                f"ACC={avg:.3f}  Task-0={t0_v:.3f}  Task-N={tN_v:.3f}",
                flush=True,
            )
            # Memaug-specific diagnostic print: gate + floor + attention
            if cfg.use_memory_augmented:
                learned_trace = diagnostics.get("per_task_learned_gate_mean", [])
                effective_trace = diagnostics.get(
                    "per_task_effective_gate_mean", []
                )
                floor_trace = diagnostics.get("per_task_maturity_floor", [])
                ent_trace = diagnostics.get("per_task_attention_entropy", [])
                per_task = diagnostics.get("per_task_memory_added", [])
                mat_target = diagnostics.get("maturity_target", 0)
                if per_task:
                    growth_str = ", ".join(
                        f"t{e['task_index']}={e['memory_size_after']}"
                        for e in per_task
                    )
                    print(f"    memory: {growth_str}", flush=True)
                # Show learned vs floor vs effective so the operator
                # can see if the model is genuinely opening the gate
                # above the floor, or just riding it.
                if learned_trace:
                    print(
                        f"    learned_gate (model wants):  "
                        f"t0={learned_trace[0]:.3f}  "
                        f"tmid={learned_trace[len(learned_trace)//2]:.3f}  "
                        f"tlast={learned_trace[-1]:.3f}",
                        flush=True,
                    )
                if floor_trace:
                    print(
                        f"    maturity_floor (target={mat_target}):  "
                        f"t0={floor_trace[0]:.3f}  "
                        f"tmid={floor_trace[len(floor_trace)//2]:.3f}  "
                        f"tlast={floor_trace[-1]:.3f}",
                        flush=True,
                    )
                if effective_trace:
                    print(
                        f"    effective_gate (applied):    "
                        f"t0={effective_trace[0]:.3f}  "
                        f"tmid={effective_trace[len(effective_trace)//2]:.3f}  "
                        f"tlast={effective_trace[-1]:.3f}",
                        flush=True,
                    )
                if ent_trace and any(e > 0 for e in ent_trace):
                    last_nz = [e for e in ent_trace if e > 0]
                    print(
                        f"    attention entropy (last non-zero): "
                        f"{last_nz[-1]:.3f}  "
                        f"(uniform would be ln(N_mem))",
                        flush=True,
                    )
            rows.append(row)
            seeds_used.append(seed)
            storage_diagnostics.append({
                "method": cfg_name,
                "seed": int(seed),
                **{
                    k: v for k, v in diagnostics.items()
                    if k != "loaded_from"
                },
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
    out_path = args.output_dir / f"{ts}_31_T{T}_memory_augmented.json"
    payload = {
        "experiment": "31_memory_augmented_eval",
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
    print(f"=== Memory-augmented native pilot — T={T}, n={len(seeds)} ===")
    print("=" * 86)
    print(
        f"{'config':<36s} {'ACC':>10s} {'Task-0':>10s} "
        f"{'Task-N':>10s} {'FGT':>10s} {'memory':>10s}"
    )
    print("-" * 86)
    ref_block = next(
        (s for s in summary_blocks if s["method"] == _REF_METHOD), None,
    )
    bm = ref_block["metric_means"] if ref_block else None
    for s in summary_blocks:
        means = s["metric_means"]
        mem_sizes = [
            d.get("final_memory_size", 0)
            for d in storage_diagnostics if d["method"] == s["method"]
        ]
        avg_mem = (
            statistics.fmean(mem_sizes) if mem_sizes else 0
        )
        mem_str = f"{avg_mem:.0f} avg" if avg_mem > 0 else "N/A"
        tag = "(ref)" if s["method"] == _REF_METHOD else ""
        print(
            f"{(s['method'] + ' ' + tag).strip():<36s} "
            f"{means['average_accuracy']:>10.3f} "
            f"{means['task0_retention']:>10.3f} "
            f"{means['taskN_final']:>10.3f} "
            f"{means['forgetting_proxy']:>+10.3f} "
            f"{mem_str:>10s}"
        )
        if bm is not None and s["method"] != _REF_METHOD:
            d_acc = (means["average_accuracy"] - bm["average_accuracy"]) * 100
            d_t0 = (means["task0_retention"] - bm["task0_retention"]) * 100
            d_tn = (means["taskN_final"] - bm["taskN_final"]) * 100
            print(
                f"{'  (Δ vs reference, pp):':<36s} "
                f"{d_acc:>+10.2f} "
                f"{d_t0:>+10.2f} "
                f"{d_tn:>+10.2f}"
            )

    print()
    print(f"Output JSON: {out_path}")


if __name__ == "__main__":
    main()
