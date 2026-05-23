"""Experiment 04 — SynapseLayer + resistance + reward mixer on Split-MNIST.

Targets the two Phase-3 mechanisms meant to address the Phase-2
failure mode where the synapse layer just tracked the latest task:

- Evidence-based resistance via SynapseLayer's ``resistance_beta``.
- Real reward signal via a RewardMixer composed of any of
  ExternalReward / ConsistencyReward / SurpriseReward.

The script supports four ``--mode`` selections so we can ablate
each component on the same harness:

| mode             | beta | reward source                         |
|------------------|------|---------------------------------------|
| v1               | 0    | fixed 1.0                             |
| resistance       | >0   | fixed 1.0                             |
| reward_only      | 0    | RewardMixer (external + consistency)  |
| resistance_full  | >0   | RewardMixer (ext + consistency + surp)|

Run from the repo root, e.g.::

    python experiments/04_synapse_with_resistance.py --mode resistance_full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from continual_synapse.baselines.naive_finetune import MLPClassifier, MLPConfig  # noqa: E402
from continual_synapse.baselines.synapse_finetune import SynapseAugmentedMLP  # noqa: E402
from continual_synapse.evaluation.benchmarks import SplitMNIST  # noqa: E402
from continual_synapse.evaluation.reporting import (  # noqa: E402
    compute_metrics,
    print_summary,
    save_run,
)
from continual_synapse.evaluation.runner import ContinualRunner, set_seed  # noqa: E402
from continual_synapse.reward.consistency import ConsistencyReward  # noqa: E402
from continual_synapse.reward.external import ExternalReward  # noqa: E402
from continual_synapse.reward.mixer import RewardMixer  # noqa: E402
from continual_synapse.reward.surprise import SurpriseReward  # noqa: E402
from continual_synapse.synapse_layer.layer import SynapseLayer  # noqa: E402
from continual_synapse.synapse_layer.modulation import SynapseModulation  # noqa: E402


_MODES = ("v1", "resistance", "reward_only", "resistance_full")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mode", choices=_MODES, default="resistance_full")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs-per-task", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-hidden-layers", type=int, default=3)
    p.add_argument("--synapse-lr", type=float, default=1e-3)
    p.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Evidence-based resistance coefficient (active in "
        "modes 'resistance' and 'resistance_full').",
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=1e-3,
        help="Developmental-trajectory decay (active when a "
        "RewardMixer is used).",
    )
    p.add_argument("--consistency-decay", type=float, default=0.99)
    p.add_argument("--w-consistency", type=float, default=1.0)
    p.add_argument("--w-surprise", type=float, default=0.5)
    p.add_argument("--init-gate", type=float, default=0.0)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda", "mps"],
    )
    p.add_argument(
        "--cache-dir", default=str(_REPO_ROOT / "data" / "hf_cache")
    )
    p.add_argument(
        "--output-dir", default=str(_REPO_ROOT / "results" / "logs")
    )
    return p.parse_args()


def _build_reward_computer(mode: str, args: argparse.Namespace, dim: int):
    """Return ``(reward_computer, mixer_or_none)`` for the chosen mode."""
    if mode in ("v1", "resistance"):
        return None, None
    consistency = ConsistencyReward(n_neurons=dim, decay=args.consistency_decay)
    if mode == "reward_only":
        mixer = RewardMixer(
            external=ExternalReward(default=1.0),
            consistency=consistency,
            gamma=args.gamma,
            w_consistency=args.w_consistency,
        )
    elif mode == "resistance_full":
        surprise = SurpriseReward(n_neurons=dim)
        mixer = RewardMixer(
            external=ExternalReward(default=1.0),
            consistency=consistency,
            surprise=surprise,
            gamma=args.gamma,
            w_consistency=args.w_consistency,
            w_surprise=args.w_surprise,
        )
    else:
        raise ValueError(f"Unknown mode {mode!r}")
    return mixer, mixer


def _effective_beta(mode: str, beta: float) -> float:
    """Resistance is enabled only in modes that include 'resistance'."""
    if mode in ("resistance", "resistance_full"):
        return beta
    return 0.0


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    bench = SplitMNIST.from_huggingface(cache_dir=args.cache_dir)
    base = MLPClassifier(
        MLPConfig(
            input_dim=784,
            hidden_dim=args.hidden_dim,
            num_classes=bench.num_classes_per_task,
            num_hidden_layers=args.num_hidden_layers,
        )
    )
    synapse = SynapseLayer(
        n_neurons=args.hidden_dim,
        learning_rate=args.synapse_lr,
        resistance_beta=_effective_beta(args.mode, args.beta),
    )
    modulator = SynapseModulation(init_gate=args.init_gate)

    reward_computer, mixer = _build_reward_computer(
        args.mode, args, dim=args.hidden_dim
    )
    model = SynapseAugmentedMLP(
        base, synapse, modulator, reward_computer=reward_computer
    )

    reward_log: list[float] = []

    def after_batch(i, task, m, x, y):
        applied = m.apply_hebbian_update()
        reward_log.append(applied)

    runner = ContinualRunner(
        optimizer_factory=lambda params: torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum
        ),
        epochs_per_task=args.epochs_per_task,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        on_after_batch=after_batch,
    )
    result = runner.run(model, bench)
    summary = compute_metrics(result)
    print_summary(result, summary, method=f"synapse_{args.mode}")

    print(
        f"\nMode: {args.mode}"
        f"\nEffective beta: {_effective_beta(args.mode, args.beta)}"
        f"\nFinal modulator gate: {modulator.gate.item():+.4f}"
        f"\nFinal synapse strength range: "
        f"[{synapse.strengths.min().item():+.4f}, "
        f"{synapse.strengths.max().item():+.4f}]"
        f"\nFinal evidence range: "
        f"[{synapse.evidence.min().item():+.4f}, "
        f"{synapse.evidence.max().item():+.4f}]"
        f"\nHebbian updates applied: {int(synapse.global_step.item())}"
    )
    if reward_log:
        avg_first = sum(reward_log[: len(reward_log) // 5]) / max(
            1, len(reward_log) // 5
        )
        avg_last = sum(reward_log[-len(reward_log) // 5 :]) / max(
            1, len(reward_log) // 5
        )
        print(
            f"\nReward trajectory:"
            f"\n  first 20% avg: {avg_first:+.4f}"
            f"\n  last 20% avg:  {avg_last:+.4f}"
        )
    if mixer is not None:
        print(
            f"\nMixer α(t) at end: {mixer.alpha:.4f}"
            f"\nMixer steps:        {mixer.step}"
        )

    path = save_run(
        result,
        experiment="04_synapse_with_resistance",
        method=f"synapse_{args.mode}",
        config=vars(args),
        output_dir=args.output_dir,
        summary=summary,
    )
    print(f"\nSaved run log to {path}")


if __name__ == "__main__":
    main()
