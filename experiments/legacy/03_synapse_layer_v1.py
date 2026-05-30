"""Experiment 03 — SynapseLayer v1 on Split-MNIST.

Augments the Phase-1 MLP baseline with a dense SynapseLayer and a
gated linear modulator (see DESIGN.md §3.2). The Hebbian update is
applied after every optimizer step via the runner's
``on_after_batch`` hook. The gate starts at zero so the model
behaves exactly like the naive baseline on the first forward pass
and only deviates once the synapse buffer has accumulated state
and the gate has had a chance to learn a useful scale.

Run from the repo root:

    python experiments/03_synapse_layer_v1.py --synapse-lr 1e-3

A JSON log is written to ``results/logs/``.
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
from continual_synapse.synapse_layer.layer import SynapseLayer  # noqa: E402
from continual_synapse.synapse_layer.modulation import SynapseModulation  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs-per-task", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-hidden-layers", type=int, default=3)
    p.add_argument(
        "--synapse-lr",
        type=float,
        default=1e-3,
        help="Hebbian learning rate (eta in DESIGN.md eq. 3.2).",
    )
    p.add_argument(
        "--init-gate",
        type=float,
        default=0.0,
        help="Initial value of the modulator gate. 0 preserves the base "
        "model exactly at init.",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda", "mps"],
    )
    p.add_argument(
        "--cache-dir",
        default=str(_REPO_ROOT / "data" / "hf_cache"),
        help="HuggingFace datasets cache directory.",
    )
    p.add_argument(
        "--output-dir",
        default=str(_REPO_ROOT / "results" / "logs"),
        help="Where to write the JSON run log.",
    )
    return p.parse_args()


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
        n_neurons=args.hidden_dim, learning_rate=args.synapse_lr
    )
    modulator = SynapseModulation(init_gate=args.init_gate)
    model = SynapseAugmentedMLP(base, synapse, modulator)

    runner = ContinualRunner(
        optimizer_factory=lambda params: torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum
        ),
        epochs_per_task=args.epochs_per_task,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        on_after_batch=lambda i, task, m, x, y: m.apply_hebbian_update(),
    )
    result = runner.run(model, bench)
    summary = compute_metrics(result)
    print_summary(result, summary, method="synapse_v1")

    print(
        f"\nFinal modulator gate: {modulator.gate.item():+.4f}"
        f"\nFinal synapse strength range: "
        f"[{synapse.strengths.min().item():+.4f}, "
        f"{synapse.strengths.max().item():+.4f}]"
        f"\nHebbian updates applied: {int(synapse.global_step.item())}"
    )

    path = save_run(
        result,
        experiment="03_synapse_layer_v1",
        method="synapse_v1",
        config=vars(args),
        output_dir=args.output_dir,
        summary=summary,
    )
    print(f"\nSaved run log to {path}")


if __name__ == "__main__":
    main()
