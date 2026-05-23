"""Experiment 01 — naive sequential fine-tuning on Split-MNIST.

Reference baseline for the project. The MLP is trained on each of
the five binary digit-pair tasks in turn with no continual-learning
protection; we expect to see clear catastrophic forgetting on the
earlier tasks.

Run from the repo root:

    python experiments/01_baseline_forgetting.py

A JSON log is written to ``results/logs/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Make the package importable without `pip install -e .`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from continual_synapse.baselines.naive_finetune import MLPClassifier, MLPConfig  # noqa: E402
from continual_synapse.evaluation.benchmarks import SplitMNIST  # noqa: E402
from continual_synapse.evaluation.reporting import (  # noqa: E402
    compute_metrics,
    print_summary,
    save_run,
)
from continual_synapse.evaluation.runner import ContinualRunner, set_seed  # noqa: E402


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
    model = MLPClassifier(
        MLPConfig(
            input_dim=784,
            hidden_dim=args.hidden_dim,
            num_classes=bench.num_classes_per_task,
            num_hidden_layers=args.num_hidden_layers,
        )
    )
    runner = ContinualRunner(
        optimizer_factory=lambda params: torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum
        ),
        epochs_per_task=args.epochs_per_task,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )
    result = runner.run(model, bench)
    summary = compute_metrics(result)
    print_summary(result, summary, method="naive_finetune")

    path = save_run(
        result,
        experiment="01_baseline_forgetting",
        method="naive_finetune",
        config=vars(args),
        output_dir=args.output_dir,
        summary=summary,
    )
    print(f"\nSaved run log to {path}")


if __name__ == "__main__":
    main()
