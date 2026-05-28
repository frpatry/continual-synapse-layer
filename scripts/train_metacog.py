"""Train the PRE and POST metacognitive layers (Phase 2d).

Reads ``data/metacog/{full|sample}_{pre|post}_{train,val}.jsonl``,
trains one or both layers, writes the best checkpoints to
``data/metacog/checkpoints/{pre,post}_layer.pt`` and a small
``training_metadata.json`` summary alongside.

Run from the repo root::

    # Train both layers on the full dataset (default).
    python scripts/train_metacog.py --layer both --epochs 30

    # Quick smoke on the committed sample fixtures.
    python scripts/train_metacog.py --use-sample --epochs 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from agi.metacognition.training import (  # noqa: E402
    TrainingConfig,
    save_training_metadata,
    train,
)


def _train_one(
    mode: str,
    data_dir: Path,
    checkpoint_dir: Path,
    prefix: str,
    args: argparse.Namespace,
) -> dict:
    """Train one layer; return the JSON-ready metadata blob."""
    print(f"\n=== Training {mode.upper()} layer ===")
    config = TrainingConfig(
        mode=mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
    )
    t0 = time.perf_counter()
    history = train(
        config=config,
        train_path=data_dir / f"{prefix}{mode}_train.jsonl",
        val_path=data_dir / f"{prefix}{mode}_val.jsonl",
        checkpoint_path=checkpoint_dir / f"{mode}_layer.pt",
    )
    elapsed = time.perf_counter() - t0
    best = max(history, key=lambda m: m.val_accuracy)
    final = history[-1]
    print(
        f"Trained {len(history)} epochs in {elapsed:.1f}s "
        f"(best val_acc={best.val_accuracy:.4f} at epoch {best.epoch})"
    )
    print(f"Per-class F1: {best.per_class_f1}")
    print(f"Calibration error: {best.calibration_error:.4f}")
    return {
        "config": {
            "mode": config.mode,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "confidence_loss_weight": config.confidence_loss_weight,
            "early_stopping_patience": config.early_stopping_patience,
            "lr_scheduler_patience": config.lr_scheduler_patience,
            "lr_scheduler_factor": config.lr_scheduler_factor,
            "seed": config.seed,
        },
        "best_epoch": best.epoch,
        "best_val_accuracy": best.val_accuracy,
        "best_per_class_f1": best.per_class_f1,
        "best_calibration_error": best.calibration_error,
        "final_train_loss": final.train_loss,
        "final_val_loss": final.val_loss,
        "epochs_trained": len(history),
        "wall_seconds": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train metacognitive PRE/POST layers on synthetic data. "
            "Checkpoints + metadata land in "
            "data/metacog/checkpoints/."
        )
    )
    parser.add_argument(
        "--layer", choices=["pre", "post", "both"], default="both",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/metacog"),
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=Path("data/metacog/checkpoints"),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-sample", action="store_true",
        help=(
            "Use the committed sample fixtures instead of the full "
            "dataset (fast smoke; accuracy will be poor)."
        ),
    )
    args = parser.parse_args()

    prefix = "sample_" if args.use_sample else "full_"
    layers = ["pre", "post"] if args.layer == "both" else [args.layer]

    all_metadata: dict[str, dict] = {}
    t_total = time.perf_counter()
    for mode in layers:
        all_metadata[mode] = _train_one(
            mode, args.data_dir, args.checkpoint_dir, prefix, args,
        )
    total_elapsed = time.perf_counter() - t_total

    # Phase 2d.1 metadata stamps — record that confidence is now
    # trained via BCE on on-the-fly correctness, not MSE on the
    # synthetic per-example targets. Future loss revisions should
    # bump this version string so old checkpoints can be matched
    # to the loss that produced them.
    all_metadata["_meta"] = {
        "loss_function_version": "phase_2d.1_bce_on_correctness",
        "notes": (
            "Confidence head trained on P(correct) via BCE; "
            "TrainingExample.confidence (synthetic) is ignored "
            "during training."
        ),
    }

    metadata_path = args.checkpoint_dir / "training_metadata.json"
    save_training_metadata(metadata_path, all_metadata)
    print(f"\nMetadata → {metadata_path}")
    print(f"Total wall: {total_elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
