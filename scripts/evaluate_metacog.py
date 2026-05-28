"""Evaluate a trained metacognitive checkpoint on any JSONL dataset.

Run from the repo root::

    python scripts/evaluate_metacog.py \\
        --layer pre \\
        --checkpoint data/metacog/checkpoints/pre_layer.pt \\
        --data data/metacog/full_pre_val.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from torch.utils.data import DataLoader  # noqa: E402

from agi.metacognition.data_generation import load_dataset  # noqa: E402
from agi.metacognition.training import (  # noqa: E402
    MetacogDataset,
    TrainingConfig,
    evaluate,
    load_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained metacog layer checkpoint on a "
            "held-out JSONL dataset."
        )
    )
    parser.add_argument("--layer", choices=["pre", "post"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    config = TrainingConfig(mode=args.layer, batch_size=args.batch_size)
    model = load_checkpoint(args.checkpoint, mode=args.layer)
    examples = load_dataset(args.data)
    dataset = MetacogDataset(examples, mode=args.layer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    metrics = evaluate(model, loader, config)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
