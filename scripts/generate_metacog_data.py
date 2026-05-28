"""Generate PRE and POST metacog training datasets.

Two modes:

- ``--mode full``    (default) generates ``--n-per-class`` examples per
  class, written to ``data/metacog/full_{pre,post}_{train,val}.jsonl``.
  These full datasets are gitignored (per machine).
- ``--mode sample``  generates 25 examples per class — the resulting
  small files (~80 train / ~20 val per dataset) are committed and used
  by the unit tests as fixtures.

PRE dataset uses 3 classes (no ``hallucinated`` — there is no
generation to hallucinate from before the LLM runs); POST uses all 4.

Run from the repo root::

    python scripts/generate_metacog_data.py --mode sample
    python scripts/generate_metacog_data.py --mode full --n-per-class 10000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from agi.metacognition.data_generation import (  # noqa: E402
    POST_CLASSES,
    PRE_CLASSES,
    SyntheticDataGenerator,
    save_dataset,
    split_train_val,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic metacog training data. "
            "PRE dataset = 10 features × 3 classes. "
            "POST dataset = 18 features × 4 classes."
        )
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=10_000,
        help="Examples per class (default: 10000 in full mode, 25 in sample mode)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master RNG seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/metacog"),
        help="Where to write the JSONL files",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "sample"],
        default="full",
        help="`sample` overrides --n-per-class to 25 and uses prefix `sample_`",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction held out for validation (default: 0.2)",
    )
    args = parser.parse_args()

    if args.mode == "sample":
        args.n_per_class = 25
    prefix = "sample_" if args.mode == "sample" else "full_"

    t0 = time.perf_counter()
    gen = SyntheticDataGenerator(seed=args.seed)

    # ----- PRE dataset (3 classes) -----
    pre_examples = gen.generate_batch(args.n_per_class, list(PRE_CLASSES))
    pre_train, pre_val = split_train_val(
        pre_examples, val_ratio=args.val_ratio, seed=args.seed,
    )
    save_dataset(pre_train, args.output_dir / f"{prefix}pre_train.jsonl", mode="pre")
    save_dataset(pre_val, args.output_dir / f"{prefix}pre_val.jsonl", mode="pre")

    # ----- POST dataset (4 classes) -----
    post_examples = gen.generate_batch(args.n_per_class, list(POST_CLASSES))
    post_train, post_val = split_train_val(
        post_examples, val_ratio=args.val_ratio, seed=args.seed,
    )
    save_dataset(post_train, args.output_dir / f"{prefix}post_train.jsonl", mode="post")
    save_dataset(post_val, args.output_dir / f"{prefix}post_val.jsonl", mode="post")

    elapsed = time.perf_counter() - t0
    print(
        f"PRE  ({len(PRE_CLASSES)} classes): "
        f"{len(pre_train)} train / {len(pre_val)} val "
        f"→ {args.output_dir / f'{prefix}pre_*.jsonl'}"
    )
    print(
        f"POST ({len(POST_CLASSES)} classes): "
        f"{len(post_train)} train / {len(post_val)} val "
        f"→ {args.output_dir / f'{prefix}post_*.jsonl'}"
    )
    print(f"Elapsed: {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
