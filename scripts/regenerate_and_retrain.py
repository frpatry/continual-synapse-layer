"""Phase 2h orchestration: regenerate synthetic data → retrain
metacog layers → re-run real-Qwen validation → final report.

Each step shells out to its own CLI so the wall-time + log output
match what you'd get running them by hand.

Run from the repo root::

    python scripts/regenerate_and_retrain.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def _run(cmd: list[str]) -> float:
    """Run ``cmd``, stream output, return wall seconds."""
    print(f"\n$ {' '.join(cmd)}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, check=False)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        print(
            f"\nFAILED (exit {result.returncode}). Aborting Phase 2h "
            "orchestration.",
            file=sys.stderr,
        )
        sys.exit(result.returncode)
    return elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/metacog"),
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=Path("data/metacog/checkpoints"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("results/agi/phase_2h_recalibration_report.md"),
    )
    parser.add_argument(
        "--raw-jsonl", type=Path,
        default=Path("results/agi/phase_2h_validation_raw.jsonl"),
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip the 16-min real-Qwen revalidation step.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t_total = time.perf_counter()

    # Step 1: regenerate synthetic data with recalibrated distributions.
    gen_seconds = _run([
        sys.executable, "scripts/generate_metacog_data.py",
        "--mode", "full",
        "--output-dir", str(args.data_dir),
        "--seed", str(args.seed),
    ])

    # Step 2: retrain both metacog layers.
    train_seconds = _run([
        sys.executable, "scripts/train_metacog.py",
        "--layer", "both",
        "--epochs", str(args.epochs),
        "--data-dir", str(args.data_dir),
        "--checkpoint-dir", str(args.checkpoint_dir),
        "--seed", str(args.seed),
    ])

    val_seconds = 0.0
    if not args.skip_validation:
        # Step 3: re-run the 100-case real-Qwen validation against the
        # new checkpoints.
        val_seconds = _run([
            sys.executable, "-m",
            "experiments.agi.phase_2_validation.run_validation",
            "--checkpoint-dir", str(args.checkpoint_dir),
            "--output", str(args.report),
            "--results-jsonl", str(args.raw_jsonl),
        ])

    total = time.perf_counter() - t_total
    print(
        f"\nPhase 2h orchestration complete in {total:.1f}s "
        f"(generate {gen_seconds:.1f}s, train {train_seconds:.1f}s, "
        f"validate {val_seconds:.1f}s)."
    )
    if not args.skip_validation:
        print(f"Read the verdict in {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
