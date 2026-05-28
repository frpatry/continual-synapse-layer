"""Compute distribution drift between real-Qwen validation dump
and the current synthetic generators, write a markdown report.

Run from the repo root::

    python scripts/analyze_distribution_drift.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from agi.metacognition.distribution_analysis import (  # noqa: E402
    _raw_collect_helpers,
    compute_drift_metrics,
    compute_stats_from_generator,
    compute_stats_from_raw_jsonl,
    generate_drift_report,
    rank_drift_by_severity,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real-jsonl", type=Path,
        default=Path("results/agi/phase_2_validation_raw.jsonl"),
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("results/agi/distribution_drift_report.md"),
    )
    parser.add_argument(
        "--n-per-class", type=int, default=1000,
        help="Synthetic samples per class for the comparison",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    if not args.real_jsonl.exists():
        print(
            f"ERROR: real-Qwen JSONL not found at {args.real_jsonl}. "
            "Run Phase 2d.2 (`python -m experiments.agi.phase_2_validation"
            ".run_validation`) first.",
            file=sys.stderr,
        )
        return 1

    raw_jsonl_fn, raw_gen_fn = _raw_collect_helpers()
    print(f"Reading real-Qwen samples from {args.real_jsonl} ...")
    real_values = raw_jsonl_fn(args.real_jsonl)
    real_stats = compute_stats_from_raw_jsonl(args.real_jsonl)

    print(
        f"Sampling {args.n_per_class}/class from current synthetic "
        "generators ..."
    )
    syn_values = raw_gen_fn(n_per_class=args.n_per_class, seed=args.seed)
    syn_stats = compute_stats_from_generator(
        n_per_class=args.n_per_class, seed=args.seed,
    )

    drift = compute_drift_metrics(
        real_stats, syn_stats,
        real_values=real_values, synthetic_values=syn_values,
    )
    ranked = rank_drift_by_severity(drift)
    generate_drift_report(
        real_stats, syn_stats, drift, ranked, args.report, top_n=args.top_n,
    )

    print("\nTop 10 drifted (status, feature) pairs:")
    print(
        f"{'status':<15} {'feature':<32} {'sev':>6} "
        f"{'real mean':>10} {'syn mean':>10} {'KS':>6}"
    )
    print("-" * 88)
    for status, fname, sev, m in ranked[:10]:
        ks = m.get("ks_stat", float("nan"))
        ks_str = f"{ks:.3f}" if ks == ks else "  —  "
        print(
            f"{status:<15} {fname:<32} {sev:>6.2f} "
            f"{m['real_mean']:>10.3f} {m['syn_mean']:>10.3f} {ks_str:>6}"
        )
    print(f"\nFull report → {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
