"""Experiment 40 — Phase 5.5.5: Split-MNIST CI comparison + verdict.

Aggregates the JSON outputs of exps 36 (naive), 37 (EWC), 38 (DER),
and 39 (CLS Variant C) into a single comparison matrix, runs
pairwise Wilcoxon rank-sum tests between CLS and each baseline,
and auto-classifies the cross-paradigm verdict per the
decision rules from the Phase 5.5 spec:

- CLS > naive, EWC by 10+ pp  → architecture generalizes
- CLS within 5pp of DER       → matches modern memory-based methods
- CLS > DER                   → cross-paradigm win, MAJOR result
- CLS < EWC                   → fundamental issue, debug needed

Run from the repo root::

    python experiments/40_split_mnist_ci_comparison.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _latest_json(
    log_dir: Path, pattern: str,
) -> Path:
    """Return the most-recent JSON in ``log_dir`` matching the
    method's filename pattern. The Phase 5.5 experiments name
    their outputs ``<timestamp>_<expnum>_split_mnist_ci_<method>.json``,
    so we sort by mtime and pick the newest."""
    candidates = sorted(log_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No JSON found in {log_dir} matching {pattern}. "
            f"Re-run the relevant sub-phase first."
        )
    return candidates[-1]


def _load_method(
    log_dir: Path, pattern: str,
) -> dict[str, Any]:
    path = _latest_json(log_dir, pattern)
    with path.open() as f:
        data = json.load(f)
    return {"path": str(path), "data": data}


def _seed_accs(data: dict[str, Any], key: str) -> list[float]:
    return [s[key] for s in data["per_seed"]]


def _seed_accs_from_per_seed_best(
    data: dict[str, Any], key: str,
) -> list[float]:
    """EWC stores per_seed_best instead of per_seed."""
    return [s[key] for s in data["per_seed_best"]]


def _print_row(
    name: str, summary_acc_mean: float, summary_acc_std: float,
    per_class_means: list[float], fgt: float,
) -> None:
    per_class_str = "[" + ", ".join(
        f"{c}:{per_class_means[c]:.2f}" for c in range(len(per_class_means))
    ) + "]"
    print(
        f"{name:<11} | {summary_acc_mean:.3f} ± {summary_acc_std:.3f}     "
        f"| {per_class_str:<58} | {fgt:+.3f}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--log-dir", type=Path,
        default=_REPO_ROOT / "results" / "logs" / "split_mnist_ci",
    )
    args = p.parse_args()

    # ----- load each method's most-recent JSON -----
    naive = _load_method(args.log_dir, "*36_split_mnist_ci_naive.json")
    ewc   = _load_method(args.log_dir, "*37_split_mnist_ci_ewc.json")
    der   = _load_method(args.log_dir, "*38_split_mnist_ci_der.json")
    cls   = _load_method(args.log_dir, "*39_split_mnist_ci_cls.json")

    print("Loaded sources:")
    print(f"  naive: {naive['path']}")
    print(f"  EWC:   {ewc['path']}")
    print(f"  DER:   {der['path']}")
    print(f"  CLS:   {cls['path']}")

    # ----- per-method seed values -----
    naive_seeds = _seed_accs(naive["data"], "final_acc")
    ewc_seeds   = _seed_accs_from_per_seed_best(ewc["data"], "final_acc")
    der_seeds   = _seed_accs(der["data"], "final_acc")
    # CLS reports neo_final_acc (the dual-system's headline number);
    # the hippocampe accuracy is also captured in the summary for
    # transparency.
    cls_seeds   = _seed_accs(cls["data"], "neo_final_acc")

    cls_hipp_seeds = _seed_accs(cls["data"], "hipp_final_acc")

    # ----- aggregate header -----
    print()
    print("=== Split-MNIST class-incremental comparison (T=5, n=3 each) ===")
    print()
    hdr = (
        f"{'Method':<11} | {'ACC (mean ± std)':<19} "
        f"| {'Per-class final accuracy':<58} | FGT (acc[0]-acc[-1])"
    )
    print(hdr)
    print("-" * len(hdr))

    def _mean_std(xs: list[float]) -> tuple[float, float]:
        return (
            statistics.fmean(xs),
            statistics.stdev(xs) if len(xs) > 1 else 0.0,
        )

    naive_m, naive_s = _mean_std(naive_seeds)
    ewc_m,   ewc_s   = _mean_std(ewc_seeds)
    der_m,   der_s   = _mean_std(der_seeds)
    cls_m,   cls_s   = _mean_std(cls_seeds)

    _print_row(
        "naive", naive_m, naive_s,
        naive["data"]["summary"]["per_class_means"],
        naive["data"]["summary"]["fgt_mean"],
    )
    _print_row(
        f"EWC λ={ewc['data']['best_lambda']}", ewc_m, ewc_s,
        ewc["data"]["summary"]["per_class_means"],
        ewc["data"]["summary"]["fgt_mean"],
    )
    _print_row(
        "DER", der_m, der_s,
        der["data"]["summary"]["per_class_means"],
        der["data"]["summary"]["fgt_mean"],
    )
    _print_row(
        "CLS Var C", cls_m, cls_s,
        cls["data"]["summary"]["per_class_means_neo"],
        cls["data"]["summary"]["fgt_mean"],
    )

    # CLS also reports the hippocampe alone for transparency.
    cls_hipp_m, cls_hipp_s = _mean_std(cls_hipp_seeds)
    print(
        f"{'  └ HIPP':<11} | {cls_hipp_m:.3f} ± {cls_hipp_s:.3f}     "
        f"| (hippocampe-only readout — not the headline number; "
        f"per-class in JSON)"
    )

    # ----- pairwise Wilcoxon rank-sum -----
    try:
        from scipy.stats import ranksums  # type: ignore[import-untyped]
        scipy_ok = True
    except Exception:
        scipy_ok = False
        ranksums = None  # type: ignore[assignment]

    print()
    print("=== Pairwise Wilcoxon (CLS vs each baseline) ===")
    if scipy_ok:
        for label, ref_seeds in (
            ("naive", naive_seeds),
            (f"EWC λ={ewc['data']['best_lambda']}", ewc_seeds),
            ("DER",   der_seeds),
        ):
            r = ranksums(cls_seeds, ref_seeds)
            print(
                f"  CLS vs {label:<14}: statistic={r.statistic:.3f}  "
                f"p={r.pvalue:.4f}"
            )
    else:
        print("  scipy not available — skipping Wilcoxon.")

    # ----- verdict -----
    delta_naive = cls_m - naive_m
    delta_ewc   = cls_m - ewc_m
    delta_der   = cls_m - der_m
    print()
    print("=== Verdict ===")
    print(
        f"  CLS - naive = {delta_naive:+.3f}  "
        f"CLS - EWC = {delta_ewc:+.3f}  "
        f"CLS - DER = {delta_der:+.3f}"
    )
    print()
    verdict_lines: list[str] = []
    if delta_naive >= 0.10 and delta_ewc >= 0.10:
        verdict_lines.append(
            "✓ CLS beats naive AND EWC by ≥10pp — architecture "
            "GENERALIZES to class-incremental."
        )
    elif delta_naive < 0.10 or delta_ewc < 0.10:
        verdict_lines.append(
            "✗ CLS does NOT beat naive/EWC by ≥10pp — architecture "
            "may not be doing meaningful work in this paradigm."
        )

    if delta_der >= 0.0:
        verdict_lines.append(
            "✓✓ CLS ≥ DER — CROSS-PARADIGM WIN. Major result."
        )
    elif abs(delta_der) <= 0.05:
        verdict_lines.append(
            "✓ CLS within 5pp of DER — matches modern memory-based "
            "methods cross-paradigm."
        )
    else:
        verdict_lines.append(
            f"~ CLS trails DER by {-delta_der:.3f} (>5pp) — investigate."
        )

    if cls_m < ewc_m:
        verdict_lines.append(
            "✗✗ CLS < EWC — fundamental issue with class-incremental "
            "adaptation, debug needed."
        )

    for line in verdict_lines:
        print(f"  {line}")

    # ----- persist a small aggregate JSON for the writeup -----
    out_dir = args.log_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    import time
    ts = int(time.time())
    aggregate_path = out_dir / f"{ts}_40_split_mnist_ci_comparison.json"
    aggregate = {
        "experiment": "40_split_mnist_ci_comparison",
        "phase": "5.5.5",
        "timestamp": ts,
        "sources": {
            "naive": naive["path"],
            "ewc":   ewc["path"],
            "der":   der["path"],
            "cls":   cls["path"],
        },
        "results": {
            "naive": {"seeds": naive_seeds, "mean": naive_m, "std": naive_s},
            "ewc":   {
                "seeds": ewc_seeds, "mean": ewc_m, "std": ewc_s,
                "best_lambda": ewc["data"]["best_lambda"],
            },
            "der":   {"seeds": der_seeds, "mean": der_m, "std": der_s},
            "cls_variant_c": {
                "seeds_neo": cls_seeds,
                "mean_neo":  cls_m, "std_neo": cls_s,
                "seeds_hipp": cls_hipp_seeds,
                "mean_hipp": cls_hipp_m, "std_hipp": cls_hipp_s,
            },
        },
        "deltas": {
            "cls_minus_naive": delta_naive,
            "cls_minus_ewc":   delta_ewc,
            "cls_minus_der":   delta_der,
        },
        "verdict_lines": verdict_lines,
    }
    if scipy_ok:
        aggregate["wilcoxon_vs_baselines"] = {
            "cls_vs_naive": {
                "statistic": float(ranksums(cls_seeds, naive_seeds).statistic),
                "p":         float(ranksums(cls_seeds, naive_seeds).pvalue),
            },
            "cls_vs_ewc": {
                "statistic": float(ranksums(cls_seeds, ewc_seeds).statistic),
                "p":         float(ranksums(cls_seeds, ewc_seeds).pvalue),
            },
            "cls_vs_der": {
                "statistic": float(ranksums(cls_seeds, der_seeds).statistic),
                "p":         float(ranksums(cls_seeds, der_seeds).pvalue),
            },
        }
    with aggregate_path.open("w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nWrote comparison JSON to {aggregate_path}")


if __name__ == "__main__":
    main()
