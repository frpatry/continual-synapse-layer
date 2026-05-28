"""One-shot helper: compare two real-Qwen validation raw JSONLs
(before / after a recalibration) and write a comparison report.

Used by Phase 2h to write the final
``results/agi/phase_2h_recalibration_report.md`` with explicit
before/after metrics rather than just the after-only template
that ``run_validation`` produces.

Run from the repo root::

    python scripts/compare_validation_runs.py \\
        --before results/agi/phase_2_validation_raw.jsonl \\
        --after  results/agi/phase_2h_validation_raw.jsonl \\
        --output results/agi/phase_2h_recalibration_report.md \\
        --label-before "Phase 2d.2 (before recalibration)" \\
        --label-after  "Phase 2h (after recalibration)"
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from agi.metacognition.distribution_analysis import _raw_collect_helpers  # noqa: E402

# Reuse the analysis helpers we already have.
sys.path.insert(0, str(_REPO_ROOT))
from experiments.agi.phase_2_validation.analysis import (  # noqa: E402
    _STATUSES,
    compute_confusion_matrix,
    compute_per_class_metrics,
    compute_real_calibration,
    identify_error_patterns,
    overall_accuracy,
)


def _load(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _confusion_md(label: str, m: dict[str, Counter]) -> list[str]:
    lines = [f"### {label}", ""]
    header = "| expected ↓ / predicted → | " + " | ".join(_STATUSES) + " |"
    sep = "|---|" + "|".join(["---:"] * len(_STATUSES)) + "|"
    lines += [header, sep]
    for s in _STATUSES:
        row = [s]
        for p in _STATUSES:
            row.append(str(m.get(s, Counter()).get(p, 0)))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def _per_class_md(label: str, m: dict[str, dict[str, float]]) -> list[str]:
    lines = [f"### {label}", ""]
    lines.append("| class | precision | recall | F1 | TP | FP | FN |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for cls, met in m.items():
        lines.append(
            f"| {cls} | {met['precision']:.3f} | {met['recall']:.3f} "
            f"| {met['f1']:.3f} | {met['tp']} | {met['fp']} | {met['fn']} |"
        )
    lines.append("")
    return lines


def _decide_verdict(post_acc: float) -> tuple[str, str]:
    if post_acc >= 0.75:
        return "GREEN", "Phase 2e (LoRA training) can begin."
    if post_acc >= 0.60:
        return (
            "YELLOW",
            "Generalises but with non-trivial residual errors. "
            "Consider further recalibration or expanding the test "
            "set before committing GPU compute.",
        )
    return (
        "RED",
        "Structural problem — synthetic distributions still don't "
        "transfer. Re-examine feature engineering before Phase 2e.",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--label-before", type=str,
        default="Before",
    )
    parser.add_argument(
        "--label-after", type=str,
        default="After",
    )
    args = parser.parse_args()

    before = _load(args.before)
    after = _load(args.after)

    # Headline accuracy + ECE
    pre_acc_b, pre_n_b = overall_accuracy(before, "pre", ("hallucinated",))
    pre_acc_a, pre_n_a = overall_accuracy(after, "pre", ("hallucinated",))
    post_acc_b, post_n_b = overall_accuracy(before, "post")
    post_acc_a, post_n_a = overall_accuracy(after, "post")
    pre_ece_b = compute_real_calibration(before, "pre")
    pre_ece_a = compute_real_calibration(after, "pre")
    post_ece_b = compute_real_calibration(before, "post")
    post_ece_a = compute_real_calibration(after, "post")

    pre_cm_b = compute_confusion_matrix(before, "pre")
    pre_cm_a = compute_confusion_matrix(after, "pre")
    post_cm_b = compute_confusion_matrix(before, "post")
    post_cm_a = compute_confusion_matrix(after, "post")

    pre_metr_b = compute_per_class_metrics(before, "pre")
    pre_metr_a = compute_per_class_metrics(after, "pre")
    post_metr_b = compute_per_class_metrics(before, "post")
    post_metr_a = compute_per_class_metrics(after, "post")

    pre_err_a = identify_error_patterns(after, "pre")
    post_err_a = identify_error_patterns(after, "post")

    verdict, rationale = _decide_verdict(post_acc_a)

    lines: list[str] = []
    lines.append("# Phase 2h — Distribution Recalibration Report")
    lines.append("")
    lines.append(
        f"Compares the metacog's real-Qwen validation performance "
        f"**{args.label_before}** vs **{args.label_after}** on the "
        f"same 100 hand-crafted test cases."
    )
    lines.append("")
    lines.append(
        "**Recalibration source:** drift analysis at "
        "`results/agi/distribution_drift_report.md` "
        "(computed by `scripts/analyze_distribution_drift.py`). The "
        "Phase 2c v1 synthetic distributions were updated to match "
        "empirical Qwen2.5-1.5B distributions across "
        "`alignment_novel_token_ratio`, `attention_to_facts_mean`, "
        "`alignment_max_cosine`, `response_length_tokens`, and the "
        "unknown-with-facts alignment branch."
    )
    lines.append("")

    lines.append("## Headline — before vs after")
    lines.append("")
    lines.append("| metric | before | after | Δ |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| PRE accuracy (excl. hallucinated) | "
        f"{pre_acc_b:.3f} ({pre_n_b}n) | {pre_acc_a:.3f} ({pre_n_a}n) "
        f"| {pre_acc_a - pre_acc_b:+.3f} |"
    )
    lines.append(
        f"| POST accuracy (all classes) | "
        f"{post_acc_b:.3f} ({post_n_b}n) | {post_acc_a:.3f} ({post_n_a}n) "
        f"| {post_acc_a - post_acc_b:+.3f} |"
    )
    lines.append(
        f"| PRE real-data ECE | {pre_ece_b:.3f} | {pre_ece_a:.3f} "
        f"| {pre_ece_a - pre_ece_b:+.3f} |"
    )
    lines.append(
        f"| POST real-data ECE | {post_ece_b:.3f} | {post_ece_a:.3f} "
        f"| {post_ece_a - post_ece_b:+.3f} |"
    )
    lines.append("")
    lines.append(f"### Verdict: **{verdict}**")
    lines.append("")
    lines.append(rationale)
    lines.append("")

    lines.append("## Per-class POST F1 — before vs after")
    lines.append("")
    lines.append("| class | F1 before | F1 after | Δ |")
    lines.append("|---|---:|---:|---:|")
    for cls in _STATUSES:
        f1_b = post_metr_b[cls]["f1"]
        f1_a = post_metr_a[cls]["f1"]
        lines.append(
            f"| {cls} | {f1_b:.3f} | {f1_a:.3f} | {f1_a - f1_b:+.3f} |"
        )
    lines.append("")

    lines.append("## POST confusion matrices")
    lines.append("")
    lines += _confusion_md(args.label_before, post_cm_b)
    lines += _confusion_md(args.label_after, post_cm_a)

    lines.append("## PRE confusion matrices")
    lines.append("")
    lines += _confusion_md(args.label_before, pre_cm_b)
    lines += _confusion_md(args.label_after, pre_cm_a)

    lines.append("## POST per-class metrics — full")
    lines.append("")
    lines += _per_class_md(args.label_before, post_metr_b)
    lines += _per_class_md(args.label_after, post_metr_a)

    lines.append("## PRE per-class metrics — full")
    lines.append("")
    lines += _per_class_md(args.label_before, pre_metr_b)
    lines += _per_class_md(args.label_after, pre_metr_a)

    lines.append("## Top POST error patterns after recalibration")
    lines.append("")
    if not post_err_a:
        lines.append("_(none)_")
    else:
        for key in sorted(post_err_a.keys(), key=lambda k: -len(post_err_a[k])):
            exp, pred = key
            lines.append(
                f"- **{exp} → {pred}** ({len(post_err_a[key])} cases): "
                f"{', '.join(post_err_a[key][:8])}"
            )
    lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"Comparison report → {args.output}")
    print()
    print(f"Verdict: {verdict}")
    print(
        f"  POST acc: {post_acc_b:.3f} → {post_acc_a:.3f} "
        f"(Δ {post_acc_a - post_acc_b:+.3f})"
    )
    print(
        f"  PRE  acc: {pre_acc_b:.3f} → {pre_acc_a:.3f} "
        f"(Δ {pre_acc_a - pre_acc_b:+.3f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
