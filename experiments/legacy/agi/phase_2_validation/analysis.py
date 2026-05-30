"""Analysis + markdown reporting for Phase 2d.2 validation results."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


_STATUSES: tuple[str, ...] = (
    "known", "unknown", "uncertain", "hallucinated",
)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def compute_confusion_matrix(
    results: list[dict], layer: str = "post",
) -> dict[str, Counter]:
    """expected_status → Counter(predicted_status)."""
    matrix: dict[str, Counter] = {s: Counter() for s in _STATUSES}
    for r in results:
        if "error" in r:
            continue
        expected = r["expected_status"]
        predicted = r[f"{layer}_predicted_status"]
        matrix.setdefault(expected, Counter())[predicted] += 1
    return matrix


def compute_per_class_metrics(
    results: list[dict], layer: str = "post",
) -> dict[str, dict[str, float]]:
    """precision / recall / F1 per epistemic class."""
    valid = [r for r in results if "error" not in r]
    out: dict[str, dict[str, float]] = {}
    for s in _STATUSES:
        tp = sum(
            1 for r in valid
            if r["expected_status"] == s and r[f"{layer}_predicted_status"] == s
        )
        fp = sum(
            1 for r in valid
            if r["expected_status"] != s and r[f"{layer}_predicted_status"] == s
        )
        fn = sum(
            1 for r in valid
            if r["expected_status"] == s and r[f"{layer}_predicted_status"] != s
        )
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        out[s] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
        }
    return out


def compute_real_calibration(
    results: list[dict], layer: str = "post", n_bins: int = 10,
) -> float:
    """ECE between the layer's confidence and the *actual*
    correctness on the case (expected vs predicted status)."""
    valid = [r for r in results if "error" not in r]
    if not valid:
        return 0.0
    confs = np.array([r[f"{layer}_confidence"] for r in valid], dtype=float)
    correct = np.array(
        [
            r["expected_status"] == r[f"{layer}_predicted_status"]
            for r in valid
        ],
        dtype=float,
    )
    if confs.size == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n_total = float(confs.size)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confs >= lo) & (confs <= hi)
        else:
            mask = (confs >= lo) & (confs < hi)
        if not mask.any():
            continue
        bin_size = float(mask.sum())
        bin_acc = float(correct[mask].mean())
        bin_conf = float(confs[mask].mean())
        ece += (bin_size / n_total) * abs(bin_acc - bin_conf)
    return float(ece)


def overall_accuracy(
    results: list[dict],
    layer: str = "post",
    exclude_statuses: Iterable[str] = (),
) -> tuple[float, int]:
    """Returns (accuracy, n_used)."""
    exclude = set(exclude_statuses)
    valid = [
        r for r in results
        if "error" not in r and r["expected_status"] not in exclude
    ]
    if not valid:
        return 0.0, 0
    n_correct = sum(
        1 for r in valid
        if r["expected_status"] == r[f"{layer}_predicted_status"]
    )
    return n_correct / len(valid), len(valid)


def identify_error_patterns(
    results: list[dict], layer: str = "post",
) -> dict[tuple[str, str], list[str]]:
    """``(expected, predicted) → [case_ids]`` for misclassifications."""
    patterns: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in results:
        if "error" in r:
            continue
        exp = r["expected_status"]
        pred = r[f"{layer}_predicted_status"]
        if exp != pred:
            patterns[(exp, pred)].append(r["case_id"])
    return dict(patterns)


# ----------------------------------------------------------------------
# Markdown report
# ----------------------------------------------------------------------

def _format_confusion_matrix_md(matrix: dict[str, Counter]) -> str:
    header = "| expected ↓ / predicted → | " + " | ".join(_STATUSES) + " |"
    sep = "|---|" + "|".join(["---:"] * len(_STATUSES)) + "|"
    lines = [header, sep]
    for s in _STATUSES:
        row = [s]
        for p in _STATUSES:
            row.append(str(matrix.get(s, Counter()).get(p, 0)))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _format_per_class_table_md(metrics: dict[str, dict]) -> str:
    header = "| class | precision | recall | F1 | TP | FP | FN |"
    sep = "|---|---:|---:|---:|---:|---:|---:|"
    rows = [header, sep]
    for cls, m in metrics.items():
        rows.append(
            f"| {cls} | {m['precision']:.3f} | {m['recall']:.3f} "
            f"| {m['f1']:.3f} | {m['tp']} | {m['fp']} | {m['fn']} |"
        )
    return "\n".join(rows)


def _format_error_patterns_md(
    patterns: dict[tuple[str, str], list[str]], by_case: dict[str, dict],
    max_examples: int = 3,
) -> str:
    if not patterns:
        return "_(none)_"
    lines: list[str] = []
    sorted_keys = sorted(patterns.keys(), key=lambda k: -len(patterns[k]))
    for key in sorted_keys:
        exp, pred = key
        ids = patterns[key]
        lines.append(
            f"- **{exp} → {pred}** ({len(ids)} case(s)): "
            f"{', '.join(ids[:8])}{' ...' if len(ids) > 8 else ''}"
        )
        for case_id in ids[:max_examples]:
            r = by_case.get(case_id)
            if r is None:
                continue
            response = (r["response"] or "").strip().replace("\n", " ")
            if len(response) > 140:
                response = response[:137] + "..."
            lines.append(
                f"    - `{case_id}` query=*{r['query']}* "
                f"→ response: *{response}*"
            )
    return "\n".join(lines)


def _decide_phase_2e_readiness(
    pre_acc: float, post_acc: float,
) -> tuple[str, str]:
    """Return ``(verdict, rationale)``."""
    if post_acc >= 0.75 and pre_acc >= 0.75:
        return (
            "READY",
            "Both layers clear the 0.75 real-data accuracy bar; "
            "synthetic-trained metacog generalises to real Qwen outputs. "
            "Phase 2e can begin.",
        )
    if post_acc >= 0.60:
        return (
            "CAUTION",
            f"POST acc={post_acc:.3f} between 0.60 and 0.75 — generalises "
            "but with non-trivial error patterns. Inspect the confusion "
            "matrix before committing GPU compute to Phase 2e.",
        )
    return (
        "RECALIBRATION_NEEDED",
        f"POST acc={post_acc:.3f} below 0.60. Synthetic distributions "
        "don't match real Qwen feature distributions. Recalibrate "
        "Phase 2c generators or collect real-LLM training data before "
        "Phase 2e.",
    )


def generate_report(
    results: list[dict],
    output_path: Path,
    *,
    wall_seconds: float | None = None,
) -> None:
    """Write the full markdown report to ``output_path``."""
    by_case = {r["case_id"]: r for r in results if "error" not in r}
    n_total = len(results)
    n_errors = sum(1 for r in results if "error" in r)
    n_valid = n_total - n_errors

    pre_conf = compute_confusion_matrix(results, layer="pre")
    post_conf = compute_confusion_matrix(results, layer="post")
    pre_metrics = compute_per_class_metrics(results, layer="pre")
    post_metrics = compute_per_class_metrics(results, layer="post")

    # PRE accuracy excludes the hallucinated cohort (PRE has no
    # generation to evaluate; it can't predict hallucinated).
    pre_acc, pre_n = overall_accuracy(
        results, layer="pre", exclude_statuses=("hallucinated",),
    )
    post_acc, post_n = overall_accuracy(results, layer="post")
    pre_ece = compute_real_calibration(results, layer="pre")
    post_ece = compute_real_calibration(results, layer="post")

    pre_errors = identify_error_patterns(results, layer="pre")
    post_errors = identify_error_patterns(results, layer="post")

    verdict, rationale = _decide_phase_2e_readiness(pre_acc, post_acc)

    md: list[str] = []
    md.append("# Phase 2d.2 — Real-Qwen Validation Report")
    md.append("")
    md.append(
        "**Pipeline.** For each of 100 hand-crafted cases (25 per "
        "epistemic status), the validation script seeded a fresh "
        "X-Ray memory, ran the query through Qwen2.5-1.5B-Instruct "
        "via `generate_with_signals`, extracted the 18 metacog "
        "features, and asked the trained PRE + POST layers to "
        "classify."
    )
    md.append("")
    md.append("## Headline")
    md.append("")
    md.append(f"- Cases run: **{n_valid}** valid / {n_total} total ({n_errors} errored).")
    if wall_seconds is not None:
        md.append(f"- Wall time: **{wall_seconds:.1f} s** ({wall_seconds / max(n_valid, 1):.2f} s/case).")
    md.append(
        f"- **PRE  accuracy** (excludes hallucinated cohort): "
        f"**{pre_acc:.3f}** over {pre_n} cases."
    )
    md.append(
        f"- **POST accuracy** (all classes): **{post_acc:.3f}** "
        f"over {post_n} cases."
    )
    md.append(
        f"- **PRE  real-data ECE**: {pre_ece:.3f}  "
        f"·  **POST real-data ECE**: {post_ece:.3f}"
    )
    md.append("")
    md.append(f"### Verdict: **{verdict}**")
    md.append("")
    md.append(rationale)
    md.append("")

    md.append("## PRE confusion matrix")
    md.append("")
    md.append(_format_confusion_matrix_md(pre_conf))
    md.append("")
    md.append("## POST confusion matrix")
    md.append("")
    md.append(_format_confusion_matrix_md(post_conf))
    md.append("")

    md.append("## Per-class metrics — PRE")
    md.append("")
    md.append(_format_per_class_table_md(pre_metrics))
    md.append("")
    md.append("## Per-class metrics — POST")
    md.append("")
    md.append(_format_per_class_table_md(post_metrics))
    md.append("")

    md.append("## Error patterns — PRE")
    md.append("")
    md.append(_format_error_patterns_md(pre_errors, by_case))
    md.append("")
    md.append("## Error patterns — POST")
    md.append("")
    md.append(_format_error_patterns_md(post_errors, by_case))
    md.append("")

    md.append("## Sample responses")
    md.append("")
    sample_ids = []
    for s in _STATUSES:
        cohort = [r for r in results if r.get("expected_status") == s and "error" not in r]
        if cohort:
            sample_ids.append(cohort[0]["case_id"])
            wrong = [
                r for r in cohort
                if r["post_predicted_status"] != r["expected_status"]
            ]
            if wrong:
                sample_ids.append(wrong[0]["case_id"])
    seen: set[str] = set()
    for cid in sample_ids:
        if cid in seen:
            continue
        seen.add(cid)
        r = by_case.get(cid)
        if r is None:
            continue
        response_block = (r["response"] or "").strip()
        if len(response_block) > 280:
            response_block = response_block[:277] + "..."
        md.append(
            f"### `{cid}` — expected `{r['expected_status']}`, "
            f"PRE→`{r['pre_predicted_status']}` POST→`{r['post_predicted_status']}`"
        )
        md.append(f"- Query: *{r['query']}*")
        md.append(f"- Memory seeded: {r['memory_size_seeded']}, retrieved: {r['retrieval_size']}")
        md.append(
            f"- Confidences: PRE={r['pre_confidence']:.3f}  "
            f"POST={r['post_confidence']:.3f}"
        )
        md.append(f"- Response: > {response_block}")
        md.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(md))


__all__ = [
    "compute_confusion_matrix",
    "compute_per_class_metrics",
    "compute_real_calibration",
    "generate_report",
    "identify_error_patterns",
    "overall_accuracy",
]
