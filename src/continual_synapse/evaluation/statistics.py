"""Statistical summary and significance tests for multi-seed runs.

Implements the protocol called out in PROJECT_PLAN.md §8:

- Per-method mean ± std for ACC, FGT, BWT, FWT.
- Pairwise Wilcoxon signed-rank tests on a chosen metric.
- Bonferroni correction across the number of pairs compared.

Wilcoxon is paired (per-seed): for each seed we have a metric value
for method A and method B, and the test asks whether A - B is
systematically non-zero. This requires the same seed list across
both methods — the helpers enforce that explicitly.

The Wilcoxon test is delegated to ``scipy.stats.wilcoxon``; the rest
of the module is plain numpy so the bulk of the file is testable
without scipy. We import scipy lazily inside :func:`pairwise_wilcoxon`
so importing the module does not require scipy.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np

from continual_synapse.evaluation.multi_seed import MultiSeedRun
from continual_synapse.evaluation.reporting import MetricsSummary, compute_metrics


_METRIC_NAMES = (
    "average_accuracy",
    "average_forgetting",
    "backward_transfer",
    "forward_transfer",
)


@dataclass
class MethodSummary:
    """Mean and std for a single method across its seeds."""

    method: str
    n_seeds: int
    metric_means: dict[str, float]
    metric_stds: dict[str, float]
    per_seed_metrics: dict[str, list[float]]


def summarise_method(run: MultiSeedRun) -> MethodSummary:
    """Compute mean/std and per-seed metric arrays for a multi-seed run."""
    per_seed: dict[str, list[float]] = {m: [] for m in _METRIC_NAMES}
    for result in run.results:
        summary: MetricsSummary = compute_metrics(result)
        per_seed["average_accuracy"].append(summary.average_accuracy)
        per_seed["average_forgetting"].append(summary.average_forgetting)
        per_seed["backward_transfer"].append(summary.backward_transfer)
        per_seed["forward_transfer"].append(
            float("nan") if summary.forward_transfer is None
            else summary.forward_transfer
        )

    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for metric, values in per_seed.items():
        arr = np.asarray(values, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        means[metric] = float(finite.mean()) if finite.size else float("nan")
        stds[metric] = float(finite.std(ddof=1)) if finite.size > 1 else 0.0

    return MethodSummary(
        method=run.method,
        n_seeds=run.n_seeds,
        metric_means=means,
        metric_stds=stds,
        per_seed_metrics=per_seed,
    )


@dataclass
class PairwiseResult:
    """One row of the pairwise Wilcoxon comparison table."""

    method_a: str
    method_b: str
    metric: str
    n: int
    statistic: float
    p_value: float
    p_value_bonferroni: float
    significant_05: bool


def pairwise_wilcoxon(
    summaries: Iterable[MethodSummary],
    metric: str = "average_accuracy",
    alpha: float = 0.05,
) -> list[PairwiseResult]:
    """Pairwise Wilcoxon signed-rank with Bonferroni correction.

    Args:
        summaries: Iterable of :class:`MethodSummary` (one per method).
        metric: Which per-seed metric to compare. Must be one of
            ``_METRIC_NAMES``.
        alpha: Significance level *before* Bonferroni correction.
            The reported ``significant_05`` flag uses
            ``p_value_bonferroni < alpha``.

    Returns:
        A list of :class:`PairwiseResult`, one entry per unordered
        method pair. Bonferroni multiplier is the number of pairs
        compared (``k choose 2`` for ``k`` methods).
    """
    if metric not in _METRIC_NAMES:
        raise ValueError(
            f"unknown metric {metric!r}; expected one of {_METRIC_NAMES}"
        )

    from scipy.stats import wilcoxon  # type: ignore[import-untyped]

    summaries = list(summaries)
    if len(summaries) < 2:
        return []

    n_pairs = len(summaries) * (len(summaries) - 1) // 2
    results: list[PairwiseResult] = []
    for sa, sb in combinations(summaries, 2):
        a_vals = np.asarray(sa.per_seed_metrics[metric], dtype=np.float64)
        b_vals = np.asarray(sb.per_seed_metrics[metric], dtype=np.float64)
        mask = np.isfinite(a_vals) & np.isfinite(b_vals)
        a_vals = a_vals[mask]
        b_vals = b_vals[mask]
        n = int(a_vals.size)
        if n < 1 or np.allclose(a_vals, b_vals):
            stat = 0.0
            p = 1.0
        else:
            test = wilcoxon(a_vals, b_vals, zero_method="wilcox")
            stat = float(test.statistic)
            p = float(test.pvalue)
        p_corr = min(1.0, p * n_pairs)
        results.append(
            PairwiseResult(
                method_a=sa.method,
                method_b=sb.method,
                metric=metric,
                n=n,
                statistic=stat,
                p_value=p,
                p_value_bonferroni=p_corr,
                significant_05=p_corr < alpha,
            )
        )
    return results


def format_summary_table(summaries: Iterable[MethodSummary]) -> str:
    """Render method × metric mean ± std as a fixed-width string."""
    summaries = list(summaries)
    if not summaries:
        return "(no methods)\n"

    headers = ["method"] + [m for m in _METRIC_NAMES]
    rows: list[list[str]] = [headers]
    for s in summaries:
        row = [f"{s.method} (n={s.n_seeds})"]
        for m in _METRIC_NAMES:
            mean = s.metric_means[m]
            std = s.metric_stds[m]
            if np.isnan(mean):
                row.append("n/a")
            else:
                row.append(f"{mean:+.3f} ± {std:.3f}")
        rows.append(row)

    widths = [max(len(r[c]) for r in rows) for c in range(len(headers))]
    lines = []
    for ri, r in enumerate(rows):
        line = "  ".join(c.ljust(widths[ci]) for ci, c in enumerate(r))
        lines.append(line)
        if ri == 0:
            lines.append("-" * len(line))
    return "\n".join(lines) + "\n"


def format_pairwise_table(comparisons: Iterable[PairwiseResult]) -> str:
    """Render pairwise comparison table."""
    rows = [["method_a", "method_b", "n", "stat", "p", "p (bonf)", "sig"]]
    for c in comparisons:
        rows.append(
            [
                c.method_a,
                c.method_b,
                str(c.n),
                f"{c.statistic:.2f}",
                f"{c.p_value:.4f}",
                f"{c.p_value_bonferroni:.4f}",
                "*" if c.significant_05 else "",
            ]
        )
    if len(rows) == 1:
        return "(no pairs)\n"
    widths = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
    lines = []
    for ri, r in enumerate(rows):
        line = "  ".join(c.ljust(widths[ci]) for ci, c in enumerate(r))
        lines.append(line)
        if ri == 0:
            lines.append("-" * len(line))
    return "\n".join(lines) + "\n"
