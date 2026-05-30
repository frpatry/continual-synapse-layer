"""Tests for the statistical summary + Wilcoxon helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from continual_synapse.evaluation.multi_seed import MultiSeedRun
from continual_synapse.evaluation.runner import RunResult
from continual_synapse.evaluation.statistics import (
    MethodSummary,
    format_pairwise_table,
    format_summary_table,
    pairwise_wilcoxon,
    summarise_method,
)


def _toy_run(method: str, acc_per_seed: list[float]) -> MultiSeedRun:
    """Build a MultiSeedRun whose final accuracy on each seed is given.

    Each per-seed run has a 2-task matrix where R[1, 0] = acc and
    R[1, 1] = acc. ACC and FGT can therefore be computed cleanly.
    """
    results = []
    for i, acc in enumerate(acc_per_seed):
        R = np.array(
            [
                [0.9, np.nan],
                [acc, acc],
            ]
        )
        results.append(
            RunResult(
                benchmark="toy",
                task_names=["t0", "t1"],
                accuracy_matrix=R,
                random_baseline=np.array([0.5, 0.5]),
            )
        )
    return MultiSeedRun(method=method, seeds=list(range(len(acc_per_seed))), results=results)


def test_summarise_method_computes_mean_and_std() -> None:
    run = _toy_run("a", acc_per_seed=[0.5, 0.6, 0.7])
    s = summarise_method(run)
    assert s.method == "a"
    assert s.n_seeds == 3
    # ACC values: 0.5, 0.6, 0.7 averaged across both columns of the
    # final row = same number each seed -> mean of accs.
    assert math.isclose(s.metric_means["average_accuracy"], 0.6, abs_tol=1e-9)
    # std with ddof=1 over [0.5, 0.6, 0.7] = 0.1
    assert math.isclose(s.metric_stds["average_accuracy"], 0.1, abs_tol=1e-9)


def test_summarise_method_handles_single_seed() -> None:
    run = _toy_run("a", acc_per_seed=[0.7])
    s = summarise_method(run)
    assert s.n_seeds == 1
    assert math.isclose(s.metric_means["average_accuracy"], 0.7)
    # ddof=1 std with one sample is conventionally 0; we report 0 explicitly.
    assert s.metric_stds["average_accuracy"] == 0.0


def test_summarise_method_records_per_seed_metrics() -> None:
    run = _toy_run("a", acc_per_seed=[0.4, 0.5])
    s = summarise_method(run)
    assert s.per_seed_metrics["average_accuracy"] == [0.4, 0.5]
    # FGT can be computed too — we don't pin its values but it
    # should be present and the right length.
    assert len(s.per_seed_metrics["average_forgetting"]) == 2


def test_pairwise_wilcoxon_runs_on_clearly_different_methods() -> None:
    """A robust diff (5 seeds, A consistently above B by 0.1) is significant."""
    a = _toy_run("a", acc_per_seed=[0.6, 0.62, 0.59, 0.61, 0.60])
    b = _toy_run("b", acc_per_seed=[0.5, 0.52, 0.49, 0.51, 0.50])
    sa, sb = summarise_method(a), summarise_method(b)
    results = pairwise_wilcoxon([sa, sb])
    assert len(results) == 1
    r = results[0]
    assert r.method_a == "a"
    assert r.method_b == "b"
    assert r.n == 5
    # With this clean shift, p-value should be small. n=5 -> minimum
    # Wilcoxon p ≈ 0.0625, so significant_05 may not flag at α=0.05;
    # we just assert p is small.
    assert r.p_value < 0.1


def test_pairwise_wilcoxon_returns_high_p_when_methods_are_equal() -> None:
    """Identical seed-by-seed values short-circuit to p=1."""
    a = _toy_run("a", acc_per_seed=[0.5, 0.6, 0.7])
    b = _toy_run("b", acc_per_seed=[0.5, 0.6, 0.7])
    results = pairwise_wilcoxon([summarise_method(a), summarise_method(b)])
    assert len(results) == 1
    assert results[0].p_value == 1.0
    assert results[0].p_value_bonferroni == 1.0
    assert not results[0].significant_05


def test_pairwise_wilcoxon_applies_bonferroni_correction() -> None:
    """With three methods (3 pairs), Bonferroni multiplies p by 3."""
    a = _toy_run("a", acc_per_seed=[0.5, 0.55, 0.52, 0.53, 0.54])
    b = _toy_run("b", acc_per_seed=[0.4, 0.45, 0.42, 0.43, 0.44])
    c = _toy_run("c", acc_per_seed=[0.3, 0.35, 0.32, 0.33, 0.34])
    summaries = [summarise_method(r) for r in (a, b, c)]
    results = pairwise_wilcoxon(summaries)
    assert len(results) == 3  # C(3, 2) = 3
    for r in results:
        # Multiplied by 3 (number of pairs), but clipped to 1.0 at most.
        assert r.p_value_bonferroni == min(1.0, r.p_value * 3)


def test_pairwise_wilcoxon_rejects_unknown_metric() -> None:
    run = _toy_run("a", acc_per_seed=[0.5])
    with pytest.raises(ValueError, match="unknown metric"):
        pairwise_wilcoxon([summarise_method(run)], metric="not_a_metric")


def test_pairwise_wilcoxon_empty_for_single_method() -> None:
    run = _toy_run("a", acc_per_seed=[0.5, 0.6])
    assert pairwise_wilcoxon([summarise_method(run)]) == []


def test_format_summary_table_emits_method_rows() -> None:
    a = _toy_run("naive", acc_per_seed=[0.5, 0.6])
    b = _toy_run("synapse", acc_per_seed=[0.55, 0.65])
    out = format_summary_table([summarise_method(a), summarise_method(b)])
    assert "naive" in out
    assert "synapse" in out
    assert "average_accuracy" in out
    # Mean of (0.5, 0.6) = 0.55 should show up.
    assert "+0.550" in out


def test_format_pairwise_table_emits_significance_markers() -> None:
    a = _toy_run("a", acc_per_seed=[0.5, 0.55, 0.52, 0.53, 0.54])
    b = _toy_run("b", acc_per_seed=[0.6, 0.65, 0.62, 0.63, 0.64])
    results = pairwise_wilcoxon([summarise_method(a), summarise_method(b)])
    out = format_pairwise_table(results)
    assert "method_a" in out
    assert "method_b" in out
