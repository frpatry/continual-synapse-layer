"""Tests for the experiment reporting helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from continual_synapse.evaluation.reporting import (
    MetricsSummary,
    compute_metrics,
    save_run,
)
from continual_synapse.evaluation.runner import RunResult


def _make_result() -> RunResult:
    R = np.array(
        [
            [0.90, 0.55, np.nan],
            [0.80, 0.85, 0.60],
            [0.60, 0.70, 0.95],
        ]
    )
    return RunResult(
        benchmark="toy",
        task_names=["t0", "t1", "t2"],
        accuracy_matrix=R,
        random_baseline=np.array([0.5, 0.5, 0.5]),
    )


def test_compute_metrics_populates_all_fields() -> None:
    summary = compute_metrics(_make_result())
    assert isinstance(summary, MetricsSummary)
    assert summary.forward_transfer is not None
    assert summary.per_task_final == {"t0": 0.60, "t1": 0.70, "t2": 0.95}
    # ACC = mean of (0.60, 0.70, 0.95)
    assert abs(summary.average_accuracy - (0.60 + 0.70 + 0.95) / 3) < 1e-9


def test_compute_metrics_handles_missing_zero_shot() -> None:
    R = np.array(
        [
            [0.90, np.nan, np.nan],
            [0.80, 0.85, np.nan],
            [0.60, 0.70, 0.95],
        ]
    )
    result = RunResult(
        benchmark="toy",
        task_names=["t0", "t1", "t2"],
        accuracy_matrix=R,
        random_baseline=np.array([0.5, 0.5, 0.5]),
    )
    summary = compute_metrics(result)
    assert summary.forward_transfer is None


def test_save_run_writes_round_trippable_json(tmp_path: Path) -> None:
    result = _make_result()
    path = save_run(
        result,
        experiment="test_exp",
        method="dummy",
        config={"seed": 0, "lr": 0.01},
        output_dir=tmp_path,
    )
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["experiment"] == "test_exp"
    assert payload["method"] == "dummy"
    assert payload["benchmark"] == "toy"
    assert payload["task_names"] == ["t0", "t1", "t2"]
    # NaN entries serialise as JSON null.
    assert payload["accuracy_matrix"][0][2] is None
    # Non-NaN entries round-trip exactly.
    assert payload["accuracy_matrix"][2] == [0.60, 0.70, 0.95]
    assert "git_sha" in payload  # may be None outside a repo, key must exist
    assert payload["metrics"]["average_accuracy"] == compute_metrics(result).average_accuracy
