"""Reporting helpers shared by experiment scripts.

Keeps the per-experiment scripts in ``experiments/`` small: each one
configures hyperparameters, runs the harness, then hands the result
to :func:`print_summary` and :func:`save_run` for human and machine
consumption respectively.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from continual_synapse.evaluation.metrics import (
    average_accuracy,
    average_forgetting,
    backward_transfer,
    forward_transfer,
)
from continual_synapse.evaluation.runner import RunResult


@dataclass
class MetricsSummary:
    """Continual-learning metrics computed from a :class:`RunResult`."""

    average_accuracy: float
    average_forgetting: float
    backward_transfer: float
    forward_transfer: float | None
    per_task_final: dict[str, float] = field(default_factory=dict)


def compute_metrics(result: RunResult) -> MetricsSummary:
    """Compute all four continual-learning metrics from a run.

    ``forward_transfer`` is ``None`` when the runner did not record
    zero-shot accuracy on upcoming tasks (``record_zero_shot=False``).
    """
    R = result.accuracy_matrix
    per_task_final = {
        name: float(R[-1, j]) for j, name in enumerate(result.task_names)
    }

    fwt: float | None
    try:
        fwt = forward_transfer(R, result.random_baseline)
    except ValueError:
        fwt = None

    return MetricsSummary(
        average_accuracy=average_accuracy(R),
        average_forgetting=average_forgetting(R),
        backward_transfer=backward_transfer(R),
        forward_transfer=fwt,
        per_task_final=per_task_final,
    )


def print_summary(
    result: RunResult,
    summary: MetricsSummary | None = None,
    method: str = "",
) -> None:
    """Pretty-print a one-screen summary of a continual-learning run."""
    summary = summary or compute_metrics(result)
    header = f"Run summary — benchmark={result.benchmark}"
    if method:
        header += f", method={method}"
    print(header)
    print("-" * len(header))
    print("Final per-task accuracy:")
    for name, acc in summary.per_task_final.items():
        print(f"  {name:<24s} {acc:6.3f}")
    print()
    print(f"  Average accuracy (ACC):  {summary.average_accuracy:6.3f}")
    print(f"  Average forgetting (FGT): {summary.average_forgetting:+6.3f}")
    print(f"  Backward transfer (BWT):  {summary.backward_transfer:+6.3f}")
    if summary.forward_transfer is not None:
        print(f"  Forward transfer (FWT):   {summary.forward_transfer:+6.3f}")
    else:
        print("  Forward transfer (FWT):   n/a (zero-shot not recorded)")


def save_run(
    result: RunResult,
    *,
    experiment: str,
    method: str,
    config: dict[str, Any],
    output_dir: Path | str = "results/logs",
    summary: MetricsSummary | None = None,
) -> Path:
    """Serialise a run as JSON. Returns the path written.

    Filename: ``<output_dir>/<unix_ts>_<experiment>_<method>.json``.
    NaN entries in the accuracy matrix become JSON ``null`` so the
    file can be re-parsed by any standard JSON tool.
    """
    summary = summary or compute_metrics(result)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    path = out / f"{ts}_{experiment}_{method}.json"

    payload: dict[str, Any] = {
        "experiment": experiment,
        "method": method,
        "benchmark": result.benchmark,
        "timestamp": ts,
        "git_sha": _git_sha(),
        "config": config,
        "task_names": result.task_names,
        "accuracy_matrix": _matrix_to_json(result.accuracy_matrix),
        "random_baseline": result.random_baseline.tolist(),
        "metrics": asdict(summary),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def _matrix_to_json(arr: np.ndarray) -> list[list[float | None]]:
    out: list[list[float | None]] = []
    for row in arr:
        out.append([None if math.isnan(v) else float(v) for v in row])
    return out


def _git_sha() -> str | None:
    """Return the current commit SHA, or ``None`` outside a repo."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return sha.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
