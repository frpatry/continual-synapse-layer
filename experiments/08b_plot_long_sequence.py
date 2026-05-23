"""Plot trajectories and aggregate diagnostics from experiment 08.

Reads the most recent JSON log from
``results/logs/long_sequence/`` and writes plots into
``results/figures/long_sequence/``:

- ``per_task_trajectory.png`` — accuracy on tasks 1, ``T/2``, and
  ``T`` over the training progression, one curve per method
  (mean across seeds).
- ``avg_accuracy_vs_tasks_seen.png`` — running average accuracy
  on all tasks seen so far, as a function of how many tasks have
  been trained.
- ``consolidations_per_task.png`` — number of consolidation
  cycles accumulated through each task (cold-storage variant
  only).
- ``forgetting_vs_first_task.png`` — accuracy on task 1 as the
  model trains on subsequent tasks. Lower curve = more
  forgetting; the qualitative shape matters more than the mean.

Run after experiment 08::

    python experiments/08b_plot_long_sequence.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LOG_DIR = _REPO_ROOT / "results" / "logs" / "long_sequence"
_DEFAULT_FIG_DIR = _REPO_ROOT / "results" / "figures" / "long_sequence"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Path to the experiment 08 JSON log. Defaults to the "
        "newest file in results/logs/long_sequence/.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_FIG_DIR,
    )
    return p.parse_args()


def _latest_log(log_dir: Path) -> Path:
    candidates = sorted(log_dir.glob("*_08_long_sequence_decisive.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No experiment 08 logs found in {log_dir}"
        )
    return candidates[-1]


def _accuracy_matrices(method: dict) -> np.ndarray:
    """Stack one ``(T, T)`` matrix per seed into a ``(seeds, T, T)`` array."""
    mats = []
    for run in method["results"]:
        arr = np.array(
            [
                [np.nan if v is None else float(v) for v in row]
                for row in run["accuracy_matrix"]
            ],
            dtype=np.float64,
        )
        mats.append(arr)
    return np.stack(mats, axis=0)


def _running_avg_accuracy(R: np.ndarray) -> np.ndarray:
    """Per training step i: mean accuracy on tasks 0..i."""
    T = R.shape[-1]
    out = np.empty(R.shape[:-1])  # (seeds, T)
    for i in range(T):
        out[..., i] = np.nanmean(R[..., i, : i + 1], axis=-1)
    return out


def _accuracy_on_task(R: np.ndarray, j: int) -> np.ndarray:
    """Accuracy on task j across training progression. NaN before training j."""
    T = R.shape[-1]
    out = np.full(R.shape[:-1], np.nan)
    out[..., j:] = R[..., j:, j]
    return out


def plot_per_task_trajectories(payload: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    methods = payload["methods"]
    T = len(methods[0]["results"][0]["task_names"])
    task_indices = [0, T // 2, T - 1]

    for ax, j in zip(axes, task_indices):
        for method in methods:
            R = _accuracy_matrices(method)  # (seeds, T, T)
            traj = _accuracy_on_task(R, j)  # (seeds, T)
            mean = np.nanmean(traj, axis=0)
            std = np.nanstd(traj, axis=0)
            x = np.arange(T)
            line, = ax.plot(x, mean, label=method["method"])
            ax.fill_between(
                x,
                mean - std,
                mean + std,
                alpha=0.15,
                color=line.get_color(),
            )
        ax.set_title(f"Accuracy on Task {j + 1}")
        ax.set_xlabel("Tasks trained so far")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Accuracy")
    axes[-1].legend(loc="lower left", fontsize=8)
    fig.suptitle(
        "Per-task accuracy over the training progression "
        f"(n={len(methods[0]['results'])} seeds, mean ± std)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_running_average(payload: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    methods = payload["methods"]
    T = len(methods[0]["results"][0]["task_names"])

    for method in methods:
        R = _accuracy_matrices(method)
        run_avg = _running_avg_accuracy(R)  # (seeds, T)
        mean = np.nanmean(run_avg, axis=0)
        std = np.nanstd(run_avg, axis=0)
        x = np.arange(1, T + 1)
        line, = ax.plot(x, mean, label=method["method"], marker="o", ms=3)
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.15,
            color=line.get_color(),
        )
    ax.set_title(
        "Average accuracy on tasks seen so far"
        f" (n={len(methods[0]['results'])} seeds)"
    )
    ax.set_xlabel("Number of tasks trained")
    ax.set_ylabel("Average accuracy on tasks 1..i")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_consolidations(payload: dict, out_path: Path) -> None:
    diagnostics = payload.get("diagnostics", {})
    cs_diag = diagnostics.get("synapse_full_cold_storage", [])
    if not cs_diag or not any(d.get("per_task") for d in cs_diag):
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for d in cs_diag:
        per_task = d.get("per_task", [])
        if not per_task:
            continue
        idxs = [int(p["task_index"]) for p in per_task]
        counts = [int(p["consolidation_count"]) for p in per_task]
        ax.plot(idxs, counts, marker="o", ms=3, alpha=0.6, label=f"seed {d['seed']}")
    ax.set_title("Cold-storage consolidation cycles vs task index")
    ax.set_xlabel("Task index (0-indexed)")
    ax.set_ylabel("Cumulative consolidations")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_forgetting_first_task(payload: dict, out_path: Path) -> None:
    """Same data as the first panel of per-task trajectories, but on
    its own at a larger size — this is the headline forgetting plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = payload["methods"]
    T = len(methods[0]["results"][0]["task_names"])
    for method in methods:
        R = _accuracy_matrices(method)
        traj = _accuracy_on_task(R, 0)
        mean = np.nanmean(traj, axis=0)
        std = np.nanstd(traj, axis=0)
        x = np.arange(T)
        line, = ax.plot(x, mean, label=method["method"], marker="o", ms=3)
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.15,
            color=line.get_color(),
        )
    ax.set_title(
        "Accuracy on Task 1 vs subsequent training "
        f"(n={len(methods[0]['results'])} seeds, mean ± std)"
    )
    ax.set_xlabel("Tasks trained so far")
    ax.set_ylabel("Accuracy on Task 1")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    log_path = args.log_path or _latest_log(_DEFAULT_LOG_DIR)
    print(f"Reading {log_path}")
    payload = json.loads(log_path.read_text())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_per_task_trajectories(
        payload, args.output_dir / "per_task_trajectory.png"
    )
    plot_running_average(
        payload, args.output_dir / "avg_accuracy_vs_tasks_seen.png"
    )
    plot_consolidations(
        payload, args.output_dir / "consolidations_per_task.png"
    )
    plot_forgetting_first_task(
        payload, args.output_dir / "forgetting_vs_first_task.png"
    )
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
