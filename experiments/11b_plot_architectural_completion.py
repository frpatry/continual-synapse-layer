"""Plot diagnostics from experiment 11 (architectural completion).

Reads the most-recent JSON log under
``results/logs/architectural_completion/`` and writes plots into
``results/figures/architectural_completion/``:

- ``avg_accuracy_vs_tasks_seen.png`` — running average accuracy
  per method (already standard).
- ``per_task_trajectory.png`` — 3-panel view of Task 1, T/2, T
  accuracy over training (already standard).
- ``forgetting_vs_first_task.png`` — Task 1 trajectory at larger
  size (already standard).
- ``gap_vs_tasks_seen.png`` — cs_full − naive ACC over training
  (already standard).
- ``store_byte_size_per_task.png`` (new) — total stored-document
  bytes per method, per task. Whether the compression sweep is
  actually bounding memory shows up here.
- ``precision_distribution_evolution.png`` (new) — for the cs_sweep
  and cs_full methods only, stacked-bar of how many entries are
  at each precision tier, snapshotted per task.

Run after experiment 11::

    python experiments/11b_plot_architectural_completion.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LOG_DIR = _REPO_ROOT / "results" / "logs" / "architectural_completion"
_DEFAULT_FIG_DIR = _REPO_ROOT / "results" / "figures" / "architectural_completion"

_PRECISION_TIERS = (32, 16, 8, 4)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--log-path", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_FIG_DIR)
    return p.parse_args()


def _latest_log(log_dir: Path) -> Path:
    candidates = sorted(log_dir.glob("*_11_architectural_completion*.json"))
    if not candidates:
        raise FileNotFoundError(f"No experiment 11 logs in {log_dir}")
    return candidates[-1]


def _accuracy_matrices(method: dict) -> np.ndarray:
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
    T = R.shape[-1]
    out = np.empty(R.shape[:-1])
    for i in range(T):
        out[..., i] = np.nanmean(R[..., i, : i + 1], axis=-1)
    return out


def _accuracy_on_task(R: np.ndarray, j: int) -> np.ndarray:
    T = R.shape[-1]
    out = np.full(R.shape[:-1], np.nan)
    out[..., j:] = R[..., j:, j]
    return out


def plot_running_average(payload: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    methods = payload["methods"]
    T = len(methods[0]["results"][0]["task_names"])
    for method in methods:
        R = _accuracy_matrices(method)
        run_avg = _running_avg_accuracy(R)
        mean = np.nanmean(run_avg, axis=0)
        std = np.nanstd(run_avg, axis=0)
        x = np.arange(1, T + 1)
        line, = ax.plot(x, mean, label=method["method"], marker="o", ms=3)
        ax.fill_between(
            x, mean - std, mean + std, alpha=0.15, color=line.get_color()
        )
    ax.set_title(
        f"Running average accuracy (n={len(methods[0]['results'])} seeds)"
    )
    ax.set_xlabel("Number of tasks trained")
    ax.set_ylabel("Avg accuracy on tasks 1..i")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_per_task_trajectories(payload: dict, out_path: Path) -> None:
    methods = payload["methods"]
    T = len(methods[0]["results"][0]["task_names"])
    task_indices = [0, T // 2, T - 1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, j in zip(axes, task_indices):
        for method in methods:
            R = _accuracy_matrices(method)
            traj = _accuracy_on_task(R, j)
            mean = np.nanmean(traj, axis=0)
            std = np.nanstd(traj, axis=0)
            x = np.arange(T)
            line, = ax.plot(x, mean, label=method["method"])
            ax.fill_between(
                x, mean - std, mean + std, alpha=0.15, color=line.get_color()
            )
        ax.set_title(f"Accuracy on Task {j + 1}")
        ax.set_xlabel("Tasks trained so far")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Accuracy")
    axes[-1].legend(loc="lower left", fontsize=8)
    fig.suptitle(
        f"Per-task accuracy (n={len(methods[0]['results'])} seeds, mean ± std)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_forgetting_first_task(payload: dict, out_path: Path) -> None:
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
            x, mean - std, mean + std, alpha=0.15, color=line.get_color()
        )
    ax.set_title(
        f"Accuracy on Task 1 vs subsequent training "
        f"(n={len(methods[0]['results'])} seeds)"
    )
    ax.set_xlabel("Tasks trained so far")
    ax.set_ylabel("Accuracy on Task 1")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_gap(payload: dict, out_path: Path) -> None:
    by_name = {m["method"]: m for m in payload["methods"]}
    if "cs_full" not in by_name or "naive" not in by_name:
        return
    Ra = _accuracy_matrices(by_name["cs_full"])
    Rb = _accuracy_matrices(by_name["naive"])
    if Ra.shape[0] != Rb.shape[0]:
        return
    avg_a = _running_avg_accuracy(Ra)
    avg_b = _running_avg_accuracy(Rb)
    gap = avg_a - avg_b
    mean = np.nanmean(gap, axis=0)
    std = np.nanstd(gap, axis=0)
    T = gap.shape[-1]
    x = np.arange(1, T + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.plot(x, mean, marker="o", ms=3, color="tab:purple", label="gap mean")
    ax.fill_between(
        x, mean - std, mean + std, alpha=0.2, color="tab:purple", label="± 1 std"
    )
    ax.set_title(
        f"cs_full − naive ACC gap (n={Ra.shape[0]} seeds, mean ± std)"
    )
    ax.set_xlabel("Number of tasks trained")
    ax.set_ylabel("ACC(cs_full) − ACC(naive)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_memory_per_task(payload: dict, out_path: Path) -> None:
    """Per-method memory footprint (decoded document bytes) per task."""
    diagnostics = payload.get("diagnostics", {})
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for method_name, seed_diags in diagnostics.items():
        if not seed_diags:
            continue
        all_per_task = [d.get("per_task", []) for d in seed_diags]
        if not any(all_per_task):
            continue
        # Each per_task may have store_byte_size; collect per task index.
        T = max(len(per_task) for per_task in all_per_task)
        bytes_grid = np.full((len(all_per_task), T), np.nan)
        for s, per_task in enumerate(all_per_task):
            for entry in per_task:
                idx = int(entry["task_index"])
                if 0 <= idx < T and "store_byte_size" in entry:
                    bytes_grid[s, idx] = int(entry["store_byte_size"])
        if np.isnan(bytes_grid).all():
            continue
        mean = np.nanmean(bytes_grid, axis=0)
        std = np.nanstd(bytes_grid, axis=0)
        x = np.arange(T)
        line, = ax.plot(x, mean / 1024, marker="o", ms=3, label=method_name)
        ax.fill_between(
            x,
            (mean - std) / 1024,
            (mean + std) / 1024,
            alpha=0.15,
            color=line.get_color(),
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_title("Cold-storage document size per task (mean across seeds)")
    ax.set_xlabel("Task index")
    ax.set_ylabel("Total decoded document bytes (KB)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_precision_distribution(payload: dict, out_path: Path) -> None:
    """For methods with a sweep, stacked-bar of precision tier counts over tasks."""
    diagnostics = payload.get("diagnostics", {})
    methods_with_sweep = [
        m for m, ds in diagnostics.items()
        if any(
            entry.get("last_compression_counts")
            for d in ds
            for entry in d.get("per_task", [])
        )
    ]
    if not methods_with_sweep:
        return

    fig, axes = plt.subplots(
        len(methods_with_sweep),
        1,
        figsize=(9, 3.5 * len(methods_with_sweep)),
        sharex=True,
    )
    if len(methods_with_sweep) == 1:
        axes = [axes]

    for ax, method_name in zip(axes, methods_with_sweep):
        seed_diags = diagnostics[method_name]
        # Use the first seed for the precision-distribution view; the
        # numbers are similar across seeds and a stacked-bar averaging
        # would be misleading.
        per_task = seed_diags[0].get("per_task", [])
        T = len(per_task)
        if T == 0:
            continue
        x = np.arange(T)
        # Build a (T, 4) matrix of counts at each precision tier.
        counts = np.zeros((T, len(_PRECISION_TIERS)), dtype=int)
        for i, entry in enumerate(per_task):
            lc = entry.get("last_compression_counts", {})
            # Keys are stringified ints from JSON; coerce.
            lc_int = {int(k): int(v) for k, v in lc.items()}
            for j, p in enumerate(_PRECISION_TIERS):
                counts[i, j] = lc_int.get(p, 0)

        bottom = np.zeros(T)
        for j, p in enumerate(_PRECISION_TIERS):
            ax.bar(
                x,
                counts[:, j],
                bottom=bottom,
                width=0.85,
                label=f"{p}-bit",
                alpha=0.85,
            )
            bottom += counts[:, j]
        ax.set_title(
            f"{method_name}: cold-storage precision distribution (seed 0)"
        )
        ax.set_ylabel("# entries")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(loc="upper left", fontsize=8, ncol=4)
    axes[-1].set_xlabel("Task index")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    log_path = args.log_path or _latest_log(_DEFAULT_LOG_DIR)
    print(f"Reading {log_path}")
    payload = json.loads(log_path.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_running_average(
        payload, args.output_dir / "avg_accuracy_vs_tasks_seen.png"
    )
    plot_per_task_trajectories(
        payload, args.output_dir / "per_task_trajectory.png"
    )
    plot_forgetting_first_task(
        payload, args.output_dir / "forgetting_vs_first_task.png"
    )
    plot_gap(payload, args.output_dir / "gap_vs_tasks_seen.png")
    plot_memory_per_task(
        payload, args.output_dir / "store_byte_size_per_task.png"
    )
    plot_precision_distribution(
        payload, args.output_dir / "precision_distribution_evolution.png"
    )
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
