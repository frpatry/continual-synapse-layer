"""Plot diagnostics from experiment 12 (audit-driven architectural fixes).

Reads the most-recent JSON log under ``results/logs/audit_fixes/``
and writes four plots into ``results/figures/audit_fixes/``:

- ``reward_distribution.png`` — per-method box-and-whisker built
  from the reward summary (min / p10 / p50 / p90 / max) with the
  mean overlaid. Naive has no reward stream and is skipped.
- ``sparse_density_per_task.png`` — fraction of non-zero strengths
  at the end of each task, per method, mean ± std across seeds.
- ``per_task_accuracy_trajectory.png`` — running average accuracy
  on tasks 1..i as more tasks are trained, all five methods
  overlaid (mean ± std band).
- ``acc_vs_std_scatter.png`` — one point per method at
  (mean ACC, std ACC) across seeds; visualises the
  accuracy / stability trade-off.

Run after experiment 12 completes::

    python experiments/12b_plot_audit_fixes.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LOG_DIR = _REPO_ROOT / "results" / "logs" / "audit_fixes"
_DEFAULT_FIG_DIR = _REPO_ROOT / "results" / "figures" / "audit_fixes"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--log-path", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_FIG_DIR)
    return p.parse_args()


def _latest_log(log_dir: Path) -> Path:
    candidates = sorted(log_dir.glob("*_12_audit_fixes*.json"))
    if not candidates:
        raise FileNotFoundError(f"No experiment 12 logs in {log_dir}")
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


def plot_reward_distribution(payload: dict, out_path: Path) -> None:
    """Box-and-whisker per method from the persisted quantile summary."""
    reward_summary = payload.get("reward_summary", {})
    method_order = [m["method"] for m in payload["methods"]]
    methods = [m for m in method_order if reward_summary.get(m, {}).get("n", 0) > 0]
    if not methods:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(methods))
    for pos, name in zip(positions, methods):
        h = reward_summary[name]
        # Whiskers from min to max.
        ax.plot([pos, pos], [h["min"], h["max"]], color="black", linewidth=1.0)
        # Box from p10 to p90 with median line.
        box_low, box_high = h["p10"], h["p90"]
        rect = plt.Rectangle(
            (pos - 0.3, box_low),
            0.6,
            box_high - box_low,
            facecolor="tab:blue",
            edgecolor="black",
            alpha=0.4,
        )
        ax.add_patch(rect)
        ax.plot(
            [pos - 0.3, pos + 0.3],
            [h["p50"], h["p50"]],
            color="black",
            linewidth=1.5,
        )
        ax.plot(
            pos,
            h["mean"],
            marker="D",
            color="tab:red",
            markersize=6,
            label="mean" if pos == positions[0] else None,
        )

    ax.axhline(0.0, color="grey", linewidth=0.8, alpha=0.6, linestyle="--")
    ax.set_xticks(positions)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("Per-batch reward")
    ax.set_title(
        "Reward distribution per method "
        "(whiskers: min/max, box: p10–p90, line: median, diamond: mean)"
    )
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_sparse_density_per_task(payload: dict, out_path: Path) -> None:
    diagnostics = payload.get("diagnostics", {})
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for method_name, seed_diags in diagnostics.items():
        per_seed = [d.get("per_task", []) for d in seed_diags]
        if not any(per_seed):
            continue
        T = max(len(pt) for pt in per_seed)
        grid = np.full((len(per_seed), T), np.nan)
        for s, per_task in enumerate(per_seed):
            for entry in per_task:
                idx = int(entry["task_index"])
                if 0 <= idx < T and "sparse_density" in entry:
                    grid[s, idx] = float(entry["sparse_density"])
        if np.isnan(grid).all():
            continue
        mean = np.nanmean(grid, axis=0)
        std = np.nanstd(grid, axis=0)
        x = np.arange(T)
        line, = ax.plot(x, mean, marker="o", ms=3, label=method_name)
        ax.fill_between(
            x, mean - std, mean + std, alpha=0.15, color=line.get_color()
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        "Sparse density per task (fraction of non-zero strengths, mean ± std)"
    )
    ax.set_xlabel("Task index")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_per_task_accuracy_trajectory(payload: dict, out_path: Path) -> None:
    """Running-average accuracy across all tasks, every method overlaid."""
    methods = payload["methods"]
    if not methods:
        return
    T = len(methods[0]["results"][0]["task_names"])
    fig, ax = plt.subplots(figsize=(9, 5))
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
        f"Per-task accuracy trajectory "
        f"(n={len(methods[0]['results'])} seeds, mean ± std)"
    )
    ax.set_xlabel("Number of tasks trained")
    ax.set_ylabel("Avg accuracy on tasks 1..i")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_acc_vs_std_scatter(payload: dict, out_path: Path) -> None:
    summaries = payload.get("summaries", [])
    if not summaries:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    for s in summaries:
        name = s["method"]
        acc = s["metric_means"].get("average_accuracy")
        std = s["metric_stds"].get("average_accuracy")
        if acc is None or std is None:
            continue
        ax.scatter(acc, std, s=80, label=name)
        ax.annotate(
            name,
            (acc, std),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )
    ax.set_xlabel("Mean ACC across seeds")
    ax.set_ylabel("Std ACC across seeds")
    ax.set_title("Accuracy vs stability trade-off (one point per method)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    log_path = args.log_path or _latest_log(_DEFAULT_LOG_DIR)
    print(f"Reading {log_path}")
    payload = json.loads(log_path.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_reward_distribution(
        payload, args.output_dir / "reward_distribution.png"
    )
    plot_sparse_density_per_task(
        payload, args.output_dir / "sparse_density_per_task.png"
    )
    plot_per_task_accuracy_trajectory(
        payload, args.output_dir / "per_task_accuracy_trajectory.png"
    )
    plot_acc_vs_std_scatter(
        payload, args.output_dir / "acc_vs_std_scatter.png"
    )
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
