"""Plot the headline scaling result from experiment 21.

Reads every ``<ts>_21_scaling_T*.json`` under
``results/logs/scaling/`` that shares the most recent timestamp
prefix, and produces:

- ``avg_accuracy_vs_task_length.png`` — one line per method,
  x = task length, y = ACC mean ± std (error bars). The
  headline plot the user asked for: shows the crossover point
  if one exists.
- ``forgetting_vs_task_length.png`` — same axes, but FGT mean
  ± std. Useful complement: a method that "wins" on ACC by
  forgetting less should show a downward FGT trajectory.
- ``task0_retention_vs_task_length.png`` — Task-0 ACC at the
  end of training, per method, per length. This is the
  "does the protection actually protect the first task across
  long sequences?" plot.

Run from the repo root::

    python experiments/21b_plot_scaling.py

By default scans ``results/logs/scaling/`` for the most recent
shared-timestamp T={15,30,50} triple and plots it. Override with
``--log-paths path1 path2 ...`` to plot an arbitrary set.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LOG_DIR = _REPO_ROOT / "results" / "logs" / "scaling"
_DEFAULT_FIG_DIR = _REPO_ROOT / "results" / "figures" / "scaling"

# Match the timestamp prefix in 21-style filenames so we can group
# all task-length runs from a single launch together.
_TS_RE = re.compile(r"^(\d+)_21_scaling_T(\d+)\.json$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--log-paths", type=Path, nargs="+", default=None,
                   help="Override auto-discovery: explicit list of "
                        "T-keyed JSON paths to combine.")
    p.add_argument("--log-dir", type=Path, default=_DEFAULT_LOG_DIR)
    p.add_argument("--output-dir", type=Path, default=_DEFAULT_FIG_DIR)
    return p.parse_args()


def _discover_latest_log_group(log_dir: Path) -> list[Path]:
    """Find the largest-timestamp set of T-keyed JSONs in log_dir.

    Returns the JSON paths sorted ascending by task length.
    """
    if not log_dir.exists():
        raise FileNotFoundError(f"log dir does not exist: {log_dir}")
    by_ts: dict[int, list[tuple[int, Path]]] = {}
    for p in log_dir.iterdir():
        match = _TS_RE.match(p.name)
        if match is None:
            continue
        ts = int(match.group(1))
        T = int(match.group(2))
        by_ts.setdefault(ts, []).append((T, p))
    if not by_ts:
        raise FileNotFoundError(
            f"no 21_scaling_T*.json files under {log_dir}"
        )
    latest_ts = max(by_ts)
    sorted_pairs = sorted(by_ts[latest_ts], key=lambda pair: pair[0])
    return [path for _, path in sorted_pairs]


def _load_results(paths: list[Path]) -> dict[int, dict]:
    """Map task-length T -> the parsed log payload."""
    out: dict[int, dict] = {}
    for p in paths:
        d = json.loads(p.read_text())
        T = int(d["num_tasks"])
        out[T] = d
    return out


def _collect_metric(
    results: dict[int, dict], method: str, metric: str
) -> tuple[list[int], list[float], list[float]]:
    """Return (task_lengths, means, stds) for one method across lengths."""
    Ts = sorted(results.keys())
    means = []
    stds = []
    for T in Ts:
        summary = next(
            (s for s in results[T]["summaries"] if s["method"] == method),
            None,
        )
        if summary is None:
            means.append(float("nan"))
            stds.append(float("nan"))
            continue
        means.append(summary["metric_means"].get(metric, float("nan")))
        stds.append(summary["metric_stds"].get(metric, float("nan")))
    return Ts, means, stds


def _collect_task0_retention(
    results: dict[int, dict], method: str
) -> tuple[list[int], list[float], list[float]]:
    """Per-method Task-0 ACC at end of training (R[T-1, 0]), mean ± std
    across seeds, per task length."""
    Ts = sorted(results.keys())
    means: list[float] = []
    stds: list[float] = []
    for T in Ts:
        method_block = next(
            (m for m in results[T]["methods"] if m["method"] == method),
            None,
        )
        if method_block is None:
            means.append(float("nan"))
            stds.append(float("nan"))
            continue
        vals = []
        for r in method_block["results"]:
            row = r["accuracy_matrix"][T - 1]
            v = row[0]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                vals.append(float(v))
        if not vals:
            means.append(float("nan"))
            stds.append(float("nan"))
            continue
        arr = np.asarray(vals, dtype=np.float64)
        means.append(float(arr.mean()))
        stds.append(float(arr.std(ddof=1)) if arr.size > 1 else 0.0)
    return Ts, means, stds


def plot_metric_vs_length(
    results: dict[int, dict],
    methods: list[str],
    metric: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods:
        Ts, means, stds = _collect_metric(results, method, metric)
        x = np.asarray(Ts, dtype=np.float64)
        m = np.asarray(means, dtype=np.float64)
        s = np.asarray(stds, dtype=np.float64)
        line, = ax.plot(x, m, marker="o", ms=6, label=method)
        ax.errorbar(
            x, m, yerr=s,
            color=line.get_color(), linestyle="none",
            capsize=4, capthick=1.2, alpha=0.85,
        )
    ax.set_xlabel("Number of tasks")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_task0_retention(
    results: dict[int, dict],
    methods: list[str],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods:
        Ts, means, stds = _collect_task0_retention(results, method)
        x = np.asarray(Ts, dtype=np.float64)
        m = np.asarray(means, dtype=np.float64)
        s = np.asarray(stds, dtype=np.float64)
        line, = ax.plot(x, m, marker="o", ms=6, label=method)
        ax.errorbar(
            x, m, yerr=s,
            color=line.get_color(), linestyle="none",
            capsize=4, capthick=1.2, alpha=0.85,
        )
    ax.set_xlabel("Number of tasks (T)")
    ax.set_ylabel("Task-0 ACC at end of training (R[T-1, 0])")
    ax.set_title("Task-0 retention vs task-sequence length")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _print_summary_table(
    results: dict[int, dict],
    methods: list[str],
) -> None:
    print("Cross-length summary (mean ± std across seeds):")
    print()
    header_T = sorted(results.keys())
    print(f"  {'method':<34s} " + "  ".join(f"T={T:>3d}".ljust(16) for T in header_T))
    print("  " + "-" * (35 + 18 * len(header_T)))
    print("  Aggregate ACC:")
    for method in methods:
        Ts, means, stds = _collect_metric(results, method, "average_accuracy")
        cells = [f"  {method:<34s} "]
        for T in header_T:
            try:
                idx = Ts.index(T)
                cells.append(f"{means[idx]:.3f}±{stds[idx]:.3f}".ljust(16))
            except ValueError:
                cells.append("—".ljust(16))
        print("".join(cells))
    print()
    print("  Task-0 ACC at end (R[T-1, 0]):")
    for method in methods:
        Ts, means, stds = _collect_task0_retention(results, method)
        cells = [f"  {method:<34s} "]
        for T in header_T:
            try:
                idx = Ts.index(T)
                cells.append(f"{means[idx]:.3f}±{stds[idx]:.3f}".ljust(16))
            except ValueError:
                cells.append("—".ljust(16))
        print("".join(cells))


def main() -> None:
    args = parse_args()
    if args.log_paths is not None:
        paths = list(args.log_paths)
    else:
        paths = _discover_latest_log_group(args.log_dir)
        print(f"Auto-discovered log group: {[str(p) for p in paths]}")

    results = _load_results(paths)
    if not results:
        raise SystemExit("no parseable logs found")

    # Method ordering: take it from the first log's methods_requested (or
    # methods_completed) so the plot lines render in the experiment's
    # declared order rather than alphabetical.
    first = results[min(results)]
    methods = (
        first.get("methods_completed")
        or first.get("methods_requested")
        or [m["method"] for m in first["methods"]]
    )
    methods = [m for m in methods if any(
        any(mm["method"] == m for mm in d["methods"]) for d in results.values()
    )]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_metric_vs_length(
        results, methods, metric="average_accuracy",
        title=f"ACC vs task-sequence length "
              f"(n={first['summaries'][0]['n_seeds']} seeds)",
        ylabel="Avg ACC across all seen tasks (end of training)",
        out_path=args.output_dir / "avg_accuracy_vs_task_length.png",
    )
    plot_metric_vs_length(
        results, methods, metric="average_forgetting",
        title=f"Forgetting vs task-sequence length "
              f"(n={first['summaries'][0]['n_seeds']} seeds)",
        ylabel="Avg forgetting across all seen tasks (end of training)",
        out_path=args.output_dir / "forgetting_vs_task_length.png",
    )
    plot_task0_retention(
        results, methods,
        out_path=args.output_dir / "task0_retention_vs_task_length.png",
    )
    print()
    _print_summary_table(results, methods)
    print()
    print(f"Saved plots to {args.output_dir}/")


if __name__ == "__main__":
    main()
