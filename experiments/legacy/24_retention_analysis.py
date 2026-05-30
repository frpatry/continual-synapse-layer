"""Experiment 24 — per-task retention analysis of Phase B configs.

Central argument: average ACC masks the difference between
``cs_gated_cosine_developmental`` variants. Two configs can land
within 1 pp of each other on aggregate ACC while differing by
20+ pp on Task-0 retention. This script makes that signal
visible.

Reads the Phase B JSON checkpoints written by experiment 23
(``results/logs/phase_b_validation/*_23_phase_b_T{T}.json``) for
whichever task lengths are available (T ∈ {15, 30, 50}), extracts
the last row of each per-seed accuracy matrix
(``R[T-1, k]`` for k=0..T-1 = final accuracy on each task after
all training is complete), and produces:

Plots under ``results/figures/phase_b/``:
- ``retention_curve_T{T}.png``: x = task index (0 = first learned,
  T-1 = last), y = final ACC, one line per config with ± std
  ribbon, chance-line (1 / num_classes) drawn for reference.
- ``retention_heatmap_T{T}.png`` for the longest task length
  available, picking the best config (highest Task-0 final ACC).
  Full ``(T, T)`` R matrix averaged across seeds; the diagonal
  shows "task just learned" ACC, off-diagonal shows degradation.

Metrics per (config, T) printed to stdout and persisted to
``results/analysis/phase_b_retention.json``:
- task0_final_acc                ``R[T-1, 0]``
- task_mid_final_acc             ``R[T-1, T // 2]``
- task_last_final_acc            ``R[T-1, T-1]`` (plasticity check)
- integrated_retention           ``mean(R[T-1, :])`` (= aggregate ACC)
- old_half_retention             ``mean(R[T-1, : T // 2])``
- new_half_retention             ``mean(R[T-1, T // 2 :])``
- plasticity_stability_ratio     ``new_half / old_half``
  (>1 = more plastic than stable; <1 = more stable than plastic)

Pairwise Wilcoxon on Task-0 final ACC with Bonferroni × number-of-
pairs correction, per T. Output table mirrors the format of
``statistics.format_pairwise_table``.

Run from the repo root::

    python experiments/24_retention_analysis.py

Optional ``--log-paths`` to point at specific JSON files instead
of auto-discovering the most recent shared-timestamp group.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LOG_DIR = _REPO_ROOT / "results" / "logs" / "phase_b_validation"
_DEFAULT_FIG_DIR = _REPO_ROOT / "results" / "figures" / "phase_b"
_DEFAULT_ANALYSIS_PATH = (
    _REPO_ROOT / "results" / "analysis" / "phase_b_retention.json"
)

# Match the timestamp prefix in exp-23 filenames so we can group all
# task-length runs from a single launch together.
_TS_RE = re.compile(r"^(\d+)_23_phase_b_T(\d+)\.json$")


# ---------- dataclasses ----------


@dataclass
class RetentionMetrics:
    """All retention statistics for one (config, T) cell.

    The ``per_seed`` lists are kept so downstream code (Wilcoxon,
    article tables) can re-aggregate without re-loading the raw
    accuracy matrix.
    """

    config: str
    num_tasks: int
    n_seeds: int
    task0_final_acc: float
    task0_final_acc_std: float
    task0_final_acc_per_seed: list[float]
    task_mid_final_acc: float
    task_mid_final_acc_std: float
    task_last_final_acc: float
    task_last_final_acc_std: float
    integrated_retention: float
    integrated_retention_std: float
    old_half_retention: float
    old_half_retention_std: float
    new_half_retention: float
    new_half_retention_std: float
    plasticity_stability_ratio: float
    plasticity_stability_ratio_std: float
    # Cross-seed mean and std of R[T-1, k] for every k. Used by the
    # retention-curve plot and by anyone who wants to re-derive any of
    # the above metrics post-hoc.
    final_acc_per_task_mean: list[float]
    final_acc_per_task_std: list[float]


@dataclass
class Task0Wilcoxon:
    config_a: str
    config_b: str
    n: int
    statistic: float
    p_value: float
    p_value_bonferroni: float
    significant_05: bool


# ---------- CLI ----------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--log-paths", type=Path, nargs="+", default=None,
        help="Override auto-discovery: explicit list of T-keyed JSON paths.",
    )
    p.add_argument("--log-dir", type=Path, default=_DEFAULT_LOG_DIR)
    p.add_argument("--fig-dir", type=Path, default=_DEFAULT_FIG_DIR)
    p.add_argument(
        "--analysis-path", type=Path, default=_DEFAULT_ANALYSIS_PATH,
        help="JSON sink for the full computed analysis.",
    )
    return p.parse_args()


# ---------- log discovery + loading ----------


def discover_phase_b_logs(log_dir: Path) -> dict[int, Path]:
    """Find the latest shared-timestamp set of T-keyed Phase-B JSONs.

    Returns a map ``{T: path}``. Empty when no logs exist.
    """
    if not log_dir.exists():
        return {}
    by_ts: dict[int, dict[int, Path]] = {}
    for p in log_dir.iterdir():
        match = _TS_RE.match(p.name)
        if match is None:
            continue
        ts = int(match.group(1))
        T = int(match.group(2))
        by_ts.setdefault(ts, {})[T] = p
    if not by_ts:
        return {}
    latest_ts = max(by_ts)
    return dict(sorted(by_ts[latest_ts].items()))


def load_logs(paths: list[Path]) -> dict[int, dict]:
    """Map T -> parsed payload. Skips files that fail to parse."""
    out: dict[int, dict] = {}
    for p in paths:
        try:
            d = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARN: could not load {p}: {e}")
            continue
        T = int(d.get("num_tasks", -1))
        if T <= 0:
            print(f"  WARN: log at {p} has no num_tasks; skipping.")
            continue
        out[T] = d
    return dict(sorted(out.items()))


def list_configs(payload: dict) -> list[str]:
    """Configs present in the log, in declared order (not alphabetical)."""
    ordered = (
        payload.get("configs_completed")
        or payload.get("configs_requested")
        or [m["method"] for m in payload.get("methods", [])]
    )
    available = {m["method"] for m in payload.get("methods", [])}
    return [c for c in ordered if c in available]


# ---------- per-seed final-row extraction ----------


def extract_final_acc_matrix(
    payload: dict, config: str
) -> np.ndarray:
    """Return ``(n_seeds, T)`` array of R[T-1, k] per seed for ``config``.

    Rows are seeds (in the order they were run); columns are task
    indices 0..T-1. NaNs in the underlying matrix are preserved.
    """
    method_block = next(
        (m for m in payload["methods"] if m["method"] == config), None
    )
    if method_block is None:
        return np.zeros((0, 0))
    rows: list[list[float]] = []
    T = int(payload["num_tasks"])
    for r in method_block["results"]:
        am = r["accuracy_matrix"]
        last_row = am[T - 1]
        rows.append([float("nan") if v is None else float(v) for v in last_row])
    if not rows:
        return np.zeros((0, T))
    return np.asarray(rows, dtype=np.float64)


def extract_full_accuracy_matrix(
    payload: dict, config: str
) -> np.ndarray:
    """Return ``(n_seeds, T, T)`` array of full R matrices for ``config``."""
    method_block = next(
        (m for m in payload["methods"] if m["method"] == config), None
    )
    if method_block is None:
        return np.zeros((0, 0, 0))
    T = int(payload["num_tasks"])
    stacks: list[np.ndarray] = []
    for r in method_block["results"]:
        am = r["accuracy_matrix"]
        arr = np.asarray(
            [
                [float("nan") if v is None else float(v) for v in row]
                for row in am
            ],
            dtype=np.float64,
        )
        stacks.append(arr)
    if not stacks:
        return np.zeros((0, T, T))
    return np.stack(stacks, axis=0)


# ---------- metric computation ----------


def _nanmean_std(values: np.ndarray) -> tuple[float, float]:
    """Mean and std (ddof=1) of finite entries; (NaN, 0) when empty."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan"), 0.0
    mean = float(finite.mean())
    std = float(finite.std(ddof=1)) if finite.size > 1 else 0.0
    return mean, std


def compute_retention_metrics(
    config: str, final_acc_matrix: np.ndarray
) -> RetentionMetrics:
    """Compute the seven retention metrics for one (config, T) cell.

    Args:
        config: Method/config name (passed through).
        final_acc_matrix: ``(n_seeds, T)`` array of R[T-1, :] per seed.
    """
    n_seeds, T = final_acc_matrix.shape
    mid = T // 2
    # Per-seed scalar metrics (one value per seed each).
    task0 = final_acc_matrix[:, 0]
    task_mid = final_acc_matrix[:, mid]
    task_last = final_acc_matrix[:, T - 1]
    integrated = np.nanmean(final_acc_matrix, axis=1)  # = avg ACC
    old_half = np.nanmean(final_acc_matrix[:, :mid], axis=1)
    new_half = np.nanmean(final_acc_matrix[:, mid:], axis=1)
    # Plasticity-stability ratio: guard against div-by-zero per seed.
    with np.errstate(invalid="ignore", divide="ignore"):
        plast_ratio = np.where(
            old_half > 0, new_half / old_half, np.nan
        )

    task0_m, task0_s = _nanmean_std(task0)
    mid_m, mid_s = _nanmean_std(task_mid)
    last_m, last_s = _nanmean_std(task_last)
    int_m, int_s = _nanmean_std(integrated)
    old_m, old_s = _nanmean_std(old_half)
    new_m, new_s = _nanmean_std(new_half)
    ratio_m, ratio_s = _nanmean_std(plast_ratio)

    # Cross-seed mean ± std per task index (for the retention curve).
    per_task_mean = np.nanmean(final_acc_matrix, axis=0)
    if n_seeds > 1:
        per_task_std = np.nanstd(final_acc_matrix, axis=0, ddof=1)
    else:
        per_task_std = np.zeros(T)

    return RetentionMetrics(
        config=config,
        num_tasks=T,
        n_seeds=n_seeds,
        task0_final_acc=task0_m,
        task0_final_acc_std=task0_s,
        task0_final_acc_per_seed=task0.tolist(),
        task_mid_final_acc=mid_m,
        task_mid_final_acc_std=mid_s,
        task_last_final_acc=last_m,
        task_last_final_acc_std=last_s,
        integrated_retention=int_m,
        integrated_retention_std=int_s,
        old_half_retention=old_m,
        old_half_retention_std=old_s,
        new_half_retention=new_m,
        new_half_retention_std=new_s,
        plasticity_stability_ratio=ratio_m,
        plasticity_stability_ratio_std=ratio_s,
        final_acc_per_task_mean=per_task_mean.tolist(),
        final_acc_per_task_std=per_task_std.tolist(),
    )


# ---------- Wilcoxon on Task-0 ACC ----------


def task0_pairwise_wilcoxon(
    metrics_by_config: dict[str, RetentionMetrics],
    alpha: float = 0.05,
) -> list[Task0Wilcoxon]:
    """Pairwise Wilcoxon signed-rank on per-seed Task-0 ACC with
    Bonferroni correction.

    Not using ``statistics.pairwise_wilcoxon`` because that helper
    validates the metric name against a fixed list that doesn't
    include task-specific accuracies.
    """
    from scipy.stats import wilcoxon  # type: ignore[import-untyped]

    configs = list(metrics_by_config.keys())
    if len(configs) < 2:
        return []
    n_pairs = len(configs) * (len(configs) - 1) // 2
    out: list[Task0Wilcoxon] = []
    for ca, cb in combinations(configs, 2):
        a = np.asarray(
            metrics_by_config[ca].task0_final_acc_per_seed, dtype=np.float64
        )
        b = np.asarray(
            metrics_by_config[cb].task0_final_acc_per_seed, dtype=np.float64
        )
        mask = np.isfinite(a) & np.isfinite(b)
        a = a[mask]
        b = b[mask]
        n = int(a.size)
        if n < 1 or np.allclose(a, b):
            stat = 0.0
            p = 1.0
        else:
            test = wilcoxon(a, b, zero_method="wilcox")
            stat = float(test.statistic)
            p = float(test.pvalue)
        p_corr = min(1.0, p * n_pairs)
        out.append(
            Task0Wilcoxon(
                config_a=ca, config_b=cb, n=n,
                statistic=stat, p_value=p,
                p_value_bonferroni=p_corr,
                significant_05=p_corr < alpha,
            )
        )
    return out


# ---------- plots ----------


def _line_colours(n: int) -> list[tuple[float, float, float, float]]:
    """Use the default matplotlib cycle so colours align with other
    plots in the repo (12b, 21b)."""
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]


def plot_retention_curve(
    metrics_by_config: dict[str, RetentionMetrics],
    num_tasks: int,
    chance_level: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    colours = _line_colours(len(metrics_by_config))
    for (config, m), colour in zip(metrics_by_config.items(), colours):
        x = np.arange(num_tasks)
        mean = np.asarray(m.final_acc_per_task_mean, dtype=np.float64)
        std = np.asarray(m.final_acc_per_task_std, dtype=np.float64)
        ax.plot(x, mean, marker="o", ms=4, color=colour, label=config)
        ax.fill_between(x, mean - std, mean + std, color=colour, alpha=0.18)
    ax.axhline(
        chance_level, color="grey", linewidth=0.8, alpha=0.7,
        linestyle="--", label=f"chance ({chance_level:.2f})",
    )
    ax.set_xlabel("Task index (0 = first learned, T-1 = last)")
    ax.set_ylabel(f"Final accuracy R[T-1, k] (mean ± std, n={list(metrics_by_config.values())[0].n_seeds})")
    ax.set_title(
        f"Per-task retention at end of training (T={num_tasks})"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_retention_heatmap(
    payload: dict, config: str, out_path: Path,
) -> None:
    """Plot the full ``(T, T)`` R matrix averaged across seeds."""
    import warnings

    full = extract_full_accuracy_matrix(payload, config)
    if full.size == 0:
        return
    T = int(payload["num_tasks"])
    # Upper-triangular cells where R[i, j] for j > i+1 are typically
    # NaN (never evaluated); np.nanmean on an all-NaN slice warns but
    # the masked imshow handles it correctly. Suppress the noise.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning,
                                message="Mean of empty slice")
        avg = np.nanmean(full, axis=0)  # (T, T)

    fig, ax = plt.subplots(figsize=(8, 7))
    # NaN cells (un-evaluated; e.g., zero-shot when disabled) plotted
    # as a distinct grey rather than the colormap's lowest value.
    masked = np.ma.masked_invalid(avg)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="lightgrey")
    im = ax.imshow(
        masked, origin="upper", cmap=cmap, vmin=0.0, vmax=1.0,
        aspect="auto",
    )
    ax.set_xlabel("Task evaluated (k)")
    ax.set_ylabel("Checkpoint: after training task (i)")
    ax.set_title(
        f"R[i, k] heatmap for {config} at T={T}\n"
        f"(mean across seeds, diagonal = task just learned)"
    )
    # Optionally annotate the diagonal and the bottom-left corner.
    ax.plot(
        [0, T - 1], [0, T - 1], color="white", linestyle=":", alpha=0.5,
        linewidth=1.0,
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Accuracy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------- stdout reporting ----------


def _fmt_metric(mean: float, std: float, width: int = 12) -> str:
    s = f"{mean:.3f}±{std:.3f}"
    return s.rjust(width)


def print_retention_table(
    metrics: dict[int, dict[str, RetentionMetrics]],
) -> None:
    print()
    for T in sorted(metrics.keys()):
        print(f"=== Retention metrics at T={T} ===")
        per_config = metrics[T]
        print(
            f"  {'config':<32s} "
            f"{'Task-0':>12s} {'Task-T/2':>12s} {'Task-T-1':>12s} "
            f"{'aggregate':>12s} {'old-half':>12s} {'new-half':>12s} "
            f"{'P/S ratio':>12s}"
        )
        print("  " + "-" * 128)
        for config, m in per_config.items():
            print(
                f"  {config:<32s} "
                f"{_fmt_metric(m.task0_final_acc, m.task0_final_acc_std)} "
                f"{_fmt_metric(m.task_mid_final_acc, m.task_mid_final_acc_std)} "
                f"{_fmt_metric(m.task_last_final_acc, m.task_last_final_acc_std)} "
                f"{_fmt_metric(m.integrated_retention, m.integrated_retention_std)} "
                f"{_fmt_metric(m.old_half_retention, m.old_half_retention_std)} "
                f"{_fmt_metric(m.new_half_retention, m.new_half_retention_std)} "
                f"{_fmt_metric(m.plasticity_stability_ratio, m.plasticity_stability_ratio_std)}"
            )
        print()


def print_task0_wilcoxon(
    wilcoxon_by_T: dict[int, list[Task0Wilcoxon]],
) -> None:
    print()
    for T in sorted(wilcoxon_by_T.keys()):
        print(f"=== Wilcoxon pairwise on Task-0 ACC at T={T} (Bonferroni-corrected) ===")
        ws = wilcoxon_by_T[T]
        if not ws:
            print("  (no pairs to compare — need ≥ 2 configs)")
            continue
        print(
            f"  {'config_a':<32s} {'config_b':<32s} "
            f"{'n':>3s}  {'stat':>8s}  {'p_raw':>9s}  {'p_bonf':>9s}  {'sig@0.05':>9s}"
        )
        print("  " + "-" * 116)
        for w in ws:
            sig = "SIG" if w.significant_05 else "n.s."
            print(
                f"  {w.config_a:<32s} {w.config_b:<32s} "
                f"{w.n:>3d}  {w.statistic:>8.3f}  "
                f"{w.p_value:>9.5f}  {w.p_value_bonferroni:>9.5f}  {sig:>9s}"
            )
        print()


# ---------- analysis JSON ----------


def build_analysis_json(
    metrics: dict[int, dict[str, RetentionMetrics]],
    wilcoxon_by_T: dict[int, list[Task0Wilcoxon]],
    best_config_per_T: dict[int, str],
    log_paths: dict[int, Path],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "experiment": "24_retention_analysis",
        "task_lengths": sorted(metrics.keys()),
        "source_logs": {str(T): str(p) for T, p in log_paths.items()},
        "best_config_per_T": {str(T): c for T, c in best_config_per_T.items()},
        "metrics": {},
        "wilcoxon_task0": {},
    }
    for T, per_config in metrics.items():
        out["metrics"][str(T)] = {
            cfg: asdict(m) for cfg, m in per_config.items()
        }
    for T, ws in wilcoxon_by_T.items():
        out["wilcoxon_task0"][str(T)] = [asdict(w) for w in ws]
    return out


# ---------- main ----------


def main() -> None:
    args = parse_args()

    if args.log_paths is not None:
        paths = list(args.log_paths)
        logs = load_logs(paths)
    else:
        discovered = discover_phase_b_logs(args.log_dir)
        if not discovered:
            raise SystemExit(
                f"No Phase B logs found under {args.log_dir}. Run exp 23 "
                f"first, or pass --log-paths explicitly."
            )
        print("Auto-discovered Phase B logs (latest shared timestamp):")
        for T, p in discovered.items():
            print(f"  T={T:>3d}: {p}")
        paths = list(discovered.values())
        logs = load_logs(paths)

    if not logs:
        raise SystemExit("No parseable Phase B logs found.")

    # Configs are identical across T (Phase B holds them constant); take
    # them from the first log we found.
    first = logs[min(logs)]
    configs = list_configs(first)
    if not configs:
        raise SystemExit(
            "No config methods in the first Phase B log; can't analyse."
        )

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.analysis_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Per-T metric computation ----
    metrics_by_T: dict[int, dict[str, RetentionMetrics]] = {}
    for T, payload in logs.items():
        per_config: dict[str, RetentionMetrics] = {}
        for cfg in list_configs(payload):
            final_acc = extract_final_acc_matrix(payload, cfg)
            if final_acc.size == 0:
                print(f"  WARN: no results for {cfg} at T={T}")
                continue
            per_config[cfg] = compute_retention_metrics(cfg, final_acc)
        metrics_by_T[T] = per_config

    # ---- Task-0 Wilcoxon per T ----
    wilcoxon_by_T: dict[int, list[Task0Wilcoxon]] = {
        T: task0_pairwise_wilcoxon(per_config)
        for T, per_config in metrics_by_T.items()
    }

    # ---- Best config per T (by Task-0 final ACC) ----
    best_config_per_T: dict[int, str] = {}
    for T, per_config in metrics_by_T.items():
        if not per_config:
            continue
        best = max(per_config.values(), key=lambda m: m.task0_final_acc)
        best_config_per_T[T] = best.config

    # ---- Plots ----
    # Retention curves per T.
    # The chance level is 1 / num_classes; PermutedMNIST has 10 classes.
    # Defensive: probe the first method's task_names length if available.
    chance = 0.1
    for T, per_config in metrics_by_T.items():
        if not per_config:
            continue
        plot_retention_curve(
            per_config, num_tasks=T, chance_level=chance,
            out_path=args.fig_dir / f"retention_curve_T{T}.png",
        )
        print(f"  Wrote {args.fig_dir / f'retention_curve_T{T}.png'}")
    # Heatmap: longest T available, best config.
    if logs:
        longest_T = max(logs.keys())
        best_cfg = best_config_per_T.get(longest_T)
        if best_cfg is not None:
            heatmap_path = args.fig_dir / f"retention_heatmap_T{longest_T}.png"
            plot_retention_heatmap(logs[longest_T], best_cfg, heatmap_path)
            print(f"  Wrote {heatmap_path}  (best config at T={longest_T}: {best_cfg})")

    # ---- Stdout report ----
    print_retention_table(metrics_by_T)
    print_task0_wilcoxon(wilcoxon_by_T)

    # ---- Persist analysis JSON ----
    log_paths_map = {T: p for T, p in zip(sorted(logs.keys()), paths)}
    # Re-pair properly: walk discovered/explicit paths and match by T.
    log_paths_map = {}
    for p in paths:
        try:
            d = json.loads(p.read_text())
            T = int(d.get("num_tasks", -1))
            if T > 0:
                log_paths_map[T] = p
        except Exception:
            continue
    analysis = build_analysis_json(
        metrics_by_T, wilcoxon_by_T, best_config_per_T, log_paths_map,
    )
    args.analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True, default=str))
    print(f"\nSaved analysis JSON to {args.analysis_path}")


if __name__ == "__main__":
    main()
