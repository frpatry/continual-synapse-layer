"""Per-class per-feature distribution analysis for Phase 2h.

The Phase 2d.2 real-Qwen validation showed a 47-point POST
accuracy gap between synthetic and real distributions. To close
the gap data-drivenly, we need to:

1. Measure the empirical feature distributions per epistemic
   class in the real-Qwen validation dump
   (``results/agi/phase_2_validation_raw.jsonl``).
2. Measure the same distributions on the *current* synthetic
   generators.
3. Rank ``(status, feature)`` pairs by how badly the two differ.
4. Update the most-drifted generator distributions in
   :mod:`agi.metacognition.data_generation` until the ranking
   collapses.

This module owns steps 1-3. The recalibration in step 4 is a
manual edit to ``data_generation.py``, informed by the drift
report this module produces.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import ks_2samp


@dataclass
class FeatureStats:
    """Summary statistics for one (class, feature) sample."""

    mean: float
    std: float
    median: float
    q25: float
    q75: float
    min: float
    max: float
    n_samples: int

    @classmethod
    def from_values(cls, values: list[float]) -> "FeatureStats":
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
        return cls(
            mean=float(arr.mean()),
            std=float(arr.std()),
            median=float(np.median(arr)),
            q25=float(np.quantile(arr, 0.25)),
            q75=float(np.quantile(arr, 0.75)),
            min=float(arr.min()),
            max=float(arr.max()),
            n_samples=int(arr.size),
        )


# ----------------------------------------------------------------------
# Stats over real-Qwen validation dump
# ----------------------------------------------------------------------

def compute_stats_from_raw_jsonl(
    path: Path,
    status_field: str = "expected_status",
) -> dict[str, dict[str, FeatureStats]]:
    """Read the Phase 2d.2 validation dump and compute per-class
    per-feature statistics on the *real* Qwen features.

    The dump's per-case records have the four feature dicts
    nested under ``memory_features`` / ``query_features`` /
    ``generation_features`` / ``alignment_features``. We flatten
    them into a single feature dict that mirrors the shape the
    synthetic generators produce.
    """
    by_status: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    by_status_values_raw: dict = defaultdict(lambda: defaultdict(list))
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if "error" in data:
                continue
            status = data[status_field]
            for group in (
                "memory_features",
                "query_features",
                "generation_features",
                "alignment_features",
            ):
                for fname, fval in data.get(group, {}).items():
                    by_status_values_raw[status][fname].append(float(fval))

    out: dict[str, dict[str, FeatureStats]] = {}
    for status, feats in by_status_values_raw.items():
        out[status] = {
            fname: FeatureStats.from_values(values)
            for fname, values in feats.items()
        }
    return out


# ----------------------------------------------------------------------
# Stats over current synthetic generators
# ----------------------------------------------------------------------

def compute_stats_from_generator(
    n_per_class: int = 1000,
    seed: int = 42,
    classes: Iterable[str] | None = None,
) -> dict[str, dict[str, FeatureStats]]:
    """Generate ``n_per_class`` examples per class and compute
    per-class per-feature stats. Mirrors ``compute_stats_from_raw_jsonl``
    output shape so the two can be diffed directly."""
    from .data_generation import POST_CLASSES, SyntheticDataGenerator

    cls_list = list(classes) if classes is not None else list(POST_CLASSES)
    gen = SyntheticDataGenerator(seed=seed)

    by_status: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for status in cls_list:
        method_name = f"generate_{status}_example"
        method = getattr(gen, method_name)
        for _ in range(n_per_class):
            ex = method()
            for fname, fval in ex.features.items():
                by_status[status][fname].append(float(fval))

    return {
        status: {
            fname: FeatureStats.from_values(values)
            for fname, values in feats.items()
        }
        for status, feats in by_status.items()
    }


# ----------------------------------------------------------------------
# Drift metrics
# ----------------------------------------------------------------------

def _raw_values_from_jsonl(
    path: Path, status_field: str = "expected_status",
) -> dict[str, dict[str, list[float]]]:
    """Same loader as ``compute_stats_from_raw_jsonl`` but returns
    raw value lists (needed for the KS test)."""
    by_status: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if "error" in data:
                continue
            status = data[status_field]
            for group in (
                "memory_features",
                "query_features",
                "generation_features",
                "alignment_features",
            ):
                for fname, fval in data.get(group, {}).items():
                    by_status[status][fname].append(float(fval))
    return by_status


def _raw_values_from_generator(
    n_per_class: int = 1000,
    seed: int = 42,
    classes: Iterable[str] | None = None,
) -> dict[str, dict[str, list[float]]]:
    from .data_generation import POST_CLASSES, SyntheticDataGenerator

    cls_list = list(classes) if classes is not None else list(POST_CLASSES)
    gen = SyntheticDataGenerator(seed=seed)
    by_status: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for status in cls_list:
        method = getattr(gen, f"generate_{status}_example")
        for _ in range(n_per_class):
            ex = method()
            for fname, fval in ex.features.items():
                by_status[status][fname].append(float(fval))
    return by_status


def compute_drift_metrics(
    real_stats: dict[str, dict[str, FeatureStats]],
    synthetic_stats: dict[str, dict[str, FeatureStats]],
    real_values: dict[str, dict[str, list[float]]] | None = None,
    synthetic_values: dict[str, dict[str, list[float]]] | None = None,
) -> dict[tuple[str, str], dict[str, float]]:
    """Per-(status, feature) drift metrics.

    Returns a dict keyed by ``(status, feature)`` with:

    - ``mean_diff``                  : real_mean − syn_mean
    - ``mean_diff_normalized``       : ``mean_diff / (syn_std + ε)``
    - ``std_ratio``                  : ``real_std / (syn_std + ε)``
    - ``median_diff``                : real_median − syn_median
    - ``ks_stat`` / ``ks_pvalue``    : two-sample KS statistic + p
      (only when both ``real_values`` and ``synthetic_values`` are
      provided — otherwise ``NaN``)
    - ``real_n``, ``synthetic_n``    : sample counts
    """
    drift: dict[tuple[str, str], dict[str, float]] = {}
    for status, feats in real_stats.items():
        if status not in synthetic_stats:
            continue
        for fname, real in feats.items():
            if fname not in synthetic_stats[status]:
                continue
            syn = synthetic_stats[status][fname]
            entry: dict[str, float] = {
                "mean_diff": real.mean - syn.mean,
                "mean_diff_normalized": (real.mean - syn.mean) / (syn.std + 1e-8),
                "std_ratio": real.std / (syn.std + 1e-8),
                "median_diff": real.median - syn.median,
                "real_mean": real.mean,
                "real_std": real.std,
                "syn_mean": syn.mean,
                "syn_std": syn.std,
                "real_n": real.n_samples,
                "synthetic_n": syn.n_samples,
                "ks_stat": float("nan"),
                "ks_pvalue": float("nan"),
            }
            if (
                real_values is not None
                and synthetic_values is not None
                and status in real_values
                and fname in real_values[status]
                and status in synthetic_values
                and fname in synthetic_values[status]
            ):
                ks = ks_2samp(
                    real_values[status][fname],
                    synthetic_values[status][fname],
                )
                entry["ks_stat"] = float(ks.statistic)
                entry["ks_pvalue"] = float(ks.pvalue)
            drift[(status, fname)] = entry
    return drift


def rank_drift_by_severity(
    drift: dict[tuple[str, str], dict[str, float]],
) -> list[tuple[str, str, float, dict[str, float]]]:
    """Sort ``(status, feature)`` pairs by absolute normalised
    mean-diff (highest first). Ties broken by KS statistic."""
    ranked: list[tuple[str, str, float, dict[str, float]]] = []
    for (status, fname), metrics in drift.items():
        severity = abs(metrics["mean_diff_normalized"])
        ranked.append((status, fname, float(severity), metrics))
    # Secondary key: KS stat (only if finite).
    def _key(row: tuple[str, str, float, dict[str, float]]) -> tuple[float, float]:
        ks = row[3].get("ks_stat", 0.0)
        if not np.isfinite(ks):
            ks = 0.0
        return (-row[2], -float(ks))
    ranked.sort(key=_key)
    return ranked


# ----------------------------------------------------------------------
# Markdown report
# ----------------------------------------------------------------------

def generate_drift_report(
    real_stats: dict[str, dict[str, FeatureStats]],
    synthetic_stats: dict[str, dict[str, FeatureStats]],
    drift: dict[tuple[str, str], dict[str, float]],
    ranked: list[tuple[str, str, float, dict[str, float]]],
    output_path: Path,
    *,
    top_n: int = 20,
) -> None:
    """Write the drift report to ``output_path``."""
    md: list[str] = []
    md.append("# Phase 2h — Distribution Drift Report")
    md.append("")
    md.append(
        "Compares per-class per-feature statistics between the "
        "**real-Qwen** validation dump "
        "(`results/agi/phase_2_validation_raw.jsonl`, 100 cases × ~17 features) "
        "and the **current** synthetic generators in "
        "`src/agi/metacognition/data_generation.py` (1000 samples / class)."
    )
    md.append("")
    md.append(
        "*Severity* = `|mean_diff / syn_std|`. Higher means the real "
        "mean is further from the synthetic distribution's centre, "
        "measured in synthetic-standard-deviation units. A severity "
        "above ~1.0 means the synthetic generator is likely to "
        "produce examples that look out-of-distribution to a model "
        "trained on it."
    )
    md.append("")

    md.append(f"## Top {top_n} most-drifted (status, feature) pairs")
    md.append("")
    md.append(
        "| status | feature | real mean ± std | syn mean ± std | "
        "Δmean | normΔ (severity) | KS |"
    )
    md.append("|---|---|---|---|---:|---:|---:|")
    for status, fname, severity, m in ranked[:top_n]:
        ks_str = (
            f"{m['ks_stat']:.3f}" if np.isfinite(m.get("ks_stat", float("nan")))
            else "—"
        )
        md.append(
            f"| {status} | `{fname}` | "
            f"{m['real_mean']:.3f} ± {m['real_std']:.3f} | "
            f"{m['syn_mean']:.3f} ± {m['syn_std']:.3f} | "
            f"{m['mean_diff']:+.3f} | {severity:.2f} | {ks_str} |"
        )
    md.append("")

    md.append("## Per-class feature snapshots")
    md.append("")
    statuses = sorted(real_stats.keys())
    for status in statuses:
        md.append(f"### `{status}` — real ({list(real_stats[status].values())[0].n_samples} samples) vs synthetic")
        md.append("")
        md.append("| feature | real mean | real std | syn mean | syn std | normΔ |")
        md.append("|---|---:|---:|---:|---:|---:|")
        # Sort features by severity within this status.
        rows = []
        for fname, real in real_stats[status].items():
            if fname not in synthetic_stats.get(status, {}):
                continue
            syn = synthetic_stats[status][fname]
            d = drift.get((status, fname))
            if d is None:
                continue
            rows.append((status, fname, abs(d["mean_diff_normalized"]), real, syn, d))
        rows.sort(key=lambda r: -r[2])
        for _s, fname, _sev, real, syn, d in rows:
            md.append(
                f"| `{fname}` | {real.mean:.3f} | {real.std:.3f} | "
                f"{syn.mean:.3f} | {syn.std:.3f} | "
                f"{d['mean_diff_normalized']:+.2f} |"
            )
        md.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(md))


__all__ = [
    "FeatureStats",
    "compute_drift_metrics",
    "compute_stats_from_generator",
    "compute_stats_from_raw_jsonl",
    "generate_drift_report",
    "rank_drift_by_severity",
]


def _raw_collect_helpers():
    """Re-exported so the CLI can pull both stats and raw values
    in one place. Implementation: load JSONL once, return both."""
    return _raw_values_from_jsonl, _raw_values_from_generator
