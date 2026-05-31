"""Phase 6i' — S-level calibration sweep.

Phase 6i shipped the S-layer architecture (4 modules, 34 tests) but
the default parameters produced exactly ONE S of size 2 across the
entire M=5 protocol — parametrically inert.

The math diagnosed three knobs to investigate:

  pair_candidacy steady state    = boost · coact_PP / (1 - decay)
                                  = 0.1 · (0.5·0.5) / 0.05 = 0.5
  → exactly at theta_s_emergence=0.5 (only lucky pairs cross)

  p_to_s_candidacy steady state  = boost · coact_PS / (1 - decay)
                                  = 0.1 · (0.5·0.2) / 0.05 = 0.2
  → below theta_s_growth=0.3 (S never grows)

This sweep tests configurations in order, isolating one knob at a
time, and stops at the first config to reach 5/5/5 (alive ≥ 2 P AND
recall ≥ 40 % for all 5 patterns) so the *critical* knob — the one
that flips the verdict — can be identified.

Configs (in order, each layered on top of the previous):
  6i'-A  baseline (Phase 6i defaults; already known 4/3)
  6i'-B  + theta_s_emergence 0.5 → 0.3
  6i'-C  + theta_s_growth   0.3 → 0.15
  6i'-D  + s_pass_boost     0.1 → 0.2
  6i'-E  + gamma_s_to_p     1.0 → 1.5
  6i'-F  + gamma_s_to_p     1.5 → 2.0
  6i'-G  + s_pass_decay     0.95 → 0.90

Protocol per config: Phase 6c interleaved (training K=80 + consolidation
K=1500) across M=5 disjoint patterns. Fresh Substrate each time; same
seed; same patterns. Phase 6c regime (rho_floor=0.3, k_protect=0).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments" / "substrate"))

from substrate.substrate import Substrate  # noqa: E402

from phase_5b_bridge_at_scale import (  # noqa: E402
    classify_p_entities,
    define_M_patterns,
    summarize_classification,
)
from phase_6c_consolidation_mode import (  # noqa: E402
    consolidation_mode_phase,
    recall_mode_test,
    training_mode_phase,
)


# ---------------------------------------------------------------------------
# Config grid
# ---------------------------------------------------------------------------


@dataclass
class SweepConfig:
    name: str
    theta_s_emergence: float = 0.5
    theta_s_growth: float = 0.3
    s_pass_boost: float = 0.1
    s_pass_decay: float = 0.95
    gamma_s_to_p: float = 1.0
    notes: str = ""


CONFIGS = [
    SweepConfig(
        name="6i'-A",
        notes="baseline Phase 6i defaults — known 4/3, 1 inert S",
    ),
    SweepConfig(
        name="6i'-B",
        theta_s_emergence=0.3,
        notes="lower emergence threshold — more pairs cross",
    ),
    SweepConfig(
        name="6i'-C",
        theta_s_emergence=0.3,
        theta_s_growth=0.15,
        notes="+ lower growth threshold — S can grow beyond size=2",
    ),
    SweepConfig(
        name="6i'-D",
        theta_s_emergence=0.3,
        theta_s_growth=0.15,
        s_pass_boost=0.2,
        notes="+ higher pass boost — candidacy builds faster",
    ),
    SweepConfig(
        name="6i'-E",
        theta_s_emergence=0.3,
        theta_s_growth=0.15,
        s_pass_boost=0.2,
        gamma_s_to_p=1.5,
        notes="+ stronger S→P feedback",
    ),
    SweepConfig(
        name="6i'-F",
        theta_s_emergence=0.3,
        theta_s_growth=0.15,
        s_pass_boost=0.2,
        gamma_s_to_p=2.0,
        notes="+ even stronger S→P feedback",
    ),
    SweepConfig(
        name="6i'-G",
        theta_s_emergence=0.3,
        theta_s_growth=0.15,
        s_pass_boost=0.2,
        s_pass_decay=0.90,
        gamma_s_to_p=2.0,
        notes="+ faster decay → higher steady state via shorter time-constant",
    ),
]


# ---------------------------------------------------------------------------
# Single-config runner
# ---------------------------------------------------------------------------


def n_alive(counts: list[int]) -> int:
    return sum(1 for c in counts if c >= 2)


def n_recall(recalls: list[float]) -> int:
    return sum(1 for r in recalls if r >= 0.4)


def run_one_config(
    cfg: SweepConfig,
    n_neurons: int = 500,
    M: int = 5,
    pattern_size: int = 10,
    K_per_pattern: int = 80,
    K_consolidate: int = 1500,
    seed: int = 42,
) -> dict[str, Any]:
    """Fresh substrate, full Phase 6c protocol, returns metrics + trajectory."""
    substrate = Substrate(
        n_neurons=n_neurons,
        k_connectivity=30,
        eta=0.01,
        lambda_decay=0.001,
        threshold=0.3,
        sparsity_target=0.05,
        # Phase 2a-c standard.
        theta_emergence=0.5,
        n_min_passes=3,
        alpha_n_to_p=1.0,
        p_threshold=0.2,
        p_sparsity_target=0.5,
        min_coactivation_to_create_pp=0.01,
        eta_pp=0.05,
        gamma_p_to_n=1.0,
        enable_feedback_p_to_n=True,
        starting_age=0.0,
        # Phase 6c regime (overrides for parity with Phase 6c).
        rho_floor=0.3,
        k_protect=0,
        # The sweep knobs:
        theta_s_emergence=cfg.theta_s_emergence,
        theta_s_growth=cfg.theta_s_growth,
        s_pass_boost=cfg.s_pass_boost,
        s_pass_decay=cfg.s_pass_decay,
        gamma_s_to_p=cfg.gamma_s_to_p,
        # Other S defaults left at substrate defaults.
        alpha_p_to_s=0.3,
        s_threshold=0.2,
        s_sparsity_target=0.20,
        s_min_active=1,
        s_max_active=3,
        eta_s=0.005,
        lambda_s_decay=0.001,
        s_viability_threshold=0.1,
        seed=seed,
    )

    patterns = define_M_patterns(n_neurons, M, pattern_size, seed=0)

    # Track "first step" trajectory diagnostics.
    first_step_two_s: Optional[int] = None
    first_step_size_three: Optional[int] = None

    history: list[dict[str, Any]] = []

    # We need step-level tracking for "first step" — but we already run
    # training_mode_phase / consolidation_mode_phase which loop internally.
    # Approximate by checking at end of each stage.
    for stage in range(M):
        training_mode_phase(substrate, patterns[stage], K=K_per_pattern)
        consolidation_mode_phase(substrate, K_steps=K_consolidate)

        # Per-stage checkpoints.
        classification = classify_p_entities(substrate.p_entities, patterns)
        summary = summarize_classification(classification, M)
        p_counts = [summary[f"P_pattern_{i}"] for i in range(M)]
        recalls = [recall_mode_test(substrate, p) for p in patterns]
        bridges = summary["P_bridges_total"]
        s_count_now = substrate.s_count()
        s_sizes = [s.size() for s in substrate.s_entities.values()]
        s_active = sum(
            1 for s in substrate.s_entities.values() if s.activation > 0.0
        )
        max_s_size = max(s_sizes) if s_sizes else 0
        mean_s_size = float(np.mean(s_sizes)) if s_sizes else 0.0

        if first_step_two_s is None and s_count_now >= 2:
            first_step_two_s = substrate.step_count
        if first_step_size_three is None and max_s_size >= 3:
            first_step_size_three = substrate.step_count

        history.append({
            "stage": stage,
            "step_count": substrate.step_count,
            "P_counts": p_counts,
            "recalls": recalls,
            "bridges": bridges,
            "s_count": s_count_now,
            "s_active": s_active,
            "mean_s_size": mean_s_size,
            "max_s_size": max_s_size,
        })

    final = history[-1]
    p_counts = final["P_counts"]
    recalls = final["recalls"]
    n_alive_val = n_alive(p_counts)
    n_recall_val = n_recall(recalls)

    s_sizes_final = [s.size() for s in substrate.s_entities.values()]
    s_active_final = sum(
        1 for s in substrate.s_entities.values() if s.activation > 0.0
    )

    return {
        "config_name": cfg.name,
        "config": {
            "theta_s_emergence": cfg.theta_s_emergence,
            "theta_s_growth": cfg.theta_s_growth,
            "s_pass_boost": cfg.s_pass_boost,
            "s_pass_decay": cfg.s_pass_decay,
            "gamma_s_to_p": cfg.gamma_s_to_p,
        },
        "notes": cfg.notes,
        "history": history,
        "P_counts_final": p_counts,
        "recalls_final": recalls,
        "n_alive": n_alive_val,
        "n_recall": n_recall_val,
        "bridges_final": final["bridges"],
        "s_count_final": substrate.s_count(),
        "s_active_final": s_active_final,
        "mean_s_size_final": (
            float(np.mean(s_sizes_final)) if s_sizes_final else 0.0
        ),
        "max_s_size_final": max(s_sizes_final) if s_sizes_final else 0,
        "first_step_two_s": first_step_two_s,
        "first_step_size_three": first_step_size_three,
    }


# ---------------------------------------------------------------------------
# Main: run sweep, stop at 5/5/5
# ---------------------------------------------------------------------------


def main() -> int:
    print("=== Phase 6i' — S-level calibration sweep ===\n")
    M = 5
    sweep_results: list[dict[str, Any]] = []
    stopping_config: Optional[str] = None
    sweep_log: list[str] = []

    for cfg in CONFIGS:
        line = f"\n--- Running {cfg.name}: {cfg.notes} ---"
        print(line)
        sweep_log.append(line)
        line = (
            f"    knobs: theta_emerge={cfg.theta_s_emergence}, "
            f"theta_growth={cfg.theta_s_growth}, "
            f"boost={cfg.s_pass_boost}, "
            f"decay={cfg.s_pass_decay}, "
            f"gamma_s2p={cfg.gamma_s_to_p}"
        )
        print(line)
        sweep_log.append(line)

        result = run_one_config(cfg)
        sweep_results.append(result)

        summary_line = (
            f"  → {result['n_alive']}/{result['n_recall']} "
            f"(alive/recall), S count={result['s_count_final']}, "
            f"mean S size={result['mean_s_size_final']:.2f}, "
            f"max S size={result['max_s_size_final']}, "
            f"first-2-S step={result['first_step_two_s']}, "
            f"first-size3 step={result['first_step_size_three']}"
        )
        print(summary_line)
        sweep_log.append(summary_line)

        if result["n_alive"] == M and result["n_recall"] == M:
            stopping_config = cfg.name
            line = (
                f"\n*** {cfg.name} REACHED 5/5/5 — stopping sweep early ***"
            )
            print(line)
            sweep_log.append(line)
            break

    # ---- Analysis ----
    print("\n=== Sweep summary ===")
    print(f"{'config':<8s} {'alive':>6s} {'recall':>7s} {'S_cnt':>6s} "
          f"{'mean_sz':>8s} {'max_sz':>7s}  verdict")
    print("-" * 75)
    for r in sweep_results:
        alive = r["n_alive"]
        recall = r["n_recall"]
        if alive == M and recall == M:
            verdict = "PASS"
        elif alive == M and recall == M - 1:
            verdict = "STRONG_PARTIAL"
        elif alive >= 4 and recall >= 4:
            verdict = "MATCHES_6C"
        elif alive >= 4 or recall >= 4:
            verdict = "PARTIAL"
        else:
            verdict = "WORSE_THAN_6C"
        print(f"{r['config_name']:<8s} {alive:>6d} {recall:>7d} "
              f"{r['s_count_final']:>6d} {r['mean_s_size_final']:>8.2f} "
              f"{r['max_s_size_final']:>7d}  {verdict}")
    print()

    # ---- Critical-knob analysis ----
    print("=== Critical-knob identification ===")
    critical_knob: Optional[str] = None
    pivot_lines = []
    if stopping_config is not None:
        # Find which knob differs from the prior config that fell short.
        stopping_idx = next(
            i for i, r in enumerate(sweep_results)
            if r["config_name"] == stopping_config
        )
        if stopping_idx == 0:
            critical_knob = "(baseline already passed — no knob changed)"
        else:
            prior = sweep_results[stopping_idx - 1]["config"]
            curr = sweep_results[stopping_idx]["config"]
            diffs = [k for k in curr if curr[k] != prior[k]]
            critical_knob = ", ".join(diffs)
        pivot_lines.append(
            f"Critical knob (pushed verdict to PASS): {critical_knob}"
        )
    else:
        # No PASS — best result and what improved over baseline.
        best = max(sweep_results,
                   key=lambda r: (r["n_alive"] + r["n_recall"],
                                  r["s_count_final"]))
        pivot_lines.append(
            f"No config reached 5/5/5. Best: {best['config_name']} at "
            f"{best['n_alive']}/{best['n_recall']} with "
            f"{best['s_count_final']} S (mean size "
            f"{best['mean_s_size_final']:.2f})"
        )
    for line in pivot_lines:
        print(line)
        sweep_log.append(line)
    print()

    # ---- Persist ----
    results_dir = _REPO_ROOT / "results" / "substrate" / "phase_6i_prime"
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "results.json").open("w") as f:
        json.dump(
            {
                "sweep_configs": [
                    {
                        "name": cfg.name,
                        "knobs": {
                            "theta_s_emergence": cfg.theta_s_emergence,
                            "theta_s_growth": cfg.theta_s_growth,
                            "s_pass_boost": cfg.s_pass_boost,
                            "s_pass_decay": cfg.s_pass_decay,
                            "gamma_s_to_p": cfg.gamma_s_to_p,
                        },
                        "notes": cfg.notes,
                    }
                    for cfg in CONFIGS[:len(sweep_results)]
                ],
                "results": sweep_results,
                "stopping_config": stopping_config,
                "critical_knob": critical_knob,
            },
            f,
            indent=2,
        )
    with (results_dir / "sweep_log.txt").open("w") as f:
        f.write("\n".join(sweep_log))

    # ---- Plot 1: alive/recall comparison ----
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    names = [r["config_name"] for r in sweep_results]
    alive_vals = [r["n_alive"] for r in sweep_results]
    recall_vals = [r["n_recall"] for r in sweep_results]
    x = np.arange(len(names))
    width = 0.4
    axes[0].bar(x - width / 2, alive_vals, width,
                color="tab:blue", label="alive (≥2 P)")
    axes[0].bar(x + width / 2, recall_vals, width,
                color="tab:green", label="recall (≥40%)")
    axes[0].axhline(M, color="black", linestyle="--", alpha=0.4,
                    label=f"target ({M}/{M})")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].set_ylabel("count / 5")
    axes[0].set_title("Phase 6i' sweep — alive/recall per config")
    axes[0].set_ylim(0, M + 0.5)
    axes[0].grid(alpha=0.3, axis="y")
    axes[0].legend()
    for i, (a, r) in enumerate(zip(alive_vals, recall_vals)):
        axes[0].text(i - width / 2, a + 0.05, str(a),
                     ha="center", va="bottom", fontsize=9)
        axes[0].text(i + width / 2, r + 0.05, str(r),
                     ha="center", va="bottom", fontsize=9)

    # ---- Plot 2: S diagnostics per config ----
    s_count_vals = [r["s_count_final"] for r in sweep_results]
    mean_size_vals = [r["mean_s_size_final"] for r in sweep_results]
    max_size_vals = [r["max_s_size_final"] for r in sweep_results]
    axes[1].bar(x - width / 2, s_count_vals, width / 2,
                color="tab:purple", label="S count")
    axes[1].bar(x, mean_size_vals, width / 2,
                color="tab:orange", label="mean S size")
    axes[1].bar(x + width / 2, max_size_vals, width / 2,
                color="tab:red", label="max S size")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].set_ylabel("count / size")
    axes[1].set_title("Phase 6i' sweep — S-layer diagnostics per config")
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(results_dir / "comparison.png", dpi=120)
    plt.close(fig)

    # ---- Plot 3: per-stage S trajectory for each config ----
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10.colors[:len(sweep_results)]
    for color, r in zip(colors, sweep_results):
        stages = [h["stage"] for h in r["history"]]
        s_counts = [h["s_count"] for h in r["history"]]
        ax.plot(stages, s_counts, "o-", color=color, label=r["config_name"])
    ax.set(xlabel="stage", ylabel="S count",
           title="S-entity count trajectory per config")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "per_stage_diagnostics.png", dpi=120)
    plt.close(fig)

    print(f"Results → {results_dir}")

    if stopping_config:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
