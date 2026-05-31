"""Phase 6i — S-level (third ontological tier) test at M=5.

After 10 phases ceiling'd at 4/4 (Phase 6c), the empirical case for a
third ontological tier was complete. Phase 6i implements S entities
(schemas), the SPassTracker (recursive emergence at P), s_dynamics
(propagation + adaptive k-WTA), and s_to_p_feedback (top-down boost).

Test: same Phase 6c-style protocol (interleaved cadence, the best
balanced of the cadence experiments) at M=5 disjoint patterns.
Substrate default rho_floor stays at 0.7 (the Phase 6f default
that's now in the codebase), but we explicitly override to 0.3
(Phase 6c's regime) and disable k_protect (to match 6c exactly).

The hypothesis being tested: does adding S+S→P feedback push the
substrate over the 5/5 line at M=5 that no parameter combination
ever crossed?

Verdict:
  PASS              5/5 alive AND 5/5 recall ≥ 40 %
                    → S-level resolves the scaling. Architecture
                    complete at M=5 with three tiers.
  STRONG_PARTIAL    5/5 alive AND 4/5 recall (or vice versa)
                    → S-level helps; minor refinement may close it.
  BETTER_THAN_6C    strictly better than 4/4 on at least one axis
                    → real progress, more work to do.
  MATCHES_6C        4/4 — no improvement over 6c.
  WORSE             worse than 6c.

Outputs:
  results/substrate/phase_6i/phase_6i_results.{png,json}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

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
# 11-way baseline summary
# ---------------------------------------------------------------------------

BASELINES = {
    "5b":  ([0, 3, 0, 4, 2], [0.0, 0.20, 0.40, 0.80, 0.80]),
    "6a":  ([2, 5, 1, 5, 0], [0.40, 0.20, 0.60, 1.00, 0.20]),
    "6b":  ([2, 3, 2, 4, 2], [0.0, 0.20, 0.40, 0.80, 0.80]),
    "6c":  ([2, 3, 2, 4, 1], [0.40, 0.20, 0.40, 0.80, 0.80]),
    "6d":  ([2, 3, 2, 4, 2], [0.0, 0.20, 0.40, 0.80, 0.80]),
    "6e":  ([0, 0, 0, 0, 0], [0.0, 0.0, 0.0, 0.0, 0.0]),
    "6e'": ([0, 5, 1, 4, 0], [0.20, 0.40, 0.40, 0.80, 0.40]),
    "6f":  ([2, 3, 1, 4, 2], [0.20, 0.20, 0.40, 0.80, 0.80]),
    "6g":  ([11, 27, 16, 0, 0], [0.50, 0.80, 0.80, 0.40, 0.0]),
    "6h":  ([2, 3, 1, 4, 1], [0.40, 0.20, 0.40, 0.80, 0.80]),
}


def n_alive(counts: list[int]) -> int:
    return sum(1 for c in counts if c >= 2)


def n_recall(recalls: list[float]) -> int:
    return sum(1 for r in recalls if r >= 0.4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=== Phase 6i: S-level test at M=5 ===\n")
    n_neurons = 500
    M = 5
    pattern_size = 10
    K_per_pattern = 80
    K_consolidate = 1500

    # Note: substrate ctor reuses ``n_min_passes`` for BOTH N→P and
    # P→S emergence pass thresholds.
    substrate = Substrate(
        n_neurons=n_neurons,
        k_connectivity=30,
        eta=0.01,
        lambda_decay=0.001,
        threshold=0.3,
        sparsity_target=0.05,
        # Phase 2a/b/c standard:
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
        # Phase 6c regime (override Bio-Inspired defaults):
        rho_floor=0.3,
        k_protect=0,           # disable Phase 6f protection to match 6c
        # Phase 6i S-level (explicit here for visibility of the new knobs):
        theta_s_emergence=0.5,
        theta_s_growth=0.3,
        alpha_p_to_s=0.3,
        s_threshold=0.2,
        s_sparsity_target=0.20,
        s_min_active=1,
        s_max_active=3,
        gamma_s_to_p=1.0,
        eta_s=0.005,
        lambda_s_decay=0.001,
        s_viability_threshold=0.1,
        seed=42,
    )

    patterns = define_M_patterns(n_neurons, M, pattern_size, seed=0)
    print(f"M={M} patterns, N={n_neurons}, rho_floor=0.3, k_protect=0")
    print(f"S-level: alpha_p_to_s=0.3, gamma_s_to_p=1.0, "
          f"s_sparsity=[{substrate.s_min_active}, {substrate.s_max_active}]")
    print(f"Cadence: Phase 6c interleaved "
          f"(K_train={K_per_pattern}, K_consolidate={K_consolidate})\n")

    history: list[dict[str, Any]] = []

    for stage in range(M):
        print(f"--- Stage {stage}: train (feedback OFF) ---")
        training_mode_phase(substrate, patterns[stage], K=K_per_pattern)

        print(f"--- Stage {stage}: consolidate ({K_consolidate} steps) ---")
        consolidation_mode_phase(substrate, K_steps=K_consolidate)

        classification = classify_p_entities(substrate.p_entities, patterns)
        summary = summarize_classification(classification, M)
        p_counts = [summary[f"P_pattern_{i}"] for i in range(M)]
        recalls = [recall_mode_test(substrate, p) for p in patterns]
        bridges = summary["P_bridges_total"]
        s_count_now = substrate.s_count()
        # Per-S diagnostics: size + activation distribution.
        s_sizes = [s.size() for s in substrate.s_entities.values()]
        s_active = sum(
            1 for s in substrate.s_entities.values() if s.activation > 0.0
        )
        history.append({
            "stage": stage,
            "P_counts": p_counts,
            "recalls": recalls,
            "bridges": bridges,
            "s_count": s_count_now,
            "s_active": s_active,
            "s_sizes": s_sizes,
            "step_count": substrate.step_count,
            "system_age": substrate.system_age,
        })

        print(f"  P counts:   {p_counts}")
        print(f"  Recalls:    {[f'{r * 100:.0f}%' for r in recalls]}")
        print(f"  Bridges:    {bridges}")
        print(f"  S count:    {s_count_now}  "
              f"(active: {s_active}, sizes: {sorted(s_sizes, reverse=True)})")
        print()

    final = history[-1]
    p_counts = final["P_counts"]
    recalls = final["recalls"]
    n_alive_6i = n_alive(p_counts)
    n_recall_6i = n_recall(recalls)
    final_s_count = final["s_count"]

    # ---- 11-way comparison ----
    print("=== 11-way comparison ===")
    phase_names = list(BASELINES.keys()) + ["6i"]
    header = " ".join(f"{name:>10s}" for name in phase_names)
    print(f"{'pattern':<8} {header}")
    print("-" * (10 + len(header)))
    for i in range(M):
        cells = []
        for name in phase_names[:-1]:
            cnt, rec = BASELINES[name]
            cells.append(f"{cnt[i]}P/{rec[i] * 100:.0f}%")
        cells.append(f"{p_counts[i]}P/{recalls[i] * 100:.0f}%")
        print(f"P{i:<7} " + " ".join(f"{c:>10s}" for c in cells))
    print("-" * (10 + len(header)))

    n_alive_summary = {name: n_alive(BASELINES[name][0]) for name in BASELINES}
    n_alive_summary["6i"] = n_alive_6i
    n_recall_summary = {name: n_recall(BASELINES[name][1]) for name in BASELINES}
    n_recall_summary["6i"] = n_recall_6i

    print(f"{'alive':<8} "
          + " ".join(f"{n_alive_summary[n]:>10d}" for n in phase_names))
    print(f"{'recall':<8} "
          + " ".join(f"{n_recall_summary[n]:>10d}" for n in phase_names))
    print(f"\nS entities at end: {final_s_count}")
    print()

    # ---- Verdict ----
    print("=== Verdict (Phase 6i) ===")
    alive_6c = n_alive_summary["6c"]
    recall_6c = n_recall_summary["6c"]

    if n_alive_6i == M and n_recall_6i == M:
        verdict = "PASS"
        reason = (
            "5/5 alive AND 5/5 recall. The S-level addition (third "
            "ontological tier with bottom-up emergence + top-down "
            "S→P feedback) RESOLVES the M=5 scaling problem. After 10 "
            "phases of parameter exploration ceiling'd at 4/4, the "
            "data was right to motivate architectural depth. "
            "Three-tier substrate (N + P + S) with three modes "
            "(training / recall / consolidation) is the complete "
            "minimal architecture at M=5."
        )
    elif (n_alive_6i == M and n_recall_6i == M - 1) \
            or (n_alive_6i == M - 1 and n_recall_6i == M):
        verdict = "STRONG_PARTIAL"
        weak_a = [i for i, c in enumerate(p_counts) if c < 2]
        weak_r = [i for i, r in enumerate(recalls) if r < 0.4]
        reason = (
            f"{n_alive_6i} alive AND {n_recall_6i} recall. Very close "
            f"to PASS. Weak: alive={weak_a}, recall={weak_r}. S-level "
            f"contributes meaningfully; minor tuning (γ_s_to_p, "
            f"alpha_p_to_s, or s_max_active) may close the gap."
        )
    elif (n_alive_6i > alive_6c and n_recall_6i >= recall_6c) \
            or (n_recall_6i > recall_6c and n_alive_6i >= alive_6c):
        verdict = "BETTER_THAN_6C"
        reason = (
            f"{n_alive_6i}/{n_recall_6i} strictly better than 6c's "
            f"{alive_6c}/{recall_6c}. S-level helps but not enough "
            f"for clean PASS. Real progress."
        )
    elif n_alive_6i == alive_6c and n_recall_6i == recall_6c:
        verdict = "MATCHES_6C"
        reason = (
            f"Same as Phase 6c ({alive_6c}/{recall_6c}). S-level "
            f"added no measurable improvement. Either S didn't emerge "
            f"during this protocol, or the feedback gain (γ_s_to_p) "
            f"is too low to influence outcomes. Investigate."
        )
    else:
        verdict = "WORSE_THAN_6C"
        reason = (
            f"{n_alive_6i}/{n_recall_6i} worse than 6c's "
            f"{alive_6c}/{recall_6c}. S-level introduces interference. "
            f"Possible: S→P feedback boost destabilises P-level "
            f"competition. Tune γ_s_to_p down or restrict S "
            f"emergence further."
        )
    print(f"{verdict}: {reason}")
    print()

    # ---- Persist ----
    results_dir = _REPO_ROOT / "results" / "substrate" / "phase_6i"
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "phase_6i_results.json").open("w") as f:
        json.dump(
            {
                "config": {
                    "n_neurons": n_neurons,
                    "M": M,
                    "pattern_size": pattern_size,
                    "K_per_pattern": K_per_pattern,
                    "K_consolidate": K_consolidate,
                    "rho_floor": substrate.rho_floor,
                    "k_protect": substrate.k_protect,
                    "alpha_p_to_s": substrate.alpha_p_to_s,
                    "gamma_s_to_p": substrate.gamma_s_to_p,
                    "s_sparsity_target": substrate.s_sparsity_target,
                    "s_min_active": substrate.s_min_active,
                    "s_max_active": substrate.s_max_active,
                    "schedule": "Phase 6c interleaved",
                    "seed": 42,
                },
                "history": history,
                "final": {
                    "P_counts": p_counts,
                    "recalls": recalls,
                    "bridges": final["bridges"],
                    "n_alive": n_alive_6i,
                    "n_recall": n_recall_6i,
                    "s_count": final_s_count,
                    "s_active": final["s_active"],
                    "s_sizes": final["s_sizes"],
                },
                "baselines": {
                    name: {"counts": cnt, "recalls": rec}
                    for name, (cnt, rec) in BASELINES.items()
                },
                "n_alive_by_phase": n_alive_summary,
                "n_recall_by_phase": n_recall_summary,
                "verdict": verdict,
                "reason": reason,
            },
            f,
            indent=2,
        )

    # ---- Plot ----
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    x = np.arange(M)
    width = 0.08
    colors = ["tab:red", "tab:orange", "tab:olive", "tab:green",
              "tab:blue", "tab:purple", "tab:cyan", "tab:brown",
              "tab:pink", "tab:gray", "black"]

    for idx, name in enumerate(phase_names):
        counts = p_counts if name == "6i" else BASELINES[name][0]
        offset = (idx - len(phase_names) / 2) * width
        axes[0].bar(x + offset, counts, width,
                    label=name, color=colors[idx])
    axes[0].axhline(2, color="gray", linestyle=":", alpha=0.4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"P{i}" for i in range(M)])
    axes[0].set_ylabel("# P entities")
    axes[0].set_title(f"Per-pattern P count — 11 phases")
    axes[0].grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8, ncol=6)

    for idx, name in enumerate(phase_names):
        recs_pct = ([r * 100 for r in recalls] if name == "6i"
                    else [r * 100 for r in BASELINES[name][1]])
        offset = (idx - len(phase_names) / 2) * width
        axes[1].bar(x + offset, recs_pct, width,
                    label=name, color=colors[idx])
    axes[1].axhline(40, color="black", linestyle="--", alpha=0.4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"P{i}" for i in range(M)])
    axes[1].set_ylabel("recall (%)")
    axes[1].set_title(
        f"Per-pattern recall — 11 phases: Phase 6i = {verdict}"
    )
    axes[1].set_ylim(0, 105)
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].legend(fontsize=8, ncol=6)

    plt.tight_layout()
    plt.savefig(results_dir / "phase_6i_results.png", dpi=120)
    plt.close(fig)
    print(f"Results → {results_dir}")

    if verdict == "PASS":
        return 0
    if verdict in ("STRONG_PARTIAL", "BETTER_THAN_6C"):
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
