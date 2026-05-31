"""Phase 6a — Re-run Phase 5b under the ρ floor (default 0.3).

Phase 5b found two distinct failure modes for sequential M=5 training:
  (a) emergence failure for late patterns — substrate aged enough that
      effective η·ρ ≈ 0.001 couldn't push any pair past emergence
  (b) weak-attractor decay — patterns with only 1–2 P dissolve over
      subsequent training (24000+ steps)

Phase 6a hypothesizes a single fix: a floor on ρ(age). With floor=0.3,
plasticity never drops below 30% of its young value, so:
  - Late patterns (high age) keep enough effective eta to emerge.
  - Older patterns keep enough plasticity to consolidate during the
    spontaneous-replay window (the Phase 5b finding where pattern 3
    grew during pattern 4's training).

Protocol: identical to Phase 5b. Train 5 disjoint patterns sequentially,
measure per-stage P counts + bridges + recall, then combined-cue tests.
Only difference: rho_floor=0.3 (substrate default in Phase 6a).

Verdict comparison vs Phase 5b baseline:
  PASS       all 5 patterns have ≥ 2 P AND all recall ≥ 0.4
             AND bridges modest (avg < 10/pair) → both bottlenecks fixed
  PARTIAL    improvement over 5b but not all patterns recovered
             → bottleneck (a) fixed, (b) remains → motivates Phase 6b (S)
  NO_CHANGE  similar to 5b → floor insufficient → next: try floor=0.5,
             or accept architecture needs S-level
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

# Re-use Phase 5b's helpers — same protocol, only the substrate ctor
# differs. Import them via path manipulation since experiments/ aren't
# a package.
sys.path.insert(0, str(_REPO_ROOT / "experiments" / "substrate"))

from substrate.substrate import Substrate  # noqa: E402

from phase_5b_bridge_at_scale import (  # noqa: E402
    classify_p_entities,
    define_M_patterns,
    summarize_classification,
    test_combined_recall,
    test_individual_recall,
    train_one_pattern,
)


# ---------------------------------------------------------------------------
# Reference Phase 5b numbers (from results/substrate/phase_5b/phase_5b_results.json)
# Embedded here for the side-by-side report so the comparison shows even
# if the prior JSON is missing or moved.
# ---------------------------------------------------------------------------

PHASE_5B_BASELINE = {
    "rho_floor": 0.0,
    "pattern_p_counts": [0, 3, 0, 4, 2],
    "pattern_recalls": [0.0, 0.20, 0.40, 0.80, 0.80],
    "bridges_total": 29,
    "avg_bridges_per_pair": 2.9,
    "verdict": "FAIL_capacity",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=== Phase 6a: Re-run Phase 5b protocol with ρ floor ===\n")
    rho_floor = 0.3
    n_neurons = 500
    M = 5
    pattern_size = 10
    K_per_pattern = 80

    substrate = Substrate(
        n_neurons=n_neurons,
        k_connectivity=30,
        eta=0.01,
        lambda_decay=0.001,
        threshold=0.3,
        sparsity_target=0.05,
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
        rho_floor=rho_floor,
        seed=42,
    )

    patterns = define_M_patterns(n_neurons, M, pattern_size, seed=0)
    print(f"Substrate: N={n_neurons}, rho_floor={rho_floor}")
    print(f"M={M} disjoint patterns of size {pattern_size} each.\n")

    history: list[dict[str, Any]] = []

    for stage in range(M):
        print(f"--- Training pattern {stage} (K={K_per_pattern} cycles) ---")
        train_one_pattern(substrate, patterns[stage], K=K_per_pattern)

        classification = classify_p_entities(substrate.p_entities, patterns)
        summary = summarize_classification(classification, M)

        recalls = []
        for i in range(stage + 1):
            comp = test_individual_recall(substrate, patterns[i])
            recalls.append({"pattern": i, "completion": comp})

        history.append({
            "stage": stage,
            "trained_patterns_so_far": stage + 1,
            "P_counts": summary,
            "recalls": recalls,
            "system_age": substrate.system_age,
        })

        p_counts_str = " ".join(
            f"P{i}={summary[f'P_pattern_{i}']}" for i in range(M)
        )
        print(f"  P counts: {p_counts_str}")
        bridges_str = ", ".join(
            f"({i},{j})={summary[f'P_bridge_{i}_{j}']}"
            for i in range(stage + 1) for j in range(i + 1, stage + 1)
        ) or "(none)"
        print(f"  Bridges:  total={summary['P_bridges_total']}, "
              f"breakdown: {bridges_str}")
        recall_str = " ".join(
            f"P{r['pattern']}:{r['completion'] * 100:.0f}%" for r in recalls
        )
        print(f"  Recalls:  {recall_str}")
        print()

    # ---- Combined-cue tests ----
    print("--- Combined-cue tests ---")
    combined_results: list[dict[str, Any]] = []
    for n_combined in range(2, M + 1):
        activations = test_combined_recall(substrate, patterns, n_combined)
        act_str = " ".join(f"{a * 100:.0f}%" for a in activations)
        print(f"  n={n_combined}: per-pattern activation = {act_str}  "
              f"(mean={np.mean(activations) * 100:.0f}%, "
              f"min={min(activations) * 100:.0f}%)")
        combined_results.append({
            "n_combined": n_combined,
            "activations_per_pattern": activations,
            "mean_activation": float(np.mean(activations)),
            "min_activation": float(min(activations)),
        })
    print()

    # ---- Final analysis ----
    final = history[-1]
    summary = final["P_counts"]
    pattern_p_counts = [summary[f"P_pattern_{i}"] for i in range(M)]
    pattern_recalls = [r["completion"] for r in final["recalls"]]
    bridges_total = summary["P_bridges_total"]
    n_possible_pairs = M * (M - 1) // 2
    avg_bridges_per_pair = bridges_total / max(n_possible_pairs, 1)

    all_patterns_have_p = all(c >= 2 for c in pattern_p_counts)
    all_patterns_recall = all(r >= 0.4 for r in pattern_recalls)
    bridges_modest = avg_bridges_per_pair < 10.0

    # ---- Side-by-side report ----
    print("=== Phase 5b baseline vs Phase 6a (rho_floor=0.3) ===")
    print(f"{'metric':35s} {'5b (no floor)':>16s} {'6a (floor=0.3)':>18s}")
    print("-" * 73)
    print(f"{'pattern P counts':35s} "
          f"{str(PHASE_5B_BASELINE['pattern_p_counts']):>16s} "
          f"{str(pattern_p_counts):>18s}")
    recalls_5b_str = "[" + ", ".join(
        f"{r * 100:.0f}%" for r in PHASE_5B_BASELINE["pattern_recalls"]
    ) + "]"
    recalls_6a_str = "[" + ", ".join(
        f"{r * 100:.0f}%" for r in pattern_recalls
    ) + "]"
    print(f"{'pattern recalls':35s} "
          f"{recalls_5b_str:>16s} {recalls_6a_str:>18s}")
    print(f"{'bridges total':35s} "
          f"{PHASE_5B_BASELINE['bridges_total']:>16d} {bridges_total:>18d}")
    print(f"{'avg bridges per pair':35s} "
          f"{PHASE_5B_BASELINE['avg_bridges_per_pair']:>16.1f} "
          f"{avg_bridges_per_pair:>18.1f}")
    print(f"{'all patterns have ≥2 P':35s} "
          f"{'False':>16s} {str(all_patterns_have_p):>18s}")
    print(f"{'all patterns recall ≥ 40 %':35s} "
          f"{'False':>16s} {str(all_patterns_recall):>18s}")
    print(f"{'bridges modest':35s} "
          f"{'True':>16s} {str(bridges_modest):>18s}")
    print()

    # ---- Verdict ----
    print("=== Verdict (Phase 6a) ===")
    if all_patterns_have_p and all_patterns_recall and bridges_modest:
        verdict = "PASS"
        reason = (
            "Floor=0.3 resolves both Phase 5b bottlenecks: every pattern "
            "now emerges enough P AND recalls reliably AND bridges stay "
            "modest. 2-level architecture suffices with the formula "
            "adjustment — no S-level needed at M=5. Critical-period "
            "asymmetry is partially lost as a side-effect (see Phase 3.1 "
            "re-run); that's the tradeoff."
        )
    elif (
        sum(1 for c in pattern_p_counts if c >= 2)
        > sum(1 for c in PHASE_5B_BASELINE["pattern_p_counts"] if c >= 2)
    ):
        verdict = "PARTIAL"
        recovered = [
            i for i in range(M)
            if PHASE_5B_BASELINE["pattern_p_counts"][i] < 2
            and pattern_p_counts[i] >= 2
        ]
        lost = [
            i for i in range(M)
            if pattern_p_counts[i] < 2
        ]
        reason = (
            f"Improvement over Phase 5b: patterns {recovered} recovered "
            f"(emerged P that previously didn't); patterns {lost} still "
            f"missing. Bottleneck (a) emergence-failure resolved for "
            f"some/all late patterns; bottleneck (b) weak-attractor "
            f"decay may still apply. If pattern recovery is partial, "
            f"motivates Phase 6b (S level) for the residual weak-"
            f"attractor case."
        )
    else:
        verdict = "NO_CHANGE"
        reason = (
            "Floor=0.3 did not measurably improve over Phase 5b. Either "
            "the floor is too low (try 0.5) or the bottleneck isn't "
            "pure plasticity collapse — could be cumulative attractor "
            "interference that needs a fundamentally different fix "
            "(S level, scheduled replay, per-pattern age reset)."
        )
    print(f"{verdict}: {reason}")

    # ---- Persist ----
    results_dir = _REPO_ROOT / "results" / "substrate" / "phase_6a"
    results_dir.mkdir(parents=True, exist_ok=True)

    with (results_dir / "phase_6a_results.json").open("w") as f:
        json.dump(
            {
                "config": {
                    "n_neurons": n_neurons,
                    "M": M,
                    "pattern_size": pattern_size,
                    "K_per_pattern": K_per_pattern,
                    "rho_floor": rho_floor,
                    "T_present": 15,
                    "T_rest": 60,
                    "seed": 42,
                },
                "phase_5b_baseline": PHASE_5B_BASELINE,
                "phase_6a_results": {
                    "pattern_p_counts": pattern_p_counts,
                    "pattern_recalls": pattern_recalls,
                    "bridges_total": bridges_total,
                    "avg_bridges_per_pair": avg_bridges_per_pair,
                    "all_patterns_have_p": all_patterns_have_p,
                    "all_patterns_recall": all_patterns_recall,
                    "bridges_modest": bridges_modest,
                },
                "history": history,
                "combined_tests": combined_results,
                "verdict": verdict,
                "reason": reason,
            },
            f,
            indent=2,
        )

    # ---- Plots ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Side-by-side per-pattern P count comparison.
    x = np.arange(M)
    width = 0.4
    axes[0].bar(x - width / 2, PHASE_5B_BASELINE["pattern_p_counts"],
                width, label="Phase 5b (no floor)", color="tab:red")
    axes[0].bar(x + width / 2, pattern_p_counts,
                width, label="Phase 6a (floor=0.3)", color="tab:blue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"P{i}" for i in range(M)])
    axes[0].set_ylabel("# P entities")
    axes[0].set_title("Per-pattern P counts: 5b vs 6a")
    axes[0].grid(alpha=0.3, axis="y")
    axes[0].legend()

    # Side-by-side recall comparison.
    axes[1].bar(x - width / 2, PHASE_5B_BASELINE["pattern_recalls"],
                width, label="Phase 5b", color="tab:red")
    axes[1].bar(x + width / 2, pattern_recalls,
                width, label="Phase 6a", color="tab:blue")
    axes[1].axhline(0.4, color="gray", linestyle=":", alpha=0.6,
                    label="recall floor")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"P{i}" for i in range(M)])
    axes[1].set_ylabel("recall completion")
    axes[1].set_title("Per-pattern recall: 5b vs 6a")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].legend()

    # Combined-cue test in 6a.
    n_list = [r["n_combined"] for r in combined_results]
    means = [r["mean_activation"] for r in combined_results]
    mins = [r["min_activation"] for r in combined_results]
    axes[2].plot(n_list, means, "o-", color="tab:blue", label="mean")
    axes[2].plot(n_list, mins, "s-", color="tab:red", label="min")
    axes[2].set(
        xlabel="patterns presented simultaneously",
        ylabel="P activation fraction",
        title="Phase 6a — combined-cue test",
        ylim=(-0.05, 1.05),
    )
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    plt.suptitle(
        f"Phase 6a — ρ floor {rho_floor} on the Phase 5b protocol: {verdict}",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(results_dir / "phase_6a_results.png", dpi=120)
    plt.close(fig)
    print(f"\nResults → {results_dir}")

    if verdict == "PASS":
        return 0
    if verdict == "PARTIAL":
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
