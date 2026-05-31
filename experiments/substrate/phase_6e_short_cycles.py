"""Phase 6e — Short cycles / spaced revisits.

Tests the cadence-refinement Option 1 from the Phase 6c/6d analysis:
distribute training across multiple shorter rounds, with consolidation
between rounds. Each pattern gets revisited multiple times, staying
within the resurrection window the Phase 6d analysis identified.

Conservation of total budgets vs Phases 6c/6d:

    Per pattern total training:    4 rounds × K=20 = 80   ← matches baseline
    Total consolidation:           4 × K=1875 = 7500     ← matches baseline

The ONLY thing that changes is the DISTRIBUTION:

    Per round:
      for pattern in [P0, P1, P2, P3, P4]:
        training_mode(pattern, K=20)        # short visit (1500 steps)
      consolidation_mode(K=1875)            # mid-round refresh

Biology: spaced learning. The single most-established principle in
cognitive psychology — distributed practice beats massed practice.
Patterns revisited within their resurrection window stay viable.

Pre-experiment expectations:
  * Each pattern visited 4 times → attractor reset every ~9k steps
  * Consolidation distributed → all patterns refreshed when each
    still has resurrection-viable infrastructure
  * Risk: substrate age advances fast in early rounds (5×1500 training
    + 1875 consol per round = ~9k steps) so ρ hits the floor early
    and stays there. Differential timing within a round may still
    leave the first-trained pattern of each round the weakest.

Verdict:
  PASS              5/5 alive AND 5/5 recall ≥ 40 %
                    → architecture complete with spaced revisits
  STRONG_PARTIAL    5/5 alive AND 4/5 recall ≥ 40 %
  PARTIAL           ≥4 alive AND ≥3 recall ≥ 40 %
  NO_CHANGE         Recall outcome matches 6c/6d baseline → Option 2
                    (fresh-pattern lockout requiring src changes)
                    empirically motivated as the next step.

Outputs:
  results/substrate/phase_6e/phase_6e_results.{png,json}
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

# Re-use everything Phase 5b/6c set up.
from phase_5b_bridge_at_scale import (  # noqa: E402
    classify_p_entities,
    define_M_patterns,
    make_external,
    summarize_classification,
)
from phase_6c_consolidation_mode import (  # noqa: E402
    consolidation_mode_phase,
    recall_mode_test,
    training_mode_phase,
)


# ---------------------------------------------------------------------------
# Embedded baselines (6-way report)
# ---------------------------------------------------------------------------

BASELINES = {
    "5b": ([0, 3, 0, 4, 2], [0.0, 0.20, 0.40, 0.80, 0.80]),
    "6a": ([2, 5, 1, 5, 0], [0.40, 0.20, 0.60, 1.00, 0.20]),
    "6b": ([2, 3, 2, 4, 2], [0.0, 0.20, 0.40, 0.80, 0.80]),
    "6c": ([2, 3, 2, 4, 1], [0.40, 0.20, 0.40, 0.80, 0.80]),
    "6d": ([2, 3, 2, 4, 2], [0.0, 0.20, 0.40, 0.80, 0.80]),
}


def n_alive(counts: list[int]) -> int:
    return sum(1 for c in counts if c >= 2)


def n_recall(recalls: list[float]) -> int:
    return sum(1 for r in recalls if r >= 0.4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=== Phase 6e: Short cycles (spaced revisits) ===\n")
    n_neurons = 500
    M = 5
    pattern_size = 10
    n_rounds = 4
    K_train_per_visit = 20
    K_consolidate_per_round = 1875

    total_train_per_pattern = n_rounds * K_train_per_visit
    total_consolidate = n_rounds * K_consolidate_per_round
    print(f"M={M} patterns, {n_rounds} rounds")
    print(f"Per round: train each pattern K={K_train_per_visit} "
          f"+ consolidate K={K_consolidate_per_round}")
    print(f"Total training per pattern: {total_train_per_pattern} "
          f"(matches baseline 80)")
    print(f"Total consolidation:        {total_consolidate} "
          f"(matches baseline 7500)\n")

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
        rho_floor=0.3,
        seed=42,
    )

    patterns = define_M_patterns(n_neurons, M, pattern_size, seed=0)
    history: list[dict[str, Any]] = []

    for round_idx in range(n_rounds):
        print(f"--- Round {round_idx + 1}/{n_rounds} ---")

        # Short training visit for each pattern, in order.
        for pat_idx, pattern in enumerate(patterns):
            training_mode_phase(substrate, pattern, K=K_train_per_visit)

        # Mid-round consolidation.
        consolidation_mode_phase(substrate, K_steps=K_consolidate_per_round)

        # Checkpoint: P counts + recall per pattern.
        classification = classify_p_entities(substrate.p_entities, patterns)
        summary = summarize_classification(classification, M)
        p_counts = [summary[f"P_pattern_{i}"] for i in range(M)]
        recalls = [recall_mode_test(substrate, p) for p in patterns]
        bridges = summary["P_bridges_total"]

        history.append({
            "round": round_idx + 1,
            "P_counts": p_counts,
            "recalls": recalls,
            "bridges": bridges,
            "system_age": substrate.system_age,
        })

        print(f"  P counts: {p_counts}")
        print(f"  Recalls:  {[f'{r * 100:.0f}%' for r in recalls]}")
        print(f"  Bridges:  {bridges}")
        print(f"  system_age: {substrate.system_age:.0f}")
        print()

    # ---- Final ----
    final = history[-1]
    p_counts = final["P_counts"]
    recalls = final["recalls"]
    n_alive_6e = n_alive(p_counts)
    n_recall_6e = n_recall(recalls)

    # ---- 6-way comparison ----
    print("=== 6-way comparison ===")
    print(f"{'pattern':<8} {'5b':>10s} {'6a':>10s} {'6b':>10s} "
          f"{'6c':>10s} {'6d':>10s} {'6e':>10s}")
    print("-" * 70)
    for i in range(M):
        cells = []
        for name in ["5b", "6a", "6b", "6c", "6d"]:
            cnt, rec = BASELINES[name]
            cells.append(f"{cnt[i]}P/{rec[i] * 100:.0f}%")
        cells.append(f"{p_counts[i]}P/{recalls[i] * 100:.0f}%")
        print(f"P{i:<7} " + " ".join(f"{c:>10s}" for c in cells))
    print("-" * 70)

    n_alive_summary = {
        name: n_alive(BASELINES[name][0]) for name in BASELINES
    }
    n_alive_summary["6e"] = n_alive_6e
    n_recall_summary = {
        name: n_recall(BASELINES[name][1]) for name in BASELINES
    }
    n_recall_summary["6e"] = n_recall_6e

    print(f"{'alive (≥2P)':<8} "
          f"{n_alive_summary['5b']:>10d} {n_alive_summary['6a']:>10d} "
          f"{n_alive_summary['6b']:>10d} {n_alive_summary['6c']:>10d} "
          f"{n_alive_summary['6d']:>10d} {n_alive_summary['6e']:>10d}")
    print(f"{'recall (≥40%)':<8} "
          f"{n_recall_summary['5b']:>10d} {n_recall_summary['6a']:>10d} "
          f"{n_recall_summary['6b']:>10d} {n_recall_summary['6c']:>10d} "
          f"{n_recall_summary['6d']:>10d} {n_recall_summary['6e']:>10d}")
    print()

    # ---- Verdict ----
    print("=== Verdict (Phase 6e) ===")
    if n_alive_6e == M and n_recall_6e == M:
        verdict = "PASS"
        reason = (
            "5/5 alive AND 5/5 recall. Short cycles (spaced revisits) "
            "resolves the resurrection-window/competition coupling. "
            "Architecture COMPLETE at M=5: 2-level + 3-mode + spaced "
            "training schedule. The biological learning principle "
            "(distributed practice > massed practice) works in this "
            "substrate as a real architectural fix, not a hack."
        )
    elif n_alive_6e == M and n_recall_6e == M - 1:
        verdict = "STRONG_PARTIAL"
        weak = [i for i, r in enumerate(recalls) if r < 0.4]
        reason = (
            f"5/5 alive AND {n_recall_6e}/5 recall. Pattern(s) {weak} "
            f"below recall floor. Close to PASS; more rounds (5 or 6) "
            f"or larger per-round K_consolidate may close the gap."
        )
    elif n_alive_6e >= 4 and n_recall_6e >= 3:
        verdict = "PARTIAL"
        weak = [i for i, r in enumerate(recalls) if r < 0.4]
        reason = (
            f"{n_alive_6e} alive AND {n_recall_6e} recall. Pattern(s) "
            f"{weak} still struggle. Spaced revisits help but don't "
            f"fully resolve. Phase 6f (fresh-pattern lockout with src "
            f"changes) is the natural next experiment."
        )
    elif n_recall_6e == n_recall_summary["6c"] or n_recall_6e == n_recall_summary["6d"]:
        verdict = "NO_CHANGE"
        reason = (
            f"Same recall count as 6c/6d. Spaced revisits ALONE didn't "
            f"resolve the coupling. Empirically motivates Phase 6f: "
            f"fresh-pattern lockout (mark just-emerged P entities as "
            f"protected from cross-attractor activation for K steps "
            f"after emergence). Requires modifying src/substrate/."
        )
    else:
        verdict = "MIXED"
        reason = (
            f"{n_alive_6e} alive / {n_recall_6e} recall — outside "
            f"clean verdict buckets. See per-round history for trajectory."
        )
    print(f"{verdict}: {reason}")
    print()

    # ---- Persist ----
    results_dir = _REPO_ROOT / "results" / "substrate" / "phase_6e"
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "phase_6e_results.json").open("w") as f:
        json.dump(
            {
                "config": {
                    "n_neurons": n_neurons,
                    "M": M,
                    "pattern_size": pattern_size,
                    "n_rounds": n_rounds,
                    "K_train_per_visit": K_train_per_visit,
                    "K_consolidate_per_round": K_consolidate_per_round,
                    "total_train_per_pattern": total_train_per_pattern,
                    "total_consolidation": total_consolidate,
                    "rho_floor": 0.3,
                    "schedule": (
                        "round-robin: 4 rounds, each round trains each "
                        "pattern K=20 then consolidates K=1875"
                    ),
                    "seed": 42,
                },
                "per_round_history": history,
                "final": {
                    "P_counts": p_counts,
                    "recalls": recalls,
                    "bridges": final["bridges"],
                    "n_alive": n_alive_6e,
                    "n_recall": n_recall_6e,
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

    # ---- Plots ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    rounds = [h["round"] for h in history]

    # Top-left: per-pattern P count over rounds.
    pattern_colors = plt.cm.tab10.colors[:M]
    for i in range(M):
        counts = [h["P_counts"][i] for h in history]
        axes[0, 0].plot(rounds, counts, "o-",
                        color=pattern_colors[i], label=f"P{i}")
    axes[0, 0].axhline(2, color="gray", linestyle=":", alpha=0.4)
    axes[0, 0].set(xlabel="round", ylabel="# P entities",
                   title="Per-pattern P count across rounds")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    # Top-right: per-pattern recall over rounds.
    for i in range(M):
        recs = [h["recalls"][i] * 100 for h in history]
        axes[0, 1].plot(rounds, recs, "o-",
                        color=pattern_colors[i], label=f"P{i}")
    axes[0, 1].axhline(40, color="black", linestyle="--", alpha=0.4)
    axes[0, 1].set(xlabel="round", ylabel="recall (%)",
                   title="Per-pattern recall across rounds",
                   ylim=(-5, 105))
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    # Bottom-left: alive count across all phases.
    phase_names = ["5b", "6a", "6b", "6c", "6d", "6e"]
    alive_counts = [n_alive_summary[name] for name in phase_names]
    colors = ["tab:red", "tab:orange", "tab:olive",
              "tab:green", "tab:blue", "tab:purple"]
    bars = axes[1, 0].bar(phase_names, alive_counts, color=colors)
    axes[1, 0].axhline(M, color="black", linestyle="--", alpha=0.4,
                       label=f"target ({M}/{M})")
    axes[1, 0].set(ylabel="patterns alive (≥2 P)",
                   title="Alive count across phases", ylim=(0, M + 0.5))
    for bar, v in zip(bars, alive_counts):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                        str(v), ha="center", va="bottom")
    axes[1, 0].legend()

    # Bottom-right: recall count across all phases.
    recall_counts = [n_recall_summary[name] for name in phase_names]
    bars = axes[1, 1].bar(phase_names, recall_counts, color=colors)
    axes[1, 1].axhline(M, color="black", linestyle="--", alpha=0.4,
                       label=f"target ({M}/{M})")
    axes[1, 1].set(ylabel="patterns with recall ≥ 40%",
                   title="Recall count across phases", ylim=(0, M + 0.5))
    for bar, v in zip(bars, recall_counts):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                        str(v), ha="center", va="bottom")
    axes[1, 1].legend()

    plt.suptitle(f"Phase 6e — short cycles (spaced revisits): {verdict}",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "phase_6e_results.png", dpi=120)
    plt.close(fig)
    print(f"Results → {results_dir}")

    if verdict == "PASS":
        return 0
    if verdict in ("STRONG_PARTIAL", "PARTIAL", "MIXED"):
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
