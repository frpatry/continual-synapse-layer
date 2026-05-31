"""Phase 6h — Parameter combination: 6c cadence + 6f protection + low floor.

Final parameter-tuning experiment before S-level (Phase 6i). Tests
whether the right combination of known-good knobs is sufficient at
M=5, or whether parameter exploration is genuinely exhausted.

The combination:
  - rho_floor=0.3       (Phase 6c's regime — best recall historically)
  - k_protect=5000      (Phase 6f's protection — keeps fresh P alive)
  - Interleaved cadence (Phase 6c's proven best balanced)
  - N=500               (Phase 6c's size — N=1000 didn't help in 6g)

Each component has a specific empirical justification:
  - Phase 6c at N=500 with rho_floor=0.3 produced 4/4 — the best
    balanced result, missing only P4 (1 P / 80 % — close to alive).
  - Phase 6f's protection mechanism specifically prevents dissolution
    of freshly-emerged P entities during their first consolidation.
    P4 in 6c had only 1 P because its second P dissolved during the
    final stage's consolidation.
  - rho_floor=0.7 (Phase 6f's choice) caused late-pattern emergence
    failure (Phase 6g math). 0.3 keeps the decay rate moderate.

Hypothesis: Phase 6c's 4/4 + protection mechanism = 5/5/5.
  - Protection keeps P4's 2nd entity from dissolving → 5 alive
  - Phase 6c's recall pathway still works → 5 recall

If FAILS to reach 5/5/5 even with this combination, parameter
exploration is exhausted and S-level is the empirically necessary
next architectural step.

Verdict mapping:
  PASS            5 alive AND 5 recall ≥ 40 %
                  → parameter combination IS the answer.
  STRONG_PARTIAL  5 alive AND 4 recall (or 4/5)
                  → very close; minor tuning might close it.
  MATCHES_6C      4 alive AND 4 recall (or close)
                  → protection didn't add over 6c's baseline.
                  Parameter space exhausted → S-level next.
  WORSE           strictly worse than 6c's 4/4
                  → combination has a new problem; S-level still next.
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
# Embedded baselines (10-way report)
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
}


def n_alive(counts: list[int]) -> int:
    return sum(1 for c in counts if c >= 2)


def n_recall(recalls: list[float]) -> int:
    return sum(1 for r in recalls if r >= 0.4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=== Phase 6h: Param combination (6c cadence + 6f protection + low floor) ===\n")
    n_neurons = 500
    M = 5
    pattern_size = 10
    K_per_pattern = 80
    K_consolidate = 1500

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
        rho_floor=0.3,      # OVERRIDE: Phase 6c's regime (not the 0.7 default)
        k_protect=5000,     # KEEP: Phase 6f's protection mechanism
        seed=42,
    )

    patterns = define_M_patterns(n_neurons, M, pattern_size, seed=0)
    print(f"M={M} patterns, N={n_neurons}, "
          f"rho_floor=0.3, k_protect=5000")
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
        n_protected = sum(
            1 for p in substrate.p_entities.values()
            if p.is_protected(substrate.step_count)
        )
        history.append({
            "stage": stage,
            "P_counts": p_counts,
            "recalls": recalls,
            "bridges": bridges,
            "protected": n_protected,
            "step_count": substrate.step_count,
            "system_age": substrate.system_age,
        })

        print(f"  P counts:   {p_counts}")
        print(f"  Recalls:    {[f'{r * 100:.0f}%' for r in recalls]}")
        print(f"  Bridges:    {bridges}, protected: {n_protected}")
        print()

    final = history[-1]
    p_counts = final["P_counts"]
    recalls = final["recalls"]
    n_alive_6h = n_alive(p_counts)
    n_recall_6h = n_recall(recalls)

    # ---- 10-way comparison ----
    print("=== 10-way comparison ===")
    phase_names = list(BASELINES.keys()) + ["6h"]
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
    n_alive_summary["6h"] = n_alive_6h
    n_recall_summary = {name: n_recall(BASELINES[name][1]) for name in BASELINES}
    n_recall_summary["6h"] = n_recall_6h

    print(f"{'alive':<8} "
          + " ".join(f"{n_alive_summary[n]:>10d}" for n in phase_names))
    print(f"{'recall':<8} "
          + " ".join(f"{n_recall_summary[n]:>10d}" for n in phase_names))
    print()

    # ---- Verdict ----
    print("=== Verdict (Phase 6h) ===")
    alive_6c = n_alive_summary["6c"]
    recall_6c = n_recall_summary["6c"]

    if n_alive_6h == M and n_recall_6h == M:
        verdict = "PASS"
        reason = (
            "5/5 alive AND 5/5 recall. The parameter combination "
            "(6c cadence + 6f protection + rho_floor=0.3) IS the "
            "architectural answer. The Bio-Inspired choice in 6f "
            "was right in principle (protect fresh patterns) but "
            "wrong on the floor parameter. With rho_floor lowered "
            "back to 0.3, protection adds to Phase 6c's 4/4 and "
            "reaches 5/5. Architecture COMPLETE at M=5; no S-level "
            "needed."
        )
    elif (n_alive_6h == M and n_recall_6h == M - 1) or \
         (n_alive_6h == M - 1 and n_recall_6h == M):
        verdict = "STRONG_PARTIAL"
        weak_a = [i for i, c in enumerate(p_counts) if c < 2]
        weak_r = [i for i, r in enumerate(recalls) if r < 0.4]
        reason = (
            f"{n_alive_6h} alive AND {n_recall_6h} recall. Very "
            f"close. Weak: alive={weak_a}, recall={weak_r}. Minor "
            f"tuning (longer protection, more consolidation) might "
            f"reach PASS."
        )
    elif n_alive_6h >= alive_6c and n_recall_6h >= recall_6c \
            and (n_alive_6h > alive_6c or n_recall_6h > recall_6c):
        verdict = "BETTER_THAN_6C"
        reason = (
            f"{n_alive_6h}/{n_recall_6h} strictly better than 6c's "
            f"{alive_6c}/{recall_6c}. Protection adds, but not enough "
            f"for clean PASS. Continued parameter exploration may "
            f"help further; S-level remains a viable alternative."
        )
    elif n_alive_6h == alive_6c and n_recall_6h == recall_6c:
        verdict = "MATCHES_6C"
        reason = (
            f"Same as Phase 6c ({alive_6c}/{recall_6c}). Protection "
            f"mechanism added no measurable improvement at this "
            f"cadence + floor. Parameter exploration empirically "
            f"exhausted. S-level (Phase 6i) is the next architectural "
            f"step."
        )
    elif n_alive_6h < alive_6c or n_recall_6h < recall_6c:
        verdict = "WORSE_THAN_6C"
        reason = (
            f"{n_alive_6h}/{n_recall_6h} worse than Phase 6c's "
            f"{alive_6c}/{recall_6c}. The combination introduces "
            f"a new problem — likely protection interfering with "
            f"the natural attractor dynamics 6c relied on. S-level "
            f"is the next architectural step."
        )
    else:
        verdict = "MIXED"
        reason = (
            f"{n_alive_6h}/{n_recall_6h} — better on one axis, "
            f"worse on the other. No clean win."
        )
    print(f"{verdict}: {reason}")
    print()

    # ---- Persist ----
    results_dir = _REPO_ROOT / "results" / "substrate" / "phase_6h"
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "phase_6h_results.json").open("w") as f:
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
                    "schedule": "Phase 6c interleaved",
                    "seed": 42,
                },
                "history": history,
                "final": {
                    "P_counts": p_counts,
                    "recalls": recalls,
                    "bridges": final["bridges"],
                    "n_alive": n_alive_6h,
                    "n_recall": n_recall_6h,
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

    # ---- Plot (concise: just per-pattern + alive/recall bars) ----
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    phase_subset = ["6a", "6b", "6c", "6d", "6e'", "6f", "6h"]
    colors = ["tab:orange", "tab:olive", "tab:green", "tab:blue",
              "tab:cyan", "tab:brown", "black"]
    x = np.arange(M)
    width = 0.12
    for idx, name in enumerate(phase_subset):
        counts = p_counts if name == "6h" else BASELINES[name][0]
        recs_pct = ([r * 100 for r in recalls] if name == "6h"
                    else [r * 100 for r in BASELINES[name][1]])
        offset = (idx - len(phase_subset) / 2) * width
        axes[0].bar(x + offset, counts, width, color=colors[idx], label=name)
        axes[1].bar(x + offset, recs_pct, width, color=colors[idx], label=name)
    axes[0].axhline(2, color="gray", linestyle=":", alpha=0.4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"P{i}" for i in range(M)])
    axes[0].set_ylabel("# P entities")
    axes[0].set_title(
        f"Phase 6h vs key prior phases: per-pattern P count — {verdict}"
    )
    axes[0].grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=9, ncol=7)

    axes[1].axhline(40, color="black", linestyle="--", alpha=0.4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"P{i}" for i in range(M)])
    axes[1].set_ylabel("recall (%)")
    axes[1].set_title("Phase 6h vs key prior phases: per-pattern recall")
    axes[1].set_ylim(0, 105)
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].legend(fontsize=9, ncol=7)

    plt.tight_layout()
    plt.savefig(results_dir / "phase_6h_results.png", dpi=120)
    plt.close(fig)
    print(f"Results → {results_dir}")

    if verdict == "PASS":
        return 0
    if verdict in ("STRONG_PARTIAL", "BETTER_THAN_6C"):
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
