"""Phase 6f — Bio-Inspired comprehensive fix.

Two coordinated architectural changes (both via Substrate defaults):

  1. rho_floor: 0.3 → 0.7
     Eliminates artificial late-learning suppression. Plasticity stays
     at ≥ 70% of its young-substrate level at every age.
     Tradeoff: Phase 3.1's critical-period demonstration regresses
     (both young & mature share ρ ≈ 0.7 from age=2 onward → no
     differential). Acceptable for an AI substrate whose goal is
     learning + retention, not developmental fidelity.

  2. Fresh-pattern protection (NEW mechanism)
     PEntity gains ``protected_until`` field; Substrate gains
     ``k_protect`` ctor param (default 5000). When _emerge_p creates
     a new P, ``protected_until = step_count + k_protect``. During
     that window _decay_and_dissolve_p still applies weight decay
     but skips the dissolution check. Lets fresh attractors stabilize
     through subsequent consolidation cycles instead of being out-
     competed by older established attractors.
     Bio analog: protein-synthesis-dependent LTP late phase + synaptic
     tagging — newly-potentiated synapses are actively maintained
     against competition for ~30 min–1 h post-induction.

Combined hypothesis: A + B together resolve the scaling issues
identified across Phases 5b through 6e'. Predict 5/5 alive AND 5/5
recall at M=5.

Protocol: Phase 6c's interleaved cadence (the best balanced of the
cadence experiments) — train pattern K=80 with feedback OFF,
consolidate K=1500 with feedback ON, advance to next pattern.

Verdict comparison (8-way) — 5b ... 6e' + 6f:
  PASS              5/5 alive AND 5/5 recall ≥ 40%
                    → architecture complete at M=5
  STRONG_PARTIAL    5/5 alive AND 4/5 recall
  PARTIAL           strictly better than best previous phase
                    (6c's 4/4) on at least one metric
  NO_CHANGE         matches some prior phase's outcome
                    → Bio-Inspired fix not sufficient → S level or
                      deeper architectural change needed.
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
# Embedded baselines (8-way report)
# ---------------------------------------------------------------------------

BASELINES = {
    "5b":  ([0, 3, 0, 4, 2], [0.0, 0.20, 0.40, 0.80, 0.80]),
    "6a":  ([2, 5, 1, 5, 0], [0.40, 0.20, 0.60, 1.00, 0.20]),
    "6b":  ([2, 3, 2, 4, 2], [0.0, 0.20, 0.40, 0.80, 0.80]),
    "6c":  ([2, 3, 2, 4, 1], [0.40, 0.20, 0.40, 0.80, 0.80]),
    "6d":  ([2, 3, 2, 4, 2], [0.0, 0.20, 0.40, 0.80, 0.80]),
    "6e":  ([0, 0, 0, 0, 0], [0.0, 0.0, 0.0, 0.0, 0.0]),
    "6e'": ([0, 5, 1, 4, 0], [0.20, 0.40, 0.40, 0.80, 0.40]),
}


def n_alive(counts: list[int]) -> int:
    return sum(1 for c in counts if c >= 2)


def n_recall(recalls: list[float]) -> int:
    return sum(1 for r in recalls if r >= 0.4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=== Phase 6f: Bio-Inspired comprehensive fix ===\n")
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
        # ---- Phase 6f new defaults (also Substrate's new defaults) ----
        rho_floor=0.7,    # was 0.3 in Phase 6a
        k_protect=5000,   # NEW
        # ---------------------------------------------------------------
        seed=42,
    )

    patterns = define_M_patterns(n_neurons, M, pattern_size, seed=0)
    print(f"M={M} patterns, K_train={K_per_pattern} each, "
          f"K_consolidate={K_consolidate}")
    print(f"Substrate: rho_floor={substrate.rho_floor}, "
          f"k_protect={substrate.k_protect}\n")

    history: list[dict[str, Any]] = []

    for stage in range(M):
        print(f"--- Stage {stage}: train (feedback OFF, K={K_per_pattern}) ---")
        training_mode_phase(substrate, patterns[stage], K=K_per_pattern)

        print(f"--- Stage {stage}: consolidate ({K_consolidate} steps) ---")
        consolidation_mode_phase(substrate, K_steps=K_consolidate)

        classification = classify_p_entities(substrate.p_entities, patterns)
        summary = summarize_classification(classification, M)
        p_counts = [summary[f"P_pattern_{i}"] for i in range(M)]
        recalls = [recall_mode_test(substrate, p) for p in patterns]
        bridges = summary["P_bridges_total"]

        # Protected P diagnostic — how many fresh P are still in window?
        protected_count = sum(
            1 for p in substrate.p_entities.values()
            if p.is_protected(substrate.step_count)
        )

        history.append({
            "stage": stage,
            "P_counts": p_counts,
            "recalls": recalls,
            "bridges": bridges,
            "protected_count": protected_count,
            "step_count": substrate.step_count,
            "system_age": substrate.system_age,
        })

        print(f"  P counts:   {p_counts}")
        print(f"  Recalls:    {[f'{r * 100:.0f}%' for r in recalls]}")
        print(f"  Bridges:    {bridges}")
        print(f"  Protected P (still in window): {protected_count}")
        print(f"  step_count: {substrate.step_count}, "
              f"system_age: {substrate.system_age:.0f}")
        print()

    final = history[-1]
    p_counts = final["P_counts"]
    recalls = final["recalls"]
    n_alive_6f = n_alive(p_counts)
    n_recall_6f = n_recall(recalls)

    # ---- 8-way comparison ----
    print("=== 8-way comparison ===")
    phase_names = ["5b", "6a", "6b", "6c", "6d", "6e", "6e'", "6f"]
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

    n_alive_summary = {
        name: n_alive(BASELINES[name][0]) for name in BASELINES
    }
    n_alive_summary["6f"] = n_alive_6f
    n_recall_summary = {
        name: n_recall(BASELINES[name][1]) for name in BASELINES
    }
    n_recall_summary["6f"] = n_recall_6f

    print(f"{'alive':<8} "
          + " ".join(f"{n_alive_summary[n]:>10d}" for n in phase_names))
    print(f"{'recall':<8} "
          + " ".join(f"{n_recall_summary[n]:>10d}" for n in phase_names))
    print()

    # ---- Verdict ----
    print("=== Verdict (Phase 6f) ===")
    best_alive_prior = max(n_alive_summary[n] for n in phase_names[:-1])
    best_recall_prior = max(n_recall_summary[n] for n in phase_names[:-1])

    if n_alive_6f == M and n_recall_6f == M:
        verdict = "PASS"
        reason = (
            "5/5 alive AND 5/5 recall. The Bio-Inspired fix "
            "(rho_floor=0.7 + fresh-pattern protection) RESOLVES "
            "the M=5 scaling problem. Architecture COMPLETE: the "
            "substrate learns (all patterns emerge), doesn't forget "
            "(all patterns recall), and is stable enough as a "
            "foundation for the user's next-step goal of generating "
            "new ideas. The two mechanisms tackle the two coupled "
            "constraints exposed by Phases 5b–6e' independently: "
            "higher floor keeps plasticity available for emergence, "
            "protection keeps fresh attractors alive against "
            "established competition."
        )
    elif n_alive_6f == M and n_recall_6f == M - 1:
        verdict = "STRONG_PARTIAL"
        weak = [i for i, r in enumerate(recalls) if r < 0.4]
        reason = (
            f"5/5 alive AND {n_recall_6f}/5 recall. Pattern(s) {weak} "
            f"below the recall floor. Very close to PASS — likely "
            f"resolvable by tuning k_protect higher or extending "
            f"consolidation per stage."
        )
    elif n_alive_6f > best_alive_prior or n_recall_6f > best_recall_prior:
        verdict = "PARTIAL"
        reason = (
            f"Bio-Inspired fix produces {n_alive_6f}/{n_recall_6f} (vs "
            f"best prior {best_alive_prior}/{best_recall_prior}). "
            f"Strictly better than any prior phase on at least one "
            f"metric. Real progress, not yet PASS."
        )
    else:
        verdict = "NO_CHANGE"
        reason = (
            f"{n_alive_6f}/{n_recall_6f} — matches some prior phase, "
            f"no improvement. The Bio-Inspired fix is insufficient at "
            f"M=5. Empirically motivates either S-level addition or "
            f"acceptance of the current memory-strength spectrum as "
            f"intrinsic to this architecture."
        )
    print(f"{verdict}: {reason}")
    print()

    # ---- Persist ----
    results_dir = _REPO_ROOT / "results" / "substrate" / "phase_6f"
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "phase_6f_results.json").open("w") as f:
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
                    "schedule": (
                        "Phase 6c-style interleaved: train K=80 with "
                        "feedback OFF, consolidate K=1500 with feedback "
                        "ON, advance to next pattern."
                    ),
                    "seed": 42,
                },
                "history": history,
                "final": {
                    "P_counts": p_counts,
                    "recalls": recalls,
                    "bridges": final["bridges"],
                    "n_alive": n_alive_6f,
                    "n_recall": n_recall_6f,
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
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    x = np.arange(M)
    width = 0.10

    # Top-left: per-pattern P count 8-way.
    colors = ["tab:red", "tab:orange", "tab:olive", "tab:green",
              "tab:blue", "tab:purple", "tab:cyan", "black"]
    for idx, name in enumerate(phase_names):
        if name == "6f":
            counts = p_counts
        else:
            counts = BASELINES[name][0]
        axes[0, 0].bar(x + (idx - len(phase_names) / 2) * width, counts, width,
                       label=name, color=colors[idx])
    axes[0, 0].axhline(2, color="gray", linestyle=":", alpha=0.4)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([f"P{i}" for i in range(M)])
    axes[0, 0].set_ylabel("# P entities")
    axes[0, 0].set_title("Per-pattern P count across 8 phases")
    axes[0, 0].grid(alpha=0.3, axis="y")
    axes[0, 0].legend(fontsize=8, ncol=4)

    # Top-right: per-pattern recall 8-way.
    for idx, name in enumerate(phase_names):
        if name == "6f":
            recs = [r * 100 for r in recalls]
        else:
            recs = [r * 100 for r in BASELINES[name][1]]
        axes[0, 1].bar(x + (idx - len(phase_names) / 2) * width, recs, width,
                       label=name, color=colors[idx])
    axes[0, 1].axhline(40, color="black", linestyle="--", alpha=0.4)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([f"P{i}" for i in range(M)])
    axes[0, 1].set_ylabel("recall (%)")
    axes[0, 1].set_title(f"Per-pattern recall across 8 phases: {verdict}")
    axes[0, 1].set_ylim(0, 105)
    axes[0, 1].grid(alpha=0.3, axis="y")
    axes[0, 1].legend(fontsize=8, ncol=4)

    # Bottom-left: alive-count summary.
    alive_counts = [n_alive_summary[n] for n in phase_names]
    bars = axes[1, 0].bar(phase_names, alive_counts, color=colors)
    axes[1, 0].axhline(M, color="black", linestyle="--", alpha=0.4,
                       label=f"target ({M}/{M})")
    axes[1, 0].set(ylabel="patterns alive (≥2 P)",
                   title="Alive count across 8 phases",
                   ylim=(0, M + 0.5))
    for bar, v in zip(bars, alive_counts):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                        str(v), ha="center", va="bottom")
    axes[1, 0].legend()

    # Bottom-right: recall-count summary.
    recall_counts = [n_recall_summary[n] for n in phase_names]
    bars = axes[1, 1].bar(phase_names, recall_counts, color=colors)
    axes[1, 1].axhline(M, color="black", linestyle="--", alpha=0.4,
                       label=f"target ({M}/{M})")
    axes[1, 1].set(ylabel="patterns with recall ≥ 40%",
                   title="Recall count across 8 phases",
                   ylim=(0, M + 0.5))
    for bar, v in zip(bars, recall_counts):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                        str(v), ha="center", va="bottom")
    axes[1, 1].legend()

    plt.suptitle(f"Phase 6f — Bio-Inspired fix: {verdict}", fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "phase_6f_results.png", dpi=120)
    plt.close(fig)
    print(f"Results → {results_dir}")

    if verdict == "PASS":
        return 0
    if verdict in ("STRONG_PARTIAL", "PARTIAL"):
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
