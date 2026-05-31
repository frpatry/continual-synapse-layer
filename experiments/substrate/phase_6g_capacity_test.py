"""Phase 6g — Capacity test at N=1000.

After 8 phases (5b through 6f) at N=500, M=5, no configuration reaches
5/5/5. The best balanced is Phase 6c at 4/4. Two competing hypotheses
for that ceiling:

  (a) CAPACITY-BOUND. N=500 is too small for 5 disjoint patterns to
      coexist with viable per-pattern attractors. The brain has ~86B
      neurons; our N=500 is comically small for any non-trivial
      multi-pattern task. If we 2× the substrate, the ceiling lifts.

  (b) ARCHITECTURAL LIMIT. The 2-level (N + P + P-P + P→N feedback)
      ontology genuinely can't represent 5 disjoint patterns
      simultaneously regardless of N — there's an intrinsic
      cross-pattern interference mechanism that scaling can't fix.
      Resolving this would require S-level grouping.

Phase 6g tests (a) directly: keep architecture and protocol identical
to Phase 6f, but double N. If the M=5 wall is capacity-bound, we
should see 5/5/5 (or close). If it's architectural, results match 6f.

Protocol — proportional scaling of the substrate, untouched protocol:
  N=1000               (was 500)
  k_connectivity=60    (was 30; same connectivity density)
  pattern_size=20      (was 10; same pattern coverage 2% of N)
  sparsity_target=0.05 (same %; absolute k-WTA = 50 vs 25)

  Cadence (Phase 6c-style, the best baseline):
    For each pattern:
      training_mode(K=80, feedback OFF)
      consolidation_mode(K=1500, feedback ON)

Verdict:
  PASS              5/5 alive AND 5/5 recall ≥ 40 %
                    → capacity was the wall; architecture works
  STRONG_PARTIAL    5/5 alive AND 4/5 recall (or vice versa)
                    → capacity helps but architecture has some
                      intrinsic ceiling beyond pure scaling
  NO_CHANGE         outcome matches Phase 6f or worse
                    → architectural limit confirmed; S-level
                      empirically required for the next phase

Runtime estimate: 4× per-step cost from N² scaling × 37500 steps
≈ 15–25 min CPU.
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
# Embedded baselines (9-way report)
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
}


def n_alive(counts: list[int]) -> int:
    return sum(1 for c in counts if c >= 2)


def n_recall(recalls: list[float]) -> int:
    return sum(1 for r in recalls if r >= 0.4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=== Phase 6g: Capacity test at N=1000 ===\n")
    n_neurons = 1000
    k_connectivity = 60
    M = 5
    pattern_size = 20
    K_per_pattern = 80
    K_consolidate = 1500

    substrate = Substrate(
        n_neurons=n_neurons,
        k_connectivity=k_connectivity,
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
        rho_floor=0.7,       # Phase 6f Bio-Inspired default
        k_protect=5000,      # Phase 6f Bio-Inspired default
        seed=42,
    )

    patterns = define_M_patterns(n_neurons, M, pattern_size, seed=0)
    print(f"Substrate: N={n_neurons}, k_connectivity={k_connectivity}, "
          f"sparsity_target=0.05 (k-WTA budget = "
          f"{int(0.05 * n_neurons)})")
    print(f"M={M} disjoint patterns of size {pattern_size} each "
          f"({M * pattern_size}/{n_neurons} = "
          f"{M * pattern_size / n_neurons * 100:.1f}% coverage)")
    print(f"Protocol: Phase 6c interleaved cadence, "
          f"K_train={K_per_pattern}, K_consolidate={K_consolidate}\n")

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
        print(f"  Protected:  {protected_count}")
        print()

    final = history[-1]
    p_counts = final["P_counts"]
    recalls = final["recalls"]
    n_alive_6g = n_alive(p_counts)
    n_recall_6g = n_recall(recalls)

    # ---- 9-way comparison ----
    print("=== 9-way comparison ===")
    phase_names = ["5b", "6a", "6b", "6c", "6d", "6e", "6e'", "6f", "6g"]
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
    n_alive_summary["6g"] = n_alive_6g
    n_recall_summary = {name: n_recall(BASELINES[name][1]) for name in BASELINES}
    n_recall_summary["6g"] = n_recall_6g

    print(f"{'alive':<8} "
          + " ".join(f"{n_alive_summary[n]:>10d}" for n in phase_names))
    print(f"{'recall':<8} "
          + " ".join(f"{n_recall_summary[n]:>10d}" for n in phase_names))
    print()

    # ---- Verdict ----
    print("=== Verdict (Phase 6g) ===")
    best_alive_prior = max(n_alive_summary[n] for n in phase_names[:-1])
    best_recall_prior = max(n_recall_summary[n] for n in phase_names[:-1])

    if n_alive_6g == M and n_recall_6g == M:
        verdict = "PASS"
        reason = (
            "5/5 alive AND 5/5 recall at N=1000. CAPACITY was the wall, "
            "not architecture. 2-level + 3-mode + Bio-Inspired floor + "
            "fresh-pattern protection IS sufficient for M=5 sequential "
            "disjoint patterns given adequate substrate sizing. "
            "Architecture COMPLETE."
        )
    elif n_alive_6g == M and n_recall_6g == M - 1:
        verdict = "STRONG_PARTIAL"
        weak = [i for i, r in enumerate(recalls) if r < 0.4]
        reason = (
            f"5/5 alive AND {n_recall_6g}/5 recall at N=1000. "
            f"Pattern(s) {weak} still below recall floor. Capacity "
            f"helps but doesn't fully resolve. Either further scaling "
            f"(N=2000) or architectural addition (S-level) likely "
            f"needed for full PASS."
        )
    elif (n_alive_6g > best_alive_prior
          or n_recall_6g > best_recall_prior):
        verdict = "PARTIAL"
        reason = (
            f"{n_alive_6g} alive / {n_recall_6g} recall at N=1000. "
            f"Strictly better than best prior phase "
            f"({best_alive_prior}/{best_recall_prior}). Capacity "
            f"gives diminishing returns; some intrinsic architectural "
            f"limit remains."
        )
    else:
        verdict = "NO_CHANGE"
        reason = (
            f"{n_alive_6g}/{n_recall_6g} — matches or worse than "
            f"prior best ({best_alive_prior}/{best_recall_prior}). "
            f"Capacity is NOT the wall. 2× substrate sizing produced "
            f"no measurable improvement. Architectural limit "
            f"empirically confirmed → S-level grouping is the next "
            f"empirically motivated step."
        )
    print(f"{verdict}: {reason}")
    print()

    # ---- Persist ----
    results_dir = _REPO_ROOT / "results" / "substrate" / "phase_6g"
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "phase_6g_results.json").open("w") as f:
        json.dump(
            {
                "config": {
                    "n_neurons": n_neurons,
                    "k_connectivity": k_connectivity,
                    "M": M,
                    "pattern_size": pattern_size,
                    "K_per_pattern": K_per_pattern,
                    "K_consolidate": K_consolidate,
                    "rho_floor": substrate.rho_floor,
                    "k_protect": substrate.k_protect,
                    "schedule": "Phase 6c interleaved (train K=80, consolidate K=1500)",
                    "seed": 42,
                },
                "history": history,
                "final": {
                    "P_counts": p_counts,
                    "recalls": recalls,
                    "bridges": final["bridges"],
                    "n_alive": n_alive_6g,
                    "n_recall": n_recall_6g,
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
    width = 0.09

    colors = ["tab:red", "tab:orange", "tab:olive", "tab:green",
              "tab:blue", "tab:purple", "tab:cyan", "tab:brown", "black"]

    for idx, name in enumerate(phase_names):
        counts = p_counts if name == "6g" else BASELINES[name][0]
        axes[0, 0].bar(x + (idx - len(phase_names) / 2) * width, counts,
                       width, label=name, color=colors[idx])
    axes[0, 0].axhline(2, color="gray", linestyle=":", alpha=0.4)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([f"P{i}" for i in range(M)])
    axes[0, 0].set_ylabel("# P entities")
    axes[0, 0].set_title("Per-pattern P count — 9 phases")
    axes[0, 0].grid(alpha=0.3, axis="y")
    axes[0, 0].legend(fontsize=8, ncol=5)

    for idx, name in enumerate(phase_names):
        recs_pct = ([r * 100 for r in recalls] if name == "6g"
                    else [r * 100 for r in BASELINES[name][1]])
        axes[0, 1].bar(x + (idx - len(phase_names) / 2) * width, recs_pct,
                       width, label=name, color=colors[idx])
    axes[0, 1].axhline(40, color="black", linestyle="--", alpha=0.4)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([f"P{i}" for i in range(M)])
    axes[0, 1].set_ylabel("recall (%)")
    axes[0, 1].set_title(f"Per-pattern recall — 9 phases: {verdict}")
    axes[0, 1].set_ylim(0, 105)
    axes[0, 1].grid(alpha=0.3, axis="y")
    axes[0, 1].legend(fontsize=8, ncol=5)

    alive_counts = [n_alive_summary[n] for n in phase_names]
    bars = axes[1, 0].bar(phase_names, alive_counts, color=colors)
    axes[1, 0].axhline(M, color="black", linestyle="--", alpha=0.4,
                       label=f"target ({M}/{M})")
    axes[1, 0].set(ylabel="patterns alive (≥2 P)",
                   title="Alive count across 9 phases",
                   ylim=(0, M + 0.5))
    for bar, v in zip(bars, alive_counts):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                        str(v), ha="center", va="bottom")
    axes[1, 0].legend()

    recall_counts = [n_recall_summary[n] for n in phase_names]
    bars = axes[1, 1].bar(phase_names, recall_counts, color=colors)
    axes[1, 1].axhline(M, color="black", linestyle="--", alpha=0.4,
                       label=f"target ({M}/{M})")
    axes[1, 1].set(ylabel="patterns with recall ≥ 40%",
                   title="Recall count across 9 phases",
                   ylim=(0, M + 0.5))
    for bar, v in zip(bars, recall_counts):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                        str(v), ha="center", va="bottom")
    axes[1, 1].legend()

    plt.suptitle(
        f"Phase 6g — capacity test at N=1000: {verdict}", fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(results_dir / "phase_6g_results.png", dpi=120)
    plt.close(fig)
    print(f"Results → {results_dir}")

    if verdict == "PASS":
        return 0
    if verdict in ("STRONG_PARTIAL", "PARTIAL"):
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
