"""Phase 6e' — Spaced revisits at viable per-visit dose.

Phase 6e collapsed (0/5 alive) because K=20 per visit produced
W ≈ 0.47, just below the 0.5 emergence threshold. The substrate
never crystallized anything regardless of how many visits we ran.

Phase 6e' separates the two confounded variables in 6e:
  1. CADENCE        (interleaved vs single vs spaced revisits)
  2. PER-VISIT DOSE (how much training fits in a single visit)

Same total budget as Phases 6c / 6d / 6e:
  Per-pattern training: 2 rounds × K=40 = 80      (matches baseline)
  Total consolidation:  2 × K=3750 = 7500         (matches baseline)

Difference vs 6e: K=40 per visit = 600 active steps, which puts
post-visit W around 0.85 — comfortably above the emergence
threshold. If the spaced-revisits CONCEPT works, this should reveal
it without the dose confound.

Per-visit math:
  600 active steps × ρ·η·cov ≈ 0.3 · 0.01 · 0.49 ≈ 0.00147 / step
  First-order convergence to W_eq ≈ 4.9
  W(600) ≈ 0.05 + (4.9 − 0.05) · (1 − e^(−600/3333)) ≈ 0.85   ← above 0.5

Inter-visit decay for pattern 0:
  4 other patterns × K=40 × 75 = 12000 quiet steps
  + 3750 consolidation steps with feedback ON
  = 15750 steps before next pattern-0 visit
  Decay factor (passive): 0.9997^15750 ≈ 0.0086
  W → 0.85 → 0.007 (passive)

  But consolidation phase MIGHT refresh: if attractor still has
  viable W (just above resurrection threshold) at start of
  consolidation, spontaneous firing can refresh.

The key hypothesis: even if W decays substantially between visits,
the brief above-threshold window during each visit is enough to:
  (a) emerge P entities (W > 0.5 AND passes >= 3 = trivially met
      across 40 cycles with hysteresis)
  (b) leave behind P-P / N-N infrastructure that consolidation can
      refresh

Verdict:
  PASS              5/5 alive AND 5/5 recall ≥ 40 %
                    → spaced-revisits concept WORKS at viable dose.
                    Architecture COMPLETE: 2-level + 3-mode +
                    dose-aware spaced cadence.
  STRONG_PARTIAL    5/5 alive AND 4/5 recall ≥ 40 %
  PARTIAL           ≥4 alive AND ≥3 recall
  NO_CHANGE         Outcome similar to 6c/6d → cadence isn't the
                    fix → Phase 6f (fresh-pattern lockout) is the
                    empirically motivated next step.
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
# Embedded baselines (7-way report)
# ---------------------------------------------------------------------------

BASELINES = {
    "5b": ([0, 3, 0, 4, 2], [0.0, 0.20, 0.40, 0.80, 0.80]),
    "6a": ([2, 5, 1, 5, 0], [0.40, 0.20, 0.60, 1.00, 0.20]),
    "6b": ([2, 3, 2, 4, 2], [0.0, 0.20, 0.40, 0.80, 0.80]),
    "6c": ([2, 3, 2, 4, 1], [0.40, 0.20, 0.40, 0.80, 0.80]),
    "6d": ([2, 3, 2, 4, 2], [0.0, 0.20, 0.40, 0.80, 0.80]),
    "6e": ([0, 0, 0, 0, 0], [0.0, 0.0, 0.0, 0.0, 0.0]),
}


def n_alive(counts: list[int]) -> int:
    return sum(1 for c in counts if c >= 2)


def n_recall(recalls: list[float]) -> int:
    return sum(1 for r in recalls if r >= 0.4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=== Phase 6e': Spaced revisits at viable per-visit dose ===\n")
    n_neurons = 500
    M = 5
    pattern_size = 10
    n_rounds = 2
    K_train_per_visit = 40
    K_consolidate_per_round = 3750

    total_train_per_pattern = n_rounds * K_train_per_visit
    total_consolidate = n_rounds * K_consolidate_per_round
    print(f"M={M} patterns, {n_rounds} rounds")
    print(f"Per round: train each pattern K={K_train_per_visit} "
          f"+ consolidate K={K_consolidate_per_round}")
    print(f"Per visit: {K_train_per_visit * 15} active steps "
          f"(vs Phase 6e's 300 → predicted W ≈ 0.85 vs 6e's 0.47)")
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
        for pat_idx, pattern in enumerate(patterns):
            training_mode_phase(substrate, pattern, K=K_train_per_visit)
        consolidation_mode_phase(substrate, K_steps=K_consolidate_per_round)

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

    final = history[-1]
    p_counts = final["P_counts"]
    recalls = final["recalls"]
    n_alive_6ep = n_alive(p_counts)
    n_recall_6ep = n_recall(recalls)

    # ---- 7-way comparison ----
    print("=== 7-way comparison ===")
    phase_names = ["5b", "6a", "6b", "6c", "6d", "6e", "6e'"]
    header = " ".join(f"{name:>10s}" for name in phase_names)
    print(f"{'pattern':<8} {header}")
    print("-" * (10 + len(header)))
    for i in range(M):
        cells = []
        for name in ["5b", "6a", "6b", "6c", "6d", "6e"]:
            cnt, rec = BASELINES[name]
            cells.append(f"{cnt[i]}P/{rec[i] * 100:.0f}%")
        cells.append(f"{p_counts[i]}P/{recalls[i] * 100:.0f}%")
        print(f"P{i:<7} " + " ".join(f"{c:>10s}" for c in cells))
    print("-" * (10 + len(header)))

    n_alive_summary = {
        name: n_alive(BASELINES[name][0]) for name in BASELINES
    }
    n_alive_summary["6e'"] = n_alive_6ep
    n_recall_summary = {
        name: n_recall(BASELINES[name][1]) for name in BASELINES
    }
    n_recall_summary["6e'"] = n_recall_6ep

    print(f"{'alive':<8} "
          + " ".join(
              f"{n_alive_summary[name]:>10d}" for name in phase_names
          ))
    print(f"{'recall':<8} "
          + " ".join(
              f"{n_recall_summary[name]:>10d}" for name in phase_names
          ))
    print()

    # ---- Verdict ----
    print("=== Verdict (Phase 6e') ===")
    if n_alive_6ep == M and n_recall_6ep == M:
        verdict = "PASS"
        reason = (
            "5/5 alive AND 5/5 recall. The spaced-revisits CONCEPT "
            "works at viable per-visit dose. Phase 6e's failure was a "
            "parameter (K=20 too short), not a principle. Architecture "
            "COMPLETE at M=5: 2-level + 3-mode + dose-aware spaced "
            "cadence. The biological learning principle (distributed "
            "practice > massed practice) translates faithfully to this "
            "substrate when each session is individually viable."
        )
    elif n_alive_6ep == M and n_recall_6ep == M - 1:
        verdict = "STRONG_PARTIAL"
        weak = [i for i, r in enumerate(recalls) if r < 0.4]
        reason = (
            f"5/5 alive AND {n_recall_6ep}/5 recall. Pattern(s) {weak} "
            f"below recall floor. Very close to PASS. The cadence "
            f"concept is partially validated; minor tuning likely closes "
            f"the gap."
        )
    elif n_alive_6ep >= 4 and n_recall_6ep >= 3:
        verdict = "PARTIAL"
        weak = [i for i, r in enumerate(recalls) if r < 0.4]
        reason = (
            f"{n_alive_6ep} alive AND {n_recall_6ep} recall. Pattern(s) "
            f"{weak} still weak. Improvement over Phase 6e's collapse "
            f"shows dose threshold matters, but cadence alone isn't "
            f"closing the gap to PASS."
        )
    elif (n_recall_6ep == n_recall_summary["6c"]
          or n_recall_6ep == n_recall_summary["6d"]):
        verdict = "NO_CHANGE"
        reason = (
            f"Same recall count as Phase 6c/6d. Cadence (even at "
            f"viable per-visit dose) doesn't fix the recall problem. "
            f"Phase 6f (fresh-pattern lockout, requires src changes) "
            f"is now empirically motivated as the next step."
        )
    else:
        verdict = "MIXED"
        reason = (
            f"{n_alive_6ep} alive / {n_recall_6ep} recall — outside "
            f"clean verdict buckets."
        )
    print(f"{verdict}: {reason}")
    print()

    # ---- Persist ----
    results_dir = _REPO_ROOT / "results" / "substrate" / "phase_6e_prime"
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "phase_6e_prime_results.json").open("w") as f:
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
                        "round-robin: 2 rounds × (train each pattern K=40, "
                        "then consolidate K=3750). Per-visit active steps = "
                        "600 (vs Phase 6e's 300), placing post-visit W ≈ "
                        "0.85 above the 0.5 emergence threshold."
                    ),
                    "seed": 42,
                },
                "per_round_history": history,
                "final": {
                    "P_counts": p_counts,
                    "recalls": recalls,
                    "bridges": final["bridges"],
                    "n_alive": n_alive_6ep,
                    "n_recall": n_recall_6ep,
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
    pattern_colors = plt.cm.tab10.colors[:M]

    # Top-left: per-pattern P count across rounds.
    for i in range(M):
        counts = [h["P_counts"][i] for h in history]
        axes[0, 0].plot(rounds, counts, "o-",
                        color=pattern_colors[i], label=f"P{i}")
    axes[0, 0].axhline(2, color="gray", linestyle=":", alpha=0.4)
    axes[0, 0].set(xlabel="round", ylabel="# P entities",
                   title="Per-pattern P count across rounds")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    # Top-right: per-pattern recall across rounds.
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

    # Bottom-left: alive count 7-way.
    colors = ["tab:red", "tab:orange", "tab:olive", "tab:green",
              "tab:blue", "tab:purple", "tab:cyan"]
    alive_counts = [n_alive_summary[name] for name in phase_names]
    bars = axes[1, 0].bar(phase_names, alive_counts, color=colors)
    axes[1, 0].axhline(M, color="black", linestyle="--", alpha=0.4,
                       label=f"target ({M}/{M})")
    axes[1, 0].set(ylabel="patterns alive (≥2 P)",
                   title="Alive count across all 7 phases",
                   ylim=(0, M + 0.5))
    for bar, v in zip(bars, alive_counts):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                        str(v), ha="center", va="bottom")
    axes[1, 0].legend()

    # Bottom-right: recall count 7-way.
    recall_counts = [n_recall_summary[name] for name in phase_names]
    bars = axes[1, 1].bar(phase_names, recall_counts, color=colors)
    axes[1, 1].axhline(M, color="black", linestyle="--", alpha=0.4,
                       label=f"target ({M}/{M})")
    axes[1, 1].set(ylabel="patterns with recall ≥ 40%",
                   title="Recall count across all 7 phases",
                   ylim=(0, M + 0.5))
    for bar, v in zip(bars, recall_counts):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                        str(v), ha="center", va="bottom")
    axes[1, 1].legend()

    plt.suptitle(
        f"Phase 6e' — spaced revisits at viable dose: {verdict}",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(results_dir / "phase_6e_prime_results.png", dpi=120)
    plt.close(fig)
    print(f"Results → {results_dir}")

    if verdict == "PASS":
        return 0
    if verdict in ("STRONG_PARTIAL", "PARTIAL", "MIXED"):
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
