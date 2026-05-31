"""Phase 6d — Post-hoc consolidation only.

Tests the cadence-refinement hypothesis from Phase 6c's analysis:
instead of interleaving consolidation after every pattern (which
created fresh-vs-mature competition that cost pattern 4 one P
entity), run ONE consolidation phase after all M patterns have been
trained. Total consolidation budget held constant vs Phase 6c
(7500 = 5 × 1500), so the only variable is the *timing*.

Biology framing: brains don't consolidate all daily experiences
equally. During sleep, replay is dominated by the strongest
attractor traces; weaker / less-tagged memories may not get
refreshed and may be lost. This is an EMERGENT selectivity — no
explicit gating mechanism, just attractor strength competing for
spontaneous reactivation.

Pre-experiment predictions (worth recording so we can compare):
  * After all training (no consolidation), the substrate should look
    like Phase 6b's post-training state: ~5 patterns emerged, recall
    pathways hollowed for the oldest patterns. P0 recall likely 0 %
    going in.
  * Post-hoc consolidation gives the SAME total budget as 6c but
    applied at the end. Two regimes are possible:
      (i)  ALL patterns get enough refresh time → 5/5 alive + recall
      (ii) Strongest (most recent) patterns dominate spontaneous
           reactivation → emergent selectivity, weakest patterns
           don't recover. The decay math suggests (ii) is likely:
           by stage 4, P0's N-N pathways have been pure-decaying for
           30000 steps with no refresh window. With ρ_floor=0.3 and
           λ=0.001, geometric decay factor ≈ 0.9997 per step, so
           after 30000 steps W ≈ original × 0.0001 — likely below
           the viability threshold + below the propagation threshold
           that lets spontaneous attractor reactivation kick in.

Verdict:
  PASS                             5/5 alive + 5/5 recall ≥ 40 %
  STRONG_PARTIAL                   5/5 alive + 4/5 recall ≥ 40 %
  PARTIAL_emergent_selectivity     ≥4 recall ≥ 40 % but not 5 — and
                                   weakest = oldest (biology working
                                   as theorized: weakest memories
                                   aren't consolidated)
  PARTIAL                          3 recall ≥ 40 %
  NO_CHANGE                        same as 6b (cadence change didn't help)

Outputs:
  results/substrate/phase_6d/phase_6d_results.{png,json}
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
# Embedded baselines (5-way report)
# ---------------------------------------------------------------------------

PHASE_5B = {
    "pattern_p_counts": [0, 3, 0, 4, 2],
    "pattern_recalls": [0.0, 0.20, 0.40, 0.80, 0.80],
}
PHASE_6A = {
    "pattern_p_counts": [2, 5, 1, 5, 0],
    "pattern_recalls": [0.40, 0.20, 0.60, 1.00, 0.20],
}
PHASE_6B = {
    "pattern_p_counts": [2, 3, 2, 4, 2],
    "pattern_recalls": [0.0, 0.20, 0.40, 0.80, 0.80],
}
PHASE_6C = {
    "pattern_p_counts": [2, 3, 2, 4, 1],
    "pattern_recalls": [0.40, 0.20, 0.40, 0.80, 0.80],
}


def n_alive(counts: list[int]) -> int:
    return sum(1 for c in counts if c >= 2)


def n_recall_pass(recalls: list[float]) -> int:
    return sum(1 for r in recalls if r >= 0.4)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=== Phase 6d: Post-hoc consolidation only ===\n")
    n_neurons = 500
    M = 5
    pattern_size = 10
    K_per_pattern = 80
    K_consolidate_total = 7500  # same total budget as Phase 6c (5 × 1500)

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
    print(f"M={M} patterns, K_train={K_per_pattern} each, "
          f"post-hoc K_consolidate={K_consolidate_total}\n")

    # ---- Phase A: train all M patterns sequentially, NO consolidation ----
    print("Phase A: sequential training (feedback OFF, NO interleaved consolidation)")
    training_history: list[dict[str, Any]] = []
    for stage in range(M):
        print(f"  Stage {stage}: train pattern {stage} (K={K_per_pattern})")
        training_mode_phase(substrate, patterns[stage], K=K_per_pattern)
        classification = classify_p_entities(substrate.p_entities, patterns)
        summary = summarize_classification(classification, M)
        training_history.append({"stage": stage, "P_counts": summary})
        counts = [summary[f"P_pattern_{i}"] for i in range(M)]
        print(f"    P counts: {counts}, bridges: {summary['P_bridges_total']}")
    print()

    # ---- Pre-consolidation snapshot ----
    pre_classification = classify_p_entities(substrate.p_entities, patterns)
    pre_summary = summarize_classification(pre_classification, M)
    pre_counts = [pre_summary[f"P_pattern_{i}"] for i in range(M)]
    pre_bridges = pre_summary["P_bridges_total"]
    print(f"Pre-consolidation snapshot:")
    print(f"  P counts per pattern: {pre_counts}")
    print(f"  Total bridges:        {pre_bridges}")
    pre_recalls = [recall_mode_test(substrate, p) for p in patterns]
    print(f"  Pre-consol recalls:   "
          f"{[f'{r * 100:.0f}%' for r in pre_recalls]}")
    print()

    # ---- Phase B: single post-hoc consolidation ----
    print(f"Phase B: post-hoc CONSOLIDATION ({K_consolidate_total} steps, "
          f"external OFF + feedback ON + plasticity ON)")
    consolidation_mode_phase(substrate, K_steps=K_consolidate_total)
    print()

    # ---- Post-consolidation measurement ----
    post_classification = classify_p_entities(substrate.p_entities, patterns)
    post_summary = summarize_classification(post_classification, M)
    post_counts = [post_summary[f"P_pattern_{i}"] for i in range(M)]
    post_bridges = post_summary["P_bridges_total"]
    print(f"Post-consolidation snapshot:")
    print(f"  P counts per pattern: {post_counts}")
    print(f"  Total bridges:        {post_bridges}")
    post_recalls = [recall_mode_test(substrate, p) for p in patterns]
    print(f"  Post-consol recalls:  "
          f"{[f'{r * 100:.0f}%' for r in post_recalls]}")
    print()

    # ---- Δ from consolidation ----
    delta_counts = [post_counts[i] - pre_counts[i] for i in range(M)]
    delta_recalls = [post_recalls[i] - pre_recalls[i] for i in range(M)]
    print("Δ from post-hoc consolidation:")
    print(f"  P-count changes: {delta_counts}")
    print(f"  Recall changes:  "
          f"{[f'{d * 100:+.0f}pp' for d in delta_recalls]}")
    print(f"  Bridge change:   {post_bridges - pre_bridges:+d}")
    print()

    # ---- 5-way comparison ----
    print("=== 5-way comparison ===")
    print(f"{'pattern':<10} {'5b':>13s} {'6a':>13s} {'6b':>13s} "
          f"{'6c':>13s} {'6d':>15s}")
    print("-" * 80)
    for i in range(M):
        def fmt(d: dict, i: int) -> str:
            c = d["pattern_p_counts"][i]
            r = d["pattern_recalls"][i]
            return f"{c}P / {r * 100:.0f}%"
        p6d = f"{post_counts[i]}P / {post_recalls[i] * 100:.0f}%"
        print(f"P{i:<9} {fmt(PHASE_5B, i):>13s} {fmt(PHASE_6A, i):>13s} "
              f"{fmt(PHASE_6B, i):>13s} {fmt(PHASE_6C, i):>13s} {p6d:>15s}")
    print("-" * 80)

    n_alive_by_phase = {
        "5b": n_alive(PHASE_5B["pattern_p_counts"]),
        "6a": n_alive(PHASE_6A["pattern_p_counts"]),
        "6b": n_alive(PHASE_6B["pattern_p_counts"]),
        "6c": n_alive(PHASE_6C["pattern_p_counts"]),
        "6d": n_alive(post_counts),
    }
    n_recall_by_phase = {
        "5b": n_recall_pass(PHASE_5B["pattern_recalls"]),
        "6a": n_recall_pass(PHASE_6A["pattern_recalls"]),
        "6b": n_recall_pass(PHASE_6B["pattern_recalls"]),
        "6c": n_recall_pass(PHASE_6C["pattern_recalls"]),
        "6d": n_recall_pass(post_recalls),
    }
    print(f"{'alive (≥2 P)':<10} "
          f"{n_alive_by_phase['5b']:>13d} {n_alive_by_phase['6a']:>13d} "
          f"{n_alive_by_phase['6b']:>13d} {n_alive_by_phase['6c']:>13d} "
          f"{n_alive_by_phase['6d']:>15d}")
    print(f"{'recall (≥40%)':<10} "
          f"{n_recall_by_phase['5b']:>13d} {n_recall_by_phase['6a']:>13d} "
          f"{n_recall_by_phase['6b']:>13d} {n_recall_by_phase['6c']:>13d} "
          f"{n_recall_by_phase['6d']:>15d}")
    print()

    # ---- Verdict ----
    print("=== Verdict (Phase 6d) ===")
    n_alive_6d = n_alive_by_phase["6d"]
    n_recall_6d = n_recall_by_phase["6d"]

    if n_alive_6d == M and n_recall_6d == M:
        verdict = "PASS"
        reason = (
            "5/5 alive AND 5/5 recall. Post-hoc cadence fully resolves "
            "the Phase 6c interleaving cost. The 2-level + 3-mode "
            "architecture with end-of-training consolidation is "
            "complete at M=5. NO S-level required."
        )
    elif n_alive_6d == M and n_recall_6d == M - 1:
        verdict = "STRONG_PARTIAL"
        weak = [i for i, r in enumerate(post_recalls) if r < 0.4]
        reason = (
            f"5/5 alive, {n_recall_6d}/5 recall. Pattern(s) {weak} below "
            f"the recall floor. Close to PASS; longer K_consolidate or "
            f"targeted replay could close the gap."
        )
    elif n_recall_6d >= 4:
        # Check whether the weakest-recall patterns are also the oldest:
        # that's the biological-selectivity signature.
        weak_pattern_idxs = [
            i for i, r in enumerate(post_recalls) if r < 0.4
        ]
        all_weak_are_old = (
            len(weak_pattern_idxs) > 0
            and max(weak_pattern_idxs) <= M // 2
        )
        if all_weak_are_old:
            verdict = "PARTIAL_emergent_selectivity"
            reason = (
                f"{n_alive_6d}/5 alive, {n_recall_6d}/5 recall. The "
                f"under-performers are the OLDEST patterns "
                f"({weak_pattern_idxs}) — consistent with the biological "
                f"hypothesis that strongest (most recent / most tagged) "
                f"attractors dominate spontaneous reactivation, leaving "
                f"the weakest behind. Not a clean PASS but architecturally "
                f"defensible."
            )
        else:
            verdict = "PARTIAL"
            reason = (
                f"{n_alive_6d}/5 alive, {n_recall_6d}/5 recall. "
                f"Pattern(s) {weak_pattern_idxs} weak but not strictly "
                f"the oldest — selectivity not purely age-driven."
            )
    elif n_recall_6d == n_recall_by_phase["6c"]:
        verdict = "NO_CHANGE"
        reason = (
            f"Same recall count as Phase 6c ({n_recall_by_phase['6c']}). "
            f"Cadence change didn't help. Try Option 2 (stabilization "
            f"gap) or Option 3 (targeted per-pattern consolidation)."
        )
    else:
        verdict = "MIXED"
        reason = (
            f"{n_alive_6d} alive / {n_recall_6d} recall — neither "
            f"matches a clean verdict bucket."
        )
    print(f"{verdict}: {reason}")
    print()

    # ---- Persist ----
    results_dir = _REPO_ROOT / "results" / "substrate" / "phase_6d"
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "phase_6d_results.json").open("w") as f:
        json.dump(
            {
                "config": {
                    "n_neurons": n_neurons,
                    "M": M,
                    "pattern_size": pattern_size,
                    "K_per_pattern": K_per_pattern,
                    "K_consolidate_post_hoc": K_consolidate_total,
                    "rho_floor": 0.3,
                    "schedule": "all trainings first, single post-hoc "
                                "consolidation; total consolidation budget "
                                "= Phase 6c's interleaved 5 × 1500",
                    "seed": 42,
                },
                "training_history": training_history,
                "pre_consolidation": {
                    "P_counts": pre_counts,
                    "recalls": pre_recalls,
                    "bridges": pre_bridges,
                },
                "post_consolidation": {
                    "P_counts": post_counts,
                    "recalls": post_recalls,
                    "bridges": post_bridges,
                },
                "delta_from_consolidation": {
                    "P_counts": delta_counts,
                    "recalls": delta_recalls,
                    "bridges": post_bridges - pre_bridges,
                },
                "comparison": {
                    "phase_5b": PHASE_5B,
                    "phase_6a": PHASE_6A,
                    "phase_6b": PHASE_6B,
                    "phase_6c": PHASE_6C,
                    "phase_6d": {
                        "pattern_p_counts": post_counts,
                        "pattern_recalls": post_recalls,
                    },
                },
                "n_alive_by_phase": n_alive_by_phase,
                "n_recall_by_phase": n_recall_by_phase,
                "verdict": verdict,
                "reason": reason,
            },
            f,
            indent=2,
        )

    # ---- Plot ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    x = np.arange(M)
    width = 0.15

    # Top-left: P counts 5-way.
    axes[0, 0].bar(x - 2 * width, PHASE_5B["pattern_p_counts"], width,
                   color="tab:red", label="5b")
    axes[0, 0].bar(x - 1 * width, PHASE_6A["pattern_p_counts"], width,
                   color="tab:orange", label="6a")
    axes[0, 0].bar(x, PHASE_6B["pattern_p_counts"], width,
                   color="tab:olive", label="6b")
    axes[0, 0].bar(x + 1 * width, PHASE_6C["pattern_p_counts"], width,
                   color="tab:green", label="6c")
    axes[0, 0].bar(x + 2 * width, post_counts, width,
                   color="tab:blue", label="6d")
    axes[0, 0].axhline(2, color="gray", linestyle=":", alpha=0.4)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([f"P{i}" for i in range(M)])
    axes[0, 0].set_ylabel("# P entities")
    axes[0, 0].set_title("Per-pattern P count — 5-way")
    axes[0, 0].grid(alpha=0.3, axis="y")
    axes[0, 0].legend(fontsize=8, ncol=5)

    # Top-right: recall 5-way.
    axes[0, 1].bar(x - 2 * width,
                   [r * 100 for r in PHASE_5B["pattern_recalls"]],
                   width, color="tab:red", label="5b")
    axes[0, 1].bar(x - 1 * width,
                   [r * 100 for r in PHASE_6A["pattern_recalls"]],
                   width, color="tab:orange", label="6a")
    axes[0, 1].bar(x,
                   [r * 100 for r in PHASE_6B["pattern_recalls"]],
                   width, color="tab:olive", label="6b")
    axes[0, 1].bar(x + 1 * width,
                   [r * 100 for r in PHASE_6C["pattern_recalls"]],
                   width, color="tab:green", label="6c")
    axes[0, 1].bar(x + 2 * width,
                   [r * 100 for r in post_recalls],
                   width, color="tab:blue", label="6d")
    axes[0, 1].axhline(40, color="black", linestyle="--", alpha=0.4)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([f"P{i}" for i in range(M)])
    axes[0, 1].set_ylabel("recall (%)")
    axes[0, 1].set_title(f"Recall — 5-way: {verdict}")
    axes[0, 1].set_ylim(0, 105)
    axes[0, 1].grid(alpha=0.3, axis="y")
    axes[0, 1].legend(fontsize=8, ncol=5)

    # Bottom-left: ΔP from consolidation.
    axes[1, 0].bar(x, delta_counts, color="tab:purple")
    axes[1, 0].axhline(0, color="black", linewidth=0.8)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([f"P{i}" for i in range(M)])
    axes[1, 0].set_ylabel("Δ P count")
    axes[1, 0].set_title("Effect of post-hoc consolidation on P count")
    axes[1, 0].grid(alpha=0.3, axis="y")

    # Bottom-right: Δrecall from consolidation.
    axes[1, 1].bar(x, [d * 100 for d in delta_recalls], color="tab:cyan")
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([f"P{i}" for i in range(M)])
    axes[1, 1].set_ylabel("Δ recall (pp)")
    axes[1, 1].set_title("Effect of post-hoc consolidation on recall")
    axes[1, 1].grid(alpha=0.3, axis="y")

    plt.suptitle(
        f"Phase 6d — post-hoc consolidation only: {verdict}",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(results_dir / "phase_6d_results.png", dpi=120)
    plt.close(fig)
    print(f"Results → {results_dir}")

    if verdict == "PASS":
        return 0
    if verdict in ("STRONG_PARTIAL", "PARTIAL_emergent_selectivity",
                   "PARTIAL", "MIXED"):
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
