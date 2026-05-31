"""Phase 6b — Feedback gating during new-pattern training.

Tests the specific hypothesis that Phase 6a's new bottleneck (later
patterns can't emerge against established-attractor feedback) is
caused by the feedback channel itself, not by anything more
fundamental in the architecture.

Phase 6a's failure analysis:
  * Floor=0.3 kept older patterns viable.
  * Older P entities fired spontaneously during stage-N training.
  * P→N feedback (γ=1.0) boosted older N to high activation.
  * Older N crowded pattern-N out of k-WTA.
  * Pattern-N internal pairs rarely co-fired → no emergence.

If that mechanism is right, then DISABLING feedback during the
substrate's training mode (while keeping it on for recall) should
unblock late-pattern emergence:
  * Older P still fire spontaneously, but their activation no longer
    propagates back to N → older N stay quiescent during training.
  * Pattern-N face no in-substrate competition.
  * Pattern-N internal pairs co-fire reliably → emerge as expected.

Biology analog: acetylcholine modulates top-down feedback during
attentive encoding, suppressing recurrent reactivation so the
sensory stream dominates new learning.

Protocol: identical to Phase 5b / 6a, except
``substrate.enable_feedback_p_to_n = False`` for every training step
and ``True`` for every recall step.

Verdict:
  PASS         5/5 patterns have ≥2 P AND all recall ≥40 %.
               Feedback gating IS the fix → per-mode parameter
               discipline generalizes (training γ=0, recall γ=1.0).
               NO S LEVEL NEEDED at M=5.
  PARTIAL      ≥4 patterns alive (improvement over 6a's 3) but
               not full recovery. Feedback partially responsible.
  NO_CHANGE    ≤3 patterns alive — feedback NOT the bottleneck.
               S level (Phase 6c) empirically motivated.

Outputs:
  results/substrate/phase_6b/phase_6b_results.{png,json}
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

# Re-use Phase 5b helpers — same protocol, only the training feedback
# differs. Recall test redefined here to keep feedback explicitly ON.
from phase_5b_bridge_at_scale import (  # noqa: E402
    classify_p_entities,
    define_M_patterns,
    make_external,
    summarize_classification,
)


# ---------------------------------------------------------------------------
# Embedded baselines for side-by-side report
# ---------------------------------------------------------------------------

PHASE_5B = {
    "label": "5b (no floor)",
    "pattern_p_counts": [0, 3, 0, 4, 2],
    "pattern_recalls": [0.0, 0.20, 0.40, 0.80, 0.80],
    "bridges_total": 29,
}

PHASE_6A = {
    "label": "6a (floor=0.3, feedback always)",
    "pattern_p_counts": [2, 5, 1, 5, 0],
    "pattern_recalls": [0.40, 0.20, 0.60, 1.00, 0.20],
    "bridges_total": 34,
}


# ---------------------------------------------------------------------------
# Training with feedback gated OFF
# ---------------------------------------------------------------------------


def train_one_pattern_no_feedback(
    substrate: Substrate,
    pattern: np.ndarray,
    K: int = 80,
    T_present: int = 15,
    T_rest: int = 60,
) -> None:
    """Train a single pattern with P→N feedback temporarily disabled,
    then restore the substrate's prior feedback setting."""
    saved = substrate.enable_feedback_p_to_n
    substrate.enable_feedback_p_to_n = False
    try:
        external = make_external(substrate.n_neurons, pattern, strength=0.7)
        for _ in range(K):
            for _ in range(T_present):
                substrate.step(external_input=external)
            for _ in range(T_rest):
                substrate.step(external_input=None)
    finally:
        substrate.enable_feedback_p_to_n = saved


def test_individual_recall_with_feedback(
    substrate: Substrate,
    pattern: np.ndarray,
    cue_fraction: float = 0.5,
    T_settle: int = 30,
    cue_seed: int = 123,
) -> float:
    """Partial-recall test that forces feedback ON during readout.

    Mirrors Phase 5b's test but with explicit feedback enabling; the
    substrate's current ``enable_feedback_p_to_n`` is saved + restored
    so toggling during the experiment doesn't bleed across calls."""
    rng = np.random.default_rng(cue_seed)
    n_cue = max(1, int(len(pattern) * cue_fraction))
    cue_indices = np.sort(rng.choice(pattern, size=n_cue, replace=False))
    cue_set = {int(x) for x in cue_indices}
    target_indices = np.array(
        [int(n) for n in pattern if int(n) not in cue_set], dtype=int,
    )
    if len(target_indices) == 0:
        return 0.0

    saved_eta = substrate.eta
    saved_eta_pp = substrate.eta_pp
    saved_p_sparsity = substrate.p_sparsity_target
    saved_feedback = substrate.enable_feedback_p_to_n
    saved_acts = substrate.activations.copy()
    saved_p_acts = {pid: p.activation for pid, p in substrate.p_entities.items()}

    substrate.eta = 0.0
    substrate.eta_pp = 0.0
    substrate.p_sparsity_target = 1.0
    substrate.enable_feedback_p_to_n = True  # recall always uses feedback
    substrate.activations = np.zeros(substrate.n_neurons, dtype=np.float32)
    for p in substrate.p_entities.values():
        p.activation = 0.0

    cue_input = np.zeros(substrate.n_neurons, dtype=np.float32)
    cue_input[cue_indices] = 1.0
    for _ in range(T_settle):
        substrate.step(external_input=cue_input)

    target_acts = substrate.activations[target_indices]
    completion = float((target_acts > 0.1).mean())

    substrate.eta = saved_eta
    substrate.eta_pp = saved_eta_pp
    substrate.p_sparsity_target = saved_p_sparsity
    substrate.enable_feedback_p_to_n = saved_feedback
    substrate.activations = saved_acts
    for pid, act in saved_p_acts.items():
        if pid in substrate.p_entities:
            substrate.p_entities[pid].activation = act
    return completion


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=== Phase 6b: Feedback gating during new-pattern training ===\n")
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
        enable_feedback_p_to_n=True,  # default; toggled per-step
        starting_age=0.0,
        rho_floor=0.3,                # Phase 6a default
        seed=42,
    )

    patterns = define_M_patterns(n_neurons, M, pattern_size, seed=0)
    print(f"Substrate: N={n_neurons}, rho_floor=0.3, "
          f"feedback OFF during training, ON during recall")
    print(f"M={M} disjoint patterns of size {pattern_size} each.\n")

    history: list[dict[str, Any]] = []

    for stage in range(M):
        print(f"--- Stage {stage}: train pattern {stage} (feedback OFF) ---")
        train_one_pattern_no_feedback(substrate, patterns[stage],
                                      K=K_per_pattern)

        classification = classify_p_entities(substrate.p_entities, patterns)
        summary = summarize_classification(classification, M)

        recalls = []
        for i in range(stage + 1):
            comp = test_individual_recall_with_feedback(substrate, patterns[i])
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
        print(f"  Bridges:  total={summary['P_bridges_total']}")
        recall_str = " ".join(
            f"P{r['pattern']}:{r['completion'] * 100:.0f}%" for r in recalls
        )
        print(f"  Recalls:  {recall_str}")
        print()

    # ---- Final analysis ----
    final = history[-1]
    summary = final["P_counts"]
    pattern_p_counts = [summary[f"P_pattern_{i}"] for i in range(M)]
    pattern_recalls = [r["completion"] for r in final["recalls"]]
    bridges_total = summary["P_bridges_total"]

    n_alive = sum(1 for c in pattern_p_counts if c >= 2)
    all_have_p = n_alive == M
    all_recall = all(r >= 0.4 for r in pattern_recalls)

    # ---- Side-by-side table ----
    print("=== Side-by-side comparison: 5b vs 6a vs 6b ===")
    print(f"{'pattern':<10} {'5b (no floor)':>15s} "
          f"{'6a (floor)':>15s} {'6b (floor+gate)':>18s}")
    print("-" * 65)
    for i in range(M):
        p5b = f"{PHASE_5B['pattern_p_counts'][i]}P / " \
              f"{PHASE_5B['pattern_recalls'][i] * 100:.0f}%"
        p6a = f"{PHASE_6A['pattern_p_counts'][i]}P / " \
              f"{PHASE_6A['pattern_recalls'][i] * 100:.0f}%"
        p6b = f"{pattern_p_counts[i]}P / {pattern_recalls[i] * 100:.0f}%"
        print(f"P{i:<9} {p5b:>15s} {p6a:>15s} {p6b:>18s}")
    print("-" * 65)
    print(f"{'alive':<10} {sum(1 for c in PHASE_5B['pattern_p_counts'] if c >= 2):>15d} "
          f"{sum(1 for c in PHASE_6A['pattern_p_counts'] if c >= 2):>15d} "
          f"{n_alive:>18d}")
    print(f"{'bridges':<10} {PHASE_5B['bridges_total']:>15d} "
          f"{PHASE_6A['bridges_total']:>15d} {bridges_total:>18d}")
    print()

    # ---- Verdict ----
    print("=== Verdict (Phase 6b) ===")
    n_alive_5b = sum(1 for c in PHASE_5B["pattern_p_counts"] if c >= 2)
    n_alive_6a = sum(1 for c in PHASE_6A["pattern_p_counts"] if c >= 2)

    if all_have_p and all_recall:
        verdict = "PASS"
        reason = (
            "5/5 patterns emerge AND recall under feedback gating. The "
            "Phase 6a bottleneck WAS feedback dominance — per-mode "
            "parameter discipline (training γ=0, recall γ=1.0) is the "
            "architectural fix. 2-level substrate suffices at M=5; NO "
            "S-level needed for compositional scaling. Biology analog: "
            "acetylcholine-mediated suppression of top-down feedback "
            "during attentive encoding is a real, working principle."
        )
    elif n_alive > n_alive_6a:
        verdict = "PARTIAL"
        recovered = [
            i for i in range(M)
            if PHASE_6A["pattern_p_counts"][i] < 2 and pattern_p_counts[i] >= 2
        ]
        still_lost = [i for i in range(M) if pattern_p_counts[i] < 2]
        reason = (
            f"Improvement over Phase 6a ({n_alive_6a} → {n_alive} alive). "
            f"Patterns {recovered} recovered; patterns {still_lost} still "
            f"missing. Feedback gating is PART of the fix; remaining gap "
            f"suggests additional architectural changes (S level, "
            f"per-pattern age reset, or scheduled replay) may help."
        )
    elif n_alive == n_alive_6a:
        verdict = "NO_CHANGE"
        reason = (
            f"Same count alive as Phase 6a ({n_alive_6a}). Feedback gating "
            f"is NOT the dominant bottleneck — the obstacle to "
            f"compositional scaling lies elsewhere. Empirically motivates "
            f"Phase 6c (S-level grouping) as the next architectural move."
        )
    else:
        verdict = "REGRESSION"
        reason = (
            f"Phase 6b ({n_alive} alive) is WORSE than 6a ({n_alive_6a}). "
            f"Suggests feedback during training was contributing positively "
            f"to emergence — perhaps by reinforcing pattern N via "
            f"self-feedback during their own training window. Counter-"
            f"intuitive; consider analyzing per-pattern P trajectories."
        )
    print(f"{verdict}: {reason}")

    # ---- Persist ----
    results_dir = _REPO_ROOT / "results" / "substrate" / "phase_6b"
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "phase_6b_results.json").open("w") as f:
        json.dump(
            {
                "config": {
                    "n_neurons": n_neurons,
                    "M": M,
                    "pattern_size": pattern_size,
                    "K_per_pattern": K_per_pattern,
                    "rho_floor": 0.3,
                    "feedback_during_training": False,
                    "feedback_during_recall": True,
                    "seed": 42,
                },
                "history": history,
                "phase_6b_results": {
                    "pattern_p_counts": pattern_p_counts,
                    "pattern_recalls": pattern_recalls,
                    "bridges_total": bridges_total,
                    "n_alive": n_alive,
                    "all_have_p": all_have_p,
                    "all_recall": all_recall,
                },
                "comparison": {
                    "phase_5b": PHASE_5B,
                    "phase_6a": PHASE_6A,
                },
                "verdict": verdict,
                "reason": reason,
            },
            f,
            indent=2,
        )

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(M)
    width = 0.25

    axes[0].bar(x - width, PHASE_5B["pattern_p_counts"], width,
                color="tab:red", label="5b (no floor)")
    axes[0].bar(x, PHASE_6A["pattern_p_counts"], width,
                color="tab:orange", label="6a (floor)")
    axes[0].bar(x + width, pattern_p_counts, width,
                color="tab:green", label="6b (floor + gate)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"P{i}" for i in range(M)])
    axes[0].set_ylabel("# P entities per pattern")
    axes[0].set_title("P emergence across phases")
    axes[0].axhline(2, color="gray", linestyle=":", alpha=0.4,
                    label="≥2 alive threshold")
    axes[0].grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)

    axes[1].bar(x - width,
                [r * 100 for r in PHASE_5B["pattern_recalls"]],
                width, color="tab:red", label="5b")
    axes[1].bar(x,
                [r * 100 for r in PHASE_6A["pattern_recalls"]],
                width, color="tab:orange", label="6a")
    axes[1].bar(x + width,
                [r * 100 for r in pattern_recalls],
                width, color="tab:green", label="6b")
    axes[1].axhline(40, color="black", linestyle="--", alpha=0.4,
                    label="recall floor")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"P{i}" for i in range(M)])
    axes[1].set_ylabel("recall completion (%)")
    axes[1].set_title(f"Recall comparison — Phase 6b: {verdict}")
    axes[1].set_ylim(0, 105)
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].legend(fontsize=8)

    plt.suptitle(
        f"Phase 6b — feedback gating: {verdict}", fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(results_dir / "phase_6b_results.png", dpi=120)
    plt.close(fig)
    print(f"\nResults → {results_dir}")

    if verdict == "PASS":
        return 0
    if verdict in ("PARTIAL", "NO_CHANGE"):
        return 1
    return 2  # REGRESSION


if __name__ == "__main__":
    raise SystemExit(main())
