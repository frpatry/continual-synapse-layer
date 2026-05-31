"""Phase 6c — Explicit consolidation mode between pattern trainings.

Tests candidate H6: brain-aligned learning requires THREE distinct
operational modes with different parameter regimes, not just training
and recall.

The three modes (from data Phases 5b/6a/6b have already mapped out):

  Training mode      external ON,  feedback OFF, plasticity ON
                     (Phase 6b discipline — new pattern dominates k-WTA)
  Recall  mode       external partial cue ON, feedback ON, plasticity OFF
                     (Phase 2c discipline — feedback drives completion)
  Consolidation mode external OFF, feedback ON, plasticity ON  ← NEW
                     (this experiment — attractors self-reactivate and
                      refresh their P-P / N-N pathways via spontaneous
                      Hebbian on co-firing during quiet windows)

Phase 6b showed that gating feedback during training FIXED emergence
(5/5 patterns alive) but HOLLOWED OUT older patterns' recall pathways
(P0 dropped to 0% recall despite having 2 P entities). The hypothesis:
without feedback during the long idle phases, older P entities never
co-fire spontaneously, so their P-P and N-N edges decay until they
exist as isolated structures with no recall machinery.

Phase 6c's fix: insert an explicit consolidation phase between each
pattern's training. During consolidation:
  - No external input (so newly-trained pattern isn't reinforced and
    doesn't dominate)
  - Feedback ON (so any attractor that fires from noise gets boosted
    back to its components)
  - Plasticity ON (so the spontaneous co-firing accumulates as Hebbian
    growth on the existing P-P / N-N infrastructure)

Biology analog: slow-wave sleep replay. The hippocampal-cortical
dialogue reactivates recent and remote memories under conditions of
suppressed sensory input and shifted neuromodulatory tone (low ACh,
high sleep-related coherence), refreshing weights that would
otherwise decay.

Verdict:
  PASS               5/5 emerge AND 5/5 recall ≥ 40 %
                     → H6 candidate VALIDATED. 2-level + 3-modes is
                       the minimal complete architecture at M=5.
  STRONG_PARTIAL     5/5 emerge AND ≥ 4/5 recall ≥ 40 %
                     → Consolidation works; minor tuning may close gap.
  PARTIAL            5/5 emerge AND ≥ 3/5 recall ≥ 40 %
                     → Helps but insufficient alone.
  NO_CHANGE          Similar to Phase 6b (no recall improvement)
                     → Consolidation not the right mechanism.

Outputs:
  results/substrate/phase_6c/phase_6c_results.{png,json}
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

# Re-use helpers from earlier phases.
from phase_5b_bridge_at_scale import (  # noqa: E402
    classify_p_entities,
    define_M_patterns,
    make_external,
    summarize_classification,
)


# ---------------------------------------------------------------------------
# Embedded baselines for the 4-way report
# ---------------------------------------------------------------------------

PHASE_5B = {
    "label": "5b (no floor)",
    "pattern_p_counts": [0, 3, 0, 4, 2],
    "pattern_recalls": [0.0, 0.20, 0.40, 0.80, 0.80],
    "bridges_total": 29,
}
PHASE_6A = {
    "label": "6a (floor, feedback always)",
    "pattern_p_counts": [2, 5, 1, 5, 0],
    "pattern_recalls": [0.40, 0.20, 0.60, 1.00, 0.20],
    "bridges_total": 34,
}
PHASE_6B = {
    "label": "6b (floor + feedback gated)",
    "pattern_p_counts": [2, 3, 2, 4, 2],
    "pattern_recalls": [0.0, 0.20, 0.40, 0.80, 0.80],
    "bridges_total": 30,
}


# ---------------------------------------------------------------------------
# Three operational modes
# ---------------------------------------------------------------------------


def training_mode_phase(
    substrate: Substrate,
    pattern: np.ndarray,
    K: int = 80,
    T_present: int = 15,
    T_rest: int = 60,
) -> None:
    """Mode 1: training. External=pattern ON, feedback OFF, plasticity ON.
    See Phase 6b for the discipline rationale."""
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


def consolidation_mode_phase(
    substrate: Substrate,
    K_steps: int = 1500,
) -> None:
    """Mode 3 (NEW): consolidation. External OFF, feedback ON, plasticity ON.

    Spontaneous attractor co-firing during quiet steps refreshes P-P
    and N-N weights via Hebbian. No external pattern is reinforced —
    the substrate replays whatever it already has."""
    saved = substrate.enable_feedback_p_to_n
    substrate.enable_feedback_p_to_n = True
    try:
        for _ in range(K_steps):
            substrate.step(external_input=None)
    finally:
        substrate.enable_feedback_p_to_n = saved


def recall_mode_test(
    substrate: Substrate,
    pattern: np.ndarray,
    cue_fraction: float = 0.5,
    T_settle: int = 30,
    cue_seed: int = 123,
) -> float:
    """Mode 2: recall. External partial cue, feedback ON, plasticity OFF,
    p_sparsity_target relaxed to 1.0 (Phase 2c finding)."""
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
    substrate.enable_feedback_p_to_n = True
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
    print("=== Phase 6c: Explicit consolidation mode (candidate H6) ===\n")
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
        enable_feedback_p_to_n=True,  # toggled per mode
        starting_age=0.0,
        rho_floor=0.3,
        seed=42,
    )

    patterns = define_M_patterns(n_neurons, M, pattern_size, seed=0)
    print(f"M={M} disjoint patterns of size {pattern_size} each")
    print(f"Per stage: training K={K_per_pattern} + consolidation K={K_consolidate}\n")

    history: list[dict[str, Any]] = []

    for stage in range(M):
        print(f"--- Stage {stage}: TRAINING pattern {stage} (feedback OFF) ---")
        training_mode_phase(substrate, patterns[stage], K=K_per_pattern)

        # Snapshot AFTER training, BEFORE consolidation — so we can see
        # whether consolidation actually refreshes recall.
        cls_post_train = classify_p_entities(substrate.p_entities, patterns)
        summary_post_train = summarize_classification(cls_post_train, M)

        print(f"--- Stage {stage}: CONSOLIDATION ({K_consolidate} steps, "
              f"external OFF + feedback ON + plasticity ON) ---")
        consolidation_mode_phase(substrate, K_steps=K_consolidate)

        cls_post_consol = classify_p_entities(substrate.p_entities, patterns)
        summary_post_consol = summarize_classification(cls_post_consol, M)

        # Recalls measured AFTER consolidation.
        recalls = []
        for i in range(stage + 1):
            comp = recall_mode_test(substrate, patterns[i])
            recalls.append({"pattern": i, "completion": comp})

        history.append({
            "stage": stage,
            "trained_patterns_so_far": stage + 1,
            "P_counts_post_train": summary_post_train,
            "P_counts_post_consolidation": summary_post_consol,
            "recalls": recalls,
            "system_age": substrate.system_age,
        })

        # Diagnostic: did consolidation grow P counts?
        deltas = []
        for i in range(M):
            d = (summary_post_consol[f"P_pattern_{i}"]
                 - summary_post_train[f"P_pattern_{i}"])
            if d != 0:
                deltas.append(f"P{i}+={d}" if d > 0 else f"P{i}{d}")
        delta_str = ", ".join(deltas) if deltas else "no change"

        post_train_str = " ".join(
            f"P{i}={summary_post_train[f'P_pattern_{i}']}" for i in range(M)
        )
        post_consol_str = " ".join(
            f"P{i}={summary_post_consol[f'P_pattern_{i}']}" for i in range(M)
        )
        recall_str = " ".join(
            f"P{r['pattern']}:{r['completion'] * 100:.0f}%" for r in recalls
        )

        print(f"  Post-train  P counts: {post_train_str}")
        print(f"  Post-consol P counts: {post_consol_str}  (Δ: {delta_str})")
        print(f"  Bridges total post-consol: "
              f"{summary_post_consol['P_bridges_total']}")
        print(f"  Recalls (after consolidation): {recall_str}")
        print()

    # ---- Final analysis ----
    final = history[-1]
    summary = final["P_counts_post_consolidation"]
    pattern_p_counts = [summary[f"P_pattern_{i}"] for i in range(M)]
    pattern_recalls = [r["completion"] for r in final["recalls"]]
    bridges_total = summary["P_bridges_total"]

    n_alive = sum(1 for c in pattern_p_counts if c >= 2)
    n_recall_pass = sum(1 for r in pattern_recalls if r >= 0.4)
    all_have_p = n_alive == M
    all_recall = n_recall_pass == M

    # ---- 4-way comparison table ----
    print("=== 4-way comparison ===")
    print(f"{'pattern':<10} {'5b':>15s} {'6a':>15s} {'6b':>15s} {'6c':>18s}")
    print("-" * 80)
    for i in range(M):
        p5b = f"{PHASE_5B['pattern_p_counts'][i]}P / " \
              f"{PHASE_5B['pattern_recalls'][i] * 100:.0f}%"
        p6a = f"{PHASE_6A['pattern_p_counts'][i]}P / " \
              f"{PHASE_6A['pattern_recalls'][i] * 100:.0f}%"
        p6b = f"{PHASE_6B['pattern_p_counts'][i]}P / " \
              f"{PHASE_6B['pattern_recalls'][i] * 100:.0f}%"
        p6c = f"{pattern_p_counts[i]}P / {pattern_recalls[i] * 100:.0f}%"
        print(f"P{i:<9} {p5b:>15s} {p6a:>15s} {p6b:>15s} {p6c:>18s}")
    print("-" * 80)
    n_alive_per_phase = {
        "5b": sum(1 for c in PHASE_5B["pattern_p_counts"] if c >= 2),
        "6a": sum(1 for c in PHASE_6A["pattern_p_counts"] if c >= 2),
        "6b": sum(1 for c in PHASE_6B["pattern_p_counts"] if c >= 2),
        "6c": n_alive,
    }
    n_recall_per_phase = {
        "5b": sum(1 for r in PHASE_5B["pattern_recalls"] if r >= 0.4),
        "6a": sum(1 for r in PHASE_6A["pattern_recalls"] if r >= 0.4),
        "6b": sum(1 for r in PHASE_6B["pattern_recalls"] if r >= 0.4),
        "6c": n_recall_pass,
    }
    print(f"{'alive (≥2 P)':<10} "
          f"{n_alive_per_phase['5b']:>15d} {n_alive_per_phase['6a']:>15d} "
          f"{n_alive_per_phase['6b']:>15d} {n_alive_per_phase['6c']:>18d}")
    print(f"{'recall (≥40%)':<10} "
          f"{n_recall_per_phase['5b']:>15d} {n_recall_per_phase['6a']:>15d} "
          f"{n_recall_per_phase['6b']:>15d} {n_recall_per_phase['6c']:>18d}")
    print(f"{'bridges':<10} "
          f"{PHASE_5B['bridges_total']:>15d} {PHASE_6A['bridges_total']:>15d} "
          f"{PHASE_6B['bridges_total']:>15d} {bridges_total:>18d}")
    print()

    # ---- Verdict ----
    print("=== Verdict (Phase 6c) ===")
    if all_have_p and all_recall:
        verdict = "PASS"
        reason = (
            "5/5 patterns emerge AND 5/5 recall ≥ 40 %. Candidate H6 "
            "(per-mode discipline) is VALIDATED. The 2-level substrate "
            "+ 3-mode architecture (training / recall / consolidation) "
            "is the minimal sufficient design at M=5. No S-level needed; "
            "the bottleneck across Phases 5b/6a/6b was the substrate's "
            "missing third operational mode, not architectural depth."
        )
    elif all_have_p and n_recall_pass >= 4:
        verdict = "STRONG_PARTIAL"
        weak = [
            i for i, r in enumerate(pattern_recalls) if r < 0.4
        ]
        reason = (
            f"5/5 emerge AND {n_recall_pass}/5 recall ≥ 40 %. "
            f"Pattern(s) {weak} still below recall floor. Consolidation "
            f"resolves most of Phase 6b's gap; minor tuning (longer "
            f"K_consolidate, additional consolidation cycles, or per-"
            f"pattern targeted replay) likely closes the remaining gap."
        )
    elif all_have_p and n_recall_pass >= 3:
        verdict = "PARTIAL"
        reason = (
            f"5/5 emerge but only {n_recall_pass}/5 recall ≥ 40 %. "
            f"Consolidation helps but isn't sufficient alone. May need "
            f"a structured replay schedule, separate hippocampal-like "
            f"buffer, or S-level grouping."
        )
    else:
        verdict = "NO_CHANGE"
        reason = (
            f"Recall outcome similar to Phase 6b ({n_recall_per_phase['6b']} "
            f"→ {n_recall_pass}). Spontaneous consolidation didn't refresh "
            f"the recall pathways as hypothesized — either attractors "
            f"weren't firing during consolidation (noise floor too low) "
            f"or refresh didn't reach the right structures. S-level may "
            f"be needed after all."
        )
    print(f"{verdict}: {reason}")

    # ---- Persist ----
    results_dir = _REPO_ROOT / "results" / "substrate" / "phase_6c"
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "phase_6c_results.json").open("w") as f:
        json.dump(
            {
                "config": {
                    "n_neurons": n_neurons,
                    "M": M,
                    "pattern_size": pattern_size,
                    "K_per_pattern": K_per_pattern,
                    "K_consolidate": K_consolidate,
                    "rho_floor": 0.3,
                    "training_mode": "external ON, feedback OFF, plasticity ON",
                    "consolidation_mode": "external OFF, feedback ON, plasticity ON",
                    "recall_mode": "external partial cue, feedback ON, plasticity OFF, p_sparsity=1.0",
                    "seed": 42,
                },
                "history": history,
                "phase_6c_results": {
                    "pattern_p_counts": pattern_p_counts,
                    "pattern_recalls": pattern_recalls,
                    "bridges_total": bridges_total,
                    "n_alive": n_alive,
                    "n_recall_pass": n_recall_pass,
                    "all_have_p": all_have_p,
                    "all_recall": all_recall,
                },
                "comparison": {
                    "phase_5b": PHASE_5B,
                    "phase_6a": PHASE_6A,
                    "phase_6b": PHASE_6B,
                },
                "n_alive_per_phase": n_alive_per_phase,
                "n_recall_per_phase": n_recall_per_phase,
                "verdict": verdict,
                "reason": reason,
            },
            f,
            indent=2,
        )

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    x = np.arange(M)
    width = 0.2

    axes[0].bar(x - 1.5 * width, PHASE_5B["pattern_p_counts"], width,
                color="tab:red", label="5b")
    axes[0].bar(x - 0.5 * width, PHASE_6A["pattern_p_counts"], width,
                color="tab:orange", label="6a")
    axes[0].bar(x + 0.5 * width, PHASE_6B["pattern_p_counts"], width,
                color="tab:olive", label="6b")
    axes[0].bar(x + 1.5 * width, pattern_p_counts, width,
                color="tab:green", label="6c")
    axes[0].axhline(2, color="gray", linestyle=":", alpha=0.4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"P{i}" for i in range(M)])
    axes[0].set_ylabel("# P entities per pattern")
    axes[0].set_title("Per-pattern P emergence — 4-way comparison")
    axes[0].grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=9)

    axes[1].bar(x - 1.5 * width,
                [r * 100 for r in PHASE_5B["pattern_recalls"]],
                width, color="tab:red", label="5b")
    axes[1].bar(x - 0.5 * width,
                [r * 100 for r in PHASE_6A["pattern_recalls"]],
                width, color="tab:orange", label="6a")
    axes[1].bar(x + 0.5 * width,
                [r * 100 for r in PHASE_6B["pattern_recalls"]],
                width, color="tab:olive", label="6b")
    axes[1].bar(x + 1.5 * width,
                [r * 100 for r in pattern_recalls],
                width, color="tab:green", label="6c")
    axes[1].axhline(40, color="black", linestyle="--", alpha=0.4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"P{i}" for i in range(M)])
    axes[1].set_ylabel("recall completion (%)")
    axes[1].set_title(f"Per-pattern recall — 4-way comparison: {verdict}")
    axes[1].set_ylim(0, 105)
    axes[1].grid(alpha=0.3, axis="y")
    axes[1].legend(fontsize=9)

    plt.suptitle(
        f"Phase 6c — consolidation mode (candidate H6): {verdict}",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(results_dir / "phase_6c_results.png", dpi=120)
    plt.close(fig)
    print(f"\nResults → {results_dir}")

    if verdict == "PASS":
        return 0
    if verdict in ("STRONG_PARTIAL", "PARTIAL"):
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
