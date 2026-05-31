"""Phase 5a — Compositional generalization (P5, spatial form).

The hard question: after the substrate has learned pattern A alone, then
pattern B alone (never A+B together), can it represent A and B
*simultaneously* when both are presented as a combined cue?

PASS would show that the 2-level (N + P + P-P + P→N) architecture
supports compositional representation without any S-level (group)
machinery. FAIL motivates Phase 5b (introduce S).

Protocol
--------
1. Substrate (N=300, k=30; sparsity_target gives k-WTA=15 winners).
2. Two DISJOINT patterns A and B, each of size 15.
3. Train pattern A alone for K=100 presentations.
4. Train pattern B alone for K=100 presentations.
   * A is never re-presented; this is the catastrophic-forgetting risk.
5. Sanity: report whether A's P entities survived B training.
6. Test: with plasticity silenced and recall_p_sparsity=1.0
   (per Phase 2c finding), present A+B as a combined cue and let the
   substrate settle for T_settle=30 steps.
7. Classify every live P entity by its components:
     * in_A    — both components ∈ A
     * in_B    — both components ∈ B
     * bridge  — one in A, one in B (should be 0 — A+B never co-presented)
     * other   — neither
   For each class, measure the fraction of P entities with
   activation > 0.1.

Verdict
-------
  PASS         P_A_fraction ≥ 0.7 AND P_B_fraction ≥ 0.7
  WEAK         one class ≥ 0.7, the other < 0.5 (k-WTA dominance)
  INCONCLUSIVE both ≥ 0.4 but neither ≥ 0.7 (parameter regime)
  FAIL         neither ≥ 0.4 — architecture insufficient → Phase 5b

Catastrophic-forgetting expectation
-----------------------------------
During B training (≈ 1500 steps), A's P weights decay at rate
``ρ · λ · W ≈ 0.10 · 0.005 · W = 5e-4 W`` per step. Geometric over
1500 steps: ``W(1500) ≈ W(0) · (1 − 5e-4)^1500 ≈ 0.47``. So A's P
entities should retain ~47 % of their weight, well above the
viability threshold 0.1. They should survive.

Outputs
-------
  results/substrate/phase_5a/phase_5a_results.png
  results/substrate/phase_5a/phase_5a_results.json
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

from substrate.p_entity import PEntity  # noqa: E402
from substrate.substrate import Substrate  # noqa: E402


# ---------------------------------------------------------------------------
# Stimulus helpers
# ---------------------------------------------------------------------------


def define_disjoint_patterns(
    n_neurons: int, pattern_size: int, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Two disjoint patterns of ``pattern_size`` each, drawn from the
    same permutation so they can't overlap by construction."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_neurons)
    a = np.sort(perm[:pattern_size])
    b = np.sort(perm[pattern_size:2 * pattern_size])
    assert len(set(a.tolist()) & set(b.tolist())) == 0
    return a, b


def make_external(
    n_neurons: int, pattern: np.ndarray, strength: float = 0.7,
) -> np.ndarray:
    inp = np.zeros(n_neurons, dtype=np.float32)
    inp[pattern] = strength
    return inp


# ---------------------------------------------------------------------------
# P-entity classification
# ---------------------------------------------------------------------------


def classify_p_entities(
    p_entities: dict[int, PEntity],
    pattern_a: np.ndarray,
    pattern_b: np.ndarray,
) -> dict[str, list[int]]:
    """Bucket every live P by where its component N sit relative to A and B."""
    set_a = {int(x) for x in pattern_a}
    set_b = {int(x) for x in pattern_b}
    out: dict[str, list[int]] = {
        "in_A": [], "in_B": [], "bridge": [], "other": [],
    }
    for pid, p in p_entities.items():
        c1, c2 = p.components
        c1_a, c1_b = c1 in set_a, c1 in set_b
        c2_a, c2_b = c2 in set_a, c2 in set_b
        if c1_a and c2_a:
            out["in_A"].append(pid)
        elif c1_b and c2_b:
            out["in_B"].append(pid)
        elif (c1_a and c2_b) or (c1_b and c2_a):
            out["bridge"].append(pid)
        else:
            out["other"].append(pid)
    return out


def measure_p_activation_by_class(
    substrate: Substrate,
    pattern_a: np.ndarray,
    pattern_b: np.ndarray,
    activation_threshold: float = 0.1,
) -> dict[str, Any]:
    classes = classify_p_entities(substrate.p_entities, pattern_a, pattern_b)

    def stats(class_name: str) -> tuple[float, int, int]:
        pids = classes[class_name]
        if not pids:
            return 0.0, 0, 0
        active = sum(
            1 for pid in pids
            if substrate.p_entities[pid].activation > activation_threshold
        )
        return active / len(pids), active, len(pids)

    a_frac, a_active, a_count = stats("in_A")
    b_frac, b_active, b_count = stats("in_B")
    bridge_frac, bridge_active, bridge_count = stats("bridge")
    other_frac, other_active, other_count = stats("other")
    return {
        "P_A_activation_fraction": a_frac,
        "P_A_active": a_active,
        "P_A_count": a_count,
        "P_B_activation_fraction": b_frac,
        "P_B_active": b_active,
        "P_B_count": b_count,
        "P_bridge_activation_fraction": bridge_frac,
        "P_bridge_active": bridge_active,
        "P_bridge_count": bridge_count,
        "P_other_activation_fraction": other_frac,
        "P_other_active": other_active,
        "P_other_count": other_count,
    }


# ---------------------------------------------------------------------------
# Training + combined-cue test
# ---------------------------------------------------------------------------


def train_one_pattern(
    substrate: Substrate,
    pattern: np.ndarray,
    K: int = 100,
    T_present: int = 15,
    T_rest: int = 60,
) -> None:
    external = make_external(substrate.n_neurons, pattern, strength=0.7)
    for _ in range(K):
        for _ in range(T_present):
            substrate.step(external_input=external)
        for _ in range(T_rest):
            substrate.step(external_input=None)


def test_combined_presentation(
    substrate: Substrate,
    pattern_a: np.ndarray,
    pattern_b: np.ndarray,
    T_settle: int = 30,
    recall_p_sparsity: float = 1.0,
) -> dict[str, Any]:
    """Clamp both patterns simultaneously, settle, measure P activations
    by class.

    Plasticity silenced + recall-time P-sparsity relaxed (the same
    conditions Phase 2c PASSed under). Substrate state is restored
    after the measurement so callers can re-use the trained substrate."""
    saved_eta = substrate.eta
    saved_eta_pp = substrate.eta_pp
    saved_p_sparsity = substrate.p_sparsity_target
    saved_acts = substrate.activations.copy()
    saved_p_acts = {pid: p.activation for pid, p in substrate.p_entities.items()}

    substrate.eta = 0.0
    substrate.eta_pp = 0.0
    substrate.p_sparsity_target = float(recall_p_sparsity)
    substrate.activations = np.zeros_like(substrate.activations)
    for p in substrate.p_entities.values():
        p.activation = 0.0

    combined_cue = np.zeros(substrate.n_neurons, dtype=np.float32)
    combined_cue[pattern_a] = 1.0
    combined_cue[pattern_b] = 1.0

    for _ in range(T_settle):
        substrate.step(external_input=combined_cue)

    metrics = measure_p_activation_by_class(substrate, pattern_a, pattern_b)

    substrate.eta = saved_eta
    substrate.eta_pp = saved_eta_pp
    substrate.p_sparsity_target = saved_p_sparsity
    substrate.activations = saved_acts
    for pid, act in saved_p_acts.items():
        if pid in substrate.p_entities:
            substrate.p_entities[pid].activation = act

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=== Phase 5a: Compositional generalization (P5 spatial) ===\n")
    n_neurons = 300
    pattern_size = 15
    K_per_pattern = 100

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
        seed=42,
    )

    pattern_a, pattern_b = define_disjoint_patterns(
        n_neurons, pattern_size, seed=0,
    )
    print(f"Pattern A (size {len(pattern_a)}): {pattern_a.tolist()}")
    print(f"Pattern B (size {len(pattern_b)}): {pattern_b.tolist()}")
    print(f"Overlap A ∩ B: {len(set(pattern_a.tolist()) & set(pattern_b.tolist()))} "
          f"(should be 0)")
    print()

    # ---- Phase 1: train A ----
    print(f"--- Train pattern A ({K_per_pattern} cycles) ---")
    train_one_pattern(substrate, pattern_a, K=K_per_pattern)
    after_a = classify_p_entities(substrate.p_entities, pattern_a, pattern_b)
    n_pA_after_A = len(after_a["in_A"])
    n_pB_after_A = len(after_a["in_B"])
    n_bridge_after_A = len(after_a["bridge"])
    n_total_after_A = sum(len(v) for v in after_a.values())
    print(f"After A training:  P_A={n_pA_after_A}, P_B={n_pB_after_A}, "
          f"P_bridge={n_bridge_after_A}, total={n_total_after_A}")
    print()

    # ---- Phase 2: train B (A never re-presented) ----
    print(f"--- Train pattern B ({K_per_pattern} cycles, A NEVER re-presented) ---")
    train_one_pattern(substrate, pattern_b, K=K_per_pattern)
    after_b = classify_p_entities(substrate.p_entities, pattern_a, pattern_b)
    n_pA_after_B = len(after_b["in_A"])
    n_pB_after_B = len(after_b["in_B"])
    n_bridge_after_B = len(after_b["bridge"])
    n_total_after_B = sum(len(v) for v in after_b.values())
    print(f"After B training:  P_A={n_pA_after_B}, P_B={n_pB_after_B}, "
          f"P_bridge={n_bridge_after_B}, total={n_total_after_B}")

    if n_pA_after_A > 0:
        retention = n_pA_after_B / n_pA_after_A
        if retention >= 0.7:
            print(f"  ✓ P_A retention through B training: "
                  f"{n_pA_after_A} → {n_pA_after_B}  ({retention * 100:.0f} %)")
        elif retention >= 0.3:
            print(f"  ⚠ P_A retention partial: "
                  f"{n_pA_after_A} → {n_pA_after_B}  ({retention * 100:.0f} %)")
        else:
            print(f"  ✗ P_A retention SEVERELY DEGRADED — catastrophic forgetting risk: "
                  f"{n_pA_after_A} → {n_pA_after_B}  ({retention * 100:.0f} %)")
    print()

    # ---- Phase 3: combined-cue compositional test ----
    print("--- Test: A + B presented simultaneously (never co-presented in training) ---")
    metrics = test_combined_presentation(
        substrate, pattern_a, pattern_b,
        T_settle=30, recall_p_sparsity=1.0,
    )
    print(f"P_A:      {metrics['P_A_active']:>2d} / {metrics['P_A_count']:<2d} active  "
          f"({metrics['P_A_activation_fraction'] * 100:.0f} %)")
    print(f"P_B:      {metrics['P_B_active']:>2d} / {metrics['P_B_count']:<2d} active  "
          f"({metrics['P_B_activation_fraction'] * 100:.0f} %)")
    print(f"P_bridge: {metrics['P_bridge_active']:>2d} / {metrics['P_bridge_count']:<2d} active  "
          f"({metrics['P_bridge_activation_fraction'] * 100:.0f} %)  "
          f"← should be 0 / 0 if A+B never co-presented")
    print(f"P_other:  {metrics['P_other_active']:>2d} / {metrics['P_other_count']:<2d} active  "
          f"({metrics['P_other_activation_fraction'] * 100:.0f} %)")
    print()

    a_frac = metrics["P_A_activation_fraction"]
    b_frac = metrics["P_B_activation_fraction"]

    print("=== Verdict (P5 spatial) ===")
    if a_frac >= 0.7 and b_frac >= 0.7:
        verdict = "PASS"
        reason = (
            "Both pattern families activate simultaneously under combined cue.\n"
            "Compositional generalization demonstrated by the 2-level "
            "(N + P + P-P + P→N) architecture — no S-level needed for the\n"
            "spatial form of P5."
        )
    elif (a_frac >= 0.7 and b_frac < 0.5) or (b_frac >= 0.7 and a_frac < 0.5):
        verdict = "WEAK"
        reason = (
            "Asymmetric activation — one family dominates the other. "
            "Likely k-WTA competition at P level even with relaxation; "
            "consider raising recall_p_sparsity above 1.0 or training "
            "both patterns at higher P-budget."
        )
    elif a_frac >= 0.4 and b_frac >= 0.4:
        verdict = "INCONCLUSIVE"
        reason = (
            "Both classes activate weakly. Either pattern under-trained "
            "(try K=200) or substrate too small for compositional capacity "
            "(try N=500). Architecture not yet falsified."
        )
    else:
        verdict = "FAIL"
        reason = (
            "Neither family activates. 2-level architecture insufficient "
            "for compositional generalization — motivates Phase 5b "
            "(introduce S entities for higher-order grouping)."
        )
    print(f"{verdict}: {reason}")

    # ---- persistence ----
    results_dir = _REPO_ROOT / "results" / "substrate" / "phase_5a"
    results_dir.mkdir(parents=True, exist_ok=True)

    with (results_dir / "phase_5a_results.json").open("w") as f:
        json.dump(
            {
                "config": {
                    "n_neurons": n_neurons,
                    "pattern_size": pattern_size,
                    "K_per_pattern": K_per_pattern,
                    "T_present": 15,
                    "T_rest": 60,
                    "T_settle": 30,
                    "recall_p_sparsity": 1.0,
                    "seed": 42,
                },
                "pattern_a": pattern_a.tolist(),
                "pattern_b": pattern_b.tolist(),
                "after_train_A": {
                    "P_A": n_pA_after_A, "P_B": n_pB_after_A,
                    "P_bridge": n_bridge_after_A, "total": n_total_after_A,
                },
                "after_train_B": {
                    "P_A": n_pA_after_B, "P_B": n_pB_after_B,
                    "P_bridge": n_bridge_after_B, "total": n_total_after_B,
                },
                "catastrophic_forgetting": {
                    "P_A_before_B_training": n_pA_after_A,
                    "P_A_after_B_training": n_pA_after_B,
                    "retention_fraction": (
                        n_pA_after_B / n_pA_after_A if n_pA_after_A > 0 else None
                    ),
                },
                "compositional_test": metrics,
                "verdict": verdict,
                "reason": reason,
            },
            f,
            indent=2,
        )

    # ---- plots ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    stages = ["After Train A", "After Train B"]
    counts_a = [n_pA_after_A, n_pA_after_B]
    counts_b = [n_pB_after_A, n_pB_after_B]
    counts_bridge = [n_bridge_after_A, n_bridge_after_B]
    x = np.arange(len(stages))
    width = 0.25
    axes[0].bar(x - width, counts_a, width, label="P_A", color="tab:blue")
    axes[0].bar(x, counts_b, width, label="P_B", color="tab:green")
    axes[0].bar(x + width, counts_bridge, width, label="P_bridge", color="tab:red")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(stages)
    axes[0].set_ylabel("# live P entities")
    axes[0].set_title("P-entity counts through sequential training")
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].legend()
    for i, (a, b, br) in enumerate(zip(counts_a, counts_b, counts_bridge)):
        axes[0].text(i - width, a + 0.1, str(a), ha="center", va="bottom", fontsize=9)
        axes[0].text(i, b + 0.1, str(b), ha="center", va="bottom", fontsize=9)
        axes[0].text(i + width, br + 0.1, str(br), ha="center", va="bottom", fontsize=9)

    classes = ["P_A", "P_B", "P_bridge", "P_other"]
    fractions = [
        metrics["P_A_activation_fraction"],
        metrics["P_B_activation_fraction"],
        metrics["P_bridge_activation_fraction"],
        metrics["P_other_activation_fraction"],
    ]
    counts = [
        metrics["P_A_count"], metrics["P_B_count"],
        metrics["P_bridge_count"], metrics["P_other_count"],
    ]
    colors = ["tab:blue", "tab:green", "tab:red", "tab:gray"]
    bars = axes[1].bar(classes, fractions, color=colors)
    axes[1].axhline(0.7, color="black", linestyle="--", alpha=0.4,
                    label="PASS threshold")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("activation fraction")
    axes[1].set_title(f"Combined-cue test (A + B simultaneous): {verdict}")
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].legend()
    for i, (frac, count) in enumerate(zip(fractions, counts)):
        axes[1].text(i, frac + 0.02,
                     f"{frac * 100:.0f} %\n(n={count})",
                     ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(results_dir / "phase_5a_results.png", dpi=120)
    plt.close(fig)
    print(f"\nResults → {results_dir}")

    if verdict == "PASS":
        return 0
    if verdict == "WEAK":
        return 1
    if verdict == "INCONCLUSIVE":
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
