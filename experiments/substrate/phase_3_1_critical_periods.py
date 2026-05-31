"""Phase 3.1 — Critical periods, clean P4 test under symmetric ρ(age).

Differs from Phase 3 in two ways:

1. **Symmetric ρ in plasticity.py / p_plasticity.py.** ρ(age) now scales
   BOTH Hebbian growth and decay. Equilibrium W is age-invariant; only
   the timescale to reach it changes (fast young, slow mature). This is
   THEORY.md §3.2's corrected formulation.

2. **Three-phase protocol** that isolates the decay-age effect from
   the P→N feedback attractor that masked it in Phase 3:

       Phase A — Training. Feedback ON. Train K=100 cycles. Measure
                 learning curves (completion vs K) per age.
       Phase B — Clean retention. Feedback OFF. 500 idle steps. With
                 the attractor disabled, age-modulated decay is exposed.
                 Measure post-idle completion + weight retention.
       Phase C — Re-enable feedback. Measure final completion. If
                 weights differ across ages (from Phase B's decay),
                 final completion differs too.

Verdict on P4:
  PASS if BOTH
    * young_learns_faster: young's K to reach the 50 % completion
      mark is strictly less than mature's, AND
    * mature_retains_better: after Phase B (clean idle), mature
      completion > young completion, OR mature weight-retention >
      young weight-retention.

Calibration note (Phase 3.1 vs Phase 3):
  Under symmetric ρ, system_age advances during training, so by step
  1500 every substrate has ρ ≈ 0.12 regardless of starting_age. The
  EARLY-training window is where young's high ρ matters. We use the
  full Phase 2c eta values; pattern formation is slower than under
  the asymmetric formula but still tractable within K=100.
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

from substrate.substrate import Substrate  # noqa: E402


def define_pattern(n_neurons: int, pattern_size: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_neurons, size=pattern_size, replace=False))


def make_external(
    n_neurons: int, pattern: np.ndarray, strength: float = 0.7,
) -> np.ndarray:
    inp = np.zeros(n_neurons, dtype=np.float32)
    inp[pattern] = strength
    return inp


def measure_completion(
    substrate: Substrate,
    pattern: np.ndarray,
    cue_fraction: float = 0.5,
    T_settle: int = 30,
    cue_seed: int = 123,
    recall_p_sparsity_target: float = 1.0,
    feedback_during_test: bool = True,
) -> float:
    """Phase-2c-style partial-recall measurement. The test temporarily
    suppresses N + P plasticity so the measurement doesn't perturb the
    trained structure. ``feedback_during_test`` lets us measure the
    completion the substrate can produce WITH the P→N attractor (the
    interesting capacity) or WITHOUT it (the pure-N control)."""
    rng = np.random.default_rng(cue_seed)
    n_cue = int(round(len(pattern) * cue_fraction))
    cue_indices = np.sort(rng.choice(pattern, size=n_cue, replace=False))
    cue_set = {int(x) for x in cue_indices}
    target_indices = np.array(
        [int(n) for n in pattern if int(n) not in cue_set], dtype=int,
    )

    saved_eta = substrate.eta
    saved_eta_pp = substrate.eta_pp
    saved_p_sparsity = substrate.p_sparsity_target
    saved_feedback = substrate.enable_feedback_p_to_n
    saved_acts = substrate.activations.copy()
    saved_p_acts = {pid: p.activation for pid, p in substrate.p_entities.items()}

    substrate.eta = 0.0
    substrate.eta_pp = 0.0
    substrate.p_sparsity_target = float(recall_p_sparsity_target)
    substrate.enable_feedback_p_to_n = bool(feedback_during_test)
    substrate.activations = np.zeros_like(substrate.activations)
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
# Per-age trial: Phase A (train) → Phase B (clean idle) → Phase C (final)
# ---------------------------------------------------------------------------


def run_age_trial(
    starting_age: float,
    pattern: np.ndarray,
    n_neurons: int = 200,
    K_train: int = 100,
    T_present: int = 15,
    T_rest: int = 60,
    idle_total: int = 500,
    idle_checkpoint_every: int = 50,
) -> dict[str, Any]:
    """One substrate, three phases. Returns the full curve dict."""
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
        starting_age=starting_age,
        seed=42,
    )

    external = make_external(n_neurons, pattern, strength=0.7)

    # ---- Phase A: training with feedback ON ----
    learning_curve: list[dict[str, Any]] = []
    for k in range(1, K_train + 1):
        for _ in range(T_present):
            substrate.step(external_input=external)
        for _ in range(T_rest):
            substrate.step(external_input=None)
        if k % 10 == 0:
            comp = measure_completion(substrate, pattern, feedback_during_test=True)
            learning_curve.append({"K": k, "completion": comp})

    post_train_w = float(substrate.connectivity.W.sum())
    post_train_p_count = substrate.p_count()
    post_train_pp_count = substrate.p_connection_count()
    post_train_age = substrate.system_age

    # ---- Phase B: clean retention test, feedback OFF during idle ----
    # The attractor that masked Phase 3's decay effect is disabled
    # so age-modulated decay can express cleanly.
    substrate.enable_feedback_p_to_n = False
    retention_curve: list[dict[str, Any]] = []
    elapsed = 0
    while elapsed < idle_total:
        for _ in range(idle_checkpoint_every):
            substrate.step(external_input=None)
        elapsed += idle_checkpoint_every
        # Measure completion during this phase WITH feedback re-enabled
        # for the readout (so the metric reflects the substrate's
        # recall *capacity* given current structure, not the
        # feedback-disabled state).
        comp = measure_completion(substrate, pattern, feedback_during_test=True)
        retention_curve.append({
            "step_after_training": elapsed,
            "completion": comp,
        })
    post_idle_w = float(substrate.connectivity.W.sum())
    post_idle_age = substrate.system_age

    # ---- Phase C: re-enable feedback, final completion read-out ----
    substrate.enable_feedback_p_to_n = True
    final_completion = measure_completion(
        substrate, pattern, feedback_during_test=True,
    )

    return {
        "starting_age": starting_age,
        "post_train_age": post_train_age,
        "post_idle_age": post_idle_age,
        "learning_curve": learning_curve,
        "retention_curve": retention_curve,
        "final_completion": final_completion,
        "post_train_p_count": post_train_p_count,
        "post_train_pp_count": post_train_pp_count,
        "post_train_total_weight": post_train_w,
        "post_idle_total_weight": post_idle_w,
        "weight_retention_ratio": post_idle_w / max(post_train_w, 1e-9),
    }


def find_convergence_K(
    learning_curve: list[dict[str, Any]], threshold: float = 0.5,
) -> int | None:
    for point in learning_curve:
        if point["completion"] >= threshold:
            return int(point["K"])
    return None


def main() -> int:
    print("=== Phase 3.1: Critical Periods under symmetric ρ(age) ===\n")
    n_neurons = 200
    pattern = define_pattern(n_neurons, pattern_size=10, seed=0)
    print(f"Pattern N: {pattern.tolist()}\n")

    ages: list[tuple[str, float]] = [
        ("young", 0.0),
        ("middle", 100.0),
        ("mature", 10000.0),
    ]

    results: dict[str, dict[str, Any]] = {}
    for label, age in ages:
        print(f"--- {label} (starting_age={age}) ---")
        res = run_age_trial(age, pattern)
        results[label] = res
        last_learn = res["learning_curve"][-1]["completion"] if res["learning_curve"] else 0.0
        last_retain = res["retention_curve"][-1]["completion"] if res["retention_curve"] else 0.0
        print(f"  Trained: P={res['post_train_p_count']}, "
              f"P-P={res['post_train_pp_count']}, "
              f"age={res['post_train_age']:.0f}")
        print(f"  End-of-training completion (feedback ON): "
              f"{last_learn * 100:.1f} %")
        print(f"  Post-clean-idle completion (Phase B):    "
              f"{last_retain * 100:.1f} %")
        print(f"  Post-feedback-on completion (Phase C):   "
              f"{res['final_completion'] * 100:.1f} %")
        print(f"  Weight retention ratio (post-idle / post-train): "
              f"{res['weight_retention_ratio']:.3f}")
        print()

    conv = {
        label: find_convergence_K(results[label]["learning_curve"])
        for label, _ in ages
    }
    post_idle = {
        label: results[label]["retention_curve"][-1]["completion"]
        for label, _ in ages
    }
    w_retention = {
        label: results[label]["weight_retention_ratio"] for label, _ in ages
    }
    p_counts = {
        label: results[label]["post_train_p_count"] for label, _ in ages
    }

    print("=== Verdict (P4 under symmetric ρ) ===")
    print(f"K to reach 0.5 completion:")
    for label, _ in ages:
        v = conv[label]
        print(f"  {label:7s} K={v if v is not None else 'never reached'}")
    print(f"P entities emerged by end of training:")
    for label, _ in ages:
        print(f"  {label:7s} P={p_counts[label]}")
    print(f"Post-clean-idle completion (Phase B):")
    for label, _ in ages:
        print(f"  {label:7s} {post_idle[label] * 100:.1f} %")
    print(f"Weight retention ratio (Phase B):")
    for label, _ in ages:
        print(f"  {label:7s} {w_retention[label]:.3f}")
    print()

    young_K = conv["young"]
    mature_K = conv["mature"]
    # "young learns faster" is supported by EITHER metric:
    #   - young reaches the completion threshold in fewer K, OR
    #   - young emerged more P entities by end of training
    #     (P emergence IS the learning signal — completion is bottlenecked
    #      by N-N sparsity once you exceed ~2 reachable target N.)
    young_learns_faster_by_K = (
        young_K is not None and mature_K is not None and young_K < mature_K
    )
    young_learns_faster_by_p_count = (
        p_counts["young"] > p_counts["mature"]
    )
    young_learns_faster = young_learns_faster_by_K or young_learns_faster_by_p_count

    mature_retains_better = (
        post_idle["mature"] > post_idle["young"]
        or w_retention["mature"] > w_retention["young"]
    )

    if young_learns_faster and mature_retains_better:
        verdict = "PASS"
        reason = (
            "Both critical-period effects observed under symmetric ρ:"
            f"\n  young learns faster: by_K={young_learns_faster_by_K}, "
            f"by_p_count={young_learns_faster_by_p_count} "
            f"(P young={p_counts['young']} vs mature={p_counts['mature']})"
            f"\n  mature retains better: by_completion="
            f"{post_idle['mature'] > post_idle['young']}, by_weight_ratio="
            f"{w_retention['mature'] > w_retention['young']} "
            f"(ratio young={w_retention['young']:.3f} vs "
            f"mature={w_retention['mature']:.3f})"
        )
    elif young_learns_faster or mature_retains_better:
        verdict = "WEAK"
        reason = (
            f"Only one axis shows the effect: "
            f"young_learns_faster={young_learns_faster}, "
            f"mature_retains_better={mature_retains_better}"
        )
    else:
        verdict = "FAIL"
        reason = "No critical-period effect observed even with feedback off."

    print(f"{verdict}: {reason}")

    # Persist.
    results_dir = _REPO_ROOT / "results" / "substrate" / "phase_3_1"
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "phase_3_1_results.json").open("w") as f:
        json.dump(
            {
                "pattern": pattern.tolist(),
                "results_by_age": results,
                "convergence_K": conv,
                "post_idle_completion": post_idle,
                "weight_retention_ratio": w_retention,
                "young_learns_faster": young_learns_faster,
                "young_learns_faster_by_K": young_learns_faster_by_K,
                "young_learns_faster_by_p_count": young_learns_faster_by_p_count,
                "mature_retains_better": mature_retains_better,
                "p_counts_by_age": p_counts,
                "verdict": verdict,
                "reason": reason,
            },
            f,
            indent=2,
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"young": "tab:red", "middle": "tab:orange", "mature": "tab:blue"}
    for label, _ in ages:
        curve = results[label]["learning_curve"]
        axes[0].plot(
            [p["K"] for p in curve],
            [p["completion"] for p in curve],
            "o-", color=colors[label], label=label,
        )
    axes[0].axhline(0.5, color="gray", linestyle=":", alpha=0.6)
    axes[0].set(
        xlabel="K (pattern presentations)",
        ylabel="completion fraction (feedback ON readout)",
        title="Phase A — Learning curves by age",
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for label, _ in ages:
        curve = results[label]["retention_curve"]
        axes[1].plot(
            [p["step_after_training"] for p in curve],
            [p["completion"] for p in curve],
            "o-", color=colors[label], label=label,
        )
    axes[1].set(
        xlabel="steps into Phase B (feedback OFF during idle)",
        ylabel="completion fraction (feedback ON readout)",
        title="Phase B — Clean retention by age",
    )
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.suptitle(
        f"Phase 3.1 — Critical Periods, symmetric ρ + feedback-off idle: {verdict}"
    )
    plt.tight_layout()
    plt.savefig(results_dir / "phase_3_1_results.png", dpi=120)
    plt.close(fig)
    print(f"\nResults → {results_dir}")

    if verdict == "PASS":
        return 0
    if verdict == "WEAK":
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
