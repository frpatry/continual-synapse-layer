"""Phase 3 — Critical periods (test of P4).

Three identical substrates instantiated at starting_age 0, 100, 10000.
Each goes through:
  1. K=100 training presentations of the same pattern (Phase-2c protocol).
  2. After training, 500 idle steps (no external input).

We record:
  * Learning curve  — completion fraction at each K = 10, 20, ... 100
  * Retention curve — completion fraction at each idle-step = 50, 100, ... 500

P4 verdict (spec):
  * PASS if BOTH young learns faster (smaller K to reach 0.5 completion)
    AND mature retains better (higher completion at end of idle phase)
  * WEAK if only one axis shows the effect
  * FAIL if neither

Math-based expectation:
  THEORY.md §3.2's decay factor 1/(1+log(1+age)) is HIGH for young
  (=1.0) and LOW for mature (=0.10). Higher decay competes more with
  Hebbian growth, so naively mature should *also* learn faster
  (decay-adjusted equilibrium W is higher). Only the retention half
  has unambiguous theoretical support. A WEAK verdict here is itself
  a result — it tells THEORY.md §3.2 that the chosen formula is
  insufficient on its own to produce the biological critical-period
  story; an additional age-dependent term on eta (growth rate) or
  on emergence threshold would be needed.

Outputs:
  results/substrate/phase_3/phase_3_results.png
  results/substrate/phase_3/phase_3_results.json
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


# ---------------------------------------------------------------------------
# Stimulus + measurement helpers
# ---------------------------------------------------------------------------


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
) -> float:
    """Phase-2c-style partial recall: clamp half the pattern as cue,
    settle, measure fraction of target N reactivated.

    Plasticity is silenced and substrate state is restored after the
    test so successive checkpoints don't perturb the structure.
    Recall-time P-sparsity is relaxed to 1.0 by default (matches
    Phase 2c's PASS configuration — the test should measure what the
    substrate *can* recover, not what the training-time k-WTA at P
    permits at recall time)."""
    rng = np.random.default_rng(cue_seed)
    n_cue = int(round(len(pattern) * cue_fraction))
    cue_indices = np.sort(rng.choice(pattern, size=n_cue, replace=False))
    cue_set = {int(x) for x in cue_indices}
    target_indices = np.array(
        [int(n) for n in pattern if int(n) not in cue_set], dtype=int,
    )

    # Snapshot mutable state so the test is non-destructive.
    saved_eta = substrate.eta
    saved_eta_pp = substrate.eta_pp
    saved_p_sparsity = substrate.p_sparsity_target
    saved_acts = substrate.activations.copy()
    saved_p_acts = {pid: p.activation for pid, p in substrate.p_entities.items()}

    substrate.eta = 0.0
    substrate.eta_pp = 0.0
    substrate.p_sparsity_target = float(recall_p_sparsity_target)
    substrate.activations = np.zeros_like(substrate.activations)
    for p in substrate.p_entities.values():
        p.activation = 0.0

    cue_input = np.zeros(substrate.n_neurons, dtype=np.float32)
    cue_input[cue_indices] = 1.0
    for _ in range(T_settle):
        substrate.step(external_input=cue_input)

    target_acts = substrate.activations[target_indices]
    completion = float((target_acts > 0.1).mean())

    # Restore.
    substrate.eta = saved_eta
    substrate.eta_pp = saved_eta_pp
    substrate.p_sparsity_target = saved_p_sparsity
    substrate.activations = saved_acts
    for pid, act in saved_p_acts.items():
        if pid in substrate.p_entities:
            substrate.p_entities[pid].activation = act

    return completion


# ---------------------------------------------------------------------------
# Per-age experiment
# ---------------------------------------------------------------------------


def run_age_condition(
    starting_age: float,
    pattern: np.ndarray,
    n_neurons: int = 200,
    K_repeats: int = 100,
    T_present: int = 15,
    T_rest: int = 60,
    idle_total: int = 500,
    idle_checkpoint_every: int = 50,
) -> dict[str, Any]:
    """Train one substrate at the given starting_age, then run an
    idle phase. Return both curves + post-training diagnostics."""
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

    learning_curve: list[dict[str, Any]] = []
    for k in range(1, K_repeats + 1):
        for _ in range(T_present):
            substrate.step(external_input=external)
        for _ in range(T_rest):
            substrate.step(external_input=None)
        if k % 10 == 0:
            comp = measure_completion(substrate, pattern)
            learning_curve.append({"K": k, "completion": comp})

    post_train_w = float(substrate.connectivity.W.sum())
    post_train_p_count = substrate.p_count()
    post_train_pp_count = substrate.p_connection_count()

    retention_curve: list[dict[str, Any]] = []
    elapsed = 0
    while elapsed < idle_total:
        for _ in range(idle_checkpoint_every):
            substrate.step(external_input=None)
        elapsed += idle_checkpoint_every
        comp = measure_completion(substrate, pattern)
        retention_curve.append({
            "step_after_training": elapsed,
            "completion": comp,
        })

    post_idle_w = float(substrate.connectivity.W.sum())

    return {
        "starting_age": starting_age,
        "learning_curve": learning_curve,
        "retention_curve": retention_curve,
        "post_train_p_count": post_train_p_count,
        "post_train_pp_count": post_train_pp_count,
        "post_train_total_weight": post_train_w,
        "post_idle_total_weight": post_idle_w,
        "weight_retention_ratio": post_idle_w / max(post_train_w, 1e-9),
    }


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def find_convergence_K(
    learning_curve: list[dict[str, Any]], threshold: float = 0.5,
) -> int | None:
    """Smallest K at which completion ≥ threshold, or None."""
    for point in learning_curve:
        if point["completion"] >= threshold:
            return int(point["K"])
    return None


def main() -> int:
    print("=== Phase 3: Critical Periods (P4) ===\n")
    n_neurons = 200
    pattern = define_pattern(n_neurons, pattern_size=10, seed=0)
    print(f"Pattern N: {pattern.tolist()}")
    print()

    ages: list[tuple[str, float]] = [
        ("young", 0.0),
        ("middle", 100.0),
        ("mature", 10000.0),
    ]

    results: dict[str, dict[str, Any]] = {}
    for label, age in ages:
        print(f"--- {label} substrate (starting_age={age}) ---")
        res = run_age_condition(age, pattern)
        results[label] = res
        final_learn = res["learning_curve"][-1]["completion"]
        final_idle = res["retention_curve"][-1]["completion"]
        print(f"  P entities: {res['post_train_p_count']}, "
              f"P-P: {res['post_train_pp_count']}")
        print(f"  Final learning completion: {final_learn * 100:.1f} %")
        print(f"  After 500 idle steps:      {final_idle * 100:.1f} %")
        print(f"  Weight retention ratio:    {res['weight_retention_ratio']:.3f}")
        print()

    conv = {
        label: find_convergence_K(results[label]["learning_curve"])
        for label, _ in ages
    }
    retention_completion = {
        label: results[label]["retention_curve"][-1]["completion"]
        for label, _ in ages
    }

    print("=== Verdict (P4) ===")
    print(f"K to reach completion 0.5:")
    for label, _ in ages:
        v = conv[label]
        print(f"  {label:7s} K={v if v is not None else 'never reached'}")
    print(f"Completion after 500 idle steps:")
    for label, _ in ages:
        print(f"  {label:7s} {retention_completion[label] * 100:.1f} %")
    print(f"Weight retention ratio (post_idle_W / post_train_W):")
    for label, _ in ages:
        print(f"  {label:7s} {results[label]['weight_retention_ratio']:.3f}")
    print()

    # Decide: young learns faster if it reaches 0.5 in strictly fewer K
    # than mature.
    young_K = conv["young"]
    mature_K = conv["mature"]
    young_learns_faster = (
        young_K is not None and mature_K is not None and young_K < mature_K
    )
    # Mature retains better: higher completion at the end of idle, OR
    # — as a secondary signal — higher weight retention ratio.
    mature_retains_better = (
        retention_completion["mature"] > retention_completion["young"]
        or results["mature"]["weight_retention_ratio"]
            > results["young"]["weight_retention_ratio"]
    )

    if young_learns_faster and mature_retains_better:
        verdict = "PASS"
        reason = "Both critical-period effects observed."
    elif young_learns_faster or mature_retains_better:
        verdict = "WEAK"
        reason = (
            f"Only one axis shows the effect: "
            f"young_learns_faster={young_learns_faster}, "
            f"mature_retains_better={mature_retains_better}"
        )
    else:
        verdict = "FAIL"
        reason = "No critical-period effect observed on either axis."

    print(f"{verdict}: {reason}")

    if verdict in ("WEAK", "FAIL"):
        print()
        print("Suggested follow-up if iterating on this prediction:")
        if not young_learns_faster:
            print("  - The chosen decay formula 1/(1+log(1+age)) makes "
                  "MATURE substrates learn faster (decay competes less "
                  "with Hebbian growth). A biological critical period "
                  "needs age-modulated *growth rate*, not just decay — "
                  "try multiplying eta by 1/(1+log(1+age)) too, or by "
                  "a steeper schedule like (1+age)^-0.5.")
        if not mature_retains_better:
            print("  - Retention axis failed — either the age range "
                  "is too compressed (try [0, 1e6]) or the idle phase "
                  "isn't long enough to resolve the decay difference "
                  "(try idle_total=5000).")

    # Persist.
    results_dir = _REPO_ROOT / "results" / "substrate" / "phase_3"
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "phase_3_results.json").open("w") as f:
        json.dump(
            {
                "pattern": pattern.tolist(),
                "results_by_age": results,
                "convergence_K": conv,
                "retention_completion": retention_completion,
                "young_learns_faster": young_learns_faster,
                "mature_retains_better": mature_retains_better,
                "verdict": verdict,
                "reason": reason,
            },
            f,
            indent=2,
        )

    # Plots.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"young": "tab:red", "middle": "tab:orange", "mature": "tab:blue"}
    for label, _ in ages:
        curve = results[label]["learning_curve"]
        axes[0].plot(
            [p["K"] for p in curve],
            [p["completion"] for p in curve],
            "o-", color=colors[label], label=label,
        )
    axes[0].axhline(0.5, color="gray", linestyle=":", alpha=0.6,
                    label="K-convergence threshold")
    axes[0].set(
        xlabel="K (pattern presentations)",
        ylabel="completion fraction",
        title="Learning curves by age",
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
        xlabel="steps after training",
        ylabel="completion fraction",
        title="Retention by age (idle phase)",
    )
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.suptitle(f"Phase 3 — Critical Periods (P4): {verdict}")
    plt.tight_layout()
    plt.savefig(results_dir / "phase_3_results.png", dpi=120)
    plt.close(fig)
    print(f"\nResults → {results_dir}")

    if verdict == "PASS":
        return 0
    if verdict == "WEAK":
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
