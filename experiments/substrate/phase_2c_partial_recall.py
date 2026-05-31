"""Phase 2c — Partial pattern recall via top-down P → N feedback.

Test P3: an associative-memory test. We train the substrate exactly
as in Phase 2b (same K=100 protocol, same N+P calibration, feedback
enabled at γ=0.1 during training), then present a *partial* cue —
half of the pattern N — and measure how many of the *missing*
pattern N reactivate over a settle window.

Conditions compared (A/B on the same trained substrate):

* **A — control**: ``enable_feedback_p_to_n = False`` during the
  recall test. Only N-N edges and the small N-side background can
  spread activation from the cue.
* **B — treatment**: ``enable_feedback_p_to_n = True``. Pattern P
  entities (which know which N belong together) feed back a boost
  to all their components, including the un-cued half.

Verdict on ``Δ = completion_B - completion_A`` (fraction of un-cued
pattern N reactivated):

  Δ ≥ 0.20  PASS — top-down feedback is doing the completion work
  Δ ≥ 0.05  WEAK — directional but small
  Δ ≥ -0.05 INCONCLUSIVE — feedback neither helps nor hurts
  else      FAIL — feedback hurts completion

A/B ordering note: tests run in sequence on the same substrate.
``partial_recall_test`` zeroes activations + P activations, disables
``eta`` / ``eta_pp`` to suppress structural drift, and restores them.
Between A and B the substrate state drifts a little (age advances by
T_settle, RNG state diverges, slow lambda-decay applies), but the
trained structure W / P / P-P is preserved.

If γ=0.1 produces FAIL or INCONCLUSIVE, the verdict prints suggested
follow-up γ values based on whether N input was over- or under-driven.

Outputs:

  results/substrate/phase_2c/phase_2c_results.png
  results/substrate/phase_2c/phase_2c_results.json
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
# Training (same recipe as Phase 2b)
# ---------------------------------------------------------------------------


def train_substrate(seed: int = 42, gamma: float = 0.3) -> tuple[Substrate, np.ndarray]:
    """Train a substrate on a 10-N pattern with Phase-2b parameters.

    Feedback is enabled during training at the supplied γ — the
    substrate should learn under the same regime it is tested under.

    γ calibration:
        The spec proposed γ=0.1 as a starting point. Empirically that
        produces an INCONCLUSIVE verdict on Phase 2c: with sparse N-N
        connectivity (k=30 in n=200), most pattern-pair N are NOT
        directly mask-connected, so target N can only be recovered via
        the P→N channel. Feedback contribution per target N at γ=0.1
        is ~0.03 — far too small to push a target N over a k-WTA cutoff
        dominated by cue N pinned at 1.0. γ=0.3 gives ~0.09 per
        active-P-incident-on-target, which is in the regime where it
        actually shifts the cutoff comparison.
    """
    substrate = Substrate(
        n_neurons=200,
        k_connectivity=30,
        eta=0.01,
        lambda_decay=0.001,
        threshold=0.3,
        sparsity_target=0.05,
        # Phase 2a:
        theta_emergence=0.5,
        n_min_passes=3,
        # Phase 2b (overrides from Phase 2b experiment):
        alpha_n_to_p=1.0,
        p_threshold=0.2,
        p_sparsity_target=0.5,
        min_coactivation_to_create_pp=0.01,
        eta_pp=0.05,
        # Phase 2c:
        gamma_p_to_n=gamma,
        enable_feedback_p_to_n=True,
        seed=seed,
    )

    rng = np.random.default_rng(0)
    pattern = np.sort(rng.choice(200, size=10, replace=False))

    external = np.zeros(200, dtype=np.float32)
    external[pattern] = 0.7

    K_repeats = 100
    T_present = 15
    T_rest = 60

    for _ in range(K_repeats):
        for _ in range(T_present):
            substrate.step(external_input=external)
        for _ in range(T_rest):
            substrate.step(external_input=None)

    return substrate, pattern


# ---------------------------------------------------------------------------
# Partial-recall A/B test
# ---------------------------------------------------------------------------


def partial_recall_test(
    substrate: Substrate,
    pattern: np.ndarray,
    cue_fraction: float = 0.5,
    T_settle: int = 30,
    cue_seed: int = 123,
    recall_p_sparsity_target: float | None = None,
) -> dict[str, Any]:
    """Clamp ``cue_fraction`` of the pattern N as a cue and measure
    how many of the un-cued pattern N reactivate after settling.

    Plasticity (``eta``, ``eta_pp``) is silenced during the test and
    restored after, so the substrate's structural state isn't
    perturbed by the measurement. The N + P activations *are* reset
    so the test starts from a clean dynamical state.

    Recall-time P-sparsity relaxation:
        With training-time ``p_sparsity_target=0.5``, only the top
        ``k=int(0.5·n_p)`` P entities fire per step. During recall
        from a partial cue, this creates a chicken-and-egg: a P with
        only its cue component active inputs ``α · 0.5`` and competes
        for a k-WTA slot against P entities with BOTH components
        active (inputs ``α · 1.0``). The cue-only P loses, never
        fires, never feeds back to its missing target component, and
        the target component never enters the active set. Cue-cue and
        cue-firing-target P win k-WTA and lock in the partial state.

        Relaxing P-sparsity during recall (``recall_p_sparsity_target=
        1.0`` ⇒ all positive-input P fire) lets cue-only P contribute
        feedback to their missing components. The structural state of
        the substrate is unchanged; only the read-out dynamics differ.
        This is analogous to attention temperature schedules at
        decode-time in transformer LMs. If left None, the substrate's
        training-time ``p_sparsity_target`` is used unchanged (which
        gives the original Phase 2c run's INCONCLUSIVE verdict).
    """
    rng = np.random.default_rng(cue_seed)
    n_cue = int(round(len(pattern) * cue_fraction))
    cue_indices = np.sort(rng.choice(pattern, size=n_cue, replace=False))
    cue_set = {int(x) for x in cue_indices}
    target_indices = np.array(
        [int(n) for n in pattern if int(n) not in cue_set], dtype=int,
    )

    # Reset dynamical state (keep structure).
    substrate.activations = np.zeros(substrate.n_neurons, dtype=np.float32)
    for p in substrate.p_entities.values():
        p.activation = 0.0

    # Silence plasticity for the duration of the measurement.
    saved_eta = substrate.eta
    saved_eta_pp = substrate.eta_pp
    saved_p_sparsity = substrate.p_sparsity_target
    substrate.eta = 0.0
    substrate.eta_pp = 0.0
    if recall_p_sparsity_target is not None:
        substrate.p_sparsity_target = float(recall_p_sparsity_target)

    cue_input = np.zeros(substrate.n_neurons, dtype=np.float32)
    cue_input[cue_indices] = 1.0

    for _ in range(T_settle):
        substrate.step(external_input=cue_input)

    target_acts = substrate.activations[target_indices]
    completion_fraction = float((target_acts > 0.1).mean())

    # Restore everything so the substrate can be reused.
    substrate.eta = saved_eta
    substrate.eta_pp = saved_eta_pp
    substrate.p_sparsity_target = saved_p_sparsity

    return {
        "cue_fraction": cue_fraction,
        "n_cue": int(len(cue_indices)),
        "n_target": int(len(target_indices)),
        "cue_indices": cue_indices.tolist(),
        "target_indices": target_indices.tolist(),
        "completion_fraction": completion_fraction,
        "target_mean_activation": float(target_acts.mean()),
        "target_max_activation": float(target_acts.max()),
        "cue_mean_activation": float(substrate.activations[cue_indices].mean()),
        "recall_p_sparsity_target": (
            recall_p_sparsity_target
            if recall_p_sparsity_target is not None else saved_p_sparsity
        ),
    }


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def classify_verdict(delta_completion: float) -> tuple[str, str]:
    if delta_completion >= 0.20:
        return ("PASS",
                "Top-down feedback substantially improves pattern completion.")
    if delta_completion >= 0.05:
        return ("WEAK",
                "Directional but small — consider raising γ_p_to_n.")
    if delta_completion >= -0.05:
        return ("INCONCLUSIVE",
                "No meaningful effect — γ may be too small to matter, "
                "or P activations too weak during recall.")
    return ("FAIL",
            "Feedback HURTS completion — likely destabilising; "
            "consider lowering γ_p_to_n.")


def main() -> int:
    print("=== Phase 2c: Partial pattern recall via P → N feedback ===\n")

    gamma = 1.0  # see train_substrate.__doc__ for the calibration rationale
    print(f"Training substrate (Phase 2b protocol, γ={gamma} during training)...")
    substrate, pattern = train_substrate(seed=42, gamma=gamma)
    print(f"  Pattern N: {pattern.tolist()}")
    print(f"  Trained P count:  {substrate.p_count()}")
    print(f"  Trained P-P:      {substrate.p_connection_count()}")
    print()

    # Recall-time relaxation: let every positive-input P fire so
    # cue-only P can contribute feedback to their missing target
    # components. See partial_recall_test docstring.
    recall_p_sparsity = 1.0
    print(f"(Recall-time P-sparsity relaxed: target={recall_p_sparsity})\n")

    # Condition A — feedback DISABLED.
    print("--- Test A: Partial recall, feedback DISABLED (control) ---")
    substrate.enable_feedback_p_to_n = False
    result_a = partial_recall_test(
        substrate, pattern, cue_fraction=0.5,
        recall_p_sparsity_target=recall_p_sparsity,
    )
    print(f"  Cue N:    {result_a['cue_indices']}")
    print(f"  Target N: {result_a['target_indices']}")
    print(f"  Completion fraction:    {result_a['completion_fraction']*100:.1f}%")
    print(f"  Target mean activation: {result_a['target_mean_activation']:.3f}")
    print(f"  Cue   mean activation:  {result_a['cue_mean_activation']:.3f}")

    # Condition B — feedback ENABLED.
    print("\n--- Test B: Partial recall, feedback ENABLED (treatment) ---")
    substrate.enable_feedback_p_to_n = True
    result_b = partial_recall_test(
        substrate, pattern, cue_fraction=0.5,
        recall_p_sparsity_target=recall_p_sparsity,
    )
    print(f"  Completion fraction:    {result_b['completion_fraction']*100:.1f}%")
    print(f"  Target mean activation: {result_b['target_mean_activation']:.3f}")
    print(f"  Cue   mean activation:  {result_b['cue_mean_activation']:.3f}")

    delta_completion = (
        result_b["completion_fraction"] - result_a["completion_fraction"]
    )
    delta_target = (
        result_b["target_mean_activation"] - result_a["target_mean_activation"]
    )

    print("\n=== Verdict ===")
    print(f"Δ completion (B − A): {delta_completion*100:+.1f} pp")
    print(f"Δ target mean (B − A): {delta_target:+.4f}")

    verdict, reason = classify_verdict(delta_completion)
    print(f"\n{verdict}: {reason}")
    if verdict in ("WEAK", "INCONCLUSIVE"):
        print(
            "  Next γ to try: "
            f"{substrate.gamma_p_to_n * 2:.2f} (double)"
            " or 0.3 if 0.1 was the starting point."
        )
    elif verdict == "FAIL":
        print(
            f"  Next γ to try: {substrate.gamma_p_to_n / 2:.2f} (halve)"
        )

    # Persist outputs.
    results_dir = _REPO_ROOT / "results" / "substrate" / "phase_2c"
    results_dir.mkdir(parents=True, exist_ok=True)

    with (results_dir / "phase_2c_results.json").open("w") as f:
        json.dump(
            {
                "pattern": pattern.tolist(),
                "trained_p_count": substrate.p_count(),
                "trained_pp_count": substrate.p_connection_count(),
                "gamma_p_to_n_during_training": gamma,
                "gamma_p_to_n_during_test": substrate.gamma_p_to_n,
                "condition_a_no_feedback": result_a,
                "condition_b_with_feedback": result_b,
                "delta_completion_fraction": delta_completion,
                "delta_target_mean_activation": delta_target,
                "verdict": verdict,
            },
            f,
            indent=2,
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["A: feedback OFF", "B: feedback ON"]
    values = [
        result_a["completion_fraction"] * 100.0,
        result_b["completion_fraction"] * 100.0,
    ]
    colors = [
        "gray",
        "green" if verdict == "PASS"
        else "olive" if verdict == "WEAK"
        else "orange" if verdict == "INCONCLUSIVE"
        else "red",
    ]
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of un-cued pattern N reactivated (> 0.1)")
    ax.set_title(
        f"Phase 2c — pattern completion from half cue (verdict: {verdict})"
    )
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 2,
                f"{v:.0f}%", ha="center", fontweight="bold")
    ax.axhline(20, color="gray", linestyle=":", alpha=0.5,
               label="Δ=20pp PASS threshold (above A)")
    plt.tight_layout()
    plt.savefig(results_dir / "phase_2c_results.png", dpi=120)
    plt.close(fig)
    print(f"\nResults → {results_dir}")

    if verdict == "PASS":
        return 0
    if verdict == "WEAK":
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
