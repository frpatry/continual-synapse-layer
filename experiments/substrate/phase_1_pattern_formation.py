"""Phase 1 / 1.1 — P1 (pattern formation through repeated exposure).

Phase 1 protocol (unchanged):

1. Create a Phase-1 substrate (500 N, sparse k=50 connectivity).
2. Pick a fixed "pattern" — a subset of 25 N indices.
3. Repeatedly drive the pattern N with strong external input
   (K presentations × ``T_present`` steps each), interleaved with
   ``T_rest`` quiet steps between presentations.
4. At checkpoints, measure pattern completion:
   - reset activations,
   - clamp HALF of the pattern as a cue,
   - let the substrate settle for ``T_settle`` steps,
   - measure how active the *non-cued* pattern N become vs the
     background (non-pattern) N.

Phase 1.1 additions:

* Substrate now applies k-winners-take-all (k = ``sparsity_target`` *
  n_neurons) after the soft threshold — H5 is enforced structurally,
  not just emergent.
* We re-run the P1 protocol under TWO configurations to compare
  k-WTA's effect on robustness:

    - **defaults**:    eta=0.01,   lambda_decay=0.001, threshold=0.30,
                       external_strength=1.0
                       (Phase 1 saturated under these; with k-WTA the
                        substrate should remain sparse and learn.)
    - **calibrated**:  eta=0.0015, lambda_decay=0.004, threshold=0.35,
                       external_strength=0.7
                       (Phase 1's tuned values; should keep working.)

Verdict from the completion score (target_mean − background_mean):

  > 0.10   PASS — pattern formation working as P1 predicts
  > 0.02   WEAK — directional signal, parameter tuning warranted
  ≤ 0.02   FAIL — theory or parameters need revisiting

Outputs:

  results/substrate/phase_1_1/phase_1_with_kwta_defaults.{json,png}
  results/substrate/phase_1_1/phase_1_with_kwta_calibrated.{json,png}
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
# Stimulus + measurement helpers (unchanged from Phase 1)
# ---------------------------------------------------------------------------


def define_pattern(
    n_neurons: int = 500, pattern_size: int = 25, seed: int = 0,
) -> np.ndarray:
    """Pick ``pattern_size`` random N indices."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_neurons, size=pattern_size, replace=False))


def make_external_input(
    n_neurons: int, pattern: np.ndarray, strength: float = 1.0,
) -> np.ndarray:
    """External drive that pushes pattern N up to ``strength`` and
    leaves all others at zero."""
    inp = np.zeros(n_neurons, dtype=np.float32)
    inp[pattern] = strength
    return inp


def measure_completion(
    substrate: Substrate,
    pattern: np.ndarray,
    cue_fraction: float = 0.5,
    T_settle: int = 30,
    cue_seed: int = 123,
) -> dict:
    """Pattern-completion test (non-destructive).

    Reset → clamp cue (half the pattern) → settle → measure how
    much non-cued pattern N activate relative to background.
    Snapshots and restores substrate state so the test does not
    bias the trained weights.
    """
    n_neurons = substrate.n_neurons
    rng = np.random.default_rng(cue_seed)

    n_cue = max(1, int(round(len(pattern) * cue_fraction)))
    cue_indices = rng.choice(pattern, size=n_cue, replace=False)
    target_indices = np.array(
        [p for p in pattern if p not in set(cue_indices.tolist())]
    )

    saved_W = substrate.connectivity.W.copy()
    saved_age = substrate.system_age
    saved_activations = substrate.activations.copy()
    saved_step = substrate.step_count
    saved_drift = substrate.background.drift_state
    saved_rng_state = substrate.background.rng.bit_generator.state

    substrate.activations = np.zeros(n_neurons, dtype=np.float32)
    cue_input = np.zeros(n_neurons, dtype=np.float32)
    cue_input[cue_indices] = 1.0

    final = None
    for _ in range(T_settle):
        final = substrate.step(external_input=cue_input)

    substrate.connectivity.W[:] = saved_W
    substrate.system_age = saved_age
    substrate.activations = saved_activations
    substrate.step_count = saved_step
    substrate.background.drift_state = saved_drift
    substrate.background.rng.bit_generator.state = saved_rng_state

    target_acts = final[target_indices]
    non_pattern = np.array(
        [i for i in range(n_neurons) if i not in set(pattern.tolist())]
    )
    bg_acts = final[non_pattern]

    return {
        "n_cue": int(len(cue_indices)),
        "n_target": int(len(target_indices)),
        "target_mean": float(target_acts.mean()),
        "target_max": float(target_acts.max()),
        "target_above_threshold_frac": float((target_acts > 0.3).mean()),
        "background_mean": float(bg_acts.mean()),
        "background_above_threshold_frac": float((bg_acts > 0.3).mean()),
        "completion_score": float(target_acts.mean() - bg_acts.mean()),
    }


# ---------------------------------------------------------------------------
# Per-config trial runner
# ---------------------------------------------------------------------------


def run_trial(
    *,
    label: str,
    eta: float,
    lambda_decay: float,
    threshold: float,
    external_strength: float,
    sparsity_target: float = 0.05,
    n_neurons: int = 500,
    pattern_size: int = 25,
    T_present: int = 20,
    T_rest: int = 10,
    K_repeats: int = 50,
    seed: int = 42,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Run one Phase-1.1 trial. Writes a per-trial PNG+JSON pair
    and returns a small dict summarizing the run for cross-trial
    comparison.
    """
    print()
    print(f"=== Phase 1.1 trial: {label} ===")
    print(f"N={n_neurons}, pattern_size={pattern_size}")
    print(f"K_repeats={K_repeats}, T_present={T_present}, T_rest={T_rest}")
    print(
        f"eta={eta}, lambda_decay={lambda_decay}, threshold={threshold}, "
        f"sparsity_target={sparsity_target}, "
        f"external_strength={external_strength}"
    )
    print()

    substrate = Substrate(
        n_neurons=n_neurons,
        eta=eta,
        lambda_decay=lambda_decay,
        threshold=threshold,
        sparsity_target=sparsity_target,
        seed=seed,
    )
    pattern = define_pattern(n_neurons, pattern_size, seed=0)
    print(f"Pattern N (first 10): {pattern[:10]}")
    print(
        f"Initial connectivity: "
        f"{substrate.connectivity.connection_count()} connections"
    )

    checkpoint_intervals = {0, 10, 20, 35, 50}
    history = {"step": [], "sparsity": [], "total_weight": []}
    completion_results: list[dict] = []

    print("Measuring baseline completion (K=0) ...")
    baseline = measure_completion(substrate, pattern)
    completion_results.append({"K": 0, **baseline})
    print(f"  baseline completion_score = {baseline['completion_score']:+.4f}")

    external_input = make_external_input(
        n_neurons, pattern, strength=external_strength,
    )

    for k in range(1, K_repeats + 1):
        for _ in range(T_present):
            substrate.step(external_input=external_input)
            history["step"].append(substrate.step_count)
            history["sparsity"].append(substrate.sparsity())
            history["total_weight"].append(substrate.total_weight())
        for _ in range(T_rest):
            substrate.step(external_input=None)
            history["step"].append(substrate.step_count)
            history["sparsity"].append(substrate.sparsity())
            history["total_weight"].append(substrate.total_weight())
        if k in checkpoint_intervals:
            result = measure_completion(substrate, pattern)
            completion_results.append({"K": k, **result})
            print(
                f"K={k:2d}: completion_score={result['completion_score']:+.4f}  "
                f"target_mean={result['target_mean']:.3f}  "
                f"bg_mean={result['background_mean']:.3f}  "
                f"target>0.3 frac={result['target_above_threshold_frac']:.2f}"
            )

    if out_dir is None:
        out_dir = _REPO_ROOT / "results" / "substrate" / "phase_1_1"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    axes[0].plot(history["step"], history["sparsity"], linewidth=0.6)
    axes[0].axhline(0.05, color="r", linestyle="--", label="H5 target ≈ 5 %")
    axes[0].set(
        xlabel="step", ylabel="sparsity (frac active)",
        title=f"Substrate sparsity over time — {label}",
    )
    axes[0].legend()

    axes[1].plot(history["step"], history["total_weight"], linewidth=0.6)
    axes[1].set(
        xlabel="step", ylabel="total structural weight",
        title=f"Accumulated learning over time — {label}",
    )

    Ks = [r["K"] for r in completion_results]
    target_means = [r["target_mean"] for r in completion_results]
    bg_means = [r["background_mean"] for r in completion_results]
    scores = [r["completion_score"] for r in completion_results]
    axes[2].plot(Ks, target_means, "g-o", label="target (non-cued pattern N)")
    axes[2].plot(Ks, bg_means, "r-o", label="background (non-pattern N)")
    axes[2].plot(Ks, scores, "b--", label="score (target − background)")
    axes[2].axhline(0.10, color="grey", linestyle=":", label="PASS threshold")
    axes[2].set(
        xlabel="K (pattern presentations)", ylabel="mean activation",
        title=f"P1 — Pattern completion vs exposure — {label}",
    )
    axes[2].legend()

    plt.tight_layout()
    plot_path = out_dir / f"phase_1_with_kwta_{label}.png"
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"Plot → {plot_path}")

    json_path = out_dir / f"phase_1_with_kwta_{label}.json"
    with json_path.open("w") as f:
        json.dump(
            {
                "label": label,
                "config": {
                    "n_neurons": n_neurons,
                    "pattern_size": pattern_size,
                    "K_repeats": K_repeats,
                    "T_present": T_present,
                    "T_rest": T_rest,
                    "seed": seed,
                    "eta": eta,
                    "lambda_decay": lambda_decay,
                    "threshold": threshold,
                    "sparsity_target": sparsity_target,
                    "external_strength": external_strength,
                },
                "completion_results": completion_results,
                "final_total_weight": substrate.total_weight(),
                "final_sparsity": substrate.sparsity(),
                "final_system_age": substrate.system_age,
            },
            f,
            indent=2,
        )
    print(f"JSON → {json_path}")

    final_score = completion_results[-1]["completion_score"]
    if final_score > 0.10:
        verdict = "PASS"
    elif final_score > 0.02:
        verdict = "WEAK"
    else:
        verdict = "FAIL"

    return {
        "label": label,
        "verdict": verdict,
        "final_score": final_score,
        "final_sparsity": substrate.sparsity(),
        "final_total_weight": substrate.total_weight(),
        "completion_results": completion_results,
    }


# ---------------------------------------------------------------------------
# Entry point — runs both trials, prints comparison
# ---------------------------------------------------------------------------


def _verdict_reason(verdict: str) -> str:
    if verdict == "PASS":
        return (
            "target activations are significantly above the background; "
            "pattern formation works as THEORY.md P1 predicts."
        )
    if verdict == "WEAK":
        return (
            "directional signal but completion is marginal — parameter "
            "tuning likely needed (η, λ_decay, threshold, sparsity)."
        )
    return (
        "no detectable pattern formation; either H1+H3 are wrong or "
        "the specific implementation parameters miss the operating "
        "regime."
    )


def main() -> int:
    out_dir = _REPO_ROOT / "results" / "substrate" / "phase_1_1"

    trials = [
        {
            "label": "defaults",
            "eta": 0.01,
            "lambda_decay": 0.001,
            "threshold": 0.3,
            "external_strength": 1.0,
        },
        {
            "label": "calibrated",
            "eta": 0.0015,
            "lambda_decay": 0.004,
            "threshold": 0.35,
            "external_strength": 0.7,
        },
    ]

    print("====================================================")
    print("Phase 1.1 — P1 under k-WTA H5 enforcement")
    print("Two configurations: defaults vs Phase-1 calibrated")
    print("====================================================")

    results = []
    for trial in trials:
        results.append(run_trial(out_dir=out_dir, **trial))

    # Cross-trial comparison report.
    print()
    print("====================================================")
    print("Phase 1.1 — Verdict summary")
    print("====================================================")
    for r in results:
        print(
            f"  [{r['verdict']}]  {r['label']:<10}  "
            f"final_score={r['final_score']:+.4f}  "
            f"final_sparsity={r['final_sparsity']:.3f}  "
            f"final_total_W={r['final_total_weight']:.1f}"
        )
    for r in results:
        print()
        print(f"  {r['label']}: {_verdict_reason(r['verdict'])}")

    # Cross-trial interpretation note.
    defaults_v = next(r["verdict"] for r in results if r["label"] == "defaults")
    calibrated_v = next(
        r["verdict"] for r in results if r["label"] == "calibrated"
    )
    print()
    print("Interpretation:")
    if defaults_v == "PASS" and calibrated_v == "PASS":
        print(
            "  k-WTA is a true structural fix — defaults that previously "
            "blew up now PASS, and the calibrated regime is preserved. "
            "Robustness window extended."
        )
    elif defaults_v != "PASS" and calibrated_v == "PASS":
        print(
            "  k-WTA alone is insufficient for the aggressive defaults; "
            "calibration is still needed (η is dominant). Confirms that "
            "k-WTA caps sparsity but does not cap weight magnitude — "
            "decay still has to do that work."
        )
    elif defaults_v == "PASS" and calibrated_v != "PASS":
        print(
            "  Unexpected: k-WTA breaks the previously-tuned regime. "
            "Investigate interference between k-WTA and small-η Hebbian — "
            "possibly k-WTA prunes too aggressively given low drive."
        )
    else:
        print(
            "  Both configs fail to PASS — k-WTA changes the operating "
            "point materially; new calibration pass needed in Phase 1.2."
        )

    final_scores = [r["final_score"] for r in results]
    # Exit OK iff at least one config PASS-es. Telemetry-only return
    # code; the cross-trial report above is the real deliverable.
    return 0 if max(final_scores) > 0.10 else (
        1 if max(final_scores) > 0.02 else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
