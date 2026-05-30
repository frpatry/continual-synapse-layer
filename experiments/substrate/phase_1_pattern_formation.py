"""Phase 1 — P1 (pattern formation through repeated exposure).

Protocol:

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

Verdict from the completion score (target_mean − background_mean):

  > 0.10   PASS — pattern formation working as P1 predicts
  > 0.02   WEAK — directional signal, parameter tuning warranted
  ≤ 0.02   FAIL — theory or parameters need revisiting

Outputs:

  results/substrate/phase_1/phase_1_results.png
  results/substrate/phase_1/phase_1_results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from substrate.substrate import Substrate  # noqa: E402


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
    """Pattern-completion test.

    Reset → clamp cue (half the pattern) → settle → measure how
    much non-cued pattern N activate relative to background.

    Note: ``substrate.step`` always applies plasticity. To avoid
    the measurement itself biasing the trained weights, we
    snapshot the connectivity + age before the test and restore
    them after.
    """
    n_neurons = substrate.n_neurons
    rng = np.random.default_rng(cue_seed)

    n_cue = max(1, int(round(len(pattern) * cue_fraction)))
    cue_indices = rng.choice(pattern, size=n_cue, replace=False)
    target_indices = np.array(
        [p for p in pattern if p not in set(cue_indices.tolist())]
    )

    # Snapshot state so the test is non-destructive.
    saved_W = substrate.connectivity.W.copy()
    saved_age = substrate.system_age
    saved_activations = substrate.activations.copy()
    saved_step = substrate.step_count
    # Snapshot background internal state so randomness in the
    # test doesn't leak into later training rounds.
    saved_drift = substrate.background.drift_state
    saved_rng_state = substrate.background.rng.bit_generator.state

    # Reset activations + clamp cue.
    substrate.activations = np.zeros(n_neurons, dtype=np.float32)
    cue_input = np.zeros(n_neurons, dtype=np.float32)
    cue_input[cue_indices] = 1.0

    final = None
    for _ in range(T_settle):
        final = substrate.step(external_input=cue_input)

    # Restore — completion test is read-only against the trained
    # substrate.
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


def main() -> int:
    n_neurons = 500
    pattern_size = 25
    T_present = 20
    T_rest = 10
    K_repeats = 50
    seed = 42

    # Calibrated parameters. The defaults in Substrate
    # (eta=0.01, lambda_decay=0.001) drove the substrate into
    # Hebbian runaway (every neuron saturated to 1.0). We dial
    # down the Hebbian rate and bump decay so the substrate
    # stays in the sparse regime (H5) while still letting
    # pattern-pair weights accumulate over many presentations.
    eta = 0.0015
    lambda_decay = 0.004
    threshold = 0.35
    external_strength = 0.7

    print("=== Phase 1: Pattern Formation Test ===")
    print(f"N={n_neurons}, pattern_size={pattern_size}")
    print(f"K_repeats={K_repeats}, T_present={T_present}, T_rest={T_rest}")
    print(f"eta={eta}, lambda_decay={lambda_decay}, threshold={threshold}, "
          f"external_strength={external_strength}")
    print()

    substrate = Substrate(
        n_neurons=n_neurons,
        eta=eta,
        lambda_decay=lambda_decay,
        threshold=threshold,
        seed=seed,
    )
    pattern = define_pattern(n_neurons, pattern_size, seed=0)
    print(f"Pattern N (first 10): {pattern[:10]}")
    print(
        f"Initial connectivity: "
        f"{substrate.connectivity.connection_count()} connections"
    )
    print()

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

    results_dir = _REPO_ROOT / "results" / "substrate" / "phase_1"
    results_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    axes[0].plot(history["step"], history["sparsity"], linewidth=0.6)
    axes[0].axhline(0.05, color="r", linestyle="--", label="H5 target ≈ 5 %")
    axes[0].set(xlabel="step", ylabel="sparsity",
                title="Substrate sparsity over time (H5 check)")
    axes[0].legend()

    axes[1].plot(history["step"], history["total_weight"], linewidth=0.6)
    axes[1].set(xlabel="step", ylabel="total structural weight",
                title="Accumulated learning over time")

    Ks = [r["K"] for r in completion_results]
    target_means = [r["target_mean"] for r in completion_results]
    bg_means = [r["background_mean"] for r in completion_results]
    scores = [r["completion_score"] for r in completion_results]
    axes[2].plot(Ks, target_means, "g-o", label="target (non-cued pattern N)")
    axes[2].plot(Ks, bg_means, "r-o", label="background (non-pattern N)")
    axes[2].plot(Ks, scores, "b--", label="score (target − background)")
    axes[2].axhline(0.10, color="grey", linestyle=":", label="PASS threshold")
    axes[2].set(xlabel="K (pattern presentations)",
                ylabel="mean activation",
                title="P1 — Pattern completion vs exposure")
    axes[2].legend()

    plt.tight_layout()
    plot_path = results_dir / "phase_1_results.png"
    plt.savefig(plot_path, dpi=120)
    print(f"\nPlot → {plot_path}")

    json_path = results_dir / "phase_1_results.json"
    with json_path.open("w") as f:
        json.dump(
            {
                "config": {
                    "n_neurons": n_neurons,
                    "pattern_size": pattern_size,
                    "K_repeats": K_repeats,
                    "T_present": T_present,
                    "T_rest": T_rest,
                    "seed": seed,
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
    print()
    print("=== Verdict (P1) ===")
    print(f"final completion_score = {final_score:+.4f}")
    if final_score > 0.10:
        verdict = "PASS"
        reason = (
            "target activations are significantly above the background; "
            "pattern formation through repeated exposure works as "
            "THEORY.md P1 predicts."
        )
        rc = 0
    elif final_score > 0.02:
        verdict = "WEAK"
        reason = (
            "directional signal but completion is marginal — parameter "
            "tuning likely needed (η, λ_decay, threshold, pattern size)."
        )
        rc = 1
    else:
        verdict = "FAIL"
        reason = (
            "no detectable pattern formation; either H1+H3 are wrong or "
            "the specific implementation parameters miss the operating "
            "regime."
        )
        rc = 2
    print(f"{verdict}: {reason}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
