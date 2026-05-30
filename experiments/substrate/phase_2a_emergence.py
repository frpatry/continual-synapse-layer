"""Phase 2a — P2 (P entity emergence is selective for pattern pairs).

Protocol:

1. Small substrate (N=200, k=30, k-WTA at 5 %).
2. Define a 10-N pattern.
3. Present the pattern with strong external input across many
   alternating present / rest cycles.
4. At checkpoints, count how many P entities have emerged and
   classify them by whether both component N belong to the pattern
   ("correct") or not ("noise").

Verdict from the final P distribution:

  in_pattern / total >= 0.7   PASS — emergence is selective
  in_pattern / total >= 0.4   WEAK — some signal but noisy
  otherwise / total == 0      FAIL — mechanism didn't fire
  else                         FAIL — emergence is essentially random

Spacing-effect calibration:

  The Substrate defaults are ``n_min_passes=3`` with
  ``pass_decay=0.95`` and hysteresis ``[0.05, 0.10]``. During a
  presentation, candidacy reaches a steady-state of roughly
  ``boost * coact / (1 - decay) ≈ 0.5`` for two pattern-active N.
  Decaying from 0.5, ``0.95^T_rest`` must drop us below ``th_low=0.05``
  to allow a second pass to be counted on the next presentation:
  ``T_rest >= log(0.1) / log(0.95) ≈ 45``.

  We use ``T_rest = 60`` here — comfortably past that minimum so
  three distinct passes accumulate within the experiment's K cycles.
  This is the "spacing interval" of the substrate; calling it out so
  it doesn't read as an arbitrary tweak.

Outputs:

  results/substrate/phase_2a/phase_2a_results.png
  results/substrate/phase_2a/phase_2a_results.json
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


def main() -> int:
    # ---- configuration ----
    n_neurons = 200
    k_connectivity = 30
    pattern_size = 10
    T_present = 15
    T_rest = 60  # see spacing-effect calibration in module docstring
    K_repeats = 100
    checkpoint_every = 10

    substrate = Substrate(
        n_neurons=n_neurons,
        k_connectivity=k_connectivity,
        eta=0.01,
        lambda_decay=0.001,
        threshold=0.3,
        sparsity_target=0.05,
        # P-emergence params: spec defaults.
        theta_emergence=0.5,
        n_min_passes=3,
        pass_boost=0.1,
        pass_decay=0.95,
        pass_theta_high=0.1,
        pass_theta_low=0.05,
        p_weight_decay=0.005,
        p_viability_threshold=0.1,
        seed=42,
    )

    # ---- pattern + ground-truth pair set ----
    rng = np.random.default_rng(0)
    pattern = np.sort(rng.choice(n_neurons, size=pattern_size, replace=False))
    pattern_set = {int(x) for x in pattern}
    # ``pattern`` is sorted, so (pattern[i], pattern[j]) for i<j is
    # already canonical and matches PEntity.components convention.
    pattern_pairs: set[tuple[int, int]] = {
        (int(pattern[i]), int(pattern[j]))
        for i in range(pattern_size)
        for j in range(i + 1, pattern_size)
    }

    print("=== Phase 2a: P emergence test (P2) ===")
    print(f"N={n_neurons}, k_connectivity={k_connectivity}, "
          f"pattern_size={pattern_size}")
    print(f"K_repeats={K_repeats}, T_present={T_present}, T_rest={T_rest}")
    print(f"Pattern N: {pattern.tolist()}")
    print(f"Max possible pattern pairs (C(n,2)): {len(pattern_pairs)}")
    print()

    # ---- training + checkpointing ----
    external = np.zeros(n_neurons, dtype=np.float32)
    external[pattern] = 0.7

    history: list[dict] = []

    for k in range(1, K_repeats + 1):
        for _ in range(T_present):
            substrate.step(external_input=external)
        for _ in range(T_rest):
            substrate.step(external_input=None)

        if k % checkpoint_every == 0:
            p_entities = list(substrate.p_entities.values())
            n_total = len(p_entities)
            n_in_pattern = sum(
                1 for p in p_entities if p.components in pattern_pairs
            )
            # A "half-in-pattern" P bridges a pattern N and a noise N
            # — useful diagnostic for whether spillover is happening.
            n_half_in_pattern = sum(
                1 for p in p_entities
                if (p.components[0] in pattern_set)
                ^ (p.components[1] in pattern_set)
            )
            n_off_pattern = n_total - n_in_pattern - n_half_in_pattern
            history.append({
                "K": k,
                "step": substrate.step_count,
                "p_total": n_total,
                "p_in_pattern": n_in_pattern,
                "p_half_in_pattern": n_half_in_pattern,
                "p_off_pattern": n_off_pattern,
            })
            frac_correct = n_in_pattern / max(n_total, 1)
            print(
                f"K={k:3d}: P_total={n_total:3d}  "
                f"in_pattern={n_in_pattern:3d}  "
                f"half_in_pattern={n_half_in_pattern:3d}  "
                f"off_pattern={n_off_pattern:3d}  "
                f"frac_correct={frac_correct:.2%}"
            )

    # ---- outputs ----
    results_dir = _REPO_ROOT / "results" / "substrate" / "phase_2a"
    results_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    Ks = [h["K"] for h in history]
    axes[0].plot(Ks, [h["p_total"] for h in history], "b-o", label="total P")
    axes[0].plot(Ks, [h["p_in_pattern"] for h in history], "g-o",
                 label="both N in pattern")
    axes[0].plot(Ks, [h["p_half_in_pattern"] for h in history], "y-o",
                 label="bridging (one in, one out)")
    axes[0].plot(Ks, [h["p_off_pattern"] for h in history], "r-o",
                 label="both N off pattern")
    axes[0].set(xlabel="K (pattern presentations)",
                ylabel="number of live P entities",
                title="P emergence over time")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    final = history[-1] if history else {
        "p_in_pattern": 0, "p_half_in_pattern": 0, "p_off_pattern": 0,
    }
    bars = ["in pattern", "bridging", "off pattern"]
    counts = [
        final["p_in_pattern"],
        final["p_half_in_pattern"],
        final["p_off_pattern"],
    ]
    colors = ["green", "gold", "red"]
    axes[1].bar(bars, counts, color=colors)
    axes[1].set(
        ylabel=f"# P at K={final.get('K', '?')}",
        title="Final P distribution by component membership",
    )
    for i, c in enumerate(counts):
        axes[1].text(i, c, str(c), ha="center", va="bottom")

    plt.tight_layout()
    plot_path = results_dir / "phase_2a_results.png"
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"\nPlot → {plot_path}")

    json_path = results_dir / "phase_2a_results.json"
    with json_path.open("w") as f:
        json.dump(
            {
                "config": {
                    "n_neurons": n_neurons,
                    "k_connectivity": k_connectivity,
                    "pattern_size": pattern_size,
                    "K_repeats": K_repeats,
                    "T_present": T_present,
                    "T_rest": T_rest,
                    "seed": 42,
                },
                "pattern_n": pattern.tolist(),
                "pattern_pairs_max": len(pattern_pairs),
                "history": history,
                "final_p_count": substrate.p_count(),
                "emergence_event_count": len(substrate.p_emergence_history),
            },
            f,
            indent=2,
        )
    print(f"JSON → {json_path}")

    # ---- verdict ----
    print()
    print("=== Verdict (P2) ===")
    if not history:
        print("FAIL: no checkpoints collected.")
        return 2
    final = history[-1]
    total = final["p_total"]
    correct = final["p_in_pattern"]
    print(f"Total P emerged: {total}")
    print(f"Pattern-pair P:  {correct}")
    print(f"Bridging P:      {final['p_half_in_pattern']}")
    print(f"Noise P:         {final['p_off_pattern']}")
    if total == 0:
        print("FAIL: no P emerged — emergence mechanism didn't fire.")
        return 2
    frac = correct / total
    print(f"Correctness fraction: {frac:.2%}")
    if frac >= 0.7:
        print("PASS: P emergence concentrated on pattern pairs (≥ 70 %).")
        return 0
    if frac >= 0.4:
        print("WEAK: directional but noisy — consider spacing / threshold tuning.")
        return 1
    print("FAIL: P emergence essentially random with respect to the pattern.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
