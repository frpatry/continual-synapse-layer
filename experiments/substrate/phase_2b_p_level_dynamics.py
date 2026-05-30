"""Phase 2b — P-level dynamics + P-P connection emergence.

Builds on Phase 2a (P entities emerge from co-active N pairs) and
checks three things:

1. P entities have their own activation dynamics (not a derived mean
   of components) — verified by ``p_sparsity()`` being bounded by k-WTA.
2. P-P connections form via covariance Hebbian. The creation gate
   (``min_coactivation_to_create_pp``) prevents noise-floor edges.
3. Because Phase 2a P emergence is ~100 % selective for pattern pairs,
   *all* P-P connections must end up between within-pattern P entities.
   Selectivity at P-P level is therefore inherited from P-emergence
   selectivity — the empirical claim is that the inheritance actually
   holds.

Verdict on the final fraction of P-P connections that connect two
within-pattern P entities:

  >= 0.80  PASS — selective P-P emergence
  >= 0.50  WEAK — directional but noisy
  else     FAIL — random or no connections


Calibration (overrides to Substrate defaults; rationale below):

* ``p_sparsity_target = 0.5`` (default 0.05). N-level 0.05 works
  because n_neurons is large. The live P pool is small — Phase 2a
  stabilises at ~11 P. ``k = max(1, int(0.05·11)) = 1`` would keep
  only ONE P active per step → P pairs are never co-active → P-P
  connections never form → trivial FAIL. With 0.5, k=5 of ~11 P co-
  fire per step — dense enough for Hebbian, sparse enough to keep
  the covariance baseline meaningful.

* ``alpha_n_to_p = 1.0`` (default 0.3). Pattern N stabilise around
  activation 0.5 (k-WTA holds them there). With alpha=0.3 the P
  input from N is 0.15, *below* p_threshold=0.3 → P never fires.
  alpha=1.0 puts the input at ~0.5, comfortably above p_threshold=0.2.

* ``p_threshold = 0.2`` (default 0.3). Combined with alpha=1.0 so
  pattern-pair P clear the threshold reliably while pure-noise input
  still gets zeroed.

* ``min_coactivation_to_create_pp = 0.01`` (default 0.1). With k-WTA
  capping P.activation at ~0.3, raw co-activation is 0.3·0.3 ≈ 0.09.
  The default gate of 0.1 blocks every legitimate co-activation as
  "noise". 0.01 admits real P-P events while still filtering the
  noise floor (sub-0.1 activations × sub-0.1 activations).

* ``eta_pp = 0.05`` (default 0.005). The covariance Hebbian signal
  is small (~0.07) because k-WTA flattens activations. The default
  eta_pp is calibrated for the N-level activation range; 10× is
  needed to grow P-P edges to observable magnitude within K=100.

The Substrate's *defaults* stay at the spec values for API
consistency with the N-level recipe; this experiment captures what
the spec was trying to express at the P scale.


Phase 2a regression note:

  Phase 2b changes how P.activation is computed. In Phase 2a it was
  ``(N_i + N_j) / 2`` every step → growth term in
  ``_decay_and_dissolve_p`` was always non-zero for active pattern P
  → all 11 pattern-pair P persisted. In Phase 2b, P.activation is
  k-WTA-thresholded → most P sit at 0 most steps → growth ≈ 0 →
  decay eventually wins for some P. Selectivity (no false-positive
  noise P) is preserved by construction; the *count* may drop. We
  re-run the Phase 2a experiment in our verification suite to measure
  the actual count regression.


Outputs:

  results/substrate/phase_2b/phase_2b_results.png
  results/substrate/phase_2b/phase_2b_results.json
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


def classify_pp_connection(
    p_id_a: int,
    p_id_b: int,
    p_entities: dict,
    pattern_set: set[int],
) -> str:
    """Tag a single P-P edge as ``within_pattern`` / ``bridging`` / ``noise``.

    Pulled out so the experiment + verdict use exactly the same rule.
    """
    p_a = p_entities.get(p_id_a)
    p_b = p_entities.get(p_id_b)
    if p_a is None or p_b is None:
        # One endpoint dissolved already — bookkeeping mishap, skip.
        return "stale"
    a_in_pattern = (p_a.components[0] in pattern_set
                    and p_a.components[1] in pattern_set)
    b_in_pattern = (p_b.components[0] in pattern_set
                    and p_b.components[1] in pattern_set)
    if a_in_pattern and b_in_pattern:
        return "within_pattern"
    if a_in_pattern or b_in_pattern:
        return "bridging"
    return "noise"


def main() -> int:
    # ---- configuration ----
    n_neurons = 200
    k_connectivity = 30
    pattern_size = 10
    T_present = 15
    T_rest = 60  # same spacing math as Phase 2a (see that experiment's docstring)
    K_repeats = 100
    checkpoint_every = 10

    substrate = Substrate(
        # N-level (Phase 1.1 calibrated for stability under k-WTA):
        n_neurons=n_neurons,
        k_connectivity=k_connectivity,
        eta=0.01,
        lambda_decay=0.001,
        threshold=0.3,
        sparsity_target=0.05,
        # Phase 2a emergence:
        theta_emergence=0.5,
        n_min_passes=3,
        pass_boost=0.1,
        pass_decay=0.95,
        # Phase 2b P-level dynamics (overrides — see module docstring):
        alpha_n_to_p=1.0,                   # default 0.3 — too weak
        p_threshold=0.2,                    # default 0.3
        p_sparsity_target=0.5,              # default 0.05 — too sparse for ~11 P
        p_background_noise_sigma=0.01,
        eta_pp=0.05,                        # default 0.005 — too slow
        lambda_pp_decay=0.001,
        min_coactivation_to_create_pp=0.01, # default 0.1 — too strict
        seed=42,
    )

    # ---- pattern + pattern-N set ----
    rng = np.random.default_rng(0)
    pattern = np.sort(rng.choice(n_neurons, size=pattern_size, replace=False))
    pattern_set = {int(x) for x in pattern}

    print("=== Phase 2b: P-level dynamics + P-P emergence ===")
    print(f"N={n_neurons}, k_connectivity={k_connectivity}, "
          f"pattern_size={pattern_size}")
    print(f"K_repeats={K_repeats}, T_present={T_present}, T_rest={T_rest}")
    print(f"p_sparsity_target={substrate.p_sparsity_target}  "
          f"(experiment override: see module docstring)")
    print(f"Pattern N: {pattern.tolist()}")
    print()

    # ---- training ----
    external = np.zeros(n_neurons, dtype=np.float32)
    external[pattern] = 0.7

    history: list[dict] = []

    for k in range(1, K_repeats + 1):
        for _ in range(T_present):
            substrate.step(external_input=external)
        for _ in range(T_rest):
            substrate.step(external_input=None)

        if k % checkpoint_every == 0:
            counts = {"within_pattern": 0, "bridging": 0, "noise": 0,
                      "stale": 0}
            total_weight = 0.0
            for (a, b, w) in substrate.p_connectivity.all_pairs():
                tag = classify_pp_connection(
                    a, b, substrate.p_entities, pattern_set,
                )
                counts[tag] += 1
                total_weight += w
            p_total = substrate.p_count()
            pp_total = substrate.p_connection_count()
            frac_within = (counts["within_pattern"] / pp_total
                           if pp_total else 0.0)
            mean_p_act = float(np.mean(
                [p.activation for p in substrate.p_entities.values()]
            )) if substrate.p_entities else 0.0
            history.append({
                "K": k,
                "step": substrate.step_count,
                "p_total": p_total,
                "pp_total": pp_total,
                "pp_within_pattern": counts["within_pattern"],
                "pp_bridging": counts["bridging"],
                "pp_noise": counts["noise"],
                "pp_total_weight": total_weight,
                "p_sparsity": substrate.p_sparsity(),
                "mean_p_activation": mean_p_act,
            })
            print(
                f"K={k:3d}: P={p_total:3d}  "
                f"P-P={pp_total:3d}  "
                f"within={counts['within_pattern']:3d}  "
                f"bridging={counts['bridging']:3d}  "
                f"noise={counts['noise']:3d}  "
                f"frac_within={frac_within:.2%}  "
                f"p_sparsity={substrate.p_sparsity():.2f}"
            )

    # ---- outputs ----
    results_dir = _REPO_ROOT / "results" / "substrate" / "phase_2b"
    results_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 11))

    Ks = [h["K"] for h in history]
    axes[0].plot(Ks, [h["p_total"] for h in history], "b-o", label="P entities")
    axes[0].plot(Ks, [h["pp_total"] for h in history], "m-o", label="P-P connections")
    axes[0].set(xlabel="K (presentations)", ylabel="count",
                title="P and P-P counts over time")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(Ks, [h["pp_within_pattern"] for h in history], "g-o",
                 label="within-pattern P-P")
    axes[1].plot(Ks, [h["pp_bridging"] for h in history], "y-o",
                 label="bridging P-P")
    axes[1].plot(Ks, [h["pp_noise"] for h in history], "r-o",
                 label="noise P-P")
    axes[1].set(xlabel="K", ylabel="# P-P connections",
                title="P-P emergence: classification over time")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(Ks, [h["p_sparsity"] for h in history], "c-o",
                 label="P-level sparsity")
    axes[2].plot(Ks, [h["mean_p_activation"] for h in history], "k-o",
                 label="mean P activation")
    axes[2].axhline(substrate.p_sparsity_target, color="grey",
                    linestyle=":", label="k-WTA target")
    axes[2].set(xlabel="K", ylabel="value",
                title="P activation dynamics")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plot_path = results_dir / "phase_2b_results.png"
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"\nPlot → {plot_path}")

    json_path = results_dir / "phase_2b_results.json"
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
                    "p_sparsity_target": substrate.p_sparsity_target,
                },
                "pattern_n": pattern.tolist(),
                "history": history,
                "final_p_count": substrate.p_count(),
                "final_pp_count": substrate.p_connection_count(),
            },
            f,
            indent=2,
        )
    print(f"JSON → {json_path}")

    # ---- verdict ----
    print()
    print("=== Verdict (Phase 2b) ===")
    if not history:
        print("FAIL: no checkpoints collected.")
        return 2
    final = history[-1]
    pp = final["pp_total"]
    within = final["pp_within_pattern"]
    print(f"Final P entities: {final['p_total']}")
    print(f"Final P-P connections: {pp}")
    print(f"  within-pattern: {within}")
    print(f"  bridging:       {final['pp_bridging']}")
    print(f"  noise:          {final['pp_noise']}")
    print(f"Final P sparsity: {final['p_sparsity']:.2f}  "
          f"(k-WTA target {substrate.p_sparsity_target:.2f})")
    print(f"Mean P activation: {final['mean_p_activation']:.3f}")
    if pp == 0:
        print("FAIL: no P-P connections formed — Hebbian gate may be too strict "
              "or P-level k-WTA admits too few simultaneous winners.")
        return 2
    frac = within / pp
    print(f"P-P within-pattern fraction: {frac:.2%}")
    if frac >= 0.80:
        print("PASS: P-P connections concentrate on within-pattern P pairs.")
        return 0
    if frac >= 0.50:
        print("WEAK: directional but noisy.")
        return 1
    print("FAIL: P-P connectivity essentially random.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
