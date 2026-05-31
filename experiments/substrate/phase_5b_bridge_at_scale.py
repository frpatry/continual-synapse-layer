"""Phase 5b — Bridge phenomenon at scale (M sequentially-trained patterns).

Extends Phase 5a (which trained 2 patterns and found that 14 bridge P
entities emerged spontaneously during idle periods between patterns)
to ask: how does this scale with M?

Three questions:
  1. P-count per pattern: does each pattern get its share, or do
     later patterns crowd out earlier ones?
  2. Bridge population: does it grow O(M), O(M²), or worse?
  3. Recall fidelity: do earlier patterns survive later training,
     and can the substrate recall each pattern individually after
     all training is done?

The 2-level architecture is "good enough" for the spatial form of P5
at M=2 (Phase 5a). The empirical question now is whether it stays
good enough at higher M, or whether interference / bridge explosion
forces us toward S-level grouping (Phase 6).

Protocol:
  1. Substrate N=500, k_connectivity=30, full Phase 2c parameters.
  2. Five disjoint patterns of size 10 each.
  3. Train pattern 0, then 1, then 2, then 3, then 4 (K=80 each).
  4. After each stage, record P-counts per pattern + bridges per pair
     + individual recall for every already-trained pattern.
  5. After all training, run combined-cue tests with 2, 3, 4, 5
     patterns presented simultaneously.

Verdict:
  PASS            all patterns retain ≥ 2 P AND recall ≥ 0.4 AND
                  bridges avg < 10 per pair
  WEAK_recall     P preserved but recall < 0.4 on some patterns
  FAIL_capacity   some pattern lost all its P (catastrophic forgetting)
                  → motivates Phase 6 / S-level consolidation
  FAIL_bridges    bridges > 10 per pair on average (explosion)
                  → motivates S-level grouping / abstraction
  INCONCLUSIVE    mixed signals

Outputs:
  results/substrate/phase_5b/phase_5b_results.{png,json}
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
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


def define_M_patterns(
    n_neurons: int, M: int, pattern_size: int, seed: int = 0,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_neurons)
    patterns = [
        np.sort(perm[i * pattern_size:(i + 1) * pattern_size])
        for i in range(M)
    ]
    # Sanity: disjoint by construction.
    total = sum(len(p) for p in patterns)
    assert len({int(x) for p in patterns for x in p}) == total
    return patterns


def make_external(
    n_neurons: int, pattern: np.ndarray, strength: float = 0.7,
) -> np.ndarray:
    inp = np.zeros(n_neurons, dtype=np.float32)
    inp[pattern] = strength
    return inp


# ---------------------------------------------------------------------------
# Classification: which pattern(s) does each P entity belong to?
# ---------------------------------------------------------------------------


def classify_p_entities(
    p_entities: dict[int, PEntity], patterns: list[np.ndarray],
) -> dict[str, list[int]]:
    """Bucket each P by its components. Buckets:
        in_pattern_i     — both components in patterns[i]
        bridge_i_j       — one in patterns[i], one in patterns[j] (i<j)
        other            — any component outside every pattern
    """
    pattern_sets = [{int(x) for x in p} for p in patterns]
    out: dict[str, list[int]] = defaultdict(list)
    for pid, p in p_entities.items():
        c1, c2 = p.components
        i1 = next((i for i, ps in enumerate(pattern_sets) if c1 in ps), -1)
        i2 = next((i for i, ps in enumerate(pattern_sets) if c2 in ps), -1)
        if i1 == -1 or i2 == -1:
            out["other"].append(pid)
        elif i1 == i2:
            out[f"in_pattern_{i1}"].append(pid)
        else:
            lo, hi = sorted([i1, i2])
            out[f"bridge_{lo}_{hi}"].append(pid)
    return dict(out)


def summarize_classification(
    classification: dict[str, list[int]], M: int,
) -> dict[str, int]:
    out: dict[str, int] = {}
    for i in range(M):
        out[f"P_pattern_{i}"] = len(classification.get(f"in_pattern_{i}", []))
    bridges_total = 0
    for i in range(M):
        for j in range(i + 1, M):
            count = len(classification.get(f"bridge_{i}_{j}", []))
            bridges_total += count
            out[f"P_bridge_{i}_{j}"] = count
    out["P_bridges_total"] = bridges_total
    out["P_other"] = len(classification.get("other", []))
    out["P_total"] = (
        sum(out[f"P_pattern_{i}"] for i in range(M))
        + bridges_total
        + out["P_other"]
    )
    return out


# ---------------------------------------------------------------------------
# Training + recall tests
# ---------------------------------------------------------------------------


def train_one_pattern(
    substrate: Substrate,
    pattern: np.ndarray,
    K: int = 80,
    T_present: int = 15,
    T_rest: int = 60,
) -> None:
    external = make_external(substrate.n_neurons, pattern, strength=0.7)
    for _ in range(K):
        for _ in range(T_present):
            substrate.step(external_input=external)
        for _ in range(T_rest):
            substrate.step(external_input=None)


def _suspend_plasticity(substrate: Substrate) -> tuple[float, float, float]:
    """Return saved (eta, eta_pp, p_sparsity) and zero / relax current."""
    saved = (
        substrate.eta, substrate.eta_pp, substrate.p_sparsity_target,
    )
    substrate.eta = 0.0
    substrate.eta_pp = 0.0
    substrate.p_sparsity_target = 1.0
    return saved


def _restore_plasticity(
    substrate: Substrate, saved: tuple[float, float, float],
) -> None:
    substrate.eta, substrate.eta_pp, substrate.p_sparsity_target = saved


def _reset_dynamic_state(substrate: Substrate) -> dict[int, float]:
    saved_p_acts = {pid: p.activation for pid, p in substrate.p_entities.items()}
    substrate.activations = np.zeros(substrate.n_neurons, dtype=np.float32)
    for p in substrate.p_entities.values():
        p.activation = 0.0
    return saved_p_acts


def _restore_dynamic_state(
    substrate: Substrate,
    saved_acts: np.ndarray,
    saved_p_acts: dict[int, float],
) -> None:
    substrate.activations = saved_acts
    for pid, act in saved_p_acts.items():
        if pid in substrate.p_entities:
            substrate.p_entities[pid].activation = act


def test_individual_recall(
    substrate: Substrate,
    pattern: np.ndarray,
    cue_fraction: float = 0.5,
    T_settle: int = 30,
    cue_seed: int = 123,
) -> float:
    """Phase-2c-style partial-recall on one pattern. Returns fraction
    of un-cued pattern N reactivated (activation > 0.1)."""
    rng = np.random.default_rng(cue_seed)
    n_cue = max(1, int(len(pattern) * cue_fraction))
    cue_indices = np.sort(rng.choice(pattern, size=n_cue, replace=False))
    cue_set = {int(x) for x in cue_indices}
    target_indices = np.array(
        [int(n) for n in pattern if int(n) not in cue_set], dtype=int,
    )
    if len(target_indices) == 0:
        return 0.0

    saved_plast = _suspend_plasticity(substrate)
    saved_acts = substrate.activations.copy()
    saved_p_acts = _reset_dynamic_state(substrate)

    cue_input = np.zeros(substrate.n_neurons, dtype=np.float32)
    cue_input[cue_indices] = 1.0
    for _ in range(T_settle):
        substrate.step(external_input=cue_input)

    target_acts = substrate.activations[target_indices]
    completion = float((target_acts > 0.1).mean())

    _restore_plasticity(substrate, saved_plast)
    _restore_dynamic_state(substrate, saved_acts, saved_p_acts)
    return completion


def test_combined_recall(
    substrate: Substrate,
    patterns: list[np.ndarray],
    n_combined: int,
    T_settle: int = 30,
) -> list[float]:
    """Clamp the first ``n_combined`` patterns together as a combined cue.
    Returns the activation-fraction per pattern (in_pattern_i class)."""
    saved_plast = _suspend_plasticity(substrate)
    saved_acts = substrate.activations.copy()
    saved_p_acts = _reset_dynamic_state(substrate)

    combined_cue = np.zeros(substrate.n_neurons, dtype=np.float32)
    for p in patterns[:n_combined]:
        combined_cue[p] = 1.0
    for _ in range(T_settle):
        substrate.step(external_input=combined_cue)

    classification = classify_p_entities(substrate.p_entities, patterns)
    out: list[float] = []
    for i in range(n_combined):
        pids = classification.get(f"in_pattern_{i}", [])
        if not pids:
            out.append(0.0)
        else:
            active = sum(
                1 for pid in pids
                if substrate.p_entities[pid].activation > 0.1
            )
            out.append(active / len(pids))

    _restore_plasticity(substrate, saved_plast)
    _restore_dynamic_state(substrate, saved_acts, saved_p_acts)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=== Phase 5b: Bridge phenomenon at scale "
          "(M sequential patterns) ===\n")
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
        enable_feedback_p_to_n=True,
        starting_age=0.0,
        seed=42,
    )

    patterns = define_M_patterns(n_neurons, M, pattern_size, seed=0)
    print(f"M={M} disjoint patterns of size {pattern_size} each:")
    for i, p in enumerate(patterns):
        print(f"  pattern {i}: {p.tolist()}")
    print()

    history: list[dict[str, Any]] = []

    for stage in range(M):
        print(f"--- Training pattern {stage} (K={K_per_pattern} cycles) ---")
        train_one_pattern(substrate, patterns[stage], K=K_per_pattern)

        classification = classify_p_entities(substrate.p_entities, patterns)
        summary = summarize_classification(classification, M)

        # Individual recall for every pattern we've already trained.
        recalls = []
        for i in range(stage + 1):
            comp = test_individual_recall(substrate, patterns[i])
            recalls.append({"pattern": i, "completion": comp})

        stage_record = {
            "stage": stage,
            "trained_patterns_so_far": stage + 1,
            "P_counts": summary,
            "recalls": recalls,
            "system_age": substrate.system_age,
        }
        history.append(stage_record)

        # Print per-pattern P count + bridges + recall.
        p_counts_str = " ".join(
            f"P{i}={summary[f'P_pattern_{i}']}" for i in range(M)
        )
        print(f"  P counts: {p_counts_str}")
        bridges_str = ", ".join(
            f"({i},{j})={summary[f'P_bridge_{i}_{j}']}"
            for i in range(stage + 1) for j in range(i + 1, stage + 1)
        ) or "(none)"
        print(f"  Bridges: total={summary['P_bridges_total']}, "
              f"breakdown: {bridges_str}")
        recall_str = " ".join(
            f"P{r['pattern']}:{r['completion'] * 100:.0f}%" for r in recalls
        )
        print(f"  Recalls: {recall_str}")
        print()

    # ---- Combined-cue tests ----
    print("--- Combined-cue tests (n patterns simultaneously) ---")
    combined_results: list[dict[str, Any]] = []
    for n_combined in range(2, M + 1):
        activations = test_combined_recall(substrate, patterns, n_combined)
        act_str = " ".join(f"{a * 100:.0f}%" for a in activations)
        print(f"  n={n_combined}: per-pattern activation = {act_str}  "
              f"(mean={np.mean(activations) * 100:.0f}%, "
              f"min={min(activations) * 100:.0f}%)")
        combined_results.append({
            "n_combined": n_combined,
            "activations_per_pattern": activations,
            "mean_activation": float(np.mean(activations)),
            "min_activation": float(min(activations)),
        })
    print()

    # ---- Verdict ----
    final = history[-1]
    summary = final["P_counts"]
    pattern_p_counts = [summary[f"P_pattern_{i}"] for i in range(M)]
    pattern_recalls = [r["completion"] for r in final["recalls"]]
    bridges_total = summary["P_bridges_total"]
    n_possible_pairs = M * (M - 1) // 2
    avg_bridges_per_pair = bridges_total / max(n_possible_pairs, 1)

    all_patterns_have_p = all(c >= 2 for c in pattern_p_counts)
    all_patterns_recall = all(r >= 0.4 for r in pattern_recalls)
    bridges_modest = avg_bridges_per_pair < 10.0

    print("=== Verdict (P5 at scale) ===")
    print(f"Pattern P counts: {pattern_p_counts}")
    print(f"Pattern recalls:  "
          f"{[f'{r * 100:.0f}%' for r in pattern_recalls]}")
    print(f"Bridges: {bridges_total} total, avg {avg_bridges_per_pair:.1f}/pair "
          f"({n_possible_pairs} possible pairs)")
    print(f"  all_patterns_have_p (≥2 each): {all_patterns_have_p}")
    print(f"  all_patterns_recall (≥40 %):  {all_patterns_recall}")
    print(f"  bridges_modest (<10/pair):    {bridges_modest}")
    print()

    if all_patterns_have_p and all_patterns_recall and bridges_modest:
        verdict = "PASS"
        reason = (
            "2-level architecture scales gracefully to M=5. Every pattern "
            "retains P entities, every pattern recalls, bridges stay modest. "
            "No S-level needed at this scale."
        )
    elif not all_patterns_have_p:
        verdict = "FAIL_capacity"
        lost = [i for i, c in enumerate(pattern_p_counts) if c < 2]
        reason = (
            f"Pattern(s) {lost} lost (almost) all their P entities — "
            f"catastrophic forgetting under sequential training. "
            f"Motivates Phase 6: S-level consolidation to protect "
            f"established patterns from later training's decay."
        )
    elif not bridges_modest:
        verdict = "FAIL_bridges"
        reason = (
            f"Bridge population dominated (avg {avg_bridges_per_pair:.1f} per "
            f"pair, {bridges_total} total bridges vs "
            f"{sum(pattern_p_counts)} within-pattern P). "
            f"Spontaneous-attractor co-firing during idle phases is "
            f"manufacturing spurious cross-pattern associations faster "
            f"than the substrate can manage. Motivates Phase 6: S-level "
            f"grouping to abstract patterns above the bridge layer."
        )
    elif not all_patterns_recall:
        verdict = "WEAK_recall"
        weak = [i for i, r in enumerate(pattern_recalls) if r < 0.4]
        reason = (
            f"P entities preserved across patterns but recall < 40 % on "
            f"pattern(s) {weak}. Suggests interference at recall time — "
            f"may be resolvable with longer T_settle or stricter cue, but "
            f"hints at a real bandwidth limit on the 2-level architecture."
        )
    else:
        verdict = "INCONCLUSIVE"
        reason = "Mixed signals — need more replications or different M."

    print(f"{verdict}: {reason}")

    results_dir = _REPO_ROOT / "results" / "substrate" / "phase_5b"
    results_dir.mkdir(parents=True, exist_ok=True)

    with (results_dir / "phase_5b_results.json").open("w") as f:
        json.dump(
            {
                "config": {
                    "n_neurons": n_neurons,
                    "M": M,
                    "pattern_size": pattern_size,
                    "K_per_pattern": K_per_pattern,
                    "T_present": 15,
                    "T_rest": 60,
                    "T_settle": 30,
                    "seed": 42,
                },
                "patterns": [p.tolist() for p in patterns],
                "history": history,
                "combined_tests": combined_results,
                "final_summary": {
                    "pattern_p_counts": pattern_p_counts,
                    "pattern_recalls": pattern_recalls,
                    "bridges_total": bridges_total,
                    "avg_bridges_per_pair": avg_bridges_per_pair,
                    "all_patterns_have_p": all_patterns_have_p,
                    "all_patterns_recall": all_patterns_recall,
                    "bridges_modest": bridges_modest,
                },
                "verdict": verdict,
                "reason": reason,
            },
            f,
            indent=2,
        )

    # ---- Plots ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    stages = list(range(M))
    pattern_colors = plt.cm.tab10.colors[:M]

    # Top-left: P per pattern over stages.
    for i in range(M):
        counts = [s["P_counts"][f"P_pattern_{i}"] for s in history]
        axes[0, 0].plot(stages, counts, "o-",
                        color=pattern_colors[i], label=f"pattern {i}")
    axes[0, 0].set(
        xlabel="stage (number of patterns trained so far)",
        ylabel="# P entities for that pattern",
        title="Per-pattern P count through sequential training",
    )
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    # Top-right: total bridge count.
    bridge_totals = [s["P_counts"]["P_bridges_total"] for s in history]
    axes[0, 1].plot(stages, bridge_totals, "o-", color="tab:red", label="bridges")
    pattern_p_totals = [
        sum(s["P_counts"][f"P_pattern_{i}"] for i in range(M)) for s in history
    ]
    axes[0, 1].plot(stages, pattern_p_totals, "o-", color="tab:blue",
                    label="within-pattern P")
    axes[0, 1].set(
        xlabel="stage", ylabel="count",
        title="Bridge vs within-pattern P populations",
    )
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    # Bottom-left: per-pattern recall through stages.
    for i in range(M):
        recalls_i = []
        for s in history:
            r = next(
                (rr["completion"] for rr in s["recalls"] if rr["pattern"] == i),
                None,
            )
            recalls_i.append(r)
        axes[1, 0].plot(stages, recalls_i, "o-",
                        color=pattern_colors[i], label=f"pattern {i}")
    axes[1, 0].axhline(0.4, color="gray", linestyle=":", alpha=0.5)
    axes[1, 0].set(
        xlabel="stage", ylabel="recall completion fraction",
        title="Per-pattern recall through sequential training",
        ylim=(-0.05, 1.05),
    )
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    # Bottom-right: combined-cue mean + min per concurrency.
    n_list = [r["n_combined"] for r in combined_results]
    means = [r["mean_activation"] for r in combined_results]
    mins = [r["min_activation"] for r in combined_results]
    axes[1, 1].plot(n_list, means, "o-", color="tab:blue", label="mean activation")
    axes[1, 1].plot(n_list, mins, "s-", color="tab:red", label="min activation")
    axes[1, 1].set(
        xlabel="patterns presented simultaneously",
        ylabel="P-entity activation fraction",
        title="Combined-cue test by concurrency",
        ylim=(-0.05, 1.05),
    )
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    plt.suptitle(f"Phase 5b — M={M} sequential disjoint patterns: {verdict}",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "phase_5b_results.png", dpi=120)
    plt.close(fig)
    print(f"\nResults → {results_dir}")

    if verdict == "PASS":
        return 0
    if verdict in ("WEAK_recall", "INCONCLUSIVE"):
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
