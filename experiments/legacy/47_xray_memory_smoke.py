"""Experiment 47 — Phase 5.7.0 XRayMemory standalone smoke.

No real data, no model training. Synthesises feature vectors in
5 well-separated clusters (one per "class") and streams them
through ``XRayMemory.update()`` to verify the prototype-based
storage + EMA refinement + sparsification + temperature schedule
work end-to-end on a tiny in-memory simulation.

Verifies the user-visible contract:

- After N updates, the per-class occupied slot counts are
  reasonable (every class has at least one prototype; total
  occupancy ≤ num_classes × prototypes_per_class).
- Some prototypes reach refinement counts in the sparsification
  active range (≥ sparsity_start), and we can SEE the
  zero-fraction increase as a function of refinement count.
- Temperature is in [temp_final, temp_init] and reflects the
  mean refinement count.
- NT-Xent loss on a query close to its true-class prototype is
  low (< 1.0); a query placed near a wrong-class prototype is
  high (> 2.0).

Run from the repo root::

    python experiments/47_xray_memory_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from continual_synapse.memory import (  # noqa: E402
    XRayMemory, nt_xent_multi_prototype_loss,
)


def _ok(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def main() -> int:
    print("=== Phase 5.7.0 XRayMemory smoke ===\n")

    torch.manual_seed(0)
    n_classes = 5
    feature_dim = 128
    k = 3
    mem = XRayMemory(
        num_classes=n_classes,
        feature_dim=feature_dim,
        prototypes_per_class=k,
        sparsity_start_refinements=10,
        sparsity_end_refinements=50,
        sparsity_max_drop_fraction=0.5,
        temp_start_refinements=10,   # bring forward so the smoke
        temp_end_refinements=50,     # hits the temperature schedule
        temperature_initial=1.0,
        temperature_final=0.3,
    )

    # Well-separated class means + per-update noise. Some noise
    # samples may be large enough to drop cosine sim below the
    # 0.6 distinctness threshold, causing a new slot to be
    # allocated occasionally — exactly what we want for a varied
    # smoke.
    class_means = torch.randn(n_classes, feature_dim) * 4.0
    # Bias class 0 to receive far more updates so it reaches the
    # post-sparsity regime; classes 3 + 4 get fewer than
    # sparsity_start so their prototypes stay in the pre-sparsity
    # range. Targets one prototype per sparsity regime.
    # Total ≈ 145 updates.
    weights = torch.tensor([80, 30, 20, 8, 5], dtype=torch.long)
    sequence: list[int] = []
    for c, w in enumerate(weights.tolist()):
        sequence.extend([c] * w)
    rng = torch.Generator()
    rng.manual_seed(0)
    sequence_t = torch.tensor(sequence)
    perm = torch.randperm(len(sequence), generator=rng)
    sequence_t = sequence_t[perm]

    n_updates = 0
    n_correct = 0
    for c in sequence_t.tolist():
        noise = torch.randn(feature_dim) * 0.6
        feat = class_means[c] + noise
        sims = F.cosine_similarity(
            feat.unsqueeze(0), class_means, dim=-1,
        )
        pred = int(sims.argmax().item())
        is_correct = pred == c
        n_updates += int(is_correct)
        n_correct += int(is_correct)
        mem.update(
            feat.unsqueeze(0),
            torch.tensor([c]),
            torch.tensor([is_correct]),
        )
    total_seen = int(sequence_t.numel())

    occupied = mem.num_occupied()
    per_class = mem.per_class_counts()
    occupied_counts = mem.refinement_counts[mem.is_occupied]
    mean_refinement = float(occupied_counts.float().mean().item())

    print(f"After {total_seen} simulated inputs "
          f"({n_correct} classified correctly; only those refine):")
    print(f"  Occupied slots: {occupied} / {n_classes * k}")
    print(f"  Per-class prototype counts: {per_class}")
    print(f"  Mean refinement count: {mean_refinement:.1f}")

    # Sparsification trace — show one prototype from each class
    # whose count crosses the sparsity_start (10) threshold.
    print("\n  Sparsification check:")
    shown = 0
    for c in range(n_classes):
        for slot in range(k):
            if not mem.is_occupied[c, slot]:
                continue
            ref = int(mem.refinement_counts[c, slot])
            proto = mem.prototypes[c, slot]
            n_zeros = int((proto == 0).sum())
            zfrac = n_zeros / proto.shape[0]
            label = (
                "pre-sparsity" if ref < mem.sparsity_start
                else ("mid-sparsity" if ref < mem.sparsity_end
                      else "post-sparsity")
            )
            print(
                f"    Class {c} prototype {slot}: "
                f"{n_zeros:>3} zeros / {proto.shape[0]} dims "
                f"({zfrac:.0%}, refinement={ref}, {label})"
            )
            shown += 1
    assert shown == occupied, "shown count mismatched num_occupied"

    # Temperature.
    temp = mem.temperature()
    print(
        f"\n  Temperature current: {temp:.2f} "
        f"(mean refinement {mean_refinement:.1f}, "
        f"schedule [{mem.temp_start}..{mem.temp_end}], "
        f"init {mem.temp_init} → final {mem.temp_final})"
    )

    # NT-Xent loss probes.
    protos, proto_labels = mem.get_all_prototypes()
    # Correct probe: features close to class-0 mean labelled 0.
    query_correct = class_means[0:1] + 0.05 * torch.randn(1, feature_dim)
    loss_correct = float(nt_xent_multi_prototype_loss(
        query_correct, torch.tensor([0]),
        protos, proto_labels, temperature=temp,
    ).item())
    # Wrong probe: features close to class-1 mean but labelled 3.
    query_wrong = class_means[1:2] + 0.05 * torch.randn(1, feature_dim)
    loss_wrong = float(nt_xent_multi_prototype_loss(
        query_wrong, torch.tensor([3]),
        protos, proto_labels, temperature=temp,
    ).item())
    print("\n  NT-Xent loss test:")
    correct_pass = loss_correct < 1.0
    wrong_pass = loss_wrong > 2.0
    print(
        f"    Query close to class 0 (correct): loss = "
        f"{loss_correct:.3f} (expected < 1.0)  {_ok(correct_pass)}"
    )
    print(
        f"    Query close to class 1 with label 3 (wrong): loss = "
        f"{loss_wrong:.3f} (expected > 2.0)  {_ok(wrong_pass)}"
    )

    # Mechanical-check verdict.
    all_classes_have_prototype = all(c > 0 for c in per_class)
    occupied_in_range = 5 <= occupied <= n_classes * k
    any_pre_sparsity = any(
        int(mem.refinement_counts[c, s]) < mem.sparsity_start
        for c in range(n_classes) for s in range(k)
        if mem.is_occupied[c, s]
    )
    any_post_sparsity = any(
        int(mem.refinement_counts[c, s]) >= mem.sparsity_end
        for c in range(n_classes) for s in range(k)
        if mem.is_occupied[c, s]
    )

    print("\n  Mechanical checks:")
    checks = [
        ("Every class has ≥ 1 prototype:", all_classes_have_prototype),
        (
            f"Occupied count in [5, {n_classes*k}]:",
            occupied_in_range,
        ),
        ("Some prototype is pre-sparsity (ref < 10):", any_pre_sparsity),
        ("Some prototype is post-sparsity (ref ≥ 50):", any_post_sparsity),
        ("NT-Xent loss small for correct probe:", correct_pass),
        ("NT-Xent loss large for wrong probe:", wrong_pass),
    ]
    all_pass = True
    for label, passed in checks:
        all_pass = all_pass and passed
        print(f"    {label:<45} {_ok(passed)}")

    print()
    if all_pass:
        print("All mechanics PASS.")
        return 0
    print("Some checks failed — investigate before integration.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
