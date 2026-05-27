"""Experiment 43 — Phase 5.6.1: CIFAR CNN architecture verification.

No training. Just instantiate :class:`CIFARHippocampus` and
:class:`CIFARNeocortex`, run a dummy forward + backward pass,
print parameter counts and multi-level feature shapes, and
sanity-check the per-entry memory storage cost under two
options:

- Option A (full feature maps): the spatial activations are
  stored verbatim. Most informative for the consolidation
  consistency loss but expensive per entry.
- Option B (GAP feature maps): every level is globally
  average-pooled to a 1-D vector before storage. ~64× cheaper
  per entry; loses the spatial structure.

The storage-cost table prints both so Phase 5.6.2 (memory
adapter) can pick the right tradeoff. Default expectation is
Option B for the first pass.

Run from the repo root::

    python experiments/43_test_cifar_cnn.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from continual_synapse.architectures import (  # noqa: E402
    CIFARHippocampus, CIFARNeocortex,
)


def _ok(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _describe_feats(
    feats: dict[str, torch.Tensor], indent: str = "    ",
) -> None:
    """Print per-level shape + storage cost (GAP vs full map) per
    sample, in floats."""
    for name, t in feats.items():
        c, h, w = t.shape[1], t.shape[2], t.shape[3]
        gap_floats = c
        full_floats = c * h * w
        shape_str = str(tuple(t.shape))
        print(
            f"{indent}{name:<5} shape {shape_str:<22}  "
            f"GAP={gap_floats:>5d} floats  "
            f"full={full_floats:>6d} floats"
        )


def _per_entry_storage(
    hipp_feats: dict[str, torch.Tensor],
    neo_feats:  dict[str, torch.Tensor],
    num_classes: int = 100,
    n_classes_mask_int: int = 10,
) -> tuple[int, int]:
    """Return (option_A_bytes, option_B_bytes) per stored entry.

    Components:
        input:        3*32*32 uint8 (1 byte each)
        hipp feats:   sum across levels (4 bytes each, float32)
        neo  feats:   sum across levels (4 bytes each, float32)
        soft target:  num_classes float32
        label:        1 int64 (8 bytes)
        classes_seen: 10 int64 (80 bytes; we report as ints)
    """
    in_bytes = 3 * 32 * 32  # uint8
    hipp_gap = sum(t.shape[1] for t in hipp_feats.values())
    hipp_full = sum(
        t.shape[1] * t.shape[2] * t.shape[3] for t in hipp_feats.values()
    )
    neo_gap = sum(t.shape[1] for t in neo_feats.values())
    neo_full = sum(
        t.shape[1] * t.shape[2] * t.shape[3] for t in neo_feats.values()
    )
    soft_bytes = 4 * num_classes
    label_bytes = 8
    mask_bytes = 8 * n_classes_mask_int

    opt_a = (
        in_bytes
        + 4 * (hipp_full + neo_full)
        + soft_bytes + label_bytes + mask_bytes
    )
    opt_b = (
        in_bytes
        + 4 * (hipp_gap + neo_gap)
        + soft_bytes + label_bytes + mask_bytes
    )
    return opt_a, opt_b


def main() -> int:
    print("=== Phase 5.6.1: CIFAR CNN Architecture Verification ===\n")
    torch.manual_seed(0)
    B = 8
    x = torch.randn(B, 3, 32, 32)

    all_pass = True

    # ---------- Hippocampus ----------
    hipp = CIFARHippocampus(num_classes=100)
    hipp_params = _count_params(hipp)
    print("Hippocampus:")
    print(
        f"  Total params: {hipp_params:,} "
        f"(expected ~100K-150K range)"
    )
    h_logits = hipp(x)
    print(f"  Sample input shape: {tuple(x.shape)}")
    print(f"  Forward output shape: {tuple(h_logits.shape)}")
    print()
    print("  Multi-level features:")
    hipp_feats = hipp.features(x)
    _describe_feats(hipp_feats)

    hipp_check_shape = h_logits.shape == (B, 100)
    hipp_check_feats_b = all(
        t.shape[0] == B for t in hipp_feats.values()
    )
    hipp_check_finite = bool(torch.isfinite(h_logits).all().item())
    all_pass &= hipp_check_shape and hipp_check_feats_b and hipp_check_finite

    # ---------- Neocortex ----------
    print()
    neo = CIFARNeocortex(num_classes=100)
    neo_params = _count_params(neo)
    print("Neocortex (Reduced ResNet-18):")
    print(
        f"  Total params: {neo_params:,} "
        f"(expected ~11M)"
    )
    n_logits = neo(x)
    print(f"  Forward output shape: {tuple(n_logits.shape)}")
    print()
    print("  Multi-level features:")
    neo_feats = neo.features(x)
    _describe_feats(neo_feats)

    neo_check_shape = n_logits.shape == (B, 100)
    neo_check_feats_b = all(
        t.shape[0] == B for t in neo_feats.values()
    )
    neo_check_finite = bool(torch.isfinite(n_logits).all().item())
    all_pass &= neo_check_shape and neo_check_feats_b and neo_check_finite

    # ---------- Storage cost summary ----------
    opt_a, opt_b = _per_entry_storage(hipp_feats, neo_feats)
    print()
    print("Per-entry storage cost (Option A = full feature maps, "
          "Option B = GAP):")
    print(
        f"  inputs (3*32*32 uint8 = 3,072 B) + hipp feats + neo "
        f"feats + soft_target (100 f32 = 400 B) + label (8 B) + "
        f"classes_seen (10 int64 = 80 B)"
    )
    print(
        f"  Option A (full): {opt_a:,} bytes / entry "
        f"≈ {opt_a / 1024:.1f} KB"
    )
    print(
        f"  Option B (GAP):  {opt_b:,} bytes / entry "
        f"≈ {opt_b / 1024:.1f} KB"
    )
    print(
        f"  Ratio A/B: {opt_a / opt_b:.1f}×"
    )
    for n_entries in (500, 5000):
        print(
            f"  For {n_entries:,} entries: "
            f"Option A ≈ {opt_a * n_entries / (1024 ** 2):.1f} MB, "
            f"Option B ≈ {opt_b * n_entries / (1024 ** 2):.1f} MB"
        )

    # ---------- Validation checks ----------
    print()
    print("Validation checks:")
    print(
        f"  - Hippocampus output has 100 logits per sample: "
        f"{_ok(hipp_check_shape)}"
    )
    print(
        f"  - Neocortex output has 100 logits per sample:   "
        f"{_ok(neo_check_shape)}"
    )
    feats_batch_ok = hipp_check_feats_b and neo_check_feats_b
    print(
        f"  - Multi-level features all have batch dim {B}:    "
        f"{_ok(feats_batch_ok)}"
    )
    finite_ok = hipp_check_finite and neo_check_finite
    print(
        f"  - No NaN/Inf in forward pass:                     "
        f"{_ok(finite_ok)}"
    )

    # Backward pass: dummy CE loss, confirm gradients flow.
    target = torch.randint(0, 100, (B,))
    hipp_loss = torch.nn.functional.cross_entropy(h_logits, target)
    neo_loss  = torch.nn.functional.cross_entropy(n_logits, target)
    hipp_loss.backward()
    neo_loss.backward()
    hipp_grads_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in hipp.parameters() if p.requires_grad
    )
    neo_grads_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in neo.parameters() if p.requires_grad
    )
    grads_ok = hipp_grads_ok and neo_grads_ok
    all_pass &= grads_ok
    print(
        f"  - Backward pass works (gradient flow check):      "
        f"{_ok(grads_ok)}"
    )

    print()
    if all_pass:
        print("Pipeline ready for memory adapter phase (5.6.2).")
        return 0
    print(
        "One or more checks failed — see PASS/FAIL rows above."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
