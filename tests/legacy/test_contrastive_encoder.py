"""Tests for the contrastive encoder + InfoNCE pretraining utilities."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from continual_synapse.episodic.contrastive_encoder import (
    ContrastiveEncoder,
    apply_permutation,
    info_nce_loss,
    random_permutation,
)


# ---- 1. encoder output shape ----


def test_encoder_output_shape() -> None:
    """encode(x) returns features at the configured feature_dim,
    regardless of the projection-head width."""
    enc = ContrastiveEncoder(
        input_dim=64, hidden_dim=32, feature_dim=16, projection_dim=8,
    )
    x = torch.randn(5, 64)
    features = enc.encode(x)
    assert features.shape == (5, 16)
    assert enc.feature_dim == 16


# ---- 2. forward returns both views ----


def test_forward_returns_features_and_projected() -> None:
    """forward(x) returns (features, projected) — features at
    feature_dim, projected at projection_dim. Both come from the
    same forward pass (cheaper than two separate calls)."""
    enc = ContrastiveEncoder(
        input_dim=20, hidden_dim=16, feature_dim=12, projection_dim=4,
    )
    x = torch.randn(3, 20)
    features, projected = enc(x)
    assert features.shape == (3, 12)
    assert projected.shape == (3, 4)


# ---- 3. InfoNCE loss responds to alignment ----


def test_info_nce_loss_decreases_with_aligned_pairs() -> None:
    """Identical positive pairs should give a loss close to the
    log(2B−1) lower bound; orthogonal pairs should give a higher
    loss. Verifies the contrastive objective actually responds to
    the alignment quality of the input pairs."""
    torch.manual_seed(0)
    B, D = 8, 16
    # Aligned: z2 = z1 ⇒ positives are maximally aligned, large
    # gap to any random negative.
    z1 = torch.randn(B, D)
    aligned_loss = float(info_nce_loss(z1, z1.clone(), temperature=0.1))

    # Misaligned: z2 random ⇒ positives are no closer than random
    # negatives, so the cross-entropy can't place mass on them.
    z2_random = torch.randn(B, D)
    misaligned_loss = float(info_nce_loss(z1, z2_random, temperature=0.1))

    assert aligned_loss < misaligned_loss, (
        f"aligned ({aligned_loss:.3f}) should be < misaligned "
        f"({misaligned_loss:.3f})"
    )
    # The aligned case should be near the theoretical floor for
    # 2B−1 distractors (≈ 2.7 for B=8). Loose upper bound to keep
    # the test from flaking on small numeric drift.
    assert aligned_loss < 1.0


# ---- 4. symmetry of positives ----


def test_info_nce_loss_is_symmetric() -> None:
    """Swapping (z1, z2) ↔ (z2, z1) must give the same loss — the
    positive-pair relationship is symmetric and so is the stacked
    similarity matrix."""
    torch.manual_seed(1)
    z1 = torch.randn(5, 8)
    z2 = torch.randn(5, 8)
    loss_a = float(info_nce_loss(z1, z2, temperature=0.2))
    loss_b = float(info_nce_loss(z2, z1, temperature=0.2))
    assert math.isclose(loss_a, loss_b, rel_tol=1e-5, abs_tol=1e-6), (
        f"info_nce_loss must be symmetric: {loss_a} vs {loss_b}"
    )


# ---- 5. permutation invertibility ----


def test_permutation_is_invertible() -> None:
    """Applying a permutation and its inverse returns the original
    tensor. Guards apply_permutation against an off-by-one or wrong-
    axis bug."""
    torch.manual_seed(2)
    x = torch.randn(4, 10)
    perm = torch.randperm(10)
    permuted = apply_permutation(x, perm)
    # Inverse permutation: arg-sort of the original permutation.
    inv = torch.argsort(perm)
    restored = apply_permutation(permuted, inv)
    torch.testing.assert_close(restored, x)


# ---- 6. random_permutation distribution ----


def test_random_permutation_is_uniform() -> None:
    """random_permutation should produce permutations whose value
    distribution at each position is approximately uniform over
    indices — a loose check that we're not silently producing
    biased permutations (e.g. all-identity or all-reversed)."""
    torch.manual_seed(3)
    dim = 10
    n = 2000
    perms = random_permutation(dim, n)
    assert perms.shape == (n, dim)
    # Every row must be a valid permutation: distinct values [0, dim).
    for row in perms[:20]:  # spot-check 20 rows
        assert sorted(row.tolist()) == list(range(dim))
    # Per-position distribution: the value at position 0 should hit
    # each index ~n/dim times. Allow generous tolerance (±25%).
    counts = torch.bincount(perms[:, 0], minlength=dim)
    expected = n / dim
    tol = 0.25 * expected
    for c in counts.tolist():
        assert abs(c - expected) < tol, (
            f"position-0 value counts uneven: got {counts.tolist()}, "
            f"expected ~{expected:.0f} per index"
        )
