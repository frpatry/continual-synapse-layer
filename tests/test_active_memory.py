"""Tests for ActiveEpisodicMemory — gradient-free episodic store."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from continual_synapse.episodic.active_memory import ActiveEpisodicMemory


# ---- 1. empty memory ----


def test_empty_memory_returns_max_novelty() -> None:
    """No stored entries → novelty = 1 for any input (the contract
    that lets a fresh memory accept everything on first encounter)."""
    mem = ActiveEpisodicMemory(feature_dim=4, n_classes=3)
    features = torch.randn(5, 4)
    novelty = mem.compute_novelty(features)
    assert novelty.shape == (5,)
    assert torch.all(novelty == 1.0)


# ---- 2. novelty-thresholded allocation ----


def test_allocate_when_novel() -> None:
    """A second sample very similar to the first should NOT trigger
    a second allocation; a far-away sample should."""
    mem = ActiveEpisodicMemory(
        feature_dim=4, n_classes=3, novelty_threshold=0.5,
    )
    # First insert always allocates (empty memory ⇒ novelty=1).
    x0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    y0 = torch.tensor([0], dtype=torch.long)
    n = mem.maybe_allocate(x0, y0)
    assert n == 1 and len(mem) == 1

    # Same direction, tiny perturbation: novelty ≈ 0, no allocate.
    x1 = torch.tensor([[1.0, 0.01, 0.0, 0.0]])
    y1 = torch.tensor([0], dtype=torch.long)
    n = mem.maybe_allocate(x1, y1)
    assert n == 0 and len(mem) == 1

    # Orthogonal direction: novelty = 1, must allocate.
    x2 = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    y2 = torch.tensor([2], dtype=torch.long)
    n = mem.maybe_allocate(x2, y2)
    assert n == 1 and len(mem) == 2


# ---- 3. no double allocation ----


def test_no_double_allocation() -> None:
    """Storing the same input twice should only allocate once — the
    second call sees the first one's entry and detects zero novelty."""
    mem = ActiveEpisodicMemory(
        feature_dim=4, n_classes=2, novelty_threshold=0.1,
    )
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    y = torch.tensor([0], dtype=torch.long)
    assert mem.maybe_allocate(x, y) == 1
    assert mem.maybe_allocate(x, y) == 0
    assert len(mem) == 1


# ---- 4. retrieval returns correct label ----


def test_retrieval_returns_correct_label_for_seen_input() -> None:
    """Store one (embedding, label=3) entry. A query identical to
    that embedding must produce a distribution whose argmax is 3."""
    mem = ActiveEpisodicMemory(
        feature_dim=4, n_classes=5, novelty_threshold=0.5, retrieval_k=1,
    )
    target = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
    mem.maybe_allocate(target, torch.tensor([3], dtype=torch.long))

    probs, conf = mem.retrieve(target)
    assert probs.shape == (1, 5)
    assert probs.argmax(dim=-1).item() == 3
    # Cosine sim of x with x is ≈ 1 ⇒ confidence near 1.
    assert float(conf) > 0.99


# ---- 5. empty memory retrieval ----


def test_retrieval_handles_empty_memory() -> None:
    """Empty memory must return a uniform distribution and zero
    confidence — callers shouldn't have to special-case it."""
    mem = ActiveEpisodicMemory(feature_dim=4, n_classes=4)
    probs, conf = mem.retrieve(torch.randn(3, 4))
    assert probs.shape == (3, 4)
    torch.testing.assert_close(
        probs, torch.full((3, 4), 0.25), rtol=0, atol=1e-8,
    )
    assert torch.all(conf == 0.0)


# ---- 6. top-k weighted vote ----


def test_retrieval_top_k_weighted_vote() -> None:
    """Store 5 entries, 3 of label 1 close to the query and 2 of
    label 4 farther. Label 1 must dominate the vote."""
    mem = ActiveEpisodicMemory(
        feature_dim=2, n_classes=5, novelty_threshold=0.0, retrieval_k=5,
    )
    # Three near-target embeddings labelled 1.
    near = torch.tensor(
        [[1.0, 0.0], [0.99, 0.05], [0.98, 0.10]],
    )
    mem.maybe_allocate(near, torch.tensor([1, 1, 1], dtype=torch.long))
    # Two far-target embeddings labelled 4 (orthogonal direction).
    far = torch.tensor(
        [[0.0, 1.0], [0.05, 0.99]],
    )
    mem.maybe_allocate(far, torch.tensor([4, 4], dtype=torch.long))
    assert len(mem) == 5

    query = torch.tensor([[1.0, 0.0]])
    probs, _ = mem.retrieve(query)
    # Label 1 weight should dominate label 4 weight.
    assert probs[0, 1] > probs[0, 4]
    assert probs.argmax(dim=-1).item() == 1


# ---- 7. max_entries cap ----


def test_max_entries_caps_growth() -> None:
    """With max_entries=5, inserting 20 distinct novel samples
    should leave the store at exactly 5 — and the extras are
    silently dropped, not raised on."""
    mem = ActiveEpisodicMemory(
        feature_dim=8, n_classes=3,
        novelty_threshold=0.0,  # accept everything that doesn't already match
        max_entries=5,
    )
    # 20 distinct random embeddings — orthogonal-enough on average
    # that they're all novel relative to each other.
    torch.manual_seed(0)
    xs = F.normalize(torch.randn(20, 8), dim=-1) * 10  # large norms
    ys = torch.arange(20, dtype=torch.long) % 3
    n_total = 0
    for i in range(20):
        n_total += mem.maybe_allocate(xs[i : i + 1], ys[i : i + 1])
    assert len(mem) == 5
    assert n_total == 5


# ---- 8. cache invalidation ----


def test_cache_invalidation_after_insert() -> None:
    """The internal normalised-embeddings cache must be rebuilt after
    every insert, or retrieval will silently ignore newly-allocated
    entries."""
    mem = ActiveEpisodicMemory(
        feature_dim=2, n_classes=3, novelty_threshold=0.5, retrieval_k=2,
    )
    # First insert + retrieve populates the cache.
    mem.maybe_allocate(
        torch.tensor([[1.0, 0.0]]), torch.tensor([0], dtype=torch.long),
    )
    probs_one, _ = mem.retrieve(torch.tensor([[1.0, 0.0]]))
    # All weight on label 0 — only one entry.
    assert probs_one.argmax(dim=-1).item() == 0
    assert mem._normalized_cache is not None
    cache_one = mem._normalized_cache

    # Second insert: cache must be invalidated.
    mem.maybe_allocate(
        torch.tensor([[0.0, 1.0]]), torch.tensor([2], dtype=torch.long),
    )
    assert mem._normalized_cache is None, (
        "cache must be cleared on insert so retrieve sees the new entry"
    )
    probs_two, _ = mem.retrieve(torch.tensor([[0.0, 1.0]]))
    # The new entry should now be reachable: query == new entry ⇒
    # argmax = its label (2).
    assert probs_two.argmax(dim=-1).item() == 2
    # And the cache is repopulated, with an extra row for the new entry.
    assert mem._normalized_cache is not None
    assert mem._normalized_cache.shape[0] == 2
    assert mem._normalized_cache.shape[0] != cache_one.shape[0]
