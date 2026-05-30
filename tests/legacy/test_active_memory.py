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
    n = mem.maybe_allocate(features=x0, raw_inputs=x0, labels=y0)
    assert n == 1 and len(mem) == 1

    # Same direction, tiny perturbation: novelty ≈ 0, no allocate.
    x1 = torch.tensor([[1.0, 0.01, 0.0, 0.0]])
    y1 = torch.tensor([0], dtype=torch.long)
    n = mem.maybe_allocate(features=x1, raw_inputs=x1, labels=y1)
    assert n == 0 and len(mem) == 1

    # Orthogonal direction: novelty = 1, must allocate.
    x2 = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    y2 = torch.tensor([2], dtype=torch.long)
    n = mem.maybe_allocate(features=x2, raw_inputs=x2, labels=y2)
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
    assert mem.maybe_allocate(features=x, raw_inputs=x, labels=y) == 1
    assert mem.maybe_allocate(features=x, raw_inputs=x, labels=y) == 0
    assert len(mem) == 1


# ---- 4. retrieval returns correct label ----


def test_retrieval_returns_correct_label_for_seen_input() -> None:
    """Store one (embedding, label=3) entry. A query identical to
    that embedding must produce a distribution whose argmax is 3."""
    mem = ActiveEpisodicMemory(
        feature_dim=4, n_classes=5, novelty_threshold=0.5, retrieval_k=1,
    )
    target = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
    mem.maybe_allocate(
        features=target, raw_inputs=target,
        labels=torch.tensor([3], dtype=torch.long),
    )

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
    mem.maybe_allocate(
        features=near, raw_inputs=near,
        labels=torch.tensor([1, 1, 1], dtype=torch.long),
    )
    # Two far-target embeddings labelled 4 (orthogonal direction).
    far = torch.tensor(
        [[0.0, 1.0], [0.05, 0.99]],
    )
    mem.maybe_allocate(
        features=far, raw_inputs=far,
        labels=torch.tensor([4, 4], dtype=torch.long),
    )
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
        n_total += mem.maybe_allocate(
            features=xs[i : i + 1],
            raw_inputs=xs[i : i + 1],
            labels=ys[i : i + 1],
        )
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
    e0 = torch.tensor([[1.0, 0.0]])
    mem.maybe_allocate(
        features=e0, raw_inputs=e0,
        labels=torch.tensor([0], dtype=torch.long),
    )
    probs_one, _ = mem.retrieve(torch.tensor([[1.0, 0.0]]))
    # All weight on label 0 — only one entry.
    assert probs_one.argmax(dim=-1).item() == 0
    assert mem._normalized_cache is not None
    cache_one = mem._normalized_cache

    # Second insert: cache must be invalidated.
    e1 = torch.tensor([[0.0, 1.0]])
    mem.maybe_allocate(
        features=e1, raw_inputs=e1,
        labels=torch.tensor([2], dtype=torch.long),
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


# ---- raw_inputs storage + re_encode_all ----


def test_raw_inputs_stored_alongside_embeddings() -> None:
    """Every successful allocation must append to ALL four lists
    (embeddings, raw_inputs, labels, task_ids) — they're parallel
    arrays that re_encode_all relies on."""
    mem = ActiveEpisodicMemory(
        feature_dim=4, n_classes=3, novelty_threshold=0.0,
    )
    # In a realistic call, features live in the model's feature
    # space and raw_inputs in the original input space. For this
    # test we use distinct tensors so we can verify each list got
    # the right one.
    features = torch.randn(3, 4)
    raw = torch.randn(3, 8)  # different input dim from feature dim
    labels = torch.tensor([0, 1, 2], dtype=torch.long)
    n = mem.maybe_allocate(
        features=features, raw_inputs=raw, labels=labels, task_id=7,
    )
    assert n == 3
    assert len(mem.embeddings) == 3
    assert len(mem.raw_inputs) == 3
    assert len(mem.labels) == 3
    assert len(mem.task_ids) == 3
    # Stored content matches what was passed (per-entry, on CPU).
    for i in range(3):
        torch.testing.assert_close(mem.embeddings[i].cpu(), features[i])
        torch.testing.assert_close(mem.raw_inputs[i].cpu(), raw[i])
        assert mem.labels[i] == int(labels[i].item())
        assert mem.task_ids[i] == 7


def test_re_encode_all_updates_embeddings() -> None:
    """Allocate under a fixed feature extractor (model_v1), swap the
    extractor (model_v2), call re_encode_all. The stored embeddings
    must now match model_v2(raw_inputs) — the whole point of the
    feature-drift fix."""

    # Two synthetic linear feature extractors. Distinct weights so
    # they map the same raw input to different feature vectors.
    torch.manual_seed(0)
    w1 = torch.randn(8, 4)
    w2 = torch.randn(8, 4) * 3.0 + 1.0  # very different

    def encode_v1(x):
        return x @ w1

    def encode_v2(x):
        return x @ w2

    mem = ActiveEpisodicMemory(
        feature_dim=4, n_classes=3, novelty_threshold=0.0,
    )
    raw = torch.randn(5, 8)
    feats_v1 = encode_v1(raw)
    mem.maybe_allocate(
        features=feats_v1, raw_inputs=raw,
        labels=torch.tensor([0, 1, 2, 0, 1], dtype=torch.long),
    )
    # Sanity: before re-encoding, embeddings are model_v1's output.
    for i in range(5):
        torch.testing.assert_close(
            mem.embeddings[i], feats_v1[i].cpu(),
            rtol=1e-5, atol=1e-6,
        )

    mem.re_encode_all(encode_v2, batch_size=2)
    feats_v2 = encode_v2(raw)
    # After re-encoding, embeddings come from model_v2.
    assert len(mem.embeddings) == 5
    for i in range(5):
        torch.testing.assert_close(
            mem.embeddings[i], feats_v2[i].to(torch.float32).cpu(),
            rtol=1e-4, atol=1e-5,
        )
        # And they DIFFER from the v1 outputs.
        assert not torch.allclose(
            mem.embeddings[i], feats_v1[i].cpu(), atol=1e-3,
        )


def test_re_encode_all_empty_memory_noop() -> None:
    """Re-encoding an empty store must be a no-op (not a crash) so
    the on_task_end callback can fire unconditionally without
    needing to guard on memory size."""
    mem = ActiveEpisodicMemory(feature_dim=4, n_classes=3)
    # Should not raise; should not invent embeddings out of thin air.
    mem.re_encode_all(lambda x: x[:, :4])
    assert len(mem) == 0
    assert mem.embeddings == []
    assert mem.raw_inputs == []


def test_re_encode_all_invalidates_cache() -> None:
    """After re-encoding, the next retrieval must rebuild the
    normalised-cache from the freshly-computed embeddings, not the
    stale ones — otherwise retrieval keeps voting based on the old
    feature space."""

    # Use simple linear extractors. v1 is identity-on-first-4-dims
    # so retrieval lines up cleanly; v2 swaps two coordinates so
    # retrieval should change post re-encoding.
    def encode_v1(x):
        return x[:, :4]

    def encode_v2(x):
        # Swap dimensions 0 and 1 → entries become a different
        # location in feature space.
        out = x[:, :4].clone()
        out[:, [0, 1]] = out[:, [1, 0]]
        return out

    mem = ActiveEpisodicMemory(
        feature_dim=4, n_classes=3, novelty_threshold=0.0, retrieval_k=1,
    )
    raw = torch.eye(4, 8)  # 4 axis-aligned raw inputs
    feats_v1 = encode_v1(raw)
    mem.maybe_allocate(
        features=feats_v1, raw_inputs=raw,
        labels=torch.tensor([0, 1, 2, 0], dtype=torch.long),
    )

    # Prime the cache via a retrieve call.
    mem.retrieve(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert mem._normalized_cache is not None

    mem.re_encode_all(encode_v2)
    assert mem._normalized_cache is None, (
        "re_encode_all must invalidate the normalised cache so the "
        "next retrieval re-reads the refreshed embeddings."
    )
    # And the cache is rebuilt on the next retrieve, with the new
    # embeddings (post-swap), not the originals.
    mem.retrieve(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert mem._normalized_cache is not None
    assert mem._normalized_cache.shape == (4, 4)


def test_re_encode_all_handles_large_memory() -> None:
    """Re-encoding 1000 entries with a batch_size of 128 must
    process every entry (no off-by-one truncation at the last
    chunk) and leave the lists internally consistent."""

    def encode(x):
        # Trivial extractor: take first 4 dims.
        return x[:, :4]

    mem = ActiveEpisodicMemory(
        feature_dim=4, n_classes=3, novelty_threshold=0.0,
    )
    torch.manual_seed(0)
    raw = torch.randn(1000, 8)
    feats = encode(raw)
    # Allocate in chunks of 100 to keep maybe_allocate's novelty
    # mask simple (a single 1000-row call hits the same novelty
    # path — both work; chunked here for readability).
    for i in range(0, 1000, 100):
        mem.maybe_allocate(
            features=feats[i : i + 100],
            raw_inputs=raw[i : i + 100],
            labels=torch.randint(0, 3, (100,), dtype=torch.long),
        )
    assert len(mem) == 1000

    mem.re_encode_all(encode, batch_size=128)
    assert len(mem.embeddings) == 1000
    assert len(mem.raw_inputs) == 1000
    # Spot-check: every embedding still matches encode(raw_input)
    # (the extractor is deterministic, so re-encoding is idempotent).
    for i in (0, 1, 127, 128, 999):
        torch.testing.assert_close(
            mem.embeddings[i], feats[i].to(torch.float32).cpu(),
            rtol=1e-5, atol=1e-6,
        )
