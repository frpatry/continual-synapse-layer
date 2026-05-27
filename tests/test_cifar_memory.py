"""Unit tests for CIFARMultiLevelMemory (Phase 5.6.2)."""

from __future__ import annotations

import torch

from continual_synapse.architectures import CIFARHippocampus, CIFARNeocortex
from continual_synapse.memory import CIFARMultiLevelMemory


def _make_models() -> tuple[CIFARHippocampus, CIFARNeocortex]:
    torch.manual_seed(0)
    return CIFARHippocampus(num_classes=100), CIFARNeocortex(num_classes=100)


def _record_n(
    memory: CIFARMultiLevelMemory, n: int,
    hipp: CIFARHippocampus, neo: CIFARNeocortex,
    *, batch_size: int = 8, base_class: int = 0,
) -> None:
    """Record ``n`` items via ``batch_size``-sized chunks. Inputs
    are i.i.d. random tensors; labels cycle through ``range(100)``
    starting at ``base_class`` so the spread is even by default.
    """
    remaining = n
    counter = 0
    while remaining > 0:
        b = min(batch_size, remaining)
        x = torch.randn(b, 3, 32, 32)
        y = torch.tensor(
            [(base_class + counter + i) % 100 for i in range(b)],
            dtype=torch.long,
        )
        memory.record_batch(
            x, y, hipp, neo, classes_seen_so_far=list(range(100)),
        )
        counter += b
        remaining -= b


def test_memory_grows_to_cap_then_reservoir_samples():
    """Feed 100 items into a cap-50 memory; final size must be 50."""
    hipp, neo = _make_models()
    mem = CIFARMultiLevelMemory(max_total=50, rng_seed=0)
    _record_n(mem, 100, hipp, neo, batch_size=10)
    assert len(mem) == 50, f"expected len 50, got {len(mem)}"
    assert mem.n_seen == 100, f"expected n_seen 100, got {mem.n_seen}"


def test_record_batch_extracts_correct_shapes():
    """Single batch of 8 → every stored field has the right shape."""
    hipp, neo = _make_models()
    mem = CIFARMultiLevelMemory(max_total=100, rng_seed=0)
    x = torch.randn(8, 3, 32, 32)
    y = torch.arange(8, dtype=torch.long)
    n_added = mem.record_batch(
        x, y, hipp, neo, classes_seen_so_far=[0, 1, 2, 3],
    )
    assert n_added == 8
    assert len(mem) == 8

    # Per-entry shapes (channel counts come from the architecture spec).
    assert mem.inputs[0].shape        == (3, 32, 32)
    assert mem.hipp_low_gap[0].shape  == (32,)
    assert mem.hipp_mid_gap[0].shape  == (64,)
    assert mem.hipp_high_gap[0].shape == (128,)
    assert mem.neo_low_gap[0].shape   == (128,)
    assert mem.neo_mid_gap[0].shape   == (256,)
    assert mem.neo_high_gap[0].shape  == (512,)
    assert mem.soft_targets[0].shape  == (100,)
    # Soft target is a valid distribution.
    assert torch.isclose(
        mem.soft_targets[0].sum(), torch.tensor(1.0), atol=1e-5,
    )
    assert (mem.soft_targets[0] >= 0).all().item()
    # Label is int, classes_seen is list[int].
    assert isinstance(mem.labels[0], int)
    assert mem.classes_seen[0] == [0, 1, 2, 3]


def test_sample_batch_returns_correct_format():
    """sample_batch(32) returns a dict with the expected keys + shapes."""
    hipp, neo = _make_models()
    mem = CIFARMultiLevelMemory(max_total=100, rng_seed=0)
    _record_n(mem, 64, hipp, neo, batch_size=16)

    sample = mem.sample_batch(32, device="cpu")
    assert sample is not None

    assert sample["inputs"].shape        == (32, 3, 32, 32)
    assert sample["hipp_low_gap"].shape  == (32, 32)
    assert sample["hipp_mid_gap"].shape  == (32, 64)
    assert sample["hipp_high_gap"].shape == (32, 128)
    assert sample["neo_low_gap"].shape   == (32, 128)
    assert sample["neo_mid_gap"].shape   == (32, 256)
    assert sample["neo_high_gap"].shape  == (32, 512)
    assert sample["soft_targets"].shape  == (32, 100)
    assert sample["labels"].shape == (32,)
    assert sample["labels"].dtype == torch.long
    assert isinstance(sample["classes_seen"], list)
    assert len(sample["classes_seen"]) == 32
    assert all(isinstance(cs, list) for cs in sample["classes_seen"])


def test_reservoir_uniform_distribution():
    """Feed 10000 items in 10 contiguous 'tasks' of 1000 each, cap
    100. Every task should be reasonably represented in the final
    buffer (each task expects ~10 entries, std ≈ 3.15)."""
    hipp, neo = _make_models()
    mem = CIFARMultiLevelMemory(max_total=100, rng_seed=42)

    # Use label as the "task" ID so we can count per-task.
    for task_id in range(10):
        # Each "task" has class labels equal to its task_id (so
        # post-storage per_class_counts gives task counts directly).
        x = torch.randn(1000, 3, 32, 32)
        y = torch.full((1000,), task_id, dtype=torch.long)
        # Feed in chunks of 200 to avoid one giant batch.
        for start in range(0, 1000, 200):
            mem.record_batch(
                x[start : start + 200], y[start : start + 200],
                hipp, neo, classes_seen_so_far=list(range(task_id + 1)),
            )

    assert len(mem) == 100, f"buffer must be saturated; got {len(mem)}"
    assert mem.n_seen == 10000

    counts = mem.per_class_counts()
    # Every task must appear at least once.
    for task_id in range(10):
        assert task_id in counts, (
            f"task {task_id} missing entirely — reservoir is broken"
        )
    # And no task should dominate beyond ~3 std above expected mean
    # (expected = 10, std ≈ 3.15; 30 is a generous bound).
    for task_id, c in counts.items():
        assert 1 <= c <= 30, (
            f"task {task_id} has {c} entries — outside the plausible "
            f"reservoir range [1, 30] for n=10000 cap=100"
        )
    # Sanity: counts sum to buffer size.
    assert sum(counts.values()) == 100


def test_no_gradients_in_record():
    """record_batch must not leave gradients on any model param."""
    hipp, neo = _make_models()
    # Pre-zero just in case anything sticky.
    for m in (hipp, neo):
        for p in m.parameters():
            p.grad = None

    mem = CIFARMultiLevelMemory(max_total=10, rng_seed=0)
    x = torch.randn(4, 3, 32, 32)
    y = torch.arange(4, dtype=torch.long)
    mem.record_batch(x, y, hipp, neo, classes_seen_so_far=[0, 1, 2, 3])

    for name, p in list(hipp.named_parameters()) + list(neo.named_parameters()):
        assert p.grad is None, (
            f"{name} accumulated a gradient during record_batch — "
            f"the no_grad decorator is not in effect"
        )
