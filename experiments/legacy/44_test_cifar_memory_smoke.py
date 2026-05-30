"""Experiment 44 — Phase 5.6.2 smoke test on real CIFAR-100.

Builds randomly-initialised CIFARHippocampus + CIFARNeocortex,
streams 5 tasks × 100 samples each from the SplitCIFAR100CI
benchmark into a cap-200 CIFARMultiLevelMemory, and verifies:

- Reservoir cap is respected at the end.
- The 5 tasks are reasonably represented (not dominated by one).
- ``sample_batch(32)`` returns the expected dict shape.

This is the end-to-end "the pieces talk to each other on real
data" check before adding consolidation + interleaved replay
in Phase 5.6.3.

Run from the repo root::

    python experiments/44_test_cifar_memory_smoke.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from continual_synapse.architectures import (  # noqa: E402
    CIFARHippocampus, CIFARNeocortex,
)
from continual_synapse.benchmarks import (  # noqa: E402
    SplitCIFAR100ClassIncremental,
)
from continual_synapse.memory import CIFARMultiLevelMemory  # noqa: E402


def _ok(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def main() -> int:
    print("=== Phase 5.6.2 smoke check ===\n")
    torch.manual_seed(0)

    t0 = time.time()
    bench = SplitCIFAR100ClassIncremental.from_huggingface(num_tasks=10)
    print(f"Loaded benchmark in {time.time() - t0:.1f}s.\n")

    hipp = CIFARHippocampus(num_classes=100)
    neo  = CIFARNeocortex(num_classes=100)
    mem = CIFARMultiLevelMemory(max_total=200, num_classes=100, rng_seed=0)
    print(
        f"Memory cap: {mem.max_total}\n"
        f"Models: hipp={sum(p.numel() for p in hipp.parameters()):,} params,  "
        f"neo={sum(p.numel() for p in neo.parameters()):,} params\n"
    )

    # Stream 5 tasks × 100 samples each.
    n_tasks = 5
    samples_per_task = 100
    classes_seen: list[int] = []
    for task_id in range(n_tasks):
        for c in bench.task_classes(task_id):
            classes_seen.append(int(c))
        ds = bench.task_train_dataset(task_id, augment=True)
        # Take the first `samples_per_task` items in order — using
        # a DataLoader would shuffle which is fine but slower.
        xs = torch.stack([ds[i][0] for i in range(samples_per_task)])
        ys = torch.tensor(
            [ds[i][1] for i in range(samples_per_task)],
            dtype=torch.long,
        )
        n_added = mem.record_batch(
            xs, ys, hipp, neo,
            classes_seen_so_far=list(classes_seen),
        )
        print(
            f"  task {task_id} (classes "
            f"{bench.task_classes(task_id)[0]}-"
            f"{bench.task_classes(task_id)[-1]}): "
            f"fed {samples_per_task} → added/replaced {n_added}, "
            f"|mem|={len(mem)}, n_seen={mem.n_seen}"
        )

    # ---- cap check ----
    print()
    expected_total_seen = n_tasks * samples_per_task
    cap_ok = (len(mem) == mem.max_total) and (mem.n_seen == expected_total_seen)
    print(
        f"Total inputs seen: {mem.n_seen} ({n_tasks} tasks × "
        f"{samples_per_task})"
    )
    print(
        f"Final memory size: {len(mem)} "
        f"(cap {mem.max_total} respected: {_ok(cap_ok)})"
    )

    # ---- per-task class distribution ----
    print()
    print("Per-task class distribution in memory:")
    ranges = [
        (task_id * 10, (task_id + 1) * 10)
        for task_id in range(n_tasks)
    ]
    counts = mem.per_class_range_counts(ranges)
    # Expected per task = max_total * (samples_per_task / total_seen)
    # = 200 * (100 / 500) = 40.
    expected_per_task = mem.max_total * samples_per_task / expected_total_seen
    for task_id, c in enumerate(counts):
        print(
            f"  task {task_id} (classes "
            f"{task_id*10}-{task_id*10 + 9}):  {c:>3} entries  "
            f"(~{expected_per_task:.0f} expected)"
        )
    # Reservoir sampling: each task should land within a generous
    # band around the expected mean. With n=500 stream, cap=200,
    # 100 items per task: variance ≈ 100 * (200/500) * (300/500) ≈ 24,
    # std ≈ 4.9. ±3 std ≈ ±15.
    min_count = min(counts)
    max_count = max(counts)
    spread_ok = (max_count - min_count) <= 30
    sum_ok = sum(counts) == mem.max_total
    print(
        f"\n  Spread (max - min) = {max_count - min_count}  "
        f"(reservoir gives ~{expected_per_task:.0f} ± ~5 per task; "
        f"≤ 30 = healthy): {_ok(spread_ok)}"
    )
    print(
        f"  Sum across tasks = {sum(counts)} (must equal "
        f"buffer size {mem.max_total}): {_ok(sum_ok)}"
    )

    # ---- sample_batch shape check ----
    print()
    print("Sample batch of 32:")
    sample = mem.sample_batch(32, device="cpu")
    assert sample is not None
    checks = [
        ("inputs",        sample["inputs"].shape,        (32, 3, 32, 32)),
        ("hipp_low_gap",  sample["hipp_low_gap"].shape,  (32, 32)),
        ("hipp_mid_gap",  sample["hipp_mid_gap"].shape,  (32, 64)),
        ("hipp_high_gap", sample["hipp_high_gap"].shape, (32, 128)),
        ("neo_low_gap",   sample["neo_low_gap"].shape,   (32, 128)),
        ("neo_mid_gap",   sample["neo_mid_gap"].shape,   (32, 256)),
        ("neo_high_gap",  sample["neo_high_gap"].shape,  (32, 512)),
        ("soft_targets",  sample["soft_targets"].shape,  (32, 100)),
        ("labels",        sample["labels"].shape,        (32,)),
    ]
    all_pass = cap_ok and spread_ok and sum_ok
    for name, got, expected in checks:
        ok = got == expected
        all_pass = all_pass and ok
        got_str = str(tuple(got))
        print(
            f"  {name:<14} shape: {got_str:<18}"
            f"  (expected {expected}): {_ok(ok)}"
        )
    classes_seen_ok = (
        isinstance(sample["classes_seen"], list)
        and len(sample["classes_seen"]) == 32
    )
    all_pass = all_pass and classes_seen_ok
    print(
        f"  classes_seen (list, len 32):                   "
        f"{_ok(classes_seen_ok)}"
    )

    # ---- storage cost summary ----
    print()
    bytes_per_entry = (
        sample["inputs"][0].element_size() * sample["inputs"][0].nelement()
        + 4 * (32 + 64 + 128 + 128 + 256 + 512 + 100)
        + 8 + 80
    )
    total_kb = bytes_per_entry * len(mem) / 1024
    print(
        f"Storage cost: ~{bytes_per_entry:,} bytes / entry, "
        f"{total_kb:.1f} KB total for {len(mem)} entries."
    )

    print()
    if all_pass:
        print("Memory ready for consolidation phase (5.6.3).")
        return 0
    print("Smoke check failed — see PASS/FAIL rows above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
