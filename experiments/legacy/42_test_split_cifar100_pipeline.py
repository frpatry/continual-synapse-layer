"""Experiment 42 — Phase 5.6.0: Split-CIFAR-100 CI pipeline verification.

No model, no training — just verify that
:class:`SplitCIFAR100ClassIncremental` loads correctly, per-task
class assignments are right, train/test partitioning is correct,
augmentation actually produces different draws, and the class-
incremental eval loader contains the right cumulative class set.

If every check PASSes, the pipeline is ready for Phase 5.6.1
(CNN architectures).

Run from the repo root::

    python experiments/42_test_split_cifar100_pipeline.py

Note: experiment number 42 (not 37 as the spec suggested) because
exp 37 is taken by ``37_split_mnist_ci_ewc.py``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from continual_synapse.benchmarks import (  # noqa: E402
    SplitCIFAR100ClassIncremental,
)


def _ok(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def main() -> int:
    print("=== Split-CIFAR-100 CI Pipeline Verification ===")
    print()

    t0 = time.time()
    bench = SplitCIFAR100ClassIncremental.from_huggingface(num_tasks=10)
    print(f"Loaded in {time.time() - t0:.1f}s.\n")

    all_pass = True

    # ----- 1) Per-task class assignments + sample shapes -----
    for task_id in range(bench.num_tasks):
        classes = bench.task_classes(task_id)
        ds = bench.task_train_dataset(task_id, augment=True)
        x, y = ds[0]
        sample_shape = tuple(x.shape)
        n_samples = len(ds)
        expected_classes = list(range(task_id * 10, (task_id + 1) * 10))
        classes_ok = list(classes) == expected_classes
        shape_ok = sample_shape == (3, 32, 32)
        all_pass = all_pass and classes_ok and shape_ok
        print(
            f"Task {task_id}: classes "
            f"[{classes[0]}..{classes[-1]}]  "
            f"train samples: {n_samples}  "
            f"sample shape: {sample_shape}  "
            f"{_ok(classes_ok and shape_ok)}"
        )

    # ----- 2) Augmentation actually changes the tensor -----
    print()
    print("Augmentation check:")
    aug_ds = bench.task_train_dataset(0, augment=True)
    no_aug_ds = bench.task_train_dataset(0, augment=False)
    torch.manual_seed(0)
    sample_a, _ = aug_ds[0]
    sample_b, _ = aug_ds[0]
    sample_c, _ = aug_ds[0]
    sample_clean, _ = no_aug_ds[0]
    # Three augmented draws of the same image should be different
    # (with very high probability — the only way they all coincide
    # is no crop offset AND no flip, three times in a row, which
    # is (1/9 * 1/2)^3 ≈ 1.7e-4).
    ab_diff = float((sample_a - sample_b).abs().sum().item())
    ac_diff = float((sample_a - sample_c).abs().sum().item())
    bc_diff = float((sample_b - sample_c).abs().sum().item())
    # The non-augmented version should be DIFFERENT from the
    # augmented ones (because of the random crop / flip) — we
    # just check the augmented draws aren't identical.
    different = (ab_diff > 1e-3) and (ac_diff > 1e-3) and (bc_diff > 1e-3)
    all_pass = all_pass and different
    print(
        f"  Task 0 train index 0 — 3 augmented draws differ by "
        f"L1 ≈ ({ab_diff:.1f}, {ac_diff:.1f}, {bc_diff:.1f}) on a "
        f"3072-element tensor. Augmentation actually applied: "
        f"{_ok(different)}"
    )

    # Sanity-check normalisation: post-transform means should be
    # near zero and stds near one (CIFAR-100 normalisation).
    batch = torch.stack([no_aug_ds[i][0] for i in range(64)])
    mean_per_ch = batch.mean(dim=(0, 2, 3)).tolist()
    std_per_ch  = batch.std (dim=(0, 2, 3)).tolist()
    norm_ok = all(-0.5 <= m <= 0.5 for m in mean_per_ch) and all(
        0.5 <= s <= 1.5 for s in std_per_ch
    )
    all_pass = all_pass and norm_ok
    print(
        f"  Normalisation: 64-sample channel means="
        f"[{mean_per_ch[0]:+.2f}, {mean_per_ch[1]:+.2f}, "
        f"{mean_per_ch[2]:+.2f}]  stds=[{std_per_ch[0]:.2f}, "
        f"{std_per_ch[1]:.2f}, {std_per_ch[2]:.2f}]  "
        f"{_ok(norm_ok)}"
    )

    # ----- 3) Eval loader contains the right cumulative class set -----
    print()
    print("Eval loader check:")
    for up_to in (0, 5, 9):
        loader = bench.get_eval_loader(up_to_task=up_to, batch_size=512)
        all_labels: list[torch.Tensor] = []
        for _x, y in loader:
            all_labels.append(y)
        labels = torch.cat(all_labels)
        present = sorted(int(c) for c in labels.unique().tolist())
        expected = list(range(0, 10 * (up_to + 1)))
        classes_ok = present == expected
        all_pass = all_pass and classes_ok
        # Each CIFAR-100 class has 100 test samples → 10 classes/task
        # × (up_to + 1) tasks × 100 samples each.
        expected_count = 10 * (up_to + 1) * 100
        count_ok = labels.shape[0] == expected_count
        all_pass = all_pass and count_ok
        print(
            f"  Eval after task {up_to}: contains classes "
            f"[{present[0]}..{present[-1]}]  "
            f"sample count: {labels.shape[0]} (expected "
            f"{expected_count})  {_ok(classes_ok and count_ok)}"
        )

    # ----- 4) num_classes_seen helper -----
    print()
    print("num_classes_seen helper:")
    for k in (0, 4, 9):
        got = bench.num_classes_seen(after_task=k)
        expected = 10 * (k + 1)
        check = got == expected
        all_pass = all_pass and check
        print(
            f"  after_task={k}: got {got}, expected {expected}  "
            f"{_ok(check)}"
        )

    # ----- verdict -----
    print()
    if all_pass:
        print("Pipeline ready for CNN architecture phase (5.6.1).")
        return 0
    print(
        "Pipeline has one or more failing checks — see PASS/FAIL "
        "rows above. Debug before moving forward."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
