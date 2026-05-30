"""Tests for the PermutedMNIST benchmark."""

from __future__ import annotations

import pytest
import torch

from continual_synapse.evaluation.benchmarks import (
    PermutedMNIST,
    _make_permutations,
)


def _make_fake_mnist(
    n_train: int = 20, n_test: int = 8, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    train_x = torch.randint(0, 256, (n_train, 28, 28), generator=g)
    train_y = torch.randint(0, 10, (n_train,), generator=g)
    test_x = torch.randint(0, 256, (n_test, 28, 28), generator=g)
    test_y = torch.randint(0, 10, (n_test,), generator=g)
    return train_x, train_y, test_x, test_y


def test_permutation_zero_is_identity() -> None:
    perms = _make_permutations(num_tasks=3, n_pixels=10, seed=0)
    assert torch.equal(perms[0], torch.arange(10))


def test_permutations_are_valid_and_distinct() -> None:
    perms = _make_permutations(num_tasks=4, n_pixels=20, seed=7)
    for p in perms:
        # Each is a valid permutation: same set, possibly reordered.
        assert torch.equal(p.sort().values, torch.arange(20))
    # Non-identity perms differ from identity and from each other.
    assert not torch.equal(perms[1], perms[0])
    assert not torch.equal(perms[2], perms[1])


def test_permutations_are_deterministic_for_fixed_seed() -> None:
    a = _make_permutations(num_tasks=3, n_pixels=8, seed=42)
    b = _make_permutations(num_tasks=3, n_pixels=8, seed=42)
    for pa, pb in zip(a, b):
        assert torch.equal(pa, pb)


def test_benchmark_produces_requested_number_of_tasks() -> None:
    perms = _make_permutations(num_tasks=5, n_pixels=28 * 28, seed=0)
    bench = PermutedMNIST(*_make_fake_mnist(), permutations=perms)
    tasks = bench.tasks()
    assert len(tasks) == 5
    assert [t.name for t in tasks] == [
        "permuted_mnist_perm0",
        "permuted_mnist_perm1",
        "permuted_mnist_perm2",
        "permuted_mnist_perm3",
        "permuted_mnist_perm4",
    ]


def test_each_task_uses_correct_permutation() -> None:
    perms = _make_permutations(num_tasks=3, n_pixels=28 * 28, seed=0)
    bench = PermutedMNIST(*_make_fake_mnist(n_train=4, n_test=2), permutations=perms)
    tasks = bench.tasks()
    # First task uses identity: x.flatten() unchanged.
    raw_train = bench._train_images.to(torch.float32).flatten(1) / 255.0
    task0_x, _ = tasks[0].train.tensors
    torch.testing.assert_close(task0_x, raw_train)

    # Second task permutes columns by perms[1].
    expected_perm1 = raw_train[:, perms[1]]
    task1_x, _ = tasks[1].train.tensors
    torch.testing.assert_close(task1_x, expected_perm1)


def test_label_space_is_shared_across_tasks() -> None:
    perms = _make_permutations(num_tasks=3, n_pixels=28 * 28, seed=0)
    bench = PermutedMNIST(*_make_fake_mnist(), permutations=perms)
    assert bench.num_classes_per_task == 10
    # All tasks use the same label space (0..9) and the same labels per
    # sample (only inputs differ).
    tasks = bench.tasks()
    _, y0 = tasks[0].train.tensors
    _, y1 = tasks[1].train.tensors
    assert torch.equal(y0, y1)
    assert int(y0.max()) <= 9


def test_input_shape_and_classes_tuple() -> None:
    perms = _make_permutations(num_tasks=2, n_pixels=28 * 28, seed=0)
    bench = PermutedMNIST(*_make_fake_mnist(), permutations=perms)
    assert bench.input_shape == (784,)
    tasks = bench.tasks()
    assert tasks[0].classes == tuple(range(10))


def test_train_and_test_are_permuted_with_the_same_permutation() -> None:
    perms = _make_permutations(num_tasks=2, n_pixels=28 * 28, seed=0)
    bench = PermutedMNIST(*_make_fake_mnist(), permutations=perms)
    tasks = bench.tasks()
    train_x, _ = tasks[1].train.tensors
    test_x, _ = tasks[1].test.tensors
    raw_train = bench._train_images.to(torch.float32).flatten(1) / 255.0
    raw_test = bench._test_images.to(torch.float32).flatten(1) / 255.0
    torch.testing.assert_close(train_x, raw_train[:, perms[1]])
    torch.testing.assert_close(test_x, raw_test[:, perms[1]])


def test_pixel_values_normalised_to_unit_range() -> None:
    perms = _make_permutations(num_tasks=1, n_pixels=28 * 28, seed=0)
    bench = PermutedMNIST(*_make_fake_mnist(), permutations=perms)
    x, _ = bench.tasks()[0].train.tensors
    assert float(x.max()) <= 1.0
    assert float(x.min()) >= 0.0


def test_num_tasks_property_matches_permutation_count() -> None:
    perms = _make_permutations(num_tasks=7, n_pixels=10, seed=0)
    g = torch.Generator().manual_seed(0)
    train_x = torch.randint(0, 256, (8, 10), generator=g)
    train_y = torch.randint(0, 10, (8,), generator=g)
    bench = PermutedMNIST(train_x, train_y, train_x, train_y, perms)
    assert bench.num_tasks == 7


def test_constructor_rejects_bad_permutation_shape() -> None:
    g = torch.Generator().manual_seed(0)
    train_x = torch.randint(0, 256, (4, 10), generator=g)
    train_y = torch.randint(0, 10, (4,), generator=g)
    bad_perm = torch.arange(10).reshape(2, 5)
    with pytest.raises(ValueError, match="1-D"):
        PermutedMNIST(train_x, train_y, train_x, train_y, [bad_perm])


def test_from_huggingface_rejects_non_positive_num_tasks() -> None:
    with pytest.raises(ValueError, match="num_tasks"):
        PermutedMNIST.from_huggingface(num_tasks=0)
