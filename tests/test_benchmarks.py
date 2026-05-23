"""Tests for the Split-MNIST benchmark."""

from __future__ import annotations

import pytest
import torch

from continual_synapse.evaluation.benchmarks import SplitMNIST, _filter_and_remap


def _make_fake_mnist(
    n_train_per_class: int = 6, n_test_per_class: int = 2, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a synthetic MNIST-shaped dataset with 10 balanced classes."""
    g = torch.Generator().manual_seed(seed)

    def make(n_per_class: int) -> tuple[torch.Tensor, torch.Tensor]:
        images = []
        labels = []
        for digit in range(10):
            x = torch.randint(0, 256, (n_per_class, 28, 28), generator=g)
            images.append(x)
            labels.append(torch.full((n_per_class,), digit, dtype=torch.int64))
        return torch.cat(images), torch.cat(labels)

    train_x, train_y = make(n_train_per_class)
    test_x, test_y = make(n_test_per_class)
    return train_x, train_y, test_x, test_y


def test_split_mnist_produces_five_tasks() -> None:
    bench = SplitMNIST(*_make_fake_mnist())
    tasks = bench.tasks()

    assert len(tasks) == 5
    assert [t.classes for t in tasks] == [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    assert [t.name for t in tasks] == [
        "split_mnist_01",
        "split_mnist_23",
        "split_mnist_45",
        "split_mnist_67",
        "split_mnist_89",
    ]


def test_split_mnist_task_only_contains_its_digit_pair() -> None:
    n_train = 6
    n_test = 2
    bench = SplitMNIST(*_make_fake_mnist(n_train, n_test))
    for task in bench.tasks():
        # Two classes (binary), each balanced -> 2 * n_train_per_class samples.
        assert len(task.train) == 2 * n_train
        assert len(task.test) == 2 * n_test
        _, train_y = task.train.tensors
        _, test_y = task.test.tensors
        assert set(train_y.tolist()) <= {0, 1}
        assert set(test_y.tolist()) <= {0, 1}
        # Both labels must appear in each split (no class collapse).
        assert set(train_y.tolist()) == {0, 1}


def test_split_mnist_flattens_images_to_784() -> None:
    bench = SplitMNIST(*_make_fake_mnist())
    task = bench.tasks()[0]
    x, _ = task.train.tensors
    assert x.shape[1:] == (784,)
    assert x.dtype == torch.float32
    # Pixel values should have been normalised into [0, 1].
    assert float(x.max()) <= 1.0
    assert float(x.min()) >= 0.0


def test_input_shape_and_num_classes() -> None:
    bench = SplitMNIST(*_make_fake_mnist())
    assert bench.input_shape == (784,)
    assert bench.num_classes_per_task == 2


def test_split_mnist_rejects_mismatched_labels() -> None:
    g = torch.Generator().manual_seed(0)
    images = torch.randint(0, 256, (10, 28, 28), generator=g)
    labels = torch.zeros(9, dtype=torch.int64)
    with pytest.raises(ValueError, match="disagree on N"):
        SplitMNIST(images, labels, images, torch.zeros(10, dtype=torch.int64))


def test_split_mnist_rejects_2d_labels() -> None:
    g = torch.Generator().manual_seed(0)
    images = torch.randint(0, 256, (10, 28, 28), generator=g)
    labels = torch.zeros((10, 1), dtype=torch.int64)
    with pytest.raises(ValueError, match="1-D"):
        SplitMNIST(images, labels, images, labels)


def test_filter_and_remap_handles_normalised_input() -> None:
    """If the caller passes already-normalised floats, no rescaling."""
    images = torch.rand(20, 28, 28)
    labels = torch.tensor([0, 1] * 10, dtype=torch.int64)
    ds = _filter_and_remap(images, labels, (0, 1))
    x, _ = ds.tensors
    assert float(x.max()) <= 1.0
    # Did not divide by 255: max should remain near its original value.
    assert float(x.max()) > 0.1
