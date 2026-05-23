"""Continual-learning benchmarks.

Phase 1 implements Split-MNIST: five sequential binary-classification
tasks over the digit pairs (0,1), (2,3), (4,5), (6,7), (8,9). Labels
are remapped to {0, 1} inside each task so the same two-class head
can be reused across tasks (the standard task-incremental setup).

The benchmark is split into a data-agnostic core that operates on
tensors and a loader that fetches MNIST from the HuggingFace
``datasets`` hub. This keeps unit tests fast and offline: tests
construct a ``SplitMNIST`` from synthetic tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch
from torch.utils.data import TensorDataset


SplitPair = tuple[int, int]


@dataclass(frozen=True)
class Task:
    """One step of a continual-learning sequence.

    Attributes:
        name: Human-readable identifier, used in logs and plots.
        train: Training set as a ``TensorDataset`` yielding ``(x, y)``.
        test: Held-out evaluation set with the same schema as ``train``.
        classes: Original (pre-remap) class labels for this task.
            For Split-MNIST these are the two digit values.
    """

    name: str
    train: TensorDataset
    test: TensorDataset
    classes: tuple[int, ...]


class ContinualBenchmark(Protocol):
    """Protocol every continual-learning benchmark must satisfy."""

    name: str

    def tasks(self) -> list[Task]:
        """Return the ordered sequence of tasks for this benchmark."""

    @property
    def num_classes_per_task(self) -> int:
        """Number of output classes the model must predict per task."""

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Shape of a single input sample, excluding the batch dim."""


def _build_task(
    name: str,
    classes: SplitPair,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
) -> Task:
    """Filter samples whose label is in ``classes`` and remap to {0, 1}.

    Inputs are expected as tensors of shape ``(N, ...)`` for images and
    ``(N,)`` for labels. The returned ``Task`` stores flattened images
    (``N, 784``) of dtype float32 in [0, 1] and int64 binary labels.
    """
    return Task(
        name=name,
        train=_filter_and_remap(train_images, train_labels, classes),
        test=_filter_and_remap(test_images, test_labels, classes),
        classes=classes,
    )


def _filter_and_remap(
    images: torch.Tensor,
    labels: torch.Tensor,
    classes: SplitPair,
) -> TensorDataset:
    cls_a, cls_b = classes
    mask = (labels == cls_a) | (labels == cls_b)
    x = images[mask].to(torch.float32)
    # Normalise to [0, 1] if the input looks like 8-bit pixel data.
    if x.numel() and x.max() > 1.5:
        x = x / 255.0
    x = x.flatten(start_dim=1)
    y = (labels[mask] == cls_b).to(torch.int64)
    return TensorDataset(x, y)


class SplitMNIST:
    """Split-MNIST benchmark over five digit-pair binary tasks.

    The class is constructed directly from tensors so that tests can
    feed synthetic data. Use :meth:`from_huggingface` to fetch real
    MNIST images at experiment time.
    """

    name: str = "split_mnist"
    SPLITS: tuple[SplitPair, ...] = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))

    def __init__(
        self,
        train_images: torch.Tensor,
        train_labels: torch.Tensor,
        test_images: torch.Tensor,
        test_labels: torch.Tensor,
        splits: Sequence[SplitPair] | None = None,
    ) -> None:
        self._validate_inputs(train_images, train_labels, "train")
        self._validate_inputs(test_images, test_labels, "test")
        self._train_images = train_images
        self._train_labels = train_labels
        self._test_images = test_images
        self._test_labels = test_labels
        self._splits = tuple(splits) if splits is not None else self.SPLITS

    @staticmethod
    def _validate_inputs(
        images: torch.Tensor, labels: torch.Tensor, kind: str
    ) -> None:
        if images.ndim < 2:
            raise ValueError(
                f"{kind} images must have at least 2 dims, got {images.ndim}"
            )
        if labels.ndim != 1:
            raise ValueError(
                f"{kind} labels must be 1-D, got shape {tuple(labels.shape)}"
            )
        if images.shape[0] != labels.shape[0]:
            raise ValueError(
                f"{kind} images and labels disagree on N: "
                f"{images.shape[0]} vs {labels.shape[0]}"
            )

    def tasks(self) -> list[Task]:
        """Construct the ordered list of binary tasks."""
        return [
            _build_task(
                name=f"{self.name}_{a}{b}",
                classes=(a, b),
                train_images=self._train_images,
                train_labels=self._train_labels,
                test_images=self._test_images,
                test_labels=self._test_labels,
            )
            for (a, b) in self._splits
        ]

    @property
    def num_classes_per_task(self) -> int:
        return 2

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (int(torch.tensor(self._train_images.shape[1:]).prod().item()),)

    @classmethod
    def from_huggingface(cls, cache_dir: str | None = None) -> "SplitMNIST":
        """Load MNIST via the HuggingFace ``datasets`` library.

        This is a lazy import so that the package can be imported in
        environments without network access. Tests should prefer
        constructing ``SplitMNIST`` from synthetic tensors directly.
        """
        from datasets import load_dataset  # type: ignore[import-untyped]

        ds = load_dataset("ylecun/mnist", cache_dir=cache_dir)
        train_images, train_labels = _hf_split_to_tensors(ds["train"])
        test_images, test_labels = _hf_split_to_tensors(ds["test"])
        return cls(train_images, train_labels, test_images, test_labels)


def _hf_split_to_tensors(
    hf_split,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a HuggingFace MNIST split to ``(images, labels)`` tensors."""
    import numpy as np

    images = np.stack(
        [np.asarray(sample, dtype=np.uint8) for sample in hf_split["image"]]
    )
    labels = np.asarray(hf_split["label"], dtype=np.int64)
    return torch.from_numpy(images), torch.from_numpy(labels)


# ---- PermutedMNIST: long-sequence continual benchmark ----


class PermutedMNIST:
    """Permuted-MNIST benchmark over ``num_tasks`` pixel-permutation tasks.

    Each task is the same MNIST 10-class classification problem, but
    the pixel order is permuted differently. The label space is
    shared across tasks (digits 0–9), so the standard setup is a
    single shared 10-class head. The first task uses the identity
    permutation; subsequent tasks use independent random
    permutations of the 784 pixel indices.

    Why this benchmark for the long-sequence decisive test:
    - 15+ tasks is a regime where the synapse layer's working
      memory should genuinely saturate.
    - Shared head means catastrophic forgetting is *aggressive*:
      every task's gradient updates the same classifier.
    - Each permutation produces a quite different activation
      pattern in the trunk, which is the property cold-storage
      retrieval is supposed to leverage (different inputs pull
      different archived patterns).
    """

    name: str = "permuted_mnist"

    def __init__(
        self,
        train_images: torch.Tensor,
        train_labels: torch.Tensor,
        test_images: torch.Tensor,
        test_labels: torch.Tensor,
        permutations: Sequence[torch.Tensor],
    ) -> None:
        self._validate_inputs(train_images, train_labels, "train")
        self._validate_inputs(test_images, test_labels, "test")
        for i, p in enumerate(permutations):
            if p.ndim != 1:
                raise ValueError(f"permutation {i} must be 1-D")
        self._train_images = train_images
        self._train_labels = train_labels
        self._test_images = test_images
        self._test_labels = test_labels
        self._permutations = list(permutations)

    @staticmethod
    def _validate_inputs(
        images: torch.Tensor, labels: torch.Tensor, kind: str
    ) -> None:
        if images.ndim < 2:
            raise ValueError(
                f"{kind} images must have at least 2 dims, got {images.ndim}"
            )
        if labels.ndim != 1:
            raise ValueError(
                f"{kind} labels must be 1-D, got shape {tuple(labels.shape)}"
            )
        if images.shape[0] != labels.shape[0]:
            raise ValueError(
                f"{kind} images and labels disagree on N: "
                f"{images.shape[0]} vs {labels.shape[0]}"
            )

    @property
    def num_classes_per_task(self) -> int:
        return 10

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (
            int(torch.tensor(self._train_images.shape[1:]).prod().item()),
        )

    @property
    def num_tasks(self) -> int:
        return len(self._permutations)

    def tasks(self) -> list[Task]:
        """Build one Task per permutation, sharing the underlying data."""
        return [
            self._build_permuted_task(i, perm)
            for i, perm in enumerate(self._permutations)
        ]

    def _build_permuted_task(
        self, index: int, perm: torch.Tensor
    ) -> Task:
        return Task(
            name=f"{self.name}_perm{index}",
            train=_apply_permutation(
                self._train_images, self._train_labels, perm
            ),
            test=_apply_permutation(
                self._test_images, self._test_labels, perm
            ),
            classes=tuple(range(10)),
        )

    @classmethod
    def from_huggingface(
        cls,
        num_tasks: int = 15,
        seed: int = 42,
        cache_dir: str | None = None,
    ) -> "PermutedMNIST":
        """Load MNIST and generate ``num_tasks`` permutations deterministically.

        Permutation 0 is the identity; permutations 1..num_tasks-1
        come from a CPU generator seeded by ``seed``. This makes
        runs reproducible across calls while still varying enough
        between tasks to make the problem hard.
        """
        if num_tasks <= 0:
            raise ValueError(f"num_tasks must be positive, got {num_tasks}")

        from datasets import load_dataset  # type: ignore[import-untyped]

        ds = load_dataset("ylecun/mnist", cache_dir=cache_dir)
        train_images, train_labels = _hf_split_to_tensors(ds["train"])
        test_images, test_labels = _hf_split_to_tensors(ds["test"])

        n_pixels = int(
            torch.tensor(train_images.shape[1:]).prod().item()
        )
        perms = _make_permutations(num_tasks, n_pixels, seed=seed)
        return cls(
            train_images, train_labels, test_images, test_labels, perms
        )


def _make_permutations(
    num_tasks: int, n_pixels: int, seed: int
) -> list[torch.Tensor]:
    """Deterministic permutation sequence with permutation 0 = identity."""
    g = torch.Generator().manual_seed(int(seed))
    perms: list[torch.Tensor] = [torch.arange(n_pixels, dtype=torch.long)]
    for _ in range(num_tasks - 1):
        perms.append(torch.randperm(n_pixels, generator=g))
    return perms


def _apply_permutation(
    images: torch.Tensor,
    labels: torch.Tensor,
    perm: torch.Tensor,
) -> TensorDataset:
    """Flatten, permute pixel order, normalise to [0, 1] if needed.

    Returns a :class:`TensorDataset` of ``(x, y)`` where ``x`` has
    shape ``(N, n_pixels)`` and ``y`` is the original integer label.
    """
    x = images.to(torch.float32).flatten(start_dim=1)
    if x.numel() and x.max() > 1.5:
        x = x / 255.0
    x = x[:, perm]
    y = labels.to(torch.int64)
    return TensorDataset(x, y)
