"""Split-CIFAR-100 class-incremental benchmark.

Standard 10 × 10 split: each of the 10 tasks gets a contiguous block
of 10 classes from CIFAR-100. Class-incremental setup — the model
predicts over the **union** of classes seen so far at every eval
point, without being told which task a sample came from.

Standard augmentation pipeline applied to training samples only:
- ``RandomCrop(32, padding=4)`` with constant-zero padding
- ``RandomHorizontalFlip(p=0.5)``

Both train and eval samples are normalised with CIFAR-100 channel
mean / std after augmentation.

Storage policy: raw uint8 images are kept in memory (≈ 153 MB for
the train split, 30 MB for the test split). Tensors are cast to
float32 in ``[0, 1]`` per-sample inside ``__getitem__`` to keep
the resident-set small.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


# Standard CIFAR-100 channel statistics — used by ResNet/CNN
# baselines across the literature.
CIFAR100_MEAN: tuple[float, float, float] = (0.5071, 0.4867, 0.4408)
CIFAR100_STD: tuple[float, float, float] = (0.2675, 0.2565, 0.2761)


class _CIFAR100Dataset(Dataset):
    """In-memory CIFAR-100 subset with optional augmentation.

    Holds the raw images as uint8 (HxWxC=32x32x3 → stored as
    CxHxW=3x32x32 for cheap channel-first slicing). On
    ``__getitem__`` the sample is cast to float32 in ``[0, 1]``,
    augmented (if enabled), then normalised. Augmentation uses
    PyTorch primitives — no torchvision dep required.
    """

    def __init__(
        self,
        images_u8: Tensor,
        labels: Tensor,
        augment: bool,
        mean: Sequence[float] = CIFAR100_MEAN,
        std:  Sequence[float] = CIFAR100_STD,
    ) -> None:
        if images_u8.dtype != torch.uint8:
            raise ValueError(
                f"images_u8 must be uint8, got {images_u8.dtype}"
            )
        if images_u8.ndim != 4 or images_u8.shape[1] != 3:
            raise ValueError(
                f"images_u8 must be (N, 3, H, W); got "
                f"{tuple(images_u8.shape)}"
            )
        if labels.ndim != 1 or labels.shape[0] != images_u8.shape[0]:
            raise ValueError(
                f"labels must be 1-D of length N; got "
                f"{tuple(labels.shape)} for N={images_u8.shape[0]}"
            )

        self._images = images_u8
        self._labels = labels.to(torch.long)
        self._augment = bool(augment)
        self._mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self._std  = torch.tensor(std,  dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return self._images.shape[0]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        img = self._images[idx].to(torch.float32) / 255.0  # (3, 32, 32)
        if self._augment:
            # RandomCrop(32, padding=4) with constant-zero
            # padding (matches torchvision default).
            padded = F.pad(img, (4, 4, 4, 4), mode="constant", value=0.0)
            top = int(torch.randint(0, 9, (1,)).item())
            left = int(torch.randint(0, 9, (1,)).item())
            img = padded[:, top : top + 32, left : left + 32]
            # RandomHorizontalFlip(p=0.5).
            if torch.rand(()).item() < 0.5:
                img = torch.flip(img, dims=[-1])
        img = (img - self._mean) / self._std
        return img, self._labels[idx]


class SplitCIFAR100ClassIncremental:
    """Split CIFAR-100 into ``num_tasks`` sequential tasks, each
    containing ``100 // num_tasks`` contiguous classes from the
    optional ``class_order`` permutation (defaults to the natural
    0..99 order).

    The class is constructed directly from tensors so unit tests
    can feed small synthetic inputs. Use :meth:`from_huggingface`
    to fetch real CIFAR-100 at experiment time.
    """

    name: str = "split_cifar100_class_incremental"

    def __init__(
        self,
        train_images: Tensor,
        train_labels: Tensor,
        test_images: Tensor,
        test_labels: Tensor,
        num_tasks: int = 10,
        class_order: Sequence[int] | None = None,
    ) -> None:
        if num_tasks <= 0:
            raise ValueError(f"num_tasks must be positive, got {num_tasks}")
        n_classes_total = 100
        if n_classes_total % num_tasks != 0:
            raise ValueError(
                f"100 classes must split evenly across num_tasks; "
                f"100 / {num_tasks} is not an integer."
            )

        # Validate image / label shapes — uint8 (N, 3, 32, 32) for
        # images, int64 (N,) for labels.
        for tag, imgs, lbls in (
            ("train", train_images, train_labels),
            ("test",  test_images,  test_labels),
        ):
            if imgs.dtype != torch.uint8:
                raise ValueError(
                    f"{tag} images must be uint8; got {imgs.dtype}"
                )
            if imgs.ndim != 4 or imgs.shape[1:] != (3, 32, 32):
                raise ValueError(
                    f"{tag} images must be (N, 3, 32, 32); got "
                    f"{tuple(imgs.shape)}"
                )
            if lbls.ndim != 1 or lbls.shape[0] != imgs.shape[0]:
                raise ValueError(
                    f"{tag} labels disagree with images on N"
                )

        self._train_images = train_images
        self._train_labels = train_labels.to(torch.long)
        self._test_images  = test_images
        self._test_labels  = test_labels.to(torch.long)
        self._num_tasks = int(num_tasks)
        self._classes_per_task = n_classes_total // self._num_tasks
        if class_order is None:
            self._class_order: list[int] = list(range(n_classes_total))
        else:
            order = list(class_order)
            if sorted(order) != list(range(n_classes_total)):
                raise ValueError(
                    "class_order must be a permutation of 0..99"
                )
            self._class_order = order

    @property
    def num_tasks(self) -> int:
        return self._num_tasks

    @property
    def classes_per_task(self) -> int:
        return self._classes_per_task

    @property
    def num_classes_total(self) -> int:
        return 100

    def task_classes(self, task_id: int) -> list[int]:
        """Return the classes that belong to ``task_id`` under
        the configured ``class_order``."""
        if not 0 <= task_id < self._num_tasks:
            raise IndexError(
                f"task_id must be in [0, {self._num_tasks}); "
                f"got {task_id}"
            )
        start = task_id * self._classes_per_task
        return self._class_order[start : start + self._classes_per_task]

    def classes_seen_through(self, task_id: int) -> list[int]:
        """Union of classes seen after training task ``task_id``
        (inclusive). The class-incremental eval at this point
        must predict over this set."""
        if not 0 <= task_id < self._num_tasks:
            raise IndexError(
                f"task_id must be in [0, {self._num_tasks}); "
                f"got {task_id}"
            )
        return self._class_order[
            : (task_id + 1) * self._classes_per_task
        ]

    def num_classes_seen(self, after_task: int) -> int:
        """Convenience: ``self.classes_per_task * (after_task + 1)``."""
        return self._classes_per_task * (after_task + 1)

    @staticmethod
    def _mask_for_classes(
        labels: Tensor, classes: Iterable[int],
    ) -> Tensor:
        mask = torch.zeros(labels.shape[0], dtype=torch.bool)
        for c in classes:
            mask = mask | (labels == int(c))
        return mask

    def task_train_dataset(
        self, task_id: int, *, augment: bool = True,
    ) -> _CIFAR100Dataset:
        """Returns the per-task train Dataset. Augmentation is
        on by default (the train-time setting); pass
        ``augment=False`` for diagnostic uses (e.g., feature
        extraction)."""
        classes = self.task_classes(task_id)
        mask = self._mask_for_classes(self._train_labels, classes)
        return _CIFAR100Dataset(
            self._train_images[mask],
            self._train_labels[mask],
            augment=augment,
        )

    def task_test_dataset(self, task_id: int) -> _CIFAR100Dataset:
        """Test samples whose label is in ``task_id``'s class set.
        No augmentation."""
        classes = self.task_classes(task_id)
        mask = self._mask_for_classes(self._test_labels, classes)
        return _CIFAR100Dataset(
            self._test_images[mask],
            self._test_labels[mask],
            augment=False,
        )

    def eval_dataset(self, up_to_task: int) -> _CIFAR100Dataset:
        """Class-incremental eval dataset: every test sample whose
        label belongs to a task in ``[0, up_to_task]``. No
        augmentation."""
        classes = self.classes_seen_through(up_to_task)
        mask = self._mask_for_classes(self._test_labels, classes)
        return _CIFAR100Dataset(
            self._test_images[mask],
            self._test_labels[mask],
            augment=False,
        )

    def get_task_train_loader(
        self, task_id: int, batch_size: int = 64,
        shuffle: bool = True, num_workers: int = 0,
        augment: bool = True,
    ) -> DataLoader:
        return DataLoader(
            self.task_train_dataset(task_id, augment=augment),
            batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers,
        )

    def get_eval_loader(
        self, up_to_task: int, batch_size: int = 256,
        num_workers: int = 0,
    ) -> DataLoader:
        return DataLoader(
            self.eval_dataset(up_to_task),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers,
        )

    # ---------- loading ----------

    @classmethod
    def from_huggingface(
        cls,
        num_tasks: int = 10,
        class_order: Sequence[int] | None = None,
        cache_dir: str | None = None,
    ) -> "SplitCIFAR100ClassIncremental":
        """Load CIFAR-100 via the HuggingFace ``datasets`` library
        and construct the benchmark. The lazy import keeps this
        package importable in environments without network
        access; unit tests should prefer constructing
        ``SplitCIFAR100ClassIncremental`` from synthetic tensors
        directly."""
        from datasets import load_dataset  # type: ignore[import-untyped]

        # ``uoft-cs/cifar100`` is the canonical HF mirror with
        # the standard fine-label split. The bare ``cifar100``
        # name resolves to the same dataset on recent ``datasets``
        # releases; either should work.
        ds = load_dataset("uoft-cs/cifar100", cache_dir=cache_dir)
        train_images, train_labels = _hf_cifar100_to_tensors(ds["train"])
        test_images,  test_labels  = _hf_cifar100_to_tensors(ds["test"])
        return cls(
            train_images, train_labels, test_images, test_labels,
            num_tasks=num_tasks, class_order=class_order,
        )


# ---------- helpers ----------


def _hf_cifar100_to_tensors(hf_split) -> tuple[Tensor, Tensor]:
    """Convert a HuggingFace CIFAR-100 split to ``(images_u8, labels)``.

    The HF dataset returns PIL images and an integer label column.
    Both label-column names that appear in the wild — ``fine_label``
    (the standard) and ``label`` (older snapshots) — are accepted.
    Returns:
        images_u8: ``(N, 3, 32, 32)`` uint8 tensor
        labels:    ``(N,)`` int64 tensor
    """
    img_col = "img" if "img" in hf_split.column_names else "image"
    if "fine_label" in hf_split.column_names:
        label_col = "fine_label"
    elif "label" in hf_split.column_names:
        label_col = "label"
    else:
        raise RuntimeError(
            f"CIFAR-100 HF split has no recognised label column; "
            f"got {hf_split.column_names}"
        )

    images_np = np.stack([
        np.asarray(sample, dtype=np.uint8) for sample in hf_split[img_col]
    ])
    # images_np shape: (N, 32, 32, 3) → transpose to (N, 3, 32, 32).
    images_np = images_np.transpose(0, 3, 1, 2)
    images = torch.from_numpy(images_np)  # uint8

    labels_np = np.asarray(hf_split[label_col], dtype=np.int64)
    labels = torch.from_numpy(labels_np)
    return images, labels
