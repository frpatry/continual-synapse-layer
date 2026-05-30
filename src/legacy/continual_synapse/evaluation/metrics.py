"""Continual-learning metrics.

All metrics operate on an *accuracy matrix* ``R`` of shape ``(T, T)``
where ``R[i, j]`` is the accuracy on task ``j`` after training through
task ``i`` (both 0-indexed). Entries that the runner did not record
should be filled with ``numpy.nan`` and will be excluded from the
relevant computations.

Definitions follow PROJECT_PLAN.md section 8.1, which in turn matches
the common conventions in Lopez-Paz & Ranzato (2017) and subsequent
continual-learning literature.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_matrix(R: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(R, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(
            f"Accuracy matrix must be square 2-D, got shape {arr.shape}"
        )
    return arr


def average_accuracy(R: ArrayLike) -> float:
    """Mean accuracy across all tasks after training the final task.

    ``ACC = (1/T) * sum_j R[T-1, j]``.
    """
    arr = _as_matrix(R)
    final_row = arr[-1, :]
    if np.isnan(final_row).any():
        raise ValueError(
            "average_accuracy requires the final row to be fully populated"
        )
    return float(final_row.mean())


def average_forgetting(R: ArrayLike) -> float:
    """Mean drop from peak accuracy to final accuracy, over old tasks.

    ``FGT = (1/(T-1)) * sum_{j<T-1} (max_{i>=j} R[i, j] - R[T-1, j])``.

    Only the first ``T-1`` tasks contribute: the final task cannot
    have been forgotten because it was just trained.
    """
    arr = _as_matrix(R)
    T = arr.shape[0]
    if T < 2:
        return 0.0
    drops: list[float] = []
    for j in range(T - 1):
        column = arr[j : T, j]
        if np.isnan(column).any():
            raise ValueError(
                f"average_forgetting requires R[i, {j}] for i in [{j}, {T-1}]"
            )
        peak = float(column.max())
        final = float(arr[-1, j])
        drops.append(peak - final)
    return float(np.mean(drops))


def backward_transfer(R: ArrayLike) -> float:
    """Mean influence of later training on earlier tasks.

    ``BWT = (1/(T-1)) * sum_{j<T-1} (R[T-1, j] - R[j, j])``.

    Negative values indicate forgetting; positive values indicate that
    later tasks helped earlier ones.
    """
    arr = _as_matrix(R)
    T = arr.shape[0]
    if T < 2:
        return 0.0
    diffs: list[float] = []
    for j in range(T - 1):
        if np.isnan(arr[-1, j]) or np.isnan(arr[j, j]):
            raise ValueError(
                f"backward_transfer requires R[{T-1}, {j}] and R[{j}, {j}]"
            )
        diffs.append(float(arr[-1, j] - arr[j, j]))
    return float(np.mean(diffs))


def forward_transfer(
    R: ArrayLike,
    random_baseline: ArrayLike,
) -> float:
    """Mean head-start each task gets from earlier training.

    ``FWT = (1/(T-1)) * sum_{j>=1} (R[j-1, j] - random_baseline[j])``.

    The accuracy matrix must contain *pre-training* zero-shot entries
    at ``R[j-1, j]`` for ``j >= 1``. The runner records these by
    evaluating on each task before it begins training on it.
    """
    arr = _as_matrix(R)
    T = arr.shape[0]
    if T < 2:
        return 0.0
    baseline = np.asarray(random_baseline, dtype=np.float64)
    if baseline.shape != (T,):
        raise ValueError(
            f"random_baseline must have shape ({T},), got {baseline.shape}"
        )
    diffs: list[float] = []
    for j in range(1, T):
        if np.isnan(arr[j - 1, j]):
            raise ValueError(
                f"forward_transfer requires zero-shot R[{j-1}, {j}]"
            )
        diffs.append(float(arr[j - 1, j] - baseline[j]))
    return float(np.mean(diffs))
