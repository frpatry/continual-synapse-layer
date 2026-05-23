"""Tests for the continual-learning metrics module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from continual_synapse.evaluation.metrics import (
    average_accuracy,
    average_forgetting,
    backward_transfer,
    forward_transfer,
)


def _hand_built_matrix() -> np.ndarray:
    """A small 3-task matrix with known metric values.

    R[i, j] = accuracy on task j after training task i. NaN for j > i
    so we can exercise the zero-shot policy.

    After task 0: acc on task 0 = 0.90
    After task 1: acc on task 0 = 0.80, task 1 = 0.85
    After task 2: acc on task 0 = 0.60, task 1 = 0.70, task 2 = 0.95
    """
    return np.array(
        [
            [0.90, np.nan, np.nan],
            [0.80, 0.85, np.nan],
            [0.60, 0.70, 0.95],
        ]
    )


def test_average_accuracy_is_mean_of_final_row() -> None:
    R = _hand_built_matrix()
    assert math.isclose(average_accuracy(R), (0.60 + 0.70 + 0.95) / 3)


def test_average_forgetting_compares_peak_to_final() -> None:
    R = _hand_built_matrix()
    # Task 0 peak = 0.90 (at i=0), final = 0.60 -> drop 0.30
    # Task 1 peak = 0.85, final = 0.70 -> drop 0.15
    expected = (0.30 + 0.15) / 2
    assert math.isclose(average_forgetting(R), expected, rel_tol=1e-9)


def test_backward_transfer_uses_first_post_training_accuracy() -> None:
    R = _hand_built_matrix()
    # (R[T-1, 0] - R[0, 0]) + (R[T-1, 1] - R[1, 1]) / (T - 1)
    expected = ((0.60 - 0.90) + (0.70 - 0.85)) / 2
    assert math.isclose(backward_transfer(R), expected, rel_tol=1e-9)


def test_forward_transfer_uses_zero_shot_entries() -> None:
    R = np.array(
        [
            [0.90, 0.55, np.nan],
            [0.80, 0.85, 0.60],
            [0.60, 0.70, 0.95],
        ]
    )
    baseline = np.array([0.5, 0.5, 0.5])
    # Task 1: R[0, 1] - 0.5 = 0.05
    # Task 2: R[1, 2] - 0.5 = 0.10
    expected = (0.05 + 0.10) / 2
    assert math.isclose(
        forward_transfer(R, baseline), expected, rel_tol=1e-9
    )


def test_metrics_reject_non_square_matrix() -> None:
    bad = np.zeros((2, 3))
    with pytest.raises(ValueError, match="square"):
        average_accuracy(bad)


def test_forgetting_returns_zero_for_single_task() -> None:
    R = np.array([[0.9]])
    assert average_forgetting(R) == 0.0
    assert backward_transfer(R) == 0.0


def test_forward_transfer_requires_zero_shot_entries() -> None:
    R = _hand_built_matrix()  # zero-shot entries are NaN
    with pytest.raises(ValueError, match="zero-shot"):
        forward_transfer(R, np.array([0.5, 0.5, 0.5]))
