"""End-to-end tests for the continual-learning runner."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset

from continual_synapse.baselines.naive_finetune import MLPClassifier, MLPConfig
from continual_synapse.evaluation.benchmarks import Task
from continual_synapse.evaluation.runner import (
    ContinualRunner,
    RunResult,
    set_seed,
)


class _TwoTaskBenchmark:
    """Trivial in-memory benchmark for smoke-testing the runner.

    Two binary tasks built from linearly separable random clusters.
    """

    name = "tiny_two_task"
    num_classes_per_task = 2
    input_shape = (8,)

    def __init__(self, seed: int = 0) -> None:
        g = torch.Generator().manual_seed(seed)
        self._g = g

    def _make_task(self, name: str, offset: float) -> Task:
        n = 32
        # Class 0: mean -offset, class 1: mean +offset.
        x0 = torch.randn(n, 8, generator=self._g) - offset
        x1 = torch.randn(n, 8, generator=self._g) + offset
        x = torch.cat([x0, x1])
        y = torch.cat([torch.zeros(n, dtype=torch.int64), torch.ones(n, dtype=torch.int64)])
        # Split 80/20 train/test deterministically.
        idx = torch.randperm(x.shape[0], generator=self._g)
        x, y = x[idx], y[idx]
        split = int(0.8 * x.shape[0])
        return Task(
            name=name,
            train=TensorDataset(x[:split], y[:split]),
            test=TensorDataset(x[split:], y[split:]),
            classes=(0, 1),
        )

    def tasks(self) -> list[Task]:
        return [
            self._make_task("task_a", offset=1.5),
            self._make_task("task_b", offset=2.0),
        ]


def _build_runner(seed: int = 0, record_zero_shot: bool = True) -> ContinualRunner:
    return ContinualRunner(
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.1, momentum=0.0),
        epochs_per_task=2,
        batch_size=16,
        eval_batch_size=32,
        seed=seed,
        record_zero_shot=record_zero_shot,
    )


def test_runner_returns_filled_lower_triangle() -> None:
    bench = _TwoTaskBenchmark()
    model = MLPClassifier(MLPConfig(input_dim=8, hidden_dim=16, num_classes=2))
    result = _build_runner().run(model, bench)

    assert isinstance(result, RunResult)
    assert result.benchmark == "tiny_two_task"
    assert result.task_names == ["task_a", "task_b"]
    R = result.accuracy_matrix
    assert R.shape == (2, 2)
    # Lower triangle (incl. diagonal) is filled.
    assert not np.isnan(R[0, 0])
    assert not np.isnan(R[1, 0])
    assert not np.isnan(R[1, 1])
    # Zero-shot upper-diagonal entry was recorded.
    assert not np.isnan(R[0, 1])


def test_runner_can_skip_zero_shot() -> None:
    bench = _TwoTaskBenchmark()
    model = MLPClassifier(MLPConfig(input_dim=8, hidden_dim=16, num_classes=2))
    result = _build_runner(record_zero_shot=False).run(model, bench)
    R = result.accuracy_matrix
    assert np.isnan(R[0, 1])
    assert not np.isnan(R[1, 0])


def test_runner_is_reproducible_with_seed() -> None:
    """Two identically-seeded runs must produce the same accuracy matrix.

    The model is constructed *after* seeding so that weight init is
    reproducible too: the runner only controls training-time RNG.
    """
    def one_run() -> np.ndarray:
        set_seed(7)
        model = MLPClassifier(MLPConfig(input_dim=8, hidden_dim=16, num_classes=2))
        bench = _TwoTaskBenchmark(seed=42)
        return _build_runner(seed=7).run(model, bench).accuracy_matrix

    np.testing.assert_allclose(
        np.nan_to_num(one_run()), np.nan_to_num(one_run())
    )


def test_runner_can_learn_first_task() -> None:
    """Sanity check: a small MLP should fit a linearly-separable task."""
    bench = _TwoTaskBenchmark(seed=1)
    model = MLPClassifier(MLPConfig(input_dim=8, hidden_dim=32, num_classes=2))
    result = _build_runner(seed=1).run(model, bench)
    # Final-task accuracy should clearly beat chance on a separable problem.
    assert result.accuracy_matrix[-1, -1] > 0.7


def test_random_baseline_matches_class_balance() -> None:
    bench = _TwoTaskBenchmark()
    model = MLPClassifier(MLPConfig(input_dim=8, hidden_dim=16, num_classes=2))
    result = _build_runner().run(model, bench)
    # Tasks are balanced 50/50 by construction, so majority-class baseline
    # should be 0.5 give or take the test-split rounding.
    assert result.random_baseline.shape == (2,)
    for b in result.random_baseline:
        assert 0.4 <= b <= 0.7
