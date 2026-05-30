"""Tests for the multi-seed runner helper."""

from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from continual_synapse.baselines.naive_finetune import MLPClassifier, MLPConfig
from continual_synapse.evaluation.benchmarks import Task
from continual_synapse.evaluation.multi_seed import (
    MultiSeedRun,
    run_multi_seed,
)
from continual_synapse.evaluation.runner import ContinualRunner, set_seed


class _TinyBenchmark:
    """Synthetic two-task benchmark for multi-seed smoke tests.

    Like real benchmarks (e.g. SplitMNIST), ``tasks()`` is
    deterministic: each call seeds a fresh local generator so
    re-invoking the benchmark always yields the same task set.
    """

    name = "ms_smoke"
    num_classes_per_task = 2
    input_shape = (4,)

    def _make(self, name: str, offset: float, g: torch.Generator) -> Task:
        x0 = torch.randn(32, 4, generator=g) - offset
        x1 = torch.randn(32, 4, generator=g) + offset
        x = torch.cat([x0, x1])
        y = torch.cat(
            [torch.zeros(32, dtype=torch.int64), torch.ones(32, dtype=torch.int64)]
        )
        idx = torch.randperm(x.shape[0], generator=g)
        x, y = x[idx], y[idx]
        return Task(
            name=name,
            train=TensorDataset(x[:48], y[:48]),
            test=TensorDataset(x[48:], y[48:]),
            classes=(0, 1),
        )

    def tasks(self) -> list[Task]:
        g = torch.Generator().manual_seed(42)
        return [self._make("a", 1.5, g), self._make("b", -1.5, g)]


def _factory(seed: int) -> tuple[MLPClassifier, ContinualRunner]:
    set_seed(seed)
    model = MLPClassifier(MLPConfig(input_dim=4, hidden_dim=8, num_classes=2))
    runner = ContinualRunner(
        optimizer_factory=lambda p: torch.optim.SGD(p, lr=0.05),
        epochs_per_task=1,
        batch_size=16,
        eval_batch_size=16,
        seed=seed,
    )
    return model, runner


def test_run_multi_seed_returns_one_result_per_seed() -> None:
    bench = _TinyBenchmark()
    run = run_multi_seed("naive", _factory, bench, seeds=[0, 1, 2])
    assert isinstance(run, MultiSeedRun)
    assert run.method == "naive"
    assert run.n_seeds == 3
    assert run.seeds == [0, 1, 2]
    assert len(run.results) == 3
    for r in run.results:
        assert r.accuracy_matrix.shape == (2, 2)


def test_run_multi_seed_is_reproducible_across_invocations() -> None:
    bench = _TinyBenchmark()
    seeds = [0, 1]
    run_a = run_multi_seed("naive", _factory, bench, seeds=seeds)
    run_b = run_multi_seed("naive", _factory, bench, seeds=seeds)
    for ra, rb in zip(run_a.results, run_b.results):
        import numpy as np

        np.testing.assert_allclose(
            np.nan_to_num(ra.accuracy_matrix),
            np.nan_to_num(rb.accuracy_matrix),
        )


def test_run_multi_seed_calls_progress_callback() -> None:
    bench = _TinyBenchmark()
    calls: list[tuple[str, int, int]] = []
    run_multi_seed(
        "x",
        _factory,
        bench,
        seeds=[0, 1, 2],
        progress=lambda m, i, n: calls.append((m, i, n)),
    )
    assert calls == [("x", 0, 3), ("x", 1, 3), ("x", 2, 3)]


def test_run_multi_seed_calls_on_seed_complete_with_result() -> None:
    """The completion callback receives the per-seed RunResult so
    experiment scripts can print metrics as each seed lands."""
    from continual_synapse.evaluation.runner import RunResult

    bench = _TinyBenchmark()
    payloads: list[tuple[str, int, int, int, RunResult]] = []
    run_multi_seed(
        "m",
        _factory,
        bench,
        seeds=[5, 7],
        on_seed_complete=lambda m, i, n, seed, result: payloads.append(
            (m, i, n, seed, result)
        ),
    )
    assert len(payloads) == 2
    assert [p[0] for p in payloads] == ["m", "m"]
    assert [p[1] for p in payloads] == [0, 1]  # seed_index
    assert [p[2] for p in payloads] == [2, 2]  # n_seeds
    assert [p[3] for p in payloads] == [5, 7]  # actual seed values
    # Result tensors are populated for both seeds.
    for _, _, _, _, result in payloads:
        assert result.accuracy_matrix.shape == (2, 2)
