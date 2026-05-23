"""Sequential training and evaluation harness.

The runner takes a model and a :class:`ContinualBenchmark`, then for
each task in order:

1. Optionally records the model's zero-shot accuracy on every
   not-yet-seen task (needed to compute forward transfer).
2. Trains the model for ``epochs_per_task`` epochs on the task's
   training set.
3. Evaluates accuracy on every task seen so far and writes the
   results into the accuracy matrix.

The runner is deliberately model-agnostic. Two optional hook
points let continual-learning methods inject behaviour without
subclassing:

- ``regulariser(model) -> Tensor`` is added to the per-batch loss.
  EWC and similar methods use this to penalise parameter drift.
- ``on_task_end(i, task, model)`` fires after each task's training
  loop. EWC uses it to compute Fisher and snapshot parameters.
- ``on_after_batch(i, task, model, x, y)`` fires after every
  optimizer step. The synapse-augmented MLP uses it to apply the
  Hebbian update from the activations cached during forward.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from continual_synapse.evaluation.benchmarks import ContinualBenchmark, Task


logger = logging.getLogger(__name__)

OptimizerFactory = Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer]
Regulariser = Callable[[nn.Module], torch.Tensor]
TaskEndCallback = Callable[[int, Task, nn.Module], None]
AfterBatchCallback = Callable[
    [int, Task, nn.Module, torch.Tensor, torch.Tensor], None
]


@dataclass
class RunResult:
    """Outcome of a single continual-learning run.

    Attributes:
        benchmark: Name of the benchmark, copied from
            ``ContinualBenchmark.name``.
        task_names: Ordered task names, length ``T``.
        accuracy_matrix: ``(T, T)`` float matrix. ``R[i, j]`` holds
            the accuracy on task ``j`` after training task ``i``.
            Entries the runner did not record are ``numpy.nan``.
        random_baseline: Per-task accuracy expected from random
            guessing, length ``T``. For balanced binary tasks this
            is ``0.5`` everywhere; recorded so that forward-transfer
            computations are self-contained.
    """

    benchmark: str
    task_names: list[str]
    accuracy_matrix: np.ndarray
    random_baseline: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def num_tasks(self) -> int:
        return len(self.task_names)


def set_seed(seed: int) -> None:
    """Seed the standard RNGs. Idempotent."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class ContinualRunner:
    """Train a model sequentially over a benchmark and record accuracies."""

    optimizer_factory: OptimizerFactory = field(
        default_factory=lambda: lambda params: torch.optim.SGD(
            params, lr=0.01, momentum=0.9
        )
    )
    loss_fn: nn.Module = field(default_factory=nn.CrossEntropyLoss)
    epochs_per_task: int = 1
    batch_size: int = 64
    eval_batch_size: int = 256
    device: str = "cpu"
    seed: int | None = None
    record_zero_shot: bool = True
    regulariser: Regulariser | None = None
    on_task_end: TaskEndCallback | None = None
    on_after_batch: AfterBatchCallback | None = None

    def run(self, model: nn.Module, benchmark: ContinualBenchmark) -> RunResult:
        """Train ``model`` sequentially on ``benchmark`` and return results."""
        if self.seed is not None:
            set_seed(self.seed)

        tasks = benchmark.tasks()
        T = len(tasks)
        if T == 0:
            raise ValueError("Benchmark produced no tasks")

        model = model.to(self.device)
        optimizer = self.optimizer_factory(model.parameters())

        R = np.full((T, T), np.nan, dtype=np.float64)

        for i, task in enumerate(tasks):
            if self.record_zero_shot and i + 1 < T:
                R[i, i + 1] = self._evaluate(model, tasks[i + 1])

            self._train_one_task(model, optimizer, task, task_index=i)

            if self.on_task_end is not None:
                self.on_task_end(i, task, model)

            for j in range(i + 1):
                R[i, j] = self._evaluate(model, tasks[j])

            logger.info(
                "task=%s (%d/%d) acc_seen=%s",
                task.name,
                i + 1,
                T,
                R[i, : i + 1].tolist(),
            )

        baseline = self._random_baseline(tasks, benchmark.num_classes_per_task)

        return RunResult(
            benchmark=benchmark.name,
            task_names=[t.name for t in tasks],
            accuracy_matrix=R,
            random_baseline=baseline,
        )

    def _train_one_task(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        task: Task,
        task_index: int,
    ) -> None:
        loader = DataLoader(
            task.train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )
        model.train()
        for _ in range(self.epochs_per_task):
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                logits = model(x)
                loss = self.loss_fn(logits, y)
                if self.regulariser is not None:
                    loss = loss + self.regulariser(model)
                loss.backward()
                optimizer.step()
                if self.on_after_batch is not None:
                    self.on_after_batch(task_index, task, model, x, y)

    @torch.no_grad()
    def _evaluate(self, model: nn.Module, task: Task) -> float:
        loader = DataLoader(
            task.test,
            batch_size=self.eval_batch_size,
            shuffle=False,
        )
        model.eval()
        correct = 0
        total = 0
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            preds = model(x).argmax(dim=1)
            correct += int((preds == y).sum().item())
            total += int(y.numel())
        if total == 0:
            return float("nan")
        return correct / total

    @staticmethod
    def _random_baseline(
        tasks: list[Task], num_classes_per_task: int
    ) -> np.ndarray:
        """Empirical majority-class accuracy on each task's test set.

        For perfectly balanced binary tasks this is ``0.5``. We use
        the empirical majority share rather than a hardcoded value so
        the baseline still makes sense on slightly imbalanced splits.
        """
        baselines: list[float] = []
        for task in tasks:
            assert isinstance(task.test, TensorDataset)
            _, y = task.test.tensors
            if y.numel() == 0:
                baselines.append(1.0 / max(num_classes_per_task, 1))
                continue
            counts = torch.bincount(y, minlength=num_classes_per_task)
            baselines.append(float(counts.max().item() / y.numel()))
        return np.asarray(baselines, dtype=np.float64)
