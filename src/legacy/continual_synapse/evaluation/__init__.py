"""Continual-learning evaluation harness: benchmarks, metrics, runner."""

from continual_synapse.evaluation.benchmarks import (
    ContinualBenchmark,
    PermutedMNIST,
    SplitMNIST,
    Task,
)
from continual_synapse.evaluation.metrics import (
    average_accuracy,
    average_forgetting,
    backward_transfer,
    forward_transfer,
)
from continual_synapse.evaluation.multi_seed import MultiSeedRun, run_multi_seed
from continual_synapse.evaluation.runner import ContinualRunner, RunResult

__all__ = [
    "ContinualBenchmark",
    "PermutedMNIST",
    "SplitMNIST",
    "Task",
    "average_accuracy",
    "average_forgetting",
    "backward_transfer",
    "forward_transfer",
    "ContinualRunner",
    "RunResult",
    "MultiSeedRun",
    "run_multi_seed",
]
