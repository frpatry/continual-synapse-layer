"""Multi-seed evaluation helper.

Single-seed numbers are at noise scale for Split-MNIST with five
tasks of ~12 k samples each. PROJECT_PLAN.md §8.2 requires at
least five seeds per (method, benchmark) for statistical claims.
This module gives experiments a tiny utility to spin a method
across seeds without re-implementing the loop each time.

The caller supplies a *factory* that, given a seed, returns a
freshly-constructed ``(model, runner)`` pair. The factory owns
all seed-dependent setup (model init, optimiser-factory closures,
reward computers, etc.). The runner is expected to call
``set_seed`` itself at the start of ``run()`` — that re-seeds the
training-time RNG even if the factory already did it for init.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from torch import nn

from continual_synapse.evaluation.benchmarks import ContinualBenchmark
from continual_synapse.evaluation.runner import ContinualRunner, RunResult


MethodFactory = Callable[[int], tuple[nn.Module, ContinualRunner]]


@dataclass
class MultiSeedRun:
    """Outcome of running one method across multiple seeds."""

    method: str
    seeds: list[int]
    results: list[RunResult]

    @property
    def n_seeds(self) -> int:
        return len(self.results)


def run_multi_seed(
    method: str,
    factory: MethodFactory,
    benchmark: ContinualBenchmark,
    seeds: list[int],
    *,
    progress: Callable[[str, int, int], None] | None = None,
) -> MultiSeedRun:
    """Run ``factory(seed)`` once per seed and collect the results.

    Args:
        method: Display name for logs and serialisation.
        factory: Callable taking a seed and returning a fresh
            ``(model, runner)`` pair. All seed-dependent state lives
            inside the factory; this function does not touch RNGs.
        benchmark: The benchmark to run on. The same instance is
            reused across seeds (its data is deterministic and the
            runner's RNG handles shuffling).
        seeds: Seeds to iterate over. Order is preserved in the
            returned ``results`` list.
        progress: Optional callable ``(method, seed_index, n_seeds)``
            invoked before each run. Useful for ``tqdm``-style
            progress reporting from experiment scripts.

    Returns:
        A ``MultiSeedRun`` carrying the original method name, seeds,
        and one ``RunResult`` per seed.
    """
    results: list[RunResult] = []
    for i, seed in enumerate(seeds):
        if progress is not None:
            progress(method, i, len(seeds))
        model, runner = factory(seed)
        result = runner.run(model, benchmark)
        results.append(result)
    return MultiSeedRun(method=method, seeds=list(seeds), results=results)
