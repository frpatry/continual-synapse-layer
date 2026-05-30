"""Continual-learning benchmarks for vision-rich settings.

This subpackage hosts benchmarks added during Phase 5.6 (Split-
CIFAR-100). The pre-existing MNIST-family benchmarks (Split-MNIST,
Permuted-MNIST, SplitMNISTClassIncremental) currently live under
``continual_synapse.evaluation.benchmarks``; that module isn't
moved here yet to avoid churn unrelated to the cross-benchmark
work. A later consolidation pass can fold all benchmarks under
this single namespace.
"""

from .split_cifar100_ci import SplitCIFAR100ClassIncremental

__all__ = ["SplitCIFAR100ClassIncremental"]
