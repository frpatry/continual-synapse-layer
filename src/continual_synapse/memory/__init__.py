"""Memory adapters for the Phase 5.6 cross-benchmark.

Currently hosts :class:`CIFARMultiLevelMemory` — a reservoir-sampled
buffer that stores GAP'd multi-level features from both the
hippocampe and the neocortex on CIFAR-100 class-incremental.
"""

from .cifar_multi_level_memory import CIFARMultiLevelMemory

__all__ = ["CIFARMultiLevelMemory"]
