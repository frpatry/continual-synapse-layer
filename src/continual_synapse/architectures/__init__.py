"""Vision architectures used by the Phase 5.6 cross-benchmark.

Currently hosts:
- :class:`CIFARHippocampus` — a small fast CNN designed for the
  hippocampe role on CIFAR-style 32x32 inputs.
- :class:`CIFARNeocortex` — a Reduced ResNet-18 (CIFAR variant,
  3x3 stem, no initial maxpool) designed for the slow-learner
  neocortex role.

Both expose a ``features(x)`` method returning a ``{low, mid, high}``
dict of intermediate feature maps for memory-storage purposes.
"""

from .cifar_cnn import (
    BasicBlock,
    CIFARHippocampus,
    CIFARNeocortex,
)

__all__ = ["BasicBlock", "CIFARHippocampus", "CIFARNeocortex"]
