"""Memory adapters for the Phase 5.6 / 5.7 work.

Currently hosts:

- :class:`CIFARMultiLevelMemory` (Phase 5.6.2): reservoir-sampled
  buffer storing raw input + GAP'd multi-level features from
  both substrates + soft target + classes-seen mask.
- :class:`XRayMemory` (Phase 5.7.0): per-class prototype memory
  with EMA refinement, progressive sparsification, temperature
  scheduling. No raw inputs stored.
- :func:`nt_xent_multi_prototype_loss`: matching supervised
  contrastive loss for XRayMemory.
"""

from .cifar_multi_level_memory import CIFARMultiLevelMemory
from .xray_memory import XRayMemory, nt_xent_multi_prototype_loss

__all__ = [
    "CIFARMultiLevelMemory",
    "XRayMemory",
    "nt_xent_multi_prototype_loss",
]
