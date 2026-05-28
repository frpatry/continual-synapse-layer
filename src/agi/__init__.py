"""AGI architecture — Phase 1 foundation.

This subpackage is **parallel to** the legacy ``continual_synapse/``
work, not an extension of it. The CL phases produced useful
mechanistic insights (input replay works, prototype-only memory
underperforms, etc.); the AGI project starts from a different
premise — a frozen pretrained LLM augmented with privacy-
preserving episodic memory — and treats the CL work as a
reference library, not a base class to inherit from.

Phase 1.0 lays the foundation:
- :class:`FrozenFoundation` — Qwen-0.5B-Instruct wrapper, frozen,
  with stable-key extraction.
- :class:`FactExtractor` — pattern-based structured-fact extraction.
- :class:`XRayEpisodicMemory` — keys + structured facts, never
  raw text.
- :class:`AGISystem` — observe / respond pipeline.

Phases 2-4 will add reward-modulated plasticity, multi-timescale
memory, and reasoning on top.

Note: :class:`FrozenFoundation` and :class:`AGISystem` import
``transformers``; the bare memory + extraction layers do not. To
keep test isolation clean (pure-memory tests shouldn't drag in
~250 MB of transformer dependencies), this top-level ``__init__``
does NOT auto-import the foundation / integration modules. Import
them explicitly:

    from agi.foundation import FrozenFoundation
    from agi.integration import AGISystem
"""

from .extraction import FactExtractor
from .memory.xray_episodic import EpisodicEntry, XRayEpisodicMemory

__all__ = [
    "FactExtractor",
    "XRayEpisodicMemory",
    "EpisodicEntry",
]
