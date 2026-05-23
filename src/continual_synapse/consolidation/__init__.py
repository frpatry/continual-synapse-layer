"""Synapse-to-cold-storage transfer pipeline (Phase 4)."""

from continual_synapse.consolidation.trigger import (
    ConsolidationTrigger,
    compute_pressure,
)

__all__ = ["ConsolidationTrigger", "compute_pressure"]
