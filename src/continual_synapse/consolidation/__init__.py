"""Synapse-to-cold-storage transfer pipeline (Phase 4)."""

from continual_synapse.consolidation.pipeline import consolidate_to_storage
from continual_synapse.consolidation.reconstruction import (
    fetch_entries_for_query,
    reconstruct_strengths,
)
from continual_synapse.consolidation.trigger import (
    ConsolidationTrigger,
    compute_pressure,
)

__all__ = [
    "ConsolidationTrigger",
    "compute_pressure",
    "consolidate_to_storage",
    "fetch_entries_for_query",
    "reconstruct_strengths",
]
