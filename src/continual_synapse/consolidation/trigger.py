"""Pressure metric and consolidation trigger.

DESIGN.md §3.5 defines the pressure metric used to decide when a
synapse is ripe for archival:

    pressure_ij = |strength_ij| * evidence_ij / (1 + access_count_ij)

High-strength, high-evidence, rarely-accessed synapses score
highest — they have learned something durable that the layer is
not actively using and that we'd want to preserve in cold storage.

The trigger fires when the *average* pressure across the synapse
state exceeds a configurable threshold. The intent is "when the
working set is rich enough that some of it should be offloaded".
Using the mean (rather than max) keeps the trigger robust to a few
outlier synapses with extreme strengths.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from continual_synapse.synapse_layer.layer import SynapseLayer


def compute_pressure(synapse: SynapseLayer) -> torch.Tensor:
    """Return the per-synapse pressure matrix.

    ``|strength_ij| * evidence_ij / (1 + access_count_ij)``, all
    in float32, same shape as the synapse state buffers. No grad
    history is retained.
    """
    with torch.no_grad():
        access = synapse.access_count.to(torch.float32)
        evidence = synapse.evidence
        strength = synapse.strengths.abs()
        return strength * evidence / (1.0 + access)


@dataclass
class ConsolidationTrigger:
    """Decide when a consolidation cycle should fire.

    Attributes:
        avg_pressure_threshold: Fire when ``compute_pressure(syn).mean()``
            exceeds this value.
        min_steps_between: Refractory period in consolidate() steps —
            don't fire again within this many steps of the previous
            consolidation. Prevents repeated firings on the same
            state when the threshold is set low.
        candidate_quantile: When the trigger fires, this fraction of
            synapses (those with pressure in the top quantile) are
            the *candidates* the consolidation pipeline will archive.
            ``0.1`` means the top 10 % by pressure.
    """

    avg_pressure_threshold: float = 0.05
    min_steps_between: int = 10
    candidate_quantile: float = 0.1
    _last_fire_step: int = -10_000  # initial value safely far in the past

    def __post_init__(self) -> None:
        if self.avg_pressure_threshold < 0:
            raise ValueError("avg_pressure_threshold must be >= 0")
        if self.min_steps_between < 0:
            raise ValueError("min_steps_between must be >= 0")
        if not 0.0 < self.candidate_quantile <= 1.0:
            raise ValueError(
                "candidate_quantile must be in (0, 1]"
            )

    def should_fire(self, synapse: SynapseLayer) -> bool:
        """Return True if it's time to consolidate ``synapse``.

        The "mean pressure across the synapse state" is taken over the
        *active* synapses — i.e. those with a non-zero strength —
        rather than the full ``(n, n)`` buffer. Sparse top-k mode zeros
        out most entries; including those zeros in the mean
        artificially dilutes pressure by ``1 / density`` and was the
        root cause of the cs_full_sparse pathology surfaced after
        experiment 12 (consolidation barely fired in sparse mode,
        cold storage stayed empty, modulator gate ran away negative).
        Dense mode has an all-True active mask, so the masked mean
        equals the unmasked mean bit-exact.
        """
        step = int(synapse.global_step.item())
        if step - self._last_fire_step < self.min_steps_between:
            return False
        pressures = compute_pressure(synapse)
        active_mask = synapse.strengths != 0
        if not active_mask.any():
            return False
        avg = float(pressures[active_mask].mean().item())
        return avg >= self.avg_pressure_threshold

    def candidate_mask(self, synapse: SynapseLayer) -> torch.Tensor:
        """Boolean mask of the synapses that should be archived.

        Returned as a ``(n, n)`` bool tensor with ``True`` where the
        pressure is in the top ``candidate_quantile`` of all synapses.
        Ties are broken by ``torch.quantile``'s implementation; the
        result is deterministic for a fixed state.
        """
        pressure = compute_pressure(synapse).flatten()
        if pressure.numel() == 0:
            return torch.zeros_like(synapse.strengths, dtype=torch.bool)
        cutoff_q = 1.0 - self.candidate_quantile
        cutoff = pressure.quantile(cutoff_q).item()
        mask = compute_pressure(synapse) >= cutoff
        return mask

    def mark_fired(self, synapse: SynapseLayer) -> None:
        """Record that a consolidation cycle has just been performed.

        The trigger uses this to enforce ``min_steps_between``.
        """
        self._last_fire_step = int(synapse.global_step.item())
