"""SynapseLayer — dense Hebbian state container with evidence-based resistance.

Evolves from the Phase-2 v1 (strength only, no resistance, fixed
reward) toward the full Phase-3 design described in DESIGN.md
section 3.2. This iteration adds:

- An ``evidence`` buffer per connection, accumulating the magnitude
  of co-activations seen.
- An evidence-based resistance term in the update rule:
  ``Δs_ij = η · R · a_i · a_j / (1 + β · evidence_ij)``. High-evidence
  synapses resist further change, which is the primary mechanism
  meant to address the Phase-2 failure mode where the synapse
  layer simply tracked the latest task.

Still deferred to follow-up work in Phase 3:

- ``confidence``, ``age``, ``access_count`` state fields.
- Sparse top-k partner selection.

The module remains a state container only: it never produces a
correction vector. Read-out is the modulator's job.

Update rule (DESIGN.md eq. 3.2)::

    Δs_ij = (η / B) · R · Σ_b a_{b,i} · a_{b,j} · 1 / (1 + β · E_ij)
    E_ij  ← E_ij + Σ_b |a_{b,i}| · |a_{b,j}| / B

With ``β = 0`` (the default) the strength update is exactly
identical to Phase 2 v1 — making this change a strict superset.
Evidence still accumulates so callers can inspect it.
"""

from __future__ import annotations

import torch
from torch import nn


class SynapseLayer(nn.Module):
    """Dense Hebbian state with optional evidence-based resistance.

    Buffers (all ``(n, n)`` unless noted, float32 unless noted):

    - ``strengths``: learned Hebbian weights.
    - ``evidence``: accumulated co-activation magnitude.
    - ``global_step``: long scalar; number of ``consolidate`` calls.

    Args:
        n_neurons: Width of the activation vector this layer observes.
        learning_rate: ``η`` in the update rule.
        resistance_beta: ``β``. With ``0`` (the default) the layer
            behaves like Phase-2 v1 — evidence accumulates but does
            not affect updates, so existing experiments remain
            reproducible. Increase β to make high-evidence synapses
            harder to overwrite.
    """

    def __init__(
        self,
        n_neurons: int,
        learning_rate: float = 1e-3,
        resistance_beta: float = 0.0,
    ) -> None:
        super().__init__()
        if n_neurons <= 0:
            raise ValueError(f"n_neurons must be positive, got {n_neurons}")
        if learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {learning_rate}"
            )
        if resistance_beta < 0:
            raise ValueError(
                f"resistance_beta must be >= 0, got {resistance_beta}"
            )
        self.n_neurons = n_neurons
        self.learning_rate = float(learning_rate)
        self.resistance_beta = float(resistance_beta)
        self.register_buffer(
            "strengths", torch.zeros(n_neurons, n_neurons, dtype=torch.float32)
        )
        self.register_buffer(
            "evidence", torch.zeros(n_neurons, n_neurons, dtype=torch.float32)
        )
        self.register_buffer(
            "global_step", torch.zeros((), dtype=torch.long)
        )

    @torch.no_grad()
    def consolidate(
        self, activations: torch.Tensor, reward: float = 1.0
    ) -> None:
        """Apply a single Hebbian update with optional resistance.

        Args:
            activations: ``(B, n_neurons)`` tensor of activations.
                Caller is responsible for detaching from autograd.
            reward: Scalar reward modulating the update magnitude.

        Order of operations is deliberate:

        1. Compute the resistance factor from *current* evidence.
           A synapse's resistance reflects what it has already
           learned, not what the present update is about to teach.
        2. Apply the resisted update to ``strengths``.
        3. Grow ``evidence`` with the new co-activation magnitudes.
        """
        if activations.ndim != 2:
            raise ValueError(
                f"Expected 2-D activations (B, n), got shape "
                f"{tuple(activations.shape)}"
            )
        if activations.shape[1] != self.n_neurons:
            raise ValueError(
                f"Activation dim {activations.shape[1]} does not match "
                f"n_neurons={self.n_neurons}"
            )
        if activations.shape[0] == 0:
            return

        a = activations.detach().to(self.strengths.dtype)
        batch_size = a.shape[0]
        raw_outer = a.transpose(-1, -2) @ a / batch_size
        abs_outer = a.abs().transpose(-1, -2) @ a.abs() / batch_size

        if self.resistance_beta == 0.0:
            # Fast path identical to v1; preserves the Phase-2
            # numerical behaviour bit-for-bit.
            self.strengths.add_(
                raw_outer, alpha=self.learning_rate * float(reward)
            )
        else:
            resistance = 1.0 / (1.0 + self.resistance_beta * self.evidence)
            self.strengths.add_(
                raw_outer * resistance,
                alpha=self.learning_rate * float(reward),
            )

        self.evidence.add_(abs_outer)
        self.global_step.add_(1)

    def reset(self) -> None:
        """Zero all buffers. Used in tests and ablations."""
        with torch.no_grad():
            self.strengths.zero_()
            self.evidence.zero_()
            self.global_step.zero_()

    def extra_repr(self) -> str:
        return (
            f"n_neurons={self.n_neurons}, "
            f"learning_rate={self.learning_rate}, "
            f"resistance_beta={self.resistance_beta}, "
            f"global_step={int(self.global_step.item())}"
        )
