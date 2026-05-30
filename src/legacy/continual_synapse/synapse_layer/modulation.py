"""Output modulation: read synapse state into a correction vector.

Given activations ``A`` of shape ``(B, n)`` from the hooked layer
and synapse strengths ``S`` of shape ``(n, n)``, the modulator
produces a correction tensor of shape ``(B, n)`` that is added to
the activations before the final classifier head.

The correction is a linear transform of the activations through
the strength matrix, scaled by a learnable scalar gate::

    correction = gate * (A @ S)

Two properties matter for Phase 2:

1. **Near-zero at init.** ``gate`` initialises to ``0`` and ``S``
   initialises to zero in :class:`SynapseLayer`, so the correction
   is exactly zero on the very first forward pass and the base
   model's behaviour is preserved verbatim.

2. **Gate is learnable, strengths are not.** The gate participates
   in autograd and is updated by the optimizer like any other model
   parameter. Strengths are a buffer, updated only by the Hebbian
   rule. This factoring keeps the gradient-trained "trust in the
   synapse signal" knob separate from the Hebbian-driven "what the
   synapses have learned" content.
"""

from __future__ import annotations

import torch
from torch import nn


class SynapseModulation(nn.Module):
    """Compute a gated linear correction from synapse state.

    Args:
        init_gate: Initial value of the learnable scalar that scales
            the correction. Defaults to ``0.0`` so the correction
            has no effect at the very first forward pass.
    """

    def __init__(self, init_gate: float = 0.0) -> None:
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(float(init_gate)))

    def forward(
        self, activations: torch.Tensor, strengths: torch.Tensor
    ) -> torch.Tensor:
        """Return ``gate * (activations @ strengths)``.

        Args:
            activations: ``(B, n)`` activations from the hooked
                layer of the base model.
            strengths: ``(n, n)`` synapse strength matrix from
                :class:`SynapseLayer`. Must live on the same device
                as ``activations``.
        """
        if activations.ndim != 2:
            raise ValueError(
                f"activations must be 2-D (B, n), got shape "
                f"{tuple(activations.shape)}"
            )
        if strengths.ndim != 2 or strengths.shape[0] != strengths.shape[1]:
            raise ValueError(
                f"strengths must be square 2-D, got shape "
                f"{tuple(strengths.shape)}"
            )
        if activations.shape[1] != strengths.shape[0]:
            raise ValueError(
                f"activation dim {activations.shape[1]} does not match "
                f"strength dim {strengths.shape[0]}"
            )
        return self.gate * (activations @ strengths)
