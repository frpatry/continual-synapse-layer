"""External reward component.

The simplest of the three reward signals in DESIGN.md §3.4. An
:class:`ExternalReward` stores a single scalar value that the
training loop can update between batches. The "external" framing
covers everything that arrives from outside the synapse layer:
explicit user feedback, task-success metrics, scheduled curricula,
etc.

In Phase 3 v1 we do not yet wire any specific external signal into
the experiment scripts. The class exists so the reward mixer can
treat all three components uniformly and so that later phases can
plug in real signals without touching the mixer.
"""

from __future__ import annotations

import torch


class ExternalReward:
    """Pass-through reward source.

    Args:
        default: Initial reward value. Defaults to ``1.0`` so the
            mixer behaves the same as the Phase-2 v1 setup when no
            other reward sources are configured.
    """

    def __init__(self, default: float = 1.0) -> None:
        self._value = float(default)

    @property
    def value(self) -> float:
        return self._value

    def set(self, value: float) -> None:
        """Update the value the next call will return."""
        self._value = float(value)

    def __call__(self, activations: torch.Tensor | None = None) -> float:
        # `activations` is accepted but ignored — kept in the signature
        # so the mixer can call all components uniformly.
        return self._value
