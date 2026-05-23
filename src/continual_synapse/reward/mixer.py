"""Reward mixer — combines components with a developmental trajectory.

Implements DESIGN.md §3.4::

    R(t) = α(t) · R_external + (1 - α(t)) · (w_c · R_consistency + w_s · R_surprise)
    α(t) = 1 / (1 + γ · validated_evidence(t))

In Phase 3 v1 we use the call count as a proxy for the
"validated_evidence" term from DESIGN.md. The full definition
("number of times external reward confirmed an internal signal")
needs joint stats over multiple reward sources and a clear notion
of "confirmation", neither of which is settled yet. The call-count
proxy preserves the qualitative behaviour the spec calls for —
``α`` is high early on (external dominates) and decreases as
experience accumulates — and the mixer's design lets us swap in a
better metric later without touching callers.

Edge cases the implementation handles intentionally:

- If only external is configured, the mixer returns the external
  value verbatim (no decay) — without internal signals there is
  nothing for ``(1 - α)`` to weight, and decaying the only
  available signal to zero would be obviously wrong.
- If only internal signals are configured, the mixer returns
  their weighted sum and ignores ``α``.
- If both are configured, the literal formula above applies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from continual_synapse.reward.consistency import ConsistencyReward
from continual_synapse.reward.external import ExternalReward
from continual_synapse.reward.surprise import SurpriseReward


@dataclass
class RewardMixer:
    """Compose reward components into a single scalar.

    Attributes:
        external: Optional pass-through component.
        consistency: Optional cosine-similarity-with-EMA component.
        surprise: Optional online-predictor-error component.
        gamma: ``γ`` in the developmental-trajectory formula. ``0``
            disables decay (``α`` stays at 1).
        w_consistency: Internal-signal weight for consistency.
        w_surprise: Internal-signal weight for surprise. Defaults to
            ``0`` so a "consistency-only" mixer is the natural one-
            line configuration.
    """

    external: ExternalReward | None = None
    consistency: ConsistencyReward | None = None
    surprise: SurpriseReward | None = None
    gamma: float = 0.001
    w_consistency: float = 1.0
    w_surprise: float = 0.0
    _step: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            self.external is None
            and self.consistency is None
            and self.surprise is None
        ):
            raise ValueError(
                "RewardMixer needs at least one component "
                "(external, consistency, or surprise)"
            )
        if self.gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {self.gamma}")

    @property
    def step(self) -> int:
        return self._step

    @property
    def alpha(self) -> float:
        """Current external/internal blend factor."""
        return 1.0 / (1.0 + self.gamma * self._step)

    def reset(self) -> None:
        """Reset step counter and per-component history."""
        self._step = 0
        if self.consistency is not None:
            self.consistency.reset()
        if self.surprise is not None:
            self.surprise.reset()

    def __call__(self, activations: torch.Tensor) -> float:
        r_ext = self.external(activations) if self.external is not None else None

        r_int: float | None
        if self.consistency is None and self.surprise is None:
            r_int = None
        else:
            r_int = 0.0
            if self.consistency is not None:
                r_int += self.w_consistency * self.consistency(activations)
            if self.surprise is not None:
                r_int += self.w_surprise * self.surprise(activations)

        if r_ext is not None and r_int is not None:
            alpha = self.alpha
            reward = alpha * r_ext + (1.0 - alpha) * r_int
        elif r_ext is not None:
            reward = r_ext
        else:
            assert r_int is not None  # at least one source must exist
            reward = r_int

        self._step += 1
        return float(reward)
