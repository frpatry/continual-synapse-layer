"""SEntity — schema (third ontological tier; grouping of P entities).

Phase 6i: after 10 phases of parameter exploration at M=5 ceiling'd at
4/4 (Phase 6c), the data empirically motivated adding a third
ontological tier above P. S entities group P entities that frequently
co-activate, providing:

- A higher-level "category" that can be cued, recalled, or composed
  with other S in future phases.
- A top-down feedback channel (S → P → N) that lets schema-level
  patterns drive pattern completion at the N level via the P layer.
- Recursive use of the same emergence machinery (PassTracker on P
  pairs, similar to PassTracker on N pairs that birthed P).

Design (per spec / discussion):
- Q1=a: S contains ONLY P (clean hierarchy; S does not contain N
  directly; future S-S connections are a separate concern).
- Q2=a: S emerges via pairwise PassTracker recursive at P level.
- Q2a=i: S grows via threshold (new P added when it co-activates
  sufficiently with an existing S).
- Q3=b: S has own activation bottom-up + S → P feedback.
- Q4=d: k-WTA at S level is adaptive (sparsity ~0.20 with min/max
  bounds, e.g. [1, 3]) so a small live S pool can still have
  meaningful selection.
- Q5=a: no special recall protocol — S participates transparently
  during recall via its existing S → P → N feedback cascade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set


@dataclass
class SEntity:
    """Schema entity grouping a set of P entity IDs.

    Attributes:
        id: stable integer identifier assigned by the substrate.
        contents: set of P entity IDs that this S contains. Starts
            non-empty (S emerges from a P pair). Grows monotonically
            via the growth mechanism; shrinks only when a contained
            P entity dissolves (handled by the substrate's cleanup).
        activation: current activation in ``[0, 1]``. Derived each
            step as ``alpha_p_to_s · mean(P.activation for P in contents)``,
            then thresholded + k-WTA'd at S level.
        weight: structural weight in ``[0, ∞)``. Initialised to 1.0
            on emergence. Grows Hebbian-like with own activation, decays
            age-modulated, dissolves below ``s_viability_threshold``.
        age_at_emergence: substrate ``system_age`` at emergence time,
            kept for diagnostics.
    """

    id: int
    contents: Set[int] = field(default_factory=set)
    activation: float = 0.0
    weight: float = 1.0
    age_at_emergence: float = 0.0

    def __post_init__(self) -> None:
        # Defensive: accept iterables but always store as set.
        if not isinstance(self.contents, set):
            self.contents = set(self.contents)

    def reset_activation(self) -> None:
        self.activation = 0.0

    def add_member(self, p_id: int) -> None:
        """Add a P entity id to this schema."""
        self.contents.add(int(p_id))

    def remove_member(self, p_id: int) -> None:
        """Remove a P entity id (e.g. when that P has dissolved)."""
        self.contents.discard(int(p_id))

    def size(self) -> int:
        return len(self.contents)
