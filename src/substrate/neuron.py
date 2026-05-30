"""N — the atomic neuron entity (THEORY.md §2.1, §2.3).

Phase 1 keeps the entity itself deliberately thin: numpy arrays
in :class:`Substrate` carry the per-neuron state for vectorised
dynamics. :class:`N` exists to:

- Pin the conceptual identity (id) of each neuron — referenced by
  later phases when P / S / C entities point back at the N they
  emerged from.
- Hold the *structural property* (``weight``) that THEORY §2.3
  describes — accumulated excitability, ``[0, ∞)``, distinct from
  the instantaneous ``activation`` ∈ ``[0, 1]``.

The activation field on the dataclass is informational only —
the canonical per-step activation lives in
:attr:`Substrate.activations`. Tests assert the contract.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class N:
    """Atomic neuron entity.

    Attributes:
        id: unique identifier inside its substrate (0..n_neurons-1).
        activation: instantaneous state in ``[0, 1]``. Mirrors the
            value held in :attr:`Substrate.activations[id]`; kept
            here for inspection convenience.
        weight: accumulated structural property (intrinsic
            excitability), ``[0, ∞)``. Distinct from connection
            weights between neurons — those live in
            :class:`ConnectivityMatrix`.
    """

    id: int
    activation: float = 0.0
    weight: float = 1.0

    def reset_activation(self) -> None:
        """Set ``activation`` back to 0 (does not touch
        ``weight``)."""
        self.activation = 0.0
