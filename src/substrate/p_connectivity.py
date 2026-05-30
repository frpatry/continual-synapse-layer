"""Dynamic sparse P-P connectivity.

Unlike the fixed-shape :class:`ConnectivityMatrix` between N entities,
the P-level connectivity is genuinely dynamic — P entities emerge and
dissolve at runtime, so the connection set must grow and shrink with
them. We store the live connections in a Python dict keyed by the
canonical pair ``(min_p_id, max_p_id)``.

Phase 2b semantics:
* Connections form (insert) when two co-active P entities co-fire and
  the covariance Hebbian signal at P level pushes the weight above
  zero — see :mod:`substrate.p_plasticity`.
* Connections strengthen / weaken on every step via the same Hebbian
  / age-modulated-decay rule as N-N.
* A connection is *dropped* from the dict the moment its weight is
  clipped to zero. Keeping zero-weight entries would let the dict
  grow without bound across very long runs.
* When a P entity dissolves, every connection touching its id is
  removed in :meth:`remove_entity` so the bookkeeping stays consistent.
"""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, Tuple


class PConnectivity:
    """Sparse, symmetric, dynamic weight store for P-P connections."""

    def __init__(self) -> None:
        # (lower_id, higher_id) → weight
        self._weights: Dict[Tuple[int, int], float] = {}

    # ---------- key helpers ----------

    @staticmethod
    def _canonical_key(p_id_a: int, p_id_b: int) -> Tuple[int, int]:
        """Canonical key so (a, b) and (b, a) collapse to the same entry."""
        return (min(p_id_a, p_id_b), max(p_id_a, p_id_b))

    # ---------- mutation ----------

    def add_connection(
        self, p_id_a: int, p_id_b: int, weight: float = 0.0,
    ) -> None:
        """Set (or replace) the weight of the (a, b) connection.

        No-op if ``a == b`` (no self-connections).
        """
        if p_id_a == p_id_b:
            return
        key = self._canonical_key(p_id_a, p_id_b)
        self._weights[key] = float(weight)

    def update_weight(
        self, p_id_a: int, p_id_b: int, delta: float,
    ) -> None:
        """Add ``delta`` to the (a, b) weight.

        * Creates the entry if it doesn't exist and the resulting weight
          is strictly positive.
        * Clips the weight to non-negative (P-P weights are excitatory
          only in Phase 2b).
        * Drops the entry from the dict the moment it would become zero.
        * No-op for self-connections.
        """
        if p_id_a == p_id_b:
            return
        key = self._canonical_key(p_id_a, p_id_b)
        current = self._weights.get(key, 0.0)
        new_weight = max(0.0, current + float(delta))
        if new_weight > 0.0:
            self._weights[key] = new_weight
        else:
            # Drop zero entries so the dict can't grow without bound.
            self._weights.pop(key, None)

    def remove_entity(self, p_id: int) -> None:
        """Remove every connection touching ``p_id``.

        Called when a PEntity dissolves so the connectivity store
        doesn't accumulate references to dead P ids.
        """
        to_remove = [k for k in self._weights if p_id in k]
        for k in to_remove:
            del self._weights[k]

    # ---------- queries ----------

    def get_weight(self, p_id_a: int, p_id_b: int) -> float:
        """Return the weight of (a, b), or 0.0 if no connection exists."""
        if p_id_a == p_id_b:
            return 0.0
        return self._weights.get(self._canonical_key(p_id_a, p_id_b), 0.0)

    def neighbors_of(self, p_id: int) -> Dict[int, float]:
        """Return ``{neighbor_id: weight}`` for every connection of ``p_id``."""
        result: Dict[int, float] = {}
        for (a, b), w in self._weights.items():
            if a == p_id:
                result[b] = w
            elif b == p_id:
                result[a] = w
        return result

    def connection_count(self) -> int:
        return len(self._weights)

    def all_pairs(self) -> Iterator[Tuple[int, int, float]]:
        """Iterate over ``(p_id_a, p_id_b, weight)`` triples."""
        for (a, b), w in self._weights.items():
            yield (a, b, w)
