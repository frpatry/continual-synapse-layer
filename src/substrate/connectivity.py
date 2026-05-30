"""Sparse random connectivity between N entities.

Per decision D1 in the spec: ``k`` random connections per N
(default ``k=50`` at ``n=500``), brain-plausible at this scale
(≈10 % of the substrate as direct neighbours).

The matrix here stores **implicit** structural weights between N
pairs — they are not yet reified as P entities (that's Phase 2).
For Phase 1 they are just floats indexed by ``(source, target)``.

Convention: ``W[i, j]`` is the weight on the connection from
``i`` (source) to ``j`` (target). Activation propagation reads
``W.T @ a`` to compute the incoming sum at each target.
"""

from __future__ import annotations

import numpy as np


class ConnectivityMatrix:
    """Sparse random connectivity with structural weights.

    The connectivity *topology* (which pairs are connected) is
    fixed at construction time — Phase 1 does not allow new
    synapses to grow or existing ones to disappear; only the
    *weights* on existing connections change via plasticity.
    Diagonal is always zero (no self-connections).
    """

    def __init__(
        self,
        n_neurons: int,
        k: int = 50,
        seed: int = 42,
    ) -> None:
        if n_neurons <= 0:
            raise ValueError(f"n_neurons must be positive, got {n_neurons}")
        if not 0 < k < n_neurons:
            raise ValueError(
                f"k must be in (0, n_neurons); got k={k}, n_neurons={n_neurons}"
            )
        self.n = int(n_neurons)
        self.k = int(k)
        self.rng = np.random.default_rng(seed)

        # Boolean mask of valid connections.
        self.mask = np.zeros((self.n, self.n), dtype=bool)
        for i in range(self.n):
            # k random targets, never self.
            candidates = np.arange(self.n)
            candidates = candidates[candidates != i]
            chosen = self.rng.choice(candidates, size=self.k, replace=False)
            self.mask[i, chosen] = True

        # Initial weights: small positive, only on mask.
        self.W = np.zeros((self.n, self.n), dtype=np.float32)
        self.W[self.mask] = self.rng.uniform(
            0.0, 0.1, size=int(self.mask.sum()),
        ).astype(np.float32)

    def get_weights(self) -> np.ndarray:
        """Return the current ``(n, n)`` weight matrix
        (modification reflects on the live array)."""
        return self.W

    def update_weights(self, delta: np.ndarray) -> None:
        """Apply ``delta`` to the weights, respecting the mask.

        Weights are clipped to ``[0, ∞)`` after the update — the
        theory keeps structural weights non-negative.
        """
        if delta.shape != self.W.shape:
            raise ValueError(
                f"delta shape {delta.shape} doesn't match W shape {self.W.shape}"
            )
        self.W += delta.astype(np.float32) * self.mask
        np.clip(self.W, 0.0, None, out=self.W)

    def connection_count(self) -> int:
        """Number of active connections in the topology."""
        return int(self.mask.sum())
