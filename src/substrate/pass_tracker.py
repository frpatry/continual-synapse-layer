"""PassTracker — emergence candidacy bookkeeping for N pairs.

Per THEORY.md §3.3 (emergence mechanism), a P entity emerges from a
pair of N entities once that pair has demonstrated sustained
co-activation across distinct events. Two ingredients:

* ``candidacy_strength[i, j]`` — accumulates with current co-activation,
  decays per step. A smooth low-pass filter on the co-activation
  signal.
* ``validation_passes[i, j]`` — count of *distinct* sessions where
  the candidacy rose into the "in-pass" regime (Q2 = A spacing-effect
  resolution). Two long bursts count as 2 passes, not 1 — a pair has
  to "be reminded" across separated events to crystallise.

We use **hysteresis** for the in-pass state — rising past
``theta_quiet_high`` enters the pass; only falling past
``theta_quiet_low`` (with ``low < high``) exits. This prevents a
candidacy oscillating around a single threshold from being miscounted
as many distinct passes.

Per implementation choice γ (preferred), the *final* emergence
criterion is structural: the connectivity weight ``W[i, j]`` must
exceed ``theta_emergence``. The pass count is an additional spacing
gate on top of that.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class PassTracker:
    """Sparse-mask-aware tracker of pass candidacy for every (i, j).

    Memory: arrays are dense ``(n, n)`` for vectorised updates, but
    only entries inside the connectivity mask carry signal — others
    are forced to zero on every update.
    """

    def __init__(
        self,
        connectivity_mask: np.ndarray,
        boost_factor: float = 0.1,
        decay_factor: float = 0.95,
        theta_quiet_high: float = 0.1,
        theta_quiet_low: float = 0.05,
    ) -> None:
        if connectivity_mask.ndim != 2 or (
            connectivity_mask.shape[0] != connectivity_mask.shape[1]
        ):
            raise ValueError("connectivity_mask must be a square 2D array")
        if not (0.0 <= theta_quiet_low < theta_quiet_high):
            raise ValueError(
                "require 0 <= theta_quiet_low < theta_quiet_high "
                "(hysteresis needs a strict gap)"
            )

        self.mask = connectivity_mask
        self.n = int(connectivity_mask.shape[0])
        self.boost = float(boost_factor)
        self.decay = float(decay_factor)
        self.th_high = float(theta_quiet_high)
        self.th_low = float(theta_quiet_low)

        self.candidacy_strength: np.ndarray = np.zeros(
            (self.n, self.n), dtype=np.float32,
        )
        self.validation_passes: np.ndarray = np.zeros(
            (self.n, self.n), dtype=np.int32,
        )
        self.in_pass: np.ndarray = np.zeros(
            (self.n, self.n), dtype=bool,
        )

    # ---------- per-step update ----------

    def update(self, neuron_activations: np.ndarray) -> None:
        """Advance candidacy + pass state by one substrate step.

        Order:
          1. compute masked co-activation matrix
          2. apply decay + boost to candidacy
          3. detect rising / falling hysteresis transitions
          4. increment counters on rising edges
          5. update ``in_pass`` state
        """
        coact = np.outer(neuron_activations, neuron_activations).astype(np.float32)
        coact = coact * self.mask
        # No self-pairs.
        np.fill_diagonal(coact, 0.0)

        self.candidacy_strength = (
            self.candidacy_strength * self.decay + self.boost * coact
        )

        # Hysteresis: rise above ``th_high`` to enter, fall below
        # ``th_low`` to exit. Because ``th_low < th_high``, the two
        # conditions are mutually exclusive at any instant, so the
        # update order below is safe.
        rising_above_high = (~self.in_pass) & (self.candidacy_strength > self.th_high)
        falling_below_low = self.in_pass & (self.candidacy_strength < self.th_low)

        # Count distinct entries into the pass regime.
        self.validation_passes += rising_above_high.astype(np.int32)

        # Update in-pass state. Equivalent to:
        #   self.in_pass = (self.in_pass | rising) & ~falling
        self.in_pass = (self.in_pass | rising_above_high) & ~falling_below_low

    # ---------- emergence query ----------

    def find_emergence_candidates(
        self,
        connectivity_W: np.ndarray,
        theta_emergence: float = 0.5,
        n_min_passes: int = 3,
        existing_p_pairs: Optional[set] = None,
    ) -> list[tuple[int, int]]:
        """Return pairs ``(i, j)`` ready to emerge into a P entity.

        Joint criterion:

        * ``connectivity_W[i, j] > theta_emergence`` — structural
          weight has crossed the γ threshold;
        * ``validation_passes[i, j] >= n_min_passes`` — pair has
          been validated across enough distinct events;
        * pair is in the connectivity mask (no off-mask pairs);
        * pair is not already represented by an existing P.

        Returned pairs are canonical (``min, max``) and deduplicated:
        if both ``(i, j)`` and ``(j, i)`` would qualify (symmetric
        portion of the mask), the single canonical entry is returned.
        """
        weight_high = connectivity_W > theta_emergence
        passes_sufficient = self.validation_passes >= n_min_passes
        candidates_mask = weight_high & passes_sufficient & self.mask

        i_indices, j_indices = np.where(candidates_mask)

        pairs: set[tuple[int, int]] = set()
        for i, j in zip(i_indices, j_indices):
            ii, jj = int(i), int(j)
            if ii == jj:
                continue
            pair = (min(ii, jj), max(ii, jj))
            if existing_p_pairs is not None and pair in existing_p_pairs:
                continue
            pairs.add(pair)

        return sorted(pairs)
