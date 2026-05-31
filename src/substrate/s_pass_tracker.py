"""SPassTracker — recursive emergence machinery at the P level.

Same conceptual shape as :class:`PassTracker` (which tracks N-N
co-activation candidacy for P emergence), but applied one tier up:
tracks P-P co-activation candidacy for S emergence and tracks
P-S co-activation candidacy for S *growth* (a P joining an existing S).

Unlike the N-level tracker which uses a dense fixed mask, P entities
emerge/dissolve dynamically — so this tracker is sparse / dict-based.
Two parallel sets of bookkeeping:

* ``pair_candidacy[(p_a, p_b)] → float``        for new-S emergence
* ``p_to_s_candidacy[(p_id, s_id)] → float``    for adding P to existing S

Each has a corresponding ``*_passes`` counter and ``*_in_pass`` boolean
that implements hysteresis (rising past ``theta_high`` enters the pass,
falling past ``theta_low`` exits — preventing chatter from counting as
multiple passes).

Cleanup hooks (``cleanup_dissolved_p`` / ``cleanup_dissolved_s``)
remove tracker entries referencing dissolved entities so the dicts
don't accumulate dead references over long runs.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple


class SPassTracker:
    """Sparse P-level co-activation tracker for S emergence + growth."""

    def __init__(
        self,
        boost: float = 0.1,
        decay: float = 0.95,
        theta_high: float = 0.1,
        theta_low: float = 0.05,
        min_active_p: float = 0.1,
        min_active_s: float = 0.1,
    ) -> None:
        if not (0.0 <= theta_low < theta_high):
            raise ValueError(
                "require 0 <= theta_low < theta_high (hysteresis gap)"
            )
        self.boost = float(boost)
        self.decay = float(decay)
        self.th_high = float(theta_high)
        self.th_low = float(theta_low)
        self.min_active_p = float(min_active_p)
        self.min_active_s = float(min_active_s)

        # For S EMERGENCE: pairs of P entities that co-fire.
        # Key: canonical (min_p_id, max_p_id).
        self.pair_candidacy: Dict[Tuple[int, int], float] = defaultdict(float)
        self.pair_passes: Dict[Tuple[int, int], int] = defaultdict(int)
        self.pair_in_pass: Dict[Tuple[int, int], bool] = defaultdict(bool)

        # For S GROWTH: P entities that co-fire with an existing S.
        # Key: (p_id, s_id) — NOT canonical (they're different namespaces).
        self.p_to_s_candidacy: Dict[Tuple[int, int], float] = defaultdict(float)
        self.p_to_s_passes: Dict[Tuple[int, int], int] = defaultdict(int)
        self.p_to_s_in_pass: Dict[Tuple[int, int], bool] = defaultdict(bool)

    @staticmethod
    def _canonical_pair(a: int, b: int) -> Tuple[int, int]:
        return (min(a, b), max(a, b))

    # ---------- per-step update ----------

    def update(self, p_entities: Dict, s_entities: Dict) -> None:
        """One step: decay both candidacy dicts, then boost on co-activation
        of currently-active P-P pairs and active-P × active-S pairs."""

        # ---- Decay + cleanup pair candidacy ----
        for key in list(self.pair_candidacy.keys()):
            self.pair_candidacy[key] *= self.decay
            if self.pair_candidacy[key] < 1e-6:
                self.pair_candidacy.pop(key, None)
                self.pair_passes.pop(key, None)
                self.pair_in_pass.pop(key, None)

        # ---- Decay + cleanup P-to-S candidacy ----
        for key in list(self.p_to_s_candidacy.keys()):
            self.p_to_s_candidacy[key] *= self.decay
            if self.p_to_s_candidacy[key] < 1e-6:
                self.p_to_s_candidacy.pop(key, None)
                self.p_to_s_passes.pop(key, None)
                self.p_to_s_in_pass.pop(key, None)

        # ---- Boost on current co-activations ----
        active_p_ids = [
            pid for pid, p in p_entities.items()
            if p.activation > self.min_active_p
        ]
        # Pair boost: every co-active P pair.
        for i, pid_a in enumerate(active_p_ids):
            a_act = p_entities[pid_a].activation
            for pid_b in active_p_ids[i + 1:]:
                b_act = p_entities[pid_b].activation
                key = self._canonical_pair(int(pid_a), int(pid_b))
                self.pair_candidacy[key] += self.boost * (a_act * b_act)

        # P-to-S boost: every active P × active S where P not already in S.
        active_s_ids = [
            sid for sid, s in s_entities.items()
            if s.activation > self.min_active_s
        ]
        for pid in active_p_ids:
            p_act = p_entities[pid].activation
            for sid in active_s_ids:
                s = s_entities[sid]
                if int(pid) in s.contents:
                    continue
                s_act = s.activation
                key = (int(pid), int(sid))
                self.p_to_s_candidacy[key] += self.boost * (p_act * s_act)

        # ---- Hysteresis pass detection on both channels ----
        for key, strength in self.pair_candidacy.items():
            if not self.pair_in_pass[key] and strength > self.th_high:
                self.pair_in_pass[key] = True
                self.pair_passes[key] += 1
            elif self.pair_in_pass[key] and strength < self.th_low:
                self.pair_in_pass[key] = False

        for key, strength in self.p_to_s_candidacy.items():
            if not self.p_to_s_in_pass[key] and strength > self.th_high:
                self.p_to_s_in_pass[key] = True
                self.p_to_s_passes[key] += 1
            elif self.p_to_s_in_pass[key] and strength < self.th_low:
                self.p_to_s_in_pass[key] = False

    # ---------- candidate queries ----------

    def find_s_emergence_candidates(
        self,
        theta_s_emergence: float,
        n_min_passes: int,
        existing_s_pairs: Optional[Set[Tuple[int, int]]] = None,
    ) -> List[Tuple[int, int]]:
        """Return P-P pairs ready to crystallise into a new S entity.

        Joint criterion: candidacy strength above ``theta_s_emergence``
        AND ``n_min_passes`` distinct entries into the pass regime.
        Skips pairs already represented by an existing S.
        """
        candidates: List[Tuple[int, int]] = []
        for (p_a, p_b), strength in self.pair_candidacy.items():
            if strength <= theta_s_emergence:
                continue
            if self.pair_passes.get((p_a, p_b), 0) < n_min_passes:
                continue
            if existing_s_pairs and (p_a, p_b) in existing_s_pairs:
                continue
            candidates.append((p_a, p_b))
        return sorted(candidates)

    def find_s_growth_candidates(
        self,
        theta_growth: float,
        n_min_passes: int,
    ) -> List[Tuple[int, int]]:
        """Return (P, S) pairs where P has been co-active enough with S
        to justify joining S's contents."""
        candidates: List[Tuple[int, int]] = []
        for (p_id, s_id), strength in self.p_to_s_candidacy.items():
            if strength <= theta_growth:
                continue
            if self.p_to_s_passes.get((p_id, s_id), 0) < n_min_passes:
                continue
            candidates.append((p_id, s_id))
        return sorted(candidates)

    # ---------- cleanup on dissolution ----------

    def cleanup_dissolved_p(self, dissolved_p_ids: Iterable[int]) -> None:
        """Remove tracker entries referencing any dissolved P id."""
        dead = {int(x) for x in dissolved_p_ids}
        if not dead:
            return
        # Pair entries with a dissolved P on either side.
        for key in [k for k in self.pair_candidacy
                    if k[0] in dead or k[1] in dead]:
            self.pair_candidacy.pop(key, None)
            self.pair_passes.pop(key, None)
            self.pair_in_pass.pop(key, None)
        # P-to-S entries with a dissolved P.
        for key in [k for k in self.p_to_s_candidacy if k[0] in dead]:
            self.p_to_s_candidacy.pop(key, None)
            self.p_to_s_passes.pop(key, None)
            self.p_to_s_in_pass.pop(key, None)

    def cleanup_dissolved_s(self, dissolved_s_ids: Iterable[int]) -> None:
        """Remove P-to-S entries pointing at any dissolved S id."""
        dead = {int(x) for x in dissolved_s_ids}
        if not dead:
            return
        for key in [k for k in self.p_to_s_candidacy if k[1] in dead]:
            self.p_to_s_candidacy.pop(key, None)
            self.p_to_s_passes.pop(key, None)
            self.p_to_s_in_pass.pop(key, None)
