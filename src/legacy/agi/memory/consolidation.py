"""Sleep-analog consolidation for the X-Ray episodic memory.

Two scopes:

- ``light`` — precision decay only. Cheap; triggered when the
  memory has been idle past a short threshold.
- ``full``  — decay + similarity-based merging + contradiction
  detection. More expensive; triggered when the store has grown
  past a soft size cap.

The lifecycle pressure mirrors how the brain consolidates during
sleep: cheap maintenance happens at every quiet moment, the
heavier reorganisation runs less often and on the full store.

Phase 2c bis ships the decay + merge logic + a stub for
contradiction detection — the resolution policy comes later.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from .precision import (
    DECAY_SCHEDULE,
    PrecisionLevel,
)

if TYPE_CHECKING:
    from .xray_episodic import EpisodicEntry, XRayEpisodicMemory


def consolidate(memory: "XRayEpisodicMemory", scope: str = "light") -> None:
    """Run a consolidation pass over ``memory``.

    ``scope``:
      * ``"light"`` — only :func:`apply_precision_decay`.
      * ``"full"``  — decay, :func:`merge_similar_entries`, and
        :func:`detect_contradictions`.
    """
    if scope not in ("light", "full"):
        raise ValueError(f"scope must be 'light' or 'full', got {scope!r}")

    apply_precision_decay(memory)
    if scope == "full":
        merge_similar_entries(memory)
        detect_contradictions(memory)
    memory.last_consolidation = datetime.now()


def apply_precision_decay(memory: "XRayEpisodicMemory") -> None:
    """Demote any entry whose idle time exceeds its level's
    decay threshold (importance-weighted by access count).

    Importance weighting: a high access-count entry decays *more
    slowly*. Concretely, the effective threshold is
    ``base_threshold * (1 + log1p(access_count))``. Capped
    implicitly by the maximum log growth — a million accesses
    still only stretches the threshold by ~15x.

    Walks ``memory.entries`` linearly and calls
    :meth:`EpisodicEntry.compress_to` with no source embedding
    (the decay path requantises the already-degraded key — no
    fact re-encoding here; that's the reconsolidation path's job).
    """
    now = datetime.now()
    for entry in memory.entries:
        if entry.precision_level >= PrecisionLevel.L5:
            continue
        last = entry.last_accessed or entry.timestamp
        idle = now - last
        base = DECAY_SCHEDULE[entry.precision_level]
        importance = 1.0 + math.log1p(entry.access_count)
        threshold = base * importance
        if idle > threshold:
            next_level = PrecisionLevel(int(entry.precision_level) + 1)
            entry.compress_to(next_level)


def merge_similar_entries(
    memory: "XRayEpisodicMemory",
    similarity_threshold: float = 0.95,
) -> int:
    """Greedy O(N²) pass that merges near-duplicate entries.

    For every unordered pair ``(a, b)`` with cosine similarity
    ≥ ``similarity_threshold``:
      * the higher-precision entry survives (lower numeric
        ``precision_level``; ties broken by ``access_count``),
      * the loser's facts are folded into the survivor via the
        existing :meth:`XRayEpisodicMemory.merge_facts` semantics
        (highest-precision wins on scalar conflicts; list-valued
        keys union),
      * the survivor's ``access_count`` is bumped by the loser's
        count (the loser's history of being useful carries
        forward),
      * the loser is removed.

    Returns the number of entries deleted. O(N²) is fine for the
    Phase 2c bis target (≤ 10k entries) — vector-index-backed
    O(N log N) variants are a Phase 3 concern.
    """
    if len(memory.entries) < 2:
        return 0

    # We walk indices in a stable order and mark losers for
    # removal at the end — modifying the list mid-loop confuses
    # the indexing.
    losers: set[int] = set()
    n = len(memory.entries)

    for i in range(n):
        if i in losers:
            continue
        entry_i = memory.entries[i]
        if entry_i.precision_level == PrecisionLevel.L5:
            continue
        key_i = F.normalize(entry_i.effective_key, dim=-1)
        for j in range(i + 1, n):
            if j in losers:
                continue
            entry_j = memory.entries[j]
            if entry_j.precision_level == PrecisionLevel.L5:
                continue
            key_j = F.normalize(entry_j.effective_key, dim=-1)
            sim = float((key_i * key_j).sum().item())
            if sim < similarity_threshold:
                continue
            # Survivor selection: lower precision_level number
            # wins; on tie, higher access_count wins; final tie
            # breaks toward the older entry (i, j ordering).
            keep_i = (
                int(entry_i.precision_level) < int(entry_j.precision_level)
                or (
                    entry_i.precision_level == entry_j.precision_level
                    and entry_i.access_count >= entry_j.access_count
                )
            )
            survivor, loser_idx = (
                (entry_i, j) if keep_i else (entry_j, i)
            )
            other = entry_j if keep_i else entry_i
            survivor.facts = memory.merge_facts(
                [(survivor, 1.0), (other, 1.0)],
            )
            survivor.access_count += other.access_count
            losers.add(loser_idx)
            if not keep_i:
                # i has been swallowed by j — no point comparing
                # i to further entries.
                break

    if not losers:
        return 0
    memory.entries = [
        e for k, e in enumerate(memory.entries) if k not in losers
    ]
    return len(losers)


def detect_contradictions(memory: "XRayEpisodicMemory") -> int:
    """Log entries that share a scalar fact key with conflicting
    values.

    Phase 2c bis stub: we *detect* and *log* into
    ``memory.contradiction_log``, but resolution is deferred to a
    later phase. Each conflict is recorded as
    ``{"key": ..., "values": [v1, v2], "entries": [idx1, idx2]}``.

    Returns the number of new conflicts logged this pass.
    """
    new_count = 0
    # Build a fact_key → list[(entry_idx, value)] index over
    # scalar values only — list / dict values are ambiguous to
    # call "contradictory" without semantic comparison.
    index: dict[str, list[tuple[int, object]]] = {}
    for idx, entry in enumerate(memory.entries):
        if entry.precision_level == PrecisionLevel.L5:
            continue
        for k, v in entry.facts.items():
            if isinstance(v, (list, tuple, dict)):
                continue
            index.setdefault(k, []).append((idx, v))

    for k, pairs in index.items():
        if len(pairs) < 2:
            continue
        # Group identical values; any group of distinct value-
        # buckets ≥ 2 means at least one pair contradicts.
        seen: dict[object, int] = {}
        for idx, v in pairs:
            seen.setdefault(v, idx)
        if len(seen) >= 2:
            memory.contradiction_log.append({
                "key": k,
                "values": list(seen.keys()),
                "entries": list(seen.values()),
            })
            new_count += 1
    return new_count
