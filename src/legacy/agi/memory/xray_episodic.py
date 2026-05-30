"""Privacy-preserving episodic memory for the AGI architecture.

Stores ``(key, structured_facts)`` per entry — no raw text ever.
The key is a stable vector produced by the frozen foundation
(see :class:`agi.foundation.FrozenFoundation`); the facts are
the structured output of the extractor pipeline.

Retrieval is cosine similarity over the stored keys, returning
the top-k entries above a configurable threshold. Phase 2c bis
adds three lifecycle features on top of the Phase 1.0 baseline:

1. **Precision-decay storage** — each entry carries a
   :class:`PrecisionLevel`. Quantised storage shrinks the
   on-memory footprint over time as entries idle.
2. **Reconsolidation on retrieval** — when an entry is matched
   by :meth:`retrieve`, the entry is re-encoded from its facts
   (which stay at full fidelity in the dict) blended with the
   current query, and promoted one precision level up. Mirrors
   biological reconsolidation: retrieval *modifies* the trace.
3. **Sleep-analog consolidation** — :meth:`maybe_consolidate`
   triggers batch housekeeping (decay / optional merge /
   contradiction-log stub) when the memory has been idle for a
   threshold or the store has grown past a soft cap.

Privacy contract — enforced at the class API:
- :meth:`add_entry` stores ``(key, facts_copy, timestamp, …)``
  and nothing else. The fact dict is shallow-copied so the
  caller can't mutate the stored copy out from under us.
- :class:`EpisodicEntry` has no field for raw text; the unit
  test suite checks for ``raw_text`` / ``original_input`` /
  ``samples`` / ``utterance`` attributes and asserts they
  aren't present.

Public APIs (signature-stable since Phase 1.0):
  * ``XRayEpisodicMemory(key_dim, retrieval_threshold=0.7)``
  * ``add_entry(key, facts) → EpisodicEntry | None``
  * ``retrieve(query_key, top_k=3) → list[(entry, sim)]``
  * ``merge_facts(retrieved) → dict``
  * ``new_session() → int``

Phase 2c bis additions (all opt-in):
  * ``XRayEpisodicMemory(..., foundation=None)`` — when set,
    enables reconsolidation on retrieve (re-encodes the entry's
    facts via the foundation).
  * ``maybe_consolidate()`` — idle / size-triggered light
    consolidation.
  * ``force_consolidate(scope="full")`` — explicit consolidation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional, TYPE_CHECKING

import torch
import torch.nn.functional as F

from .precision import (
    PRECISION_MODIFIER,
    PrecisionLevel,
    RECONSOLIDATION_BLEND_RATIO,
    dequantize_to_float32,
    estimate_storage_bytes,
    quantize_to_level,
    serialize_facts,
)

if TYPE_CHECKING:  # avoid hard transformers dependency at import time
    from agi.foundation import FrozenFoundation


@dataclass
class EpisodicEntry:
    """A single episodic-memory record.

    Phase 1.0 fields:
        key: ``(key_dim,)`` Float32 tensor on CPU. *May be None*
            when ``precision_level != L0`` — use :attr:`effective_key`
            to always get a Float32 representation.
        facts: Structured facts extracted from the utterance.
            E.g. ``{"name": "Francois", "location": "Montréal"}``.
            A dict, never the original text.
        timestamp: When this entry was written.
        access_count: Bumped every time the entry is returned by
            :meth:`XRayEpisodicMemory.retrieve` above the
            threshold. Useful for "which facts are actually being
            used" + drives the importance-weighted decay schedule.
        creation_session: Integer session id from the memory's
            :attr:`XRayEpisodicMemory.current_session` at write
            time.

    Phase 2c bis additions:
        last_accessed: Wall time of the last successful retrieval.
            Defaults to ``timestamp`` at creation. Drives the
            decay schedule via ``(now - last_accessed)``.
        precision_level: Current storage tier (L0 = full Float32,
            L5 = existence trace).
        _stored_key: Quantised storage payload, set when
            ``precision_level != L0``. Format depends on the
            level — see :mod:`agi.memory.precision`.
        embedding_dim: Dimensionality of the original key. Cached
            so dequantisation can trim padding cleanly even after
            ``key`` is freed.
    """

    key: Optional[torch.Tensor]
    facts: dict
    timestamp: datetime
    access_count: int = 0
    creation_session: int = 0
    last_accessed: Optional[datetime] = None
    precision_level: PrecisionLevel = PrecisionLevel.L0
    _stored_key: Any = None
    embedding_dim: int = 0

    def __post_init__(self) -> None:
        # last_accessed defaults to the creation timestamp so a
        # brand-new entry isn't immediately eligible for decay.
        if self.last_accessed is None:
            self.last_accessed = self.timestamp
        # Auto-derive embedding_dim from the key when at L0.
        if self.embedding_dim == 0 and isinstance(self.key, torch.Tensor):
            self.embedding_dim = int(self.key.shape[-1])

    # ---------- precision plumbing ----------

    @property
    def effective_key(self) -> torch.Tensor:
        """Always return a Float32 1-D tensor — dequantised on
        demand when storage is at a non-L0 level.

        At ``L0`` this is a no-copy passthrough of ``self.key``.
        At higher levels it allocates a fresh tensor each call.
        """
        if self.precision_level == PrecisionLevel.L0:
            if isinstance(self.key, torch.Tensor):
                return self.key
            # Defensive: precision_level was set to L0 but key got
            # cleared by a buggy code path — fall back to zeros so
            # cosine retrieval can't crash on a None.
            return torch.zeros(self.embedding_dim, dtype=torch.float32)
        return dequantize_to_float32(
            self._stored_key, self.precision_level, self.embedding_dim,
        )

    def compress_to(
        self,
        target_level: PrecisionLevel,
        source_embedding: Optional[torch.Tensor] = None,
    ) -> None:
        """Move this entry to ``target_level`` in place.

        ``source_embedding`` lets the caller supply a fresh
        Float32 vector to quantise from — used by the
        reconsolidation path when promoting back to a higher
        precision. When ``None``, the entry's current
        :attr:`effective_key` is used (the decay path: the
        already-degraded key is requantised at the lower
        precision).
        """
        if source_embedding is None:
            source_embedding = self.effective_key

        if target_level == PrecisionLevel.L0:
            self.key = source_embedding.detach().to(torch.float32).cpu().clone()
            self._stored_key = None
            self.embedding_dim = int(self.key.shape[-1])
        else:
            self._stored_key = quantize_to_level(source_embedding, target_level)
            # Hold on to the *dimensionality* but drop the Float32
            # tensor — the whole point of quantisation is the size
            # saving.
            self.embedding_dim = int(source_embedding.shape[-1])
            self.key = None

        self.precision_level = target_level

    def storage_bytes(self) -> int:
        """Estimated payload footprint at the current level —
        excludes Python object overhead. Useful for
        memory-saving diagnostics."""
        return estimate_storage_bytes(self.precision_level, self.embedding_dim)


class XRayEpisodicMemory:
    """Flat episodic memory with cosine-similarity retrieval +
    precision-decay storage.

    Phase 1.0 used a plain Python list; that's still the
    backing structure. Scale-up phases will swap in a vector
    index. Phase 2c bis adds the precision lifecycle without
    breaking the existing API.
    """

    IDLE_CONSOLIDATION_THRESHOLD: timedelta = timedelta(seconds=30)
    """Idle interval after which :meth:`maybe_consolidate`
    triggers a *light* consolidation pass (decay only)."""

    SIZE_CONSOLIDATION_THRESHOLD: int = 10_000
    """Soft cap. Past this many entries, :meth:`maybe_consolidate`
    runs a *full* pass (decay + merge + contradiction stub)."""

    def __init__(
        self,
        key_dim: int,
        retrieval_threshold: float = 0.7,
        *,
        foundation: Optional["FrozenFoundation"] = None,
    ) -> None:
        if key_dim <= 0:
            raise ValueError(f"key_dim must be positive, got {key_dim}")
        self.key_dim = int(key_dim)
        self.retrieval_threshold = float(retrieval_threshold)
        self.entries: list[EpisodicEntry] = []
        self.current_session: int = 0
        # Phase 2c bis: lifecycle bookkeeping.
        self.foundation = foundation
        self.last_consolidation: datetime = datetime.now()
        self.last_activity: datetime = datetime.now()
        self.contradiction_log: list[dict] = []

    # ---------- session bookkeeping ----------

    def new_session(self) -> int:
        """Increment the session counter. Useful for marking the
        boundary between two distinct conversations so later
        phases can reason about session locality. Returns the
        new session id."""
        self.current_session += 1
        return self.current_session

    # ---------- write path ----------

    def add_entry(self, key: torch.Tensor, facts: dict) -> Optional[EpisodicEntry]:
        """Add an entry. No-op (returns ``None``) when ``facts``
        is empty — there's no point indexing a record that holds
        no information.

        The key is detached + moved to CPU and the facts dict is
        shallow-copied so the caller can't mutate the stored copy
        out from under us.
        """
        if not facts:
            return None
        if key.shape[-1] != self.key_dim:
            raise ValueError(
                f"key has last-dim {key.shape[-1]}, expected {self.key_dim}"
            )
        now = datetime.now()
        entry = EpisodicEntry(
            key=key.detach().to(torch.float32).cpu(),
            facts=dict(facts),
            timestamp=now,
            last_accessed=now,
            creation_session=self.current_session,
            embedding_dim=int(key.shape[-1]),
        )
        self.entries.append(entry)
        self.last_activity = now
        return entry

    # ---------- read path ----------

    @torch.no_grad()
    def retrieve(
        self, query_key: torch.Tensor, top_k: int = 3,
    ) -> list[tuple[EpisodicEntry, float]]:
        """Return up to ``top_k`` entries with precision-adjusted
        cosine similarity ≥ ``retrieval_threshold``, sorted by
        score desc.

        Score = ``cosine(query, entry.effective_key) *
        PRECISION_MODIFIER[entry.precision_level]``. Entries at
        ``L5`` (existence-only) are skipped entirely — they have
        no usable key.

        Bumps ``access_count`` and refreshes ``last_accessed``
        on every returned entry. When a foundation is wired in,
        each returned entry is also *reconsolidated* — re-encoded
        from its facts blended with the query, and promoted one
        precision level up (see :meth:`_reconsolidate`).
        """
        self.last_activity = datetime.now()

        # Skip L5 entries — no usable key.
        active = [
            (i, e) for i, e in enumerate(self.entries)
            if e.precision_level != PrecisionLevel.L5
        ]
        if not active:
            return []

        query_n = F.normalize(
            query_key.detach().to(torch.float32).cpu(), dim=-1,
        )
        all_keys = torch.stack([e.effective_key for _i, e in active])
        all_keys_n = F.normalize(all_keys, dim=-1)
        raw_sims = (all_keys_n @ query_n).tolist()

        # Apply per-entry precision modifier.
        scored: list[tuple[EpisodicEntry, float]] = []
        for (_i, entry), raw_sim in zip(active, raw_sims):
            modifier = PRECISION_MODIFIER[entry.precision_level]
            scored.append((entry, float(raw_sim) * modifier))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        out: list[tuple[EpisodicEntry, float]] = []
        for entry, score in scored[:top_k]:
            if score < self.retrieval_threshold:
                break
            out.append((entry, score))

        # Bookkeeping + reconsolidation for every returned match.
        for entry, _score in out:
            self._reconsolidate(entry, query_n)
        return out

    @torch.no_grad()
    def _reconsolidate(
        self,
        entry: EpisodicEntry,
        query_embedding: torch.Tensor,
    ) -> None:
        """Refresh an entry on successful retrieval — biological-
        reconsolidation analogue.

        Every successful match:
          * bumps ``access_count`` and ``last_accessed``,
          * (if entry is below L0 AND a foundation is wired in)
            re-encodes the entry from its facts via the
            foundation, blends 10% of the current query into the
            fresh embedding, and promotes the entry one precision
            level up.

        *Mustache analogy.* If the system recognises "François"
        today via memory M, retrieval reconsolidates M with a
        small blend of today's context. Tomorrow when François
        shows up with a mustache, M's key will have drifted
        slightly toward the mustachioed view of him — accumulated
        repeated contexts get folded back into the trace. Over
        many accesses the entry becomes a soft *average* of its
        most-frequent retrieval contexts, while still anchored to
        the original facts via the re-encoding step.

        No-op on the promotion side when:
          * the entry is already at ``L0`` (nothing to promote
            into) — only the freshness bookkeeping runs.
          * no foundation is wired in — we can't re-encode the
            facts without it; only the freshness bookkeeping
            runs.
        """
        entry.access_count += 1
        entry.last_accessed = datetime.now()

        if entry.precision_level == PrecisionLevel.L0:
            return
        if self.foundation is None:
            return

        facts_text = serialize_facts(entry.facts)
        fresh = self.foundation.get_key(facts_text).to(torch.float32).cpu()
        q = query_embedding.to(torch.float32).cpu()
        refreshed = (1.0 - RECONSOLIDATION_BLEND_RATIO) * fresh \
            + RECONSOLIDATION_BLEND_RATIO * q

        next_level = PrecisionLevel(int(entry.precision_level) - 1)
        entry.compress_to(next_level, source_embedding=refreshed)

    def merge_facts(
        self,
        retrieved: list[tuple[EpisodicEntry, float]],
    ) -> dict:
        """Combine facts across retrieved entries. Highest-
        similarity entry wins on conflicting scalar keys
        (name/age/location). List-valued keys (preferences) are
        unioned across entries while preserving order of first
        appearance.

        ``preferences`` may arrive as a string (from the
        LLM-driven extractor — e.g. ``"coffee"``) or as a list
        (from the regex extractor — e.g. ``["coffee", "short
        answers"]``). The naive ``for p in v`` loop iterates a
        string as individual characters, which corrupted the
        merged record in Phase 1.1's demo. Coerce a bare string
        to a single-element list before unioning.
        """
        merged: dict = {}
        for entry, _sim in retrieved:
            for k, v in entry.facts.items():
                if k == "preferences":
                    existing = merged.setdefault("preferences", [])
                    if isinstance(v, str):
                        v_iter: list = [v]
                    else:
                        v_iter = list(v)
                    for p in v_iter:
                        if p not in existing:
                            existing.append(p)
                elif k not in merged:
                    merged[k] = v
        return merged

    # ---------- consolidation triggers ----------

    def maybe_consolidate(self) -> bool:
        """Run consolidation if a trigger is met.

        Triggers:
          * size > :attr:`SIZE_CONSOLIDATION_THRESHOLD` → full pass
          * idle time > :attr:`IDLE_CONSOLIDATION_THRESHOLD` → light pass

        Returns ``True`` when a consolidation actually ran. Safe
        to call frequently (e.g. once per metacog pre-evaluate);
        the size + idle checks are cheap.
        """
        from .consolidation import consolidate

        now = datetime.now()
        if len(self.entries) > self.SIZE_CONSOLIDATION_THRESHOLD:
            consolidate(self, scope="full")
            return True
        if (now - self.last_activity) > self.IDLE_CONSOLIDATION_THRESHOLD:
            consolidate(self, scope="light")
            return True
        return False

    def force_consolidate(self, scope: str = "full") -> None:
        """Run consolidation unconditionally with the chosen
        scope (``"light"`` or ``"full"``). Useful from tests and
        from manual diagnostic flows that don't want to wait for
        a trigger."""
        from .consolidation import consolidate
        consolidate(self, scope=scope)

    # ---------- diagnostics ----------

    def __len__(self) -> int:
        return len(self.entries)

    def precision_distribution(self) -> dict[PrecisionLevel, int]:
        """Count of entries at each precision level — small
        diagnostic for the auto-organisation demo / tests."""
        counts: dict[PrecisionLevel, int] = {lvl: 0 for lvl in PrecisionLevel}
        for entry in self.entries:
            counts[entry.precision_level] += 1
        return counts

    def total_storage_bytes(self) -> int:
        """Sum of :meth:`EpisodicEntry.storage_bytes` across the
        whole memory. Excludes per-entry Python overhead."""
        return sum(e.storage_bytes() for e in self.entries)
