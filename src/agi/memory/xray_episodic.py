"""Privacy-preserving episodic memory for the AGI architecture.

Stores ``(key, structured_facts)`` per entry — no raw text ever.
The key is a stable vector produced by the frozen foundation
(see :class:`agi.foundation.FrozenFoundation`); the facts are
the structured output of :class:`agi.extraction.FactExtractor`.

Retrieval is cosine similarity over the stored keys, returning
the top-k entries above a configurable threshold. Multi-
timescale storage (working / short / long) is a Phase 3 concern;
Phase 1.0 uses one flat list with timestamps + access counts so
later phases can graft an ageing/promotion policy on top.

Privacy contract — enforced at the class API:
- :meth:`add_entry` stores ``(key.detach().cpu(), facts_copy,
  timestamp, ...)`` and *nothing else*. The fact dict is shallow-
  copied so the caller can't mutate it post-store.
- :class:`EpisodicEntry` has no field for raw text; the unit
  test suite checks for ``raw_text`` / ``original_input`` /
  ``samples`` attributes and asserts they aren't present.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class EpisodicEntry:
    """A single episodic-memory record.

    Attributes:
        key: ``(key_dim,)`` float tensor on CPU. The frozen
            foundation's stable representation of the source
            utterance.
        facts: Structured facts extracted from the utterance.
            E.g. ``{"name": "Francois", "location": "Montréal"}``.
            A dict, never the original text.
        timestamp: When this entry was written.
        access_count: Bumped every time the entry is returned
            by :meth:`XRayEpisodicMemory.retrieve` above the
            threshold. Useful diagnostic for "which facts are
            actually being used".
        creation_session: Integer session id from the memory's
            :attr:`XRayEpisodicMemory.current_session` at write
            time. Helps later phases reason about same-session
            vs across-session retrieval.
    """

    key: torch.Tensor
    facts: dict
    timestamp: datetime
    access_count: int = 0
    creation_session: int = 0


class XRayEpisodicMemory:
    """Flat episodic memory with cosine-similarity retrieval.

    Phase 1.0 uses a Python list. Scale-up phases will swap the
    backend for a vector index (FAISS / hnswlib) and add ageing/
    promotion across timescales.
    """

    def __init__(
        self,
        key_dim: int,
        retrieval_threshold: float = 0.7,
    ) -> None:
        if key_dim <= 0:
            raise ValueError(f"key_dim must be positive, got {key_dim}")
        self.key_dim = int(key_dim)
        self.retrieval_threshold = float(retrieval_threshold)
        self.entries: list[EpisodicEntry] = []
        self.current_session: int = 0

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
        entry = EpisodicEntry(
            key=key.detach().to(torch.float32).cpu(),
            facts=dict(facts),
            timestamp=datetime.now(),
            creation_session=self.current_session,
        )
        self.entries.append(entry)
        return entry

    # ---------- read path ----------

    @torch.no_grad()
    def retrieve(
        self, query_key: torch.Tensor, top_k: int = 3,
    ) -> list[tuple[EpisodicEntry, float]]:
        """Return up to ``top_k`` entries with cosine similarity
        ≥ ``retrieval_threshold``, sorted by similarity desc.

        Bumps ``access_count`` on every returned entry — useful
        signal for "which memories are actually being used" that
        later phases can plug into a forgetting / promotion
        policy.
        """
        if not self.entries:
            return []
        query_n = F.normalize(
            query_key.detach().to(torch.float32).cpu(), dim=-1,
        )
        all_keys = torch.stack([e.key for e in self.entries])
        all_keys_n = F.normalize(all_keys, dim=-1)
        sims = all_keys_n @ query_n  # (N,)

        order = sims.argsort(descending=True).tolist()
        out: list[tuple[EpisodicEntry, float]] = []
        for i in order[:top_k]:
            s = float(sims[i].item())
            if s < self.retrieval_threshold:
                break
            entry = self.entries[i]
            entry.access_count += 1
            out.append((entry, s))
        return out

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

    # ---------- diagnostics ----------

    def __len__(self) -> int:
        return len(self.entries)
