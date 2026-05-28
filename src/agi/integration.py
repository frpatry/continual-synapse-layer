"""Unified AGI system: foundation + memory + fact extraction.

Phase 1.0 pipeline:

  observe(text) → extract structured facts → store (key, facts)
  respond(query) → retrieve relevant entries → merge facts →
                   augment prompt → generate via foundation

Phase 1.1 adds:

  - :class:`ConversationManager` for multi-turn coherence within
    a session (history threaded into the response prompt).
  - :class:`LLMFactExtractor` as the default extractor, with the
    regex :class:`FactExtractor` retained as a fallback when the
    LLM returns nothing parseable.
  - Session lifecycle: :meth:`new_session` clears the in-session
    conversation history (memory persists across sessions).
  - On-disk persistence: :meth:`save` / :meth:`load` serialise
    only the privacy-preserving fields of each memory entry
    (key + structured facts + bookkeeping). Raw text and
    conversation history are deliberately never written to disk.

No reward signal, no parameter updates, no separate reasoning
module — those come in Phases 2/4.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import torch

from .conversation import ConversationManager
from .extraction import FactExtractor
from .foundation import FrozenFoundation
from .llm_extraction import LLMFactExtractor
from .memory.xray_episodic import EpisodicEntry, XRayEpisodicMemory


class AGISystem:
    """Compose foundation + episodic memory + extractors + history.

    The class deliberately takes a *constructed* foundation
    rather than constructing one internally — this lets tests
    pass a mock (``MockFoundation``) without touching Qwen.

    ``use_llm_extraction`` toggles whether the foundation is
    prompted for fact extraction. When ``True`` (default), the
    LLM extractor runs first and the regex extractor runs as a
    fallback if the LLM returns no parseable facts. When
    ``False``, only the regex extractor runs — useful when the
    foundation can't yet handle JSON-formatted output (e.g. tiny
    mocks in unit tests, or non-instruct models).
    """

    def __init__(
        self,
        foundation: FrozenFoundation,
        *,
        retrieval_threshold: float = 0.7,
        use_llm_extraction: bool = True,
        max_recent_turns: int = 6,
    ) -> None:
        self.foundation = foundation
        self.memory = XRayEpisodicMemory(
            key_dim=foundation.key_dim,
            retrieval_threshold=retrieval_threshold,
        )
        self.regex_extractor = FactExtractor()
        self.use_llm_extraction = bool(use_llm_extraction)
        self.llm_extractor: LLMFactExtractor | None = (
            LLMFactExtractor(foundation) if self.use_llm_extraction else None
        )
        self.conversation = ConversationManager(max_recent_turns=max_recent_turns)

    # ---------- session lifecycle ----------

    def new_session(self) -> int:
        """Start a fresh conversation. Memory persists across
        sessions; only the in-session conversation history is
        cleared. Returns the new memory session id."""
        self.conversation.clear()
        return self.memory.new_session()

    # ---------- persistence ----------

    def save(self, path: str) -> None:
        """Persist memory entries to ``path`` as JSON.

        Only the privacy-preserving fields are written: the
        stable key vector, the structured facts, the timestamp,
        the access count, and the creation session. Raw user
        text and the in-session conversation history are NOT
        written — this is the on-disk extension of the
        Phase 1.0 privacy contract.
        """
        state = {
            "entries": [
                {
                    # ``effective_key`` is always Float32, even when
                    # the entry has been compressed to a higher
                    # precision level — the save format thus stays
                    # stable across the Phase 2c bis precision
                    # ladder. Loaded entries default to L0; the
                    # precision history is intentionally NOT
                    # persisted (a fresh session restarts the decay
                    # clock).
                    "key": entry.effective_key.tolist(),
                    "facts": entry.facts,
                    "timestamp": entry.timestamp.isoformat(),
                    "access_count": entry.access_count,
                    "creation_session": entry.creation_session,
                }
                for entry in self.memory.entries
            ],
            "current_session": self.memory.current_session,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load(self, path: str) -> None:
        """Restore memory from a file written by :meth:`save`.

        Replaces the current in-memory entries entirely. The
        conversation history is not affected — call
        :meth:`new_session` if you want a clean slate before
        the next turn.
        """
        with open(path) as f:
            state = json.load(f)
        restored: list[EpisodicEntry] = []
        for entry_data in state["entries"]:
            entry = EpisodicEntry(
                key=torch.tensor(entry_data["key"], dtype=torch.float32),
                facts=dict(entry_data["facts"]),
                timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                access_count=int(entry_data.get("access_count", 0)),
                creation_session=int(entry_data.get("creation_session", 0)),
            )
            restored.append(entry)
        self.memory.entries = restored
        self.memory.current_session = int(state.get("current_session", 0))

    # ---------- observe ----------

    def observe(self, text: str) -> dict:
        """Extract structured facts from ``text`` and store
        ``(key, facts)`` in memory.

        Extraction strategy:
          1. If LLM extraction is enabled, prompt the foundation
             first.
          2. Fall back to the regex extractor when the LLM
             returns no parseable facts.
        Empty extraction is a no-op — the memory is never indexed
        against a fact-less entry.
        """
        facts: dict = {}
        if self.llm_extractor is not None:
            facts = self.llm_extractor.extract(text)
        if not facts:
            facts = self.regex_extractor.extract(text)
        if not facts:
            return {}
        key = self.foundation.get_key(text)
        self.memory.add_entry(key, facts)
        return facts

    # ---------- respond ----------

    def respond(
        self,
        query: str,
        *,
        max_new_tokens: int = 80,
        temperature: float = 0.7,
        top_k: int = 3,
        include_history: bool = False,
    ) -> dict[str, Any]:
        """Answer ``query`` augmented with retrieved memory.

        When ``include_history`` is True, the recent conversation
        turns (from :class:`ConversationManager`) are threaded
        into the prompt ahead of the new user turn. The default
        is False to preserve the Phase 1.0 contract for callers
        that use ``respond`` directly without going through
        :meth:`chat`.

        Returns a dict with:
            response:        the foundation's generated string
            memory_used:     bool — did we augment with context?
            retrieved_facts: list of ``(facts, similarity)``
            merged_facts:    union of retrieved fact dicts
            memory_size:     current memory length
        """
        query_key = self.foundation.get_key(query)
        retrieved = self.memory.retrieve(query_key, top_k=top_k)

        history_block = (
            self.conversation.get_recent_context() if include_history else ""
        )

        if retrieved:
            merged = self.memory.merge_facts(retrieved)
            context_str = self._format_facts_as_context(merged)
            system_block = (
                f"<|im_start|>system\n"
                f"You have the following information about the user:\n"
                f"{context_str}\n"
                f"Use this information naturally when relevant.\n"
                f"<|im_end|>\n"
            )
        else:
            merged = {}
            system_block = ""

        # When history is included, the *current* user turn is
        # already the last entry in the conversation manager
        # (chat() adds it before calling respond). Avoid duplicating
        # it in the prompt.
        if include_history and history_block:
            prompt = (
                f"{system_block}"
                f"{history_block}\n"
                f"<|im_start|>assistant\n"
            )
        else:
            prompt = (
                f"{system_block}"
                f"<|im_start|>user\n{query}\n<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        response = self.foundation.generate(
            prompt, max_new_tokens=max_new_tokens, temperature=temperature,
        )
        return {
            "response": response,
            "memory_used": bool(retrieved),
            "retrieved_facts": [
                (entry.facts, sim) for entry, sim in retrieved
            ],
            "merged_facts": merged,
            "memory_size": len(self.memory),
        }

    # ---------- chat (observe + respond + history) ----------

    def chat(
        self,
        text: str,
        *,
        max_new_tokens: int = 80,
        temperature: float = 0.7,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Single entry point for a conversational turn.

        Wires together observation, retrieval, generation, and
        conversation-history bookkeeping in the order a real
        chat loop wants them:

          1. Extract facts from the user's message and store
             them in memory.
          2. Append the user turn to the conversation history.
          3. Generate a response using retrieved facts + recent
             history.
          4. Append the assistant turn to the history.

        The returned dict adds ``extracted_facts`` to the
        Phase 1.0 :meth:`respond` payload so callers can log
        what the extractor found this turn.
        """
        extracted = self.observe(text)
        self.conversation.add_turn("user", text)
        result = self.respond(
            text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            include_history=True,
        )
        self.conversation.add_turn("assistant", result["response"])
        result["extracted_facts"] = extracted
        return result

    # ---------- helpers ----------

    @staticmethod
    def _format_facts_as_context(facts: dict) -> str:
        """Render the structured-fact dict as natural-language
        bullet points for the system prompt.

        Common Phase 1.0 keys (name/age/location/preferences) get
        dedicated labels; arbitrary keys from LLM extraction get
        a generic ``- {key}: {value}`` rendering so we don't
        silently drop them on the floor.
        """
        lines: list[str] = []
        known: set[str] = set()
        if "name" in facts:
            lines.append(f"- Name: {facts['name']}")
            known.add("name")
        if "age" in facts:
            lines.append(f"- Age: {facts['age']}")
            known.add("age")
        if "location" in facts:
            lines.append(f"- Location: {facts['location']}")
            known.add("location")
        if "preferences" in facts:
            prefs = facts["preferences"]
            if isinstance(prefs, list):
                lines.append(f"- Preferences: {', '.join(str(p) for p in prefs)}")
            else:
                lines.append(f"- Preferences: {prefs}")
            known.add("preferences")
        for k, v in facts.items():
            if k in known:
                continue
            if isinstance(v, list):
                rendered = ", ".join(str(x) for x in v)
            else:
                rendered = str(v)
            lines.append(f"- {k.capitalize()}: {rendered}")
        return "\n".join(lines) if lines else "(no specific info)"
