"""Unified AGI system: foundation + memory + fact extraction.

Phase 1.0 pipeline:

  observe(text) → extract structured facts → store (key, facts)
  respond(query) → retrieve relevant entries → merge facts →
                   augment prompt → generate via foundation

No reward signal, no parameter updates, no separate reasoning
module — those come in Phases 2/4. The system here is just the
foundation + memory + extractor wired together to demonstrate
that the privacy-preserving prototype-style memory holds up on
a *real* downstream task (fact retention) end to end.
"""

from __future__ import annotations

from typing import Any

from .extraction import FactExtractor
from .foundation import FrozenFoundation
from .memory.xray_episodic import XRayEpisodicMemory


class AGISystem:
    """Compose foundation + episodic memory + fact extractor.

    The class deliberately takes a *constructed* foundation
    rather than constructing one internally — this lets tests
    pass a mock (``MockFoundation``) without touching Qwen.
    """

    def __init__(
        self,
        foundation: FrozenFoundation,
        *,
        retrieval_threshold: float = 0.7,
    ) -> None:
        self.foundation = foundation
        self.memory = XRayEpisodicMemory(
            key_dim=foundation.key_dim,
            retrieval_threshold=retrieval_threshold,
        )
        self.extractor = FactExtractor()

    # ---------- observe ----------

    def observe(self, text: str) -> dict:
        """Extract structured facts from ``text`` and store
        ``(key, facts)`` in memory.

        Returns the extracted-fact dict for the caller's debug /
        logging convenience. Raw text never crosses into the
        memory.
        """
        facts = self.extractor.extract(text)
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
    ) -> dict[str, Any]:
        """Answer ``query`` augmented with retrieved memory.

        Returns a dict with:
            response:        the foundation's generated string
            memory_used:     bool — did we augment with context?
            retrieved_facts: list of ``(facts, similarity)``
            merged_facts:    union of retrieved fact dicts
            memory_size:     current memory length
        """
        query_key = self.foundation.get_key(query)
        retrieved = self.memory.retrieve(query_key, top_k=top_k)

        if retrieved:
            merged = self.memory.merge_facts(retrieved)
            context_str = self._format_facts_as_context(merged)
            prompt = (
                f"<|im_start|>system\n"
                f"You have the following information about the user:\n"
                f"{context_str}\n"
                f"Use this information naturally when relevant.\n"
                f"<|im_end|>\n"
                f"<|im_start|>user\n{query}\n<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        else:
            merged = {}
            prompt = (
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

    # ---------- helpers ----------

    @staticmethod
    def _format_facts_as_context(facts: dict) -> str:
        """Render the structured-fact dict as natural-language
        bullet points for the system prompt."""
        lines: list[str] = []
        if "name" in facts:
            lines.append(f"- Name: {facts['name']}")
        if "age" in facts:
            lines.append(f"- Age: {facts['age']}")
        if "location" in facts:
            lines.append(f"- Location: {facts['location']}")
        if "preferences" in facts:
            prefs = facts["preferences"]
            if isinstance(prefs, list):
                lines.append(f"- Preferences: {', '.join(prefs)}")
            else:
                lines.append(f"- Preferences: {prefs}")
        return "\n".join(lines) if lines else "(no specific info)"
