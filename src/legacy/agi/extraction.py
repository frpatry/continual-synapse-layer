"""Structured fact extraction from natural-language input.

Phase 1.0 uses simple regex patterns covering a few common fact
types (name, age, location, preference) in French and English.
Later phases will replace this with foundation-LLM-driven
extraction for arbitrary fact types — for now the simple
patterns are enough to demonstrate the storage + retrieval
pipeline end-to-end.

The privacy contract is enforced at this layer: we extract
*structured facts* from text and pass them to the memory.
Raw text never crosses the API boundary into the memory layer.
"""

from __future__ import annotations

import re
from typing import Iterable


class FactExtractor:
    """Pattern-based extractor for a handful of fact types.

    Designed to be permissive on common phrasings and conservative
    on edge cases — when uncertain, return nothing rather than a
    false positive. Each pattern is tried independently; multiple
    fact types can be extracted from a single utterance.
    """

    # Name introductions in French + English. ``(\w+)`` captures
    # the first whitespace-bounded token after the trigger — fine
    # for single-word names; multi-word names need richer patterns
    # later.
    NAME_PATTERNS: tuple[str, ...] = (
        r"mon nom est\s+([A-Za-zÀ-ÖØ-öø-ÿ]+)",
        r"je m'appelle\s+([A-Za-zÀ-ÖØ-öø-ÿ]+)",
        r"my name is\s+([A-Za-zÀ-ÖØ-öø-ÿ]+)",
        r"\bi['' ]?am\s+([A-Z][A-Za-zÀ-ÖØ-öø-ÿ]+)",
        r"\bi'?m\s+([A-Z][A-Za-zÀ-ÖØ-öø-ÿ]+)",
        r"call me\s+([A-Za-zÀ-ÖØ-öø-ÿ]+)",
    )

    AGE_PATTERNS: tuple[str, ...] = (
        r"j'ai\s+(\d+)\s+ans",
        r"i am\s+(\d+)\s+years?\s*old",
        r"i'?m\s+(\d+)\s+years?\s*old",
    )

    # Location: capitalised place name following the trigger.
    # The trigger phrase is matched case-insensitively (via inline
    # ``(?i:...)`` scoped flag); the place-name capture stays
    # case-sensitive so we only grab tokens that look like proper
    # nouns rather than common nouns.
    LOCATION_PATTERNS: tuple[str, ...] = (
        r"(?i:je vis|je suis|j'?habite)\s+(?:à|en|au[xs]?|chez)\s+"
        r"([A-ZÀ-Ö][A-Za-zÀ-ÖØ-öø-ÿ]+(?:[ \-][A-ZÀ-Ö][A-Za-zÀ-ÖØ-öø-ÿ]+)*)",
        r"(?i:i live\s+in|i'?m from)\s+"
        r"([A-Z][A-Za-zÀ-ÖØ-öø-ÿ]+(?:[ \-][A-Z][A-Za-zÀ-ÖØ-öø-ÿ]+)*)",
    )

    PREFERENCE_PATTERNS: tuple[str, ...] = (
        r"je préfère\s+(.+?)(?:[\.,!\?]|$)",
        r"j'aime\s+(.+?)(?:[\.,!\?]|$)",
        r"i prefer\s+(.+?)(?:[\.,!\?]|$)",
        r"i like\s+(.+?)(?:[\.,!\?]|$)",
    )

    def extract(self, text: str) -> dict:
        """Return a dict of extracted facts. Empty dict when
        nothing matches — never None, so callers can iterate."""
        facts: dict = {}

        for pat in self.NAME_PATTERNS:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                facts["name"] = m.group(1).capitalize()
                break

        for pat in self.AGE_PATTERNS:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                facts["age"] = int(m.group(1))
                break

        # Location patterns are case-sensitive (capitalised place
        # name as the marker) — don't pass IGNORECASE here.
        for pat in self.LOCATION_PATTERNS:
            m = re.search(pat, text)
            if m:
                facts["location"] = m.group(1).strip()
                break

        # Preferences can stack — collect every match across the
        # patterns rather than break after the first.
        prefs: list[str] = []
        for pat in self.PREFERENCE_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                pref = m.group(1).strip()
                if pref and pref not in prefs:
                    prefs.append(pref)
        if prefs:
            facts["preferences"] = prefs

        return facts
