"""Canned response templates for the metacognitive orchestrator.

The metacognitive layer can decide that the system should NOT
generate a free-form response (epistemic status = ``unknown`` or
``hallucinated``). In those cases the orchestrator falls back to
a small library of pre-written replies — short, honest, in the
user's language.

Phase 2a ships a minimum useful set (admit-ignorance + ask-for-
clarification + softened-uncertainty) in French and English.
:meth:`add_template` lets later phases extend without modifying
this file.
"""

from __future__ import annotations


_DEFAULT_TEMPLATES: dict[str, str] = {
    # Polite admission of ignorance — no follow-up prompt.
    "ignorance_polite_fr": "Je n'ai pas cette information dans ma mémoire.",
    "ignorance_polite_en": "I don't have that information in my memory.",
    # Admission of ignorance with an invitation to share more.
    "ignorance_curious_fr": (
        "Je ne sais pas — peux-tu m'en dire plus?"
    ),
    "ignorance_curious_en": (
        "I don't know — can you tell me more?"
    ),
    # Soft uncertainty — used with ``answer_with_caveat``.
    # ``{guess}`` is the best-effort answer to prepend the caveat to.
    "uncertainty_fr": (
        "Je ne suis pas certain, mais je crois que {guess}."
    ),
    "uncertainty_en": (
        "I'm not certain, but I believe {guess}."
    ),
    # Ask the user to disambiguate the topic.
    "asking_clarification_fr": "Peux-tu préciser {topic}?",
    "asking_clarification_en": "Can you clarify {topic}?",
}


class ResponseTemplates:
    """A small key → template-string store with safe formatting.

    Templates are stored as Python ``str.format``-style strings.
    :meth:`retrieve` formats with the given keyword arguments;
    calling with no kwargs returns the template verbatim (no
    accidental ``KeyError`` from unfilled placeholders when the
    caller doesn't need them).
    """

    def __init__(self) -> None:
        self.templates: dict[str, str] = dict(_DEFAULT_TEMPLATES)

    def retrieve(self, template_key: str, **kwargs) -> str:
        """Return the template under ``template_key``, formatted
        with ``kwargs``.

        Raises ``KeyError`` if the key is unknown. Returns the
        template unformatted when no kwargs are passed so callers
        without substitutions don't need to know the template's
        placeholders.
        """
        if template_key not in self.templates:
            raise KeyError(
                f"Unknown template key {template_key!r}; "
                f"known keys: {sorted(self.templates)}"
            )
        template = self.templates[template_key]
        if not kwargs:
            return template
        return template.format(**kwargs)

    def add_template(self, key: str, value: str) -> None:
        """Register a new template under ``key`` (or overwrite)."""
        self.templates[str(key)] = str(value)

    def list_templates(self) -> list[str]:
        """All currently-registered template keys, sorted."""
        return sorted(self.templates)
