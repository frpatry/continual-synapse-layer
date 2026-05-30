"""Foundation-LLM-driven structured fact extraction.

Phase 1.0 shipped a regex :class:`agi.extraction.FactExtractor`
hand-tuned for a handful of fact types in French and English.
Phase 1.1 added a foundation-LLM extractor with a schema-listing
prompt; the Phase 1.1 demo against Qwen-0.5B revealed two
failure modes that motivated Phase 1.2:

1. The model treated the prompt's *list of example fact types*
   as a schema to fill in, confabulating values for every key on
   every input.
2. The model emitted fact dicts even on pure questions like
   "Quel est mon nom?" — the wrong direction for a "facts the
   user stated about themselves" extractor.

Phase 1.2 addresses both:

- A few-shot prompt with two positive *and two negative* examples
  (the negatives are pure questions that should yield ``{}``)
  plus an explicit "no hallucination" instruction.
- A pre-LLM heuristic gate (:meth:`_is_question_or_too_short`)
  that skips the foundation call entirely on short utterances
  and obvious questions. Saves compute and removes the chief
  source of schema-fill hallucinations.

The regex extractor remains as a fallback: :class:`AGISystem`
runs the LLM extractor first and falls back to regex when the
LLM returns nothing parseable. Privacy contract unchanged — the
extractor sees raw text but returns only structured facts.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a hard transformers dependency at import time
    from .foundation import FrozenFoundation


def _find_first_json_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` substring of ``text``,
    or ``None`` if no balanced object is found.

    Tolerates nested braces and quoted strings. Not a full JSON
    parser — ``json.loads`` is what actually validates the
    payload; this function just locates a candidate slice.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


# Question-starting tokens in French + English. Used by
# :meth:`LLMFactExtractor._is_question_or_too_short`. Matched
# against the lowercased start of the (stripped) input.
_QUESTION_STARTS: tuple[str, ...] = (
    # French
    "quel", "quelle", "quels", "quelles",
    "comment", "pourquoi", "où",
    "quand", "qui", "que ", "qu'",
    "est-ce", "peux-tu", "peut-on", "pouvez-vous",
    # English
    "what", "where", "when", "why", "how", "who", "which",
    "can you", "could you", "would you",
    "do you", "did you", "is it", "are you",
)


class LLMFactExtractor:
    """Use the frozen foundation to extract structured facts.

    The extractor prompts the foundation for a JSON object
    summarising any user-relevant facts in the input. The
    extractor is deliberately tolerant — when the foundation's
    output is unparseable or empty, :meth:`extract` returns an
    empty dict and lets the caller decide what to do (typically:
    fall back to the regex extractor).

    Attributes:
        last_gated: True if the most recent :meth:`extract` call
            short-circuited via the question/length heuristic
            (no foundation call was made). The demo reads this
            to surface "gated" events in its per-turn log.
    """

    # Few-shot prompt with explicit no-hallucination instruction.
    # The negatives (questions → ``{}``) are crucial — they teach
    # the model that not every input has facts to extract.
    #
    # Split into a constant prefix and a per-call suffix so we
    # can concatenate cleanly without ``str.format()`` choking
    # on the literal ``{...}`` inside the few-shot JSON examples.
    _PROMPT_PREFIX = (
        "<|im_start|>system\n"
        "Tu es un extracteur d'informations factuelles. Le user va "
        "te donner un message. Tu dois retourner UNIQUEMENT les "
        "faits que le user a EXPLICITEMENT déclarés sur lui-même "
        "dans ce message. Si le user ne déclare aucun fait sur "
        "lui-même, retourne {}.\n"
        "\n"
        "IMPORTANT:\n"
        "- N'invente AUCUN fait. Ne complète AUCUN schéma.\n"
        "- Extrais seulement ce qui est explicitement dit.\n"
        "- Réponds avec UN SEUL objet JSON valide. Rien d'autre.\n"
        "\n"
        "Exemples:\n"
        "Input: \"Bonjour, je m'appelle Sarah et je vis à Paris.\"\n"
        "Output: {\"name\": \"Sarah\", \"location\": \"Paris\"}\n"
        "\n"
        "Input: \"J'aime beaucoup le café le matin.\"\n"
        "Output: {\"preferences\": [\"café le matin\"]}\n"
        "\n"
        "Input: \"Quel temps fait-il aujourd'hui?\"\n"
        "Output: {}\n"
        "\n"
        "Input: \"Peux-tu me résumer ce concept?\"\n"
        "Output: {}\n"
        "\n"
        "Input: \"Je travaille comme médecin à Lyon depuis 5 ans.\"\n"
        "Output: {\"profession\": \"médecin\", \"location\": \"Lyon\", "
        "\"years_experience\": 5}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Input: \""
    )
    _PROMPT_SUFFIX = (
        "\"\n"
        "Output: "
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    def __init__(self, foundation: "FrozenFoundation") -> None:
        self.foundation = foundation
        self.last_gated: bool = False

    # ---------- gating ----------

    @staticmethod
    def _is_question_or_too_short(text: str) -> bool:
        """Heuristic — skip extraction when the input is a clear
        question or too short to plausibly contain a user fact.

        Cheap and conservative: a false negative (we run
        extraction on a question) costs one wasted LLM call, but
        a false positive (we skip extraction on a real fact)
        loses information. Tuned to gate aggressively on
        questions because Qwen tends to schema-fill those.
        """
        text_lower = text.lower().strip()

        # Too short to plausibly state a fact.
        if len(text_lower.split()) < 3:
            return True

        # Ends with a question mark.
        if text_lower.rstrip().endswith("?"):
            return True

        # Starts with a question word (French + English).
        for start in _QUESTION_STARTS:
            if text_lower.startswith(start):
                return True

        return False

    # ---------- prompt assembly ----------

    def _build_prompt(self, text: str) -> str:
        return self._PROMPT_PREFIX + text + self._PROMPT_SUFFIX

    # ---------- extract ----------

    def extract(self, text: str) -> dict:
        """Return a dict of extracted facts. Empty dict when:

        - The input is gated as a question / too short (no
          foundation call is made; :attr:`last_gated` is set
          ``True``).
        - The foundation's output is not parseable as a JSON
          object.

        Otherwise returns the parsed JSON object with null /
        empty values dropped.
        """
        if self._is_question_or_too_short(text):
            self.last_gated = True
            print("  [extraction gated: question or short input]")
            return {}
        self.last_gated = False

        prompt = self._build_prompt(text)
        # Greedy decoding (temperature=0) — extraction should be
        # deterministic for a given input.
        response = self.foundation.generate(
            prompt, max_new_tokens=150, temperature=0.0,
        )
        candidate = _find_first_json_object(response)
        if candidate is None:
            return {}
        try:
            facts = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            return {}
        if not isinstance(facts, dict):
            return {}
        # Drop null/empty values — they're noise, and downstream
        # storage / merging assumes a fact has a usable value.
        cleaned: dict = {}
        for k, v in facts.items():
            if v is None:
                continue
            if isinstance(v, str) and v.strip() == "":
                continue
            if isinstance(v, (list, dict)) and len(v) == 0:
                continue
            cleaned[str(k)] = v
        return cleaned
