"""Turn-by-turn conversation history for the AGI architecture.

Phase 1.1 adds multi-turn coherence on top of the Phase 1.0
observe/respond pipeline. The :class:`ConversationManager` is a
deliberately thin shim: it keeps an in-memory list of turns and
formats the most recent slice into Qwen chat-template tokens so
the foundation can condition its next reply on prior turns.

Privacy note: conversation turns are *ephemeral* — they live
only in process memory and are cleared on :meth:`clear` (and on
:meth:`AGISystem.new_session`). The persistent on-disk layer
(:meth:`AGISystem.save`) deliberately does NOT include them; the
privacy contract — never store raw user text past a session —
extends to this layer. The episodic memory continues to hold
*only* structured facts + stable keys.
"""

from __future__ import annotations

from datetime import datetime


class ConversationManager:
    """Track turn-by-turn conversation history within a session.

    The history is bounded to the most recent ``max_recent_turns``
    entries when formatted into prompt context, but the full
    history is retained in :attr:`turns` for in-process inspection.
    Bounding the prompt slice keeps the foundation's context
    window predictable across long conversations; bounding the
    *retained* history is a Phase 3 concern (consolidation /
    summarisation), not 1.1.
    """

    def __init__(self, max_recent_turns: int = 6) -> None:
        if max_recent_turns <= 0:
            raise ValueError(
                f"max_recent_turns must be positive, got {max_recent_turns}"
            )
        self.turns: list[dict] = []
        self.max_recent = int(max_recent_turns)

    def add_turn(self, role: str, content: str) -> None:
        """Append a turn. ``role`` should be ``"user"`` or
        ``"assistant"`` (matching Qwen's chat template); we do not
        validate strictly so callers can experiment with a
        ``"system"`` role if needed."""
        self.turns.append(
            {"role": role, "content": content, "timestamp": datetime.now()}
        )

    def get_recent_context(self) -> str:
        """Format the most recent turns as Qwen chat-template
        tokens for prompt prefixing.

        Returns an empty string when there's no history — callers
        can concatenate unconditionally.
        """
        if not self.turns:
            return ""
        recent = self.turns[-self.max_recent:]
        formatted = [
            f"<|im_start|>{turn['role']}\n{turn['content']}\n<|im_end|>"
            for turn in recent
        ]
        return "\n".join(formatted)

    def clear(self) -> None:
        """Drop all conversation history. Called on
        :meth:`AGISystem.new_session` so a fresh conversation
        doesn't carry the prior session's turns into its prompt."""
        self.turns.clear()

    def __len__(self) -> int:
        return len(self.turns)
