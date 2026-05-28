"""Tests for ConversationManager (turn history + prompt formatting)."""

from __future__ import annotations

import pytest

from agi.conversation import ConversationManager


def test_empty_history_renders_empty_context():
    cm = ConversationManager()
    assert cm.get_recent_context() == ""
    assert len(cm) == 0


def test_add_turn_grows_history():
    cm = ConversationManager()
    cm.add_turn("user", "hi")
    cm.add_turn("assistant", "hello")
    assert len(cm) == 2
    assert cm.turns[0]["role"] == "user"
    assert cm.turns[0]["content"] == "hi"
    assert cm.turns[1]["role"] == "assistant"


def test_turn_records_timestamp():
    cm = ConversationManager()
    cm.add_turn("user", "x")
    assert "timestamp" in cm.turns[0]


def test_context_uses_qwen_chat_template_tokens():
    cm = ConversationManager()
    cm.add_turn("user", "hi")
    cm.add_turn("assistant", "hello")
    ctx = cm.get_recent_context()
    assert "<|im_start|>user" in ctx
    assert "<|im_start|>assistant" in ctx
    assert "<|im_end|>" in ctx
    assert "hi" in ctx
    assert "hello" in ctx


def test_context_is_bounded_to_max_recent_turns():
    """Only the last ``max_recent`` turns appear in the formatted
    context; earlier turns remain in :attr:`turns` for inspection
    but don't leak into the prompt."""
    cm = ConversationManager(max_recent_turns=2)
    cm.add_turn("user", "u1")
    cm.add_turn("assistant", "a1")
    cm.add_turn("user", "u2")
    cm.add_turn("assistant", "a2")
    ctx = cm.get_recent_context()
    assert "u1" not in ctx
    assert "a1" not in ctx
    assert "u2" in ctx
    assert "a2" in ctx
    # Full retained history is unchanged.
    assert len(cm) == 4


def test_clear_drops_all_turns():
    cm = ConversationManager()
    cm.add_turn("user", "x")
    cm.add_turn("assistant", "y")
    cm.clear()
    assert len(cm) == 0
    assert cm.get_recent_context() == ""


def test_invalid_max_recent_raises():
    with pytest.raises(ValueError):
        ConversationManager(max_recent_turns=0)
    with pytest.raises(ValueError):
        ConversationManager(max_recent_turns=-1)
