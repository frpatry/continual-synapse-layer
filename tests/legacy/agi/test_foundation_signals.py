"""Tests for FrozenFoundation.generate_with_signals (Phase 2b).

All tests in this module load Qwen2.5-1.5B-Instruct via the
real :class:`FrozenFoundation`; on first run that's a ~3 GB
download. They are therefore gated behind the same env var as
the existing key-determinism tests in
``tests/agi/test_foundation.py`` — set ``AGI_FOUNDATION_TESTS=1``
to actually run them.

The gated tests share a single foundation via a session-scoped
fixture so the model loads once even when several tests run.
"""

from __future__ import annotations

import math
import os
import time

import pytest

from agi.foundation import FrozenFoundation, GenerationInfo


pytestmark = pytest.mark.skipif(
    os.environ.get("AGI_FOUNDATION_TESTS") != "1",
    reason=(
        "Set AGI_FOUNDATION_TESTS=1 to run the real-Qwen "
        "generate_with_signals tests (~3 GB download first time, "
        "~10-30s per test on CPU)."
    ),
)


# Module-scoped foundation so the model loads once across the
# five tests in this file. We can't use pytest's built-in
# fixture scoping niceties without changing other tests, so we
# wire it up by hand via a module-level cache.

_foundation_cache: dict = {}


def _foundation() -> FrozenFoundation:
    if "fnd" not in _foundation_cache:
        _foundation_cache["fnd"] = FrozenFoundation()
    return _foundation_cache["fnd"]


def _short_prompt() -> str:
    """A trivially short prompt — keeps the CPU run under a few
    seconds per test."""
    return "Hello, can you say one short sentence in English?"


def test_generate_with_signals_returns_valid_info():
    """Smoke: the call returns a populated GenerationInfo with
    sensible field values (non-empty text, non-negative entropy,
    response length matches the generated id list)."""
    fnd = _foundation()
    info = fnd.generate_with_signals(
        _short_prompt(), max_new_tokens=16, temperature=0.0,
    )
    assert isinstance(info, GenerationInfo)
    assert info.response_text.strip() != ""
    assert info.response_length_tokens > 0
    assert info.response_length_tokens == len(info.generated_token_ids)
    assert info.mean_token_entropy >= 0.0
    assert info.max_token_entropy >= info.mean_token_entropy
    assert info.generation_time_seconds > 0


def test_generate_with_signals_without_fact_ranges():
    """No ``fact_token_ranges`` → attention bookkeeping is
    skipped (faster path) and both fields stay at 0.0."""
    fnd = _foundation()
    info = fnd.generate_with_signals(
        _short_prompt(),
        fact_token_ranges=None,
        max_new_tokens=8,
        temperature=0.0,
    )
    assert info.attention_to_facts_mean == 0.0
    assert info.attention_to_facts_max == 0.0


def test_generate_with_signals_with_fact_ranges():
    """With a fact span pointed at by ``fact_token_ranges``, the
    attention-to-facts signal must be strictly positive — every
    decoder layer puts at least *some* mass on any in-range
    position by construction (softmax weights are positive)."""
    fnd = _foundation()
    prompt = (
        "FACT: name=François. "
        "Question: comment je m'appelle? "
        "Reply briefly."
    )
    # Locate the FACT span in token IDs. Tokenise the prefix up
    # to the start of FACT to get the start index, then up to
    # the end of "François." to get the end index.
    enc_full = fnd.tokenizer(prompt, return_tensors="pt")
    enc_pre = fnd.tokenizer("", return_tensors="pt")
    enc_through = fnd.tokenizer(
        "FACT: name=François.", return_tensors="pt",
    )
    fact_start = int(enc_pre["input_ids"].shape[1])
    fact_end = int(enc_through["input_ids"].shape[1])
    assert fact_end > fact_start, "fact span is empty"
    assert fact_end <= int(enc_full["input_ids"].shape[1])

    info = fnd.generate_with_signals(
        prompt,
        fact_token_ranges=[(fact_start, fact_end)],
        max_new_tokens=12,
        temperature=0.0,
    )
    assert info.attention_to_facts_mean > 0.0
    assert info.attention_to_facts_max >= info.attention_to_facts_mean


def test_entropy_values_in_reasonable_range():
    """``mean_entropy`` is bounded by ``log(vocab_size)``; for
    Qwen2.5 vocab ~152k the upper bound is ~11.93 nats. We
    require at least that the values are finite and
    non-negative — a hard upper-bound check confirms the
    log-softmax path isn't accidentally double-logging."""
    fnd = _foundation()
    vocab_size = int(fnd.model.config.vocab_size)
    upper = math.log(vocab_size) + 1e-3  # tiny float slack
    info = fnd.generate_with_signals(
        _short_prompt(), max_new_tokens=8, temperature=0.0,
    )
    assert 0.0 <= info.mean_token_entropy <= upper
    assert 0.0 <= info.max_token_entropy <= upper


def test_timing_vs_plain_generate():
    """``generate_with_signals`` runs the same forward but
    materialises ``scores`` (always) and ``attentions`` (when
    fact ranges are passed). The overhead vs plain ``generate``
    should stay under a generous 4x ceiling on CPU; the
    primary point of the test is to catch a 10x+ regression
    from e.g. accidentally holding the full attention tuple
    on the device.
    """
    fnd = _foundation()
    prompt = _short_prompt()
    n_tokens = 16

    # Warm-up call so the first allocation / kernel-pick doesn't
    # bias the measurement.
    fnd.generate(prompt, max_new_tokens=4, temperature=0.0)

    t0 = time.perf_counter()
    fnd.generate(prompt, max_new_tokens=n_tokens, temperature=0.0)
    plain_seconds = time.perf_counter() - t0

    info = fnd.generate_with_signals(
        prompt, max_new_tokens=n_tokens, temperature=0.0,
    )
    signals_seconds = info.generation_time_seconds

    # Don't strictly enforce a tight ratio (CPU + Python timing
    # is noisy at sub-second scale). Cap at 4x and log the
    # ratio for visibility in pytest -v output.
    ratio = signals_seconds / max(plain_seconds, 1e-6)
    print(
        f"\n[timing] plain={plain_seconds:.3f}s "
        f"signals={signals_seconds:.3f}s ratio={ratio:.2f}x"
    )
    assert ratio < 4.0, (
        f"generate_with_signals took {ratio:.2f}x plain generate "
        f"({signals_seconds:.3f}s vs {plain_seconds:.3f}s) — "
        f"likely a regression in the signal-capture path"
    )
