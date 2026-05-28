"""Lightweight tests for FrozenFoundation.

These touch the real model and therefore download Qwen-0.5B
(~1 GB) on first run. The first test is marked slow and only
exercises the import path + device-fallback logic; the actual
model-loading + key-generation test runs only when the env var
``AGI_FOUNDATION_TESTS=1`` is set (so the default ``pytest``
run stays fast and offline).
"""

from __future__ import annotations

import os

import pytest
import torch

from agi.foundation import _pick_default_device_dtype


def test_default_device_dtype_picker():
    """Pure-Python pick: CUDA → fp16, otherwise CPU + fp32. No
    model load needed; this is the contract the rest of the code
    relies on."""
    device, dtype = _pick_default_device_dtype()
    if torch.cuda.is_available():
        assert device == "cuda"
        assert dtype == torch.float16
    else:
        assert device == "cpu"
        assert dtype == torch.float32


@pytest.mark.skipif(
    os.environ.get("AGI_FOUNDATION_TESTS") != "1",
    reason=(
        "Set AGI_FOUNDATION_TESTS=1 to run the actual Qwen "
        "load + key-determinism test (~1 GB download first time)."
    ),
)
def test_foundation_key_is_deterministic_for_same_text():
    """Frozen model + greedy / no-randomness pooling → the same
    text MUST produce the same key vector every call."""
    from agi.foundation import FrozenFoundation
    fnd = FrozenFoundation()
    k1 = fnd.get_key("Hello world")
    k2 = fnd.get_key("Hello world")
    assert k1.shape == (fnd.key_dim,)
    assert torch.allclose(k1, k2, atol=1e-5)


@pytest.mark.skipif(
    os.environ.get("AGI_FOUNDATION_TESTS") != "1",
    reason="Requires AGI_FOUNDATION_TESTS=1 (loads Qwen-0.5B).",
)
def test_foundation_keys_differ_for_distinct_texts():
    from agi.foundation import FrozenFoundation
    fnd = FrozenFoundation()
    a = fnd.get_key("My name is Francois")
    b = fnd.get_key("The weather is cold")
    # Same shape but materially different vectors.
    assert a.shape == b.shape
    assert not torch.allclose(a, b, atol=1e-2)
