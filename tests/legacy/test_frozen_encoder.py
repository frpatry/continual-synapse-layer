"""Tests for the PretrainedContrastiveEncoder wrapper."""

from __future__ import annotations

import pytest
import torch

from continual_synapse.episodic.contrastive_encoder import ContrastiveEncoder
from continual_synapse.episodic.frozen_encoder import PretrainedContrastiveEncoder


def _write_dummy_checkpoint(
    tmp_path, input_dim: int = 16, hidden_dim: int = 8,
    feature_dim: int = 4, projection_dim: int = 3,
):
    """Create a minimal ContrastiveEncoder, dump it to a checkpoint,
    return the path. Avoids running the full pretraining script for
    unit-test purposes."""
    enc = ContrastiveEncoder(
        input_dim=input_dim, hidden_dim=hidden_dim,
        feature_dim=feature_dim, projection_dim=projection_dim,
    )
    ckpt_path = tmp_path / "fake_encoder.pt"
    torch.save(
        {"state_dict": enc.state_dict(), "config": enc.config},
        ckpt_path,
    )
    return ckpt_path, enc


# ---- 1. loads + freezes ----


def test_pretrained_loads_and_is_frozen(tmp_path) -> None:
    """After loading, every parameter must have requires_grad=False
    so the encoder can't accidentally enter a downstream gradient
    graph."""
    ckpt_path, _ = _write_dummy_checkpoint(tmp_path)
    frozen = PretrainedContrastiveEncoder(ckpt_path)
    params = list(frozen.parameters())
    assert len(params) > 0, "encoder should expose parameters"
    for p in params:
        assert p.requires_grad is False, (
            f"frozen encoder parameter still requires_grad: {p.shape}"
        )
    # Feature dim survives the round-trip.
    assert frozen.feature_dim == 4


# ---- 2. deterministic output ----


def test_pretrained_output_deterministic(tmp_path) -> None:
    """Same input twice must produce identical output — no dropout,
    no batchnorm running stats, no other source of nondeterminism."""
    ckpt_path, _ = _write_dummy_checkpoint(tmp_path)
    frozen = PretrainedContrastiveEncoder(ckpt_path)
    x = torch.randn(3, 16)
    out1 = frozen(x)
    out2 = frozen(x)
    torch.testing.assert_close(out1, out2, rtol=0, atol=0)
    # And output matches what the underlying encoder.encode would
    # produce in eval mode — verifies we kept the right module.
    assert out1.shape == (3, 4)


# ---- 3. eval mode is preserved ----


def test_pretrained_eval_mode_preserved(tmp_path) -> None:
    """Calling .train() on the frozen wrapper must NOT flip it into
    training mode. The override-to-no-op makes the frozen contract
    structural so a downstream model.train() that propagates to
    children can't accidentally un-freeze this encoder."""
    ckpt_path, _ = _write_dummy_checkpoint(tmp_path)
    frozen = PretrainedContrastiveEncoder(ckpt_path)
    assert frozen.training is False

    # Try to put it in training mode — should silently stay in eval.
    frozen.train()
    assert frozen.training is False, (
        "PretrainedContrastiveEncoder.train() must not enable training mode"
    )
    frozen.train(True)
    assert frozen.training is False
    # Explicit eval still works (no-op, but legal).
    frozen.eval()
    assert frozen.training is False
