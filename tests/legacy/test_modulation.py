"""Tests for SynapseModulation."""

from __future__ import annotations

import pytest
import torch

from continual_synapse.synapse_layer.modulation import SynapseModulation


def test_correction_is_exactly_zero_at_init() -> None:
    """Gate=0 and arbitrary strengths -> zero correction.

    This is the guarantee that the base model's behaviour is
    preserved on the very first forward pass.
    """
    mod = SynapseModulation()  # default init_gate=0.0
    a = torch.randn(4, 5)
    s = torch.randn(5, 5)
    correction = mod(a, s)
    assert torch.all(correction == 0.0)
    assert correction.shape == (4, 5)


def test_correction_is_linear_in_gate() -> None:
    mod = SynapseModulation()
    a = torch.tensor([[1.0, 0.0]])
    s = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    with torch.no_grad():
        mod.gate.fill_(0.5)
    out = mod(a, s)
    # a @ s = [[1, 2]]; scaled by 0.5 -> [[0.5, 1.0]]
    torch.testing.assert_close(out, torch.tensor([[0.5, 1.0]]))


def test_gradient_flows_through_gate() -> None:
    mod = SynapseModulation(init_gate=0.1)
    a = torch.randn(3, 4)
    s = torch.randn(4, 4)
    out = mod(a, s)
    out.sum().backward()
    assert mod.gate.grad is not None
    # ∂(Σ gate · a@s)/∂gate = Σ (a @ s)
    expected = (a @ s).sum()
    torch.testing.assert_close(mod.gate.grad, expected)


def test_strengths_do_not_receive_gradient_by_default() -> None:
    """SynapseLayer.strengths is a buffer (no requires_grad), so even
    if a stray tensor is passed in, it does not silently start being
    optimized. We can still pass a requires_grad strengths tensor
    if someone wants to experiment, but by default the buffer path
    must not propagate gradients we don't expect."""
    mod = SynapseModulation(init_gate=0.5)
    a = torch.randn(2, 3, requires_grad=True)
    s = torch.zeros(3, 3)  # mimicking a buffer
    assert not s.requires_grad
    out = mod(a, s)
    out.sum().backward()
    assert s.grad is None


def test_rejects_non_2d_activations() -> None:
    mod = SynapseModulation()
    with pytest.raises(ValueError, match="activations must be 2-D"):
        mod(torch.zeros(3), torch.zeros(3, 3))


def test_rejects_non_square_strengths() -> None:
    mod = SynapseModulation()
    with pytest.raises(ValueError, match="square"):
        mod(torch.zeros(2, 3), torch.zeros(3, 2))


def test_rejects_mismatched_dimensions() -> None:
    mod = SynapseModulation()
    with pytest.raises(ValueError, match="does not match"):
        mod(torch.zeros(2, 4), torch.zeros(3, 3))


def test_custom_init_gate_is_respected() -> None:
    mod = SynapseModulation(init_gate=0.25)
    assert float(mod.gate.item()) == 0.25
