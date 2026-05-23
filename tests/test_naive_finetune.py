"""Tests for the MLP baseline."""

from __future__ import annotations

import pytest
import torch

from continual_synapse.baselines.naive_finetune import MLPClassifier, MLPConfig


def test_forward_output_shape_matches_num_classes() -> None:
    model = MLPClassifier(MLPConfig(input_dim=10, hidden_dim=8, num_classes=3))
    x = torch.randn(4, 10)
    out = model(x)
    assert out.shape == (4, 3)


def test_default_config_is_three_hidden_layers() -> None:
    model = MLPClassifier()
    # Each hidden block is Linear + ReLU (+ optional Dropout). Default has
    # no dropout, so we expect exactly 3 Linear layers in the backbone.
    linear_layers = [m for m in model.backbone if isinstance(m, torch.nn.Linear)]
    assert len(linear_layers) == 3


def test_features_returns_hidden_activations() -> None:
    model = MLPClassifier(MLPConfig(input_dim=10, hidden_dim=8, num_classes=3))
    x = torch.randn(2, 10)
    feats = model.features(x)
    assert feats.shape == (2, 8)


def test_zero_hidden_layers_rejected() -> None:
    with pytest.raises(ValueError, match="num_hidden_layers"):
        MLPClassifier(MLPConfig(num_hidden_layers=0))


def test_gradient_flows_through_backbone() -> None:
    model = MLPClassifier(MLPConfig(input_dim=4, hidden_dim=4, num_classes=2))
    x = torch.randn(3, 4)
    y = torch.tensor([0, 1, 1])
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum().item() > 0 for g in grads if g is not None)
