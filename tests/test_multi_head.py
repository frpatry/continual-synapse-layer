"""Tests for the multi-head MLP baseline."""

from __future__ import annotations

import pytest
import torch

from continual_synapse.baselines.multi_head import MultiHeadMLPClassifier
from continual_synapse.baselines.naive_finetune import MLPConfig


def test_constructs_with_correct_number_of_heads() -> None:
    model = MultiHeadMLPClassifier(
        num_tasks=5, config=MLPConfig(input_dim=4, hidden_dim=8, num_classes=2)
    )
    assert len(model.heads) == 5
    assert model.active_head == 0


def test_forward_uses_active_head() -> None:
    model = MultiHeadMLPClassifier(
        num_tasks=3, config=MLPConfig(input_dim=4, hidden_dim=8, num_classes=2)
    )
    # Make each head's weight distinguishable.
    with torch.no_grad():
        for i, head in enumerate(model.heads):
            head.weight.fill_(float(i + 1))
            head.bias.zero_()
    x = torch.ones(1, 4)
    feats = model.features(x)
    for i in range(3):
        model.set_active_head(i)
        out = model(x)
        expected = (i + 1) * feats.sum()
        # Each output entry is (i+1) * sum(features).
        torch.testing.assert_close(out, expected.expand(1, 2))


def test_set_active_head_validates_range() -> None:
    model = MultiHeadMLPClassifier(
        num_tasks=2, config=MLPConfig(input_dim=4, hidden_dim=8, num_classes=2)
    )
    with pytest.raises(ValueError, match="out of range"):
        model.set_active_head(2)
    with pytest.raises(ValueError, match="out of range"):
        model.set_active_head(-1)


def test_constructor_rejects_bad_num_tasks() -> None:
    with pytest.raises(ValueError, match="num_tasks"):
        MultiHeadMLPClassifier(num_tasks=0)


def test_backbone_is_shared_gradients_flow_through_it() -> None:
    """Training task 0 should produce gradients on the backbone but
    only on head 0; head 1 should remain untouched."""
    model = MultiHeadMLPClassifier(
        num_tasks=2, config=MLPConfig(input_dim=4, hidden_dim=8, num_classes=2)
    )
    model.set_active_head(0)
    x = torch.randn(3, 4)
    y = torch.tensor([0, 1, 1])
    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()
    # Backbone params have gradients.
    for p in model.backbone.parameters():
        assert p.grad is not None
        assert p.grad.abs().sum().item() > 0
    # Head 0 has gradients; head 1 does not.
    assert model.heads[0].weight.grad is not None
    assert model.heads[0].weight.grad.abs().sum().item() > 0
    assert model.heads[1].weight.grad is None


def test_features_dim_matches_classifier_input() -> None:
    cfg = MLPConfig(input_dim=4, hidden_dim=8, num_classes=2)
    model = MultiHeadMLPClassifier(num_tasks=3, config=cfg)
    feats = model.features(torch.randn(2, 4))
    assert feats.shape == (2, 8)
    logits = model.classify(feats)
    assert logits.shape == (2, 2)
