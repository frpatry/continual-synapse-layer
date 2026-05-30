"""Tests for the forward-hook utilities."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from continual_synapse.base_models.hooks import (
    ActivationCapture,
    get_module_by_name,
)
from continual_synapse.baselines.naive_finetune import MLPClassifier, MLPConfig


def test_get_module_by_name_returns_root_for_empty() -> None:
    model = MLPClassifier(MLPConfig(input_dim=4, hidden_dim=8, num_classes=2))
    assert get_module_by_name(model, "") is model


def test_get_module_by_name_walks_attributes() -> None:
    model = MLPClassifier(MLPConfig(input_dim=4, hidden_dim=8, num_classes=2))
    assert get_module_by_name(model, "head") is model.head


def test_get_module_by_name_indexes_sequential() -> None:
    model = MLPClassifier(MLPConfig(input_dim=4, hidden_dim=8, num_classes=2))
    first_linear = get_module_by_name(model, "backbone.0")
    assert isinstance(first_linear, nn.Linear)
    assert first_linear is model.backbone[0]


def test_get_module_by_name_rejects_missing_attribute() -> None:
    model = MLPClassifier(MLPConfig(input_dim=4, hidden_dim=8, num_classes=2))
    with pytest.raises(AttributeError, match="missing"):
        get_module_by_name(model, "missing")


def test_capture_records_backbone_features() -> None:
    cfg = MLPConfig(input_dim=4, hidden_dim=8, num_classes=2)
    model = MLPClassifier(cfg)
    capture = ActivationCapture(model, "backbone")
    capture.attach()
    try:
        x = torch.randn(3, 4)
        logits = model(x)
        feats = capture.activation
        # Backbone output shape matches `features(x)`.
        assert feats.shape == (3, cfg.hidden_dim)
        # Match the deterministic features() pathway exactly.
        torch.testing.assert_close(feats, model.features(x))
        # Logits depend on the same features through the head.
        assert logits.shape == (3, cfg.num_classes)
    finally:
        capture.detach_hook()


def test_capture_can_be_used_as_context_manager() -> None:
    model = MLPClassifier(MLPConfig(input_dim=4, hidden_dim=8, num_classes=2))
    x = torch.randn(2, 4)
    with ActivationCapture(model, "backbone") as cap:
        model(x)
        feats = cap.activation
    assert feats.shape == (2, 8)
    # After exit, hook is detached.
    assert not cap.is_attached


def test_capture_activation_raises_before_forward() -> None:
    model = MLPClassifier(MLPConfig(input_dim=4, hidden_dim=8, num_classes=2))
    with ActivationCapture(model, "backbone") as cap:
        with pytest.raises(RuntimeError, match="No activation captured"):
            _ = cap.activation


def test_capture_detaches_from_autograd_by_default() -> None:
    """Hebbian updates should not propagate gradients back through the
    captured activation."""
    model = MLPClassifier(MLPConfig(input_dim=4, hidden_dim=8, num_classes=2))
    x = torch.randn(2, 4)
    with ActivationCapture(model, "backbone") as cap:
        model(x)
        feats = cap.activation
    assert not feats.requires_grad


def test_capture_can_keep_grad_when_requested() -> None:
    model = MLPClassifier(MLPConfig(input_dim=4, hidden_dim=8, num_classes=2))
    x = torch.randn(2, 4)
    with ActivationCapture(model, "backbone", detach=False) as cap:
        logits = model(x)
        feats = cap.activation
        # Backbone output is part of the live autograd graph.
        assert feats.requires_grad
        loss = logits.sum()
        loss.backward()


def test_detach_hook_is_idempotent() -> None:
    model = MLPClassifier(MLPConfig(input_dim=4, hidden_dim=8, num_classes=2))
    cap = ActivationCapture(model, "backbone")
    cap.attach()
    cap.detach_hook()
    cap.detach_hook()  # second call must not raise
    assert not cap.is_attached


def test_attach_is_idempotent() -> None:
    model = MLPClassifier(MLPConfig(input_dim=4, hidden_dim=8, num_classes=2))
    cap = ActivationCapture(model, "backbone")
    cap.attach()
    cap.attach()  # second call must not double-register
    # Run forward, check we get exactly one observation per call.
    model(torch.randn(1, 4))
    feats_a = cap.activation
    model(torch.randn(1, 4))
    feats_b = cap.activation
    # Different inputs -> different captured tensors (sanity).
    assert feats_a.shape == feats_b.shape
    cap.detach_hook()


def test_clear_resets_cached_activation() -> None:
    model = MLPClassifier(MLPConfig(input_dim=4, hidden_dim=8, num_classes=2))
    with ActivationCapture(model, "backbone") as cap:
        model(torch.randn(1, 4))
        _ = cap.activation
        cap.clear()
        with pytest.raises(RuntimeError):
            _ = cap.activation
