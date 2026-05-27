"""Tests for ExternalMemory + MemoryAugmentedMLP."""

from __future__ import annotations

import pytest
import torch

from continual_synapse.memory_augmented.memory_augmented_model import (
    ExternalMemory,
    MemoryAugmentedMLP,
)


# ---- 1. empty memory returns zeros ----


def test_empty_memory_returns_zeros() -> None:
    """ExternalMemory.read on an empty store returns a (B, value_dim)
    zero tensor and a (B, 0) empty attention tensor — callers don't
    have to special-case this regime."""
    mem = ExternalMemory(key_dim=8, value_dim=16)
    query = torch.randn(5, 8)
    retrieved, weights = mem.read(query)
    assert retrieved.shape == (5, 16)
    assert torch.all(retrieved == 0)
    assert weights.shape == (5, 0)


# ---- 2. memory grows on write ----


def test_memory_grows_on_write() -> None:
    """Each write appends to keys / values / task_ids and the
    buffers grow correspondingly."""
    mem = ExternalMemory(key_dim=4, value_dim=4)
    assert len(mem) == 0
    mem.write(
        keys=torch.randn(3, 4),
        values=torch.randn(3, 4),
        task_id=0,
    )
    assert len(mem) == 3
    assert mem.task_ids.tolist() == [0, 0, 0]
    mem.write(
        keys=torch.randn(2, 4),
        values=torch.randn(2, 4),
        task_id=7,
    )
    assert len(mem) == 5
    # Task ids are tracked per entry.
    assert mem.task_ids.tolist() == [0, 0, 0, 7, 7]


# ---- 3. attention focuses on the matching key ----


def test_read_returns_attended_values() -> None:
    """Write 5 (key, value) pairs with one key clearly closer to the
    query. The retrieved value should be close to that key's value
    — attention is doing its job. We can't ask for exact equality
    (softmax assigns non-zero weight to every entry) but the
    retrieved vector should be much closer to the target than to
    any of the other four values."""
    mem = ExternalMemory(key_dim=4, value_dim=4)
    keys = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 0.0],  # target row 2
            [0.0, 0.0, 0.0, 10.0],
            [-10.0, 0.0, 0.0, 0.0],
        ]
    )
    # Distinctive value per key — value[i] == one-hot row i scaled.
    values = torch.eye(5, 4)[:5]  # (5, 4) — value[2] = [0, 0, 1, 0]
    mem.write(keys=keys, values=values, task_id=0)

    # Query strongly favors key row 2.
    query = torch.tensor([[0.0, 0.0, 10.0, 0.0]])
    retrieved, weights = mem.read(query)
    assert retrieved.shape == (1, 4)
    # Attention weight on row 2 should dominate.
    assert weights[0].argmax().item() == 2
    # And retrieved should be close to values[2] = [0, 0, 1, 0].
    closest_idx = (
        (retrieved[0].unsqueeze(0) - values).pow(2).sum(dim=-1).argmin()
    )
    assert int(closest_idx) == 2


# ---- 4. empty memory → output equals classifier(encoder(x)) ----


def test_forward_with_empty_memory_uses_encoder_only() -> None:
    """When memory is empty, the gate/combiner path is skipped and
    forward output equals classify(features(x)) exactly. This is
    the bit-identical-to-bare-MLP contract for the empty-memory
    regime."""
    torch.manual_seed(0)
    model = MemoryAugmentedMLP(
        input_dim=4, hidden_dim=8, n_classes=3,
        key_dim=4, value_dim=4, n_encoder_layers=1,
    )
    model.eval()
    x = torch.randn(2, 4)
    out_forward = model(x)
    # Direct path:
    h = model.encoder(x)
    expected = model.classifier(h)
    torch.testing.assert_close(out_forward, expected, rtol=1e-6, atol=1e-6)


# ---- 5. non-empty memory deterministically changes the output ----


def test_forward_with_memory_uses_attention() -> None:
    """Writing entries into memory and forwarding the same input must
    change the model's output (gate=0.5 at init mixes in the
    combined branch). Verifies the memory path is actually wired
    into forward."""
    torch.manual_seed(0)
    model = MemoryAugmentedMLP(
        input_dim=4, hidden_dim=8, n_classes=3,
        key_dim=4, value_dim=4, n_encoder_layers=1,
    )
    model.eval()
    x = torch.randn(3, 4)
    out_empty = model(x).clone()
    # Write some entries.
    model.write_batch_to_memory(torch.randn(10, 4), task_id=0)
    out_populated = model(x)
    assert not torch.allclose(out_empty, out_populated, atol=1e-4), (
        "model output should change once memory contains entries"
    )


# ---- 6. gradients flow through every memory-access head ----


def test_gradients_flow_through_memory_access() -> None:
    """With memory populated, backward must produce non-zero grads on
    every parameter that touches the memory path: query_proj,
    value_proj's downstream effect via stored values, context_combiner,
    memory_gate. This is what makes the access mechanism *learnable*."""
    torch.manual_seed(0)
    model = MemoryAugmentedMLP(
        input_dim=4, hidden_dim=8, n_classes=3,
        key_dim=4, value_dim=4, n_encoder_layers=1,
    )
    model.train()
    # Populate memory first so the gate/combiner path is active.
    model.write_batch_to_memory(torch.randn(5, 4), task_id=0)

    x = torch.randn(2, 4)
    y = torch.tensor([0, 1], dtype=torch.long)
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()

    for name in [
        "query_proj", "context_combiner", "memory_gate", "classifier",
    ]:
        module = getattr(model, name)
        grads = [
            p.grad for p in module.parameters() if p.requires_grad
        ]
        assert grads, f"{name} should have parameters with grad"
        assert any(
            g is not None and g.abs().sum().item() > 0 for g in grads
        ), f"{name} should accumulate non-zero gradients"


# ---- 7. stored memory entries are NOT trainable ----


def test_gradients_do_not_flow_into_stored_memory() -> None:
    """The stored keys and values are buffers, not parameters; they
    must not appear in model.parameters() and they must not get
    gradients accumulated on them after backward."""
    model = MemoryAugmentedMLP(
        input_dim=4, hidden_dim=8, n_classes=3,
        key_dim=4, value_dim=4, n_encoder_layers=1,
    )
    model.write_batch_to_memory(torch.randn(5, 4), task_id=0)
    # The buffer tensors are not parameters.
    param_ids = {id(p) for p in model.parameters()}
    assert id(model.memory.keys) not in param_ids
    assert id(model.memory.values) not in param_ids
    # They don't require grad — backward can't even attach to them.
    assert model.memory.keys.requires_grad is False
    assert model.memory.values.requires_grad is False

    # And after a real backward, their .grad attribute is None.
    x = torch.randn(2, 4)
    logits = model(x)
    logits.sum().backward()
    assert model.memory.keys.grad is None
    assert model.memory.values.grad is None


# ---- 8. write_batch_to_memory accumulates no gradients ----


def test_write_batch_to_memory_uses_no_grad() -> None:
    """Calling write_batch_to_memory mid-training must not pollute
    any model parameter's .grad. The function is @torch.no_grad and
    we verify by clearing grads, writing, then asserting every
    parameter's grad is still None."""
    model = MemoryAugmentedMLP(
        input_dim=4, hidden_dim=8, n_classes=3,
        key_dim=4, value_dim=4, n_encoder_layers=1,
    )
    for p in model.parameters():
        p.grad = None
    model.write_batch_to_memory(torch.randn(4, 4), task_id=0)
    for name, p in model.named_parameters():
        assert p.grad is None, (
            f"write_batch_to_memory leaked a gradient into {name}"
        )
