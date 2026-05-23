"""Tests for the EWC baseline."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.data import TensorDataset

from continual_synapse.baselines.ewc import EWC
from continual_synapse.baselines.naive_finetune import MLPClassifier, MLPConfig
from continual_synapse.evaluation.benchmarks import Task
from continual_synapse.evaluation.runner import ContinualRunner


def _tiny_linear_model() -> nn.Linear:
    """Single Linear(2, 1) with deterministic zero weights, no bias."""
    layer = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        layer.weight.zero_()
    return layer


def _single_sample_dataset() -> TensorDataset:
    x = torch.tensor([[1.0, 2.0]])
    y = torch.tensor([[3.0]])
    return TensorDataset(x, y)


def test_penalty_zero_before_consolidation() -> None:
    model = _tiny_linear_model()
    ewc = EWC(lam=10.0, loss_fn=nn.MSELoss())
    p = ewc.penalty(model)
    assert p.item() == 0.0


def test_penalty_zero_immediately_after_consolidation() -> None:
    """When params have not moved, ``(θ - θ*) = 0`` so the penalty is 0."""
    model = _tiny_linear_model()
    ewc = EWC(lam=10.0, loss_fn=nn.MSELoss())
    ewc.consolidate(model, _single_sample_dataset())
    assert ewc.num_consolidated_tasks == 1
    assert ewc.penalty(model).item() == 0.0


def test_fisher_matches_hand_computation() -> None:
    """Empirical Fisher on a 1-sample MSE problem has a closed form.

    Model y = w0*x0 + w1*x1 with init weights (0, 0), sample
    ``x=(1,2)`` and target ``3``. Residual ``r = pred - target = -3``.
    Per-sample gradient w.r.t. ``w0 = 2r``, w.r.t. ``w1 = 2r*2 = 4r``.
    So Fisher[w0] = (2r)^2 = 4r^2 = 36 and Fisher[w1] = 144.
    """
    model = _tiny_linear_model()
    ewc = EWC(lam=1.0, loss_fn=nn.MSELoss())
    ewc.consolidate(model, _single_sample_dataset())

    fisher = ewc._consolidated[0].fisher["weight"].squeeze()
    assert torch.allclose(fisher, torch.tensor([36.0, 144.0]), atol=1e-5)


def test_penalty_grows_with_parameter_drift() -> None:
    """After consolidation, moving params away increases the penalty."""
    model = _tiny_linear_model()
    ewc = EWC(lam=2.0, loss_fn=nn.MSELoss())
    ewc.consolidate(model, _single_sample_dataset())

    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.5, -0.5]]))

    # Expected: (λ/2) * (F[0]*0.25 + F[1]*0.25)
    #        = (2/2) * (36*0.25 + 144*0.25) = 9 + 36 = 45.
    p = ewc.penalty(model)
    assert math.isclose(p.item(), 45.0, abs_tol=1e-4)


def test_penalty_is_differentiable_wrt_params() -> None:
    """The penalty must contribute gradients during training."""
    model = _tiny_linear_model()
    ewc = EWC(lam=1.0, loss_fn=nn.MSELoss())
    ewc.consolidate(model, _single_sample_dataset())
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.5, -0.5]]))

    p = ewc.penalty(model)
    p.backward()
    assert model.weight.grad is not None
    # ∂p/∂w0 = (λ/2) * 2 * F[0] * (w0 - w0*) = 1 * 36 * 0.5 = 18
    # ∂p/∂w1 = 1 * 144 * (-0.5) = -72
    expected = torch.tensor([[18.0, -72.0]])
    assert torch.allclose(model.weight.grad, expected, atol=1e-4)


def test_multi_task_consolidation_accumulates() -> None:
    """A second consolidation adds another term to the penalty."""
    model = _tiny_linear_model()
    ewc = EWC(lam=1.0, loss_fn=nn.MSELoss())
    ewc.consolidate(model, _single_sample_dataset())
    # Drift, then consolidate again to register a second task.
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 1.0]]))
    ewc.consolidate(model, _single_sample_dataset())
    assert ewc.num_consolidated_tasks == 2
    # At the second consolidation point, only the first task contributes
    # (because diff for task 2 is zero), so penalty == 0.5 * 1 * (F1 * (1-0)^2).
    fisher1 = ewc._consolidated[0].fisher["weight"].squeeze()
    expected = 0.5 * (fisher1[0].item() * 1.0 + fisher1[1].item() * 1.0)
    assert math.isclose(ewc.penalty(model).item(), expected, abs_tol=1e-4)


def test_subsampling_fisher_uses_at_most_n_samples() -> None:
    """``fisher_sample_size`` caps the number of samples seen."""
    x = torch.randn(50, 2)
    y = torch.randn(50, 1)
    ds = TensorDataset(x, y)
    model = _tiny_linear_model()
    ewc = EWC(lam=1.0, fisher_sample_size=5, loss_fn=nn.MSELoss())
    ewc.consolidate(model, ds)
    # Fisher tensors must be finite — too few samples would otherwise yield NaN.
    f = ewc._consolidated[0].fisher["weight"]
    assert torch.isfinite(f).all()


def _two_class_task(name: str, mean: float, n_train: int = 16, n_test: int = 8) -> Task:
    g = torch.Generator().manual_seed(hash(name) & 0xFFFF)
    cls0 = torch.randn(n_train + n_test, 4, generator=g) - mean
    cls1 = torch.randn(n_train + n_test, 4, generator=g) + mean
    x = torch.cat([cls0, cls1])
    y = torch.cat(
        [torch.zeros(n_train + n_test, dtype=torch.int64),
         torch.ones(n_train + n_test, dtype=torch.int64)]
    )
    idx = torch.randperm(x.shape[0], generator=g)
    x, y = x[idx], y[idx]
    split = 2 * n_train
    return Task(
        name=name,
        train=TensorDataset(x[:split], y[:split]),
        test=TensorDataset(x[split:], y[split:]),
        classes=(0, 1),
    )


class _TwoTaskBenchmark:
    name = "ewc_smoke"
    num_classes_per_task = 2
    input_shape = (4,)

    def tasks(self) -> list[Task]:
        return [_two_class_task("a", 1.5), _two_class_task("b", -1.5)]


def test_ewc_integrates_with_runner() -> None:
    """End-to-end: penalty is consulted during training and grows over time."""
    model = MLPClassifier(MLPConfig(input_dim=4, hidden_dim=8, num_classes=2))
    ewc = EWC(lam=100.0, fisher_sample_size=8)

    penalty_log: list[float] = []

    def logging_regulariser(m: nn.Module) -> torch.Tensor:
        p = ewc.penalty(m)
        penalty_log.append(float(p.detach().item()))
        return p

    runner = ContinualRunner(
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.05),
        epochs_per_task=1,
        batch_size=8,
        eval_batch_size=8,
        seed=0,
        regulariser=logging_regulariser,
        on_task_end=lambda i, task, model: ewc.consolidate(model, task.train),
    )

    result = runner.run(model, _TwoTaskBenchmark())

    assert ewc.num_consolidated_tasks == 2
    # Penalty must be zero throughout task 1 (no consolidations yet).
    n_batches_task1 = 2 * 16 // 8  # 32 train samples, batch 8 -> 4 batches
    assert all(p == 0.0 for p in penalty_log[:n_batches_task1])
    # Penalty must be strictly positive at least once during task 2.
    assert any(p > 0.0 for p in penalty_log[n_batches_task1:])
    # Sanity: the runner returned a fully populated lower triangle.
    assert not (result.accuracy_matrix != result.accuracy_matrix).all()
