"""Elastic Weight Consolidation (Kirkpatrick et al., 2017).

EWC penalises parameter drift away from values that were important
for previous tasks. After training task ``t``, we estimate the
diagonal of the empirical Fisher Information matrix on that task's
data and snapshot the current parameter values. During subsequent
training, a quadratic penalty is added to the loss:

    L_total(θ) = L_data(θ) + (λ / 2) * Σ_t Σ_i F_{t,i} * (θ_i - θ*_{t,i})²

Where ``θ*_t`` are the parameters at the end of task ``t`` and
``F_t`` is the diagonal empirical Fisher on task ``t``.

This module is wired into :class:`ContinualRunner` via two hook
points: ``regulariser`` (called per batch to add the penalty term)
and ``on_task_end`` (called after each task's training to estimate
Fisher and snapshot parameters). See ``experiments/02_ewc_baseline.py``
for a complete usage example.

Notes:
- We store Fisher and a parameter snapshot per consolidated task.
  For ``T`` tasks and ``P`` parameters this is ``O(T * P)`` memory.
  Online-EWC variants accumulate into a single running Fisher; we
  keep the per-task form to match the original paper.
- Fisher is estimated *sample-by-sample* with the true labels
  (``empirical Fisher``). Sample-by-sample is required because the
  Fisher is an expectation of *per-sample* squared gradients;
  batching and then squaring would give a biased estimator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class _Consolidated:
    """One consolidated task's Fisher diagonal and parameter snapshot.

    Both dicts are keyed by ``model.named_parameters()`` names and
    contain detached tensors on the same device as the model.
    """

    fisher: dict[str, torch.Tensor]
    star: dict[str, torch.Tensor]


@dataclass
class EWC:
    """Elastic Weight Consolidation regulariser.

    Attributes:
        lam: Penalty coefficient ``λ``. The original paper uses
            values in the thousands for MNIST; tune per benchmark.
        fisher_sample_size: Cap on samples drawn from each task's
            training set during Fisher estimation. ``None`` uses the
            full set. Sub-sampling speeds up consolidation with
            limited accuracy cost in practice.
        device: Device on which Fisher tensors are stored.
        loss_fn: Loss used to compute the log-likelihood gradient
            for Fisher estimation. Defaults to cross-entropy, which
            matches every classification benchmark in this project.
    """

    lam: float
    fisher_sample_size: int | None = None
    device: str = "cpu"
    loss_fn: nn.Module = field(default_factory=nn.CrossEntropyLoss)
    _consolidated: list[_Consolidated] = field(default_factory=list)

    @property
    def num_consolidated_tasks(self) -> int:
        return len(self._consolidated)

    def consolidate(self, model: nn.Module, dataset: Dataset) -> None:
        """Estimate Fisher on ``dataset`` and snapshot the model's params.

        Called after each task's training loop. The model is briefly
        put in ``eval`` mode so that dropout / batchnorm noise does
        not pollute the Fisher estimate, then restored.
        """
        named = {
            n: p for n, p in model.named_parameters() if p.requires_grad
        }
        if not named:
            raise ValueError("Model has no trainable parameters")

        fisher: dict[str, torch.Tensor] = {
            n: torch.zeros_like(p, device=self.device) for n, p in named.items()
        }
        star: dict[str, torch.Tensor] = {
            n: p.detach().clone().to(self.device) for n, p in named.items()
        }

        was_training = model.training
        model.eval()
        loader = DataLoader(dataset, batch_size=1, shuffle=True)

        n_used = 0
        for x, y in loader:
            if (
                self.fisher_sample_size is not None
                and n_used >= self.fisher_sample_size
            ):
                break
            x = x.to(self.device)
            y = y.to(self.device)
            model.zero_grad(set_to_none=True)
            loss = self.loss_fn(model(x), y)
            loss.backward()
            for n, p in named.items():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
            n_used += 1

        if n_used == 0:
            raise ValueError("Fisher estimation saw zero samples")
        for n in fisher:
            fisher[n] /= float(n_used)

        model.zero_grad(set_to_none=True)
        if was_training:
            model.train()

        self._consolidated.append(_Consolidated(fisher=fisher, star=star))

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Return ``(λ/2) * Σ_t Σ_i F_{t,i} * (θ_i - θ*_{t,i})²``.

        When no task has been consolidated yet, returns a zero scalar
        on the regulariser's device. The returned tensor participates
        in autograd through the current model parameters.
        """
        if not self._consolidated:
            return torch.zeros((), device=self.device)

        params = dict(model.named_parameters())
        total = torch.zeros((), device=self.device)
        for entry in self._consolidated:
            for name, f in entry.fisher.items():
                if name not in params:
                    continue
                diff = params[name] - entry.star[name]
                total = total + (f * diff.pow(2)).sum()
        return 0.5 * self.lam * total

    def __call__(self, model: nn.Module) -> torch.Tensor:
        return self.penalty(model)
