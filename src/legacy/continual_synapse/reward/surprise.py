"""Surprise reward component — prediction error from a tiny online model.

DESIGN.md §3.4 calls for "prediction error from a small auxiliary
model". This module implements that with a single linear predictor
that learns to forecast the next batch's mean activation from the
previous batch's mean activation. Each call:

1. Predict the current mean activation from the cached previous one.
2. Compute the cosine-distance surprise (clipped to ``[0, 1]``).
3. Take one SGD step on the predictor's MSE loss against the truth.
4. Cache the current activation as "previous" for the next call.

The first call has no previous observation; it returns ``0.0`` and
seeds the cache. The second call uses the zero-initialised
predictor, which produces a zero prediction; cosine distance is
ill-defined against a zero vector, so we return the maximum
surprise (``1.0``).

The predictor is a separate :class:`torch.nn.Linear` with its own
optimizer, deliberately isolated from the main model's parameters
and from any autograd graph that built the activations. The
activations passed in are detached before they ever touch the
predictor.
"""

from __future__ import annotations

import torch
from torch import nn


_EPS = 1e-8


class SurpriseReward(nn.Module):
    """Online prediction-error reward.

    Args:
        n_neurons: Width of the activation vector being observed.
        predictor_lr: SGD learning rate for the online predictor.
            ``0.01`` is a reasonable starting value; lower values
            make surprise decay more slowly after a regime change.
    """

    def __init__(self, n_neurons: int, predictor_lr: float = 0.01) -> None:
        super().__init__()
        if n_neurons <= 0:
            raise ValueError("n_neurons must be positive")
        if predictor_lr <= 0:
            raise ValueError(
                f"predictor_lr must be positive, got {predictor_lr}"
            )
        self.n_neurons = int(n_neurons)
        self.predictor_lr = float(predictor_lr)
        self.predictor = nn.Linear(n_neurons, n_neurons, bias=False)
        with torch.no_grad():
            self.predictor.weight.zero_()
        self._optimizer = torch.optim.SGD(
            self.predictor.parameters(), lr=self.predictor_lr
        )
        self._prev: torch.Tensor | None = None

    @property
    def has_history(self) -> bool:
        return self._prev is not None

    def reset(self) -> None:
        """Drop history and re-zero the predictor weights."""
        self._prev = None
        with torch.no_grad():
            self.predictor.weight.zero_()

    def __call__(self, activations: torch.Tensor) -> float:
        if activations.ndim != 2:
            raise ValueError(
                f"activations must be 2-D (B, n), got shape "
                f"{tuple(activations.shape)}"
            )
        if activations.shape[1] != self.n_neurons:
            raise ValueError(
                f"activation dim {activations.shape[1]} does not match "
                f"n_neurons={self.n_neurons}"
            )

        current = activations.detach().mean(dim=0)

        if self._prev is None:
            self._prev = current.clone()
            return 0.0

        prev = self._prev.detach().clone()
        with torch.no_grad():
            pred = self.predictor(prev.unsqueeze(0)).squeeze(0)
            surprise = _cosine_distance(pred, current)

        # The predictor needs gradient flow for its own SGD step. This
        # call site is typically reached from inside a `@torch.no_grad`
        # block (the augmented MLP's apply_hebbian_update), so we
        # explicitly re-enable autograd just for the predictor update.
        with torch.enable_grad():
            self._optimizer.zero_grad()
            pred_for_loss = self.predictor(prev.unsqueeze(0)).squeeze(0)
            loss = (pred_for_loss - current).pow(2).mean()
            loss.backward()
            self._optimizer.step()

        self._prev = current.clone()
        return float(surprise)


def _cosine_distance(pred: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
    """Return ``1 - max(0, cos_sim(pred, actual))`` so the result is in [0, 1].

    Defined as the maximum surprise (``1.0``) when either vector has
    near-zero norm — that's the early-training case where the
    predictor has not yet learned anything useful.
    """
    pred_norm = pred.norm()
    actual_norm = actual.norm()
    if pred_norm.item() < _EPS or actual_norm.item() < _EPS:
        return torch.tensor(1.0)
    sim = (pred * actual).sum() / (pred_norm * actual_norm)
    return 1.0 - torch.clamp(sim, min=0.0)
