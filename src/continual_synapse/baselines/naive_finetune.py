"""Naive sequential fine-tuning baseline.

A plain three-hidden-layer MLP trained with cross-entropy on each
task in turn, with no continual-learning protection. This is the
reference against which every other method is compared.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MLPConfig:
    """Hyperparameters for :class:`MLPClassifier`.

    Attributes:
        input_dim: Number of input features (flattened image size).
        hidden_dim: Width of each hidden layer.
        num_classes: Output dimensionality.
        num_hidden_layers: Number of hidden layers (depth). The plan
            specifies three. Kept configurable for sweeps.
        dropout: Dropout probability applied after each hidden ReLU.
            Set to 0 for a deterministic baseline.
    """

    input_dim: int = 784
    hidden_dim: int = 256
    num_classes: int = 2
    num_hidden_layers: int = 3
    dropout: float = 0.0


class MLPClassifier(nn.Module):
    """Three-hidden-layer ReLU MLP used as the Phase-1 backbone.

    The module exposes intermediate hidden activations through
    :meth:`features` so that Phase 2 synapse hooks can target the
    penultimate layer without re-running the forward pass.
    """

    def __init__(self, config: MLPConfig | None = None) -> None:
        super().__init__()
        cfg = config or MLPConfig()
        if cfg.num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")
        self.config = cfg

        layers: list[nn.Module] = []
        in_dim = cfg.input_dim
        for _ in range(cfg.num_hidden_layers):
            layers.append(nn.Linear(in_dim, cfg.hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            in_dim = cfg.hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(cfg.hidden_dim, cfg.num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate-layer activations for input ``x``."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
