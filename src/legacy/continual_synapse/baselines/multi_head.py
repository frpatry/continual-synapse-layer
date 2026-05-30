"""Multi-head MLP baseline.

A shared trunk plus one classification head per task. The active
head is selected at training and evaluation time via
:meth:`set_active_head`; the runner's ``on_task_change`` callback
is the usual driver.

Multi-head setups isolate the task-discriminative gradient inside
each task's own head, so the trunk receives a much cleaner signal
than the shared-head Phase-3 setup where every task fights over
the same final classifier. This module is the test bed for whether
the shared-head bottleneck is what was hiding any synapse-layer
signal on Split-MNIST.
"""

from __future__ import annotations

import torch
from torch import nn

from continual_synapse.baselines.naive_finetune import MLPConfig


class MultiHeadMLPClassifier(nn.Module):
    """Shared backbone + one head per task.

    Args:
        num_tasks: Number of classification heads to instantiate.
            The active head defaults to ``0`` and is changed by
            :meth:`set_active_head`.
        config: Layer-shape hyperparameters. ``num_classes`` applies
            to each per-task head; the heads are all identical in
            shape but have independent parameters.
    """

    def __init__(
        self,
        num_tasks: int,
        config: MLPConfig | None = None,
    ) -> None:
        super().__init__()
        cfg = config or MLPConfig()
        if cfg.num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")
        if num_tasks <= 0:
            raise ValueError(f"num_tasks must be positive, got {num_tasks}")
        self.config = cfg
        self.num_tasks = int(num_tasks)

        layers: list[nn.Module] = []
        in_dim = cfg.input_dim
        for _ in range(cfg.num_hidden_layers):
            layers.append(nn.Linear(in_dim, cfg.hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            in_dim = cfg.hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.heads = nn.ModuleList(
            [
                nn.Linear(cfg.hidden_dim, cfg.num_classes)
                for _ in range(self.num_tasks)
            ]
        )
        self._active_head: int = 0

    @property
    def active_head(self) -> int:
        return self._active_head

    def set_active_head(self, index: int) -> None:
        """Select which task's head is used by forward/classify."""
        if not 0 <= index < self.num_tasks:
            raise ValueError(
                f"head index {index} out of range [0, {self.num_tasks})"
            )
        self._active_head = int(index)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        return self.heads[self._active_head](features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify(self.features(x))
