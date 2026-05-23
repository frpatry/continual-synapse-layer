"""PyTorch forward-hook utilities for the synapse layer.

The synapse layer needs to observe activations from a chosen layer
of the base model. ``ActivationCapture`` wraps the boilerplate of
registering a forward hook on a named submodule and exposing the
most recent output tensor. It is designed to work uniformly with
small MLPs (Phase 1) and HuggingFace transformer modules (Phase 2+).

Typical usage::

    capture = ActivationCapture(model, "backbone")
    capture.attach()
    try:
        logits = model(x)
        features = capture.activation
    finally:
        capture.detach()

Or as a context manager::

    with ActivationCapture(model, "backbone") as capture:
        logits = model(x)
        features = capture.activation
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle


def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    """Walk a dotted attribute path to retrieve a submodule.

    ``""`` (the empty string) returns ``model`` itself. Numeric path
    components are interpreted as sequence indices, matching the
    behaviour of ``model.named_modules()``.
    """
    if name == "":
        return model
    module: Any = model
    for part in name.split("."):
        if part.isdigit() and isinstance(module, (nn.Sequential, nn.ModuleList)):
            module = module[int(part)]
        else:
            if not hasattr(module, part):
                raise AttributeError(
                    f"Module {type(module).__name__!r} has no attribute {part!r} "
                    f"(while resolving {name!r})"
                )
            module = getattr(module, part)
    if not isinstance(module, nn.Module):
        raise TypeError(
            f"Resolved {name!r} is {type(module).__name__}, not nn.Module"
        )
    return module


class ActivationCapture:
    """Capture the output of a named submodule via a forward hook.

    Attributes:
        model: The model whose submodule will be hooked.
        target_name: Dotted name of the submodule to observe.
        detach: If True (default), the captured tensor is detached
            from autograd. Hebbian updates do not need gradients
            through the activation, so detaching saves memory.
        clone: If True (default), the captured tensor is cloned so
            that subsequent in-place ops on the model's intermediate
            buffers do not corrupt it. Set False for a small speedup
            when you know no such ops will happen.
    """

    def __init__(
        self,
        model: nn.Module,
        target_name: str,
        *,
        detach: bool = True,
        clone: bool = True,
    ) -> None:
        self.model = model
        self.target_name = target_name
        self.detach = detach
        self.clone = clone
        self._handle: RemovableHandle | None = None
        self._activation: torch.Tensor | None = None

    @property
    def activation(self) -> torch.Tensor:
        """Most recent output tensor of the hooked module.

        Raises ``RuntimeError`` if no forward pass has occurred since
        the hook was attached, so the caller catches mis-ordered usage
        instead of silently consuming a stale ``None``.
        """
        if self._activation is None:
            raise RuntimeError(
                f"No activation captured yet for {self.target_name!r}. "
                "Run a forward pass with the hook attached first."
            )
        return self._activation

    @property
    def is_attached(self) -> bool:
        return self._handle is not None

    def attach(self) -> "ActivationCapture":
        """Register the forward hook. Idempotent: re-attach is a no-op."""
        if self._handle is not None:
            return self
        target = get_module_by_name(self.model, self.target_name)
        self._handle = target.register_forward_hook(self._hook)
        return self

    def detach_hook(self) -> None:
        """Remove the forward hook. Safe to call repeatedly."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def clear(self) -> None:
        """Drop the cached activation without removing the hook."""
        self._activation = None

    def _hook(self, _module: nn.Module, _inputs: Any, output: Any) -> None:
        if isinstance(output, tuple):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                f"Hook on {self.target_name!r} expected a Tensor output, "
                f"got {type(output).__name__}"
            )
        tensor = output
        if self.detach:
            tensor = tensor.detach()
        if self.clone:
            tensor = tensor.clone()
        self._activation = tensor

    def __enter__(self) -> "ActivationCapture":
        return self.attach()

    def __exit__(self, *_exc: Any) -> None:
        self.detach_hook()
