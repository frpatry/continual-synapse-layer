"""Base model loaders and PyTorch hook helpers (Phase 2+)."""

from continual_synapse.base_models.hooks import (
    ActivationCapture,
    get_module_by_name,
)

__all__ = ["ActivationCapture", "get_module_by_name"]
