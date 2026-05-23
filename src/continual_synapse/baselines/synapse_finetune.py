"""MLP backbone augmented with a SynapseLayer and modulator.

Composes the Phase-1 :class:`MLPClassifier` with the Phase-2
:class:`SynapseLayer` and :class:`SynapseModulation`. The augmented
model is interchangeable with the base MLP in the runner: same
``forward(x) -> logits`` signature, same ``.features(x)`` accessor.
The Hebbian update is triggered explicitly by the trainer via
:meth:`apply_hebbian_update`, which the runner calls per batch
through its ``on_after_batch`` hook.

Read-out path::

    f_base    = base.features(x)            # untouched MLP backbone
    correct   = mod(f_base, syn.strengths)  # gate * (f_base @ S)
    logits    = base.head(f_base + correct)

Hebbian path (after the optimizer step)::

    syn.consolidate(f_base.detach(), reward)

Hebbian observation uses the *pre-correction* base activations so
the synapse layer records raw co-activations of the base model
rather than self-reinforcing its own correction.
"""

from __future__ import annotations

import torch
from torch import nn

from continual_synapse.baselines.naive_finetune import MLPClassifier
from continual_synapse.synapse_layer.layer import SynapseLayer
from continual_synapse.synapse_layer.modulation import SynapseModulation


class SynapseAugmentedMLP(nn.Module):
    """:class:`MLPClassifier` with an additive synapse correction.

    The base model's parameters remain optimisable like any other
    backbone; the synapse strengths are Hebbian-only; the modulator
    gate is gradient-trained.

    Args:
        base: The underlying MLP classifier.
        synapse: State container whose ``n_neurons`` must equal
            ``base.config.hidden_dim``.
        modulator: Read-out that turns synapse state into a
            correction. Defaults to a fresh ``SynapseModulation()``
            with ``gate=0`` so the model is functionally identical
            to the base MLP at init.
    """

    def __init__(
        self,
        base: MLPClassifier,
        synapse: SynapseLayer,
        modulator: SynapseModulation | None = None,
    ) -> None:
        super().__init__()
        if synapse.n_neurons != base.config.hidden_dim:
            raise ValueError(
                f"SynapseLayer n_neurons={synapse.n_neurons} does not match "
                f"base.config.hidden_dim={base.config.hidden_dim}"
            )
        self.base = base
        self.synapse = synapse
        self.modulator = modulator if modulator is not None else SynapseModulation()
        self._last_features: torch.Tensor | None = None

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return base features plus the synapse correction.

        The features cache used for Hebbian updates records the
        pre-correction base output; downstream callers that want
        the corrected representation use this method's return value.
        """
        f_base = self.base.features(x)
        self._last_features = f_base.detach()
        return f_base + self.modulator(f_base, self.synapse.strengths)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.head(self.features(x))

    @torch.no_grad()
    def apply_hebbian_update(self, reward: float = 1.0) -> None:
        """Push the most recently observed features into the synapse layer.

        Must be called after a forward pass (training or evaluation).
        Raises if no features have been cached yet so the caller
        notices a missing forward instead of silently skipping the
        update.
        """
        if self._last_features is None:
            raise RuntimeError(
                "No features cached. Run a forward pass before calling "
                "apply_hebbian_update()."
            )
        self.synapse.consolidate(self._last_features, reward=reward)
        self._last_features = None
