"""MLP backbone augmented with a SynapseLayer, modulator, and optional cold storage.

Composes the Phase-1 :class:`MLPClassifier` (or any base model with
``features`` / ``classify`` methods) with the Phase-2
:class:`SynapseLayer` + :class:`SynapseModulation` and the Phase-4
:class:`ColdStorage` + :class:`ConsolidationTrigger`.

Read-out path (cold storage off — Phase 2/3 behaviour)::

    f_base    = base.features(x)
    correct   = mod(f_base, syn.strengths)
    logits    = base.classify(f_base + correct)

Read-out path (cold storage on — Phase 4)::

    f_base    = base.features(x)
    retrieved = reconstruct_strengths(store, f_base.mean(0), k, n_neurons)
    correct   = mod(f_base, syn.strengths + retrieved)
    logits    = base.classify(f_base + correct)

The retrieved strengths are added to (not multiplied with) the
working-memory strengths so the modulator's gate continues to
scale the whole correction uniformly. Retrieval is *context-
dependent*: the query is the current batch's mean pre-correction
activation, so similar inputs recover their archived patterns and
different inputs do not.

Hebbian path (after the optimizer step)::

    syn.record_access(f_base.detach())
    syn.consolidate(f_base.detach(), reward)
    consolidate_to_storage(syn, store, trigger, f_base.mean(0))  # optional

Hebbian observation uses the *pre-correction* base activations so
the synapse layer records raw co-activations of the base model
rather than self-reinforcing its own correction.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from continual_synapse.baselines.naive_finetune import MLPClassifier
from continual_synapse.cold_storage.store import ColdStorage
from continual_synapse.consolidation.pipeline import consolidate_to_storage
from continual_synapse.consolidation.reconstruction import reconstruct_strengths
from continual_synapse.consolidation.trigger import ConsolidationTrigger
from continual_synapse.synapse_layer.layer import SynapseLayer
from continual_synapse.synapse_layer.modulation import SynapseModulation


RewardComputer = Callable[[torch.Tensor], float]


class SynapseAugmentedMLP(nn.Module):
    """MLP base composed with synapse layer, modulator, and optional cold storage.

    The optional pieces are independent: experiments can wire up the
    bare synapse (Phase 2), synapse + reward (Phase 3), or synapse +
    reward + cold storage (Phase 4) without changing the model class.

    Args:
        base: The underlying classifier (e.g. ``MLPClassifier`` or
            ``MultiHeadMLPClassifier``).
        synapse: Working-memory state. ``n_neurons`` must match
            ``base.config.hidden_dim``.
        modulator: Read-out layer.
        reward_computer: Optional reward callable for the Hebbian
            update.
        cold_storage: Optional long-term archive (Phase 4). When
            present, ``features()`` augments strengths with a
            context-dependent retrieval.
        consolidation_trigger: Optional trigger that, together with
            ``cold_storage``, decides when consolidation cycles
            fire from ``apply_hebbian_update``.
        retrieval_k: Number of cold-storage entries combined per
            forward pass when cold storage is active.
    """

    def __init__(
        self,
        base: MLPClassifier,
        synapse: SynapseLayer,
        modulator: SynapseModulation | None = None,
        reward_computer: RewardComputer | None = None,
        cold_storage: ColdStorage | None = None,
        consolidation_trigger: ConsolidationTrigger | None = None,
        retrieval_k: int = 4,
    ) -> None:
        super().__init__()
        if synapse.n_neurons != base.config.hidden_dim:
            raise ValueError(
                f"SynapseLayer n_neurons={synapse.n_neurons} does not match "
                f"base.config.hidden_dim={base.config.hidden_dim}"
            )
        if retrieval_k <= 0:
            raise ValueError(f"retrieval_k must be positive, got {retrieval_k}")
        self.base = base
        self.synapse = synapse
        self.modulator = modulator if modulator is not None else SynapseModulation()
        self.reward_computer = reward_computer
        self.cold_storage = cold_storage
        self.consolidation_trigger = consolidation_trigger
        self.retrieval_k = int(retrieval_k)
        self._last_features: torch.Tensor | None = None
        self._consolidation_count: int = 0

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return base features plus the (optionally cold-augmented) correction.

        The cached features fed into Hebbian updates are the
        pre-correction base output. Retrieval queries cold storage
        with the current batch's mean activation so the correction
        is context-dependent rather than universal.
        """
        f_base = self.base.features(x)
        self._last_features = f_base.detach()

        if self.cold_storage is not None and self.cold_storage.count() > 0:
            with torch.no_grad():
                query = f_base.detach().mean(dim=0)
                retrieved = reconstruct_strengths(
                    self.cold_storage,
                    query,
                    k=self.retrieval_k,
                    n_neurons=self.synapse.n_neurons,
                )
            effective_strengths = self.synapse.strengths + retrieved
        else:
            effective_strengths = self.synapse.strengths

        return f_base + self.modulator(f_base, effective_strengths)

    def set_active_head(self, index: int) -> None:
        """Forward head selection to the base model, if it supports it."""
        if not hasattr(self.base, "set_active_head"):
            raise AttributeError(
                f"base model {type(self.base).__name__} does not "
                "support multi-head selection"
            )
        self.base.set_active_head(index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base.classify(self.features(x))

    @torch.no_grad()
    def apply_hebbian_update(self, reward: float | None = None) -> float:
        """Push the most recently observed features into the synapse layer.

        When cold storage and a trigger are configured, also attempt
        a consolidation cycle. The cycle is a no-op if the trigger
        declines.
        """
        if self._last_features is None:
            raise RuntimeError(
                "No features cached. Run a forward pass before calling "
                "apply_hebbian_update()."
            )
        features = self._last_features
        if reward is None:
            if self.reward_computer is not None:
                reward = float(self.reward_computer(features))
            else:
                reward = 1.0
        self.synapse.record_access(features)
        self.synapse.consolidate(features, reward=reward)

        if (
            self.cold_storage is not None
            and self.consolidation_trigger is not None
        ):
            embedding = features.mean(dim=0).to(torch.float32)
            entry_id = consolidate_to_storage(
                self.synapse,
                self.cold_storage,
                self.consolidation_trigger,
                activation_embedding=embedding,
            )
            if entry_id is not None:
                self._consolidation_count += 1

        self._last_features = None
        return reward

    @property
    def consolidation_count(self) -> int:
        """Number of consolidation cycles that have fired on this instance."""
        return self._consolidation_count
