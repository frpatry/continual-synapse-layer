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
        retrieval_refresh_interval: int = 1,
    ) -> None:
        super().__init__()
        if synapse.n_neurons != base.config.hidden_dim:
            raise ValueError(
                f"SynapseLayer n_neurons={synapse.n_neurons} does not match "
                f"base.config.hidden_dim={base.config.hidden_dim}"
            )
        if retrieval_k <= 0:
            raise ValueError(f"retrieval_k must be positive, got {retrieval_k}")
        if retrieval_refresh_interval <= 0:
            raise ValueError(
                f"retrieval_refresh_interval must be positive, "
                f"got {retrieval_refresh_interval}"
            )
        self.base = base
        self.synapse = synapse
        self.modulator = modulator if modulator is not None else SynapseModulation()
        self.reward_computer = reward_computer
        self.cold_storage = cold_storage
        self.consolidation_trigger = consolidation_trigger
        self.retrieval_k = int(retrieval_k)
        self.retrieval_refresh_interval = int(retrieval_refresh_interval)
        self._last_features: torch.Tensor | None = None
        self._consolidation_count: int = 0
        # Retrieval cache: avoid querying Chroma every forward pass.
        # `retrieval_refresh_interval=N` means refresh-then-reuse-(N-1)
        # times. So `N=1` refreshes every forward (no caching);
        # `N=4` refreshes on forwards 1, 5, 9, ...
        self._retrieved_cache: torch.Tensor | None = None
        self._reuses_remaining: int = 0
        self._cache_invalidated: bool = True

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
            retrieved = self._get_or_refresh_retrieval(f_base)
            effective_strengths = self.synapse.strengths + retrieved
        else:
            effective_strengths = self.synapse.strengths

        return f_base + self.modulator(f_base, effective_strengths)

    def _get_or_refresh_retrieval(
        self, f_base: torch.Tensor
    ) -> torch.Tensor:
        """Return the cached cold-storage strengths, refreshing if stale.

        Refresh fires when:
        - The cache has been invalidated (post-consolidation or never set).
        - We have used up the configured number of reuses.

        This is the hot path on the cold-storage variant; the cache
        avoids one Chroma query per forward, which is the dominant
        per-batch cost when the store has hundreds of entries.
        """
        needs_refresh = (
            self._cache_invalidated
            or self._retrieved_cache is None
            or self._reuses_remaining <= 0
        )
        if needs_refresh:
            with torch.no_grad():
                query = f_base.detach().mean(dim=0)
                self._retrieved_cache = reconstruct_strengths(
                    self.cold_storage,  # type: ignore[arg-type]
                    query,
                    k=self.retrieval_k,
                    n_neurons=self.synapse.n_neurons,
                )
            # After this refresh, allow `interval - 1` reuses before
            # the next refresh. `interval = 1` means zero reuses, i.e.
            # refresh on every forward.
            self._reuses_remaining = max(self.retrieval_refresh_interval - 1, 0)
            self._cache_invalidated = False
        else:
            self._reuses_remaining -= 1
        assert self._retrieved_cache is not None
        return self._retrieved_cache

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
                # Mark the retrieval cache as stale so the next
                # forward picks up the just-archived pattern.
                self._cache_invalidated = True

        self._last_features = None
        return reward

    @property
    def consolidation_count(self) -> int:
        """Number of consolidation cycles that have fired on this instance."""
        return self._consolidation_count
