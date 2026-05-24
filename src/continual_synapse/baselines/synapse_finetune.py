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
from continual_synapse.cold_storage.compression import CompressionSchedule
from continual_synapse.cold_storage.store import ColdStorage
from continual_synapse.consolidation.pipeline import consolidate_to_storage
from continual_synapse.consolidation.reconstruction import reconstruct_strengths
from continual_synapse.consolidation.trigger import ConsolidationTrigger
from continual_synapse.reward.external import ExternalReward
from continual_synapse.reward.mixer import RewardMixer
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
        n_passes: Per-batch multi-pass count for noise-filtered
            co-activation (PROJECT_PLAN.md §4.2.1). Default ``1``
            reproduces single-pass Phase-3 behaviour bit-exact. When
            ``> 1`` *and* the model is in training mode, ``features()``
            performs ``n_passes`` forwards on the same input, pushes
            each pre-correction observation through ``synapse.observe``,
            and lets ``synapse.consolidate`` average them. The
            noise-filtering benefit only materialises when forwards
            are stochastic (e.g. dropout enabled) — for a fully
            deterministic forward, ``n_passes > 1`` produces ``n_passes``
            identical observations whose average equals a single pass.
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
        n_passes: int = 1,
        compression_sweep_interval: int = 0,
        compression_schedule: CompressionSchedule | None = None,
        # ---- Amplification variant flags (defaults preserve current behavior) ----
        amplification_alpha: float = 0.0,
        confidence_exponent: float = 0.0,
        no_drain_on_consolidate: bool = False,
        repeat_consolidation_threshold: float = 1.0,
        retrieval_feedback_threshold: float = 0.0,
        retrieval_feedback_decay: float = 0.95,
        retrieval_feedback_bump: float = 0.5,
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
        if n_passes <= 0:
            raise ValueError(f"n_passes must be positive, got {n_passes}")
        if compression_sweep_interval < 0:
            raise ValueError(
                f"compression_sweep_interval must be >= 0, got "
                f"{compression_sweep_interval}"
            )
        if amplification_alpha < 0:
            raise ValueError(
                f"amplification_alpha must be >= 0, got {amplification_alpha}"
            )
        if confidence_exponent < 0:
            raise ValueError(
                f"confidence_exponent must be >= 0, got {confidence_exponent}"
            )
        if not 0.0 < repeat_consolidation_threshold <= 1.0:
            raise ValueError(
                f"repeat_consolidation_threshold must be in (0, 1], "
                f"got {repeat_consolidation_threshold}"
            )
        if retrieval_feedback_threshold < 0:
            raise ValueError(
                f"retrieval_feedback_threshold must be >= 0, got "
                f"{retrieval_feedback_threshold}"
            )
        if not 0.0 < retrieval_feedback_decay <= 1.0:
            raise ValueError(
                f"retrieval_feedback_decay must be in (0, 1], got "
                f"{retrieval_feedback_decay}"
            )
        self.base = base
        self.synapse = synapse
        self.modulator = modulator if modulator is not None else SynapseModulation()
        self.reward_computer = reward_computer
        self.cold_storage = cold_storage
        self.consolidation_trigger = consolidation_trigger
        self.retrieval_k = int(retrieval_k)
        self.retrieval_refresh_interval = int(retrieval_refresh_interval)
        self.n_passes = int(n_passes)
        # Periodic compression sweep. 0 (the default) disables the
        # sweep entirely, reproducing the Phase-4b behaviour where
        # every entry stayed at 32-bit forever. Non-zero values
        # trigger ColdStorage.re_evaluate_all_entries every N training
        # batches (counted from the synapse's global_step deltas).
        self.compression_sweep_interval = int(compression_sweep_interval)
        self.compression_schedule = (
            compression_schedule
            if compression_schedule is not None
            else CompressionSchedule()
        )
        # Amplification variant flags. When defaults are kept, every code
        # path below is bit-exact equivalent to the pre-amplification
        # implementation — verified by the existing test suite plus the
        # dedicated defaults-preserve-behavior tests.
        self.amplification_alpha = float(amplification_alpha)
        self.confidence_exponent = float(confidence_exponent)
        self.no_drain_on_consolidate = bool(no_drain_on_consolidate)
        self.repeat_consolidation_threshold = float(repeat_consolidation_threshold)
        self.retrieval_feedback_threshold = float(retrieval_feedback_threshold)
        self.retrieval_feedback_decay = float(retrieval_feedback_decay)
        self.retrieval_feedback_bump = float(retrieval_feedback_bump)
        # Per-call buffer of (entry_id, pre_bump_access_count) tuples filled
        # by _get_or_refresh_retrieval when a refresh fires. Cleared by
        # apply_hebbian_update after any retrieval-feedback bookkeeping.
        self._last_retrieved_meta: list[tuple[int, int]] = []
        self._merge_count: int = 0
        # EMA of the per-batch loss, fed by apply_hebbian_update(loss=...)
        # (or by the cached-logits fallback below). Compared against the
        # current batch's loss to decide whether the just-retrieved
        # entries deserve a "you helped" access_count bump.
        self._loss_ema: float | None = None
        self._retrieval_feedback_event_count: int = 0
        self._batches_since_compression_sweep: int = 0
        self._compression_sweep_count: int = 0
        self._last_compression_counts: dict[int, int] = {}
        self._last_features: torch.Tensor | None = None
        # Detached training-mode logits from the most recent forward.
        # Used by `apply_hebbian_update(training_target=...)` to drive
        # an ExternalReward from per-batch accuracy without re-running
        # the forward pass. Set to None in eval mode so callers that
        # forget to switch back to train can't accidentally feed
        # eval-time logits into the training reward.
        self._last_logits: torch.Tensor | None = None
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

        When ``n_passes > 1`` and the module is in ``training`` mode,
        this method also performs ``n_passes`` forward passes on
        ``x`` to populate the synapse layer's observation buffer.
        The pass that's returned (and that contributes to the loss)
        is the first one; the remaining ``n_passes - 1`` are
        observation-only re-runs.
        """
        f_base = self.base.features(x)
        self._last_features = f_base.detach()

        # Multi-pass observation path (training mode only). Even at
        # n_passes=1 we deliberately do NOT call observe(), to keep
        # the buffer-empty path bit-exact compatible with Phase 3.
        in_multi_pass = self.training and self.n_passes > 1
        if in_multi_pass:
            self.synapse.observe(f_base.detach())
            for _ in range(self.n_passes - 1):
                f_extra = self.base.features(x)
                self.synapse.observe(f_extra.detach())
            # Signal to apply_hebbian_update that the buffer is the
            # source of truth; the cached _last_features is no longer
            # representative of what we'll consolidate.
            self._last_features = None

        if self.cold_storage is not None and self.cold_storage.count() > 0:
            # Multi-pass query consistency (audit fix 3/3): when the
            # Hebbian update will consume the buffer-averaged
            # activations, the retrieval query should match — using
            # the noisy first-forward mean here was an architectural
            # inconsistency between read and write paths.
            query_source = (
                self.synapse.buffer_average() if in_multi_pass else f_base
            )
            retrieved = self._get_or_refresh_retrieval(query_source)
            if self.amplification_alpha == 0.0:
                # Default additive composition — bit-exact equivalent to
                # the pre-amplification path.
                effective_strengths = self.synapse.strengths + retrieved
            else:
                # Multiplicative amplification: the retrieval modulates
                # the existing strengths up or down rather than adding a
                # second pattern on top. Normalising `retrieved` to
                # [-1, +1] (max-abs scaling) keeps the multiplier in
                # [1 - alpha, 1 + alpha], so alpha = 1 corresponds to
                # "double the strength where retrieved is fully positive
                # and zero it where retrieved is fully negative". The
                # working-memory pattern is what actually gets scaled;
                # cold storage acts as a gain map.
                max_abs = retrieved.abs().max().clamp_min(1e-8)
                retrieved_normalized = retrieved / max_abs
                effective_strengths = self.synapse.strengths * (
                    1.0 + self.amplification_alpha * retrieved_normalized
                )
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
                # Capture (entry_id, pre_bump_access_count) for any
                # downstream consumer (retrieval-success feedback, per-task
                # average-access-count diagnostics). The retrieval cache
                # logic below is unchanged.
                retrieved_meta: list[tuple[str, int]] = []
                self._retrieved_cache = reconstruct_strengths(
                    self.cold_storage,  # type: ignore[arg-type]
                    query,
                    k=self.retrieval_k,
                    n_neurons=self.synapse.n_neurons,
                    confidence_exponent=self.confidence_exponent,
                    out_retrieved_meta=retrieved_meta,
                )
                self._last_retrieved_meta = retrieved_meta
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
        logits = self.base.classify(self.features(x))
        if self.training:
            self._last_logits = logits.detach()
        else:
            # Eval forward — leave any cached training logits alone but
            # do not overwrite with eval-time values. The cache is only
            # meaningful for the next apply_hebbian_update call, which
            # only fires from on_after_batch during training.
            pass
        return logits

    @torch.no_grad()
    def apply_hebbian_update(
        self,
        reward: float | None = None,
        training_target: torch.Tensor | None = None,
        loss: float | None = None,
    ) -> float:
        """Push the most recently observed features into the synapse layer.

        Multi-pass mode (``n_passes > 1``): the synapse's observation
        buffer was populated by ``features()`` during this batch's
        forwards; we hand off to ``synapse.consolidate(reward=...)``
        with no explicit activations so it averages the buffer. The
        reward computer and ``record_access`` use the same average so
        every downstream signal sees the same denoised features.

        Single-pass mode (``n_passes == 1``, the default): pass the
        cached ``_last_features`` to ``consolidate(activations=...,
        reward=...)``. Bit-exact Phase-3 behaviour.

        Args:
            reward: Explicit reward. When given, bypasses both the
                training-target accuracy path and the reward_computer.
            training_target: Optional ``(B,)`` int64 labels for the
                most recent training batch. When supplied AND the
                reward_computer is a :class:`RewardMixer` with an
                :class:`ExternalReward`, the model computes per-batch
                training accuracy from the cached logits and pushes it
                into the external reward source BEFORE the mixer reads
                it. This makes the audit-flagged "external reward is
                always 1.0" pathway carry a genuine signal. No-op
                when the reward_computer is anything else or the
                training_target is None.
            loss: Explicit per-batch loss for retrieval-success
                feedback (amplification variant change 5). When
                ``retrieval_feedback_threshold > 0`` and this batch's
                loss falls below ``EMA(loss) * threshold``, the entries
                that contributed to this batch's retrieval get a
                small access_count bump — a "you helped" signal. If
                ``loss`` is None but ``training_target`` and cached
                logits are both available, the model computes
                cross-entropy itself so callers don't have to pass it
                explicitly. No-op when the threshold is zero or no
                retrieval happened this batch.

        When cold storage and a trigger are configured, also attempt
        a consolidation cycle. The cycle is a no-op if the trigger
        declines.
        """
        # Pick the features to use for reward / access-counting /
        # consolidation. Multi-pass: average the buffer. Single-pass:
        # use the cached features.
        use_buffer = self.synapse.buffer_size > 0
        if use_buffer:
            features = self.synapse.buffer_average()
        else:
            if self._last_features is None:
                raise RuntimeError(
                    "No features cached. Run a forward pass before calling "
                    "apply_hebbian_update()."
                )
            features = self._last_features

        # If the caller passed training labels and the reward stack is
        # configured to take external feedback, drive external from
        # per-batch accuracy. This must happen BEFORE the mixer is
        # invoked so the updated value is what gets blended.
        if (
            training_target is not None
            and self._last_logits is not None
            and isinstance(self.reward_computer, RewardMixer)
            and isinstance(self.reward_computer.external, ExternalReward)
        ):
            preds = self._last_logits.argmax(dim=1)
            target = training_target.to(preds.device)
            if preds.shape == target.shape and preds.numel() > 0:
                acc = float((preds == target).float().mean().item())
                self.reward_computer.external.set(acc)

        if reward is None:
            if self.reward_computer is not None:
                reward = float(self.reward_computer(features))
            else:
                reward = 1.0
        self.synapse.record_access(features)

        if use_buffer:
            # Let the layer drain its own buffer.
            self.synapse.consolidate(reward=reward)
        else:
            self.synapse.consolidate(features, reward=reward)

        if (
            self.cold_storage is not None
            and self.consolidation_trigger is not None
        ):
            embedding = features.mean(dim=0).to(torch.float32)
            outcome = consolidate_to_storage(
                self.synapse,
                self.cold_storage,
                self.consolidation_trigger,
                activation_embedding=embedding,
                drain=not self.no_drain_on_consolidate,
                merge_threshold=self.repeat_consolidation_threshold,
            )
            if outcome.fired:
                self._consolidation_count += 1
                if outcome.was_merged:
                    self._merge_count += 1
                # Mark the retrieval cache as stale so the next
                # forward picks up the just-archived pattern.
                self._cache_invalidated = True

        # Periodic compression sweep (Phase 4b follow-up). Fires every
        # `compression_sweep_interval` apply_hebbian_update calls when
        # cold storage is configured. Without this sweep, the compression
        # schedule's tier transitions never happen — entries stay at
        # the freshly-stored precision forever (see decisions_log entry
        # "Architectural completion Part 2").
        if (
            self.cold_storage is not None
            and self.compression_sweep_interval > 0
        ):
            self._batches_since_compression_sweep += 1
            if (
                self._batches_since_compression_sweep
                >= self.compression_sweep_interval
            ):
                current_step = int(self.synapse.global_step.item())
                self._last_compression_counts = (
                    self.cold_storage.re_evaluate_all_entries(
                        current_step=current_step,
                        schedule=self.compression_schedule,
                    )
                )
                self._batches_since_compression_sweep = 0
                self._compression_sweep_count += 1
                # Retrieval cache may hold a tensor decoded from the
                # old precision; invalidate so the next forward
                # re-fetches from the freshly-quantised entries.
                self._cache_invalidated = True

        # Retrieval-success feedback (amplification variant change 5).
        # Skipped entirely at the default threshold of 0.0.
        if (
            self.retrieval_feedback_threshold > 0.0
            and self.cold_storage is not None
            and self._last_retrieved_meta
        ):
            effective_loss: float | None = loss
            if effective_loss is None and (
                training_target is not None and self._last_logits is not None
            ):
                with torch.no_grad():
                    target = training_target.to(self._last_logits.device)
                    if target.numel() > 0:
                        effective_loss = float(
                            torch.nn.functional.cross_entropy(
                                self._last_logits, target
                            ).item()
                        )
            if effective_loss is not None:
                self._update_loss_ema(effective_loss)
                ema = self._loss_ema  # local for type narrowing
                if ema is not None and effective_loss < ema * self.retrieval_feedback_threshold:
                    self._bump_retrieved_access_counts(self.retrieval_feedback_bump)
                    self._retrieval_feedback_event_count += 1
        # The buffer is consumed-per-batch; clear regardless of the
        # threshold so stale IDs from this batch don't bleed into the
        # next one.
        self._last_retrieved_meta = []

        # Either way, the next forward should start with a clean
        # multi-pass buffer. Defensive — consolidate() drained it
        # in the multi-pass path; this is a no-op when buffer is
        # already empty (single-pass mode).
        self.synapse.clear_buffer()
        # The cached `_last_features` was consumed; clear so a
        # missing forward before the next update is caught loudly.
        self._last_features = None
        # Same for the cached logits — the next training forward
        # will re-populate them. Eval forwards will not.
        self._last_logits = None
        return reward

    @property
    def compression_sweep_count(self) -> int:
        """Number of compression sweeps that have fired on this instance."""
        return self._compression_sweep_count

    @property
    def last_compression_counts(self) -> dict[int, int]:
        """``{precision: count}`` after the most recent sweep, ``{}`` if none."""
        return dict(self._last_compression_counts)

    @property
    def consolidation_count(self) -> int:
        """Number of consolidation cycles that have fired on this instance.

        Counts both new-entry and merged-into-existing cycles. The
        ``merge_count`` property is the subset that resolved by
        merging; new-entry cycles are ``consolidation_count -
        merge_count``.
        """
        return self._consolidation_count

    @property
    def merge_count(self) -> int:
        """Of the fired consolidation cycles, how many merged into an
        existing cold-storage entry rather than creating a new one.

        Always 0 unless ``repeat_consolidation_threshold < 1.0``.
        """
        return self._merge_count

    @property
    def retrieval_feedback_event_count(self) -> int:
        """How many times a retrieval-success bump fired.

        Always 0 unless ``retrieval_feedback_threshold > 0``.
        """
        return self._retrieval_feedback_event_count

    @property
    def last_retrieved_meta(self) -> list[tuple[str, int]]:
        """``(entry_id, pre_bump_access_count)`` tuples for the most
        recent retrieval. Cleared at the end of
        :meth:`apply_hebbian_update`; populated by retrieval-cache
        refreshes inside :meth:`features`.
        """
        return list(self._last_retrieved_meta)

    @property
    def loss_ema(self) -> float | None:
        """Running EMA of the per-batch loss, or ``None`` before the
        first ``apply_hebbian_update`` that supplied or derived a
        loss. Only updated when ``retrieval_feedback_threshold > 0``.
        """
        return self._loss_ema

    def _update_loss_ema(self, loss: float) -> None:
        """Cold-start EMA on the first call; exponential thereafter."""
        if self._loss_ema is None:
            self._loss_ema = float(loss)
        else:
            decay = self.retrieval_feedback_decay
            self._loss_ema = decay * self._loss_ema + (1.0 - decay) * float(loss)

    def _bump_retrieved_access_counts(self, bump: float) -> None:
        """Add ``bump`` to ``access_count`` for every entry recorded in
        ``_last_retrieved_meta``. access_count is stored as a number
        in metadata; downstream consumers ``int(...)`` it, so floating
        increments accumulate gracefully.
        """
        if self.cold_storage is None:
            return
        for entry_id, pre_bump in self._last_retrieved_meta:
            try:
                # The retrieval cache may have re-bumped this entry once
                # already (reconstruct_strengths defaults bump_access_count
                # to True). Read fresh, add the feedback bump.
                entry = self.cold_storage.get_by_id(entry_id)
            except KeyError:
                continue
            new_meta = dict(entry.metadata)
            current = float(new_meta.get("access_count", 0))
            new_meta["access_count"] = current + float(bump)
            self.cold_storage.update_metadata(entry_id, new_meta)
