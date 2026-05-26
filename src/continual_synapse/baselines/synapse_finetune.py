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
from continual_synapse.consolidation.pipeline import (
    _drain_candidates,
    consolidate_to_storage,
)
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
        # ---- Task-aware variant flags (defaults preserve current behavior) ----
        task_aware_decay: float = 0.0,
        task_warmup_batches: int = 0,
        task_warmup_downweight: float = 1.0,
        # ---- Reward-modulated amplification flag ----
        reward_modulated_amplification: bool = False,
        # ---- Gradient-gating (experiment 15) flags ----
        gate_modulation_enabled: bool = True,
        gradient_gating_enabled: bool = False,
        gradient_gating_alpha: float = 0.9,
        familiarity_mode: str = "magnitude",
        # ---- Developmental maturity (experiment 19) ----
        maturity_target_consolidations: int = 0,
        # ---- Path-C: per-class consolidation ----
        consolidation_mode: str = "aggregate",
        min_samples_per_class: int = 5,
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
        if task_aware_decay < 0:
            raise ValueError(
                f"task_aware_decay must be >= 0, got {task_aware_decay}"
            )
        if task_warmup_batches < 0:
            raise ValueError(
                f"task_warmup_batches must be >= 0, got {task_warmup_batches}"
            )
        if task_warmup_downweight < 0:
            raise ValueError(
                f"task_warmup_downweight must be >= 0, got "
                f"{task_warmup_downweight}"
            )
        if not 0.0 <= gradient_gating_alpha <= 1.0:
            raise ValueError(
                f"gradient_gating_alpha must be in [0, 1], got "
                f"{gradient_gating_alpha}"
            )
        if familiarity_mode not in ("magnitude", "cosine"):
            raise ValueError(
                f"familiarity_mode must be 'magnitude' or 'cosine', "
                f"got {familiarity_mode!r}"
            )
        if maturity_target_consolidations < 0:
            raise ValueError(
                f"maturity_target_consolidations must be >= 0, got "
                f"{maturity_target_consolidations}"
            )
        if consolidation_mode not in ("aggregate", "per_class"):
            raise ValueError(
                f"consolidation_mode must be 'aggregate' or 'per_class', "
                f"got {consolidation_mode!r}"
            )
        if min_samples_per_class <= 0:
            raise ValueError(
                f"min_samples_per_class must be positive, got "
                f"{min_samples_per_class}"
            )
        self.consolidation_mode = str(consolidation_mode)
        self.min_samples_per_class = int(min_samples_per_class)
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
        self.task_aware_decay = float(task_aware_decay)
        self.task_warmup_batches = int(task_warmup_batches)
        self.task_warmup_downweight = float(task_warmup_downweight)
        self.reward_modulated_amplification = bool(reward_modulated_amplification)
        # Gradient-gating (experiment 15). When gate_modulation_enabled
        # is False the output is the bare base.classify(f_base) (the
        # synapse correction is skipped) AND the modulator gate is
        # frozen at zero so the optimizer cannot revive it. When
        # gradient_gating_enabled is True, the model exposes
        # apply_gradient_gating() which scales base.parameters()
        # gradients by ``1 - alpha * familiarity`` where familiarity
        # = ||f_base @ effective_strengths|| normalised by its running
        # max. The on_pre_optimizer_step runner hook drives this from
        # experiment 15.
        self.gate_modulation_enabled = bool(gate_modulation_enabled)
        self.gradient_gating_enabled = bool(gradient_gating_enabled)
        self.gradient_gating_alpha = float(gradient_gating_alpha)
        self.familiarity_mode = str(familiarity_mode)
        # Developmental maturity (experiment 19). Default 0 means
        # "maturity is irrelevant" — apply_gradient_gating then uses
        # the raw alpha as before. With target > 0, the effective
        # alpha at any moment is scaled by
        # min(consolidation_count / target, 1) so the gating ramps up
        # from "no protection" (when the system has accumulated no
        # consolidations yet) to full protection (when the count
        # reaches target). Models the idea that the system has weak
        # conviction about what it knows when it knows little.
        self.maturity_target_consolidations = int(maturity_target_consolidations)
        self._last_maturity: float = 1.0
        # Snapshot of the most recent cosine-similarity vector against
        # cold storage, populated by apply_gradient_gating in cosine
        # mode. Empty list otherwise. Exposed via last_similarities.
        self._last_similarities: list[float] = []
        if not self.gate_modulation_enabled:
            with torch.no_grad():
                self.modulator.gate.data.zero_()
            self.modulator.gate.requires_grad_(False)
        # Familiarity tracking state.
        self._last_effective_strengths: torch.Tensor | None = None
        self._familiarity_max: float = 1e-6
        self._last_familiarity: float = 0.0
        self._last_gradient_scale: float = 1.0
        # Cache of the most recently applied Hebbian reward, used by
        # reward-modulated amplification on the *next* forward pass.
        # None on cold start ⇒ the modulation factor defaults to 1.0
        # so the first batch behaves like standard (non-modulated)
        # amplification. Populated at the end of every successful
        # apply_hebbian_update.
        self._last_reward: float | None = None
        # Task-aware state. The model is "task-agnostic" until the
        # caller calls notify_task_change(). _current_task_id == -1
        # writes the same "untagged" sentinel into consolidations as
        # the omitted-kwarg default in consolidate_to_storage. The
        # warmup counter starts very large so the warmup window is
        # inactive until notify_task_change resets it to 0.
        self._current_task_id: int = -1
        self._batches_since_task_change: int = 10**9
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
            # Task-warmup downweight: during the first ``task_warmup_batches``
            # apply_hebbian_update calls after a task change, scale the
            # cold-storage contribution by ``task_warmup_downweight`` so
            # the new task isn't immediately overwritten by archive
            # patterns. With the default downweight=1.0 this is a no-op,
            # so non-task-aware methods are unaffected. The counter
            # itself only advances when apply_hebbian_update fires (it
            # is incremented there), so eval-mode forwards do not
            # progress the warmup window.
            if (
                self.task_warmup_batches > 0
                and self._batches_since_task_change < self.task_warmup_batches
                and self.task_warmup_downweight != 1.0
            ):
                retrieved = retrieved * self.task_warmup_downweight
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
                # Reward-modulated amplification: when enabled, scale
                # ``alpha`` by the most recently applied per-batch
                # reward so high-confidence updates produce stronger
                # amplification, low-confidence updates produce weaker
                # amplification, and negative-reward batches (anti-
                # correlated retrieval) produce anti-amplification.
                # The reward is from the *previous* batch's
                # apply_hebbian_update — a one-batch lag, acceptable for
                # a continuous signal. Cold-start ⇒ no prior reward ⇒
                # default 1.0 (standard amplification behavior).
                effective_alpha = self.amplification_alpha
                if self.reward_modulated_amplification:
                    last = self._last_reward if self._last_reward is not None else 1.0
                    effective_alpha = effective_alpha * float(last)
                effective_strengths = self.synapse.strengths * (
                    1.0 + effective_alpha * retrieved_normalized
                )
        else:
            effective_strengths = self.synapse.strengths

        # Cache for gradient gating (experiment 15). Detached so the
        # autograd graph isn't extended; the gating computation runs
        # under no_grad anyway.
        self._last_effective_strengths = effective_strengths.detach()

        if not self.gate_modulation_enabled:
            # cs_gated: bare base output, synapse correction muted.
            # The synapse layer still learns via apply_hebbian_update;
            # the cold storage still retrieves and stays warm; the
            # familiarity used by gradient gating still derives from
            # the same effective_strengths we cached above. Only the
            # *output-side* modulation is skipped.
            return f_base
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
                    current_task_id=(
                        self._current_task_id
                        if self._current_task_id >= 0
                        else None
                    ),
                    task_recency_decay=self.task_aware_decay,
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

                Second purpose (path-A label storage): when cold
                storage and a consolidation trigger are also
                configured, the dominant ground-truth class of this
                batch (``torch.mode(training_target)``) plus a per-
                class histogram are passed through to
                :func:`consolidate_to_storage` so any entry created
                this cycle carries ``metadata["true_label"]`` and
                ``metadata["label_histogram_json"]``. ``None`` keeps
                the pre-path-A behaviour (no label written, schema
                matches older checkpoints).
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
            # Path-C divergence: in per_class mode we group this batch's
            # (features, training_target) pairs by class and create one
            # cold-storage entry per class meeting the
            # min_samples_per_class threshold. Sibling entries created in
            # the same event share the same trigger-fire gate, the same
            # candidate mask, and the same quantised strengths/document
            # (synapse state doesn't change between sibling calls because
            # we pass drain=False; we drain once explicitly afterwards).
            # When training_target is None or no class meets the
            # threshold, per_class silently falls back to aggregate so
            # existing eval pipelines keep working.
            use_per_class = (
                self.consolidation_mode == "per_class"
                and training_target is not None
                and training_target.numel() > 0
            )
            if use_per_class:
                self._consolidate_per_class(features, training_target)
            else:
                embedding = features.mean(dim=0).to(torch.float32)
                true_label: int | None = None
                label_histogram: list[int] | None = None
                if training_target is not None and training_target.numel() > 0:
                    targets_long = training_target.detach().to(torch.long).cpu()
                    true_label = int(torch.mode(targets_long).values.item())
                    if self._last_logits is not None:
                        num_classes = int(self._last_logits.shape[-1])
                        label_histogram = torch.bincount(
                            targets_long, minlength=num_classes
                        ).tolist()
                outcome = consolidate_to_storage(
                    self.synapse,
                    self.cold_storage,
                    self.consolidation_trigger,
                    activation_embedding=embedding,
                    drain=not self.no_drain_on_consolidate,
                    merge_threshold=self.repeat_consolidation_threshold,
                    task_id=self._current_task_id,
                    true_label=true_label,
                    label_histogram=label_histogram,
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
        # NOTE: ``_last_retrieved_meta`` is deliberately NOT cleared here.
        # The retrieval cache may be reused across many forwards
        # (refresh_interval > 1); the IDs captured at the last refresh
        # remain "the active retrieval set" until the next refresh
        # overwrites them. Per-task diagnostics in experiment 13 read
        # the property at every on_after_batch hook.

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
        # Advance the task-warmup counter exactly once per training
        # batch (this method only fires from on_after_batch hooks).
        # Eval-mode forwards do not get here so the warmup window
        # does not "leak" through evaluation.
        self._batches_since_task_change += 1
        # Cache for reward-modulated amplification on the NEXT forward.
        # Stored even when reward_modulated_amplification is False —
        # the cache is cheap and lets a caller flip the flag mid-run
        # without a cold-start window. Read sites still respect the
        # flag, so non-modulated methods are unaffected.
        self._last_reward = float(reward)
        return reward

    def _consolidate_per_class(
        self,
        features: torch.Tensor,
        training_target: torch.Tensor,
    ) -> None:
        """Path-C: group ``features`` by ``training_target`` and create
        one cold-storage entry per class that meets
        ``min_samples_per_class``.

        All sibling entries from one event share the same trigger gate
        (we check ``should_fire`` once up front), the same candidate
        mask, and the same quantised strengths/document — because
        ``drain=False`` is passed on every internal call and the
        synapse state is therefore unchanged across siblings. A single
        manual drain at the end produces the same final state as one
        ``drain=True`` consolidation in aggregate mode.

        ``label_histogram_json`` is deliberately not written in
        per-class entries: each entry is class-pure (one-hot), so the
        histogram is redundant and only bloats metadata.

        Falls through silently (no consolidation this batch) when the
        trigger declines, when no class meets the threshold, or when
        the candidate mask is empty.
        """
        if not self.consolidation_trigger.should_fire(self.synapse):
            return

        # Group this batch by class. unique returns sorted classes; we
        # iterate in that order so the resulting entry ids/storage
        # order are deterministic for a fixed seed (useful in tests
        # and in any diagnostic that lists entries by creation order).
        targets_long = training_target.detach().to(torch.long).cpu()
        unique_classes, counts = torch.unique(targets_long, return_counts=True)

        n_fired = 0
        n_merged = 0
        for c_val, n_val in zip(unique_classes.tolist(), counts.tolist()):
            if n_val < self.min_samples_per_class:
                continue
            mask_c = training_target.to(targets_long.device) == int(c_val)
            class_emb = (
                features[mask_c.to(features.device)]
                .mean(dim=0)
                .to(torch.float32)
            )
            outcome = consolidate_to_storage(
                self.synapse,
                self.cold_storage,
                self.consolidation_trigger,
                activation_embedding=class_emb,
                # force=True bypasses the per-call should_fire check
                # (we already gated above) so the second-and-later
                # siblings, which trip the min_steps_between
                # refractory set by sibling #1's mark_fired, still
                # write their entry.
                force=True,
                # drain=False: keep synapse state intact across
                # siblings so they all see the same candidate mask
                # and strengths. A single drain runs after the loop.
                drain=False,
                merge_threshold=self.repeat_consolidation_threshold,
                task_id=self._current_task_id,
                true_label=int(c_val),
                # No label_histogram in per_class mode — each entry
                # is class-pure, so the histogram is one-hot and
                # carries no extra signal over true_label.
                label_histogram=None,
            )
            if outcome.fired:
                n_fired += 1
                if outcome.was_merged:
                    n_merged += 1

        # Single drain after all siblings, mirroring the aggregate
        # mode's drain semantics (one event = one drain). When
        # no_drain_on_consolidate is set, skip it just like aggregate
        # mode would.
        if n_fired > 0 and not self.no_drain_on_consolidate:
            mask = self.consolidation_trigger.candidate_mask(self.synapse)
            if mask.any():
                _drain_candidates(self.synapse, mask)

        if n_fired > 0:
            # Count one bump per sibling stored, matching aggregate
            # mode's "one cycle = one bump" semantic extended to
            # multi-entry cycles. Diagnostics that read
            # consolidation_count are forward-compatible: the count
            # grows faster in per_class mode because more entries
            # are produced per event, which is the intended
            # behaviour.
            self._consolidation_count += n_fired
            self._merge_count += n_merged
            self._cache_invalidated = True

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

    @property
    def last_familiarity(self) -> float:
        """Familiarity score from the most recent
        :meth:`apply_gradient_gating` call (or 0.0 if not yet called).
        Always in ``[0, 1]``. ``0`` means novel pattern (full
        plasticity); ``1`` means most-familiar-yet (gradients reduced
        to ``1 - gradient_gating_alpha`` of original).
        """
        return self._last_familiarity

    @property
    def last_gradient_scale(self) -> float:
        """The scalar gradients were multiplied by on the most recent
        :meth:`apply_gradient_gating` call, or 1.0 if not yet called.
        Defined as ``1 - gradient_gating_alpha * last_familiarity``;
        ``1.0`` means gradients passed through unchanged.
        """
        return self._last_gradient_scale

    @property
    def familiarity_max(self) -> float:
        """Running max of the raw familiarity signal. Used as the
        denominator in the per-batch normalisation. Useful diagnostic
        to detect a single-spike scale-pinning event.
        """
        return self._familiarity_max

    @property
    def last_similarities(self) -> list[float]:
        """Cosine similarities computed by the most recent
        :meth:`apply_gradient_gating` call in ``"cosine"`` mode, one
        per stored cold-storage entry in insertion order. Empty list
        in magnitude mode or before the first call. Useful for
        per-task distribution diagnostics in experiment 16.
        """
        return list(self._last_similarities)

    @property
    def last_maturity(self) -> float:
        """Developmental-maturity factor used by the most recent
        :meth:`apply_gradient_gating` call. Defined as
        ``min(consolidation_count / maturity_target_consolidations, 1)``
        when the target is positive; ``1.0`` otherwise (i.e. when
        maturity scaling is disabled, the maturity factor is the
        identity). Reported per task in experiment 19 to show the
        ramp-up trajectory.
        """
        return self._last_maturity

    @torch.no_grad()
    def apply_gradient_gating(self) -> float:
        """Scale ``base.parameters()`` gradients by Hebbian familiarity.

        Called by the runner's ``on_pre_optimizer_step`` hook between
        ``loss.backward()`` and ``optimizer.step()``. No-op when
        ``gradient_gating_enabled`` is False (preserves existing
        methods bit-exact). No-op when no forward has been run yet
        and there is nothing in the multi-pass buffer either —
        defensive; the runner always forwards first.

        Activation source for the familiarity computation:

        - Single-pass training (``n_passes == 1``): use the cached
          ``_last_features`` populated by ``features()``.
        - Multi-pass training (``n_passes > 1``): ``features()``
          deliberately clears ``_last_features`` (the buffer is the
          source of truth) and pushes ``n_passes`` activations into
          ``synapse._activation_buffer``. Average the buffer here so
          the familiarity reflects the same denoised activations the
          Hebbian update will consume. This matches the multi-pass
          query-consistency fix in ``features()`` itself (audit
          fix 3/3).

        Familiarity is computed according to ``familiarity_mode``:

        - ``"magnitude"`` (default, backward-compat with cs_gated):

              raw = || features @ effective_strengths ||
              max ← max(max, raw)
              fam = min(raw / max, 1)

          An aggregate magnitude measure that saturates as the synapse
          layer accumulates patterns. The mechanism behind the original
          exp-15 cs_gated baseline.

        - ``"cosine"`` (experiment 16's cs_gated_cosine):

              activation = features.mean(0)
              sims = cold_storage.compute_similarities(activation)
              fam  = max(0, max(sims))  if sims else 0

          A pattern-specific recognition measure. Cosine is naturally
          bounded in ``[-1, +1]``; the ``max(0, ·)`` clamp turns
          anti-correlation into "no familiarity" rather than negative
          familiarity. No adaptive normalisation needed.

        Final scaling is the same either way:

              scale = 1 - alpha * fam

        Returns the gradient scale that was applied (useful for the
        on_after_batch diagnostics in experiments 15 and 16).
        """
        if not self.gradient_gating_enabled:
            return 1.0
        if self._last_effective_strengths is None and self.familiarity_mode != "cosine":
            return 1.0
        features_for_familiarity = self._last_features
        if features_for_familiarity is None:
            # Multi-pass path: features() set _last_features = None so
            # apply_hebbian_update reads from the buffer. Mirror that
            # convention here.
            if self.synapse.buffer_size > 0:
                features_for_familiarity = self.synapse.buffer_average()
            else:
                return 1.0

        if self.familiarity_mode == "cosine":
            # Pattern-specific recognition via cosine similarity against
            # every stored cold-storage embedding.
            if self.cold_storage is None or self.cold_storage.count() == 0:
                self._last_similarities = []
                self._last_familiarity = 0.0
                self._last_gradient_scale = 1.0
                return 1.0
            query = features_for_familiarity.mean(dim=0).detach()
            sims = self.cold_storage.compute_similarities(query.tolist())
            self._last_similarities = list(sims)
            if not sims:
                self._last_familiarity = 0.0
                self._last_gradient_scale = 1.0
                return 1.0
            raw_familiarity = max(0.0, float(max(sims)))
            self._familiarity_max = max(self._familiarity_max, raw_familiarity)
            familiarity = float(raw_familiarity)  # already in [0, 1]
        else:
            # "magnitude" — original cs_gated path.
            if self._last_effective_strengths is None:
                return 1.0
            raw_signal = features_for_familiarity @ self._last_effective_strengths
            raw_familiarity = float(raw_signal.norm().item())
            self._familiarity_max = max(self._familiarity_max, raw_familiarity)
            familiarity = min(raw_familiarity / self._familiarity_max, 1.0)

        # Developmental maturity scales the effective alpha by how much
        # consolidation experience the system has accumulated. With
        # target=0 (the default) maturity is fixed at 1.0 and the
        # scaling reduces to the standard formula bit-exact. With
        # target>0, effective_alpha grows linearly from 0 to alpha as
        # consolidation_count grows from 0 to target, then stays
        # saturated.
        if self.maturity_target_consolidations > 0:
            maturity = min(
                self._consolidation_count
                / float(self.maturity_target_consolidations),
                1.0,
            )
        else:
            maturity = 1.0
        self._last_maturity = float(maturity)
        effective_alpha = self.gradient_gating_alpha * maturity
        gradient_scale = 1.0 - effective_alpha * familiarity
        self._last_familiarity = float(familiarity)
        self._last_gradient_scale = float(gradient_scale)
        for param in self.base.parameters():
            if param.grad is not None:
                param.grad.mul_(gradient_scale)
        return gradient_scale

    @property
    def last_reward(self) -> float | None:
        """Reward applied by the most recent ``apply_hebbian_update``,
        or ``None`` before the first such call. Cached for
        :attr:`reward_modulated_amplification` to consume on the next
        forward; also useful as a diagnostic.
        """
        return self._last_reward

    @property
    def current_task_id(self) -> int:
        """Identifier of the task currently being trained. ``-1`` until
        :meth:`notify_task_change` is first called. Tagged into new
        cold-storage entries; combined with ``task_aware_decay`` to
        weight retrieval by task recency.
        """
        return self._current_task_id

    @property
    def batches_since_task_change(self) -> int:
        """How many ``apply_hebbian_update`` calls have happened since
        the most recent :meth:`notify_task_change`. Capped by the
        warmup window: while this is ``< task_warmup_batches`` the
        cold-storage retrieval is scaled by ``task_warmup_downweight``.
        """
        return self._batches_since_task_change

    def notify_task_change(self, task_id: int) -> None:
        """Tell the model "I'm about to train on task ``task_id`` next."

        Three things happen:
        - ``_current_task_id`` is set to ``task_id``. Subsequent
          consolidations get tagged with this id (so task-recency
          weighting can apply), and retrieval queries use this as the
          "current" reference when computing
          ``exp(-decay * (current - entry_task_id))``.
        - The retrieval cache is invalidated. The next forward
          re-queries cold storage so the cached pre-task-change
          retrieval doesn't bleed into the new task's first batches.
        - The warmup counter resets to ``0`` so the next
          ``task_warmup_batches`` training batches see the
          ``task_warmup_downweight`` scaling applied to retrieval.

        Safe to call repeatedly (including for the same task_id, e.g.
        before each evaluation hop in the runner) — the cache-clear
        is idempotent and the warmup window only matters during
        training, which can't enter from this call.
        """
        self._current_task_id = int(task_id)
        self._cache_invalidated = True
        self._batches_since_task_change = 0

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
