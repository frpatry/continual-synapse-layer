"""Episodic predictor — base model + active memory at inference.

Pairs an :class:`ActiveEpisodicMemory` with a base classifier
(typically :class:`MLPClassifier`). At inference, the model's softmax
is blended with a retrieval-based label distribution, with the
blend weight scaled by retrieval confidence — the memory only
contributes when something genuinely similar exists in it.

During training, :meth:`training_step_observe` extracts features
(``torch.no_grad`` — the storage decision must not affect the
model's gradients) and lets the memory decide whether to allocate.
The base model's own optimisation is left untouched; the wrapper
holds **no** training-time machinery beyond observation.

The blend rule, per sample:

    λ_eff = blend_max * clamp((conf − threshold) / (1 − threshold), 0, 1)
            if conf > threshold else 0

    blended_probs = (1 − λ_eff) * softmax(model_logits)
                  + λ_eff * retrieval_probs
    return log(blended_probs + eps)

Returning log-probabilities (instead of raw blended logits) keeps
the downstream argmax / cross-entropy compatible with the existing
runner's evaluation code — exactly the same trick
:class:`RetrievalEnsemble` uses.

The base model is put in ``eval()`` for the duration of
:meth:`predict` and restored to its prior training flag on exit,
matching the contract every other retrieval wrapper in this repo
follows.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from continual_synapse.episodic.active_memory import ActiveEpisodicMemory


class EpisodicPredictor:
    """Blend a base model's softmax with active episodic retrieval.

    Args:
        base_model: A model exposing ``.features(x) -> (B, D)`` and
            ``.classify(features) -> (B, C)``. The
            :class:`MLPClassifier` baseline satisfies this; the
            :class:`SynapseAugmentedMLP` wrapper does **not** (its
            ``features`` includes a synapse correction; it's the
            wrong call). Phase 3 explicitly uses a plain MLP so the
            dual-substrate test isn't contaminated.
        memory: An :class:`ActiveEpisodicMemory` instance. Built
            with ``feature_dim`` matching ``base_model.config.hidden_dim``.
        blend_threshold: Retrieval confidence (max cos sim within
            the top-k) below which the memory contributes zero
            weight. Default ``0.5`` — "don't trust retrieval unless
            it's at least somewhat similar".
        blend_max: Maximum weight the memory can receive when
            confidence is at its ceiling (1.0). Default ``0.5`` —
            even at perfect similarity, the model's own softmax
            still gets half the vote, hedging against memory
            mistakes.
        eps: Numerical floor for the ``log(blended_probs)`` step.
    """

    def __init__(
        self,
        base_model: nn.Module,
        memory: ActiveEpisodicMemory,
        blend_threshold: float = 0.5,
        blend_max: float = 0.5,
        eps: float = 1e-8,
        keying_encoder: nn.Module | None = None,
    ) -> None:
        if not 0.0 <= blend_threshold <= 1.0:
            raise ValueError(
                f"blend_threshold must be in [0, 1], got {blend_threshold}"
            )
        if not 0.0 <= blend_max <= 1.0:
            raise ValueError(
                f"blend_max must be in [0, 1], got {blend_max}"
            )
        self.base_model = base_model
        self.memory = memory
        self.blend_threshold = float(blend_threshold)
        self.blend_max = float(blend_max)
        self.eps = float(eps)
        # Optional frozen keying encoder for memory storage/retrieval.
        # When None (default), the predictor uses base_model.features
        # for both classification AND memory query — the original
        # behaviour. When set (e.g. a PretrainedContrastiveEncoder),
        # the memory uses the keying encoder's stable feature space
        # while the base model's own features still drive
        # classification. This is the Option-B dual-substrate fix
        # for feature drift: separate the "compute" feature space
        # (base_model, trainable) from the "memory" feature space
        # (keying_encoder, frozen).
        self.keying_encoder = keying_encoder

    # ---- feature extraction / classification (small adapters) ----

    def feature_extract(self, x: torch.Tensor) -> torch.Tensor:
        """Memory-query features.

        When ``keying_encoder`` is set, returns its output — this is
        the stable feature space that memory storage and retrieval
        both live in. When ``keying_encoder`` is ``None`` (default),
        falls back to ``base_model.features(x)`` (the original
        behaviour, bit-identical to pre-Option-B).

        Note: this is **not** the same as ``base_model.features(x)``
        when a keying encoder is configured. The classification
        path inside :meth:`predict` still uses
        ``base_model.features`` directly — the base model's head was
        trained on its own features, so feeding it the keying
        encoder's outputs would break the classifier.
        """
        if self.keying_encoder is not None:
            return self.keying_encoder(x)
        return self.base_model.features(x)

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """Model head applied to precomputed features. Same
        ``classify`` API as :class:`MLPClassifier`."""
        return self.base_model.classify(features)

    # ---- inference ----

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return blended logits (technically log-probabilities) for
        the input batch. See module docstring for the blend formula.

        Two feature spaces flow through this method when a frozen
        ``keying_encoder`` is configured:

        - **Classification path**: ``base_model.features(x)`` →
          ``base_model.classify(...)``. The base classifier was
          trained on its own features; feeding it anything else
          breaks the head.
        - **Memory query path**: ``self.feature_extract(x)``, which
          routes to ``keying_encoder(x)`` when set, otherwise the
          base features. Memory storage and retrieval must share
          this space.

        When no keying encoder is set, both paths happen to use the
        same base features — that's the original (bit-identical-
        to-pre-Option-B) behaviour.

        The base model is forced into ``eval()`` for the duration so
        dropout / batchnorm don't contaminate the classification or
        retrieval features; the prior ``training`` flag is restored
        on exit.
        """
        was_training = self.base_model.training
        self.base_model.eval()
        try:
            # Classification path — always uses base_model.
            base_features = self.base_model.features(x)
            logits_model = self.base_model.classify(base_features)
            model_probs = F.softmax(logits_model, dim=-1)

            # Memory query path — uses keying_encoder when set, else
            # reuses the base features we just computed (no double
            # forward).
            if self.keying_encoder is not None:
                query_features = self.keying_encoder(x)
            else:
                query_features = base_features

            retrieval_probs, confidence = self.memory.retrieve(query_features)
            # Make sure retrieval output lives on the same device as
            # the model's outputs (the memory keeps embeddings on CPU
            # by default).
            retrieval_probs = retrieval_probs.to(model_probs.device)
            confidence = confidence.to(model_probs.device)

            # Effective blend weight per sample. Below threshold:
            # zero contribution from retrieval. Above threshold:
            # linearly ramp from 0 to blend_max as confidence goes
            # from threshold to 1.
            denom = max(1e-6, 1.0 - self.blend_threshold)
            scaled = ((confidence - self.blend_threshold) / denom).clamp(
                min=0.0, max=1.0,
            )
            engaged = (confidence > self.blend_threshold).to(scaled.dtype)
            lambda_eff = (engaged * self.blend_max * scaled).unsqueeze(-1)

            blended_probs = (
                (1.0 - lambda_eff) * model_probs
                + lambda_eff * retrieval_probs
            )
            return (blended_probs + self.eps).log()
        finally:
            self.base_model.train(was_training)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x)

    # ---- training-time observation ----

    @torch.no_grad()
    def training_step_observe(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        task_id: int | None = None,
    ) -> int:
        """Extract features without grad and let the memory decide
        whether to allocate. Returns the number of entries created.

        Does NOT touch the model's parameters or the optimiser state
        — the caller is responsible for running standard backprop on
        the task loss separately. The memory substrate accumulates
        independently of the compute substrate; that's the whole
        point of the dual-substrate design.

        ``x`` flows into the memory as both the feature query (after
        :meth:`feature_extract`) and the stored ``raw_inputs`` —
        the latter is what :meth:`re_encode_memory` re-uses at task
        boundaries to keep stored embeddings aligned with the
        current model's feature space.

        Args:
            x: Input batch ``(B, ...)``.
            y: Long-int class targets ``(B,)``.
            task_id: Optional task identifier stored on each allocated
                entry. Used only by diagnostics; retrieval doesn't
                consult it.
        """
        features = self.feature_extract(x)
        return self.memory.maybe_allocate(
            features=features,
            raw_inputs=x,
            labels=y,
            task_id=task_id,
        )

    @torch.no_grad()
    def re_encode_memory(
        self,
        device: torch.device | str | None = None,
        batch_size: int = 256,
    ) -> int:
        """Refresh every stored embedding under the current model.

        Convenience wrapper around
        :meth:`ActiveEpisodicMemory.re_encode_all` that handles the
        eval-mode boilerplate so the encoded features don't see
        dropout. The previous training flag is restored on exit so a
        re-encode in the middle of a training loop doesn't leak.

        Short-circuits to ``0`` when a frozen ``keying_encoder`` is
        configured: by construction, a frozen encoder produces the
        same output for the same raw input on every call, so
        re-encoding can't change the stored embeddings. The caller
        is welcome to invoke this unconditionally — the no-op fast
        path is cheaper than gating in callers.

        Returns the number of entries re-encoded (0 when nothing
        was done, either because the memory is empty or because
        the encoder is frozen).
        """
        if self.keying_encoder is not None:
            return 0
        n = len(self.memory)
        if n == 0:
            return 0
        was_training = self.base_model.training
        self.base_model.eval()
        try:
            self.memory.re_encode_all(
                feature_extractor=self.feature_extract,
                device=device,
                batch_size=batch_size,
            )
        finally:
            self.base_model.train(was_training)
        return n
