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

    # ---- feature extraction / classification (small adapters) ----

    def feature_extract(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate-layer features. Uses the MLP's ``features(x)``
        method directly so the synapse-augmented variant's modulator-
        corrected features can't sneak in by mistake."""
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

        The base model is forced into ``eval()`` for the duration so
        dropout / batchnorm don't contaminate the retrieval features;
        the prior ``training`` flag is restored on exit.
        """
        was_training = self.base_model.training
        self.base_model.eval()
        try:
            features = self.feature_extract(x)
            logits_model = self.classify(features)
            model_probs = F.softmax(logits_model, dim=-1)

            retrieval_probs, confidence = self.memory.retrieve(features)
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

        Args:
            x: Input batch ``(B, ...)``.
            y: Long-int class targets ``(B,)``.
            task_id: Optional task identifier stored on each allocated
                entry. Used only by diagnostics; retrieval doesn't
                consult it.
        """
        features = self.feature_extract(x)
        return self.memory.maybe_allocate(features, y, task_id=task_id)
