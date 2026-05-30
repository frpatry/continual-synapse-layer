"""Consistency reward component.

Cosine similarity between the current batch's mean activation and a
running exponential moving average of mean activations seen so far.
The first call initialises the EMA and returns ``1.0`` (fully
consistent, no prior to compare against).

Intuition: when the task switches, the current activation pattern
differs from the EMA of past patterns, the cosine similarity drops,
the reward drops, and the Hebbian update is dampened. Combined with
:class:`SynapseLayer`'s evidence-based resistance, this is the
mechanism meant to address the Phase-2 failure mode where the
synapse layer simply tracked whatever task was being trained most
recently.

Cosine similarity ranges in ``[-1, 1]``; for non-negative activations
(e.g., post-ReLU) it is naturally in ``[0, 1]``. The class does not
clip the output — callers that need a strictly non-negative reward
should wrap the value or apply ``max(0, ...)`` themselves.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class ConsistencyReward:
    """Cosine similarity vs. an EMA of past mean activations.

    Args:
        n_neurons: Activation dimension. Used to size the EMA buffer
            on first call (we keep it ``None`` until then so the
            class works for any dim without preallocating).
        decay: EMA decay ``λ``. ``EMA ← λ · EMA + (1-λ) · current``.
            Defaults to ``0.99``, i.e. ~100 batches of history.
        center: Affine centering applied after cosine similarity:
            ``r' = (r - center) / scale``. Defaults to ``0.0`` and
            ``scale=1.0`` (no transform) — bit-exact backward
            compatibility. The architectural audit (2026-05-23)
            found cosine sim saturates at ~0.97 on post-ReLU
            features so the raw signal carries almost no
            modulation; setting ``center=0.95, scale=0.05``
            rescales the saturated band into a useful ``[0, 1]``
            working range. Negative values are possible on task
            switches; ``clip_min`` / ``clip_max`` bound the
            transformed output.
        scale: Divisor in the centering transform. Must be > 0.
        clip_min: Optional lower bound applied to the final
            (post-centering) reward. ``None`` (default) means no
            clipping.
        clip_max: Optional upper bound. ``None`` (default) means
            no clipping.
    """

    def __init__(
        self,
        n_neurons: int,
        decay: float = 0.99,
        center: float = 0.0,
        scale: float = 1.0,
        clip_min: float | None = None,
        clip_max: float | None = None,
    ) -> None:
        if n_neurons <= 0:
            raise ValueError("n_neurons must be positive")
        if not 0.0 < decay < 1.0:
            raise ValueError(
                f"decay must lie in (0, 1), got {decay}"
            )
        if scale <= 0.0:
            raise ValueError(f"scale must be > 0, got {scale}")
        if (
            clip_min is not None
            and clip_max is not None
            and clip_min > clip_max
        ):
            raise ValueError(
                f"clip_min ({clip_min}) must be <= clip_max ({clip_max})"
            )
        self.n_neurons = int(n_neurons)
        self.decay = float(decay)
        self.center = float(center)
        self.scale = float(scale)
        self.clip_min = clip_min if clip_min is None else float(clip_min)
        self.clip_max = clip_max if clip_max is None else float(clip_max)
        self._ema: torch.Tensor | None = None

    @property
    def ema(self) -> torch.Tensor | None:
        return self._ema

    def reset(self) -> None:
        self._ema = None

    def __call__(self, activations: torch.Tensor) -> float:
        if activations.ndim != 2:
            raise ValueError(
                f"activations must be 2-D (B, n), got shape "
                f"{tuple(activations.shape)}"
            )
        if activations.shape[1] != self.n_neurons:
            raise ValueError(
                f"activation dim {activations.shape[1]} does not match "
                f"n_neurons={self.n_neurons}"
            )

        with torch.no_grad():
            current = activations.detach().mean(dim=0)
            if self._ema is None:
                self._ema = current.clone()
                # First call uses the default-1.0 seeding semantics.
                # Run the same centering / clipping pipeline so that
                # callers configuring center/scale see consistent
                # output magnitudes from the very first batch.
                return self._transform(1.0)
            sim = F.cosine_similarity(
                current.unsqueeze(0), self._ema.unsqueeze(0), dim=1
            ).item()
            self._ema.mul_(self.decay).add_(current, alpha=1.0 - self.decay)
            return self._transform(float(sim))

    def _transform(self, raw: float) -> float:
        """Apply the configured affine transform + clip to ``raw``."""
        r = (raw - self.center) / self.scale
        if self.clip_min is not None and r < self.clip_min:
            r = self.clip_min
        if self.clip_max is not None and r > self.clip_max:
            r = self.clip_max
        return float(r)
