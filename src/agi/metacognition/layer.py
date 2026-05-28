"""Small MLP that maps engineered features → epistemic state.

Two instances are used in production:

- **Pre-layer**  reads 9 features (memory + query) and judges
  whether the system *should* answer before generation.
- **Post-layer** reads 18 features (memory + query + generation +
  alignment) and judges whether the generated response is
  trustworthy.

Both layers share the same architecture but are *trained
separately* — the pre-layer never sees generation-time signals,
the post-layer always does. Phase 2a ships the random-init
networks; Phase 2b will train them on synthetic data.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .features import (
    POST_FEATURE_DIM,
    PRE_FEATURE_DIM,
    assemble_feature_vector,
)
from .state import EpistemicStatus, MetacognitiveState, RecommendedAction


# Stable ordering of statuses → logit indices. Changing this
# breaks any trained checkpoint of either layer.
_STATUS_ORDER: tuple[EpistemicStatus, ...] = (
    "known",
    "uncertain",
    "unknown",
    "hallucinated",
)

# Status → action mapping. `known` answers directly, `uncertain`
# answers with a caveat, both `unknown` and `hallucinated` defer
# to a template.
_ACTION_MAP: dict[EpistemicStatus, RecommendedAction] = {
    "known": "answer",
    "uncertain": "answer_with_caveat",
    "unknown": "admit_ignorance",
    "hallucinated": "admit_ignorance",
}


def _input_dim(mode: str) -> int:
    if mode == "pre":
        return PRE_FEATURE_DIM
    if mode == "post":
        return POST_FEATURE_DIM
    raise ValueError(f"mode must be 'pre' or 'post', got {mode!r}")


class MetacognitiveLayer(nn.Module):
    """A two-hidden-layer MLP with a 4-way status head and a
    sigmoid confidence head.

    Architecture::

        Linear(n_in → 64) → ReLU → Dropout(0.1)
        Linear(64   → 64) → ReLU
        ├── Linear(64 → 4) ──→ status logits
        └── Linear(64 → 1) ──→ sigmoid → confidence

    The two heads share the trunk so confidence is conditioned
    on the same internal representation as the status decision.
    """

    def __init__(
        self,
        mode: Literal["pre", "post"],
        *,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if mode not in ("pre", "post"):
            raise ValueError(f"mode must be 'pre' or 'post', got {mode!r}")
        self.mode = mode
        self.in_dim = _input_dim(mode)
        self.hidden_dim = int(hidden_dim)

        self.trunk = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.status_head = nn.Linear(self.hidden_dim, len(_STATUS_ORDER))
        self.confidence_head = nn.Linear(self.hidden_dim, 1)

    def forward(
        self, features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, confidence)``.

        ``features`` may be 1-D ``(in_dim,)`` for a single sample
        or 2-D ``(batch, in_dim)`` for a batch. The returned
        logits are shape ``(..., 4)`` and confidence is shape
        ``(...,)`` (one scalar per sample).
        """
        if features.dim() == 1:
            x = features.unsqueeze(0)
            squeeze = True
        else:
            x = features
            squeeze = False
        if x.shape[-1] != self.in_dim:
            raise ValueError(
                f"feature dim {x.shape[-1]} does not match "
                f"layer in_dim {self.in_dim} (mode={self.mode!r})"
            )
        h = self.trunk(x)
        logits = self.status_head(h)
        confidence = torch.sigmoid(self.confidence_head(h)).squeeze(-1)
        if squeeze:
            logits = logits.squeeze(0)
            confidence = confidence.squeeze(0)
        return logits, confidence

    @torch.no_grad()
    def predict(
        self,
        features: torch.Tensor | dict,
        *,
        raw_features: dict | None = None,
    ) -> MetacognitiveState:
        """Run inference and assemble a :class:`MetacognitiveState`.

        ``features`` may be either:

        - A 1-D float tensor matching the layer's input
          dimensionality (``9`` for pre, ``18`` for post), OR
        - A feature **dict** with the same keys
          ``assemble_feature_vector`` expects, in which case the
          tensor is built internally.

        When a dict is passed, ``raw_features`` defaults to that
        dict. When a tensor is passed, ``raw_features`` defaults
        to ``{}`` unless the caller supplies it explicitly.
        """
        if isinstance(features, dict):
            raw = features if raw_features is None else raw_features
            tensor = assemble_feature_vector(features, mode=self.mode)
        else:
            raw = raw_features or {}
            tensor = features

        self.eval()
        logits, confidence = self.forward(tensor)

        status_idx = int(logits.argmax(dim=-1).item())
        status: EpistemicStatus = _STATUS_ORDER[status_idx]
        action: RecommendedAction = _ACTION_MAP[status]

        # Memory features are the first 6 slots in both modes.
        # Use them to populate the scalar memory_coverage /
        # memory_quality fields on the state. Coverage uses
        # n_facts_retrieved (slot 0) clamped via a soft cap;
        # quality uses max_similarity (slot 1) directly.
        flat = tensor.detach().flatten().tolist()
        n_facts = flat[0] if len(flat) > 0 else 0.0
        max_sim = flat[1] if len(flat) > 1 else 0.0
        memory_coverage = float(min(1.0, n_facts / 3.0))
        memory_quality = float(max(0.0, min(1.0, max_sim)))

        # Alignment is only meaningful in post mode (slots 13-15);
        # take the mean of the three alignment slots as the
        # scalar summary. None for the pre-layer.
        if self.mode == "post" and len(flat) >= 16:
            align_slice = flat[13:16]
            gen_alignment: float | None = (
                sum(align_slice) / len(align_slice) if align_slice else 0.0
            )
        else:
            gen_alignment = None

        return MetacognitiveState(
            epistemic_status=status,
            confidence=float(confidence.item()),
            memory_coverage=memory_coverage,
            memory_quality=memory_quality,
            generation_alignment=gen_alignment,
            recommended_action=action,
            raw_features=dict(raw),
        )
