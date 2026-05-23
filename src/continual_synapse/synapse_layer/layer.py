"""SynapseLayer — dense Hebbian state with metacognitive buffers.

Iterates on Phase-2 v1 by adding the full Phase-3 state schema from
DESIGN.md section 3.2:

- ``strengths``: learned Hebbian weight.
- ``evidence``: accumulated co-activation magnitude.
- ``confidence``: how *sustained* a co-activation has been across
  consecutive batches.
- ``age``: number of consolidations since the buffer was reset.
- ``access_count``: how often the synapse contributed non-trivially
  to the modulator's correction vector.

The new state fields (``confidence``, ``age``, ``access_count``)
are *populated* mechanically but not yet fed back into the update
rule. They exist so that Phase 4's pressure-based consolidation
can compute its scoring formula and so that ablation experiments
can correlate any future rule that uses them with the behaviour we
observe now.

The update rule (DESIGN.md eq. 3.2 with normalised evidence)::

    Δs_ij = η · R · raw_ij · 1 / (1 + β · normalised_evidence_ij)
    normalised_evidence = evidence / (max(evidence) + ε)
    E_ij ← E_ij + |a_i| · |a_j|

Normalising by ``max(evidence)`` makes ``β`` dataset-independent:
``β`` is now the dampening factor for the *most-evidenced* synapse
in the layer, regardless of how large raw evidence happens to grow
on the chosen benchmark.
"""

from __future__ import annotations

import torch
from torch import nn

from continual_synapse.synapse_layer.topk import (
    apply_topk_mask_inplace,
    compute_topk_mask,
)


_EVIDENCE_NORM_EPS = 1e-6


class SynapseLayer(nn.Module):
    """Dense Hebbian state with evidence-normalised resistance.

    Buffers (all ``(n, n)`` unless noted):

    - ``strengths``: float32.
    - ``evidence``: float32.
    - ``confidence``: float32.
    - ``age``: int64.
    - ``access_count``: int64.
    - ``global_step``: long scalar.
    - ``_prev_abs_outer``: float32, internal cache of the previous
      batch's ``|a_i| · |a_j|`` for the confidence rule.

    Args:
        n_neurons: Width of the activation vector this layer observes.
        learning_rate: ``η`` in the update rule.
        resistance_beta: ``β``. With ``0`` (the default) the strength
            path is bit-identical to Phase-2 v1. Larger values
            dampen updates for synapses whose normalised evidence is
            close to 1.
    """

    def __init__(
        self,
        n_neurons: int,
        learning_rate: float = 1e-3,
        resistance_beta: float = 0.0,
        sparse: bool = False,
        top_k: int = 64,
    ) -> None:
        super().__init__()
        if n_neurons <= 0:
            raise ValueError(f"n_neurons must be positive, got {n_neurons}")
        if learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {learning_rate}"
            )
        if resistance_beta < 0:
            raise ValueError(
                f"resistance_beta must be >= 0, got {resistance_beta}"
            )
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        self.n_neurons = n_neurons
        self.learning_rate = float(learning_rate)
        self.resistance_beta = float(resistance_beta)
        self.sparse = bool(sparse)
        self.top_k = int(top_k)

        zeros_f = torch.zeros(n_neurons, n_neurons, dtype=torch.float32)
        zeros_l = torch.zeros(n_neurons, n_neurons, dtype=torch.long)

        self.register_buffer("strengths", zeros_f.clone())
        self.register_buffer("evidence", zeros_f.clone())
        self.register_buffer("confidence", zeros_f.clone())
        self.register_buffer("age", zeros_l.clone())
        self.register_buffer("access_count", zeros_l.clone())
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))
        self.register_buffer("_prev_abs_outer", zeros_f.clone())

    @torch.no_grad()
    def consolidate(
        self, activations: torch.Tensor, reward: float = 1.0
    ) -> None:
        """Apply one Hebbian update and advance every state field.

        Args:
            activations: ``(B, n_neurons)`` tensor of activations.
                Caller is responsible for detaching from autograd.
            reward: Scalar reward modulating the update magnitude.
        """
        if activations.ndim != 2:
            raise ValueError(
                f"Expected 2-D activations (B, n), got shape "
                f"{tuple(activations.shape)}"
            )
        if activations.shape[1] != self.n_neurons:
            raise ValueError(
                f"Activation dim {activations.shape[1]} does not match "
                f"n_neurons={self.n_neurons}"
            )
        if activations.shape[0] == 0:
            return

        a = activations.detach().to(self.strengths.dtype)
        batch_size = a.shape[0]
        raw_outer = a.transpose(-1, -2) @ a / batch_size
        abs_outer = a.abs().transpose(-1, -2) @ a.abs() / batch_size

        # ---- strength update with normalised-evidence resistance ----
        if self.resistance_beta == 0.0:
            # Fast path identical to Phase-2 v1; bit-for-bit.
            self.strengths.add_(
                raw_outer, alpha=self.learning_rate * float(reward)
            )
        else:
            ev_max = self.evidence.max().clamp_min(_EVIDENCE_NORM_EPS)
            normalised_evidence = self.evidence / ev_max
            resistance = 1.0 / (1.0 + self.resistance_beta * normalised_evidence)
            self.strengths.add_(
                raw_outer * resistance,
                alpha=self.learning_rate * float(reward),
            )

        # ---- confidence: min(prev, curr) co-activation magnitude ----
        # Skip on the very first batch: there is no previous batch to
        # compare against, so confidence stays at zero.
        if int(self.global_step.item()) > 0:
            self.confidence.add_(torch.minimum(self._prev_abs_outer, abs_outer))

        # ---- age ticks for every synapse, every call ----
        self.age.add_(1)

        # ---- evidence accumulates, then prev_abs_outer caches ----
        self.evidence.add_(abs_outer)
        self._prev_abs_outer.copy_(abs_outer)
        self.global_step.add_(1)

        # ---- sparse eviction after every update ----
        # Mask is computed from the *post-update* strengths so that
        # eviction reflects what the synapse layer most recently
        # learned, not what it knew before this batch.
        if self.sparse and self.top_k < self.n_neurons:
            mask = compute_topk_mask(self.strengths, self.top_k)
            apply_topk_mask_inplace(
                [
                    self.strengths,
                    self.evidence,
                    self.confidence,
                    self.age,
                    self.access_count,
                    self._prev_abs_outer,
                ],
                mask,
            )

    @torch.no_grad()
    def record_access(
        self, features: torch.Tensor, threshold: float = 1e-3
    ) -> None:
        """Increment ``access_count`` for synapses that contributed.

        A synapse ``(i, j)`` is counted as having contributed in this
        batch when ``mean_b(|features[b, i]|) · |strengths[i, j]|``
        exceeds ``threshold``. The threshold is intentionally on the
        same scale as the post-modulation correction so that
        synapses with effectively-zero contribution don't get counted.

        Args:
            features: ``(B, n_neurons)`` activations the modulator saw.
            threshold: Minimum |contribution| to count as access.
        """
        if features.ndim != 2 or features.shape[1] != self.n_neurons:
            raise ValueError(
                f"features must be (B, {self.n_neurons}), got shape "
                f"{tuple(features.shape)}"
            )
        mean_abs = features.detach().abs().mean(dim=0)  # (n,)
        contribution = mean_abs.unsqueeze(1) * self.strengths.abs()
        self.access_count.add_((contribution > threshold).long())

    def reset(self) -> None:
        """Zero every buffer."""
        with torch.no_grad():
            for buf_name in (
                "strengths",
                "evidence",
                "confidence",
                "age",
                "access_count",
                "global_step",
                "_prev_abs_outer",
            ):
                getattr(self, buf_name).zero_()

    def extra_repr(self) -> str:
        sparsity = (
            f"sparse=True, top_k={self.top_k}" if self.sparse else "sparse=False"
        )
        return (
            f"n_neurons={self.n_neurons}, "
            f"learning_rate={self.learning_rate}, "
            f"resistance_beta={self.resistance_beta}, "
            f"{sparsity}, "
            f"global_step={int(self.global_step.item())}"
        )
