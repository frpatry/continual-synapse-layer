"""Phase 5.7.0 — prototype-based memory for class-incremental CL.

XRayMemory replaces raw-input memory with per-class *prototypes*:
k=3 learned feature-space vectors per class, refined via EMA on
correctly-classified inputs. Privacy by design — no raw inputs
are ever stored.

Two biology-inspired schedules tied to per-prototype refinement
count drive the novice → expert trajectory:

- **Progressive sparsification** ("abstraction"): once a prototype
  has been refined ``sparsity_start`` times, its smallest-magnitude
  dimensions are progressively zeroed out, reaching
  ``sparsity_max_drop_fraction`` zeros by ``sparsity_end``.

- **Temperature scaling** ("decision sharpening"): the contrastive
  comparison temperature decays from ``temperature_initial`` to
  ``temperature_final`` as the mean prototype refinement count
  grows from ``temp_start_refinements`` to ``temp_end_refinements``.

Implementation note: XRayMemory inherits from ``nn.Module`` so that
``register_buffer`` works (the user's design spec called for
``register_buffer`` semantics, which require ``nn.Module``).
Inheriting also gives ``.to(device)`` and ``state_dict()`` for
free, useful for Colab GPU runs and checkpoint resume.

The matching contrastive loss ``nt_xent_multi_prototype_loss`` is
defined in this same module so the two pieces stay coupled.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# Threshold below which a new feature is considered "different
# enough" from the nearest existing prototype to warrant a new
# slot allocation rather than an EMA update. Pulled out as a
# module-level constant so tests can patch it if needed.
_DISTINCTNESS_THRESHOLD: float = 0.6


class XRayMemory(nn.Module):
    """Per-class prototype memory with EMA refinement.

    Storage:
        prototypes:        (num_classes, k, feature_dim) float32
        refinement_counts: (num_classes, k) int64 — how many
            EMA updates each (class, slot) has absorbed
        is_occupied:       (num_classes, k) bool — which (class,
            slot) pairs currently hold a prototype

    The class deliberately exposes NO ``inputs`` attribute. There
    is no path by which raw inputs could leak in or out — only
    feature vectors (post-encoder) ever touch the buffers, and
    even those are EMA-blended rather than stored verbatim.
    """

    def __init__(
        self,
        num_classes: int = 100,
        feature_dim: int = 128,
        prototypes_per_class: int = 3,
        sparsity_start_refinements: int = 10,
        sparsity_end_refinements: int = 50,
        sparsity_max_drop_fraction: float = 0.5,
        temp_start_refinements: int = 50,
        temp_end_refinements: int = 200,
        temperature_initial: float = 1.0,
        temperature_final: float = 0.3,
        distinctness_threshold: float = _DISTINCTNESS_THRESHOLD,
    ) -> None:
        super().__init__()
        if num_classes <= 0 or feature_dim <= 0 or prototypes_per_class <= 0:
            raise ValueError(
                "num_classes, feature_dim, prototypes_per_class "
                "must all be positive"
            )
        if sparsity_end_refinements <= sparsity_start_refinements:
            raise ValueError(
                "sparsity_end_refinements must be > sparsity_start_refinements"
            )
        if temp_end_refinements <= temp_start_refinements:
            raise ValueError(
                "temp_end_refinements must be > temp_start_refinements"
            )
        if not 0.0 <= sparsity_max_drop_fraction < 1.0:
            raise ValueError(
                "sparsity_max_drop_fraction must be in [0, 1)"
            )

        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.k = int(prototypes_per_class)

        self.register_buffer(
            "prototypes",
            torch.zeros(self.num_classes, self.k, self.feature_dim),
        )
        self.register_buffer(
            "refinement_counts",
            torch.zeros(self.num_classes, self.k, dtype=torch.long),
        )
        self.register_buffer(
            "is_occupied",
            torch.zeros(self.num_classes, self.k, dtype=torch.bool),
        )

        # Schedule parameters (kept as Python floats / ints, not
        # buffers — they don't move across devices).
        self.sparsity_start = int(sparsity_start_refinements)
        self.sparsity_end = int(sparsity_end_refinements)
        self.sparsity_max_drop = float(sparsity_max_drop_fraction)
        self.temp_start = int(temp_start_refinements)
        self.temp_end = int(temp_end_refinements)
        self.temp_init = float(temperature_initial)
        self.temp_final = float(temperature_final)
        self.distinctness_threshold = float(distinctness_threshold)

    # ---------- write path ----------

    @torch.no_grad()
    def update(
        self,
        features: Tensor,
        labels: Tensor,
        correct_mask: Tensor,
    ) -> int:
        """Refine prototypes from a batch of (feature, label, was_correct)
        tuples.

        Only entries where ``correct_mask[i]`` is True are used —
        the "reward signal" is correct classification.

        Allocation/refinement policy per entry:
        1. If no prototype exists yet for the true label → allocate
           slot 0 with the feature verbatim.
        2. Otherwise compute cosine similarity to every existing
           prototype of the true label. If the highest similarity
           is below ``distinctness_threshold`` AND there is an
           empty slot available, allocate a NEW slot. Otherwise
           EMA-update the nearest existing prototype.
        3. After an EMA update, apply progressive sparsification
           if the refinement count is in the active range.

        Returns: number of prototype updates (allocations + EMA
        updates combined) the call performed.
        """
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"features must be (B, {self.feature_dim}); got "
                f"{tuple(features.shape)}"
            )
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError(
                f"labels must be (B,) matching features; got "
                f"{tuple(labels.shape)} for B={features.shape[0]}"
            )
        if correct_mask.shape != labels.shape:
            raise ValueError(
                "correct_mask must have the same shape as labels"
            )

        features = features.to(self.prototypes.device)
        labels = labels.to(self.prototypes.device)
        correct_mask = correct_mask.to(self.prototypes.device)

        n_updated = 0
        for i in range(features.shape[0]):
            if not bool(correct_mask[i].item()):
                continue
            label = int(labels[i].item())
            if not 0 <= label < self.num_classes:
                raise ValueError(
                    f"label {label} out of range [0, {self.num_classes})"
                )
            feat = features[i]

            occupied = self.is_occupied[label]
            if not bool(occupied.any().item()):
                # First prototype for this class — slot 0.
                self.prototypes[label, 0] = feat.clone()
                self.is_occupied[label, 0] = True
                self.refinement_counts[label, 0] = 1
                n_updated += 1
                continue

            # Find nearest occupied prototype via cosine similarity.
            existing = self.prototypes[label]
            existing_n = F.normalize(existing, dim=-1)
            feat_n = F.normalize(feat, dim=-1)
            sims = (existing_n * feat_n.unsqueeze(0)).sum(dim=-1)
            sims = sims.masked_fill(~occupied, float("-inf"))
            best_slot = int(sims.argmax().item())
            best_sim = float(sims[best_slot].item())

            if (
                best_sim < self.distinctness_threshold
                and not bool(occupied.all().item())
            ):
                # Sufficiently different from anything we've seen for
                # this class → allocate a new slot.
                new_slot = int((~occupied).nonzero()[0].item())
                self.prototypes[label, new_slot] = feat.clone()
                self.is_occupied[label, new_slot] = True
                self.refinement_counts[label, new_slot] = 1
                n_updated += 1
                continue

            # EMA-update the nearest occupied slot. Weight decreases
            # with refinement count so older prototypes change less.
            n_prev = int(self.refinement_counts[label, best_slot].item())
            ema_weight = 1.0 / (1.0 + float(n_prev) ** 0.5)
            self.prototypes[label, best_slot] = (
                (1.0 - ema_weight) * self.prototypes[label, best_slot]
                + ema_weight * feat
            )
            self.refinement_counts[label, best_slot] = n_prev + 1
            n_updated += 1

            # Progressive sparsification kicks in once the slot has
            # crossed ``sparsity_start``.
            refinement = n_prev + 1
            if refinement >= self.sparsity_start:
                drop_fraction = self._sparsification_schedule(refinement)
                if drop_fraction > 0.0:
                    proto = self.prototypes[label, best_slot]
                    n_dims = proto.shape[0]
                    n_to_keep = max(1, int(n_dims * (1.0 - drop_fraction)))
                    _, top_idx = proto.abs().topk(n_to_keep)
                    mask = torch.zeros_like(proto)
                    mask[top_idx] = 1.0
                    self.prototypes[label, best_slot] = proto * mask

        return n_updated

    # ---------- schedules ----------

    def _sparsification_schedule(self, refinement: int) -> float:
        """Linear ramp from 0 at ``sparsity_start`` to
        ``sparsity_max_drop`` at ``sparsity_end``. Clamped at both
        ends."""
        if refinement < self.sparsity_start:
            return 0.0
        if refinement >= self.sparsity_end:
            return self.sparsity_max_drop
        span = self.sparsity_end - self.sparsity_start
        progress = (refinement - self.sparsity_start) / span
        return progress * self.sparsity_max_drop

    def temperature(self, mean_refinement_count: float | None = None) -> float:
        """Current contrastive temperature.

        With no argument, uses the mean refinement count across
        all currently-occupied (class, slot) pairs. Empty memory
        → ``temperature_initial``.

        Schedule: linear interp from ``temp_init`` at
        ``temp_start_refinements`` to ``temp_final`` at
        ``temp_end_refinements``; clamped at both ends.
        """
        if mean_refinement_count is None:
            occupied_counts = self.refinement_counts[self.is_occupied]
            if occupied_counts.numel() == 0:
                return self.temp_init
            mean_refinement_count = float(occupied_counts.float().mean().item())

        if mean_refinement_count < self.temp_start:
            return self.temp_init
        if mean_refinement_count >= self.temp_end:
            return self.temp_final
        span = self.temp_end - self.temp_start
        progress = (mean_refinement_count - self.temp_start) / span
        return self.temp_init + progress * (self.temp_final - self.temp_init)

    # ---------- read path ----------

    def get_all_prototypes(self) -> tuple[Tensor, Tensor]:
        """Return ``(prototypes, labels)`` for every occupied slot.

        prototypes: ``(n_occupied, feature_dim)``
        labels:     ``(n_occupied,)`` int64 — class label per prototype

        Empty memory → empty tensors with the right shapes.
        """
        occupied_idx = self.is_occupied.nonzero(as_tuple=False)
        if occupied_idx.numel() == 0:
            return (
                torch.empty(
                    0, self.feature_dim,
                    device=self.prototypes.device,
                    dtype=self.prototypes.dtype,
                ),
                torch.empty(
                    0, dtype=torch.long, device=self.prototypes.device,
                ),
            )
        class_idx = occupied_idx[:, 0]
        slot_idx  = occupied_idx[:, 1]
        protos = self.prototypes[class_idx, slot_idx]
        return protos, class_idx.clone()

    def num_occupied(self) -> int:
        return int(self.is_occupied.sum().item())

    def per_class_counts(self) -> list[int]:
        return [
            int(self.is_occupied[c].sum().item())
            for c in range(self.num_classes)
        ]


# ---------- multi-prototype NT-Xent loss ----------


def nt_xent_multi_prototype_loss(
    features: Tensor,
    labels: Tensor,
    prototypes: Tensor,
    prototype_labels: Tensor,
    temperature: float = 0.5,
) -> Tensor:
    """Multi-positive NT-Xent (SupCon-style) loss against prototypes.

    For each query in ``features`` with label ``labels[i]``:
        - every prototype with matching label is a *positive*
        - every prototype with a different label is a *negative*
    The loss is the negative mean log-softmax probability assigned
    to the positives, averaged across queries that have at least
    one positive in memory.

    Args:
        features:         ``(B, D)`` query features, L2-normalised
            inside the function.
        labels:           ``(B,)`` int true class label per query.
        prototypes:       ``(P, D)`` stored prototypes (any class).
        prototype_labels: ``(P,)`` int class label per prototype.
        temperature:      similarity scaling (lower = sharper).

    Returns:
        Scalar loss. ``0`` (no grad) when the memory has zero
        prototypes or when none of the query labels have a matching
        prototype available.
    """
    device = features.device
    if prototypes.shape[0] == 0:
        return torch.zeros((), device=device)

    features_n = F.normalize(features, dim=-1)
    prototypes_n = F.normalize(prototypes.to(device), dim=-1)
    prototype_labels = prototype_labels.to(device)
    labels = labels.to(device)

    sims = (features_n @ prototypes_n.T) / max(temperature, 1e-8)
    pos_mask = (
        labels.unsqueeze(1) == prototype_labels.unsqueeze(0)
    )  # (B, P) bool

    n_positives = pos_mask.float().sum(dim=-1)
    has_positives = n_positives > 0
    if not bool(has_positives.any().item()):
        return torch.zeros((), device=device)

    log_probs = F.log_softmax(sims, dim=-1)
    pos_log_probs_sum = (log_probs * pos_mask.float()).sum(dim=-1)
    # ``clamp(min=1.0)`` is just a safe divisor — entries with
    # n_positives==0 are filtered out via has_positives below.
    pos_log_probs_mean = pos_log_probs_sum / n_positives.clamp(min=1.0)

    return -pos_log_probs_mean[has_positives].mean()
