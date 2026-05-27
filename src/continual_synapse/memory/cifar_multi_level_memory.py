"""CIFAR multi-level memory for Phase 5.6.

Adapts the Phase 4 / Phase 5.5 ``MultiLevelMemory`` to the CIFAR
class-incremental setup. Per entry:

- ``inputs``       : ``(3, 32, 32)`` — the input image. Stored as
  whatever dtype the caller passes (typically the normalized
  float32 produced by the CIFAR data pipeline, which means no
  re-normalisation is needed at replay time).
- ``hipp_low_gap`` : ``(C_low_hipp,)``  float32 — GAP of hipp.features['low']
- ``hipp_mid_gap`` : ``(C_mid_hipp,)``  float32
- ``hipp_high_gap``: ``(C_high_hipp,)`` float32
- ``neo_low_gap``  : ``(C_low_neo,)``   float32
- ``neo_mid_gap``  : ``(C_mid_neo,)``   float32
- ``neo_high_gap`` : ``(C_high_neo,)``  float32
- ``soft_targets`` : ``(num_classes,)`` float32 — softmax of the
  hippocampe's logits at storage time
- ``labels``       : int — ground-truth class
- ``classes_seen`` : ``list[int]`` — class set the hippocampe had
  seen by storage time (used for masked-KL distillation later)

Reservoir sampling caps the buffer at ``max_total``. The
hippocampe's GAP features are the ones the Phase 5.6.3
consolidation step will anchor against; the neocortex's GAP
features are stored speculatively for future use (e.g., neo
feature-level distillation) and are not read by Variant C
consolidation as currently designed.

Note on input dtype: the spec aspired to store raw uint8 inputs
(~3 KB / entry). In practice the caller is in the training loop
and has only the augmented + normalised float32 tensor. We
accept whatever dtype is handed in and document the tradeoff:
storing the float input is ~3-4× heavier but avoids re-augmenting
/ re-normalising at replay time. At ``max_total=5000`` the buffer
is still under 90 MB.
"""

from __future__ import annotations

import random
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _gap(feature_map: Tensor) -> Tensor:
    """Global-average-pool a (B, C, H, W) feature map to (B, C)."""
    return F.adaptive_avg_pool2d(feature_map, 1).flatten(1)


class CIFARMultiLevelMemory:
    """Reservoir-sampled multi-level feature memory for Split-CIFAR-100 CI."""

    def __init__(
        self,
        max_total: int = 5000,
        num_classes: int = 100,
        rng_seed: int | None = None,
    ) -> None:
        if max_total <= 0:
            raise ValueError(f"max_total must be positive, got {max_total}")
        self.max_total = int(max_total)
        self.num_classes = int(num_classes)
        # Reservoir counter: how many items have been seen total.
        # NOT the same as len(self) once the buffer saturates.
        self.n_seen = 0

        self.inputs:        list[Tensor] = []
        self.hipp_low_gap:  list[Tensor] = []
        self.hipp_mid_gap:  list[Tensor] = []
        self.hipp_high_gap: list[Tensor] = []
        self.neo_low_gap:   list[Tensor] = []
        self.neo_mid_gap:   list[Tensor] = []
        self.neo_high_gap:  list[Tensor] = []
        self.soft_targets:  list[Tensor] = []
        self.labels:        list[int]    = []
        self.classes_seen:  list[list[int]] = []

        # Per-instance RNG so multiple memories in the same process
        # (e.g., across seeds) don't fight over the module-level
        # random state.
        self._rng: random.Random | random._RandomLike  # type: ignore[name-defined]
        self._rng = random.Random(rng_seed) if rng_seed is not None else random

    # ---------- write path ----------

    @torch.no_grad()
    def record_batch(
        self,
        inputs: Tensor,
        labels: Tensor,
        hippocampus: nn.Module,
        neocortex: nn.Module,
        classes_seen_so_far: Sequence[int],
    ) -> int:
        """Snapshot a batch into the memory via reservoir sampling.

        Args:
            inputs: ``(B, 3, 32, 32)`` — float32 (normalised) or
                uint8. The caller picks; the dtype is preserved on
                storage so replay can feed it back unchanged.
            labels: ``(B,)`` int — ground-truth class indices.
            hippocampus: model exposing ``features(x)`` returning a
                ``{low, mid, high}`` dict of 4-D feature maps, and
                a ``__call__(x)`` returning logits over
                ``num_classes``.
            neocortex: same protocol; only ``features(x)`` is used.
            classes_seen_so_far: classes the hippocampe had seen by
                the time this batch is recorded. Stored verbatim
                per entry so the consolidation step's masked KL
                later renormalises against the right subset.

        Returns:
            Count of entries added or replaced this call. NOT the
            same as ``len(self)`` change when the reservoir is
            saturated.
        """
        if inputs.ndim != 4 or inputs.shape[1:] != (3, 32, 32):
            raise ValueError(
                f"inputs must be (B, 3, 32, 32); got "
                f"{tuple(inputs.shape)}"
            )
        if labels.ndim != 1 or labels.shape[0] != inputs.shape[0]:
            raise ValueError("labels must be (B,) matching inputs")

        device = next(hippocampus.parameters()).device
        x = inputs.to(device)
        if x.dtype == torch.uint8:
            x = x.to(torch.float32) / 255.0

        # Switch both models to eval temporarily — BatchNorm in
        # train mode would update its running stats on this
        # forward, which we don't want for a pure "snapshot" call.
        # ``torch.no_grad`` on its own doesn't suppress those
        # running-stats updates.
        hipp_was_training = hippocampus.training
        neo_was_training = neocortex.training
        hippocampus.eval()
        neocortex.eval()
        try:
            hipp_feats = hippocampus.features(x)
            hipp_logits = hippocampus(x)
            hipp_soft = F.softmax(hipp_logits, dim=-1).detach().cpu()
            neo_feats = neocortex.features(x)
        finally:
            hippocampus.train(hipp_was_training)
            neocortex.train(neo_was_training)

        h_low  = _gap(hipp_feats["low"]).detach().cpu()
        h_mid  = _gap(hipp_feats["mid"]).detach().cpu()
        h_high = _gap(hipp_feats["high"]).detach().cpu()
        n_low  = _gap(neo_feats["low"]).detach().cpu()
        n_mid  = _gap(neo_feats["mid"]).detach().cpu()
        n_high = _gap(neo_feats["high"]).detach().cpu()

        # Always store inputs CPU-side so the buffer doesn't pin
        # GPU memory.
        inputs_cpu = inputs.detach().cpu()
        labels_cpu = labels.detach().cpu()
        classes_snapshot: list[int] = [int(c) for c in classes_seen_so_far]

        B = inputs.shape[0]
        n_added = 0
        for i in range(B):
            entry = {
                "input":         inputs_cpu[i].clone(),
                "hipp_low_gap":  h_low[i].clone(),
                "hipp_mid_gap":  h_mid[i].clone(),
                "hipp_high_gap": h_high[i].clone(),
                "neo_low_gap":   n_low[i].clone(),
                "neo_mid_gap":   n_mid[i].clone(),
                "neo_high_gap":  n_high[i].clone(),
                "soft_target":   hipp_soft[i].clone(),
                "label":         int(labels_cpu[i].item()),
                "classes_seen":  list(classes_snapshot),
            }
            self.n_seen += 1
            if len(self.inputs) < self.max_total:
                self._append_entry(entry)
                n_added += 1
            else:
                # Reservoir: pick j uniformly in [0, n_seen);
                # if j < max_total, replace at j.
                j = self._rng.randrange(self.n_seen)
                if j < self.max_total:
                    self._replace_entry(j, entry)
                    n_added += 1
        return n_added

    def _append_entry(self, entry: dict) -> None:
        self.inputs.append(entry["input"])
        self.hipp_low_gap.append(entry["hipp_low_gap"])
        self.hipp_mid_gap.append(entry["hipp_mid_gap"])
        self.hipp_high_gap.append(entry["hipp_high_gap"])
        self.neo_low_gap.append(entry["neo_low_gap"])
        self.neo_mid_gap.append(entry["neo_mid_gap"])
        self.neo_high_gap.append(entry["neo_high_gap"])
        self.soft_targets.append(entry["soft_target"])
        self.labels.append(entry["label"])
        self.classes_seen.append(entry["classes_seen"])

    def _replace_entry(self, idx: int, entry: dict) -> None:
        self.inputs[idx]        = entry["input"]
        self.hipp_low_gap[idx]  = entry["hipp_low_gap"]
        self.hipp_mid_gap[idx]  = entry["hipp_mid_gap"]
        self.hipp_high_gap[idx] = entry["hipp_high_gap"]
        self.neo_low_gap[idx]   = entry["neo_low_gap"]
        self.neo_mid_gap[idx]   = entry["neo_mid_gap"]
        self.neo_high_gap[idx]  = entry["neo_high_gap"]
        self.soft_targets[idx]  = entry["soft_target"]
        self.labels[idx]        = entry["label"]
        self.classes_seen[idx]  = entry["classes_seen"]

    # ---------- read path ----------

    def sample_batch(
        self, batch_size: int, device: torch.device | str = "cpu",
    ) -> dict | None:
        """Return a uniform random batch as stacked tensors, or
        ``None`` when the memory is empty.

        Keys returned:
            inputs:        (n, 3, 32, 32) — same dtype as stored
            hipp_low_gap / hipp_mid_gap / hipp_high_gap: (n, C_*)
            neo_low_gap  / neo_mid_gap  / neo_high_gap : (n, C_*)
            soft_targets: (n, num_classes)
            labels:       (n,) int64
            classes_seen: list[list[int]] of length n
        """
        if not self.inputs:
            return None
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        n = min(batch_size, len(self.inputs))
        idx = self._rng.sample(range(len(self.inputs)), n)
        device_t = torch.device(device)

        def _stack(lst: list[Tensor]) -> Tensor:
            return torch.stack([lst[i] for i in idx]).to(device_t)

        return {
            "inputs":        _stack(self.inputs),
            "hipp_low_gap":  _stack(self.hipp_low_gap),
            "hipp_mid_gap":  _stack(self.hipp_mid_gap),
            "hipp_high_gap": _stack(self.hipp_high_gap),
            "neo_low_gap":   _stack(self.neo_low_gap),
            "neo_mid_gap":   _stack(self.neo_mid_gap),
            "neo_high_gap":  _stack(self.neo_high_gap),
            "soft_targets":  _stack(self.soft_targets),
            "labels":        torch.tensor(
                [self.labels[i] for i in idx],
                dtype=torch.long, device=device_t,
            ),
            "classes_seen":  [self.classes_seen[i] for i in idx],
        }

    # ---------- diagnostics ----------

    def __len__(self) -> int:
        return len(self.inputs)

    def per_class_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for c in self.labels:
            counts[c] = counts.get(c, 0) + 1
        return counts

    def per_class_range_counts(
        self, ranges: Iterable[tuple[int, int]],
    ) -> list[int]:
        """Count how many stored entries fall in each ``[lo, hi)``
        class range. Convenient for "how many entries per task"
        summaries with contiguous class blocks."""
        out: list[int] = []
        for lo, hi in ranges:
            c = sum(1 for lbl in self.labels if lo <= lbl < hi)
            out.append(c)
        return out
