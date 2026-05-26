"""Active episodic memory with gradient-free allocation.

The memory substrate in the dual-substrate architecture. Stores
``(embedding, label, task_id)`` tuples and grows during training
whenever an input's novelty crosses a threshold. "Novelty" is one
minus the maximum cosine similarity between the input's penultimate
features and any stored embedding — so a sample that's well covered
by existing memory has novelty near 0, and a sample that looks like
nothing seen before has novelty near 1.

The store is **gradient-free**:

- Allocation decisions are made from a no-grad cosine comparison
  and never propagate into the model's parameters.
- The stored embeddings are detached snapshots; no autograd state
  is retained.
- The class supports a hard ``max_entries`` cap for ablations, but
  the default is unbounded — Phase D's experiments will measure
  empirical growth before any size-aware policy is added.

At inference, :meth:`retrieve` returns a label distribution from a
similarity-weighted top-k vote plus a confidence scalar (the max
similarity within the top-k). The blending into the model's softmax
happens in :class:`EpisodicPredictor`, not here — this class stays
narrow.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class ActiveEpisodicMemory:
    """In-memory episodic store with cosine-novelty allocation.

    Args:
        feature_dim: Dimensionality of stored embeddings. Used only
            for validation; the storage tensors are built lazily on
            first insert.
        n_classes: Number of output classes. Determines the shape of
            the retrieval probability vector.
        novelty_threshold: A sample is allocated iff its novelty
            (1 − max cosine sim to anything stored) is **strictly
            greater than** this value. ``0.7`` means "store anything
            that's less than 30 % similar to the closest existing
            entry". Defaults match the dual-substrate v1 spec.
        retrieval_k: Top-k entries consulted in the weighted vote.
        max_entries: Optional hard cap on the number of stored
            entries. ``None`` (default) means unbounded.
    """

    def __init__(
        self,
        feature_dim: int,
        n_classes: int,
        novelty_threshold: float = 0.7,
        retrieval_k: int = 5,
        max_entries: int | None = None,
    ) -> None:
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if n_classes <= 0:
            raise ValueError(f"n_classes must be positive, got {n_classes}")
        if not 0.0 <= novelty_threshold <= 1.0:
            raise ValueError(
                f"novelty_threshold must be in [0, 1], got {novelty_threshold}"
            )
        if retrieval_k <= 0:
            raise ValueError(f"retrieval_k must be positive, got {retrieval_k}")
        if max_entries is not None and max_entries <= 0:
            raise ValueError(
                f"max_entries must be positive or None, got {max_entries}"
            )

        self.feature_dim = int(feature_dim)
        self.n_classes = int(n_classes)
        self.novelty_threshold = float(novelty_threshold)
        self.retrieval_k = int(retrieval_k)
        self.max_entries = max_entries

        # Growing storage. Lists of CPU tensors / Python ints keep the
        # data structure simple and pickle-friendly; the cache below
        # holds a normalised stacked tensor for fast retrieval.
        self.embeddings: list[torch.Tensor] = []
        self.labels: list[int] = []
        self.task_ids: list[int | None] = []
        self._normalized_cache: torch.Tensor | None = None

    # ---- bookkeeping ----

    def _invalidate_cache(self) -> None:
        """Drop the cached normalised stack. Called on every insert."""
        self._normalized_cache = None

    def _normalized_embeddings(self) -> torch.Tensor | None:
        """Return all stored embeddings as a single ``(N, D)`` tensor,
        L2-normalised. ``None`` for an empty store. Cached until the
        next insert."""
        if not self.embeddings:
            return None
        if self._normalized_cache is None:
            stacked = torch.stack(self.embeddings)
            self._normalized_cache = F.normalize(stacked, dim=-1)
        return self._normalized_cache

    def __len__(self) -> int:
        return len(self.embeddings)

    # ---- novelty + allocation ----

    @torch.no_grad()
    def compute_novelty(self, features: torch.Tensor) -> torch.Tensor:
        """Return per-sample novelty score in ``[0, 1]``.

        ``novelty_i = 1 − max_j cos(features_i, stored_j)``, clamped to
        ``[0, 1]``. The clamp catches numerical drift around the
        boundaries (cosine similarity is bounded in ``[-1, 1]`` so the
        un-clamped novelty is in ``[0, 2]``; in practice large negative
        sims essentially never appear for L2-normalised vectors of
        real features but the clamp keeps the API contract tight).

        Empty memory returns all-ones — every sample is maximally
        novel when nothing is stored.
        """
        if features.ndim != 2:
            raise ValueError(
                f"features must be 2-D (B, D), got shape {tuple(features.shape)}"
            )
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"features dim {features.shape[1]} does not match "
                f"feature_dim={self.feature_dim}"
            )
        B = features.shape[0]
        normed_stored = self._normalized_embeddings()
        if normed_stored is None:
            return torch.ones(B, device=features.device)
        normed_q = F.normalize(features, dim=-1)
        sims = normed_q @ normed_stored.to(features.device).T  # (B, N)
        max_sims = sims.max(dim=-1).values  # (B,)
        return (1.0 - max_sims).clamp(min=0.0, max=1.0)

    @torch.no_grad()
    def maybe_allocate(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        task_id: int | None = None,
    ) -> int:
        """Allocate a new entry for each sample whose novelty exceeds
        the threshold. Returns the count of entries created this call.

        Per-sample novelty is computed once against the store **as of
        function entry**, so two highly-similar novel samples in the
        same batch will both allocate (rather than the second one
        seeing the first one's freshly-inserted entry). This matches
        the spec's "gradient-free, fast" property — making allocation
        order-dependent within a batch would couple it back to data
        ordering.

        Args:
            features: ``(B, feature_dim)`` penultimate activations.
                Detached and cloned before storage.
            labels: ``(B,)`` long integer class labels.
            task_id: Optional task identifier stored alongside each
                allocated entry. Used by diagnostics; not consulted
                by retrieval.
        """
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"features must be (B, {self.feature_dim}), got "
                f"{tuple(features.shape)}"
            )
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError(
                f"labels must be (B,) matching features batch; got "
                f"labels.shape={tuple(labels.shape)}, "
                f"features.shape={tuple(features.shape)}"
            )

        novelty = self.compute_novelty(features)
        novel_mask = novelty > self.novelty_threshold
        n_allocated = 0
        labels_long = labels.detach().to(torch.long).cpu()
        for i in range(features.shape[0]):
            if not bool(novel_mask[i].item()):
                continue
            if (
                self.max_entries is not None
                and len(self.embeddings) >= self.max_entries
            ):
                break
            self.embeddings.append(features[i].detach().clone().cpu())
            self.labels.append(int(labels_long[i].item()))
            self.task_ids.append(task_id)
            n_allocated += 1
        if n_allocated > 0:
            self._invalidate_cache()
        return n_allocated

    # ---- retrieval ----

    @torch.no_grad()
    def retrieve(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(retrieval_probs, confidence)`` for a query batch.

        ``retrieval_probs`` is a ``(B, n_classes)`` distribution from a
        similarity-weighted top-k vote: each of the top-``k`` nearest
        entries contributes ``max(cos_sim, 0)`` weight to its own
        class bucket, and the result is L1-normalised per row. The
        clip at zero stops anti-correlated entries from subtracting
        from a class's tally; the L1 normalisation keeps the row a
        proper distribution.

        ``confidence`` is ``(B,)``, the maximum similarity within the
        top-k entries — the "how relevant is the memory to this
        query?" signal that :class:`EpisodicPredictor` uses to scale
        its blend weight.

        Empty memory falls back to a uniform distribution and zero
        confidence so downstream callers don't have to special-case
        the no-entries regime.
        """
        if features.ndim != 2:
            raise ValueError(
                f"features must be 2-D (B, D), got shape {tuple(features.shape)}"
            )
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"features dim {features.shape[1]} does not match "
                f"feature_dim={self.feature_dim}"
            )
        B = features.shape[0]
        device = features.device
        normed_stored = self._normalized_embeddings()
        if normed_stored is None:
            uniform = torch.full(
                (B, self.n_classes), 1.0 / self.n_classes, device=device,
            )
            zero_conf = torch.zeros(B, device=device)
            return uniform, zero_conf

        normed_stored = normed_stored.to(device)
        normed_q = F.normalize(features, dim=-1)
        sims = normed_q @ normed_stored.T  # (B, N)
        k = min(self.retrieval_k, sims.shape[1])
        top_sims, top_idx = sims.topk(k, dim=-1)  # (B, k)

        stored_labels = torch.tensor(
            self.labels, dtype=torch.long, device=device,
        )  # (N,)
        top_labels = stored_labels[top_idx]  # (B, k)

        weights = top_sims.clamp(min=0.0)
        probs = torch.zeros(B, self.n_classes, device=device)
        probs.scatter_add_(1, top_labels, weights)
        denom = probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        probs = probs / denom

        confidence = top_sims.max(dim=-1).values  # (B,)
        return probs, confidence
