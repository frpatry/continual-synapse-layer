"""Cold Storage v2 — inference-time retrieval ensemble.

Turns the cold-storage embeddings from a passive store into an
active second-opinion predictor. At inference, the model's
penultimate-layer activation for each test sample is compared
(cosine similarity) against every stored embedding; when a
sample's most-similar stored entry passes a threshold ``tau``,
the model's softmax is blended with a similarity-weighted vote
over the top-k entries' labels.

The "labels" for stored embeddings are derived once at init by
passing each stored embedding through the *current* model's
classifier head and taking argmax. This is the CLS-style
retrieval ensemble pattern: the model still does the labelling,
but old activation patterns get to vote for what they used to
predict.

Reference: motivated by the exp 21 / exp 23 finding that
``cs_gated_cosine_developmental`` at T=50 retains Task-0 at
0.619 while EWC retains it at 0.834. The cold storage in the
gated variant is currently only used for gradient gating during
training; at inference the stored patterns contribute nothing.
This module exposes them.

Design knobs:
- ``k``: number of nearest stored entries to consider per sample.
- ``tau``: cosine-similarity threshold for "this sample looks
  familiar enough to consult the archive". Below the threshold,
  the model's prediction is used unmodified.
- ``lambda_blend``: blend weight for the retrieval vote. Higher
  values trust the archive more; ``1.0`` ignores the model's
  softmax entirely on high-similarity samples.

All defaults are intentionally conservative (``k=5``, ``tau=0.7``,
``lambda_blend=0.3``) — preserves model behaviour for samples that
don't look like anything in the archive.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from continual_synapse.baselines.naive_finetune import MLPClassifier
from continual_synapse.baselines.synapse_finetune import SynapseAugmentedMLP
from continual_synapse.cold_storage.store import ColdStorage


@dataclass
class RetrievalEnsembleConfig:
    """Hyperparameters for :class:`RetrievalEnsemble`. Exposed as a
    dataclass so experiment scripts can iterate sweeps cleanly."""

    name: str
    k: int = 5
    tau: float = 0.7
    lambda_blend: float = 0.3


class RetrievalEnsemble:
    """Blend a model's softmax with a similarity-weighted vote over
    cold-storage entries.

    The wrapper is callable via :meth:`predict` (or ``__call__``).
    It does not modify ``model`` or ``cold_storage`` — purely a
    read-only wrapper around a trained system.

    Args:
        model: Either a bare :class:`MLPClassifier` (single-head)
            or a :class:`SynapseAugmentedMLP` wrapping one. Must
            expose ``features(x)`` that returns the penultimate-
            layer activations (same vector space as the stored
            embeddings).
        cold_storage_embeddings: ``(N, D)`` tensor of stored
            embeddings, one row per cold-storage entry.
        cold_storage_labels: ``(N,)`` long tensor of derived
            labels (argmax of the current model's classifier head
            applied to each stored embedding). Constructed by
            :meth:`from_model_and_storage` at init time.
        k: How many nearest entries to consider per sample.
        tau: Cosine-similarity threshold for engaging the retrieval
            vote. Samples whose top-1 stored similarity falls at or
            below ``tau`` fall through to the unmodified model
            prediction.
        lambda_blend: Mixing weight for the retrieval vote when
            engaged. Final probabilities are
            ``(1 - λ) * softmax_model + λ * retrieval_probs``.
        eps: Numerical-stability floor for log / division.
    """

    def __init__(
        self,
        model: nn.Module,
        cold_storage_embeddings: torch.Tensor,
        cold_storage_labels: torch.Tensor,
        k: int = 5,
        tau: float = 0.7,
        lambda_blend: float = 0.3,
        eps: float = 1e-8,
    ) -> None:
        if cold_storage_embeddings.dim() != 2:
            raise ValueError(
                f"cold_storage_embeddings must be 2-D (N, D), got shape "
                f"{tuple(cold_storage_embeddings.shape)}"
            )
        if cold_storage_labels.dim() != 1:
            raise ValueError(
                f"cold_storage_labels must be 1-D (N,), got shape "
                f"{tuple(cold_storage_labels.shape)}"
            )
        if cold_storage_embeddings.shape[0] != cold_storage_labels.shape[0]:
            raise ValueError(
                f"embeddings ({cold_storage_embeddings.shape[0]}) and labels "
                f"({cold_storage_labels.shape[0]}) disagree on N."
            )
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if not 0.0 <= tau <= 1.0:
            raise ValueError(f"tau must be in [0, 1], got {tau}")
        if not 0.0 <= lambda_blend <= 1.0:
            raise ValueError(
                f"lambda_blend must be in [0, 1], got {lambda_blend}"
            )
        self.model = model
        self.embeddings = cold_storage_embeddings
        self.labels = cold_storage_labels.to(torch.long)
        self.k = int(k)
        self.tau = float(tau)
        self.lambda_blend = float(lambda_blend)
        self.eps = float(eps)
        # Populated by from_model_and_storage; None for direct __init__
        # because the constructor can't introspect where the labels
        # came from when handed a raw tensor.
        self._label_source_breakdown: dict[str, int] | None = None

    @property
    def label_source_breakdown(self) -> dict[str, int] | None:
        """Per-entry tally of where the labels came from at construction.

        Returns a dict with keys ``true_label``, ``derived``, and
        ``total`` (summing to ``total``). ``None`` for ensembles built
        through ``__init__`` directly — only populated when constructed
        via :meth:`from_model_and_storage`, which is the path that
        actually knows the provenance of each label.

        Useful for logging path-A coverage in experiment summaries,
        e.g. ``487 true_label, 153 derived (76% coverage)`` on a
        mixed-vintage store.
        """
        if self._label_source_breakdown is None:
            return None
        return dict(self._label_source_breakdown)

    # ---- construction from training artifacts ----

    @classmethod
    def from_model_and_storage(
        cls,
        model: nn.Module,
        cold_storage: ColdStorage,
        k: int = 5,
        tau: float = 0.7,
        lambda_blend: float = 0.3,
        device: torch.device | str | None = None,
        label_source: str = "auto",
    ) -> "RetrievalEnsemble":
        """Pull every embedding from ``cold_storage`` and build a ready
        ensemble.

        Args:
            label_source: How to assign a label to each stored entry.

                - ``"auto"`` (default): use ``metadata["true_label"]``
                  when present, fall back to deriving via the current
                  model's classifier head otherwise. Path-A entries
                  use their ground-truth label; path-B / pre-path-A
                  entries fall through to the original derivation.
                  Mixed-vintage stores work seamlessly.
                - ``"true_label"``: strict path-A. Every entry must
                  carry ``metadata["true_label"]``; raises
                  :class:`ValueError` otherwise. Use this on stores
                  trained with label storage enabled when you want a
                  hard guarantee that no derived labels leak in.
                - ``"derived"``: ignore any stored ``true_label`` and
                  derive every label via the current model's head.
                  This is the path-B behaviour, kept available for
                  A/B ablations against path A.

        ``device``: where to keep the stored embeddings and labels.
        Defaults to whichever device the model's first parameter lives
        on, so cosine sim doesn't cross device.

        On return, :attr:`label_source_breakdown` reports how many
        labels came from each source.
        """
        if label_source not in ("auto", "true_label", "derived"):
            raise ValueError(
                f"label_source must be one of "
                f"'auto', 'true_label', 'derived'; got {label_source!r}"
            )

        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        device = torch.device(device)

        entries = cold_storage.all_entries() if cold_storage.count() > 0 else []
        if not entries:
            n_neurons = _infer_n_neurons(model)
            ens = cls(
                model=model,
                cold_storage_embeddings=torch.zeros(0, n_neurons, device=device),
                cold_storage_labels=torch.zeros(0, dtype=torch.long, device=device),
                k=k, tau=tau, lambda_blend=lambda_blend,
            )
            ens._label_source_breakdown = {
                "true_label": 0, "derived": 0, "total": 0,
            }
            return ens

        embeddings = torch.tensor(
            [list(e.embedding) for e in entries],
            dtype=torch.float32, device=device,
        )
        n = len(entries)
        stored = [e.metadata.get("true_label") for e in entries]
        n_stored = sum(1 for v in stored if v is not None)

        if label_source == "true_label":
            if n_stored != n:
                missing_idx = next(i for i, v in enumerate(stored) if v is None)
                raise ValueError(
                    f"label_source='true_label' requires every entry to "
                    f"carry metadata['true_label']; {n - n_stored} of "
                    f"{n} entries are missing it (e.g. id="
                    f"{entries[missing_idx].id!r}). Use label_source="
                    f"'auto' to mix derived labels in as a fallback."
                )
            labels = torch.tensor(
                [int(v) for v in stored], dtype=torch.long, device=device,
            )
            breakdown = {"true_label": n, "derived": 0, "total": n}
        elif label_source == "derived":
            labels = cls._derive_labels(model, embeddings)
            breakdown = {"true_label": 0, "derived": n, "total": n}
        else:  # "auto"
            if n_stored == 0:
                labels = cls._derive_labels(model, embeddings)
                breakdown = {"true_label": 0, "derived": n, "total": n}
            elif n_stored == n:
                labels = torch.tensor(
                    [int(v) for v in stored],
                    dtype=torch.long, device=device,
                )
                breakdown = {"true_label": n, "derived": 0, "total": n}
            else:
                # Mixed-vintage: derive everything (one batched head
                # call), then overlay stored labels where present.
                derived = cls._derive_labels(model, embeddings)
                labels = derived.clone()
                for i, v in enumerate(stored):
                    if v is not None:
                        labels[i] = int(v)
                breakdown = {
                    "true_label": n_stored,
                    "derived": n - n_stored,
                    "total": n,
                }

        ens = cls(
            model=model,
            cold_storage_embeddings=embeddings,
            cold_storage_labels=labels,
            k=k, tau=tau, lambda_blend=lambda_blend,
        )
        ens._label_source_breakdown = breakdown
        return ens

    @staticmethod
    def _derive_labels(
        model: nn.Module, embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Argmax of the model's classifier head applied to each
        stored embedding."""
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                logits = _classify_features(model, embeddings)
            return logits.argmax(dim=-1).to(torch.long)
        finally:
            if was_training:
                model.train()

    # ---- inference ----

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return blended logits for batch ``x``.

        Samples whose top-1 stored cosine similarity is at or below
        ``tau`` get the model's unmodified logits. Above ``tau``,
        the model's softmax is blended with a similarity-weighted
        vote over the top-``k`` nearest stored entries' labels.
        """
        was_training = self.model.training
        self.model.eval()
        try:
            h = _model_features(self.model, x)  # (B, D)
            logits_model = _classify_features(self.model, h)  # (B, C)

            # Empty cold storage ⇒ no retrieval is even possible.
            if self.embeddings.shape[0] == 0:
                return logits_model

            # Cosine similarity. F.normalize zeroes the denominator
            # for zero-norm rows (eps in the denom prevents NaN); a
            # zero query / zero stored entry will produce 0 similarity.
            h_norm = F.normalize(h, dim=-1, eps=1e-12)
            stored_norm = F.normalize(self.embeddings, dim=-1, eps=1e-12)
            sim = h_norm @ stored_norm.T  # (B, N)

            # Top-k similarities + labels. k capped at N if needed so
            # caller doesn't have to special-case small archives.
            k_eff = min(self.k, sim.shape[1])
            top_k_sim, top_k_idx = sim.topk(k_eff, dim=-1)
            top_k_labels = self.labels[top_k_idx]  # (B, k_eff)
            max_sim = top_k_sim.max(dim=-1).values  # (B,)

            # Weighted vote per class. scatter_add accumulates each
            # neighbour's similarity into the corresponding class bin.
            # Use weights clipped at 0 so anti-correlated entries don't
            # subtract from the tally.
            B, C = logits_model.shape
            weights = top_k_sim.clamp_min(0.0)
            retrieval_counts = torch.zeros(
                B, C, dtype=weights.dtype, device=weights.device,
            )
            retrieval_counts.scatter_add_(1, top_k_labels, weights)
            denom = retrieval_counts.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            retrieval_probs = retrieval_counts / denom  # (B, C)

            # Blend on the probability simplex, then log to return
            # logit-shaped tensor that argmax-equivalent downstream
            # code can handle exactly like the original logits.
            softmax_model = F.softmax(logits_model, dim=-1)
            blended_probs = (
                (1.0 - self.lambda_blend) * softmax_model
                + self.lambda_blend * retrieval_probs
            )
            blended_logits = torch.log(blended_probs.clamp_min(self.eps))

            # Conditional fallback: low-sim samples keep model logits.
            engage = (max_sim > self.tau).unsqueeze(-1)  # (B, 1)
            return torch.where(engage, blended_logits, logits_model)
        finally:
            if was_training:
                self.model.train()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict(x)


# ---------- private helpers ----------


def _infer_n_neurons(model: nn.Module) -> int:
    """Best-effort: pull hidden dimension from either model variant."""
    if isinstance(model, SynapseAugmentedMLP):
        return int(model.synapse.n_neurons)
    if isinstance(model, MLPClassifier):
        return int(model.config.hidden_dim)
    if hasattr(model, "config") and hasattr(model.config, "hidden_dim"):
        return int(model.config.hidden_dim)
    raise ValueError(
        "could not infer n_neurons from model — neither "
        "SynapseAugmentedMLP, MLPClassifier, nor anything with "
        "model.config.hidden_dim"
    )


def _model_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Get the penultimate-layer activations from either model variant.

    SynapseAugmentedMLP.features(x) in eval mode returns f_base unchanged
    when gate_modulation_enabled is False (the cs_gated setting). For
    gate_modulation_enabled=True, features() returns f_base + correction
    — we deliberately take the BASE features only, so the retrieval
    query lives in the same vector space the cold-storage embeddings
    were stored in (which is f_base.mean(0)).
    """
    if isinstance(model, SynapseAugmentedMLP):
        return model.base.features(x)
    return model.features(x)


def _classify_features(model: nn.Module, features: torch.Tensor) -> torch.Tensor:
    """Run the classifier head on penultimate features for either variant."""
    if isinstance(model, SynapseAugmentedMLP):
        return model.base.classify(features)
    return model.classify(features)
