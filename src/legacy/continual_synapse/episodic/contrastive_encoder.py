"""Contrastive encoder for the frozen-keying-encoder pivot (Option B).

The dual-substrate pilot showed that a trainable encoder drifts so
heavily during continual training that its feature space stops
preserving inter-task geometry — memory ends up dominated by
task-0-flavoured prototypes even with re-encoding. The fix here is
to commit to a **frozen** encoder whose feature space was built
once, up front, to be invariant to the permutation transform that
defines the continual benchmark.

Pretraining objective:

- Sample a mini-batch ``x`` of MNIST images, flatten each to 784 dims.
- Sample two random pixel permutations ``p1, p2``.
- Apply: ``x1 = x[:, p1]``, ``x2 = x[:, p2]``.
- Encode + project both views: ``z1 = projection(encoder(x1))``,
  ``z2 = projection(encoder(x2))``.
- Treat ``(z1[i], z2[i])`` as positive pairs (same image, two
  permutations); every other pair in the SimCLR-stacked
  ``[z1; z2]`` is a negative.

This forces the encoder to map any two permutations of the same
image to nearby points, which by construction means the encoder is
permutation-invariant — exactly the property the dual-substrate
memory needs as its keying function. The projection head exists
only during pretraining; at deploy time we throw it away and use
the encoder's penultimate output directly (the SimCLR convention).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveEncoder(nn.Module):
    """Encoder + projection-head pair for SimCLR-style pretraining.

    Args:
        input_dim: Flattened input dimensionality (784 for MNIST).
        hidden_dim: Width of the two hidden layers in the encoder.
        feature_dim: Encoder output dim — this is the "useful" output
            that downstream code (memory, retrieval) consumes after
            the projection head is discarded.
        projection_dim: Width of the projection head's two layers.
            Only used during contrastive training. Following SimCLR,
            this is smaller than ``feature_dim`` — the head adds a
            non-linear bottleneck whose only job is to make the
            contrastive objective tractable.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        feature_dim: int = 128,
        projection_dim: int = 64,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if projection_dim <= 0:
            raise ValueError(
                f"projection_dim must be positive, got {projection_dim}"
            )

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim),
        )
        # Stash so save/load can round-trip without inspecting layers.
        self._config = {
            "input_dim": int(input_dim),
            "hidden_dim": int(hidden_dim),
            "feature_dim": int(feature_dim),
            "projection_dim": int(projection_dim),
        }
        self._feature_dim = int(feature_dim)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def config(self) -> dict:
        """Ctor arguments needed to rebuild this module from disk."""
        return dict(self._config)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return encoder features only — no projection head.

        The deploy-time inference path. The
        :class:`PretrainedContrastiveEncoder` wrapper that
        :mod:`continual_synapse.episodic.frozen_encoder` exposes
        ends up calling exactly this submodule under the hood.
        """
        return self.encoder(x)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(features, projected)``.

        The pretraining loop uses both:
        - ``features`` feeds the linear probe at the end of training
          (sanity check on representation quality).
        - ``projected`` feeds :func:`info_nce_loss` for the
          contrastive objective itself.
        """
        features = self.encoder(x)
        projected = self.projection(features)
        return features, projected


def info_nce_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """SimCLR-style InfoNCE loss.

    ``z1[i]`` and ``z2[i]`` are two augmented views of the same
    input — a positive pair. Every other pair in the stacked
    ``[z1; z2]`` (shape ``(2B, D)``) is a negative.

    Implementation notes:

    - Both views are L2-normalised so the dot products in the
      similarity matrix are cosine similarities; the temperature
      then sets the sharpness of the softmax.
    - Self-similarity (the diagonal of the ``(2B, 2B)`` matrix) is
      masked to ``-inf`` so it can't be selected as the positive.
    - The target for row ``i`` in ``[0, B)`` is index ``i + B``
      (its paired view); for ``i in [B, 2B)`` it's ``i − B``.
      ``(i + B) % (2B)`` encodes both halves in one expression.

    Args:
        z1: ``(B, D)`` first-view projections.
        z2: ``(B, D)`` second-view projections. Must match z1's
            batch and feature dims.
        temperature: ``τ``. Lower → sharper softmax, more
            discriminative gradients. SimCLR's default is ``0.1``.
    """
    if z1.shape != z2.shape:
        raise ValueError(
            f"z1 {tuple(z1.shape)} and z2 {tuple(z2.shape)} must match"
        )
    if z1.ndim != 2:
        raise ValueError(f"z1 must be (B, D), got {tuple(z1.shape)}")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    B = z1.shape[0]
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    z = torch.cat([z1, z2], dim=0)  # (2B, D)

    sim = z @ z.T / temperature  # (2B, 2B)
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, float("-inf"))

    targets = torch.arange(2 * B, device=z.device)
    targets = (targets + B) % (2 * B)
    return F.cross_entropy(sim, targets)


def random_permutation(
    dim: int,
    n: int,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate ``n`` independent random permutations of length ``dim``.

    Returns a ``(n, dim)`` long tensor whose row ``k`` is a
    permutation of ``[0, dim)``. Independent across rows.

    The pretraining loop uses ``n=2`` per batch (one perm for each
    augmented view).
    """
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    return torch.stack(
        [
            torch.randperm(dim, device=device, generator=generator)
            for _ in range(n)
        ]
    )


def apply_permutation(
    x: torch.Tensor, perm: torch.Tensor
) -> torch.Tensor:
    """Permute the last dimension of ``x`` by ``perm``.

    ``x`` is typically ``(B, input_dim)`` flat MNIST images and
    ``perm`` a ``(input_dim,)`` index tensor; the result is
    ``x[:, perm]``, i.e. every row gets the same permutation
    applied. Same-perm-per-batch is the SimCLR convention here:
    different batches see different perms, and within a batch the
    contrastive loss compares the same image under two different
    perms (one for each augmented view).
    """
    if perm.ndim != 1:
        raise ValueError(f"perm must be 1-D, got {tuple(perm.shape)}")
    if x.shape[-1] != perm.shape[0]:
        raise ValueError(
            f"x last dim {x.shape[-1]} disagrees with perm length "
            f"{perm.shape[0]}"
        )
    return x.index_select(dim=-1, index=perm)
