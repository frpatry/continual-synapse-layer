"""Sparse top-k partner selection for SynapseLayer.

DESIGN.md section 3.2: each source neuron keeps only its ``k``
strongest target partners. When a new co-activation pushes a
synapse's |strength| above the weakest currently-retained partner's,
the weakest is evicted. This brings memory from ``O(n²)`` to
``O(n · k)`` at the cost of dropping rarely-firing connections —
the trade is favourable at transformer scale (n ~ 768) and
roughly neutral at MLP scale (n = 256, k = 64).

The implementation deliberately keeps dense ``(n, n)`` buffers in
:class:`SynapseLayer` and zeros out the off-top-k entries after
each consolidation. This avoids the bookkeeping cost of a true
sparse representation while preserving the user-visible behaviour
of top-k selection. A future iteration may switch to a real sparse
layout if the dense zeros become a memory bottleneck.
"""

from __future__ import annotations

from typing import Sequence

import torch


def compute_topk_mask(strengths: torch.Tensor, k: int) -> torch.Tensor:
    """Boolean mask of the top-k entries by ``|strength|`` per row.

    Args:
        strengths: ``(n, n)`` float tensor (typically the synapse
            strength matrix). The mask is computed row-wise — row
            ``i`` corresponds to "source neuron ``i``'s outgoing
            partners".
        k: Number of partners to keep per source. Values ``>= n``
            yield an all-True mask (no eviction). Values ``<= 0``
            yield an all-False mask (everything evicted).

    Returns:
        A ``(n, n)`` boolean tensor on the same device as
        ``strengths``. Ties are broken by ``torch.topk``'s
        deterministic index ordering.
    """
    if strengths.ndim != 2:
        raise ValueError(
            f"strengths must be 2-D, got shape {tuple(strengths.shape)}"
        )
    n_rows, n_cols = strengths.shape
    if k <= 0:
        return torch.zeros_like(strengths, dtype=torch.bool)
    if k >= n_cols:
        return torch.ones_like(strengths, dtype=torch.bool)

    _, topk_indices = strengths.abs().topk(k, dim=1)
    mask = torch.zeros_like(strengths, dtype=torch.bool)
    mask.scatter_(1, topk_indices, True)
    return mask


def apply_topk_mask_inplace(
    buffers: Sequence[torch.Tensor], mask: torch.Tensor
) -> None:
    """Zero entries outside ``mask`` for every supplied buffer, in place.

    Used to keep the state buffers of :class:`SynapseLayer`
    consistent after eviction. Floating buffers are multiplied by
    the mask (cast to their dtype); integer buffers receive the
    same treatment. The mask itself is never modified.
    """
    for buf in buffers:
        if buf.shape != mask.shape:
            raise ValueError(
                f"Buffer shape {tuple(buf.shape)} does not match mask "
                f"shape {tuple(mask.shape)}"
            )
        buf.mul_(mask.to(buf.dtype))
