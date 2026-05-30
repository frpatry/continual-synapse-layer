"""Frozen keying-encoder wrappers for the dual-substrate memory.

The dual-substrate hypothesis needs a stable feature space to key
the episodic memory against. Trainable encoders drift, which
breaks the memory's "same task ⇒ same neighbourhood" invariant
(see the failure mode documented in the T=15 n=2 pilot
``results/logs/episodic/1779818599_28_T15_dual_substrate.json``).

The classes in this module wrap an encoder, freeze its weights, and
expose a clean ``forward(x) -> features`` API for use as
:class:`EpisodicPredictor`'s ``keying_encoder``. Phase 1 of Option
B ships :class:`PretrainedContrastiveEncoder`, which loads a
SimCLR-pretrained encoder from disk; other frozen-encoder
variants (random-projection ablation, snapshot-then-freeze) can
land here later without touching the predictor.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from continual_synapse.episodic.contrastive_encoder import ContrastiveEncoder


class PretrainedContrastiveEncoder(nn.Module):
    """Loads a contrastive-pretrained encoder and freezes it.

    Reads the checkpoint produced by
    ``experiments/29_pretrain_contrastive_encoder.py`` (state_dict +
    config), reconstructs the :class:`ContrastiveEncoder`, then
    discards the projection head and keeps only the encoder portion
    — the SimCLR convention.

    All parameters have ``requires_grad=False`` after construction
    and the module is set to ``eval()``, so downstream callers don't
    accidentally pull this encoder into the model's gradient graph
    or trigger dropout / batch-norm updates. ``forward`` runs under
    ``torch.no_grad`` for extra belt-and-braces protection.

    Args:
        checkpoint_path: Filesystem path to the encoder ``.pt``
            written by exp 29. The file must contain ``state_dict``
            and ``config`` keys.
    """

    def __init__(self, checkpoint_path: str | Path) -> None:
        super().__init__()
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Pretrained encoder checkpoint not found at {ckpt_path}. "
                f"Run experiments/29_pretrain_contrastive_encoder.py first."
            )
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "config" not in ckpt or "state_dict" not in ckpt:
            raise KeyError(
                f"Checkpoint at {ckpt_path} missing required keys "
                f"'config' and/or 'state_dict'. Has keys: "
                f"{sorted(ckpt.keys())}"
            )
        config = ckpt["config"]
        encoder_full = ContrastiveEncoder(**config)
        encoder_full.load_state_dict(ckpt["state_dict"])

        # Strip the projection head — only the encoder portion is
        # used at deploy time. (Keeping projection alive would also
        # work but bloats the parameter count and is misleading
        # about the deploy contract.)
        self._encoder = encoder_full.encoder
        self._feature_dim = int(config["feature_dim"])
        self._source_path = str(ckpt_path)

        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def source_path(self) -> str:
        """Filesystem path the encoder was loaded from. Useful for
        the experiment startup banner so the operator can confirm
        which encoder is in use."""
        return self._source_path

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._encoder(x)

    def train(self, mode: bool = True):  # noqa: D401 — override to no-op
        """Override ``train()`` to keep the encoder permanently in
        eval mode. The base ``nn.Module`` API otherwise lets a
        caller's ``model.train()`` propagate down and accidentally
        re-enable dropout / batch-norm here; this guard makes the
        frozen contract structural rather than discipline-based.
        """
        return super().train(False)
