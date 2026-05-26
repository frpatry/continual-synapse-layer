"""Dual-substrate episodic memory.

The compute substrate (the model's weights) is free to learn —
standard backprop, no protection mechanisms. The memory substrate is
:class:`ActiveEpisodicMemory`: an actively-growing store that
accumulates (embedding, label) tuples whenever a sufficiently novel
input arrives during training, with no gradient signal involved.

At inference, :class:`EpisodicPredictor` blends the model's softmax
with a retrieval-based label distribution computed by the memory,
with the blend weight scaled by retrieval confidence — the memory
only contributes when something genuinely similar exists in it.

The bet: the plasticity-stability trade-off is an artefact of
asking the same substrate (network weights) to compute the right
answer AND remember previous distributions. Separate them and the
trade-off may dissolve.
"""

from continual_synapse.episodic.active_memory import ActiveEpisodicMemory
from continual_synapse.episodic.episodic_predictor import EpisodicPredictor

__all__ = ["ActiveEpisodicMemory", "EpisodicPredictor"]
