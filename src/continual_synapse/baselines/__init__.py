"""Continual-learning baseline models (naive fine-tune, EWC, replay, ...)."""

from continual_synapse.baselines.ewc import EWC
from continual_synapse.baselines.multi_head import MultiHeadMLPClassifier
from continual_synapse.baselines.naive_finetune import MLPClassifier
from continual_synapse.baselines.synapse_finetune import SynapseAugmentedMLP

__all__ = [
    "EWC",
    "MLPClassifier",
    "MultiHeadMLPClassifier",
    "SynapseAugmentedMLP",
]
