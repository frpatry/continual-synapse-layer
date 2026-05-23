"""Continual-learning baseline models (naive fine-tune, EWC, replay, ...)."""

from continual_synapse.baselines.ewc import EWC
from continual_synapse.baselines.naive_finetune import MLPClassifier

__all__ = ["EWC", "MLPClassifier"]
