"""Reward signal subsystem (Phase 3) + reward-as-confidence (path D)."""

from continual_synapse.reward.confidence_reward import (
    compute_reward_signal,
    developmental_alpha,
    normalize_reward_batch,
)
from continual_synapse.reward.consistency import ConsistencyReward
from continual_synapse.reward.external import ExternalReward
from continual_synapse.reward.mixer import RewardMixer
from continual_synapse.reward.surprise import SurpriseReward

__all__ = [
    "ConsistencyReward",
    "ExternalReward",
    "RewardMixer",
    "SurpriseReward",
    "compute_reward_signal",
    "developmental_alpha",
    "normalize_reward_batch",
]
