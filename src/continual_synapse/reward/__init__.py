"""Reward signal subsystem (Phase 3)."""

from continual_synapse.reward.consistency import ConsistencyReward
from continual_synapse.reward.external import ExternalReward
from continual_synapse.reward.surprise import SurpriseReward

__all__ = ["ConsistencyReward", "ExternalReward", "SurpriseReward"]
