"""Reward signal subsystem (Phase 3)."""

from continual_synapse.reward.consistency import ConsistencyReward
from continual_synapse.reward.external import ExternalReward

__all__ = ["ConsistencyReward", "ExternalReward"]
