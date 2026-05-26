"""Inference-time wrappers for trained continual-learning models.

These wrappers take an already-trained model (and optionally its
cold-storage state) and produce predictions, possibly enhanced by
retrieval. They never modify the underlying model — strictly
eval-time use.
"""

from continual_synapse.inference.retrieval_ensemble import RetrievalEnsemble

__all__ = ["RetrievalEnsemble"]
