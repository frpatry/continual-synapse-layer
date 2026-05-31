"""Brain-aligned learning substrate — Phase 1.

This subpackage is *parallel to* the legacy ``continual_synapse``
and ``agi`` work, not an extension of it. The earlier code is kept
as reference; the substrate is a clean restart following the
theory document at ``THEORY.md`` (Phase 0).

Phase 1 scope (N entities only):
- :class:`N`                  — atomic neuron entity (``neuron.py``)
- :class:`ConnectivityMatrix` — sparse implicit connectivity (``connectivity.py``)
- activation dynamics + metastable background (``dynamics.py``)
- covariance Hebbian + age-modulated decay (``plasticity.py``)
- :class:`Substrate`          — orchestrator (``substrate.py``)

No P / S / C entities, no emergence, no spaces / zones. Those
arrive in Phase 2+ once P1 (pattern formation through repeated
exposure) is validated empirically.

Pure NumPy. No PyTorch, no transformers. CPU-only.
"""

from .connectivity import ConnectivityMatrix
from .dynamics import (
    DEFAULT_SPARSITY_TARGET,
    GlobalBackground,
    k_winners_take_all,
    propagate_activation,
    soft_threshold,
)
from .neuron import N
from .p_connectivity import PConnectivity
from .p_dynamics import compute_p_input, propagate_p_activations
from .p_entity import PEntity
from .p_plasticity import apply_pp_plasticity
from .p_to_n_feedback import compute_p_to_n_feedback
from .pass_tracker import PassTracker
from .plasticity import (
    age_modulated_decay,
    apply_plasticity,
    covariance_hebbian_update,
    rho_age,
)
from .substrate import Substrate

__all__ = [
    "DEFAULT_SPARSITY_TARGET",
    "ConnectivityMatrix",
    "GlobalBackground",
    "N",
    "PConnectivity",
    "PEntity",
    "PassTracker",
    "Substrate",
    "age_modulated_decay",
    "apply_plasticity",
    "apply_pp_plasticity",
    "compute_p_input",
    "compute_p_to_n_feedback",
    "covariance_hebbian_update",
    "k_winners_take_all",
    "propagate_activation",
    "propagate_p_activations",
    "rho_age",
    "soft_threshold",
]
