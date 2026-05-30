"""Memory-augmented native architecture.

Where every prior approach in this project added memory as a bolt-on
(post-hoc retrieval ensembles, distillation against stored soft
targets), this module treats memory as a **first-class participant
in the forward pass**. The model contains parameterised heads for
querying memory (``query_proj``), combining retrieved values with
current features (``context_combiner``), and gating how much of the
combined result to trust (``memory_gate``). All four heads —
including ``value_proj`` which produces the values that get
written — are trained end-to-end via the task loss from batch 0,
even when memory is empty. The stored entries themselves are
gradient-free ``register_buffer`` snapshots written at the end of
each task.

The bet: a model trained to use memory natively can offload
long-term retention to external storage, leaving its parametric
weights free for current-task learning.
"""

from continual_synapse.memory_augmented.memory_augmented_model import (
    ExternalMemory,
    MemoryAugmentedMLP,
)

__all__ = ["ExternalMemory", "MemoryAugmentedMLP"]
