"""Functional regularization for continual learning (LwF-style).

The architectural bet (in counterpoint to the dual-substrate /
retrieval-ensemble line, which failed to preserve Task-0 across
every variant we tried): **constrain the model's function on
selected past inputs, not its weights.** At the end of each task
we snapshot the model's soft predictions on a sample of that
task's inputs; during subsequent task training, a knowledge-
distillation loss against those frozen soft targets keeps the
model's behaviour on those inputs from drifting.

The mechanism is structurally a training-time intervention (like
EWC and cosine gating, both of which do hold Task-0) but its
restoring force acts on **function** rather than **weights**,
which is the part EWC gets wrong.
"""

from continual_synapse.functional.functional_memory import (
    FunctionalMemory,
    distillation_loss,
)

__all__ = ["FunctionalMemory", "distillation_loss"]
