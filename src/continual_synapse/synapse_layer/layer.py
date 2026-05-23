"""SynapseLayer v1 — dense Hebbian state container.

This is the first iteration of the synapse layer described in
DESIGN.md section 3.2. The Phase-2 spec is intentionally minimal:

- Dense ``n × n`` strength matrix. Sparse top-k partner selection
  is deferred to Phase 3.
- A single state field (``strength``). ``confidence``, ``evidence``,
  ``age``, and ``access_count`` are deferred to Phase 3.
- Hebbian update with a fixed reward of 1.0. The reward signal
  system is deferred to Phase 3.
- Updates are triggered explicitly by calling :meth:`consolidate`
  from the training loop. Pressure-based automatic triggering is
  deferred to Phase 4.

The module is deliberately split from the read-out side (the
:mod:`continual_synapse.synapse_layer.modulation` module): this
class only holds state and updates it. Producing a correction
vector from the state belongs to the modulator. This keeps the
state representation independent of the modulation strategy and
makes both pieces easier to ablate.

Update rule (DESIGN.md eq. (3.2), with ``β=0`` for v1)::

    Δs_ij = (η / B) · R · Σ_b a_{b,i} · a_{b,j}

i.e. the batch-mean outer product of activations, scaled by
learning rate and reward. The batch-mean form keeps the update
magnitude independent of batch size.
"""

from __future__ import annotations

import torch
from torch import nn


class SynapseLayer(nn.Module):
    """Dense Hebbian state for an ``n``-neuron activation vector.

    The module holds two buffers:

    - ``strengths``: ``(n, n)`` float32. ``strengths[j, i]`` is the
      learned weight of the connection from neuron ``j`` to neuron
      ``i``. Initialised to zero so the layer has no effect at start.
    - ``global_step``: long scalar, incremented once per call to
      :meth:`consolidate`. Useful for logging and for later phases'
      age-based logic.

    The module is registered as an ``nn.Module`` so its buffers
    follow standard PyTorch device / dtype semantics (``.to(...)``,
    ``state_dict``, etc.). It deliberately does **not** override
    ``forward``: SynapseLayer is a state container, not a layer that
    transforms activations. Read-out is the modulator's job.

    Args:
        n_neurons: Width of the activation vector this layer
            observes. Equal to the hooked module's output dim.
        learning_rate: ``η`` in the update rule. The Phase-2 spec
            recommends starting small (``1e-3``) to keep the layer
            from destabilising the base model.
    """

    def __init__(self, n_neurons: int, learning_rate: float = 1e-3) -> None:
        super().__init__()
        if n_neurons <= 0:
            raise ValueError(f"n_neurons must be positive, got {n_neurons}")
        if learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {learning_rate}"
            )
        self.n_neurons = n_neurons
        self.learning_rate = float(learning_rate)
        self.register_buffer(
            "strengths", torch.zeros(n_neurons, n_neurons, dtype=torch.float32)
        )
        self.register_buffer(
            "global_step", torch.zeros((), dtype=torch.long)
        )

    @torch.no_grad()
    def consolidate(
        self, activations: torch.Tensor, reward: float = 1.0
    ) -> None:
        """Apply a single Hebbian update from a batch of activations.

        Args:
            activations: ``(B, n_neurons)`` tensor of activations
                observed for this batch. The caller is responsible
                for detaching from the autograd graph (the
                :class:`ActivationCapture` helper does this by
                default).
            reward: Scalar reward modulating the update magnitude.
                Phase 2 uses a fixed ``1.0``; later phases compute a
                real reward signal.

        The update is computed in float32 regardless of the input
        dtype to keep the running strength buffer well-conditioned
        when the base model uses lower precision.
        """
        if activations.ndim != 2:
            raise ValueError(
                f"Expected 2-D activations (B, n), got shape "
                f"{tuple(activations.shape)}"
            )
        if activations.shape[1] != self.n_neurons:
            raise ValueError(
                f"Activation dim {activations.shape[1]} does not match "
                f"n_neurons={self.n_neurons}"
            )
        if activations.shape[0] == 0:
            return

        a = activations.detach().to(self.strengths.dtype)
        outer = a.transpose(-1, -2) @ a / a.shape[0]
        self.strengths.add_(outer, alpha=self.learning_rate * float(reward))
        self.global_step.add_(1)

    def reset(self) -> None:
        """Zero strengths and global_step. Used in tests and ablations."""
        with torch.no_grad():
            self.strengths.zero_()
            self.global_step.zero_()

    def extra_repr(self) -> str:
        return (
            f"n_neurons={self.n_neurons}, "
            f"learning_rate={self.learning_rate}, "
            f"global_step={int(self.global_step.item())}"
        )
