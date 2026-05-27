"""Memory-augmented MLP: native external memory in the forward pass.

Two classes:

- :class:`ExternalMemory` is a key-value store backed by
  ``register_buffer`` tensors (not parameters). It exposes a
  scaled-dot-product :meth:`read` and a gradient-free :meth:`write`.
  The READ is differentiable end-to-end through the attention
  weights and the retrieved values, so the surrounding model can
  learn how to query. The WRITE is detached; stored entries are
  snapshots, not parameters, and don't accumulate gradients.
- :class:`MemoryAugmentedMLP` wraps an encoder, attention heads
  (``query_proj``, ``value_proj``, ``context_combiner``,
  ``memory_gate``), and a classifier. The forward pass *always*
  routes through the memory-access mechanism, even when memory is
  empty — empty reads return zeros and the gate's empty-memory
  output equals the bare encoder output, but the query_proj head
  still gets exercised so the gradient path through it warms up
  from batch 0.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExternalMemory(nn.Module):
    """Append-only key-value store with attention-based read.

    Args:
        key_dim: Dimensionality of stored keys (and of the queries
            that read them).
        value_dim: Dimensionality of stored values.

    The two storage tensors and the ``task_ids`` index are
    registered as buffers, so they:

    - move with the module via ``model.to(device)``,
    - serialise into the module's ``state_dict()`` (so
      checkpoints round-trip the memory contents),
    - **do not** appear in ``model.parameters()`` — so they don't
      receive gradient updates from the task loss.

    The buffers start as empty tensors of the appropriate dtype +
    shape; each :meth:`write` re-builds them via ``torch.cat`` and
    reassigns the buffer attribute (``nn.Module.__setattr__``
    handles the dispatch to ``_buffers`` correctly).
    """

    def __init__(self, key_dim: int, value_dim: int) -> None:
        super().__init__()
        if key_dim <= 0:
            raise ValueError(f"key_dim must be positive, got {key_dim}")
        if value_dim <= 0:
            raise ValueError(f"value_dim must be positive, got {value_dim}")
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.register_buffer("keys", torch.empty(0, key_dim))
        self.register_buffer("values", torch.empty(0, value_dim))
        self.register_buffer(
            "task_ids", torch.empty(0, dtype=torch.long),
        )

    def __len__(self) -> int:
        return int(self.keys.shape[0])

    # ---- write (gradient-free) ----

    @torch.no_grad()
    def write(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        task_id: int,
    ) -> None:
        """Append ``(keys, values)`` pairs tagged with ``task_id``.

        Inputs are detached before storage. The buffer attribute is
        reassigned (not mutated in place) so the appended shape is
        respected — registered buffers support reassignment via
        ``nn.Module.__setattr__``.
        """
        if keys.ndim != 2 or keys.shape[1] != self.key_dim:
            raise ValueError(
                f"keys must be (B, {self.key_dim}), got "
                f"{tuple(keys.shape)}"
            )
        if values.ndim != 2 or values.shape[1] != self.value_dim:
            raise ValueError(
                f"values must be (B, {self.value_dim}), got "
                f"{tuple(values.shape)}"
            )
        if keys.shape[0] != values.shape[0]:
            raise ValueError(
                f"keys and values batch sizes disagree: "
                f"{keys.shape[0]} vs {values.shape[0]}"
            )
        new_task_ids = torch.full(
            (keys.shape[0],), int(task_id),
            dtype=torch.long, device=keys.device,
        )
        self.keys = torch.cat([self.keys, keys.detach()], dim=0)
        self.values = torch.cat([self.values, values.detach()], dim=0)
        self.task_ids = torch.cat([self.task_ids, new_task_ids], dim=0)

    # ---- read (differentiable) ----

    def read(
        self,
        query: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(retrieved_values, attention_weights)``.

        Scaled-dot-product attention:

            scores  = (Q @ K.T) / (temperature * sqrt(d_k))
            weights = softmax(scores)
            retrieved = weights @ V

        Empty memory returns a ``(B, value_dim)`` tensor of zeros and
        a ``(B, 0)`` attention tensor — callers don't have to
        special-case the no-entries regime.

        The read is fully differentiable: gradients flow from
        ``retrieved`` back through ``weights`` (and through the
        score computation) into the caller's ``query`` tensor —
        which is exactly how the wrapping model trains its
        ``query_proj`` head.
        """
        if query.ndim != 2 or query.shape[1] != self.key_dim:
            raise ValueError(
                f"query must be (B, {self.key_dim}), got "
                f"{tuple(query.shape)}"
            )
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        B = query.shape[0]
        if self.keys.shape[0] == 0:
            return (
                torch.zeros(B, self.value_dim, device=query.device),
                torch.zeros(B, 0, device=query.device),
            )

        scale = temperature * (self.key_dim ** 0.5)
        scores = (query @ self.keys.T) / scale
        weights = F.softmax(scores, dim=-1)
        retrieved = weights @ self.values
        return retrieved, weights


class MemoryAugmentedMLP(nn.Module):
    """MLP with native external-memory access in the forward pass.

    Architecture::

        x -> encoder -> h
        h -> query_proj -> query
        retrieved, _ = memory.read(query)
        combined = context_combiner([h, retrieved])
        gate = sigmoid(memory_gate(h))                     # (B, 1)
        effective_h = (1 - gate) * h + gate * combined
        logits = classifier(effective_h)

    The ``write_batch_to_memory`` helper computes ``query_proj(h)``
    and ``value_proj(h)`` under ``no_grad`` and appends those into
    the external memory; called once at the end of each task by
    the experiment driver.

    Empty-memory contract: when the memory has no entries,
    ``memory.read`` returns zero retrieved values and the
    ``if len(memory) > 0`` guard in the forward bypasses the
    gate / combiner path, so the output equals
    ``classifier(encoder(x))`` exactly. ``query_proj`` still runs
    in that regime (we compute it before the read), so its weight
    receives a (zero) gradient signal from the read math — but the
    real training signal for the access heads kicks in once memory
    is non-empty (the first batch of task 1 onward).
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        n_classes: int = 10,
        key_dim: int = 64,
        value_dim: int = 64,
        n_encoder_layers: int = 2,
        gate_init: float = 0.0,
        maturity_target: int = 750,
    ) -> None:
        super().__init__()
        if n_encoder_layers < 1:
            raise ValueError(
                f"n_encoder_layers must be >= 1, got {n_encoder_layers}"
            )
        if maturity_target <= 0:
            raise ValueError(
                f"maturity_target must be positive, got {maturity_target}"
            )
        # Encoder: n_encoder_layers Linear+ReLU stack ending in a
        # plain Linear so the last layer's output isn't squashed by
        # a ReLU (matches the conventional "features = penultimate"
        # idiom in this repo's MLPClassifier).
        layers: list[nn.Module] = []
        prev = input_dim
        for _ in range(n_encoder_layers - 1):
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            prev = hidden_dim
        layers.append(nn.Linear(prev, hidden_dim))
        self.encoder = nn.Sequential(*layers)

        # Learnable memory-access heads.
        self.query_proj = nn.Linear(hidden_dim, key_dim)
        self.value_proj = nn.Linear(hidden_dim, value_dim)
        self.context_combiner = nn.Sequential(
            nn.Linear(hidden_dim + value_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.memory_gate = nn.Linear(hidden_dim, 1)
        # Initialise the gate's bias so the initial sigmoid output
        # equals sigmoid(gate_init). Default 0.0 → sigmoid(0) = 0.5,
        # i.e. the model starts undecided about how much to weight
        # retrieved context.
        with torch.no_grad():
            self.memory_gate.bias.fill_(float(gate_init))

        # Classifier.
        self.classifier = nn.Linear(hidden_dim, n_classes)

        # External memory (buffers, not parameters).
        self.memory = ExternalMemory(key_dim, value_dim)

        # Developmental maturity: as memory fills, we impose a
        # rising floor on the effective gate. The model can learn
        # to use memory MORE than the floor (the gate is still a
        # trainable parameter), but it cannot learn to use memory
        # LESS than the floor. Without this, the empty-memory
        # regime at the start of training tells the model "memory
        # contributes nothing", and the gate trains toward 0 — a
        # local optimum the model never escapes even when memory
        # later contains useful content. The maturity floor is the
        # structural intervention that breaks that attractor.
        self.maturity_target = int(maturity_target)

        self._hidden_dim = int(hidden_dim)
        self._n_classes = int(n_classes)

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def n_classes(self) -> int:
        return self._n_classes

    # ---- forward pass ----

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Bare-encoder features. Kept for compatibility with the
        retrieval/probing code that expects a ``.features(x)``
        method on every model in this repo."""
        return self.encoder(x)

    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """Bare-classifier output applied to precomputed features.
        Skips the memory-access path — useful for evaluating the
        encoder + classifier alone."""
        return self.classifier(features)

    def _memory_maturity_floor(self) -> float:
        """Return the developmental floor on the effective gate.

        Sigmoid-shaped function of ``len(memory) / maturity_target``,
        centred at the target with sharpness 5. Concretely:

        - ``len(memory) = 0``        → floor ≈ 0.007  (effectively 0)
        - ``len(memory) = target/2`` → floor ≈ 0.076
        - ``len(memory) = target``   → floor = 0.500
        - ``len(memory) = 2·target`` → floor ≈ 0.993  (effectively 1)

        The model's learned gate can still climb above the floor
        (memory is genuinely useful at that point) but cannot fall
        below it (the structural intervention against the
        learned-to-ignore-empty-memory attractor).
        """
        ratio = len(self.memory) / max(1, self.maturity_target)
        return float(torch.sigmoid(torch.tensor(5.0 * (ratio - 1.0))).item())

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Forward with the native memory-access path.

        Args:
            x: Input batch ``(B, input_dim)``.
            return_diagnostics: When True, additionally returns a
                dict with:

                - ``learned_gate_mean``: mean over the batch of the
                  *raw* sigmoid gate (what the model wants).
                - ``maturity_floor``: scalar floor imposed by
                  :meth:`_memory_maturity_floor` at the current
                  memory size.
                - ``effective_gate_mean``: mean over the batch of
                  the gate actually applied
                  (``max(learned, floor)``).
                - ``attention_entropy``: mean over the batch of the
                  attention distribution's entropy in nats.

                Reading ``learned`` vs ``effective`` per task tells
                you whether the model is genuinely opening the gate
                beyond the floor (good — finding memory useful) or
                whether the floor is doing all the work (the model
                is "resigned" to memory rather than enjoying it).

        Returns:
            ``logits`` of shape ``(B, n_classes)``, optionally with
            the diagnostics dict.
        """
        h = self.encoder(x)
        query = self.query_proj(h)
        retrieved, weights = self.memory.read(query)

        diagnostics: dict[str, float] = {
            "learned_gate_mean": 0.0,
            "maturity_floor": 0.0,
            "effective_gate_mean": 0.0,
            "attention_entropy": 0.0,
        }
        if len(self.memory) > 0:
            combined_input = torch.cat([h, retrieved], dim=-1)
            combined = self.context_combiner(combined_input)
            learned_gate = torch.sigmoid(self.memory_gate(h))  # (B, 1)
            # Maturity floor: structurally-imposed lower bound on
            # the effective gate. The model can climb above it
            # (gradient still flows through learned_gate when
            # learned > floor); it can't drop below it.
            floor_value = self._memory_maturity_floor()
            floor = torch.full_like(learned_gate, floor_value)
            effective_gate = torch.maximum(learned_gate, floor)
            effective_h = (1.0 - effective_gate) * h + effective_gate * combined
            if return_diagnostics:
                diagnostics["learned_gate_mean"] = float(learned_gate.mean().item())
                diagnostics["maturity_floor"] = floor_value
                diagnostics["effective_gate_mean"] = float(effective_gate.mean().item())
                # Entropy of the (B, N_mem) attention rows; mean over batch.
                entropy = -(weights * (weights.clamp_min(1e-12)).log()).sum(dim=-1)
                diagnostics["attention_entropy"] = float(entropy.mean().item())
        else:
            effective_h = h

        logits = self.classifier(effective_h)
        if return_diagnostics:
            return logits, diagnostics
        return logits

    # ---- writing ----

    @torch.no_grad()
    def write_batch_to_memory(
        self, x: torch.Tensor, task_id: int,
    ) -> None:
        """Project the batch through the encoder + query/value heads
        and append the resulting (key, value) pairs to memory.

        All projections run under ``torch.no_grad`` so the write
        accumulates no gradients on any parameter. The keys go
        through the same ``query_proj`` head queries use at read
        time — this is what makes the stored representation
        compatible with what the model has learned to ask for.
        """
        h = self.encoder(x)
        keys = self.query_proj(h)
        values = self.value_proj(h)
        self.memory.write(keys, values, task_id=task_id)
