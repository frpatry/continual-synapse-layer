"""Functional memory + Hinton-style distillation loss.

The memory stores ``(raw_input, soft_target, task_id)`` triples
sampled at the end of each task; the distillation loss penalises
the current model's deviation from the stored soft target on
those inputs.

Critical design choices:

- **Soft targets are computed from the model AT TASK END** —
  before any subsequent training updates them. The stored
  distribution is the teacher signal; the current model is the
  student during distillation.
- **Memory keys are raw inputs**, not features. This sidesteps
  every feature-drift problem the dual-substrate line hit: the
  current model is what computes the loss, so there's no
  alignment problem between stored features and queried
  features.
- **Distillation uses temperature scaling and the T² gradient-
  rebalancing factor** from Hinton et al. 2015. The T² keeps the
  gradient on the distillation branch on the same scale as
  cross-entropy so the loss weighting hyperparameter doesn't
  silently change with T.

This module is intentionally narrow: no training-loop logic, no
config dataclass, no driver. The experiment script
(``experiments/30_functional_regularization_eval.py``) owns the
loop and the configs.
"""

from __future__ import annotations

import random
from typing import Callable

import torch
import torch.nn.functional as F


class FunctionalMemory:
    """Store ``(raw_input, soft_target, task_id)`` triples for LwF.

    Args:
        samples_per_task: Number of inputs to snapshot at the end
            of each task. Picked uniformly from the task's training
            pool. Default ``100`` is a reasonable starting point;
            the eval script exposes this as a CLI sweep knob.
        max_total: Optional hard cap on memory size. ``None``
            (default) grows the memory unboundedly across tasks.
            When set, an entry is evicted uniformly at random
            before any new entry is added past the cap — keeps the
            implementation simple at the cost of class- and task-
            balance guarantees. Smarter eviction (LRU, per-task
            stratified) can land later if growth becomes a real
            problem.
        rng_seed: Optional seed for the internal random module so
            sampling is reproducible across runs. ``None`` (the
            default) uses Python's module-level random state.
    """

    def __init__(
        self,
        samples_per_task: int = 100,
        max_total: int | None = None,
        rng_seed: int | None = None,
    ) -> None:
        if samples_per_task <= 0:
            raise ValueError(
                f"samples_per_task must be positive, got {samples_per_task}"
            )
        if max_total is not None and max_total <= 0:
            raise ValueError(
                f"max_total must be positive or None, got {max_total}"
            )

        self.samples_per_task = int(samples_per_task)
        self.max_total = max_total
        self.inputs: list[torch.Tensor] = []
        self.soft_targets: list[torch.Tensor] = []
        self.task_ids: list[int] = []
        # Per-instance RNG so multiple FunctionalMemory objects
        # don't fight over the module-level seed (the eval driver
        # runs n_seeds models in sequence; each gets its own
        # memory).
        self._rng = random.Random(rng_seed) if rng_seed is not None else random

    def __len__(self) -> int:
        return len(self.inputs)

    @torch.no_grad()
    def record_task_end(
        self,
        model_forward: Callable[[torch.Tensor], torch.Tensor],
        task_inputs: torch.Tensor,
        task_id: int,
        device: torch.device | str = "cpu",
    ) -> int:
        """Snapshot the model's soft predictions on a sample of the
        task's inputs and store them.

        Returns the number of entries added this call. If the memory
        is at ``max_total`` capacity, a random existing entry is
        evicted before each new entry is appended — the count
        returned is "entries written this call", not "size delta".

        Args:
            model_forward: Callable mapping a batch of inputs
                ``(B, ...)`` to logits ``(B, K)``. The function is
                wrapped in ``torch.no_grad`` for the snapshot.
            task_inputs: ``(N, ...)`` pool to sample from. Typically
                the task's training inputs. Must have at least
                ``samples_per_task`` rows for the sample to be
                full-size, but smaller pools are accepted (the
                sample size is clamped).
            task_id: Integer identifier stored alongside each
                entry. Used by diagnostics to track per-task
                contribution to memory.
            device: Where to push the sampled inputs before calling
                ``model_forward``. Stored tensors are moved to CPU
                regardless.
        """
        if task_inputs.ndim < 2:
            raise ValueError(
                f"task_inputs must be at least 2-D (N, ...), got shape "
                f"{tuple(task_inputs.shape)}"
            )
        n = min(self.samples_per_task, task_inputs.shape[0])
        if n == 0:
            return 0
        # Sample without replacement.
        idx = torch.randperm(task_inputs.shape[0])[:n]
        sampled = task_inputs[idx].to(device)
        logits = model_forward(sampled)
        if logits.ndim != 2:
            raise RuntimeError(
                f"model_forward returned shape {tuple(logits.shape)}; "
                f"expected (B, K) logits."
            )
        soft = F.softmax(logits, dim=-1)
        for i in range(n):
            if (
                self.max_total is not None
                and len(self.inputs) >= self.max_total
            ):
                evict = self._rng.randrange(len(self.inputs))
                self.inputs.pop(evict)
                self.soft_targets.pop(evict)
                self.task_ids.pop(evict)
            self.inputs.append(sampled[i].detach().cpu())
            self.soft_targets.append(soft[i].detach().cpu())
            self.task_ids.append(int(task_id))
        return n

    def sample_batch(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return a uniformly-sampled ``(inputs, soft_targets)``
        batch, or ``None`` when the memory is empty.

        ``None`` signals the caller to skip the distillation loss
        entirely (the term is well-defined to be zero when there's
        nothing to remember, and the train-step code uses this
        contract rather than computing a 0 tensor).
        """
        if not self.inputs:
            return None
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        n = min(batch_size, len(self.inputs))
        idx = self._rng.sample(range(len(self.inputs)), n)
        inputs = torch.stack([self.inputs[i] for i in idx]).to(device)
        targets = torch.stack([self.soft_targets[i] for i in idx]).to(device)
        return inputs, targets

    # ---- diagnostics ----

    def per_task_counts(self) -> dict[int, int]:
        """Histogram of stored entries by task_id. Useful for the
        per-task memory contribution diagnostic the eval driver
        prints."""
        counts: dict[int, int] = {}
        for tid in self.task_ids:
            counts[tid] = counts.get(tid, 0) + 1
        return counts


def distillation_loss(
    current_logits: torch.Tensor,
    stored_soft_targets: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Hinton-style knowledge-distillation loss.

    The student is the current model's output on the stored inputs;
    the teacher is the soft-target distribution stored at task end.
    Both are softened by ``temperature`` (T > 1 flattens the
    distribution, exposing more inter-class similarity structure)
    and compared via KL divergence. The ``T²`` factor on the loss
    keeps its gradient magnitude on the same scale as the
    cross-entropy on the task loss, so the loss-weighting
    hyperparameter ``λ`` doesn't silently change with ``T``.

    Args:
        current_logits: ``(B, K)`` raw logits from the current
            model on the stored inputs.
        stored_soft_targets: ``(B, K)`` softmax probabilities
            recorded at task end. Must sum to ~1 along the last
            dimension.
        temperature: ``T`` in Hinton's notation. Default ``2.0``
            matches the original paper. Setting ``T = 1`` reduces
            this to a plain KL between the two predictive
            distributions.

    Returns:
        A scalar ``KL(soft_teacher || soft_student) * T²``,
        averaged across the batch.
    """
    if current_logits.shape != stored_soft_targets.shape:
        raise ValueError(
            f"current_logits {tuple(current_logits.shape)} and "
            f"stored_soft_targets {tuple(stored_soft_targets.shape)} "
            f"must match"
        )
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    # Student: softmax(z_student / T) — use log_softmax for KL stability.
    log_student = F.log_softmax(current_logits / temperature, dim=-1)
    # Teacher: stored soft_target is already softmax(z_teacher); we
    # need softmax(z_teacher / T). Since softmax is invariant to
    # additive constants, softmax(log(p) / T) = softmax(z/T) — so
    # we recover the temperature-scaled teacher distribution by
    # going through log → / T → softmax. The clamp keeps log() from
    # blowing up on exact zeros (which softmax-then-store can
    # produce for very confident inputs).
    log_teacher_over_T = stored_soft_targets.clamp(min=1e-8).log() / temperature
    soft_teacher = F.softmax(log_teacher_over_T, dim=-1)
    return F.kl_div(
        log_student, soft_teacher, reduction="batchmean",
    ) * (temperature ** 2)
