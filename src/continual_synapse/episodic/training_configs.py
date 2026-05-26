"""Dual-substrate episodic training configurations.

One config to start (``cs_episodic_dual_substrate``) — the
hypothesis under test. The compute substrate is a plain
:class:`MLPClassifier` trained with standard backprop; no synapse
layer, no cosine gating, no Hebbian state, no EWC. The memory
substrate is an :class:`ActiveEpisodicMemory` that grows during
training via gradient-free novelty-thresholded allocation. At
inference, an :class:`EpisodicPredictor` blends the model's softmax
with the memory's retrieval distribution.

The config is intentionally narrow: it bundles the memory and
predictor hyperparameters but leaves model construction and the
training loop to the experiment driver (exp 28), matching the
pattern :mod:`reward.training_configs` established for the
reward-axis configs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from continual_synapse.episodic.active_memory import ActiveEpisodicMemory
from continual_synapse.episodic.episodic_predictor import EpisodicPredictor


@dataclass(frozen=True)
class EpisodicConfig:
    """A named dual-substrate configuration.

    Attributes:
        name: Identifier doubled as the dict key in
            :data:`EPISODIC_CONFIGS`.
        novelty_threshold: Cosine-novelty above which a sample
            triggers allocation in the memory. ``0.7`` (i.e. "less
            than 30 % similar to anything stored") is the
            dual-substrate v1 default.
        retrieval_k: Top-k entries consulted at retrieval time.
        max_entries: Optional hard cap on memory size. ``None``
            (default) keeps the memory unbounded — Phase D's first
            pilot measures empirical growth before any size policy
            lands.
        blend_threshold: Retrieval-confidence threshold below which
            the memory contributes nothing to the blended output.
        blend_max: Maximum weight the retrieval distribution receives
            when confidence is at its ceiling.
    """

    name: str
    novelty_threshold: float = 0.7
    retrieval_k: int = 5
    max_entries: int | None = None
    blend_threshold: float = 0.5
    blend_max: float = 0.5

    def build_memory(
        self, feature_dim: int, n_classes: int
    ) -> ActiveEpisodicMemory:
        return ActiveEpisodicMemory(
            feature_dim=feature_dim,
            n_classes=n_classes,
            novelty_threshold=self.novelty_threshold,
            retrieval_k=self.retrieval_k,
            max_entries=self.max_entries,
        )

    def build_predictor(
        self, base_model, memory: ActiveEpisodicMemory
    ) -> EpisodicPredictor:
        return EpisodicPredictor(
            base_model=base_model,
            memory=memory,
            blend_threshold=self.blend_threshold,
            blend_max=self.blend_max,
        )

    def make_after_batch(
        self, predictor: EpisodicPredictor
    ) -> Callable:
        """Return an ``on_after_batch`` closure that lets the
        memory observe each training batch.

        The closure ignores the ``model`` argument the runner passes
        in (it's the bare ``MLPClassifier`` — same model the predictor
        already wraps) and routes to the closed-over predictor's
        ``training_step_observe``. The runner's ``task_index`` becomes
        the ``task_id`` on each newly-allocated entry.
        """

        def on_after_batch(task_index, task, model, x, y) -> None:  # noqa: ARG001
            predictor.training_step_observe(
                x, y, task_id=int(task_index),
            )

        return on_after_batch


EPISODIC_CONFIGS: dict[str, EpisodicConfig] = {
    "cs_episodic_dual_substrate": EpisodicConfig(
        name="cs_episodic_dual_substrate",
    ),
}
"""Registry of dual-substrate configs. v1 ships one entry — the
hypothesis under test. Ablations (different novelty thresholds,
capped memory, different blend curves) can be added by name in
follow-up commits without touching the experiment driver."""
