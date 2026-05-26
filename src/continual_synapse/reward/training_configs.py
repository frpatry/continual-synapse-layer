"""Reward-as-confidence training configurations.

Three new configs that compose the per-sample reward signal with
different gating + alpha-schedule choices, plus the unchanged
baseline for uniform handling in the eval driver:

- ``cs_gated_cosine_developmental`` (baseline, unchanged):
  cosine gating ON, no reward signal — the scout_a095_validated
  config that anchors the comparison.
- ``cs_reward_developmental``: cosine gating OFF, reward ON with
  the developmental ``α`` schedule. Isolates ``R``'s contribution.
- ``cosine_reward_developmental``: cosine gating ON, reward ON
  developmental. The composition we expect to be best.
- ``reward_only_static``: cosine gating OFF, reward ON with a
  constant ``α = 0.5``. Ablates the developmental component.

Each :class:`RewardConfig` knows how to build its training
callbacks (``on_pre_optimizer_step``, ``on_after_batch``,
``on_task_change``). The model-construction side is left to the
experiment script because it depends on architecture flags
(``gate_modulation_enabled``, ``familiarity_mode``, etc.) that the
training loop doesn't otherwise need to know about; the config
just exposes the small set of choices that differ between the four
named entries.

Importing pattern:

    from continual_synapse.reward.training_configs import REWARD_CONFIGS
    cfg = REWARD_CONFIGS["cs_reward_developmental"]
    callbacks = cfg.make_callbacks()
    runner = ContinualRunner(..., **callbacks)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch

from continual_synapse.reward.confidence_reward import (
    compute_reward_signal,
    developmental_alpha,
)

AlphaMode = Literal["constant", "developmental", "static"]


@dataclass(frozen=True)
class RewardConfig:
    """A named training configuration on the reward-as-confidence axis.

    Attributes:
        name: Short config identifier. Doubles as the dict key in
            :data:`REWARD_CONFIGS`.
        gradient_gating_enabled: Whether the
            ``on_pre_optimizer_step`` callback should call
            :meth:`SynapseAugmentedMLP.apply_gradient_gating`. Maps
            to the same-named constructor arg.
        alpha_mode: How the reward's ``α`` is set.

            - ``"constant"``: no reward signal is computed. The
              Hebbian update uses the historical scalar-R path
              (the existing ``RewardMixer`` output). Used for the
              baseline so exp 27 can iterate uniformly over all
              four configs.
            - ``"developmental"``: ``α = developmental_alpha(model.current_maturity)``
              each batch. Ramps from ``alpha_min`` at ``maturity=0``
              to ``alpha_max`` (capped, default ``0.85``) at full
              maturity.
            - ``"static"``: ``α = static_alpha`` constant for all
              batches. Used by the ``reward_only_static`` ablation.
        static_alpha: ``α`` value when ``alpha_mode == "static"``.
        gamma: Weight on the calibration term in
            :func:`compute_reward_signal`. Ignored when
            ``alpha_mode == "constant"``.
    """

    name: str
    gradient_gating_enabled: bool
    alpha_mode: AlphaMode
    static_alpha: float = 0.5
    gamma: float = 0.3

    def uses_reward_signal(self) -> bool:
        """True iff the on_after_batch callback should compute R and
        pass it to ``apply_hebbian_update``."""
        return self.alpha_mode != "constant"

    def alpha_for(self, model) -> float:
        """The ``α`` that :func:`compute_reward_signal` should see
        for this batch."""
        if self.alpha_mode == "static":
            return self.static_alpha
        if self.alpha_mode == "developmental":
            return developmental_alpha(model.current_maturity)
        raise ValueError(
            f"alpha is undefined for alpha_mode={self.alpha_mode!r}"
        )

    def make_callbacks(self) -> dict[str, Callable]:
        """Return ``{on_pre_optimizer_step, on_after_batch,
        on_task_change}`` closures wired for this config.

        The on_pre_optimizer_step closure conditionally fires
        gradient gating based on ``gradient_gating_enabled``. The
        on_after_batch closure decides between constant-R and
        per-sample-R based on ``uses_reward_signal()`` and computes
        the reward signal from the model's cached last logits when
        it does. The on_task_change closure forwards to
        :meth:`SynapseAugmentedMLP.notify_task_change` so cold-
        storage entries get ``task_id`` tagged identically to path-A.
        """
        config = self  # closure capture

        def on_pre_optimizer_step(task_index, task, model) -> None:
            if config.gradient_gating_enabled:
                model.apply_gradient_gating()

        def on_after_batch(task_index, task, model, x, y) -> None:
            if (
                config.uses_reward_signal()
                and model._last_logits is not None
                and y is not None
                and y.numel() > 0
            ):
                alpha = config.alpha_for(model)
                R = compute_reward_signal(
                    model._last_logits,
                    y.to(torch.long),
                    alpha=alpha,
                    gamma=config.gamma,
                )
                model.apply_hebbian_update(
                    training_target=y,
                    reward_signal=R,
                    reward_mode="per_sample",
                )
            else:
                # Baseline / safe-fallback path: existing scalar-R
                # behaviour. training_target=y still flows through
                # so path-A label storage continues to work.
                model.apply_hebbian_update(training_target=y)

        def on_task_change(task_index, task, model) -> None:
            model.notify_task_change(int(task_index))

        return {
            "on_pre_optimizer_step": on_pre_optimizer_step,
            "on_after_batch": on_after_batch,
            "on_task_change": on_task_change,
        }


REWARD_CONFIGS: dict[str, RewardConfig] = {
    "cs_gated_cosine_developmental": RewardConfig(
        name="cs_gated_cosine_developmental",
        gradient_gating_enabled=True,
        alpha_mode="constant",
    ),
    "cs_reward_developmental": RewardConfig(
        name="cs_reward_developmental",
        gradient_gating_enabled=False,
        alpha_mode="developmental",
    ),
    "cosine_reward_developmental": RewardConfig(
        name="cosine_reward_developmental",
        gradient_gating_enabled=True,
        alpha_mode="developmental",
    ),
    "reward_only_static": RewardConfig(
        name="reward_only_static",
        gradient_gating_enabled=False,
        alpha_mode="static",
        static_alpha=0.5,
    ),
}
"""Registry of named training configurations on the reward axis.

The baseline (``cs_gated_cosine_developmental``) is included so the
experiment driver can iterate over a single dict uniformly. It uses
the historical constant-R Hebbian path (``alpha_mode="constant"``);
its training behaviour is identical to the unchanged
``scout_a095_validated`` config from path A and earlier."""
