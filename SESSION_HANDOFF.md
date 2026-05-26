# Session handoff — 2026-05-26 (functional regularization, Phase F infrastructure ready)

## Where we are

The dual-substrate / retrieval-ensemble line is closed. Across path-A
(true labels), path-B (labels-as-of-now), path-C (per-class
prototypes), and the episodic dual-substrate line in three variants
(trainable encoder, trainable + re-encoding, frozen contrastive
encoder), every post-hoc memory approach failed to preserve Task-0.
The best across the entire line is **Task-0 ≈ 0.39**, vs the
unchanged baseline ``cs_gated_cosine_developmental`` at **0.798**.

The pattern across the whole project is now unambiguous:

- **Interventions during training preserve knowledge.** Cosine
  gating (``cs_gated_cosine_developmental``) and EWC hold Task-0
  in the 0.8 range. Both intervene in the gradient step itself —
  gating scales ``base.parameters()`` gradients down on familiar
  inputs; EWC adds a Fisher-weighted penalty to the loss.
- **Post-hoc retrieval-based corrections do not.** Every variant
  that tried to recover lost knowledge at inference time, with the
  base model's weights free to drift during training, capped at
  ~0.4 Task-0.

The new pivot, codified in the latest decisions_log entry: **Learning
Without Forgetting (Li & Hoiem 2017)**, adapted to the continual
setting. Store ``(input, soft_target)`` pairs at the end of each
task; during subsequent task training, add a knowledge-distillation
loss against the stored soft targets. The model's weights are free
to move; what's anchored is its **function on selected past inputs**.

This is structurally a training-time intervention — the regulariser's
gradient flows through the loss into all the model's weights at the
same step the task gradient does. But its restoring force acts on
function rather than weights, which is the part EWC arguably gets
wrong. The composition ``cs_gated_cosine_functional`` is the most
interesting cell: both mechanisms intervene during training but on
different axes (weight scaling vs function anchoring); they may be
additive.

## What ships in this session (incremental, all committed)

Three commits land the functional-reg line:

- `1399b4a` — Phase 0: pivot rationale + decisions_log entry
  consolidating the "interventions during training work" /
  "post-hoc don't" pattern across all the prior architectures.
- `6abbcfc` — Phase 1: ``FunctionalMemory`` + ``distillation_loss``
  utility module with the Hinton T² rebalancing. 7 unit tests.
- `fb7b3ee` — Phase 2: ``experiments/30_functional_regularization_eval.py``
  driver. Three configs (baseline reload, ``cs_functional_only``,
  ``cs_gated_cosine_functional``). Custom training loop with the
  per-batch ``task_loss + λ·reg_loss`` composition. Smoke-tested
  at T=2 n=1.

The earlier dual-substrate / episodic infrastructure (commits
through `90dbd78`) all stay in the codebase — `src/continual_synapse/episodic/`
is intact, exp 28 + exp 29 still work, the contrastive encoder
checkpoint at ``results/pretrained/contrastive_encoder.pt`` is left
on disk. The new functional-reg line is purely additive and lives
under ``src/continual_synapse/functional/``.

## Running the pilot

```bash
source .venv/bin/activate
python experiments/30_functional_regularization_eval.py --T 15 --n_seeds 2
```

(The venv must be activated first — the system has no ``python`` on
PATH outside the venv.)

ETA: at T=2 n=1, ``cs_functional_only`` took 3.4s, ``cs_gated_cosine_functional``
took 40s. Extrapolating to T=15 n=2 and adding the baseline reload:

- cs_gated_cosine_developmental (reload from disk + eval): ~5s × 2 seeds
- cs_functional_only (train + eval): ~30s × 2 seeds
- cs_gated_cosine_functional (train + eval, synapse + LwF overhead): ~5 min × 2 seeds

**Total ETA: ~12 minutes.**

## What to watch in the printed output

The headline summary looks like:

```
=== Functional regularization pilot — T=15, n=2 ===
config                              ACC     Task-0    Task-N    FGT     memory
cs_gated_cosine_developmental      0.815    0.798     0.906    +0.11   N/A (ref)
cs_functional_only                 X.XXX    X.XXX     X.XXX    X.XX    1500 avg
cs_gated_cosine_functional         X.XXX    X.XXX     X.XXX    X.XX    1500 avg
```

Per-seed diagnostics also print:

- **per-task memory growth**: ``t0=+100(100), t1=+100(200), ..., t14=+100(1500)``.
  Exactly 100 entries should be added at every task end — confirms
  ``record_task_end`` fires correctly.
- **avg reg_loss per task**: should be 0 on task 0 (empty memory)
  and strictly positive on tasks 1..N (distillation active against
  stored soft targets). The first non-zero value lands at task 1.
- **per-task avg task loss**: should track the usual MNIST training
  curve.

The smoke at T=2 n=1 produced exactly this pattern (memory grew
t0=+100, t1=+100; reg_loss=0 on task 0, ~0.006-0.008 on task 1).

## Decision criteria

- **Strong win**: Either functional variant gives Task-0 ≥ baseline
  (0.798) AND ACC ≥ 0.78 → functional regularisation works in
  isolation or composition. Proceed to a T=50 + hyperparameter
  sweep in a later session.
- **Composition bonus**: if ``cs_gated_cosine_functional`` beats
  **both** ``cs_gated_cosine_developmental`` and ``cs_functional_only``,
  the two mechanisms are additive. That's a publishable composition
  finding on its own.
- **Moderate**: Task-0 in [0.75, 0.79] AND ACC ≥ 0.78 → roughly
  matches baseline; tuning needed (sweep ``--lambda-reg``,
  ``--samples-per-task``, ``--temperature``).
- **Null**: Task-0 < 0.60 for both functional variants → functional
  regularisation is insufficient at this scale on this benchmark.
  Document the negative result.

## Hyperparameters to sweep if the pilot is positive

The three knobs most likely to matter:

- ``--lambda-reg`` ∈ {0.5, 1.0, 2.0}: balance between task loss
  and distillation. Higher → more retention, less plasticity. The
  default 1.0 weights them equally.
- ``--samples-per-task`` ∈ {50, 100, 300}: how much of each task
  to remember. Bigger memory = more rehearsal coverage but more
  compute per training step. The default 100 → 1500 entries at
  T=15, manageable on CPU.
- ``--temperature`` ∈ {1.0, 2.0, 4.0}: distillation softness.
  Higher temperature exposes more inter-class similarity structure
  in the teacher signal. The default 2.0 matches Hinton's
  original paper. ``T=1`` reduces to plain KL between
  unsoftened distributions; ``T=4`` flattens substantially.

The composition config ``cs_gated_cosine_functional`` has more
hyperparameter interaction (gating's ``α``, maturity target, all
the scout_a095 knobs). The sensible v1 sweep keeps those at the
scout_a095 defaults and varies only the LwF hyperparameters.

## Two integration bugs surfaced + fixed during smoke

These are listed in the Phase 2 commit message but worth flagging
here too in case anyone hits a related issue:

1. **``_last_features`` cache pollution** — ``SynapseAugmentedMLP.forward``
   updates ``_last_features`` and ``_last_logits`` on every call.
   The composition config's memory forward (``model(x_old)``) would
   overwrite the task batch's cached values, so the subsequent
   ``apply_gradient_gating`` + ``apply_hebbian_update`` would
   scale gradients on the *memory* batch's features rather than
   the *current task*'s. Fixed by snapshotting + restoring the
   caches around the memory forward.
2. **Multi-pass observation buffer pollution** — every training-mode
   forward on a ``SynapseAugmentedMLP`` pushes ``n_passes`` (=5)
   activation observations into the synapse's buffer. The buffer's
   batched stack+mean step requires all observations to share a
   shape. The memory forward (and ``record_task_end``'s 100-sample
   snapshot) would leave differently-shaped observations in the
   buffer that crashed the next task's training forward. Fixed by
   forcing ``model.eval()`` around both the per-batch memory
   forward and the end-of-task snapshot — eval mode gates the
   observation hook on ``self.training``, so the buffer stays
   clean. Gradients still flow through the memory forward.

## Reference numbers for the comparison

Baseline ``cs_gated_cosine_developmental`` from the path-D pilot at
T=15 n=3: **ACC=0.8143, Task-0=0.8047, Task-N=0.9063**. Exp 30
reloads the same checkpoints (under
``results/checkpoints/phase_d/``) so the baseline row in the new
JSON should reproduce these numbers within rounding.

Across the path-A/B/C/D and dual-substrate lines, no variant ever
broke Task-0 ≥ 0.5 on T=15. The functional regularisation pivot is
the first principled attempt at a training-time intervention on
function. If it falls in the strong-or-moderate band, that's a
genuine result.

## What's still in flight from prior sessions

- T=50 n=5 validation on the existing Phase B configs (deferred).
- Decay-subsystem honesty note in the README (deferred).
- The contrastive encoder pretraining script (exp 29) and its
  T=15 dual-substrate pilot (exp 28 with the frozen keying
  encoder) are committed and runnable but conclusively negative;
  see ``results/logs/episodic/1779821036_28_T15_dual_substrate.json``
  for the diagnostic that closes that line.

## Files of interest (new this session)

- ``src/continual_synapse/functional/functional_memory.py`` — the
  ``FunctionalMemory`` class + ``distillation_loss`` function.
- ``src/continual_synapse/functional/__init__.py`` — package
  exports.
- ``experiments/30_functional_regularization_eval.py`` — the
  pilot driver with three configs.
- ``tests/test_functional_memory.py`` — 7 unit tests.

## Suite status

**451 tests passing** at end of session.

| pre-session (frozen encoder) | + Phase 1 functional memory | total |
|---:|---:|---:|
| 444 | +7 | 451 |

(Phase 2 added the driver script — manual run only, not exercised
by pytest. The Phase 2 commit message documents the smoke at T=2
that validated the training loop end-to-end.) No new dependencies
introduced.

## If functional regularization works at T=15

Next session would:
1. Validate at T=50 n=5 with the best-performing config and
   default hyperparameters.
2. Sweep ``--lambda-reg`` × ``--samples-per-task`` × ``--temperature``
   at T=15 n=5 to characterise the trade-off and identify any
   particularly strong cell.
3. Run exp 24 retention analysis on the T=50 JSON for the
   retention curve + Wilcoxon Bonferroni vs the existing T=50
   reference data.

## If it fails

If both functional variants land below Task-0 = 0.60, the honest
read is that the plasticity-stability trade-off as captured in
our experiments is **intrinsic** to the problem as we've defined
it — not to any specific architectural choice we've made. The
Pareto-frontier framing from the Phase B verdict (``decisions_log``
2026-05-26) is the publishable story in that case: a thorough
characterisation of why every reasonable architectural attempt
fails to close the EWC Task-0 gap is itself a contribution.
