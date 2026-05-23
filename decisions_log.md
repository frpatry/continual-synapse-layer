# Decisions Log

A running log of architectural decisions made during the project. See
PROJECT_PLAN.md section 9 for the format. Entries are appended in
reverse chronological order (newest first).

---

## [2026-05-23] Phase 3 (partial): evidence-based resistance + reward signal

This session implemented the two mechanisms scoped for "addressing
the Phase-2 failure mode": evidence-based resistance and a real
reward signal (external + consistency + surprise + mixer with
developmental trajectory). Deferred to a follow-up Phase-3 session:
``confidence``, ``age``, ``access_count``, and sparse top-k partner
selection.

### Why this session was narrower than the full Phase-3 task list

**Decision:** Scope the session to the two mechanisms the user
explicitly named, defer confidence and sparse top-k.

**Rationale:** Phase 2 closed with a specific, measurable failure
mode ("synapse just records the latest task"). Resistance directly
attacks that failure (a high-evidence synapse no longer shifts);
the reward signal attacks it indirectly (consistency drops on
task switches, dampening updates). Confidence and sparse top-k are
useful but orthogonal — they belong in their own session so the
ablation is clean. Recorded the deferrals in
[[phase-3-deferred]].

### Evidence semantics: absolute co-activation

**Decision:** ``evidence[i, j] += mean_b(|a_{b,i}| · |a_{b,j}|)``.

**Alternatives considered:**
- Signed outer product ``mean_b(a_{b,i} · a_{b,j})``: gets cancelled
  when samples have opposite-signed activations on the same neuron
  pair. We want "this pair fires together a lot" regardless of sign.
- Indicator above a threshold: requires a hyperparameter and loses
  magnitude information. Not used.

**Trade-off:** Absolute-value form means evidence is monotonically
non-decreasing per neuron pair. Without a decay term, evidence
grows unboundedly across many updates and resistance asymptotically
freezes the strength matrix. For Phase 3 this is acceptable
(Split-MNIST runs are short); a Phase-4 long-horizon experiment may
need a decay or windowed-mean variant.

### β=0 default for strict v1 compatibility

**Decision:** ``SynapseLayer(resistance_beta=0.0)`` takes a fast
path that is bit-identical to Phase-2 v1. Evidence still accumulates
but it does not multiply into the strength update.

**Rationale:** The user explicitly wanted Phase 2 numbers
reproducible. The test
``test_beta_zero_strength_update_matches_v1_exactly`` locks it.

### β must be calibrated to the evidence scale

**Observation (not a permanent decision):** With activations from
the MLP backbone and Split-MNIST, evidence reaches ~2800 after one
full run. ``β = 1`` then gives resistance ≈ 1/2800 — strengths
never move and the synapse layer is functionally dead. The useful
range we found is ``β = 0.001 … 0.1``.

This calibration sensitivity is a real ergonomic problem for the
system. Two ways to address it later:

1. Normalise evidence to ``[0, 1]`` (e.g., divide by a running max).
   Then ``β`` has a benchmark-independent meaning.
2. Schedule evidence with a decay so it settles at a known scale.

Marking as a known issue. For Phase-3 reporting we tune β manually.

### Reward semantics: at least one source, but no required mix

**Decision:** The ``RewardMixer`` accepts any non-empty subset of
``{external, consistency, surprise}``. Trying to construct one
with no components raises.

**Rationale:** Three sources give the user three independent
ablation knobs and Phase-3 experiments need to exercise that
freedom. The "at least one" rule prevents silently misconfigured
mixers from returning zero forever.

### Mixer formula has explicit edge cases

**Decision:** When only ``external`` is configured, return its
value verbatim (no ``α``-decay). When only internals are
configured, return their weighted sum (no ``α`` involvement).
When both are present, apply the literal formula
``α · external + (1 - α) · (w_c · cons + w_s · surp)``.

**Rationale:** Decaying the only available signal to zero (as the
literal formula would do with only external) is obviously wrong.
DESIGN.md's formula assumes both kinds of sources are present, so
when one is missing the natural reading is "the other dominates".
Documented in the module docstring so the next reader knows it
was intentional, not a bug.

### ``validated_evidence(t)`` proxied by step count

**Decision:** Phase 3 v1's ``α(t)`` denominator uses the mixer's
own call count rather than the DESIGN.md formula
("count of times external reward confirmed an internal signal").

**Rationale:** The DESIGN.md formula needs a clear definition of
"confirmation" (a threshold? a correlation? a sign agreement?) and
a joint statistic over reward sources. None of those are settled
yet, and a wrong choice now would be load-bearing. The call-count
proxy preserves the qualitative trajectory (α high early, decaying
later) and keeps the door open for a real ``validated_evidence``
implementation when we have data to ground it. Module docstring
calls this out so the substitution is obvious.

### Surprise reward: enable_grad inside no_grad caller

**Decision:** ``SurpriseReward.__call__`` wraps its predictor's
``loss.backward()`` in ``torch.enable_grad()``.

**Why:** ``SynapseAugmentedMLP.apply_hebbian_update`` is decorated
``@torch.no_grad`` (the Hebbian update is gradient-free), and the
reward computer is called from inside it. SurpriseReward needs
gradient flow for its own online SGD step on a tiny linear
predictor. We re-enable autograd only for that local update; the
surrounding no-grad context for the synapse update is unchanged.

Caught by experiment 04 mode=resistance_full; the unit tests for
SurpriseReward exercise it directly and so passed without revealing
the interaction. Added a docstring note.

### Phase 3 numbers, honestly

Single seed, Split-MNIST, 2 epochs per task:

| Method                                | ACC   | FGT   |
|---------------------------------------|-------|-------|
| Naive baseline (Phase 1)              | 0.604 | 0.483 |
| EWC, λ=1000 (Phase 1)                 | 0.636 | 0.434 |
| Synapse v1 (Phase 2)                  | 0.578 | 0.521 |
| + resistance, β=0.01                  | 0.611 | 0.478 |
| + reward (β=0)                        | 0.577 | 0.521 |
| + resistance + reward, β=0.01         | 0.613 | 0.474 |

What this says:

- **Resistance is doing the work.** Turning β from 0 to 0.01 shifts
  ACC by +3.5 pp and forgetting by -4.3 pp.
- **Reward signal alone is not enough.** Without resistance,
  consistency rarely dips below 0.97 — the EMA tracks task switches
  fast enough to keep reward near 1.0. The dampening is too weak to
  matter on its own.
- **The full system slightly beats baseline** (0.613 vs 0.604) but
  still trails EWC (0.636). The next Phase-3 mechanisms
  (confidence, sparse top-k) are where the next gains should come
  from. We have not run multi-seed yet; the +0.9 pp gap vs baseline
  is at single-seed noise scale.

### Deferred to follow-up Phase-3 session

- ``confidence``, ``age``, ``access_count`` state fields.
- Sparse top-k partner selection in
  ``synapse_layer/topk.py``.
- Multi-seed runs and statistical significance — proper ablation
  needs at least 5 seeds with Wilcoxon signed-rank per
  PROJECT_PLAN.md §8.
- Notebook visualising synapse state, evidence, and α(t) over time.

---

## [2026-05-23] Phase 2: SynapseLayer v1

This session implemented the first iteration of the additive
synapse layer, the gated modulator, the augmented model wrapper,
and experiment 03. The numbers say v1 does not yet beat baseline;
the Phase-2 checkpoint criteria are still met. Phase 3 will
revisit performance once metacognition is in place.

### State and read-out are separate modules

**Decision:** Split the synapse system into two cooperating
modules — `SynapseLayer` holds state (the strength matrix and
global step counter); `SynapseModulation` reads state to produce
a correction. Neither is a "synapse layer" on its own.

**Alternatives considered:**
- One monolithic `SynapseLayer` whose `forward(x)` returns the
  corrected activations. Conflates two independently-ablatable
  choices (what to store vs. how to read it).
- Subclassing the base MLP. Forces inheritance and ties the synapse
  design to a specific backbone shape.

**Rationale:** Phase 3 introduces confidence, evidence, and sparse
top-k, all of which are state-side changes. Phase 4 may try other
read-out strategies (e.g., dot-product against archived clusters)
without touching state. Keeping them in separate modules means
each phase can ablate one axis at a time.

### `gate=0` at init guarantees identical-to-baseline behaviour

**Decision:** The modulator gate is a scalar `nn.Parameter`
initialised to `0.0`. Combined with `strengths=0` at init, the
correction is exactly zero on the very first forward pass.

**Rationale:** The user instruction was explicit ("Initialize near
zero so base model behavior is preserved at init"). A literal zero
is the strongest guarantee available and makes the
identical-to-base test bit-exact rather than approximate. The gate
unfreezes naturally on the second batch once the first Hebbian
update gives the strengths some signal; the result is a clean
"start as baseline, then deviate" trajectory.

**Trade-off:** One batch of delay before the gate can move. In
practice the delay is invisible — far smaller than the time it
takes for either the gate or the strengths to reach a useful scale.

### Manual `consolidate()` over hidden side-effects

**Decision:** `SynapseLayer.consolidate(activations, reward=1.0)`
must be called explicitly by the training loop. The Phase-2 spec
calls this out ("Update triggered manually for now").

**Rationale:** Hidden side-effects (e.g., the synapse updating
inside `forward`) make the code harder to read and test, and they
prevent evaluation passes — where we don't want to update the
synapse — from sharing the same forward path. The explicit call
also gives Phase 3 an obvious place to plug the reward signal.

### `on_after_batch` runner hook over a SynapseRunner subclass

**Decision:** Add a third callback to `ContinualRunner`:
`on_after_batch(i, task, model, x, y)`, fired after every
optimizer step. The augmented model wires it to
`apply_hebbian_update()`.

**Alternatives considered:**
- Subclass `ContinualRunner` to a `SynapseRunner`. Duplicates the
  batch loop and prevents composing EWC + synapse cleanly later.
- Make the augmented model trigger updates internally via a flag
  on `forward`. Mixes inference and training paths.

**Rationale:** Three small callbacks (`regulariser`, `on_task_end`,
`on_after_batch`) cover EWC, Hebbian updates, and any future
per-batch / per-task method we need. Phase 3 will likely add no
new hook points.

### No strength clipping in v1

**Decision:** No clipping or saturating function on
`SynapseLayer.strengths`. Numerical stability is guarded by a
small default `learning_rate=1e-3` plus the stability test that
runs 100 updates and asserts strengths stay finite.

**Rationale:** Clipping hides bugs. The Phase-2 spec assumes
small learning rates; the experiment 03 sweep below shows the
strength range moves with `synapse_lr` exactly as expected.
Phase 3's evidence-based resistance (`1 / (1 + β · evidence)`)
will provide a principled self-saturating mechanism.

### Hebbian observes pre-correction features

**Decision:** The features fed into the synapse's Hebbian update
are the *pre-correction* base output, not the post-correction
activations used by the classifier head.

**Rationale:** If the synapse observed its own correction, the
update would self-reinforce: high gate × high strengths →
larger correction → larger reported activations → larger update,
ad infinitum. Observing the raw base activations breaks the loop
and matches DESIGN.md ("Hooks into a chosen layer of the base
model"). The unit test
`test_apply_hebbian_update_uses_pre_correction_features` locks
this in.

### v1 numbers honestly do not beat baseline

**Observation:** On a single seed with `epochs_per_task=2`:

| Method (lr / λ / synapse_lr) | ACC | FGT |
|---|---|---|
| Naive | 0.604 | 0.483 |
| EWC, λ=1000 | 0.636 | 0.434 |
| EWC, λ=100000 | 0.493 | 0.254 (plasticity collapse) |
| Synapse v1, synapse_lr=1e-6 (inert) | 0.604 | 0.483 |
| Synapse v1, synapse_lr=1e-4 | 0.608 | 0.480 |
| Synapse v1, synapse_lr=1e-3 | 0.578 | 0.521 |
| Synapse v1, synapse_lr=5e-3 | 0.529 | 0.581 |

The dense-Hebbian-with-fixed-reward formulation has no incentive
to preserve past tasks; it simply records co-firing of whatever
the current task is producing. Without confidence-based resistance
(Phase 3) the synapse correction follows the latest task and
slightly *amplifies* the forgetting of older tasks.

This is what the Phase-2 spec predicts ("Does the synapse layer
measurably affect output? Yes. Reduce forgetting yet? Not
necessarily."). Recording it here so Phase 3 can use the gap as a
quantitative baseline to beat.

---

## [2026-05-23] Phase 1 close-out: EWC, experiments, CI, README

This session closed the remaining Phase 1 deliverables.

### EWC implementation

**Decision:** Implement diagonal *empirical* Fisher Information,
estimated sample-by-sample on each task's training set, with
per-task storage of Fisher and parameter snapshots.

**Alternatives considered:**
- **True Fisher** (expectation under the model's predictive
  distribution): more correct, slower, requires sampling y from
  softmax(logits). Empirical Fisher uses the true labels and is the
  standard choice in practical EWC reproductions (e.g., the original
  Kirkpatrick code, most public EWC repos).
- **Mini-batch Fisher** (average gradient within a batch then
  square): faster but biased — the Fisher is the expectation of
  *per-sample* squared gradients, and squaring a batch-averaged
  gradient does not equal averaging squared per-sample gradients.
- **Online EWC** (accumulate Fisher into a single running matrix
  with a forgetting factor): saves O(T) memory. The original paper
  keeps per-task Fisher; for Phase 1 we follow the paper. Phase 5
  re-evaluations may revisit this.

**Trade-offs:** Sample-by-sample Fisher is slow (`fisher_sample_size`
caps the cost). Per-task storage is `O(T * P)` where `P` is the
parameter count. Both are fine at MLP scale; will revisit at
transformer scale.

### Runner extension points (vs. subclassing)

**Decision:** Add two optional callbacks to `ContinualRunner` —
`regulariser(model) -> Tensor` (per batch) and
`on_task_end(i, task, model)` (after each task). Continual-learning
methods are wired in by passing these callbacks at construction
time rather than subclassing.

**Rationale:** Callbacks compose: EWC + replay + the synapse layer
can all be active in the same run without inheritance gymnastics.
The runner stays one class. Cost is two extra fields and a few
lines of plumbing.

### EWC numbers on shared-head Split-MNIST

**Observation (not a permanent decision):** With the Phase-1
shared 2-class head, EWC exhibits the classic stability/plasticity
trade-off in a stark way:

| `λ` | ACC | Forgetting |
|---|---|---|
| 0 (naive) | 0.604 | 0.483 |
| 1000 | 0.636 | 0.434 |
| 10000 | 0.610 | 0.466 |
| 100000 | 0.493 | 0.254 |

Strong λ preserves old tasks but freezes the head and stops new
learning. Standard EWC reproductions use multi-head Split-MNIST
where the head per task absorbs task-specific gradients. We are
keeping the shared-head setup for now because it makes the
catastrophic-forgetting story unambiguous; Phase 5 will rerun the
EWC comparison with the multi-head variant as a separate baseline.

### Experiment logs: JSON, ignored by git

**Decision:** Each experiment run writes a JSON file to
`results/logs/<unix_ts>_<experiment>_<method>.json` capturing
config, accuracy matrix, metrics, and git SHA. The directory is
tracked via `.gitkeep`, but the logs themselves are git-ignored.

**Rationale:** Logs are easily regenerated from the script; keeping
them in git would bloat history without adding signal. A future
phase that needs to pin "the numbers for the article" can drop a
chosen subset under `results/tables/` with explicit naming.

### CI: minimal install, no heavy deps

**Decision:** The GitHub Actions workflow installs only
`torch`, `numpy`, and `pytest`. It does not install `datasets`,
`chromadb`, `transformers`, `matplotlib`, etc.

**Rationale:** The test suite uses synthetic tensors — it does not
download datasets or touch chromadb. Skipping those installs cuts
CI time and avoids flakes from external services.

**Trade-off:** Experiments are not exercised in CI. We'll need a
separate "long" workflow if/when we want nightly experiment runs.

### Reporting helper placement

**Decision:** Shared experiment-side code lives in
`src/continual_synapse/evaluation/reporting.py`. Experiments under
`experiments/` import it via the package.

**Rationale:** Keeps experiment scripts small and lets the
reporting code be unit-tested like the rest of the package. The
alternative — `experiments/_common.py` — would require sys.path
gymnastics and would not be testable through the standard pytest
configuration.

---

## [2026-05-23] Pinned versions bumped for Python 3.13

**Decision:** Replace the pins in PROJECT_PLAN.md section 5.1 with
versions that install on Python 3.13:

| Package | Plan pin | Used pin |
|---|---|---|
| torch | 2.4.0 | 2.12.0 |
| transformers | 4.45.0 | 4.57.1 |
| datasets | 3.0.0 | 4.0.0 |
| chromadb | 0.5.5 | 1.2.1 |
| numpy | 1.26.4 | 2.4.6 |
| scipy | 1.13.0 | 1.16.3 |
| matplotlib | 3.9.0 | 3.10.7 |
| seaborn | 0.13.0 | 0.13.2 |
| pytest | 8.3.0 | 9.0.3 |
| jupyter | 1.0.0 | 1.1.1 |
| tqdm | 4.66.0 | 4.67.1 |

**Rationale:** PROJECT_PLAN.md section 5.1 was written assuming
Python 3.10–3.12. The local environment is Python 3.13 (the only
non-3.7 interpreter installed) and torch 2.4.0 has no cp313 wheel.
Bumping to current versions is the smallest change that lets us
run the test suite. The torch + numpy + pytest combo is verified
to install and pass all 24 Phase-1 tests; the other libraries are
pinned to recent stable versions but only the ones used in Phase 1
have been exercised so far.

**Trade-offs:**
- numpy 2.x is an API break vs 1.x (e.g., `np.float_` is gone). Our
  Phase-1 code already targets the 2.x surface so this is fine, but
  it is something to keep in mind when porting reference EWC code
  from older repos.
- chromadb 1.x is a major version jump; the Phase-4 storage code
  should be written against the 1.x API directly.

**Reversibility:** Easy. Replace the pin file and rebuild the venv.

---

## [2026-05-23] Phase 1 scaffolding: benchmark, MLP, runner

**Decision:** Set up the Phase-1 evaluation harness (Split-MNIST,
3-layer MLP, sequential runner, ACC/forgetting/BWT/FWT metrics) as
described in PROJECT_PLAN.md section 7 / Phase 1.

**Rationale:** This is the foundation for every later phase. Without
a working continual-evaluation loop and a measurable forgetting
baseline, we cannot tell whether the synapse layer is doing anything.

### Sub-decisions made along the way

#### Split-MNIST: shared 2-class head, task-incremental setup

**Decision:** Each task is a binary classification with labels
remapped to ``{0, 1}`` inside the task. The model uses a single
2-output head shared across tasks.

**Alternatives considered:**
- **Multi-head**: one 2-class head per task. Standard in some
  continual-learning papers (e.g., the original EWC paper).
- **Single 10-class head**: keep raw digit labels. More realistic but
  changes the catastrophic-forgetting signature and complicates
  Phase-2 modulation.

**Rationale:** Shared 2-class head is the cleanest setup to *show*
catastrophic forgetting. The output space is identical across tasks
but the input distribution shifts, so the classifier necessarily has
to overwrite its decision boundary, which is exactly the failure
mode we want to measure. We can revisit multi-head if Phase 2 needs
it for the synapse-layer modulation story.

**Reversibility:** Easy. The benchmark API exposes ``classes`` on
each ``Task`` and the label remapping is in a single helper
(``_filter_and_remap``).

#### Data loader: HuggingFace ``datasets``, not torchvision

**Decision:** Use ``datasets.load_dataset("ylecun/mnist")`` rather
than ``torchvision.datasets.MNIST``.

**Rationale:** ``datasets`` is already in the pinned requirements
(section 5.1). ``torchvision`` is not, and adding it would expand
the dependency surface for one dataset. The Phase-2 transformer work
already uses ``datasets`` for Split-AG-News.

**Trade-off:** First-time download is slightly slower than
torchvision's. Negligible for a one-off experiment.

#### Benchmark construction: tensor-first, loader-second

**Decision:** ``SplitMNIST.__init__`` takes raw tensors. A separate
``from_huggingface`` classmethod fetches MNIST. Tests construct from
synthetic tensors.

**Rationale:** Keeps unit tests fast and offline. The HuggingFace
import is deferred to inside ``from_huggingface`` so importing the
package does not require ``datasets`` to be installed.

#### Metrics module: numpy-only, NaN-aware

**Decision:** Metrics operate on a ``(T, T)`` numpy accuracy matrix.
Un-recorded entries are ``NaN``; functions that need a particular
entry raise if it is missing.

**Rationale:** Decouples metrics from torch/PyTorch and makes the
matrix easy to serialise (CSV, JSON). The NaN sentinel matches what
the runner writes when it skips zero-shot evaluation.

#### Runner: zero-shot evaluation is optional

**Decision:** Before training task ``i+1``, the runner records
``R[i, i+1]`` as the model's current zero-shot accuracy on the next
task. This is needed for forward transfer. The behaviour is gated by
``record_zero_shot`` (default ``True``) so that lighter experiments
can skip it.

#### Runner: model-agnostic, optimizer-via-factory

**Decision:** ``ContinualRunner`` takes a ``model`` and an
``optimizer_factory`` callable. The model is trained sequentially
with the *same* optimizer instance across tasks.

**Rationale:** Naive sequential fine-tuning is exactly "keep going
with the same optimizer state", which is the worst-case continual
baseline we want to measure. Methods that need to reset the optimizer
between tasks, or attach extra state (EWC, replay buffers), will
subclass or wrap this runner in later phases.

#### MLP: 3 hidden layers, 256 wide, ReLU, no dropout by default

**Decision:** Default config matches the Phase-1 spec literally.
Dropout is configurable but off by default to keep the baseline
deterministic.

**Rationale:** Catastrophic forgetting is most legible on a small,
deterministic baseline. Dropout adds noise that can mask the
forgetting signal.
