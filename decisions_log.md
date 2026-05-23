# Decisions Log

A running log of architectural decisions made during the project. See
PROJECT_PLAN.md section 9 for the format. Entries are appended in
reverse chronological order (newest first).

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
