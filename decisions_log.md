# Decisions Log

A running log of architectural decisions made during the project. See
PROJECT_PLAN.md section 9 for the format. Entries are appended in
reverse chronological order (newest first).

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
