# Session handoff — 2026-05-27 (memory-augmented null + decision point)

## TL;DR — read this first

**The functional-regularization result is the project's ceiling so far.**
`cs_gated_cosine_functional` at T=15 n=4: **ACC=0.904, Task-0=0.908,
FGT=0.003**. Audited end-to-end (six checks passed). Equivalent to
Dark Experience Replay (DER).

**Memory-augmented native (this session's pivot) hit a clean null.**
Even with the developmental maturity floor forcing memory engagement,
the architecture is *parasitic* relative to its own no-memory control
across all three maturity sweep settings. Headline numbers below.

**The project has reached a decision point.** Seven architecturally
distinct innovations have now been tried; only one (functional reg)
ties the DER baseline; none surpasses it. Three honest paths forward
documented at the bottom of this file.

## Current state of the codebase

All commits clean. Last commit `89380e9` patches an exp 31 loader
bug surfaced during this session's debugging (silent
checkpoint-mismatch on `maturity_target` was producing meaningless
numbers; loader now raises a clear RuntimeError on mismatch). Full
test suite: **461 passing**.

## Headline result from this session (Phase G, memory-augmented native)

After the fix to retrain from scratch (deleting stale Phase 2
checkpoints), three clean T=15 n=3 runs across the maturity sweep:

| config | maturity_target | ACC | Task-0 | Task-N | FGT |
|---|---:|---:|---:|---:|---:|
| **memory_augmented_no_memory (control)** | n/a | **0.469** | **0.235** | 0.961 | +0.727 |
| memory_augmented_native | 300 (aggressive) | 0.405 | 0.165 | 0.953 | +0.788 |
| memory_augmented_native | 750 (default) | 0.419 | 0.178 | 0.954 | +0.775 |
| memory_augmented_native | 1200 (gentle) | 0.436 | 0.191 | 0.954 | +0.763 |

JSON outputs in `results/logs/memory_augmented/`:
- `1779875332_31_T15_memory_augmented.json` (target=750 with control)
- `1779875759_31_T15_memory_augmented.json` (target=300)
- `1779875842_31_T15_memory_augmented.json` (target=1200)

Three findings stack into a clean null verdict:

1. **Memory is parasitic, not helpful.** Native is worse than the
   no-memory control across every maturity target — ACC drops 3-6
   pp, Task-0 drops 4-7 pp. Forcing the model to engage with
   memory makes things worse.
2. **Less floor pressure is monotonically better.** target=1200
   (gentlest floor) beats target=300 (most aggressive floor) by
   3 pp ACC. The architecture wants to *escape* the memory, not
   embrace it. This is the opposite of what the developmental
   hypothesis predicted.
3. **Task-N stays at ~0.954 across all four cells.** The model
   learns each new task fine; the damage lands entirely on
   retention. Memory is supposed to *help* retention but is
   *hurting* it.

Per the failure-mode playbook from the prior handoff, this matches
mode #3: the `value_proj` + `context_combiner` produces noisy
retrieved content that the model can't usefully exploit. Forcing
the gate open just dilutes the model's own predictions with
garbage.

## Seven architecturally distinct attempts vs DER

The full picture of architectural variants tried on
Permuted-MNIST T=15 with the small-MLP setup:

| line | best Task-0 | vs cs_gated_cosine_functional (0.908) |
|---|---:|---|
| path-A true-label retrieval | ~0.39 | far worse |
| path-B labels-as-of-now | ~0.20 | far worse |
| path-C per-class consolidation | ~0.39 | far worse (broke baseline) |
| path-D per-sample reward | 0.73 | worse |
| episodic dual-substrate (trainable encoder) | 0.39 | far worse |
| episodic dual-substrate (frozen contrastive encoder) | 0.25 | far worse |
| **functional regularization (LwF)** | **0.908** | **ties** — DER-equivalent |
| memory-augmented native (this session) | 0.19 | far worse |

The pattern is unambiguous: on Permuted-MNIST with a small MLP,
DER-equivalent appears to be the ceiling. Every architectural
innovation either ties it or fails outright. The trade-off as
captured in this benchmark + this model class appears to be
intrinsic to the setup, not a function of architectural choice.

## Validated mechanisms (don't lose these)

The codebase still contains the working DER-equivalent pipeline
plus seven architectural baselines. All are reproducible.

### The breakthrough cell — `cs_gated_cosine_functional`

Audited at T=15 n=4 with six independent checks (no train/test
contamination, +2.23pp generalization gap, healthy loss
magnitudes, flat per-task retention curve, exact memory math,
tight 4-seed replication). Code in
`src/continual_synapse/functional/` + `experiments/30_*.py`.
Reproduce with:

```bash
source .venv/bin/activate
python experiments/30_functional_regularization_eval.py --T 15 --n_seeds 3
```

Checkpoints persist in `results/checkpoints/phase_f/`.

### The memory-augmented infrastructure (works, just null)

The MemoryAugmentedMLP architecture is correctly implemented and
all 10 unit tests pass. The developmental maturity floor mechanism
works as designed (verified by tests + smoke). The null is a
*scientific* finding, not a *code* failure. Code in
`src/continual_synapse/memory_augmented/` + `experiments/31_*.py`.

## Suite status

**461 tests passing** at end of session. Test breakdown across the
project (rough categories):

- Core synapse / cold storage / consolidation: ~200
- Path A/B/C/D functional regularization: ~50
- Episodic dual-substrate + contrastive encoder: ~30
- Memory-augmented native (this session, including +2 maturity
  floor tests): 10
- Everything else (baselines, retrieval, reward, infra): balance

No new dependencies introduced this session.

## What the next session has to decide

Three honest paths forward. The current state is publishable as-is
(option 3); the other two extend the search before writing up.

### Option 1: Change the benchmark (medium cost, clean test)

Permuted-MNIST is brutally hostile to memory mechanisms — pixel
permutation destroys all spatial structure, leaving only
pixel-statistics for any encoder to grab onto. The contrastive-
encoder pretraining pilot (commit `646c845` / exp 29) collapsed
for exactly this reason — there's no spatial information to base
class-discriminative features on.

A spatially-structured benchmark (Split-CIFAR-10 / Split-CIFAR-100
/ Split-MiniImageNet) would let memory-augmented architectures
actually find useful neighbours. The headline question becomes:
does DER-equivalent generalise as a ceiling across benchmarks, or
is it specific to Permuted-MNIST?

Cost: ~1 day of code (new dataset loader in
`src/continual_synapse/evaluation/benchmarks.py` mirroring the
`PermutedMNIST` / `SplitMNIST` pattern; new exp 32 or sweep flags
in exp 30/31). Need to re-establish baseline numbers for the new
benchmark before any architectural conclusion.

Risk: changes the comparison story; existing baselines need
re-running. Best executed with a clear research question
(e.g. "does memory-augmented match DER on CIFAR-100?") rather
than open exploration.

### Option 2: Scale the architecture (high cost, harder to interpret)

A transformer-based encoder with cross-attention to memory
(Memorizing Transformers-style) has dramatically more parameters
and an inductive bias that fits attention-over-tokens naturally.

Cost: ~1-2 days of code + significantly longer training runs. The
small-MLP runs in this project take seconds-to-minutes per seed;
a transformer would be hours per seed.

Risk: might just push the DER-equivalent ceiling slightly higher
without changing the *gap* between methods. If transformer-DER hits
0.95 and transformer-memory-augmented hits 0.95, the architectural
finding hasn't changed.

Probably not the right next step — committing more compute without
a strong prior that the architectural class is what's limiting.

### Option 3: Stop and write up (lowest cost, most defensible)

The story is already publishable:

- A clean, audited DER-equivalent result
  (`cs_gated_cosine_functional`, T=15 n=4, ACC=0.904,
  Task-0=0.908, six-step audit passed).
- A systematic refutation of seven architectural innovations that
  failed to surpass DER.
- The Pareto-frontier framing from the Phase B verdict as a
  structural finding (the two informative hyperparameters of
  cosine gating trade Task-0 against aggregate ACC).
- A consolidated methodological lesson: "interventions during
  training preserve knowledge (cosine gating, EWC, functional
  reg); post-hoc retrieval-based corrections do not (every
  variant we tried)".

This is publishable negative-results / characterization work. The
field's publication bias toward novelty wins makes rigorous
"these N architectural variants don't beat DER" papers rare and
valuable.

### My recommendation (logged from this session)

Option 3 is the most defensible scientifically. Seven independent
attempts is enough evidence that this benchmark + model class
hits a real ceiling. Continuing to architect new mechanisms
hoping the next one will break through has reached diminishing
returns — every recent attempt has fallen into the same null
bucket, and the post-hoc explanations are getting strained.

Option 1 is the cleanest extension if you want to try one more
thing — Permuted-MNIST is genuinely hostile to memory mechanisms,
and testing on a spatially-structured benchmark would either
validate that the ceiling is benchmark-specific (genuine new
finding) or confirm it generalises (strengthens option 3's
story).

Option 2 should probably wait until after one of options 1 or 3.

The user has final say. The project is in a good place either
way.

## Files of interest (final state)

Existing tests + code that future sessions should not break:

- `src/continual_synapse/functional/functional_memory.py` — the
  DER-equivalent ceiling cell. Don't touch.
- `src/continual_synapse/memory_augmented/memory_augmented_model.py`
  — the architecture that hit null. Working; just doesn't help on
  Permuted-MNIST.
- `experiments/30_functional_regularization_eval.py` — reproduces
  the breakthrough cell.
- `experiments/31_memory_augmented_eval.py` — reproduces the
  memory-augmented null with maturity sweep capability.
- `decisions_log.md` — full chronological narrative. The most
  recent entries cover the memory-augmented pivot rationale and
  the (forthcoming, if you do it) memory-augmented null verdict.

## Open todos (deferred)

- **Write the memory-augmented null verdict into decisions_log.**
  The pivot entry exists (commit `dda0881`); the verdict entry
  doesn't yet. Should be a single entry under today's date
  covering: (1) the three sweep numbers above, (2) the
  parasitic-memory diagnosis, (3) the seven-attempts pattern,
  (4) the decision point.
- **n=10 validation on the Phase B configs at T=50** (unchanged
  from prior handoffs — deferred indefinitely).
- **T=50 n=3 functional regularization pilot** (started in a
  prior session, hit harness timeout, can be resumed in chunks
  per the prior handoff instructions). Would extend the
  DER-equivalent result to a longer benchmark.
- **README honesty note** about the decay subsystem (still
  deferred).
