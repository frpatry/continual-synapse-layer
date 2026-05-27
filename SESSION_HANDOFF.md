# Session handoff — 2026-05-27 (longer-training hypothesis)

## TL;DR — read this first

**The previous session closed the architectural-search line with a
null result.** Seven architecturally distinct attempts on
Permuted-MNIST T=15 small-MLP; only one (`cs_gated_cosine_functional`,
DER-equivalent at ACC=0.904, Task-0=0.908) ties the literature, none
surpasses it.

**New hypothesis for this session: every config we've run uses
`epochs_per_task=1`.** Models may simply be undertrained. Longer
training per task could (a) lift the DER-equivalent ceiling on
`cs_gated_cosine_functional`, and (b) give the memory-augmented
architecture's access heads (`query_proj`, `context_combiner`,
`memory_gate`) enough gradient signal to escape the bad local
optimum that produced the null verdict last session.

**Goal of this session: characterize the
ceiling-vs-epochs-per-task curve.** Sweep `--epochs-per-task` ∈
{1, 3, 5, 10} on the two configs that matter
(`cs_gated_cosine_functional` and `memory_augmented_native`) and
see whether more epochs raises the ceiling, leaves it flat, or
breaks things.

## Current state of the codebase

- Last commit `5a291b5` (clean working tree, **461 tests passing**).
- All seven prior architectural lines committed and reproducible.
- The memory-augmented loader-guard patch (`89380e9`) prevents
  silent maturity_target mismatches.

## Why this hypothesis is worth testing

Three concrete reasons the existing 1-epoch-per-task results may
be misleading:

1. **`cs_gated_cosine_functional` audit numbers showed
   under-convergence.** The audit (commit referenced in the prior
   handoff) reported per-task `task_loss` in the 0.5-0.9 range —
   notably above the 0.1-0.3 that fully-converged MNIST hits. The
   gating-induced slowdown was offered as the explanation, but
   undertraining is the simpler one. If we just need to train more,
   the DER-equivalent ceiling might rise meaningfully.

2. **The memory-augmented gate failure mode is consistent with
   undertraining.** The gate trained DOWN to ~0.16 in the
   pre-floor run, and even with the maturity floor forcing
   engagement, the access heads (`query_proj`, `context_combiner`)
   may simply not have seen enough gradient updates to learn a
   useful query/combine policy. 1 epoch × 938 batches × 15 tasks
   = 14,070 batches total for an architecture that needs to
   simultaneously learn the classification task AND learn how to
   use external memory. That's tight.

3. **The control config also showed Task-N collapse on Task-0
   (forgetting signature)**. `memory_augmented_no_memory` got
   Task-N=0.961 but Task-0=0.235 — classic naive-finetune
   catastrophic forgetting. More epochs per task makes catastrophic
   forgetting *worse* for unprotected models, but should make the
   protected/memory-using configs *relatively* better — widening
   the gap if memory is genuinely helping at all.

## Concrete experiment plan

### Step 1 — characterize the trend on the breakthrough cell

```bash
source .venv/bin/activate

# Reproduce the existing 1-epoch baseline (sanity — should hit ACC≈0.904)
# Already in results/logs/functional/; skip if you trust the prior audit.

# 3 epochs (~18 min — fits the harness 50-min cap comfortably)
python experiments/30_functional_regularization_eval.py \
    --T 15 --n_seeds 3 \
    --configs cs_gated_cosine_functional \
    --epochs-per-task 3

# 5 epochs (~30 min — still fits)
python experiments/30_functional_regularization_eval.py \
    --T 15 --n_seeds 3 \
    --configs cs_gated_cosine_functional \
    --epochs-per-task 5

# 10 epochs (~60 min — likely hits the harness cap; run as 3 separate
# single-seed jobs if so)
python experiments/30_functional_regularization_eval.py \
    --T 15 --n_seeds 1 --seed-base 0 \
    --configs cs_gated_cosine_functional \
    --epochs-per-task 10
# Repeat with --seed-base 1, then 2.
```

**Important:** exp 30's checkpoint loader will refuse to load
stale `phase_f/` checkpoints under a different
`--epochs-per-task` — actually, the loader doesn't currently
check this. You'll need to either (a) `rm` the relevant
checkpoints before each new-epoch run, or (b) the smarter
patch: add `epochs_per_task` to the exp 30 checkpoint
metadata + loader guard, mirroring the exp 31 fix in commit
`89380e9`. The patch is small and removes a footgun.

### Step 2 — same sweep on `memory_augmented_native`

```bash
# 3 epochs
rm results/checkpoints/phase_g/memory_augmented_native_T15_seed*.pt
python experiments/31_memory_augmented_eval.py \
    --T 15 --n_seeds 3 \
    --configs memory_augmented_native \
    --epochs-per-task 3 \
    --maturity-target 750

# 5 epochs
rm results/checkpoints/phase_g/memory_augmented_native_T15_seed*.pt
python experiments/31_memory_augmented_eval.py \
    --T 15 --n_seeds 3 \
    --configs memory_augmented_native \
    --epochs-per-task 5 \
    --maturity-target 750

# 10 epochs (likely needs single-seed splits)
rm results/checkpoints/phase_g/memory_augmented_native_T15_seed*.pt
python experiments/31_memory_augmented_eval.py \
    --T 15 --n_seeds 1 --seed-base 0 \
    --configs memory_augmented_native \
    --epochs-per-task 10 \
    --maturity-target 750
# Repeat for seeds 1, 2.
```

Exp 31's loader does check `maturity_target` (commit `89380e9`)
but NOT `epochs_per_task`. Same patch opportunity as above.

### Step 3 — interpret

For each config, build a small table:

| epochs_per_task | ACC | Task-0 | Task-N | wall_time |
|---:|---:|---:|---:|---:|
| 1 | (existing) | (existing) | (existing) | (existing) |
| 3 | ... | ... | ... | ... |
| 5 | ... | ... | ... | ... |
| 10 | ... | ... | ... | ... |

Three possible patterns:

- **Monotonic improvement**: ceiling was undertraining-limited.
  The architectural ceiling thesis was wrong. The breakthrough is
  to retrain everything at the saturation epochs-per-task.
- **Saturation by 3-5 epochs, then flat**: 1 epoch was indeed
  under-converged but the architectural ceiling is real. The
  saturated number is the honest version of each cell.
- **Improvement then degradation**: the model overfits the current
  task at high epochs-per-task, which makes catastrophic
  forgetting worse. There's a sweet spot.

The most interesting cell to watch is `memory_augmented_native` at
high epochs — if Task-0 rises from 0.19 (current) to anywhere
above ~0.30, the architectural pivot wasn't dead, it was just
undertrained. If it stays flat, the seven-attempts null was the
right read.

## Wall-clock budget

The harness background-task cap is ~50 minutes per invocation
(established empirically across this project — path-A succeeded
at 19 min, path-C got killed at ~50 min, T=50 baseline got killed
at ~60 min into seed 1). Plan accordingly:

| config | 1 epoch / seed | 3 epochs | 5 epochs | 10 epochs |
|---|---:|---:|---:|---:|
| cs_gated_cosine_functional | ~6 min | ~18 min | ~30 min | ~60 min ⚠ |
| memory_augmented_native | ~30 sec | ~90 sec | ~3 min | ~6 min |

For exp 30 at 10 epochs, n=3 in one invocation would be ~3 hours
— far over the cap. Split into single-seed jobs. exp 31 is small
enough that n=3 at 10 epochs (~18 min) fits comfortably.

## Pre-flight: add `epochs_per_task` to both loader guards

Recommended ~10-line patch to exp 30 and exp 31 before doing the
sweep, mirroring the maturity_target guard in commit `89380e9`.
Without it, the second invocation in each sweep will silently
load the first invocation's checkpoint (different epochs) and
just re-evaluate it — exactly the bug that wasted the last
maturity-target sweep.

The change for exp 30: in `_save_functional_checkpoint`'s payload,
add `"epochs_per_task": int(args.epochs_per_task)`; in the load
path, check it matches and raise a clear error if not. Same
shape as the exp 31 maturity guard.

## What stays from the prior session

The prior session's three-option decision is **deferred, not
cancelled**:

- Option 1 (change benchmark to Split-CIFAR-10/100)
- Option 2 (scale to transformer)
- Option 3 (stop and write up)

If the epochs-per-task sweep shows monotonic improvement on either
config, the project is back in active research mode and the
three options become re-relevant *with new data*. If the sweep
saturates flat (the architectural ceiling thesis was right), the
three options are exactly the same decision as before — and option
3 (write up) becomes the stronger choice because we'll have ruled
out one more confounder.

## Files of interest

- `experiments/30_functional_regularization_eval.py` — exposes
  `--epochs-per-task` (default 1).
- `experiments/31_memory_augmented_eval.py` — exposes
  `--epochs-per-task` (default 1).
- `decisions_log.md` — full narrative for every prior pivot. The
  memory-augmented null verdict still needs to land here (see
  open todos).

## Open todos (carried from prior session)

- **Memory-augmented null verdict in decisions_log.** Pivot entry
  exists (`dda0881`); the verdict entry doesn't. Should cover the
  three sweep numbers, the parasitic-memory diagnosis, the
  seven-attempts pattern.
- **Add `epochs_per_task` to exp 30 + exp 31 loader guards.**
  ~10 lines each, prevents the silent-stale-checkpoint footgun
  during the sweep.
- **n=10 validation at T=50** (still deferred).
- **T=50 n=3 functional reg pilot** (started, hit harness
  timeout — resumable in chunks).
- **README decay-subsystem honesty note** (still deferred).

## Suite status

**461 tests passing** at end of prior session. No changes this
session beyond the handoff rewrite.
