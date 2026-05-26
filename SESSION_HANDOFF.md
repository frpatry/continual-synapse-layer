# Session handoff — 2026-05-26 (frozen-encoder pretraining ready)

## Where we are

The dual-substrate architecture is implemented and partially tested.
What we've learned, in order:

1. **Substrate separation makes the system viable** but doesn't
   automatically dissolve the plasticity-stability trade-off. The
   first pilot (`results/logs/episodic/1779817131_28_T15_dual_substrate.json`,
   `blend_max=0.5`, no re-encoding) showed Task-0 = 0.218 — much
   worse than baseline 0.798.
2. **Feature drift in the trainable encoder is real.** Adding
   re-encoding at task boundaries
   (`results/logs/episodic/1779818599_28_T15_dual_substrate.json`,
   `blend_max=1.0`, re-encoding ON) lifted Task-0 a bit (0.389) but
   killed Task-N (0.755, down from 0.928). Diagnostic from the run:
   ```
   per-task memory size: t0=92, t1=92, t2=92, ..., t11=92, t12=101, t13=101, t14=103
   ```
   The trainable encoder's feature space drifts so heavily that
   tasks 1-11 register as ≤ novelty_threshold against the existing
   memory — so the store is dominated by task-0 prototypes, and
   high `blend_max` then over-weights stale task-0 retrieval on
   every input.
3. **Pivot diagnosis**: the keying function for the memory must be
   stable across continual training. A trainable encoder will
   always drift; we need a frozen one.

The new direction (Option B): pretrain a **permutation-invariant**
encoder via self-supervised contrastive learning on raw MNIST, then
freeze it and use it as the keying function for the episodic
memory. The base classifier still trains freely; the memory's
feature space is locked in by construction.

## Why permutation-invariant pretraining?

Permuted-MNIST defines every task as a different random permutation
of the 784 input pixels. If the encoder is **invariant to those
permutations** by design, then "same digit under different
permutations" maps to nearby points in feature space — which is
exactly the inter-task geometry the memory needs to preserve. The
pretraining objective makes this invariance the explicit loss:
two random perms of the same image are positive pairs in a SimCLR
InfoNCE loss.

This is methodologically clean: no future-task leakage (we only
use raw MNIST, which we've always had access to as prior
knowledge), no peeking at the continual benchmark's labels, no
synthetic data hacks.

## What ships in this session (incremental, all committed)

Six prior commits + four new ones land the frozen-encoder line:

- `96244cb` — Phase 0: dual-substrate pivot doc + decisions_log
- `342b3c4` — Phase 1: ActiveEpisodicMemory + 8 tests
- `7057d6f` — Phase 2: EpisodicPredictor + 6 tests
- `56fe242` — Phase 3: cs_episodic_dual_substrate config
- `118ddba` — Phase 4: exp 28 driver (not run in-session)
- `864d773` — Phase 5: handoff update
- `65ca031` — feature-drift fix Phase 1: raw_inputs + re_encode_all + 5 tests
- `4841649` — feature-drift fix Phase 2: re-encode wired into exp 28
- (new) `646c845` — Option B Phase 1: ContrastiveEncoder + InfoNCE
  + experiments/29 pretraining script + 6 tests
- (new) `34ed2ef` — Option B Phase 2: PretrainedContrastiveEncoder
  wrapper + 3 tests
- (new) `fab71d4` — Option B Phase 3: keying_encoder wired into
  EpisodicPredictor + exp 28 (smoke-tested at T=2)
- (this commit) — Option B Phase 4: handoff update

## Running the pilot (two commands, in order)

```bash
source .venv/bin/activate

# Step 1 — pretrain the frozen encoder. Run once.
python experiments/29_pretrain_contrastive_encoder.py --epochs 50

# Step 2 — continual eval with the frozen encoder as the memory's
# keying function.
python experiments/28_episodic_dual_substrate_eval.py \
    --T 15 --n_seeds 2 --keying-encoder pretrained_contrastive
```

**Important:** the venv must be activated first — the system has
no ``python`` on PATH outside the venv.

### Step 1 — what to expect from `experiments/29`

At the end of pretraining the script runs three sanity-check
assertions:

```
linear-probe MNIST test accuracy: 0.XX  (floor 0.90)
same-digit cross-perm sim:       0.XX  (floor 0.50)
gap (same − different):           0.XX  (floor 0.20)
```

If any of these fail, the script raises ``AssertionError`` but
**still saves the checkpoint** so you can inspect it. Expected
ballpark on CPU at 50 epochs:

- linear-probe ≥ 0.95 (MNIST is easy enough that a frozen
  contrastive encoder with a single linear head should clear it)
- same-digit similarity ≥ 0.70 (the contrastive objective directly
  optimises this)
- gap ≥ 0.30 (different-digit pairs are negatives in the loss; if
  the encoder collapsed all classes to one cluster, the gap would
  be near 0 — the assertion catches that failure mode)

ETA on CPU: 10–15 minutes at default `--epochs 50`. Outputs:
``results/pretrained/contrastive_encoder.pt`` (~few MB).

### Step 2 — what to expect from `experiments/28`

The startup banner should now show:

```
Keying encoder: pretrained_contrastive (loaded from ...)
Re-encoding mode: DISABLED (frozen keying encoder — re_encode_memory short-circuits)
```

The per-task ``re-encoded N entries`` lines won't appear (frozen
encoder ⇒ structural no-op; the print is suppressed in this mode).
The memory growth pattern is the key diagnostic — at the end you
should see something like:

```
per-task memory size: t0=N0, t1=N1, ..., t14=N14
```

**Uniform growth across all 15 tasks** is the dual-substrate
hypothesis confirmed: the frozen encoder maps different-perm
inputs to different feature regions, so the novelty gate fires
correctly on every task. If memory grows only on task 0 (like the
trainable-encoder runs), the encoder isn't permutation-invariant
enough and we'd need to retrain it.

ETA: similar to the previous pilot (~12 min at T=15, n=2 on CPU).
Maybe slightly faster because re-encoding is disabled.

## Decision criteria for the Option B pilot

- **Strong win**: ACC ≥ 0.78 AND Task-0 ≥ 0.70 AND memory grows
  uniformly across all 15 tasks → the dual-substrate hypothesis is
  confirmed with rigorous methodology. Worth investing in T=50
  n=5 in a later session.
- **Moderate**: Task-0 ≥ 0.55 with uniform memory growth → the
  architecture works but needs tuning (sweep `--blend-max`,
  `--novelty-threshold`).
- **Null**: Task-0 ≤ 0.40 OR memory still concentrates in Task 0
  → the encoder isn't permutation-invariant enough. First check
  the exp-29 sanity numbers (linear probe accuracy, same-digit
  similarity, gap). If those passed, the issue is downstream —
  maybe the encoder is permutation-invariant on raw MNIST but the
  retrieval threshold needs lowering, or the InfoNCE temperature
  needs tuning.

## Reference numbers to compare against

| pilot | ACC | Task-0 | Task-N | memory growth pattern |
|---|---:|---:|---:|---|
| baseline (cs_gated_cosine_developmental) | 0.816 | 0.798 | 0.906 | n/a |
| dual-substrate, blend_max=0.5, **trainable encoder, no re-encoding** | 0.666 | 0.218 | 0.931 | unknown |
| dual-substrate, blend_max=0.5, **trainable encoder, no re-encoding** (rerun) | 0.692 | 0.341 | 0.928 | unknown |
| dual-substrate, blend_max=1.0, **trainable encoder, re-encoding ON** | 0.589 | 0.389 | 0.755 | task-0-dominated (92 entries through task 11) |
| **dual-substrate, blend_max=0.5 (default), pretrained_contrastive keying** | **TBD** | **TBD** | **TBD** | **TBD — operator runs** |

The operator's question to answer: does the pretrained_contrastive
row push Task-0 from ~0.3-0.4 toward ≥ 0.55 (moderate) or ≥ 0.70
(strong)? AND does the memory grow on every task?

## What's been kept across the storage-line work

The path-A / path-B / path-C / path-D / reward / re-encoding
infrastructure all remains in the codebase. None of the new files
shadow the old; the new line lives entirely under
``src/continual_synapse/episodic/`` and
``experiments/28_*``, ``experiments/29_*``. All test changes are
additive.

## What's explicitly NOT done in this pilot

- The frozen encoder uses raw MNIST. The continual benchmark uses
  permuted MNIST. They share the digit-class label space but no
  permutation info — the encoder doesn't see the benchmark's
  specific permutations. This is the methodological discipline
  the user called out: no future-task leakage.
- No fine-tuning of the encoder during continual training. The
  frozen contract is structural (`PretrainedContrastiveEncoder.train()`
  is overridden to no-op), so even a stray `predictor.train()`
  call can't accidentally re-enable gradients on the encoder.
- Re-encoding is disabled in this mode. A frozen encoder produces
  identical output every call, so the round-trip is a no-op by
  construction. `predictor.re_encode_memory()` short-circuits to
  return 0 immediately when `keying_encoder` is set.

## Files of interest (new this session)

- ``src/continual_synapse/episodic/contrastive_encoder.py``:
  ``ContrastiveEncoder``, ``info_nce_loss``, ``random_permutation``,
  ``apply_permutation``.
- ``src/continual_synapse/episodic/frozen_encoder.py``:
  ``PretrainedContrastiveEncoder`` (loads + freezes + eval-locks).
- ``experiments/29_pretrain_contrastive_encoder.py``: the
  pretraining script with three sanity-check assertions.
- ``experiments/28_episodic_dual_substrate_eval.py``: now accepts
  ``--keying-encoder`` and ``--pretrained-encoder-path``.

## Suite status

**444 tests passing** at the end of this session:

| pre-session (re-encode fix) | + contrastive_encoder | + frozen_encoder | total |
|---:|---:|---:|---:|
| 435 | +6 | +3 | 444 |

No new dependencies introduced (the linear probe in exp 29 uses a
single ``torch.nn.Linear`` rather than sklearn).

## If Option B works at T=15

The next session would:
1. Run T=50 n=5 with the same `--keying-encoder pretrained_contrastive`
   to validate at scale (expected ~few hours wall-clock).
2. Run exp 24 retention analysis on the T=50 JSON for the retention
   curve + heatmap + Wilcoxon Bonferroni comparison vs the existing
   T=50 baseline data.
3. Consider promoting `ActiveEpisodicMemory` to ChromaDB-backed
   storage if entry counts get into the thousands.

## If Option B fails

Three diagnostic branches, in order of cost:

1. **Tune the InfoNCE temperature / projection_dim / epochs** in
   exp 29 and re-pretrain (cheap, ~30 min).
2. **Try a different frozen encoder** — random projection
   (`nn.Linear(784, feature_dim)` with no training) as an extreme
   ablation. If random projection beats trainable encoder, the
   issue is drift, not representation quality.
3. **Pivot away from the dual-substrate story entirely** — the
   Pareto-frontier framing from the Phase B verdict remains the
   fallback story for the article.

## Open todos (deferred, not blocking)

- n=10 validation on Phase B configs at T=50 (unchanged).
- Decay-subsystem honesty note in README (unchanged).
- Promote ``ActiveEpisodicMemory`` to disk-backed storage if T=50
  pilot needs it (deferred until we know whether the architecture
  works at T=15).
