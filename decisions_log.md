# Decisions Log

A running log of architectural decisions made during the project. See
PROJECT_PLAN.md section 9 for the format. Entries are appended in
reverse chronological order (newest first).

---

## [2026-05-24] Pressure-metric dilution fix — cs_full_sparse goes from catastrophic to baseline-equivalent

### The pathology

Experiment 12's ``cs_full_sparse`` method produced ACC
``0.534 ± 0.149`` (vs ``0.673 ± 0.007`` for dense ``cs_full``)
with a monotonic-by-seed-value crash (per-seed ACC ``0.652,
0.679, 0.578, 0.423, 0.336``). Post-exp seed-isolation
investigation found no code-level state leak; the Python lifecycle
was clean. The smoking gun came from the diagnostics: sparse mode
fired only **1–4 consolidations per 15-task run** (vs 123–153 for
dense), so cold storage stayed nearly empty, retrieval barely
contributed, and the modulator gate gradient went unstable in
seeds 3 and 4 (final gate values ``−0.44`` and ``−0.52``
respectively — large negative gate means the synapse correction
gets *subtracted* from the base activations, actively pulling
the model away from what it had learned).

Root cause: ``ConsolidationTrigger.should_fire`` averaged
pressure across the full ``(n, n)=(256, 256)=65_536`` buffer.
Sparse top-k mode zeros out ~75 % of entries (``top_k=64``,
``density=0.250``), so the zeros diluted the mean by ``1 /
density ≈ 4×``. The trigger almost never crossed
``pressure_threshold=0.005`` in sparse mode, even though the
*active* synapses had pressures well above it.

### The fix (commit ``4b64038``)

In ``src/continual_synapse/consolidation/trigger.py``, mask out
zero-strength entries before averaging::

    pressures = compute_pressure(synapse)
    active_mask = synapse.strengths != 0
    if not active_mask.any():
        return False
    avg = float(pressures[active_mask].mean().item())
    return avg >= self.avg_pressure_threshold

Dense mode has ``active_mask`` all-True so the masked mean equals
the unmasked mean **bit-exact**. Every prior dense experiment's
threshold dynamics are unchanged — verified by the existing 321
tests continuing to pass. Three new tests in
``test_consolidation_trigger.py`` lock the contract:

- ``test_should_fire_mean_pressure_active_mask_matches_dense_at_full_density``
  — dense backward compat.
- ``test_should_fire_not_artificially_deflated_in_sparse_mode``
  — equivalent active synapses produce equivalent threshold
  sensitivity regardless of how many zero entries surround them.
- ``test_should_fire_false_when_no_active_synapses`` — short-circuit
  to ``False`` when every strength is zero (e.g. immediately after
  a full drain), instead of accepting ``mean(zeros) == 0 >= 0``
  as a "fire" signal.

### Result: cs_full_sparse re-run with the fix

15-task Permuted-MNIST, dropout=0.5, 5 seeds, same hyperparameters
as experiment 12. Output:
``results/logs/audit_fixes/1779620809_sparse_fix_T15.json``.

| Method | per-seed ACC | mean ± std | mean FGT |
|---|---|---|---|
| cs_full_sparse (exp12 — broken) | 0.652, 0.679, 0.578, 0.423, 0.336 | 0.534 ± 0.149 | +0.431 ± 0.156 |
| **cs_full_sparse (fixed)** | **0.680, 0.654, 0.654, 0.672, 0.674** | **0.667 ± 0.012** | **+0.291 ± 0.013** |
| cs_full (exp12 — reference dense) | 0.664, 0.672, 0.669, 0.683, 0.676 | 0.673 ± 0.007 | +0.294 ± 0.008 |

Headline: **ACC up 13.3 pp** (0.534 → 0.667). **Std down 12×**
(0.149 → 0.012). The fixed sparse method is now within 0.6 pp ACC
and 0.005 std of the dense reference. The catastrophic crash on
seeds 3–4 is eliminated; every seed lands in a tight band.

Mechanism diagnostics (final-task snapshot, per seed):

| Variant | consolidations | store entries | final gate range |
|---|---|---|---|
| Original cs_full_sparse | 1–4 | 1–4 | −0.52 to −0.003 |
| **Fixed cs_full_sparse** | **82–119** (avg ~101) | 82–119 | **−0.084 to +0.017** |
| Dense cs_full | 123–153 (avg ~138) | 123–153 | −0.063 to +0.015 |

Consolidation rate jumped ~30× to within the same order of
magnitude as dense. The modulator gate now stays in the healthy
``[−0.1, +0.02]`` band — no more runaway. Sparse density is
unchanged at exactly ``0.250 = 64/256`` (as expected;
``apply_topk_mask_inplace`` is independent of the trigger).

### Cost: sparse is no longer "the fast variant"

Per-seed wall-clock went from ~80 s (when the trigger almost
never fired) to ~205 s (when it fires ~100 times per seed and
each fire writes to chromadb + invalidates the retrieval cache).
Total per-method runtime is now comparable to dense ``cs_full``
(~17 min for 5 seeds, vs ~19 min for cs_full at exp12). The prior
"sparse is cheap" estimate was an artifact of the bug — the
correct architecture runs at the same scale as dense.

### What this overturns in the audit narrative

The post-experiment-11 audit entry (this file, dated 2026-05-23)
listed component #9 (sparse top-k) as "✗ implemented but never
activated" and component #24 (candidate selection) as "⚠ uses
quantile not threshold; sensible deviation". Component #9 *was*
activated in experiment 12 and produced the pathology described
here; the fix completes that work. The pressure-metric dilution
should also be added to the audit table as a separate row — the
trigger's averaging convention assumed dense semantics implicitly
and wasn't flagged when the top-k code path was added.

### Implication for the architectural call

Experiment 11 / 12 already established that the cold-storage
architecture is statistically indistinguishable from naive on
ACC at 15-task PermutedMNIST. The fixed sparse path does not
change that headline (0.667 vs naive's 0.662 = +0.5 pp,
overlapping std bands). It does, however, restore sparse mode
as a *legitimate* memory-bounded alternative — at the same ACC
and the same memory footprint per active synapse, with the
expected ``O(n·k)`` strength density. So sparse remains a real
engineering option for future scaling work, not an architectural
dead-end. The Option B negative-results writeup stands; sparse
goes from "broken" to "viable but not differentiating" in the
follow-up article's mechanism section.

### Follow-ups (intentionally deferred)

- **Pairwise vs naive at 15 tasks.** Not run as part of this fix
  because the question being answered was "does the fix work?",
  not "does sparse beat naive?". The single-method run produces
  no pairwise table. If we want a clean comparison, re-run with
  ``--methods naive cs_full cs_full_sparse`` once.
- **Extended-sequence sparse.** Whether sparse at 30 tasks
  generalises like dense did is unknown. Not on the critical
  path for the negative-results writeup.
- **Trigger threshold semantics more broadly.** The audit's
  component #24 (quantile vs absolute threshold for candidate
  selection) is a related design choice; both ``should_fire`` and
  ``candidate_mask`` could be revisited together if/when sparse
  is the headline mechanism.

---

## [2026-05-23] Three audit-fix implementations: consistency-reward transform, external-reward wiring, multi-pass query consistency

Three implementation changes addressing audit items #11, #15, #16,
and #19 from the [post-experiment-11 audit entry below](#). Each
fix is gated behind explicit per-experiment opt-in so prior
results remain bit-exact reproducible. All three were committed
before experiment 12 was launched; this entry documents the
implementation choices, not the results (which will be logged
separately).

### Commit summary

| Commit | Audit items addressed | Scope |
|--------|-----------------------|-------|
| `e362cd1` | #16 | `ConsistencyReward` gains optional center / scale / clip parameters. |
| `7c337a1` | #15, #19, #11 | `SynapseAugmentedMLP` caches training-mode logits; `apply_hebbian_update` accepts `training_target=` and pushes per-batch accuracy into the mixer's `ExternalReward`. The multi-pass cold-storage retrieval query uses `synapse.buffer_average()` instead of the first forward's `f_base.mean`. |
| `fb57a5a` | (harness) | `experiments/12_audit_fixes.py`: five-method comparison (naive, cs_full, cs_full_sparse, cs_full_real_reward, cs_full_complete). |

### Fix 1 — `ConsistencyReward` center / scale / clip (`e362cd1`)

**Implementation:** Four new optional kwargs — `center`, `scale`,
`clip_min`, `clip_max` — with defaults `0.0`, `1.0`, `-inf`,
`+inf`. A `_transform(x)` helper applies
`clip((x - center) / scale, clip_min, clip_max)` and is also
applied to the first-call seeded `1.0` so the very first reward
sample obeys the same transform as subsequent ones. Defaults
preserve the pre-fix behaviour bit-exact.

**Audit-fix configuration** (used by `cs_full_real_reward` and
`cs_full_complete` in experiment 12): `center=0.95, scale=0.05,
clip_min=-1.0, clip_max=1.0`. The choice of `center=0.95` was
based on the saturation point observed in earlier diagnostics
(consistency hovered at ≈0.97 once the EMA warmed up);
`scale=0.05` re-stretches the active range to roughly [-1, +1];
the clip prevents runaway values at task switches.

**Why parameterised rather than hard-coded:** Existing experiments
03–11 should not change behaviour. The constructor defaults are
the identity transform, so any caller that does not opt in gets
the pre-fix output exactly.

**Tests:** 5 new tests in `tests/test_reward_components.py` covering
default identity, recentered output, clip behaviour, and
first-call transform.

### Fix 2 — External-reward wiring from per-batch accuracy (`7c337a1`)

**Implementation:**
- `SynapseAugmentedMLP.forward` caches `self._last_logits` (detached)
  when in training mode. Eval-mode forwards do not cache.
- `apply_hebbian_update` gains a new kwarg `training_target=None`.
  When supplied, **and** `self.reward_computer` is a `RewardMixer`
  with a non-None `external`, the method computes per-batch
  accuracy from `_last_logits` and `training_target`, and writes
  it to `self.reward_computer.external.value` before the mixer
  computes the blended reward.
- `_last_logits` is cleared at the end of `apply_hebbian_update`
  so that a stale logit batch is never re-used.

**Why the `isinstance(RewardMixer)` check:** The training_target
path is opt-in via the kwarg AND requires the canonical reward
stack. Callers that compose or subclass `RewardMixer` will not
trigger the auto-wire path — they retain manual control over
external. This is a robustness trade-off documented here for
future reference.

**Why per-batch accuracy as the external signal:** The audit
flagged that #15 was effectively `1.0` for the lifetime of every
prior experiment. Per-batch accuracy gives a 0–1 signal that
genuinely varies across the training trajectory and across task
boundaries (task switches drop it sharply). Other choices — loss,
loss delta, calibrated confidence — were considered; per-batch
accuracy was selected for being the most directly interpretable
and the one most likely to differ from `1.0` at task switches.

**Tests:** 5 new tests in `tests/test_synapse_finetune.py` covering
the logits cache, the training_target path, and the no-op
behaviour when `training_target` is omitted.

### Fix 3 — Multi-pass query consistency (`7c337a1`, second hunk)

**Implementation:** In `SynapseAugmentedMLP.features`, when
`self.n_passes > 1` the cold-storage retrieval query vector is
now built from `synapse.buffer_average()` (the average over all
N forwards' activations) instead of the first forward's
`f_base.mean(...)`. When `n_passes == 1` the original single-pass
path is preserved bit-exact.

**Why this matters:** Before the fix, the Hebbian update used the
denoised buffer-averaged activations while the retrieval query
used the noisy first-forward mean. The two parts of the same
mechanism were looking at different signals. The fix unifies them
so that "the activation the synapse layer learned from" equals
"the activation we used to query cold storage".

### What is opt-in vs always-on

- Fix 1 (consistency transform): off by default; opted into via
  constructor kwargs.
- Fix 2 (external-reward wiring): off by default; opted into by
  the caller passing `training_target=` to `apply_hebbian_update`.
- Fix 3 (multi-pass query): always-on when `n_passes > 1`. There
  is no opt-out because the single-pass code path is preserved
  unchanged and is the previous behaviour for that mode.

### Verification before launching experiment 12

A 3-task × 2-seed smoke run of `experiments/12_audit_fixes.py`
confirmed the mechanisms work at runtime:
- `cs_full_real_reward` per-batch reward std was ≈54× larger than
  `cs_full` (std 0.164 vs 0.003), and the reward range included
  negative values down to −0.604.
- `cs_full_sparse` end-of-run strength density was 0.250
  (= 64/256, matching `top_k=64`), vs 1.000 for dense.

Both fixes did what they were designed to do at the mechanism
level. Whether they translate into an ACC change is what
experiment 12 will answer.

### Test count

296 → 301 tests passing (+5 from the new test files).

---

## [2026-05-23] Post-experiment-11 audit: 48-component verification of the architecture

A structured read-only audit walked DESIGN.md and PROJECT_PLAN.md
component-by-component (48 items total: synapse layer, modulation,
reward components, consolidation, cold storage, runner, wrappers)
and cross-referenced each against the implementation as of commit
`9dec81f` (the experiment-11 architectural-call commit). This
entry records the audit's findings as facts about the codebase at
that point in time. It is intentionally separate from any
interpretation of how these findings relate to prior experimental
results — that interpretation, if warranted, will be logged after
experiment 12 reports.

### Component status table

Status legend:
- ✓ implemented and active in at least one experiment
- ✗ not implemented, or implemented but never activated
- ⚠ implemented and active but with a runtime deviation from spec

| # | Component (DESIGN / PLAN reference) | Status | Note |
|---|-------------------------------------|:------:|------|
| 2 | Per-synapse confidence (`SynapseLayer.confidence`) | ✗ | Populated mechanically by `consolidate()`; no consumer in the codebase. |
| 3 | Per-synapse age (`SynapseLayer.age`) | ✗ | Incremented in `consolidate()`; no consumer in the codebase. |
| 9 | Sparse top-k partner selection (DESIGN §3.2) | ✗ | `SynapseLayer(sparse=True, top_k=…)` implemented and unit-tested; every experiment 03–11 instantiated with `sparse=False`. |
| 10 | Top-k eviction logic | ✗ | Same as #9 — implementation present, never activated. |
| 11 | Multi-pass averaging (DESIGN §3.2, PLAN §4.2.1) | ⚠ | Buffer + `observe()` / `buffer_average()` present and exercised. Query path into cold storage in `SynapseAugmentedMLP.features` uses the first forward's `f_base.mean` rather than `buffer_average()`. |
| 15 | External reward (`ExternalReward`) | ⚠ | Class implemented; no caller updates `external.value` away from its `default=1.0` in any experiment script. Effectively constant at runtime. |
| 16 | Consistency reward (`ConsistencyReward`) | ⚠ | Saturates around 0.97 once the EMA warms up (centered on a `1.0` similarity floor; no transform applied). Output is near-constant per batch. |
| 19 | Reward mixing α(t) (`RewardMixer`) | ⚠ | Mechanism present and correct, but the two input streams (#15 and #16) are both near-constant at runtime, so the mixed signal is also near-constant. |
| 24 | Consolidation candidate selection | ⚠ | Implemented via `ConsolidationTrigger(candidate_quantile=…)` (a quantile, e.g. lowest-resistance 5%). DESIGN spec uses an absolute threshold. Sensible deviation; not previously recorded as such. |
| 25 | K-means clustering of candidates | ✗ | DESIGN specifies clustering; `consolidation/pipeline.py` documents one Chroma entry per cycle (no clustering) at lines 10–13. Deferred in source comment, not in this log until now. |
| 28 | Cold-storage metadata fidelity tracking | ⚠ | Compression tier and timestamp tracked; per-entry decompression fidelity is not tracked at runtime. |
| 35 | Decompression fidelity in retrieval | ⚠ | Decompression is correct; we don't measure how much fidelity loss accumulates as entries cycle through 32 → 16 → 8 → 4 bit tiers. |

The remaining 36 components verified clean against the spec.
(Other deviations such as the retrieval cache at `interval=20`
batches and Chroma HNSW determinism are noted in the source but
fall outside the audit's component list.)

### What the audit did not do

The audit did not re-run or re-interpret prior experiments. The
status entries above are observations about the codebase. Whether
any of these gaps materially affected experiment 11's headline
result (cs_full ≈ naive on ACC) is what experiment 12 is being
run to answer; that interpretation will be logged separately when
the results land.

### Why this is logged now

Several of these items have been informal knowledge since earlier
phases (k-means deferral, confidence/age unused) but were never
collected into one auditable place. The post-experiment-11 audit
is the first time they were verified together against current
code, and the count is large enough that a single entry is the
correct unit of record.

---

## [2026-05-23] Architectural completion Part 3: experiment 11 result and the final architectural call

### The five-method comparison (15 tasks Permuted-MNIST, dropout=0.5, 5 seeds)

| Method          | ACC          | FGT          | Note                          |
|-----------------|--------------|--------------|-------------------------------|
| naive           | 0.662 ± 0.018| 0.296 ± 0.019| control                       |
| cs_current      | 0.660 ± 0.017| 0.298 ± 0.020| Phase-4 cold storage          |
| cs_multi_pass   | 0.655 ± 0.008| 0.304 ± 0.008| + n_passes=5                  |
| cs_sweep        | 0.666 ± 0.016| 0.292 ± 0.018| + periodic compression sweep  |
| cs_full         | 0.664 ± 0.008| 0.294 ± 0.008| spec-complete (both)          |

Bonferroni-corrected Wilcoxon: **every pairwise p ≥ 0.625**. Even the
smallest p (cs_multi_pass vs cs_full, p_raw=0.0625, p_corr=0.625)
doesn't approach significance. All five methods are within 1.1 pp ACC
of each other; std bands overlap throughout the training trajectory
(see ``avg_accuracy_vs_tasks_seen.png``).

### Honest decomposition of the prior "cold storage" finding

How much of Phase 4b/4c's "no benefit" was an artifact of the
silent simplifications versus a real property of the architecture?

**Answer: very little of it was artifactual.** The spec-complete
``cs_full`` (multi-pass + compression sweep both on) matches the
spec-incomplete ``cs_current`` (matches what Phase 4b actually tested)
within 0.4 pp ACC and 0.4 pp FGT. The accuracy story is essentially
identical whether you run the mechanism as designed or as
accidentally simplified.

Per-mechanism contribution to ACC vs naive baseline:
- + multi-pass alone: **−0.7 pp** (cs_multi_pass = 0.655 vs naive 0.662)
- + compression sweep alone: **+0.4 pp** (cs_sweep = 0.666)
- + both (cs_full): **+0.2 pp** (cs_full = 0.664)

These are all inside one standard deviation. The architecture
delivers no measurable accuracy benefit even when implemented to
spec.

### What the mechanisms actually do (the real findings)

The architectural completion is not empty — the mechanisms are
working as designed. Two non-accuracy findings stand on their own:

**1. Compression sweep bounds memory at 3.7× lower footprint.**
The ``store_byte_size_per_task.png`` plot is the clearest result
of this whole session: cs_current and cs_multi_pass grow unbounded
to ~37 MB of stored documents by task 14 (linear in task count);
cs_sweep and cs_full stay flat at ~10 MB throughout. The schedule
moves entries through 32 → 16 → 8 → 4 bit tiers exactly as
designed (see ``precision_distribution_evolution.png``). This is a
real engineering contribution that the silent Phase-4 simplification
had been hiding.

**2. Multi-pass averaging reduces variance roughly 2×.** The std on
ACC drops from 0.018 (naive) to 0.008 (cs_multi_pass and cs_full).
The Phase 4b "variance reduction" claim that didn't generalise to
30 tasks for cs_current *does* hold when multi-pass is enabled —
because the variance source that multi-pass actually targets is
intra-sample dropout noise, which the dropout-0 Phase 4b setup
never exercised. With dropout=0.5 here, multi-pass picks up its
intended effect.

Neither of these accounts for the absence of an accuracy gain.

### The variance-reduction effect, revisited

Phase 4b reported a 2.2× std reduction at 15 tasks (dropout=0)
that did not generalise to 30 tasks. The current experiment
clarifies that finding: with multi-pass *on*, the variance
reduction is real and reproducible (0.018 → 0.008, a 2.25× ratio
that matches Phase 4b almost exactly). With multi-pass *off* but
cold storage on (cs_current here), no variance reduction occurs
(0.017 vs 0.018 naive). The 2.2× ratio in Phase 4b was therefore
likely an artifact of the seed-specific noise structure in that
exact 15-task run, not a property of cold storage. Multi-pass is
the actual mechanism for variance reduction.

### Architectural call

The session prompt offered two options:
- **Option A:** synapse + cold storage shows clear gain → proceed
  to Phase 5 contribution.
- **Option B:** confirms negative result on the actual architecture
  → pivot to negative-results writeup.

**The data calls Option B.** With the spec-complete architecture
implemented, tested at 15 tasks × 5 seeds × 5 methods with proper
multi-seed statistics:

- No method significantly differs from any other on ACC.
- The best variant (cs_sweep) beats naive by 0.4 pp ACC; not
  Bonferroni-significant and not large in absolute terms.
- The spec-complete architecture is statistically indistinguishable
  from the naive baseline.

This is the cleanest possible negative result: the mechanisms work,
the variance reduction is real, the memory bounding is real — but
the architectural hypothesis ("memory-inspired CL beats simple
sequential fine-tuning on standard benchmarks") is not supported by
the data.

### Why we did not run the 30-task follow-up

The Phase 4c 30-task run with the simplified architecture already
showed a flat-zero gap. The 15-task run with the spec-complete
architecture also shows a flat-zero gap. Two data points across
task counts and architecture variants both deliver the same answer.
A third (30 tasks × spec-complete) is unlikely to flip the call and
costs ~3 hours of compute. The decision is to instead allocate that
compute to the follow-up article and Phase 5 polish.

### What the follow-up article should say

The article narrative (per PROJECT_PLAN.md §10.3) becomes:

> *"We implemented a memory-inspired continual-learning architecture
> (an additive synapse layer with evidence-based resistance,
> context-dependent reward, sparse top-k partner selection, and a
> cold-storage archive with progressive quantization-based compression).
> We tested it on Split-MNIST at 5 tasks and Permuted-MNIST at 15
> and 30 tasks against sequential fine-tuning and EWC, with 5 seeds
> per condition and pairwise Wilcoxon signed-rank testing with
> Bonferroni correction. The architecture's accuracy is
> statistically indistinguishable from naive sequential fine-tuning
> across all conditions tested. Two non-accuracy findings stand:
> the progressive compression schedule bounds storage at 3.7× lower
> memory footprint than the uncompressed variant at equivalent
> accuracy, and multi-pass observation reduces seed-to-seed variance
> by about 2× when the base model has stochastic forwards (dropout).
> A meta-finding from the engineering process is worth recording:
> two specified mechanisms (multi-pass averaging and compression
> re-evaluation) were silently simplified in v1 implementations and
> only surfaced by a structured read-only code audit late in the
> project. We document both for future reference and as a cautionary
> tale about spec-to-code traceability in research engineering."*

### Concrete next steps (Phase 5 reframed)

1. **Polish the public-facing repo** (README, docstrings, two
   walkthrough notebooks) — Phase 6 work moved up.
2. **Write the article** with the narrative above.
3. **Optional: one more bench (Split-CIFAR-10)** to broaden the
   "no benefit" claim before publishing. ~1 day of work.
4. **Demo video** showing the implementation clearly, especially
   the compression-bounding and variance-reduction effects which
   are real engineering contributions.

The negative result is now the headline; the engineering
contributions sit alongside it as supporting findings.

---

## [2026-05-23] Architectural completion Part 1: multi-pass implemented (and the silent simplification it closes)

### Meta-finding: the silent Phase-2 simplification

A read-only audit on 2026-05-23 surfaced a discrepancy between
DESIGN.md / PROJECT_PLAN.md and the implementation: the multi-pass
co-activation mechanism described in DESIGN.md §3.2 and pseudo-coded
in PROJECT_PLAN.md §4.2.1 (lines 129–149 with ``activation_buffer``,
``observe()``, ``consolidate()`` averaging) was **never built**.
Phase-2 v1 (``synapse_layer/layer.py``, commit 92cefdd, 2026-05-23
session 4) silently substituted single-pass batch-mean — outer
products were computed from one forward of one batch, never from
multiple forwards averaged together. The simplification was not
recorded as a deferral; subsequent phases all built on top of it
without noticing.

This is a real failure mode for spec-to-code traceability. Every
other deferred item from DESIGN.md (confidence, sparse top-k,
k-means clustering, etc.) was logged as a deferral in this very
file. Multi-pass was not. The cause was likely that single-pass
batch-mean and multi-pass averaging look superficially similar in
the v1 dense-MLP setting (both reduce noise across samples) and
the gap was easy to miss without re-reading the spec carefully.

The audit also flagged a second silent simplification — the
compression schedule never re-fires post-insert, so every cold-
storage entry stays at 32-bit forever. That one is addressed in
Part 2 of this architectural-completion work; see the next
entry below.

### What this means for Phase 4b and 4c results

The "cold storage shows no benefit" and "the gap is flat at zero"
findings from Phase 4b/4c were collected with the *single-pass*
synapse layer — i.e. an architecturally incomplete system. The
finding still stands as a fact about that exact configuration,
but it does not yet falsify the architectural hypothesis,
because the design called for multi-pass.

Experiment 11 (next session's compute) re-runs the same
benchmarks with the spec-complete architecture (multi-pass +
compression sweep) so we can quantify how much of the prior
result was an artifact of the simplifications versus the
underlying mechanism.

### Multi-pass design decisions

**SynapseLayer dual-mode consolidate.** The new
``consolidate(activations=None, reward=1.0)`` keeps the Phase-3
explicit-activations path bit-exact when the buffer is empty;
when the buffer is non-empty, the activations argument is
forbidden and the buffer is averaged via ``stack(...).mean(0)``
before the outer products are computed. Two paths, one method,
no behavioural drift for existing callers. ``observe()`` is
the new entry point; ``buffer_size``, ``buffer_average()``, and
``clear_buffer()`` round out the public surface for callers that
want to inspect or reset state.

**``n_passes`` on the layer is informational.** The SynapseLayer
itself averages whatever the buffer holds — it does not enforce
that exactly ``n_passes`` observations were made. The ``n_passes``
field is mainly documentation for callers and shows up in
``extra_repr``. The wrapper (SynapseAugmentedMLP) is where the
operational semantics live.

**Multi-pass is training-mode-only.** In
``SynapseAugmentedMLP.features``, the extra forwards run only
when ``self.training and self.n_passes > 1``. During evaluation
the buffer stays empty so the runner's repeated ``forward(x)``
calls in ``_evaluate`` never pollute synapse state. This is
both correctness (eval should not change the model) and a
performance optimisation.

**The first forward is the loss-contributing one.** When N>1
forwards happen, the first call's ``f_base`` is the one that
becomes part of the modulated output and reaches the classifier
head; the additional N-1 forwards are observation-only re-runs
of ``base.features(x)``. This keeps autograd clean — only one
forward graph contributes to backward.

**Deterministic vs stochastic forwards.** For a fully
deterministic forward (no dropout), all N passes give identical
activations and ``stack(N copies).mean()`` equals any single
copy. So ``n_passes > 1`` on a deterministic MLP is provably a
no-op. The benefit only materialises when forwards are stochastic
— enabling dropout in the base MLPConfig is the natural switch.
Experiment 11 will exercise the stochastic variant explicitly.

**Backward compatibility.** Every Phase-3 test still passes, and a
new dedicated test (``test_consolidate_single_pass_unchanged_when_buffer_empty``)
locks the buffer-empty path bit-exact equivalent to the legacy
``consolidate(activations, reward)`` API.

---

## [2026-05-23] Phase 4c: 30-task extended sequence — the gap is flat at zero

To check whether the synapse + cold-storage gap to naive baseline
was stable, narrowing, or widening with more tasks, the
experiment from Phase 4b was re-run with 30 Permuted-MNIST tasks
(naive and synapse_full_cold_storage only — 5 seeds, same
hyperparameters, same architecture).

### 30-task numbers (5 seeds, Permuted-MNIST shared head)

| Method                          | ACC          | FGT          |
|---------------------------------|--------------|--------------|
| naive                           | 0.276 ± 0.015| 0.705 ± 0.015|
| synapse_full_cold_storage       | 0.277 ± 0.018| 0.704 ± 0.019|

Wilcoxon signed-rank: p_corr = 0.8125 (no significance).
Difference in mean ACC: 0.001. Difference in std: 0.003.

Cold-storage consolidations per seed: 405–461 (≈ 15 cycles per
task average). Store growth scaled roughly linearly with task
count.

### Answer to the headline question

**The gap is flat, neither narrowing nor widening.** At every
training step from 1 to 30, the cold_storage − naive average-
accuracy gap sits in [-0.02, +0.01] with mean very close to zero.
See ``results/figures/extended_sequence/gap_vs_tasks_seen.png``.

### What this overturns from Phase 4b

The Phase 4b architectural call leaned heavily on **two**
secondary findings since the headline ACC was already a near-tie:

1. *Cold storage repairs the synapse regression* (+7.5 pp vs
   synapse_full alone). This is unchanged — synapse_full alone
   was not re-run at 30 tasks, but there's no reason to expect
   the +7.5 pp recovery effect changed.

2. *Cold storage reduces variance 8×* (std 0.092 → 0.011 at 15
   tasks). **This effect did NOT generalise.** At 30 tasks:

   | Tasks | naive std | cold-storage std | ratio |
   |-------|-----------|------------------|-------|
   | 15    | 0.024     | 0.011            | 2.2×  |
   | 30    | 0.015     | 0.018            | 0.83× |

   The "variance reduction" was a 15-task observation that
   reversed by 30 tasks. The Phase-4b std numbers may have
   been a seed-specific quirk or a too-early measurement; with
   the working set saturated for longer the cold-storage variant
   is no more stable than naive (slightly less, in fact).

### Implication for Phase 5 framing

Phase 4b's architectural call was "proceed to Phase 5 with the
contribution explicitly reframed around robustness and variance
reduction." That framing now has thin empirical support:

- **ACC ≈ naive** at both 15 and 30 tasks. No absolute benefit.
- **Variance reduction did not generalise** beyond 15 tasks.
- **The synapse-repair contribution is real** but it just brings
  the architecture *to* naive parity, not past it.
- **Runtime cost is 7× naive** (1547 s vs 219 s at 30 tasks)
  with no measurable accuracy benefit.

The honest reading: the cold-storage variant is **expensive
parity** with the naive baseline on Permuted-MNIST, with no
generalising stability benefit. Phase 4b's optimism about
"graceful degradation profile" was driven mostly by EWC's
catastrophic collapse (a hyperparameter artifact); against the
naive baseline alone, there is no distinguishing positive
finding to report.

This sharpens the Phase 5 question: rather than "proceed with
reframed contribution", the more honest call is:

- **Option B-revised (pivot to negative-results writeup) is now
  the better call** unless the next session uncovers a stress
  condition where the cold-storage variant clearly wins.

Two concrete next experiments would be decisive:

1. **EWC λ-sweep on Permuted-MNIST 15 and 30 tasks.** Find the λ
   where EWC stays well-tuned; compare to synapse + cold storage
   at that point. If EWC tuned well decisively beats us, the
   "robust vs EWC" story collapses entirely.

2. **A stress condition designed to favour episodic memory.**
   e.g., task interleaving (revisit task 1 after task 10 and
   measure recall), or distribution shift mid-stream. If cold
   storage's context-dependent retrieval helps anywhere, it
   should help here.

If both of those come back neutral or negative, the writeup
pivots: "We implemented a memory-inspired continual-learning
architecture (synapse layer + cold storage), tested it
rigorously on Permuted-MNIST at 15 and 30 tasks against naive
sequential fine-tuning, and found no measurable benefit. The
implementation is a clean reference for future work but the
architectural hypothesis was not supported by the data."

### Status: architectural call now leans Option B

Pending the two follow-up probes above. If either reveals a
defensible positive condition, Option A returns.

---

## [2026-05-23] Phase 4b: long-sequence decisive test + architectural call

This is the test the entire project was building toward: 15-task
Permuted-MNIST with a shared 10-class head, the regime where the
synapse layer's working memory genuinely saturates and cold-storage
retrieval has real opportunity to make a difference.

### Headline numbers (5 seeds, Permuted-MNIST, 15 tasks)

| Method                          | ACC          | FGT          |
|---------------------------------|--------------|--------------|
| naive                           | 0.475 ± 0.024| 0.518 ± 0.025|
| ewc (λ=1000)                    | 0.098 ± 0.000| 0.346 ± 0.036|
| synapse_full (no cold storage)  | 0.391 ± 0.092| 0.584 ± 0.060|
| **synapse_full_cold_storage**   | **0.466 ± 0.011**| 0.527 ± 0.012|

Bonferroni-corrected pairwise Wilcoxon: at n=5, nothing crosses
α=0.05 (floor p_corr=0.375). All readings below are point-
estimate observations confirmed by the trajectory plots in
``results/figures/long_sequence/``.

### What the data says

**1. Cold storage repairs synapse_full's regression.**
synapse_full alone: 0.391 ACC. Adding cold storage brings it to
0.466 — a +7.5 pp recovery. Without cold storage the synapse
architecture is *worse* than naive on this benchmark; with cold
storage it is statistically indistinguishable from naive.

**2. Cold storage stabilises variance dramatically.**
synapse_full std: 0.092. With cold storage: 0.011. An 8× reduction.
The trajectory plots make this visible — the red band (cold
storage) is tight; the green band (no cold storage) sprawls across
nearly the full range. This is the most unambiguous positive
contribution of the cold-storage architecture.

**3. EWC catastrophically collapses on this benchmark.**
ACC 0.098 — below the 0.10 random-guess baseline for 10-class
MNIST. The Task-1 plot shows the failure mode clearly: EWC
preserves Task 1 *perfectly* at 0.93 for the first 5 tasks, then
falls off a cliff to ~0.10 by task 7 and stays there. λ=1000 is
too aggressive for 15 sequential tasks on a shared head; the
quadratic penalty over-constrains the model until it freezes in a
configuration that cannot learn the new task either. Note this is
a hyperparameter artifact, not an issue with our EWC implementation
— a properly tuned λ (likely much smaller) would not collapse.

**4. No method significantly outperforms the naive baseline.**
naive 0.475 vs synapse_full_cold_storage 0.466 — within one
standard deviation either way. Permuted-MNIST 15-task shared head
is a hard benchmark; sequential fine-tuning's "forget gracefully"
strategy is already a strong baseline.

**5. The cold-storage variant has the most graceful degradation
profile.** Both the Task-1 trajectory and the running-average plot
show synapse_full_cold_storage and naive degrading on essentially
identical curves, with cold storage holding a slightly tighter
band. EWC has the cliff; synapse_full has the noisy descent. The
qualitative "graceful long-sequence robustness" claim from the
project's framing is supported, though the absolute accuracy
claim is not.

### Architectural decision

The session prompt offered two options:

- **Option A:** synapse shows a clear gain → proceed to Phase 5.
- **Option B:** all methods degrade similarly → pivot to negative-
  results writeup.

The data fits *neither* cleanly. The honest reading: synapse +
cold storage delivers a *real but limited* contribution — it
**stabilises** the synapse architecture and gives it **graceful
long-sequence robustness**, but it does not deliver an absolute
accuracy gain over the naive baseline on Permuted-MNIST.

**Call: proceed to Phase 5 with the contribution explicitly
reframed.**

Reasons in favour:

- Cold storage achieves the milestone the session prompt called
  "a defensible positive result": *"Cold storage maintaining
  better long-term accuracy than synapse_full alone (but still
  under EWC)"*. Here it is better than synapse_full alone and
  EWC has collapsed, so technically it is above EWC by a wide
  margin — though for the wrong reason (EWC's λ failure).
- The variance-reduction effect (8×) is dramatic, unambiguous,
  and unique to the cold-storage variant in this benchmark.
- The qualitatively different degradation profile vs EWC's
  cliff is a real finding about the robustness of memory-
  inspired CL architectures.

Reasons against (and the caveats to carry into Phase 5):

- We do not beat the naive baseline absolutely. The
  publication-grade headline number is not "synapse beats X" but
  "synapse + cold storage matches naive while being more stable
  and more graceful."
- EWC's collapse is a hyperparameter artifact. A Phase-5 EWC
  λ-sweep is mandatory before any "robustness vs EWC" claim is
  defensible.
- The implementation cost is real: cold-storage runtime is ~8×
  naive (761 s vs 93 s for 5 seeds at 15 tasks). The retrieval
  cache helped (3.7× speedup over uncached) but Chroma overhead
  still dominates.

### Phase 5 framing (the reframing)

The next phase should *not* be "rigorous comparison to existing
methods on standard benchmarks". Phase 4b shows the synapse
architecture does not win that comparison.

Instead, Phase 5 should investigate the contributions the cold-
storage variant *does* deliver:

1. **Robustness under stress.** Long sequences, distribution
   shift, mid-stream task injection. Quantify the conditions
   under which synapse + cold storage outperforms regularization-
   based methods that hit collapse modes.
2. **Variance characterisation.** The 8× std reduction is
   striking. Investigate whether it holds across benchmarks
   and seeds, and what causes it (retrieval-induced smoothing?
   pressure-trigger noise dampening?).
3. **EWC λ-sweep.** Find the collapse boundary for EWC on each
   benchmark and compare to synapse + cold storage at the
   collapse point. If EWC has a narrow viable λ band and
   synapse + cold storage covers a broader regime, that's a
   real story.
4. **Memory footprint trade-offs.** Cold storage entry count vs
   accuracy. Compression schedule's effect on long-term recall.
5. **Honest reporting.** The follow-up article should lead with
   the robustness contribution, not with a claim of beating
   baselines.

### What's deferred to Phase 5

- Multi-benchmark evaluation (Split-CIFAR, Permuted-MNIST,
  Split-AG-News).
- EWC λ-sweep for each benchmark.
- Higher-seed runs (10+) for stronger statistical power.
- Reward-signal investigation (still open from Phase 3).
- DistilBERT integration (still open from Phase 3.5).
- Phase 6 polish: notebooks, demo video, follow-up article.

The plots in ``results/figures/long_sequence/`` and the full
JSON log in ``results/logs/long_sequence/`` are the artifacts
the Phase 5 evaluation should build on.

---

## [2026-05-23] Phase 4 (partial): cold storage architecture + short-benchmark result

This session implemented the full Phase-4 architecture — cold
storage, compression pipeline, pressure-based consolidation
trigger, consolidation pipeline (synapse → store), reconstructive
retrieval (store → synapse), and integration into
``SynapseAugmentedMLP`` — plus the 5-task multi-head experiment 07
that the project plan flagged as the short-benchmark test. The
decisive 15+ task long-sequence experiment (08) is deferred to
the next session per PROJECT_PLAN.md §7 / Phase 4 priorities.

The architectural call (proceed to Phase 5 vs pivot to negative
writeup) **remains deferred** until experiment 08 results are in.

### Cold-storage backend: Chroma in-memory, document-as-bytes

**Decision:** Use ``chromadb.Client()`` (in-memory, no server)
as the vector store. Per-entry storage:

- ``embedding``: activation pattern at consolidation time (used
  for retrieval).
- ``document``: base-64-encoded compressed strengths matrix.
- ``metadata``: precision, n_neurons, age, access_count,
  created_at_step, num_candidates.

**Rationale:** Chroma's metadata fields don't accept raw bytes,
so the compressed strengths live in ``document`` (Chroma's free-
form string slot) as base-64. This keeps the schema declarative
without a sidecar store.

### Compression schedule: 32 → 16 → 8 → 4 bit by age, bumped by access

**Decision:** ``CompressionSchedule(age_thresholds=(100, 500,
2000), tier_precisions=(32, 16, 8, 4), access_count_floor=5)``.
Quantisation is symmetric per-tensor max-abs; 4-bit values are
packed two per byte and sign-extended on dequantise.

**Trade-off:** 4-bit halves storage from 8-bit but the per-element
quantisation error is up to ~14 % of the value range. Acceptable
because the archive holds gist, not detail — and any entry we
care about gets bumped up a tier via ``access_count``.

### Pressure metric: avg-pressure trigger with refractory + top-quantile candidates

**Decision:** Mean of ``|s_ij| · e_ij / (1 + a_ij)`` across the
synapse matrix, with a configurable threshold. ``min_steps_between``
adds a refractory period; ``candidate_quantile`` selects which
synapses get archived once the cycle fires.

**Tuning observation:** With our 256×256 synapse matrix on
Split-MNIST, mean pressure ends at ~0.013 even after a full run.
The PROJECT_PLAN.md default of "around 0.1" produced zero
consolidations. Threshold 0.005 fires roughly every 30 batches.
The right threshold is dataset- and capacity-dependent; a more
principled auto-calibration (e.g., percentile-based or relative
to long-running mean) is open work for Phase 5.

### Consolidation pipeline: one entry per cycle, no k-means in v1

**Decision:** Each fired cycle archives a single Chroma entry
containing the candidate synapses' strengths (non-candidates
zeroed) at the fresh-precision tier (32-bit). K-means clustering
of candidates (DESIGN.md §3.5) is deferred to a refinement — one
entry per cycle keeps the data path tractable for the first
working version.

**Drain semantics:** strength, evidence, access_count are zeroed
on the candidate positions; age and confidence are preserved. The
spec says "reset strength, evidence, access counters" and we
follow it literally — age/confidence track the *cell's* lifecycle,
not the *pattern's*.

### Reconstructive retrieval: context-dependent, similarity-weighted

**Decision:** Forward path consults cold storage with the current
batch's mean activation as the query. Top-k retrieved entries are
decompressed and averaged with weights ``1 / (1 + distance)``;
the resulting strengths matrix is added to the synapse's working
matrix before the modulator applies its gate.

**Why context-dependent matters:** Phase 3.5 surfaced the
hypothesis that the synapse layer's universal correction matrix
conflicts with multi-head's per-task representations. With cold
storage, different tasks pull different slices of the archive
because they produce different activation patterns. The same
modulator now produces task-conditioned corrections without the
synapse layer itself needing per-task heads.

**Side-effect:** Each retrieval bumps the retrieved entries'
``access_count`` metadata, which feeds back into both the
compression schedule (frequently-accessed entries stay sharper)
and the pressure metric (frequently-accessed entries are less
likely to be re-consolidated).

### Backward-compatible integration

**Decision:** ``SynapseAugmentedMLP`` accepts optional
``cold_storage``, ``consolidation_trigger``, and ``retrieval_k``.
When ``cold_storage=None`` the model reproduces Phase-3 behaviour
bit-for-bit (verified by 15 unchanged tests).

### Phase 4 numbers, honestly (n=5 seeds, multi-head Split-MNIST)

| Method                          | ACC          | FGT          |
|---------------------------------|--------------|--------------|
| naive (multi-head)              | 0.833 ± 0.055| 0.197 ± 0.068|
| ewc (multi-head, λ=1000)        | 0.956 ± 0.020| 0.041 ± 0.023|
| synapse_full (multi-head)       | 0.810 ± 0.065| 0.227 ± 0.081|
| synapse_full_cold_storage       | 0.812 ± 0.066| 0.224 ± 0.083|

The cold-storage variant moves ACC by 0.2 pp and FGT by 0.3 pp
on this 5-task benchmark — well inside one std and Bonferroni-
corrected p of 1.0 vs synapse_full. **Cold storage shows no
measurable benefit on 5-task Split-MNIST.**

This matches the expected framing: with only 5 tasks, the synapse
layer's working set (256² = 65 k synapses) never saturates. The
trigger fires (~30 batch intervals at threshold 0.005), entries
accumulate in cold storage, retrieval queries succeed — but the
retrieved patterns don't carry information the working set has
already lost, so the augmentation is a noisy no-op.

### What's next: experiment 08 (long sequence)

The decisive test for the project hypothesis is a 15-20 task
sequence where capacity saturation forces real reliance on the
archive. The Phase-4 v1 architecture is in place to run that test;
the next session will:

1. Build a Permuted-MNIST (or Rotated-MNIST) benchmark with 15+
   tasks.
2. Re-run the same four methods at 5 seeds.
3. Report per-task accuracy trajectories, memory footprint over
   time, and consolidation-cycle counts in addition to ACC/FGT.
4. Make the architectural call (proceed to Phase 5 vs pivot to
   negative writeup) with the full data.

### Deferred to future sessions

- Experiment 08 (long-sequence benchmark) — next session.
- K-means clustering of consolidation candidates — refinement.
- Auto-tuned pressure threshold (percentile-based or relative) —
  removes a hyperparameter that was already painful to set.
- Reward-signal investigation (consistency EMA tuning) — still
  open from Phase 3.
- DistilBERT integration + Split-AG-News — still open from
  Phase 3.5.

---

## [2026-05-23] Phase 3.5: multi-head probe and deferred architectural call

This exploratory session tested whether the synapse architecture
performs better when the shared-head bottleneck is removed. The
DistilBERT/Split-AG-News probe was scoped out; the multi-head
result was decisive enough that adding another data point would
not change the conclusion (see "Why we skipped DistilBERT" below).

### Multi-head Split-MNIST numbers (5 seeds, MultiHeadMLPClassifier)

For reference, the Phase-3 shared-head numbers are repeated in the
first row of the comparison table.

| Method                     | ACC          | FGT          |
|----------------------------|--------------|--------------|
| naive (single-head)        | 0.600 ± 0.004| 0.488 ± 0.005|
| naive (multi-head)         | 0.833 ± 0.055| 0.197 ± 0.068|
| ewc (multi-head, λ=1000)   | 0.956 ± 0.020| 0.041 ± 0.023|
| synapse_resistance (m-h)   | 0.810 ± 0.067| 0.227 ± 0.083|
| synapse_full (m-h)         | 0.810 ± 0.065| 0.227 ± 0.081|
| synapse_full_sparse (m-h)  | 0.800 ± 0.066| 0.239 ± 0.083|

### What the data says

1. **The shared-head bottleneck was real and large.** Going
   multi-head adds +23 pp ACC to the naive baseline by itself.
   This confirms the Phase-3 hypothesis: with a shared 2-class
   head, the head's softmax boundary was the main forgetting
   surface, and the synapse correction at the penultimate layer
   could not compensate for it.

2. **EWC under multi-head reaches 95.6 % ACC**, near published
   Split-MNIST EWC numbers. The combination is well-matched:
   inactive heads have zero Fisher, the active backbone has
   strong Fisher, and the quadratic penalty pins the trunk while
   each head specialises.

3. **The synapse layer does not benefit from removing the
   bottleneck.** synapse_resistance and synapse_full sit at
   0.810 — slightly *below* naive 0.833. The synapse correction
   is shared across all tasks (one strengths matrix), and in
   multi-head the dominant continual-learning mechanism IS the
   per-task head; the global synapse correction adds noise rather
   than signal.

4. **Sparse top-k still doesn't hurt much.** Within the synapse
   family, synapse_full vs synapse_full_sparse differs by 1 pp
   ACC (well inside one std). The memory benefit is intact.

5. **Bonferroni-corrected significance.** As in experiment 05,
   n=5 puts the floor at p ≥ 0.0625 × 10 = 0.625, so nothing
   crosses α=0.05. But the point-estimate gaps are large
   (EWC – synapse_full ≈ 14.6 pp ACC) and the std bands of
   naive/synapse vs EWC barely overlap, so the qualitative
   picture is unambiguous even before significance is reached.

### Why we skipped DistilBERT

The Phase-3.5 prompt called for a quick DistilBERT/Split-AG-News
probe as a second favourable condition. We chose to skip it for
this session because:

- Installing ``transformers`` and tokenizers is ~500 MB of new
  dependencies and ~20 min of CPU runtime even for a tiny probe.
- The multi-head result is already decisive on the synapse-alone
  question: even with the bottleneck removed, the synapse layer
  *regresses* below naive. A DistilBERT probe is likely
  confirmatory rather than flipping. Asymmetric value would not
  justify the cost here.
- The architectural question that really matters (synapse + cold
  storage as a coordinated long-horizon memory system) is *not*
  what DistilBERT on a 2-task AG-News subset would test.

The DistilBERT probe is deferred to a future Phase-3.6 session if
it ever becomes the bottleneck for the architectural call.

### Architectural call (Option A vs Option B) is DEFERRED

Original framing in this session:
- **Option A:** synapse shows clear gain in either favourable
  setting → proceed to Phase 4.
- **Option B:** synapse remains neutral → pivot to negative-
  results writeup.

Neither is appropriate yet, because **both options misframe what
the project hypothesis actually is.** The hypothesis is not
"synapse layer beats EWC on standard CL benchmarks". The
hypothesis from [DESIGN.md §3.1](DESIGN.md) is "synapse layer +
cold-storage layer as a coordinated long-horizon memory system
beats sequential fine-tuning". On a 5-task Split-MNIST sequence,
synapse capacity is nowhere near strained — there is nothing for
the cold storage to do.

The decisive test is therefore Phase 4 with a long task sequence
(10+ tasks). That is the regime where:
- Synapse strengths saturate and the pressure metric matters.
- Old patterns leave the synapse layer's working set and need to
  be reconstructed from cold storage.
- The full architecture — observe → consolidate → store →
  retrieve — is exercised.

Until that test runs, "synapse_full is below naive on multi-head
Split-MNIST" is a real, honest data point but not yet evidence
that the full architecture fails.

### Next session: Phase 4 with long-sequence experiment

Phase 4 was scoped in PROJECT_PLAN.md §7 as cold-storage
implementation. We're adding a specific deliverable: an experiment
that runs the full system on a 10+-task sequence and measures
whether long-term retrieval recovers performance on tasks that
have left the synapse layer's working set.

The architectural call (A or B) gets made after that experiment,
with the multi-head finding above as one input to the decision
rather than the only input.

---

## [2026-05-23] Phase 3 close-out: state schema, β normalisation, sparse top-k, multi-seed

This session closed the remaining Phase-3 deliverables: the full
state schema (confidence, age, access_count), β calibration via
normalised evidence, sparse top-k partner selection, multi-seed
runner + statistical-significance helpers, and the headline
experiment 05 comparison. The reward signal was *not* touched this
session per the explicit non-goal.

### State-field semantics (confidence, age, access_count)

**Decision:** Populate the three new state fields mechanically but
do *not* feed them back into the update rule yet.

**Update logic chosen:**
- ``confidence[i, j] += min(prev_abs_outer[i, j], curr_abs_outer[i, j])``
  per call after the first. Rewards co-firing that is *sustained*
  across consecutive batches; a pair that flickers gets credit only
  for the weaker of its two batches.
- ``age[i, j]`` ticks +1 on every consolidate.
- ``access_count[i, j]`` ticks +1 when
  ``mean_b(|features[b, i]|) · |strengths[i, j]| > threshold``
  (default 1e-3), recorded via a new ``record_access`` method that
  the augmented MLP calls before each consolidate.

**Rationale for not yet using them in the update rule:** the
Phase-3 spec only requires "populate them mechanically". Adding
them to the rule prematurely would entangle independent variables
and make the Phase-4 pressure metric harder to attribute. By the
end of this session, the buffers exist and evolve correctly; future
sessions can plug them in cleanly.

### β calibration: normalise evidence by current max

**Decision:** Resistance now uses
``evidence / (max(evidence) + ε)`` instead of raw evidence, so
β has the same effective meaning regardless of how large evidence
grows on the chosen benchmark.

**Alternative considered:** schedule β decay based on observed
evidence scale. Rejected as more knobs to tune; the normalisation
approach has the same effect with one fewer hyperparameter.

**Backward-compat remap:** the old un-normalised β=0.01 on
Split-MNIST mapped to ``1 / (1 + 0.01 · 2800) ≈ 0.034`` pass-through
on the most-evidenced synapse. Under the new math, β≈28 reproduces
that exactly. We picked β=10 as the new default (~9% pass-through),
giving milder dampening. Single-seed verification confirmed the
numbers stay in the same band (ACC 0.608/FGT 0.479 at β=10 vs the
old β=0.01 single-seed 0.611/0.478).

**Edge case noted:** the very first consolidate sees
``evidence == 0`` everywhere, so ``max(evidence) = 0`` would divide
by zero. Handled by ``clamp_min(ε)``.

### Sparse top-k: dense buffers + zero-mask strategy

**Decision:** Keep dense ``(n, n)`` buffers in SynapseLayer and
*zero* entries outside the top-k mask after each consolidate. Top-k
is opt-in via ``sparse=False`` default + ``top_k=64``.

**Alternatives considered:**
- True sparse representation (e.g., COO indices for the top-k
  per-row partners). Lower memory but heavy bookkeeping for the
  per-batch eviction; the cost-benefit at MLP scale (n=256, k=64)
  is unfavourable.
- Pure dense always. Memory grows to O(n²) — fine at n=256 but
  prohibitive at transformer scale (n=768 means 590 k synapses).

**Rationale:** The zero-mask strategy delivers the user-visible
top-k semantics without changing the storage layout. At transformer
scale we may revisit, but for Phase 3 / 4 development this is the
right trade. The dense buffers also make state inspection in tests
and notebooks trivial.

**Eviction rule:** the mask is computed from *post-update* strengths.
A synapse that crosses above the previous weakest's |strength|
displaces it; all five state fields (strengths, evidence,
confidence, age, access_count) plus the prev_abs_outer cache are
zeroed for evicted positions, so a position that returns to the
top-k later starts with a fresh slate.

### Multi-seed and significance protocol

**Decision:** Multi-seed runs via a factory-based ``run_multi_seed``
helper; Wilcoxon signed-rank with Bonferroni correction across the
``k choose 2`` method pairs.

**Choices made:**
- Factory takes ``seed`` and returns ``(model, runner)``. The
  factory owns all seed-dependent setup (model init, EWC instance,
  synapse layer, reward computer). The benchmark is reused across
  seeds because real benchmarks (SplitMNIST) are deterministic.
- Wilcoxon is paired per-seed — the same seed list must be used
  across methods. Enforced implicitly by the factory pattern.
- Bonferroni multiplies p-values by the number of pairs (``k choose 2``)
  and clips to 1.0. Less powerful than Holm-Bonferroni but simpler
  and conservative.
- scipy is imported lazily inside ``pairwise_wilcoxon`` so the rest
  of the statistics module remains importable without scipy. CI
  workflow gained ``scipy==1.16.3`` in the install step.

### Phase 3 headline numbers (5 seeds, Split-MNIST)

| Method                | ACC            | FGT            |
|-----------------------|----------------|----------------|
| naive                 | 0.600 ± 0.004  | 0.488 ± 0.005  |
| ewc (λ=1000)          | 0.623 ± 0.017  | 0.458 ± 0.023  |
| synapse_resistance    | 0.597 ± 0.010  | 0.493 ± 0.011  |
| synapse_full          | 0.598 ± 0.010  | 0.492 ± 0.011  |
| synapse_full_sparse   | 0.596 ± 0.009  | 0.495 ± 0.010  |

**After Bonferroni correction, no pairwise comparison reaches
α=0.05 significance.** The minimum raw p-value the Wilcoxon
signed-rank test can produce at n=5 is 0.0625; multiplied by 10
pairs that's 0.625. We are statistically blind at this seed count.
Pre-correction, the EWC-vs-naive comparison sits at p=0.125 (not
significant either).

**Honest assessment.**

1. The Phase-3 synapse variants are statistically indistinguishable
   from naive on Split-MNIST. The single-seed +0.9 pp ACC gain
   reported at the end of the previous Phase-3 session was noise —
   the 5-seed mean for synapse_resistance is 0.597, *below* naive's
   0.600.
2. EWC continues to dominate at +2.3 pp ACC over naive (mean), but
   even this is not Bonferroni-significant at 5 seeds.
3. Sparse top-k (k=64) does not measurably hurt the synapse system
   (0.598 → 0.596 ACC, within noise). The expected memory benefit
   is real for transformer-scale use.

**Why the synapse layer fails to help here:**

- Split-MNIST with a *shared* 2-class head puts the head squarely
  in the catastrophic-forgetting path. The synapse correction at
  the penultimate layer can't compensate when the head's softmax
  boundary is the bottleneck.
- The reward signal really is too weak as currently configured —
  consistency rarely drops below 0.97 across task switches because
  the EMA tracks too fast. The deferred reward-signal investigation
  is genuinely needed.
- Without confidence/age/access_count being *consumed* (just
  populated), the resistance mechanism leans on a single field
  (normalised evidence). The richer state schema is there for
  Phase 4 to use.

**What this means for the project narrative:**

- The Phase-3 deliverables are met: full state schema, normalised
  resistance, sparse top-k, multi-seed protocol all working and
  tested.
- The Phase-2 failure mode ("synapse just tracks the latest task")
  is fixed — it no longer regresses 2.6 pp. It's now neutral.
- The Phase-3 success criterion ("beats baseline") is NOT met. We
  recorded this honestly so the project's narrative reflects
  reality. Phase 4 (cold storage) is where we either find a real
  benefit or write the negative-results follow-up article called
  out in PROJECT_PLAN.md §10.3.

### What's still deferred

Carried forward to future sessions:

- **Reward signal investigation.** Consistency EMA tuning was an
  explicit non-goal for this session.
- **DistilBERT integration** via ActivationCapture and **Split-AG-News
  benchmark** — Phase-2 tasks that may shift the picture if the
  shared-head bottleneck is the real culprit.
- **Phase 3 walkthrough notebook** — Phase 6 polish.
- **Consume confidence/age/access_count in the update rule** —
  Phase 4 will use them in the pressure metric, but a more
  sophisticated update rule that uses them directly is open.

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
