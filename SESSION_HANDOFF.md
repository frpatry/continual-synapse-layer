# Session handoff — 2026-05-26 (dual-substrate episodic infrastructure ready)

## Where we are

We've pivoted away from "make the synapse layer's reward signal /
storage / consolidation cleverer" and toward a fundamentally
different architecture: **separate the substrate that computes from
the substrate that remembers**.

The empirical motivation, captured in
``results/logs/reward_confidence/1779815135_27_T15_path_d.json``:
the reward-as-confidence path (path D) showed the R signal IS
structured — ``cosine_reward_developmental`` beats the baseline ACC
by **+1.5 pp** at T=15 — but it does so by **trading −7.3 pp of
Task-0 retention**. Three architectural iterations (path A
true labels, path C per-class consolidation, path D per-sample
reward) have all hit the same wall: any change that helps the model
learn faster bleeds into Task-0 forgetting; any change that protects
Task-0 starves plasticity. The plausible reading is that the
trade-off is an **artefact of asking one substrate (the network
weights) to carry two responsibilities**.

The new design (Option 2 in the user's plan): the network is free
to learn (standard backprop, no protection mechanisms), and an
**active episodic memory** grows alongside it via gradient-free
allocation. At inference, the model's softmax is blended with a
retrieval-based label distribution from the memory, with the blend
weight scaled by retrieval confidence. The memory contributes when
something genuinely similar exists in it, and stays silent
otherwise.

The bet: if the trade-off is substrate-coupling, this separation
should dissolve it. The model can over-write its representation of
an old task (catastrophic at the weights level), but the memory's
entries from that task remain queryable and contribute at inference
whenever a similar input arrives.

## What ships in this session (incremental, all committed)

Five commits land the dual-substrate infrastructure:

- `96244cb` — Phase 0: pivot documented in decisions_log; the
  cosine_reward T=15 headline table, three-iteration retrospective,
  and the four-tier decision criteria for the new pilot are all
  captured in the
  "Pivot to dual-substrate episodic architecture" entry.
- `342b3c4` — Phase 1: ``ActiveEpisodicMemory`` — gradient-free
  allocation via cosine novelty threshold; weighted top-k vote at
  retrieval. 8 unit tests covering allocation, retrieval, cache
  invalidation, and the ``max_entries`` cap.
- `7057d6f` — Phase 2: ``EpisodicPredictor`` — blends base-model
  softmax with retrieval distribution; ``λ_eff`` scales from 0 to
  ``blend_max`` with retrieval confidence above
  ``blend_threshold``. Returns log-probabilities for downstream
  compatibility. ``training_step_observe`` runs the storage
  decision under ``torch.no_grad``. 6 unit tests.
- `56fe242` — Phase 3: ``cs_episodic_dual_substrate`` config in
  ``src/continual_synapse/episodic/training_configs.py``.
  ``EpisodicConfig`` dataclass + ``EPISODIC_CONFIGS`` registry.
  Smoke-tested at T=2 n=1: ACC=1.000, memory grows on novelty,
  no allocations on repeat inputs.
- `118ddba` — Phase 4: ``experiments/28_episodic_dual_substrate_eval.py``
  driver — manual run only (NOT executed in-session). Trains the
  episodic config, re-evaluates the unchanged
  ``cs_gated_cosine_developmental`` baseline from existing exp-27
  checkpoints when available, writes exp-23-compatible JSON for
  the exp-24 retention analyser, plus a storage-diagnostics block
  (per-task memory growth, early-vs-late novelty mean).
- (this commit) — Phase 5: this handoff update.

## Running the pilot (manual, when ready)

```bash
source .venv/bin/activate
python experiments/28_episodic_dual_substrate_eval.py --T 15 --n_seeds 2
```

(Defaults: episodic config with ``novelty_threshold=0.7``,
``retrieval_k=5``, ``blend_threshold=0.5``, ``blend_max=0.5``;
unbounded memory; baseline auto-loaded from
``results/checkpoints/phase_d/`` if present.)

ETA estimate: the dual-substrate training is a plain MLP forward +
backward per batch plus a no-grad feature extract for the memory.
Should be **faster than scout_a095** (no n_passes=5 multi-pass
synapse buffer; no cold-storage Chroma I/O in the training loop).
Path-A T=15 n=3 was 19 min total; expect maybe ~12 min total for
the new pilot at n_seeds=2, plus a few seconds per baseline seed
for the eval-only reload.

## What to read in the printed summary

The script ends with:

```
=== Dual-substrate episodic — T=15, n=2 ===
config                                       ACC   Task-0   Task-N   memory
------------------------------------------------------------------------------
cs_gated_cosine_developmental (ref)        0.814    0.805    0.906        N/A
cs_episodic_dual_substrate                 X.XXX    X.XXX    X.XXX     XXX avg
  (Δ vs baseline, pp):                     +X.XX    +X.XX    +X.XX
```

And per-seed, look for:

- ``final memory size = N`` after each training. The "reasonable"
  band for T=15 is **50–500** entries; anything below 30 says the
  novelty threshold is too high (memory isn't capturing the input
  space), anything above ~2000 says the threshold is too low and
  storage is overflowing.
- ``per-task memory size: t0=…, t1=…, …``. A healthy curve grows
  fast on the first task and slows as later-task inputs find more
  matches; near-linear growth across all tasks means the threshold
  isn't discriminating between novel and seen, and a near-flat
  curve after task 0 means the threshold is too high.
- ``novelty mean (first 100 batches): 0.XX; last 100: 0.YY``.
  Expected: 0.YY ≪ 0.XX. The first-100 average will be near 1.0
  (memory starts empty); the last-100 average should be much lower
  as the memory fills.

## Decision criteria for the pilot

(Carried from the decisions_log entry, repeated here for the
operator who reads only this file.)

- **Strong win**: ACC ≥ baseline AND Task-0 ≥ baseline + 5 pp
  AND memory grows to a reasonable size (50–500 entries for T=15)
  → green-light T=50, n=5 in a later session.
- **Moderate**: ACC roughly matches baseline AND Task-0 ≥ baseline
  by any positive margin. Counts as a win because the base model
  has zero protection mechanisms — any non-collapse on Task-0
  proves the memory substrate is doing the retention work, even
  if the magnitude is small.
- **Concerning**: Task-N collapses (means retrieval is dominating
  poorly on new tasks). The blend logic or the novelty threshold
  needs tuning. Sweep ``--novelty-threshold`` (try 0.5, 0.8) and
  ``--blend-max`` (try 0.3, 0.7) before pivoting again.
- **Null**: Both ACC and Task-0 worse than baseline → the dual-
  substrate hypothesis is wrong; retrieval over a free-running
  model doesn't help and the trade-off is intrinsic to the
  function being learned, not to the substrate that holds it.
  Pivot direction: investigate stronger memory mechanisms
  (parametric memory à la GEM) or accept the Phase-B Pareto
  frontier as the final story for the article.

## Reference numbers from the path-D T=15 pilot

(Mean across 3 seeds; for the comparison the new pilot prints.)

| config                                  | ACC    | Task-0 | Task-N |
|-----------------------------------------|-------:|-------:|-------:|
| cs_gated_cosine_developmental           | 0.8143 | 0.8047 | 0.9063 |
| cs_reward_developmental                 | 0.6683 | 0.2004 | 0.9317 |
| cosine_reward_developmental             | 0.8291 | 0.7317 | 0.8985 |
| reward_only_static                      | 0.6683 | 0.2004 | 0.9317 |

The new pilot should be compared against
``cs_gated_cosine_developmental`` (the unchanged baseline).
Anything that beats Task-0 = 0.8047 wins on retention; anything
that holds ACC ≥ 0.8143 wins on aggregate.

## What's been kept from prior lines of work

The reward / path-A / path-C / path-D infrastructure is **not
reverted**:

- ``true_label`` + ``label_histogram_json`` metadata on
  cold-storage entries (cheap, possibly reusable).
- ``RetrievalEnsemble.label_source`` config + breakdown.
- ``SynapseAugmentedMLP(consolidation_mode="per_class", ...)``
  as an opt-in mode (not recommended with current cosine gating).
- ``apply_hebbian_update(reward_signal=..., reward_mode="per_sample")``
  + ``current_maturity`` property.
- ``REWARD_CONFIGS`` registry with the four named configs.
- ``ConsolidationTrigger.mode={pressure,count}``.
- Exp 25, 27 driver scripts.

All of those still work; the dual-substrate work just adds a parallel
``episodic`` subpackage and a new experiment driver.

## What's explicitly DISABLED in cs_episodic_dual_substrate

- No ``SynapseLayer``
- No ``SynapseAugmentedMLP`` wrapper (we use plain ``MLPClassifier``)
- No cosine gating (``apply_gradient_gating``)
- No Hebbian state, no ``apply_hebbian_update``
- No EWC, no parameter-protection regulariser
- No reward computer or reward mixer
- No cold storage (the memory is in-memory Python lists, not
  Chroma-backed for v1)

That's the architectural bet: the model is free.

## Files of interest

- ``src/continual_synapse/episodic/active_memory.py`` (new) —
  ``ActiveEpisodicMemory`` with gradient-free allocation.
- ``src/continual_synapse/episodic/episodic_predictor.py`` (new) —
  blend-at-inference wrapper.
- ``src/continual_synapse/episodic/training_configs.py`` (new) —
  ``EpisodicConfig`` + ``EPISODIC_CONFIGS`` registry.
- ``experiments/28_episodic_dual_substrate_eval.py`` (new) — the
  manual driver.
- ``decisions_log.md`` — the architectural rationale and the
  prior-three-paths retrospective.

## Suite status

**430 tests passing** at the end of this session:

| baseline pre-session | + Phase 1 active_memory | + Phase 2 predictor | total |
|---:|---:|---:|---:|
| 416 | +8 | +6 | 430 |

(Phase 0 was docs only, Phase 3 added the config + a smoke test
without unit tests, Phase 4 added the driver — manual run, not
exercised by pytest.) No new dependencies introduced.

## If the dual-substrate hypothesis fails

The Pareto-frontier framing from the Phase B verdict
(``decisions_log`` 2026-05-26) remains the fallback story for the
follow-up article. The reward-as-confidence work also stands as a
publishable finding even if it doesn't ship as the final design —
the Chebyshev anti-correlation between R and feature magnitude
(commit ``5d525d3``) reveals a hidden coupling worth writing up.

## Open todos (deferred, not blocking)

- **n=10 validation** on the Phase B configs for clean Wilcoxon
  Bonferroni at T=50. (Unchanged from prior handoffs.)
- **Decay-subsystem honesty note** in README. (Unchanged.)
- **Promote ActiveEpisodicMemory to disk-backed storage** via
  ``ColdStorage`` if the dual-substrate hypothesis works at T=15
  and we need to scale to T=50 with many more entries.
