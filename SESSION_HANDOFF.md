# Session handoff — 2026-05-26 (path-A pilot complete)

## Where we are

Cold Storage v2 path-A pilot at T=15 (n=3) is **complete and clean**.
The mechanism works exactly as designed — every consolidated entry on
the new checkpoints carries a ground-truth ``true_label`` (100%
coverage across all 3 seeds), and ``label_source="true_label"``
succeeded without falling back to derived labels even once.

**But the decision criterion failed.** None of the three retrieval
configurations improved Task-0 retention by ≥ +5 pp over the
no-retrieval baseline. Δ Task-0 was essentially neutral
(−0.03 / −0.72 / −0.85 pp). Wilcoxon pairwise on Task-0 is saturated
at p_bonf = 1.00.

The qualitative win is that path A removed path B's catastrophic
collapse entirely: v2_aggressive recovered ~8 pp of aggregate ACC
(from 0.713 path-B to 0.796 path-A). Mechanism real; gain in the
headline metric absent at T=15.

See decisions_log entry "2026-05-26 Cold Storage v2 path-A pilot —
mechanism works, criterion fails" for the long-form narrative.

## Immediate next decision

**Three options on the table** for whoever picks this up next:

1. **Step 4 — soft voting via ``label_histogram_json``.** Path-A
   checkpoints already store the per-class histogram from each
   consolidation batch (step 1 wrote it; step 2 deliberately ignored
   it). Replace the hard argmax vote in ``RetrievalEnsemble`` with a
   soft distribution-weighted vote. **No retraining required** —
   re-runs eval on the existing path-A checkpoints. If "hard labels
   collapse useful within-batch ambiguity" is the failure mode, this
   recovers the lost signal. ~30 min eval per seed.
2. **T=50 path-A pilot.** Headroom is much bigger at T=50
   (baseline Task-0 ~0.46–0.62 vs ~0.80 at T=15), so the same
   mechanism could potentially produce a measurable Task-0 lift
   where T=15 didn't have room. Cost: ~3 h per seed × n=5 = ~15 h
   training plus a few minutes of eval. Risk: if soft voting (option
   1) wins, this gets rerun anyway with the better mechanism.
3. **Abandon Cold Storage v2 retrieval-ensemble entirely** and move
   to a different direction (embedding-space regulariser, parametric
   memory à la GEM, or treat the Phase B Pareto frontier as the
   final story for the article). The Wilcoxon p-values say there's
   no rescue hiding in the current variant.

Suggested ordering: try option 1 first (cheap), use its result to
decide between option 2 and option 3.

## What completed this session

Three commits ship the path-A line of work:

- `78f514c` — step 1: ``consolidate_to_storage`` + ``apply_hebbian_update``
  capture ``true_label`` and ``label_histogram`` per consolidation batch.
  Backward-compatible; 7 new tests.
- `a5cbd90` — step 2: ``RetrievalEnsemble.from_model_and_storage``
  gains ``label_source ∈ {auto, true_label, derived}`` and exposes
  ``label_source_breakdown``. 7 new tests.
- (this commit) — step 3: exp 25 plumbed with ``--label-source`` and
  ``--run-tag``; pilot run; comparison vs path B; exp 24 retention
  analysis on the new JSON.

T=15 path-A pilot artifacts:

- 3 fresh checkpoints at ``results/checkpoints/phase_a/scout_a095_T15_seed{0,1,2}.pt``
  (15 MB each — local only, not committed; regenerable with the exp
  25 command below)
- Results JSON: ``results/logs/retrieval_ensemble/1779796716_25_T15_path_a.json``
- Retention analysis: ``results/analysis/retrieval_ensemble_retention_path_a.json``
- Figures: ``results/figures/retrieval_ensemble/path_a/{retention_curve,retention_heatmap}_T15.png``
- Pilot log: ``/tmp/exp25_path_a.log`` (also local only)

## Path A vs Path B at T=15 — headline table

| config         | path-B ACC | path-A ACC | Δ ACC      | Task-0 path-A | Δ Task-0 vs path-A baseline |
|----------------|-----------:|-----------:|-----------:|--------------:|----------------------------:|
| baseline       | 0.8137     | 0.8143     | +0.06 pp   | 0.8047 (ref)  | 0.00 pp                     |
| v2_mild        | 0.8066     | 0.8124     | +0.58 pp   | 0.8044        | −0.03 pp                    |
| v2_moderate    | 0.7663     | 0.8003     | +3.40 pp   | 0.7974        | −0.72 pp                    |
| v2_aggressive  | 0.7134     | 0.7956     | +8.22 pp   | 0.7961        | −0.85 pp                    |

Label source breakdown per seed (100% coverage on all):
``104 / 105 / 108 true_label, 0 derived``.

## Re-running the pilot from scratch

```bash
rm -rf results/checkpoints/phase_a
python experiments/25_retrieval_ensemble_eval.py \
    --task-lengths 15 --seeds 0 1 2 \
    --checkpoint-dir results/checkpoints/phase_a \
    --label-source true_label --run-tag path_a
```

Total wall-clock: ~19 minutes (3 seeds × ~5 min training + per-seed
eval). The earlier 1.5 h estimate was conservative.

## Re-generating the retention analysis from the existing JSON

```bash
python experiments/24_retention_analysis.py \
    --log-paths results/logs/retrieval_ensemble/1779796716_25_T15_path_a.json \
    --fig-dir results/figures/retrieval_ensemble/path_a \
    --analysis-path results/analysis/retrieval_ensemble_retention_path_a.json
```

## Phase B reference numbers (T=50, n=5)

Unchanged from the previous handoff — kept here for the Pareto-
frontier writeup.

| Config | Aggregate ACC | Task-0 retention |
|---|---:|---:|
| baseline (target=50, α=0.9) | 0.516 | 0.463 |
| scout_combined (target=100, α=0.95) | **0.565** | mid-pack |
| scout_a095_validated (target=50, α=0.95) | mid-pack | **0.619** |
| scout_mat100 (target=100, α=0.9) | dominated | dominated |
| **ewc_lam_10** (reference, separately trained) | comparable to scout_combined | **0.834** |

Gap our architecture needs to close to dethrone EWC on Task-0
retention at T=50: **+21.5 pp**.

## Open todos (deferred, not blocking)

- **Step 4 (soft voting via histogram).** Re-evaluate the same path-A
  checkpoints with a histogram-aware ``RetrievalEnsemble``. Path: add
  a ``vote_mode ∈ {"hard", "soft"}`` flag (or a third label-source
  value), read ``metadata.get("label_histogram_json")``, accumulate
  per-class soft votes weighted by similarity. No retraining.
- **Decay-subsystem honesty note** in README and the follow-up
  article: stored strengths matrices computed and discarded under
  ``cs_gated_cosine_developmental`` — the decay machinery is dormant
  Phase 4 / cs_full infrastructure. (Unchanged from previous handoff.)
- **n=10 validation** on scout_combined and scout_a095 for clean
  Wilcoxon Bonferroni at T=50. (Unchanged.)

## Key design choices preserved across path A

1. Features extracted via ``base.features(x)``, **not** the
   modulator-augmented ``model.features(x)`` — same vector space the
   stored embeddings live in.
2. Cosine sims clipped at 0 in the weighted vote so anti-correlated
   entries don't subtract from a class's tally.
3. ``RetrievalEnsemble.predict`` puts the model in ``eval()`` and
   restores the prior ``training`` flag.
4. Checkpoint format = single ``.pt`` with ``model_state_dict`` +
   serialised ``cold_storage_entries`` (embedding + document +
   metadata) + ``config``. Path-A entries now carry ``true_label``
   and ``label_histogram_json`` in metadata; backward-compatible
   readers use ``metadata.get(...)``.
5. Training passes ``on_task_change=notify_task_change`` (task_id
   tagging) AND ``training_target=y`` (path-A label storage).
6. ``label_source="auto"`` (the default) is bit-identical to the
   pre-path-A ``"derived"`` behaviour on any store without
   ``true_label`` metadata. Verified end-to-end against the path-B
   ``scout_a095_T15_seed0.pt`` checkpoint.

## Files of interest

- ``src/continual_synapse/inference/retrieval_ensemble.py`` (step 2)
- ``src/continual_synapse/consolidation/pipeline.py`` (step 1)
- ``src/continual_synapse/baselines/synapse_finetune.py`` (step 1)
- ``experiments/25_retrieval_ensemble_eval.py`` (step 3)
- ``experiments/24_retention_analysis.py`` (works on the new JSON
  via ``--log-paths``)
- ``tests/test_retrieval_ensemble.py``, ``tests/test_consolidation_pipeline.py``,
  ``tests/test_synapse_finetune.py`` (14 new tests across steps 1+2)
- ``DESIGN.md``, ``PROJECT_PLAN.md``, ``decisions_log.md``

## Suite status

394 tests passing (380 baseline + 7 step-1 + 7 step-2). No new
dependencies introduced.
