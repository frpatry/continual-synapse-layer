# Session handoff — 2026-05-26

## Where we are

Phase B closed (decisions_log 2026-05-26 entry). Cold Storage v2
inference-time retrieval ensemble built and **smoke-tested through to
data**: the exp 25 T=15 n=3 pilot completed in this session and the
result is unambiguous — **path B is dead**. Label-derivation accuracy
averaged **18.7%** across 3 seeds (well below the 50% viability
threshold), and all three retrieval configurations degraded baseline
ACC by between −0.7 and −10.0 pp.

The infrastructure (RetrievalEnsemble class, checkpoint persistence,
label-accuracy diagnostic, exp 25 train-and-eval pipeline) is solid
and re-usable. What needs replacing is the **labels-as-of-now**
derivation — it doesn't work because the model's classifier head has
drifted too far from the consolidation-time state.

## Immediate next decision

**Pivot to path A** (retrain with true-label storage at consolidation
time) or move to a different mechanism entirely. Do not invest more
seeds in the current path-B pilot — the diagnostic has settled it.

If pivoting to path A, the first concrete steps are:

1. Modify `src/continual_synapse/baselines/synapse_finetune.py`
   `apply_hebbian_update` to accept a `training_labels` kwarg (already
   has `training_target`; we'd repurpose or add) and compute the
   majority class in the batch as the dominant true label.
2. Modify `src/continual_synapse/consolidation/pipeline.py`
   `consolidate_to_storage` to accept and store `dominant_label`
   in entry metadata.
3. Rebuild scout_a095 checkpoints with the new instrumentation
   (~70 min for n=3 × T=15, ~5 h for n=3 × T=50).
4. Re-run exp 25 with a flag that uses `metadata["dominant_label"]`
   for the ensemble's label vector instead of deriving via
   `argmax(head(stored))`.

## What completed this session

- `2c55c2a` — `src/continual_synapse/inference/retrieval_ensemble.py`
  + 10 unit tests
- `9ecaa1d` — `experiments/25_retrieval_ensemble_eval.py`
  (combined train-and-evaluate)
- `f5bb465` — exp 25 gains path-B label-derivation accuracy
  diagnostic + `on_task_change` task_id tagging in training
- T=15 pilot ran end-to-end:
  - 3 checkpoints at `results/checkpoints/phase_b/scout_a095_T15_seed{0,1,2}.pt`
  - results JSON at `results/logs/retrieval_ensemble/1779790326_25_T15.json`
  - log at `/tmp/exp25_pilot.log`

## Pilot result reference

```
=== Label derivation accuracy (path B sanity check)  T=15 ===
seed 0: 10.9% (11/101 correctly relabeled)
seed 1: 26.2% (27/103 correctly relabeled)
seed 2: 19.1% (21/110 correctly relabeled)
average: 18.7%
  ⚠ unrecoverable — needs path A retraining

=== Retrieval Ensemble v2 — Pilot Results  T=15 ===
  baseline scout_a095:                    ACC=0.814  Task-0=0.814
  v2_mild      (k=5, τ=0.70, λ=0.30):     ACC=0.807   (Δ −0.7 pp)
  v2_moderate  (k=5, τ=0.80, λ=0.50):     ACC=0.766   (Δ −4.7 pp)
  v2_aggressive(k=5, τ=0.50, λ=0.50):     ACC=0.713   (Δ −10.0 pp)
```

To regenerate retention curves from the pilot data without re-running:

```bash
python experiments/24_retention_analysis.py \
    --log-paths results/logs/retrieval_ensemble/1779790326_25_T15.json \
    --fig-dir results/figures/retrieval_ensemble \
    --analysis-path results/analysis/retrieval_ensemble_retention.json
```

## Phase B reference numbers (T=50, n=5)

| Config | Aggregate ACC | Task-0 retention |
|---|---:|---:|
| baseline (target=50, α=0.9) | 0.516 | 0.463 |
| scout_combined (target=100, α=0.95) | **0.565** | mid-pack |
| scout_a095_validated (target=50, α=0.95) | mid-pack | **0.619** |
| scout_mat100 (target=100, α=0.9) | dominated | dominated |
| **ewc_lam_10** (reference, separately trained) | comparable to scout_combined | **0.834** |

Gap our architecture needs to close to dethrone EWC on Task-0
retention at T=50: **+21.5 pp**.

Full per-task / per-seed numbers in:
- `results/analysis/phase_b_retention.json`
- `results/logs/scaling/1779714283_21_scaling_T{15,30,50}.json`
- `results/logs/phase_b_validation/1779744145_23_phase_b_T15.json`
- (`T=30` and `T=50` Phase B JSONs may exist if your exp 23 run
  has finished; check `ls results/logs/phase_b_validation/`)

## Open todos (deferred, not blocking)

- **Decay-subsystem honesty note** in README and the follow-up
  article: stored strengths matrices computed and discarded under
  `cs_gated_cosine_developmental` — the decay machinery is dormant
  Phase 4 / cs_full infrastructure. See decisions_log 2026-05-26
  section "Honesty note".
- **n=10 validation** on scout_combined and scout_a095 for clean
  Wilcoxon Bonferroni (current Phase B is n=5; statistical floor
  hits 0.1875 with Bonferroni × 3).
- **Path A** retraining pipeline (only if we commit to that
  direction over an alternative mechanism).

## Key design choices in the Cold Storage v2 implementation

1. Features extracted via `base.features(x)`, **not** the
   modulator-augmented `model.features(x)` — same vector space the
   stored embeddings live in. Verified by a dedicated test.
2. Labels derived once at ensemble init via
   `argmax(model.base.classify(stored_embedding))`. This is the
   **path-B** assumption that the head still classifies old
   activations coherently; the diagnostic shows it doesn't at
   T=15 on Permuted-MNIST.
3. Negative cosine sims are **clipped at 0** in the weighted vote
   so anti-correlated entries don't subtract from a class's tally.
4. Eval mode is strict — `RetrievalEnsemble.predict` puts the
   model in `eval()` and restores the prior `training` flag.
5. Checkpoint format = single `.pt` with `model_state_dict` +
   serialised `cold_storage_entries` (embedding + document +
   metadata) + `config`. Re-runs with checkpoints in place skip
   training entirely (~20 min eval-only iteration).
6. Training now passes `on_task_change=notify_task_change` so
   stored entries are properly tagged with `task_id`. Required by
   the label-accuracy diagnostic and any future path-A code that
   wants to filter entries by source task.

## Files of interest

- `src/continual_synapse/inference/retrieval_ensemble.py`
- `experiments/25_retrieval_ensemble_eval.py`
- `experiments/24_retention_analysis.py` (works on exp 25 JSON
  unchanged via `--log-paths`)
- `results/figures/phase_b/retention_curve_T15.png`
- `DESIGN.md`, `PROJECT_PLAN.md`, `decisions_log.md`
- `tests/test_retrieval_ensemble.py`

## Suite status

380 tests passing. No new dependencies introduced.
