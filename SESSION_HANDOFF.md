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

### Path A sketch — concrete steps

Path B failed at T=15 with label-derivation accuracy = 18.7%.
The premise was that the current model head still classifies old
activations coherently; on a forgotten head it doesn't. Path A
removes the dependency on the current head by anchoring labels in
ground truth, captured at consolidation time when the labels are
still trustworthy.

1. **Capture true_label at consolidation.** Modify the cold-storage
   consolidation path to take the dominant true label from the
   batch that triggered the consolidation (majority class across
   the batch's `y` tensor) and write it into entry metadata.
   Add the field to `StoredEntry`-equivalent metadata schema with
   a `None` (or `-1`) default so existing checkpoints stay
   loadable. Backward-compatible: any code that doesn't pass a
   label gets the old behaviour.
2. **Teach `RetrievalEnsemble` to prefer true_label when present.**
   Add a `label_source` config flag with two values:
   - `"true_label"` (path A): read each entry's
     `metadata["true_label"]`, fall back to derived label only when
     missing (so we can still load older path-B checkpoints).
   - `"derived"` (path B, current default): existing behaviour
     — `argmax(model.base.classify(stored_embedding))`.
3. **Retrain scout_a095 with label storage enabled.** T=15, n=3
   seeds, ~1.5 h. Save fresh checkpoints alongside the path-B
   ones (different filename pattern or directory so we don't
   clobber the existing artifacts).
4. **Re-run exp 25 evaluation with the new checkpoints.** Same
   three retrieval configs (`v2_mild`, `v2_moderate`,
   `v2_aggressive`) + `scout_a095_baseline` anchor. Use
   `--label-source true_label` (or whatever flag we add).
5. **Decision gate (same threshold as path B):**
   - If at least one config shows `Task-0 ≥ baseline + 5 pp`
     AND aggregate ACC drops `≤ 2 pp`: proceed to T=50 with n=5.
   - Else: design revision needed (e.g. different blending rule,
     different similarity metric, per-class normalisation, etc.).
6. **Re-run exp 24 retention analysis on the new JSON.**
   Same command as path B, just pointed at the new log file —
   generates the retention curve and prints the summary table.

**Expected outcome:** with true labels stored at consolidation,
label quality is anchored in ground truth (100% accuracy at store
time, doesn't drift with subsequent task training). If the
embedding space remains semantically meaningful — which the
gradient-gating system already relies on for its familiarity
signal — retrieval should now help Task-0 retention measurably
rather than voting near-random as it did in path B.

If path A also fails the decision gate, the failure mode is
informative: it would mean the embedding space itself has
drifted enough that even ground-truth-labelled neighbours no
longer match query semantics. That points away from
retrieval-ensemble entirely and toward methods that constrain
the embedding space directly (e.g. embedding-space regulariser
during training, parametric memory à la GEM).

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
