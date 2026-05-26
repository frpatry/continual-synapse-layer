# Session handoff — 2026-05-26 (reward-as-confidence infrastructure ready)

## Where we are

The Cold Storage v2 retrieval-ensemble line of work is **closed as
a dead end** on the current architecture. Three storage variants
tested at T=15 — path B (labels-as-of-now), path A (true-label
aggregate), path C (class-pure prototypes) — and none of them
produced a measurable Task-0 lift over the no-retrieval baseline.

**Path-C added a new finding**: storage quality is not the
bottleneck. The label-derivation diagnostic jumped from path-A's
26-29% to path-C's 85.4% — per-class prototypes ARE dramatically
more class-discriminative. But the retrieval ensemble's Task-0
delta stayed neutral, AND inflating cold storage by 15× broke
``scout_a095``'s training dynamics (baseline ACC dropped from
0.815 to 0.766) because cosine gating saturates when too many
familiar-looking patterns are stored.

Conclusion: the limit is upstream of storage. The Hebbian update's
reward signal ``R`` has been a constant 1.0 throughout the entire
project. Every sample contributes equally to synapse updates,
whether it's a routine correct prediction or a confidently-wrong
edge case. **The next direction is to make R per-sample.**

See decisions_log entries "Cold Storage v2 path-A pilot — mechanism
works, criterion fails" and "Path-C pilot — per-class consolidation
not the right pivot" for the long-form narrative on the storage-
quality dead end.

## What ships in this session (incremental, all committed)

Storage-line work:

- `78f514c` — path-A step 1: ``true_label`` + ``label_histogram``
  captured at consolidation time; backward-compatible.
- `a5cbd90` — path-A step 2: ``RetrievalEnsemble.label_source``
  config + ``label_source_breakdown`` diagnostic.
- `1860569` — path-A T=15 pilot: 100% true_label coverage,
  retrieval neutral, decision-criterion fail.
- `1885443` — path-C step 1: ``consolidation_mode="per_class"``
  refactor + 5 tests; aggregate mode bit-identical.
- `a5e6b54` — exp 25 wiring for ``--consolidation-mode`` +
  storage diagnostics (used by the partial path-C pilot).
- `90702c0` — path-C verdict documented; pivot to reward
  direction.

Reward-as-confidence infrastructure (Phases 1–4 of the new line):

- `b9fcc26` — utility module
  ``src/continual_synapse/reward/confidence_reward.py``:
  ``compute_reward_signal``, ``normalize_reward_batch``,
  ``developmental_alpha``. 8 unit tests.
- `5665e5c` — ``apply_hebbian_update`` integration:
  ``reward_signal`` + ``reward_mode`` kwargs. Constant mode
  bit-identical to pre-path-D. Sqrt-pre-weighting gives the
  exact ``Σ_i R_i a_i a_i.T / B`` math without a synapse-layer
  API change. Adds ``current_maturity`` property (live
  recompute, independent of ``apply_gradient_gating``'s cache —
  required for gating-disabled configs). 4 new tests.
- `bf18661` — registry
  ``src/continual_synapse/reward/training_configs.py``:
  ``REWARD_CONFIGS`` dict with four named entries (baseline +
  three reward variants). Each ``RewardConfig.make_callbacks()``
  returns ready ``on_pre_optimizer_step / on_after_batch /
  on_task_change`` closures. Smoke-tested at T=2 n=1 on a
  synthetic 4-class blob benchmark: all four configs reach
  ACC = 1.000.
- `5cf0030` — driver script
  ``experiments/27_reward_as_confidence_eval.py``. Exp-23-
  compatible output JSON (consumable by
  ``experiments/24_retention_analysis.py`` unchanged). Adds the
  Task-N final ACC plasticity diagnostic and the per-event
  reward variance recorder (early / mid / late summaries +
  late-collapse warning). Script is **not run** in this session
  — manual command below.
- (this commit) — final handoff update with running state and
  decision criteria.

## Immediate next direction — reward-as-confidence

Replace the constant ``R = 1.0`` in the Hebbian update with a
per-sample informativeness signal computed from the model's own
output:

```
R_i = (1 - γ) * [α * error_i + (1 - α) * uncertainty_i]
      + γ * |calibration_i|
```

Where ``α`` is **developmental** — low (0.2) when the model is
young (mostly weights uncertainty), rising to a capped maximum
(0.85) as the consolidation count grows (mostly weights error).
The cap prevents late-stage stagnation; even a "mature" system
should keep listening to uncertainty.

Three planned configs (alongside the unchanged ``cs_gated_cosine_developmental``
baseline):

1. ``cs_reward_developmental``: reward signal ON, cosine gating OFF
   — isolates R alone
2. ``cosine_reward_developmental``: both ON — the composition we
   expect to be best
3. ``reward_only_static``: reward signal ON with α = 0.5 constant
   — ablates the developmental component

The infrastructure has now landed (commits `b9fcc26`, `5665e5c`,
`bf18661`, `5cf0030` above). The experiment itself is **not run**
in-session — you trigger it manually in a terminal (see "Running
the pilot" below).

## What's been kept from the dead-end paths

The path-A and path-C code is **not reverted**:

- ``true_label`` + ``label_histogram_json`` metadata stays on
  cold-storage entries (cheap; useful baseline for any future
  retrieval revival).
- ``RetrievalEnsemble.label_source`` flag stays.
- ``SynapseAugmentedMLP(consolidation_mode="per_class", ...)``
  stays as an opt-in mode. **Don't use it with the current cosine
  gating** — see the path-C decisions_log entry for the
  interaction that breaks baseline training. Could be revisited
  if cosine gating is replaced.

All three are no-ops at their defaults (``aggregate`` mode,
``label_source="auto"``, no ``training_target`` passed). Loading
older checkpoints still works.

## Running the pilot (manual, when ready)

```bash
source .venv/bin/activate
python experiments/27_reward_as_confidence_eval.py --T 15 --n_seeds 3
```

The ``source .venv/bin/activate`` is required — the system has
no ``python`` on PATH outside the venv. Once active your shell
prompt shows ``(.venv)`` and ``python`` resolves to the venv's
interpreter. All ``python ...`` commands in this handoff and in
``decisions_log.md`` assume the venv is active.

(Defaults: 4 configs — baseline + 3 reward variants, output dir
``results/logs/reward_confidence/``, checkpoint dir
``results/checkpoints/phase_d/``.)

ETA: path-A took 19 minutes at T=15 n=3; the reward signal adds
one small extra compute step per batch (softmax + entropy +
calibration); expect ~25 minutes. Per-class consolidation mode
should NOT be used with these configs (cosine gating still
active in two of them).

## Decision criteria for the reward-as-confidence pilot

Baseline reference is ``cs_gated_cosine_developmental`` at T=15
n=3 (the unchanged scout_a095_validated config; numbers from path
A's pilot, since path A's training is bit-identical to baseline
when path A's only change — ``training_target=y`` flowing into
``apply_hebbian_update`` — has no effect without an
``ExternalReward``):

| metric | baseline value at T=15 n=3 |
|---|---:|
| aggregate ACC | 0.8143 |
| Task-0 retention | 0.8047 |
| Task-N final (new diagnostic) | TBD — measured during the pilot |
| Forgetting | TBD |

Decision tiers:

- **Strong win**: ≥1 reward config beats baseline on Task-0 by
  ≥ +3 pp without losing more than 2 pp of aggregate ACC →
  green-light a T=50 n=5 run in a separate session.
- **Modest signal**: ≥1 reward config moves Task-0 by ≥ +1 pp
  in the right direction → tune α schedule / γ / floor and
  rerun before T=50.
- **Null**: no reward config moves Task-0 from baseline → the
  per-sample-R hypothesis is wrong; pivot again (see below).
- **Catastrophic**: any reward config crashes baseline training
  (aggregate ACC drops > 5 pp) → the per-sample weighting is
  destabilising; investigate normalization / floor / α cap
  before the next attempt. ``reward_only_static`` is the
  control here — if only the developmental variant crashes,
  α is the problem; if static also crashes, the formula itself
  needs work.

Additional diagnostic from the exp 27 script:
``reward_statistics_per_consolidation`` records mean and variance
of R at each consolidation event. If late-training variance
collapses (> 50% drop from early variance), the signal has
saturated and the developmental α cap was insufficient.

## If the reward direction also fails

The Pareto-frontier framing from the Phase B verdict (decisions_log,
2026-05-26) remains the fallback story for the follow-up article.
The architecture's two informative knobs (``maturity_target_consolidations``
and ``gradient_gating_alpha``) trade Task-0 retention against
aggregate ACC, with EWC λ=10 owning Task-0 retention outright
at T=50. That's a publishable result even if no future work
closes the +21.5 pp gap.

## Open todos (deferred, not blocking)

- **n=10 validation** on scout_combined and scout_a095 for clean
  Wilcoxon Bonferroni at T=50. (Unchanged from prior handoffs.)
- **Decay-subsystem honesty note** in README and the follow-up
  article: under ``cs_gated_cosine_developmental`` the stored
  strengths matrices are computed and discarded. (Unchanged.)
- **Soft voting via ``label_histogram_json``** (would have been
  step 4 of the path-A line). Cheap, no retraining needed,
  but probably moot given path C disproved the storage-quality
  hypothesis. Document but don't pursue.

## Key design choices preserved across the storage-line work

1. Features extracted via ``base.features(x)`` — same vector
   space as stored embeddings.
2. Cosine sims clipped at 0 in retrieval weighted-vote.
3. ``RetrievalEnsemble.predict`` puts the model in ``eval()``
   and restores the prior ``training`` flag.
4. Checkpoint format = single ``.pt`` with ``model_state_dict``
   + serialised ``cold_storage_entries`` + ``config``.
   Path-A entries carry ``true_label`` + ``label_histogram_json``;
   path-C entries carry ``true_label`` only (one-hot histogram
   is redundant). All loadable via ``metadata.get(...)``.
5. ``label_source="auto"`` (default) is bit-identical to the
   pre-path-A ``derived`` behaviour on any store without
   ``true_label`` metadata.

## Files of interest

- ``src/continual_synapse/baselines/synapse_finetune.py`` —
  ``apply_hebbian_update`` (now with ``reward_signal`` +
  ``reward_mode`` kwargs), ``SynapseAugmentedMLP`` (now exposing
  ``current_maturity``).
- ``src/continual_synapse/reward/confidence_reward.py`` (new
  this session) — the per-sample R utility:
  ``compute_reward_signal``, ``normalize_reward_batch``,
  ``developmental_alpha``.
- ``src/continual_synapse/reward/training_configs.py`` (new this
  session) — ``REWARD_CONFIGS`` registry + ``RewardConfig``
  dataclass with ``.make_callbacks()`` factory.
- ``experiments/27_reward_as_confidence_eval.py`` (new this
  session) — the training + eval driver. Exp-23-compatible
  output JSON; sidecar carrying per-consolidation reward
  variance.
- ``src/continual_synapse/inference/retrieval_ensemble.py`` —
  unchanged; retrieval line is closed but code stays.
- ``decisions_log.md`` — full narrative for paths A / B / C
  and the rationale for the reward pivot.

## Suite status

**411 tests passing** at the end of this session:

| baseline | + Phase 1 reward utility | + Phase 2 integration | total |
|---:|---:|---:|---:|
| 399 | +8 | +4 | 411 |

(Phase 3 added one module + smoke test, no unit tests. Phase 4
added the eval script; per the user's spec it is not exercised
by the test suite — manual run only.) No new dependencies
introduced this session.
