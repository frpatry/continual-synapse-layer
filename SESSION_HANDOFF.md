# Session handoff — 2026-05-26 (memory-augmented native architecture ready)

## Where we are

The functional-regularization pivot worked but capped at
**DER-equivalent** results: ``cs_gated_cosine_functional`` at T=15
n=4 hits ACC=0.904, Task-0=0.908, FGT=0.003 — a clean, audited,
strong result that ties (rather than surpasses) what Dark Experience
Replay achieves in the literature. The six-step audit confirmed
the result is real (no contamination, +2.23pp generalization gap,
healthy loss magnitudes, flat per-task retention curve, exact memory
math, tight 4-seed replication) — it's just not a breakthrough on
the existing methodological frontier.

The pattern across **every architecture this project has tried** is
the same: bolting memory onto a trained model — whether at inference
(retrieval ensembles), as a distillation target (LwF / DER), or
through gradient modulation (cosine gating) — hits a fundamental
ceiling. Models that don't *learn to use memory* cannot leverage it
effectively, even when the memory contains correct information.

The pivot for this session: an architecture where memory access is
**part of the forward pass during training**. Inspired by DNC
(Graves 2016), MANN (Santoro 2016), Memorizing Transformers
(Wu 2022), and the project's original Cold Storage vision — which
proposed queryable storage trained jointly with the network, an
idea we never properly implemented.

## What ships in this session (all committed)

- `dda0881` — Phase 0: pivot rationale + decisions_log entry
  codifying the "bolted-on memory hits a ceiling" pattern across
  retrieval ensembles, dual-substrate, and functional reg.
- `f632a16` — Phase 1: ``MemoryAugmentedMLP`` + ``ExternalMemory``
  in ``src/continual_synapse/memory_augmented/`` with 8 unit tests
  covering empty/non-empty paths, gradient flow through every
  access head, and the frozen-stored-entries contract.
- `c4aef77` — Phase 2: ``experiments/31_memory_augmented_eval.py``
  with three configs (the proposal, the architectural control,
  and the cs_gated_cosine_functional reference) + per-task
  diagnostics (gate trace, attention entropy). Smoke-tested at
  T=2.
- (this commit) — Phase 3: handoff update + run command + decision
  tiers + the post-pilot diagnostic playbook.

The earlier path-A/B/C/D + dual-substrate + functional infrastructure
all stays in the codebase under its original subpackages. The new
memory-augmented work is purely additive.

## Architecture sketch

```
x -> encoder -> h
h -> query_proj -> query
retrieved, weights = attention(query, memory.keys, memory.values)
h, retrieved -> context_combiner -> combined
gate = sigmoid(memory_gate(h))                  # (B, 1)
effective_h = (1 - gate) * h + gate * combined
logits = classifier(effective_h)
```

The four memory-access heads — ``query_proj``, ``value_proj``,
``context_combiner``, ``memory_gate`` — are ``nn.Linear`` parameters
trained end-to-end via the task loss from batch 0. The stored
``(keys, values, task_ids)`` are ``register_buffer`` snapshots
written at task end under ``torch.no_grad``; they don't appear in
``model.parameters()`` and don't accumulate gradients.

The empty-memory regime returns zero retrieved values, and the
forward's ``if len(memory) > 0`` guard bypasses the gate/combiner
path entirely — so before any writes have happened the model's
output equals ``classifier(encoder(x))`` bit-exactly. ``query_proj``
still runs in that regime (we compute it before the read), so the
gradient path through it warms up from batch 0; the real signal
for the access heads kicks in once memory is non-empty (first
batch of task 1 onward).

## Running the pilot

```bash
source .venv/bin/activate
python experiments/31_memory_augmented_eval.py --T 15 --n_seeds 3
```

ETA: cs_gated_cosine_functional reloads from
``results/checkpoints/phase_f/`` (~5s/seed). The two memory-augmented
configs train fresh (plain MLP + attention head, no synapse machinery)
— ~30s/seed each at T=15. Total: ~5 minutes for three configs × three
seeds. Well within harness limits, no chunking needed.

## What to watch in the printed output

Per-seed lines for the memory-augmented configs:

```
trained in Xs; final memory size = N
eval done in Ys   ACC=X.XXX  Task-0=X.XXX  Task-N=X.XXX
memory: t0=N0, t1=N1, ..., t14=N14
gate_mean: t0=0.XXX  tmid=0.XXX  tlast=0.XXX
attention entropy (last non-zero): X.XXX  (uniform would be ln(N_mem))
```

The two single most diagnostic lines:

1. **gate_mean trajectory.** At init the gate is ``sigmoid(0)=0.5``.
   If it stays near 0.5 or rises, the model is learning to use
   memory. If it drops toward 0, the model has learned to ignore
   the memory path — the architecture is structurally fine but
   not actually contributing.
2. **Attention entropy** of the last non-zero task. Maximum entropy
   for ``N`` stored entries is ``ln(N)``: at T=15 with 1500 entries,
   that's ≈ 7.31. If observed entropy is close to that (within
   ~0.5), the model is spreading attention uniformly = not finding
   useful structure. If it's significantly lower (say < 5.0), the
   model is focusing on specific entries = retrieval is finding
   useful neighbours.

## Decision criteria

Baseline to clear is ``cs_gated_cosine_functional``:
ACC=0.904, Task-0=0.908.

- **Strong win**: ACC ≥ 0.92 AND Task-0 ≥ 0.92 → genuine surpass
  of the DER-equivalent baseline by ≥ +1.6 pp on both axes. The
  +5 pp goal from the decisions_log entry would be Task-0 ≥ 0.96
  — that's the headline target. Strong-win opens a T=50 + n=10
  validation in a later session.
- **Moderate**: ACC and Task-0 within ±2 pp of baseline → the
  architecture matches but doesn't surpass. Tuning candidates
  (``gate_init``, ``key_dim``, ``samples_per_task``) before any
  follow-up.
- **Null**: ACC or Task-0 significantly below baseline (≥ 5 pp
  worse) → integration issues; debug per the diagnostic playbook
  below or pivot back.

The architectural control (``memory_augmented_no_memory``) is
essential: it uses the same model but never writes to memory. If
it matches or beats ``memory_augmented_native``, the architecture's
gains aren't coming from memory — they're coming from parameter
count or shape, which would invalidate the result.

## Post-pilot diagnostic playbook

If the result is moderate or null, the printed diagnostics tell
you what to try next:

1. **gate_mean stuck near 0 across all tasks** → model isn't
   learning to use memory. Try ``--gate-init 2.0`` (biases the
   initial sigmoid output toward open ≈ 0.88). Also worth trying
   a higher learning rate just for the gate.
2. **attention entropy near ``ln(N_mem)`` throughout** → model
   can't distinguish useful entries. Increase ``--key-dim`` to
   128 or 256 — wider keys give the query more capacity to be
   selective.
3. **ACC degrades over tasks (per-seed eval drops across the
   run)** → memory is hurting more than helping. Likely cause:
   ``value_proj`` is producing noisy values that the combiner
   can't denoise. Try increasing ``--value-dim`` or adding a
   nonlinearity in ``value_proj``.
4. **memory_augmented_no_memory ≈ memory_augmented_native** →
   the memory mechanism is contributing nothing; the architecture
   alone happens to be a reasonable continual-learning regulariser
   (probably via the gate adding parameters in an attention-like
   way). Memory isn't the explanation; investigate the architectural
   inductive bias separately.

The smoke at T=2 already showed a soft version of failure mode #1
(gate trained DOWN to 0.03 with default ``gate_init=0``). At T=2
the forgetting pressure is too weak to give the model a reason to
use memory. The T=15 pilot is where the architectural bet actually
gets tested — there's enough forgetting that retrieval should
provide gradient signal toward opening the gate.

## Reference numbers for the comparison

| metric | cs_gated_cosine_developmental (Phase B, T=15) | cs_gated_cosine_functional (DER-equivalent, T=15 n=4) |
|---|---:|---:|
| ACC | 0.814 | **0.904** |
| Task-0 | 0.798 | **0.908** |
| Task-N | 0.906 | 0.911 |
| FGT proxy | +0.108 | **+0.003** |

``cs_gated_cosine_functional`` is the bar to clear. If
``memory_augmented_native`` lands at e.g. ACC=0.92, Task-0=0.93,
that's the headline result — a method that surpasses DER-equivalent
on Permuted-MNIST without using a distillation loss.

## What's explicitly excluded from `memory_augmented_native`

- No synapse layer, no cosine gating, no Hebbian state, no EWC.
- No functional regularisation / distillation loss.
- No retrieval blend at inference (memory is consulted during
  training too — that's the whole point).

The architectural control config (`memory_augmented_no_memory`)
uses the same model but writes zero entries; the reference
(`cs_gated_cosine_functional`) is the existing pipeline loaded
from disk.

## Files of interest (new this session)

- ``src/continual_synapse/memory_augmented/memory_augmented_model.py``
  — ``ExternalMemory`` + ``MemoryAugmentedMLP``.
- ``experiments/31_memory_augmented_eval.py`` — the driver with
  three configs.
- ``tests/test_memory_augmented.py`` — 8 unit tests.

## Suite status

**459 tests passing** at end of session.

| pre-session (functional reg) | + Phase 1 memory-augmented | total |
|---:|---:|---:|
| 451 | +8 | 459 |

(Phase 2's exp 31 is exercised by the T=2 smoke, not by pytest —
matches the pattern of every prior exp script in this repo.) No
new dependencies introduced.

## If the pilot is moderate or null

The diagnostic playbook above gives concrete sweep candidates
(``gate_init``, ``key_dim``, ``value_dim``). If none of those
recover a strong result at T=15, the honest read is that on this
benchmark — Permuted-MNIST with a small MLP — native-attention
memory doesn't structurally beat DER-equivalent distillation. The
methodological contribution from the functional-reg pilot remains
defensible (a clean re-derivation of DER plus the six-step audit),
and the publishable story shifts to a thorough characterisation of
what does and doesn't work across an unusually large set of
continual-learning architectures.

If the pilot is a strong win, the next-session work is:

1. T=50 n=5 validation to confirm scale.
2. Sweep ``key_dim`` × ``value_dim`` × ``gate_init`` ×
   ``samples_per_task`` at T=15 n=5 to characterise the
   architectural Pareto frontier.
3. Compose with cosine gating
   (``memory_augmented_native + gated_cosine``) as a follow-on
   experiment — same composition test that worked for
   ``cs_gated_cosine_functional`` over ``cs_functional_only``.

## Open todos (deferred, not blocking)

- T=50 n=3 pilot for functional reg (was started, hit harness
  timeout — see prior handoff notes; can be resumed in chunks).
- Decay-subsystem honesty note in README (still deferred).
