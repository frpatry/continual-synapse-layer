# Session handoff — Phase 1.0 AGI architecture COMPLETE

## TL;DR — read this first

**Major pivot completed last session.** The repo now has two parallel
tracks:

1. **Legacy** (`src/continual_synapse/`, `experiments/[01-49]*.py`,
   `tests/test_*.py`) — the continual-learning work. Final empirical
   conclusion: input-replay (DER / CLS-CI v2) matches the literature
   at ~0.86 on Split-MNIST CI and ~0.86 on Permuted-MNIST T=50.
   Prototype-only memory (Phase 5.7) failed to transfer (0.193 final
   ACC on Split-MNIST CI — see commit `65059cb`). **Do not extend
   this code** — it's reference-only.

2. **AGI** (`src/agi/`, `experiments/agi/`, `tests/agi/`) — the new
   project. Phase 1.0 wired up at commit `610c250`: frozen
   Qwen-0.5B-Instruct + structured-fact extractor + privacy-preserving
   episodic memory + observe/respond pipeline. Four-scenario demo
   PASSED all user-defined criteria. **This is the active line.**

## Where to start

```bash
git checkout main && git pull
# Verify environment:
source .venv/bin/activate
python -m pytest tests/agi/ -q          # 27 pass, 2 skipped (Qwen-load gated)
# Replay the demo (uses cached Qwen-0.5B; ~30s on CPU after first run):
python experiments/agi/phase_1_0_fact_retention.py
```

## Repo layout (post-pivot)

```
src/
├── agi/                                # NEW — Phase 1.0 onwards
│   ├── __init__.py                     # lazy: exports memory + extractor only
│   ├── foundation.py                   # FrozenFoundation (Qwen-0.5B wrapper)
│   ├── extraction.py                   # FactExtractor (regex, FR+EN)
│   ├── integration.py                  # AGISystem (observe/respond)
│   └── memory/
│       ├── __init__.py
│       └── xray_episodic.py            # XRayEpisodicMemory, EpisodicEntry
└── continual_synapse/                  # LEGACY — reference only

experiments/
├── agi/
│   ├── __init__.py
│   └── phase_1_0_fact_retention.py     # Phase 1.0 demo (PASS)
├── 01_*.py … 49_*.py                   # LEGACY CL experiments

tests/
├── agi/
│   ├── test_extraction.py              # 10 tests
│   ├── test_episodic_memory.py         # 10 tests
│   ├── test_integration.py             # 6 tests (uses MockFoundation)
│   └── test_foundation.py              # 3 tests; 2 gated by AGI_FOUNDATION_TESTS=1
└── test_*.py                           # LEGACY CL tests (still pass)

colab/
└── cifar100_cls_pilot.ipynb            # LEGACY (CIFAR pilot infra)
```

## Phase 1.0 — what landed

Commit `610c250`. Four-scenario demo on `Qwen/Qwen2-0.5B-Instruct`,
frozen, CPU (fp32) or CUDA (fp16):

| Scenario | Result |
|---|---|
| 1. Introduction `"Mon nom est Francois..."` | Extracted `{name: Francois, location: Montréal}`; memory size 1 |
| 2. **Retrieval test** `"Comment je m'appelle?"` | Assistant: **"Je m'appelle François."** ← memory_used=True, sim=0.816 ✓ |
| 3. Control (no memory) `"Comment je m'appelle?"` | Assistant deflects, does NOT fabricate ✓ |
| 4. Multi-fact retrieval + summary | Retrieval+merge worked; summarisation deflected (Qwen-0.5B quality limit, NOT mechanism failure) |

**Privacy verified programmatically**: every `EpisodicEntry` exposes
only `(key, facts, timestamp, access_count, creation_session)` — no
`raw_text`, `original_input`, `raw`, `samples`, or `utterance`
fields. The unit test suite asserts the same.

## Key architectural decisions made in Phase 1.0

1. **`src/agi/__init__.py` does NOT eagerly import `FrozenFoundation`
   or `AGISystem`.** That keeps pure-memory tests fast and offline.
   Import them explicitly: `from agi.foundation import FrozenFoundation`.

2. **Device + dtype auto-picked**: CUDA + fp16 when available, else
   CPU + fp32. Apple Silicon MPS NOT auto-selected (HF generation
   paths flaky on MPS for small models). Pass `device="mps"`
   explicitly to override.

3. **`transformers==4.57.1` was pinned in `pyproject.toml` already
   but not installed in the .venv** — fixed during 1.0 work.
   `dtype=` (not `torch_dtype=`) is the load kwarg.

4. **FactExtractor uses scoped inline `(?i:...)` flags** so trigger
   phrases match case-insensitively while place-name captures stay
   case-sensitive (avoids extracting common nouns as locations).

5. **`AGISystem._format_facts_as_context`** uses hand-crafted Qwen
   chat tokens (`<|im_start|>...`). A polish pass could switch to
   `tokenizer.apply_chat_template`; not blocking.

## Phase 1.1 — natural next steps (USER-DECIDED, not yet specified)

Phase 1.0 only proves the foundation + memory pipeline works on a
toy demo. The user has explicitly indicated this is a 9-12 month
project; Phase 1.0 is one milestone of many. Plausible Phase 1.1
directions (do NOT pre-commit to one — wait for user spec):

- **Foundation-LLM-driven fact extraction.** Replace the regex
  `FactExtractor` with prompting the foundation to extract structured
  facts from arbitrary text. Generalises beyond name/age/location.
- **Multi-session persistence.** Pickle the memory to disk between
  runs; reload at session start. Tests session_id behaviour the
  Phase 1.0 code already tracks via `new_session()`.
- **Negative-fact handling.** User says "I'm NOT from Paris" — the
  current extractor would falsely store `location=Paris`.
- **Larger foundation.** Qwen-0.5B is too small to summarise across
  retrieved facts (Scenario 4). Try Qwen-1.5B-Instruct or similar.
- **Phase 2.0** (per the user's roadmap): reward-modulated plasticity.

## CL legacy snapshot — preserve, don't extend

The legacy CL track ended with:
- **CLS-CI v2 on Split-MNIST CI** (n=10): NEO ACC = **0.861 ± 0.020**
  (commit `fa8b88c`). Matches/approaches DER (0.879).
- **CIFAR-100 CI pilot** (CLS-CI v2 + DER) stalled at ~0.17-0.19 final
  ACC across multiple lambda/lr/architecture pivots
  (commits `513c0fb` through `a19d157`). The Colab notebook
  `colab/cifar100_cls_pilot.ipynb` has the launchers.
- **Prototype-based XRay attempt** (Phase 5.7) failed on Split-MNIST
  CI at the same 0.193 plateau (commit `65059cb`). Root cause
  diagnosed in that commit message: prototypes go stale as the
  encoder drifts. The AGI pivot preserved the XRay name in the new
  `XRayEpisodicMemory` (which works because the foundation is frozen,
  so keys don't drift).

## Open todos (carried forward, lowest priority)

- **Suppress the `temperature/top_p/top_k may be ignored` warning**
  from Qwen generation when `do_sample=False`. Pass an explicit
  `GenerationConfig` to silence cleanly.
- **Optionally migrate `_format_facts_as_context`** to
  `tokenizer.apply_chat_template` for robustness across HF versions.

## Important — pyproject.toml / environment

- `transformers==4.57.1` is in `pyproject.toml` but is **NOT** in
  the default install path; if a fresh venv is set up, run
  `pip install transformers==4.57.1` explicitly (or run a full
  `pip install -e .[dev]` once the project is properly packaged).
- `scikit-learn==1.8.0` was added earlier for Phase 3.0 CL work.
  Phase 1.0 AGI does not depend on it.

## Most recent commits (for fast-forward)

```
610c250 feat: Phase 1.0 AGI architecture — Foundation + X-Ray Episodic Memory   ← HEAD
65059cb exp: Phase 5.7.2 — XRay validation on Split-MNIST CI (FAILURE, kept)
374c20a feat: Phase 5.7.1 — integrate X-Ray Memory into training pipeline
afc98ed feat: Phase 5.7.0 XRayMemory — prototype-based memory with refinement
fa8b88c exp: Phase 5.5.6 — CLS-CI v2 n=10 confirmation + ablation
41a114b exp: Phase 5.5.6 — CLS-CI v2 with interleaved replay
```

## Suite status

- **Legacy tests** (continual_synapse/): all pass.
- **AGI tests** (tests/agi/): 27 pass, 2 skipped (Qwen-load gated by
  `AGI_FOUNDATION_TESTS=1` env var).
- Combined: clean `pytest` run is fast (~5-10 s) when foundation
  tests are skipped.

---

*Generated by Claude Code (executor role) at the end of the Phase 1.0
session. The conversational-Claude handoff (separate file:
`results/CLAUDE_CONVERSATION_HANDOFF.md`, untracked) covers the
strategic discussion arc. This file covers the codebase state for
the next executor session.*
