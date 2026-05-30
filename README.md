# Brain-Aligned Learning Substrate

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A research investigation into what an artificial system must look like
if it is to learn the way biological brains learn.

## Status

Phase 2a complete — Pass entity emergence verified on the P2 test
(100 % selectivity on pattern pairs in a 200-N substrate).

The full theoretical framework, hypotheses, and predictions live in
[**THEORY.md**](THEORY.md).

## What this is

A pure research project investigating five hypotheses about
brain-aligned learning:

- **H1**: Unified substrate — computation and memory live in the same
  fabric, not in separate "model" + "database" modules.
- **H2**: Local learning rules — weights update from quantities
  available at the synapse, not from a global error signal.
- **H3**: Forward-pass-as-learning — every inference modifies the
  substrate; there is no separate "train" vs "infer" mode.
- **H4**: Metastable background dynamics — the substrate is never
  fully silent, even without external input.
- **H5**: Sparse distributed representations — at any moment, only a
  small fraction (≈ 1–5 %) of the substrate is active, structurally
  enforced via k-winners-take-all.

The investigation is documented in `THEORY.md`. Active implementation
lives in `src/substrate/`. Experimental validation lives in
`experiments/substrate/`.

## What this is NOT

This project deliberately excludes:

- Pretrained LLMs as substrate
- Backpropagation
- External vector databases
- RAG-style retrieval-augmented patterns
- Standard fine-tuning or LoRA adapters

These exclusions are justified in `THEORY.md` Section 8.

## Repository structure

```
THEORY.md                    Canonical theory document
README.md                    This file

src/substrate/               Active implementation (Phase 1–2a)
experiments/substrate/       Active experiments
tests/substrate/             Active tests (65 green)
results/substrate/           Experimental outputs

src/legacy/                  Archived: prior research directions
experiments/legacy/          Archived: prior experiments
tests/legacy/                Archived: prior tests (skipped by pytest)
colab/legacy/                Archived: prior Colab notebooks
docs/archived/               Archived: prior documentation
```

The `legacy/` trees preserve two earlier directions — a
continual-learning study (`src/legacy/continual_synapse/`) and an
AGI-architecture study (`src/legacy/agi/`) — for history and future
writeups. They are not part of the active investigation.

## Running

```bash
pip install -e .
pytest tests/substrate/

# Phase 1 — pattern formation through repeated exposure:
python experiments/substrate/phase_1_pattern_formation.py

# Phase 2a — P (Pass) entity emergence from co-activated N pairs:
python experiments/substrate/phase_2a_emergence.py
```

The substrate is pure NumPy; no PyTorch / transformers dependency
in the active code path.

## Probability of success

Honest assessment in `THEORY.md` Section 9. The project is high-risk
fundamental research: most plausible outcomes are failure modes. The
investigation itself — the careful articulation of what brain-aligned
learning would require, and the empirical examination of each
prediction — is the value, independent of whether the substrate
ultimately scales.

## License

See [LICENSE](LICENSE).
