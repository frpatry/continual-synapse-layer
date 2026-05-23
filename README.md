# continual-synapse-layer

[![tests](https://github.com/frpatry/continual-synapse-layer/actions/workflows/tests.yml/badge.svg)](https://github.com/frpatry/continual-synapse-layer/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An exploratory research project investigating whether an additive,
synapse-inspired memory layer on top of a frozen pre-trained model
can reduce catastrophic forgetting in continual-learning settings.
The system observes activations from a frozen base model, maintains
per-connection state with sparse top-k partner selection and
evidence-based resistance to change, and emits a correction vector
that modulates the base model's output. See [DESIGN.md](DESIGN.md)
for the technical specification and [PROJECT_PLAN.md](PROJECT_PLAN.md)
for the six-phase roadmap.

**Status:** Phase 1 in progress — Split-MNIST benchmark, MLP
baseline, sequential evaluation harness, ACC/forgetting/BWT/FWT
metrics, and EWC reproduction are in place. The synapse layer
itself is Phase 2 work. See [decisions_log.md](decisions_log.md)
for the running design log.

## Installation

```bash
git clone https://github.com/frpatry/continual-synapse-layer.git
cd continual-synapse-layer
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.13. See [PROJECT_PLAN.md §5.2](PROJECT_PLAN.md)
for hardware recommendations. CPU is sufficient for Phase 1.

## Quickstart

Reproduce the catastrophic-forgetting baseline on Split-MNIST:

```bash
python experiments/01_baseline_forgetting.py
```

The script downloads MNIST via HuggingFace `datasets` on first run,
trains a 3-layer MLP sequentially on five binary digit-pair tasks,
prints all four standard continual-learning metrics, and writes a
JSON log to `results/logs/`. Expect ~50% average forgetting with the
default hyperparameters.

Compare against Elastic Weight Consolidation:

```bash
python experiments/02_ewc_baseline.py --lam 1000
```

See [experiments/README.md](experiments/README.md) for the full
list of scripts, hyperparameters, and the JSON log schema.

## Repository layout

```
src/continual_synapse/
├── synapse_layer/     # Additive layer (Phase 2+)
├── cold_storage/      # Long-term archive (Phase 4)
├── consolidation/     # Synapse → storage transfer (Phase 4)
├── reward/            # Reward signal mixer (Phase 3)
├── base_models/       # HF loaders + hook helpers (Phase 2+)
├── baselines/         # naive_finetune, ewc, replay, diff_plasticity
└── evaluation/        # benchmarks, metrics, runner, reporting
experiments/           # Numbered runnable experiments
tests/                 # pytest suite (Phase 1: 35 tests)
results/{logs,figures,tables}/  # Generated artifacts
```

## Development

```bash
pip install -r requirements.txt
python -m pytest tests/
```

The test suite runs in under two seconds on CPU and has no network
or dataset dependencies — synthetic tensors stand in for MNIST. CI
runs the same suite on every push.

## License

MIT — see [LICENSE](LICENSE).

## References

- Kirkpatrick et al. 2017 — *Overcoming catastrophic forgetting in
  neural networks* (EWC).
- Miconi et al. 2018 — *Differentiable Plasticity* (Phase 5 baseline).
- McClelland, McNaughton, O'Reilly 1995 — *Why there are
  complementary learning systems* (biological inspiration).
- Parisi et al. 2019 — *Continual Lifelong Learning with Neural
  Networks: A Review*.

See [PROJECT_PLAN.md §13](PROJECT_PLAN.md) for the full reading
list with notes.
