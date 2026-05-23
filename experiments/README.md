# Experiments

Numbered experiments map to the phase-by-phase deliverables in
`PROJECT_PLAN.md`. Each script writes a JSON log to `results/logs/`
that records the config, accuracy matrix, and metrics so runs are
reproducible after the fact.

| Script | Method | Benchmark | Phase |
|---|---|---|---|
| `01_baseline_forgetting.py` | Naive sequential fine-tuning | Split-MNIST | 1 |
| `02_ewc_baseline.py` | EWC | Split-MNIST | 1 |
| `03_synapse_layer_v1.py` | Naive + SynapseLayer v1 (Hebbian, dense, fixed reward) | Split-MNIST | 2 |

## Running

From the repo root, after installing dependencies:

```bash
python experiments/01_baseline_forgetting.py
python experiments/02_ewc_baseline.py --lam 1000
python experiments/03_synapse_layer_v1.py --synapse-lr 1e-3
```

The first run downloads MNIST via HuggingFace `datasets` into
`data/hf_cache/`. Subsequent runs reuse the cache.

All scripts accept `--help` for the full list of hyperparameters.
Defaults match the Phase-1 numbers used in `decisions_log.md`.

## Output format

Each run produces `results/logs/<unix_ts>_<experiment>_<method>.json`
with the following schema:

```json
{
  "experiment": "01_baseline_forgetting",
  "method": "naive_finetune",
  "benchmark": "split_mnist",
  "timestamp": 1716000000,
  "git_sha": "abc123...",
  "config": { ... CLI args ... },
  "task_names": ["split_mnist_01", ...],
  "accuracy_matrix": [[0.99, null, ...], ...],
  "random_baseline": [0.5, 0.5, 0.5, 0.5, 0.5],
  "metrics": {
    "average_accuracy": 0.65,
    "average_forgetting": 0.20,
    "backward_transfer": -0.18,
    "forward_transfer": 0.05,
    "per_task_final": {"split_mnist_01": 0.40, ...}
  }
}
```

`null` entries are the zero-shot positions the runner did not
record (the upper triangle above ``R[i, i+1]``).
