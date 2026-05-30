# Design: Additive Synapse Layer for Continual Learning

> An exploratory research engineering project investigating whether a synapse-inspired memory layer added on top of pre-trained transformers can address catastrophic forgetting in continual learning settings.

This document describes the technical design, architecture, evaluation methodology, and references for the project. The implementation is built incrementally, with each phase producing a working artifact that can be evaluated against existing methods.

## 1. Motivation

Standard transformer-based language models do not learn after training is complete. When fine-tuned sequentially on new tasks, they suffer from catastrophic forgetting: the weights that encoded previous knowledge get overwritten by gradient updates on new data. This is a direct consequence of how knowledge is stored in dense neural networks.

The biological brain does not have this problem. It uses synapses, which are physical structures with their own state and local learning rules, sitting on top of a neural substrate. Synapses adjust based on what they directly observe, not via a global error signal. Different memories use mostly disjoint populations of neurons, so new learning does not destroy old learning.

This project explores whether some of these properties can be replicated as an additive layer on top of an existing pre-trained transformer, without modifying the underlying model. The goal is not to outperform existing methods but to build a credible, well-evaluated implementation that contributes to the public conversation about continual learning architectures.

The original architectural sketch is described in the article *What If We're Trying to Make an Orange Out of an Apple? Notes on AI Memory From a Curious Outsider*.

## 2. Design principles

1. **Additive, not replacement.** The pre-trained transformer stays frozen. The synapse layer sits on top, observes activations, accumulates state, and modulates outputs.
2. **Reproduce baselines first, innovate second.** Phase 1 reproduces known methods (EWC, naive fine-tuning). Innovation builds on verified baselines.
3. **Small models for fast iteration.** DistilBERT or GPT-2 small are sufficient for development. Larger models come only after the system demonstrably works on smaller ones.
4. **Honest evaluation over impressive claims.** Multiple seeds, statistical significance tests, comparison to existing methods. Negative results are reported as such.
5. **Sparse representations where possible.** Top-k partner selection per neuron to avoid quadratic memory growth.

## 3. Architecture

### 3.1 System overview

The system consists of three layers in addition to the base model:

```
┌─────────────────────────────────────────────────┐
│         COLD STORAGE LAYER                       │
│  Vector DB + progressive compression             │
│  Long-term archive of consolidated patterns      │
└─────────────────────────────────────────────────┘
              ↕ consolidate / retrieve
┌─────────────────────────────────────────────────┐
│         SYNAPSE LAYER                            │
│  Per-connection state with sparse top-k          │
│  Hebbian updates modulated by reward             │
│  Confidence and evidence tracking                │
└─────────────────────────────────────────────────┘
              ↑ observe          ↓ modulate
┌─────────────────────────────────────────────────┐
│         BASE MODEL (frozen)                      │
│  Pre-trained transformer                         │
│  PyTorch hooks capture activations               │
└─────────────────────────────────────────────────┘
              ↑ input
```

### 3.2 Synapse layer

The core contribution. Hooks into a chosen layer of the base model (typically the final hidden state), maintains per-connection state, and produces a correction vector that modulates the model's output.

**Per-connection state:**

| Field | Type | Purpose |
|---|---|---|
| `strength` | float32 | The learned weight, updated by Hebbian rule |
| `confidence` | float32 | How much evidence supports this connection |
| `accumulated_evidence` | float32 | Sum of co-activation magnitudes seen |
| `age` | int64 | Number of update cycles since creation |
| `access_count` | int64 | Number of times this connection contributed to output |

**Sparse representation:** for an `n`-neuron layer, full connectivity would be `n²` connections, which is wasteful. Each neuron maintains only its top-`k` strongest partners (k=64 initially). When a new co-activation exceeds the weakest current partner, eviction occurs.

**Update rule:**

```
Δw_ij = η · R · a_i · a_j / (1 + β · evidence_ij)
```

Where:
- `η` is the learning rate
- `R` is the reward signal (see 3.4)
- `a_i, a_j` are the activations of neurons i and j
- `evidence_ij` is the accumulated evidence on this connection
- `β` controls how strongly accumulated evidence resists further change

**Multi-pass co-activation:** because transformer activations are continuous-valued and noisy, co-activations are computed across multiple inference passes of the same input. Pairs that consistently fire together across passes are treated as co-activated. Single-pass noise is filtered out.

**Output modulation:** the synapse layer outputs a correction vector that gets added to the base model's hidden state before the LM head. The correction is initialized near zero so the base model's behavior is preserved at initialization.

### 3.3 Cold storage layer

Long-term archive for consolidated synapse states. When the synapse layer's working memory saturates, important patterns are transferred to cold storage and the source synapses are partially drained, freeing them to capture new experience.

**Implementation:** local-first vector database (Chroma) for development. Each archived cluster is stored as an embedding plus metadata (age, strength sum, evidence sum, access count).

**Compression strategy:** progressive quantization based on age and access frequency. Recently archived clusters stay at full precision. Older clusters that are rarely accessed get progressively quantized: 32-bit → 16-bit → 8-bit → 4-bit.

**Retrieval:** when current activations suggest information that may live in cold storage, the system queries the vector database with the activation pattern. The top-k matches are retrieved, decompressed, and used to partially reconstruct synapse state. Reconstruction is approximate by design.

### 3.4 Reward signal

The signal that decides which co-activations to consolidate. Multi-component, with a developmental trajectory that shifts weighting over the system's lifetime.

**Components:**
- `external_reward`: explicit feedback from training signal or user
- `consistency_reward`: cosine similarity between current activations and exponential moving average of past activations
- `surprise_signal`: prediction error from a small auxiliary model

**Mixing function with developmental trajectory:**

```
R(t) = α(t) · R_external + (1 - α(t)) · (w_c · R_consistency + w_s · R_surprise)
```

Where `α(t)` decreases as validated evidence accumulates:

```
α(t) = 1.0 / (1.0 + γ · validated_evidence(t))
```

Early in the system's life, external feedback dominates. As the system accumulates validated experience, internal signals (consistency, surprise) gain weight. This mirrors human cognitive development from external validation toward intellectual autonomy.

### 3.5 Consolidation trigger

A pressure metric tracks which synapses are candidates for archival:

```
pressure_ij = (strength_ij × evidence_ij) / (1 + access_count_ij)
```

High-strength, high-evidence, rarely-accessed synapses score highest. When the average pressure across all synapses exceeds a threshold, a consolidation cycle is triggered.

**During consolidation:**
1. Identify synapses with pressure above threshold
2. Cluster them by activation signature (k-means)
3. Compute cluster embeddings
4. Insert into cold storage with metadata
5. Drain source synapses (reset strength, evidence, access counters)

## 4. Implementation roadmap

The project is built in six phases. Each phase produces a working artifact and adds capability incrementally.

### Phase 1 — Foundation

Reproduce known continual learning baselines. Establish the evaluation harness.

- Set up Split-MNIST benchmark
- Implement naive sequential fine-tuning baseline
- Implement EWC (Elastic Weight Consolidation) baseline
- Verify reproductions against published results
- Build the continual learning evaluation runner

### Phase 2 — Basic synapse layer

Implement v1 of the SynapseLayer with minimal state.

- PyTorch hook utilities to capture activations
- SynapseLayer with `strength` state only
- Basic Hebbian update (fixed reward)
- Output modulation via correction vector
- Integration with small MLP and DistilBERT

### Phase 3 — Confidence and resistance

Add the metacognitive components.

- Extend state to include confidence, evidence, age, access count
- Implement evidence-based resistance to revision
- Implement reward signal system (external, consistency, surprise)
- Implement developmental trajectory mixing
- Add sparse top-k partner selection

### Phase 4 — Cold storage and consolidation

Add the long-term archive and consolidation cycles.

- Set up Chroma local instance
- Implement compression pipeline (progressive quantization)
- Implement consolidation trigger (pressure threshold)
- Implement consolidation pipeline (drain to storage)
- Implement reconstructive retrieval

### Phase 5 — Rigorous evaluation

Compare against existing methods on multiple benchmarks.

- Reproduce Differentiable Plasticity baseline
- Reproduce Experience Replay baseline
- Run evaluation matrix: 4 benchmarks × 5 methods × 5 seeds
- Compute standard CL metrics (ACC, FGT, BWT, FWT)
- Statistical significance tests (Wilcoxon signed-rank with Bonferroni)
- Generate publication-quality figures and tables

### Phase 6 — Documentation and release

Polish for public release.

- README with installation, quickstart, results summary
- Demo notebooks (4 total)
- Comprehensive docstrings and type hints
- Demo video
- Follow-up article documenting findings

## 5. Evaluation methodology

### 5.1 Benchmarks

Standard continual learning benchmarks used:

| Benchmark | Tasks | Modality |
|---|---|---|
| Split-MNIST | 5 binary classification tasks | Vision (toy) |
| Permuted-MNIST | 10 tasks with permuted pixels | Vision (toy) |
| Split-CIFAR-10 | 5 binary classification tasks | Vision |
| Split-AG-News | 4 topic classification tasks | Text |

### 5.2 Metrics

For a sequence of `T` tasks with `a_{i,j}` = accuracy on task `j` after training through task `i`:

- **Average Accuracy (ACC):** mean of `a_{T,j}` for j = 1 to T
- **Average Forgetting (FGT):** mean of `(max_i a_{i,j} - a_{T,j})` for j = 1 to T-1
- **Backward Transfer (BWT):** mean of `(a_{T,j} - a_{j,j})` for j = 1 to T-1
- **Forward Transfer (FWT):** mean of `(a_{j-1,j} - random_accuracy_j)` for j = 2 to T

### 5.3 Statistical significance

- Minimum 5 random seeds per (method, benchmark) combination
- Report mean ± standard deviation
- Wilcoxon signed-rank test for paired comparisons
- Bonferroni correction for multiple comparisons
- Explicit significance levels (p < 0.05)

### 5.4 Reporting standards

Three commitments:
1. If a method does not significantly outperform a baseline on a metric, state so explicitly
2. Report variance, not just means
3. Report all benchmarks tested, not just favorable ones

## 6. Tech stack

```
# Core dependencies (pinned)
torch==2.4.0
transformers==4.45.0
datasets==3.0.0
chromadb==0.5.5
numpy==1.26.4
scipy==1.13.0
matplotlib==3.9.0
seaborn==0.13.0
pytest==8.3.0
jupyter==1.0.0
tqdm==4.66.0
```

### 6.1 Base model choices by phase

| Phase | Recommended base model | Reason |
|---|---|---|
| Phase 1 | Small MLP (3 layers) | Fast iteration on MNIST |
| Phase 2-4 | DistilBERT-base (66M) | Standard, fast on consumer GPU |
| Phase 5 | GPT-2 medium (355M) | Larger scale validation |
| Optional later | Llama-3 1B or 3B | LLM-scale validation |

### 6.2 Hardware requirements

- Minimum: 16GB RAM, CPU-only (slow but workable for early phases)
- Recommended: Single GPU with 12GB+ VRAM
- Cloud option: Google Colab Pro, Lambda Labs, vast.ai

## 7. Repository structure

```
continual-synapse-layer/
├── README.md
├── DESIGN.md (this document)
├── requirements.txt
├── LICENSE
│
├── src/continual_synapse/
│   ├── synapse_layer/
│   │   ├── layer.py
│   │   ├── update_rules.py
│   │   ├── topk.py
│   │   └── modulation.py
│   │
│   ├── cold_storage/
│   │   ├── store.py
│   │   ├── compression.py
│   │   └── retrieval.py
│   │
│   ├── consolidation/
│   │   ├── trigger.py
│   │   ├── pipeline.py
│   │   └── reconstruction.py
│   │
│   ├── reward/
│   │   ├── external.py
│   │   ├── consistency.py
│   │   ├── surprise.py
│   │   └── mixer.py
│   │
│   ├── base_models/
│   │   ├── loaders.py
│   │   └── hooks.py
│   │
│   ├── baselines/
│   │   ├── naive_finetune.py
│   │   ├── ewc.py
│   │   ├── replay.py
│   │   └── diff_plasticity.py
│   │
│   └── evaluation/
│       ├── benchmarks.py
│       ├── metrics.py
│       ├── runner.py
│       └── statistics.py
│
├── experiments/
│   ├── 01_baseline_forgetting.py
│   ├── 02_ewc_baseline.py
│   ├── 03_synapse_layer_v1.py
│   ├── 04_synapse_with_resistance.py
│   ├── 05_with_cold_storage.py
│   ├── 06_full_system.py
│   └── 07_comparisons.py
│
├── notebooks/
│   ├── 01_demo_catastrophic_forgetting.ipynb
│   ├── 02_synapse_layer_walkthrough.ipynb
│   ├── 03_cold_storage_visualization.ipynb
│   └── 04_results_analysis.ipynb
│
├── tests/
│
├── docs/
│
└── results/
    ├── figures/
    ├── tables/
    └── logs/
```

## 8. Key references

The following papers establish the foundation and provide baselines. Worth reading before contributing or evaluating the work.

1. **Kirkpatrick et al. (2017).** *Overcoming catastrophic forgetting in neural networks.* PNAS. Establishes EWC, foundational baseline.
2. **Miconi, Stanley, Clune (2018).** *Differentiable Plasticity: Training Plastic Neural Networks with Backpropagation.* ICML. Closest existing work to the additive synapse layer concept.
3. **McClelland, McNaughton, O'Reilly (1995).** *Why There Are Complementary Learning Systems in the Hippocampus and Neocortex.* Psychological Review. The CLS theory that inspires the two-system design.
4. **Ba, Hinton, Mnih, Leibo, Ionescu (2016).** *Using Fast Weights to Attend to the Recent Past.* NeurIPS. Fast/slow weights distinction.
5. **Parisi, Kemker, Part, Kanan, Wermter (2019).** *Continual Lifelong Learning with Neural Networks: A Review.* Neural Networks. Field overview.
6. **Hadsell, Rao, Rusu, Pascanu (2020).** *Embracing Change: Continual Learning in Deep Neural Networks.* Nature Reviews. Recent perspective.

## 9. Status and contributions

This is an exploratory research engineering project by a non-academic author. It is not intended as a publication-ready paper. Pull requests, issues, and critical feedback are welcome.

The implementation prioritizes:
- Clarity of code over performance optimization
- Reproducibility over novelty claims
- Honest reporting of results over impressive numbers

For background on the motivation and the original architectural sketch, see the article *What If We're Trying to Make an Orange Out of an Apple? Notes on AI Memory From a Curious Outsider*.

## 10. License

MIT License. See `LICENSE` file.
