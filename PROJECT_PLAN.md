# Project Plan: Additive Synapse Layer for Continual Learning

**Author:** Francois Patry
**Status:** Specification — pre-implementation
**Target completion:** 6 months
**Repository name (suggested):** `continual-synapse-layer`

---

## 0. How to use this document

This document is the master specification for a research engineering project. It is written to be given to Claude Code at the start of the project and referenced throughout. It contains:

- The mission and success criteria
- A complete technical architecture specification
- A six-phase implementation roadmap with concrete deliverables
- The exact tech stack with version recommendations
- The repository structure
- Evaluation methodology
- Suggested Claude Code prompts at each phase
- Risk management and decision logging guidance

When working with Claude Code, point it at this document at the start of every session. Add phase-specific notes to a `decisions_log.md` as the project evolves so context persists across sessions.

---

## 1. Mission

Build a working proof-of-concept of an additive synapse layer that sits on top of an existing pre-trained transformer, observes its activations, accumulates state over time, and demonstrably reduces catastrophic forgetting on standard continual learning benchmarks.

The goal is not to produce a publishable paper. The goal is to produce a credible public artifact (working code + measurable results + clear documentation) that demonstrates serious engineering and research capability.

## 2. Success criteria

The project is considered successful if all of the following are true at month 6:

1. **Public GitHub repository** with clean code, full README, runnable demos, and a permissive open-source license.
2. **Reproducible results** on at least one standard continual learning benchmark (Split-MNIST or Split-CIFAR-10), with the proposed system measurably reducing average forgetting compared to a naive fine-tuning baseline.
3. **Comparison to at least two existing methods** (recommended: EWC and Differentiable Plasticity), with rigorous methodology even if the proposed system does not outperform them on every metric.
4. **At least one Jupyter notebook** that walks through the system end to end and produces visualizations.
5. **A follow-up article** (~2000-3000 words) documenting the findings honestly, including negative results if applicable.
6. **A demo video** (5-10 minutes) walking through the system.

Note: Outperforming existing methods is not a success criterion. Producing rigorous, honest results is. Negative results that are well-documented are still a contribution.

## 3. Project philosophy

- **Ship something honest, not something perfect.** A working system with modest results beats a vaporware perfect system.
- **Document as you go.** Every architectural decision goes in `decisions_log.md` with the reasoning. Future-you and Claude Code need this context.
- **Reproduce baselines first, innovate second.** Spend the first two months proving you can reproduce existing methods. Innovation built on shaky baselines is worthless.
- **Smaller models, faster iteration.** Use the smallest models that demonstrate the concept. DistilBERT or GPT-2 small beats Llama 3 8B for prototyping.
- **Evaluation rigor is non-negotiable.** Bad evaluation makes a great system look mediocre, and a mediocre system look great. Get this right.

---

## 4. Technical architecture

### 4.1 High-level system diagram

```
┌─────────────────────────────────────────────────┐
│         COLD STORAGE LAYER                       │
│  Vector DB (Chroma) + Compression                │
│  - Long-term archive of consolidated patterns    │
│  - Progressive quantization (32-bit → 4-bit)    │
└─────────────────────────────────────────────────┘
              ↕ consolidate / retrieve
┌─────────────────────────────────────────────────┐
│         SYNAPSE LAYER (the new contribution)     │
│  Per-connection state:                           │
│  - strength (float32)                            │
│  - confidence (float32)                          │
│  - accumulated_evidence (float32)                │
│  - age (int64)                                   │
│  - access_count (int64)                          │
│                                                  │
│  Updates: Δw = η · R · a_i · a_j / (1 + β·E)    │
│  Modulates: base_output += correction(state)     │
└─────────────────────────────────────────────────┘
              ↑ observe          ↓ modulate
┌─────────────────────────────────────────────────┐
│         BASE MODEL (frozen)                      │
│  Pre-trained transformer (DistilBERT or GPT-2)  │
│  PyTorch hooks capture layer activations         │
└─────────────────────────────────────────────────┘
              ↑ input
```

### 4.2 Component specifications

#### 4.2.1 SynapseLayer

**Purpose:** The core of the system. Sits between the base model and the final output. Observes activations via hooks, maintains state per connection, modulates output.

**Key design decisions:**

- **Sparse representation.** For an n-neuron layer, full connectivity is n² connections. For BERT-base (768 hidden), this is 590k synapses per layer. Manageable in memory but wasteful. Use top-k selection: each neuron maintains synapses only to its top-k most co-activated partners (k=64 to start).
- **Hook target.** Hook into the final hidden state of the base model (output of the last transformer block, before the LM head). This keeps the implementation simple and focused.
- **Update timing.** Synapse updates happen at end of each input sequence, not per-token. This batches the work and enables multi-pass averaging.
- **Output modulation.** The synapse layer outputs a *correction vector* that gets added to the base model's hidden state before the LM head. Initially small (initialized near zero) so the base model's behavior is preserved.

**Pseudo-class structure:**

```python
class SynapseLayer(nn.Module):
    """
    Additive synapse layer for continual learning.

    Observes activations from a base model layer via PyTorch hooks.
    Maintains per-connection state across calls.
    Produces a correction vector that modulates the base model output.
    """
    def __init__(
        self,
        n_neurons: int,
        top_k: int = 64,
        learning_rate: float = 0.01,
        resistance_beta: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()
        # Sparse state, stored as edge lists
        # For each neuron i, top-k partner indices and their strengths
        self.register_buffer("partner_indices", torch.zeros(n_neurons, top_k, dtype=torch.long))
        self.register_buffer("strengths", torch.zeros(n_neurons, top_k))
        self.register_buffer("confidence", torch.zeros(n_neurons, top_k))
        self.register_buffer("evidence", torch.zeros(n_neurons, top_k))
        self.register_buffer("ages", torch.zeros(n_neurons, top_k, dtype=torch.long))
        # Buffer for multi-pass averaging
        self.activation_buffer = []
        self.global_step = 0

    def observe(self, activations: torch.Tensor):
        """Called via hook. Adds to buffer for later consolidation."""
        self.activation_buffer.append(activations.detach())

    def consolidate(self, reward: float):
        """Process accumulated buffer, update synapse state."""
        # Average activations across buffer for multi-pass detection
        avg_acts = torch.stack(self.activation_buffer).mean(dim=0)
        # Compute co-activations (top-k partners per neuron)
        co_acts = self._compute_topk_coactivations(avg_acts)
        # Apply Hebbian update modulated by reward and resistance
        resistance = 1.0 / (1.0 + self.resistance_beta * self.evidence)
        delta = self.learning_rate * reward * co_acts * resistance
        self.strengths += delta
        self.evidence += co_acts.abs()
        self.ages += 1
        self.activation_buffer = []
        self.global_step += 1

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Compute correction vector from current state."""
        correction = self._compute_correction(hidden_state)
        return hidden_state + correction
```

This skeleton evolves through phases. Initial implementation in Phase 2 is simpler than this. The full version arrives in Phase 4.

#### 4.2.2 Cold Storage System

**Purpose:** Long-term archive for consolidated synapse states. Stores compressed versions, supports approximate retrieval.

**Implementation choice:** Use [Chroma](https://www.trychroma.com/) for local development. It's a local-first vector database, no infrastructure required.

**Compression strategy:** Each archived synapse cluster gets stored as a vector embedding. Precision reduction over time using quantization: 32-bit → 16-bit → 8-bit → 4-bit. Triggered by age and access frequency.

**Retrieval mechanism:** When the synapse layer needs to access information that was offloaded, query Chroma by current activation pattern. Return top-k most similar archived clusters. Reconstruct partial state in the synapse layer.

#### 4.2.3 Consolidation Trigger

**Purpose:** Decide when to flush synapse state to cold storage.

**Trigger condition:** Track a `consolidation_pressure` metric per synapse. When the average pressure across all synapses exceeds a threshold, trigger consolidation cycle.

**Pressure formula:**
```
pressure_ij = (strength_ij × evidence_ij) / (1 + access_count_ij)
```

High-strength, high-evidence, rarely-accessed synapses are prime candidates for archival.

**During consolidation:**
1. Identify synapses with pressure > threshold
2. Bundle into clusters (k-means on activation patterns that triggered them)
3. Compute cluster embedding
4. Insert into Chroma with metadata (age, strength sum, evidence sum)
5. Drain the original synapses (set strength to small value, reset evidence)

#### 4.2.4 Reward Signal System

**Purpose:** Provide the modulation signal that decides which co-activations to consolidate.

**Components:**
- `external_reward`: Explicit user feedback (thumbs up/down style). Default 0 if not provided.
- `consistency_reward`: Measure of agreement between new pattern and existing knowledge. Computed as cosine similarity between current activations and weighted average of past activations.
- `surprise_signal`: Measure of how much the current pattern deviates from predictions. Computed via simple predictive model.

**Mixing function:**
```
R(t) = α(t) · R_external + (1 - α(t)) · (w_c · R_consistency + w_s · R_surprise)
```

Where `α(t)` is the developmental trajectory parameter. Starts at 1.0 (fully external), decreases over time as `validated_evidence` accumulates:

```
α(t) = 1.0 / (1.0 + γ · validated_evidence(t))
```

`validated_evidence(t)` is the count of times external reward confirmed an internal signal.

## 5. Tech stack

### 5.1 Required dependencies

Pin specific versions to avoid breakage. Updated for current ecosystem state:

```
# requirements.txt
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

### 5.2 Hardware requirements

- **Minimum:** Any modern laptop with 16GB RAM. Use CPU for initial development. Painful but workable.
- **Recommended:** Single GPU with 12GB+ VRAM. RTX 3060/4060 sufficient for development with DistilBERT-class models.
- **Cloud option:** Google Colab Pro (~$10/month) gives access to T4 GPUs. Sufficient for the entire project. Lambda Labs or vast.ai for stronger GPUs as needed.

### 5.3 Base model recommendations

Use smaller models for faster iteration:

| Phase | Recommended base | Why |
|---|---|---|
| Phase 1 (baseline reproduction) | Small MLP, ~3 hidden layers | Fast iteration on MNIST |
| Phase 2-4 (synapse layer dev) | DistilBERT-base (66M params) | Standard NLP model, fast on consumer GPU |
| Phase 5 (extended testing) | GPT-2 medium (355M params) | Larger scale validation |
| Optional later | Llama-3 1B or 3B | If you want to claim LLM-scale results |

Avoid larger models until the system actually works on smaller ones.

---

## 6. Repository structure

```
continual-synapse-layer/
├── README.md                          # Public-facing intro, install, quickstart
├── PROJECT_PLAN.md                    # This document
├── ARCHITECTURE.md                    # Detailed technical specification
├── decisions_log.md                   # Running log of design decisions
├── LICENSE                            # MIT or Apache 2.0
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── src/
│   └── continual_synapse/
│       ├── __init__.py
│       ├── synapse_layer/
│       │   ├── __init__.py
│       │   ├── layer.py               # Main SynapseLayer module
│       │   ├── update_rules.py        # Hebbian + reward modulation
│       │   ├── topk.py                # Sparse top-k selection
│       │   └── modulation.py          # Output correction logic
│       │
│       ├── cold_storage/
│       │   ├── __init__.py
│       │   ├── store.py               # Chroma interface
│       │   ├── compression.py         # Quantization functions
│       │   └── retrieval.py           # Approximate recall
│       │
│       ├── consolidation/
│       │   ├── __init__.py
│       │   ├── trigger.py             # Pressure-based trigger
│       │   ├── pipeline.py            # Synapse → storage transfer
│       │   └── reconstruction.py      # Storage → synapse retrieval
│       │
│       ├── reward/
│       │   ├── __init__.py
│       │   ├── external.py            # User feedback wrapper
│       │   ├── consistency.py         # Internal consistency signal
│       │   ├── surprise.py            # Prediction error signal
│       │   └── mixer.py               # Developmental trajectory α(t)
│       │
│       ├── base_models/
│       │   ├── __init__.py
│       │   ├── loaders.py             # HF model loading utilities
│       │   └── hooks.py               # PyTorch hook helpers
│       │
│       ├── baselines/
│       │   ├── __init__.py
│       │   ├── naive_finetune.py      # Sequential fine-tuning
│       │   ├── ewc.py                 # Elastic Weight Consolidation
│       │   ├── replay.py              # Experience replay
│       │   └── diff_plasticity.py     # Differentiable Plasticity
│       │
│       └── evaluation/
│           ├── __init__.py
│           ├── benchmarks.py          # Split-MNIST, Permuted-MNIST, Split-CIFAR
│           ├── metrics.py             # Accuracy, forgetting, BWT, FWT
│           ├── runner.py              # Orchestration for continual eval
│           └── statistics.py          # Significance tests
│
├── experiments/
│   ├── README.md                      # How to reproduce each experiment
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
│   ├── __init__.py
│   ├── test_synapse_layer.py
│   ├── test_update_rules.py
│   ├── test_cold_storage.py
│   ├── test_consolidation.py
│   └── test_reward_mixer.py
│
├── docs/
│   ├── installation.md
│   ├── quickstart.md
│   ├── architecture_overview.md
│   ├── evaluation_methodology.md
│   └── glossary.md
│
├── results/
│   ├── figures/                       # Generated plots
│   ├── tables/                        # Generated CSV tables
│   └── logs/                          # Experiment run logs
│
└── scripts/
    ├── download_datasets.py
    ├── run_full_evaluation.sh
    └── generate_paper_figures.py
```

---

## 7. Six-phase implementation roadmap

Each phase has: goal, specific tasks, deliverables, checkpoint criteria, estimated time, and a suggested Claude Code initial prompt.

### Phase 1 — Foundation (Weeks 1-8, ~4-8 weeks)

**Goal:** Set up the development environment, reproduce known baselines, prove you can measure catastrophic forgetting.

**Tasks:**

1. Initialize Git repository, set up Python environment with pinned versions
2. Implement Split-MNIST benchmark
   - 5 sequential tasks, each a binary classification on a pair of digits
   - Standard split: (0,1), (2,3), (4,5), (6,7), (8,9)
3. Implement a small MLP (3 hidden layers, ReLU)
4. Implement the continual learning evaluation harness
   - Train on each task sequentially
   - After each task, evaluate on all previously seen tasks
   - Compute: per-task accuracy, average accuracy, average forgetting, backward transfer
5. Run baseline: naive sequential fine-tuning
   - Document the catastrophic forgetting clearly (plots)
6. Implement EWC (Elastic Weight Consolidation) as a baseline
   - Reference: Kirkpatrick et al. 2017
   - Compute Fisher Information matrix, add quadratic penalty
7. Run EWC, compare to naive baseline
8. Set up basic CI (pytest on push)

**Deliverables:**
- Repo skeleton with all directories
- Working Split-MNIST evaluation
- Baseline naive fine-tuning results documented in a notebook
- EWC reproduction with results within 5% of published numbers
- First version of README with what the project is about

**Checkpoint criteria (end of Phase 1):**
- Can you run `python experiments/01_baseline_forgetting.py` and get a clean output showing catastrophic forgetting?
- Can you run `python experiments/02_ewc_baseline.py` and see EWC reduce forgetting?
- Does `pytest` pass on the foundational utilities?

If any of these is no, do not move to Phase 2.

**Suggested Claude Code prompt to start Phase 1:**

```
I am starting a research engineering project to build an additive
synapse layer for continual learning. We are in Phase 1 of the
PROJECT_PLAN.md (read it for full context).

Goal for this session: set up the repository structure, install
dependencies with pinned versions, and implement Split-MNIST
benchmark with a simple MLP baseline.

Specifically:
1. Create the directory structure as specified in PROJECT_PLAN.md
   section 6
2. Set up requirements.txt with the versions in section 5.1
3. Implement src/continual_synapse/evaluation/benchmarks.py with
   Split-MNIST as the first benchmark
4. Implement a simple 3-layer MLP baseline
5. Implement src/continual_synapse/evaluation/runner.py that trains
   on tasks sequentially and evaluates after each
6. Write tests for these components

Use type hints throughout. Add docstrings. Keep functions small
and testable. Log decisions to decisions_log.md.
```

---

### Phase 2 — Basic Synapse Layer (Weeks 9-16, ~4-6 weeks)

**Goal:** Implement v1 of the SynapseLayer. Show it has measurable effect on continual learning.

**Tasks:**

1. Implement PyTorch hook utilities
   - Forward hook to capture activations from any named layer
   - Designed to work with both small MLPs and HuggingFace transformers
2. Implement SynapseLayer v1 (basic version)
   - Dense connectivity (no sparsity yet) for the small MLP version
   - State: just `strength` for now (no confidence, no evidence yet)
   - Hebbian update: `Δw = η · R · a_i · a_j`
   - Reward fixed at 1.0 (no reward system yet)
   - Update triggered at end of each batch
3. Implement basic output modulation
   - Linear projection from synapse activations to a correction vector
   - Added to base model output before final classification
4. Integrate synapse layer with the MLP baseline
5. Run on Split-MNIST, compare against naive fine-tuning
6. Move to DistilBERT and a small text classification continual benchmark
   - Hook into the final hidden state
   - Use a Split-AG-News benchmark (split classes into sequential tasks)
7. Document early results, positive or negative

**Deliverables:**
- Working `SynapseLayer` module
- Hook integration with both MLP and DistilBERT
- Results: synapse layer v1 vs naive baseline on Split-MNIST
- Results: synapse layer v1 on DistilBERT + Split-AG-News
- Notebook walking through the v1 design

**Checkpoint criteria (end of Phase 2):**
- Does the synapse layer measurably affect output (sanity check, not necessarily reduce forgetting yet)?
- Is the update rule numerically stable (no explosion, no collapse to zero)?
- Can you run the same evaluation harness from Phase 1 on the synapse-augmented model?

If the synapse layer is destroying base model performance, debug. Likely culprits: learning rate too high, correction vector too large, modulation point too aggressive.

**Suggested Claude Code prompt to start Phase 2:**

```
We are starting Phase 2 of PROJECT_PLAN.md. Phase 1 is complete:
we have working Split-MNIST evaluation and EWC baseline results.

Goal for Phase 2: implement v1 of the SynapseLayer. Reference
ARCHITECTURE.md section 4.2.1 for the design.

Specific tasks:
1. Implement PyTorch hook utilities in
   src/continual_synapse/base_models/hooks.py
2. Implement SynapseLayer v1 in
   src/continual_synapse/synapse_layer/layer.py (basic version,
   dense connectivity, just strength state, fixed reward)
3. Implement output modulation in
   src/continual_synapse/synapse_layer/modulation.py
4. Wire it into the MLP baseline
5. Run Split-MNIST evaluation
6. Document findings in decisions_log.md

Important: keep the modulation initially weak (correction vector
should be near-zero at init) so the base model's behavior is
preserved. Then let the synapse layer learn corrections.
```

---

### Phase 3 — Confidence and Resistance (Weeks 17-22, ~4-6 weeks)

**Goal:** Add the metacognitive components: confidence dimension and evidence-based resistance to change. Add a real reward system.

**Tasks:**

1. Extend `SynapseLayer` state with `confidence`, `evidence`, `age`, `access_count`
2. Implement confidence update logic
   - Confidence grows when activations consistently fire together
   - Confidence decreases when contradicted
   - Simple version: confidence ∝ stability of co-activation across recent batches
3. Implement evidence-based resistance
   - `Δw = η · R · a_i · a_j / (1 + β · evidence_ij)`
   - High-evidence synapses resist change
4. Implement the reward signal system
   - External reward: pass-through from training signal initially
   - Consistency reward: cosine similarity between current activations and EMA of past activations
   - Surprise reward: prediction error from a small auxiliary model
   - Mixer with developmental trajectory α(t)
5. Add sparse top-k representation
   - Each neuron keeps only its top-k strongest partners
   - This reduces memory and compute significantly
6. Run experiments comparing variants:
   - Synapse v1 (no confidence, no resistance)
   - + Confidence dimension
   - + Resistance
   - + Full reward system
   - + Sparse top-k

**Deliverables:**
- Full `SynapseLayer` with all state dimensions
- Reward mixer with developmental trajectory
- Ablation study: which components contribute what
- Notebook visualizing synapse state over time

**Checkpoint criteria (end of Phase 3):**
- Does adding confidence + resistance reduce variance in performance across runs?
- Does the developmental trajectory parameter α(t) actually decrease over time as intended?
- Are sparse top-k representations correctly maintaining the most important synapses?

**Suggested Claude Code prompt to start Phase 3:**

```
We are starting Phase 3. We have basic SynapseLayer working from
Phase 2. Now adding metacognition and reward system.

Reference: PROJECT_PLAN.md section 4.2.1 (synapse state) and 4.2.4
(reward signal).

Tasks for this session:
1. Extend SynapseLayer state buffers with confidence, evidence,
   age, access_count
2. Implement resistance in update_rules.py
3. Implement reward system in src/continual_synapse/reward/
   - external.py, consistency.py, surprise.py, mixer.py
4. Implement sparse top-k partner selection in
   src/continual_synapse/synapse_layer/topk.py
5. Update SynapseLayer to use the sparse representation
6. Write tests for the new components

Note: be careful with the sparse top-k update. When a co-activation
exceeds the current top-k threshold for a neuron, we need to evict
the weakest current partner. Make this efficient.
```

---

### Phase 4 — Cold Storage and Consolidation (Weeks 23-28, ~4-6 weeks)

**Goal:** Add the long-term archive layer and the consolidation cycle that flushes synapse state to cold storage.

**Tasks:**

1. Set up Chroma local instance
2. Implement cold storage interface
   - `store_cluster(embedding, metadata)`: add to Chroma
   - `retrieve_similar(query_embedding, k)`: find approximate matches
   - `update_metadata(id, metadata)`: track access patterns
3. Implement compression pipeline
   - Quantization functions: 32-bit → 16-bit → 8-bit → 4-bit
   - Age-based compression schedule: items get progressively quantized
4. Implement consolidation trigger
   - Compute pressure metric per synapse
   - Identify candidates exceeding threshold
   - Cluster candidates (use simple k-means on activation signatures)
5. Implement the consolidation pipeline
   - Bundle clusters into archive entries
   - Compute cluster embeddings
   - Insert to Chroma
   - Drain source synapses
6. Implement reconstructive retrieval
   - On retrieval, query Chroma with current activations
   - Reconstruct partial synapse state from top matches
   - Account for compression: reconstruction is approximate
7. Run long-horizon experiments
   - Extend Split-MNIST to many sequential tasks
   - Verify that long-term memory (via cold storage) actually works

**Deliverables:**
- Working Chroma integration
- Compression pipeline with measurable size reduction
- Consolidation cycles triggered automatically by pressure
- Reconstructive recall demonstrably preserves gist if not detail
- Notebook visualizing the consolidation cycle and retrieval

**Checkpoint criteria (end of Phase 4):**
- After consolidation, does the synapse layer have measurably lower memory footprint?
- After retrieval, do reconstructed synapses approximate the originals (some loss expected)?
- On long task sequences, does the cold storage layer prevent total forgetting of early tasks?

**Suggested Claude Code prompt to start Phase 4:**

```
We are starting Phase 4. SynapseLayer with full state and reward
system is working from Phase 3. Now adding the cold storage and
consolidation system.

Reference: PROJECT_PLAN.md sections 4.2.2 (cold storage) and
4.2.3 (consolidation trigger).

Tasks for this session:
1. Set up Chroma local instance and write the storage interface
   in src/continual_synapse/cold_storage/store.py
2. Implement compression in
   src/continual_synapse/cold_storage/compression.py
3. Implement consolidation trigger in
   src/continual_synapse/consolidation/trigger.py
4. Implement consolidation pipeline in
   src/continual_synapse/consolidation/pipeline.py
5. Implement reconstructive retrieval in
   src/continual_synapse/consolidation/reconstruction.py
6. Wire everything together: when consolidation triggers, synapse
   layer flushes to cold storage. When new query arrives, optional
   retrieval from cold storage to augment synapse state.
7. Write tests

Important: the consolidation cycle should be parameterized so we
can test with frequent low-threshold triggers during development
and infrequent high-threshold triggers in production.
```

---

### Phase 5 — Rigorous Evaluation (Weeks 29-32, ~3-4 weeks)

**Goal:** Compare the full system to existing methods. Generate publication-quality plots and tables.

**Tasks:**

1. Reproduce Differentiable Plasticity (Miconi et al. 2018) as a comparison baseline
   - Open-source implementations exist, reference them
   - Verify your reproduction matches published numbers within reasonable tolerance
2. Implement Experience Replay baseline
   - Maintain a buffer of past examples, interleave with new task data
3. Run all methods on multiple benchmarks:
   - Split-MNIST (5 tasks)
   - Permuted-MNIST (10 tasks)
   - Split-CIFAR-10 (5 tasks)
   - Split-AG-News (4 tasks)
4. For each benchmark, run multiple seeds (at least 5) for statistical significance
5. Compute standard continual learning metrics:
   - Average accuracy after final task
   - Average forgetting
   - Backward transfer (BWT)
   - Forward transfer (FWT)
6. Statistical tests: pair-wise Wilcoxon signed-rank between methods
7. Generate plots:
   - Per-task accuracy over time
   - Average accuracy comparison
   - Forgetting comparison
   - Memory footprint over time
8. Generate tables in publication format

**Deliverables:**
- All baselines reproduced and verified
- Full evaluation matrix: 4 benchmarks × 5 methods × 5 seeds
- Statistical significance tests
- Publication-quality figures in `results/figures/`
- Publication-quality tables in `results/tables/`

**Checkpoint criteria (end of Phase 5):**
- Can you state in one sentence per method how it performs vs the others?
- Are your results statistically significant or within noise?
- Have you been honest about negative or null results?

This is the phase where overclaiming is most tempting and most dangerous. Be ruthlessly honest.

**Suggested Claude Code prompt to start Phase 5:**

```
We are starting Phase 5. Full system is implemented from Phases
1-4. Now doing rigorous evaluation against existing methods.

Tasks:
1. Implement Differentiable Plasticity baseline in
   src/continual_synapse/baselines/diff_plasticity.py
   Reference: Miconi et al. 2018, look at official Uber AI repo
2. Implement Experience Replay baseline
3. Build evaluation runner that orchestrates: 4 benchmarks ×
   5 methods × 5 seeds
4. Compute metrics: avg accuracy, forgetting, BWT, FWT
5. Run Wilcoxon signed-rank tests between methods
6. Generate plots in results/figures/
7. Generate tables in results/tables/

Be rigorous about random seeds: use the same seeds across
methods for fair comparison. Save raw results to results/logs/
as JSON so we can re-analyze later without re-running.
```

---

### Phase 6 — Documentation and Publication (Weeks 33-36, ~2-4 weeks)

**Goal:** Polish the project for public release. Write the follow-up article. Create the demo video.

**Tasks:**

1. Polish the README
   - What this is, why it matters
   - Quick installation
   - Minimum viable example (5 lines of code)
   - Link to docs
   - Citation if applicable
2. Polish all module docstrings
3. Add comprehensive type hints
4. Run `pre-commit` hooks (black, ruff, mypy)
5. Create demo notebooks
   - `01_demo_catastrophic_forgetting.ipynb`: shows the baseline problem
   - `02_synapse_layer_walkthrough.ipynb`: builds understanding
   - `03_cold_storage_visualization.ipynb`: shows the cold storage cycle
   - `04_results_analysis.ipynb`: reproduces all paper figures
6. Write the follow-up article
   - Honest summary of what worked and what didn't
   - Comparison to the original sketch in the first article
   - Lessons learned
   - 2000-3000 words
7. Create demo video
   - 5-10 minutes
   - Show the system in action
   - Walk through the key results
8. Publish to GitHub
9. Cross-post the follow-up article to LinkedIn and your site
10. Engage with any responses

**Deliverables:**
- Polished public GitHub repository
- Demo video on YouTube
- Follow-up article published
- Social media post announcing the project

**Checkpoint criteria (end of Phase 6):**
- Can a stranger clone the repo, install in 5 minutes, and run the demo?
- Does the follow-up article accurately reflect what you built?
- Are you honest about negative results, if any?

**Suggested Claude Code prompt to start Phase 6:**

```
We are starting Phase 6, the final polish and publication phase.
Full system and evaluation are complete.

Tasks:
1. Polish README.md following PROJECT_PLAN.md template
2. Add docstrings to all public functions
3. Run black and ruff on the entire codebase
4. Add type hints where missing
5. Create the four demo notebooks in notebooks/
6. Help me draft the follow-up article (2000-3000 words)
7. Help me outline the demo video

Focus: make this so polished that a researcher who reviews it
would treat it as a serious project. Boring documentation that
works beats clever documentation that confuses.
```

---

## 8. Evaluation methodology

### 8.1 Standard continual learning metrics

For a sequence of T tasks, with `a_{i,j}` = accuracy on task j after training through task i:

- **Average Accuracy (ACC):** mean of `a_{T,j}` for j = 1 to T
- **Average Forgetting (FGT):** mean of `(max_i a_{i,j} - a_{T,j})` for j = 1 to T-1
- **Backward Transfer (BWT):** mean of `(a_{T,j} - a_{j,j})` for j = 1 to T-1
- **Forward Transfer (FWT):** mean of `(a_{j-1,j} - random_accuracy_j)` for j = 2 to T

Report all four for every method on every benchmark.

### 8.2 Statistical significance

- Run each (method, benchmark) combination with 5 random seeds minimum
- Report mean ± std deviation
- Use Wilcoxon signed-rank test for paired comparisons between methods
- Use Bonferroni correction when comparing multiple methods
- Be explicit about significance levels (typically p < 0.05)

### 8.3 Honest reporting

Three rules:
1. If your method does not significantly outperform a baseline on a metric, say so explicitly
2. If your method has higher variance than baselines, report it
3. If your method works on one benchmark but not another, report both

The integrity of the project depends on this honesty more than on the results.

---

## 9. Decision logging

Maintain `decisions_log.md` with entries like this:

```
## [2026-06-15] Chose Chroma over Pinecone for cold storage
**Decision:** Use Chroma as the vector database backend.
**Alternatives considered:** Pinecone, Weaviate, FAISS.
**Rationale:** Chroma is local-first, no infrastructure. Pinecone
requires API keys and has rate limits. Weaviate is heavier setup.
FAISS is just an index, not a full DB with metadata.
**Trade-offs:** May need to switch to Pinecone for production if
we ever go to that scale. For 6-month research project, Chroma
is right.
**Reversibility:** Easy. The interface in cold_storage/store.py
abstracts the backend.
```

This becomes invaluable in month 4 when you ask yourself "why did I do that?" and when you onboard collaborators or write the article.

---

## 10. Risk management

### 10.1 Technical risks

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Synapse layer destabilizes base model | High in Phase 2 | High | Start with very small modulation. Tune learning rate carefully. |
| Sparse top-k is buggy | Medium in Phase 3 | High | Write extensive tests. Visualize sparse state. |
| Cold storage retrieval is meaningless | Medium in Phase 4 | High | Build the retrieval interface around a clear use case from day one. |
| Results are negative | Medium overall | Medium | Honest reporting. Negative results still publishable as a project. |
| Reproducing baselines is harder than expected | High | Medium | Allocate extra time in Phase 5. Use existing open-source implementations where possible. |

### 10.2 Project risks

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Loss of motivation around month 3 | High | Critical | Set monthly mini-milestones. Public commitments. |
| Scope creep | High | Medium | This document is the contract. New features go to "future work" not "version 1". |
| Hardware constraints slow iteration | Medium | Medium | Move to cloud GPUs if local becomes a bottleneck. |
| External pressure (job, life) | High | Medium | Reduce scope before quitting. Phase 1+2 alone is already valuable. |

### 10.3 Exit criteria

If after Phase 2 you realize the project isn't viable:
- Publish what you have as "exploration of additive plasticity for continual learning"
- The article + Phase 1-2 work is still a credible portfolio piece

If after Phase 4 results are clearly negative:
- Publish as "negative results in additive plasticity layers for continual learning"
- Negative results are a real contribution

---

## 11. Working with Claude Code

### 11.1 General principles

- **Start every session by referencing this document.** "We are working on the continual-synapse-layer project, currently in Phase X. Please read PROJECT_PLAN.md before we start."
- **Keep `decisions_log.md` updated.** End every Claude Code session with "Please update decisions_log.md with any architectural decisions we made this session."
- **Use small focused prompts.** Don't ask Claude Code to "implement Phase 3". Ask it to implement one component at a time.
- **Verify outputs.** Run the tests. Look at the plots. Don't trust without checking.

### 11.2 Suggested workflow per session

1. Open this document, identify the phase you're in
2. Pick 1-3 tasks from that phase
3. Start Claude Code session, paste the suggested prompt for that phase
4. After Claude Code makes changes, run the tests
5. Iterate on bugs and design questions
6. Before ending session, ask Claude Code to update decisions_log.md
7. Commit and push

### 11.3 Prompts for common situations

**When stuck on a design decision:**
```
We are at decision point X. The options are A, B, and C.
A has the property of ... but the downside of ...
B has the property of ... but the downside of ...
C has the property of ... but the downside of ...

Given the goals in PROJECT_PLAN.md (specifically section X.Y),
which would you recommend and why? Be specific about how this
decision affects later phases.
```

**When debugging:**
```
The test test_synapse_layer.py::test_consolidation is failing
with the error [...]. Here is the relevant code: [...].

Walk through the logic step by step. Identify likely causes.
Suggest the minimal change to fix. Do not refactor anything else.
```

**When results look wrong:**
```
I ran experiment 03_synapse_layer_v1.py and got these results:
[paste results]

These look [too good / too bad / strange] because [why].

Help me investigate. Possible causes I'd like you to consider:
1. Bug in the metric computation
2. Bug in the training loop
3. Hyperparameter issue
4. Actually a real result

For each, suggest a quick diagnostic.
```

---

## 12. After publication

### 12.1 Capitalize on the project

- Pin the GitHub repo to your profile
- Add it to your LinkedIn featured section
- Mention it in your LinkedIn headline
- Reference it in every cold outreach to AI companies
- Update the original article ("Orange out of an apple") with a postscript linking to the implementation

### 12.2 Open dialogue

- Reach out to authors of papers you reference (Miconi et al. for Differentiable Plasticity, etc.)
- "I built a related thing, would love your thoughts on X"
- Researchers often respond to thoughtful technical engagement from outsiders

### 12.3 Followups

The natural next projects:
- Apply to a larger base model (Llama-3 1B or 3B)
- Apply to a different domain (vision continual learning)
- Investigate one of the open questions identified in the article (reward signal weighting, consolidation threshold tuning)

Each of these is another 3-6 month project that builds on the foundation.

---

## 13. Appendix: Key references to read first

Before starting Phase 1, read these papers:

1. **Kirkpatrick et al. 2017 — "Overcoming catastrophic forgetting in neural networks"** (EWC). Foundational.
2. **Miconi et al. 2018 — "Differentiable Plasticity"** (Uber AI). Closest existing work to your additive layer concept.
3. **McClelland, McNaughton, O'Reilly 1995 — "Why there are complementary learning systems"** (CLS theory). The biological inspiration.
4. **Parisi et al. 2019 — "Continual Lifelong Learning with Neural Networks: A Review"** Good overview of the field.
5. **Hadsell et al. 2020 — "Embracing Change: Continual Learning in Deep Neural Networks"** (Nature Reviews). Recent perspective.

Read with these questions in mind:
- What benchmarks do they use? (You'll use the same ones)
- What metrics do they report? (You'll report the same)
- What are their open questions? (Your project may touch some)
- How do they handle the trade-offs you'll face?

Spend a full week on this reading. Take notes. Add a section to `decisions_log.md` summarizing each paper's key insight.

---

## End of plan

This document is your north star. Read it before every Claude Code session. Update it as you learn. The contract with yourself is that scope changes get added here explicitly, not silently in your head.

Good luck.
