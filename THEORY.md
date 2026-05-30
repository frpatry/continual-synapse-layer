# THEORY.md — Brain-Aligned Learning Substrate

**Version 0.1 — first complete formalization**

This is the theoretical foundation for a research investigation into what a system must look like if it is to learn the way biological brains learn. The investigation begins from first principles rather than from existing implementations.

This is Phase 0 of the project: theory before code. Nothing in this document has been validated experimentally. Every claim is provisional and falsifiable.

---

## 0. What this document is — and isn't

**This document is:**
- A precise statement of hypotheses about what brain-like learning requires
- A complete-enough ontology to begin Phase 1 implementation
- A set of falsifiable predictions
- A commitment to a research direction, not a product roadmap

**This document is not:**
- A literature review (we deliberately rederive concepts from first principles)
- A specification ready for engineering implementation
- A theory of consciousness, intelligence, or AGI
- A claim that the proposed substrate IS what the brain is doing — only that it shares enough structural properties to be worth investigating

**Methodological commitment:**
We choose conceptually clean designs over expedient ones. When two paths are available — one that matches existing ML patterns and one that respects our hypotheses more purely — we take the harder path. If implementation proves intractable, we document that finding and reconsider, rather than compromising the theory silently.

---

## 1. Core hypotheses

We claim that any system capable of brain-like learning must simultaneously satisfy five hypotheses. Removing any one of them defines a system fundamentally different from a brain, regardless of empirical performance on narrow tasks.

### H1 — Unified substrate
*Computation and memory share the same physical substrate.*

There is no separation between "the model that computes" and "the database that remembers". Activity within the substrate IS the computation AND constitutes the memory trace. Memory is not stored separately and retrieved into computation; it lives in the computational substrate itself as patterns of state and structural connectivity.

**Operationalization:** all information about prior experience must be encoded in the same set of elements that perform inference. No auxiliary storage modules are permitted in the substrate proper.

### H2 — Local learning rules
*All updates to structural state occur based on locally-available information only.*

A connection's update depends solely on the activity of its two endpoints (or, more generally, the activity of the elements immediately participating in it). No global error signal. No backpropagation through a graph. No loss function defined over the substrate's output.

**Operationalization:** for any structural weight w connecting elements e_a and e_b, the update Δw at time t depends only on quantities directly accessible from e_a, e_b, and (optionally) the substrate's global oscillatory state. Δw does NOT depend on a gradient computed by differentiating an objective function with respect to w.

### H3 — Forward-pass-as-learning
*Every inference modifies the substrate.*

There is no clean "training mode" vs "inference mode". Each act of computation leaves a trace in the substrate. The thinker is changed by thinking.

**Operationalization:** for every activation event, at least one structural weight in the substrate changes. The substrate's state at time t+1 is never identical to its state at time t whenever any activity occurred between t and t+1.

### H4 — Metastable background dynamics
*The system is never fully off.*

Even absent input, the substrate maintains spontaneous activity. New inputs perturb an already-active substrate, not a silent one. This background activity provides the dynamical context into which new patterns can graft.

**Operationalization:** the substrate's dynamics have one or more attractors that, in the absence of external input, generate ongoing low-level activity. The probability that all elements are simultaneously inactive for any sustained period approaches zero.

### H5 — Sparse distributed representations
*Each concept is encoded in patterns across many elements; each element participates in many concepts.*

No single element represents a concept. Each concept is a sparse pattern (small fraction of total elements active). The same elements participate in many different concepts under different conditions.

**Operationalization:** at any moment, the fraction of elements with activation above some threshold is small (target: 1-5% of substrate). For any given element, the number of distinct concepts whose activation patterns include it is large (target: many).

---

## 2. Ontology

The substrate consists of entities at four hierarchical levels. Each level shares the same fundamental properties; only the type of relation differs.

### 2.1 Element types

**N — Neurons (atoms)**

The fundamental unit. Has identity (unique ID), an activation state, and a structural weight (described below). N are present in the substrate from initialization.

**P — Pairs (binary relations)**

A pair P connects two elements (typically two N, but recursively could connect any two entities of any level). P has its own identity, activation, and structural weight. P are emergent — none exist at initialization; they appear through the emergence mechanism (Section 3.4).

**S — Schemas (unordered groupings)**

A schema S contains a set of elements (of any level — N, P, S, or C). S represents co-activation: the elements within S tend to be active together, without a specific order. S has identity, activation, and structural weight. Emergent.

**C — Paths (ordered sequences)**

A path C contains an ordered sequence of elements (of any level). C represents temporal succession: the elements in C tend to be activated in this order. C has identity, activation, and structural weight. Emergent.

### 2.2 Recursivity (fractal property)

Each of S, C can contain elements of any level, including other S and other C. This recursivity is unbounded in principle but limited in practice by available emergence opportunities.

Examples of valid compositions:
- An S of N: a co-activation pattern of neurons
- A C of P: a sequence of binary relations activating in order
- An S of C: a co-activation of several temporal sequences (e.g., several motor sequences executable in parallel)
- A C of S: a sequence of co-activation patterns (e.g., a reasoning chain)

The substrate does not distinguish between "primitive" and "composite" elements in its dynamics. All elements obey the same rules.

### 2.3 Element properties

Every element X (regardless of level) has:

```
X.id           — unique identifier
X.activation   ∈ [0, 1]    — instantaneous state, fluid, changes per timestep
X.weight       ∈ [0, ∞)    — accumulated structural property, slow
X.contents     — for P, S, C only: which sub-elements are contained
X.ordering     — for C only: the order of contained elements
```

The dual nature of activation (fast, fluid) and weight (slow, structural) is essential. Activation is the substrate's current "thought"; weight is its accumulated experience.

### 2.4 Vue 1 — Entities are distinct

A pair P_ab is NOT merely a label for the co-activation of N_a and N_b. It is a distinct entity with its own state, separate from N_a and N_b. When P_ab is active, this is an event distinct from N_a and N_b being individually active.

This commitment matters: it means the substrate can hold representational structure at multiple levels simultaneously, with feedback between levels.

---

## 3. Dynamics

### 3.1 Activation propagation

At each time step t, the activation of each element evolves based on:
- Activations of related elements
- Its own structural weight
- Background noise (per H4)
- External input (when present)

The precise functional form is to be determined experimentally. A minimum starting form:

```
X.activation(t+1) = σ(
    Σ_{Y ∈ neighbors(X)} influence(Y → X) · Y.activation(t)
    + background_drive
    + external_input(X, t)
)
```

where σ is a saturating nonlinearity producing values in [0,1], and `influence(Y → X)` reflects both the structural weight of the relation between X and Y and contextual modulation.

The exact form will be refined during Phase 1.

### 3.2 Structural weight growth (multi-level plasticity)

Per H3, every activation event modifies the substrate. Specifically, structural weights evolve continuously according to a local rule applied at every level:

```
For each element X:
  ΔX.weight = f(X.activation) - g(X.weight, system_age)
```

The growth term `f(X.activation)` reinforces weight in proportion to activation intensity. Higher and more sustained activation produces larger increments.

The decay term `g(X.weight, system_age)` represents passive forgetting. It depends on:
- The current weight (heavier weights decay more slowly, reflecting consolidation)
- **The age of the system** — younger systems have higher plasticity (faster decay AND faster growth), older systems have slower plasticity (consolidated structure resists change)

This age-dependence is non-trivial. It captures developmental "critical periods" observed in biology: child brains are highly plastic and forget quickly; adult brains have stabilized memories that resist change. Our system should exhibit the same trajectory: rapid early learning and forgetting, gradual consolidation, eventual robustness.

The precise form of g (and the role of system_age) is open for experimental refinement.

### 3.3 The emergence mechanism

Per our commitment to truly dynamic emergence (E-d), new entities (P, S, C) are not pre-allocated. They come into existence when specific conditions are met.

**Conditions for emergence:**

For each candidate combination (two elements that might form a P, multiple elements that might form an S, an ordered sequence that might form a C), the substrate implicitly tracks:

```
candidacy_strength(combination) — accumulates with co-activation, decays with non-use
validation_passes(combination)  — counts distinct "passes" of co-activation
```

A "pass" is defined by a quiescence-to-activity transition: candidacy_strength must drop below a quiescence threshold and then rise again. Sustained continuous activity counts as one pass, regardless of duration.

**Emergence trigger:**

A new entity is instantiated when:

```
candidacy_strength > θ_emergence
AND validation_passes ≥ N_min
```

Both conditions must hold simultaneously. This requires not only sufficient cumulative activation, but recurrence across distinct events. This mirrors the biological consolidation that requires repeated exposure across time (the spacing effect).

**Implementation preference (γ): phase transition from structural weights**

Per H1, we prefer to encode candidacy in the substrate itself rather than maintain an external tracker. The implementation we will attempt first treats emergence as a phase transition: when the structural weights of a sub-graph exceed a coupling threshold, the sub-graph "crystallizes" into a new entity at the next higher level.

**Fallback (α): explicit candidacy tracker**

If γ proves mathematically intractable or empirically unstable, we fall back to maintaining explicit candidacy_strength and validation_passes as auxiliary state, accepting a small departure from strict H1 compliance.

### 3.4 Forgetting and entity dissolution

Per the decay term g in Section 3.2, structural weights decay continuously. An entity whose weight drops below a viability threshold is removed from the substrate.

This removal is true dissolution (E-d permits substrate to shrink, not only grow). The element's ID may be later reused for a different emergent entity.

**Age-modulated forgetting:**

The decay rate g depends on system age. This produces two phenomena:
- Young systems exhibit rapid turnover: many entities emerge and dissolve in short order
- Mature systems exhibit stable structure: established entities persist for long durations even when not recently activated

This is the formal expression of "critical periods" and adult cognitive stability.

### 3.5 Background dynamics (H4)

The substrate maintains spontaneous activity even without external input. This is implemented as a stochastic drive term in the activation update (Section 3.1). The exact form of background_drive will be determined experimentally; minimum requirement is that it produce non-trivial dynamics in the absence of input.

---

## 4. Spaces / Zones — deferred

A potentially important refinement, deferred for now: the substrate may benefit from spatial/topological organization, where certain activations preferentially occur in certain "zones" of the substrate.

Three implementation strategies were identified:
- Z-a: pre-defined zones
- Z-b: content-based hashing into zones
- Z-c: zones emerge organically from substrate dynamics

If introduced, Z-c is preferred (most aligned with H1). We defer this question entirely until Phase 1 results indicate whether unstructured (no zones) substrate suffices or fails.

---

## 5. Open questions to be resolved during Phase 1

The theory above is complete in form but underspecified in numerical detail. Phase 1 must determine:

**5.1 Activation function form**

The exact form of σ in Section 3.1. Sigmoidal? Threshold? Soft-thresholded ReLU? The choice affects what kinds of patterns can be represented and how they interact.

**5.2 Influence function**

How `influence(Y → X)` combines structural weight and activation. A starting point: `influence = weight · contextual_modulation`. Contextual modulation may depend on the elements' levels and types.

**5.3 Specific form of f and g**

The growth function f(activation) and decay function g(weight, age). Starting candidates: f linear in activation, g exponential decay rate scaled by age.

**5.4 Emergence thresholds**

θ_co (co-activation significance threshold), θ_emergence (candidacy strength to trigger emergence), N_min (passes required). These will be tuned empirically.

**5.5 Background dynamics**

The form and intensity of background_drive in the activation update.

**5.6 System age dynamics**

How "system age" advances and how exactly it modulates plasticity. Linear? Logarithmic? Non-monotonic with discontinuous critical periods?

**5.7 Substrate scale**

How many N at initialization? The minimum that exhibits the phenomena we predict.

---

## 6. Predictions (falsifiable)

If the theory is correct, the following should be observable in any implementation respecting H1-H5:

### P1 — Pattern formation through repeated exposure

Repeated exposure to a stimulus pattern should produce a stable activation pattern in the substrate that recurs when the stimulus recurs. After sufficient exposure, the pattern should be recallable from partial cues (pattern completion).

**Falsification:** if no stable pattern forms regardless of exposure duration and parameter tuning, H1+H3 are likely wrong or our specific implementation is broken.

### P2 — Emergence of higher-level entities

After exposure to many low-level patterns, the substrate should spontaneously produce higher-level entities (P, S, C) representing those patterns. The number and quality of emergent entities should correlate with the structural regularity of the input.

**Falsification:** if no emergence occurs despite sufficient stimulus and time, the emergence mechanism (Section 3.3) is wrong.

### P3 — Continuous learning without catastrophic forgetting

Sequential exposure to multiple distinct stimulus families should produce accumulated representations. Old patterns should remain accessible (perhaps with reduced fidelity) after new patterns are learned.

**Falsification:** if the substrate exhibits catastrophic forgetting like standard neural networks, then local learning rules (H2) alone are insufficient and some additional mechanism (perhaps related to age-modulated plasticity, Section 3.4) is required.

### P4 — Critical period effects

A young substrate (low system_age) should learn faster and forget faster than a mature substrate. Patterns established when young should be most robust. Late-life exposure to genuinely novel patterns should produce weaker representations than early-life exposure.

**Falsification:** if learning dynamics are identical at all system ages, the age-dependence we postulated is unnecessary.

### P5 — Compositional generalization

A substrate exposed to atomic patterns A, B, C should produce meaningful (not random) responses to novel compositions like AB or ABC. The emergent higher-level entities (especially S and C) should support this composition.

**Falsification:** if responses to novel compositions are no better than random or are simply lookups of the most-similar trained pattern, the substrate is acting as memory rather than as a learner.

### P6 — Stable background dynamics

In the absence of input, the substrate should exhibit non-trivial spontaneous activity that respects sparse activation (H5) and does not converge to a fixed point or saturate.

**Falsification:** if the substrate either silences completely or saturates without input, H4 is not properly implemented.

---

## 7. Methodology

### 7.1 Phase plan

**Phase 0 (current):** This document and any refinements to it. No code.

**Phase 1:** Minimal substrate exhibiting H1+H2+H3 (one level deep — only N, plasticity, basic activation dynamics). Test P1 (pattern formation).

**Phase 2:** Extended substrate with emergence (P and S). Test P2 (emergence) and partial P3 (forgetting).

**Phase 3:** Age dynamics added. Test P4 (critical periods).

**Phase 4:** C (paths) added. Test partial P5 (sequential composition).

**Phase 5:** Compositional behavior and substrate scaling. Test full P5 and P6.

**Phase 6+:** Speculative — language interface, integration with external observers.

Each phase has a falsifiable claim. Failure at any phase sends us back to theory revision before continuing.

### 7.2 What counts as success

Phase-level success is measured against the specific predictions above. Project-level success is measured against an honest assessment: have we produced something that learns in a qualitatively different way than current ML systems?

**We do not measure success by:**
- Benchmark accuracy on classification tasks
- Comparison to LLMs on language tasks
- Production-readiness
- Scale

**We measure success by:**
- Conceptual clarity gained
- Phenomena observed that are absent in current systems
- Honesty about what works and what fails

### 7.3 What counts as failure

If, after sustained effort, we cannot produce a substrate satisfying H1-H5 that exhibits even pattern formation (P1), the theory is likely fundamentally wrong. We document the negative result and reconsider.

Failure modes anticipated:
- Substrate refuses to stabilize (always saturated or always silent)
- Emergence never occurs regardless of parameters
- Catastrophic forgetting persists despite all interventions
- Compositional generalization fails even with extensive exposure

These would be useful negative results in themselves.

---

## 8. What we exclude — and why

To preserve the integrity of the investigation, we exclude approaches that, while perhaps useful for other goals, would compromise the test of our hypotheses:

- **Pretrained LLMs (Qwen, GPT, etc.) as the substrate.** Their pretraining is precisely the kind of separation between learning and inference that H3 denies. Using them as substrate contaminates any conclusion about H3.

- **Backpropagation.** Direct violation of H2.

- **External vector databases or memory modules.** Direct violation of H1.

- **RAG / retrieval-augmented generation patterns.** These embed the violation of H1 architecturally.

- **Standard fine-tuning, LoRA adapters.** These are training events separated from inference and violate H3.

- **Reusing existing implementations.** Per our methodological commitment, we redevelop from first principles. We may *read* literature (Hebb, Hopfield, Kanerva, Friston, modern interpretability work) but we do not import their code or directly transcribe their algorithms.

**What we may use as auxiliary tools (not as the system being studied):**
- PyTorch, NumPy, JAX as numerical substrates for our implementation
- Visualization libraries to inspect substrate dynamics
- An external language model (e.g., Qwen) purely as a translation layer between our substrate's representations and human-readable text — only if and when such an interface becomes necessary, and only in a way that does not contaminate the substrate itself

---

## 9. Honest probability assessment

Given that this investigation rejects most standard ML tools and methods, and given that we are a single researcher with consumer-grade compute:

- Producing a substrate that learns in any non-trivial way: roughly 30%
- Producing a substrate that demonstrates pattern formation (P1): roughly 50%
- Producing a substrate that demonstrates emergence (P2): roughly 15%
- Producing a substrate that avoids catastrophic forgetting (P3): roughly 8%
- Producing a substrate that exhibits critical periods (P4): roughly 5%
- Producing a substrate with compositional generalization (P5): roughly 3%
- Project produces useful insight worth publishing even if the headline goal fails: 35-45%

These probabilities have been explicitly accepted by the researcher.

---

## 10. Status

This is version 0.1 of the theory. It is complete enough to begin Phase 1 implementation but expected to be revised as Phase 1 results come in.

The theory has been co-developed in conversation. Key contributors:
- The framing of H1-H5 emerged from a fundamental insight about reactivation being the missing piece in all prior approaches
- The N/P/S/C ontology was proposed and refined collaboratively
- The recursive (fractal) structure across levels was a deliberate design choice
- The emergence mechanism (validation_passes counting distinct passes, not duration) reflects the biological spacing effect
- The age-dependence of plasticity captures critical period phenomena

Next concrete step: begin Phase 1 — implement a minimal substrate with N only, basic activation dynamics, and the local learning rule. Test P1 (pattern formation).

---

*This document supersedes all prior project plans. The previous direction (memory-augmented LLM with metacognitive scaffolding) is archived as legacy work; the new direction begins here.*
