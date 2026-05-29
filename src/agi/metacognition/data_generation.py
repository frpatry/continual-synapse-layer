"""Synthetic training data for the metacognitive layers (Phase 2c).

Generates labelled feature vectors that plausibly represent the four
epistemic statuses (``known``, ``unknown``, ``uncertain``,
``hallucinated``). Rule-based, with per-class distributions calibrated
to *empirically observed* Qwen2.5-1.5B feature distributions from the
Phase 2d.2 real-Qwen validation dump (
``results/agi/phase_2_validation_raw.jsonl``), per the Phase 2h
recalibration. The drift report this generator was tuned against lives
at ``results/agi/distribution_drift_report.md``.

**Phase 2h recalibration (key changes from the Phase 2c v1):**

- ``alignment_novel_token_ratio`` was universally too low across all
  classes in v1 — Qwen produces verbose, fluent responses with
  conversational filler that synthetic underestimated. Boosted for
  known/uncertain/unknown; in hallucinated, the v1 generator was
  *inverted* (predicted high novelty when real cohort actually has
  low novelty, because half of the "hallucinated" test cases trigger
  Qwen2.5's safety-trained polite refusal rather than confabulation).
- ``attention_to_facts_mean`` was wildly too high for known/uncertain
  in v1 — Qwen's attention is more diffuse than the synthetic prior
  suggested.
- ``alignment_max_cosine`` for known was too high in v1 — real Qwen
  responses are diluted by filler and don't match facts as tightly.
- ``response_length_tokens`` was too short for unknown/hallucinated in
  v1 — Qwen's polite refusals are LONG ("I'm sorry, but I cannot…
  Could you give me more context?").
- ``unknown`` with non-empty memory was modelled with alignment ≡ 0 in
  v1, but real Qwen still computes a moderate cosine to the irrelevant
  facts; bumped to mid-range.

**Limitations** that the recalibration does NOT fix:

- The ``hallucinated`` cohort observed in real data is partially
  contaminated with safety-trained refusals; the synthetic generator
  models this contamination so the metacog routes both confabulations
  and refusals to ``admit_ignorance``. A future phase that wants to
  distinguish them needs a cleaner test set.
- Query-side features (``query_length_tokens``, ``has_named_entity``,
  ``query_specificity``) are test-set characteristics, not Qwen
  behaviour; left as-is.

Two-stage strategy: use this synthetic data for training; then validate
against real-LLM data with the Phase 2d.2 pipeline (Phase 2h re-runs
this).

Feature naming — the dicts below are intentionally aligned with
:mod:`agi.metacognition.features` constants
(``MEMORY_FEATURE_NAMES`` / ``QUERY_FEATURE_NAMES`` /
``GENERATION_FEATURE_NAMES`` / ``ALIGNMENT_FEATURE_NAMES`` /
``RESERVED_FEATURE_NAMES``) so the generated examples can be fed
straight into ``assemble_feature_vector``. A unit test pins this
correspondence.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Literal, Tuple

import numpy as np

from .features import (
    ALIGNMENT_FEATURE_NAMES,
    GENERATION_FEATURE_NAMES,
    MEMORY_FEATURE_NAMES,
    POST_FEATURE_ORDER,
    PRE_FEATURE_ORDER,
    QUERY_FEATURE_NAMES,
    RESERVED_FEATURE_NAMES,
)


EpistemicStatus = Literal["known", "unknown", "uncertain", "hallucinated"]

# Class lists used by the CLI script + tests.
PRE_CLASSES: tuple[EpistemicStatus, ...] = ("known", "unknown", "uncertain")
POST_CLASSES: tuple[EpistemicStatus, ...] = (
    "known", "unknown", "uncertain", "hallucinated",
)


@dataclass
class TrainingExample:
    """A single (features, label, confidence) triple plus metadata.

    Attributes:
        features: feature_name → float. Keys depend on the persisted
            mode: for ``"pre"`` only the 10 :data:`PRE_FEATURE_ORDER`
            keys are present; for ``"post"`` all 18
            :data:`POST_FEATURE_ORDER` keys are present.
        status: ground-truth epistemic class.
        confidence: ground-truth target for the metacog confidence
            head, in ``[0, 1]``. Higher = more confident classification
            (for clear-cut classes); lower for the inherently
            ambiguous ``uncertain`` class.
        metadata: per-example provenance bookkeeping (generator name,
            internal seed snippet) — kept for debugging only.
    """

    features: dict
    status: EpistemicStatus
    confidence: float
    metadata: dict


class SyntheticDataGenerator:
    """Generate per-class feature vectors for metacog training.

    The four ``generate_*_example`` methods always emit the full
    18-feature dict (memory ∪ query ∪ generation ∪ alignment ∪
    reserved). Mode-specific projection is the I/O layer's job —
    see :func:`save_dataset`.

    Distributions are documented inline next to each draw so a future
    calibration pass can find and tweak them quickly.
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Per-class generators
    # ------------------------------------------------------------------

    def generate_known_example(self) -> TrainingExample:
        """``known``: memory has pertinent facts and the LLM
        response leans on them.

        Phase 2h diagnostic shape (post-recalibration):
          * 1-4 facts retrieved
          * max_similarity ≈ 0.7-0.95 (memory key matches query)
          * alignment_max_cosine ≈ 0.40-0.65 (filler dilutes)
          * alignment_novel_token_ratio ≈ 0.70-0.90 (Qwen's filler)
          * attention_to_facts_mean ≈ 0.05 (Qwen barely attends)
          * response_length_tokens ≈ 30
        """
        memory = self._memory_known()
        query = self._query_default()
        gen = self._generation_known(has_facts=True)
        align = self._alignment_known()
        return self._assemble(
            memory, query, gen, align,
            status="known",
            confidence=float(self.rng.beta(8, 2)),
            metadata={"generator": "known"},
        )

    def generate_unknown_example(self) -> TrainingExample:
        """``unknown``: memory empty or with irrelevant facts; Qwen
        almost always issues a polite refusal.

        Phase 2h diagnostic shape (post-recalibration):
          * 0-2 facts retrieved (~50/50 empty vs irrelevant)
          * when facts present: max_similarity ≈ 0.40-0.65
            (irrelevant facts still embed-close to query)
          * response_length ≈ 50 (polite multi-sentence refusal)
          * alignment_novel_token_ratio ≈ 0.5 when has_facts
            (refusal vocab ∉ facts); 0 when no facts (per the
            empty-facts contract)
        """
        has_facts = bool(self.rng.binomial(1, 0.5))
        n_facts = int(self.rng.integers(1, 3)) if has_facts else 0
        memory = self._memory_unknown(n_facts, has_facts)
        query = self._query_default()
        admitted = bool(self.rng.binomial(1, 0.90))
        gen = self._generation_unknown(admitted, has_facts)
        align = (
            self._alignment_unknown_with_facts()
            if has_facts
            else self._alignment_zero()
        )
        return self._assemble(
            memory, query, gen, align,
            status="unknown",
            confidence=float(self.rng.beta(8, 2)),
            metadata={
                "generator": "unknown",
                "admitted": admitted,
                "has_facts": has_facts,
            },
        )

    def generate_uncertain_example(self) -> TrainingExample:
        """``uncertain``: medium-quality memory hits, often older /
        partially degraded, with non-trivial similarity variance.

        Diagnostic shape:
          * 1-3 facts retrieved
          * medium similarity (0.4-0.7)
          * elevated similarity variance (multiple weakly-matching
            facts)
          * older recency, lower precision
          * response shows hedging (medium entropy, medium alignment)
          * confidence target is itself medium — uncertainty is
            inherently harder to be confident about
        """
        memory = self._memory_uncertain()
        query = self._query_default()
        gen = self._generation_uncertain()
        align = self._alignment_uncertain()
        return self._assemble(
            memory, query, gen, align,
            status="uncertain",
            confidence=float(self.rng.beta(3, 3)),
            metadata={"generator": "uncertain"},
        )

    def generate_hallucinated_example(self) -> TrainingExample:
        """``hallucinated``: low memory support BUT a long, confident
        response with low attention to facts and HIGH novel-token
        ratio.

        Diagnostic shape (the post-layer's signature catch):
          * low memory features
          * long response (LLM generated *something*)
          * low ``attention_to_facts_mean``
          * low ``alignment_max_cosine`` / ``alignment_mean_cosine``
          * HIGH ``alignment_novel_token_ratio`` (response invents
            content)
          * low-to-medium ``mean_token_entropy`` — the LLM is
            *confidently wrong*
        """
        n_facts = int(self.rng.integers(0, 3))
        has_facts = n_facts > 0
        memory = self._memory_hallucinated(n_facts, has_facts)
        query = self._query_default()
        gen = self._generation_hallucinated()
        align = self._alignment_hallucinated(has_facts)
        return self._assemble(
            memory, query, gen, align,
            status="hallucinated",
            confidence=float(self.rng.beta(8, 2)),
            metadata={"generator": "hallucinated"},
        )

    # ------------------------------------------------------------------
    # Batch + assembly helpers
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        n_per_class: int,
        classes: List[EpistemicStatus],
    ) -> List[TrainingExample]:
        """Generate ``n_per_class`` examples for each class, shuffled.

        The shuffled order means downstream train/val splitters can
        slice without re-sorting; the per-class count is exact (not
        sampled) so dataset balance is guaranteed.
        """
        examples: List[TrainingExample] = []
        generators: dict[EpistemicStatus, Callable[[], TrainingExample]] = {
            "known": self.generate_known_example,
            "unknown": self.generate_unknown_example,
            "uncertain": self.generate_uncertain_example,
            "hallucinated": self.generate_hallucinated_example,
        }
        for cls in classes:
            if cls not in generators:
                raise ValueError(
                    f"unknown class {cls!r}; expected one of "
                    f"{sorted(generators)}"
                )
            for _ in range(n_per_class):
                examples.append(generators[cls]())
        self.rng.shuffle(examples)
        return examples

    # ---------- internal feature builders ----------

    def _memory_known(self) -> dict:
        # Phase 2h recalibration vs v1:
        #  - similarity_variance: ×0.1 → ×0.02 (real ~0)
        #  - max_recency_days: exp(10) → exp(2) (real 0)
        #  - mean_access_count: Poisson(5)+1 → Poisson(1)+1 (real 1)
        # (max_similarity left at 0.7-0.95 — real 0.76 sits in range.)
        max_sim = float(self.rng.beta(8, 2) * 0.25 + 0.7)
        mean_sim = max_sim * float(self.rng.uniform(0.85, 0.98))
        return {
            "n_facts_retrieved": float(self.rng.integers(1, 5)),
            "max_similarity": max_sim,
            "mean_similarity": mean_sim,
            "similarity_variance": float(self.rng.beta(2, 5) * 0.02),
            "max_recency_days": float(self.rng.exponential(2)),
            "mean_access_count": float(self.rng.poisson(1) + 1),
            "precision_quality": float(self.rng.beta(8, 2) * 0.2 + 0.8),
        }

    def _memory_unknown(self, n_facts: int, has_facts: bool) -> dict:
        # Phase 2h recalibration vs v1:
        #  - When has_facts: max/mean_similarity 0.06 → 0.40-0.65
        #    (real 0.46 — even irrelevant facts retrieve at non-trivial
        #    cosine because Qwen's embeddings cluster semantically).
        #  - precision_quality: bumped to ~1.0 when has_facts
        #    (test set always has fresh memory).
        if not has_facts:
            return {
                "n_facts_retrieved": 0.0,
                "max_similarity": 0.0,
                "mean_similarity": 0.0,
                "similarity_variance": 0.0,
                "max_recency_days": 0.0,
                "mean_access_count": 0.0,
                "precision_quality": 0.0,
            }
        max_sim = float(self.rng.beta(5, 5) * 0.4 + 0.30)  # ~0.30-0.70
        mean_sim = max_sim * float(self.rng.uniform(0.90, 1.0))
        return {
            "n_facts_retrieved": float(n_facts),
            "max_similarity": max_sim,
            "mean_similarity": mean_sim,
            "similarity_variance": float(self.rng.beta(2, 8) * 0.005),
            "max_recency_days": float(self.rng.exponential(2)),
            "mean_access_count": float(self.rng.poisson(1) + 1),
            "precision_quality": float(self.rng.beta(8, 2) * 0.2 + 0.8),
        }

    def _memory_uncertain(self) -> dict:
        # Phase 2h recalibration vs v1:
        #  - max_similarity: 0.55 → 0.60-0.85 (real 0.70).
        #  - mean_similarity: matched proportionally (real 0.69).
        #  - similarity_variance: 0.11 → ~0.002 (real ~0; the test
        #    set's "competing facts" both retrieve at nearly identical
        #    cosine because facts in our test cases are encoded
        #    similarly).
        #  - max_recency_days: exp(60) → exp(2) (test set is fresh).
        #  - precision_quality: bumped to ~1.0.
        max_sim = float(self.rng.beta(7, 4) * 0.25 + 0.60)
        mean_sim = max_sim * float(self.rng.uniform(0.92, 1.0))
        return {
            "n_facts_retrieved": float(self.rng.integers(1, 4)),
            "max_similarity": max_sim,
            "mean_similarity": mean_sim,
            "similarity_variance": float(self.rng.beta(2, 8) * 0.005),
            "max_recency_days": float(self.rng.exponential(2)),
            "mean_access_count": float(self.rng.poisson(1) + 1),
            "precision_quality": float(self.rng.beta(8, 2) * 0.2 + 0.8),
        }

    def _memory_hallucinated(self, n_facts: int, has_facts: bool) -> dict:
        if not has_facts:
            return {
                "n_facts_retrieved": 0.0,
                "max_similarity": 0.0,
                "mean_similarity": 0.0,
                "similarity_variance": 0.0,
                "max_recency_days": 0.0,
                "mean_access_count": 0.0,
                "precision_quality": 0.0,
            }
        return {
            "n_facts_retrieved": float(n_facts),
            "max_similarity": float(self.rng.beta(2, 5) * 0.5),
            "mean_similarity": float(self.rng.beta(2, 5) * 0.4),
            "similarity_variance": float(self.rng.beta(3, 5) * 0.15),
            "max_recency_days": float(self.rng.exponential(90)),
            "mean_access_count": float(self.rng.poisson(1)),
            "precision_quality": float(self.rng.beta(2, 6) * 0.4),
        }

    def _query_default(self) -> dict:
        """Query features are class-agnostic — the metacog layer can
        only learn class from memory + generation + alignment signals.
        The pre-layer uses query features as weak side-info."""
        return {
            "query_length_tokens": float(self.rng.poisson(15) + 3),
            "has_named_entity": float(self.rng.binomial(1, 0.7)),
            "query_specificity": float(self.rng.beta(5, 2)),
        }

    def _generation_known(self, has_facts: bool = True) -> dict:
        # Phase 2h recalibration vs v1:
        #  - attention_to_facts_mean: Beta(8,2) (~0.80) → ~0.05
        #    (real Qwen barely attends to fact spans even on the
        #    cleanest known cases — observed 0.05 ± 0.025).
        #  - mean_token_entropy: gamma(2,1) → gamma(1.5,0.6) (real
        #    0.82, syn was 1.90).
        #  - response_length kept around 25-35 (real 26).
        mean_h = float(self.rng.gamma(1.5, 0.6))
        # ``attention_to_facts`` is only meaningful when a fact span
        # was actually injected into the prompt.
        attention = (
            float(self.rng.beta(3, 8) * 0.15 + 0.02)
            if has_facts else 0.0
        )
        return {
            "mean_token_entropy": mean_h,
            "max_token_entropy": mean_h * float(self.rng.uniform(2.0, 4.0)),
            "response_length_tokens": float(self.rng.poisson(25) + 5),
            "attention_to_facts_mean": attention,
        }

    def _generation_unknown(self, admitted: bool, has_facts: bool) -> dict:
        # Phase 2h recalibration vs v1:
        #  - admitted branch is now LONG (real refusals are 30-70
        #    tokens, not 8-15): Poisson(40)+10 ≈ 50 (real 48).
        #  - mean_token_entropy: lowered to gamma(1.5, 0.7) (real
        #    1.13, syn was 2.91).
        #  - attention_to_facts: 0 when no facts, tiny when has_facts
        #    (real 0.016).
        if admitted:
            mean_h = float(self.rng.gamma(1.5, 0.7))
            length = float(self.rng.poisson(40) + 10)
        else:
            # Long, high-entropy attempt despite no info — rare.
            mean_h = float(self.rng.gamma(4, 0.8))
            length = float(self.rng.poisson(40) + 10)
        attention = (
            float(self.rng.beta(3, 9) * 0.04 + 0.005)
            if has_facts else 0.0
        )
        return {
            "mean_token_entropy": mean_h,
            "max_token_entropy": mean_h * float(self.rng.uniform(2.0, 4.0)),
            "response_length_tokens": length,
            "attention_to_facts_mean": attention,
        }

    def _generation_uncertain(self) -> dict:
        # Phase 2h recalibration vs v1:
        #  - response_length: Poisson(18)+5 (~23) → Poisson(35)+8
        #    (~43; real 42).
        #  - attention_to_facts: Beta(4,4) (~0.5) → ~0.06 (real 0.06).
        #  - mean_token_entropy: gamma(3,1) → gamma(2,0.7) (real 1.25).
        mean_h = float(self.rng.gamma(2, 0.7))
        return {
            "mean_token_entropy": mean_h,
            "max_token_entropy": mean_h * float(self.rng.uniform(2.0, 4.0)),
            "response_length_tokens": float(self.rng.poisson(35) + 8),
            "attention_to_facts_mean": float(self.rng.beta(3, 9) * 0.15 + 0.02),
        }

    def _generation_hallucinated(self) -> dict:
        # Phase 2h recalibration vs v1:
        #  - response_length: Poisson(30)+10 (~40) → Poisson(70)+15
        #    (~85; real 85 ± 23). Real Qwen confabulations / refusals
        #    are LONG.
        #  - attention_to_facts: Beta(2,8)*0.3 (~0.06) → ~0.005 (real
        #    0.003; even when a fact is in the prompt, the
        #    confabulating Qwen ignores it).
        mean_h = float(self.rng.gamma(2, 0.7))
        return {
            "mean_token_entropy": mean_h,
            "max_token_entropy": mean_h * float(self.rng.uniform(2.0, 4.0)),
            "response_length_tokens": float(self.rng.poisson(70) + 15),
            "attention_to_facts_mean": float(self.rng.beta(2, 10) * 0.02 + 0.001),
        }

    def _alignment_known(self) -> dict:
        # Phase 2h recalibration vs v1:
        #  - alignment_max_cosine: 0.7-0.95 → 0.40-0.65 (real 0.54).
        #    Qwen's filler dilutes the embedding match.
        #  - alignment_novel_token_ratio: ~0.06 → ~0.80 (real 0.81).
        #    Qwen's verbose answers introduce many novel tokens.
        # Phase 2h.1:
        #  - verbatim_fact_match: Beta(8,2) → ~0.80. Qwen, even when
        #    wrapping the answer in filler, almost always uses the
        #    exact fact value verbatim ("Vous avez 32 ans …").
        max_c = float(self.rng.beta(5, 5) * 0.25 + 0.40)
        mean_c = max_c * float(self.rng.uniform(0.95, 1.0))
        return {
            "alignment_max_cosine": max_c,
            "alignment_mean_cosine": mean_c,
            "alignment_novel_token_ratio": float(
                self.rng.beta(7, 3) * 0.2 + 0.70
            ),
            "verbatim_fact_match": float(self.rng.beta(8, 2)),
        }

    def _alignment_zero(self) -> dict:
        return {
            "alignment_max_cosine": 0.0,
            "alignment_mean_cosine": 0.0,
            "alignment_novel_token_ratio": 0.0,
            "verbatim_fact_match": 0.0,
        }

    def _alignment_unknown_with_facts(self) -> dict:
        """Phase 2h NEW: real Qwen produces non-zero alignment for
        polite refusals when irrelevant facts are in memory — the
        refusal text shares some embedding space with the fact text,
        and the refusal words don't appear in the fact (high novelty).

        Real (n=5 cases with facts):
          - alignment_max/mean_cosine ≈ 0.28 (with substantial std)
          - alignment_novel_token_ratio ≈ 0.59

        Phase 2h.1:
          - verbatim_fact_match: Beta(2,8)*0.3 → ~0.06. The refusal
            text rarely quotes the irrelevant facts back at the user.
        """
        max_c = float(self.rng.beta(4, 7) * 0.4 + 0.10)  # ~0.20-0.50
        mean_c = max_c * float(self.rng.uniform(0.90, 1.0))
        return {
            "alignment_max_cosine": max_c,
            "alignment_mean_cosine": mean_c,
            "alignment_novel_token_ratio": float(
                self.rng.beta(5, 4) * 0.4 + 0.40
            ),
            "verbatim_fact_match": float(self.rng.beta(2, 8) * 0.3),
        }

    def _alignment_uncertain(self) -> dict:
        # Phase 2h recalibration vs v1:
        #  - alignment_novel_token_ratio: ~0.17 → ~0.92 (real 0.93).
        #  - alignment_max_cosine: kept around 0.40-0.60 (real 0.51).
        # Phase 2h.1:
        #  - verbatim_fact_match: Beta(4,4)*0.6 → ~0.30. With
        #    multiple competing facts, the response may quote one
        #    of them — but rarely all. Wide spread captures that.
        max_c = float(self.rng.beta(5, 5) * 0.3 + 0.35)  # ~0.35-0.65
        mean_c = max_c * float(self.rng.uniform(0.92, 1.0))
        return {
            "alignment_max_cosine": max_c,
            "alignment_mean_cosine": mean_c,
            "alignment_novel_token_ratio": float(
                self.rng.beta(9, 2) * 0.15 + 0.82
            ),
            "verbatim_fact_match": float(self.rng.beta(4, 4) * 0.6),
        }

    def _alignment_hallucinated(self, has_facts: bool) -> dict:
        # Phase 2h recalibration vs v1: the "hallucinated" cohort
        # observed in real data is a BIMODAL mix of:
        #   (a) true confabulations  (60% of with-facts cases): the
        #       model invents content, alignment_novel is HIGH.
        #   (b) safety-trained polite refusals (40% of with-facts
        #       cases AND 100% of no-facts cases): the model says
        #       "I'm sorry, I don't have that information" — alignment_
        #       novel is LOW because the refusal vocabulary is sparse.
        # Modelling both modes so the metacog learns to route both to
        # ``admit_ignorance`` rather than over-fit one mode.
        if not has_facts:
            # No facts → empty-facts gate zeroes alignment. Real
            # behaviour confirmed (80% of real hallucinated cases).
            return self._alignment_zero()
        # With facts — pick the mode.
        # Phase 2h.1:
        #  - verbatim_fact_match: very low in both modes (Beta(2,8)*0.2,
        #    mean ~0.04). Both refusals and confabulations rarely echo
        #    the actual fact value back at the user. This is the
        #    diagnostic signal separating hallucination from "correct
        #    answer + filler" (KNOWN class, where verbatim ~0.8).
        verbatim = float(self.rng.beta(2, 8) * 0.2)
        is_refusal = self.rng.random() < 0.55
        if is_refusal:
            return {
                "alignment_max_cosine": float(self.rng.beta(3, 8) * 0.3 + 0.05),
                "alignment_mean_cosine": float(self.rng.beta(3, 8) * 0.25 + 0.05),
                "alignment_novel_token_ratio": float(
                    self.rng.beta(2, 8) * 0.3
                ),
                "verbatim_fact_match": verbatim,
            }
        return {
            "alignment_max_cosine": float(self.rng.beta(2, 6) * 0.4),
            "alignment_mean_cosine": float(self.rng.beta(2, 6) * 0.3),
            "alignment_novel_token_ratio": float(
                self.rng.beta(7, 3) * 0.3 + 0.55
            ),
            "verbatim_fact_match": verbatim,
        }

    # ---------- combine + assemble ----------

    def _assemble(
        self,
        memory: dict,
        query: dict,
        gen: dict,
        align: dict,
        *,
        status: EpistemicStatus,
        confidence: float,
        metadata: dict,
    ) -> TrainingExample:
        # ``reserved_1`` is currently always 0.0 — kept in the dict
        # so the saved 18-feature post examples have the full
        # POST_FEATURE_ORDER key set.
        features = {**memory, **query, **gen, **align}
        for name in RESERVED_FEATURE_NAMES:
            features[name] = 0.0
        return TrainingExample(
            features=features,
            status=status,
            confidence=confidence,
            metadata=metadata,
        )


# ----------------------------------------------------------------------
# Persistence + split helpers
# ----------------------------------------------------------------------

def _project_features(features: dict, mode: str) -> dict:
    """Subset ``features`` to the keys expected by ``mode`` (``"pre"``
    or ``"post"``). Pre-mode gets the 10 memory + query keys; post-mode
    gets the full 18-key dict.
    """
    if mode == "pre":
        order = PRE_FEATURE_ORDER
    elif mode == "post":
        order = POST_FEATURE_ORDER
    else:
        raise ValueError(f"mode must be 'pre' or 'post', got {mode!r}")
    return {name: float(features.get(name, 0.0)) for name in order}


def save_dataset(
    examples: List[TrainingExample],
    path: Path,
    mode: str = "post",
) -> None:
    """Write ``examples`` to ``path`` as JSON-lines.

    ``mode`` selects the feature subset to persist (``"pre"`` →
    10 features per example, ``"post"`` → 18). The on-disk format
    is therefore self-consistent: a pre file always contains pre
    examples, a post file always contains post examples.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ex in examples:
            ex_dict = asdict(ex)
            ex_dict["features"] = _project_features(ex.features, mode)
            f.write(json.dumps(ex_dict) + "\n")


def load_dataset(path: Path) -> List[TrainingExample]:
    """Load a JSONL dataset written by :func:`save_dataset`."""
    path = Path(path)
    examples: List[TrainingExample] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            examples.append(TrainingExample(**data))
    return examples


def split_train_val(
    examples: List[TrainingExample],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[TrainingExample], List[TrainingExample]]:
    """Random train/val split. With a balanced input batch (equal
    examples per class), the split is *approximately* class-
    balanced — variance shrinks as N grows."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(
            f"val_ratio must be in (0, 1), got {val_ratio}"
        )
    rng = np.random.default_rng(seed)
    indices = np.arange(len(examples))
    rng.shuffle(indices)
    n_val = int(len(examples) * val_ratio)
    val_idx = set(int(i) for i in indices[:n_val])
    train = [ex for i, ex in enumerate(examples) if i not in val_idx]
    val = [ex for i, ex in enumerate(examples) if i in val_idx]
    return train, val


__all__ = [
    "EpistemicStatus",
    "POST_CLASSES",
    "PRE_CLASSES",
    "SyntheticDataGenerator",
    "TrainingExample",
    "load_dataset",
    "save_dataset",
    "split_train_val",
]
