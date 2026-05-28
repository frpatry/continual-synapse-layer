"""Synthetic training data for the metacognitive layers (Phase 2c).

Generates labelled feature vectors that plausibly represent the four
epistemic statuses (``known``, ``unknown``, ``uncertain``,
``hallucinated``). Rule-based, with per-class distributions calibrated
from the *expected* shape of features rather than from a real LLM run.

**Limitations** — these are honest and deliberate:

- The distributions are synthetic; they encode our prior on what the
  metacog signals should look like, not measured behaviour.
- The ``hallucinated`` class is the most synthetic — real-world
  hallucinations vary more than a single rule-based pattern can capture.
- Numerical calibration (mean, variance, beta shape parameters) will
  almost certainly need a second pass after Phase 2d trains on this and
  we observe what the layer actually learns vs. what real Qwen produces.

Two-stage strategy: use this synthetic data for initial training; then
validate against real-LLM data once the metacog layer is integrated
(Phase 2e+).

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
        """``known``: memory has pertinent, recent, high-precision
        facts and the LLM response leans on them.

        Diagnostic shape:
          * many facts retrieved (2-5)
          * high cosine similarity (max ~0.7-0.95)
          * high precision_quality
          * response is moderate length, low entropy
          * alignment cosine high, novel-token ratio low
        """
        memory = self._memory_known()
        query = self._query_default()
        gen = self._generation_known()
        align = self._alignment_known()
        return self._assemble(
            memory, query, gen, align,
            status="known",
            confidence=float(self.rng.beta(8, 2)),
            metadata={"generator": "known"},
        )

    def generate_unknown_example(self) -> TrainingExample:
        """``unknown``: memory empty or with low-similarity hits;
        response is either an admission (short, possibly templated)
        or an attempted answer with high entropy.

        Diagnostic shape:
          * 0-2 facts retrieved (mostly 0)
          * very low similarity if any
          * alignment all-zero (per the Phase 2b empty-facts contract)
          * response either short + admission OR long + high entropy
        """
        n_facts = int(self.rng.binomial(1, 0.3) * self.rng.integers(0, 3))
        has_facts = n_facts > 0
        memory = self._memory_unknown(n_facts, has_facts)
        query = self._query_default()
        admitted = bool(self.rng.binomial(1, 0.7))
        gen = self._generation_unknown(admitted)
        align = self._alignment_zero()
        return self._assemble(
            memory, query, gen, align,
            status="unknown",
            confidence=float(self.rng.beta(8, 2)),
            metadata={"generator": "unknown", "admitted": admitted},
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
        max_sim = float(self.rng.beta(8, 2) * 0.25 + 0.7)
        mean_sim = max_sim * float(self.rng.uniform(0.7, 0.9))
        return {
            "n_facts_retrieved": float(self.rng.integers(2, 6)),
            "max_similarity": max_sim,
            "mean_similarity": mean_sim,
            "similarity_variance": float(self.rng.beta(2, 5) * 0.1),
            "max_recency_days": float(self.rng.exponential(10)),
            "mean_access_count": float(self.rng.poisson(5) + 1),
            "precision_quality": float(self.rng.beta(8, 2)),
        }

    def _memory_unknown(self, n_facts: int, has_facts: bool) -> dict:
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
            "max_similarity": float(self.rng.beta(2, 8) * 0.3),
            "mean_similarity": float(self.rng.beta(2, 8) * 0.25),
            "similarity_variance": float(self.rng.beta(3, 5) * 0.1),
            "max_recency_days": float(self.rng.exponential(60)),
            "mean_access_count": float(self.rng.poisson(1)),
            "precision_quality": float(self.rng.beta(2, 5) * 0.5),
        }

    def _memory_uncertain(self) -> dict:
        max_sim = float(self.rng.beta(5, 5) * 0.3 + 0.4)
        mean_sim = max_sim * float(self.rng.uniform(0.6, 0.85))
        return {
            "n_facts_retrieved": float(self.rng.integers(1, 4)),
            "max_similarity": max_sim,
            "mean_similarity": mean_sim,
            # Higher variance — multiple weakly-matching facts.
            "similarity_variance": float(self.rng.beta(5, 2) * 0.15),
            "max_recency_days": float(self.rng.exponential(60)),
            "mean_access_count": float(self.rng.poisson(2)),
            "precision_quality": float(self.rng.beta(3, 5)),
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

    def _generation_known(self) -> dict:
        mean_h = float(self.rng.gamma(2, 1))
        return {
            "mean_token_entropy": mean_h,
            "max_token_entropy": mean_h * float(self.rng.uniform(1.5, 3.0)),
            "response_length_tokens": float(self.rng.poisson(25) + 5),
            "attention_to_facts_mean": float(self.rng.beta(8, 2)),
        }

    def _generation_unknown(self, admitted: bool) -> dict:
        if admitted:
            mean_h = float(self.rng.gamma(2, 1))
            length = float(self.rng.poisson(8) + 3)
        else:
            # Long, high-entropy attempt despite no info.
            mean_h = float(self.rng.gamma(5, 1))
            length = float(self.rng.poisson(15) + 5)
        return {
            "mean_token_entropy": mean_h,
            "max_token_entropy": mean_h * float(self.rng.uniform(1.5, 3.0)),
            "response_length_tokens": length,
            "attention_to_facts_mean": 0.0,
        }

    def _generation_uncertain(self) -> dict:
        # Medium entropy = hedging.
        mean_h = float(self.rng.gamma(3, 1))
        return {
            "mean_token_entropy": mean_h,
            "max_token_entropy": mean_h * float(self.rng.uniform(1.5, 3.0)),
            "response_length_tokens": float(self.rng.poisson(18) + 5),
            "attention_to_facts_mean": float(self.rng.beta(4, 4)),
        }

    def _generation_hallucinated(self) -> dict:
        # LOW entropy (confident) + LONG response.
        mean_h = float(self.rng.gamma(2, 0.7))
        return {
            "mean_token_entropy": mean_h,
            "max_token_entropy": mean_h * float(self.rng.uniform(1.5, 3.0)),
            "response_length_tokens": float(self.rng.poisson(30) + 10),
            # LOW attention to (the little) memory available.
            "attention_to_facts_mean": float(self.rng.beta(2, 8) * 0.3),
        }

    def _alignment_known(self) -> dict:
        max_c = float(self.rng.beta(8, 2) * 0.25 + 0.7)
        mean_c = max_c * float(self.rng.uniform(0.7, 0.9))
        return {
            "alignment_max_cosine": max_c,
            "alignment_mean_cosine": mean_c,
            "alignment_novel_token_ratio": float(self.rng.beta(2, 8) * 0.3),
        }

    def _alignment_zero(self) -> dict:
        return {
            "alignment_max_cosine": 0.0,
            "alignment_mean_cosine": 0.0,
            "alignment_novel_token_ratio": 0.0,
        }

    def _alignment_uncertain(self) -> dict:
        max_c = float(self.rng.beta(4, 5) * 0.4 + 0.3)  # ~0.3-0.7
        mean_c = max_c * float(self.rng.uniform(0.6, 0.85))
        return {
            "alignment_max_cosine": max_c,
            "alignment_mean_cosine": mean_c,
            "alignment_novel_token_ratio": float(self.rng.beta(3, 4) * 0.4),
        }

    def _alignment_hallucinated(self, has_facts: bool) -> dict:
        if not has_facts:
            # No facts → alignment slots are all zero per the empty-
            # facts contract. The diagnostic signal here is the
            # combination with LOW attention + LONG response (set by
            # _generation_hallucinated) + memory empty.
            return self._alignment_zero()
        # Has facts but the response ignores them.
        return {
            "alignment_max_cosine": float(self.rng.beta(2, 6) * 0.4),
            "alignment_mean_cosine": float(self.rng.beta(2, 6) * 0.3),
            # HIGH novel-token ratio — the signature hallucination tell.
            "alignment_novel_token_ratio": float(self.rng.beta(8, 2) * 0.3 + 0.6),
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
