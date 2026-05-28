"""End-to-end validation pipeline (Phase 2d.2).

For each :class:`ValidationCase`:

1. Build a fresh :class:`XRayEpisodicMemory` and seed it with the
   case's ``memory_facts`` (each fact rendered to text via
   :func:`serialize_facts` and embedded as the entry's key, so
   retrieval works by similarity to the query — the same way it
   does in real usage).
2. Retrieve the top-k entries for the case's query.
3. Build a prompt — *normal* (memory-aware, with admission
   scaffolding when memory is empty) or *raw* (no scaffolding,
   used for the hallucinated cohort to give Qwen the freedom to
   confabulate).
4. Generate the response via
   :meth:`FrozenFoundation.generate_with_signals` (entropy +
   attention-to-fact-spans captured).
5. Build the 10/18-feature dicts via the metacog feature
   extractors.
6. Run both the PRE and POST layers and record the predicted
   epistemic status + recommended action + confidence.
7. Return a self-contained dict the analysis layer can chew on.

The pipeline is intentionally read-only against memory (each
case builds its own); the foundation, pre, and post layers are
all passed in by the caller so the heavy load happens once.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import torch

from agi.foundation import FrozenFoundation
from agi.memory.precision import serialize_facts
from agi.memory.xray_episodic import XRayEpisodicMemory
from agi.metacognition.features import (
    assemble_feature_vector,
    extract_alignment_features,
    extract_generation_features,
    extract_memory_features,
    extract_query_features,
)
from agi.metacognition.layer import MetacognitiveLayer

from .test_cases import ValidationCase


# ----------------------------------------------------------------------
# Prompt builders
# ----------------------------------------------------------------------

def build_normal_prompt(
    query: str,
    retrieval: List[Tuple[Any, float]],
    tokenizer,
) -> tuple[str, list[tuple[int, int]]]:
    """Memory-aware Qwen chat prompt + token-range of the facts block."""
    facts_text = "\n".join(
        f"- {serialize_facts(entry.facts)}" for entry, _ in retrieval
    )
    sys_prefix = (
        "<|im_start|>system\n"
        "You have the following information about the user:\n"
    )
    sys_suffix = (
        "\nUse this information naturally when relevant.\n"
        "<|im_end|>\n"
    )
    user_part = (
        f"<|im_start|>user\n{query}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    full_prompt = sys_prefix + facts_text + sys_suffix + user_part

    fact_char_start = len(sys_prefix)
    fact_char_end = fact_char_start + len(facts_text)
    fact_ranges = _char_to_token_ranges(
        full_prompt, fact_char_start, fact_char_end, tokenizer,
    )
    return full_prompt, fact_ranges


def build_admission_prompt(query: str) -> tuple[str, list]:
    """Memory-empty case under scaffolding — encourages
    "I don't know" rather than confabulation."""
    prompt = (
        "<|im_start|>system\n"
        "You can only use information from your memory. If you "
        "don't have the information, simply say so. Don't make "
        "up answers.\n"
        "<|im_end|>\n"
        f"<|im_start|>user\n{query}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return prompt, []


def build_raw_prompt(
    query: str,
    retrieval: List[Tuple[Any, float]],
    tokenizer,
) -> tuple[str, list[tuple[int, int]]]:
    """No scaffolding — used by the hallucinated cohort to give
    Qwen freedom to confabulate."""
    if retrieval:
        return build_normal_prompt(query, retrieval, tokenizer)
    prompt = f"Question: {query}\nRéponse:"
    return prompt, []


def _char_to_token_ranges(
    full_text: str,
    char_start: int,
    char_end: int,
    tokenizer,
) -> list[tuple[int, int]]:
    """Translate a character span in ``full_text`` to a token
    span in the tokenised version.

    Approximate — Qwen's BPE doesn't perfectly align to character
    boundaries. We tokenise the prefix and the prefix+span
    separately and use the resulting token-count differential.
    Off-by-one errors get absorbed by ``generate_with_signals``'s
    range clamping.
    """
    if char_end <= char_start:
        return []
    pre = tokenizer(full_text[:char_start], return_tensors="pt")
    through = tokenizer(full_text[:char_end], return_tensors="pt")
    start = int(pre["input_ids"].shape[1])
    end = int(through["input_ids"].shape[1])
    if end <= start:
        return []
    return [(start, end)]


# ----------------------------------------------------------------------
# Main per-case driver
# ----------------------------------------------------------------------

def run_validation_case(
    case: ValidationCase,
    foundation: FrozenFoundation,
    pre_layer: MetacognitiveLayer,
    post_layer: MetacognitiveLayer,
    *,
    retrieval_threshold: float = 0.3,
    top_k: int = 5,
    max_new_tokens: int = 96,
) -> dict:
    """Run one case end-to-end and return a flat result dict."""
    # 1. Fresh memory seeded with the case's facts. Use a
    #    permissive retrieval threshold so synthetic facts with
    #    non-trivial semantic distance to the query still
    #    retrieve.
    memory = XRayEpisodicMemory(
        key_dim=foundation.key_dim,
        retrieval_threshold=retrieval_threshold,
        foundation=foundation,
    )
    for fact in case.memory_facts:
        fact_text = serialize_facts(fact)
        key = foundation.get_key(fact_text)
        memory.add_entry(key, fact)

    # 2. Retrieve.
    query_key = foundation.get_key(case.query)
    retrieval = memory.retrieve(query_key, top_k=top_k)

    # 3. Build the prompt.
    if case.use_scaffolding:
        if retrieval:
            prompt, fact_ranges = build_normal_prompt(
                case.query, retrieval, foundation.tokenizer,
            )
        else:
            prompt, fact_ranges = build_admission_prompt(case.query)
    else:
        prompt, fact_ranges = build_raw_prompt(
            case.query, retrieval, foundation.tokenizer,
        )

    # 4. Generate with signal capture (greedy for reproducibility).
    gen_info = foundation.generate_with_signals(
        prompt=prompt,
        fact_token_ranges=fact_ranges or None,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )

    # 5. Feature extraction.
    memory_feats = extract_memory_features(retrieval)
    query_feats = extract_query_features(case.query, foundation)
    gen_feats = extract_generation_features(gen_info)
    align_feats = extract_alignment_features(
        response=gen_info.response_text,
        facts=retrieval,
        foundation=foundation,
    )

    # 6. PRE inference.
    pre_combined = {**memory_feats, **query_feats}
    pre_tensor = assemble_feature_vector(pre_combined, mode="pre")
    pre_state = pre_layer.predict(pre_tensor, raw_features=pre_combined)

    # 7. POST inference.
    post_combined = {
        **memory_feats, **query_feats, **gen_feats, **align_feats,
    }
    post_tensor = assemble_feature_vector(post_combined, mode="post")
    post_state = post_layer.predict(post_tensor, raw_features=post_combined)

    return {
        "case_id": case.case_id,
        "query": case.query,
        "memory_size_seeded": len(case.memory_facts),
        "retrieval_size": len(retrieval),
        "response": gen_info.response_text,
        "gen_seconds": float(gen_info.generation_time_seconds),
        # Ground truth.
        "expected_status": case.expected_status,
        "expected_action": case.expected_action,
        "use_scaffolding": case.use_scaffolding,
        "notes": case.notes,
        # PRE outputs.
        "pre_predicted_status": pre_state.epistemic_status,
        "pre_predicted_action": pre_state.recommended_action,
        "pre_confidence": float(pre_state.confidence),
        # POST outputs.
        "post_predicted_status": post_state.epistemic_status,
        "post_predicted_action": post_state.recommended_action,
        "post_confidence": float(post_state.confidence),
        # Raw features for downstream diagnostics.
        "memory_features": {k: float(v) for k, v in memory_feats.items()},
        "query_features": {k: float(v) for k, v in query_feats.items()},
        "generation_features": {k: float(v) for k, v in gen_feats.items()},
        "alignment_features": {k: float(v) for k, v in align_feats.items()},
    }


__all__ = [
    "build_admission_prompt",
    "build_normal_prompt",
    "build_raw_prompt",
    "run_validation_case",
]
