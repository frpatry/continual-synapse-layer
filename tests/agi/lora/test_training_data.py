"""Tests for the synthetic-query + dataset generator."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from agi.lora.distillation import TeacherOutput
from agi.lora.training_data import (
    QUERY_TEMPLATES,
    VALUE_POOLS,
    generate_distillation_dataset,
    generate_training_queries,
)


# ---------- generate_training_queries ----------

def test_generate_queries_balanced_per_category():
    out = generate_training_queries(n_per_category=10, seed=0)
    counts = Counter(q.category for q in out)
    for cat in QUERY_TEMPLATES:
        assert counts[cat] == 10
    assert sum(counts.values()) == 10 * len(QUERY_TEMPLATES)


def test_generate_queries_diverse():
    """Most templates carry a ``{placeholder}`` in their query or
    memory; combinations with the value pools produce enough
    distinct (query, memory) tuples to cover the per-category
    target. We measure distinct *(query, frozen-memory)* pairs
    rather than queries alone — several unknown templates are
    fixed-query (e.g. "Quel est mon code postal?") and only vary
    via the memory fact, which is still useful diversity for
    distillation."""
    out = generate_training_queries(n_per_category=50, seed=1)
    distinct_pairs = {
        (q.query, tuple(sorted(tuple(sorted(f.items())) for f in q.memory_facts)))
        for q in out
    }
    # At least 20% distinct (query, memory) combinations — well
    # above the trivial "all identical" floor and tolerant of the
    # placeholder-less template subset.
    assert len(distinct_pairs) >= 0.2 * len(out)


def test_generate_queries_known_has_memory_facts():
    """Every ``known`` query should have at least one fact in
    its seeded memory (otherwise the teacher can't answer)."""
    out = generate_training_queries(n_per_category=20, seed=2)
    for q in out:
        if q.category == "known":
            assert q.memory_facts, f"known query has empty memory: {q.query}"


def test_generate_queries_unknown_no_memory_is_empty():
    out = generate_training_queries(n_per_category=20, seed=3)
    for q in out:
        if q.category == "unknown_no_memory":
            assert q.memory_facts == []


def test_generate_queries_uncertain_has_multi_facts():
    """Uncertain category always seeds 2+ competing facts."""
    out = generate_training_queries(n_per_category=20, seed=4)
    for q in out:
        if q.category == "uncertain_multi":
            assert len(q.memory_facts) >= 2


def test_generate_queries_placeholders_filled():
    """No raw ``{placeholder}`` tokens should leak into a
    generated query or fact."""
    out = generate_training_queries(n_per_category=30, seed=5)
    for q in out:
        assert "{" not in q.query, q.query
        for fact in q.memory_facts:
            for v in fact.values():
                assert "{" not in str(v), v


def test_generate_queries_deterministic_with_seed():
    a = generate_training_queries(n_per_category=10, seed=99)
    b = generate_training_queries(n_per_category=10, seed=99)
    assert [(q.query, q.category) for q in a] == [
        (q.query, q.category) for q in b
    ]


# ---------- generate_distillation_dataset (mocked teacher) ----------

class _FakeTeacher:
    """Returns a deterministic TeacherOutput per query."""

    def respond(self, query, memory):
        # ``memory.entries`` is a list of bare ``EpisodicEntry``
        # objects (the (entry, sim) tuple shape only appears in
        # ``retrieve()`` output).
        facts = (
            [e.facts for e in memory.entries[:5]]
            if memory is not None and getattr(memory, "entries", None)
            else []
        )
        # Simulate a real teacher branch: defer when no facts.
        if not facts:
            return TeacherOutput(
                query=query,
                facts_in_context=facts,
                response="Je n'ai pas cette information dans ma mémoire.",
                epistemic_status="unknown",
                action_taken="admit_ignorance",
                used_template=True,
                metacog_confidence=0.99,
            )
        return TeacherOutput(
            query=query,
            facts_in_context=facts,
            response=f"Réponse basée sur: {facts[0]}",
            epistemic_status="known",
            action_taken="answer",
            used_template=False,
            metacog_confidence=0.9,
        )


class _FakeFoundation:
    """Just enough surface for _seed_memory_from_facts."""

    def __init__(self):
        self.key_dim = 8

    def get_key(self, text):
        import torch
        h = abs(hash(text)) & 0xFFFFFFFF
        g = torch.Generator()
        g.manual_seed(h)
        v = torch.randn(self.key_dim, generator=g)
        return v / v.norm()


def test_generate_distillation_dataset_writes_jsonl(tmp_path):
    queries = generate_training_queries(n_per_category=2, seed=7)
    out_path = tmp_path / "train.jsonl"
    n = generate_distillation_dataset(
        queries, _FakeTeacher(), _FakeFoundation(), out_path,
    )
    assert n == len(queries)
    assert out_path.exists()
    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == n
    rec = json.loads(lines[0])
    # Schema check.
    for key in (
        "prompt", "target", "category", "action_taken",
        "used_template", "epistemic_status", "metacog_confidence",
    ):
        assert key in rec


def test_dataset_distinguishes_template_vs_answer_targets(tmp_path):
    """The dataset should contain both template-style and
    generated-style targets so the student learns both."""
    queries = generate_training_queries(n_per_category=10, seed=8)
    out_path = tmp_path / "mixed.jsonl"
    generate_distillation_dataset(
        queries, _FakeTeacher(), _FakeFoundation(), out_path,
    )
    used_template_count = 0
    not_template_count = 0
    for line in out_path.read_text().strip().splitlines():
        rec = json.loads(line)
        if rec["used_template"]:
            used_template_count += 1
        else:
            not_template_count += 1
    assert used_template_count > 0
    assert not_template_count > 0


# ---------- VALUE_POOLS sanity ----------

def test_value_pools_are_non_empty():
    """Each placeholder key referenced in templates has a non-
    empty value pool."""
    referenced_keys: set[str] = set()
    import re
    pat = re.compile(r"\{(\w+)\}")
    for templates in QUERY_TEMPLATES.values():
        for query_str, mem in templates:
            referenced_keys.update(pat.findall(query_str))
            if isinstance(mem, dict):
                for v in mem.values():
                    referenced_keys.update(pat.findall(str(v)))
            elif isinstance(mem, list):
                for item in mem:
                    for v in item.values():
                        referenced_keys.update(pat.findall(str(v)))
    for key in referenced_keys:
        assert key in VALUE_POOLS, f"missing value pool: {key}"
        assert VALUE_POOLS[key], f"value pool {key} is empty"
