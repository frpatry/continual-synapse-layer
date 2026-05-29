"""Diverse training-query generation + teacher-driven dataset
production for Phase 2e distillation.

The generation is fully synthetic: a small set of query templates
in French + English is combined with value pools (names, cities,
ages, …) to produce thousands of distinct (query, memory_state)
pairs. Each pair is then run through the teacher pipeline to
produce a ``(prompt, target)`` example for LoRA fine-tuning.

Categories (matched to the four epistemic statuses):

- ``known`` — memory has the answer (templates encode which fact
  field carries the answer).
- ``unknown_no_memory`` — empty memory.
- ``unknown_irrelevant_memory`` — memory has a fact that doesn't
  match the query.
- ``uncertain_multi`` — memory has multiple competing facts
  (hard-coded in the templates because the ambiguity is the
  whole point).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from agi.memory.precision import serialize_facts

if TYPE_CHECKING:
    from agi.foundation import FrozenFoundation
    from .distillation import TeacherPipeline

from .distillation import build_student_input


# ----------------------------------------------------------------------
# Templates + value pools
# ----------------------------------------------------------------------

# Each query template is a (query_str, memory_template) pair.
# memory_template is either:
#   * None                    → caller should leave memory empty
#   * a single dict           → memory will get one fact
#   * a list of dicts         → memory will get several facts
# Placeholders use ``{name}`` / ``{city}`` / etc. and are filled
# from VALUE_POOLS at generation time.

QUERY_TEMPLATES: Dict[str, List[Tuple[str, Any]]] = {
    "known": [
        ("Comment je m'appelle?", {"name": "{name}"}),
        ("Où est-ce que j'habite?", {"city": "{city}"}),
        ("Quel âge j'ai?", {"age": "{age}"}),
        ("Quelle est ma profession?", {"profession": "{profession}"}),
        ("Quel est mon hobby?", {"hobby": "{hobby}"}),
        ("Quelle est ma couleur préférée?", {"favorite_color": "{color}"}),
        ("Quel sport je pratique?", {"sport": "{sport}"}),
        ("Quelle langue je parle?", {"language": "{language}"}),
        ("Quelle est ma ville natale?", {"birth_city": "{city}"}),
        ("Comment s'appelle ma femme?", {"spouse": "{name}"}),
        ("Quel instrument je joue?", {"instrument": "{instrument}"}),
        ("Quel est mon plat préféré?", {"food_preference": "{food}"}),
        ("Comment s'appelle mon chien?", {"pet_name": "{name}", "pet_type": "chien"}),
        ("Suis-je marié?", {"marital_status": "{marital_status}"}),
        ("Quelle voiture je conduis?", {"car_make": "{car_make}"}),
        # English mirrors.
        ("What is my name?", {"name": "{name}"}),
        ("Where do I live?", {"city": "{city}"}),
        ("What's my favorite sport?", {"sport": "{sport}"}),
        ("What's my profession?", {"profession": "{profession}"}),
    ],
    "unknown_no_memory": [
        ("Quel est mon code postal?", None),
        ("Quel est mon numéro de téléphone?", None),
        ("Combien d'enfants j'ai?", None),
        ("Quelle est mon adresse exacte?", None),
        ("Comment s'appelle ma mère?", None),
        ("Quelles sont mes allergies?", None),
        ("Quel est mon groupe sanguin?", None),
        ("Quelle université j'ai fréquentée?", None),
        ("Combien je gagne par an?", None),
        ("Quelle est ma religion?", None),
        ("Pour qui ai-je voté aux dernières élections?", None),
        ("Que faisais-je hier soir?", None),
        # English.
        ("What is my postal code?", None),
        ("How many siblings do I have?", None),
        ("What is my insurance provider?", None),
    ],
    "unknown_irrelevant_memory": [
        # Memory has ONE unrelated fact.
        ("Quel est mon code postal?", {"name": "{name}"}),
        ("Combien d'enfants j'ai?", {"city": "{city}"}),
        ("Quel est mon numéro de téléphone?", {"profession": "{profession}"}),
        ("Comment s'appelle ma mère?", {"name": "{name}"}),
        ("Quelle est ma religion?", {"hobby": "{hobby}"}),
        ("Quelles sont mes allergies?", {"age": "{age}"}),
        ("Quel sport je pratique?", {"name": "{name}"}),  # NOTE: caller asks sport but memory has name
        ("Quel est mon plat préféré?", {"city": "{city}"}),
        ("Quelle université j'ai fréquentée?", {"profession": "{profession}"}),
        # English.
        ("What is my phone number?", {"name": "{name}"}),
        ("How many pets do I have?", {"city": "{city}"}),
    ],
    "uncertain_multi": [
        # Memory has multiple competing facts of the same type — hand-
        # coded because the *ambiguity* is the diagnostic shape.
        ("Quel sport je pratique?", [
            {"sport": "natation"}, {"sport": "vélo"},
        ]),
        ("Où je vis?", [
            {"city": "Lyon"}, {"city": "Paris"},
        ]),
        ("Quelle est ma couleur préférée?", [
            {"favorite_color": "bleu"}, {"favorite_color": "vert"},
        ]),
        ("Quel est mon métier?", [
            {"profession": "ingénieur"}, {"profession": "manager"},
        ]),
        ("Quelle est ma boisson préférée?", [
            {"drink": "café"}, {"drink": "thé"},
        ]),
        ("Quel instrument je joue?", [
            {"instrument": "guitare"}, {"instrument": "piano"},
        ]),
        # English mirror.
        ("What's my favorite sport?", [
            {"sport": "tennis"}, {"sport": "swimming"},
        ]),
    ],
}


VALUE_POOLS: Dict[str, List[Any]] = {
    "name": [
        "François", "Marie", "Jean", "Sophie", "Pierre", "Camille",
        "Lucas", "Emma", "Léa", "Hugo", "Chloé", "Antoine",
        "John", "Mary", "Alice", "Robert", "Linda", "James",
    ],
    "city": [
        "Lyon", "Paris", "Marseille", "Bordeaux", "Lille", "Toulouse",
        "Nice", "Nantes", "Strasbourg", "Montpellier",
        "New York", "London", "Tokyo", "Berlin", "Madrid",
    ],
    "age": list(range(20, 71)),
    "profession": [
        "ingénieur", "médecin", "enseignant", "avocat", "architecte",
        "journaliste", "designer", "chef", "pilote", "infirmier",
        "engineer", "doctor", "teacher", "lawyer", "writer",
    ],
    "hobby": [
        "photographie", "lecture", "randonnée", "cuisine", "peinture",
        "musique", "jardinage", "cyclisme",
        "photography", "reading", "hiking", "cooking",
    ],
    "color": [
        "bleu", "rouge", "vert", "jaune", "noir", "blanc",
        "blue", "red", "green",
    ],
    "sport": [
        "tennis", "football", "natation", "vélo", "course",
        "yoga", "basketball", "rugby",
    ],
    "language": [
        "français", "anglais", "espagnol", "allemand", "italien",
        "japonais", "mandarin",
    ],
    "instrument": [
        "guitare", "piano", "violon", "batterie", "flûte",
        "guitar", "piano", "drums",
    ],
    "food": [
        "pizza", "sushi", "pâtes", "tarte aux pommes", "salade",
        "burger", "pasta",
    ],
    "marital_status": [
        "marié", "célibataire", "divorcé",
    ],
    "car_make": [
        "Toyota", "Renault", "Peugeot", "BMW", "Tesla",
    ],
}


_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def _fill_str(template: str, rng: np.random.Generator) -> str:
    """Replace ``{key}`` placeholders in ``template`` with a random
    draw from ``VALUE_POOLS[key]``. Unknown keys are left in place."""
    def _sub(match: re.Match) -> str:
        key = match.group(1)
        pool = VALUE_POOLS.get(key)
        if not pool:
            return match.group(0)
        return str(pool[rng.integers(0, len(pool))])
    return _PLACEHOLDER_RE.sub(_sub, template)


def _fill_memory(
    memory_template: Optional[Any],
    rng: np.random.Generator,
) -> List[dict]:
    """Materialise the ``memory_template`` into a list of dicts.

    Returns ``[]`` when ``memory_template`` is ``None``. A single
    dict template returns a single-element list. A list-of-dicts
    template returns the list with values filled.
    """
    if memory_template is None:
        return []
    if isinstance(memory_template, dict):
        return [{k: _fill_str(str(v), rng) for k, v in memory_template.items()}]
    if isinstance(memory_template, list):
        out: List[dict] = []
        for item in memory_template:
            out.append({k: _fill_str(str(v), rng) for k, v in item.items()})
        return out
    raise TypeError(f"unexpected memory_template type {type(memory_template)!r}")


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

@dataclass
class GeneratedQuery:
    """One materialised training query with optional memory seeding."""

    query: str
    memory_facts: List[dict]
    category: str


def generate_training_queries(
    n_per_category: int = 250,
    seed: int = 42,
) -> List[GeneratedQuery]:
    """Sample ``n_per_category`` queries from each template family,
    filling placeholders from :data:`VALUE_POOLS`. Returns a shuffled
    list."""
    rng = np.random.default_rng(seed)
    out: List[GeneratedQuery] = []
    for category, templates in QUERY_TEMPLATES.items():
        for _ in range(n_per_category):
            idx = int(rng.integers(0, len(templates)))
            tmpl, mem_tmpl = templates[idx]
            out.append(
                GeneratedQuery(
                    query=_fill_str(tmpl, rng),
                    memory_facts=_fill_memory(mem_tmpl, rng),
                    category=category,
                )
            )
    # In-place shuffle so train/val splitting downstream doesn't have
    # to think about per-category ordering.
    rng.shuffle(out)
    return out


def _seed_memory_from_facts(
    foundation: "FrozenFoundation",
    facts: List[dict],
):
    """Build a fresh ``XRayEpisodicMemory`` and write each fact into
    it. Used by the dataset-generation loop, kept here so tests can
    patch it cheaply."""
    from agi.memory.xray_episodic import XRayEpisodicMemory

    memory = XRayEpisodicMemory(
        key_dim=foundation.key_dim,
        retrieval_threshold=0.3,
        foundation=foundation,
    )
    for fact in facts:
        text = serialize_facts(fact)
        key = foundation.get_key(text)
        memory.add_entry(key, fact)
    return memory


def generate_distillation_dataset(
    queries: List[GeneratedQuery],
    teacher: "TeacherPipeline",
    foundation: "FrozenFoundation",
    output_path: Path,
) -> int:
    """Run the teacher pipeline on each query and append a JSONL
    record to ``output_path``. Returns the count written.

    Each record has fields:

    - ``prompt`` — the input the student LoRA sees
    - ``target`` — the teacher's response (training target)
    - ``category`` — the source query family
    - ``action_taken`` — what the teacher pipeline decided
    - ``used_template`` — True when the response was a template
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w") as f:
        for q in queries:
            memory = _seed_memory_from_facts(foundation, q.memory_facts)
            teacher_out = teacher.respond(q.query, memory)
            student_in = build_student_input(teacher_out)
            record = {
                "prompt": student_in.prompt,
                "target": student_in.target,
                "category": q.category,
                "action_taken": teacher_out.action_taken,
                "used_template": teacher_out.used_template,
                "epistemic_status": teacher_out.epistemic_status,
                "metacog_confidence": float(teacher_out.metacog_confidence),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


__all__ = [
    "GeneratedQuery",
    "QUERY_TEMPLATES",
    "VALUE_POOLS",
    "generate_distillation_dataset",
    "generate_training_queries",
]
