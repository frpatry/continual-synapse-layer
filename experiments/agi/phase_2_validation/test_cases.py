"""100 hand-crafted validation cases for Phase 2d.2.

25 per epistemic status. Each case specifies:

- ``query``: what the user asks
- ``memory_facts``: list of fact dicts to seed the X-Ray memory
  with *before* the query is issued (empty list = empty memory)
- ``expected_status``: ground-truth epistemic class
- ``expected_action``: ground-truth recommended action
- ``use_scaffolding``: when ``False``, the prompt drops the
  "say so if you don't know" guard so Qwen has freedom to
  confabulate (used by the ``hallucinated`` cohort)
- ``notes``: why this case is interesting

The four cohorts are deliberately heterogeneous within each
class (French + English, varied fact types, single vs multi-
fact memory, etc.) to surface generalisation issues that a
template-y test set would miss.

Note on PRE evaluation for the ``hallucinated`` cohort: the
PRE layer was trained on 3 classes (no ``hallucinated``) and
runs *before* the foundation generates anything. The expected
status for the post-layer is ``hallucinated``; the pre-layer is
expected to call it ``unknown`` (memory is typically empty in
these cases). The analysis layer handles this by excluding
``hallucinated`` cases from the PRE accuracy aggregate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal


@dataclass
class ValidationCase:
    case_id: str
    query: str
    memory_facts: List[dict] = field(default_factory=list)
    expected_status: Literal[
        "known", "unknown", "uncertain", "hallucinated",
    ] = "known"
    expected_action: Literal[
        "answer", "answer_with_caveat", "admit_ignorance",
    ] = "answer"
    use_scaffolding: bool = True
    notes: str = ""


# ----------------------------------------------------------------------
# KNOWN — memory has the answer directly. Action: answer.
# ----------------------------------------------------------------------

KNOWN_CASES: list[ValidationCase] = [
    ValidationCase(
        case_id="known_001",
        query="Comment je m'appelle?",
        memory_facts=[{"name": "François"}],
        expected_status="known", expected_action="answer",
        notes="Single direct fact: name.",
    ),
    ValidationCase(
        case_id="known_002",
        query="Où est-ce que j'habite?",
        memory_facts=[{"city": "Lyon", "country": "France"}],
        expected_status="known", expected_action="answer",
        notes="Direct fact, two attributes.",
    ),
    ValidationCase(
        case_id="known_003",
        query="Quel est mon métier?",
        memory_facts=[{"profession": "ingénieur logiciel"}],
        expected_status="known", expected_action="answer",
        notes="Profession.",
    ),
    ValidationCase(
        case_id="known_004",
        query="Quel âge j'ai?",
        memory_facts=[{"age": 32}],
        expected_status="known", expected_action="answer",
        notes="Numeric scalar in memory.",
    ),
    ValidationCase(
        case_id="known_005",
        query="Quelle est ma couleur préférée?",
        memory_facts=[{"favorite_color": "bleu"}],
        expected_status="known", expected_action="answer",
        notes="Preference.",
    ),
    ValidationCase(
        case_id="known_006",
        query="Quel est mon hobby principal?",
        memory_facts=[{"hobby": "photographie"}],
        expected_status="known", expected_action="answer",
        notes="Hobby.",
    ),
    ValidationCase(
        case_id="known_007",
        query="Combien d'enfants j'ai?",
        memory_facts=[{"children_count": 2}],
        expected_status="known", expected_action="answer",
        notes="Numeric family fact.",
    ),
    ValidationCase(
        case_id="known_008",
        query="Où est-ce que je travaille?",
        memory_facts=[
            {"employer": "Google", "office_location": "Paris"},
        ],
        expected_status="known", expected_action="answer",
        notes="Two related attributes in one fact.",
    ),
    ValidationCase(
        case_id="known_009",
        query="Quelles langues je parle?",
        memory_facts=[{"languages": ["français", "anglais", "espagnol"]}],
        expected_status="known", expected_action="answer",
        notes="List-valued fact.",
    ),
    ValidationCase(
        case_id="known_010",
        query="Comment s'appelle mon chien?",
        memory_facts=[{"pet_name": "Rex", "pet_type": "chien"}],
        expected_status="known", expected_action="answer",
        notes="Pet name with disambiguator.",
    ),
    ValidationCase(
        case_id="known_011",
        query="What is my name?",
        memory_facts=[{"name": "John Smith"}],
        expected_status="known", expected_action="answer",
        notes="English variant, multi-word value.",
    ),
    ValidationCase(
        case_id="known_012",
        query="Where do I live?",
        memory_facts=[{"city": "New York", "country": "USA"}],
        expected_status="known", expected_action="answer",
        notes="English location.",
    ),
    ValidationCase(
        case_id="known_013",
        query="What's my favorite sport?",
        memory_facts=[{"sport": "tennis"}],
        expected_status="known", expected_action="answer",
        notes="English preference.",
    ),
    ValidationCase(
        case_id="known_014",
        query="Tell me about my family.",
        memory_facts=[
            {"spouse": "Marie", "children_count": 2, "pet_name": "Rex"},
        ],
        expected_status="known", expected_action="answer",
        notes="Multi-attribute synthesis from one fact.",
    ),
    ValidationCase(
        case_id="known_015",
        query="Quel est mon plat préféré?",
        memory_facts=[{"food_preference": "pizza margherita"}],
        expected_status="known", expected_action="answer",
        notes="Multi-word value.",
    ),
    ValidationCase(
        case_id="known_016",
        query="Combien d'animaux j'ai?",
        memory_facts=[{"pets_count": 3}],
        expected_status="known", expected_action="answer",
        notes="Count fact.",
    ),
    ValidationCase(
        case_id="known_017",
        query="Suis-je marié?",
        memory_facts=[{"marital_status": "marié"}],
        expected_status="known", expected_action="answer",
        notes="Yes/no factual answer.",
    ),
    ValidationCase(
        case_id="known_018",
        query="Quelle est ma ville natale?",
        memory_facts=[{"birth_city": "Marseille"}],
        expected_status="known", expected_action="answer",
        notes="Distinct from current city.",
    ),
    ValidationCase(
        case_id="known_019",
        query="Est-ce que je suis végétarien?",
        memory_facts=[{"diet": "végétarien"}],
        expected_status="known", expected_action="answer",
        notes="Yes/no with extra context.",
    ),
    ValidationCase(
        case_id="known_020",
        query="Quel instrument je joue?",
        memory_facts=[{"instrument": "guitare"}],
        expected_status="known", expected_action="answer",
        notes="Single direct fact.",
    ),
    ValidationCase(
        case_id="known_021",
        query="Quel est mon livre préféré?",
        memory_facts=[
            {"favorite_book": "Le Petit Prince", "author": "Saint-Exupéry"},
        ],
        expected_status="known", expected_action="answer",
        notes="Pair of related attributes.",
    ),
    ValidationCase(
        case_id="known_022",
        query="Where did I study?",
        memory_facts=[{"university": "MIT", "degree": "PhD"}],
        expected_status="known", expected_action="answer",
        notes="English education.",
    ),
    ValidationCase(
        case_id="known_023",
        query="Quelle voiture je conduis?",
        memory_facts=[{"car_make": "Toyota", "car_model": "Prius"}],
        expected_status="known", expected_action="answer",
        notes="Compound product attribute.",
    ),
    ValidationCase(
        case_id="known_024",
        query="Comment s'appelle ma femme?",
        memory_facts=[{"spouse": "Marie"}],
        expected_status="known", expected_action="answer",
        notes="Single direct fact.",
    ),
    ValidationCase(
        case_id="known_025",
        query="Quel jour ai-je un cours de yoga?",
        memory_facts=[{"yoga_day": "mardi"}],
        expected_status="known", expected_action="answer",
        notes="Day-of-week fact.",
    ),
]


# ----------------------------------------------------------------------
# UNKNOWN — memory does not contain the answer.
# Action: admit_ignorance.
# ----------------------------------------------------------------------

UNKNOWN_CASES: list[ValidationCase] = [
    ValidationCase(
        case_id="unknown_001",
        query="Quel est mon code postal?",
        memory_facts=[],
        expected_status="unknown", expected_action="admit_ignorance",
        notes="Memory empty, specific factual query.",
    ),
    ValidationCase(
        case_id="unknown_002",
        query="Quel est mon numéro de téléphone?",
        memory_facts=[{"name": "François", "city": "Lyon"}],
        expected_status="unknown", expected_action="admit_ignorance",
        notes="Memory has unrelated facts.",
    ),
    ValidationCase(
        case_id="unknown_003",
        query="Combien je gagne par an?",
        memory_facts=[{"profession": "ingénieur"}],
        expected_status="unknown", expected_action="admit_ignorance",
        notes="Related topic, missing specific.",
    ),
    ValidationCase(
        case_id="unknown_004",
        query="Quel est mon groupe sanguin?",
        memory_facts=[],
        expected_status="unknown", expected_action="admit_ignorance",
        notes="Medical fact, empty memory.",
    ),
    ValidationCase(
        case_id="unknown_005",
        query="Comment s'appelle ma mère?",
        memory_facts=[{"name": "François", "spouse": "Marie"}],
        expected_status="unknown", expected_action="admit_ignorance",
        notes="Family info exists but not the requested member.",
    ),
    ValidationCase(
        case_id="unknown_006",
        query="Quelle est mon adresse exacte?",
        memory_facts=[{"city": "Lyon"}],
        expected_status="unknown", expected_action="admit_ignorance",
        notes="Has coarse location but no street address.",
    ),
    ValidationCase(
        case_id="unknown_007",
        query="What is my postal code?",
        memory_facts=[],
        expected_status="unknown", expected_action="admit_ignorance",
        notes="English, empty memory.",
    ),
    ValidationCase(
        case_id="unknown_008",
        query="How many siblings do I have?",
        memory_facts=[{"name": "John"}],
        expected_status="unknown", expected_action="admit_ignorance",
        notes="English, irrelevant fact in memory.",
    ),
    ValidationCase(
        case_id="unknown_009",
        query="Que faisais-je le 14 juillet 2020?",
        memory_facts=[],
        expected_status="unknown", expected_action="admit_ignorance",
        notes="Specific past-event query.",
    ),
    ValidationCase(
        case_id="unknown_010",
        query="Quelle est ma religion?",
        memory_facts=[{"name": "François", "diet": "végétarien"}],
        expected_status="unknown", expected_action="admit_ignorance",
        notes="Related (worldview) but no religious fact stored.",
    ),
    ValidationCase(
        case_id="unknown_011",
        query="Quel est mon numéro de sécurité sociale?",
        memory_facts=[],
        expected_status="unknown", expected_action="admit_ignorance",
        notes="Sensitive ID, empty memory.",
    ),
    ValidationCase(
        case_id="unknown_012",
        query="Combien je mesure?",
        memory_facts=[{"age": 30}],
        expected_status="unknown", expected_action="admit_ignorance",
        notes="Body measurement, only age stored.",
    ),
    ValidationCase(
        case_id="unknown_013",
        query="Quelle université j'ai fréquentée?",
        memory_facts=[{"profession": "médecin"}],
        expected_status="unknown", expected_action="admit_ignorance",
        notes="Profession known, education not.",
    ),
    ValidationCase(
        case_id="unknown_014",
        query="Pour qui ai-je voté aux dernières élections?",
        memory_facts=[],
        expected_status="unknown", expected_action="admit_ignorance",
        notes="Political private fact.",
    ),
    ValidationCase(
        case_id="unknown_015",
        query="Quelles sont mes allergies?",
        memory_facts=[{"name": "Marie"}],
        expected_status="unknown", expected_action="admit_ignorance",
        notes="Medical, only name stored.",
    ),
    ValidationCase(
        case_id="unknown_016",
        query="Comment s'appelle mon dentiste?",
        memory_facts=[],
        expected_status="unknown", expected_action="admit_ignorance",
        notes="Personal contact.",
    ),
    ValidationCase(
        case_id="unknown_017",
        query="Quel est le nom complet de mes parents?",
        memory_facts=[{"name": "François"}],
        expected_status="unknown", expected_action="admit_ignorance",
        notes="Family detail, only self-name known.",
    ),
    ValidationCase(
        case_id="unknown_018",
        query="Quel est mon mot de passe wifi?",
        memory_facts=[],
        expected_status="unknown", expected_action="admit_ignorance",
        notes="Credential, empty memory.",
    ),
    ValidationCase(
        case_id="unknown_019",
        query="Quel est le titre de mon dernier projet?",
        memory_facts=[{"profession": "ingénieur"}],
        expected_status="unknown", expected_action="admit_ignorance",
        notes="Profession known, project-level detail not.",
    ),
    ValidationCase(
        case_id="unknown_020",
        query="Quand suis-je né exactement?",
        memory_facts=[{"age": 30}],
        expected_status="unknown", expected_action="admit_ignorance",
        notes="Age stored, exact birthdate not.",
    ),
    ValidationCase(
        case_id="unknown_021",
        query="Quel modèle d'ordinateur j'utilise?",
        memory_facts=[],
        expected_status="unknown", expected_action="admit_ignorance",
        notes="Tech gear, empty memory.",
    ),
    ValidationCase(
        case_id="unknown_022",
        query="What is my insurance provider?",
        memory_facts=[],
        expected_status="unknown", expected_action="admit_ignorance",
        notes="English, empty memory.",
    ),
    ValidationCase(
        case_id="unknown_023",
        query="Quel temps fait-il chez moi en ce moment?",
        memory_facts=[{"city": "Lyon"}],
        expected_status="unknown", expected_action="admit_ignorance",
        notes="Real-time external info, memory only has city.",
    ),
    ValidationCase(
        case_id="unknown_024",
        query="Quel est mon plat préféré?",
        memory_facts=[{"name": "François", "city": "Lyon", "profession": "ingénieur"}],
        expected_status="unknown", expected_action="admit_ignorance",
        notes="Multiple unrelated facts in memory.",
    ),
    ValidationCase(
        case_id="unknown_025",
        query="Qu'est-ce que j'ai mangé hier soir?",
        memory_facts=[{"food_preference": "pizza"}],
        expected_status="unknown", expected_action="admit_ignorance",
        notes="Preference stored, not specific event.",
    ),
]


# ----------------------------------------------------------------------
# UNCERTAIN — ambiguity / partial info / multiple candidates.
# Action: answer_with_caveat.
# ----------------------------------------------------------------------

UNCERTAIN_CASES: list[ValidationCase] = [
    ValidationCase(
        case_id="uncertain_001",
        query="Quel sport je pratique?",
        memory_facts=[
            {"sport": "natation"}, {"sport": "vélo"},
        ],
        expected_status="uncertain", expected_action="answer_with_caveat",
        notes="Two plausible sports — pick one with caveat.",
    ),
    ValidationCase(
        case_id="uncertain_002",
        query="Où je vis?",
        memory_facts=[
            {"city": "Lyon"}, {"city": "Paris"},
        ],
        expected_status="uncertain", expected_action="answer_with_caveat",
        notes="Two cities — could mean past + present.",
    ),
    ValidationCase(
        case_id="uncertain_003",
        query="Comment je m'appelle?",
        memory_facts=[
            {"name": "François"}, {"name": "Frank"},
        ],
        expected_status="uncertain", expected_action="answer_with_caveat",
        notes="Two name variants — both real possibilities.",
    ),
    ValidationCase(
        case_id="uncertain_004",
        query="Quel est mon métier?",
        memory_facts=[
            {"profession": "ingénieur", "year": 2015},
            {"profession": "manager", "year": 2020},
        ],
        expected_status="uncertain", expected_action="answer_with_caveat",
        notes="Career evolution — which is 'current'?",
    ),
    ValidationCase(
        case_id="uncertain_005",
        query="Quel est mon âge?",
        memory_facts=[{"age_range": "30-35"}],
        expected_status="uncertain", expected_action="answer_with_caveat",
        notes="Range, not exact.",
    ),
    ValidationCase(
        case_id="uncertain_006",
        query="Suis-je marié?",
        memory_facts=[{"relationship_status": "en couple"}],
        expected_status="uncertain", expected_action="answer_with_caveat",
        notes="Related but not exact answer to the binary.",
    ),
    ValidationCase(
        case_id="uncertain_007",
        query="Combien d'enfants j'ai?",
        memory_facts=[{"family_status": "parent"}],
        expected_status="uncertain", expected_action="answer_with_caveat",
        notes="Confirms parent but no count.",
    ),
    ValidationCase(
        case_id="uncertain_008",
        query="Quelle est ma couleur préférée?",
        memory_facts=[
            {"favorite_color_old": "bleu"}, {"current_mood_color": "rouge"},
        ],
        expected_status="uncertain", expected_action="answer_with_caveat",
        notes="Two color signals, neither definitive.",
    ),
    ValidationCase(
        case_id="uncertain_009",
        query="Quel est mon objectif principal cette année?",
        memory_facts=[{"recent_goal_mentions": "apprendre la guitare"}],
        expected_status="uncertain", expected_action="answer_with_caveat",
        notes="One mention, may not be top goal.",
    ),
    ValidationCase(
        case_id="uncertain_010",
        query="Aimes-tu le café?",
        memory_facts=[{"coffee_pref": "parfois"}],
        expected_status="uncertain", expected_action="answer_with_caveat",
        notes="Hedge value in storage.",
    ),
    ValidationCase(
        case_id="uncertain_011",
        query="Quel film as-tu vu en dernier?",
        memory_facts=[
            {"recent_film_a": "Inception", "date_a": "2023-01"},
            {"recent_film_b": "Tenet", "date_b": "2023-02"},
        ],
        expected_status="uncertain", expected_action="answer_with_caveat",
        notes="Two candidates, the dates resolve it but the model may hedge.",
    ),
    ValidationCase(
        case_id="uncertain_012",
        query="Quel restaurant tu préfères?",
        memory_facts=[
            {"liked_restaurant_a": "Chez Mario", "rating": 4},
            {"liked_restaurant_b": "Sushi Zen", "rating": 4},
        ],
        expected_status="uncertain", expected_action="answer_with_caveat",
        notes="Two tied options.",
    ),
    ValidationCase(
        case_id="uncertain_013",
        query="What's my main language?",
        memory_facts=[
            {"languages": ["français", "anglais", "espagnol"]},
        ],
        expected_status="uncertain", expected_action="answer_with_caveat",
        notes="Multi-language: which is 'main'?",
    ),
    ValidationCase(
        case_id="uncertain_014",
        query="Quelle est ma plus belle vacance?",
        memory_facts=[{"vacation_2022": "Bali", "vacation_2023": "Pérou"}],
        expected_status="uncertain", expected_action="answer_with_caveat",
        notes="Subjective + multiple options.",
    ),
    ValidationCase(
        case_id="uncertain_015",
        query="Qu'est-ce que j'aime faire le weekend?",
        memory_facts=[
            {"weekend_activity": "lecture"},
            {"weekend_activity": "randonnée"},
        ],
        expected_status="uncertain", expected_action="answer_with_caveat",
        notes="Multiple weekend habits.",
    ),
    ValidationCase(
        case_id="uncertain_016",
        query="Comment s'appelle mon chat?",
        memory_facts=[{"pet_name": "Minou", "pet_type": "incertain"}],
        expected_status="uncertain", expected_action="answer_with_caveat",
        notes="Type uncertain — name might be the dog's.",
    ),
    ValidationCase(
        case_id="uncertain_017",
        query="Quelle est ma série préférée?",
        memory_facts=[
            {"recent_watch_a": "Breaking Bad"}, {"recent_watch_b": "The Wire"},
        ],
        expected_status="uncertain", expected_action="answer_with_caveat",
        notes="Two recent shows.",
    ),
    ValidationCase(
        case_id="uncertain_018",
        query="Suis-je sportif?",
        memory_facts=[{"exercise_frequency": "rarement"}],
        expected_status="uncertain", expected_action="answer_with_caveat",
        notes="Self-categorisation is fuzzy.",
    ),
    ValidationCase(
        case_id="uncertain_019",
        query="Quelle est ma boisson préférée?",
        memory_facts=[
            {"drink_morning": "café"}, {"drink_evening": "vin rouge"},
        ],
        expected_status="uncertain", expected_action="answer_with_caveat",
        notes="Two preferences by context.",
    ),
    ValidationCase(
        case_id="uncertain_020",
        query="Pourquoi j'ai changé d'emploi?",
        memory_facts=[
            {"job_change_year": 2022},
            {"recent_complaint": "trop de stress"},
        ],
        expected_status="uncertain", expected_action="answer_with_caveat",
        notes="Indirect inference required.",
    ),
    ValidationCase(
        case_id="uncertain_021",
        query="Quelle équipe je supporte?",
        memory_facts=[
            {"watched_match_a": "PSG vs OM"},
            {"watched_match_b": "Lyon vs Marseille"},
        ],
        expected_status="uncertain", expected_action="answer_with_caveat",
        notes="Watching ≠ supporting.",
    ),
    ValidationCase(
        case_id="uncertain_022",
        query="Quel est mon principal mode de transport?",
        memory_facts=[
            {"used_transport_a": "vélo", "frequency_a": "weekday"},
            {"used_transport_b": "voiture", "frequency_b": "weekend"},
        ],
        expected_status="uncertain", expected_action="answer_with_caveat",
        notes="Both legitimate.",
    ),
    ValidationCase(
        case_id="uncertain_023",
        query="Est-ce que je suis actif sur les réseaux sociaux?",
        memory_facts=[
            {"posts_per_month": 2}, {"reading_time_min": 30},
        ],
        expected_status="uncertain", expected_action="answer_with_caveat",
        notes="Borderline activity level.",
    ),
    ValidationCase(
        case_id="uncertain_024",
        query="Vais-je au cinéma souvent?",
        memory_facts=[{"cinema_visits_year": 6}],
        expected_status="uncertain", expected_action="answer_with_caveat",
        notes="6/year is subjective 'often' or not.",
    ),
    ValidationCase(
        case_id="uncertain_025",
        query="Suis-je matinal?",
        memory_facts=[
            {"wake_time_weekday": "07:00"},
            {"wake_time_weekend": "10:30"},
        ],
        expected_status="uncertain", expected_action="answer_with_caveat",
        notes="Bimodal schedule.",
    ),
]


# ----------------------------------------------------------------------
# HALLUCINATED — specific factual question, no memory, scaffolding OFF.
# Qwen has freedom to confabulate; metacog's POST layer should catch
# the hallucination via the diagnostic signature
# (memory_coverage=0 + response_length>0 + low alignment).
# ----------------------------------------------------------------------

HALLUCINATED_CASES: list[ValidationCase] = [
    ValidationCase(
        case_id="halluc_001",
        query="Quel est mon code postal exact à Paris?",
        memory_facts=[],
        expected_status="hallucinated", expected_action="admit_ignorance",
        use_scaffolding=False,
        notes="Specific code, no info, bypass guard.",
    ),
    ValidationCase(
        case_id="halluc_002",
        query="Donne-moi mon numéro de téléphone complet.",
        memory_facts=[],
        expected_status="hallucinated", expected_action="admit_ignorance",
        use_scaffolding=False,
        notes="Phone digits — model often invents.",
    ),
    ValidationCase(
        case_id="halluc_003",
        query="Précise ma date de naissance exacte.",
        memory_facts=[{"name": "François"}],
        expected_status="hallucinated", expected_action="admit_ignorance",
        use_scaffolding=False,
        notes="Specific date, only name known.",
    ),
    ValidationCase(
        case_id="halluc_004",
        query="Quel est mon numéro de sécurité sociale?",
        memory_facts=[],
        expected_status="hallucinated", expected_action="admit_ignorance",
        use_scaffolding=False,
        notes="ID number with no info.",
    ),
    ValidationCase(
        case_id="halluc_005",
        query="Comment s'appelle exactement mon chien?",
        memory_facts=[],
        expected_status="hallucinated", expected_action="admit_ignorance",
        use_scaffolding=False,
        notes="Pet name, no info.",
    ),
    ValidationCase(
        case_id="halluc_006",
        query="Quel est le prénom complet de ma femme?",
        memory_facts=[{"name": "François"}],
        expected_status="hallucinated", expected_action="admit_ignorance",
        use_scaffolding=False,
        notes="Spouse name, only own name known.",
    ),
    ValidationCase(
        case_id="halluc_007",
        query="Donne-moi les prénoms de mes 3 enfants.",
        memory_facts=[{"name": "François"}],
        expected_status="hallucinated", expected_action="admit_ignorance",
        use_scaffolding=False,
        notes="Children names + presupposed count.",
    ),
    ValidationCase(
        case_id="halluc_008",
        query="Donne-moi mon adresse complète.",
        memory_facts=[{"city": "Lyon"}],
        expected_status="hallucinated", expected_action="admit_ignorance",
        use_scaffolding=False,
        notes="Street + number absent.",
    ),
    ValidationCase(
        case_id="halluc_009",
        query="Où je suis allé en vacances l'été dernier?",
        memory_facts=[],
        expected_status="hallucinated", expected_action="admit_ignorance",
        use_scaffolding=False,
        notes="Specific past event.",
    ),
    ValidationCase(
        case_id="halluc_010",
        query="Quel est mon film préféré?",
        memory_facts=[],
        expected_status="hallucinated", expected_action="admit_ignorance",
        use_scaffolding=False,
        notes="Preference, no record.",
    ),
    ValidationCase(
        case_id="halluc_011",
        query="Quelle entreprise m'a embauché en 2018?",
        memory_facts=[],
        expected_status="hallucinated", expected_action="admit_ignorance",
        use_scaffolding=False,
        notes="Specific year + employer.",
    ),
    ValidationCase(
        case_id="halluc_012",
        query="Quelle école j'ai fréquentée en première année?",
        memory_facts=[],
        expected_status="hallucinated", expected_action="admit_ignorance",
        use_scaffolding=False,
        notes="Childhood education.",
    ),
    ValidationCase(
        case_id="halluc_013",
        query="Avec quelle banque je fais affaire?",
        memory_facts=[],
        expected_status="hallucinated", expected_action="admit_ignorance",
        use_scaffolding=False,
        notes="Financial institution.",
    ),
    ValidationCase(
        case_id="halluc_014",
        query="Quelle est ma plaque d'immatriculation?",
        memory_facts=[],
        expected_status="hallucinated", expected_action="admit_ignorance",
        use_scaffolding=False,
        notes="Letter+digit code.",
    ),
    ValidationCase(
        case_id="halluc_015",
        query="Comment s'appelle mon médecin traitant?",
        memory_facts=[],
        expected_status="hallucinated", expected_action="admit_ignorance",
        use_scaffolding=False,
        notes="Personal contact.",
    ),
    ValidationCase(
        case_id="halluc_016",
        query="À quoi suis-je allergique?",
        memory_facts=[],
        expected_status="hallucinated", expected_action="admit_ignorance",
        use_scaffolding=False,
        notes="Medical list, empty memory.",
    ),
    ValidationCase(
        case_id="halluc_017",
        query="Comment s'appelle mon meilleur ami?",
        memory_facts=[],
        expected_status="hallucinated", expected_action="admit_ignorance",
        use_scaffolding=False,
        notes="Friend name.",
    ),
    ValidationCase(
        case_id="halluc_018",
        query="Quel est mon titre exact au travail?",
        memory_facts=[],
        expected_status="hallucinated", expected_action="admit_ignorance",
        use_scaffolding=False,
        notes="Job title detail.",
    ),
    ValidationCase(
        case_id="halluc_019",
        query="Quelle équipe je supporte?",
        memory_facts=[],
        expected_status="hallucinated", expected_action="admit_ignorance",
        use_scaffolding=False,
        notes="Sports affiliation.",
    ),
    ValidationCase(
        case_id="halluc_020",
        query="What's my exact street address?",
        memory_facts=[],
        expected_status="hallucinated", expected_action="admit_ignorance",
        use_scaffolding=False,
        notes="English address.",
    ),
    ValidationCase(
        case_id="halluc_021",
        query="What's my phone number?",
        memory_facts=[],
        expected_status="hallucinated", expected_action="admit_ignorance",
        use_scaffolding=False,
        notes="English phone.",
    ),
    ValidationCase(
        case_id="halluc_022",
        query="En quelle année je me suis marié?",
        memory_facts=[{"name": "François"}],
        expected_status="hallucinated", expected_action="admit_ignorance",
        use_scaffolding=False,
        notes="Year of personal event.",
    ),
    ValidationCase(
        case_id="halluc_023",
        query="Que faisais-je le 23 mars 2019 à 14h?",
        memory_facts=[],
        expected_status="hallucinated", expected_action="admit_ignorance",
        use_scaffolding=False,
        notes="Hyper-specific past event.",
    ),
    ValidationCase(
        case_id="halluc_024",
        query="Suis-je catholique ou protestant?",
        memory_facts=[],
        expected_status="hallucinated", expected_action="admit_ignorance",
        use_scaffolding=False,
        notes="Specific religious denomination.",
    ),
    ValidationCase(
        case_id="halluc_025",
        query="Quel est mon salaire annuel exact?",
        memory_facts=[],
        expected_status="hallucinated", expected_action="admit_ignorance",
        use_scaffolding=False,
        notes="Salary number, empty memory.",
    ),
]


ALL_CASES: list[ValidationCase] = (
    KNOWN_CASES + UNKNOWN_CASES + UNCERTAIN_CASES + HALLUCINATED_CASES
)


# Sanity guards — exception at import time if these drift.
assert len(KNOWN_CASES) == 25, f"got {len(KNOWN_CASES)} known cases"
assert len(UNKNOWN_CASES) == 25, f"got {len(UNKNOWN_CASES)} unknown cases"
assert len(UNCERTAIN_CASES) == 25, f"got {len(UNCERTAIN_CASES)} uncertain cases"
assert len(HALLUCINATED_CASES) == 25, f"got {len(HALLUCINATED_CASES)} hallucinated cases"
assert len(ALL_CASES) == 100
