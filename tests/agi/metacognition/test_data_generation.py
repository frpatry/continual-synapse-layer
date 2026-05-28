"""Tests for the synthetic metacog training-data generator.

Class-distinguishability is verified with Welch's t-test (via
``scipy.stats.ttest_ind(equal_var=False)``) on the key per-class
diagnostic features. With n=1000 per class, the calibrated
distributions should produce ``p < 0.001`` on every checked pair.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from scipy import stats

from agi.metacognition.data_generation import (
    POST_CLASSES,
    PRE_CLASSES,
    SyntheticDataGenerator,
    TrainingExample,
    load_dataset,
    save_dataset,
    split_train_val,
)
from agi.metacognition.features import (
    MEMORY_FEATURE_NAMES,
    POST_FEATURE_ORDER,
    PRE_FEATURE_ORDER,
    QUERY_FEATURE_NAMES,
)


_N_DISTRIBUTION_SAMPLES = 200  # bigger samples in the distinguishability test


# ---------- per-class shape ----------

def test_generate_known_features_in_expected_range():
    g = SyntheticDataGenerator(seed=1)
    exs = [g.generate_known_example() for _ in range(100)]
    max_sims = [ex.features["max_similarity"] for ex in exs]
    precs = [ex.features["precision_quality"] for ex in exs]
    # Beta(8,2)*0.25 + 0.7 → roughly [0.7, 0.95].
    assert min(max_sims) >= 0.70 - 1e-6
    assert max(max_sims) <= 0.95 + 1e-6
    # Beta(8,2) is concentrated near 0.8 — mean should clearly
    # exceed 0.5.
    assert sum(precs) / len(precs) > 0.5


def test_generate_unknown_features_in_expected_range():
    g = SyntheticDataGenerator(seed=2)
    exs = [g.generate_unknown_example() for _ in range(100)]
    # max_similarity should hug zero (most have n_facts == 0).
    max_sims = [ex.features["max_similarity"] for ex in exs]
    assert max(max_sims) < 0.31
    # Alignment slots are ALL zero per the empty-facts contract.
    for ex in exs:
        assert ex.features["alignment_max_cosine"] == 0.0
        assert ex.features["alignment_mean_cosine"] == 0.0
        assert ex.features["alignment_novel_token_ratio"] == 0.0


def test_generate_uncertain_features_in_expected_range():
    g = SyntheticDataGenerator(seed=3)
    exs = [g.generate_uncertain_example() for _ in range(200)]
    max_sims = [ex.features["max_similarity"] for ex in exs]
    sim_vars = [ex.features["similarity_variance"] for ex in exs]
    # Beta(5,5)*0.3 + 0.4 → roughly [0.4, 0.7].
    assert min(max_sims) >= 0.40 - 1e-6
    assert max(max_sims) <= 0.70 + 1e-6
    # Beta(5,2)*0.15 has mean ≈ (5/7)*0.15 ≈ 0.107. So mean of
    # similarity_variance across 200 samples should comfortably
    # exceed 0.05.
    assert sum(sim_vars) / len(sim_vars) > 0.05


def test_generate_hallucinated_features_diagnostic_pattern():
    """Hallucinated signature: low memory + long response + low
    attention + low alignment_max + HIGH novel_token_ratio (when
    facts are present). Sample 200 then assert the *average*
    pattern (individual draws have noise)."""
    g = SyntheticDataGenerator(seed=4)
    exs = [g.generate_hallucinated_example() for _ in range(200)]

    avg_resp_len = sum(
        ex.features["response_length_tokens"] for ex in exs
    ) / len(exs)
    avg_att = sum(
        ex.features["attention_to_facts_mean"] for ex in exs
    ) / len(exs)
    # Long: Poisson(30)+10 → mean ≈ 40.
    assert avg_resp_len > 30
    # LOW attention: Beta(2,8)*0.3 → mean ~ 0.06.
    assert avg_att < 0.2

    # For the subset WITH facts, novel-token ratio should be HIGH.
    has_facts = [
        ex for ex in exs if ex.features["n_facts_retrieved"] > 0
    ]
    if has_facts:
        avg_novel = sum(
            ex.features["alignment_novel_token_ratio"] for ex in has_facts
        ) / len(has_facts)
        # Beta(8,2)*0.3 + 0.6 → mean ≈ 0.84.
        assert avg_novel > 0.6


# ---------- distinguishability ----------

def _samples(
    gen: SyntheticDataGenerator,
    method_name: str,
    n: int,
    feature: str,
) -> list[float]:
    method = getattr(gen, method_name)
    return [method().features[feature] for _ in range(n)]


def _welch_pvalue(a: list[float], b: list[float]) -> float:
    res = stats.ttest_ind(a, b, equal_var=False)
    return float(res.pvalue)


def test_classes_are_distinguishable_max_similarity():
    """``max_similarity`` should reliably separate known (high) from
    unknown (≈0) and uncertain (medium)."""
    g = SyntheticDataGenerator(seed=10)
    known = _samples(g, "generate_known_example", _N_DISTRIBUTION_SAMPLES, "max_similarity")
    unknown = _samples(g, "generate_unknown_example", _N_DISTRIBUTION_SAMPLES, "max_similarity")
    uncertain = _samples(g, "generate_uncertain_example", _N_DISTRIBUTION_SAMPLES, "max_similarity")
    assert _welch_pvalue(known, unknown) < 1e-3
    assert _welch_pvalue(known, uncertain) < 1e-3
    assert _welch_pvalue(uncertain, unknown) < 1e-3


def test_classes_are_distinguishable_novel_token_ratio():
    """``alignment_novel_token_ratio`` distinguishes hallucinated
    (high) from the rest. Focus the contrast on hallucinated
    samples WITH facts (the empty-facts gate zeroes the slot for
    the others)."""
    g = SyntheticDataGenerator(seed=11)
    # Draw hallucinated until we have N samples whose alignment
    # slot wasn't zeroed by the empty-facts gate.
    halluc_with_facts: list[float] = []
    while len(halluc_with_facts) < _N_DISTRIBUTION_SAMPLES:
        ex = g.generate_hallucinated_example()
        if ex.features["n_facts_retrieved"] > 0:
            halluc_with_facts.append(
                ex.features["alignment_novel_token_ratio"]
            )
    known = _samples(g, "generate_known_example", _N_DISTRIBUTION_SAMPLES, "alignment_novel_token_ratio")
    uncertain = _samples(g, "generate_uncertain_example", _N_DISTRIBUTION_SAMPLES, "alignment_novel_token_ratio")
    assert _welch_pvalue(halluc_with_facts, known) < 1e-3
    assert _welch_pvalue(halluc_with_facts, uncertain) < 1e-3


def test_classes_are_distinguishable_attention_to_facts():
    """``attention_to_facts_mean`` is high for known, low for
    hallucinated."""
    g = SyntheticDataGenerator(seed=12)
    known = _samples(g, "generate_known_example", _N_DISTRIBUTION_SAMPLES, "attention_to_facts_mean")
    halluc = _samples(g, "generate_hallucinated_example", _N_DISTRIBUTION_SAMPLES, "attention_to_facts_mean")
    assert _welch_pvalue(known, halluc) < 1e-3
    assert sum(known) / len(known) > sum(halluc) / len(halluc)


# ---------- generate_batch ----------

def test_generate_batch_equilibre():
    g = SyntheticDataGenerator(seed=20)
    classes = list(POST_CLASSES)
    batch = g.generate_batch(n_per_class=100, classes=classes)
    assert len(batch) == 100 * len(classes)
    counts = Counter(ex.status for ex in batch)
    for cls in classes:
        assert counts[cls] == 100


def test_generate_batch_rejects_unknown_class():
    g = SyntheticDataGenerator(seed=21)
    with pytest.raises(ValueError):
        g.generate_batch(n_per_class=1, classes=["not-a-real-class"])  # type: ignore[list-item]


# ---------- split_train_val ----------

def test_split_train_val_keeps_balance():
    g = SyntheticDataGenerator(seed=30)
    examples = g.generate_batch(n_per_class=200, classes=list(POST_CLASSES))
    train, val = split_train_val(examples, val_ratio=0.2, seed=30)
    assert len(train) + len(val) == len(examples)
    # With n=800 total and 20% val, exact-balance isn't required —
    # each class should land in val somewhere in the range
    # [0.2 * 200 - 30, 0.2 * 200 + 30] = [10, 70].
    val_counts = Counter(ex.status for ex in val)
    for cls in POST_CLASSES:
        assert 10 <= val_counts[cls] <= 70


def test_split_train_val_rejects_bad_ratio():
    with pytest.raises(ValueError):
        split_train_val([], val_ratio=0.0)
    with pytest.raises(ValueError):
        split_train_val([], val_ratio=1.0)


# ---------- save / load ----------

def test_save_load_roundtrip_post(tmp_path):
    g = SyntheticDataGenerator(seed=40)
    examples = g.generate_batch(n_per_class=5, classes=list(POST_CLASSES))
    path = tmp_path / "post.jsonl"
    save_dataset(examples, path, mode="post")
    loaded = load_dataset(path)

    assert len(loaded) == len(examples)
    for original, restored in zip(examples, loaded):
        assert restored.status == original.status
        assert restored.confidence == pytest.approx(original.confidence)
        # Persisted feature set is the full 18.
        assert set(restored.features.keys()) == set(POST_FEATURE_ORDER)
        for k in POST_FEATURE_ORDER:
            assert restored.features[k] == pytest.approx(
                original.features[k], abs=1e-9,
            )


def test_save_load_roundtrip_pre(tmp_path):
    g = SyntheticDataGenerator(seed=41)
    examples = g.generate_batch(n_per_class=5, classes=list(PRE_CLASSES))
    path = tmp_path / "pre.jsonl"
    save_dataset(examples, path, mode="pre")
    loaded = load_dataset(path)
    for restored in loaded:
        # Pre-mode persistence projects to 10 features only.
        assert set(restored.features.keys()) == set(PRE_FEATURE_ORDER)


def test_save_dataset_rejects_bad_mode(tmp_path):
    g = SyntheticDataGenerator(seed=42)
    with pytest.raises(ValueError):
        save_dataset(
            [g.generate_known_example()],
            tmp_path / "x.jsonl",
            mode="bogus",
        )


# ---------- determinism ----------

def test_deterministic_with_seed():
    a = SyntheticDataGenerator(seed=99)
    b = SyntheticDataGenerator(seed=99)
    for _ in range(20):
        ex_a = a.generate_known_example()
        ex_b = b.generate_known_example()
        for k in ex_a.features:
            assert ex_a.features[k] == ex_b.features[k]


# ---------- feature naming alignment ----------

def test_features_match_metacog_layer_pre_input_dim():
    """A PRE example's persisted feature dict must be exactly the
    10 keys of PRE_FEATURE_ORDER — otherwise the metacog layer
    will read zeros for the missing ones."""
    g = SyntheticDataGenerator(seed=50)
    ex = g.generate_known_example()
    # Project to pre — what save_dataset does.
    from agi.metacognition.data_generation import _project_features
    pre_features = _project_features(ex.features, mode="pre")
    assert set(pre_features.keys()) == set(PRE_FEATURE_ORDER)
    assert len(pre_features) == 10
    assert set(pre_features.keys()) == set(MEMORY_FEATURE_NAMES) | set(QUERY_FEATURE_NAMES)


def test_features_match_metacog_layer_post_input_dim():
    """A POST example must carry all 18 POST_FEATURE_ORDER keys."""
    g = SyntheticDataGenerator(seed=51)
    ex = g.generate_hallucinated_example()
    from agi.metacognition.data_generation import _project_features
    post_features = _project_features(ex.features, mode="post")
    assert set(post_features.keys()) == set(POST_FEATURE_ORDER)
    assert len(post_features) == 18


# ---------- committed sample fixtures ----------

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SAMPLE_DIR = _REPO_ROOT / "data" / "metacog"


@pytest.mark.parametrize(
    "filename,expected_keys,expected_classes",
    [
        ("sample_pre_train.jsonl", PRE_FEATURE_ORDER, PRE_CLASSES),
        ("sample_pre_val.jsonl", PRE_FEATURE_ORDER, PRE_CLASSES),
        ("sample_post_train.jsonl", POST_FEATURE_ORDER, POST_CLASSES),
        ("sample_post_val.jsonl", POST_FEATURE_ORDER, POST_CLASSES),
    ],
)
def test_sample_dataset_files_present_and_well_formed(
    filename, expected_keys, expected_classes,
):
    """The four committed sample files exist and round-trip cleanly
    through ``load_dataset``."""
    path = _SAMPLE_DIR / filename
    assert path.exists(), f"missing committed sample fixture: {path}"
    loaded = load_dataset(path)
    assert len(loaded) > 0, f"empty sample fixture: {path}"
    for ex in loaded:
        assert isinstance(ex, TrainingExample)
        assert ex.status in expected_classes
        assert set(ex.features.keys()) == set(expected_keys)
        assert 0.0 <= ex.confidence <= 1.0


# ---------- CLI smoke ----------

def test_cli_sample_mode_produces_files(tmp_path):
    """End-to-end smoke: running the script with --mode sample +
    a temp output dir produces the four expected files."""
    script = _REPO_ROOT / "scripts" / "generate_metacog_data.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            "sample",
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"CLI failed: {result.stderr}"
    )
    for name in (
        "sample_pre_train.jsonl",
        "sample_pre_val.jsonl",
        "sample_post_train.jsonl",
        "sample_post_val.jsonl",
    ):
        assert (tmp_path / name).exists(), f"missing {name}"
