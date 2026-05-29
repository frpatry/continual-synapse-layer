# Phase 2h — Distribution Recalibration Report

Compares the metacog's real-Qwen validation performance **Phase 2h (recalibrated, no verbatim)** vs **Phase 2h.1 (+ verbatim_fact_match)** on the same 100 hand-crafted test cases.

**Recalibration source:** drift analysis at `results/agi/distribution_drift_report.md` (computed by `scripts/analyze_distribution_drift.py`). The Phase 2c v1 synthetic distributions were updated to match empirical Qwen2.5-1.5B distributions across `alignment_novel_token_ratio`, `attention_to_facts_mean`, `alignment_max_cosine`, `response_length_tokens`, and the unknown-with-facts alignment branch.

## Headline — before vs after

| metric | before | after | Δ |
|---|---:|---:|---:|
| PRE accuracy (excl. hallucinated) | 0.453 (75n) | 0.467 (75n) | +0.013 |
| POST accuracy (all classes) | 0.600 (100n) | 0.640 (100n) | +0.040 |
| PRE real-data ECE | 0.641 | 0.637 | -0.003 |
| POST real-data ECE | 0.395 | 0.355 | -0.039 |

### Verdict: **YELLOW**

Generalises but with non-trivial residual errors. Consider further recalibration or expanding the test set before committing GPU compute.

## Per-class POST F1 — before vs after

| class | F1 before | F1 after | Δ |
|---|---:|---:|---:|
| known | 0.622 | 0.717 | +0.095 |
| unknown | 0.529 | 0.513 | -0.017 |
| uncertain | 0.500 | 0.509 | +0.009 |
| hallucinated | 0.737 | 0.792 | +0.056 |

## POST confusion matrices

### Phase 2h (recalibrated, no verbatim)

| expected ↓ / predicted → | known | unknown | uncertain | hallucinated |
|---|---:|---:|---:|---:|
| known | 14 | 0 | 9 | 2 |
| unknown | 2 | 9 | 10 | 4 |
| uncertain | 4 | 0 | 16 | 5 |
| hallucinated | 0 | 0 | 4 | 21 |

### Phase 2h.1 (+ verbatim_fact_match)

| expected ↓ / predicted → | known | unknown | uncertain | hallucinated |
|---|---:|---:|---:|---:|
| known | 19 | 1 | 4 | 1 |
| unknown | 3 | 10 | 9 | 3 |
| uncertain | 6 | 2 | 14 | 3 |
| hallucinated | 0 | 1 | 3 | 21 |

## PRE confusion matrices

### Phase 2h (recalibrated, no verbatim)

| expected ↓ / predicted → | known | unknown | uncertain | hallucinated |
|---|---:|---:|---:|---:|
| known | 5 | 1 | 19 | 0 |
| unknown | 1 | 11 | 13 | 0 |
| uncertain | 1 | 6 | 18 | 0 |
| hallucinated | 0 | 20 | 5 | 0 |

### Phase 2h.1 (+ verbatim_fact_match)

| expected ↓ / predicted → | known | unknown | uncertain | hallucinated |
|---|---:|---:|---:|---:|
| known | 6 | 0 | 19 | 0 |
| unknown | 1 | 10 | 14 | 0 |
| uncertain | 1 | 5 | 19 | 0 |
| hallucinated | 0 | 20 | 5 | 0 |

## POST per-class metrics — full

### Phase 2h (recalibrated, no verbatim)

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| known | 0.700 | 0.560 | 0.622 | 14 | 6 | 11 |
| unknown | 1.000 | 0.360 | 0.529 | 9 | 0 | 16 |
| uncertain | 0.410 | 0.640 | 0.500 | 16 | 23 | 9 |
| hallucinated | 0.656 | 0.840 | 0.737 | 21 | 11 | 4 |

### Phase 2h.1 (+ verbatim_fact_match)

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| known | 0.679 | 0.760 | 0.717 | 19 | 9 | 6 |
| unknown | 0.714 | 0.400 | 0.513 | 10 | 4 | 15 |
| uncertain | 0.467 | 0.560 | 0.509 | 14 | 16 | 11 |
| hallucinated | 0.750 | 0.840 | 0.792 | 21 | 7 | 4 |

## PRE per-class metrics — full

### Phase 2h (recalibrated, no verbatim)

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| known | 0.714 | 0.200 | 0.312 | 5 | 2 | 20 |
| unknown | 0.289 | 0.440 | 0.349 | 11 | 27 | 14 |
| uncertain | 0.327 | 0.720 | 0.450 | 18 | 37 | 7 |
| hallucinated | 0.000 | 0.000 | 0.000 | 0 | 0 | 25 |

### Phase 2h.1 (+ verbatim_fact_match)

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| known | 0.750 | 0.240 | 0.364 | 6 | 2 | 19 |
| unknown | 0.286 | 0.400 | 0.333 | 10 | 25 | 15 |
| uncertain | 0.333 | 0.760 | 0.463 | 19 | 38 | 6 |
| hallucinated | 0.000 | 0.000 | 0.000 | 0 | 0 | 25 |

## Top POST error patterns after recalibration

- **unknown → uncertain** (9 cases): unknown_002, unknown_003, unknown_006, unknown_010, unknown_012, unknown_017, unknown_019, unknown_020
- **uncertain → known** (6 cases): uncertain_001, uncertain_003, uncertain_009, uncertain_010, uncertain_016, uncertain_019
- **known → uncertain** (4 cases): known_003, known_006, known_016, known_021
- **unknown → known** (3 cases): unknown_005, unknown_015, unknown_025
- **unknown → hallucinated** (3 cases): unknown_009, unknown_013, unknown_018
- **uncertain → hallucinated** (3 cases): uncertain_014, uncertain_021, uncertain_024
- **hallucinated → uncertain** (3 cases): halluc_003, halluc_007, halluc_008
- **uncertain → unknown** (2 cases): uncertain_005, uncertain_023
- **known → unknown** (1 cases): known_007
- **known → hallucinated** (1 cases): known_013
- **hallucinated → unknown** (1 cases): halluc_006
