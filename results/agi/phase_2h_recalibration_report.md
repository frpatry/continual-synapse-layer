# Phase 2h — Distribution Recalibration Report

Compares the metacog's real-Qwen validation performance **Phase 2d.2 (Phase 2c v1 distributions)** vs **Phase 2h (recalibrated distributions)** on the same 100 hand-crafted test cases.

**Recalibration source:** drift analysis at `results/agi/distribution_drift_report.md` (computed by `scripts/analyze_distribution_drift.py`). The Phase 2c v1 synthetic distributions were updated to match empirical Qwen2.5-1.5B distributions across `alignment_novel_token_ratio`, `attention_to_facts_mean`, `alignment_max_cosine`, `response_length_tokens`, and the unknown-with-facts alignment branch.

## Headline — before vs after

| metric | before | after | Δ |
|---|---:|---:|---:|
| PRE accuracy (excl. hallucinated) | 0.467 (75n) | 0.453 (75n) | -0.013 |
| POST accuracy (all classes) | 0.530 (100n) | 0.600 (100n) | +0.070 |
| PRE real-data ECE | 0.641 | 0.641 | -0.001 |
| POST real-data ECE | 0.467 | 0.395 | -0.073 |

### Verdict: **YELLOW**

Generalises but with non-trivial residual errors. Consider further recalibration or expanding the test set before committing GPU compute.

## Per-class POST F1 — before vs after

| class | F1 before | F1 after | Δ |
|---|---:|---:|---:|
| known | 0.591 | 0.622 | +0.031 |
| unknown | 0.214 | 0.529 | +0.315 |
| uncertain | 0.517 | 0.500 | -0.017 |
| hallucinated | 0.629 | 0.737 | +0.108 |

## POST confusion matrices

### Phase 2d.2 (Phase 2c v1 distributions)

| expected ↓ / predicted → | known | unknown | uncertain | hallucinated |
|---|---:|---:|---:|---:|
| known | 13 | 0 | 8 | 4 |
| unknown | 2 | 3 | 7 | 13 |
| uncertain | 4 | 0 | 15 | 6 |
| hallucinated | 0 | 0 | 3 | 22 |

### Phase 2h (recalibrated distributions)

| expected ↓ / predicted → | known | unknown | uncertain | hallucinated |
|---|---:|---:|---:|---:|
| known | 14 | 0 | 9 | 2 |
| unknown | 2 | 9 | 10 | 4 |
| uncertain | 4 | 0 | 16 | 5 |
| hallucinated | 0 | 0 | 4 | 21 |

## PRE confusion matrices

### Phase 2d.2 (Phase 2c v1 distributions)

| expected ↓ / predicted → | known | unknown | uncertain | hallucinated |
|---|---:|---:|---:|---:|
| known | 25 | 0 | 0 | 0 |
| unknown | 15 | 10 | 0 | 0 |
| uncertain | 25 | 0 | 0 | 0 |
| hallucinated | 5 | 20 | 0 | 0 |

### Phase 2h (recalibrated distributions)

| expected ↓ / predicted → | known | unknown | uncertain | hallucinated |
|---|---:|---:|---:|---:|
| known | 5 | 1 | 19 | 0 |
| unknown | 1 | 11 | 13 | 0 |
| uncertain | 1 | 6 | 18 | 0 |
| hallucinated | 0 | 20 | 5 | 0 |

## POST per-class metrics — full

### Phase 2d.2 (Phase 2c v1 distributions)

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| known | 0.684 | 0.520 | 0.591 | 13 | 6 | 12 |
| unknown | 1.000 | 0.120 | 0.214 | 3 | 0 | 22 |
| uncertain | 0.455 | 0.600 | 0.517 | 15 | 18 | 10 |
| hallucinated | 0.489 | 0.880 | 0.629 | 22 | 23 | 3 |

### Phase 2h (recalibrated distributions)

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| known | 0.700 | 0.560 | 0.622 | 14 | 6 | 11 |
| unknown | 1.000 | 0.360 | 0.529 | 9 | 0 | 16 |
| uncertain | 0.410 | 0.640 | 0.500 | 16 | 23 | 9 |
| hallucinated | 0.656 | 0.840 | 0.737 | 21 | 11 | 4 |

## PRE per-class metrics — full

### Phase 2d.2 (Phase 2c v1 distributions)

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| known | 0.357 | 1.000 | 0.526 | 25 | 45 | 0 |
| unknown | 0.333 | 0.400 | 0.364 | 10 | 20 | 15 |
| uncertain | 0.000 | 0.000 | 0.000 | 0 | 0 | 25 |
| hallucinated | 0.000 | 0.000 | 0.000 | 0 | 0 | 25 |

### Phase 2h (recalibrated distributions)

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| known | 0.714 | 0.200 | 0.312 | 5 | 2 | 20 |
| unknown | 0.289 | 0.440 | 0.349 | 11 | 27 | 14 |
| uncertain | 0.327 | 0.720 | 0.450 | 18 | 37 | 7 |
| hallucinated | 0.000 | 0.000 | 0.000 | 0 | 0 | 25 |

## Top POST error patterns after recalibration

- **unknown → uncertain** (10 cases): unknown_002, unknown_006, unknown_008, unknown_012, unknown_015, unknown_017, unknown_019, unknown_020
- **known → uncertain** (9 cases): known_004, known_005, known_006, known_007, known_015, known_016, known_019, known_021
- **uncertain → hallucinated** (5 cases): uncertain_002, uncertain_014, uncertain_017, uncertain_021, uncertain_024
- **unknown → hallucinated** (4 cases): unknown_003, unknown_009, unknown_013, unknown_018
- **uncertain → known** (4 cases): uncertain_001, uncertain_003, uncertain_009, uncertain_016
- **hallucinated → uncertain** (4 cases): halluc_003, halluc_006, halluc_007, halluc_008
- **known → hallucinated** (2 cases): known_003, known_013
- **unknown → known** (2 cases): unknown_005, unknown_010
