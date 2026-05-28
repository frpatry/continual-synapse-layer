# Phase 2h — Distribution Drift Report

Compares per-class per-feature statistics between the **real-Qwen** validation dump (`results/agi/phase_2_validation_raw.jsonl`, 100 cases × ~17 features) and the **current** synthetic generators in `src/agi/metacognition/data_generation.py` (1000 samples / class).

*Severity* = `|mean_diff / syn_std|`. Higher means the real mean is further from the synthetic distribution's centre, measured in synthetic-standard-deviation units. A severity above ~1.0 means the synthetic generator is likely to produce examples that look out-of-distribution to a model trained on it.

## Top 20 most-drifted (status, feature) pairs

| status | feature | real mean ± std | syn mean ± std | Δmean | normΔ (severity) | KS |
|---|---|---|---|---:|---:|---:|
| known | `max_similarity` | 0.758 ± 0.081 | 0.900 ± 0.031 | -0.142 | 4.57 | 0.802 |
| known | `query_length_tokens` | 4.200 ± 0.980 | 17.989 ± 3.874 | -13.789 | 3.56 | 1.000 |
| uncertain | `query_length_tokens` | 4.400 ± 1.673 | 18.017 ± 4.071 | -13.617 | 3.35 | 0.992 |
| unknown | `query_length_tokens` | 5.400 ± 1.414 | 18.051 ± 3.915 | -12.651 | 3.23 | 0.997 |
| hallucinated | `query_length_tokens` | 5.880 ± 1.366 | 18.042 ± 3.873 | -12.162 | 3.14 | 0.996 |
| hallucinated | `precision_quality` | 0.200 ± 0.400 | 0.068 ± 0.068 | +0.132 | 1.93 | 0.468 |
| known | `similarity_variance` | 0.000 ± 0.000 | 0.006 ± 0.003 | -0.006 | 1.83 | 1.000 |
| known | `query_specificity` | 1.000 ± 0.000 | 0.712 ± 0.162 | +0.288 | 1.77 | 1.000 |
| uncertain | `query_specificity` | 1.000 ± 0.000 | 0.713 ± 0.163 | +0.287 | 1.77 | 1.000 |
| hallucinated | `query_specificity` | 1.000 ± 0.000 | 0.714 ± 0.163 | +0.286 | 1.76 | 1.000 |
| unknown | `query_specificity` | 1.000 ± 0.000 | 0.723 ± 0.158 | +0.277 | 1.75 | 1.000 |
| known | `precision_quality` | 1.000 ± 0.000 | 0.961 ± 0.023 | +0.039 | 1.68 | 1.000 |
| uncertain | `precision_quality` | 1.000 ± 0.000 | 0.961 ± 0.023 | +0.039 | 1.67 | 1.000 |
| uncertain | `max_similarity` | 0.704 ± 0.076 | 0.760 ± 0.035 | -0.056 | 1.60 | 0.457 |
| uncertain | `has_named_entity` | 0.000 ± 0.000 | 0.708 ± 0.455 | -0.708 | 1.56 | 0.708 |
| unknown | `has_named_entity` | 0.000 ± 0.000 | 0.706 ± 0.456 | -0.706 | 1.55 | 0.706 |
| known | `has_named_entity` | 0.000 ± 0.000 | 0.682 ± 0.466 | -0.682 | 1.46 | 0.682 |
| hallucinated | `has_named_entity` | 0.040 ± 0.196 | 0.700 ± 0.458 | -0.660 | 1.44 | 0.660 |
| known | `mean_similarity` | 0.758 ± 0.081 | 0.822 ± 0.045 | -0.064 | 1.44 | 0.435 |
| known | `n_facts_retrieved` | 1.000 ± 0.000 | 2.481 ± 1.098 | -1.481 | 1.35 | 0.750 |

## Per-class feature snapshots

### `hallucinated` — real (25 samples) vs synthetic

| feature | real mean | real std | syn mean | syn std | normΔ |
|---|---:|---:|---:|---:|---:|
| `query_length_tokens` | 5.880 | 1.366 | 18.042 | 3.873 | -3.14 |
| `precision_quality` | 0.200 | 0.400 | 0.068 | 0.068 | +1.93 |
| `query_specificity` | 1.000 | 0.000 | 0.714 | 0.163 | +1.76 |
| `has_named_entity` | 0.040 | 0.196 | 0.700 | 0.458 | -1.44 |
| `similarity_variance` | 0.000 | 0.000 | 0.037 | 0.033 | -1.13 |
| `mean_similarity` | 0.154 | 0.308 | 0.075 | 0.074 | +1.07 |
| `n_facts_retrieved` | 0.200 | 0.400 | 1.000 | 0.815 | -0.98 |
| `max_recency_days` | 0.000 | 0.000 | 63.039 | 86.539 | -0.73 |
| `attention_to_facts_mean` | 0.003 | 0.006 | 0.004 | 0.002 | -0.68 |
| `max_similarity` | 0.154 | 0.308 | 0.096 | 0.093 | +0.63 |
| `mean_access_count` | 0.200 | 0.400 | 0.643 | 0.937 | -0.47 |
| `alignment_novel_token_ratio` | 0.200 | 0.400 | 0.241 | 0.332 | -0.12 |
| `alignment_mean_cosine` | 0.074 | 0.148 | 0.067 | 0.059 | +0.11 |
| `max_token_entropy` | 4.413 | 0.768 | 4.201 | 3.043 | +0.07 |
| `alignment_max_cosine` | 0.074 | 0.148 | 0.077 | 0.070 | -0.06 |
| `mean_token_entropy` | 1.445 | 0.245 | 1.397 | 0.954 | +0.05 |
| `response_length_tokens` | 85.360 | 23.279 | 85.233 | 8.575 | +0.01 |

### `known` — real (25 samples) vs synthetic

| feature | real mean | real std | syn mean | syn std | normΔ |
|---|---:|---:|---:|---:|---:|
| `max_similarity` | 0.758 | 0.081 | 0.900 | 0.031 | -4.57 |
| `query_length_tokens` | 4.200 | 0.980 | 17.989 | 3.874 | -3.56 |
| `similarity_variance` | 0.000 | 0.000 | 0.006 | 0.003 | -1.83 |
| `query_specificity` | 1.000 | 0.000 | 0.712 | 0.162 | +1.77 |
| `precision_quality` | 1.000 | 0.000 | 0.961 | 0.023 | +1.68 |
| `has_named_entity` | 0.000 | 0.000 | 0.682 | 0.466 | -1.46 |
| `mean_similarity` | 0.758 | 0.081 | 0.822 | 0.045 | -1.44 |
| `n_facts_retrieved` | 1.000 | 0.000 | 2.481 | 1.098 | -1.35 |
| `max_recency_days` | 0.000 | 0.000 | 1.962 | 1.940 | -1.01 |
| `alignment_novel_token_ratio` | 0.814 | 0.147 | 0.842 | 0.028 | -1.00 |
| `mean_access_count` | 1.000 | 0.000 | 1.987 | 1.011 | -0.98 |
| `alignment_mean_cosine` | 0.544 | 0.088 | 0.511 | 0.038 | +0.86 |
| `response_length_tokens` | 25.880 | 25.354 | 30.040 | 4.911 | -0.85 |
| `alignment_max_cosine` | 0.544 | 0.088 | 0.525 | 0.038 | +0.51 |
| `attention_to_facts_mean` | 0.055 | 0.025 | 0.060 | 0.019 | -0.29 |
| `mean_token_entropy` | 0.818 | 0.307 | 0.874 | 0.721 | -0.08 |
| `max_token_entropy` | 2.706 | 0.885 | 2.641 | 2.263 | +0.03 |

### `uncertain` — real (25 samples) vs synthetic

| feature | real mean | real std | syn mean | syn std | normΔ |
|---|---:|---:|---:|---:|---:|
| `query_length_tokens` | 4.400 | 1.673 | 18.017 | 4.071 | -3.35 |
| `query_specificity` | 1.000 | 0.000 | 0.713 | 0.163 | +1.77 |
| `precision_quality` | 1.000 | 0.000 | 0.961 | 0.023 | +1.67 |
| `max_similarity` | 0.704 | 0.076 | 0.760 | 0.035 | -1.60 |
| `has_named_entity` | 0.000 | 0.000 | 0.708 | 0.455 | -1.56 |
| `alignment_novel_token_ratio` | 0.925 | 0.073 | 0.943 | 0.016 | -1.07 |
| `mean_similarity` | 0.690 | 0.071 | 0.729 | 0.038 | -1.04 |
| `max_recency_days` | 0.000 | 0.000 | 1.934 | 1.889 | -1.02 |
| `mean_access_count` | 1.000 | 0.000 | 2.006 | 1.001 | -1.01 |
| `n_facts_retrieved` | 1.600 | 0.490 | 2.013 | 0.809 | -0.51 |
| `attention_to_facts_mean` | 0.064 | 0.025 | 0.058 | 0.018 | +0.38 |
| `alignment_mean_cosine` | 0.492 | 0.097 | 0.479 | 0.046 | +0.29 |
| `alignment_max_cosine` | 0.511 | 0.102 | 0.500 | 0.045 | +0.25 |
| `max_token_entropy` | 3.449 | 0.531 | 4.177 | 3.090 | -0.24 |
| `response_length_tokens` | 42.040 | 26.186 | 43.163 | 5.847 | -0.19 |
| `mean_token_entropy` | 1.253 | 0.336 | 1.402 | 1.011 | -0.15 |
| `similarity_variance` | 0.001 | 0.003 | 0.001 | 0.001 | -0.10 |

### `unknown` — real (25 samples) vs synthetic

| feature | real mean | real std | syn mean | syn std | normΔ |
|---|---:|---:|---:|---:|---:|
| `query_length_tokens` | 5.400 | 1.414 | 18.051 | 3.915 | -3.23 |
| `query_specificity` | 1.000 | 0.000 | 0.723 | 0.158 | +1.75 |
| `has_named_entity` | 0.000 | 0.000 | 0.706 | 0.456 | -1.55 |
| `alignment_mean_cosine` | 0.278 | 0.240 | 0.117 | 0.121 | +1.32 |
| `alignment_max_cosine` | 0.278 | 0.240 | 0.123 | 0.127 | +1.21 |
| `attention_to_facts_mean` | 0.016 | 0.017 | 0.008 | 0.008 | +1.00 |
| `mean_similarity` | 0.460 | 0.379 | 0.243 | 0.242 | +0.90 |
| `alignment_novel_token_ratio` | 0.585 | 0.480 | 0.316 | 0.314 | +0.86 |
| `max_similarity` | 0.460 | 0.379 | 0.256 | 0.255 | +0.80 |
| `similarity_variance` | 0.000 | 0.000 | 0.001 | 0.001 | -0.78 |
| `max_recency_days` | 0.000 | 0.000 | 0.958 | 1.577 | -0.61 |
| `response_length_tokens` | 47.520 | 28.118 | 50.036 | 6.354 | -0.40 |
| `mean_access_count` | 0.600 | 0.490 | 1.018 | 1.222 | -0.34 |
| `precision_quality` | 0.600 | 0.490 | 0.488 | 0.481 | +0.23 |
| `n_facts_retrieved` | 0.600 | 0.490 | 0.750 | 0.819 | -0.18 |
| `max_token_entropy` | 3.268 | 0.720 | 3.756 | 3.701 | -0.13 |
| `mean_token_entropy` | 1.131 | 0.239 | 1.254 | 1.190 | -0.10 |
