# Scientific Validation API (sci/v1)

Publication-ready similarity analysis with full statistical packages.

## Overview

The `sci/v1` API provides scientifically rigorous similarity responses:

```
sci/v1/
├── similarity      # Full statistical analysis
├── validate        # Reciprocal truth checking
├── crossval        # Multi-method cross-validation
├── evaluate        # 7-point self-evaluation
└── report          # Publication-ready output
```

## Response Structure

### Full Scientific Response

```json
{
  "query": "0a1b2c...",
  "match": {
    "address": 32768,
    "fingerprint": "8f9e0d..."
  },

  "statistics": {
    "distance": {
      "hamming": 1250,
      "total_bits": 9984
    },
    "similarity": {
      "hamming": 0.875,
      "cosine_estimate": 0.92,
      "jina_score": 0.89
    },
    "population": {
      "n": 25600,
      "mean": 4992,
      "median": 4950,
      "std_dev": 1200,
      "variance": 1440000,
      "skewness": 0.12,
      "kurtosis": 2.95,
      "quartiles": [3800, 4950, 6100],
      "iqr": 2300
    },
    "confidence": {
      "ci_90": [0.867, 0.883],
      "ci_95": [0.861, 0.889],
      "ci_99": [0.854, 0.896],
      "standard_error": 0.007
    },
    "effect": {
      "cohens_d": 3.12,
      "hedges_g": 3.10,
      "interpretation": "Large",
      "r_squared": 0.71
    },
    "position": {
      "z_score": -3.12,
      "percentile": 99.9,
      "p_value": 0.0018,
      "significant_05": true,
      "significant_01": true,
      "significant_001": false
    }
  },

  "validation": {
    "forward": {
      "rank": 1,
      "similarity": 0.875,
      "distance": 1250
    },
    "reverse": {
      "rank": 1,
      "similarity": 0.873,
      "distance": 1268
    },
    "agreement": {
      "is_mutual_nearest": true,
      "is_mutual_top_k": true,
      "k_for_mutual": 10,
      "rank_difference": 0,
      "similarity_difference": 0.002
    },
    "confidence": {
      "reciprocal_confidence": 0.98,
      "symmetry_score": 0.997,
      "asymmetry_flag": false,
      "outlier_probability": 0.02
    }
  },

  "crossval": {
    "methods": {
      "hamming": {"similarity": 0.875, "rank": 1, "weight": 1.0},
      "cosine": {"similarity": 0.92, "rank": 1, "weight": 0.9},
      "jina": {"similarity": 0.89, "rank": 1, "weight": 0.85}
    },
    "agreement": {
      "method_agreement": 0.95,
      "rank_correlation": 0.98,
      "concordance": 0.94
    },
    "consensus": {
      "similarity": 0.893,
      "confidence": 0.96
    }
  },

  "evaluation": {
    "checks": [
      {"id": 1, "name": "Statistical Significance", "passed": true, "score": 0.998},
      {"id": 2, "name": "Effect Size", "passed": true, "score": 1.0},
      {"id": 3, "name": "Reciprocal Validation", "passed": true, "score": 0.98},
      {"id": 4, "name": "Symmetry Check", "passed": true, "score": 0.997},
      {"id": 5, "name": "Method Agreement", "passed": true, "score": 0.95},
      {"id": 6, "name": "Confidence Interval", "passed": true, "score": 0.86},
      {"id": 7, "name": "Outlier Detection", "passed": true, "score": 0.98}
    ],
    "passed": 7,
    "score": 0.966,
    "verdict": "Accept",
    "recommendation": "UseAsIs"
  },

  "metadata": {
    "timestamp": 1706889600,
    "computation_time_ns": 125000,
    "api_version": "sci/v1"
  }
}
```

## Endpoints

### sci/v1/similarity

Full statistical similarity analysis.

**Request:**
```json
{
  "query": "hex_fingerprint",
  "address": 32768,
  "include_population": true
}
```

### sci/v1/validate

Reciprocal validation only.

**Request:**
```json
{
  "query": "hex_fingerprint",
  "match": "hex_fingerprint",
  "k": 10
}
```

### sci/v1/crossval

Cross-validation across methods.

**Request:**
```json
{
  "query": "hex_fingerprint",
  "match": "hex_fingerprint",
  "methods": ["hamming", "cosine", "jina"]
}
```

### sci/v1/evaluate

7-point self-evaluation.

**Request:**
```json
{
  "query": "hex_fingerprint",
  "match_address": 32768
}
```

### sci/v1/report

Generate publication-ready report.

**Request:**
```json
{
  "query": "hex_fingerprint",
  "k": 10,
  "format": "markdown|latex|json"
}
```

**Response (Markdown):**
```markdown
## Similarity Analysis Report

**Query**: `0a1b2c...` (truncated)
**Date**: 2026-02-03T12:00:00Z

### Top Match

| Metric | Value | 95% CI |
|--------|-------|--------|
| Similarity | 0.875 | [0.861, 0.889] |
| Effect Size (d) | 3.12 | Large |
| p-value | 0.0018 | ** |

### Validation

- Reciprocal: ✓ Mutual nearest neighbor
- Cross-method: ✓ 95% agreement
- 7-point check: 7/7 passed

### Verdict: **Accept**
```

## Statistical Methods

### Confidence Intervals

Using standard normal approximation:

```
CI = x̄ ± z * (σ / √n)

z_90 = 1.645
z_95 = 1.960
z_99 = 2.576
```

### Effect Size

Cohen's d interpretation:

| |d| | Interpretation |
|-----|----------------|
| < 0.2 | Negligible |
| 0.2 - 0.5 | Small |
| 0.5 - 0.8 | Medium |
| ≥ 0.8 | Large |

### Cosine Approximation

For LSH-derived binary fingerprints:

```
cos(θ) ≈ 1 - 2 * (hamming_distance / total_bits)
```

### 7-Point Self-Evaluation

| Check | Criterion | Weight |
|-------|-----------|--------|
| 1 | p < 0.05 | Pass/Fail |
| 2 | \|d\| ≥ 0.2 | Pass/Fail |
| 3 | Mutual top-k | Pass/Fail |
| 4 | Symmetry > 0.9 | Pass/Fail |
| 5 | Method agreement > 0.7 | Pass/Fail |
| 6 | CI width < 0.2 | Pass/Fail |
| 7 | Outlier prob < 0.1 | Pass/Fail |

## Usage Examples

### Python

```python
import requests

# Full analysis
response = requests.post("http://localhost:50051/sci/v1/similarity", json={
    "query": query_fp_hex,
    "address": 32768,
    "include_population": True
})

result = response.json()
print(f"Similarity: {result['statistics']['similarity']['hamming']}")
print(f"95% CI: {result['statistics']['confidence']['ci_95']}")
print(f"p-value: {result['statistics']['position']['p_value']}")
print(f"Verdict: {result['evaluation']['verdict']}")
```

### Rust

```rust
use ladybug::search::scientific::*;

let stats = compute_statistical_similarity(
    query_fp,
    match_fp,
    &population_stats,
);

println!("Cohen's d: {}", stats.effect.cohens_d);
println!("Significant: {}", stats.position.significant_05);
```

## Flight Actions

| Action | Description |
|--------|-------------|
| `sci.similarity` | Full statistical analysis |
| `sci.validate` | Reciprocal validation |
| `sci.crossval` | Cross-method validation |
| `sci.evaluate` | 7-point self-evaluation |
| `sci.report` | Publication-ready report |

## Why This Matters

1. **Reproducibility**: Full stats enable proper replication
2. **Meta-analysis**: Effect sizes + CIs for combining studies
3. **Peer review**: Publication-ready output
4. **Self-validation**: Built-in truth checking
5. **Confidence**: Know when to trust results
