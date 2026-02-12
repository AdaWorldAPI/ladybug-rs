//! Scientific Validation Module
//!
//! Provides publication-ready similarity analysis with:
//! - Full statistical package (mean, SD, CI, effect sizes)
//! - Reciprocal validation (bidirectional truth checking)
//! - Cross-validation (Hamming, Jina, Cosine agreement)
//! - Self-evaluation cycle (7-point validation)

// =============================================================================
// CONSTANTS
// =============================================================================

/// Total bits in fingerprint (256 * 64)
pub const TOTAL_BITS: u32 = 16384;

/// Default confidence level for CI
pub const DEFAULT_CONFIDENCE: f64 = 0.95;

/// Z-scores for confidence intervals
pub const Z_95: f64 = 1.96;
pub const Z_99: f64 = 2.576;

// =============================================================================
// STATISTICAL SIMILARITY
// =============================================================================

/// Complete statistical analysis of a similarity measurement
#[derive(Debug, Clone)]
pub struct StatisticalSimilarity {
    // === Raw Measurement ===
    pub hamming_distance: u32,
    pub total_bits: u32,

    // === Point Estimates ===
    pub similarity: f64,              // 1.0 - (dist/bits)
    pub cosine_estimate: f64,         // Angular interpretation
    pub jina_score: Option<f64>,      // Learned metric (if available)

    // === Descriptive Statistics ===
    pub population: PopulationStats,

    // === Confidence Intervals ===
    pub confidence: ConfidenceIntervals,

    // === Effect Size ===
    pub effect: EffectSize,

    // === Distribution Position ===
    pub position: DistributionPosition,
}

/// Population statistics from the search corpus
#[derive(Debug, Clone, Default)]
pub struct PopulationStats {
    pub n: u32,                       // Sample size
    pub mean: f64,                    // μ
    pub median: f64,                  // Robust center
    pub std_dev: f64,                 // σ
    pub variance: f64,                // σ²
    pub skewness: f64,                // Asymmetry
    pub kurtosis: f64,                // Tail weight
    pub min: f64,
    pub max: f64,
    pub quartiles: (f64, f64, f64),   // Q1, Q2, Q3
    pub iqr: f64,                     // Q3 - Q1
}

/// Confidence intervals at multiple levels
#[derive(Debug, Clone, Default)]
pub struct ConfidenceIntervals {
    pub ci_90: (f64, f64),
    pub ci_95: (f64, f64),
    pub ci_99: (f64, f64),
    pub standard_error: f64,
}

/// Effect size measurements
#[derive(Debug, Clone, Default)]
pub struct EffectSize {
    pub cohens_d: f64,                // Standardized effect
    pub hedges_g: f64,                // Bias-corrected
    pub glass_delta: f64,             // Using control SD
    pub r_squared: f64,               // Variance explained
    pub interpretation: EffectInterpretation,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub enum EffectInterpretation {
    #[default]
    Negligible,  // |d| < 0.2
    Small,       // 0.2 <= |d| < 0.5
    Medium,      // 0.5 <= |d| < 0.8
    Large,       // |d| >= 0.8
}

/// Position within the distribution
#[derive(Debug, Clone, Default)]
pub struct DistributionPosition {
    pub z_score: f64,
    pub percentile: f64,
    pub p_value: f64,
    pub p_one_tailed: f64,
    pub significant_05: bool,
    pub significant_01: bool,
    pub significant_001: bool,
}

// =============================================================================
// RECIPROCAL VALIDATION
// =============================================================================

/// Bidirectional truth validation
#[derive(Debug, Clone)]
pub struct ReciprocalValidation {
    // === Forward Search (Query → Match) ===
    pub forward_rank: u32,
    pub forward_similarity: f64,
    pub forward_distance: u32,

    // === Reverse Search (Match → Query) ===
    pub reverse_rank: u32,
    pub reverse_similarity: f64,
    pub reverse_distance: u32,

    // === Agreement Metrics ===
    pub is_mutual_nearest: bool,      // Both rank #1
    pub is_mutual_top_k: bool,        // Both in top-k
    pub k_for_mutual: u32,            // k value used
    pub rank_difference: i32,         // |fwd - rev|
    pub similarity_difference: f64,   // |fwd_sim - rev_sim|

    // === Confidence ===
    pub reciprocal_confidence: f64,   // Combined score
    pub symmetry_score: f64,          // How symmetric
    pub asymmetry_flag: bool,         // Suspicious asymmetry
    pub outlier_probability: f64,     // Noise likelihood
}

// =============================================================================
// CROSS-VALIDATION
// =============================================================================

/// Multi-method cross-validation
#[derive(Debug, Clone)]
pub struct CrossValidation {
    // === Method Scores ===
    pub hamming: MethodScore,
    pub cosine: MethodScore,
    pub jina: Option<MethodScore>,
    pub euclidean: Option<MethodScore>,

    // === Agreement ===
    pub method_agreement: f64,        // How much methods agree
    pub rank_correlation: f64,        // Spearman's ρ across methods
    pub concordance: f64,             // Kendall's W

    // === Consensus ===
    pub consensus_similarity: f64,    // Weighted average
    pub consensus_confidence: f64,    // Agreement-based confidence
    pub outlier_methods: Vec<String>, // Methods that disagree
}

#[derive(Debug, Clone, Default)]
pub struct MethodScore {
    pub name: String,
    pub raw_distance: f64,
    pub similarity: f64,
    pub rank: u32,
    pub weight: f64,                  // Method reliability weight
}

// =============================================================================
// SELF-EVALUATION CYCLE (7-POINT)
// =============================================================================

/// 7-point self-evaluation for each similarity result
#[derive(Debug, Clone)]
pub struct SelfEvaluation {
    pub checks: [EvaluationCheck; 7],
    pub passed: u8,                   // Count of passed checks
    pub score: f64,                   // 0.0 - 1.0
    pub verdict: Verdict,
    pub recommendation: Recommendation,
}

#[derive(Debug, Clone)]
pub struct EvaluationCheck {
    pub id: u8,
    pub name: &'static str,
    pub description: &'static str,
    pub passed: bool,
    pub score: f64,
    pub details: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    Accept,           // All/most checks pass
    AcceptWithCaution,// Some concerns
    Review,           // Manual review needed
    Reject,           // Failed critical checks
}

#[derive(Debug, Clone)]
pub enum Recommendation {
    UseAsIs,
    VerifyManually,
    IncreaseK,
    TryDifferentMethod,
    Discard,
}

/// The 7 evaluation checks
pub fn create_evaluation_checks(
    stats: &StatisticalSimilarity,
    reciprocal: &ReciprocalValidation,
    crossval: &CrossValidation,
) -> [EvaluationCheck; 7] {
    [
        // 1. Statistical Significance
        EvaluationCheck {
            id: 1,
            name: "Statistical Significance",
            description: "Is the similarity statistically significant?",
            passed: stats.position.significant_05,
            score: 1.0 - stats.position.p_value.min(1.0),
            details: format!("p={:.4}", stats.position.p_value),
        },

        // 2. Effect Size
        EvaluationCheck {
            id: 2,
            name: "Effect Size",
            description: "Is the effect size meaningful?",
            passed: stats.effect.interpretation != EffectInterpretation::Negligible,
            score: stats.effect.cohens_d.abs().min(2.0) / 2.0,
            details: format!("d={:.2} ({:?})", stats.effect.cohens_d, stats.effect.interpretation),
        },

        // 3. Reciprocal Agreement
        EvaluationCheck {
            id: 3,
            name: "Reciprocal Validation",
            description: "Does reverse search confirm the match?",
            passed: reciprocal.is_mutual_top_k,
            score: reciprocal.reciprocal_confidence,
            details: format!("fwd_rank={}, rev_rank={}", reciprocal.forward_rank, reciprocal.reverse_rank),
        },

        // 4. Symmetry
        EvaluationCheck {
            id: 4,
            name: "Symmetry Check",
            description: "Is the similarity symmetric?",
            passed: !reciprocal.asymmetry_flag,
            score: reciprocal.symmetry_score,
            details: format!("diff={:.3}", reciprocal.similarity_difference),
        },

        // 5. Cross-Method Agreement
        EvaluationCheck {
            id: 5,
            name: "Method Agreement",
            description: "Do different methods agree?",
            passed: crossval.method_agreement > 0.7,
            score: crossval.method_agreement,
            details: format!("agreement={:.2}", crossval.method_agreement),
        },

        // 6. Confidence Interval
        EvaluationCheck {
            id: 6,
            name: "Confidence Interval",
            description: "Is the CI reasonably narrow?",
            passed: (stats.confidence.ci_95.1 - stats.confidence.ci_95.0) < 0.2,
            score: 1.0 - (stats.confidence.ci_95.1 - stats.confidence.ci_95.0).min(1.0),
            details: format!("95% CI: [{:.3}, {:.3}]", stats.confidence.ci_95.0, stats.confidence.ci_95.1),
        },

        // 7. Outlier Check
        EvaluationCheck {
            id: 7,
            name: "Outlier Detection",
            description: "Is this not an outlier/noise?",
            passed: reciprocal.outlier_probability < 0.1,
            score: 1.0 - reciprocal.outlier_probability,
            details: format!("outlier_prob={:.2}", reciprocal.outlier_probability),
        },
    ]
}

impl SelfEvaluation {
    pub fn from_checks(checks: [EvaluationCheck; 7]) -> Self {
        let passed = checks.iter().filter(|c| c.passed).count() as u8;
        let score = checks.iter().map(|c| c.score).sum::<f64>() / 7.0;

        let verdict = match passed {
            7 => Verdict::Accept,
            5..=6 => Verdict::AcceptWithCaution,
            3..=4 => Verdict::Review,
            _ => Verdict::Reject,
        };

        let recommendation = match &verdict {
            Verdict::Accept => Recommendation::UseAsIs,
            Verdict::AcceptWithCaution => Recommendation::VerifyManually,
            Verdict::Review => Recommendation::IncreaseK,
            Verdict::Reject => Recommendation::Discard,
        };

        Self {
            checks,
            passed,
            score,
            verdict,
            recommendation,
        }
    }
}

// =============================================================================
// COMPLETE SCIENTIFIC RESPONSE
// =============================================================================

/// Full scientific validation response
#[derive(Debug, Clone)]
pub struct ScientificSimilarityResponse {
    // === Query Info ===
    pub query_fingerprint: String,    // Hex
    pub match_address: u16,
    pub match_fingerprint: String,    // Hex

    // === Core Analysis ===
    pub statistics: StatisticalSimilarity,
    pub validation: ReciprocalValidation,
    pub crossval: CrossValidation,

    // === Self-Evaluation ===
    pub evaluation: SelfEvaluation,

    // === Metadata ===
    pub timestamp: u64,
    pub computation_time_ns: u64,
    pub api_version: &'static str,
}

// =============================================================================
// COMPUTATION FUNCTIONS
// =============================================================================

/// Convert Hamming distance to cosine estimate
/// For LSH-derived fingerprints: cos(θ) ≈ 1 - 2*(d/n)
pub fn hamming_to_cosine(distance: u32, total_bits: u32) -> f64 {
    let ratio = distance as f64 / total_bits as f64;
    1.0 - 2.0 * ratio
}

/// Compute z-score
pub fn z_score(value: f64, mean: f64, std_dev: f64) -> f64 {
    if std_dev == 0.0 {
        return 0.0;
    }
    (value - mean) / std_dev
}

/// Compute p-value from z-score (two-tailed)
pub fn p_value_from_z(z: f64) -> f64 {
    // Approximation using error function
    let t = 1.0 / (1.0 + 0.2316419 * z.abs());
    let d = 0.3989423 * (-z * z / 2.0).exp();
    let p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    2.0 * p  // Two-tailed
}

/// Compute Cohen's d effect size
pub fn cohens_d(value: f64, mean: f64, std_dev: f64) -> f64 {
    if std_dev == 0.0 {
        return 0.0;
    }
    (value - mean) / std_dev
}

/// Interpret effect size
pub fn interpret_effect(d: f64) -> EffectInterpretation {
    let abs_d = d.abs();
    if abs_d < 0.2 {
        EffectInterpretation::Negligible
    } else if abs_d < 0.5 {
        EffectInterpretation::Small
    } else if abs_d < 0.8 {
        EffectInterpretation::Medium
    } else {
        EffectInterpretation::Large
    }
}

/// Compute confidence interval
pub fn confidence_interval(mean: f64, std_dev: f64, n: u32, z: f64) -> (f64, f64) {
    let se = std_dev / (n as f64).sqrt();
    let margin = z * se;
    (mean - margin, mean + margin)
}

/// Compute percentile from z-score
pub fn percentile_from_z(z: f64) -> f64 {
    // CDF approximation
    let t = 1.0 / (1.0 + 0.2316419 * z.abs());
    let d = 0.3989423 * (-z * z / 2.0).exp();
    let p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));

    if z >= 0.0 {
        (1.0 - p) * 100.0
    } else {
        p * 100.0
    }
}

// =============================================================================
// ONLINE STATISTICS (Welford's Algorithm)
// =============================================================================

/// Online statistics calculator using Welford's algorithm
#[derive(Debug, Clone, Default)]
pub struct OnlineStats {
    n: u64,
    mean: f64,
    m2: f64,      // Sum of squares of differences from mean
    min: f64,
    max: f64,

    // For median/quartiles (reservoir sampling)
    reservoir: Vec<f64>,
    reservoir_size: usize,
}

impl OnlineStats {
    pub fn new(reservoir_size: usize) -> Self {
        Self {
            reservoir_size,
            min: f64::MAX,
            max: f64::MIN,
            ..Default::default()
        }
    }

    pub fn update(&mut self, value: f64) {
        self.n += 1;
        let delta = value - self.mean;
        self.mean += delta / self.n as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;

        self.min = self.min.min(value);
        self.max = self.max.max(value);

        // Reservoir sampling for percentiles
        if self.reservoir.len() < self.reservoir_size {
            self.reservoir.push(value);
        } else {
            let idx = (rand_simple() * self.n as f64) as usize;
            if idx < self.reservoir_size {
                self.reservoir[idx] = value;
            }
        }
    }

    pub fn mean(&self) -> f64 {
        self.mean
    }

    pub fn variance(&self) -> f64 {
        if self.n < 2 {
            return 0.0;
        }
        self.m2 / (self.n - 1) as f64
    }

    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    pub fn to_population_stats(&self) -> PopulationStats {
        let mut sorted = self.reservoir.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let quartiles = if sorted.len() >= 4 {
            let q1_idx = sorted.len() / 4;
            let q2_idx = sorted.len() / 2;
            let q3_idx = 3 * sorted.len() / 4;
            (sorted[q1_idx], sorted[q2_idx], sorted[q3_idx])
        } else {
            (self.mean, self.mean, self.mean)
        };

        PopulationStats {
            n: self.n as u32,
            mean: self.mean,
            median: quartiles.1,
            std_dev: self.std_dev(),
            variance: self.variance(),
            skewness: 0.0,  // TODO: compute
            kurtosis: 0.0,  // TODO: compute
            min: self.min,
            max: self.max,
            quartiles,
            iqr: quartiles.2 - quartiles.0,
        }
    }
}

// Simple deterministic "random" for reservoir sampling
fn rand_simple() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    (nanos as f64 % 1000.0) / 1000.0
}

// =============================================================================
// API VERSION
// =============================================================================

pub const API_VERSION: &str = "sci/v1";
