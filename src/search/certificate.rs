//! CausalCertificate — auditable statistical evidence for causal claims.
//!
//! Every causal edge in the system can produce a certificate that bundles:
//! - **Effect size** (Cohen's d): how strong is the relationship?
//! - **Granger signal**: does A's past predict B's future?
//! - **Confidence interval**: bounds on the Granger signal
//! - **p-value**: significance of the causal direction
//! - **NARS truth value**: frequency × confidence evidence
//! - **Berry-Esseen bound**: approximation error for finite samples
//!
//! A certificate is "certified" only when all conditions are met simultaneously.
//! This prevents the system from acting on weak, spurious, or underpowered
//! causal claims.
//!
//! # Science
//!
//! - Cohen (1988): d = (μ₁ - μ₂) / σ_pooled
//! - Granger (1969): temporal causality test
//! - Berry-Esseen (1941): CLT convergence rate O(1/√n)
//! - Itti & Baldi (2009): Bayesian surprise as information-theoretic salience
//!
//! # Integration
//!
//! ```text
//! TemporalEffectSize ──→ CausalCertificate ←── TruthValue (NARS)
//!   (effect_d, granger)       │                    (freq, conf)
//!                             ↓
//!                        certified: bool
//! ```

use std::fmt;

use super::temporal::TemporalEffectSize;

// =============================================================================
// CAUSAL CERTIFICATE
// =============================================================================

/// A statistical certificate for a causal claim.
///
/// Bundles all evidence needed to judge whether a causal relationship
/// is genuine, strong enough to act on, and well-supported by data.
#[derive(Debug, Clone)]
pub struct CausalCertificate {
    /// Effect size (Cohen's d) — how strong is this relationship?
    /// Small: d ≈ 0.2, Medium: d ≈ 0.5, Large: d ≈ 0.8
    pub effect_size: f64,

    /// Granger signal — does A's past predict B's future beyond autocorrelation?
    /// Positive values indicate A → B temporal causality.
    pub granger_signal: f64,

    /// 95% confidence interval on the Granger signal.
    /// For the claim to be certified, CI lower bound must be > 0.
    pub granger_ci: (f64, f64),

    /// p-value for the causal direction (A→B vs B→A).
    /// Lower is stronger evidence for the claimed direction.
    pub direction_p_value: f64,

    /// Berry-Esseen approximation error bound.
    /// For n samples: error ≤ C·ρ / (σ³·√n) where C ≈ 0.4748.
    pub approximation_error: f64,

    /// NARS frequency: proportion of positive evidence (0.0–1.0).
    pub nars_frequency: f32,

    /// NARS confidence: reliability of the frequency (0.0–1.0).
    pub nars_confidence: f32,

    /// Minimum cluster size required for this edge to be reliable.
    pub required_n: usize,

    /// Actual sample size for the source series.
    pub n_source: usize,

    /// Actual sample size for the target series.
    pub n_target: usize,

    /// Is this edge certifiably causal? (computed by `certify()`)
    pub certified: bool,
}

impl CausalCertificate {
    /// Check whether all certification conditions are met.
    ///
    /// A claim is certified if and only if:
    /// 1. Effect size is non-negligible (|d| > 0.2)
    /// 2. Granger signal indicates correct temporal direction (> 0)
    /// 3. Confidence interval excludes zero (lower bound > 0)
    /// 4. Directional p-value is significant (< 0.01)
    /// 5. Sufficient data for both source and target
    pub fn certify(&self) -> bool {
        self.effect_size.abs() > 0.2             // non-negligible effect
            && self.granger_signal > 0.0          // correct temporal direction
            && self.granger_ci.0 > 0.0            // CI excludes zero
            && self.direction_p_value < 0.01      // significant direction
            && self.n_source >= self.required_n   // sufficient source data
            && self.n_target >= self.required_n   // sufficient target data
    }

    /// Create a certificate from temporal effect size measurement and NARS evidence.
    ///
    /// This is the primary constructor. It computes all derived fields
    /// (CI, p-value, Berry-Esseen bound) from the raw measurements.
    pub fn from_temporal(
        effect: &TemporalEffectSize,
        n_source: usize,
        n_target: usize,
        nars_frequency: f32,
        nars_confidence: f32,
    ) -> Self {
        let n_min = n_source.min(n_target);

        // Confidence interval: signal ± 1.96 * std_error
        let z = 1.96; // 95% CI
        let se = effect.std_error as f64;
        let signal = effect.granger_signal as f64;
        let ci_lower = signal - z * se;
        let ci_upper = signal + z * se;

        // Approximate p-value from z-score (one-sided: A→B)
        let z_score = if se > 0.0 {
            signal / se
        } else if signal > 0.0 {
            10.0 // effectively zero p-value
        } else {
            0.0
        };
        let p_value = p_from_z(z_score);

        // Berry-Esseen bound: error ≤ C / √n
        // C ≈ 0.4748 (best known constant)
        let berry_esseen = if n_min > 0 {
            0.4748 / (n_min as f64).sqrt()
        } else {
            1.0
        };

        // Required n based on effect size: n ≥ (2.8 / d)² for 80% power
        // (rule of thumb from Cohen 1988)
        let required_n = if effect.effect_d.abs() > 0.01 {
            let d = effect.effect_d.abs() as f64;
            ((2.8 / d).powi(2)).ceil() as usize
        } else {
            1000 // very small effect, need lots of data
        };

        let mut cert = Self {
            effect_size: effect.effect_d as f64,
            granger_signal: signal,
            granger_ci: (ci_lower, ci_upper),
            direction_p_value: p_value,
            approximation_error: berry_esseen,
            nars_frequency,
            nars_confidence,
            required_n,
            n_source,
            n_target,
            certified: false,
        };

        cert.certified = cert.certify();
        cert
    }

    /// Create a certificate for a claim with no temporal data.
    ///
    /// Uses only NARS evidence. The certificate will NOT be certified
    /// (no Granger signal), but it records the available evidence.
    pub fn from_nars_only(
        nars_frequency: f32,
        nars_confidence: f32,
        weight: f32,
    ) -> Self {
        Self {
            effect_size: weight as f64,
            granger_signal: 0.0,
            granger_ci: (0.0, 0.0),
            direction_p_value: 1.0,
            approximation_error: 1.0,
            nars_frequency,
            nars_confidence,
            required_n: 0,
            n_source: 0,
            n_target: 0,
            certified: false,
        }
    }

    /// Combined evidence score: NARS frequency × confidence × effect strength.
    ///
    /// A quick scalar summary of the certificate's strength (0.0–1.0).
    pub fn evidence_score(&self) -> f64 {
        let nars = self.nars_frequency as f64 * self.nars_confidence as f64;
        let effect = (self.effect_size.abs() / 2.0).min(1.0); // normalize d to 0–1
        nars * effect
    }

    /// How much evidence is "missing"? (0 = fully certified, 1 = no evidence)
    pub fn evidence_gap(&self) -> f64 {
        1.0 - self.evidence_score()
    }

    /// Effect size classification (Cohen's convention).
    pub fn effect_class(&self) -> EffectClass {
        let d = self.effect_size.abs();
        if d < 0.2 {
            EffectClass::Negligible
        } else if d < 0.5 {
            EffectClass::Small
        } else if d < 0.8 {
            EffectClass::Medium
        } else {
            EffectClass::Large
        }
    }
}

/// Cohen's d effect size classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectClass {
    /// |d| < 0.2
    Negligible,
    /// 0.2 ≤ |d| < 0.5
    Small,
    /// 0.5 ≤ |d| < 0.8
    Medium,
    /// |d| ≥ 0.8
    Large,
}

impl fmt::Display for EffectClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EffectClass::Negligible => write!(f, "negligible"),
            EffectClass::Small => write!(f, "small"),
            EffectClass::Medium => write!(f, "medium"),
            EffectClass::Large => write!(f, "large"),
        }
    }
}

impl fmt::Display for CausalCertificate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CausalCert[{}] d={:.3} g={:.3} CI=[{:.3},{:.3}] p={:.4} \
             NARS=<{:.2},{:.2}> n=({},{}) err={:.4}",
            if self.certified { "CERTIFIED" } else { "UNCERTIFIED" },
            self.effect_size,
            self.granger_signal,
            self.granger_ci.0,
            self.granger_ci.1,
            self.direction_p_value,
            self.nars_frequency,
            self.nars_confidence,
            self.n_source,
            self.n_target,
            self.approximation_error,
        )
    }
}

// =============================================================================
// HELPERS
// =============================================================================

/// Approximate one-sided p-value from z-score using the complementary error function.
///
/// Uses the rational approximation from Abramowitz & Stegun (1964), §26.2.17.
fn p_from_z(z: f64) -> f64 {
    if z <= 0.0 {
        return 0.5; // no evidence for directionality
    }

    // Abramowitz & Stegun approximation for Φ(x) complementary
    let t = 1.0 / (1.0 + 0.2316419 * z);
    let d = 0.3989422804014327; // 1/√(2π)
    let p = d * (-z * z / 2.0).exp();

    let poly = t
        * (0.319381530
            + t * (-0.356563782
                + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));

    (p * poly).max(0.0).min(0.5)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_temporal(d: f32, signal: f32, se: f32) -> TemporalEffectSize {
        TemporalEffectSize {
            effect_d: d,
            granger_signal: signal,
            lag: 1,
            std_error: se,
        }
    }

    #[test]
    fn test_certified_strong_causal() {
        // Strong effect, strong signal, low error
        let effect = make_temporal(1.2, 500.0, 50.0);
        let cert = CausalCertificate::from_temporal(&effect, 100, 100, 0.9, 0.85);

        assert!(cert.certified);
        assert_eq!(cert.effect_class(), EffectClass::Large);
        assert!(cert.evidence_score() > 0.0);
        println!("{}", cert);
    }

    #[test]
    fn test_uncertified_weak_effect() {
        // Negligible effect size
        let effect = make_temporal(0.05, 10.0, 5.0);
        let cert = CausalCertificate::from_temporal(&effect, 100, 100, 0.9, 0.85);

        assert!(!cert.certified);
        assert_eq!(cert.effect_class(), EffectClass::Negligible);
    }

    #[test]
    fn test_uncertified_wrong_direction() {
        // Negative Granger signal (B predicts A, not A predicts B)
        let effect = make_temporal(0.8, -100.0, 50.0);
        let cert = CausalCertificate::from_temporal(&effect, 100, 100, 0.9, 0.85);

        assert!(!cert.certified);
    }

    #[test]
    fn test_uncertified_wide_ci() {
        // Signal is positive but SE is so large that CI includes zero
        let effect = make_temporal(0.8, 10.0, 100.0);
        let cert = CausalCertificate::from_temporal(&effect, 100, 100, 0.9, 0.85);

        // CI = 10 ± 196 → lower = -186 (includes zero)
        assert!(!cert.certified);
        assert!(cert.granger_ci.0 < 0.0);
    }

    #[test]
    fn test_uncertified_insufficient_data() {
        // Good signal but not enough samples
        let effect = make_temporal(0.3, 500.0, 50.0);
        // required_n for d=0.3 → (2.8/0.3)² ≈ 87
        let cert = CausalCertificate::from_temporal(&effect, 10, 10, 0.9, 0.85);

        assert!(!cert.certified);
        assert!(cert.required_n > 10);
    }

    #[test]
    fn test_nars_only() {
        let cert = CausalCertificate::from_nars_only(0.9, 0.8, 0.5);

        assert!(!cert.certified); // no Granger data
        assert_eq!(cert.granger_signal, 0.0);
        assert!(cert.evidence_score() > 0.0);
    }

    #[test]
    fn test_evidence_score_range() {
        let effect = make_temporal(1.0, 500.0, 50.0);
        let cert = CausalCertificate::from_temporal(&effect, 100, 100, 1.0, 1.0);

        let score = cert.evidence_score();
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_evidence_gap() {
        let cert = CausalCertificate::from_nars_only(0.0, 0.0, 0.0);
        assert_eq!(cert.evidence_gap(), 1.0); // no evidence

        let effect = make_temporal(2.0, 500.0, 50.0);
        let cert2 = CausalCertificate::from_temporal(&effect, 100, 100, 1.0, 1.0);
        assert!(cert2.evidence_gap() < 0.5); // strong evidence
    }

    #[test]
    fn test_p_from_z() {
        // z=0 → p=0.5 (no evidence)
        assert!((p_from_z(0.0) - 0.5).abs() < 0.001);

        // z=1.96 → p ≈ 0.025
        assert!((p_from_z(1.96) - 0.025).abs() < 0.005);

        // z=3.0 → p < 0.01
        assert!(p_from_z(3.0) < 0.01);

        // Negative z → 0.5
        assert_eq!(p_from_z(-1.0), 0.5);
    }

    #[test]
    fn test_berry_esseen_scales() {
        let effect = make_temporal(0.5, 100.0, 10.0);

        let small = CausalCertificate::from_temporal(&effect, 10, 10, 0.9, 0.9);
        let large = CausalCertificate::from_temporal(&effect, 1000, 1000, 0.9, 0.9);

        // More samples → smaller approximation error
        assert!(large.approximation_error < small.approximation_error);
    }

    #[test]
    fn test_effect_class_boundaries() {
        assert_eq!(
            CausalCertificate::from_nars_only(0.5, 0.5, 0.1).effect_class(),
            EffectClass::Negligible
        );
        assert_eq!(
            CausalCertificate::from_nars_only(0.5, 0.5, 0.3).effect_class(),
            EffectClass::Small
        );
        assert_eq!(
            CausalCertificate::from_nars_only(0.5, 0.5, 0.6).effect_class(),
            EffectClass::Medium
        );
        assert_eq!(
            CausalCertificate::from_nars_only(0.5, 0.5, 0.9).effect_class(),
            EffectClass::Large
        );
    }

    #[test]
    fn test_display() {
        let cert = CausalCertificate::from_nars_only(0.85, 0.72, 1.3);
        let s = format!("{}", cert);
        assert!(s.contains("UNCERTIFIED"));
        assert!(s.contains("NARS"));
    }

    #[test]
    fn test_certify_matches_field() {
        let effect = make_temporal(1.0, 500.0, 50.0);
        let cert = CausalCertificate::from_temporal(&effect, 100, 100, 0.9, 0.85);

        // The stored `certified` field should match re-calling certify()
        assert_eq!(cert.certified, cert.certify());
    }
}
