//! Shadow Parallel Processor — run alternative reasoning path and compare.
//!
//! The shadow processor runs a second "what-if" computation in parallel,
//! using a different thinking style or parameter set, then compares results
//! for robustness checking.
//!
//! # Science
//! - Kahneman (2011): System 1 vs System 2 — two parallel processing modes
//! - Silver (2012): Ensemble methods outperform individual predictors
//! - Einhorn & Hogarth (1978): Dialectical inquiry improves decision quality
//! - Berry-Esseen (1941/42): Noise floor at d=16384 distinguishes real from random

use crate::search::hdr_cascade::{hamming_distance, WORDS};

const TOTAL_BITS: f32 = 16384.0;

/// Result of a shadow comparison between primary and shadow paths.
#[derive(Debug, Clone)]
pub struct ShadowComparison {
    /// Primary result fingerprint
    pub primary: [u64; WORDS],
    /// Shadow result fingerprint
    pub shadow: [u64; WORDS],
    /// Normalized Hamming distance between primary and shadow
    pub divergence: f32,
    /// Whether the two paths agree (divergence < threshold)
    pub agreement: bool,
    /// Confidence boost from agreement (0.0 if disagreement)
    pub confidence_boost: f32,
    /// Which path should be preferred (true = primary, false = shadow)
    pub prefer_primary: bool,
}

/// Configuration for shadow processing.
#[derive(Debug, Clone)]
pub struct ShadowConfig {
    /// Agreement threshold: paths agree if divergence < this
    pub agreement_threshold: f32,
    /// Confidence boost when paths agree
    pub boost_on_agreement: f32,
    /// Berry-Esseen noise floor — below this, differences are noise
    pub noise_floor: f32,
}

impl Default for ShadowConfig {
    fn default() -> Self {
        Self {
            agreement_threshold: 0.15,
            boost_on_agreement: 0.1,
            noise_floor: 0.004, // Berry-Esseen at d=16384
        }
    }
}

/// Shadow processor that maintains a parallel reasoning path.
#[derive(Debug, Clone)]
pub struct ShadowProcessor {
    config: ShadowConfig,
    /// Running count of agreements
    agreement_count: usize,
    /// Running count of total comparisons
    total_count: usize,
}

impl ShadowProcessor {
    pub fn new(config: ShadowConfig) -> Self {
        Self {
            config,
            agreement_count: 0,
            total_count: 0,
        }
    }

    /// Compare primary and shadow results.
    ///
    /// The `quality_primary` and `quality_shadow` parameters are optional
    /// quality scores (e.g., from CRP or truth value) used to break ties
    /// when paths disagree.
    pub fn compare(
        &mut self,
        primary: &[u64; WORDS],
        shadow: &[u64; WORDS],
        quality_primary: f32,
        quality_shadow: f32,
    ) -> ShadowComparison {
        let distance = hamming_distance(primary, shadow);
        let divergence = distance as f32 / TOTAL_BITS;

        let agreement = divergence < self.config.agreement_threshold;
        let below_noise = divergence < self.config.noise_floor;

        self.total_count += 1;
        if agreement {
            self.agreement_count += 1;
        }

        let confidence_boost = if below_noise {
            // Below noise floor — paths are essentially identical
            self.config.boost_on_agreement
        } else if agreement {
            // Paths agree but with detectable differences
            self.config.boost_on_agreement * (1.0 - divergence / self.config.agreement_threshold)
        } else {
            0.0
        };

        // Prefer whichever has higher quality; default to primary
        let prefer_primary = quality_primary >= quality_shadow;

        ShadowComparison {
            primary: *primary,
            shadow: *shadow,
            divergence,
            agreement,
            confidence_boost,
            prefer_primary,
        }
    }

    /// Get the agreement rate over all comparisons.
    pub fn agreement_rate(&self) -> f32 {
        if self.total_count == 0 {
            return 1.0;
        }
        self.agreement_count as f32 / self.total_count as f32
    }

    /// Total number of comparisons made.
    pub fn total_comparisons(&self) -> usize {
        self.total_count
    }

    /// Reset statistics.
    pub fn reset(&mut self) {
        self.agreement_count = 0;
        self.total_count = 0;
    }
}

/// Merge two fingerprints by majority vote when they agree,
/// preferring the `preferred` when they disagree.
pub fn merge_shadow(
    primary: &[u64; WORDS],
    shadow: &[u64; WORDS],
    prefer_primary: bool,
) -> [u64; WORDS] {
    let mut result = [0u64; WORDS];
    let preferred = if prefer_primary { primary } else { shadow };
    let other = if prefer_primary { shadow } else { primary };

    for i in 0..WORDS {
        // Keep bits where both agree
        let agree = !(preferred[i] ^ other[i]);
        // Where they disagree, use preferred
        result[i] = (preferred[i] & agree) | (preferred[i] & !agree);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_fp(seed: u64) -> [u64; WORDS] {
        let mut fp = [0u64; WORDS];
        let mut state = seed;
        for w in fp.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *w = state;
        }
        fp
    }

    #[test]
    fn test_shadow_identical() {
        let mut proc = ShadowProcessor::new(ShadowConfig::default());
        let fp = make_fp(42);
        let result = proc.compare(&fp, &fp, 1.0, 1.0);
        assert!(result.agreement);
        assert!(result.divergence < 0.001);
        assert!(result.confidence_boost > 0.0);
    }

    #[test]
    fn test_shadow_similar() {
        let mut proc = ShadowProcessor::new(ShadowConfig::default());
        let primary = make_fp(42);
        let mut shadow = primary;
        // Flip a few words — small divergence
        shadow[0] ^= 0xFFFF;
        shadow[1] ^= 0xFFFF;

        let result = proc.compare(&primary, &shadow, 0.8, 0.7);
        assert!(result.agreement); // Small change, should still agree
        assert!(result.prefer_primary); // Primary has higher quality
    }

    #[test]
    fn test_shadow_divergent() {
        let mut proc = ShadowProcessor::new(ShadowConfig::default());
        let primary = make_fp(42);
        let shadow = make_fp(99); // Completely different

        let result = proc.compare(&primary, &shadow, 0.8, 0.9);
        assert!(!result.agreement);
        assert!(result.confidence_boost == 0.0);
        assert!(!result.prefer_primary); // Shadow has higher quality
    }

    #[test]
    fn test_agreement_rate() {
        let mut proc = ShadowProcessor::new(ShadowConfig::default());
        let fp = make_fp(42);

        // Two agreements
        proc.compare(&fp, &fp, 1.0, 1.0);
        proc.compare(&fp, &fp, 1.0, 1.0);

        // One disagreement
        let other = make_fp(99);
        proc.compare(&fp, &other, 1.0, 1.0);

        let rate = proc.agreement_rate();
        assert!((rate - 2.0 / 3.0).abs() < 0.01);
        assert_eq!(proc.total_comparisons(), 3);
    }

    #[test]
    fn test_merge_shadow() {
        let fp = make_fp(42);
        // Merging identical should return identical
        let merged = merge_shadow(&fp, &fp, true);
        assert_eq!(hamming_distance(&fp, &merged), 0);
    }

    #[test]
    fn test_merge_prefers_primary() {
        let primary = make_fp(42);
        let shadow = make_fp(99);
        let merged = merge_shadow(&primary, &shadow, true);
        // Merged should be identical to primary when prefer_primary=true
        // (because in merge_shadow, disagreement bits go to preferred)
        assert_eq!(hamming_distance(&primary, &merged), 0);
    }
}
