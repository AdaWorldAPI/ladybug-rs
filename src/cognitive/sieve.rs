//! Socratic Sieve — Three Gates for Self-Modification
//!
//! Before a thinking style modification is applied, it must pass
//! three Socratic gates:
//!
//! 1. **Truth**: Does the change improve accuracy? (More FLOW decisions)
//! 2. **Good**: Does calibration improve? (Lower Brier score)
//! 3. **Necessary**: Did the old style actually fail? (Recent blocks)
//!
//! All three must pass. This prevents premature style switching and
//! ensures modifications are evidence-grounded.
//!
//! # Science
//!
//! - Socrates (Plato, Apology): Three filters — Is it true? Is it good? Is it necessary?
//! - Brier (1950): Verification of forecasts via quadratic scoring rule
//! - Pearl/Bareinboim (2014): Transportability — changes must preserve causal structure

use crate::cognitive::collapse_gate::{CollapseDecision, GateState};
use crate::core::Fingerprint;
use crate::nars::TruthValue;

/// Socratic sieve — three gates for self-modification.
///
/// Extended for L9 Validation with:
/// - NARS truth revision across independent evidence
/// - Brier calibration scoring
/// - XOR residual (popcount distance from expected)
/// - Dunning-Kruger detection (high confidence + high XOR residual)
pub struct SocraticSieve;

impl SocraticSieve {
    /// Gate 1: Is it TRUE? Does the proposed change improve accuracy?
    ///
    /// Compares collapse decisions before and after the style change.
    /// More FLOW decisions = better accuracy.
    pub fn gate_truth(
        before: &[CollapseDecision],
        after: &[CollapseDecision],
    ) -> bool {
        if before.is_empty() || after.is_empty() {
            return false; // Need evidence both ways
        }

        let flow_before = before.iter().filter(|d| d.state == GateState::Flow).count();
        let flow_after = after.iter().filter(|d| d.state == GateState::Flow).count();

        // Normalize by length to handle different sample sizes
        let rate_before = flow_before as f32 / before.len() as f32;
        let rate_after = flow_after as f32 / after.len() as f32;

        rate_after > rate_before
    }

    /// Gate 2: Is it GOOD? Does calibration improve?
    ///
    /// Uses Brier score: lower = better calibrated.
    /// Brier = (1/N) Σ (forecast - outcome)²
    pub fn gate_good(brier_before: f32, brier_after: f32) -> bool {
        brier_after < brier_before // Lower Brier = better calibration
    }

    /// Gate 3: Is it NECESSARY? Did the old style actually fail?
    ///
    /// Only modify if the old style was producing BLOCK decisions.
    /// Prevents gratuitous style switching when things are working.
    pub fn gate_necessary(recent_blocks: usize, threshold: usize) -> bool {
        recent_blocks >= threshold // Only modify if actually blocking
    }

    /// All three gates must pass for modification to proceed.
    ///
    /// Returns (passed, reason) tuple.
    pub fn evaluate(
        before: &[CollapseDecision],
        after: &[CollapseDecision],
        brier_before: f32,
        brier_after: f32,
        recent_blocks: usize,
        block_threshold: usize,
    ) -> (bool, &'static str) {
        if !Self::gate_truth(before, after) {
            return (false, "truth: new style does not improve FLOW rate");
        }
        if !Self::gate_good(brier_before, brier_after) {
            return (false, "good: new style does not improve calibration");
        }
        if !Self::gate_necessary(recent_blocks, block_threshold) {
            return (false, "necessary: current style is not failing enough to justify change");
        }
        (true, "all gates passed")
    }

    /// Compute Brier score from collapse decisions and ground truth outcomes.
    ///
    /// Each decision's winner_score is treated as the forecast probability.
    /// `outcomes[i]` is 1.0 if the decision was correct, 0.0 if not.
    pub fn brier_score(decisions: &[CollapseDecision], outcomes: &[f32]) -> f32 {
        if decisions.is_empty() {
            return 1.0; // Worst possible
        }

        let n = decisions.len().min(outcomes.len());
        let mut sum = 0.0;

        for i in 0..n {
            let forecast = decisions[i].winner_score.unwrap_or(0.5);
            let outcome = outcomes[i];
            let diff = forecast - outcome;
            sum += diff * diff;
        }

        sum / n as f32
    }

    // =========================================================================
    // L9 VALIDATION: Extended Socratic Sieve
    // =========================================================================

    /// L9 full validation pipeline: NARS + Brier + XOR residual + Dunning-Kruger.
    ///
    /// Returns (pass, reason, confidence) triple.
    /// `nars_tv`: merged NARS truth value from resonance-gated inference.
    /// `calibration_error`: current Brier score (lower = better).
    /// `xor_residual`: popcount distance between expected and actual fingerprint.
    ///                 High residual = the result diverges from the index.
    /// `expected_popcount`: baseline popcount for fingerprints at this centroid.
    ///                      1-2 sigma is the sweet spot.
    pub fn validate_l9(
        nars_tv: &TruthValue,
        calibration_error: f32,
        xor_residual: u32,
        expected_popcount: u32,
    ) -> (bool, &'static str, f32) {
        // 1. NARS: is the truth value strong enough?
        let nars_pass = nars_tv.confidence > 0.4 && nars_tv.frequency > 0.3;

        // 2. Calibration: is our confidence estimate reliable?
        let calibrated = calibration_error < 0.25;

        // 3. XOR residual: how far is the result from the centroid?
        // Sweet spot: within 1-2 sigma of expected popcount.
        // Beyond 2 sigma → suspicious divergence.
        let sigma = (expected_popcount as f32).sqrt().max(1.0);
        let deviation = (xor_residual as f32 - expected_popcount as f32).abs() / sigma;
        let xor_pass = deviation < 2.5; // 2.5 sigma tolerance

        // 4. Dunning-Kruger: high confidence + high XOR residual = suspicious.
        // The system THINKS it's right but the fingerprint says otherwise.
        let dunning_kruger = nars_tv.confidence > 0.85 && deviation > 1.5;

        if dunning_kruger {
            return (
                false,
                "Dunning-Kruger: high confidence but XOR residual exceeds 1.5σ",
                nars_tv.confidence * 0.3, // Slash confidence
            );
        }

        if !nars_pass {
            return (false, "NARS: insufficient evidence or agreement", nars_tv.confidence);
        }

        if !calibrated {
            return (
                false,
                "Brier: calibration error too high, gather more outcomes",
                nars_tv.confidence * (1.0 - calibration_error),
            );
        }

        if !xor_pass {
            return (
                false,
                "XOR: result diverges beyond 2.5σ from centroid",
                nars_tv.confidence * (1.0 - deviation / 5.0).max(0.1),
            );
        }

        // All passed — confidence adjusted by calibration quality
        let adjusted_confidence = nars_tv.confidence * (1.0 - calibration_error * 0.5);
        (true, "L9 validation passed", adjusted_confidence)
    }

    /// Compute XOR residual between a result fingerprint and an expected centroid.
    ///
    /// Returns (xor_residual_popcount, centroid_popcount).
    /// The XOR residual is the popcount of `result ⊕ centroid` — how many bits
    /// differ. For random 10K-bit fingerprints, expected popcount ≈ 5000.
    /// 1-2 sigma (σ ≈ 50) means residual in [4900, 5100] is normal.
    pub fn xor_residual(result: &Fingerprint, centroid: &Fingerprint) -> (u32, u32) {
        let xor_fp = result.bind(centroid); // XOR = bind
        let residual = xor_fp.popcount();
        let centroid_pop = centroid.popcount();
        (residual, centroid_pop)
    }

    /// Evaluate NARS truth revision across multiple independent evidence sources.
    ///
    /// Each source provides a TruthValue. Independent sources are revised
    /// together — this is the "multi-spectator merge" from L7 Contingency.
    pub fn nars_revision(sources: &[TruthValue]) -> TruthValue {
        if sources.is_empty() {
            return TruthValue::unknown();
        }
        let mut merged = sources[0].clone();
        for source in &sources[1..] {
            merged = merged.revision(source);
        }
        merged
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive::collapse_gate::{CollapseAction, CollapseDecision};

    fn make_decision(state: GateState, sd: f32, score: f32) -> CollapseDecision {
        CollapseDecision {
            state,
            sd,
            can_collapse: state == GateState::Flow,
            action: CollapseAction::Hold {
                sppm_key: "test".to_string(),
            },
            reason: "test".to_string(),
            winner_index: Some(0),
            winner_score: Some(score),
        }
    }

    #[test]
    fn test_gate_truth_improvement() {
        let before = vec![
            make_decision(GateState::Hold, 0.2, 0.6),
            make_decision(GateState::Block, 0.4, 0.5),
            make_decision(GateState::Flow, 0.1, 0.9),
        ];
        let after = vec![
            make_decision(GateState::Flow, 0.1, 0.9),
            make_decision(GateState::Flow, 0.08, 0.95),
            make_decision(GateState::Hold, 0.2, 0.7),
        ];

        assert!(SocraticSieve::gate_truth(&before, &after));
    }

    #[test]
    fn test_gate_truth_no_improvement() {
        let before = vec![
            make_decision(GateState::Flow, 0.1, 0.9),
            make_decision(GateState::Flow, 0.08, 0.95),
        ];
        let after = vec![
            make_decision(GateState::Hold, 0.2, 0.6),
            make_decision(GateState::Block, 0.4, 0.5),
        ];

        assert!(!SocraticSieve::gate_truth(&before, &after));
    }

    #[test]
    fn test_gate_good_calibration() {
        assert!(SocraticSieve::gate_good(0.3, 0.15)); // Lower = better
        assert!(!SocraticSieve::gate_good(0.15, 0.3)); // Higher = worse
    }

    #[test]
    fn test_gate_necessary() {
        assert!(SocraticSieve::gate_necessary(3, 3)); // At threshold
        assert!(SocraticSieve::gate_necessary(5, 3)); // Above threshold
        assert!(!SocraticSieve::gate_necessary(1, 3)); // Below threshold
    }

    #[test]
    fn test_evaluate_all_pass() {
        let before = vec![
            make_decision(GateState::Block, 0.4, 0.5),
            make_decision(GateState::Hold, 0.2, 0.6),
        ];
        let after = vec![
            make_decision(GateState::Flow, 0.1, 0.9),
            make_decision(GateState::Flow, 0.08, 0.95),
        ];

        let (passed, _reason) = SocraticSieve::evaluate(
            &before,
            &after,
            0.3,  // brier before
            0.15, // brier after (better)
            3,    // recent blocks
            2,    // threshold
        );
        assert!(passed);
    }

    #[test]
    fn test_evaluate_fails_truth() {
        let before = vec![make_decision(GateState::Flow, 0.1, 0.9)];
        let after = vec![make_decision(GateState::Block, 0.4, 0.5)];

        let (passed, reason) = SocraticSieve::evaluate(&before, &after, 0.3, 0.15, 3, 2);
        assert!(!passed);
        assert!(reason.contains("truth"));
    }

    #[test]
    fn test_evaluate_fails_necessary() {
        let before = vec![make_decision(GateState::Hold, 0.2, 0.6)];
        let after = vec![make_decision(GateState::Flow, 0.1, 0.9)];

        let (passed, reason) = SocraticSieve::evaluate(
            &before, &after,
            0.3, 0.15,
            0, // no recent blocks
            2, // threshold
        );
        assert!(!passed);
        assert!(reason.contains("necessary"));
    }

    #[test]
    fn test_brier_score_perfect() {
        let decisions = vec![
            make_decision(GateState::Flow, 0.05, 1.0),
            make_decision(GateState::Flow, 0.05, 1.0),
        ];
        let outcomes = vec![1.0, 1.0]; // Perfect predictions

        let brier = SocraticSieve::brier_score(&decisions, &outcomes);
        assert!(brier < 0.01, "Perfect predictions should have near-zero Brier: {}", brier);
    }

    #[test]
    fn test_brier_score_bad() {
        let decisions = vec![
            make_decision(GateState::Flow, 0.05, 0.9),
            make_decision(GateState::Flow, 0.05, 0.8),
        ];
        let outcomes = vec![0.0, 0.0]; // Completely wrong

        let brier = SocraticSieve::brier_score(&decisions, &outcomes);
        assert!(brier > 0.5, "Wrong predictions should have high Brier: {}", brier);
    }

    // =========================================================================
    // L9 VALIDATION TESTS
    // =========================================================================

    #[test]
    fn test_l9_validation_pass() {
        let tv = TruthValue::new(0.8, 0.7);
        let (pass, _reason, conf) = SocraticSieve::validate_l9(&tv, 0.1, 5000, 5000);
        assert!(pass);
        assert!(conf > 0.5);
    }

    #[test]
    fn test_l9_dunning_kruger_detection() {
        // Very confident (0.9) but XOR residual far from centroid
        let tv = TruthValue::new(0.8, 0.9);
        let sigma = (5000f32).sqrt(); // ~70.7
        let residual = 5000 + (sigma * 2.0) as u32; // 2σ above centroid
        let (pass, reason, _conf) = SocraticSieve::validate_l9(&tv, 0.1, residual, 5000);
        assert!(!pass);
        assert!(reason.contains("Dunning-Kruger"));
    }

    #[test]
    fn test_l9_nars_insufficient() {
        let tv = TruthValue::new(0.2, 0.2); // Too low
        let (pass, reason, _) = SocraticSieve::validate_l9(&tv, 0.1, 5000, 5000);
        assert!(!pass);
        assert!(reason.contains("NARS"));
    }

    #[test]
    fn test_l9_poor_calibration() {
        let tv = TruthValue::new(0.8, 0.7);
        let (pass, reason, _) = SocraticSieve::validate_l9(&tv, 0.4, 5000, 5000);
        assert!(!pass);
        assert!(reason.contains("Brier"));
    }

    #[test]
    fn test_l9_xor_divergent() {
        let tv = TruthValue::new(0.7, 0.6);
        // 3σ above centroid
        let sigma = (5000f32).sqrt();
        let residual = 5000 + (sigma * 3.0) as u32;
        let (pass, reason, _) = SocraticSieve::validate_l9(&tv, 0.1, residual, 5000);
        assert!(!pass);
        assert!(reason.contains("XOR"));
    }

    #[test]
    fn test_xor_residual_same_fingerprint() {
        let fp = Fingerprint::from_content("test");
        let (residual, _centroid_pop) = SocraticSieve::xor_residual(&fp, &fp);
        assert_eq!(residual, 0, "XOR of identical fingerprints should be zero");
    }

    #[test]
    fn test_xor_residual_different_fingerprints() {
        let a = Fingerprint::from_content("alpha");
        let b = Fingerprint::from_content("beta");
        let (residual, _) = SocraticSieve::xor_residual(&a, &b);
        assert!(residual > 0, "Different fingerprints should have nonzero XOR residual");
    }

    #[test]
    fn test_nars_revision_empty() {
        let merged = SocraticSieve::nars_revision(&[]);
        assert!(merged.confidence < 0.01);
    }

    #[test]
    fn test_nars_revision_single() {
        let tv = TruthValue::new(0.8, 0.7);
        let merged = SocraticSieve::nars_revision(&[tv.clone()]);
        assert!((merged.frequency - tv.frequency).abs() < 0.01);
    }

    #[test]
    fn test_nars_revision_multiple_converges() {
        let sources = vec![
            TruthValue::new(0.7, 0.6),
            TruthValue::new(0.8, 0.5),
            TruthValue::new(0.75, 0.7),
        ];
        let merged = SocraticSieve::nars_revision(&sources);
        // Revision of independent sources should increase confidence
        assert!(merged.confidence > sources[0].confidence);
    }
}
