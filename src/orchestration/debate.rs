//! Structured Debate — adversarial rounds with NARS truth revision.
//!
//! Implements a debate protocol where propositions are challenged, defended,
//! and revised through multiple rounds. Each round applies NARS revision
//! to merge pro/con evidence into an updated truth value.
//!
//! # Science
//! - Wang (2006): NARS revision merges independent evidence streams
//! - Mercier & Sperber (2011): Argumentative theory — reasoning evolved for debate
//! - Kahneman (2011): Adversarial collaboration improves calibration
//! - Tetlock (2005): Foxes (multiple perspectives) outperform hedgehogs (single theory)

use crate::nars::TruthValue;
use crate::search::hdr_cascade::{hamming_distance, WORDS};

const TOTAL_BITS: f32 = 16384.0;

/// A single argument in a debate.
#[derive(Debug, Clone)]
pub struct Argument {
    /// Fingerprint of the evidence/reasoning
    pub fingerprint: [u64; WORDS],
    /// Truth value of this argument
    pub truth: TruthValue,
    /// Whether this is pro (true) or con (false) the proposition
    pub is_pro: bool,
}

/// Result of a single debate round.
#[derive(Debug, Clone)]
pub struct RoundResult {
    /// Round number (0-indexed)
    pub round: usize,
    /// Truth value after this round's revision
    pub revised_truth: TruthValue,
    /// Pro argument strength (expectation of pro arguments)
    pub pro_strength: f32,
    /// Con argument strength (expectation of con arguments)
    pub con_strength: f32,
    /// Whether the debate has converged (truth changed less than threshold)
    pub converged: bool,
}

/// Complete debate outcome.
#[derive(Debug, Clone)]
pub struct DebateOutcome {
    /// Rounds executed
    pub rounds: Vec<RoundResult>,
    /// Final revised truth value
    pub final_truth: TruthValue,
    /// Whether the debate converged before max rounds
    pub converged: bool,
    /// Structural similarity between pro and con fingerprints
    /// (high similarity = genuine disagreement on same topic)
    pub structural_overlap: f32,
}

/// Configuration for a structured debate.
#[derive(Debug, Clone)]
pub struct DebateConfig {
    /// Maximum number of rounds
    pub max_rounds: usize,
    /// Convergence threshold: stop when |Δexpectation| < this
    pub convergence_threshold: f32,
    /// Minimum confidence to accept final truth
    pub min_confidence: f32,
}

impl Default for DebateConfig {
    fn default() -> Self {
        Self {
            max_rounds: 5,
            convergence_threshold: 0.01,
            min_confidence: 0.3,
        }
    }
}

/// Run a structured debate between pro and con arguments.
///
/// Each round:
/// 1. Bundle pro arguments → pro_truth
/// 2. Bundle con arguments → con_truth (negated)
/// 3. Revise proposition truth with both
/// 4. Check convergence
pub fn debate(
    initial_truth: &TruthValue,
    pro_args: &[Argument],
    con_args: &[Argument],
    config: &DebateConfig,
) -> DebateOutcome {
    let mut current_truth = *initial_truth;
    let mut rounds = Vec::new();
    let mut converged = false;

    // Compute structural overlap between pro and con
    let structural_overlap = if !pro_args.is_empty() && !con_args.is_empty() {
        let pro_centroid = bundle_fingerprints(pro_args);
        let con_centroid = bundle_fingerprints(con_args);
        let dist = hamming_distance(&pro_centroid, &con_centroid) as f32 / TOTAL_BITS;
        1.0 - dist // 1.0 = identical structure, 0.0 = maximally different
    } else {
        0.0
    };

    for round in 0..config.max_rounds {
        let prev_expectation = current_truth.expectation();

        // Aggregate pro arguments via NARS revision
        let pro_strength = if pro_args.is_empty() {
            0.5
        } else {
            let mut pro_truth = pro_args[0].truth;
            for arg in &pro_args[1..] {
                pro_truth = pro_truth.revision(&arg.truth);
            }
            pro_truth.expectation()
        };

        // Aggregate con arguments via NARS revision, then negate
        let con_strength = if con_args.is_empty() {
            0.5
        } else {
            let mut con_truth = con_args[0].truth;
            for arg in &con_args[1..] {
                con_truth = con_truth.revision(&arg.truth);
            }
            con_truth.expectation()
        };

        // Create round evidence: pro boosts frequency, con reduces it
        let round_evidence = TruthValue::from_evidence(
            pro_strength * pro_args.len() as f32,
            con_strength * con_args.len() as f32,
        );

        // Revise current truth with round evidence
        current_truth = current_truth.revision(&round_evidence);

        let delta = (current_truth.expectation() - prev_expectation).abs();
        let round_converged = delta < config.convergence_threshold;

        rounds.push(RoundResult {
            round,
            revised_truth: current_truth,
            pro_strength,
            con_strength,
            converged: round_converged,
        });

        if round_converged {
            converged = true;
            break;
        }
    }

    DebateOutcome {
        rounds,
        final_truth: current_truth,
        converged,
        structural_overlap,
    }
}

/// Bundle fingerprints from arguments into a centroid (majority vote).
fn bundle_fingerprints(args: &[Argument]) -> [u64; WORDS] {
    if args.is_empty() {
        return [0u64; WORDS];
    }
    if args.len() == 1 {
        return args[0].fingerprint;
    }

    let mut result = [0u64; WORDS];
    let threshold = args.len() / 2;

    for word_idx in 0..WORDS {
        let mut word = 0u64;
        for bit in 0..64 {
            let count = args.iter()
                .filter(|a| (a.fingerprint[word_idx] >> bit) & 1 == 1)
                .count();
            if count > threshold {
                word |= 1u64 << bit;
            }
        }
        result[word_idx] = word;
    }
    result
}

/// Quick verdict: is the proposition supported, opposed, or undecided?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Proposition is supported (expectation > 0.6)
    Supported,
    /// Proposition is opposed (expectation < 0.4)
    Opposed,
    /// Insufficient evidence to decide
    Undecided,
}

impl DebateOutcome {
    /// Get the verdict from the debate.
    pub fn verdict(&self) -> Verdict {
        let e = self.final_truth.expectation();
        if e > 0.6 {
            Verdict::Supported
        } else if e < 0.4 {
            Verdict::Opposed
        } else {
            Verdict::Undecided
        }
    }

    /// Number of rounds executed.
    pub fn round_count(&self) -> usize {
        self.rounds.len()
    }
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
    fn test_debate_pro_wins() {
        let initial = TruthValue::new(0.5, 0.1); // Uncertain
        let pro_args = vec![
            Argument { fingerprint: make_fp(1), truth: TruthValue::new(0.9, 0.8), is_pro: true },
            Argument { fingerprint: make_fp(2), truth: TruthValue::new(0.85, 0.7), is_pro: true },
        ];
        let con_args = vec![
            Argument { fingerprint: make_fp(3), truth: TruthValue::new(0.6, 0.3), is_pro: false },
        ];

        let outcome = debate(&initial, &pro_args, &con_args, &DebateConfig::default());
        assert!(outcome.final_truth.expectation() > 0.5);
        assert_eq!(outcome.verdict(), Verdict::Supported);
    }

    #[test]
    fn test_debate_con_wins() {
        let initial = TruthValue::new(0.5, 0.1);
        let pro_args = vec![
            Argument { fingerprint: make_fp(1), truth: TruthValue::new(0.6, 0.3), is_pro: true },
        ];
        let con_args = vec![
            Argument { fingerprint: make_fp(2), truth: TruthValue::new(0.95, 0.9), is_pro: false },
            Argument { fingerprint: make_fp(3), truth: TruthValue::new(0.9, 0.8), is_pro: false },
        ];

        let outcome = debate(&initial, &pro_args, &con_args, &DebateConfig::default());
        assert!(outcome.final_truth.expectation() < 0.5);
    }

    #[test]
    fn test_debate_converges() {
        let initial = TruthValue::new(0.5, 0.5);
        let pro_args = vec![
            Argument { fingerprint: make_fp(1), truth: TruthValue::new(0.7, 0.5), is_pro: true },
        ];
        let con_args = vec![
            Argument { fingerprint: make_fp(2), truth: TruthValue::new(0.7, 0.5), is_pro: false },
        ];

        let outcome = debate(&initial, &pro_args, &con_args, &DebateConfig::default());
        assert!(outcome.converged);
    }

    #[test]
    fn test_debate_empty_args() {
        let initial = TruthValue::new(0.7, 0.8);
        let outcome = debate(&initial, &[], &[], &DebateConfig::default());
        // With no arguments, should converge quickly
        assert!(outcome.converged);
    }

    #[test]
    fn test_debate_structural_overlap() {
        let fp1 = make_fp(42);
        let mut fp2 = fp1;
        // Flip a few bits to make similar but not identical
        fp2[0] ^= 0xFF;
        fp2[1] ^= 0xFF;

        let pro_args = vec![
            Argument { fingerprint: fp1, truth: TruthValue::new(0.8, 0.7), is_pro: true },
        ];
        let con_args = vec![
            Argument { fingerprint: fp2, truth: TruthValue::new(0.8, 0.7), is_pro: false },
        ];

        let outcome = debate(&TruthValue::new(0.5, 0.1), &pro_args, &con_args, &DebateConfig::default());
        // Similar fingerprints should have high structural overlap
        assert!(outcome.structural_overlap > 0.9);
    }

    #[test]
    fn test_verdict() {
        let outcome = DebateOutcome {
            rounds: vec![],
            final_truth: TruthValue::new(0.8, 0.9),
            converged: true,
            structural_overlap: 0.5,
        };
        assert_eq!(outcome.verdict(), Verdict::Supported);

        let outcome2 = DebateOutcome {
            rounds: vec![],
            final_truth: TruthValue::new(0.2, 0.9),
            converged: true,
            structural_overlap: 0.5,
        };
        assert_eq!(outcome2.verdict(), Verdict::Opposed);
    }
}
