//! Contradiction Detection — find beliefs with similar structure but opposing truth.
//!
//! # Science
//! - Wang (2006): NARS revision rule — two conflicting evidence streams merge
//! - Priest (2002): Paraconsistent Logic — tolerate contradiction without explosion
//! - CHAODA (Ishaq et al.): Anomaly detection on CLAM tree — structural outliers ARE contradictions

use super::TruthValue;
use crate::search::hdr_cascade::{hamming_distance, WORDS};

const TOTAL_BITS: f32 = 16384.0;

/// A detected contradiction between two beliefs.
#[derive(Debug, Clone)]
pub struct Contradiction {
    /// Index of first belief
    pub a: usize,
    /// Index of second belief
    pub b: usize,
    /// Structural similarity (1.0 = identical fingerprints)
    pub structural_similarity: f32,
    /// Truth conflict magnitude (|f_a - f_b|)
    pub truth_conflict: f32,
    /// NARS-revised resolution of the contradiction
    pub resolution: TruthValue,
}

/// Detect contradictions: fingerprints that are structurally similar
/// but have opposing NARS truth values.
///
/// `similarity_threshold`: minimum structural similarity (0.7 = 70% similar bits).
/// `conflict_threshold`: minimum truth frequency difference to count as conflict.
pub fn detect_contradictions(
    fingerprints: &[[u64; WORDS]],
    truths: &[TruthValue],
    similarity_threshold: f32,
    conflict_threshold: f32,
) -> Vec<Contradiction> {
    assert_eq!(fingerprints.len(), truths.len());

    let mut contradictions = Vec::new();
    for i in 0..fingerprints.len() {
        for j in (i + 1)..fingerprints.len() {
            let distance = hamming_distance(&fingerprints[i], &fingerprints[j]);
            let structural_sim = 1.0 - distance as f32 / TOTAL_BITS;
            let truth_conflict = (truths[i].frequency - truths[j].frequency).abs();

            // High structural similarity + high truth conflict = contradiction
            if structural_sim > similarity_threshold && truth_conflict > conflict_threshold {
                contradictions.push(Contradiction {
                    a: i,
                    b: j,
                    structural_similarity: structural_sim,
                    truth_conflict,
                    resolution: truths[i].revision(&truths[j]),
                });
            }
        }
    }
    contradictions
}

/// Deliberately induce cognitive dissonance for creativity.
///
/// Given a belief, construct its dialectical opposite with maximum tension.
///
/// # Science
/// - Festinger (1957): Cognitive dissonance theory
/// - Berlyne (1960): Optimal arousal — moderate conflict drives curiosity
/// - Peng & Nisbett (1999): Dialectical thinking
pub fn induce_dissonance(truth: &TruthValue) -> (TruthValue, f32) {
    // Maximum dissonance: flip frequency, halve confidence
    let opposite = TruthValue::new(1.0 - truth.frequency, truth.confidence * 0.5);
    let tension = (truth.frequency - opposite.frequency).abs() * truth.confidence;
    (opposite, tension)
}

/// Check belief coherence across a set of truths.
///
/// Returns coherence score (1.0 = perfectly coherent, 0.0 = contradictory).
pub fn coherence_score(truths: &[TruthValue]) -> f32 {
    if truths.len() < 2 {
        return 1.0;
    }
    let n = truths.len();
    let mut total_conflict = 0.0;
    let mut pairs = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            total_conflict += (truths[i].frequency - truths[j].frequency).abs();
            pairs += 1;
        }
    }
    1.0 - (total_conflict / pairs as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_contradictions() {
        // Two similar fingerprints with opposing truths
        let mut fp_a = [0u64; WORDS];
        fp_a[0] = 0xFFFF_FFFF_FFFF_FFFF;
        fp_a[1] = 0xFFFF_FFFF_FFFF_FFFF;
        let mut fp_b = fp_a;
        fp_b[100] ^= 0xFF; // Slightly different

        let truths = vec![
            TruthValue::new(0.9, 0.8),  // Strongly positive
            TruthValue::new(0.1, 0.8),  // Strongly negative
        ];

        let contradictions = detect_contradictions(
            &[fp_a, fp_b],
            &truths,
            0.9,  // Very similar structures
            0.5,  // At least 0.5 truth gap
        );
        assert_eq!(contradictions.len(), 1);
        assert!(contradictions[0].truth_conflict > 0.5);
    }

    #[test]
    fn test_no_contradictions_when_different() {
        let fp_a = [0xAAAA_AAAA_AAAA_AAAAu64; WORDS];
        let fp_b = [0x5555_5555_5555_5555u64; WORDS];

        let truths = vec![
            TruthValue::new(0.9, 0.8),
            TruthValue::new(0.1, 0.8),
        ];

        let contradictions = detect_contradictions(
            &[fp_a, fp_b],
            &truths,
            0.7,
            0.5,
        );
        assert!(contradictions.is_empty(), "Different structures shouldn't contradict");
    }

    #[test]
    fn test_induce_dissonance() {
        let belief = TruthValue::new(0.8, 0.7);
        let (opposite, tension) = induce_dissonance(&belief);
        assert!((opposite.frequency - 0.2).abs() < 0.01);
        assert!(tension > 0.0);
    }

    #[test]
    fn test_coherence_score() {
        let coherent = vec![
            TruthValue::new(0.8, 0.8),
            TruthValue::new(0.82, 0.7),
            TruthValue::new(0.79, 0.9),
        ];
        assert!(coherence_score(&coherent) > 0.9);

        let incoherent = vec![
            TruthValue::new(0.9, 0.8),
            TruthValue::new(0.1, 0.8),
        ];
        assert!(coherence_score(&incoherent) < 0.3);
    }
}
