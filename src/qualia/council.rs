//! Inner Council — Three-Archetype VSA Consensus via BundleCollector
//!
//! Three specialist perspectives evaluate the same decision frame,
//! each biased by a different qualia vector. The bundled result
//! is the consensus: bits that survive majority vote across all three.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │  Archetype   │  Bias              │  Qualia Profile          │
//! ├──────────────┼────────────────────┼──────────────────────────┤
//! │  Guardian    │  Safety-first      │  safe, certain, ordered  │
//! │  Catalyst    │  Growth-seeking    │  active, new, creating   │
//! │  Balanced    │  Uniform moderate  │  all axes at 0.5         │
//! └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! Each archetype XOR-binds its qualia bias with the decision fingerprint,
//! producing a perspective-shifted view. The three views are then bundled
//! via majority vote. Bits shared by 2+ archetypes form the consensus.

use crate::storage::FINGERPRINT_WORDS;
use super::meaning_axes::{AxisActivation, encode_axes};

// =============================================================================
// ARCHETYPES
// =============================================================================

/// Council archetype — a perspective lens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Archetype {
    /// Safety-first: prioritizes safe, certain, ordered, calm
    Guardian,
    /// Growth-seeking: prioritizes active, new, creating, open
    Catalyst,
    /// Moderate baseline: all axes at neutral
    Balanced,
}

impl Archetype {
    /// Generate the qualia bias fingerprint for this archetype.
    pub fn bias_fingerprint(&self) -> [u64; FINGERPRINT_WORDS] {
        let activations = self.bias_activations();
        encode_axes(&activations)
    }

    /// Raw axis activations for this archetype.
    fn bias_activations(&self) -> AxisActivation {
        let mut a = [0.0f32; 48];
        match self {
            Self::Guardian => {
                a[32] = 0.9;   // safe
                a[20] = 0.8;   // certain
                a[40] = 0.7;   // ordered
                a[25] = 0.8;   // calm
                a[17] = 0.6;   // permanent
                a[5] = 0.5;    // hard
            }
            Self::Catalyst => {
                a[2] = 0.9;    // active
                a[16] = 0.8;   // new
                a[44] = 0.7;   // creating
                a[38] = 0.8;   // open
                a[39] = 0.6;   // free
                a[41] = 0.7;   // alive
                a[42] = 0.5;   // growing
            }
            Self::Balanced => {
                // Uniform moderate — all axes slightly positive
                for v in a.iter_mut() {
                    *v = 0.3;
                }
            }
        }
        a
    }
}

// =============================================================================
// INNER COUNCIL
// =============================================================================

/// The three-archetype inner council.
pub struct InnerCouncil {
    guardian_bias: [u64; FINGERPRINT_WORDS],
    catalyst_bias: [u64; FINGERPRINT_WORDS],
    balanced_bias: [u64; FINGERPRINT_WORDS],
}

impl InnerCouncil {
    pub fn new() -> Self {
        Self {
            guardian_bias: Archetype::Guardian.bias_fingerprint(),
            catalyst_bias: Archetype::Catalyst.bias_fingerprint(),
            balanced_bias: Archetype::Balanced.bias_fingerprint(),
        }
    }

    /// Deliberate on a decision fingerprint.
    ///
    /// Each archetype XOR-binds its bias with the input, producing a
    /// perspective-shifted view. The three views are bundled via majority vote.
    ///
    /// Returns (consensus, votes) where votes[i] is each archetype's view.
    pub fn deliberate(
        &self,
        decision: &[u64; FINGERPRINT_WORDS],
    ) -> ([u64; FINGERPRINT_WORDS], [[u64; FINGERPRINT_WORDS]; 3]) {
        // Each archetype sees the decision through its own lens (XOR-bind)
        let guardian_view = xor_bind(decision, &self.guardian_bias);
        let catalyst_view = xor_bind(decision, &self.catalyst_bias);
        let balanced_view = xor_bind(decision, &self.balanced_bias);

        // Majority vote: bit survives if 2+ of 3 archetypes agree
        let consensus = majority_vote_3(
            &guardian_view,
            &catalyst_view,
            &balanced_view,
        );

        (consensus, [guardian_view, catalyst_view, balanced_view])
    }

    /// How much does each archetype agree with the consensus?
    /// Returns (guardian_sim, catalyst_sim, balanced_sim) in [0, 1].
    pub fn agreement(
        &self,
        consensus: &[u64; FINGERPRINT_WORDS],
        votes: &[[u64; FINGERPRINT_WORDS]; 3],
    ) -> [f32; 3] {
        let total_bits = (FINGERPRINT_WORDS * 64) as f32;
        [
            1.0 - hamming_distance(&votes[0], consensus) as f32 / total_bits,
            1.0 - hamming_distance(&votes[1], consensus) as f32 / total_bits,
            1.0 - hamming_distance(&votes[2], consensus) as f32 / total_bits,
        ]
    }
}

impl Default for InnerCouncil {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// EPIPHANY DETECTOR
// =============================================================================

/// Detects unexpected resonance after Burst-mode perturbation.
///
/// The false-flow → burst → epiphany pipeline:
/// 1. MUL detects stagnation (false flow, L5)
/// 2. Scheduler switches to Burst mode (inject novelty)
/// 3. Burst produces random/perturbed frames
/// 4. One hits unexpected high resonance → EpiphanyDetector fires
/// 5. MUL learns from surprise (L10 PostActionLearning)
/// 6. System recalibrates and naturally exits Burst
pub struct EpiphanyDetector {
    /// Baseline similarity from recent frames (rolling average)
    baseline_similarity: f32,
    /// Threshold multiplier: signal must be this many × baseline to trigger
    surprise_factor: f32,
    /// Recent similarity samples for baseline computation
    recent_samples: Vec<f32>,
    /// Max samples to keep
    window_size: usize,
    /// Total epiphanies detected
    pub epiphanies: u64,
}

/// An epiphany event — unexpected resonance discovery.
#[derive(Debug, Clone)]
pub struct Epiphany {
    /// The fingerprint that triggered the discovery
    pub fingerprint: [u64; FINGERPRINT_WORDS],
    /// How similar it was to the query (0-1)
    pub similarity: f32,
    /// How surprising relative to baseline (multiplier)
    pub surprise: f32,
    /// The baseline at time of detection
    pub baseline: f32,
}

impl EpiphanyDetector {
    pub fn new() -> Self {
        Self {
            baseline_similarity: 0.5,
            surprise_factor: 1.5,
            recent_samples: Vec::with_capacity(64),
            window_size: 64,
            epiphanies: 0,
        }
    }

    /// Feed a similarity score from a Burst-mode resonance result.
    /// Returns Some(Epiphany) if this is a surprising discovery.
    pub fn observe(
        &mut self,
        similarity: f32,
        fingerprint: [u64; FINGERPRINT_WORDS],
    ) -> Option<Epiphany> {
        // Update baseline
        self.recent_samples.push(similarity);
        if self.recent_samples.len() > self.window_size {
            self.recent_samples.remove(0);
        }
        self.baseline_similarity = self.recent_samples.iter().sum::<f32>()
            / self.recent_samples.len() as f32;

        // Check for surprise
        let threshold = self.baseline_similarity * self.surprise_factor;
        if similarity > threshold && self.recent_samples.len() >= 4 {
            self.epiphanies += 1;
            Some(Epiphany {
                fingerprint,
                similarity,
                surprise: similarity / self.baseline_similarity.max(0.01),
                baseline: self.baseline_similarity,
            })
        } else {
            None
        }
    }

    /// Reset after mode transition (e.g., Burst → Sprint).
    pub fn reset_baseline(&mut self) {
        self.recent_samples.clear();
        self.baseline_similarity = 0.5;
    }
}

impl Default for EpiphanyDetector {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// VSA HELPERS
// =============================================================================

fn xor_bind(
    a: &[u64; FINGERPRINT_WORDS],
    b: &[u64; FINGERPRINT_WORDS],
) -> [u64; FINGERPRINT_WORDS] {
    let mut result = [0u64; FINGERPRINT_WORDS];
    for i in 0..FINGERPRINT_WORDS {
        result[i] = a[i] ^ b[i];
    }
    result
}

fn majority_vote_3(
    a: &[u64; FINGERPRINT_WORDS],
    b: &[u64; FINGERPRINT_WORDS],
    c: &[u64; FINGERPRINT_WORDS],
) -> [u64; FINGERPRINT_WORDS] {
    let mut result = [0u64; FINGERPRINT_WORDS];
    for i in 0..FINGERPRINT_WORDS {
        // Bit is set if 2+ of 3 inputs have it set
        // majority(a,b,c) = (a & b) | (a & c) | (b & c)
        result[i] = (a[i] & b[i]) | (a[i] & c[i]) | (b[i] & c[i]);
    }
    result
}

fn hamming_distance(
    a: &[u64; FINGERPRINT_WORDS],
    b: &[u64; FINGERPRINT_WORDS],
) -> u32 {
    let mut dist = 0u32;
    for i in 0..FINGERPRINT_WORDS {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_council_consensus() {
        let council = InnerCouncil::new();

        // Create a decision fingerprint (something active + safe)
        let mut activations = [0.0f32; 48];
        activations[2] = 0.7;   // active
        activations[32] = 0.8;  // safe
        activations[41] = 0.5;  // alive
        let decision = encode_axes(&activations);

        let (consensus, votes) = council.deliberate(&decision);

        // Consensus should have bits (non-zero)
        let bits: u32 = consensus.iter().map(|w| w.count_ones()).sum();
        assert!(bits > 0, "consensus should have set bits");

        // Agreement scores should be > 0
        let agreement = council.agreement(&consensus, &votes);
        for (i, &a) in agreement.iter().enumerate() {
            assert!(a > 0.0, "archetype {} agreement should be > 0: {}", i, a);
        }
    }

    #[test]
    fn test_majority_vote_3_logic() {
        let a = [0b111u64; FINGERPRINT_WORDS]; // all 3 lower bits
        let b = [0b110u64; FINGERPRINT_WORDS]; // bits 1,2
        let c = [0b101u64; FINGERPRINT_WORDS]; // bits 0,2

        let result = majority_vote_3(&a, &b, &c);

        // Bit 0: a=1, b=0, c=1 → 2/3 → set
        // Bit 1: a=1, b=1, c=0 → 2/3 → set
        // Bit 2: a=1, b=1, c=1 → 3/3 → set
        assert_eq!(result[0] & 0b111, 0b111);
    }

    #[test]
    fn test_epiphany_detection() {
        let mut detector = EpiphanyDetector::new();
        let dummy_fp = [0u64; FINGERPRINT_WORDS];

        // Feed baseline samples (moderate similarity)
        for _ in 0..10 {
            assert!(detector.observe(0.5, dummy_fp).is_none());
        }

        // Now a surprisingly high similarity should trigger
        let result = detector.observe(0.95, dummy_fp);
        assert!(result.is_some(), "0.95 should be an epiphany when baseline is 0.5");

        let epiphany = result.unwrap();
        assert!(epiphany.surprise > 1.5);
        assert_eq!(detector.epiphanies, 1);
    }

    #[test]
    fn test_no_false_epiphany() {
        let mut detector = EpiphanyDetector::new();
        let dummy_fp = [0u64; FINGERPRINT_WORDS];

        // Feed high baseline
        for _ in 0..10 {
            detector.observe(0.8, dummy_fp);
        }

        // 0.9 is only 1.125× baseline — should NOT trigger (need 1.5×)
        let result = detector.observe(0.9, dummy_fp);
        assert!(result.is_none(), "0.9 with 0.8 baseline shouldn't trigger");
    }
}
