//! Satisfaction Gate — Maslow Hierarchy for Cognitive Layers
//!
//! Lower layers must be satisfied before higher layers fire.
//! This is NOT a binary gate — it's a continuous modulation.
//! A slightly unsatisfied L2 doesn't BLOCK L5; it raises L5's
//! threshold proportionally, making it harder but not impossible.
//!
//! This creates implicit focus of attention:
//! - Savant agent (high base thresholds) → tight focus, few layers fire
//! - Creative agent (low base thresholds) → wide awareness, many fire
//! - Both read the same satisfaction state from the shared CogRecord header
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    SATISFACTION GATE                             │
//! │                                                                 │
//! │  L10 ░░░░░░░░░░░░  Crystallization: only what's solid          │
//! │  L9  ░░░░░░░░░░░░  Validation: truth must harden               │
//! │  L8  ░░░░░░░░░░░░  Integration: must converge                  │
//! │  L7  ░░░░░░░░░░░░  Contingency: alternatives explored          │
//! │  L6  ░░░░░░░░░░░░  Delegation: agents must respond             │
//! │  ─── unsatisfied lower layers push these thresholds UP ──────  │
//! │  L5  ████████████  Execution: must produce output               │
//! │  L4  ████████████  Routing: route must be selected              │
//! │  L3  ████████████  Appraisal: gestalt must form                 │
//! │  L2  ████████████  Resonance: must find match                   │
//! │  L1  ████████████  Recognition: input must parse                │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use super::layer_stack::{LayerId, NUM_LAYERS};
use crate::core::Fingerprint;

/// Minimum satisfaction scores per layer.
/// Below this threshold, the layer is considered unsatisfied,
/// and higher layers' effective thresholds are raised.
const MINIMUM: [f32; NUM_LAYERS] = [
    0.3, // L1  Recognition:    input must parse
    0.3, // L2  Resonance:      must find some match
    0.4, // L3  Appraisal:      gestalt must form
    0.4, // L4  Routing:        route must be selected
    0.5, // L5  Execution:      must produce output
    0.5, // L6  Delegation:     agents must respond
    0.5, // L7  Contingency:    alternatives must be explored
    0.6, // L8  Integration:    must converge
    0.7, // L9  Validation:     truth must harden
    0.8, // L10 Crystallization: only crystallize what's solid
];

/// Default resonance thresholds per layer (before satisfaction modulation).
const DEFAULT_THRESHOLD: [f32; NUM_LAYERS] = [
    0.2, // L1  Low bar — everything gets recognized
    0.3, // L2  Slightly higher — need real resonance
    0.4, // L3  Moderate — gestalt needs substance
    0.4, // L4  Moderate — routing needs direction
    0.5, // L5  Medium — execution needs confidence
    0.5, // L6  Medium — delegation needs justification
    0.6, // L7  Higher — contingency needs real alternatives
    0.6, // L8  Higher — integration needs convergence
    0.7, // L9  High — validation needs strong evidence
    0.8, // L10 Highest — only crystallize certainty
];

/// Tracks satisfaction scores across the 10-layer stack.
///
/// The satisfaction state is the shared awareness substrate:
/// multiple agents read it, each with their own FieldModulation.
/// The savant sees high effective thresholds (focus). The creative
/// sees low ones (breadth). Same data, different lenses.
#[derive(Clone, Debug)]
pub struct LayerSatisfaction {
    /// Per-layer satisfaction scores [0.0, 1.0]
    pub scores: [f32; NUM_LAYERS],
}

impl LayerSatisfaction {
    /// Create with neutral satisfaction (all 0.5)
    pub fn new() -> Self {
        Self {
            scores: [0.5; NUM_LAYERS],
        }
    }

    /// Update satisfaction for a specific layer
    pub fn update(&mut self, layer: LayerId, satisfaction: f32) {
        self.scores[layer.index()] = satisfaction.clamp(0.0, 1.0);
    }

    /// Effective threshold for a layer.
    ///
    /// If any lower layer is unsatisfied, the threshold rises toward 1.0.
    /// The penalty is proportional to the deficit: a barely-unsatisfied
    /// lower layer causes a small bump, a completely unsatisfied one
    /// pushes the threshold near 1.0.
    pub fn effective_threshold(&self, layer: LayerId) -> f32 {
        let idx = layer.index();
        let base = DEFAULT_THRESHOLD[idx];

        // Find the worst deficit among lower layers
        let mut worst_penalty: f32 = 0.0;
        for i in 0..idx {
            if self.scores[i] < MINIMUM[i] {
                let deficit = MINIMUM[i] - self.scores[i];
                let penalty = deficit / MINIMUM[i];
                worst_penalty = worst_penalty.max(penalty);
            }
        }

        if worst_penalty > 0.0 {
            // Raise threshold proportionally to worst deficit
            (base + (1.0 - base) * worst_penalty).min(1.0)
        } else {
            base
        }
    }

    /// Effective threshold using async (previous-cycle) scores.
    ///
    /// Used by the 2-stroke engine: reads last cycle's lower-layer state
    /// rather than current cycle's, avoiding sequential dependency.
    pub fn effective_threshold_async(&self, layer: LayerId, prev_scores: &[f32; NUM_LAYERS]) -> f32 {
        let idx = layer.index();
        let base = DEFAULT_THRESHOLD[idx];

        let mut worst_penalty: f32 = 0.0;
        for i in 0..idx {
            if prev_scores[i] < MINIMUM[i] {
                let deficit = MINIMUM[i] - prev_scores[i];
                let penalty = deficit / MINIMUM[i];
                worst_penalty = worst_penalty.max(penalty);
            }
        }

        if worst_penalty > 0.0 {
            (base + (1.0 - base) * worst_penalty).min(1.0)
        } else {
            base
        }
    }

    /// Check if all dependencies for a layer are satisfied.
    pub fn is_ready(&self, layer: LayerId) -> bool {
        let idx = layer.index();
        (0..idx).all(|i| self.scores[i] >= MINIMUM[i])
    }

    /// Encode satisfaction state as a fingerprint.
    ///
    /// Used for resonance-gated ThinkingStyle selection: the system's
    /// current satisfaction state resonates with style fingerprints,
    /// and the most resonant style activates. No explicit dispatch.
    pub fn to_fingerprint(&self) -> Fingerprint {
        // Build a deterministic string from satisfaction scores
        // This maps each unique satisfaction state to a unique fingerprint
        let encoded: String = self
            .scores
            .iter()
            .enumerate()
            .map(|(i, &s)| format!("L{}:{:.2}", i + 1, s))
            .collect::<Vec<_>>()
            .join("|");
        Fingerprint::from_content(&encoded)
    }

    /// Get the minimum satisfaction scores (for external reference)
    pub fn minimums() -> &'static [f32; NUM_LAYERS] {
        &MINIMUM
    }

    /// Get the default thresholds (for external reference)
    pub fn defaults() -> &'static [f32; NUM_LAYERS] {
        &DEFAULT_THRESHOLD
    }
}

impl Default for LayerSatisfaction {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_thresholds() {
        let sat = LayerSatisfaction::new();
        // All satisfied at 0.5 — no penalties
        let t1 = sat.effective_threshold(LayerId::L1);
        assert!((t1 - DEFAULT_THRESHOLD[0]).abs() < 0.01);
    }

    #[test]
    fn test_unsatisfied_lower_raises_threshold() {
        let mut sat = LayerSatisfaction::new();
        // L1 is unsatisfied (below minimum 0.3)
        sat.update(LayerId::L1, 0.1);

        let t5 = sat.effective_threshold(LayerId::L5);
        let default_t5 = DEFAULT_THRESHOLD[4];
        assert!(
            t5 > default_t5,
            "Unsatisfied L1 should raise L5 threshold: {} > {}",
            t5,
            default_t5
        );
    }

    #[test]
    fn test_satisfied_no_penalty() {
        let mut sat = LayerSatisfaction::new();
        // All lower layers well above minimum
        for i in 0..5 {
            sat.scores[i] = 0.9;
        }

        let t6 = sat.effective_threshold(LayerId::L6);
        assert!(
            (t6 - DEFAULT_THRESHOLD[5]).abs() < 0.01,
            "Satisfied lower layers should not change threshold"
        );
    }

    #[test]
    fn test_is_ready() {
        let mut sat = LayerSatisfaction::new();
        // L1 satisfied, L2 not
        sat.update(LayerId::L1, 0.8);
        sat.update(LayerId::L2, 0.1);

        assert!(sat.is_ready(LayerId::L2)); // Only checks layers below L2 (= L1)
        assert!(!sat.is_ready(LayerId::L3)); // L2 is below minimum
    }

    #[test]
    fn test_worst_deficit_wins() {
        let mut sat = LayerSatisfaction::new();
        sat.update(LayerId::L1, 0.29); // Barely below 0.3 minimum
        sat.update(LayerId::L2, 0.01); // Far below 0.3 minimum

        let t5 = sat.effective_threshold(LayerId::L5);
        // L2's worse deficit should dominate
        assert!(t5 > 0.9, "Worst deficit should push threshold near 1.0: {}", t5);
    }

    #[test]
    fn test_continuous_degradation() {
        let sat_good = LayerSatisfaction::new(); // All 0.5
        let mut sat_bad = LayerSatisfaction::new();
        sat_bad.update(LayerId::L1, 0.2); // Slightly below minimum

        let t_good = sat_good.effective_threshold(LayerId::L5);
        let t_bad = sat_bad.effective_threshold(LayerId::L5);

        assert!(
            t_bad > t_good,
            "Slightly unsatisfied should raise threshold: {} > {}",
            t_bad,
            t_good
        );
        assert!(
            t_bad < 0.95,
            "Slightly unsatisfied should NOT push to near-1.0: {}",
            t_bad
        );
    }

    #[test]
    fn test_fingerprint_deterministic() {
        let sat = LayerSatisfaction::new();
        let fp1 = sat.to_fingerprint();
        let fp2 = sat.to_fingerprint();
        assert_eq!(fp1.similarity(&fp2), 1.0);
    }

    #[test]
    fn test_different_states_different_fingerprints() {
        let sat1 = LayerSatisfaction::new();
        let mut sat2 = LayerSatisfaction::new();
        sat2.update(LayerId::L5, 0.9);

        let fp1 = sat1.to_fingerprint();
        let fp2 = sat2.to_fingerprint();
        assert!(
            fp1.similarity(&fp2) < 0.9,
            "Different satisfaction states should produce different fingerprints"
        );
    }
}
