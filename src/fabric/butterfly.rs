//! Butterfly Detection - Amplification Cascade Monitoring
//!
//! Detects when small inputs cause disproportionately large effects.
//! Central to the cognitive fabric.

use std::collections::VecDeque;
use std::time::Instant;

use crate::cognitive::ThinkingStyle;
use crate::core::Fingerprint;
use crate::fabric::{FieldSnapshot, Subsystem};

/// Maximum butterflies to retain
const MAX_BUTTERFLIES: usize = 100;

/// Butterfly detector
pub struct ButterflyDetector {
    /// Detected butterflies
    butterflies: VecDeque<Butterfly>,

    /// Current sensitivity (from thinking style)
    sensitivity: f32,

    /// Minimum amplification to trigger detection
    min_amplification: f32,
}

impl ButterflyDetector {
    /// Create new detector
    pub fn new() -> Self {
        Self {
            butterflies: VecDeque::with_capacity(MAX_BUTTERFLIES),
            sensitivity: 0.5,
            min_amplification: 2.0,
        }
    }

    /// Set sensitivity from thinking style
    pub fn set_style(&mut self, style: ThinkingStyle) {
        self.sensitivity = style.butterfly_sensitivity();
        // Lower sensitivity = lower threshold = more detections
        self.min_amplification = 1.0 + self.sensitivity * 4.0;
    }

    /// Detect butterfly from field history
    pub fn detect(
        &mut self,
        history: &VecDeque<FieldSnapshot>,
        trigger: &Fingerprint,
        resonance_count: usize,
    ) -> Option<Butterfly> {
        if history.len() < 2 {
            return None;
        }

        let prev = &history[history.len() - 2];
        let current = &history[history.len() - 1];

        // Input magnitude: 1 concept added
        let input_magnitude = 1.0;

        // Effect magnitude: resonance cascade size
        let effect_magnitude = resonance_count as f32;

        // Amplification ratio
        let amplification = effect_magnitude / input_magnitude;

        if amplification >= self.min_amplification {
            // Butterfly detected!
            let butterfly = Butterfly {
                trigger: trigger.clone(),
                amplification,
                affected_count: resonance_count,
                affected_subsystems: self.infer_affected_subsystems(prev, current),
                detected_at: Instant::now(),
                field_delta: self.compute_delta(prev, current),
            };

            self.record(butterfly.clone());
            Some(butterfly)
        } else {
            None
        }
    }

    /// Predict butterfly from hypothetical change
    pub fn predict(
        &self,
        hypothetical: &Fingerprint,
        field_superposition: &Fingerprint,
        field_concept_count: usize,
    ) -> ButterflyPrediction {
        // Estimate resonance based on similarity to superposition
        let resonance_estimate = hypothetical.similarity(field_superposition);

        // Higher similarity to superposition = more potential resonance
        let estimated_cascade = (resonance_estimate * field_concept_count as f32) as usize;

        // Predicted amplification
        let predicted_amplification = estimated_cascade as f32;

        // Confidence based on field size
        let confidence = if field_concept_count > 100 {
            0.7
        } else if field_concept_count > 10 {
            0.5
        } else {
            0.3
        };

        ButterflyPrediction {
            trigger: hypothetical.clone(),
            predicted_amplification,
            predicted_cascade_size: estimated_cascade,
            will_trigger: predicted_amplification >= self.min_amplification,
            confidence,
        }
    }

    /// Get recent butterflies
    pub fn recent(&self, n: usize) -> Vec<&Butterfly> {
        self.butterflies.iter().rev().take(n).collect()
    }

    /// Get all butterflies
    pub fn all(&self) -> &VecDeque<Butterfly> {
        &self.butterflies
    }

    /// Clear history
    pub fn clear(&mut self) {
        self.butterflies.clear();
    }

    fn record(&mut self, butterfly: Butterfly) {
        if self.butterflies.len() >= MAX_BUTTERFLIES {
            self.butterflies.pop_front();
        }
        self.butterflies.push_back(butterfly);
    }

    fn infer_affected_subsystems(
        &self,
        prev: &FieldSnapshot,
        current: &FieldSnapshot,
    ) -> Vec<Subsystem> {
        let mut affected = Vec::new();

        for i in 0..5 {
            let prev_count = prev.subsystem_counts[i].1;
            let curr_count = current.subsystem_counts[i].1;

            // Subsystem affected if its concept count changed significantly
            if curr_count > prev_count + 1 {
                affected.push(current.subsystem_counts[i].0);
            }
        }

        affected
    }

    fn compute_delta(&self, prev: &FieldSnapshot, current: &FieldSnapshot) -> FieldDelta {
        FieldDelta {
            concept_count_change: current.concept_count as i32 - prev.concept_count as i32,
            superposition_drift: prev.superposition.similarity(&current.superposition),
            time_elapsed: current.timestamp.duration_since(prev.timestamp),
        }
    }
}

impl Default for ButterflyDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// A detected butterfly effect
#[derive(Clone, Debug)]
pub struct Butterfly {
    /// The concept that triggered the cascade
    pub trigger: Fingerprint,

    /// Amplification factor (effect/input)
    pub amplification: f32,

    /// Number of concepts affected
    pub affected_count: usize,

    /// Which subsystems were affected
    pub affected_subsystems: Vec<Subsystem>,

    /// When detected
    pub detected_at: Instant,

    /// Change in field state
    pub field_delta: FieldDelta,
}

/// Prediction of butterfly effect
#[derive(Clone, Debug)]
pub struct ButterflyPrediction {
    /// The hypothetical trigger
    pub trigger: Fingerprint,

    /// Predicted amplification
    pub predicted_amplification: f32,

    /// Predicted cascade size
    pub predicted_cascade_size: usize,

    /// Whether it would trigger detection
    pub will_trigger: bool,

    /// Confidence in prediction (0.0 - 1.0)
    pub confidence: f32,
}

/// Change in field state
#[derive(Clone, Debug)]
pub struct FieldDelta {
    pub concept_count_change: i32,
    pub superposition_drift: f32,
    pub time_elapsed: std::time::Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_butterfly_sensitivity() {
        let mut detector = ButterflyDetector::new();

        // Creative style = low sensitivity = catches more
        detector.set_style(ThinkingStyle::Creative);
        assert!(detector.min_amplification < 2.0);

        // Analytical = high sensitivity = catches fewer
        detector.set_style(ThinkingStyle::Analytical);
        assert!(detector.min_amplification > 3.0);
    }

    #[test]
    fn test_butterfly_prediction() {
        let detector = ButterflyDetector::new();

        let trigger = Fingerprint::from_content("test");
        let superposition = Fingerprint::from_content("test similar");

        let prediction = detector.predict(&trigger, &superposition, 100);

        // Should have some prediction
        assert!(prediction.confidence > 0.0);
    }
}
