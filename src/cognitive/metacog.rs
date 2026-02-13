//! Meta-Cognition — thinking about thinking.
//!
//! Tracks calibration of confidence estimates and provides reflexive
//! assessment of reasoning quality.
//!
//! # Science
//! - Fleming & Dolan (2012): meta-d' as type-2 sensitivity
//! - Brier (1950): Brier score for calibration
//! - Yeung & Summerfield (2012): Metacognition in human decision-making

use super::GateState;
use crate::nars::TruthValue;
use std::collections::VecDeque;

/// Maximum history window for calibration tracking.
const MAX_HISTORY: usize = 100;

/// Meta-assessment of current reasoning state.
#[derive(Debug, Clone)]
pub struct MetaAssessment {
    /// Current confidence level
    pub confidence: f32,
    /// Meta-confidence: how reliable is our confidence estimate?
    /// Low variance in recent confidence → high meta-confidence.
    pub meta_confidence: f32,
    /// Current gate state
    pub gate_state: GateState,
    /// Whether to admit ignorance (low confidence + poor calibration)
    pub should_admit_ignorance: bool,
    /// Brier score (calibration error, lower = better)
    pub calibration_error: f32,
}

/// Meta-cognition tracker.
///
/// Maintains a window of recent confidence estimates and computes
/// meta-level statistics about reasoning quality.
#[derive(Debug, Clone)]
pub struct MetaCognition {
    /// Recent confidence values
    confidence_history: VecDeque<f32>,
    /// Running calibration error (Brier score)
    calibration_error: f32,
    /// Number of predictions made
    prediction_count: u32,
    /// Sum of squared prediction errors
    brier_sum: f32,
}

impl MetaCognition {
    pub fn new() -> Self {
        Self {
            confidence_history: VecDeque::with_capacity(MAX_HISTORY),
            calibration_error: 0.0,
            prediction_count: 0,
            brier_sum: 0.0,
        }
    }

    /// Assess meta-cognitive state given current gate and truth value.
    pub fn assess(&mut self, gate: GateState, truth: &TruthValue) -> MetaAssessment {
        let confidence = truth.confidence;
        self.confidence_history.push_back(confidence);
        if self.confidence_history.len() > MAX_HISTORY {
            self.confidence_history.pop_front();
        }

        let n = self.confidence_history.len() as f32;
        let mean_conf: f32 = self.confidence_history.iter().sum::<f32>() / n;
        let variance: f32 = self
            .confidence_history
            .iter()
            .map(|c| (c - mean_conf).powi(2))
            .sum::<f32>()
            / n;

        // Meta-confidence: low variance in recent estimates → we know what we know
        let meta_confidence = 1.0 - variance.sqrt();

        MetaAssessment {
            confidence,
            meta_confidence,
            gate_state: gate,
            should_admit_ignorance: confidence < 0.3 && self.calibration_error > 0.2,
            calibration_error: self.calibration_error,
        }
    }

    /// Record a prediction outcome for Brier score calibration.
    ///
    /// `predicted_confidence`: what we said our confidence was.
    /// `actual_outcome`: 1.0 if we were right, 0.0 if wrong.
    pub fn record_outcome(&mut self, predicted_confidence: f32, actual_outcome: f32) {
        self.prediction_count += 1;
        self.brier_sum += (predicted_confidence - actual_outcome).powi(2);
        self.calibration_error = self.brier_sum / self.prediction_count as f32;
    }

    /// Current Brier score (calibration error).
    /// 0.0 = perfect calibration, 1.0 = worst possible.
    pub fn brier_score(&self) -> f32 {
        self.calibration_error
    }

    /// Mean confidence over the history window.
    pub fn mean_confidence(&self) -> f32 {
        if self.confidence_history.is_empty() {
            return 0.5;
        }
        self.confidence_history.iter().sum::<f32>() / self.confidence_history.len() as f32
    }

    /// Confidence variance over the history window.
    pub fn confidence_variance(&self) -> f32 {
        if self.confidence_history.len() < 2 {
            return 0.0;
        }
        let mean = self.mean_confidence();
        let n = self.confidence_history.len() as f32;
        self.confidence_history
            .iter()
            .map(|c| (c - mean).powi(2))
            .sum::<f32>()
            / n
    }

    /// Is reasoning well-calibrated? (Brier score < 0.15)
    pub fn is_calibrated(&self) -> bool {
        self.prediction_count > 10 && self.calibration_error < 0.15
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        self.confidence_history.clear();
        self.calibration_error = 0.0;
        self.prediction_count = 0;
        self.brier_sum = 0.0;
    }
}

impl Default for MetaCognition {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metacognition_basic() {
        let mut mc = MetaCognition::new();
        let truth = TruthValue::new(0.8, 0.7);
        let assessment = mc.assess(GateState::Flow, &truth);
        assert!((assessment.confidence - 0.7).abs() < 0.01);
        assert!(!assessment.should_admit_ignorance);
    }

    #[test]
    fn test_metacognition_ignorance() {
        let mut mc = MetaCognition::new();
        // Simulate poor calibration
        mc.calibration_error = 0.3;
        let truth = TruthValue::new(0.5, 0.2);
        let assessment = mc.assess(GateState::Block, &truth);
        assert!(assessment.should_admit_ignorance);
    }

    #[test]
    fn test_brier_score_calibration() {
        let mut mc = MetaCognition::new();
        // Good calibration: 90% confident, right 90% of the time
        for _ in 0..9 {
            mc.record_outcome(0.9, 1.0);
        }
        mc.record_outcome(0.9, 0.0);
        assert!(mc.brier_score() < 0.1);
    }

    #[test]
    fn test_brier_score_miscalibration() {
        let mut mc = MetaCognition::new();
        // Bad calibration: 90% confident but only right 50%
        for i in 0..20 {
            mc.record_outcome(0.9, if i % 2 == 0 { 1.0 } else { 0.0 });
        }
        assert!(mc.brier_score() > 0.15);
    }

    #[test]
    fn test_meta_confidence_stable() {
        let mut mc = MetaCognition::new();
        // Consistently confident → high meta-confidence
        for _ in 0..10 {
            mc.assess(GateState::Flow, &TruthValue::new(0.8, 0.8));
        }
        let assessment = mc.assess(GateState::Flow, &TruthValue::new(0.8, 0.8));
        assert!(
            assessment.meta_confidence > 0.9,
            "Stable confidence should yield high meta-confidence"
        );
    }

    #[test]
    fn test_meta_confidence_volatile() {
        let mut mc = MetaCognition::new();
        // Wildly varying confidence → low meta-confidence
        for i in 0..10 {
            let c = if i % 2 == 0 { 0.1 } else { 0.9 };
            mc.assess(GateState::Hold, &TruthValue::new(0.5, c));
        }
        let assessment = mc.assess(GateState::Hold, &TruthValue::new(0.5, 0.5));
        assert!(
            assessment.meta_confidence < 0.8,
            "Volatile confidence should yield low meta-confidence"
        );
    }
}
