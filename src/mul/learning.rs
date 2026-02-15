//! Layer 10: Post-Action Learning — Convert Navigation into Knowledge
//!
//! After taking an action guided by the compass, the OUTCOME is compared
//! to the PREDICTION. The prediction error drives trust updates, DK updates,
//! and crystallization decisions.

use super::compass::CompassDecision;

/// Post-action learning — converts compass outcomes into knowledge updates.
#[derive(Debug, Clone)]
pub struct PostActionLearning {
    /// Compass decision that was used
    pub decision: CompassDecision,
    /// Predicted outcome confidence
    pub predicted_confidence: f32,
    /// Actual outcome (success=1.0, failure=0.0)
    pub actual_outcome: f32,
}

impl PostActionLearning {
    pub fn new(
        decision: CompassDecision,
        predicted_confidence: f32,
        actual_outcome: f32,
    ) -> Self {
        Self {
            decision,
            predicted_confidence: predicted_confidence.clamp(0.0, 1.0),
            actual_outcome: actual_outcome.clamp(0.0, 1.0),
        }
    }

    /// Prediction error (surprise signal — Friston free energy).
    pub fn prediction_error(&self) -> f32 {
        (self.predicted_confidence - self.actual_outcome).abs()
    }

    /// Should this outcome be crystallized? (persisted to long-term memory)
    pub fn should_crystallize(&self) -> bool {
        let clear = self.actual_outcome > 0.8 || self.actual_outcome < 0.2;
        let surprising = self.prediction_error() > 0.3;
        clear || surprising
    }

    /// Trust delta for Layer 1 update.
    pub fn trust_delta(&self) -> i8 {
        if self.prediction_error() < 0.1 {
            1 // well-calibrated → improve trust
        } else if self.prediction_error() > 0.4 {
            -1 // badly miscalibrated → degrade trust
        } else {
            0 // acceptable range
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_well_calibrated() {
        let l = PostActionLearning::new(CompassDecision::ExecuteWithLearning, 0.8, 0.85);
        assert!(l.prediction_error() < 0.1);
        assert_eq!(l.trust_delta(), 1);
    }

    #[test]
    fn test_badly_miscalibrated() {
        let l = PostActionLearning::new(CompassDecision::ExecuteWithLearning, 0.9, 0.1);
        assert!(l.prediction_error() > 0.5);
        assert_eq!(l.trust_delta(), -1);
        assert!(l.should_crystallize()); // Surprising failure
    }

    #[test]
    fn test_crystallize_on_clear_outcome() {
        let l = PostActionLearning::new(CompassDecision::Exploratory, 0.5, 0.95);
        assert!(l.should_crystallize()); // Clear success
    }
}
