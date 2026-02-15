//! Layer 8: Free Will Modifier — Multiplicative Confidence
//!
//! The modifier is the PRODUCT of four factors (all 0.0-1.0).
//! One weak factor kills the whole thing (multiplicative, not additive).
//! This makes the system CONSERVATIVE: you need confidence on ALL axes to act freely.

use super::dk_detector::DKDetector;
use super::homeostasis::CognitiveHomeostasis;
use super::risk_vector::RiskVector;
use super::trust_qualia::TrustQualia;

/// Free will modifier — multiplicative factor that modulates agency.
#[derive(Debug, Clone, Copy)]
pub struct FreeWillModifier {
    /// From DK detector (Layer 2)
    pub dk_factor: f32,
    /// From Trust qualia (Layer 1)
    pub trust_factor: f32,
    /// From complexity assessment (inverse of epistemic risk)
    pub complexity_factor: f32,
    /// From homeostasis (Layer 6)
    pub flow_factor: f32,
}

impl FreeWillModifier {
    /// Compute from MUL state.
    pub fn compute(
        dk: &DKDetector,
        trust: &TrustQualia,
        risk: &RiskVector,
        homeostasis: &CognitiveHomeostasis,
    ) -> Self {
        Self {
            dk_factor: dk.position.humility_factor(),
            trust_factor: trust.composite(),
            complexity_factor: (1.0 - risk.epistemic).clamp(0.0, 1.0),
            flow_factor: homeostasis.state.flow_factor(),
        }
    }

    /// The modifier value (0.0-1.0).
    pub fn value(&self) -> f32 {
        (self.dk_factor * self.trust_factor * self.complexity_factor * self.flow_factor)
            .clamp(0.0, 1.0)
    }

    /// Weakest component name (for diagnostics).
    pub fn weakest(&self) -> &'static str {
        let min = self
            .dk_factor
            .min(self.trust_factor)
            .min(self.complexity_factor)
            .min(self.flow_factor);
        if (min - self.dk_factor).abs() < f32::EPSILON {
            "dunning_kruger"
        } else if (min - self.trust_factor).abs() < f32::EPSILON {
            "trust"
        } else if (min - self.complexity_factor).abs() < f32::EPSILON {
            "complexity"
        } else {
            "flow"
        }
    }

    /// Create from a single pre-computed value (for deserialization).
    ///
    /// Stores the value in dk_factor with the rest at 1.0,
    /// so that `value()` returns the original.
    pub fn from_value(value: f32) -> Self {
        let v = value.clamp(0.0, 1.0);
        Self {
            dk_factor: v,
            trust_factor: 1.0,
            complexity_factor: 1.0,
            flow_factor: 1.0,
        }
    }
}

impl Default for FreeWillModifier {
    fn default() -> Self {
        Self {
            dk_factor: 1.0,
            trust_factor: 1.0,
            complexity_factor: 1.0,
            flow_factor: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimal_modifier() {
        let m = FreeWillModifier::default();
        assert!((m.value() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_mount_stupid_kills_modifier() {
        let m = FreeWillModifier {
            dk_factor: 0.3, // MountStupid humility
            trust_factor: 0.9,
            complexity_factor: 0.9,
            flow_factor: 1.0,
        };
        assert!(m.value() < 0.3);
        assert_eq!(m.weakest(), "dunning_kruger");
    }

    #[test]
    fn test_multiplicative_conservative() {
        let m = FreeWillModifier {
            dk_factor: 0.5,
            trust_factor: 0.5,
            complexity_factor: 0.5,
            flow_factor: 0.5,
        };
        // 0.5^4 = 0.0625
        assert!(m.value() < 0.1);
    }
}
