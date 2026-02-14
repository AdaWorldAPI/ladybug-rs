//! NARS Truth Values: <frequency, confidence>
//!
//! Based on NAL (Non-Axiomatic Logic) truth functions.
//! Pure arithmetic — no I/O, no storage dependencies.

use std::fmt;

/// NARS truth value: <frequency, confidence>
///
/// - **frequency** (f): Proportion of positive evidence (0.0 - 1.0)
/// - **confidence** (c): Reliability of the frequency (0.0 - 1.0)
#[derive(Clone, Copy, PartialEq)]
pub struct TruthValue {
    pub frequency: f32,
    pub confidence: f32,
}

impl TruthValue {
    /// Evidential horizon parameter (controls confidence scaling).
    pub const HORIZON: f32 = 1.0;

    pub fn new(frequency: f32, confidence: f32) -> Self {
        debug_assert!((0.0..=1.0).contains(&frequency));
        debug_assert!((0.0..=1.0).contains(&confidence));
        Self {
            frequency,
            confidence,
        }
    }

    /// Create from positive/negative evidence counts.
    pub fn from_evidence(positive: f32, negative: f32) -> Self {
        let total = positive + negative;
        if total == 0.0 {
            return Self::unknown();
        }
        let frequency = positive / total;
        let confidence = total / (total + Self::HORIZON);
        Self {
            frequency,
            confidence,
        }
    }

    /// Create from positive evidence and total evidence.
    pub fn from_positive(positive: f32, total: f32) -> Self {
        if total == 0.0 {
            return Self::unknown();
        }
        let frequency = positive / total;
        let confidence = total / (total + Self::HORIZON);
        Self {
            frequency,
            confidence,
        }
    }

    /// Certain true: <1.0, 0.9>
    pub fn certain_true() -> Self {
        Self {
            frequency: 1.0,
            confidence: 0.9,
        }
    }

    /// Certain false: <0.0, 0.9>
    pub fn certain_false() -> Self {
        Self {
            frequency: 0.0,
            confidence: 0.9,
        }
    }

    /// Unknown: <0.5, 0.0>
    pub fn unknown() -> Self {
        Self {
            frequency: 0.5,
            confidence: 0.0,
        }
    }

    /// Convert to evidence counts.
    pub fn to_evidence(&self) -> (f32, f32) {
        let c = self.confidence.min(1.0 - 1.0 / (Self::HORIZON + 1000.0));
        let w = Self::HORIZON * c / (1.0 - c);
        let w_pos = w * self.frequency;
        let w_neg = w * (1.0 - self.frequency);
        (w_pos, w_neg)
    }

    /// Expected value (decision-making utility).
    /// e = c * (f - 0.5) + 0.5
    pub fn expectation(&self) -> f32 {
        self.confidence * (self.frequency - 0.5) + 0.5
    }

    /// Is this truth value "positive" (expectation > 0.5)?
    pub fn is_positive(&self) -> bool {
        self.expectation() > 0.5
    }

    /// Is this highly confident?
    pub fn is_confident(&self) -> bool {
        self.confidence > 0.7
    }

    // === Truth Functions ===

    /// Revision: Combine two truth values with independent evidence.
    pub fn revision(&self, other: &TruthValue) -> TruthValue {
        let (w1_pos, w1_neg) = self.to_evidence();
        let (w2_pos, w2_neg) = other.to_evidence();
        let w_pos = w1_pos + w2_pos;
        let w_neg = w1_neg + w2_neg;
        TruthValue::from_evidence(w_pos, w_neg)
    }

    /// Negation: NOT operation.
    pub fn negation(&self) -> TruthValue {
        TruthValue {
            frequency: 1.0 - self.frequency,
            confidence: self.confidence,
        }
    }

    /// Deduction: A→B, B→C ⊢ A→C
    pub fn deduction(&self, other: &TruthValue) -> TruthValue {
        let f = self.frequency * other.frequency;
        let c = self.confidence * other.confidence * f;
        TruthValue {
            frequency: f,
            confidence: c,
        }
    }

    /// Induction: A→B, A→C ⊢ B→C
    pub fn induction(&self, other: &TruthValue) -> TruthValue {
        let f = other.frequency;
        let c = self.frequency * self.confidence * other.confidence;
        TruthValue {
            frequency: f,
            confidence: c,
        }
    }

    /// Abduction: A→B, C→B ⊢ A→C
    pub fn abduction(&self, other: &TruthValue) -> TruthValue {
        let f = self.frequency;
        let c = other.frequency * self.confidence * other.confidence;
        TruthValue {
            frequency: f,
            confidence: c,
        }
    }

    /// Analogy: A→B, A↔C ⊢ C→B
    pub fn analogy(&self, other: &TruthValue) -> TruthValue {
        let f = self.frequency * other.frequency;
        let c = self.confidence * other.confidence * other.frequency;
        TruthValue {
            frequency: f,
            confidence: c,
        }
    }

    /// Comparison: A→B, C→B ⊢ A↔C
    pub fn comparison(&self, other: &TruthValue) -> TruthValue {
        let f1f2 = self.frequency * other.frequency;
        let f = f1f2 / (self.frequency + other.frequency - f1f2).max(f32::EPSILON);
        let c = self.confidence * other.confidence * f;
        TruthValue {
            frequency: f,
            confidence: c,
        }
    }

    /// Intersection: A, B ⊢ A ∧ B
    pub fn intersection(&self, other: &TruthValue) -> TruthValue {
        let f = self.frequency * other.frequency;
        let c = self.confidence * other.confidence;
        TruthValue {
            frequency: f,
            confidence: c,
        }
    }

    /// Union: A, B ⊢ A ∨ B
    pub fn union(&self, other: &TruthValue) -> TruthValue {
        let f = 1.0 - (1.0 - self.frequency) * (1.0 - other.frequency);
        let c = self.confidence * other.confidence;
        TruthValue {
            frequency: f,
            confidence: c,
        }
    }
}

impl fmt::Debug for TruthValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<{:.2}, {:.2}>", self.frequency, self.confidence)
    }
}

impl fmt::Display for TruthValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "\u{27E8}{:.0}%, {:.0}%\u{27E9}",
            self.frequency * 100.0,
            self.confidence * 100.0
        )
    }
}

impl Default for TruthValue {
    fn default() -> Self {
        Self::unknown()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.001
    }

    #[test]
    fn test_nars_truth_revision() {
        let a = TruthValue {
            frequency: 0.8,
            confidence: 0.9,
        };
        let b = TruthValue {
            frequency: 0.6,
            confidence: 0.7,
        };
        let revised = a.revision(&b);
        assert!(revised.confidence > a.confidence.max(b.confidence) * 0.9);
    }

    #[test]
    fn test_deduction() {
        let ab = TruthValue::new(0.9, 0.9);
        let bc = TruthValue::new(0.8, 0.8);
        let ac = ab.deduction(&bc);
        assert!(ac.frequency < ab.frequency);
        assert!(ac.confidence < bc.confidence);
    }

    #[test]
    fn test_negation() {
        let tv = TruthValue::new(0.8, 0.9);
        let neg = tv.negation();
        assert!(approx_eq(neg.frequency, 0.2));
        assert!(approx_eq(neg.confidence, 0.9));
    }
}
