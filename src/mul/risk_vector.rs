//! Layer 4: Epistemic vs Moral Risk — Two Orthogonal Axes
//!
//! Epistemic risk: "am I likely to be wrong?"
//! Moral risk: "if I'm wrong, how bad is the damage?"
//! These are INDEPENDENT axes — high epistemic risk + low moral risk = safe to explore.

/// Risk vector — two orthogonal risk dimensions.
#[derive(Debug, Clone, Copy)]
pub struct RiskVector {
    /// Probability of being wrong (0.0 = certain correct, 1.0 = certain wrong)
    pub epistemic: f32,
    /// Potential damage if wrong (0.0 = harmless, 1.0 = catastrophic/irreversible)
    pub moral: f32,
}

impl RiskVector {
    pub fn new(epistemic: f32, moral: f32) -> Self {
        Self {
            epistemic: epistemic.clamp(0.0, 1.0),
            moral: moral.clamp(0.0, 1.0),
        }
    }

    /// Low risk — both axes below threshold.
    pub fn low() -> Self {
        Self::new(0.1, 0.1)
    }

    /// Whether exploration is appropriate.
    /// "I might be wrong but it doesn't matter much."
    pub fn allows_exploration(&self) -> bool {
        self.epistemic > 0.5 && self.moral < 0.3
    }

    /// Whether caution is required.
    /// "The stakes are too high to risk being wrong."
    pub fn requires_caution(&self) -> bool {
        self.moral > 0.7
    }

    /// Whether a sandbox environment should be requested.
    /// "I'm uncertain AND the stakes are high."
    pub fn needs_sandbox(&self) -> bool {
        self.epistemic > 0.3 && self.moral > 0.5
    }

    /// Combined risk for gating decisions.
    /// Uses max rather than product: either axis can block.
    pub fn combined(&self) -> f32 {
        self.epistemic.max(self.moral)
    }
}

impl Default for RiskVector {
    fn default() -> Self {
        Self::new(0.5, 0.1) // moderate epistemic, low moral
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exploration_allowed() {
        let r = RiskVector::new(0.7, 0.1);
        assert!(r.allows_exploration());
        assert!(!r.requires_caution());
    }

    #[test]
    fn test_caution_required() {
        let r = RiskVector::new(0.4, 0.8);
        assert!(!r.allows_exploration());
        assert!(r.requires_caution());
        assert!(r.needs_sandbox()); // epistemic 0.4 > 0.3 AND moral 0.8 > 0.5
    }

    #[test]
    fn test_combined_risk() {
        let r = RiskVector::new(0.3, 0.8);
        assert!((r.combined() - 0.8).abs() < 0.01);
    }
}
