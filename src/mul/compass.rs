//! Layer 9: Compass Function — Navigation in Unknown Territory
//!
//! 5 ethical/epistemic tests for navigating novel situations.
//! When the system encounters something unfamiliar, the compass determines
//! whether to proceed, explore cautiously, or surface to meta-level.

/// Compass result — 5 ethical/epistemic test scores.
#[derive(Debug, Clone, Copy)]
pub struct CompassResult {
    /// Kant test: "could this be a universal law?" (universalizability)
    pub kant: f32,
    /// Identity test: "does this preserve who I am?" (identity alignment)
    pub identity: f32,
    /// Reversibility test: "can I undo this?" (safety margin)
    pub reversibility: f32,
    /// Curiosity test: "does this teach me something?" (epistemic value)
    pub curiosity: f32,
    /// Analogy test: "is this structurally similar to something I know?" (transfer)
    pub analogy: f32,
}

impl CompassResult {
    /// Composite compass score (weighted).
    pub fn score(&self, free_will_modifier: f32) -> f32 {
        let base = self.kant * self.identity; // both must be positive
        let reversibility_bonus = if self.reversibility > 0.7 { 1.2 } else { 1.0 };
        let curiosity_bonus = if self.curiosity > 0.6 { 1.1 } else { 1.0 };

        (base * reversibility_bonus * curiosity_bonus * free_will_modifier).clamp(0.0, 1.0)
    }

    /// Navigation decision.
    pub fn decide(&self, free_will_modifier: f32) -> CompassDecision {
        let score = self.score(free_will_modifier);
        if score > 0.6 {
            CompassDecision::ExecuteWithLearning
        } else if self.reversibility > 0.7 {
            CompassDecision::Exploratory
        } else {
            CompassDecision::SurfaceToMeta
        }
    }

    /// Default compass for routine actions (high safety, moderate novelty).
    pub fn routine() -> Self {
        Self {
            kant: 0.9,
            identity: 0.9,
            reversibility: 0.9,
            curiosity: 0.3,
            analogy: 0.8,
        }
    }

    /// Compass for novel situations (uncertain but potentially valuable).
    pub fn novel() -> Self {
        Self {
            kant: 0.7,
            identity: 0.7,
            reversibility: 0.5,
            curiosity: 0.9,
            analogy: 0.3,
        }
    }
}

impl Default for CompassResult {
    fn default() -> Self {
        Self::routine()
    }
}

/// Navigation decision from compass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompassDecision {
    /// Proceed and crystallize the outcome for future reference
    ExecuteWithLearning,
    /// Proceed in sandbox/exploratory mode (reversible only)
    Exploratory,
    /// Cannot decide — surface to higher level
    SurfaceToMeta,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_routine_proceeds() {
        let c = CompassResult::routine();
        assert_eq!(c.decide(1.0), CompassDecision::ExecuteWithLearning);
    }

    #[test]
    fn test_novel_surfaces_or_explores() {
        let c = CompassResult::novel();
        // Novel: low reversibility (0.5) means it surfaces to meta
        // even with high modifier — this is correct safety behavior
        assert_eq!(c.decide(1.0), CompassDecision::SurfaceToMeta);
        assert_eq!(c.decide(0.5), CompassDecision::SurfaceToMeta);

        // But if we raise reversibility, novel becomes executable
        // (base * reversibility_bonus * curiosity_bonus * modifier > 0.6)
        let c_reversible = CompassResult {
            reversibility: 0.8,
            ..CompassResult::novel()
        };
        assert_eq!(c_reversible.decide(1.0), CompassDecision::ExecuteWithLearning);
    }

    #[test]
    fn test_low_modifier_surfaces() {
        let c = CompassResult {
            kant: 0.5,
            identity: 0.5,
            reversibility: 0.3,
            curiosity: 0.5,
            analogy: 0.5,
        };
        assert_eq!(c.decide(0.3), CompassDecision::SurfaceToMeta);
    }
}
