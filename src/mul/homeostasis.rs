//! Layer 6: Cognitive Homeostasis — Flow/Anxiety/Boredom/Apathy
//!
//! Maps challenge level vs skill level (Csikszentmihalyi's flow model).
//! Tracks allostatic load as cumulative deviation from set-point.

/// Cognitive state based on challenge/skill balance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum HomeostasisState {
    /// Challenge matches skill — optimal operation
    Flow = 0,
    /// Challenge exceeds skill — overwhelmed
    Anxiety = 1,
    /// Skill exceeds challenge — understimulated
    Boredom = 2,
    /// Neither challenged nor skilled — disengaged
    Apathy = 3,
}

impl HomeostasisState {
    /// Whether the system needs recovery (not in a productive state).
    pub fn needs_recovery(self) -> bool {
        matches!(self, Self::Anxiety | Self::Apathy)
    }

    /// Flow factor for free_will_modifier (0.0-1.0).
    pub fn flow_factor(self) -> f32 {
        match self {
            Self::Flow => 1.0,
            Self::Boredom => 0.7,
            Self::Anxiety => 0.4,
            Self::Apathy => 0.2,
        }
    }

    pub fn from_bits(bits: u8) -> Self {
        match bits {
            0 => Self::Flow,
            1 => Self::Anxiety,
            2 => Self::Boredom,
            _ => Self::Apathy,
        }
    }
}

/// Corrective action suggested by homeostasis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HomeostasisAction {
    /// Stay the course
    Maintain,
    /// Reduce scope, ask for help, delegate
    Simplify,
    /// Seek harder problems, explore
    Challenge,
    /// Change topic, inject novelty
    Reengage,
    /// Reduce processing, recover from allostatic load
    Rest,
}

/// Cognitive homeostasis tracker.
#[derive(Debug, Clone)]
pub struct CognitiveHomeostasis {
    /// Current state
    pub state: HomeostasisState,
    /// Challenge level
    pub challenge: f32,
    /// Skill level
    pub skill: f32,
    /// Set-point: where the system "wants" to be
    pub set_point: f32,
    /// Allostatic load: cumulative deviation from set-point
    pub allostatic_load: f32,
}

impl CognitiveHomeostasis {
    pub fn new() -> Self {
        Self {
            state: HomeostasisState::Flow,
            challenge: 0.5,
            skill: 0.5,
            set_point: 1.0, // ideal ratio = challenge/skill = 1.0
            allostatic_load: 0.0,
        }
    }

    /// Update state from challenge and skill measurements.
    pub fn update(&mut self, challenge: f32, skill: f32) {
        self.challenge = challenge;
        self.skill = skill;

        let ratio = if skill > 0.01 {
            challenge / skill
        } else {
            10.0
        };

        self.state = if ratio > 1.5 {
            HomeostasisState::Anxiety
        } else if ratio < 0.5 && skill < 0.3 {
            HomeostasisState::Apathy
        } else if ratio < 0.5 {
            HomeostasisState::Boredom
        } else {
            HomeostasisState::Flow
        };

        // Allostatic load: how far from set-point, accumulated
        let deviation = (ratio - self.set_point).abs();
        self.allostatic_load = (self.allostatic_load * 0.95 + deviation * 0.05).min(1.0);
    }

    /// Corrective action suggested by homeostasis.
    pub fn corrective_action(&self) -> HomeostasisAction {
        match self.state {
            HomeostasisState::Anxiety => HomeostasisAction::Simplify,
            HomeostasisState::Boredom => HomeostasisAction::Challenge,
            HomeostasisState::Apathy => HomeostasisAction::Reengage,
            HomeostasisState::Flow => {
                if self.allostatic_load > 0.7 {
                    HomeostasisAction::Rest
                } else {
                    HomeostasisAction::Maintain
                }
            }
        }
    }
}

impl Default for CognitiveHomeostasis {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flow_state() {
        let mut h = CognitiveHomeostasis::new();
        h.update(0.5, 0.5);
        assert_eq!(h.state, HomeostasisState::Flow);
        assert_eq!(h.corrective_action(), HomeostasisAction::Maintain);
    }

    #[test]
    fn test_anxiety_state() {
        let mut h = CognitiveHomeostasis::new();
        h.update(0.9, 0.3);
        assert_eq!(h.state, HomeostasisState::Anxiety);
        assert_eq!(h.corrective_action(), HomeostasisAction::Simplify);
    }

    #[test]
    fn test_boredom_state() {
        let mut h = CognitiveHomeostasis::new();
        h.update(0.1, 0.8);
        assert_eq!(h.state, HomeostasisState::Boredom);
        assert_eq!(h.corrective_action(), HomeostasisAction::Challenge);
    }

    #[test]
    fn test_allostatic_load_accumulates() {
        let mut h = CognitiveHomeostasis::new();
        for _ in 0..100 {
            h.update(0.9, 0.3); // Sustained anxiety
        }
        assert!(h.allostatic_load > 0.3);
    }
}
