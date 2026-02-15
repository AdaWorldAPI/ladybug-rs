//! Layer 2: Dunning-Kruger Detector — The Humility Engine
//!
//! Maps felt competence vs demonstrated competence to detect overconfidence.
//! MountStupid is the most dangerous state: high confidence with low experience.

/// Position on the Dunning-Kruger curve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DKPosition {
    /// Feels confident but lacks evidence (DANGEROUS)
    MountStupid = 0,
    /// Aware of gaps (CAUTIOUS but honest)
    ValleyOfDespair = 1,
    /// Building real competence
    SlopeOfEnlightenment = 2,
    /// Calibrated confidence matches ability
    PlateauOfMastery = 3,
}

impl DKPosition {
    /// Humility factor — how much to REDUCE confidence.
    pub fn humility_factor(self) -> f32 {
        match self {
            Self::MountStupid => 0.3,
            Self::ValleyOfDespair => 0.7,
            Self::SlopeOfEnlightenment => 0.85,
            Self::PlateauOfMastery => 1.0,
        }
    }

    pub fn from_bits(bits: u8) -> Self {
        match bits {
            0 => Self::MountStupid,
            1 => Self::ValleyOfDespair,
            2 => Self::SlopeOfEnlightenment,
            _ => Self::PlateauOfMastery,
        }
    }
}

/// Dunning-Kruger detector state.
#[derive(Debug, Clone)]
pub struct DKDetector {
    /// Current position on the curve
    pub position: DKPosition,
    /// Felt competence (self-reported confidence)
    pub felt_competence: f32,
    /// Demonstrated competence (empirical accuracy from Brier scores)
    pub demonstrated_competence: f32,
    /// Number of samples used for demonstration (experience)
    pub sample_count: u32,
    /// Smoothed gap: felt - demonstrated (positive = overconfident)
    pub gap: f32,
}

impl DKDetector {
    pub fn new() -> Self {
        Self {
            position: DKPosition::SlopeOfEnlightenment,
            felt_competence: 0.5,
            demonstrated_competence: 0.5,
            sample_count: 0,
            gap: 0.0,
        }
    }

    /// Classify position from gap and experience.
    pub fn classify(&mut self) {
        self.gap = self.felt_competence - self.demonstrated_competence;

        self.position = if self.sample_count < 10 && self.gap > 0.2 {
            DKPosition::MountStupid
        } else if self.gap < -0.15 {
            DKPosition::ValleyOfDespair
        } else if self.gap.abs() < 0.15 && self.sample_count > 50 {
            DKPosition::PlateauOfMastery
        } else {
            DKPosition::SlopeOfEnlightenment
        };
    }

    /// Update with new observation.
    pub fn observe(&mut self, predicted_confidence: f32, was_correct: bool) {
        self.sample_count = self.sample_count.saturating_add(1);
        let outcome = if was_correct { 1.0 } else { 0.0 };

        let alpha = 0.1;
        self.demonstrated_competence =
            self.demonstrated_competence * (1.0 - alpha) + outcome * alpha;

        self.felt_competence = predicted_confidence;
        self.classify();
    }
}

impl Default for DKDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mount_stupid() {
        let mut dk = DKDetector::new();
        dk.sample_count = 3;
        dk.felt_competence = 0.9;
        dk.demonstrated_competence = 0.3;
        dk.classify();
        assert_eq!(dk.position, DKPosition::MountStupid);
        assert!((dk.position.humility_factor() - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_valley_of_despair() {
        let mut dk = DKDetector::new();
        dk.sample_count = 20;
        dk.felt_competence = 0.3;
        dk.demonstrated_competence = 0.6;
        dk.classify();
        assert_eq!(dk.position, DKPosition::ValleyOfDespair);
    }

    #[test]
    fn test_plateau_of_mastery() {
        let mut dk = DKDetector::new();
        dk.sample_count = 100;
        dk.felt_competence = 0.8;
        dk.demonstrated_competence = 0.78;
        dk.classify();
        assert_eq!(dk.position, DKPosition::PlateauOfMastery);
    }

    #[test]
    fn test_observe_updates() {
        let mut dk = DKDetector::new();
        for _ in 0..20 {
            dk.observe(0.9, true);
        }
        assert!(dk.demonstrated_competence > 0.7);
        assert!(dk.sample_count == 20);
    }
}
