//! Layer 1: Trust Qualia — The Felt-Sense of Knowing
//!
//! Trust is not a single scalar — it is a composite of 4 orthogonal dimensions:
//! competence, source reliability, environmental stability, and calibration accuracy.
//!
//! The geometric mean preserves the "weakest link" property: a single low dimension
//! drags the composite down, which is the correct safety behavior.

/// Trust qualia — composite trust from 4 orthogonal sources.
#[derive(Debug, Clone, Copy)]
pub struct TrustQualia {
    /// Competence trust: "can I do this?"
    /// Source: Brier score history, past success rate
    pub competence: f32,

    /// Source trust: "is the information reliable?"
    /// Source: NARS confidence of incoming evidence
    pub source: f32,

    /// Environment trust: "is the context stable?"
    /// Source: entropy of recent inputs (low entropy = stable)
    pub environment: f32,

    /// Calibration trust: "are my confidence estimates accurate?"
    /// Source: Brier calibration error (low error = well-calibrated)
    pub calibration: f32,
}

impl TrustQualia {
    /// Create with all dimensions at a given level.
    pub fn uniform(level: f32) -> Self {
        let level = level.clamp(0.0, 1.0);
        Self {
            competence: level,
            source: level,
            environment: level,
            calibration: level,
        }
    }

    /// Composite trust score (geometric mean preserves "weakest link" property).
    pub fn composite(&self) -> f32 {
        (self.competence * self.source * self.environment * self.calibration)
            .max(0.0)
            .powf(0.25)
    }

    /// Map composite trust to a texture level (0-4).
    pub fn texture_level(&self) -> TrustLevel {
        let c = self.composite();
        if c >= 0.85 {
            TrustLevel::Crystalline
        } else if c >= 0.65 {
            TrustLevel::Solid
        } else if c >= 0.45 {
            TrustLevel::Fuzzy
        } else if c >= 0.25 {
            TrustLevel::Murky
        } else {
            TrustLevel::Dissonant
        }
    }

    /// Which component is weakest? This is WHERE to focus improvement.
    pub fn weakest_component(&self) -> TrustComponent {
        let min = self
            .competence
            .min(self.source)
            .min(self.environment)
            .min(self.calibration);
        if (min - self.competence).abs() < f32::EPSILON {
            TrustComponent::Competence
        } else if (min - self.source).abs() < f32::EPSILON {
            TrustComponent::Source
        } else if (min - self.environment).abs() < f32::EPSILON {
            TrustComponent::Environment
        } else {
            TrustComponent::Calibration
        }
    }

    /// Build from metacognitive measurements.
    pub fn from_measurements(
        brier_score: f32,
        nars_confidence: f32,
        input_entropy: f32,
        calibration_error: f32,
    ) -> Self {
        Self {
            competence: (1.0 - brier_score).clamp(0.0, 1.0),
            source: nars_confidence.clamp(0.0, 1.0),
            environment: (1.0 - input_entropy.min(1.0)).clamp(0.0, 1.0),
            calibration: (1.0 - calibration_error).clamp(0.0, 1.0),
        }
    }

    /// Whether trust needs repair (below Fuzzy threshold).
    pub fn needs_repair(&self) -> bool {
        matches!(
            self.texture_level(),
            TrustLevel::Murky | TrustLevel::Dissonant
        )
    }
}

impl Default for TrustQualia {
    fn default() -> Self {
        Self::uniform(0.5) // Start at Fuzzy level
    }
}

/// Discrete trust level (maps to 3 bits in CogRecord W64).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TrustLevel {
    Crystalline = 0,
    Solid = 1,
    Fuzzy = 2,
    Murky = 3,
    Dissonant = 4,
}

impl TrustLevel {
    pub fn from_bits(bits: u8) -> Self {
        match bits {
            0 => Self::Crystalline,
            1 => Self::Solid,
            2 => Self::Fuzzy,
            3 => Self::Murky,
            _ => Self::Dissonant,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrustComponent {
    Competence,
    Source,
    Environment,
    Calibration,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_trust() {
        let t = TrustQualia::uniform(0.9);
        assert!((t.composite() - 0.9).abs() < 0.01);
        assert_eq!(t.texture_level(), TrustLevel::Crystalline);
    }

    #[test]
    fn test_weakest_link() {
        let t = TrustQualia {
            competence: 0.9,
            source: 0.9,
            environment: 0.1, // weak link
            calibration: 0.9,
        };
        // Geometric mean: (0.9 * 0.9 * 0.1 * 0.9)^0.25 ≈ 0.52
        assert!(t.composite() < 0.6);
        assert_eq!(t.weakest_component(), TrustComponent::Environment);
    }

    #[test]
    fn test_needs_repair() {
        assert!(!TrustQualia::uniform(0.9).needs_repair());
        assert!(!TrustQualia::uniform(0.5).needs_repair());
        assert!(TrustQualia::uniform(0.2).needs_repair());
    }

    #[test]
    fn test_texture_levels() {
        assert_eq!(TrustQualia::uniform(0.9).texture_level(), TrustLevel::Crystalline);
        assert_eq!(TrustQualia::uniform(0.7).texture_level(), TrustLevel::Solid);
        assert_eq!(TrustQualia::uniform(0.5).texture_level(), TrustLevel::Fuzzy);
        assert_eq!(TrustQualia::uniform(0.3).texture_level(), TrustLevel::Murky);
        assert_eq!(TrustQualia::uniform(0.1).texture_level(), TrustLevel::Dissonant);
    }
}
