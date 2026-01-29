//! 12 Thinking Styles - Execution Dispatch, Not Metadata
//!
//! Each style is a different "lens" through the cognitive substrate.
//! Style determines HOW operations execute, not just what they return.

use std::fmt;

/// The 12 thinking styles organized into clusters
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum ThinkingStyle {
    // === Convergent Cluster ===
    Analytical,
    Convergent,
    Systematic,
    
    // === Divergent Cluster ===
    Creative,
    Divergent,
    Exploratory,
    
    // === Attention Cluster ===
    Focused,
    Diffuse,
    Peripheral,
    
    // === Speed Cluster ===
    Intuitive,
    #[default]
    Deliberate,
    
    // === Meta Cluster ===
    Metacognitive,
}

/// Field modulation parameters
#[derive(Clone, Copy, Debug)]
pub struct FieldModulation {
    pub resonance_threshold: f32,
    pub fan_out: usize,
    pub depth_bias: f32,
    pub breadth_bias: f32,
    pub noise_tolerance: f32,
    pub speed_bias: f32,
    pub exploration: f32,
}

impl Default for FieldModulation {
    fn default() -> Self {
        Self {
            resonance_threshold: 0.5,
            fan_out: 5,
            depth_bias: 0.5,
            breadth_bias: 0.5,
            noise_tolerance: 0.2,
            speed_bias: 0.5,
            exploration: 0.2,
        }
    }
}

impl ThinkingStyle {
    pub fn field_modulation(&self) -> FieldModulation {
        match self {
            Self::Analytical => FieldModulation {
                resonance_threshold: 0.85, fan_out: 3, depth_bias: 1.0,
                breadth_bias: 0.1, noise_tolerance: 0.05, speed_bias: 0.1, exploration: 0.05,
            },
            Self::Convergent => FieldModulation {
                resonance_threshold: 0.75, fan_out: 4, depth_bias: 0.8,
                breadth_bias: 0.2, noise_tolerance: 0.1, speed_bias: 0.3, exploration: 0.1,
            },
            Self::Systematic => FieldModulation {
                resonance_threshold: 0.7, fan_out: 5, depth_bias: 0.7,
                breadth_bias: 0.3, noise_tolerance: 0.15, speed_bias: 0.2, exploration: 0.1,
            },
            Self::Creative => FieldModulation {
                resonance_threshold: 0.35, fan_out: 12, depth_bias: 0.2,
                breadth_bias: 1.0, noise_tolerance: 0.4, speed_bias: 0.5, exploration: 0.8,
            },
            Self::Divergent => FieldModulation {
                resonance_threshold: 0.4, fan_out: 10, depth_bias: 0.3,
                breadth_bias: 0.9, noise_tolerance: 0.35, speed_bias: 0.4, exploration: 0.7,
            },
            Self::Exploratory => FieldModulation {
                resonance_threshold: 0.3, fan_out: 15, depth_bias: 0.4,
                breadth_bias: 0.8, noise_tolerance: 0.5, speed_bias: 0.6, exploration: 0.9,
            },
            Self::Focused => FieldModulation {
                resonance_threshold: 0.9, fan_out: 1, depth_bias: 1.0,
                breadth_bias: 0.0, noise_tolerance: 0.02, speed_bias: 0.2, exploration: 0.0,
            },
            Self::Diffuse => FieldModulation {
                resonance_threshold: 0.45, fan_out: 8, depth_bias: 0.3,
                breadth_bias: 0.7, noise_tolerance: 0.3, speed_bias: 0.5, exploration: 0.4,
            },
            Self::Peripheral => FieldModulation {
                resonance_threshold: 0.2, fan_out: 20, depth_bias: 0.1,
                breadth_bias: 0.5, noise_tolerance: 0.6, speed_bias: 0.7, exploration: 0.6,
            },
            Self::Intuitive => FieldModulation {
                resonance_threshold: 0.5, fan_out: 3, depth_bias: 0.3,
                breadth_bias: 0.4, noise_tolerance: 0.25, speed_bias: 0.9, exploration: 0.3,
            },
            Self::Deliberate => FieldModulation {
                resonance_threshold: 0.7, fan_out: 7, depth_bias: 0.6,
                breadth_bias: 0.5, noise_tolerance: 0.1, speed_bias: 0.1, exploration: 0.2,
            },
            Self::Metacognitive => FieldModulation {
                resonance_threshold: 0.5, fan_out: 5, depth_bias: 0.5,
                breadth_bias: 0.5, noise_tolerance: 0.2, speed_bias: 0.3, exploration: 0.3,
            },
        }
    }
    
    pub fn butterfly_sensitivity(&self) -> f32 {
        match self {
            Self::Peripheral => 0.1,
            Self::Exploratory => 0.15,
            Self::Creative => 0.2,
            Self::Diffuse => 0.25,
            Self::Intuitive => 0.3,
            Self::Divergent => 0.35,
            Self::Metacognitive => 0.4,
            Self::Deliberate => 0.5,
            Self::Systematic => 0.6,
            Self::Convergent => 0.7,
            Self::Analytical => 0.8,
            Self::Focused => 0.9,
        }
    }
    
    pub fn fan_out(&self) -> usize {
        self.field_modulation().fan_out
    }
    
    pub fn confidence_threshold(&self) -> f32 {
        self.field_modulation().resonance_threshold
    }
    
    pub const ALL: [ThinkingStyle; 12] = [
        Self::Analytical, Self::Convergent, Self::Systematic,
        Self::Creative, Self::Divergent, Self::Exploratory,
        Self::Focused, Self::Diffuse, Self::Peripheral,
        Self::Intuitive, Self::Deliberate, Self::Metacognitive,
    ];
}

impl fmt::Display for ThinkingStyle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Analytical => write!(f, "🔬 Analytical"),
            Self::Convergent => write!(f, "🎯 Convergent"),
            Self::Systematic => write!(f, "📋 Systematic"),
            Self::Creative => write!(f, "🎨 Creative"),
            Self::Divergent => write!(f, "🌳 Divergent"),
            Self::Exploratory => write!(f, "🧭 Exploratory"),
            Self::Focused => write!(f, "🔦 Focused"),
            Self::Diffuse => write!(f, "☁️ Diffuse"),
            Self::Peripheral => write!(f, "👁️ Peripheral"),
            Self::Intuitive => write!(f, "⚡ Intuitive"),
            Self::Deliberate => write!(f, "⚖️ Deliberate"),
            Self::Metacognitive => write!(f, "🪞 Metacognitive"),
        }
    }
}
