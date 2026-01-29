//! Thinking styles

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum ThinkingStyle {
    Analytical,
    Creative,
    Focused,
    Diffuse,
    Intuitive,
    #[default]
    Deliberate,
}

impl ThinkingStyle {
    pub fn fan_out(&self) -> usize {
        match self {
            Self::Analytical => 3,
            Self::Creative => 10,
            Self::Focused => 1,
            Self::Diffuse => 7,
            Self::Intuitive => 2,
            Self::Deliberate => 5,
        }
    }
    
    pub fn confidence_threshold(&self) -> f32 {
        match self {
            Self::Analytical => 0.8,
            Self::Creative => 0.3,
            Self::Focused => 0.9,
            Self::Diffuse => 0.4,
            Self::Intuitive => 0.5,
            Self::Deliberate => 0.7,
        }
    }
}
