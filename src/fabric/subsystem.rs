//! Subsystems that participate in the cognitive fabric

/// Subsystems that cross-pollinate via mRNA
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Subsystem {
    /// Query planning and execution
    Query,

    /// Data compression and encoding
    Compression,

    /// Reinforcement learning
    Learning,

    /// NARS inference
    Inference,

    /// Thinking style management
    Style,
}

impl Subsystem {
    /// All subsystems
    pub const ALL: [Subsystem; 5] = [
        Self::Query,
        Self::Compression,
        Self::Learning,
        Self::Inference,
        Self::Style,
    ];

    /// Default priority (higher = pollinates first)
    pub fn priority(&self) -> u8 {
        match self {
            Self::Style => 5,       // Metacognitive observes all
            Self::Inference => 4,   // Reasoning shapes understanding
            Self::Query => 3,       // Queries drive activity
            Self::Learning => 2,    // Learning adapts
            Self::Compression => 1, // Compression reacts
        }
    }

    /// Emoji for display
    pub fn emoji(&self) -> &'static str {
        match self {
            Self::Query => "🔍",
            Self::Compression => "📦",
            Self::Learning => "🧠",
            Self::Inference => "💭",
            Self::Style => "🎨",
        }
    }
}

impl std::fmt::Display for Subsystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {:?}", self.emoji(), self)
    }
}
