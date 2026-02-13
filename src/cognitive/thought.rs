//! Thought and related cognitive types

use crate::cognitive::ThinkingStyle;
use crate::core::Fingerprint;
use crate::nars::TruthValue;

/// A thought - the atomic unit of cognition
#[derive(Clone, Debug)]
pub struct Thought {
    pub id: String,
    pub content: String,
    pub fingerprint: Fingerprint,
    pub truth: TruthValue,
    pub style: ThinkingStyle,
    pub qidx: u8,
}

impl Thought {
    pub fn new(content: &str) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            content: content.to_string(),
            fingerprint: Fingerprint::from_content(content),
            truth: TruthValue::unknown(),
            style: ThinkingStyle::default(),
            qidx: 128,
        }
    }

    pub fn with_truth(mut self, truth: TruthValue) -> Self {
        self.truth = truth;
        self
    }

    pub fn with_style(mut self, style: ThinkingStyle) -> Self {
        self.style = style;
        self
    }
}

/// Abstract concept
#[derive(Clone, Debug)]
pub struct Concept {
    pub id: String,
    pub name: String,
    pub fingerprint: Fingerprint,
    pub abstraction_level: u8,
}

/// A belief
#[derive(Clone, Debug)]
pub struct Belief {
    pub thought: Thought,
    pub source: Option<String>,
}
