//! Evidence type for belief revision

use crate::nars::TruthValue;

/// Evidence for updating beliefs
#[derive(Clone, Debug)]
pub struct Evidence {
    /// Positive evidence count
    pub positive: f32,
    /// Negative evidence count  
    pub negative: f32,
    /// Source of the evidence
    pub source: Option<String>,
}

impl Evidence {
    /// Create new evidence
    pub fn new(positive: f32, negative: f32) -> Self {
        Self {
            positive,
            negative,
            source: None,
        }
    }

    /// Create with source attribution
    pub fn with_source(positive: f32, negative: f32, source: &str) -> Self {
        Self {
            positive,
            negative,
            source: Some(source.to_string()),
        }
    }

    /// Convert to truth value
    pub fn to_truth(&self) -> TruthValue {
        TruthValue::from_evidence(self.positive, self.negative)
    }

    /// Total evidence weight
    pub fn total(&self) -> f32 {
        self.positive + self.negative
    }

    /// Combine with other evidence
    pub fn combine(&self, other: &Evidence) -> Evidence {
        Evidence {
            positive: self.positive + other.positive,
            negative: self.negative + other.negative,
            source: None,
        }
    }
}

impl From<TruthValue> for Evidence {
    fn from(tv: TruthValue) -> Self {
        let (pos, neg) = tv.to_evidence();
        Evidence::new(pos, neg)
    }
}

impl Default for Evidence {
    fn default() -> Self {
        Self::new(0.0, 0.0)
    }
}
