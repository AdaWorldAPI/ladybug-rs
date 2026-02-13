//! Graph traversal

use std::ops::RangeInclusive;

pub struct Traversal {
    pub start_id: String,
    pub edge_type: Option<String>,
    pub depth: RangeInclusive<usize>,
    pub amplification_threshold: Option<f32>,
}

impl Traversal {
    pub fn from(start_id: &str) -> Self {
        Self {
            start_id: start_id.to_string(),
            edge_type: None,
            depth: 1..=10,
            amplification_threshold: None,
        }
    }

    pub fn causes(mut self) -> Self {
        self.edge_type = Some("CAUSES".to_string());
        self
    }

    pub fn depth(mut self, range: RangeInclusive<usize>) -> Self {
        self.depth = range;
        self
    }

    pub fn amplifies(mut self, threshold: f32) -> Self {
        self.amplification_threshold = Some(threshold);
        self
    }
}
