//! Graph edges with amplification

#[derive(Clone, Debug)]
pub struct Edge {
    pub from_id: String,
    pub to_id: String,
    pub edge_type: EdgeType,
    pub weight: f32,
    pub amplification: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum EdgeType {
    Causes,
    Supports,
    Contradicts,
    Relates,
    Custom(String),
}

impl Edge {
    pub fn causes(from: &str, to: &str) -> Self {
        Self {
            from_id: from.to_string(),
            to_id: to.to_string(),
            edge_type: EdgeType::Causes,
            weight: 1.0,
            amplification: 1.0,
        }
    }

    pub fn with_amplification(mut self, amp: f32) -> Self {
        self.amplification = amp;
        self
    }
}
