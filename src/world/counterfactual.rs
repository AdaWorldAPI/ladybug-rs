//! Counterfactual reasoning

pub struct Counterfactual {
    pub baseline_version: u64,
    pub hypothesis_version: u64,
    pub affected_nodes: Vec<String>,
}

#[derive(Clone, Debug)]
pub enum Change {
    Remove(String),
    UpdateTruth { id: String, frequency: f32, confidence: f32 },
    AddEdge { from: String, to: String, edge_type: String },
}
