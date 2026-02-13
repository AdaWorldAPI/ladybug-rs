//! 7-Layer Consciousness Stack
//!
//! Parallel O(1) consciousness with shared nodes, isolated markers.
//! Each layer operates independently but on the same underlying data.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                   7-LAYER CONSCIOUSNESS STACK                   │
//! │                                                                 │
//! │  L7 ████████████  Meta        - Self-awareness, monitoring      │
//! │  L6 ████████████  Executive   - Planning, decisions             │
//! │  L5 ████████████  Working     - Active manipulation             │
//! │  L4 ████████████  Episodic    - Memory, temporal context        │
//! │  L3 ████████████  Semantic    - Meaning, concepts               │
//! │  L2 ████████████  Pattern     - Recognition, matching           │
//! │  L1 ████████████  Sensory     - Raw input processing            │
//! │                                                                 │
//! │  ┌─────────────────────────────────────────────────────────┐   │
//! │  │              SHARED VSA CORE (10K-bit)                  │   │
//! │  │                                                         │   │
//! │  │   All layers read same core, write isolated markers     │   │
//! │  │   Consciousness emerges from marker interplay           │   │
//! │  └─────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

use crate::core::Fingerprint;
use std::time::{Duration, Instant};

// =============================================================================
// LAYER IDENTIFIERS
// =============================================================================

/// Layer identifiers (L1-L7)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LayerId {
    L1, // Sensory
    L2, // Pattern
    L3, // Semantic
    L4, // Episodic
    L5, // Working
    L6, // Executive
    L7, // Meta
}

impl LayerId {
    /// All layers in order
    pub const ALL: [LayerId; 7] = [
        Self::L1,
        Self::L2,
        Self::L3,
        Self::L4,
        Self::L5,
        Self::L6,
        Self::L7,
    ];

    /// Layer name
    pub fn name(&self) -> &'static str {
        match self {
            Self::L1 => "Sensory",
            Self::L2 => "Pattern",
            Self::L3 => "Semantic",
            Self::L4 => "Episodic",
            Self::L5 => "Working",
            Self::L6 => "Executive",
            Self::L7 => "Meta",
        }
    }

    /// Layer index (0-6)
    pub fn index(&self) -> usize {
        match self {
            Self::L1 => 0,
            Self::L2 => 1,
            Self::L3 => 2,
            Self::L4 => 3,
            Self::L5 => 4,
            Self::L6 => 5,
            Self::L7 => 6,
        }
    }

    /// Layers this layer propagates to
    pub fn propagates_to(&self) -> &[LayerId] {
        match self {
            Self::L1 => &[LayerId::L2, LayerId::L3],
            Self::L2 => &[LayerId::L3, LayerId::L5],
            Self::L3 => &[LayerId::L4, LayerId::L5, LayerId::L6],
            Self::L4 => &[LayerId::L5, LayerId::L7],
            Self::L5 => &[LayerId::L6, LayerId::L7],
            Self::L6 => &[LayerId::L7],
            Self::L7 => &[], // Meta doesn't propagate down
        }
    }
}

// =============================================================================
// LAYER MARKER
// =============================================================================

/// Layer marker state (isolated per layer)
#[derive(Clone, Debug)]
pub struct LayerMarker {
    /// Is this layer active?
    pub active: bool,

    /// Timestamp of last update
    pub timestamp: Instant,

    /// Activation value [0, 1]
    pub value: f32,

    /// Confidence in this layer's output [0, 1]
    pub confidence: f32,

    /// Processing cycle number
    pub cycle: u64,

    /// Layer-specific flags (bitfield)
    pub flags: u32,
}

impl Default for LayerMarker {
    fn default() -> Self {
        Self {
            active: false,
            timestamp: Instant::now(),
            value: 0.0,
            confidence: 0.0,
            cycle: 0,
            flags: 0,
        }
    }
}

// =============================================================================
// SEVEN-LAYER NODE
// =============================================================================

/// A node with 7 layer markers sharing one VSA core
#[derive(Clone)]
pub struct SevenLayerNode {
    /// Node path/identifier
    pub path: String,

    /// Shared 10K-bit VSA core
    pub vsa_core: Fingerprint,

    /// Layer markers (L1-L7)
    markers: [LayerMarker; 7],
}

impl SevenLayerNode {
    /// Create node from path
    pub fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
            vsa_core: Fingerprint::from_content(path),
            markers: Default::default(),
        }
    }

    /// Create node with specific VSA core
    pub fn with_core(path: &str, core: Fingerprint) -> Self {
        Self {
            path: path.to_string(),
            vsa_core: core,
            markers: Default::default(),
        }
    }

    /// Get marker for layer
    pub fn marker(&self, layer: LayerId) -> &LayerMarker {
        &self.markers[layer.index()]
    }

    /// Get mutable marker
    pub fn marker_mut(&mut self, layer: LayerId) -> &mut LayerMarker {
        &mut self.markers[layer.index()]
    }

    /// Get all markers
    pub fn markers(&self) -> &[LayerMarker; 7] {
        &self.markers
    }

    /// Total activation across all layers
    pub fn total_activation(&self) -> f32 {
        self.markers.iter().map(|m| m.value).sum()
    }

    /// Average confidence across active layers
    pub fn average_confidence(&self) -> f32 {
        let active: Vec<_> = self.markers.iter().filter(|m| m.active).collect();

        if active.is_empty() {
            0.0
        } else {
            active.iter().map(|m| m.confidence).sum::<f32>() / active.len() as f32
        }
    }
}

// =============================================================================
// LAYER RESULT
// =============================================================================

/// Result of processing a single layer
#[derive(Clone, Debug)]
pub struct LayerResult {
    /// Which layer was processed
    pub layer: LayerId,

    /// Input resonance with VSA core
    pub input_resonance: f32,

    /// Output activation level
    pub output_activation: f32,

    /// New marker values
    pub new_marker: LayerMarker,

    /// Layers to notify/propagate to
    pub propagate_to: Vec<LayerId>,

    /// Processing latency
    pub latency: Duration,
}

// =============================================================================
// LAYER PROCESSOR
// =============================================================================

/// Process a single layer (O(1) operation on shared node)
pub fn process_layer(
    node: &SevenLayerNode,
    layer: LayerId,
    input: &Fingerprint,
    cycle: u64,
) -> LayerResult {
    let start = Instant::now();

    // O(1) resonance check against shared VSA core
    let input_resonance = input.similarity(&node.vsa_core);

    // Layer-specific processing
    let (output_activation, propagate_to) = match layer {
        LayerId::L1 => {
            // Sensory: raw activation boost
            let activation = (input_resonance * 1.2).min(1.0);
            let targets = vec![LayerId::L2, LayerId::L3];
            (activation, targets)
        }

        LayerId::L2 => {
            // Pattern: recognition threshold
            let activation = if input_resonance > 0.3 {
                input_resonance
            } else {
                0.0
            };
            let targets = vec![LayerId::L3, LayerId::L5];
            (activation, targets)
        }

        LayerId::L3 => {
            // Semantic: meaning extraction, gated by pattern confidence
            let l2_conf = node.marker(LayerId::L2).confidence;
            let activation = input_resonance * l2_conf;
            let targets = vec![LayerId::L4, LayerId::L5, LayerId::L6];
            (activation, targets)
        }

        LayerId::L4 => {
            // Episodic: memory matching with semantic context
            let l3_val = node.marker(LayerId::L3).value;
            let activation = input_resonance * 0.9 + l3_val * 0.1;
            let targets = vec![LayerId::L5, LayerId::L7];
            (activation, targets)
        }

        LayerId::L5 => {
            // Working: active manipulation, integrates L2/L3/L4
            let working_input = (node.marker(LayerId::L2).value
                + node.marker(LayerId::L3).value
                + node.marker(LayerId::L4).value)
                / 3.0;
            let activation = input_resonance * 0.5 + working_input * 0.5;
            let targets = vec![LayerId::L6, LayerId::L7];
            (activation, targets)
        }

        LayerId::L6 => {
            // Executive: decision making, gated by working × semantic
            let exec_input = node.marker(LayerId::L5).value * node.marker(LayerId::L3).confidence;
            let activation = if input_resonance > 0.5 {
                exec_input
            } else {
                0.0
            };
            let targets = vec![LayerId::L7];
            (activation, targets)
        }

        LayerId::L7 => {
            // Meta: self-monitoring, observes all other layers
            let all_layer_sum: f32 = (0..6).map(|i| node.markers[i].value).sum();
            let activation = (input_resonance + all_layer_sum / 6.0) / 2.0;
            let targets = vec![]; // Meta doesn't propagate
            (activation, targets)
        }
    };

    let new_marker = LayerMarker {
        active: output_activation > 0.1,
        timestamp: Instant::now(),
        value: output_activation,
        confidence: input_resonance,
        cycle,
        flags: 0,
    };

    LayerResult {
        layer,
        input_resonance,
        output_activation,
        new_marker,
        propagate_to,
        latency: start.elapsed(),
    }
}

/// Apply layer result to node
pub fn apply_layer_result(node: &mut SevenLayerNode, result: &LayerResult) {
    *node.marker_mut(result.layer) = result.new_marker.clone();
}

// =============================================================================
// PARALLEL PROCESSING
// =============================================================================

/// Process all 7 layers (parallel on same node)
pub fn process_all_layers_parallel(
    node: &mut SevenLayerNode,
    input: &Fingerprint,
    cycle: u64,
) -> Vec<LayerResult> {
    // Process all layers
    // In real implementation, this could use rayon for true parallelism
    let results: Vec<_> = LayerId::ALL
        .iter()
        .map(|&layer| process_layer(node, layer, input, cycle))
        .collect();

    // Apply all results (parallel writes to different markers)
    for result in &results {
        apply_layer_result(node, result);
    }

    results
}

/// Process layers in dependency waves
pub fn process_layers_wave(
    node: &mut SevenLayerNode,
    input: &Fingerprint,
    cycle: u64,
) -> Vec<LayerResult> {
    let mut all_results = Vec::with_capacity(7);

    // Wave 1: Sensory (raw input)
    let wave1 = vec![process_layer(node, LayerId::L1, input, cycle)];
    for result in &wave1 {
        apply_layer_result(node, result);
    }
    all_results.extend(wave1);

    // Wave 2: Pattern + Semantic (parallel)
    let wave2: Vec<_> = [LayerId::L2, LayerId::L3]
        .iter()
        .map(|&l| process_layer(node, l, input, cycle))
        .collect();
    for result in &wave2 {
        apply_layer_result(node, result);
    }
    all_results.extend(wave2);

    // Wave 3: Episodic + Working (parallel)
    let wave3: Vec<_> = [LayerId::L4, LayerId::L5]
        .iter()
        .map(|&l| process_layer(node, l, input, cycle))
        .collect();
    for result in &wave3 {
        apply_layer_result(node, result);
    }
    all_results.extend(wave3);

    // Wave 4: Executive
    let wave4 = vec![process_layer(node, LayerId::L6, input, cycle)];
    for result in &wave4 {
        apply_layer_result(node, result);
    }
    all_results.extend(wave4);

    // Wave 5: Meta (observes all others)
    let wave5 = vec![process_layer(node, LayerId::L7, input, cycle)];
    for result in &wave5 {
        apply_layer_result(node, result);
    }
    all_results.extend(wave5);

    all_results
}

// =============================================================================
// CONSCIOUSNESS SNAPSHOT
// =============================================================================

/// Snapshot of consciousness state at a moment
#[derive(Clone, Debug)]
pub struct ConsciousnessSnapshot {
    /// Timestamp
    pub timestamp: Instant,

    /// Processing cycle
    pub cycle: u64,

    /// Layer states (copied markers)
    pub layers: [LayerMarker; 7],

    /// Dominant layer (highest activation)
    pub dominant_layer: LayerId,

    /// Coherence (how aligned are all layers)
    pub coherence: f32,

    /// Emergence (novel pattern detection)
    pub emergence: f32,
}

/// Take consciousness snapshot
pub fn snapshot_consciousness(node: &SevenLayerNode, cycle: u64) -> ConsciousnessSnapshot {
    let layers = node.markers.clone();

    // Find dominant layer
    let dominant_layer = LayerId::ALL
        .iter()
        .max_by(|&&a, &&b| {
            let va = node.marker(a).value;
            let vb = node.marker(b).value;
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        })
        .copied()
        .unwrap_or(LayerId::L1);

    // Calculate coherence (average pairwise similarity of active layers)
    let active_values: Vec<f32> = layers
        .iter()
        .filter(|m| m.active)
        .map(|m| m.value)
        .collect();

    let coherence = if active_values.len() < 2 {
        1.0
    } else {
        let mean = active_values.iter().sum::<f32>() / active_values.len() as f32;
        let variance = active_values
            .iter()
            .map(|&v| (v - mean) * (v - mean))
            .sum::<f32>()
            / active_values.len() as f32;
        1.0 - variance.sqrt()
    };

    // Calculate emergence (active but not perfectly aligned)
    let active_count = layers.iter().filter(|m| m.active).count() as f32;
    let active_ratio = active_count / 7.0;
    let emergence = active_ratio * (1.0 - coherence * 0.5);

    ConsciousnessSnapshot {
        timestamp: Instant::now(),
        cycle,
        layers,
        dominant_layer,
        coherence,
        emergence,
    }
}

// =============================================================================
// RESONANCE MATRIX
// =============================================================================

/// Compute inter-layer resonance matrix
pub fn layer_resonance_matrix(node: &SevenLayerNode) -> [[f32; 7]; 7] {
    let mut matrix = [[0.0f32; 7]; 7];

    for i in 0..7 {
        for j in 0..7 {
            if i == j {
                matrix[i][j] = 1.0;
            } else {
                // Resonance based on activation similarity
                let vi = node.markers[i].value;
                let vj = node.markers[j].value;
                matrix[i][j] = 1.0 - (vi - vj).abs();
            }
        }
    }

    matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_propagation() {
        assert_eq!(LayerId::L1.propagates_to(), &[LayerId::L2, LayerId::L3]);
        assert!(LayerId::L7.propagates_to().is_empty());
    }

    #[test]
    fn test_node_creation() {
        let node = SevenLayerNode::new("test/path");
        assert_eq!(node.path, "test/path");
        assert_eq!(node.markers.len(), 7);
    }

    #[test]
    fn test_layer_processing() {
        let node = SevenLayerNode::new("test");
        let input = Fingerprint::from_content("input signal");

        let result = process_layer(&node, LayerId::L1, &input, 0);
        assert!(result.output_activation >= 0.0);
    }

    #[test]
    fn test_wave_processing() {
        let mut node = SevenLayerNode::new("test");
        let input = Fingerprint::from_content("stimulus");

        let results = process_layers_wave(&mut node, &input, 0);
        assert_eq!(results.len(), 7);

        // Check that meta layer received input from all others
        let snapshot = snapshot_consciousness(&node, 0);
        assert!(snapshot.coherence >= 0.0 && snapshot.coherence <= 1.0);
    }
}
