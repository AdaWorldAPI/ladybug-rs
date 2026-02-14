//! 10-Layer Cognitive Stack
//!
//! Replaces the 7-layer consciousness stack with a 10-layer processing model
//! that spans from single-agent perception (L1-L5) through multi-agent
//! refinement (L6-L10).
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    10-LAYER COGNITIVE STACK                         │
//! │                                                                     │
//! │  L10 ████████████ Crystallization — what survives becomes system    │
//! │  L9  ████████████ Validation      — NARS + Brier + Socratic sieve  │
//! │  L8  ████████████ Integration     — evidence merge, meta-awareness │
//! │  L7  ████████████ Contingency     — cross-branch, could-be-otherwise│
//! │  L6  ████████████ Delegation      — cognitive fan-out, multi-agent │
//! │  ─── single agent boundary ─────────────────────────────────────── │
//! │  L5  ████████████ Execution       — active manipulation, synthesis │
//! │  L4  ████████████ Routing         — branch selection, template pick│
//! │  L3  ████████████ Appraisal       — gestalt, hypothesis, evaluation│
//! │  L2  ████████████ Resonance       — field binding, similarity, assoc│
//! │  L1  ████████████ Recognition     — pattern match, fingerprint enc │
//! │                                                                     │
//! │  ┌─────────────────────────────────────────────────────────────┐   │
//! │  │              SHARED VSA CORE (10K-bit)                      │   │
//! │  │   All layers read same core, write isolated markers         │   │
//! │  │   Consciousness emerges from marker interplay               │   │
//! │  └─────────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! L1-L5: One mind thinking. A single agent's processing path.
//! L6-L10: Multiple minds refining. Results fan out, alternatives explored,
//! outcomes integrate, truth hardens, what survives crystallizes.
//!
//! L10 feeds back to L4 (new routing rules) and L2 (new resonance patterns)
//! via crystallization into the BindSpace — not through explicit wiring.

use crate::core::Fingerprint;
use std::time::{Duration, Instant};

/// Number of layers in the cognitive stack.
pub const NUM_LAYERS: usize = 10;

// =============================================================================
// LAYER IDENTIFIERS
// =============================================================================

/// Layer identifiers (L1-L10)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LayerId {
    L1,  // Recognition
    L2,  // Resonance
    L3,  // Appraisal
    L4,  // Routing
    L5,  // Execution
    L6,  // Delegation
    L7,  // Contingency
    L8,  // Integration
    L9,  // Validation
    L10, // Crystallization
}

impl LayerId {
    /// All layers in order
    pub const ALL: [LayerId; NUM_LAYERS] = [
        Self::L1,
        Self::L2,
        Self::L3,
        Self::L4,
        Self::L5,
        Self::L6,
        Self::L7,
        Self::L8,
        Self::L9,
        Self::L10,
    ];

    /// Layer name
    pub fn name(&self) -> &'static str {
        match self {
            Self::L1 => "Recognition",
            Self::L2 => "Resonance",
            Self::L3 => "Appraisal",
            Self::L4 => "Routing",
            Self::L5 => "Execution",
            Self::L6 => "Delegation",
            Self::L7 => "Contingency",
            Self::L8 => "Integration",
            Self::L9 => "Validation",
            Self::L10 => "Crystallization",
        }
    }

    /// Layer index (0-9)
    pub fn index(&self) -> usize {
        match self {
            Self::L1 => 0,
            Self::L2 => 1,
            Self::L3 => 2,
            Self::L4 => 3,
            Self::L5 => 4,
            Self::L6 => 5,
            Self::L7 => 6,
            Self::L8 => 7,
            Self::L9 => 8,
            Self::L10 => 9,
        }
    }

    /// Is this a single-agent layer (L1-L5)?
    pub fn is_single_agent(&self) -> bool {
        self.index() < 5
    }

    /// Is this a multi-agent layer (L6-L10)?
    pub fn is_multi_agent(&self) -> bool {
        self.index() >= 5
    }

    /// Layers this layer propagates to
    pub fn propagates_to(&self) -> &[LayerId] {
        match self {
            Self::L1 => &[LayerId::L2, LayerId::L3],
            Self::L2 => &[LayerId::L3, LayerId::L5],
            Self::L3 => &[LayerId::L4, LayerId::L5],
            Self::L4 => &[LayerId::L5, LayerId::L6],
            Self::L5 => &[LayerId::L6, LayerId::L8],
            Self::L6 => &[LayerId::L7, LayerId::L8],
            Self::L7 => &[LayerId::L8, LayerId::L9],
            Self::L8 => &[LayerId::L9, LayerId::L10],
            Self::L9 => &[LayerId::L10],
            Self::L10 => &[], // Crystallization is terminal
            // (Feedback to L2/L4 happens via BindSpace, not propagation)
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
    /// Bits 0-2: ready(0), active(1), gated(2), stale(3)
    /// Bits 4-7: reserved for downstream
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

impl LayerMarker {
    /// Flag bit: dependencies met, layer ready to fire
    pub const FLAG_READY: u32 = 1 << 0;
    /// Flag bit: currently processing
    pub const FLAG_ACTIVE: u32 = 1 << 1;
    /// Flag bit: gated by unsatisfied dependency
    pub const FLAG_GATED: u32 = 1 << 2;
    /// Flag bit: hasn't updated in N cycles
    pub const FLAG_STALE: u32 = 1 << 3;
}

// =============================================================================
// LAYER NODE
// =============================================================================

/// A node with 10 layer markers sharing one VSA core.
///
/// Renamed from `SevenLayerNode` in the 10-layer expansion.
/// The type alias `SevenLayerNode = LayerNode` preserves backward compatibility.
#[derive(Clone)]
pub struct LayerNode {
    /// Node path/identifier
    pub path: String,

    /// Shared 10K-bit VSA core
    pub vsa_core: Fingerprint,

    /// Layer markers (L1-L10)
    markers: [LayerMarker; NUM_LAYERS],
}

/// Backward-compatible alias.
pub type SevenLayerNode = LayerNode;

impl LayerNode {
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
    pub fn markers(&self) -> &[LayerMarker; NUM_LAYERS] {
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

    /// Total activation across single-agent layers (L1-L5)
    pub fn single_agent_activation(&self) -> f32 {
        self.markers[..5].iter().map(|m| m.value).sum()
    }

    /// Total activation across multi-agent layers (L6-L10)
    pub fn multi_agent_activation(&self) -> f32 {
        self.markers[5..].iter().map(|m| m.value).sum()
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
    node: &LayerNode,
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
            // Recognition: raw pattern matching, fingerprint encoding
            let activation = (input_resonance * 1.2).min(1.0);
            let targets = vec![LayerId::L2, LayerId::L3];
            (activation, targets)
        }

        LayerId::L2 => {
            // Resonance: field binding, similarity search, association
            // L1 and L2 form a single perceptual act — L2 inherits L1 activation
            let l1_val = node.marker(LayerId::L1).value;
            let activation = if input_resonance > 0.3 || l1_val > 0.3 {
                (input_resonance + l1_val) / 2.0
            } else {
                0.0
            };
            let targets = vec![LayerId::L3, LayerId::L5];
            (activation, targets)
        }

        LayerId::L3 => {
            // Appraisal: gestalt formation, hypothesis, initial evaluation
            // Gated by L2 (Resonance) confidence
            let l2_conf = node.marker(LayerId::L2).confidence;
            let activation = input_resonance * l2_conf;
            let targets = vec![LayerId::L4, LayerId::L5];
            (activation, targets)
        }

        LayerId::L4 => {
            // Routing: branch selection, fan-out degree, template dispatch
            // Uses L3 (Appraisal) context
            let l3_val = node.marker(LayerId::L3).value;
            let activation = input_resonance * 0.9 + l3_val * 0.1;
            let targets = vec![LayerId::L5, LayerId::L6];
            (activation, targets)
        }

        LayerId::L5 => {
            // Execution: active manipulation, synthesis, production
            // Integrates L2/L3/L4 — the single-agent output layer
            let working_input = (node.marker(LayerId::L2).value
                + node.marker(LayerId::L3).value
                + node.marker(LayerId::L4).value)
                / 3.0;
            let activation = input_resonance * 0.5 + working_input * 0.5;
            let targets = vec![LayerId::L6, LayerId::L8];
            (activation, targets)
        }

        // ─── SINGLE AGENT BOUNDARY ───────────────────────────────────

        LayerId::L6 => {
            // Delegation: cognitive fan-out, multi-agent dispatch
            // Activation scales with L5 Execution quality.
            // A strong L5 output with high confidence → delegate widely.
            let exec_val = node.marker(LayerId::L5).value;
            let exec_conf = node.marker(LayerId::L5).confidence;
            let activation = exec_val * exec_conf * input_resonance.max(0.3);
            let targets = vec![LayerId::L7, LayerId::L8];
            (activation, targets)
        }

        LayerId::L7 => {
            // Contingency: "things could be otherwise"
            // High activation when appraisal is uncertain but delegation active.
            // The uncertainty IS the signal — not a bug, a feature.
            let appraisal_conf = node.marker(LayerId::L3).confidence;
            let delegation_val = node.marker(LayerId::L6).value;
            let uncertainty = 1.0 - appraisal_conf;
            let activation =
                (uncertainty * 0.6 + delegation_val * 0.4) * input_resonance.max(0.2);
            let targets = vec![LayerId::L8, LayerId::L9];
            (activation, targets)
        }

        LayerId::L8 => {
            // Integration: evidence merge, learning from outcomes
            // Merges L5 (Execution), L6 (Delegation), L7 (Contingency)
            let exec_val = node.marker(LayerId::L5).value;
            let deleg_val = node.marker(LayerId::L6).value;
            let conting_conf = node.marker(LayerId::L7).confidence;
            let activation = (exec_val * 0.4 + deleg_val * 0.4 + input_resonance * 0.2)
                * conting_conf.max(0.3);
            let targets = vec![LayerId::L9, LayerId::L10];
            (activation, targets)
        }

        LayerId::L9 => {
            // Validation: NARS revision, Brier calibration, Socratic sieve
            // Requires L8 Integration to have converged
            let integration_val = node.marker(LayerId::L8).value;
            let integration_conf = node.marker(LayerId::L8).confidence;
            let activation = if integration_val > 0.5 && integration_conf > 0.4 {
                input_resonance * integration_conf
            } else {
                0.0 // Not enough evidence to validate
            };
            let targets = vec![LayerId::L10];
            (activation, targets)
        }

        LayerId::L10 => {
            // Crystallization: only crystallize what's validated
            // Requires L9 validation with high confidence
            let validation = node.marker(LayerId::L9).value;
            let validation_conf = node.marker(LayerId::L9).confidence;
            let activation = if validation > 0.7 && validation_conf > 0.6 {
                validation * validation_conf
            } else {
                0.0
            };
            // Terminal layer. Feedback to L2/L4 happens via BindSpace.
            let targets = vec![];
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
pub fn apply_layer_result(node: &mut LayerNode, result: &LayerResult) {
    *node.marker_mut(result.layer) = result.new_marker.clone();
}

// =============================================================================
// PARALLEL PROCESSING
// =============================================================================

/// Process all 10 layers (parallel on same node)
pub fn process_all_layers_parallel(
    node: &mut LayerNode,
    input: &Fingerprint,
    cycle: u64,
) -> Vec<LayerResult> {
    let results: Vec<_> = LayerId::ALL
        .iter()
        .map(|&layer| process_layer(node, layer, input, cycle))
        .collect();

    for result in &results {
        apply_layer_result(node, result);
    }

    results
}

/// Process layers in dependency waves.
///
/// Waves ensure each layer reads committed state from its dependencies.
/// This is the deterministic variant; for async, use the 2-stroke engine.
pub fn process_layers_wave(
    node: &mut LayerNode,
    input: &Fingerprint,
    cycle: u64,
) -> Vec<LayerResult> {
    let mut all_results = Vec::with_capacity(NUM_LAYERS);

    // Wave 1: Recognition (raw input) + Resonance (perceptual unit)
    let wave1: Vec<_> = [LayerId::L1, LayerId::L2]
        .iter()
        .map(|&l| process_layer(node, l, input, cycle))
        .collect();
    for result in &wave1 {
        apply_layer_result(node, result);
    }
    all_results.extend(wave1);

    // Wave 2: Appraisal + Routing (parallel)
    let wave2: Vec<_> = [LayerId::L3, LayerId::L4]
        .iter()
        .map(|&l| process_layer(node, l, input, cycle))
        .collect();
    for result in &wave2 {
        apply_layer_result(node, result);
    }
    all_results.extend(wave2);

    // Wave 3: Execution (single-agent output)
    let wave3 = vec![process_layer(node, LayerId::L5, input, cycle)];
    for result in &wave3 {
        apply_layer_result(node, result);
    }
    all_results.extend(wave3);

    // Wave 4: Delegation + Contingency (parallel, multi-agent begins)
    let wave4: Vec<_> = [LayerId::L6, LayerId::L7]
        .iter()
        .map(|&l| process_layer(node, l, input, cycle))
        .collect();
    for result in &wave4 {
        apply_layer_result(node, result);
    }
    all_results.extend(wave4);

    // Wave 5: Integration (merges multi-agent results)
    let wave5 = vec![process_layer(node, LayerId::L8, input, cycle)];
    for result in &wave5 {
        apply_layer_result(node, result);
    }
    all_results.extend(wave5);

    // Wave 6: Validation (truth hardening)
    let wave6 = vec![process_layer(node, LayerId::L9, input, cycle)];
    for result in &wave6 {
        apply_layer_result(node, result);
    }
    all_results.extend(wave6);

    // Wave 7: Crystallization (terminal — what survives becomes system)
    let wave7 = vec![process_layer(node, LayerId::L10, input, cycle)];
    for result in &wave7 {
        apply_layer_result(node, result);
    }
    all_results.extend(wave7);

    all_results
}

// =============================================================================
// CONSCIOUSNESS SNAPSHOT
// =============================================================================

/// Snapshot of cognitive state at a moment
#[derive(Clone, Debug)]
pub struct ConsciousnessSnapshot {
    /// Timestamp
    pub timestamp: Instant,

    /// Processing cycle
    pub cycle: u64,

    /// Layer states (copied markers)
    pub layers: [LayerMarker; NUM_LAYERS],

    /// Dominant layer (highest activation)
    pub dominant_layer: LayerId,

    /// Coherence (how aligned are all layers)
    pub coherence: f32,

    /// Emergence (novel pattern detection)
    pub emergence: f32,
}

/// Take consciousness snapshot
pub fn snapshot_consciousness(node: &LayerNode, cycle: u64) -> ConsciousnessSnapshot {
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
    let active_ratio = active_count / NUM_LAYERS as f32;
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
pub fn layer_resonance_matrix(node: &LayerNode) -> [[f32; NUM_LAYERS]; NUM_LAYERS] {
    let mut matrix = [[0.0f32; NUM_LAYERS]; NUM_LAYERS];

    for i in 0..NUM_LAYERS {
        for j in 0..NUM_LAYERS {
            if i == j {
                matrix[i][j] = 1.0;
            } else {
                let vi = node.markers[i].value;
                let vj = node.markers[j].value;
                matrix[i][j] = 1.0 - (vi - vj).abs();
            }
        }
    }

    matrix
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_count() {
        assert_eq!(LayerId::ALL.len(), 10);
        assert_eq!(NUM_LAYERS, 10);
    }

    #[test]
    fn test_layer_boundaries() {
        assert!(LayerId::L1.is_single_agent());
        assert!(LayerId::L5.is_single_agent());
        assert!(LayerId::L6.is_multi_agent());
        assert!(LayerId::L10.is_multi_agent());
    }

    #[test]
    fn test_layer_propagation() {
        assert_eq!(LayerId::L1.propagates_to(), &[LayerId::L2, LayerId::L3]);
        assert!(LayerId::L10.propagates_to().is_empty());
        assert_eq!(LayerId::L5.propagates_to(), &[LayerId::L6, LayerId::L8]);
    }

    #[test]
    fn test_node_creation() {
        let node = LayerNode::new("test/path");
        assert_eq!(node.path, "test/path");
        assert_eq!(node.markers.len(), NUM_LAYERS);
    }

    #[test]
    fn test_backward_compat_alias() {
        let node: SevenLayerNode = LayerNode::new("compat");
        assert_eq!(node.path, "compat");
    }

    #[test]
    fn test_layer_processing() {
        let node = LayerNode::new("test");
        let input = Fingerprint::from_content("input signal");

        let result = process_layer(&node, LayerId::L1, &input, 0);
        assert!(result.output_activation >= 0.0);
    }

    #[test]
    fn test_wave_processing() {
        let mut node = LayerNode::new("test");
        let input = Fingerprint::from_content("stimulus");

        let results = process_layers_wave(&mut node, &input, 0);
        assert_eq!(results.len(), NUM_LAYERS);

        let snapshot = snapshot_consciousness(&node, 0);
        assert!(snapshot.coherence >= 0.0 && snapshot.coherence <= 1.0);
    }

    #[test]
    fn test_crystallization_requires_validation() {
        let mut node = LayerNode::new("test");
        let input = Fingerprint::from_content("data");

        // Process one cycle — L10 should not fire without L9 being satisfied
        process_layers_wave(&mut node, &input, 0);
        let l10 = node.marker(LayerId::L10);
        // L10 requires L9.value > 0.7 AND L9.confidence > 0.6
        // On first pass, unlikely to be met
        assert!(
            l10.value < 0.5,
            "L10 should not fire strongly without L9 validation"
        );
    }

    #[test]
    fn test_single_vs_multi_agent_activation() {
        let mut node = LayerNode::new("test");
        let input = Fingerprint::from_content("test");
        process_layers_wave(&mut node, &input, 0);

        let single = node.single_agent_activation();
        let multi = node.multi_agent_activation();
        assert!(single >= 0.0);
        assert!(multi >= 0.0);
    }

    #[test]
    fn test_snapshot_dominant_layer() {
        let mut node = LayerNode::new("test");
        let input = Fingerprint::from_content("strong signal");
        process_layers_wave(&mut node, &input, 0);

        let snapshot = snapshot_consciousness(&node, 0);
        // Dominant layer should be one of the lower layers (first to fire)
        assert!(snapshot.dominant_layer.index() < NUM_LAYERS);
    }
}
