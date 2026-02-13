//! Grammar-Aware Cognitive Engine
//!
//! Unified integration of:
//! - 4 QuadTriangles (10K-bit VSA corners)
//! - 7-Layer Consciousness Stack
//! - 12 Thinking Styles
//! - Collapse Gate (SIMD SD)
//! - mRNA Cross-Pollination
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    GRAMMAR-AWARE COGNITIVE ENGINE                       │
//! │                                                                         │
//! │   INPUT ─────────────────────────────────────────────────────────────►  │
//! │      │                                                                  │
//! │      ▼                                                                  │
//! │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │
//! │   │   GRAMMAR   │───►│  QUAD-TRI   │───►│   7-LAYER   │                │
//! │   │   PARSER    │    │  RESONANCE  │    │    STACK    │                │
//! │   └─────────────┘    └─────────────┘    └─────────────┘                │
//! │          │                  │                  │                        │
//! │          ▼                  ▼                  ▼                        │
//! │   ┌─────────────────────────────────────────────────────┐              │
//! │   │              mRNA CROSS-POLLINATION                 │              │
//! │   │                                                     │              │
//! │   │   Grammar ←──► Thinking ←──► Memory ←──► Action    │              │
//! │   └─────────────────────────────────────────────────────┘              │
//! │                           │                                            │
//! │                           ▼                                            │
//! │   ┌─────────────────────────────────────────────────────┐              │
//! │   │              COLLAPSE GATE (SD)                     │              │
//! │   │                                                     │              │
//! │   │   FLOW ◄────────┬────────┬────────► BLOCK          │              │
//! │   │   (commit)      │  HOLD  │         (clarify)       │              │
//! │   └─────────────────────────────────────────────────────┘              │
//! │                           │                                            │
//! │                           ▼                                            │
//! │   OUTPUT ◄─────────────────────────────────────────────────────────    │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```

use crate::cognitive::{
    CognitiveProfiles, CollapseDecision, ConsciousnessSnapshot, LayerId, QuadTriangle,
    SevenLayerNode, ThinkingStyle, evaluate_gate, process_layers_wave, snapshot_consciousness,
};
use crate::core::{Fingerprint, VsaOps};
use crate::fabric::{Butterfly, ButterflyDetector, MRNA, Subsystem};
use std::time::Instant;

// =============================================================================
// GRAMMAR TRIANGLE (Input from parser)
// =============================================================================

/// Grammar role (from parser)
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GrammarRole {
    Subject,
    Predicate,
    Object,
    IndirectObject,
    Modifier,
    Determiner,
    Complement,
    Adjunct,
}

/// Grammar triangle from parser
#[derive(Clone, Debug)]
pub struct GrammarTriangle {
    pub role: GrammarRole,
    pub filler: String,
    pub frame_type: String,
    pub confidence: f32,
    pub fingerprint: Fingerprint,
}

impl GrammarTriangle {
    pub fn new(role: GrammarRole, filler: &str, frame_type: &str, confidence: f32) -> Self {
        let content = format!("{}:{}:{}", role_name(&role), filler, frame_type);
        Self {
            role,
            filler: filler.to_string(),
            frame_type: frame_type.to_string(),
            confidence,
            fingerprint: Fingerprint::from_content(&content),
        }
    }
}

fn role_name(role: &GrammarRole) -> &'static str {
    match role {
        GrammarRole::Subject => "SUBJ",
        GrammarRole::Predicate => "PRED",
        GrammarRole::Object => "OBJ",
        GrammarRole::IndirectObject => "IOBJ",
        GrammarRole::Modifier => "MOD",
        GrammarRole::Determiner => "DET",
        GrammarRole::Complement => "COMP",
        GrammarRole::Adjunct => "ADJ",
    }
}

// =============================================================================
// COGNITIVE STATE
// =============================================================================

/// Current cognitive state
#[derive(Clone)]
pub struct CognitiveState {
    /// Active thinking style
    pub style: ThinkingStyle,

    /// Quad-triangle cognitive texture
    pub quad_triangle: QuadTriangle,

    /// 7-layer consciousness node
    pub consciousness: SevenLayerNode,

    /// Processing cycle
    pub cycle: u64,

    /// Last snapshot
    pub last_snapshot: Option<ConsciousnessSnapshot>,
}

impl Default for CognitiveState {
    fn default() -> Self {
        Self {
            style: ThinkingStyle::Analytical,
            quad_triangle: QuadTriangle::neutral(),
            consciousness: SevenLayerNode::new("cognitive_state"),
            cycle: 0,
            last_snapshot: None,
        }
    }
}

// =============================================================================
// GRAMMAR-AWARE ENGINE
// =============================================================================

/// Grammar-aware cognitive engine
pub struct GrammarCognitiveEngine {
    /// Current cognitive state
    state: CognitiveState,

    /// mRNA cross-pollination substrate
    mrna: MRNA,

    /// Butterfly detector
    butterfly: ButterflyDetector,

    /// Accumulated grammar triangles
    grammar_buffer: Vec<GrammarTriangle>,

    /// Collapse history
    collapse_history: Vec<CollapseDecision>,
}

impl GrammarCognitiveEngine {
    /// Create new engine
    pub fn new() -> Self {
        Self {
            state: CognitiveState::default(),
            mrna: MRNA::new(),
            butterfly: ButterflyDetector::new(),
            grammar_buffer: Vec::new(),
            collapse_history: Vec::new(),
        }
    }

    /// Create with specific thinking style
    pub fn with_style(style: ThinkingStyle) -> Self {
        let mut engine = Self::new();
        engine.set_style(style);
        engine
    }

    /// Set thinking style (modulates all processing)
    pub fn set_style(&mut self, style: ThinkingStyle) {
        self.state.style = style;
        self.mrna.set_style(style);
        self.butterfly.set_style(style);

        // Nudge quad-triangle toward style-appropriate profile
        let target = style_to_quad_triangle(style);
        self.state.quad_triangle.nudge_toward(&target, 0.3);
    }

    /// Get current style
    pub fn style(&self) -> ThinkingStyle {
        self.state.style
    }

    /// Process grammar triangles from parser
    pub fn ingest_grammar(&mut self, triangles: Vec<GrammarTriangle>) -> IngestResult {
        let start = Instant::now();

        // Bundle all grammar fingerprints
        let fps: Vec<Fingerprint> = triangles.iter().map(|t| t.fingerprint.clone()).collect();

        let grammar_fp = if fps.is_empty() {
            Fingerprint::zero()
        } else {
            Fingerprint::bundle(&fps)
        };

        // Cross-pollinate with mRNA
        let resonances = self.mrna.pollinate_from(Subsystem::Query, &grammar_fp);

        // Check for butterfly effects
        // Note: Butterfly detection requires history access - currently disabled
        // pending architectural review to expose history from MRNA
        let butterfly: Option<Butterfly> = None;
        let _ = resonances.len(); // suppress warning

        // Process through 7-layer stack
        self.state.cycle += 1;
        let _layer_results =
            process_layers_wave(&mut self.state.consciousness, &grammar_fp, self.state.cycle);

        // Take consciousness snapshot
        let snapshot = snapshot_consciousness(&self.state.consciousness, self.state.cycle);
        self.state.last_snapshot = Some(snapshot.clone());

        // Update quad-triangle based on grammar coherence
        let grammar_coherence =
            triangles.iter().map(|t| t.confidence).sum::<f32>() / triangles.len().max(1) as f32;

        self.update_quad_triangle_from_grammar(grammar_coherence);

        // Store in buffer
        self.grammar_buffer.extend(triangles);

        IngestResult {
            resonance_count: resonances.len(),
            butterfly,
            snapshot,
            processing_time: start.elapsed(),
            grammar_coherence,
        }
    }

    /// Evaluate collapse for current candidates
    pub fn evaluate_collapse(&mut self, candidate_scores: &[f32]) -> CollapseDecision {
        // Get style-modulated thresholds
        let _modulation = self.state.style.field_modulation();

        // Evaluate gate
        let decision = evaluate_gate(candidate_scores, true);

        // Store in history
        self.collapse_history.push(decision.clone());

        // Cross-pollinate collapse decision
        if decision.can_collapse {
            if let Some(winner) = decision.winner_index {
                let collapse_fp = Fingerprint::from_content(&format!("collapse:{}", winner));
                self.mrna.pollinate_from(Subsystem::Learning, &collapse_fp);
            }
        }

        decision
    }

    /// Get quad-triangle resonance with query
    pub fn quad_resonance(&self, query: &Fingerprint) -> f32 {
        self.state.quad_triangle.query_resonance(query)
    }

    /// Get cognitive signature
    pub fn signature(&self) -> String {
        self.state.quad_triangle.signature()
    }

    /// Get flow count (how many triangles in flow state)
    pub fn flow_count(&self) -> usize {
        self.state.quad_triangle.flow_count()
    }

    /// Check if in global flow
    pub fn is_global_flow(&self) -> bool {
        self.state.quad_triangle.is_global_flow()
    }

    /// Get consciousness coherence
    pub fn coherence(&self) -> f32 {
        self.state
            .last_snapshot
            .as_ref()
            .map(|s| s.coherence)
            .unwrap_or(0.0)
    }

    /// Get emergence level
    pub fn emergence(&self) -> f32 {
        self.state
            .last_snapshot
            .as_ref()
            .map(|s| s.emergence)
            .unwrap_or(0.0)
    }

    /// Get dominant consciousness layer
    pub fn dominant_layer(&self) -> LayerId {
        self.state
            .last_snapshot
            .as_ref()
            .map(|s| s.dominant_layer)
            .unwrap_or(LayerId::L1)
    }

    /// Get mRNA superposition
    pub fn mrna_superposition(&self) -> Fingerprint {
        self.mrna.superposition()
    }

    /// Get current cycle
    pub fn cycle(&self) -> u64 {
        self.state.cycle
    }

    /// Get collapse history
    pub fn collapse_history(&self) -> &[CollapseDecision] {
        &self.collapse_history
    }

    /// Clear grammar buffer
    pub fn clear_grammar_buffer(&mut self) {
        self.grammar_buffer.clear();
    }

    /// Get grammar buffer
    pub fn grammar_buffer(&self) -> &[GrammarTriangle] {
        &self.grammar_buffer
    }

    // =========================================================================
    // INTERNAL
    // =========================================================================

    fn update_quad_triangle_from_grammar(&mut self, coherence: f32) {
        // High coherence → boost analytical processing
        if coherence > 0.7 {
            let current = self.state.quad_triangle.processing.activations();
            self.state.quad_triangle.processing.set_activations(
                (current[0] + 0.1).min(1.0), // Boost analytical
                current[1],
                current[2],
            );
        }

        // Low coherence → boost intuitive processing
        if coherence < 0.3 {
            let current = self.state.quad_triangle.processing.activations();
            self.state.quad_triangle.processing.set_activations(
                current[0],
                (current[1] + 0.1).min(1.0), // Boost intuitive
                current[2],
            );
        }

        // Update gestalt based on coherence
        let current = self.state.quad_triangle.gestalt.activations();
        self.state.quad_triangle.gestalt.set_activations(
            coherence, // Coherence corner reflects grammar coherence
            current[1], current[2],
        );
    }
}

impl Default for GrammarCognitiveEngine {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// INGEST RESULT
// =============================================================================

/// Result of grammar ingestion
#[derive(Clone, Debug)]
pub struct IngestResult {
    /// Number of resonances triggered
    pub resonance_count: usize,

    /// Butterfly effect detected (if any)
    pub butterfly: Option<Butterfly>,

    /// Consciousness snapshot
    pub snapshot: ConsciousnessSnapshot,

    /// Processing time
    pub processing_time: std::time::Duration,

    /// Grammar coherence
    pub grammar_coherence: f32,
}

// =============================================================================
// STYLE TO QUAD-TRIANGLE MAPPING
// =============================================================================

fn style_to_quad_triangle(style: ThinkingStyle) -> QuadTriangle {
    match style {
        ThinkingStyle::Analytical => CognitiveProfiles::analytical(),
        ThinkingStyle::Convergent => CognitiveProfiles::analytical(),
        ThinkingStyle::Systematic => CognitiveProfiles::procedural(),

        ThinkingStyle::Creative => CognitiveProfiles::creative(),
        ThinkingStyle::Divergent => CognitiveProfiles::creative(),
        ThinkingStyle::Exploratory => CognitiveProfiles::creative(),

        ThinkingStyle::Focused => CognitiveProfiles::analytical(),
        ThinkingStyle::Diffuse => CognitiveProfiles::empathic(),
        ThinkingStyle::Peripheral => CognitiveProfiles::empathic(),

        ThinkingStyle::Intuitive => CognitiveProfiles::empathic(),
        ThinkingStyle::Deliberate => CognitiveProfiles::procedural(),

        ThinkingStyle::Metacognitive => CognitiveProfiles::counterfactual(),
    }
}

// =============================================================================
// BATCH PROCESSING
// =============================================================================

/// Process multiple inputs in batch
pub fn process_batch(
    engine: &mut GrammarCognitiveEngine,
    inputs: Vec<Vec<GrammarTriangle>>,
) -> Vec<IngestResult> {
    inputs
        .into_iter()
        .map(|triangles| engine.ingest_grammar(triangles))
        .collect()
}

// =============================================================================
// SERIALIZATION
// =============================================================================

/// Serialize engine state to bytes
pub fn serialize_state(engine: &GrammarCognitiveEngine) -> Vec<u8> {
    let mut bytes = Vec::new();

    // Style (1 byte)
    bytes.push(engine.state.style as u8);

    // Quad-triangle (12 floats = 48 bytes)
    for f in engine.state.quad_triangle.to_floats() {
        bytes.extend_from_slice(&f.to_le_bytes());
    }

    // Cycle (8 bytes)
    bytes.extend_from_slice(&engine.state.cycle.to_le_bytes());

    bytes
}

/// Deserialize engine state from bytes
pub fn deserialize_state(bytes: &[u8]) -> Option<GrammarCognitiveEngine> {
    if bytes.len() < 57 {
        return None;
    }

    let mut engine = GrammarCognitiveEngine::new();

    // Style
    let style = match bytes[0] {
        0 => ThinkingStyle::Analytical,
        1 => ThinkingStyle::Convergent,
        2 => ThinkingStyle::Systematic,
        3 => ThinkingStyle::Creative,
        4 => ThinkingStyle::Divergent,
        5 => ThinkingStyle::Exploratory,
        6 => ThinkingStyle::Focused,
        7 => ThinkingStyle::Diffuse,
        8 => ThinkingStyle::Peripheral,
        9 => ThinkingStyle::Intuitive,
        10 => ThinkingStyle::Deliberate,
        11 => ThinkingStyle::Metacognitive,
        _ => ThinkingStyle::Analytical,
    };
    engine.set_style(style);

    // Quad-triangle
    let mut floats = [0.0f32; 12];
    for i in 0..12 {
        let start = 1 + i * 4;
        floats[i] = f32::from_le_bytes([
            bytes[start],
            bytes[start + 1],
            bytes[start + 2],
            bytes[start + 3],
        ]);
    }
    engine.state.quad_triangle = QuadTriangle::from_floats(floats);

    // Cycle
    let cycle_bytes: [u8; 8] = bytes[49..57].try_into().ok()?;
    engine.state.cycle = u64::from_le_bytes(cycle_bytes);

    Some(engine)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive::GateState;

    #[test]
    fn test_engine_creation() {
        let engine = GrammarCognitiveEngine::new();
        assert_eq!(engine.cycle(), 0);
    }

    #[test]
    fn test_style_setting() {
        let mut engine = GrammarCognitiveEngine::new();
        engine.set_style(ThinkingStyle::Creative);
        assert_eq!(engine.style(), ThinkingStyle::Creative);
    }

    #[test]
    fn test_grammar_ingestion() {
        let mut engine = GrammarCognitiveEngine::new();

        let triangles = vec![
            GrammarTriangle::new(GrammarRole::Subject, "cat", "NP", 0.9),
            GrammarTriangle::new(GrammarRole::Predicate, "sat", "VP", 0.8),
            GrammarTriangle::new(GrammarRole::Object, "mat", "NP", 0.85),
        ];

        let result = engine.ingest_grammar(triangles);
        assert!(result.grammar_coherence > 0.0);
        assert_eq!(engine.cycle(), 1);
    }

    #[test]
    fn test_collapse_evaluation() {
        let mut engine = GrammarCognitiveEngine::new();

        // Tight consensus should FLOW
        let decision = engine.evaluate_collapse(&[0.9, 0.85, 0.88]);
        assert_eq!(decision.state, GateState::Flow);

        // High variance should BLOCK (need SD > 0.35)
        // [1.0, 0.0] gives SD = 0.5, which exceeds threshold
        let decision = engine.evaluate_collapse(&[1.0, 0.0]);
        assert_eq!(decision.state, GateState::Block);
    }

    #[test]
    fn test_serialization() {
        let mut engine = GrammarCognitiveEngine::new();
        engine.set_style(ThinkingStyle::Creative);
        engine.state.cycle = 42;

        let bytes = serialize_state(&engine);
        let restored = deserialize_state(&bytes).unwrap();

        assert_eq!(restored.style(), ThinkingStyle::Creative);
        assert_eq!(restored.cycle(), 42);
    }
}
