//! Unified Cognitive Substrate
//!
//! Integrates all cognitive components into one resonance field:
//! - QuadTriangles (4 × 10K-bit VSA)
//! - 7-Layer Consciousness Stack
//! - Collapse Gate with SIMD SD
//! - 12 Thinking Styles
//! - mRNA Cross-Pollination
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    UNIFIED COGNITIVE SUBSTRATE                      │
//! │                                                                     │
//! │  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐       │
//! │  │ QuadTriangle  │    │  7-Layer      │    │   Thinking    │       │
//! │  │   (40K-bit)   │◄──►│  Consciousness│◄──►│    Style      │       │
//! │  │               │    │               │    │  Modulation   │       │
//! │  │  Processing   │    │  L7: Meta     │    │               │       │
//! │  │  Content      │    │  L6: Exec     │    │  Analytical   │       │
//! │  │  Gestalt      │    │  L5: Working  │    │  Creative     │       │
//! │  │  Crystal      │    │  L4: Episodic │    │  Focused      │       │
//! │  │               │    │  L3: Semantic │    │  Intuitive    │       │
//! │  └───────┬───────┘    │  L2: Pattern  │    │  ...          │       │
//! │          │            │  L1: Sensory  │    └───────┬───────┘       │
//! │          │            └───────┬───────┘            │               │
//! │          │                    │                    │               │
//! │          ▼                    ▼                    ▼               │
//! │  ┌─────────────────────────────────────────────────────────────┐   │
//! │  │                    mRNA RESONANCE FIELD                     │   │
//! │  │                                                             │   │
//! │  │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐       │   │
//! │  │   │ Concept │  │ Concept │  │ Concept │  │   ...   │       │   │
//! │  │   │  10K    │  │  10K    │  │  10K    │  │         │       │   │
//! │  │   └────┬────┘  └────┬────┘  └────┬────┘  └─────────┘       │   │
//! │  │        │            │            │                         │   │
//! │  │        └────────────┴────────────┘                         │   │
//! │  │                     │                                      │   │
//! │  │              SUPERPOSITION                                 │   │
//! │  │               (10K-bit)                                    │   │
//! │  └─────────────────────┬───────────────────────────────────────┘   │
//! │                        │                                           │
//! │                        ▼                                           │
//! │  ┌─────────────────────────────────────────────────────────────┐   │
//! │  │                    COLLAPSE GATE                            │   │
//! │  │                                                             │   │
//! │  │   SD < 0.15    │   0.15 ≤ SD ≤ 0.35   │    SD > 0.35       │   │
//! │  │   ───────────  │   ─────────────────  │    ───────────     │   │
//! │  │   ?? FLOW      │   ?? HOLD            │    ?? BLOCK        │   │
//! │  │   (commit)     │   (superposition)    │    (clarify)       │   │
//! │  └─────────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```

use std::sync::{Arc, RwLock};
use std::time::Instant;

use crate::core::{Fingerprint, VsaOps};
use crate::cognitive::style::{ThinkingStyle, FieldModulation};
use crate::cognitive::quad_triangle::{QuadTriangle, TriangleId, CognitiveProfiles};
use crate::cognitive::seven_layer::{SevenLayerNode, LayerId, ConsciousnessSnapshot, process_layers_wave, snapshot_consciousness};
use crate::cognitive::collapse_gate::{GateState, CollapseDecision, evaluate_gate, calculate_sd};
use crate::fabric::mrna::{MRNA, TaggedConcept, Subsystem};
use crate::fabric::butterfly::{ButterflyDetector, Butterfly};

// =============================================================================
// COGNITIVE STATE
// =============================================================================

/// Complete cognitive state at a moment
#[derive(Clone)]
pub struct CognitiveState {
    /// Quad-triangle cognitive texture
    pub quad_triangle: QuadTriangle,
    
    /// 7-layer consciousness markers
    pub consciousness: SevenLayerNode,
    
    /// Current thinking style
    pub thinking_style: ThinkingStyle,
    
    /// Last collapse decision
    pub last_collapse: Option<CollapseDecision>,
    
    /// Processing cycle
    pub cycle: u64,
    
    /// Timestamp
    pub timestamp: Instant,
}

impl CognitiveState {
    /// Create new cognitive state
    pub fn new(path: &str) -> Self {
        Self {
            quad_triangle: QuadTriangle::neutral(),
            consciousness: SevenLayerNode::new(path),
            thinking_style: ThinkingStyle::Analytical,
            last_collapse: None,
            cycle: 0,
            timestamp: Instant::now(),
        }
    }
    
    /// Get global fingerprint (quad-triangle + consciousness)
    pub fn fingerprint(&self) -> Fingerprint {
        let qt_fp = self.quad_triangle.fingerprint().clone();
        let cs_fp = self.consciousness.vsa_core.clone();
        Fingerprint::bundle(&[qt_fp, cs_fp])
    }
}

// =============================================================================
// UNIFIED COGNITIVE SUBSTRATE
// =============================================================================

/// Unified cognitive substrate
pub struct CognitiveSubstrate {
    /// mRNA resonance field
    mrna: Arc<MRNA>,
    
    /// Butterfly detector
    butterfly: ButterflyDetector,
    
    /// Current cognitive state
    state: RwLock<CognitiveState>,
    
    /// Cognitive profiles cache
    profiles: CognitiveProfilesCache,
}

/// Cached cognitive profiles
struct CognitiveProfilesCache {
    analytical: QuadTriangle,
    creative: QuadTriangle,
    empathic: QuadTriangle,
    procedural: QuadTriangle,
    counterfactual: QuadTriangle,
}

impl Default for CognitiveProfilesCache {
    fn default() -> Self {
        Self {
            analytical: CognitiveProfiles::analytical(),
            creative: CognitiveProfiles::creative(),
            empathic: CognitiveProfiles::empathic(),
            procedural: CognitiveProfiles::procedural(),
            counterfactual: CognitiveProfiles::counterfactual(),
        }
    }
}

impl CognitiveSubstrate {
    /// Create new cognitive substrate
    pub fn new() -> Self {
        Self {
            mrna: Arc::new(MRNA::new()),
            butterfly: ButterflyDetector::new(),
            state: RwLock::new(CognitiveState::new("root")),
            profiles: CognitiveProfilesCache::default(),
        }
    }
    
    /// Create with specific thinking style
    pub fn with_style(style: ThinkingStyle) -> Self {
        let substrate = Self::new();
        substrate.set_thinking_style(style);
        substrate
    }
    
    // =========================================================================
    // THINKING STYLE
    // =========================================================================
    
    /// Set thinking style (modulates all subsystems)
    pub fn set_thinking_style(&self, style: ThinkingStyle) {
        // Update mRNA field
        self.mrna.set_style(style);
        
        // Update butterfly sensitivity
        self.butterfly.set_sensitivity(style.butterfly_sensitivity());
        
        // Update cognitive state
        if let Ok(mut state) = self.state.write() {
            state.thinking_style = style;
            
            // Nudge quad-triangle toward style-appropriate profile
            let target = self.profile_for_style(style);
            state.quad_triangle.nudge_toward(target, 0.3);
        }
    }
    
    /// Get thinking style
    pub fn thinking_style(&self) -> ThinkingStyle {
        self.state.read().map(|s| s.thinking_style).unwrap_or(ThinkingStyle::Analytical)
    }
    
    /// Get profile for thinking style
    fn profile_for_style(&self, style: ThinkingStyle) -> &QuadTriangle {
        match style {
            ThinkingStyle::Analytical | ThinkingStyle::Convergent | ThinkingStyle::Systematic => {
                &self.profiles.analytical
            }
            ThinkingStyle::Creative | ThinkingStyle::Divergent | ThinkingStyle::Exploratory => {
                &self.profiles.creative
            }
            ThinkingStyle::Focused => &self.profiles.procedural,
            ThinkingStyle::Diffuse | ThinkingStyle::Peripheral => &self.profiles.empathic,
            ThinkingStyle::Intuitive => &self.profiles.creative,
            ThinkingStyle::Deliberate => &self.profiles.analytical,
            ThinkingStyle::Metacognitive => &self.profiles.counterfactual,
        }
    }
    
    // =========================================================================
    // POLLINATION
    // =========================================================================
    
    /// Pollinate field with concept (returns resonances)
    pub fn pollinate(&self, concept: &Fingerprint) -> Vec<(TaggedConcept, f32)> {
        self.mrna.pollinate(concept)
    }
    
    /// Pollinate from specific subsystem
    pub fn pollinate_from(&self, subsystem: Subsystem, concept: &Fingerprint) -> Vec<(TaggedConcept, f32)> {
        self.mrna.pollinate_from(subsystem, concept)
    }
    
    /// Check cross-pollination between subsystems
    pub fn cross_pollinate(
        &self,
        source: Subsystem,
        concept: &Fingerprint,
        target: Subsystem,
    ) -> Option<f32> {
        self.mrna.cross_pollinate(source, concept, target)
    }
    
    // =========================================================================
    // CONSCIOUSNESS PROCESSING
    // =========================================================================
    
    /// Process input through 7-layer consciousness
    pub fn process_consciousness(&self, input: &Fingerprint) -> ConsciousnessSnapshot {
        let mut state = self.state.write().expect("lock poisoned");
        state.cycle += 1;
        
        // Process through layers
        let _results = process_layers_wave(&mut state.consciousness, input, state.cycle);
        
        // Take snapshot
        snapshot_consciousness(&state.consciousness, state.cycle)
    }
    
    /// Get current consciousness snapshot
    pub fn consciousness_snapshot(&self) -> ConsciousnessSnapshot {
        let state = self.state.read().expect("lock poisoned");
        snapshot_consciousness(&state.consciousness, state.cycle)
    }
    
    // =========================================================================
    // COLLAPSE GATE
    // =========================================================================
    
    /// Evaluate collapse gate for candidates
    pub fn evaluate_collapse(&self, candidate_scores: &[f32]) -> CollapseDecision {
        let decision = evaluate_gate(candidate_scores, true);
        
        // Store last decision
        if let Ok(mut state) = self.state.write() {
            state.last_collapse = Some(decision.clone());
        }
        
        decision
    }
    
    /// Check if collapse is permitted
    pub fn can_collapse(&self, candidate_scores: &[f32]) -> bool {
        let decision = evaluate_gate(candidate_scores, true);
        decision.can_collapse
    }
    
    /// Get collapse gate state
    pub fn gate_state(&self, candidate_scores: &[f32]) -> GateState {
        let sd = calculate_sd(candidate_scores);
        crate::cognitive::collapse_gate::get_gate_state(sd)
    }
    
    // =========================================================================
    // BUTTERFLY DETECTION
    // =========================================================================
    
    /// Detect butterfly effects in resonance cascade
    pub fn detect_butterfly(&self, input: &Fingerprint, cascade_size: usize) -> Option<Butterfly> {
        self.butterfly.detect(
            &self.mrna.history(),
            input,
            cascade_size,
        )
    }
    
    /// Predict butterfly effect before execution
    pub fn predict_butterfly(&self, hypothetical: &Fingerprint) -> Option<f32> {
        let prediction = self.butterfly.predict(hypothetical, &self.mrna);
        if prediction.confidence > 0.5 {
            Some(prediction.predicted_amplification)
        } else {
            None
        }
    }
    
    // =========================================================================
    // QUAD-TRIANGLE
    // =========================================================================
    
    /// Get quad-triangle state
    pub fn quad_triangle(&self) -> QuadTriangle {
        self.state.read().expect("lock poisoned").quad_triangle.clone()
    }
    
    /// Set quad-triangle activations
    pub fn set_quad_triangle_activations(
        &self,
        processing: [f32; 3],
        content: [f32; 3],
        gestalt: [f32; 3],
        crystallization: [f32; 3],
    ) {
        if let Ok(mut state) = self.state.write() {
            state.quad_triangle = QuadTriangle::with_activations(
                processing, content, gestalt, crystallization
            );
        }
    }
    
    /// Get cognitive signature
    pub fn cognitive_signature(&self) -> String {
        self.state.read()
            .map(|s| s.quad_triangle.signature())
            .unwrap_or_else(|_| "Unknown".to_string())
    }
    
    /// Check if in global flow
    pub fn is_global_flow(&self) -> bool {
        self.state.read()
            .map(|s| s.quad_triangle.is_global_flow())
            .unwrap_or(false)
    }
    
    // =========================================================================
    // UNIFIED QUERY
    // =========================================================================
    
    /// Unified resonance query across all subsystems
    pub fn query(&self, input: &Fingerprint) -> UnifiedQueryResult {
        let state = self.state.read().expect("lock poisoned");
        
        // Quad-triangle resonance
        let qt_resonance = state.quad_triangle.query_resonance(input);
        
        // Consciousness resonance (against VSA core)
        let consciousness_resonance = input.similarity(&state.consciousness.vsa_core);
        
        // mRNA field resonance
        let field_resonances = self.mrna.pollinate(input);
        let mrna_resonance = if field_resonances.is_empty() {
            0.0
        } else {
            field_resonances.iter().map(|(_, r)| *r).sum::<f32>() / field_resonances.len() as f32
        };
        
        // Style modulation
        let modulation = state.thinking_style.field_modulation();
        
        // Combined score (weighted by style)
        let combined = (
            qt_resonance * modulation.depth_bias +
            consciousness_resonance * modulation.breadth_bias +
            mrna_resonance * (1.0 - modulation.noise_tolerance)
        ) / 3.0;
        
        UnifiedQueryResult {
            quad_triangle_resonance: qt_resonance,
            consciousness_resonance,
            mrna_resonance,
            combined_score: combined,
            thinking_style: state.thinking_style,
            cognitive_signature: state.quad_triangle.signature(),
            flow_count: state.quad_triangle.flow_count(),
        }
    }
    
    // =========================================================================
    // STATE MANAGEMENT
    // =========================================================================
    
    /// Get current cycle
    pub fn cycle(&self) -> u64 {
        self.state.read().map(|s| s.cycle).unwrap_or(0)
    }
    
    /// Get global fingerprint
    pub fn fingerprint(&self) -> Fingerprint {
        self.state.read()
            .map(|s| s.fingerprint())
            .unwrap_or_else(|_| Fingerprint::zero())
    }
    
    /// Reset to neutral state
    pub fn reset(&self) {
        if let Ok(mut state) = self.state.write() {
            *state = CognitiveState::new("root");
        }
        self.mrna.clear();
    }
}

impl Default for CognitiveSubstrate {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// UNIFIED QUERY RESULT
// =============================================================================

/// Result of unified resonance query
#[derive(Clone, Debug)]
pub struct UnifiedQueryResult {
    /// Resonance with quad-triangle
    pub quad_triangle_resonance: f32,
    
    /// Resonance with consciousness core
    pub consciousness_resonance: f32,
    
    /// Resonance with mRNA field
    pub mrna_resonance: f32,
    
    /// Combined weighted score
    pub combined_score: f32,
    
    /// Current thinking style
    pub thinking_style: ThinkingStyle,
    
    /// Cognitive signature
    pub cognitive_signature: String,
    
    /// Number of triangles in flow
    pub flow_count: usize,
}

// =============================================================================
// COGNITIVE FABRIC (Full Integration)
// =============================================================================

/// Complete cognitive fabric with all subsystems
pub struct CognitiveFabric {
    /// Core substrate
    pub substrate: CognitiveSubstrate,
    
    /// mRNA reference
    pub mrna: Arc<MRNA>,
    
    /// Butterfly detector
    pub butterfly: ButterflyDetector,
}

impl CognitiveFabric {
    /// Create new cognitive fabric
    pub fn new() -> Self {
        let substrate = CognitiveSubstrate::new();
        let mrna = substrate.mrna.clone();
        let butterfly = ButterflyDetector::new();
        
        Self { substrate, mrna, butterfly }
    }
    
    /// Create with thinking style
    pub fn with_style(style: ThinkingStyle) -> Self {
        let mut fabric = Self::new();
        fabric.substrate.set_thinking_style(style);
        fabric.butterfly.set_sensitivity(style.butterfly_sensitivity());
        fabric
    }
    
    /// Full cognitive cycle
    pub fn cognitive_cycle(&mut self, input: &Fingerprint) -> CognitiveCycleResult {
        // 1. Pollinate mRNA field
        let resonances = self.substrate.pollinate(input);
        
        // 2. Process consciousness
        let consciousness = self.substrate.process_consciousness(input);
        
        // 3. Check for butterfly
        let butterfly = self.substrate.detect_butterfly(input, resonances.len());
        
        // 4. Unified query
        let query = self.substrate.query(input);
        
        // 5. Evaluate collapse if we have resonances
        let collapse = if !resonances.is_empty() {
            let scores: Vec<f32> = resonances.iter().map(|(_, r)| *r).collect();
            Some(self.substrate.evaluate_collapse(&scores))
        } else {
            None
        };
        
        CognitiveCycleResult {
            resonances,
            consciousness,
            butterfly,
            query,
            collapse,
            cycle: self.substrate.cycle(),
        }
    }
}

impl Default for CognitiveFabric {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of full cognitive cycle
#[derive(Clone)]
pub struct CognitiveCycleResult {
    /// mRNA resonances
    pub resonances: Vec<(TaggedConcept, f32)>,
    
    /// Consciousness snapshot
    pub consciousness: ConsciousnessSnapshot,
    
    /// Detected butterfly (if any)
    pub butterfly: Option<Butterfly>,
    
    /// Unified query result
    pub query: UnifiedQueryResult,
    
    /// Collapse decision (if applicable)
    pub collapse: Option<CollapseDecision>,
    
    /// Cycle number
    pub cycle: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_substrate_creation() {
        let substrate = CognitiveSubstrate::new();
        assert_eq!(substrate.thinking_style(), ThinkingStyle::Analytical);
    }
    
    #[test]
    fn test_style_modulation() {
        let substrate = CognitiveSubstrate::new();
        
        substrate.set_thinking_style(ThinkingStyle::Creative);
        assert_eq!(substrate.thinking_style(), ThinkingStyle::Creative);
        
        // Signature should reflect creative profile
        let sig = substrate.cognitive_signature();
        assert!(!sig.is_empty());
    }
    
    #[test]
    fn test_unified_query() {
        let substrate = CognitiveSubstrate::new();
        let input = Fingerprint::from_content("test input");
        
        let result = substrate.query(&input);
        assert!(result.combined_score >= 0.0 && result.combined_score <= 1.0);
    }
    
    #[test]
    fn test_cognitive_cycle() {
        let mut fabric = CognitiveFabric::new();
        let input = Fingerprint::from_content("test stimulus");
        
        let result = fabric.cognitive_cycle(&input);
        assert!(result.cycle > 0);
    }
}
