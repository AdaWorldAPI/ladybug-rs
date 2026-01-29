//! Integrated Cognitive Fabric
//!
//! Combines all cognitive components into unified resonance substrate:
//! - 4 QuadTriangles (Processing/Content/Gestalt/Crystallization)
//! - 7-Layer Consciousness Stack
//! - 12 Thinking Styles
//! - Collapse Gate (FLOW/HOLD/BLOCK)
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    COGNITIVE FABRIC ARCHITECTURE                    │
//! │                                                                     │
//! │  ┌───────────────────────────────────────────────────────────────┐ │
//! │  │                    THINKING STYLE (12)                        │ │
//! │  │   Modulates: threshold, fan-out, exploration, speed           │ │
//! │  └───────────────────────────────────────────────────────────────┘ │
//! │                              │                                      │
//! │                              ▼                                      │
//! │  ┌───────────────────────────────────────────────────────────────┐ │
//! │  │                    QUAD-TRIANGLE (4×3×10K)                    │ │
//! │  │                                                               │ │
//! │  │   Processing ─────┬───── Content                              │ │
//! │  │        │          │          │                                │ │
//! │  │        └────── Gestalt ──────┘                                │ │
//! │  │                   │                                           │ │
//! │  │            Crystallization                                    │ │
//! │  └───────────────────────────────────────────────────────────────┘ │
//! │                              │                                      │
//! │                              ▼                                      │
//! │  ┌───────────────────────────────────────────────────────────────┐ │
//! │  │                 7-LAYER CONSCIOUSNESS                         │ │
//! │  │   L7:Meta ← L6:Exec ← L5:Work ← L4:Epis ← L3:Sem ← L2:Pat ← L1│ │
//! │  └───────────────────────────────────────────────────────────────┘ │
//! │                              │                                      │
//! │                              ▼                                      │
//! │  ┌───────────────────────────────────────────────────────────────┐ │
//! │  │                    COLLAPSE GATE                              │ │
//! │  │              SD < 0.15 → FLOW (commit)                        │ │
//! │  │         0.15 ≤ SD ≤ 0.35 → HOLD (ruminate)                    │ │
//! │  │              SD > 0.35 → BLOCK (clarify)                      │ │
//! │  └───────────────────────────────────────────────────────────────┘ │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```

use std::time::Instant;
use crate::core::{Fingerprint, VsaOps};
use super::style::{ThinkingStyle, FieldModulation};
use super::quad_triangle::{QuadTriangle, TriangleId, CognitiveProfiles};
use super::seven_layer::{SevenLayerNode, LayerId, ConsciousnessSnapshot, process_layers_wave, snapshot_consciousness};
use super::collapse_gate::{GateState, CollapseDecision, evaluate_gate};

// =============================================================================
// COGNITIVE STATE
// =============================================================================

/// Complete cognitive state snapshot
#[derive(Clone)]
pub struct CognitiveState {
    /// Current thinking style
    pub style: ThinkingStyle,
    
    /// Field modulation from style
    pub modulation: FieldModulation,
    
    /// Quad-triangle state
    pub triangles: QuadTriangle,
    
    /// 7-layer consciousness
    pub consciousness: ConsciousnessSnapshot,
    
    /// Last collapse decision
    pub last_collapse: Option<CollapseDecision>,
    
    /// Processing cycle
    pub cycle: u64,
    
    /// Global coherence (triangles × layers)
    pub coherence: f32,
    
    /// Emergence signal
    pub emergence: f32,
}

// =============================================================================
// COGNITIVE FABRIC
// =============================================================================

/// Integrated cognitive fabric
pub struct CognitiveFabric {
    /// Current thinking style
    style: ThinkingStyle,
    
    /// Quad-triangle state
    triangles: QuadTriangle,
    
    /// 7-layer node
    node: SevenLayerNode,
    
    /// Processing cycle counter
    cycle: u64,
    
    /// History of collapse decisions
    collapse_history: Vec<CollapseDecision>,
    
    /// Active resonances (above threshold)
    active_resonances: Vec<(Fingerprint, f32)>,
}

impl CognitiveFabric {
    /// Create new cognitive fabric with default style
    pub fn new(path: &str) -> Self {
        Self {
            style: ThinkingStyle::Analytical,
            triangles: QuadTriangle::neutral(),
            node: SevenLayerNode::new(path),
            cycle: 0,
            collapse_history: Vec::new(),
            active_resonances: Vec::new(),
        }
    }
    
    /// Create with specific thinking style
    pub fn with_style(path: &str, style: ThinkingStyle) -> Self {
        let triangles = match style {
            ThinkingStyle::Analytical | ThinkingStyle::Convergent | ThinkingStyle::Systematic 
                => CognitiveProfiles::analytical(),
            ThinkingStyle::Creative | ThinkingStyle::Divergent | ThinkingStyle::Exploratory 
                => CognitiveProfiles::creative(),
            ThinkingStyle::Focused => CognitiveProfiles::procedural(),
            ThinkingStyle::Diffuse | ThinkingStyle::Peripheral => CognitiveProfiles::empathic(),
            ThinkingStyle::Intuitive => CognitiveProfiles::creative(),
            ThinkingStyle::Deliberate => CognitiveProfiles::analytical(),
            ThinkingStyle::Metacognitive => QuadTriangle::neutral(),
        };
        
        Self {
            style,
            triangles,
            node: SevenLayerNode::new(path),
            cycle: 0,
            collapse_history: Vec::new(),
            active_resonances: Vec::new(),
        }
    }
    
    /// Set thinking style
    pub fn set_style(&mut self, style: ThinkingStyle) {
        self.style = style;
        
        // Adjust triangles based on style
        let target = match style {
            ThinkingStyle::Analytical | ThinkingStyle::Convergent | ThinkingStyle::Systematic 
                => CognitiveProfiles::analytical(),
            ThinkingStyle::Creative | ThinkingStyle::Divergent | ThinkingStyle::Exploratory 
                => CognitiveProfiles::creative(),
            ThinkingStyle::Focused => CognitiveProfiles::procedural(),
            ThinkingStyle::Diffuse | ThinkingStyle::Peripheral => CognitiveProfiles::empathic(),
            ThinkingStyle::Intuitive => CognitiveProfiles::creative(),
            ThinkingStyle::Deliberate => CognitiveProfiles::analytical(),
            ThinkingStyle::Metacognitive => QuadTriangle::neutral(),
        };
        
        // Blend toward new profile
        self.triangles.blend_toward(&target, 0.5);
    }
    
    /// Get current style
    pub fn style(&self) -> ThinkingStyle {
        self.style
    }
    
    /// Get field modulation for current style
    pub fn modulation(&self) -> FieldModulation {
        self.style.field_modulation()
    }
    
    /// Process input through full cognitive stack
    pub fn process(&mut self, input: &Fingerprint) -> CognitiveState {
        self.cycle += 1;
        let modulation = self.modulation();
        
        // 1. Process through 7-layer stack
        let _results = process_layers_wave(&mut self.node, input, self.cycle);
        let consciousness = snapshot_consciousness(&self.node, self.cycle);
        
        // 2. Update triangles based on layer activations
        self.update_triangles_from_layers(&consciousness);
        
        // 3. Compute active resonances (filtered by style threshold)
        self.update_active_resonances(input, modulation.resonance_threshold);
        
        // 4. Evaluate collapse gate if we have candidates
        let last_collapse = if self.active_resonances.len() >= 2 {
            let scores: Vec<f32> = self.active_resonances.iter()
                .take(3) // Triangle requires 3
                .map(|(_, s)| *s)
                .collect();
            
            if scores.len() >= 2 {
                let decision = evaluate_gate(&scores, true);
                self.collapse_history.push(decision.clone());
                Some(decision)
            } else {
                None
            }
        } else {
            None
        };
        
        // 5. Compute global coherence
        let triangle_coherence = self.triangles.coherence();
        let layer_coherence = consciousness.coherence;
        let coherence = (triangle_coherence + layer_coherence) / 2.0;
        
        // 6. Compute emergence
        let emergence = consciousness.emergence * (1.0 - coherence * 0.3);
        
        CognitiveState {
            style: self.style,
            modulation,
            triangles: self.triangles.clone(),
            consciousness,
            last_collapse,
            cycle: self.cycle,
            coherence,
            emergence,
        }
    }
    
    /// Update triangles based on layer activations
    fn update_triangles_from_layers(&mut self, snapshot: &ConsciousnessSnapshot) {
        // Map layer activations to triangle nudges
        let layers = &snapshot.layers;
        
        // Processing triangle: L2 (Pattern) → Analytical, L1 (Sensory) → Intuitive, L6 (Exec) → Procedural
        let proc_target = QuadTriangle::with_activations(
            [layers[1].value, layers[0].value, layers[5].value],
            self.triangles.content.activations(),
            self.triangles.gestalt.activations(),
            self.triangles.crystallization.activations(),
        );
        
        // Content triangle: L3 (Semantic) → Abstract, L1 (Sensory) → Concrete, L4 (Episodic) → Relational
        let cont_target = QuadTriangle::with_activations(
            self.triangles.processing.activations(),
            [layers[2].value, layers[0].value, layers[3].value],
            self.triangles.gestalt.activations(),
            self.triangles.crystallization.activations(),
        );
        
        // Gestalt triangle: L7 (Meta) → Coherence, L2 (Pattern) → Novelty, L5 (Working) → Resonance
        let gest_target = QuadTriangle::with_activations(
            self.triangles.processing.activations(),
            self.triangles.content.activations(),
            [layers[6].value, layers[1].value, layers[4].value],
            self.triangles.crystallization.activations(),
        );
        
        // Nudge toward targets
        self.triangles.processing.nudge_toward(&proc_target.processing, 0.1);
        self.triangles.content.nudge_toward(&cont_target.content, 0.1);
        self.triangles.gestalt.nudge_toward(&gest_target.gestalt, 0.1);
    }
    
    /// Update active resonances above threshold
    fn update_active_resonances(&mut self, input: &Fingerprint, threshold: f32) {
        // Clear old resonances
        self.active_resonances.retain(|(_, score)| *score > threshold * 0.5);
        
        // Add input resonance with triangles
        let tri_resonance = self.triangles.query_resonance(input);
        if tri_resonance > threshold {
            self.active_resonances.push((input.clone(), tri_resonance));
        }
        
        // Add input resonance with node core
        let node_resonance = input.similarity(&self.node.vsa_core);
        if node_resonance > threshold && (node_resonance - tri_resonance).abs() > 0.1 {
            // Different enough to be separate candidate
            self.active_resonances.push((self.node.vsa_core.clone(), node_resonance));
        }
        
        // Sort by score descending
        self.active_resonances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Keep only top candidates based on fan-out
        let fan_out = self.modulation().fan_out as usize;
        self.active_resonances.truncate(fan_out.max(3));
    }
    
    /// Check if in FLOW state (both triangles and gate)
    pub fn is_flow(&self) -> bool {
        let triangles_flow = self.triangles.flow_count() >= 3;
        let gate_flow = self.collapse_history.last()
            .map(|d| d.state == GateState::Flow)
            .unwrap_or(false);
        
        triangles_flow && gate_flow
    }
    
    /// Check if blocked (high dispersion)
    pub fn is_blocked(&self) -> bool {
        self.collapse_history.last()
            .map(|d| d.state == GateState::Block)
            .unwrap_or(false)
    }
    
    /// Get cognitive signature
    pub fn signature(&self) -> String {
        format!("{} | {}", self.style, self.triangles.signature())
    }
    
    /// Get last N collapse decisions
    pub fn recent_collapses(&self, n: usize) -> &[CollapseDecision] {
        let start = self.collapse_history.len().saturating_sub(n);
        &self.collapse_history[start..]
    }
    
    /// Reset to neutral state
    pub fn reset(&mut self) {
        self.triangles = QuadTriangle::neutral();
        self.node = SevenLayerNode::new(&self.node.path);
        self.cycle = 0;
        self.collapse_history.clear();
        self.active_resonances.clear();
    }
    
    /// Serialize state to bytes (for persistence)
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        
        // Style (1 byte)
        bytes.push(self.style as u8);
        
        // Triangles (12 floats = 48 bytes)
        for f in self.triangles.to_floats() {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        
        // Cycle (8 bytes)
        bytes.extend_from_slice(&self.cycle.to_le_bytes());
        
        bytes
    }
    
    /// Deserialize from bytes
    pub fn from_bytes(path: &str, bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 57 { // 1 + 48 + 8
            return None;
        }
        
        let style = ThinkingStyle::from_u8(bytes[0])?;
        
        let mut floats = [0.0f32; 12];
        for i in 0..12 {
            let start = 1 + i * 4;
            floats[i] = f32::from_le_bytes([
                bytes[start], bytes[start+1], bytes[start+2], bytes[start+3]
            ]);
        }
        
        let cycle = u64::from_le_bytes([
            bytes[49], bytes[50], bytes[51], bytes[52],
            bytes[53], bytes[54], bytes[55], bytes[56],
        ]);
        
        let mut fabric = Self::with_style(path, style);
        fabric.triangles = QuadTriangle::from_floats(floats);
        fabric.cycle = cycle;
        
        Some(fabric)
    }
}

impl ThinkingStyle {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Analytical),
            1 => Some(Self::Convergent),
            2 => Some(Self::Systematic),
            3 => Some(Self::Creative),
            4 => Some(Self::Divergent),
            5 => Some(Self::Exploratory),
            6 => Some(Self::Focused),
            7 => Some(Self::Diffuse),
            8 => Some(Self::Peripheral),
            9 => Some(Self::Intuitive),
            10 => Some(Self::Deliberate),
            11 => Some(Self::Metacognitive),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fabric_creation() {
        let fabric = CognitiveFabric::new("test");
        assert_eq!(fabric.cycle, 0);
    }
    
    #[test]
    fn test_style_change() {
        let mut fabric = CognitiveFabric::new("test");
        fabric.set_style(ThinkingStyle::Creative);
        assert_eq!(fabric.style(), ThinkingStyle::Creative);
    }
    
    #[test]
    fn test_process_cycle() {
        let mut fabric = CognitiveFabric::new("test");
        let input = Fingerprint::from_content("test input");
        
        let state = fabric.process(&input);
        assert_eq!(state.cycle, 1);
        assert!(state.coherence >= 0.0 && state.coherence <= 1.0);
    }
    
    #[test]
    fn test_serialization() {
        let mut fabric = CognitiveFabric::with_style("test", ThinkingStyle::Creative);
        let input = Fingerprint::from_content("data");
        fabric.process(&input);
        
        let bytes = fabric.to_bytes();
        let restored = CognitiveFabric::from_bytes("test", &bytes).unwrap();
        
        assert_eq!(restored.style(), ThinkingStyle::Creative);
        assert_eq!(restored.cycle, fabric.cycle);
    }
}
