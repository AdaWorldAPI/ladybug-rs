//! Unified Cognitive Fabric
//!
//! Full integration layer coordinating all cognitive components:
//! - Multi-style cognitive processing
//! - Style transitions and blending
//! - Cross-component resonance tracking
//! - High-level cognitive API
//!
//! This module provides the highest level abstraction over the cognitive
//! architecture, enabling seamless style changes and multi-style coordination.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                         UNIFIED COGNITIVE FABRIC                            │
//! │                                                                             │
//! │   ┌─────────────────────────────────────────────────────────────────────┐   │
//! │   │                      STYLE COORDINATOR                              │   │
//! │   │                                                                     │   │
//! │   │   Analytical ◄──► Creative ◄──► Focused ◄──► Intuitive             │   │
//! │   │       │              │             │             │                  │   │
//! │   │       └──────────────┴─────────────┴─────────────┘                  │   │
//! │   │                        │                                            │   │
//! │   │                  BLEND WEIGHTS                                      │   │
//! │   └─────────────────────────────────────────────────────────────────────┘   │
//! │                                │                                            │
//! │                                ▼                                            │
//! │   ┌─────────────────────────────────────────────────────────────────────┐   │
//! │   │                    COGNITIVE SUBSTRATE                              │   │
//! │   │         QuadTriangle + 7-Layer + Collapse Gate + mRNA              │   │
//! │   └─────────────────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```

use std::sync::{Arc, RwLock};
use std::time::Instant;
use std::collections::HashMap;

use crate::core::{Fingerprint, VsaOps};
use crate::cognitive::{
    quad_triangle::{QuadTriangle, TriangleId, CognitiveProfiles},
    seven_layer::{SevenLayerNode, LayerId, ConsciousnessSnapshot, process_layers_wave, snapshot_consciousness},
    collapse_gate::{GateState, CollapseDecision, evaluate_gate, calculate_sd},
    style::{ThinkingStyle, FieldModulation},
};
use crate::fabric::mrna::{MRNA, TaggedConcept, Subsystem};
use crate::fabric::butterfly::{ButterflyDetector, Butterfly};

// =============================================================================
// STYLE WEIGHT
// =============================================================================

/// Weight for a particular thinking style in a blend
#[derive(Clone, Copy, Debug)]
pub struct StyleWeight {
    /// The thinking style
    pub style: ThinkingStyle,
    /// Weight in blend [0.0, 1.0]
    pub weight: f32,
}

impl StyleWeight {
    /// Create new style weight
    pub fn new(style: ThinkingStyle, weight: f32) -> Self {
        Self {
            style,
            weight: weight.clamp(0.0, 1.0),
        }
    }
}

// =============================================================================
// STYLE BLEND
// =============================================================================

/// Blend of multiple thinking styles
#[derive(Clone, Debug)]
pub struct StyleBlend {
    /// Active styles with weights
    pub styles: Vec<StyleWeight>,
    /// Dominant style (highest weight)
    pub dominant: ThinkingStyle,
    /// Blend coherence [0.0, 1.0]
    pub coherence: f32,
}

impl StyleBlend {
    /// Create single-style blend
    pub fn single(style: ThinkingStyle) -> Self {
        Self {
            styles: vec![StyleWeight::new(style, 1.0)],
            dominant: style,
            coherence: 1.0,
        }
    }
    
    /// Create multi-style blend
    pub fn multi(styles: Vec<StyleWeight>) -> Self {
        if styles.is_empty() {
            return Self::single(ThinkingStyle::Analytical);
        }
        
        // Normalize weights
        let total: f32 = styles.iter().map(|s| s.weight).sum();
        let normalized: Vec<StyleWeight> = if total > 0.0 {
            styles.iter()
                .map(|s| StyleWeight::new(s.style, s.weight / total))
                .collect()
        } else {
            vec![StyleWeight::new(ThinkingStyle::Analytical, 1.0)]
        };
        
        // Find dominant
        let dominant = normalized.iter()
            .max_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap())
            .map(|s| s.style)
            .unwrap_or(ThinkingStyle::Analytical);
        
        // Calculate coherence (inverse of style spread)
        let max_weight = normalized.iter().map(|s| s.weight).fold(0.0f32, f32::max);
        let coherence = if normalized.len() == 1 {
            1.0
        } else {
            // Higher max weight = higher coherence
            max_weight
        };
        
        Self {
            styles: normalized,
            dominant,
            coherence,
        }
    }
    
    /// Get blended field modulation
    pub fn field_modulation(&self) -> FieldModulation {
        let mut result = FieldModulation::default();
        
        for sw in &self.styles {
            let mod_i = sw.style.field_modulation();
            result.depth_bias += mod_i.depth_bias * sw.weight;
            result.breadth_bias += mod_i.breadth_bias * sw.weight;
            result.speed_bias += mod_i.speed_bias * sw.weight;
            result.novelty_threshold += mod_i.novelty_threshold * sw.weight;
            result.coherence_threshold += mod_i.coherence_threshold * sw.weight;
        }
        
        result
    }
    
    /// Get blended butterfly sensitivity
    pub fn butterfly_sensitivity(&self) -> f32 {
        self.styles.iter()
            .map(|sw| sw.style.butterfly_sensitivity() * sw.weight)
            .sum()
    }
}

impl Default for StyleBlend {
    fn default() -> Self {
        Self::single(ThinkingStyle::Analytical)
    }
}

// =============================================================================
// FABRIC STATE
// =============================================================================

/// Internal state for the fabric
pub struct FabricState {
    /// Quad-triangle cognitive texture
    pub quad_triangle: QuadTriangle,
    
    /// 7-layer consciousness node
    pub consciousness: SevenLayerNode,
    
    /// Current style blend
    pub style: StyleBlend,
    
    /// mRNA cross-pollination substrate
    pub mrna: MRNA,
    
    /// Butterfly detector
    pub butterfly: ButterflyDetector,
    
    /// Processing cycle
    pub cycle: u64,
    
    /// Last collapse decision
    pub last_collapse: Option<CollapseDecision>,
    
    /// Active concepts in resonance field
    pub active_concepts: Vec<TaggedConcept>,
    
    /// Style history for tracking transitions
    pub style_history: Vec<(ThinkingStyle, Instant)>,
    
    /// Creation timestamp
    pub created_at: Instant,
}

impl FabricState {
    /// Create new fabric state
    pub fn new(path: &str) -> Self {
        Self {
            quad_triangle: QuadTriangle::neutral(),
            consciousness: SevenLayerNode::new(path),
            style: StyleBlend::default(),
            mrna: MRNA::new(128),
            butterfly: ButterflyDetector::new(100),
            cycle: 0,
            last_collapse: None,
            active_concepts: Vec::new(),
            style_history: vec![(ThinkingStyle::Analytical, Instant::now())],
            created_at: Instant::now(),
        }
    }
}

// =============================================================================
// UNIFIED FABRIC
// =============================================================================

/// Unified Cognitive Fabric - highest level cognitive abstraction
///
/// Provides multi-style processing, style transitions, and full
/// integration of all cognitive components.
pub struct UnifiedFabric {
    /// Thread-safe state
    state: Arc<RwLock<FabricState>>,
    
    /// Path identifier
    path: String,
}

impl UnifiedFabric {
    /// Create new unified fabric
    pub fn new(path: &str) -> Self {
        Self {
            state: Arc::new(RwLock::new(FabricState::new(path))),
            path: path.to_string(),
        }
    }
    
    /// Create with specific style
    pub fn with_style(path: &str, style: ThinkingStyle) -> Self {
        let fabric = Self::new(path);
        fabric.set_style(style);
        fabric
    }
    
    /// Create with multi-style blend
    pub fn with_blend(path: &str, styles: Vec<StyleWeight>) -> Self {
        let fabric = Self::new(path);
        fabric.set_blend(StyleBlend::multi(styles));
        fabric
    }
    
    // =========================================================================
    // STYLE MANAGEMENT
    // =========================================================================
    
    /// Get current style blend
    pub fn style(&self) -> StyleBlend {
        self.state.read().unwrap().style.clone()
    }
    
    /// Get dominant style
    pub fn dominant_style(&self) -> ThinkingStyle {
        self.state.read().unwrap().style.dominant
    }
    
    /// Set single style
    pub fn set_style(&self, style: ThinkingStyle) {
        let mut state = self.state.write().unwrap();
        state.style = StyleBlend::single(style);
        state.style_history.push((style, Instant::now()));
        
        // Limit history size
        if state.style_history.len() > 100 {
            state.style_history.remove(0);
        }
    }
    
    /// Set multi-style blend
    pub fn set_blend(&self, blend: StyleBlend) {
        let mut state = self.state.write().unwrap();
        let dominant = blend.dominant;
        state.style = blend;
        state.style_history.push((dominant, Instant::now()));
        
        // Limit history size
        if state.style_history.len() > 100 {
            state.style_history.remove(0);
        }
    }
    
    /// Transition to new style with gradual blend
    pub fn transition_style(&self, target: ThinkingStyle, blend_factor: f32) {
        let current = self.dominant_style();
        
        if blend_factor >= 1.0 {
            self.set_style(target);
        } else if blend_factor <= 0.0 {
            // Keep current
        } else {
            let blend = StyleBlend::multi(vec![
                StyleWeight::new(current, 1.0 - blend_factor),
                StyleWeight::new(target, blend_factor),
            ]);
            self.set_blend(blend);
        }
    }
    
    /// Get style history
    pub fn style_history(&self) -> Vec<(ThinkingStyle, Instant)> {
        self.state.read().unwrap().style_history.clone()
    }
    
    // =========================================================================
    // PROCESSING
    // =========================================================================
    
    /// Process input through unified fabric
    pub fn process(&self, input: &Fingerprint) -> FabricResult {
        let mut state = self.state.write().unwrap();
        state.cycle += 1;
        let cycle = state.cycle;
        
        // Get style modulation
        let modulation = state.style.field_modulation();
        let sensitivity = state.style.butterfly_sensitivity();
        
        // Process through 7 layers
        let layer_results = process_layers_wave(
            &mut state.consciousness,
            input,
            cycle,
        );
        
        // Update quad-triangle based on layer activations
        let profiles = state.quad_triangle.compute_profiles();
        
        // Check for butterflies
        let dominant_layer = layer_results.iter()
            .enumerate()
            .max_by(|a, b| a.1.activation.partial_cmp(&b.1.activation).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        
        let amplitude = layer_results.iter()
            .map(|r| r.activation)
            .fold(0.0f32, f32::max);
        
        let detected = state.butterfly.detect(amplitude, sensitivity);
        
        // Snapshot consciousness
        let snapshot = snapshot_consciousness(&state.consciousness, cycle);
        
        FabricResult {
            snapshot,
            profiles,
            butterflies: detected,
            cycle,
            dominant_layer: LayerId::from_index(dominant_layer),
            style_coherence: state.style.coherence,
            modulation,
        }
    }
    
    /// Ingest concept into resonance field
    pub fn ingest_concept(&self, content: &str, subsystem: Subsystem) -> TaggedConcept {
        let mut state = self.state.write().unwrap();
        
        let fingerprint = Fingerprint::from_content(content);
        let concept = TaggedConcept {
            fingerprint,
            subsystem,
            strength: 1.0,
            timestamp: state.cycle,
        };
        
        // Add to active concepts
        state.active_concepts.push(concept.clone());
        
        // Pollinate
        state.mrna.pollinate(concept.clone());
        
        concept
    }
    
    /// Evaluate collapse gate for candidates
    pub fn evaluate_collapse(&self, candidates: &[f32]) -> CollapseDecision {
        let mut state = self.state.write().unwrap();
        
        // Use style's coherence threshold for homogeneous determination
        let threshold = state.style.field_modulation().coherence_threshold;
        let sd = calculate_sd(candidates);
        let homogeneous = sd < threshold;
        
        let decision = evaluate_gate(candidates, homogeneous);
        state.last_collapse = Some(decision.clone());
        
        decision
    }
    
    /// Get superposition of active concepts
    pub fn superposition(&self) -> Fingerprint {
        let state = self.state.read().unwrap();
        
        if state.active_concepts.is_empty() {
            return Fingerprint::zero();
        }
        
        let fps: Vec<&Fingerprint> = state.active_concepts.iter()
            .map(|c| &c.fingerprint)
            .collect();
        
        Fingerprint::bundle(&fps)
    }
    
    /// Get current cycle
    pub fn cycle(&self) -> u64 {
        self.state.read().unwrap().cycle
    }
    
    /// Get last collapse decision
    pub fn last_collapse(&self) -> Option<CollapseDecision> {
        self.state.read().unwrap().last_collapse.clone()
    }
    
    /// Clear active concepts
    pub fn clear_concepts(&self) {
        let mut state = self.state.write().unwrap();
        state.active_concepts.clear();
    }
    
    /// Get path identifier
    pub fn path(&self) -> &str {
        &self.path
    }
}

impl Clone for UnifiedFabric {
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
            path: self.path.clone(),
        }
    }
}

// =============================================================================
// FABRIC RESULT
// =============================================================================

/// Result from processing through unified fabric
#[derive(Clone, Debug)]
pub struct FabricResult {
    /// Consciousness snapshot
    pub snapshot: ConsciousnessSnapshot,
    
    /// Cognitive profiles from quad-triangle
    pub profiles: CognitiveProfiles,
    
    /// Detected butterflies
    pub butterflies: Vec<Butterfly>,
    
    /// Processing cycle
    pub cycle: u64,
    
    /// Dominant layer
    pub dominant_layer: LayerId,
    
    /// Style blend coherence
    pub style_coherence: f32,
    
    /// Field modulation from style
    pub modulation: FieldModulation,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_style_change() {
        let fabric = UnifiedFabric::new("test/style_change");
        
        // Initial style
        assert_eq!(fabric.dominant_style(), ThinkingStyle::Analytical);
        
        // Change style
        fabric.set_style(ThinkingStyle::Creative);
        assert_eq!(fabric.dominant_style(), ThinkingStyle::Creative);
        
        // Check history
        let history = fabric.style_history();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].0, ThinkingStyle::Analytical);
        assert_eq!(history[1].0, ThinkingStyle::Creative);
    }
    
    #[test]
    fn test_multi_style() {
        let fabric = UnifiedFabric::with_blend("test/multi_style", vec![
            StyleWeight::new(ThinkingStyle::Analytical, 0.6),
            StyleWeight::new(ThinkingStyle::Creative, 0.4),
        ]);
        
        // Dominant should be Analytical (higher weight)
        assert_eq!(fabric.dominant_style(), ThinkingStyle::Analytical);
        
        // Blend should have 2 styles
        let blend = fabric.style();
        assert_eq!(blend.styles.len(), 2);
        
        // Coherence should reflect the blend
        assert!(blend.coherence > 0.5);
        assert!(blend.coherence < 1.0);
    }
    
    #[test]
    fn test_style_transition() {
        let fabric = UnifiedFabric::new("test/transition");
        
        // Start analytical
        assert_eq!(fabric.dominant_style(), ThinkingStyle::Analytical);
        
        // Partial transition to creative
        fabric.transition_style(ThinkingStyle::Creative, 0.3);
        
        // Still analytical dominant
        assert_eq!(fabric.dominant_style(), ThinkingStyle::Analytical);
        
        // Full transition
        fabric.transition_style(ThinkingStyle::Creative, 1.0);
        assert_eq!(fabric.dominant_style(), ThinkingStyle::Creative);
    }
    
    #[test]
    fn test_process() {
        let fabric = UnifiedFabric::new("test/process");
        
        let input = Fingerprint::from_content("test input for processing");
        let result = fabric.process(&input);
        
        assert_eq!(result.cycle, 1);
        assert!(result.style_coherence > 0.0);
    }
    
    #[test]
    fn test_concept_ingestion() {
        let fabric = UnifiedFabric::new("test/concepts");
        
        let concept = fabric.ingest_concept("quantum coherence", Subsystem::Query);
        assert_eq!(concept.subsystem, Subsystem::Query);
        assert!(concept.strength > 0.0);
        
        // Superposition should contain the concept
        let super_fp = fabric.superposition();
        let similarity = super_fp.similarity(&concept.fingerprint);
        assert!(similarity > 0.9);
    }
    
    #[test]
    fn test_collapse_evaluation() {
        let fabric = UnifiedFabric::new("test/collapse");
        
        // Homogeneous candidates -> FLOW
        let decision = fabric.evaluate_collapse(&[0.9, 0.88, 0.91]);
        assert!(matches!(decision.state, GateState::Flow));
        
        // Heterogeneous candidates -> HOLD or BLOCK
        let decision = fabric.evaluate_collapse(&[0.9, 0.3, 0.1]);
        assert!(!matches!(decision.state, GateState::Flow));
    }
    
    #[test]
    fn test_style_modulation() {
        let blend = StyleBlend::multi(vec![
            StyleWeight::new(ThinkingStyle::Analytical, 0.5),
            StyleWeight::new(ThinkingStyle::Creative, 0.5),
        ]);
        
        let mod_a = ThinkingStyle::Analytical.field_modulation();
        let mod_c = ThinkingStyle::Creative.field_modulation();
        let mod_blend = blend.field_modulation();
        
        // Blended modulation should be average
        let expected_depth = (mod_a.depth_bias + mod_c.depth_bias) / 2.0;
        let diff = (mod_blend.depth_bias - expected_depth).abs();
        assert!(diff < 0.01);
    }
}
