//! Cognitive Service — Mode-Aware Facade for Cross-Crate Integration
//!
//! Unifies the cognitive pipeline (Grammar Triangle → QuadTriangle → 10-Layer
//! Stack → CollapseGate) behind a single API that adapts to three operating modes:
//!
//! - **PassiveRag**: ladybug-rs serves as a cognitive database. crewai-rust drives
//!   orchestration; ladybug only answers resonance queries. No cognitive state mutation.
//!
//! - **Brain**: ladybug-rs drives the cognitive loop. It imports crewai-rust personas
//!   as thinking style layers and runs the full Grammar→QuadTriangle→Gate pipeline.
//!
//! - **Orchestrated**: n8n-rs manages the execution pipeline. The service responds
//!   to step commands via Blackboard, with thinking style and gate state communicated
//!   through `CognitiveSnapshot` entries.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                         COGNITIVE SERVICE                              │
//! │                                                                        │
//! │  Mode::PassiveRag          Mode::Brain           Mode::Orchestrated    │
//! │  ─────────────────         ──────────────        ──────────────────    │
//! │  query() only              full process()        step-driven           │
//! │  no state mutation         style drives           blackboard I/O       │
//! │  crewai-rust leads         ladybug leads          n8n-rs leads         │
//! │                                                                        │
//! │  All three share:                                                      │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                 │
//! │  │ Grammar △    │→│ Cognitive    │→│ Collapse     │                 │
//! │  │ NSM+Caus+Qual│  │ Fabric      │  │ Gate         │                 │
//! │  │ → Fingerprint│  │ QuadTriangle│  │ FLOW/HOLD/   │                 │
//! │  │              │  │ 10-Layer    │  │ BLOCK        │                 │
//! │  └──────────────┘  └──────────────┘  └──────────────┘                 │
//! │                                                                        │
//! │  ThinkingStyleBridge: crewai 36 ←→ ladybug 12 ←→ FieldModulation     │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```

use crate::cognitive::collapse_gate::{CollapseDecision, GateState, evaluate_gate};
use crate::cognitive::fabric::CognitiveState;
use crate::cognitive::quad_triangle::QuadTriangle;
use crate::cognitive::style::{FieldModulation, ThinkingStyle};
use crate::core::Fingerprint;
use crate::grammar::GrammarTriangle;

// =============================================================================
// OPERATING MODE
// =============================================================================

/// Operating mode for the cognitive service.
///
/// Determines how much of the cognitive pipeline is active and who drives it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CognitiveMode {
    /// Passive RAG: resonance queries only, no state mutation.
    /// Used when crewai-rust standalone drives orchestration.
    /// ladybug-rs answers "what resonates?" but doesn't advance its own cycle.
    PassiveRag,

    /// Brain mode: ladybug-rs drives the full cognitive cycle.
    /// Grammar Triangle → QuadTriangle → 10-Layer → CollapseGate.
    /// crewai-rust personas map to thinking styles via the bridge.
    Brain,

    /// Orchestrated mode: responds to external step commands.
    /// n8n-rs manages the execution pipeline. The service reads/writes
    /// cognitive state through CognitiveSnapshot entries on the blackboard.
    Orchestrated,
}

// =============================================================================
// THINKING STYLE BRIDGE
// =============================================================================

/// Maps external thinking style identifiers to ladybug's 12 internal styles.
///
/// crewai-rust has 36 styles in 6 clusters. ladybug has 12 styles in 4 clusters.
/// This bridge provides the canonical mapping and interpolation.
///
/// ```text
/// crewai-rust 6 clusters          ladybug 4 clusters
/// ═══════════════════════          ═════════════════════
/// Analytical (6 styles)     ──→   Convergent: Analytical, Convergent, Systematic
/// Creative   (6 styles)     ──→   Divergent:  Creative, Divergent, Exploratory
/// Empathic   (6 styles)     ──→   Attention:  Diffuse, Peripheral  + Intuitive
/// Direct     (6 styles)     ──→   Convergent: Focused, Systematic
/// Exploratory(6 styles)     ──→   Divergent:  Exploratory + Deliberate
/// Meta       (6 styles)     ──→   Meta:       Metacognitive + Analytical
/// ```
pub struct ThinkingStyleBridge;

/// External style identifier — either a named style from crewai-rust
/// or a cluster+weight pair for interpolation.
#[derive(Clone, Debug)]
pub enum ExternalStyle {
    /// Named style from crewai-rust's 36-style taxonomy.
    Named(String),
    /// Cluster name with optional emphasis weight (0.0-1.0).
    Cluster { name: String, weight: f32 },
    /// Direct ladybug style (bypass mapping).
    Direct(ThinkingStyle),
    /// Interpolated: blend of multiple styles with weights.
    Blend(Vec<(ThinkingStyle, f32)>),
}

impl ThinkingStyleBridge {
    /// Map a crewai-rust style name to a ladybug ThinkingStyle.
    ///
    /// Handles the 36→12 mapping. Unknown styles default to Deliberate
    /// (the default style — balanced, thorough, moderate speed).
    pub fn resolve(name: &str) -> ThinkingStyle {
        let lower = name.to_lowercase();
        let lower = lower.trim();

        match lower {
            // === crewai-rust Analytical cluster → ladybug Convergent ===
            "analytical" | "logical" | "deductive" => ThinkingStyle::Analytical,
            "critical" | "structured" | "methodical" => ThinkingStyle::Systematic,
            "convergent" | "precise" | "rigorous" => ThinkingStyle::Convergent,

            // === crewai-rust Creative cluster → ladybug Divergent ===
            "creative" | "imaginative" | "artistic" => ThinkingStyle::Creative,
            "innovative" | "lateral" | "brainstorming" => ThinkingStyle::Divergent,
            "visionary" | "generative" | "inventive" => ThinkingStyle::Exploratory,

            // === crewai-rust Empathic cluster → ladybug Attention + Speed ===
            "empathic" | "compassionate" | "understanding" => ThinkingStyle::Diffuse,
            "supportive" | "emotional" | "nurturing" => ThinkingStyle::Peripheral,
            "intuitive" | "instinctive" | "gut-feeling" => ThinkingStyle::Intuitive,

            // === crewai-rust Direct cluster → ladybug Focused + Convergent ===
            "direct" | "decisive" | "action-oriented" => ThinkingStyle::Focused,
            "pragmatic" | "efficient" | "results-driven" => ThinkingStyle::Systematic,

            // === crewai-rust Exploratory cluster → ladybug Divergent + Speed ===
            "exploratory" | "curious" | "investigative" => ThinkingStyle::Exploratory,
            "research-oriented" | "questioning" | "open-minded" => ThinkingStyle::Deliberate,

            // === crewai-rust Meta cluster → ladybug Meta ===
            "meta-cognitive" | "metacognitive" | "reflective" => ThinkingStyle::Metacognitive,
            "philosophical" | "abstract" | "systemic" => ThinkingStyle::Metacognitive,
            "integrative" | "holistic" | "synthesizing" => ThinkingStyle::Convergent,

            // === Direct ladybug style names ===
            "focused" => ThinkingStyle::Focused,
            "diffuse" => ThinkingStyle::Diffuse,
            "peripheral" => ThinkingStyle::Peripheral,
            "deliberate" => ThinkingStyle::Deliberate,
            "systematic" => ThinkingStyle::Systematic,
            "divergent" => ThinkingStyle::Divergent,

            // Unknown → safe default
            _ => ThinkingStyle::Deliberate,
        }
    }

    /// Resolve an ExternalStyle to a ThinkingStyle + FieldModulation.
    pub fn resolve_external(ext: &ExternalStyle) -> (ThinkingStyle, FieldModulation) {
        match ext {
            ExternalStyle::Named(name) => {
                let style = Self::resolve(name);
                (style, style.field_modulation())
            }
            ExternalStyle::Cluster { name, weight } => {
                let style = Self::resolve(name);
                let mut modulation = style.field_modulation();
                // Weight scales exploration and fan-out
                modulation.exploration *= weight;
                modulation.fan_out = (modulation.fan_out as f32 * weight).ceil() as usize;
                (style, modulation)
            }
            ExternalStyle::Direct(style) => (*style, style.field_modulation()),
            ExternalStyle::Blend(pairs) => {
                if pairs.is_empty() {
                    return (ThinkingStyle::Deliberate, ThinkingStyle::Deliberate.field_modulation());
                }
                // Weighted blend of field modulations
                let mut modulation = FieldModulation {
                    resonance_threshold: 0.0,
                    fan_out: 0,
                    depth_bias: 0.0,
                    breadth_bias: 0.0,
                    noise_tolerance: 0.0,
                    speed_bias: 0.0,
                    exploration: 0.0,
                };
                let mut total_weight = 0.0f32;
                let mut dominant_style = pairs[0].0;
                let mut max_weight = 0.0f32;

                for (style, weight) in pairs {
                    let m = style.field_modulation();
                    modulation.resonance_threshold += m.resonance_threshold * weight;
                    modulation.fan_out += (m.fan_out as f32 * weight) as usize;
                    modulation.depth_bias += m.depth_bias * weight;
                    modulation.breadth_bias += m.breadth_bias * weight;
                    modulation.noise_tolerance += m.noise_tolerance * weight;
                    modulation.speed_bias += m.speed_bias * weight;
                    modulation.exploration += m.exploration * weight;
                    total_weight += weight;
                    if *weight > max_weight {
                        max_weight = *weight;
                        dominant_style = *style;
                    }
                }

                if total_weight > 0.0 {
                    modulation.resonance_threshold /= total_weight;
                    modulation.depth_bias /= total_weight;
                    modulation.breadth_bias /= total_weight;
                    modulation.noise_tolerance /= total_weight;
                    modulation.speed_bias /= total_weight;
                    modulation.exploration /= total_weight;
                }

                (dominant_style, modulation)
            }
        }
    }

    /// Get the crewai-rust cluster name for a ladybug style.
    pub fn to_cluster_name(style: ThinkingStyle) -> &'static str {
        match style {
            ThinkingStyle::Analytical | ThinkingStyle::Convergent | ThinkingStyle::Systematic => {
                "Analytical"
            }
            ThinkingStyle::Creative | ThinkingStyle::Divergent | ThinkingStyle::Exploratory => {
                "Creative"
            }
            ThinkingStyle::Focused => "Direct",
            ThinkingStyle::Diffuse | ThinkingStyle::Peripheral => "Empathic",
            ThinkingStyle::Intuitive => "Empathic",
            ThinkingStyle::Deliberate => "Exploratory",
            ThinkingStyle::Metacognitive => "Meta",
        }
    }
}

// =============================================================================
// COGNITIVE SNAPSHOT — Cross-crate state exchange
// =============================================================================

/// Cognitive state snapshot for cross-crate exchange.
///
/// This is the data that flows through the Blackboard between crewai-rust,
/// n8n-rs, and ladybug-rs. Compatible with crewai-rust's BlackboardEntry
/// (active_style, coherence, flow_state, confidence, state_fingerprint).
#[derive(Clone, Debug)]
pub struct CognitiveSnapshot {
    /// Active thinking style name (crewai-rust compatible).
    pub active_style: String,

    /// Coherence: triangle coherence × layer coherence (0.0-1.0).
    pub coherence: f32,

    /// Flow state: FLOW / HOLD / BLOCK as string.
    pub flow_state: String,

    /// Confidence from NARS validation (0.0-1.0).
    pub confidence: f32,

    /// Emergence signal (0.0-1.0).
    pub emergence: f32,

    /// Number of triangles in FLOW (0-4).
    pub flow_count: usize,

    /// Processing cycle number.
    pub cycle: u64,

    /// Cognitive signature (human-readable state summary).
    pub signature: String,

    /// State fingerprint for resonance comparison (10K-bit).
    pub state_fingerprint: Option<Vec<u8>>,

    /// Quad-triangle activations (4×3 = 12 floats).
    pub triangle_activations: [f32; 12],

    /// Grammar triangle summary (if text was processed).
    pub grammar_summary: Option<String>,
}

impl CognitiveSnapshot {
    /// Create from a CognitiveFabric state.
    pub fn from_cognitive_state(state: &CognitiveState) -> Self {
        let gate_state = state
            .last_collapse
            .as_ref()
            .map(|d| match d.state {
                GateState::Flow => "FLOW",
                GateState::Hold => "HOLD",
                GateState::Block => "BLOCK",
            })
            .unwrap_or("PENDING");

        let confidence = state
            .last_collapse
            .as_ref()
            .map(|d| if d.can_collapse { d.sd } else { 0.0 })
            .unwrap_or(0.0);

        Self {
            active_style: ThinkingStyleBridge::to_cluster_name(state.style).to_string(),
            coherence: state.coherence,
            flow_state: gate_state.to_string(),
            confidence,
            emergence: state.emergence,
            flow_count: state.triangles.flow_count(),
            cycle: state.cycle,
            signature: format!("{} | {}", state.style, state.triangles.signature()),
            state_fingerprint: None,
            triangle_activations: state.triangles.to_floats(),
            grammar_summary: None,
        }
    }
}

// =============================================================================
// COGNITIVE SERVICE
// =============================================================================

/// The unified cognitive service — mode-aware facade over the full pipeline.
///
/// Wraps CognitiveFabric with Grammar Triangle integration, ThinkingStyle
/// bridging, and mode-appropriate behavior.
pub struct CognitiveService {
    /// Operating mode.
    mode: CognitiveMode,

    /// The cognitive fabric (10-layer + QuadTriangle + CollapseGate).
    fabric: super::fabric::CognitiveFabric,

    /// Last processed grammar triangle (for compound queries).
    last_grammar: Option<GrammarTriangle>,

    /// Last cognitive snapshot (for blackboard exchange).
    last_snapshot: Option<CognitiveSnapshot>,
}

impl CognitiveService {
    /// Create a new cognitive service in the given mode.
    pub fn new(mode: CognitiveMode) -> Self {
        Self {
            mode,
            fabric: super::fabric::CognitiveFabric::new("service"),
            last_grammar: None,
            last_snapshot: None,
        }
    }

    /// Create with a specific thinking style.
    pub fn with_style(mode: CognitiveMode, style: ThinkingStyle) -> Self {
        Self {
            mode,
            fabric: super::fabric::CognitiveFabric::with_style("service", style),
            last_grammar: None,
            last_snapshot: None,
        }
    }

    /// Create with an external style (crewai-rust name or blend).
    pub fn with_external_style(mode: CognitiveMode, ext: &ExternalStyle) -> Self {
        let (style, _modulation) = ThinkingStyleBridge::resolve_external(ext);
        Self::with_style(mode, style)
    }

    // =========================================================================
    // MODE
    // =========================================================================

    /// Get current operating mode.
    pub fn mode(&self) -> CognitiveMode {
        self.mode
    }

    /// Switch operating mode.
    ///
    /// Switching to PassiveRag does NOT reset state — the accumulated
    /// cognitive state remains available for queries. Switching to Brain
    /// resumes full processing from current state.
    pub fn set_mode(&mut self, mode: CognitiveMode) {
        self.mode = mode;
    }

    // =========================================================================
    // STYLE
    // =========================================================================

    /// Set thinking style from an external identifier.
    pub fn set_style_external(&mut self, name: &str) {
        let style = ThinkingStyleBridge::resolve(name);
        self.fabric.set_style(style);
    }

    /// Set thinking style directly.
    pub fn set_style(&mut self, style: ThinkingStyle) {
        self.fabric.set_style(style);
    }

    /// Get current thinking style.
    pub fn style(&self) -> ThinkingStyle {
        self.fabric.style()
    }

    /// Get current field modulation.
    pub fn modulation(&self) -> FieldModulation {
        self.fabric.modulation()
    }

    // =========================================================================
    // QUERY — Available in ALL modes
    // =========================================================================

    /// Resonance query against the cognitive state.
    ///
    /// Available in all modes. Does NOT advance the cognitive cycle.
    /// Returns similarity score against the current QuadTriangle superposition.
    pub fn query_resonance(&self, fingerprint: &Fingerprint) -> f32 {
        self.fabric.triangles.query_resonance(fingerprint)
    }

    /// Query with text input (Grammar Triangle → fingerprint → resonance).
    ///
    /// Available in all modes. Does NOT advance the cognitive cycle.
    pub fn query_text(&self, text: &str) -> TextQueryResult {
        let grammar = GrammarTriangle::from_text(text);
        let fingerprint = grammar.to_fingerprint();
        let resonance = self.fabric.triangles.query_resonance(&fingerprint);

        TextQueryResult {
            resonance,
            grammar_summary: format!("{}", grammar.summary()),
            temporality: grammar.temporality(),
            agency: grammar.agency(),
            valence: grammar.qualia("valence").unwrap_or(0.5),
            is_positive: grammar.is_positive(),
            is_future: grammar.is_future_oriented(),
        }
    }

    /// Check current gate state (without advancing cycle).
    pub fn gate_state(&self) -> Option<GateState> {
        self.fabric
            .recent_collapses(1)
            .last()
            .map(|d| d.state)
    }

    /// Check if in FLOW state.
    pub fn is_flow(&self) -> bool {
        self.fabric.is_flow()
    }

    /// Check if blocked.
    pub fn is_blocked(&self) -> bool {
        self.fabric.is_blocked()
    }

    /// Get cognitive signature.
    pub fn signature(&self) -> String {
        self.fabric.signature()
    }

    // =========================================================================
    // PROCESS — Brain and Orchestrated modes only
    // =========================================================================

    /// Process text through the full cognitive pipeline.
    ///
    /// Grammar Triangle → Fingerprint → CognitiveFabric → CollapseGate.
    ///
    /// Returns the cognitive state snapshot and gate decision.
    /// Only active in Brain or Orchestrated mode. In PassiveRag mode,
    /// returns None (use `query_text` instead).
    pub fn process_text(&mut self, text: &str) -> Option<ProcessResult> {
        if self.mode == CognitiveMode::PassiveRag {
            return None;
        }

        // 1. Grammar Triangle: text → semantic field → fingerprint
        let grammar = GrammarTriangle::from_text(text);
        let fingerprint = grammar.to_fingerprint();

        // 2. Full cognitive cycle through the fabric
        let state = self.fabric.process(&fingerprint);

        // 3. Build snapshot for cross-crate exchange
        let mut snapshot = CognitiveSnapshot::from_cognitive_state(&state);
        snapshot.grammar_summary = Some(format!("{}", grammar.summary()));

        // 4. Store state
        self.last_grammar = Some(grammar);
        self.last_snapshot = Some(snapshot.clone());

        Some(ProcessResult {
            snapshot,
            gate: state.last_collapse.as_ref().map(|d| d.state),
            can_collapse: state.last_collapse.as_ref().map(|d| d.can_collapse).unwrap_or(false),
            fingerprint,
        })
    }

    /// Process a raw fingerprint through the cognitive pipeline.
    ///
    /// Bypasses Grammar Triangle — use when the fingerprint is already
    /// computed (e.g., from a Container or CogRecord).
    pub fn process_fingerprint(&mut self, fingerprint: &Fingerprint) -> Option<ProcessResult> {
        if self.mode == CognitiveMode::PassiveRag {
            return None;
        }

        let state = self.fabric.process(fingerprint);
        let snapshot = CognitiveSnapshot::from_cognitive_state(&state);
        self.last_snapshot = Some(snapshot.clone());

        Some(ProcessResult {
            snapshot,
            gate: state.last_collapse.as_ref().map(|d| d.state),
            can_collapse: state.last_collapse.as_ref().map(|d| d.can_collapse).unwrap_or(false),
            fingerprint: fingerprint.clone(),
        })
    }

    /// Process with an external style override.
    ///
    /// Temporarily sets the style for this cycle, then reverts.
    /// Useful for n8n-rs orchestrated mode where each step can specify a style.
    pub fn process_with_style(
        &mut self,
        text: &str,
        style_name: &str,
    ) -> Option<ProcessResult> {
        let prev_style = self.fabric.style();
        self.set_style_external(style_name);
        let result = self.process_text(text);
        // Revert to previous style after processing
        self.fabric.set_style(prev_style);
        result
    }

    // =========================================================================
    // BLACKBOARD EXCHANGE
    // =========================================================================

    /// Get the last cognitive snapshot for blackboard writing.
    pub fn snapshot(&self) -> Option<&CognitiveSnapshot> {
        self.last_snapshot.as_ref()
    }

    /// Apply an external snapshot to update internal state.
    ///
    /// Used in Orchestrated mode when n8n-rs or crewai-rust writes
    /// a style/state update to the blackboard.
    pub fn apply_snapshot(&mut self, snapshot: &CognitiveSnapshot) {
        // Apply style from snapshot
        let style = ThinkingStyleBridge::resolve(&snapshot.active_style);
        self.fabric.set_style(style);

        // Apply triangle activations if non-zero
        let acts = &snapshot.triangle_activations;
        if acts.iter().any(|a| *a > 0.0) {
            self.fabric.triangles = QuadTriangle::from_floats(*acts);
        }

        self.last_snapshot = Some(snapshot.clone());
    }

    // =========================================================================
    // EVALUATE — Standalone gate evaluation
    // =========================================================================

    /// Evaluate collapse gate for arbitrary candidate scores.
    ///
    /// Available in all modes. Does not affect internal state.
    pub fn evaluate_gate(&self, candidate_scores: &[f32]) -> CollapseDecision {
        evaluate_gate(candidate_scores, true)
    }

    // =========================================================================
    // STATE
    // =========================================================================

    /// Get current processing cycle.
    pub fn cycle(&self) -> u64 {
        self.fabric.cycle
    }

    /// Reset cognitive state to neutral.
    pub fn reset(&mut self) {
        self.fabric.reset();
        self.last_grammar = None;
        self.last_snapshot = None;
    }

    /// Serialize cognitive state to bytes (for persistence).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Mode (1 byte)
        bytes.push(self.mode as u8);

        // Fabric state
        bytes.extend_from_slice(&self.fabric.to_bytes());

        bytes
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.is_empty() {
            return None;
        }

        let mode = match bytes[0] {
            0 => CognitiveMode::PassiveRag,
            1 => CognitiveMode::Brain,
            2 => CognitiveMode::Orchestrated,
            _ => return None,
        };

        let fabric = super::fabric::CognitiveFabric::from_bytes("service", &bytes[1..])?;

        Some(Self {
            mode,
            fabric,
            last_grammar: None,
            last_snapshot: None,
        })
    }
}

// =============================================================================
// RESULT TYPES
// =============================================================================

/// Result of processing text through the cognitive pipeline.
#[derive(Clone, Debug)]
pub struct ProcessResult {
    /// Cognitive state snapshot (for blackboard exchange).
    pub snapshot: CognitiveSnapshot,

    /// Collapse gate state (if evaluated).
    pub gate: Option<GateState>,

    /// Whether the gate permits collapse.
    pub can_collapse: bool,

    /// The fingerprint that was processed.
    pub fingerprint: Fingerprint,
}

/// Result of a text resonance query.
#[derive(Clone, Debug)]
pub struct TextQueryResult {
    /// Resonance score against current cognitive state (0.0-1.0).
    pub resonance: f32,

    /// Grammar triangle summary string.
    pub grammar_summary: String,

    /// Temporal orientation (-1.0 past, 0.0 present, +1.0 future).
    pub temporality: f32,

    /// Agency level (0.0 passive, 1.0 active).
    pub agency: f32,

    /// Valence (0.0 negative, 1.0 positive).
    pub valence: f32,

    /// Is the text positive in valence?
    pub is_positive: bool,

    /// Is the text future-oriented?
    pub is_future: bool,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_style_bridge_analytical_cluster() {
        assert_eq!(ThinkingStyleBridge::resolve("analytical"), ThinkingStyle::Analytical);
        assert_eq!(ThinkingStyleBridge::resolve("logical"), ThinkingStyle::Analytical);
        assert_eq!(ThinkingStyleBridge::resolve("critical"), ThinkingStyle::Systematic);
        assert_eq!(ThinkingStyleBridge::resolve("methodical"), ThinkingStyle::Systematic);
        assert_eq!(ThinkingStyleBridge::resolve("convergent"), ThinkingStyle::Convergent);
    }

    #[test]
    fn test_style_bridge_creative_cluster() {
        assert_eq!(ThinkingStyleBridge::resolve("creative"), ThinkingStyle::Creative);
        assert_eq!(ThinkingStyleBridge::resolve("imaginative"), ThinkingStyle::Creative);
        assert_eq!(ThinkingStyleBridge::resolve("innovative"), ThinkingStyle::Divergent);
        assert_eq!(ThinkingStyleBridge::resolve("visionary"), ThinkingStyle::Exploratory);
    }

    #[test]
    fn test_style_bridge_empathic_cluster() {
        assert_eq!(ThinkingStyleBridge::resolve("empathic"), ThinkingStyle::Diffuse);
        assert_eq!(ThinkingStyleBridge::resolve("supportive"), ThinkingStyle::Peripheral);
        assert_eq!(ThinkingStyleBridge::resolve("intuitive"), ThinkingStyle::Intuitive);
    }

    #[test]
    fn test_style_bridge_direct_cluster() {
        assert_eq!(ThinkingStyleBridge::resolve("direct"), ThinkingStyle::Focused);
        assert_eq!(ThinkingStyleBridge::resolve("decisive"), ThinkingStyle::Focused);
        assert_eq!(ThinkingStyleBridge::resolve("pragmatic"), ThinkingStyle::Systematic);
    }

    #[test]
    fn test_style_bridge_meta_cluster() {
        assert_eq!(ThinkingStyleBridge::resolve("metacognitive"), ThinkingStyle::Metacognitive);
        assert_eq!(ThinkingStyleBridge::resolve("reflective"), ThinkingStyle::Metacognitive);
        assert_eq!(ThinkingStyleBridge::resolve("philosophical"), ThinkingStyle::Metacognitive);
    }

    #[test]
    fn test_style_bridge_unknown_defaults_deliberate() {
        assert_eq!(ThinkingStyleBridge::resolve("unknown_style"), ThinkingStyle::Deliberate);
        assert_eq!(ThinkingStyleBridge::resolve(""), ThinkingStyle::Deliberate);
    }

    #[test]
    fn test_style_bridge_case_insensitive() {
        assert_eq!(ThinkingStyleBridge::resolve("ANALYTICAL"), ThinkingStyle::Analytical);
        assert_eq!(ThinkingStyleBridge::resolve("Creative"), ThinkingStyle::Creative);
        assert_eq!(ThinkingStyleBridge::resolve("  focused  "), ThinkingStyle::Focused);
    }

    #[test]
    fn test_style_bridge_reverse_mapping() {
        assert_eq!(ThinkingStyleBridge::to_cluster_name(ThinkingStyle::Analytical), "Analytical");
        assert_eq!(ThinkingStyleBridge::to_cluster_name(ThinkingStyle::Creative), "Creative");
        assert_eq!(ThinkingStyleBridge::to_cluster_name(ThinkingStyle::Focused), "Direct");
        assert_eq!(ThinkingStyleBridge::to_cluster_name(ThinkingStyle::Diffuse), "Empathic");
        assert_eq!(ThinkingStyleBridge::to_cluster_name(ThinkingStyle::Metacognitive), "Meta");
    }

    #[test]
    fn test_blend_modulation() {
        let blend = ExternalStyle::Blend(vec![
            (ThinkingStyle::Analytical, 0.7),
            (ThinkingStyle::Creative, 0.3),
        ]);
        let (dominant, modulation) = ThinkingStyleBridge::resolve_external(&blend);
        assert_eq!(dominant, ThinkingStyle::Analytical);
        // Blended threshold should be between analytical (0.85) and creative (0.35)
        assert!(modulation.resonance_threshold > 0.35);
        assert!(modulation.resonance_threshold < 0.85);
    }

    #[test]
    fn test_passive_rag_mode_blocks_processing() {
        let mut service = CognitiveService::new(CognitiveMode::PassiveRag);
        assert!(service.process_text("test input").is_none());
    }

    #[test]
    fn test_passive_rag_mode_allows_queries() {
        let service = CognitiveService::new(CognitiveMode::PassiveRag);
        let result = service.query_text("I want to understand this deeply");
        assert!(result.resonance >= 0.0);
        assert!(!result.grammar_summary.is_empty());
    }

    #[test]
    fn test_brain_mode_full_cycle() {
        let mut service = CognitiveService::new(CognitiveMode::Brain);
        let result = service.process_text("I want to understand this deeply").unwrap();

        assert!(result.snapshot.coherence >= 0.0);
        assert!(result.snapshot.coherence <= 1.0);
        assert_eq!(result.snapshot.cycle, 1);
        assert!(!result.snapshot.signature.is_empty());
        assert!(result.snapshot.grammar_summary.is_some());
    }

    #[test]
    fn test_brain_mode_style_switch() {
        let mut service = CognitiveService::new(CognitiveMode::Brain);

        service.set_style_external("creative");
        assert_eq!(service.style(), ThinkingStyle::Creative);

        service.set_style_external("analytical");
        assert_eq!(service.style(), ThinkingStyle::Analytical);
    }

    #[test]
    fn test_process_with_style_override() {
        let mut service = CognitiveService::new(CognitiveMode::Brain);
        service.set_style(ThinkingStyle::Analytical);

        // Process with temporary creative override
        let _result = service.process_with_style("explore new ideas freely", "creative");

        // Should revert to analytical after processing
        assert_eq!(service.style(), ThinkingStyle::Analytical);
    }

    #[test]
    fn test_cognitive_snapshot_exchange() {
        let mut service = CognitiveService::new(CognitiveMode::Brain);
        let _result = service.process_text("think about this").unwrap();

        // Get snapshot
        let snapshot = service.snapshot().unwrap().clone();
        assert!(!snapshot.active_style.is_empty());
        assert!(!snapshot.flow_state.is_empty());

        // Apply snapshot to new service (simulating cross-crate exchange)
        let mut service2 = CognitiveService::new(CognitiveMode::Orchestrated);
        service2.apply_snapshot(&snapshot);
        assert_eq!(
            ThinkingStyleBridge::resolve(&snapshot.active_style),
            service2.style()
        );
    }

    #[test]
    fn test_mode_switch_preserves_state() {
        let mut service = CognitiveService::new(CognitiveMode::Brain);
        service.process_text("accumulate some state").unwrap();

        let cycle_before = service.cycle();
        assert!(cycle_before > 0);

        // Switch to passive — state preserved
        service.set_mode(CognitiveMode::PassiveRag);
        assert_eq!(service.cycle(), cycle_before);

        // Switch back to brain — state still there
        service.set_mode(CognitiveMode::Brain);
        assert_eq!(service.cycle(), cycle_before);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut service = CognitiveService::with_style(CognitiveMode::Brain, ThinkingStyle::Creative);
        service.process_text("some data");

        let bytes = service.to_bytes();
        let restored = CognitiveService::from_bytes(&bytes).unwrap();

        assert_eq!(restored.mode(), CognitiveMode::Brain);
        assert_eq!(restored.style(), ThinkingStyle::Creative);
        assert_eq!(restored.cycle(), service.cycle());
    }

    #[test]
    fn test_grammar_triangle_to_cognitive_pipeline() {
        let mut service = CognitiveService::new(CognitiveMode::Brain);

        // Romeo & Juliet should have different cognitive signature than financial report
        let romeo = service.process_text(
            "But soft! What light through yonder window breaks? It is the east, and Juliet is the sun.",
        ).unwrap();

        let financial = service.process_text(
            "The quarterly financial report shows a 5% increase in revenue.",
        ).unwrap();

        // Both should produce valid snapshots
        assert!(romeo.snapshot.cycle < financial.snapshot.cycle);
        assert!(!romeo.snapshot.signature.is_empty());
        assert!(!financial.snapshot.signature.is_empty());
    }

    #[test]
    fn test_evaluate_gate_standalone() {
        let service = CognitiveService::new(CognitiveMode::PassiveRag);

        // Low dispersion → FLOW
        let decision = service.evaluate_gate(&[0.8, 0.82, 0.79]);
        assert_eq!(decision.state, GateState::Flow);

        // High dispersion → HOLD or BLOCK (SD > 0.15)
        let decision = service.evaluate_gate(&[0.1, 0.9, 0.5]);
        assert_ne!(decision.state, GateState::Flow);
    }
}
