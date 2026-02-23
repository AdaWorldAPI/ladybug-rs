//! Ladybug Subsystem — Lifecycle bridge for cognitive integration.
//!
//! Implements the subsystem pattern compatible with crewai-rust's `Subsystem`
//! trait. When compiled in one-binary mode (via `vendor-crewai`), the
//! orchestration module's thin adapter delegates to this implementation.
//!
//! ```text
//! SubsystemRegistry::build()
//!     │
//!     ├─ LadybugSubsystem::step_handler()  → LbStepHandler (lb.* dispatch)
//!     ├─ LadybugSubsystem::init_blackboard() → seed cognitive state
//!     ├─ LadybugSubsystem::register_agents() → A2A: cognitive analyst, style advisor
//!     └─ LadybugSubsystem::install_hooks()  → style sync, gate propagation
//!
//! At runtime:
//!     Pipeline::run(execution)
//!         → StepRouter dispatches lb.* → LbStepHandler
//!         → Handler reads/writes Blackboard via CognitiveSnapshot
//!         → Cognitive state shared across subsystems (zero-serde)
//! ```
//!
//! # Integration with orchestration/
//!
//! The `orchestration/` module provides the external-facing bridge (Agent cards,
//! A2A protocol, Persona registry, SemanticKernel). This subsystem provides
//! the _internal_ cognitive lifecycle. They connect through BindSpace:
//!
//! - orchestration writes agent cards to 0x0C, thinking templates to 0x0D
//! - subsystem writes cognitive snapshots to 0x0E (blackboard zone)
//! - step handler reads/writes through CognitiveService
//! - A2A messages flow through 0x0F channels

use super::service::{CognitiveMode, CognitiveService, ThinkingStyleBridge};
use super::step_handler::LbStepHandler;
use super::style::ThinkingStyle;
use serde_json::{json, Value};

// =============================================================================
// SUBSYSTEM
// =============================================================================

/// Ladybug cognitive subsystem.
///
/// Manages the cognitive service lifecycle and provides step handling
/// for `lb.*` step types. Compatible with crewai-rust's `Subsystem` trait.
///
/// # Modes
///
/// The subsystem adapts its behavior based on the CognitiveMode:
///
/// - **PassiveRag**: Minimal initialization. Only resonance queries enabled.
///   Used when crewai-rust drives all orchestration.
///
/// - **Brain**: Full initialization with style fingerprints, satisfaction gate,
///   and QuadTriangle state. ladybug-rs drives the cognitive loop.
///
/// - **Orchestrated**: Initialized but externally driven. Responds to step
///   commands and publishes state via blackboard snapshots.
pub struct LadybugSubsystem {
    mode: CognitiveMode,
    initial_style: ThinkingStyle,
    version: String,
}

impl LadybugSubsystem {
    /// Create a new subsystem with the given mode.
    pub fn new(mode: CognitiveMode) -> Self {
        Self {
            mode,
            initial_style: ThinkingStyle::Deliberate,
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Create with a specific initial thinking style.
    pub fn with_style(mode: CognitiveMode, style: ThinkingStyle) -> Self {
        Self {
            mode,
            initial_style: style,
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Create in Brain mode (ladybug-rs drives cognitive loop).
    pub fn brain() -> Self {
        Self::new(CognitiveMode::Brain)
    }

    /// Create in PassiveRag mode (resonance queries only).
    pub fn passive() -> Self {
        Self::new(CognitiveMode::PassiveRag)
    }

    /// Create in Orchestrated mode (external step-driven).
    pub fn orchestrated() -> Self {
        Self::new(CognitiveMode::Orchestrated)
    }

    // =========================================================================
    // SUBSYSTEM TRAIT METHODS (compatible with crewai-rust::Subsystem)
    // =========================================================================

    /// Human-readable subsystem name.
    pub fn name(&self) -> &str {
        "ladybug-rs"
    }

    /// Domain prefix this subsystem handles.
    pub fn domain(&self) -> &str {
        "lb"
    }

    /// Subsystem version.
    pub fn version(&self) -> &str {
        &self.version
    }

    /// Create the step handler for this subsystem.
    ///
    /// Returns an LbStepHandler configured with the subsystem's mode and style.
    /// In crewai-rust's Subsystem trait, this returns `Box<dyn StepHandler>`.
    pub fn step_handler(&self) -> LbStepHandler {
        LbStepHandler::new(CognitiveService::with_style(self.mode, self.initial_style))
    }

    /// Initialize blackboard state.
    ///
    /// Seeds the blackboard with a default CognitiveSnapshot so other
    /// subsystems can read cognitive state from the start.
    ///
    /// Returns the initial state as JSON (the actual trait version takes
    /// `&mut Blackboard` — the JSON is written to bb by the adapter).
    pub fn init_state(&self) -> Value {
        let style_name = ThinkingStyleBridge::to_cluster_name(self.initial_style);
        let modulation = self.initial_style.field_modulation();

        json!({
            "subsystem": "ladybug-rs",
            "version": self.version,
            "mode": format!("{:?}", self.mode),
            "cognitive_snapshot": {
                "active_style": style_name,
                "coherence": 0.5,
                "flow_state": "PENDING",
                "confidence": 0.0,
                "emergence": 0.0,
                "flow_count": 0,
                "cycle": 0,
                "signature": format!("{} | neutral", self.initial_style),
                "triangle_activations": vec![0.5f32; 12],
            },
            "modulation": {
                "resonance_threshold": modulation.resonance_threshold,
                "fan_out": modulation.fan_out,
                "depth_bias": modulation.depth_bias,
                "breadth_bias": modulation.breadth_bias,
                "noise_tolerance": modulation.noise_tolerance,
                "speed_bias": modulation.speed_bias,
                "exploration": modulation.exploration,
            },
            "capabilities": [
                "lb.query",
                "lb.resonate",
                "lb.process",
                "lb.gate",
                "lb.style",
                "lb.snapshot",
                "lb.mode",
                "lb.reset",
            ],
        })
    }

    /// Register cognitive agents in the A2A registry.
    ///
    /// Exposes two cognitive agents discoverable by other subsystems:
    /// 1. `lb:analyst` — handles resonance queries and cognitive analysis
    /// 2. `lb:advisor` — provides thinking style recommendations
    ///
    /// Returns agent descriptors as JSON (the actual trait version writes
    /// directly to the A2A registry in BindSpace 0x0F).
    pub fn agent_descriptors(&self) -> Vec<AgentDescriptor> {
        vec![
            AgentDescriptor {
                id: "lb:analyst".to_string(),
                name: "Cognitive Analyst".to_string(),
                role: "analyst".to_string(),
                capabilities: vec![
                    "resonance_query".to_string(),
                    "grammar_analysis".to_string(),
                    "gate_evaluation".to_string(),
                    "fingerprint_search".to_string(),
                ],
                step_types: vec![
                    "lb.query".to_string(),
                    "lb.resonate".to_string(),
                    "lb.gate".to_string(),
                ],
            },
            AgentDescriptor {
                id: "lb:advisor".to_string(),
                name: "Style Advisor".to_string(),
                role: "advisor".to_string(),
                capabilities: vec![
                    "style_recommendation".to_string(),
                    "cognitive_profiling".to_string(),
                    "modulation_tuning".to_string(),
                ],
                step_types: vec![
                    "lb.style".to_string(),
                    "lb.process".to_string(),
                    "lb.snapshot".to_string(),
                ],
            },
        ]
    }

    /// Shutdown hook — called when the pipeline is torn down.
    pub fn shutdown(&self) {
        // No-op for now. In future: flush BindSpace to persistent storage,
        // save final CognitiveSnapshot, close Arrow Flight connections.
    }
}

// =============================================================================
// AGENT DESCRIPTOR
// =============================================================================

/// Descriptor for a cognitive agent registered in the A2A registry.
///
/// Compatible with crewai-rust's A2A agent card format.
/// In one-binary mode, this maps to BindSpace 0x0F:slot.
#[derive(Clone, Debug)]
pub struct AgentDescriptor {
    /// Unique agent ID (format: "subsystem:name").
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Agent role (analyst, advisor, orchestrator, etc.).
    pub role: String,
    /// Capabilities this agent provides.
    pub capabilities: Vec<String>,
    /// Step types this agent can handle.
    pub step_types: Vec<String>,
}

impl AgentDescriptor {
    /// Convert to JSON for blackboard/A2A exchange.
    pub fn to_json(&self) -> Value {
        json!({
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "capabilities": self.capabilities,
            "step_types": self.step_types,
        })
    }
}

// =============================================================================
// SUBSYSTEM BUILDER
// =============================================================================

/// Builder for configuring a LadybugSubsystem.
///
/// ```rust,ignore
/// let subsystem = SubsystemBuilder::new()
///     .mode(CognitiveMode::Brain)
///     .style(ThinkingStyle::Analytical)
///     .build();
/// ```
pub struct SubsystemBuilder {
    mode: CognitiveMode,
    style: ThinkingStyle,
}

impl SubsystemBuilder {
    pub fn new() -> Self {
        Self {
            mode: CognitiveMode::Brain,
            style: ThinkingStyle::Deliberate,
        }
    }

    pub fn mode(mut self, mode: CognitiveMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn style(mut self, style: ThinkingStyle) -> Self {
        self.style = style;
        self
    }

    pub fn build(self) -> LadybugSubsystem {
        LadybugSubsystem::with_style(self.mode, self.style)
    }
}

impl Default for SubsystemBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::types::{StepStatus, UnifiedStep};
    use chrono::Utc;
    use serde_json::json;

    #[test]
    fn test_subsystem_creation() {
        let sub = LadybugSubsystem::new(CognitiveMode::Brain);
        assert_eq!(sub.name(), "ladybug-rs");
        assert_eq!(sub.domain(), "lb");
        assert!(!sub.version().is_empty());
    }

    #[test]
    fn test_subsystem_brain_mode() {
        let sub = LadybugSubsystem::brain();
        assert_eq!(sub.mode, CognitiveMode::Brain);
    }

    #[test]
    fn test_subsystem_passive_mode() {
        let sub = LadybugSubsystem::passive();
        assert_eq!(sub.mode, CognitiveMode::PassiveRag);
    }

    #[test]
    fn test_step_handler_creation() {
        let sub = LadybugSubsystem::brain();
        let mut handler = sub.step_handler();

        // Handler should be functional
        let mut step = UnifiedStep {
            step_id: "test-1".to_string(),
            execution_id: "exec-1".to_string(),
            step_type: "lb.query".to_string(),
            runtime: "lb".to_string(),
            name: "test".to_string(),
            status: StepStatus::Pending,
            input: json!({"text": "test query"}),
            output: Value::Null,
            error: None,
            started_at: Utc::now(),
            finished_at: None,
            sequence: 0,
            reasoning: None,
            confidence: None,
            alternatives: None,
        };

        handler.handle(&mut step).unwrap();
        assert_eq!(step.status, StepStatus::Completed);
    }

    #[test]
    fn test_init_state() {
        let sub = LadybugSubsystem::with_style(CognitiveMode::Brain, ThinkingStyle::Analytical);
        let state = sub.init_state();

        assert_eq!(state["subsystem"], "ladybug-rs");
        assert_eq!(state["mode"], "Brain");
        assert!(state["cognitive_snapshot"].is_object());
        assert!(state["capabilities"].is_array());
    }

    #[test]
    fn test_agent_descriptors() {
        let sub = LadybugSubsystem::brain();
        let agents = sub.agent_descriptors();

        assert_eq!(agents.len(), 2);
        assert_eq!(agents[0].id, "lb:analyst");
        assert_eq!(agents[1].id, "lb:advisor");
        assert!(agents[0].capabilities.contains(&"resonance_query".to_string()));
        assert!(agents[1].capabilities.contains(&"style_recommendation".to_string()));
    }

    #[test]
    fn test_agent_descriptor_to_json() {
        let sub = LadybugSubsystem::brain();
        let agents = sub.agent_descriptors();
        let json_val = agents[0].to_json();

        assert_eq!(json_val["id"], "lb:analyst");
        assert_eq!(json_val["name"], "Cognitive Analyst");
        assert!(json_val["capabilities"].is_array());
    }

    #[test]
    fn test_builder() {
        let sub = SubsystemBuilder::new()
            .mode(CognitiveMode::Orchestrated)
            .style(ThinkingStyle::Creative)
            .build();

        assert_eq!(sub.mode, CognitiveMode::Orchestrated);
        assert_eq!(sub.initial_style, ThinkingStyle::Creative);
    }

    #[test]
    fn test_full_lifecycle() {
        // 1. Create subsystem
        let sub = LadybugSubsystem::brain();

        // 2. Initialize state
        let init = sub.init_state();
        assert!(init["cognitive_snapshot"]["cycle"].as_u64() == Some(0));

        // 3. Create handler
        let mut handler = sub.step_handler();

        // 4. Process some steps
        let mut process_step = UnifiedStep {
            step_id: "s1".to_string(),
            execution_id: "e1".to_string(),
            step_type: "lb.process".to_string(),
            runtime: "lb".to_string(),
            name: "cognitive cycle".to_string(),
            status: StepStatus::Pending,
            input: json!({"text": "analyze the cognitive architecture"}),
            output: Value::Null,
            error: None,
            started_at: Utc::now(),
            finished_at: None,
            sequence: 0,
            reasoning: None,
            confidence: None,
            alternatives: None,
        };
        handler.handle(&mut process_step).unwrap();
        assert_eq!(process_step.status, StepStatus::Completed);

        // 5. Get snapshot
        let snap = handler.last_snapshot();
        assert!(snap.is_some());

        // 6. Verify cycle advanced
        let snap = snap.unwrap();
        assert_eq!(snap.cycle, 1);

        // 7. Shutdown
        sub.shutdown();
    }
}
