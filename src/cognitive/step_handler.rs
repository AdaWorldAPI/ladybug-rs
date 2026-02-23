//! Step Handler — Routes `lb.*` step types to CognitiveService.
//!
//! Mirrors crewai-rust's `StepHandler` trait interface. When compiled with
//! `vendor-crewai`, a thin adapter bridges this to the actual trait:
//!
//! ```text
//! UnifiedStep { step_type: "lb.query", input: {"text": "..."} }
//!     │
//!     ▼
//! LbStepHandler::handle(&mut step)
//!     │
//!     ├─ lb.query     → CognitiveService.query_text()
//!     ├─ lb.resonate  → CognitiveService.query_resonance()
//!     ├─ lb.process   → CognitiveService.process_text()
//!     ├─ lb.gate      → CognitiveService.evaluate_gate()
//!     ├─ lb.style     → CognitiveService.set_style_external()
//!     └─ lb.snapshot  → CognitiveService.snapshot()
//! ```
//!
//! All handlers write output to `step.output` as JSON and update `step.status`.
//! In one-binary mode, the blackboard carries `CognitiveSnapshot` for cross-
//! subsystem state exchange.

use serde_json::{json, Value};

use super::service::{CognitiveMode, CognitiveService, CognitiveSnapshot};
use crate::contract::types::{StepStatus, UnifiedStep};
use crate::core::Fingerprint;

// =============================================================================
// STEP RESULT TYPE (compatible with crewai-rust's StepResult)
// =============================================================================

/// Result type for step handler operations.
pub type StepResult = Result<(), StepError>;

/// Step handler error.
#[derive(Debug)]
pub struct StepError {
    pub message: String,
}

impl std::fmt::Display for StepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StepError: {}", self.message)
    }
}

impl std::error::Error for StepError {}

impl From<&str> for StepError {
    fn from(s: &str) -> Self {
        Self {
            message: s.to_string(),
        }
    }
}

impl From<String> for StepError {
    fn from(s: String) -> Self {
        Self { message: s }
    }
}

// =============================================================================
// LB STEP TYPES
// =============================================================================

/// Parsed `lb.*` step sub-type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LbStepType {
    /// `lb.query` — text resonance query (all modes).
    Query,
    /// `lb.resonate` — fingerprint similarity search (all modes).
    Resonate,
    /// `lb.process` — full cognitive cycle (Brain/Orchestrated only).
    Process,
    /// `lb.gate` — collapse gate evaluation (all modes).
    Gate,
    /// `lb.style` — change thinking style.
    Style,
    /// `lb.snapshot` — get current cognitive state.
    Snapshot,
    /// `lb.mode` — switch operating mode.
    Mode,
    /// `lb.reset` — reset cognitive state.
    Reset,
    /// Unknown sub-type.
    Unknown(String),
}

impl LbStepType {
    /// Parse step_type string into an LbStepType.
    ///
    /// Expects format "lb.<sub_type>". Returns Unknown for unrecognized sub-types.
    pub fn from_step_type(step_type: &str) -> Self {
        let sub = step_type.strip_prefix("lb.").unwrap_or(step_type);
        match sub {
            "query" => Self::Query,
            "resonate" => Self::Resonate,
            "process" => Self::Process,
            "gate" => Self::Gate,
            "style" => Self::Style,
            "snapshot" => Self::Snapshot,
            "mode" => Self::Mode,
            "reset" => Self::Reset,
            other => Self::Unknown(other.to_string()),
        }
    }

    /// Get the canonical step_type string.
    pub fn step_type(&self) -> String {
        match self {
            Self::Query => "lb.query".to_string(),
            Self::Resonate => "lb.resonate".to_string(),
            Self::Process => "lb.process".to_string(),
            Self::Gate => "lb.gate".to_string(),
            Self::Style => "lb.style".to_string(),
            Self::Snapshot => "lb.snapshot".to_string(),
            Self::Mode => "lb.mode".to_string(),
            Self::Reset => "lb.reset".to_string(),
            Self::Unknown(s) => format!("lb.{}", s),
        }
    }
}

// =============================================================================
// STEP HANDLER
// =============================================================================

/// Handler for `lb.*` step types.
///
/// Routes step commands to the appropriate CognitiveService method.
/// Compatible with crewai-rust's `StepHandler` trait — when compiled
/// in one-binary mode, a thin adapter delegates to this handler.
///
/// # Step Input/Output Contracts
///
/// ## lb.query
/// - Input:  `{ "text": "..." }`
/// - Output: `{ "resonance", "temporality", "agency", "valence", "is_positive", "is_future", "grammar_summary" }`
///
/// ## lb.resonate
/// - Input:  `{ "fingerprint": "<base64>" }` or `{ "text": "..." }`
/// - Output: `{ "similarity" }`
///
/// ## lb.process
/// - Input:  `{ "text": "...", "style": "optional_style_name" }`
/// - Output: `{ "gate", "can_collapse", "coherence", "emergence", "cycle", "signature", "flow_count", "triangle_activations" }`
///
/// ## lb.gate
/// - Input:  `{ "scores": [0.8, 0.7, 0.85] }`
/// - Output: `{ "state", "sd", "can_collapse", "scores_used" }`
///
/// ## lb.style
/// - Input:  `{ "style": "creative" }` or `{ "style": "analytical", "weight": 0.8 }`
/// - Output: `{ "style", "cluster", "modulation": { ... } }`
///
/// ## lb.snapshot
/// - Input:  `{}` (empty)
/// - Output: full CognitiveSnapshot as JSON
///
/// ## lb.mode
/// - Input:  `{ "mode": "brain" | "passive_rag" | "orchestrated" }`
/// - Output: `{ "mode": "...", "previous": "..." }`
///
/// ## lb.reset
/// - Input:  `{}` (empty)
/// - Output: `{ "reset": true }`
pub struct LbStepHandler {
    service: CognitiveService,
}

impl LbStepHandler {
    /// Create a new handler wrapping the given CognitiveService.
    pub fn new(service: CognitiveService) -> Self {
        Self { service }
    }

    /// Create a handler with default Brain mode service.
    pub fn brain() -> Self {
        Self::new(CognitiveService::new(CognitiveMode::Brain))
    }

    /// Create a handler with default PassiveRag mode service.
    pub fn passive() -> Self {
        Self::new(CognitiveService::new(CognitiveMode::PassiveRag))
    }

    /// Get a reference to the underlying service.
    pub fn service(&self) -> &CognitiveService {
        &self.service
    }

    /// Get a mutable reference to the underlying service.
    pub fn service_mut(&mut self) -> &mut CognitiveService {
        &mut self.service
    }

    /// Get the last cognitive snapshot (for blackboard writing).
    pub fn last_snapshot(&self) -> Option<&CognitiveSnapshot> {
        self.service.snapshot()
    }

    // =========================================================================
    // DISPATCH
    // =========================================================================

    /// Handle a UnifiedStep by dispatching to the appropriate sub-handler.
    ///
    /// Updates `step.status` and `step.output` in-place.
    /// Returns Ok(()) on success, Err on handler failure.
    pub fn handle(&mut self, step: &mut UnifiedStep) -> StepResult {
        step.status = StepStatus::Running;

        let result = match LbStepType::from_step_type(&step.step_type) {
            LbStepType::Query => self.handle_query(step),
            LbStepType::Resonate => self.handle_resonate(step),
            LbStepType::Process => self.handle_process(step),
            LbStepType::Gate => self.handle_gate(step),
            LbStepType::Style => self.handle_style(step),
            LbStepType::Snapshot => self.handle_snapshot(step),
            LbStepType::Mode => self.handle_mode(step),
            LbStepType::Reset => self.handle_reset(step),
            LbStepType::Unknown(t) => {
                step.status = StepStatus::Failed;
                step.error = Some(format!("Unknown lb step type: {}", t));
                return Err(StepError::from(format!("Unknown lb step type: {}", t)));
            }
        };

        if let Err(ref e) = result {
            step.status = StepStatus::Failed;
            step.error = Some(e.message.clone());
        }

        result
    }

    // =========================================================================
    // SUB-HANDLERS
    // =========================================================================

    /// `lb.query` — text resonance query (all modes).
    fn handle_query(&self, step: &mut UnifiedStep) -> StepResult {
        let text = extract_text(&step.input)?;
        let result = self.service.query_text(&text);

        step.output = json!({
            "resonance": result.resonance,
            "temporality": result.temporality,
            "agency": result.agency,
            "valence": result.valence,
            "is_positive": result.is_positive,
            "is_future": result.is_future,
            "grammar_summary": result.grammar_summary,
        });
        step.confidence = Some(result.resonance as f64);
        step.status = StepStatus::Completed;
        Ok(())
    }

    /// `lb.resonate` — fingerprint similarity search (all modes).
    fn handle_resonate(&self, step: &mut UnifiedStep) -> StepResult {
        let fp = extract_fingerprint(&step.input)?;
        let similarity = self.service.query_resonance(&fp);

        step.output = json!({
            "similarity": similarity,
        });
        step.confidence = Some(similarity as f64);
        step.status = StepStatus::Completed;
        Ok(())
    }

    /// `lb.process` — full cognitive cycle (Brain/Orchestrated only).
    fn handle_process(&mut self, step: &mut UnifiedStep) -> StepResult {
        let text = extract_text(&step.input)?;

        // Optional style override for this step
        let style_override = step.input.get("style").and_then(|v| v.as_str());

        let result = if let Some(style_name) = style_override {
            self.service.process_with_style(&text, style_name)
        } else {
            self.service.process_text(&text)
        };

        match result {
            Some(pr) => {
                step.output = json!({
                    "gate": pr.gate.map(|g| format!("{:?}", g)),
                    "can_collapse": pr.can_collapse,
                    "coherence": pr.snapshot.coherence,
                    "emergence": pr.snapshot.emergence,
                    "cycle": pr.snapshot.cycle,
                    "signature": pr.snapshot.signature,
                    "flow_count": pr.snapshot.flow_count,
                    "flow_state": pr.snapshot.flow_state,
                    "active_style": pr.snapshot.active_style,
                    "triangle_activations": pr.snapshot.triangle_activations.to_vec(),
                });
                step.confidence = Some(pr.snapshot.confidence as f64);
                step.reasoning = Some(pr.snapshot.signature.clone());
                step.status = StepStatus::Completed;
                Ok(())
            }
            None => {
                step.status = StepStatus::Failed;
                step.error = Some(format!(
                    "lb.process not available in {:?} mode",
                    self.service.mode()
                ));
                Err(StepError::from("Processing not available in PassiveRag mode"))
            }
        }
    }

    /// `lb.gate` — evaluate collapse gate for arbitrary scores (all modes).
    fn handle_gate(&self, step: &mut UnifiedStep) -> StepResult {
        let scores = extract_scores(&step.input)?;
        let decision = self.service.evaluate_gate(&scores);

        step.output = json!({
            "state": format!("{:?}", decision.state),
            "sd": decision.sd,
            "can_collapse": decision.can_collapse,
            "scores_used": scores.len(),
        });
        step.status = StepStatus::Completed;
        Ok(())
    }

    /// `lb.style` — change thinking style.
    fn handle_style(&mut self, step: &mut UnifiedStep) -> StepResult {
        let style_name = step
            .input
            .get("style")
            .and_then(|v| v.as_str())
            .ok_or("lb.style requires input.style")?;

        self.service.set_style_external(style_name);
        let current = self.service.style();
        let modulation = self.service.modulation();

        step.output = json!({
            "style": format!("{}", current),
            "cluster": super::service::ThinkingStyleBridge::to_cluster_name(current),
            "modulation": {
                "resonance_threshold": modulation.resonance_threshold,
                "fan_out": modulation.fan_out,
                "depth_bias": modulation.depth_bias,
                "breadth_bias": modulation.breadth_bias,
                "noise_tolerance": modulation.noise_tolerance,
                "speed_bias": modulation.speed_bias,
                "exploration": modulation.exploration,
            },
        });
        step.status = StepStatus::Completed;
        Ok(())
    }

    /// `lb.snapshot` — get current cognitive state.
    fn handle_snapshot(&self, step: &mut UnifiedStep) -> StepResult {
        let snapshot = self.service.snapshot();

        step.output = match snapshot {
            Some(snap) => snapshot_to_json(snap),
            None => json!({
                "active_style": format!("{}", self.service.style()),
                "mode": format!("{:?}", self.service.mode()),
                "cycle": self.service.cycle(),
                "note": "No processing cycle completed yet",
            }),
        };
        step.status = StepStatus::Completed;
        Ok(())
    }

    /// `lb.mode` — switch operating mode.
    fn handle_mode(&mut self, step: &mut UnifiedStep) -> StepResult {
        let mode_str = step
            .input
            .get("mode")
            .and_then(|v| v.as_str())
            .ok_or("lb.mode requires input.mode")?;

        let previous = self.service.mode();
        let new_mode = parse_mode(mode_str)?;
        self.service.set_mode(new_mode);

        step.output = json!({
            "mode": format!("{:?}", new_mode),
            "previous": format!("{:?}", previous),
        });
        step.status = StepStatus::Completed;
        Ok(())
    }

    /// `lb.reset` — reset cognitive state.
    fn handle_reset(&mut self, step: &mut UnifiedStep) -> StepResult {
        self.service.reset();

        step.output = json!({
            "reset": true,
            "cycle": self.service.cycle(),
        });
        step.status = StepStatus::Completed;
        Ok(())
    }
}

// =============================================================================
// HELPERS
// =============================================================================

/// Extract "text" field from step input, with fallback to stringified content.
fn extract_text(input: &Value) -> Result<String, StepError> {
    // Try input.text first
    if let Some(text) = input.get("text").and_then(|v| v.as_str()) {
        return Ok(text.to_string());
    }
    // Try input.content (DataEnvelope style)
    if let Some(content) = input.get("content").and_then(|v| v.as_str()) {
        return Ok(content.to_string());
    }
    // If input is a string directly
    if let Some(s) = input.as_str() {
        return Ok(s.to_string());
    }
    Err(StepError::from(
        "Step input must contain 'text' or 'content' string field",
    ))
}

/// Extract a Fingerprint from step input.
///
/// Supports:
/// - `{ "fingerprint": "<base64>" }` — direct base64-encoded fingerprint
/// - `{ "text": "..." }` — compute fingerprint from text via GrammarTriangle
fn extract_fingerprint(input: &Value) -> Result<Fingerprint, StepError> {
    // Try base64 fingerprint
    if let Some(b64) = input.get("fingerprint").and_then(|v| v.as_str()) {
        use base64::Engine;
        use base64::engine::general_purpose::STANDARD as BASE64;

        let bytes = BASE64
            .decode(b64)
            .map_err(|e| StepError::from(format!("Invalid base64 fingerprint: {}", e)))?;

        if bytes.len() < crate::FINGERPRINT_BYTES {
            return Err(StepError::from(format!(
                "Fingerprint too short: {} bytes, expected {}",
                bytes.len(),
                crate::FINGERPRINT_BYTES
            )));
        }

        let mut data = [0u64; crate::FINGERPRINT_U64];
        for (i, chunk) in bytes.chunks_exact(8).take(crate::FINGERPRINT_U64).enumerate() {
            data[i] = u64::from_le_bytes(chunk.try_into().unwrap());
        }
        return Ok(Fingerprint::from_raw(data));
    }

    // Fall back to text → GrammarTriangle → fingerprint
    let text = extract_text(input)?;
    let grammar = crate::grammar::GrammarTriangle::from_text(&text);
    Ok(grammar.to_fingerprint())
}

/// Extract float scores from step input.
fn extract_scores(input: &Value) -> Result<Vec<f32>, StepError> {
    let scores_val = input
        .get("scores")
        .ok_or("lb.gate requires input.scores array")?;

    let arr = scores_val
        .as_array()
        .ok_or("input.scores must be a JSON array")?;

    let scores: Vec<f32> = arr
        .iter()
        .filter_map(|v| v.as_f64().map(|f| f as f32))
        .collect();

    if scores.is_empty() {
        return Err(StepError::from("input.scores must contain at least one number"));
    }

    Ok(scores)
}

/// Parse mode string to CognitiveMode.
fn parse_mode(mode_str: &str) -> Result<CognitiveMode, StepError> {
    match mode_str.to_lowercase().as_str() {
        "passive_rag" | "passive" | "rag" => Ok(CognitiveMode::PassiveRag),
        "brain" => Ok(CognitiveMode::Brain),
        "orchestrated" | "orch" => Ok(CognitiveMode::Orchestrated),
        other => Err(StepError::from(format!(
            "Unknown mode: '{}'. Expected: passive_rag, brain, orchestrated",
            other
        ))),
    }
}

/// Convert CognitiveSnapshot to JSON Value.
pub fn snapshot_to_json(snap: &CognitiveSnapshot) -> Value {
    json!({
        "active_style": snap.active_style,
        "coherence": snap.coherence,
        "flow_state": snap.flow_state,
        "confidence": snap.confidence,
        "emergence": snap.emergence,
        "flow_count": snap.flow_count,
        "cycle": snap.cycle,
        "signature": snap.signature,
        "triangle_activations": snap.triangle_activations.to_vec(),
        "grammar_summary": snap.grammar_summary,
    })
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_step(step_type: &str, input: Value) -> UnifiedStep {
        UnifiedStep {
            step_id: "test-step-1".to_string(),
            execution_id: "exec-1".to_string(),
            step_type: step_type.to_string(),
            runtime: "lb".to_string(),
            name: "test".to_string(),
            status: StepStatus::Pending,
            input,
            output: Value::Null,
            error: None,
            started_at: Utc::now(),
            finished_at: None,
            sequence: 0,
            reasoning: None,
            confidence: None,
            alternatives: None,
        }
    }

    #[test]
    fn test_step_type_parsing() {
        assert_eq!(LbStepType::from_step_type("lb.query"), LbStepType::Query);
        assert_eq!(LbStepType::from_step_type("lb.resonate"), LbStepType::Resonate);
        assert_eq!(LbStepType::from_step_type("lb.process"), LbStepType::Process);
        assert_eq!(LbStepType::from_step_type("lb.gate"), LbStepType::Gate);
        assert_eq!(LbStepType::from_step_type("lb.style"), LbStepType::Style);
        assert_eq!(LbStepType::from_step_type("lb.snapshot"), LbStepType::Snapshot);
        assert_eq!(LbStepType::from_step_type("lb.mode"), LbStepType::Mode);
        assert_eq!(LbStepType::from_step_type("lb.reset"), LbStepType::Reset);
        assert_eq!(
            LbStepType::from_step_type("lb.unknown"),
            LbStepType::Unknown("unknown".to_string())
        );
    }

    #[test]
    fn test_handle_query() {
        let mut handler = LbStepHandler::brain();
        let mut step = make_step("lb.query", json!({"text": "I want to understand this deeply"}));

        handler.handle(&mut step).unwrap();

        assert_eq!(step.status, StepStatus::Completed);
        assert!(step.output.get("resonance").is_some());
        assert!(step.output.get("temporality").is_some());
        assert!(step.output.get("agency").is_some());
        assert!(step.output.get("valence").is_some());
        assert!(step.output.get("grammar_summary").is_some());
        assert!(step.confidence.is_some());
    }

    #[test]
    fn test_handle_resonate_from_text() {
        let mut handler = LbStepHandler::brain();
        let mut step = make_step("lb.resonate", json!({"text": "cognitive resonance search"}));

        handler.handle(&mut step).unwrap();

        assert_eq!(step.status, StepStatus::Completed);
        assert!(step.output.get("similarity").is_some());
    }

    #[test]
    fn test_handle_process_brain_mode() {
        let mut handler = LbStepHandler::brain();
        let mut step = make_step("lb.process", json!({"text": "analyze this concept carefully"}));

        handler.handle(&mut step).unwrap();

        assert_eq!(step.status, StepStatus::Completed);
        assert!(step.output.get("coherence").is_some());
        assert!(step.output.get("cycle").is_some());
        assert!(step.output.get("signature").is_some());
        assert!(step.output.get("flow_state").is_some());
        assert!(step.output.get("triangle_activations").is_some());
    }

    #[test]
    fn test_handle_process_passive_mode_fails() {
        let mut handler = LbStepHandler::passive();
        let mut step = make_step("lb.process", json!({"text": "this should fail in passive mode"}));

        let result = handler.handle(&mut step);

        assert!(result.is_err());
        assert_eq!(step.status, StepStatus::Failed);
    }

    #[test]
    fn test_handle_process_with_style() {
        let mut handler = LbStepHandler::brain();
        let mut step = make_step(
            "lb.process",
            json!({"text": "explore creative possibilities", "style": "creative"}),
        );

        handler.handle(&mut step).unwrap();

        assert_eq!(step.status, StepStatus::Completed);
        // Style should have been applied for this cycle
        let active = step.output.get("active_style").and_then(|v| v.as_str());
        assert!(active.is_some());
    }

    #[test]
    fn test_handle_gate() {
        let mut handler = LbStepHandler::brain();
        let mut step = make_step("lb.gate", json!({"scores": [0.8, 0.82, 0.79]}));

        handler.handle(&mut step).unwrap();

        assert_eq!(step.status, StepStatus::Completed);
        let state = step.output.get("state").and_then(|v| v.as_str()).unwrap();
        assert_eq!(state, "Flow");
    }

    #[test]
    fn test_handle_style() {
        let mut handler = LbStepHandler::brain();
        let mut step = make_step("lb.style", json!({"style": "creative"}));

        handler.handle(&mut step).unwrap();

        assert_eq!(step.status, StepStatus::Completed);
        assert!(step.output.get("style").is_some());
        assert!(step.output.get("cluster").is_some());
        assert!(step.output.get("modulation").is_some());
    }

    #[test]
    fn test_handle_snapshot_no_prior_processing() {
        let mut handler = LbStepHandler::brain();
        let mut step = make_step("lb.snapshot", json!({}));

        handler.handle(&mut step).unwrap();

        assert_eq!(step.status, StepStatus::Completed);
        // Should return basic state info even without processing
        assert!(step.output.get("cycle").is_some());
    }

    #[test]
    fn test_handle_snapshot_after_processing() {
        let mut handler = LbStepHandler::brain();

        // First process something
        let mut process_step = make_step("lb.process", json!({"text": "build some state first"}));
        handler.handle(&mut process_step).unwrap();

        // Now get snapshot
        let mut step = make_step("lb.snapshot", json!({}));
        handler.handle(&mut step).unwrap();

        assert_eq!(step.status, StepStatus::Completed);
        assert!(step.output.get("active_style").is_some());
        assert!(step.output.get("coherence").is_some());
        assert!(step.output.get("flow_state").is_some());
    }

    #[test]
    fn test_handle_mode_switch() {
        let mut handler = LbStepHandler::brain();
        let mut step = make_step("lb.mode", json!({"mode": "passive_rag"}));

        handler.handle(&mut step).unwrap();

        assert_eq!(step.status, StepStatus::Completed);
        assert_eq!(
            step.output.get("mode").and_then(|v| v.as_str()).unwrap(),
            "PassiveRag"
        );
        assert_eq!(
            step.output
                .get("previous")
                .and_then(|v| v.as_str())
                .unwrap(),
            "Brain"
        );
        assert_eq!(handler.service().mode(), CognitiveMode::PassiveRag);
    }

    #[test]
    fn test_handle_reset() {
        let mut handler = LbStepHandler::brain();

        // Process to accumulate state
        let mut process_step = make_step("lb.process", json!({"text": "accumulate state"}));
        handler.handle(&mut process_step).unwrap();
        assert!(handler.service().cycle() > 0);

        // Reset
        let mut step = make_step("lb.reset", json!({}));
        handler.handle(&mut step).unwrap();

        assert_eq!(step.status, StepStatus::Completed);
        assert_eq!(handler.service().cycle(), 0);
    }

    #[test]
    fn test_unknown_step_type_fails() {
        let mut handler = LbStepHandler::brain();
        let mut step = make_step("lb.nonexistent", json!({}));

        let result = handler.handle(&mut step);

        assert!(result.is_err());
        assert_eq!(step.status, StepStatus::Failed);
        assert!(step.error.is_some());
    }

    #[test]
    fn test_missing_text_input_fails() {
        let mut handler = LbStepHandler::brain();
        let mut step = make_step("lb.query", json!({"not_text": 42}));

        let result = handler.handle(&mut step);

        assert!(result.is_err());
        assert_eq!(step.status, StepStatus::Failed);
    }

    #[test]
    fn test_gate_with_invalid_scores() {
        let mut handler = LbStepHandler::brain();
        let mut step = make_step("lb.gate", json!({"scores": "not an array"}));

        let result = handler.handle(&mut step);

        assert!(result.is_err());
        assert_eq!(step.status, StepStatus::Failed);
    }

    #[test]
    fn test_sequential_steps() {
        let mut handler = LbStepHandler::brain();

        // Step 1: Set style
        let mut s1 = make_step("lb.style", json!({"style": "analytical"}));
        handler.handle(&mut s1).unwrap();

        // Step 2: Process with that style
        let mut s2 = make_step("lb.process", json!({"text": "analyze this data"}));
        handler.handle(&mut s2).unwrap();

        // Step 3: Get snapshot
        let mut s3 = make_step("lb.snapshot", json!({}));
        handler.handle(&mut s3).unwrap();

        // Step 4: Query resonance
        let mut s4 = make_step("lb.query", json!({"text": "similar analytical concept"}));
        handler.handle(&mut s4).unwrap();

        // All should complete
        assert_eq!(s1.status, StepStatus::Completed);
        assert_eq!(s2.status, StepStatus::Completed);
        assert_eq!(s3.status, StepStatus::Completed);
        assert_eq!(s4.status, StepStatus::Completed);
    }
}
