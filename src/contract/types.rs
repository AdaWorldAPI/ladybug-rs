//! Unified Execution Contract — Canonical types shared across ada-n8n, crewai-rust, and ladybug-rs.
//!
//! Source of truth: ada-n8n/src/contract/types.rs (deployed, running).
//! These types MUST serialize identically across all three repos.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Execution step status — snake_case serialization matches PostgreSQL enum values.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Skipped,
}

/// A single execution step within a workflow.
///
/// Steps are namespaced by `step_type`:
/// - `n8n.*` — n8n integration nodes
/// - `crew.*` — crewAI agent tasks
/// - `lb.*` — ladybug cognitive operations
/// - `core.*` — shared control flow
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedStep {
    pub step_id: String,
    pub execution_id: String,
    pub step_type: String,
    pub runtime: String,
    pub name: String,
    pub status: StepStatus,
    pub input: Value,
    pub output: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub started_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<DateTime<Utc>>,
    pub sequence: i32,
    // Decision trail (populated by crew.agent steps):
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alternatives: Option<Value>,
}

/// Top-level execution record spanning one workflow run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedExecution {
    pub execution_id: String,
    pub runtime: String,
    pub workflow_name: String,
    pub status: StepStatus,
    pub trigger: String,
    pub input: Value,
    pub output: Value,
    pub started_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<DateTime<Utc>>,
    pub step_count: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fork_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fork_parent: Option<String>,
}

/// Data envelope — the wire format for passing data between runtimes.
///
/// Content types:
/// - `text/plain` — raw text
/// - `application/json` — structured JSON
/// - `application/x-fingerprint` — base64-encoded 16,384-bit HDR fingerprint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataEnvelope {
    pub step_id: String,
    pub output_key: String,
    pub content_type: String,
    pub content: Value,
    pub metadata: EnvelopeMetadata,
}

/// Envelope metadata — all fields optional, skip if None.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EnvelopeMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub epoch: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}
