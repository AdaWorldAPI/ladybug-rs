//! V1 JSON backward compatibility — the current crewai-rust / n8n-rs wire format.
//!
//! These types serialize identically to the existing `UnifiedStep` and `DataEnvelope`
//! in crewai-rust and n8n-rs. The `From` conversions bridge V1 ↔ CogRecord.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::container::Container;
use crate::geometry::ContainerGeometry;
use crate::record::CogRecord;

// ============================================================================
// V1 StepStatus
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum V1StepStatus {
    #[default]
    Pending,
    Running,
    Completed,
    Failed,
    Skipped,
}

// ============================================================================
// V1 UnifiedStep (crewai-rust / n8n-rs schema)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct V1UnifiedStep {
    pub step_id: String,
    pub execution_id: String,
    pub step_type: String,
    pub name: String,
    #[serde(default)]
    pub status: V1StepStatus,
    #[serde(default)]
    pub sequence: i32,
    #[serde(default)]
    pub input: Value,
    #[serde(default)]
    pub output: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub started_at: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alternatives: Option<Value>,
}

impl Default for V1UnifiedStep {
    fn default() -> Self {
        Self {
            step_id: String::new(),
            execution_id: String::new(),
            step_type: String::new(),
            name: String::new(),
            status: V1StepStatus::Pending,
            sequence: 0,
            input: Value::Null,
            output: Value::Null,
            error: None,
            started_at: None,
            finished_at: None,
            reasoning: None,
            confidence: None,
            alternatives: None,
        }
    }
}

// ============================================================================
// V1 DataEnvelope (crewai-rust / n8n-rs schema)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct V1DataEnvelope {
    pub data: Value,
    pub metadata: V1EnvelopeMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct V1EnvelopeMetadata {
    pub source_step: String,
    #[serde(default)]
    pub confidence: f64,
    #[serde(default)]
    pub epoch: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,

    // --- 10-Layer Cognitive Awareness (backward-compatible, all optional) ---

    /// Dominant cognitive layer that produced this output (0-9 → L1-L10).
    /// Used for cross-agent routing: savant agents operate at L1-L5,
    /// orchestrator agents at L6-L10.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub dominant_layer: Option<u8>,

    /// 10-layer activation snapshot: [f32; 10] serialized as JSON array.
    /// Encodes the satisfaction state at the time of output, enabling
    /// cross-agent awareness through shared metadata.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub layer_activations: Option<Vec<f32>>,

    /// NARS frequency component (0.0-1.0) from L9 validation.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub nars_frequency: Option<f64>,

    /// Calibration error (Brier score) from MetaCognition.
    /// Lower = better calibrated. Consumers can use this to weight
    /// evidence from this source during L8 Integration.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub calibration_error: Option<f64>,
}

// ============================================================================
// V1 LadybugEnvelope (ladybug-rs schema)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct V1LadybugEnvelope {
    pub step_id: String,
    pub output_key: String,
    pub content_type: String,
    pub content: Value,
    pub metadata: V1LadybugMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct V1LadybugMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub epoch: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,

    // --- 10-Layer Cognitive Awareness ---

    /// Dominant cognitive layer (0-9 → L1-L10).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub dominant_layer: Option<u8>,

    /// 10-layer activation snapshot.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub layer_activations: Option<Vec<f32>>,

    /// NARS frequency from L9 validation.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub nars_frequency: Option<f64>,

    /// Calibration error (Brier score).
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub calibration_error: Option<f64>,
}

// ============================================================================
// V1 Delegation (crewai-rust / n8n-rs schema)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct V1StepDelegationRequest {
    pub step: V1UnifiedStep,
    pub input: V1DataEnvelope,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct V1StepDelegationResponse {
    pub output: V1DataEnvelope,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step: Option<V1UnifiedStep>,
}

// ============================================================================
// Conversions: V1UnifiedStep ↔ CogRecord
// ============================================================================

fn hash_step_id(step_id: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    step_id.hash(&mut hasher);
    hasher.finish()
}

fn infer_runtime(step_type: &str) -> String {
    match step_type.split('.').next() {
        Some("crew") => "crewai-rust",
        Some("lb") => "ladybug-rs",
        Some("n8n") => "n8n-rs",
        _ => "unknown",
    }
    .to_string()
}

impl From<&V1UnifiedStep> for CogRecord {
    fn from(step: &V1UnifiedStep) -> CogRecord {
        let mut record = CogRecord::new(ContainerGeometry::Cam);
        {
            let mut m = record.meta_view_mut();
            m.set_dn_addr(hash_step_id(&step.step_id));
            m.set_created_ms(
                step.started_at
                    .map(|t| t.timestamp_millis() as u32)
                    .unwrap_or(0),
            );
            m.set_modified_ms(
                step.finished_at
                    .map(|t| t.timestamp_millis() as u32)
                    .unwrap_or(0),
            );
            if let Some(conf) = step.confidence {
                m.set_nars_frequency(1.0);
                m.set_nars_confidence(conf as f32);
            }
        }

        // Encode step_type + output into the content container
        let json = serde_json::json!({
            "step_type": step.step_type,
            "name": step.name,
            "runtime": infer_runtime(&step.step_type),
            "output": step.output,
            "reasoning": step.reasoning,
        });
        let bytes = serde_json::to_vec(&json).unwrap_or_default();
        record.content[0] = content_from_bytes(&bytes);

        record
    }
}

impl From<&CogRecord> for V1UnifiedStep {
    fn from(record: &CogRecord) -> V1UnifiedStep {
        let m = record.meta_view();
        V1UnifiedStep {
            step_id: format!("{:016X}", m.dn_addr()),
            confidence: {
                let c = m.nars_confidence();
                if c > 0.0 {
                    Some(c as f64)
                } else {
                    None
                }
            },
            ..Default::default()
        }
    }
}

// ============================================================================
// Conversions: V1DataEnvelope ↔ CogRecord
// ============================================================================

impl From<&V1DataEnvelope> for CogRecord {
    fn from(envelope: &V1DataEnvelope) -> CogRecord {
        let mut record = CogRecord::new(ContainerGeometry::Cam);
        {
            let mut m = record.meta_view_mut();
            m.set_dn_addr(hash_step_id(&envelope.metadata.source_step));
            m.set_nars_confidence(envelope.metadata.confidence as f32);
            if let Some(freq) = envelope.metadata.nars_frequency {
                m.set_nars_frequency(freq as f32);
            }
            // Layer activations are carried in the JSON metadata, not
            // packed into container words (container layer markers are
            // 5-byte 7-layer format; the 10-layer data travels in the
            // serde envelope until container format is upgraded).
        }
        let bytes = serde_json::to_vec(&envelope.data).unwrap_or_default();
        record.content[0] = content_from_bytes(&bytes);
        record
    }
}

impl From<&CogRecord> for V1DataEnvelope {
    fn from(record: &CogRecord) -> V1DataEnvelope {
        let m = record.meta_view();

        // Extract layer markers from container (7-layer format in W12-W15).
        // We extract the available 7 layers and pad with zeros for L8-L10.
        // Full 10-layer data travels in the JSON envelope when available.
        let mut layer_activations = Vec::with_capacity(10);
        let mut dominant: Option<(u8, f32)> = None;
        for i in 0..7 {
            let marker = m.layer_marker(i);
            let activation = marker.0 as f32 / 255.0;
            layer_activations.push(activation);
            if dominant.is_none() || activation > dominant.unwrap().1 {
                dominant = Some((i as u8, activation));
            }
        }
        // Pad L8-L10 with zero (not yet stored in container metadata)
        layer_activations.extend(std::iter::repeat_n(0.0, 3));

        V1DataEnvelope {
            data: Value::Null,
            metadata: V1EnvelopeMetadata {
                source_step: format!("{:016X}", m.dn_addr()),
                confidence: m.nars_confidence() as f64,
                epoch: 0,
                version: None,
                dominant_layer: dominant.map(|(i, _)| i),
                layer_activations: Some(layer_activations),
                nars_frequency: Some(m.nars_frequency() as f64),
                calibration_error: None, // Not stored in container metadata
            },
        }
    }
}

// ============================================================================
// Conversions: V1LadybugEnvelope ↔ CogRecord
// ============================================================================

impl From<&V1LadybugEnvelope> for CogRecord {
    fn from(envelope: &V1LadybugEnvelope) -> CogRecord {
        let mut record = CogRecord::new(ContainerGeometry::Cam);
        {
            let mut m = record.meta_view_mut();
            m.set_dn_addr(hash_step_id(&envelope.step_id));
            if let Some(conf) = envelope.metadata.confidence {
                m.set_nars_confidence(conf as f32);
            }
        }
        let bytes = serde_json::to_vec(&envelope.content).unwrap_or_default();
        record.content[0] = content_from_bytes(&bytes);
        record
    }
}

impl From<&CogRecord> for V1LadybugEnvelope {
    fn from(record: &CogRecord) -> V1LadybugEnvelope {
        let m = record.meta_view();
        V1LadybugEnvelope {
            step_id: format!("{:016X}", m.dn_addr()),
            output_key: "default".to_string(),
            content_type: "application/json".to_string(),
            content: Value::Null,
            metadata: V1LadybugMetadata {
                agent_id: None,
                confidence: {
                    let c = m.nars_confidence();
                    if c > 0.0 {
                        Some(c as f64)
                    } else {
                        None
                    }
                },
                epoch: None,
                version: None,
                dominant_layer: None,
                layer_activations: None,
                nars_frequency: {
                    let f = m.nars_frequency();
                    if f > 0.0 {
                        Some(f as f64)
                    } else {
                        None
                    }
                },
                calibration_error: None,
            },
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Hash arbitrary bytes into a Container (deterministic, lossy).
fn content_from_bytes(bytes: &[u8]) -> Container {
    let mut c = Container::zero();
    // SplitMix64-style expansion from content hash
    let mut h = 0xcbf29ce484222325u64; // FNV offset
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3); // FNV prime
    }
    // Expand hash to fill container
    for w in c.words.iter_mut() {
        h = h.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = h;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        *w = z ^ (z >> 31);
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v1_step_status_serde() {
        let json = serde_json::to_string(&V1StepStatus::Completed).unwrap();
        assert_eq!(json, "\"completed\"");
        let back: V1StepStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(back, V1StepStatus::Completed);
    }

    #[test]
    fn test_v1_to_cogrecord_roundtrip() {
        let v1 = V1UnifiedStep {
            step_id: "test-123".into(),
            execution_id: "exec-1".into(),
            step_type: "crew.agent".into(),
            name: "Research".into(),
            confidence: Some(0.92),
            ..Default::default()
        };
        let record: CogRecord = (&v1).into();
        let v1_back: V1UnifiedStep = (&record).into();
        // step_id is hashed, not recoverable, but confidence roundtrips
        assert!((v1_back.confidence.unwrap() - 0.92).abs() < 0.01);
    }

    #[test]
    fn test_v1_envelope_serde() {
        let env = V1DataEnvelope {
            data: serde_json::json!({"key": "value"}),
            metadata: V1EnvelopeMetadata {
                source_step: "step-1".into(),
                confidence: 1.0,
                epoch: 12345,
                version: None,
                dominant_layer: None,
                layer_activations: None,
                nars_frequency: None,
                calibration_error: None,
            },
        };
        let json = serde_json::to_string(&env).unwrap();
        let back: V1DataEnvelope = serde_json::from_str(&json).unwrap();
        assert_eq!(back.metadata.source_step, "step-1");
    }

    #[test]
    fn test_infer_runtime() {
        assert_eq!(infer_runtime("crew.agent"), "crewai-rust");
        assert_eq!(infer_runtime("lb.resonate"), "ladybug-rs");
        assert_eq!(infer_runtime("n8n.set"), "n8n-rs");
        assert_eq!(infer_runtime("other"), "unknown");
    }

    #[test]
    fn test_v1_delegation_serde() {
        let step = V1UnifiedStep {
            step_id: "s1".into(),
            step_type: "crew.agent".into(),
            ..Default::default()
        };
        let input = V1DataEnvelope {
            data: serde_json::json!({"query": "rust"}),
            metadata: V1EnvelopeMetadata {
                source_step: "trigger".into(),
                confidence: 1.0,
                epoch: 0,
                version: None,
                dominant_layer: None,
                layer_activations: None,
                nars_frequency: None,
                calibration_error: None,
            },
        };
        let req = V1StepDelegationRequest { step, input };
        let json = serde_json::to_string(&req).unwrap();
        let back: V1StepDelegationRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.step.step_type, "crew.agent");
    }
}
