//! Cross-runtime delegation types.

use serde::{Deserialize, Serialize};

use crate::record::CogRecord;

/// Request to delegate an operation to another runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelegationRequest {
    /// CAM opcode to execute.
    pub opcode: u16,
    /// The record to operate on (serialized).
    #[serde(with = "cogrecord_serde")]
    pub record: CogRecord,
    /// Target runtime: "crewai-rust", "n8n-rs", "ladybug-rs".
    pub target_runtime: String,
    /// Optional specific agent/workflow ID.
    pub target_id: Option<String>,
}

/// Response from a delegated operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelegationResponse {
    /// The result record.
    #[serde(with = "cogrecord_serde")]
    pub record: CogRecord,
    /// Opcode that was actually executed.
    pub opcode_executed: u16,
    /// Execution time in microseconds.
    pub elapsed_us: u64,
}

/// Serde support for CogRecord (serialize as base64 containers).
mod cogrecord_serde {
    use super::*;
    use crate::container::{Container, CONTAINER_WORDS};
    use serde::ser::SerializeStruct;
    use serde::{Deserializer, Serializer};

    pub fn serialize<S: Serializer>(record: &CogRecord, serializer: S) -> Result<S::Ok, S::Error> {
        let mut s = serializer.serialize_struct("CogRecord", 2)?;
        s.serialize_field("meta", &record.meta.words.as_slice())?;
        s.serialize_field("content", &record.content.words.as_slice())?;
        s.end()
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<CogRecord, D::Error> {
        #[derive(Deserialize)]
        struct Raw {
            meta: Vec<u64>,
            content: Vec<u64>,
        }

        let raw = Raw::deserialize(deserializer)?;

        let mut meta = Container::zero();
        for (i, &w) in raw.meta.iter().take(CONTAINER_WORDS).enumerate() {
            meta.words[i] = w;
        }

        let mut content = Container::zero();
        for (i, &w) in raw.content.iter().take(CONTAINER_WORDS).enumerate() {
            content.words[i] = w;
        }

        Ok(CogRecord { meta, content })
    }
}
