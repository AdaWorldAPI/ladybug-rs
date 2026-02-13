//! Envelope ↔ Fingerprint codec.
//!
//! Converts between `DataEnvelope` wire format and BindSpace fingerprints.
//! Two directions:
//! - `to_fingerprint()` — ingest an envelope into a 16,384-bit HDR fingerprint
//! - `from_fingerprint()` — export a fingerprint as a DataEnvelope

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde_json::Value;

use crate::core::Fingerprint;
use crate::storage::bind_space::FINGERPRINT_WORDS;

use super::types::{DataEnvelope, EnvelopeMetadata};

impl DataEnvelope {
    /// Convert envelope content to a BindSpace fingerprint.
    ///
    /// - `application/x-fingerprint` — decode base64 directly
    /// - anything else — hash content text through `Fingerprint::from_content()`
    pub fn to_fingerprint(&self) -> [u64; FINGERPRINT_WORDS] {
        if self.content_type == "application/x-fingerprint" {
            if let Some(b64) = self.content.as_str() {
                if let Ok(bytes) = BASE64.decode(b64) {
                    if let Some(fp) = deserialize_fingerprint(&bytes) {
                        return fp;
                    }
                }
            }
        }

        // Fallback: hash content to fingerprint
        let text = match &self.content {
            Value::String(s) => s.clone(),
            other => serde_json::to_string(other).unwrap_or_default(),
        };
        *Fingerprint::from_content(&text).as_raw()
    }

    /// Create a DataEnvelope from a fingerprint (for `lb.*` step outputs).
    pub fn from_fingerprint(step_id: &str, fingerprint: &[u64; FINGERPRINT_WORDS]) -> Self {
        Self {
            step_id: step_id.to_string(),
            output_key: format!("{}.output", step_id),
            content_type: "application/x-fingerprint".to_string(),
            content: Value::String(BASE64.encode(serialize_fingerprint(fingerprint))),
            metadata: EnvelopeMetadata::default(),
        }
    }
}

/// Serialize a fingerprint to bytes (little-endian u64 words).
pub fn serialize_fingerprint(fp: &[u64; FINGERPRINT_WORDS]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(FINGERPRINT_WORDS * 8);
    for word in fp {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

/// Deserialize a fingerprint from bytes (little-endian u64 words).
pub fn deserialize_fingerprint(bytes: &[u8]) -> Option<[u64; FINGERPRINT_WORDS]> {
    if bytes.len() < FINGERPRINT_WORDS * 8 {
        return None;
    }
    let mut fp = [0u64; FINGERPRINT_WORDS];
    for (i, chunk) in bytes.chunks_exact(8).take(FINGERPRINT_WORDS).enumerate() {
        fp[i] = u64::from_le_bytes(chunk.try_into().ok()?);
    }
    Some(fp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fingerprint_roundtrip() {
        let original = *Fingerprint::from_content("hello world").as_raw();
        let envelope = DataEnvelope::from_fingerprint("step-1", &original);
        let recovered = envelope.to_fingerprint();
        assert_eq!(original, recovered, "Fingerprint roundtrip must be exact");
    }

    #[test]
    fn test_text_envelope_to_fingerprint() {
        let envelope = DataEnvelope {
            step_id: "step-2".to_string(),
            output_key: "step-2.output".to_string(),
            content_type: "text/plain".to_string(),
            content: Value::String("quantum computing applications".to_string()),
            metadata: EnvelopeMetadata::default(),
        };
        let fp = envelope.to_fingerprint();
        // Should match Fingerprint::from_content with same text
        let expected = *Fingerprint::from_content("quantum computing applications").as_raw();
        assert_eq!(fp, expected);
    }

    #[test]
    fn test_json_envelope_to_fingerprint() {
        let envelope = DataEnvelope {
            step_id: "step-3".to_string(),
            output_key: "step-3.output".to_string(),
            content_type: "application/json".to_string(),
            content: serde_json::json!({"key": "value", "count": 42}),
            metadata: EnvelopeMetadata::default(),
        };
        let fp = envelope.to_fingerprint();
        // Non-zero fingerprint
        assert!(
            fp.iter().any(|&w| w != 0),
            "JSON fingerprint must be non-zero"
        );
    }

    #[test]
    fn test_serialize_deserialize_fingerprint() {
        let original = [0xDEAD_BEEF_CAFE_BABEu64; FINGERPRINT_WORDS];
        let bytes = serialize_fingerprint(&original);
        assert_eq!(bytes.len(), FINGERPRINT_WORDS * 8);
        let recovered = deserialize_fingerprint(&bytes).unwrap();
        assert_eq!(original, recovered);
    }
}
