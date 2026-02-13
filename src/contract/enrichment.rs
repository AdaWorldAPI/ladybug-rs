//! Ladybug-specific enrichment types for the unified contract.
//!
//! When ladybug processes a `UnifiedStep`, it computes a 16,384-bit HDR
//! fingerprint and stores it in BindSpace. The `StepEnrichment` struct
//! captures the enrichment metadata written to `unified_embeddings`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::storage::bind_space::{Addr, FINGERPRINT_WORDS};

/// Serde helper for `[u64; FINGERPRINT_WORDS]` (256 elements).
///
/// Serde's derive macros only support arrays up to 32 elements.
/// This module serializes the fixed-size array as a `Vec<u64>`.
mod serde_fingerprint {
    use super::FINGERPRINT_WORDS;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(
        data: &[u64; FINGERPRINT_WORDS],
        ser: S,
    ) -> Result<S::Ok, S::Error> {
        data.as_slice().serialize(ser)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        de: D,
    ) -> Result<[u64; FINGERPRINT_WORDS], D::Error> {
        let v: Vec<u64> = Vec::deserialize(de)?;
        v.try_into().map_err(|v: Vec<u64>| {
            serde::de::Error::custom(format!(
                "expected {} u64 words, got {}",
                FINGERPRINT_WORDS,
                v.len()
            ))
        })
    }
}

/// Enrichment data that ladybug adds to unified steps.
///
/// Written to the `unified_embeddings` PostgreSQL table (if postgres feature)
/// and stored in BindSpace at `bind_addr`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StepEnrichment {
    pub step_id: String,
    /// 16,384-bit HDR fingerprint (256 × u64 words).
    #[serde(with = "serde_fingerprint")]
    pub fingerprint: [u64; FINGERPRINT_WORDS],
    /// Qualia index (0–255) — which surface compartment best matches.
    pub qidx: u8,
    /// 7-axis thinking style vector (cognitive style profile).
    pub thinking_style: [f32; 7],
    /// BindSpace address where the fingerprint is stored.
    pub bind_addr: Addr,
    pub enriched_at: DateTime<Utc>,
}

/// Compute qualia index from a fingerprint.
///
/// Uses the first byte's popcount distribution to assign a surface compartment
/// index in range 0–255.
pub fn compute_qidx(fingerprint: &[u64; FINGERPRINT_WORDS]) -> u8 {
    // Use high bits of first word as a fast hash to 0-255
    (fingerprint[0] >> 56) as u8
}

/// Extract a 7-axis thinking style from a UnifiedStep's decision trail.
///
/// The axes are: analytical, creative, systematic, intuitive,
/// collaborative, critical, adaptive — each in [0.0, 1.0].
pub fn extract_thinking_style(step: &super::types::UnifiedStep) -> [f32; 7] {
    let mut style = [0.5_f32; 7]; // neutral default

    // If confidence is present, it modulates the analytical axis
    if let Some(conf) = step.confidence {
        style[0] = conf as f32; // analytical
    }

    // If reasoning is present, boost systematic axis
    if step.reasoning.is_some() {
        style[2] = 0.8; // systematic
    }

    // If alternatives are present, boost creative axis
    if step.alternatives.is_some() {
        style[1] = 0.7; // creative
    }

    style
}

/// Derive a deterministic slot from a step_id string.
pub fn hash_step_id(step_id: &str) -> u8 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    step_id.hash(&mut hasher);
    hasher.finish() as u8
}
