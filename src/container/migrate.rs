//! Migration: 16K Fingerprint → CogRecord.
//!
//! With CONTAINER_WORDS = 256 a single Container holds the full 16K fingerprint.
//! The old "Extended" split into two 128-word halves is no longer needed.

use super::CONTAINER_WORDS;
use super::geometry::ContainerGeometry;
use super::meta::MetaViewMut;
use super::record::CogRecord;

/// Convert a 16K Fingerprint ([u64; 256]) to a CogRecord (Cam geometry).
///
/// All 256 words fit into the single content container (CONTAINER_WORDS = 256).
/// If the old vector had schema sidecar data in words 208-255, it is extracted
/// and placed into the metadata container using the new layout.
pub fn migrate_16k(old: &[u64; 256]) -> CogRecord {
    let mut record = CogRecord::new(ContainerGeometry::Cam);

    // Content: first 128 words = primary fingerprint signal
    record.content
        .words
        .copy_from_slice(&old[..CONTAINER_WORDS]);

    // Check if words 224-255 contain schema sidecar metadata (non-zero)
    let has_sidecar = old[224..256].iter().any(|&w| w != 0);

    if has_sidecar {
        // Extract known fields from old 16K sidecar layout (blocks 14-15):
        // Block 14 (words 224-239): Identity + Reasoning + Learning
        // Block 15 (words 240-255): Graph topology + inline edges
        extract_sidecar_to_meta(&old[224..256], &mut record.meta.words);
    }

    // Update metadata
    {
        let mut meta = MetaViewMut::new(&mut record.meta.words);
        meta.set_schema_version(1);
        meta.set_container_count(2);
        meta.set_geometry(ContainerGeometry::Cam);
        meta.update_checksum();
    }

    record
}

/// Convert a single CogRecord back to 16K Fingerprint ([u64; 256]).
pub fn to_16k(record: &CogRecord) -> [u64; 256] {
    let mut out = [0u64; 256];
    out[..CONTAINER_WORDS].copy_from_slice(&record.content.words);
    out
}

/// Extract known fields from old 16K sidecar (32 words) into new metadata.
fn extract_sidecar_to_meta(sidecar: &[u64], meta_words: &mut [u64; CONTAINER_WORDS]) {
    // Old block 14 layout (words 224-239 of 16K, mapped to sidecar[0..15]):
    //   Word 0: NARS truth (freq:u16 | conf:u16 | pos_ev:u16 | neg_ev:u16)
    //   Word 1: Rung level (u8) | gate state (u8) | layer bitmap (u8) | ...
    //   Word 2-3: Timestamps

    if sidecar.len() < 16 {
        return;
    }

    let mut meta = MetaViewMut::new(meta_words);

    // Extract NARS from old quantized u16 format → promote to f32
    let nars_word = sidecar[0];
    let freq_q = (nars_word & 0xFFFF) as u16;
    let conf_q = ((nars_word >> 16) & 0xFFFF) as u16;
    meta.set_nars_frequency(freq_q as f32 / 65535.0);
    meta.set_nars_confidence(conf_q as f32 / 65535.0);

    // Extract rung + gate
    let rung_word = sidecar[1];
    meta.set_rung_level((rung_word & 0xFF) as u8);
    meta.set_gate_state(((rung_word >> 8) & 0xFF) as u8);
    meta.set_layer_bitmap(((rung_word >> 16) & 0x7F) as u8);

    // Extract timestamps
    if sidecar.len() > 2 {
        meta.set_created_ms((sidecar[2] & 0xFFFF_FFFF) as u32);
        meta.set_modified_ms(((sidecar[2] >> 32) & 0xFFFF_FFFF) as u32);
    }
}
