//! Unified CogRecord Schema — each CogRecord = one 16,384-bit container.
//!
//! This module is the **single source of truth** for the CogRecord schema
//! shared across ladybug-rs, crewai-rust, and n8n-rs when compiled together.
//!
//! ```text
//! A node is composed of separate CogRecords (containers):
//!
//! Container 0 — Metadata CogRecord (16,384 bits = 256 words = 2 KB)
//! ┌──────────────────────────────────────────────────────────────┐
//! │  W0        DN address (identity)                             │
//! │  W1-3      type / timestamps / label                         │
//! │  W4-7      NARS truth values                                 │
//! │  W8-15     DN rung + 7-layer markers                         │
//! │  W16-31    inline edges (64 packed)                           │
//! │  W32-39    RL / Q-values / rewards                            │
//! │  W40-47    bloom filter (512 bits)                            │
//! │  W48-55    graph metrics                                      │
//! │  W56-63    qualia: 8 affect channels                          │
//! │  W64-79    rung history + collapse gate history                │
//! │  W80-95    representation language descriptor                 │
//! │  W96-111   DN-sparse adjacency (compact inline CSR)           │
//! │  W112-125  reserved                                           │
//! │  W126-127  checksum + version                                 │
//! │  W128-255  reserved (future expansion)                        │
//! └──────────────────────────────────────────────────────────────┘
//!
//! Container 1 — Content CogRecord (16,384 bits = 256 words = 2 KB)
//! ┌──────────────────────────────────────────────────────────────┐
//! │  W0-255    searchable VSA fingerprint (Hamming / XOR-bind)    │
//! └──────────────────────────────────────────────────────────────┘
//!
//! Container N — Additional CogRecords (Jina embeddings, etc.)
//! ```
//!
//! # Usage from Other Runtimes
//!
//! ```rust,no_run
//! // In crewai-rust or n8n-rs (with `ladybug` feature):
//! use ladybug_contract::schema;
//!
//! assert_eq!(schema::CONTAINER_BITS_EACH, 16384);
//! assert_eq!(schema::CONTAINER_WORDS_EACH, 256);
//! assert_eq!(schema::CONTENT_WORDS, 256);
//! ```

use crate::container::{CONTAINER_BITS, CONTAINER_BYTES, CONTAINER_WORDS};

// =============================================================================
// CONTAINER CONSTANTS
// =============================================================================

/// Bits per container (CogRecord): 16,384.
pub const CONTAINER_BITS_EACH: usize = CONTAINER_BITS;

/// Bytes per container: 2,048.
pub const CONTAINER_BYTES_EACH: usize = CONTAINER_BYTES;

/// Words per container: 256.
pub const CONTAINER_WORDS_EACH: usize = CONTAINER_WORDS;

// =============================================================================
// CONTENT CONTAINER LAYOUT
// =============================================================================

/// Content container: word offset (always 0 — full container is payload).
pub const CONTENT_OFFSET: usize = 0;

/// Content container: word count (all 256 words are searchable fingerprint).
pub const CONTENT_WORDS: usize = CONTAINER_WORDS; // 256

/// Content container: bit count.
pub const CONTENT_BITS: usize = CONTAINER_BITS; // 16,384

// =============================================================================
// METADATA FIELD LAYOUT (word offsets within metadata container)
//
// Matches crate::meta (MetaView / MetaViewMut) — the canonical bit-level layout.
// =============================================================================

/// Word range: identity (dn_addr, type, timestamps, label).
pub const META_IDENTITY: (usize, usize) = (0, 4);

/// Word range: NARS truth values (frequency, confidence, evidence).
pub const META_NARS: (usize, usize) = (4, 8);

/// Word range: DN rung + gate + 7-layer markers.
pub const META_RUNG_LAYERS: (usize, usize) = (8, 16);

/// Word range: inline edges (64 packed, 4 per word).
pub const META_EDGES: (usize, usize) = (16, 32);

/// Word range: RL / Q-values / rewards.
pub const META_RL: (usize, usize) = (32, 40);

/// Word range: bloom filter (512 bits).
pub const META_BLOOM: (usize, usize) = (40, 48);

/// Word range: graph metrics (in-degree, out-degree, pagerank, clustering).
pub const META_GRAPH: (usize, usize) = (48, 56);

/// Word range: qualia channels (8 affect dimensions).
pub const META_QUALIA: (usize, usize) = (56, 64);

/// Word range: rung history + collapse gate history.
pub const META_RUNG_HIST: (usize, usize) = (64, 80);

/// Word range: representation language descriptor.
pub const META_REPR: (usize, usize) = (80, 96);

/// Word range: DN-sparse adjacency (compact inline CSR).
pub const META_ADJ: (usize, usize) = (96, 112);

/// Word range: reserved.
pub const META_RESERVED: (usize, usize) = (112, 126);

/// Word range: checksum + version.
pub const META_CHECKSUM: (usize, usize) = (126, 128);

// =============================================================================
// SCHEMA FIELD DESCRIPTORS
// =============================================================================

/// A descriptor for a field in the CogRecord schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldDescriptor {
    /// Human-readable field name.
    pub name: &'static str,
    /// Container index (0 = meta, 1 = content).
    pub container: u8,
    /// Start word offset within the container.
    pub word_start: usize,
    /// End word offset (exclusive).
    pub word_end: usize,
    /// Bit width of the field (word_count × 64).
    pub bits: usize,
}

impl FieldDescriptor {
    /// Number of u64 words this field occupies.
    pub const fn word_count(&self) -> usize {
        self.word_end - self.word_start
    }

    /// Byte offset from the start of the node (container_index × CONTAINER_BYTES + word_start × 8).
    pub const fn byte_offset(&self) -> usize {
        (self.container as usize) * CONTAINER_BYTES + self.word_start * 8
    }

    /// Byte length of this field.
    pub const fn byte_length(&self) -> usize {
        self.word_count() * 8
    }
}

/// Metadata container field descriptors (Container 0).
pub const META_FIELDS: &[FieldDescriptor] = &[
    FieldDescriptor { name: "identity",   container: 0, word_start: META_IDENTITY.0,   word_end: META_IDENTITY.1,   bits: (META_IDENTITY.1   - META_IDENTITY.0)   * 64 },
    FieldDescriptor { name: "nars",       container: 0, word_start: META_NARS.0,       word_end: META_NARS.1,       bits: (META_NARS.1       - META_NARS.0)       * 64 },
    FieldDescriptor { name: "rung_layers",container: 0, word_start: META_RUNG_LAYERS.0,word_end: META_RUNG_LAYERS.1,bits: (META_RUNG_LAYERS.1- META_RUNG_LAYERS.0)* 64 },
    FieldDescriptor { name: "edges",      container: 0, word_start: META_EDGES.0,      word_end: META_EDGES.1,      bits: (META_EDGES.1      - META_EDGES.0)      * 64 },
    FieldDescriptor { name: "rl",         container: 0, word_start: META_RL.0,         word_end: META_RL.1,         bits: (META_RL.1         - META_RL.0)         * 64 },
    FieldDescriptor { name: "bloom",      container: 0, word_start: META_BLOOM.0,      word_end: META_BLOOM.1,      bits: (META_BLOOM.1      - META_BLOOM.0)      * 64 },
    FieldDescriptor { name: "graph",      container: 0, word_start: META_GRAPH.0,      word_end: META_GRAPH.1,      bits: (META_GRAPH.1      - META_GRAPH.0)      * 64 },
    FieldDescriptor { name: "qualia",     container: 0, word_start: META_QUALIA.0,     word_end: META_QUALIA.1,     bits: (META_QUALIA.1     - META_QUALIA.0)     * 64 },
    FieldDescriptor { name: "rung_hist",  container: 0, word_start: META_RUNG_HIST.0,  word_end: META_RUNG_HIST.1,  bits: (META_RUNG_HIST.1  - META_RUNG_HIST.0)  * 64 },
    FieldDescriptor { name: "repr",       container: 0, word_start: META_REPR.0,       word_end: META_REPR.1,       bits: (META_REPR.1       - META_REPR.0)       * 64 },
    FieldDescriptor { name: "adj",        container: 0, word_start: META_ADJ.0,        word_end: META_ADJ.1,        bits: (META_ADJ.1        - META_ADJ.0)        * 64 },
    FieldDescriptor { name: "reserved",   container: 0, word_start: META_RESERVED.0,   word_end: META_RESERVED.1,   bits: (META_RESERVED.1   - META_RESERVED.0)   * 64 },
    FieldDescriptor { name: "checksum",   container: 0, word_start: META_CHECKSUM.0,   word_end: META_CHECKSUM.1,   bits: (META_CHECKSUM.1   - META_CHECKSUM.0)   * 64 },
];

/// The complete field layout across containers.
pub const FIELDS: &[FieldDescriptor] = &[
    // Container 0: metadata (W0-W127 defined, W128-W255 reserved)
    FieldDescriptor { name: "identity",   container: 0, word_start: META_IDENTITY.0,   word_end: META_IDENTITY.1,   bits: (META_IDENTITY.1   - META_IDENTITY.0)   * 64 },
    FieldDescriptor { name: "nars",       container: 0, word_start: META_NARS.0,       word_end: META_NARS.1,       bits: (META_NARS.1       - META_NARS.0)       * 64 },
    FieldDescriptor { name: "rung_layers",container: 0, word_start: META_RUNG_LAYERS.0,word_end: META_RUNG_LAYERS.1,bits: (META_RUNG_LAYERS.1- META_RUNG_LAYERS.0)* 64 },
    FieldDescriptor { name: "edges",      container: 0, word_start: META_EDGES.0,      word_end: META_EDGES.1,      bits: (META_EDGES.1      - META_EDGES.0)      * 64 },
    FieldDescriptor { name: "rl",         container: 0, word_start: META_RL.0,         word_end: META_RL.1,         bits: (META_RL.1         - META_RL.0)         * 64 },
    FieldDescriptor { name: "bloom",      container: 0, word_start: META_BLOOM.0,      word_end: META_BLOOM.1,      bits: (META_BLOOM.1      - META_BLOOM.0)      * 64 },
    FieldDescriptor { name: "graph",      container: 0, word_start: META_GRAPH.0,      word_end: META_GRAPH.1,      bits: (META_GRAPH.1      - META_GRAPH.0)      * 64 },
    FieldDescriptor { name: "qualia",     container: 0, word_start: META_QUALIA.0,     word_end: META_QUALIA.1,     bits: (META_QUALIA.1     - META_QUALIA.0)     * 64 },
    FieldDescriptor { name: "rung_hist",  container: 0, word_start: META_RUNG_HIST.0,  word_end: META_RUNG_HIST.1,  bits: (META_RUNG_HIST.1  - META_RUNG_HIST.0)  * 64 },
    FieldDescriptor { name: "repr",       container: 0, word_start: META_REPR.0,       word_end: META_REPR.1,       bits: (META_REPR.1       - META_REPR.0)       * 64 },
    FieldDescriptor { name: "adj",        container: 0, word_start: META_ADJ.0,        word_end: META_ADJ.1,        bits: (META_ADJ.1        - META_ADJ.0)        * 64 },
    FieldDescriptor { name: "reserved",   container: 0, word_start: META_RESERVED.0,   word_end: META_RESERVED.1,   bits: (META_RESERVED.1   - META_RESERVED.0)   * 64 },
    FieldDescriptor { name: "checksum",   container: 0, word_start: META_CHECKSUM.0,   word_end: META_CHECKSUM.1,   bits: (META_CHECKSUM.1   - META_CHECKSUM.0)   * 64 },
    // Container 1: content (all 256 words)
    FieldDescriptor { name: "content",    container: 1, word_start: CONTENT_OFFSET,    word_end: CONTENT_WORDS,     bits: CONTENT_BITS },
];

/// Look up a field descriptor by name.
pub const fn field(name: &str) -> Option<&'static FieldDescriptor> {
    let mut i = 0;
    while i < FIELDS.len() {
        if const_str_eq(FIELDS[i].name, name) {
            return Some(&FIELDS[i]);
        }
        i += 1;
    }
    None
}

/// Const-compatible string equality.
const fn const_str_eq(a: &str, b: &str) -> bool {
    let a = a.as_bytes();
    let b = b.as_bytes();
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_container_constants() {
        assert_eq!(CONTAINER_BITS_EACH, 16_384);
        assert_eq!(CONTAINER_BYTES_EACH, 2048);
        assert_eq!(CONTAINER_WORDS_EACH, 256);
    }

    #[test]
    fn test_content_constants() {
        assert_eq!(CONTENT_OFFSET, 0);
        assert_eq!(CONTENT_WORDS, 256);
        assert_eq!(CONTENT_BITS, 16_384);
    }

    #[test]
    fn test_meta_fields_contiguous_to_128() {
        // Metadata container: W0-W127 are defined fields, contiguous.
        assert_eq!(META_IDENTITY.0, 0);
        assert_eq!(META_IDENTITY.1, META_NARS.0);
        assert_eq!(META_NARS.1, META_RUNG_LAYERS.0);
        assert_eq!(META_RUNG_LAYERS.1, META_EDGES.0);
        assert_eq!(META_EDGES.1, META_RL.0);
        assert_eq!(META_RL.1, META_BLOOM.0);
        assert_eq!(META_BLOOM.1, META_GRAPH.0);
        assert_eq!(META_GRAPH.1, META_QUALIA.0);
        assert_eq!(META_QUALIA.1, META_RUNG_HIST.0);
        assert_eq!(META_RUNG_HIST.1, META_REPR.0);
        assert_eq!(META_REPR.1, META_ADJ.0);
        assert_eq!(META_ADJ.1, META_RESERVED.0);
        assert_eq!(META_RESERVED.1, META_CHECKSUM.0);
        assert_eq!(META_CHECKSUM.1, 128);
    }

    #[test]
    fn test_qualia_at_w56() {
        assert_eq!(META_QUALIA.0, 56);
        assert_eq!(META_QUALIA.1, 64);
    }

    #[test]
    fn test_field_descriptors() {
        assert_eq!(FIELDS.len(), 14); // 13 meta fields + 1 content
        assert_eq!(FIELDS[0].name, "identity");
        assert_eq!(FIELDS[13].name, "content");
        assert_eq!(FIELDS[13].container, 1);
        assert_eq!(FIELDS[13].bits, 16_384);
    }

    #[test]
    fn test_field_byte_offsets() {
        let identity = &FIELDS[0];
        assert_eq!(identity.byte_offset(), 0);
        assert_eq!(identity.byte_length(), 32); // 4 words × 8 bytes

        let content = field("content").unwrap();
        assert_eq!(content.byte_offset(), 2048); // starts at container 1
        assert_eq!(content.byte_length(), 2048); // full container
    }

    #[test]
    fn test_field_lookup() {
        let nars = field("nars").unwrap();
        assert_eq!(nars.word_start, 4);
        assert_eq!(nars.word_end, 8);

        let qualia = field("qualia").unwrap();
        assert_eq!(qualia.word_start, 56);
        assert_eq!(qualia.word_end, 64);

        let content = field("content").unwrap();
        assert_eq!(content.container, 1);
        assert_eq!(content.bits, 16_384);

        assert!(field("nonexistent").is_none());
    }

    #[test]
    fn test_meta_defined_bits() {
        let meta_bits: usize = META_FIELDS.iter().map(|f| f.bits).sum();
        // W0-W127 = 128 words = 8192 bits of defined fields
        assert_eq!(meta_bits, 128 * 64);
    }
}
