//! Unified CogRecord Schema — the canonical 2×8192-bit record layout.
//!
//! This module is the **single source of truth** for the CogRecord schema
//! shared across ladybug-rs, crewai-rust, and n8n-rs when compiled together.
//!
//! ```text
//! CogRecord = [Container; 2] = 2 KB = 16,384 bits
//! ┌─────────────────────────────────────────────────────────────┐
//! │  meta    (8,192 bits = 1 KB)                                │
//! │  ├── identity:    words[0..8]     dn_addr, container_count  │
//! │  ├── nars:        words[8..16]    frequency, confidence     │
//! │  ├── edges:       words[16..48]   adjacency bitfield        │
//! │  ├── rung/rl:     words[48..64]   consciousness rung, RL    │
//! │  ├── qualia:      words[64..72]   8 affect channels         │
//! │  └── repr:        words[72..128]  geometry-specific fields   │
//! ├─────────────────────────────────────────────────────────────┤
//! │  content (8,192 bits = 1 KB)                                │
//! │  └── searchable fingerprint (Hamming / SIMD / XOR-bind)     │
//! └─────────────────────────────────────────────────────────────┘
//!       = 2 KB = 1 DN tree node = 1 Redis value = 1 Fingerprint
//! ```
//!
//! # Usage from Other Runtimes
//!
//! ```rust,no_run
//! // In crewai-rust or n8n-rs (with `ladybug` feature):
//! use ladybug_contract::schema;
//!
//! assert_eq!(schema::RECORD_BYTES, 2048);
//! assert_eq!(schema::RECORD_CONTAINERS, 2);
//! assert_eq!(schema::CONTAINER_BITS_EACH, 8192);
//! ```

use crate::container::{CONTAINER_BITS, CONTAINER_BYTES, CONTAINER_WORDS};

// =============================================================================
// RECORD SCHEMA CONSTANTS
// =============================================================================

/// Total bits in a CogRecord: 2 × 8,192 = 16,384.
pub const RECORD_BITS: usize = 2 * CONTAINER_BITS;

/// Total bytes in a CogRecord: 2 × 1,024 = 2,048.
pub const RECORD_BYTES: usize = 2 * CONTAINER_BYTES;

/// Total u64 words in a CogRecord: 2 × 128 = 256.
pub const RECORD_WORDS: usize = 2 * CONTAINER_WORDS;

/// Number of containers in a CogRecord (always 2: meta + content).
pub const RECORD_CONTAINERS: usize = 2;

/// Bits per container: 8,192.
pub const CONTAINER_BITS_EACH: usize = CONTAINER_BITS;

/// Bytes per container: 1,024.
pub const CONTAINER_BYTES_EACH: usize = CONTAINER_BYTES;

/// Words per container: 128.
pub const CONTAINER_WORDS_EACH: usize = CONTAINER_WORDS;

// =============================================================================
// METADATA FIELD LAYOUT (word offsets within meta container)
// =============================================================================

/// Word range: identity (dn_addr, container_count, geometry).
pub const META_IDENTITY: (usize, usize) = (0, 8);

/// Word range: NARS truth values (frequency, confidence, expectation).
pub const META_NARS: (usize, usize) = (8, 16);

/// Word range: edge adjacency bitfield (2048-bit neighborhood).
pub const META_EDGES: (usize, usize) = (16, 48);

/// Word range: consciousness rung + RL state.
pub const META_RUNG_RL: (usize, usize) = (48, 64);

/// Word range: qualia channels (8 affect dimensions × 8 bytes).
pub const META_QUALIA: (usize, usize) = (64, 72);

/// Word range: representation / geometry-specific fields.
pub const META_REPR: (usize, usize) = (72, 128);

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

    /// Byte offset from the start of the CogRecord.
    pub const fn byte_offset(&self) -> usize {
        (self.container as usize) * CONTAINER_BYTES + self.word_start * 8
    }

    /// Byte length of this field.
    pub const fn byte_length(&self) -> usize {
        self.word_count() * 8
    }
}

/// The complete CogRecord field layout.
pub const FIELDS: &[FieldDescriptor] = &[
    FieldDescriptor {
        name: "identity",
        container: 0,
        word_start: META_IDENTITY.0,
        word_end: META_IDENTITY.1,
        bits: (META_IDENTITY.1 - META_IDENTITY.0) * 64,
    },
    FieldDescriptor {
        name: "nars",
        container: 0,
        word_start: META_NARS.0,
        word_end: META_NARS.1,
        bits: (META_NARS.1 - META_NARS.0) * 64,
    },
    FieldDescriptor {
        name: "edges",
        container: 0,
        word_start: META_EDGES.0,
        word_end: META_EDGES.1,
        bits: (META_EDGES.1 - META_EDGES.0) * 64,
    },
    FieldDescriptor {
        name: "rung_rl",
        container: 0,
        word_start: META_RUNG_RL.0,
        word_end: META_RUNG_RL.1,
        bits: (META_RUNG_RL.1 - META_RUNG_RL.0) * 64,
    },
    FieldDescriptor {
        name: "qualia",
        container: 0,
        word_start: META_QUALIA.0,
        word_end: META_QUALIA.1,
        bits: (META_QUALIA.1 - META_QUALIA.0) * 64,
    },
    FieldDescriptor {
        name: "repr",
        container: 0,
        word_start: META_REPR.0,
        word_end: META_REPR.1,
        bits: (META_REPR.1 - META_REPR.0) * 64,
    },
    FieldDescriptor {
        name: "content",
        container: 1,
        word_start: 0,
        word_end: CONTAINER_WORDS,
        bits: CONTAINER_BITS,
    },
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
    fn test_record_constants() {
        assert_eq!(RECORD_BITS, 16_384);
        assert_eq!(RECORD_BYTES, 2048);
        assert_eq!(RECORD_WORDS, 256);
        assert_eq!(RECORD_CONTAINERS, 2);
        assert_eq!(CONTAINER_BITS_EACH, 8192);
        assert_eq!(CONTAINER_BYTES_EACH, 1024);
        assert_eq!(CONTAINER_WORDS_EACH, 128);
    }

    #[test]
    fn test_meta_fields_cover_all_words() {
        // Meta container is words 0..128. Verify fields are contiguous.
        assert_eq!(META_IDENTITY.0, 0);
        assert_eq!(META_IDENTITY.1, META_NARS.0);
        assert_eq!(META_NARS.1, META_EDGES.0);
        assert_eq!(META_EDGES.1, META_RUNG_RL.0);
        assert_eq!(META_RUNG_RL.1, META_QUALIA.0);
        assert_eq!(META_QUALIA.1, META_REPR.0);
        assert_eq!(META_REPR.1, CONTAINER_WORDS);
    }

    #[test]
    fn test_field_descriptors() {
        assert_eq!(FIELDS.len(), 7); // 6 meta fields + 1 content
        assert_eq!(FIELDS[0].name, "identity");
        assert_eq!(FIELDS[6].name, "content");
        assert_eq!(FIELDS[6].container, 1);
        assert_eq!(FIELDS[6].bits, 8192);
    }

    #[test]
    fn test_field_byte_offsets() {
        let identity = &FIELDS[0];
        assert_eq!(identity.byte_offset(), 0);
        assert_eq!(identity.byte_length(), 64); // 8 words × 8 bytes

        let content = &FIELDS[6];
        assert_eq!(content.byte_offset(), 1024); // starts at container 1
        assert_eq!(content.byte_length(), 1024); // full container
    }

    #[test]
    fn test_field_lookup() {
        let nars = field("nars").unwrap();
        assert_eq!(nars.word_start, 8);
        assert_eq!(nars.word_end, 16);

        let content = field("content").unwrap();
        assert_eq!(content.container, 1);
        assert_eq!(content.bits, 8192);

        assert!(field("nonexistent").is_none());
    }

    #[test]
    fn test_total_field_bits() {
        let total: usize = FIELDS.iter().map(|f| f.bits).sum();
        assert_eq!(total, RECORD_BITS);
    }
}
