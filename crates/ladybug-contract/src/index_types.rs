//! Index key and entry types for LadybugDB.
//!
//! These are the types that index consumers need. The actual index
//! implementation (`IndexBuilder`, `LadybugIndex`, `IndexHandle`)
//! stays in the main ladybug crate.

/// 64-bit index key: [16-bit type_id][48-bit fingerprint prefix].
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Key(u64);

impl Key {
    /// Create a key from a type ID and a fingerprint byte slice.
    pub fn new(type_id: u16, fp: &[u8]) -> Self {
        let mut prefix = 0u64;
        for (i, &b) in fp.iter().take(6).enumerate() {
            prefix |= (b as u64) << (i * 8);
        }
        Key(((type_id as u64) << 48) | (prefix & 0x0000_FFFF_FFFF_FFFF))
    }

    /// Create a key from a raw u64.
    pub fn from_raw(raw: u64) -> Self {
        Key(raw)
    }

    /// The type ID (top 16 bits).
    pub fn type_id(self) -> u16 {
        (self.0 >> 48) as u16
    }

    /// The fingerprint prefix (bottom 48 bits).
    pub fn prefix(self) -> u64 {
        self.0 & 0x0000_FFFF_FFFF_FFFF
    }

    /// The raw u64.
    pub fn raw(self) -> u64 {
        self.0
    }
}

impl std::fmt::Debug for Key {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Key(type={:#06x}, prefix={:#014x})",
            self.type_id(),
            self.prefix()
        )
    }
}

/// Entry in an index bucket.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Entry {
    /// 48-bit fingerprint prefix (stored in 64).
    pub prefix: u64,
    /// Row offset in Arrow/Lance.
    pub offset: u64,
    /// For edges: target fingerprint prefix. 0 otherwise.
    pub target: u64,
}

impl Entry {
    pub fn new(prefix: u64, offset: u64) -> Self {
        Self {
            prefix,
            offset,
            target: 0,
        }
    }

    pub fn edge(prefix: u64, offset: u64, target: u64) -> Self {
        Self {
            prefix,
            offset,
            target,
        }
    }
}

/// Type ID constants for LadybugDB's type system.
pub mod types {
    // Core entities
    pub const THOUGHT: u16 = 0x0001;
    pub const CONCEPT: u16 = 0x0002;
    pub const STYLE: u16 = 0x0003;

    // Edge types (graph relations)
    pub const EDGE_CAUSES: u16 = 0x0100;
    pub const EDGE_SUPPORTS: u16 = 0x0101;
    pub const EDGE_CONTRADICTS: u16 = 0x0102;
    pub const EDGE_BECOMES: u16 = 0x0103;
    pub const EDGE_REFINES: u16 = 0x0104;
    pub const EDGE_GROUNDS: u16 = 0x0105;
    pub const EDGE_ABSTRACTS: u16 = 0x0106;

    // Consciousness layers
    pub const LAYER_SUBSTRATE: u16 = 0x0200;
    pub const LAYER_FELT_CORE: u16 = 0x0201;
    pub const LAYER_BODY: u16 = 0x0202;
    pub const LAYER_QUALIA: u16 = 0x0203;
    pub const LAYER_VOLITION: u16 = 0x0204;
    pub const LAYER_GESTALT: u16 = 0x0205;
    pub const LAYER_META: u16 = 0x0206;

    // Thinking styles
    pub const STYLE_ANALYTICAL: u16 = 0x0300;
    pub const STYLE_INTUITIVE: u16 = 0x0301;
    pub const STYLE_FOCUSED: u16 = 0x0302;
    pub const STYLE_DIFFUSE: u16 = 0x0303;
    pub const STYLE_CONVERGENT: u16 = 0x0304;
    pub const STYLE_DIVERGENT: u16 = 0x0305;
    pub const STYLE_CONCRETE: u16 = 0x0306;
    pub const STYLE_ABSTRACT: u16 = 0x0307;
    pub const STYLE_SEQUENTIAL: u16 = 0x0308;
    pub const STYLE_HOLISTIC: u16 = 0x0309;
    pub const STYLE_VERBAL: u16 = 0x030A;
    pub const STYLE_SPATIAL: u16 = 0x030B;

    // Codebook
    pub const CODE: u16 = 0x0400;

    // Edge type ranges
    pub const EDGE_START: u16 = 0x0100;
    pub const EDGE_END: u16 = 0x01FF;

    /// Check if a type ID is an edge type.
    pub fn is_edge(t: u16) -> bool {
        (EDGE_START..=EDGE_END).contains(&t)
    }
}
