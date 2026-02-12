//! Container 0: Metadata layout — zero-copy views.
//!
//! 128 words of structural information. Never included in Hamming search.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │  W0       PackedDn address (THE identity)                    │
//! │  W1       Type: node_kind | count | geometry | flags         │
//! │           | schema_version(u16) | provenance_hash(u16)       │
//! │  W2       Timestamps (created_ms:32 | modified_ms:32)        │
//! │  W3       Label hash (u32) | tree depth:u8 | branch:u8 | rsvd│
//! │  W4-7     NARS (freq:f32 | conf:f32 | pos_ev:f32 | neg_ev)  │
//! │  W8-11    DN rung + 7-Layer compact + collapse gate          │
//! │  W12-15   7-Layer markers (5 bytes × 7 = 35 bytes)           │
//! │  W16-31   Inline edges (64 packed, 4 per word)               │
//! │  W32-39   RL / Q-values / rewards                            │
//! │  W40-47   Bloom filter (512 bits)                            │
//! │  W48-55   Graph metrics (full precision)                     │
//! │  W56-63   Qualia (18 channels × f16 + 8 slots)              │
//! │  W64-79   Rung history + collapse gate history               │
//! │  W80-95   Representation language descriptor                 │
//! │  W96-111  DN-Sparse adjacency (compact inline CSR)           │
//! │  W112-125 Reserved                                           │
//! │  W126-127 Checksum + version                                 │
//! └──────────────────────────────────────────────────────────────┘
//! ```

use super::CONTAINER_WORDS;
use super::geometry::ContainerGeometry;

// ============================================================================
// WORD OFFSETS
// ============================================================================

/// PackedDn address — the identity of this record.
pub const W_DN_ADDR: usize = 0;

/// Record type + geometry.
/// Layout: node_kind(u8) | container_count(u8) | geometry(u8) | flags(u8)
///         | schema_version(u16) | provenance_hash(u16)
pub const W_TYPE: usize = 1;

/// Timestamps: created_ms(u32) | modified_ms(u32).
pub const W_TIME: usize = 2;

/// Label hash (u32) | tree_depth(u8) | branch(u8) | reserved(u16).
pub const W_LABEL: usize = 3;

/// NARS base: words 4-7 hold freq, conf, pos_evidence, neg_evidence as f32.
pub const W_NARS_BASE: usize = 4;

/// DN rung + 7-layer compact + collapse gate state.
pub const W_DN_RUNG: usize = 8;

/// 7-layer markers base (words 12-15).
pub const W_LAYER_BASE: usize = 12;

/// Inline edges base (words 16-31). 4 edges per word × 16 words = 64 edges.
pub const W_EDGE_BASE: usize = 16;
pub const W_EDGE_END: usize = 31;

/// Maximum inline edges.
pub const MAX_INLINE_EDGES: usize = 64;

/// RL / Q-values / rewards base (words 32-39).
pub const W_RL_BASE: usize = 32;

/// Bloom filter base (words 40-47, 512 bits).
pub const W_BLOOM_BASE: usize = 40;

/// Graph metrics base (words 48-55).
pub const W_GRAPH_BASE: usize = 48;

/// Qualia base (words 56-63).
pub const W_QUALIA_BASE: usize = 56;

/// Rung history + collapse gate history (words 64-79).
pub const W_RUNG_HIST: usize = 64;

/// Representation language descriptor (words 80-95).
pub const W_REPR_BASE: usize = 80;

/// DN-Sparse adjacency (compact inline CSR, words 96-111).
pub const W_ADJ_BASE: usize = 96;

/// Reserved (words 112-125).
pub const W_RESERVED: usize = 112;

/// Checksum + version (words 126-127).
pub const W_CHECKSUM: usize = 126;

/// Current schema version.
pub const SCHEMA_VERSION: u16 = 1;

// ============================================================================
// META VIEW (read-only)
// ============================================================================

/// Zero-copy read-only view into Container 0 metadata.
pub struct MetaView<'a> {
    words: &'a [u64; CONTAINER_WORDS],
}

impl<'a> MetaView<'a> {
    /// Create from a reference to container words.
    pub fn new(words: &'a [u64; CONTAINER_WORDS]) -> Self {
        Self { words }
    }

    // -- W0: DN Address --

    /// The packed DN address (full 64-bit identity).
    #[inline]
    pub fn dn_addr(&self) -> u64 {
        self.words[W_DN_ADDR]
    }

    // -- W1: Type/Geometry --

    /// Node kind (byte 0 of W1).
    #[inline]
    pub fn node_kind(&self) -> u8 {
        (self.words[W_TYPE] & 0xFF) as u8
    }

    /// Container count (byte 1 of W1).
    #[inline]
    pub fn container_count(&self) -> u8 {
        ((self.words[W_TYPE] >> 8) & 0xFF) as u8
    }

    /// Geometry (byte 2 of W1).
    #[inline]
    pub fn geometry(&self) -> ContainerGeometry {
        let g = ((self.words[W_TYPE] >> 16) & 0xFF) as u8;
        ContainerGeometry::from_u8(g).unwrap_or_default()
    }

    /// Flags (byte 3 of W1).
    #[inline]
    pub fn flags(&self) -> u8 {
        ((self.words[W_TYPE] >> 24) & 0xFF) as u8
    }

    /// Schema version (bytes 4-5 of W1).
    #[inline]
    pub fn schema_version(&self) -> u16 {
        ((self.words[W_TYPE] >> 32) & 0xFFFF) as u16
    }

    /// Provenance hash (bytes 6-7 of W1).
    #[inline]
    pub fn provenance_hash(&self) -> u16 {
        ((self.words[W_TYPE] >> 48) & 0xFFFF) as u16
    }

    // -- W2: Timestamps --

    /// Created timestamp (ms, lower 32 bits of W2).
    #[inline]
    pub fn created_ms(&self) -> u32 {
        (self.words[W_TIME] & 0xFFFF_FFFF) as u32
    }

    /// Modified timestamp (ms, upper 32 bits of W2).
    #[inline]
    pub fn modified_ms(&self) -> u32 {
        ((self.words[W_TIME] >> 32) & 0xFFFF_FFFF) as u32
    }

    // -- W3: Label --

    /// Label hash (lower 32 bits of W3).
    #[inline]
    pub fn label_hash(&self) -> u32 {
        (self.words[W_LABEL] & 0xFFFF_FFFF) as u32
    }

    /// Tree depth (byte 4 of W3).
    #[inline]
    pub fn tree_depth(&self) -> u8 {
        ((self.words[W_LABEL] >> 32) & 0xFF) as u8
    }

    /// Branch index (byte 5 of W3).
    #[inline]
    pub fn branch(&self) -> u8 {
        ((self.words[W_LABEL] >> 40) & 0xFF) as u8
    }

    // -- W4-7: NARS --

    /// NARS frequency (f32, word 4).
    #[inline]
    pub fn nars_frequency(&self) -> f32 {
        f32::from_bits((self.words[W_NARS_BASE] & 0xFFFF_FFFF) as u32)
    }

    /// NARS confidence (f32, upper half of word 4).
    #[inline]
    pub fn nars_confidence(&self) -> f32 {
        f32::from_bits(((self.words[W_NARS_BASE] >> 32) & 0xFFFF_FFFF) as u32)
    }

    /// NARS positive evidence (f32, word 5).
    #[inline]
    pub fn nars_positive_evidence(&self) -> f32 {
        f32::from_bits((self.words[W_NARS_BASE + 1] & 0xFFFF_FFFF) as u32)
    }

    /// NARS negative evidence (f32, upper half of word 5).
    #[inline]
    pub fn nars_negative_evidence(&self) -> f32 {
        f32::from_bits(((self.words[W_NARS_BASE + 1] >> 32) & 0xFFFF_FFFF) as u32)
    }

    // -- W8: DN Rung + Gate --

    /// Rung level (byte 0 of W8). Matches `RungLevel` repr(u8) 0-9.
    #[inline]
    pub fn rung_level(&self) -> u8 {
        (self.words[W_DN_RUNG] & 0xFF) as u8
    }

    /// Collapse gate state (byte 1 of W8). 0=Flow, 1=Hold, 2=Block.
    #[inline]
    pub fn gate_state(&self) -> u8 {
        ((self.words[W_DN_RUNG] >> 8) & 0xFF) as u8
    }

    /// 7-layer compact activation bitmap (byte 2 of W8). 1 bit per layer.
    #[inline]
    pub fn layer_bitmap(&self) -> u8 {
        ((self.words[W_DN_RUNG] >> 16) & 0x7F) as u8
    }

    // -- W12-15: Layer markers --

    /// Per-layer activation data. 5 bytes per layer packed into words 12-15.
    /// Returns (strength_u8, frequency_u8, recency_u16, flags_u8) for layer 0-6.
    pub fn layer_marker(&self, layer: usize) -> (u8, u8, u16, u8) {
        debug_assert!(layer < 7);
        let byte_offset = layer * 5;
        let word_idx = W_LAYER_BASE + byte_offset / 8;
        let bit_offset = (byte_offset % 8) * 8;

        // Read 5 bytes spanning at most 2 words
        let lo = self.words[word_idx] >> bit_offset;
        let hi = if word_idx + 1 < CONTAINER_WORDS {
            self.words[word_idx + 1]
        } else {
            0
        };
        let val = if bit_offset <= 24 {
            lo
        } else {
            lo | (hi << (64 - bit_offset))
        };

        let strength = (val & 0xFF) as u8;
        let frequency = ((val >> 8) & 0xFF) as u8;
        let recency = ((val >> 16) & 0xFFFF) as u16;
        let flags = ((val >> 32) & 0xFF) as u8;
        (strength, frequency, recency, flags)
    }

    // -- W16-31: Inline edges --

    /// Read inline edge at index 0..63. Returns (verb_u8, target_u8).
    #[inline]
    pub fn inline_edge(&self, idx: usize) -> (u8, u8) {
        debug_assert!(idx < MAX_INLINE_EDGES);
        let word_idx = W_EDGE_BASE + idx / 4;
        let shift = (idx % 4) * 16;
        let packed = ((self.words[word_idx] >> shift) & 0xFFFF) as u16;
        ((packed >> 8) as u8, (packed & 0xFF) as u8)
    }

    /// Count of inline edges (non-zero entries in W16-31).
    pub fn inline_edge_count(&self) -> usize {
        let mut count = 0;
        for idx in 0..MAX_INLINE_EDGES {
            let (verb, target) = self.inline_edge(idx);
            if verb != 0 || target != 0 {
                count += 1;
            }
        }
        count
    }

    // -- W32-39: RL --

    /// Q-value for action `i` (f32). Up to 16 actions in 8 words.
    #[inline]
    pub fn q_value(&self, action: usize) -> f32 {
        debug_assert!(action < 16);
        let word_idx = W_RL_BASE + action / 2;
        let shift = (action % 2) * 32;
        f32::from_bits(((self.words[word_idx] >> shift) & 0xFFFF_FFFF) as u32)
    }

    // -- W40-47: Bloom --

    /// Check if an id is in the Bloom filter (512 bits, 3 hash functions).
    pub fn bloom_contains(&self, id: u64) -> bool {
        let h1 = id as usize % 512;
        let h2 = id.wrapping_mul(0x9E3779B97F4A7C15) as usize % 512;
        let h3 = id.wrapping_mul(0x517CC1B727220A95) as usize % 512;

        let check = |bit: usize| -> bool {
            let w = W_BLOOM_BASE + bit / 64;
            self.words[w] & (1u64 << (bit % 64)) != 0
        };

        check(h1) && check(h2) && check(h3)
    }

    // -- W48-55: Graph metrics --

    /// In-degree (u32, lower 32 of W48).
    #[inline]
    pub fn in_degree(&self) -> u32 {
        (self.words[W_GRAPH_BASE] & 0xFFFF_FFFF) as u32
    }

    /// Out-degree (u32, upper 32 of W48).
    #[inline]
    pub fn out_degree(&self) -> u32 {
        ((self.words[W_GRAPH_BASE] >> 32) & 0xFFFF_FFFF) as u32
    }

    /// PageRank (f32, lower 32 of W49).
    #[inline]
    pub fn pagerank(&self) -> f32 {
        f32::from_bits((self.words[W_GRAPH_BASE + 1] & 0xFFFF_FFFF) as u32)
    }

    /// Clustering coefficient (f32, upper 32 of W49).
    #[inline]
    pub fn clustering(&self) -> f32 {
        f32::from_bits(((self.words[W_GRAPH_BASE + 1] >> 32) & 0xFFFF_FFFF) as u32)
    }

    // -- W126-127: Checksum --

    /// XOR checksum of words 0..126.
    #[inline]
    pub fn stored_checksum(&self) -> u64 {
        self.words[W_CHECKSUM]
    }

    /// Compute checksum over words 0..126 and compare.
    pub fn verify_checksum(&self) -> bool {
        let mut xor = 0u64;
        for i in 0..W_CHECKSUM {
            xor ^= self.words[i];
        }
        xor == self.words[W_CHECKSUM]
    }
}

// ============================================================================
// META VIEW MUT (read-write)
// ============================================================================

/// Zero-copy mutable view into Container 0 metadata.
pub struct MetaViewMut<'a> {
    words: &'a mut [u64; CONTAINER_WORDS],
}

impl<'a> MetaViewMut<'a> {
    /// Create from a mutable reference to container words.
    pub fn new(words: &'a mut [u64; CONTAINER_WORDS]) -> Self {
        Self { words }
    }

    /// Get a read-only view.
    pub fn as_view(&self) -> MetaView<'_> {
        MetaView { words: self.words }
    }

    // -- W0: DN Address --

    pub fn set_dn_addr(&mut self, addr: u64) {
        self.words[W_DN_ADDR] = addr;
    }

    // -- W1: Type/Geometry --

    pub fn set_node_kind(&mut self, kind: u8) {
        self.words[W_TYPE] = (self.words[W_TYPE] & !0xFF) | (kind as u64);
    }

    pub fn set_container_count(&mut self, count: u8) {
        self.words[W_TYPE] = (self.words[W_TYPE] & !(0xFF << 8))
            | ((count as u64) << 8);
    }

    pub fn set_geometry(&mut self, g: ContainerGeometry) {
        self.words[W_TYPE] = (self.words[W_TYPE] & !(0xFF << 16))
            | ((g as u64) << 16);
    }

    pub fn set_flags(&mut self, flags: u8) {
        self.words[W_TYPE] = (self.words[W_TYPE] & !(0xFF << 24))
            | ((flags as u64) << 24);
    }

    pub fn set_schema_version(&mut self, ver: u16) {
        self.words[W_TYPE] = (self.words[W_TYPE] & !(0xFFFF << 32))
            | ((ver as u64) << 32);
    }

    pub fn set_provenance_hash(&mut self, hash: u16) {
        self.words[W_TYPE] = (self.words[W_TYPE] & !(0xFFFF << 48))
            | ((hash as u64) << 48);
    }

    // -- W2: Timestamps --

    pub fn set_created_ms(&mut self, ms: u32) {
        self.words[W_TIME] = (self.words[W_TIME] & !0xFFFF_FFFF)
            | (ms as u64);
    }

    pub fn set_modified_ms(&mut self, ms: u32) {
        self.words[W_TIME] = (self.words[W_TIME] & !(0xFFFF_FFFF << 32))
            | ((ms as u64) << 32);
    }

    // -- W3: Label --

    pub fn set_label_hash(&mut self, hash: u32) {
        self.words[W_LABEL] = (self.words[W_LABEL] & !0xFFFF_FFFF)
            | (hash as u64);
    }

    pub fn set_tree_depth(&mut self, depth: u8) {
        self.words[W_LABEL] = (self.words[W_LABEL] & !(0xFF << 32))
            | ((depth as u64) << 32);
    }

    pub fn set_branch(&mut self, branch: u8) {
        self.words[W_LABEL] = (self.words[W_LABEL] & !(0xFF << 40))
            | ((branch as u64) << 40);
    }

    // -- W4-7: NARS --

    pub fn set_nars_frequency(&mut self, freq: f32) {
        let bits = freq.to_bits() as u64;
        self.words[W_NARS_BASE] = (self.words[W_NARS_BASE] & !0xFFFF_FFFF) | bits;
    }

    pub fn set_nars_confidence(&mut self, conf: f32) {
        let bits = conf.to_bits() as u64;
        self.words[W_NARS_BASE] = (self.words[W_NARS_BASE] & !(0xFFFF_FFFF << 32))
            | (bits << 32);
    }

    pub fn set_nars_positive_evidence(&mut self, ev: f32) {
        let bits = ev.to_bits() as u64;
        self.words[W_NARS_BASE + 1] = (self.words[W_NARS_BASE + 1] & !0xFFFF_FFFF) | bits;
    }

    pub fn set_nars_negative_evidence(&mut self, ev: f32) {
        let bits = ev.to_bits() as u64;
        self.words[W_NARS_BASE + 1] = (self.words[W_NARS_BASE + 1] & !(0xFFFF_FFFF << 32))
            | (bits << 32);
    }

    // -- W8: DN Rung + Gate --

    pub fn set_rung_level(&mut self, rung: u8) {
        self.words[W_DN_RUNG] = (self.words[W_DN_RUNG] & !0xFF)
            | (rung as u64);
    }

    pub fn set_gate_state(&mut self, gate: u8) {
        self.words[W_DN_RUNG] = (self.words[W_DN_RUNG] & !(0xFF << 8))
            | ((gate as u64) << 8);
    }

    pub fn set_layer_bitmap(&mut self, bitmap: u8) {
        self.words[W_DN_RUNG] = (self.words[W_DN_RUNG] & !(0x7F << 16))
            | (((bitmap & 0x7F) as u64) << 16);
    }

    // -- W16-31: Inline edges --

    /// Write inline edge at index 0..63 as (verb, target).
    pub fn set_inline_edge(&mut self, idx: usize, verb: u8, target: u8) {
        debug_assert!(idx < MAX_INLINE_EDGES);
        let word_idx = W_EDGE_BASE + idx / 4;
        let shift = (idx % 4) * 16;
        let packed = ((verb as u64) << 8) | (target as u64);
        self.words[word_idx] = (self.words[word_idx] & !(0xFFFF << shift))
            | (packed << shift);
    }

    // -- W32-39: RL --

    pub fn set_q_value(&mut self, action: usize, val: f32) {
        debug_assert!(action < 16);
        let word_idx = W_RL_BASE + action / 2;
        let shift = (action % 2) * 32;
        let bits = val.to_bits() as u64;
        self.words[word_idx] = (self.words[word_idx] & !(0xFFFF_FFFF << shift))
            | (bits << shift);
    }

    // -- W40-47: Bloom --

    /// Insert an id into the Bloom filter.
    pub fn bloom_insert(&mut self, id: u64) {
        let h1 = id as usize % 512;
        let h2 = id.wrapping_mul(0x9E3779B97F4A7C15) as usize % 512;
        let h3 = id.wrapping_mul(0x517CC1B727220A95) as usize % 512;

        for bit in [h1, h2, h3] {
            let w = W_BLOOM_BASE + bit / 64;
            self.words[w] |= 1u64 << (bit % 64);
        }
    }

    // -- W48-55: Graph metrics --

    pub fn set_in_degree(&mut self, deg: u32) {
        self.words[W_GRAPH_BASE] = (self.words[W_GRAPH_BASE] & !0xFFFF_FFFF)
            | (deg as u64);
    }

    pub fn set_out_degree(&mut self, deg: u32) {
        self.words[W_GRAPH_BASE] = (self.words[W_GRAPH_BASE] & !(0xFFFF_FFFF << 32))
            | ((deg as u64) << 32);
    }

    pub fn set_pagerank(&mut self, pr: f32) {
        let bits = pr.to_bits() as u64;
        self.words[W_GRAPH_BASE + 1] = (self.words[W_GRAPH_BASE + 1] & !0xFFFF_FFFF)
            | bits;
    }

    pub fn set_clustering(&mut self, cc: f32) {
        let bits = cc.to_bits() as u64;
        self.words[W_GRAPH_BASE + 1] = (self.words[W_GRAPH_BASE + 1] & !(0xFFFF_FFFF << 32))
            | (bits << 32);
    }

    // -- W80-95: Representation descriptor --

    /// Set the branching factor for Tree geometry (stored in W80).
    pub fn set_branching_factor(&mut self, k: u8) {
        self.words[W_REPR_BASE] = (self.words[W_REPR_BASE] & !0xFF)
            | (k as u64);
    }

    // -- W126-127: Checksum --

    /// Recompute and store XOR checksum of words 0..126.
    pub fn update_checksum(&mut self) {
        let mut xor = 0u64;
        for i in 0..W_CHECKSUM {
            xor ^= self.words[i];
        }
        self.words[W_CHECKSUM] = xor;
    }

    /// Initialize metadata with geometry + schema version.
    pub fn init(&mut self, geometry: ContainerGeometry, content_count: u8) {
        self.set_geometry(geometry);
        self.set_container_count(content_count + 1); // +1 for meta container
        self.set_schema_version(SCHEMA_VERSION);
    }
}
