//! Container 0 metadata layout — zero-copy read/write views.
//!
//! The first container in every [`CogRecord`] is metadata, laid out as:
//!
//! ```text
//! W0       PackedDn address (identity)
//! W1       Type: node_kind(u8) | count(u8) | geometry(u8) | flags(u8)
//!          | schema_version(u16) | provenance_hash(u16)
//! W2       Timestamps: created_ms(u32) | modified_ms(u32)
//! W3       Label hash(u32) | tree_depth(u8) | branch(u8) | reserved(u16)
//! W4-7     NARS: freq(f32) | conf(f32) | pos_evidence(f32) | neg_evidence(f32)
//! W8-11    DN rung(u8) | gate_state(u8) | 7-layer compact | reserved
//! W12-15   7-Layer markers: 5 bytes × 7 layers = 35 bytes
//! W16-31   Inline edges: 64 packed, 4 per word
//! W32-39   RL / Q-values / rewards
//! W40-47   Bloom filter (512 bits)
//! W48-55   Graph metrics (full precision)
//! W56-63   Qualia: 18 channels × f16 + 8 slots
//! W64-79   Rung history + collapse gate history
//! W80-95   Representation language descriptor
//! W96-111  DN-Sparse adjacency (compact inline CSR)
//! W112-125 Reserved
//! W126-127 Checksum + version
//! ```

use crate::container::CONTAINER_WORDS;
use crate::geometry::ContainerGeometry;

// Word offsets
pub const W_DN_ADDR: usize = 0;
pub const W_TYPE: usize = 1;
pub const W_TIME: usize = 2;
pub const W_LABEL: usize = 3;
pub const W_NARS_BASE: usize = 4;
pub const W_DN_RUNG: usize = 8;
pub const W_LAYER_BASE: usize = 12;
pub const W_EDGE_BASE: usize = 16;
pub const W_EDGE_END: usize = 31;
pub const MAX_INLINE_EDGES: usize = 64;
pub const W_RL_BASE: usize = 32;
pub const W_BLOOM_BASE: usize = 40;
pub const W_GRAPH_BASE: usize = 48;
pub const W_QUALIA_BASE: usize = 56;
pub const W_RUNG_HIST: usize = 64;
pub const W_REPR_BASE: usize = 80;
pub const W_ADJ_BASE: usize = 96;
pub const W_RESERVED: usize = 112;
pub const W_CHECKSUM: usize = 126;
pub const SCHEMA_VERSION: u16 = 1;

/// Zero-copy read-only view into Container 0 metadata.
pub struct MetaView<'a> {
    words: &'a [u64; CONTAINER_WORDS],
}

impl<'a> MetaView<'a> {
    pub fn new(words: &'a [u64; CONTAINER_WORDS]) -> Self {
        Self { words }
    }

    // -- W0: DN Address --
    #[inline]
    pub fn dn_addr(&self) -> u64 {
        self.words[W_DN_ADDR]
    }

    // -- W1: Type/Geometry --
    #[inline]
    pub fn node_kind(&self) -> u8 {
        (self.words[W_TYPE] & 0xFF) as u8
    }

    #[inline]
    pub fn container_count(&self) -> u8 {
        ((self.words[W_TYPE] >> 8) & 0xFF) as u8
    }

    #[inline]
    pub fn geometry(&self) -> ContainerGeometry {
        let g = ((self.words[W_TYPE] >> 16) & 0xFF) as u8;
        ContainerGeometry::from_u8(g).unwrap_or_default()
    }

    #[inline]
    pub fn flags(&self) -> u8 {
        ((self.words[W_TYPE] >> 24) & 0xFF) as u8
    }

    #[inline]
    pub fn schema_version(&self) -> u16 {
        ((self.words[W_TYPE] >> 32) & 0xFFFF) as u16
    }

    #[inline]
    pub fn provenance_hash(&self) -> u16 {
        ((self.words[W_TYPE] >> 48) & 0xFFFF) as u16
    }

    // -- W2: Timestamps --
    #[inline]
    pub fn created_ms(&self) -> u32 {
        (self.words[W_TIME] & 0xFFFF_FFFF) as u32
    }

    #[inline]
    pub fn modified_ms(&self) -> u32 {
        ((self.words[W_TIME] >> 32) & 0xFFFF_FFFF) as u32
    }

    // -- W3: Label --
    #[inline]
    pub fn label_hash(&self) -> u32 {
        (self.words[W_LABEL] & 0xFFFF_FFFF) as u32
    }

    #[inline]
    pub fn tree_depth(&self) -> u8 {
        ((self.words[W_LABEL] >> 32) & 0xFF) as u8
    }

    #[inline]
    pub fn branch(&self) -> u8 {
        ((self.words[W_LABEL] >> 40) & 0xFF) as u8
    }

    // -- W4-7: NARS --
    #[inline]
    pub fn nars_frequency(&self) -> f32 {
        f32::from_bits((self.words[W_NARS_BASE] & 0xFFFF_FFFF) as u32)
    }

    #[inline]
    pub fn nars_confidence(&self) -> f32 {
        f32::from_bits(((self.words[W_NARS_BASE] >> 32) & 0xFFFF_FFFF) as u32)
    }

    #[inline]
    pub fn nars_positive_evidence(&self) -> f32 {
        f32::from_bits((self.words[W_NARS_BASE + 1] & 0xFFFF_FFFF) as u32)
    }

    #[inline]
    pub fn nars_negative_evidence(&self) -> f32 {
        f32::from_bits(((self.words[W_NARS_BASE + 1] >> 32) & 0xFFFF_FFFF) as u32)
    }

    // -- W8: DN Rung + Gate --
    #[inline]
    pub fn rung_level(&self) -> u8 {
        (self.words[W_DN_RUNG] & 0xFF) as u8
    }

    #[inline]
    pub fn gate_state(&self) -> u8 {
        ((self.words[W_DN_RUNG] >> 8) & 0xFF) as u8
    }

    #[inline]
    pub fn layer_bitmap(&self) -> u8 {
        ((self.words[W_DN_RUNG] >> 16) & 0x7F) as u8
    }

    // -- W12-15: Layer markers --
    /// Per-layer activation data. 5 bytes per layer packed into words 12-15.
    /// Returns (strength, frequency, recency, flags) for layer 0-6.
    pub fn layer_marker(&self, layer: usize) -> (u8, u8, u16, u8) {
        debug_assert!(layer < 7);
        let byte_offset = layer * 5;
        let word_idx = W_LAYER_BASE + byte_offset / 8;
        let bit_offset = (byte_offset % 8) * 8;

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
    #[inline]
    pub fn inline_edge(&self, idx: usize) -> (u8, u8) {
        debug_assert!(idx < MAX_INLINE_EDGES);
        let word_idx = W_EDGE_BASE + idx / 4;
        let shift = (idx % 4) * 16;
        let packed = ((self.words[word_idx] >> shift) & 0xFFFF) as u16;
        ((packed >> 8) as u8, (packed & 0xFF) as u8)
    }

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
    #[inline]
    pub fn q_value(&self, action: usize) -> f32 {
        debug_assert!(action < 16);
        let word_idx = W_RL_BASE + action / 2;
        let shift = (action % 2) * 32;
        f32::from_bits(((self.words[word_idx] >> shift) & 0xFFFF_FFFF) as u32)
    }

    // -- W40-47: Bloom --
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
    #[inline]
    pub fn in_degree(&self) -> u32 {
        (self.words[W_GRAPH_BASE] & 0xFFFF_FFFF) as u32
    }

    #[inline]
    pub fn out_degree(&self) -> u32 {
        ((self.words[W_GRAPH_BASE] >> 32) & 0xFFFF_FFFF) as u32
    }

    #[inline]
    pub fn pagerank(&self) -> f32 {
        f32::from_bits((self.words[W_GRAPH_BASE + 1] & 0xFFFF_FFFF) as u32)
    }

    #[inline]
    pub fn clustering(&self) -> f32 {
        f32::from_bits(((self.words[W_GRAPH_BASE + 1] >> 32) & 0xFFFF_FFFF) as u32)
    }

    // -- W126-127: Checksum --
    #[inline]
    pub fn stored_checksum(&self) -> u64 {
        self.words[W_CHECKSUM]
    }

    pub fn verify_checksum(&self) -> bool {
        let mut xor = 0u64;
        for i in 0..W_CHECKSUM {
            xor ^= self.words[i];
        }
        xor == self.words[W_CHECKSUM]
    }
}

/// Zero-copy mutable view into Container 0 metadata.
pub struct MetaViewMut<'a> {
    words: &'a mut [u64; CONTAINER_WORDS],
}

impl<'a> MetaViewMut<'a> {
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
        self.words[W_TYPE] = (self.words[W_TYPE] & !(0xFF << 8)) | ((count as u64) << 8);
    }

    pub fn set_geometry(&mut self, g: ContainerGeometry) {
        self.words[W_TYPE] = (self.words[W_TYPE] & !(0xFF << 16)) | ((g as u64) << 16);
    }

    pub fn set_flags(&mut self, flags: u8) {
        self.words[W_TYPE] = (self.words[W_TYPE] & !(0xFF << 24)) | ((flags as u64) << 24);
    }

    pub fn set_schema_version(&mut self, ver: u16) {
        self.words[W_TYPE] = (self.words[W_TYPE] & !(0xFFFF << 32)) | ((ver as u64) << 32);
    }

    pub fn set_provenance_hash(&mut self, hash: u16) {
        self.words[W_TYPE] = (self.words[W_TYPE] & !(0xFFFF << 48)) | ((hash as u64) << 48);
    }

    // -- W2: Timestamps --
    pub fn set_created_ms(&mut self, ms: u32) {
        self.words[W_TIME] = (self.words[W_TIME] & !0xFFFF_FFFF) | (ms as u64);
    }

    pub fn set_modified_ms(&mut self, ms: u32) {
        self.words[W_TIME] =
            (self.words[W_TIME] & !(0xFFFF_FFFF << 32)) | ((ms as u64) << 32);
    }

    // -- W3: Label --
    pub fn set_label_hash(&mut self, hash: u32) {
        self.words[W_LABEL] = (self.words[W_LABEL] & !0xFFFF_FFFF) | (hash as u64);
    }

    pub fn set_tree_depth(&mut self, depth: u8) {
        self.words[W_LABEL] = (self.words[W_LABEL] & !(0xFF << 32)) | ((depth as u64) << 32);
    }

    pub fn set_branch(&mut self, branch: u8) {
        self.words[W_LABEL] = (self.words[W_LABEL] & !(0xFF << 40)) | ((branch as u64) << 40);
    }

    // -- W4-7: NARS --
    pub fn set_nars_frequency(&mut self, freq: f32) {
        let bits = freq.to_bits() as u64;
        self.words[W_NARS_BASE] = (self.words[W_NARS_BASE] & !0xFFFF_FFFF) | bits;
    }

    pub fn set_nars_confidence(&mut self, conf: f32) {
        let bits = conf.to_bits() as u64;
        self.words[W_NARS_BASE] =
            (self.words[W_NARS_BASE] & !(0xFFFF_FFFF << 32)) | (bits << 32);
    }

    pub fn set_nars_positive_evidence(&mut self, ev: f32) {
        let bits = ev.to_bits() as u64;
        self.words[W_NARS_BASE + 1] = (self.words[W_NARS_BASE + 1] & !0xFFFF_FFFF) | bits;
    }

    pub fn set_nars_negative_evidence(&mut self, ev: f32) {
        let bits = ev.to_bits() as u64;
        self.words[W_NARS_BASE + 1] =
            (self.words[W_NARS_BASE + 1] & !(0xFFFF_FFFF << 32)) | (bits << 32);
    }

    // -- W8: DN Rung + Gate --
    pub fn set_rung_level(&mut self, rung: u8) {
        self.words[W_DN_RUNG] = (self.words[W_DN_RUNG] & !0xFF) | (rung as u64);
    }

    pub fn set_gate_state(&mut self, gate: u8) {
        self.words[W_DN_RUNG] =
            (self.words[W_DN_RUNG] & !(0xFF << 8)) | ((gate as u64) << 8);
    }

    pub fn set_layer_bitmap(&mut self, bitmap: u8) {
        self.words[W_DN_RUNG] =
            (self.words[W_DN_RUNG] & !(0x7F << 16)) | (((bitmap & 0x7F) as u64) << 16);
    }

    // -- W16-31: Inline edges --
    pub fn set_inline_edge(&mut self, idx: usize, verb: u8, target: u8) {
        debug_assert!(idx < MAX_INLINE_EDGES);
        let word_idx = W_EDGE_BASE + idx / 4;
        let shift = (idx % 4) * 16;
        let packed = ((verb as u64) << 8) | (target as u64);
        self.words[word_idx] =
            (self.words[word_idx] & !(0xFFFF << shift)) | (packed << shift);
    }

    // -- W32-39: RL --
    pub fn set_q_value(&mut self, action: usize, val: f32) {
        debug_assert!(action < 16);
        let word_idx = W_RL_BASE + action / 2;
        let shift = (action % 2) * 32;
        let bits = val.to_bits() as u64;
        self.words[word_idx] =
            (self.words[word_idx] & !(0xFFFF_FFFF << shift)) | (bits << shift);
    }

    // -- W40-47: Bloom --
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
        self.words[W_GRAPH_BASE] =
            (self.words[W_GRAPH_BASE] & !0xFFFF_FFFF) | (deg as u64);
    }

    pub fn set_out_degree(&mut self, deg: u32) {
        self.words[W_GRAPH_BASE] =
            (self.words[W_GRAPH_BASE] & !(0xFFFF_FFFF << 32)) | ((deg as u64) << 32);
    }

    pub fn set_pagerank(&mut self, pr: f32) {
        let bits = pr.to_bits() as u64;
        self.words[W_GRAPH_BASE + 1] =
            (self.words[W_GRAPH_BASE + 1] & !0xFFFF_FFFF) | bits;
    }

    pub fn set_clustering(&mut self, cc: f32) {
        let bits = cc.to_bits() as u64;
        self.words[W_GRAPH_BASE + 1] =
            (self.words[W_GRAPH_BASE + 1] & !(0xFFFF_FFFF << 32)) | (bits << 32);
    }

    // -- W80-95: Representation descriptor --
    pub fn set_branching_factor(&mut self, k: u8) {
        self.words[W_REPR_BASE] = (self.words[W_REPR_BASE] & !0xFF) | (k as u64);
    }

    // -- W126-127: Checksum --
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
