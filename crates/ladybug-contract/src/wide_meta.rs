//! 16,384-bit WideMetaView — expanded metadata for CogRecord8K.
//!
//! Extends the legacy 8,192-bit MetaView (128 words = W0–W127) to use
//! the full 16,384-bit WideContainer (256 words = W0–W255).
//!
//! ## Layout: Lower 128 words (backward-compatible with MetaView)
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
//! W112-125 Reserved (legacy)
//! W126-127 Checksum + version
//! ```
//!
//! ## Layout: Upper 128 words (NEW — schema expansion)
//!
//! ```text
//! W128-143  SPO Crystal: Subject(u64) | Predicate(u64) | Object(u64) packed triples
//!           16 words = 8 SPO triples (subject × predicate → object)
//! W144-159  Hybrid Crystal: up to 16 f32 attention weights for cross-layer binding
//! W160-175  Extended NARS evidence: horizon(f32) | decay(f32) | derived_freq(f32)
//!           | derived_conf(f32) | evidence_history[8] × f32
//! W176-191  Scent index: 16 scent vectors (source_id(u32) + strength(f32) each)
//! W192-207  Causal graph: 16 causal edges (cause(u32) | effect(u32) per word)
//! W208-223  10-Layer cognitive activations: 10 × f64 = 80 bytes (10 words)
//!           + calibration_error(f64) + meta_confidence(f64) + 4 reserved
//! W224-239  Extended edge overflow: 64 more packed edges (4 per word)
//! W240-251  DN tree spine cache: 12 words for fast path to root
//! W252-253  Extended bloom filter (additional 128 bits)
//! W254      Wide checksum (XOR of W128..W253)
//! W255      Embedding format tag (u8) + wide schema version (u8) + reserved
//! ```

use crate::wide_container::WIDE_WORDS;

// ============================================================================
// Word offsets — Upper 128 words (NEW schema expansion fields)
// ============================================================================

// Lower 128 words: same as meta.rs (W_DN_ADDR through W_CHECKSUM)
// Re-export legacy offsets for completeness
pub use crate::meta::{
    W_DN_ADDR, W_TYPE, W_TIME, W_LABEL, W_NARS_BASE, W_DN_RUNG,
    W_LAYER_BASE, W_EDGE_BASE, W_EDGE_END, MAX_INLINE_EDGES,
    W_RL_BASE, W_BLOOM_BASE, W_GRAPH_BASE, W_QUALIA_BASE,
    W_RUNG_HIST, W_REPR_BASE, W_ADJ_BASE, W_RESERVED, W_CHECKSUM,
};

/// SPO Crystal base: Subject-Predicate-Object triples. 8 triples × 2 words each.
pub const W_SPO_BASE: usize = 128;
/// SPO Crystal end.
pub const W_SPO_END: usize = 143;
/// Max SPO triples in the inline crystal.
pub const MAX_SPO_TRIPLES: usize = 8;

/// Hybrid Crystal: cross-layer attention weights.
pub const W_HYBRID_BASE: usize = 144;
pub const W_HYBRID_END: usize = 159;
/// Max f32 attention weights in the hybrid crystal.
pub const MAX_HYBRID_WEIGHTS: usize = 32; // 16 words × 2 f32 per word

/// Extended NARS evidence fields.
pub const W_EXT_NARS_BASE: usize = 160;
pub const W_EXT_NARS_END: usize = 175;

/// Scent index for gradient-free RL exploration.
pub const W_SCENT_BASE: usize = 176;
pub const W_SCENT_END: usize = 191;
/// Max scent vectors.
pub const MAX_SCENT_ENTRIES: usize = 16;

/// Causal graph edges.
pub const W_CAUSAL_BASE: usize = 192;
pub const W_CAUSAL_END: usize = 207;
/// Max causal edges.
pub const MAX_CAUSAL_EDGES: usize = 16;

/// 10-Layer cognitive activation snapshot.
pub const W_LAYER10_BASE: usize = 208;
pub const W_LAYER10_END: usize = 223;

/// Extended edge overflow (64 more packed edges).
pub const W_EXT_EDGE_BASE: usize = 224;
pub const W_EXT_EDGE_END: usize = 239;
/// Max overflow edges.
pub const MAX_EXT_EDGES: usize = 64;

/// DN tree spine cache for fast path to root.
pub const W_SPINE_BASE: usize = 240;
pub const W_SPINE_END: usize = 251;
/// Max spine entries (depth of DN tree path).
pub const MAX_SPINE_DEPTH: usize = 12;

/// Extended bloom filter (additional 128 bits).
pub const W_EXT_BLOOM_BASE: usize = 252;
pub const W_EXT_BLOOM_END: usize = 253;

/// Wide checksum (XOR of W128..W253).
pub const W_WIDE_CHECKSUM: usize = 254;

/// Embedding format tag + wide schema version.
pub const W_WIDE_TAG: usize = 255;

/// Schema version for the wide metadata layout.
pub const WIDE_SCHEMA_VERSION: u8 = 2;

// ============================================================================
// SPO Crystal Triple
// ============================================================================

/// A Subject-Predicate-Object triple packed into 2 × u64 words.
///
/// Layout per triple:
///   Word A: subject_dn(u32) | predicate_hash(u16) | confidence(u16, Q8.8)
///   Word B: object_dn(u32) | evidence_count(u16) | flags(u16)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpoTriple {
    /// DN address of the subject node.
    pub subject_dn: u32,
    /// Hash of the predicate/relation.
    pub predicate_hash: u16,
    /// NARS confidence as Q8.8 fixed-point (0..255 maps to 0.0..1.0).
    pub confidence_q8: u16,
    /// DN address of the object node.
    pub object_dn: u32,
    /// Evidence count supporting this triple.
    pub evidence_count: u16,
    /// Flags: bit 0 = negated, bit 1 = inferred, bit 2 = temporal.
    pub flags: u16,
}

impl SpoTriple {
    /// Confidence as f32 (0.0–1.0).
    #[inline]
    pub fn confidence(&self) -> f32 {
        self.confidence_q8 as f32 / 255.0
    }

    /// True if this is a negated triple.
    #[inline]
    pub fn is_negated(&self) -> bool {
        self.flags & 1 != 0
    }

    /// True if this triple was inferred (not directly observed).
    #[inline]
    pub fn is_inferred(&self) -> bool {
        self.flags & 2 != 0
    }

    /// True if this is a temporal triple.
    #[inline]
    pub fn is_temporal(&self) -> bool {
        self.flags & 4 != 0
    }

    /// Pack into two u64 words.
    fn pack(&self) -> (u64, u64) {
        let word_a = (self.subject_dn as u64)
            | ((self.predicate_hash as u64) << 32)
            | ((self.confidence_q8 as u64) << 48);
        let word_b = (self.object_dn as u64)
            | ((self.evidence_count as u64) << 32)
            | ((self.flags as u64) << 48);
        (word_a, word_b)
    }

    /// Unpack from two u64 words.
    fn unpack(word_a: u64, word_b: u64) -> Self {
        Self {
            subject_dn: (word_a & 0xFFFF_FFFF) as u32,
            predicate_hash: ((word_a >> 32) & 0xFFFF) as u16,
            confidence_q8: ((word_a >> 48) & 0xFFFF) as u16,
            object_dn: (word_b & 0xFFFF_FFFF) as u32,
            evidence_count: ((word_b >> 32) & 0xFFFF) as u16,
            flags: ((word_b >> 48) & 0xFFFF) as u16,
        }
    }

    /// True if this triple is empty (all zeros).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.subject_dn == 0 && self.object_dn == 0 && self.predicate_hash == 0
    }
}

// ============================================================================
// WideMetaView — read-only
// ============================================================================

/// Zero-copy read-only view into the full 16,384-bit WideContainer metadata.
///
/// Provides access to all legacy fields (W0–W127) plus the expanded schema
/// fields (W128–W255) for SPO Crystal, Hybrid Crystal, extended NARS,
/// scent index, causal graph, 10-layer activations, and DN spine cache.
pub struct WideMetaView<'a> {
    words: &'a [u64; WIDE_WORDS],
}

impl<'a> WideMetaView<'a> {
    pub fn new(words: &'a [u64; WIDE_WORDS]) -> Self {
        Self { words }
    }

    /// Get a MetaView over the words (now same width as WideMetaView).
    pub fn legacy(&self) -> crate::meta::MetaView<'_> {
        // Container and WideContainer are now the same width (256 words)
        crate::meta::MetaView::new(self.words)
    }

    // ====================================================================
    // Delegate all legacy field accessors
    // ====================================================================

    #[inline] pub fn dn_addr(&self) -> u64 { self.words[W_DN_ADDR] }
    #[inline] pub fn node_kind(&self) -> u8 { (self.words[W_TYPE] & 0xFF) as u8 }
    #[inline] pub fn schema_version(&self) -> u16 { ((self.words[W_TYPE] >> 32) & 0xFFFF) as u16 }
    #[inline] pub fn rung_level(&self) -> u8 { (self.words[W_DN_RUNG] & 0xFF) as u8 }
    #[inline] pub fn tree_depth(&self) -> u8 { ((self.words[W_LABEL] >> 32) & 0xFF) as u8 }
    #[inline] pub fn nars_frequency(&self) -> f32 { f32::from_bits((self.words[W_NARS_BASE] & 0xFFFF_FFFF) as u32) }
    #[inline] pub fn nars_confidence(&self) -> f32 { f32::from_bits(((self.words[W_NARS_BASE] >> 32) & 0xFFFF_FFFF) as u32) }

    // ====================================================================
    // NEW: SPO Crystal (W128–W143)
    // ====================================================================

    /// Read an SPO triple by index (0–7).
    pub fn spo_triple(&self, idx: usize) -> SpoTriple {
        debug_assert!(idx < MAX_SPO_TRIPLES);
        let base = W_SPO_BASE + idx * 2;
        SpoTriple::unpack(self.words[base], self.words[base + 1])
    }

    /// Count of non-empty SPO triples.
    pub fn spo_count(&self) -> usize {
        (0..MAX_SPO_TRIPLES)
            .filter(|&i| !self.spo_triple(i).is_empty())
            .count()
    }

    // ====================================================================
    // NEW: Hybrid Crystal (W144–W159)
    // ====================================================================

    /// Read an attention weight from the hybrid crystal.
    /// Index 0–31, stored as f32 (2 per word).
    pub fn hybrid_weight(&self, idx: usize) -> f32 {
        debug_assert!(idx < MAX_HYBRID_WEIGHTS);
        let word = W_HYBRID_BASE + idx / 2;
        let shift = (idx % 2) * 32;
        f32::from_bits(((self.words[word] >> shift) & 0xFFFF_FFFF) as u32)
    }

    /// Read all non-zero hybrid weights.
    pub fn hybrid_weights(&self) -> Vec<f32> {
        (0..MAX_HYBRID_WEIGHTS)
            .map(|i| self.hybrid_weight(i))
            .filter(|&w| w != 0.0)
            .collect()
    }

    // ====================================================================
    // NEW: Extended NARS (W160–W175)
    // ====================================================================

    /// NARS evidence horizon (decay window).
    pub fn nars_horizon(&self) -> f32 {
        f32::from_bits((self.words[W_EXT_NARS_BASE] & 0xFFFF_FFFF) as u32)
    }

    /// NARS evidence decay rate.
    pub fn nars_decay(&self) -> f32 {
        f32::from_bits(((self.words[W_EXT_NARS_BASE] >> 32) & 0xFFFF_FFFF) as u32)
    }

    /// Derived frequency (from inference chain).
    pub fn derived_frequency(&self) -> f32 {
        f32::from_bits((self.words[W_EXT_NARS_BASE + 1] & 0xFFFF_FFFF) as u32)
    }

    /// Derived confidence (from inference chain).
    pub fn derived_confidence(&self) -> f32 {
        f32::from_bits(((self.words[W_EXT_NARS_BASE + 1] >> 32) & 0xFFFF_FFFF) as u32)
    }

    /// Evidence history entry (index 0–7).
    pub fn evidence_history(&self, idx: usize) -> f32 {
        debug_assert!(idx < 8);
        let word = W_EXT_NARS_BASE + 2 + idx / 2;
        let shift = (idx % 2) * 32;
        f32::from_bits(((self.words[word] >> shift) & 0xFFFF_FFFF) as u32)
    }

    // ====================================================================
    // NEW: Scent Index (W176–W191)
    // ====================================================================

    /// Read a scent entry: (source_dn, strength).
    pub fn scent_entry(&self, idx: usize) -> (u32, f32) {
        debug_assert!(idx < MAX_SCENT_ENTRIES);
        let word = W_SCENT_BASE + idx;
        let source_dn = (self.words[word] & 0xFFFF_FFFF) as u32;
        let strength = f32::from_bits(((self.words[word] >> 32) & 0xFFFF_FFFF) as u32);
        (source_dn, strength)
    }

    /// Count of non-zero scent entries.
    pub fn scent_count(&self) -> usize {
        (0..MAX_SCENT_ENTRIES)
            .filter(|&i| {
                let (dn, _) = self.scent_entry(i);
                dn != 0
            })
            .count()
    }

    // ====================================================================
    // NEW: Causal Graph (W192–W207)
    // ====================================================================

    /// Read a causal edge: (cause_dn, effect_dn).
    pub fn causal_edge(&self, idx: usize) -> (u32, u32) {
        debug_assert!(idx < MAX_CAUSAL_EDGES);
        let word = W_CAUSAL_BASE + idx;
        let cause = (self.words[word] & 0xFFFF_FFFF) as u32;
        let effect = ((self.words[word] >> 32) & 0xFFFF_FFFF) as u32;
        (cause, effect)
    }

    // ====================================================================
    // NEW: 10-Layer Cognitive Activations (W208–W223)
    // ====================================================================

    /// Read layer activation (index 0–9) as f64.
    pub fn layer_activation(&self, layer: usize) -> f64 {
        debug_assert!(layer < 10);
        let word = W_LAYER10_BASE + layer;
        f64::from_bits(self.words[word])
    }

    /// Read all 10 layer activations.
    pub fn layer_activations(&self) -> [f64; 10] {
        let mut out = [0.0f64; 10];
        for i in 0..10 {
            out[i] = f64::from_bits(self.words[W_LAYER10_BASE + i]);
        }
        out
    }

    /// Dominant layer (index of highest activation).
    pub fn dominant_layer(&self) -> usize {
        let acts = self.layer_activations();
        acts.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Calibration error (Brier score) from metacognition layer.
    pub fn calibration_error(&self) -> f64 {
        f64::from_bits(self.words[W_LAYER10_BASE + 10])
    }

    /// Meta-confidence (confidence in own confidence estimate).
    pub fn meta_confidence(&self) -> f64 {
        f64::from_bits(self.words[W_LAYER10_BASE + 11])
    }

    // ====================================================================
    // NEW: Extended Edges (W224–W239)
    // ====================================================================

    /// Read an extended edge by index (0–63), same format as inline edges.
    pub fn ext_edge(&self, idx: usize) -> (u8, u8) {
        debug_assert!(idx < MAX_EXT_EDGES);
        let word_idx = W_EXT_EDGE_BASE + idx / 4;
        let shift = (idx % 4) * 16;
        let packed = ((self.words[word_idx] >> shift) & 0xFFFF) as u16;
        ((packed >> 8) as u8, (packed & 0xFF) as u8)
    }

    /// Total edge count (inline + extended).
    pub fn total_edge_count(&self) -> usize {
        let mut count = 0;
        for idx in 0..MAX_INLINE_EDGES {
            let word_idx = W_EDGE_BASE + idx / 4;
            let shift = (idx % 4) * 16;
            let packed = ((self.words[word_idx] >> shift) & 0xFFFF) as u16;
            if packed != 0 {
                count += 1;
            }
        }
        for idx in 0..MAX_EXT_EDGES {
            let (verb, target) = self.ext_edge(idx);
            if verb != 0 || target != 0 {
                count += 1;
            }
        }
        count
    }

    // ====================================================================
    // NEW: DN Tree Spine Cache (W240–W251)
    // ====================================================================

    /// Read a spine entry (DN address of ancestor at depth `idx`).
    /// Index 0 = immediate parent, index 11 = root (if depth ≥ 12).
    pub fn spine_entry(&self, idx: usize) -> u64 {
        debug_assert!(idx < MAX_SPINE_DEPTH);
        self.words[W_SPINE_BASE + idx]
    }

    /// Effective spine depth (number of non-zero spine entries).
    pub fn spine_depth(&self) -> usize {
        (0..MAX_SPINE_DEPTH)
            .take_while(|&i| self.words[W_SPINE_BASE + i] != 0)
            .count()
    }

    // ====================================================================
    // NEW: Wide Checksum + Tag (W254–W255)
    // ====================================================================

    /// Wide schema version (stored in W255 bits 8–15).
    pub fn wide_schema_version(&self) -> u8 {
        ((self.words[W_WIDE_TAG] >> 8) & 0xFF) as u8
    }

    /// Embedding format tag (stored in W255 bits 0–7).
    pub fn embedding_format_tag(&self) -> u8 {
        (self.words[W_WIDE_TAG] & 0xFF) as u8
    }

    /// Verify wide checksum (XOR of W128..W253).
    pub fn verify_wide_checksum(&self) -> bool {
        let mut xor = 0u64;
        for i in 128..W_WIDE_CHECKSUM {
            xor ^= self.words[i];
        }
        xor == self.words[W_WIDE_CHECKSUM]
    }

    /// True if the upper 128 words are all zero (legacy record, not expanded).
    pub fn is_legacy_only(&self) -> bool {
        self.words[128..256].iter().all(|&w| w == 0)
    }
}

// ============================================================================
// WideMetaViewMut — mutable
// ============================================================================

/// Zero-copy mutable view into the full 16,384-bit WideContainer metadata.
pub struct WideMetaViewMut<'a> {
    words: &'a mut [u64; WIDE_WORDS],
}

impl<'a> WideMetaViewMut<'a> {
    pub fn new(words: &'a mut [u64; WIDE_WORDS]) -> Self {
        Self { words }
    }

    /// Read-only access.
    pub fn as_view(&self) -> WideMetaView<'_> {
        WideMetaView { words: self.words }
    }

    /// Get a MetaViewMut over the words (now same width as WideMetaViewMut).
    pub fn legacy_mut(&mut self) -> crate::meta::MetaViewMut<'_> {
        crate::meta::MetaViewMut::new(self.words)
    }

    // ====================================================================
    // SPO Crystal writers
    // ====================================================================

    /// Write an SPO triple at index (0–7).
    pub fn set_spo_triple(&mut self, idx: usize, triple: &SpoTriple) {
        debug_assert!(idx < MAX_SPO_TRIPLES);
        let base = W_SPO_BASE + idx * 2;
        let (word_a, word_b) = triple.pack();
        self.words[base] = word_a;
        self.words[base + 1] = word_b;
    }

    /// Append an SPO triple to the first empty slot.
    /// Returns the slot index, or None if full.
    pub fn append_spo_triple(&mut self, triple: &SpoTriple) -> Option<usize> {
        for idx in 0..MAX_SPO_TRIPLES {
            let base = W_SPO_BASE + idx * 2;
            if self.words[base] == 0 && self.words[base + 1] == 0 {
                self.set_spo_triple(idx, triple);
                return Some(idx);
            }
        }
        None
    }

    // ====================================================================
    // Hybrid Crystal writers
    // ====================================================================

    /// Write an attention weight at index (0–31).
    pub fn set_hybrid_weight(&mut self, idx: usize, weight: f32) {
        debug_assert!(idx < MAX_HYBRID_WEIGHTS);
        let word = W_HYBRID_BASE + idx / 2;
        let shift = (idx % 2) * 32;
        let bits = weight.to_bits() as u64;
        self.words[word] = (self.words[word] & !(0xFFFF_FFFF << shift)) | (bits << shift);
    }

    // ====================================================================
    // Extended NARS writers
    // ====================================================================

    pub fn set_nars_horizon(&mut self, horizon: f32) {
        let bits = horizon.to_bits() as u64;
        self.words[W_EXT_NARS_BASE] = (self.words[W_EXT_NARS_BASE] & !0xFFFF_FFFF) | bits;
    }

    pub fn set_nars_decay(&mut self, decay: f32) {
        let bits = decay.to_bits() as u64;
        self.words[W_EXT_NARS_BASE] =
            (self.words[W_EXT_NARS_BASE] & !(0xFFFF_FFFF << 32)) | (bits << 32);
    }

    pub fn set_derived_frequency(&mut self, freq: f32) {
        let bits = freq.to_bits() as u64;
        self.words[W_EXT_NARS_BASE + 1] =
            (self.words[W_EXT_NARS_BASE + 1] & !0xFFFF_FFFF) | bits;
    }

    pub fn set_derived_confidence(&mut self, conf: f32) {
        let bits = conf.to_bits() as u64;
        self.words[W_EXT_NARS_BASE + 1] =
            (self.words[W_EXT_NARS_BASE + 1] & !(0xFFFF_FFFF << 32)) | (bits << 32);
    }

    pub fn set_evidence_history(&mut self, idx: usize, val: f32) {
        debug_assert!(idx < 8);
        let word = W_EXT_NARS_BASE + 2 + idx / 2;
        let shift = (idx % 2) * 32;
        let bits = val.to_bits() as u64;
        self.words[word] = (self.words[word] & !(0xFFFF_FFFF << shift)) | (bits << shift);
    }

    // ====================================================================
    // Scent Index writers
    // ====================================================================

    pub fn set_scent_entry(&mut self, idx: usize, source_dn: u32, strength: f32) {
        debug_assert!(idx < MAX_SCENT_ENTRIES);
        let word = W_SCENT_BASE + idx;
        self.words[word] = (source_dn as u64) | ((strength.to_bits() as u64) << 32);
    }

    // ====================================================================
    // Causal Graph writers
    // ====================================================================

    pub fn set_causal_edge(&mut self, idx: usize, cause_dn: u32, effect_dn: u32) {
        debug_assert!(idx < MAX_CAUSAL_EDGES);
        let word = W_CAUSAL_BASE + idx;
        self.words[word] = (cause_dn as u64) | ((effect_dn as u64) << 32);
    }

    // ====================================================================
    // 10-Layer Cognitive Activation writers
    // ====================================================================

    pub fn set_layer_activation(&mut self, layer: usize, activation: f64) {
        debug_assert!(layer < 10);
        self.words[W_LAYER10_BASE + layer] = activation.to_bits();
    }

    pub fn set_layer_activations(&mut self, activations: &[f64; 10]) {
        for (i, &a) in activations.iter().enumerate() {
            self.words[W_LAYER10_BASE + i] = a.to_bits();
        }
    }

    pub fn set_calibration_error(&mut self, error: f64) {
        self.words[W_LAYER10_BASE + 10] = error.to_bits();
    }

    pub fn set_meta_confidence(&mut self, conf: f64) {
        self.words[W_LAYER10_BASE + 11] = conf.to_bits();
    }

    // ====================================================================
    // Extended Edge writers
    // ====================================================================

    pub fn set_ext_edge(&mut self, idx: usize, verb: u8, target: u8) {
        debug_assert!(idx < MAX_EXT_EDGES);
        let word_idx = W_EXT_EDGE_BASE + idx / 4;
        let shift = (idx % 4) * 16;
        let packed = ((verb as u64) << 8) | (target as u64);
        self.words[word_idx] =
            (self.words[word_idx] & !(0xFFFF << shift)) | (packed << shift);
    }

    // ====================================================================
    // DN Tree Spine Cache writers
    // ====================================================================

    /// Set spine entry at depth `idx` to an ancestor DN address.
    pub fn set_spine_entry(&mut self, idx: usize, ancestor_dn: u64) {
        debug_assert!(idx < MAX_SPINE_DEPTH);
        self.words[W_SPINE_BASE + idx] = ancestor_dn;
    }

    /// Build the full spine from a path of ancestor addresses (parent first).
    pub fn set_spine(&mut self, ancestors: &[u64]) {
        // Clear spine
        for i in 0..MAX_SPINE_DEPTH {
            self.words[W_SPINE_BASE + i] = 0;
        }
        let n = ancestors.len().min(MAX_SPINE_DEPTH);
        for i in 0..n {
            self.words[W_SPINE_BASE + i] = ancestors[i];
        }
    }

    // ====================================================================
    // Wide Checksum + Tag writers
    // ====================================================================

    /// Set wide schema version.
    pub fn set_wide_schema_version(&mut self, version: u8) {
        self.words[W_WIDE_TAG] =
            (self.words[W_WIDE_TAG] & !(0xFF << 8)) | ((version as u64) << 8);
    }

    /// Set embedding format tag.
    pub fn set_embedding_format_tag(&mut self, tag: u8) {
        self.words[W_WIDE_TAG] = (self.words[W_WIDE_TAG] & !0xFF) | (tag as u64);
    }

    /// Recompute wide checksum (XOR of W128..W253).
    pub fn update_wide_checksum(&mut self) {
        let mut xor = 0u64;
        for i in 128..W_WIDE_CHECKSUM {
            xor ^= self.words[i];
        }
        self.words[W_WIDE_CHECKSUM] = xor;
    }

    /// Initialize the expanded metadata with schema version 2.
    pub fn init_wide(&mut self) {
        self.set_wide_schema_version(WIDE_SCHEMA_VERSION);
        self.update_wide_checksum();
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_words() -> [u64; WIDE_WORDS] {
        [0u64; WIDE_WORDS]
    }

    #[test]
    fn test_spo_triple_roundtrip() {
        let mut words = make_words();
        let triple = SpoTriple {
            subject_dn: 0x1234_5678,
            predicate_hash: 0xABCD,
            confidence_q8: 200,
            object_dn: 0x9ABC_DEF0,
            evidence_count: 42,
            flags: 0b101, // negated + temporal
        };

        let mut view = WideMetaViewMut::new(&mut words);
        view.set_spo_triple(0, &triple);
        let read = view.as_view().spo_triple(0);
        assert_eq!(read, triple);
        assert!(read.is_negated());
        assert!(!read.is_inferred());
        assert!(read.is_temporal());
    }

    #[test]
    fn test_spo_append() {
        let mut words = make_words();
        let mut view = WideMetaViewMut::new(&mut words);

        for i in 0..MAX_SPO_TRIPLES {
            let t = SpoTriple {
                subject_dn: i as u32 + 1,
                predicate_hash: 1,
                confidence_q8: 255,
                object_dn: 100 + i as u32,
                evidence_count: 1,
                flags: 0,
            };
            assert_eq!(view.append_spo_triple(&t), Some(i));
        }

        // Should be full now
        let overflow = SpoTriple {
            subject_dn: 999,
            predicate_hash: 1,
            confidence_q8: 255,
            object_dn: 999,
            evidence_count: 1,
            flags: 0,
        };
        assert_eq!(view.append_spo_triple(&overflow), None);
        assert_eq!(view.as_view().spo_count(), MAX_SPO_TRIPLES);
    }

    #[test]
    fn test_hybrid_weights() {
        let mut words = make_words();
        let mut view = WideMetaViewMut::new(&mut words);
        view.set_hybrid_weight(0, 0.95);
        view.set_hybrid_weight(5, 0.42);
        view.set_hybrid_weight(31, 1.0);

        let read = view.as_view();
        assert!((read.hybrid_weight(0) - 0.95).abs() < 1e-6);
        assert!((read.hybrid_weight(5) - 0.42).abs() < 1e-6);
        assert!((read.hybrid_weight(31) - 1.0).abs() < 1e-6);
        assert_eq!(read.hybrid_weight(1), 0.0);
    }

    #[test]
    fn test_extended_nars() {
        let mut words = make_words();
        let mut view = WideMetaViewMut::new(&mut words);
        view.set_nars_horizon(100.0);
        view.set_nars_decay(0.95);
        view.set_derived_frequency(0.8);
        view.set_derived_confidence(0.6);
        view.set_evidence_history(0, 1.0);
        view.set_evidence_history(7, 0.5);

        let read = view.as_view();
        assert!((read.nars_horizon() - 100.0).abs() < 1e-6);
        assert!((read.nars_decay() - 0.95).abs() < 1e-6);
        assert!((read.derived_frequency() - 0.8).abs() < 1e-6);
        assert!((read.derived_confidence() - 0.6).abs() < 1e-6);
        assert!((read.evidence_history(0) - 1.0).abs() < 1e-6);
        assert!((read.evidence_history(7) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_10_layer_activations() {
        let mut words = make_words();
        let mut view = WideMetaViewMut::new(&mut words);

        let activations = [0.1, 0.2, 0.9, 0.4, 0.5, 0.6, 0.3, 0.8, 0.7, 0.15];
        view.set_layer_activations(&activations);
        view.set_calibration_error(0.05);
        view.set_meta_confidence(0.92);

        let read = view.as_view();
        let read_acts = read.layer_activations();
        for i in 0..10 {
            assert!((read_acts[i] - activations[i]).abs() < 1e-12);
        }
        assert_eq!(read.dominant_layer(), 2); // 0.9 is highest
        assert!((read.calibration_error() - 0.05).abs() < 1e-12);
        assert!((read.meta_confidence() - 0.92).abs() < 1e-12);
    }

    #[test]
    fn test_dn_spine_cache() {
        let mut words = make_words();
        let mut view = WideMetaViewMut::new(&mut words);

        let path = vec![0xA0, 0xB0, 0xC0, 0xD0, 0xE0];
        view.set_spine(&path);

        let read = view.as_view();
        assert_eq!(read.spine_depth(), 5);
        assert_eq!(read.spine_entry(0), 0xA0);
        assert_eq!(read.spine_entry(4), 0xE0);
        assert_eq!(read.spine_entry(5), 0); // beyond path
    }

    #[test]
    fn test_ext_edges() {
        let mut words = make_words();
        let mut view = WideMetaViewMut::new(&mut words);
        view.set_ext_edge(0, 0x0C, 0x42);
        view.set_ext_edge(63, 0xFF, 0x01);

        let read = view.as_view();
        assert_eq!(read.ext_edge(0), (0x0C, 0x42));
        assert_eq!(read.ext_edge(63), (0xFF, 0x01));
        assert_eq!(read.ext_edge(1), (0, 0)); // empty
    }

    #[test]
    fn test_scent_index() {
        let mut words = make_words();
        let mut view = WideMetaViewMut::new(&mut words);
        view.set_scent_entry(0, 100, 0.8);
        view.set_scent_entry(5, 200, 0.3);

        let read = view.as_view();
        let (dn0, s0) = read.scent_entry(0);
        assert_eq!(dn0, 100);
        assert!((s0 - 0.8).abs() < 1e-6);
        let (dn5, s5) = read.scent_entry(5);
        assert_eq!(dn5, 200);
        assert!((s5 - 0.3).abs() < 1e-6);
        assert_eq!(read.scent_count(), 2);
    }

    #[test]
    fn test_causal_edges() {
        let mut words = make_words();
        let mut view = WideMetaViewMut::new(&mut words);
        view.set_causal_edge(0, 42, 99);

        let read = view.as_view();
        let (cause, effect) = read.causal_edge(0);
        assert_eq!(cause, 42);
        assert_eq!(effect, 99);
    }

    #[test]
    fn test_wide_checksum() {
        let mut words = make_words();
        let mut view = WideMetaViewMut::new(&mut words);
        view.set_spo_triple(0, &SpoTriple {
            subject_dn: 1,
            predicate_hash: 2,
            confidence_q8: 255,
            object_dn: 3,
            evidence_count: 4,
            flags: 0,
        });
        view.init_wide();

        let read = view.as_view();
        assert!(read.verify_wide_checksum());
        assert_eq!(read.wide_schema_version(), WIDE_SCHEMA_VERSION);
    }

    #[test]
    fn test_is_legacy_only() {
        let words = make_words();
        let read = WideMetaView::new(&words);
        assert!(read.is_legacy_only());

        let mut words2 = make_words();
        let mut view = WideMetaViewMut::new(&mut words2);
        view.set_spo_triple(0, &SpoTriple {
            subject_dn: 1,
            predicate_hash: 1,
            confidence_q8: 128,
            object_dn: 2,
            evidence_count: 1,
            flags: 0,
        });
        assert!(!view.as_view().is_legacy_only());
    }

    #[test]
    fn test_total_edge_count() {
        let mut words = make_words();
        let mut view = WideMetaViewMut::new(&mut words);

        // Set 2 inline edges via legacy offset
        let shift0 = 0 * 16;
        view.words[W_EDGE_BASE] |= 0x0C42u64 << shift0;
        let shift1 = 1 * 16;
        view.words[W_EDGE_BASE] |= 0xFF01u64 << shift1;

        // Set 1 extended edge
        view.set_ext_edge(0, 0xAB, 0xCD);

        let read = view.as_view();
        assert_eq!(read.total_edge_count(), 3);
    }
}
