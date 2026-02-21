//! CogRecord8K — 8 KB record: 4 × 16,384-bit WideContainers.
//!
//! The evolution from 2 KB CogRecord to 8 KB CogRecord8K. With AVX-512
//! VPOPCNTDQ you popcount 65,536 bits in 128 instructions — same cost
//! as the old 8,192 in 16. Each container is 2 KB, fits in L1 cache.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │ Container 0: META  (2 KB = 16,384 bits)                         │
//! │   codebook identity + DN address + hashtag zone + NARS truth     │
//! │   + edges + rung/RL + qualia + bloom + graph metrics             │
//! ├──────────────────────────────────────────────────────────────────┤
//! │ Container 1: CAM   (2 KB = 16,384 bits)                         │
//! │   content-addressable memory: searchable fingerprint             │
//! │   Hamming queryable, XOR-bindable, majority-bundlable            │
//! ├──────────────────────────────────────────────────────────────────┤
//! │ Container 2: INDEX (2 KB = 16,384 bits)                         │
//! │   B-tree index / structural position / DN-tree topology          │
//! │   edge adjacency, spine cache, scent index                       │
//! ├──────────────────────────────────────────────────────────────────┤
//! │ Container 3: EMBED (2 KB = 16,384 bits)                         │
//! │   Embedding store — dual distance metric:                        │
//! │   • Binary 16384-bit hash → VPOPCNTDQ (Hamming)                 │
//! │   • Int8 × 1024D or 2048D → VPDPBUSD (VNNI dot-product)         │
//! │   • Int4 × 4096D          → software int4 decode                 │
//! └──────────────────────────────────────────────────────────────────┘
//!       = 8 KB = 65,536 bits = 128 AVX-512 VPOPCNTDQ instructions
//! ```
//!
//! # Neo4j-rs Integration
//!
//! Cypher compiles to rotation + popcount on 4 containers:
//! - MATCH → popcount sweep on Container 1 (CAM)
//! - WHERE → filter on Container 0 (META) fields
//! - RETURN → project from any container
//! - Edge traversal → XOR-unbind on Container 2 (INDEX)
//! - Vector similarity → VNNI dot-product on Container 3 (EMBED)
//!
//! No scan, no index lookup, no serialization.
//! One VPOPCNTDQ pass per container per hop.

use crate::container::Container;
use crate::wide_container::{WideContainer, EmbeddingFormat, WIDE_BYTES};

// =============================================================================
// CONTAINER SLOT INDICES
// =============================================================================

/// Container 0: Metadata (codebook identity, DN, hashtag zone, NARS, edges).
pub const SLOT_META: usize = 0;
/// Container 1: Content-Addressable Memory (searchable fingerprint).
pub const SLOT_CAM: usize = 1;
/// Container 2: B-tree index / structural position / topology.
pub const SLOT_INDEX: usize = 2;
/// Container 3: Embedding store (binary hash or quantized vectors).
pub const SLOT_EMBED: usize = 3;
/// Total containers in a CogRecord8K.
pub const RECORD8K_CONTAINERS: usize = 4;

// =============================================================================
// SIZE CONSTANTS
// =============================================================================

/// Total bits in a CogRecord8K: 4 × 16,384 = 65,536.
pub const RECORD8K_BITS: usize = RECORD8K_CONTAINERS * crate::wide_container::WIDE_BITS;
/// Total bytes in a CogRecord8K: 4 × 2,048 = 8,192.
pub const RECORD8K_BYTES: usize = RECORD8K_CONTAINERS * WIDE_BYTES;
/// Total u64 words in a CogRecord8K: 4 × 256 = 1,024.
pub const RECORD8K_WORDS: usize = RECORD8K_CONTAINERS * crate::wide_container::WIDE_WORDS;
/// AVX-512 VPOPCNTDQ iterations for full record: 4 × 32 = 128.
pub const RECORD8K_AVX512_ITERS: usize = RECORD8K_CONTAINERS * crate::wide_container::WIDE_AVX512_ITERS;

// =============================================================================
// COGRECORD8K
// =============================================================================

/// 8 KB cognitive record: 4 × 16,384-bit WideContainers.
///
/// Stack-allocated, SIMD-aligned, zero-copy. This is the atomic unit for
/// the full cognitive pipeline: popcount-queryable across all 4 containers,
/// with dual distance metrics on the embedding container.
#[derive(Clone, Debug)]
#[repr(C, align(64))]
pub struct CogRecord8K {
    /// Container 0: META — codebook identity + DN + hashtag zone + NARS + edges.
    pub meta: WideContainer,
    /// Container 1: CAM — content-addressable memory (searchable fingerprint).
    pub cam: WideContainer,
    /// Container 2: INDEX — B-tree / structural position / DN-tree topology.
    pub index: WideContainer,
    /// Container 3: EMBED — embedding store (binary hash or int8/int4 vectors).
    pub embed: WideContainer,
}

impl CogRecord8K {
    /// Byte size of a CogRecord8K: 4 × 2,048 = 8,192 bytes.
    pub const SIZE: usize = RECORD8K_BYTES;

    /// Create a new zeroed CogRecord8K.
    pub fn new() -> Self {
        Self {
            meta: WideContainer::zero(),
            cam: WideContainer::zero(),
            index: WideContainer::zero(),
            embed: WideContainer::zero(),
        }
    }

    /// Create with explicit embedding format tag stored in meta.
    pub fn with_embedding_format(format: EmbeddingFormat) -> Self {
        let mut record = Self::new();
        record.set_embedding_format(format);
        record
    }

    // =========================================================================
    // CONTAINER ACCESS
    // =========================================================================

    /// Access container by slot index.
    pub fn container(&self, slot: usize) -> &WideContainer {
        match slot {
            SLOT_META => &self.meta,
            SLOT_CAM => &self.cam,
            SLOT_INDEX => &self.index,
            SLOT_EMBED => &self.embed,
            _ => panic!("invalid container slot: {slot}"),
        }
    }

    /// Mutable access to container by slot index.
    pub fn container_mut(&mut self, slot: usize) -> &mut WideContainer {
        match slot {
            SLOT_META => &mut self.meta,
            SLOT_CAM => &mut self.cam,
            SLOT_INDEX => &mut self.index,
            SLOT_EMBED => &mut self.embed,
            _ => panic!("invalid container slot: {slot}"),
        }
    }

    // =========================================================================
    // HAMMING DISTANCE (per-container and full-record)
    // =========================================================================

    /// Hamming distance on a single container.
    /// 32 VPOPCNTDQ instructions per container.
    #[inline]
    pub fn hamming_container(&self, other: &CogRecord8K, slot: usize) -> u32 {
        self.container(slot).hamming(other.container(slot))
    }

    /// Hamming distance on the CAM container (the primary search target).
    #[inline]
    pub fn hamming_cam(&self, other: &CogRecord8K) -> u32 {
        self.cam.hamming(&other.cam)
    }

    /// Hamming distance on the CAM container against a raw query.
    #[inline]
    pub fn hamming_cam_query(&self, query: &WideContainer) -> u32 {
        self.cam.hamming(query)
    }

    /// Full 65,536-bit Hamming distance across all 4 containers.
    /// 128 VPOPCNTDQ instructions total.
    pub fn hamming_full(&self, other: &CogRecord8K) -> u32 {
        self.meta.hamming(&other.meta)
            + self.cam.hamming(&other.cam)
            + self.index.hamming(&other.index)
            + self.embed.hamming(&other.embed)
    }

    // =========================================================================
    // EMBEDDING DISTANCE (Container 3 dual metrics)
    // =========================================================================

    /// Get the embedding format from metadata.
    pub fn embedding_format(&self) -> EmbeddingFormat {
        // Stored in meta words[255] lower 8 bits (last word of meta container).
        let tag = (self.meta.words[255] & 0xFF) as u8;
        EmbeddingFormat::from_u8(tag).unwrap_or(EmbeddingFormat::Binary16K)
    }

    /// Set the embedding format tag in metadata.
    pub fn set_embedding_format(&mut self, format: EmbeddingFormat) {
        self.meta.words[255] = (self.meta.words[255] & !0xFF) | (format as u64);
    }

    /// Compute distance on the embedding container using the appropriate metric.
    ///
    /// - Binary16K → Hamming distance (VPOPCNTDQ)
    /// - Int8 → dot product (VNNI VPDPBUSD)
    /// - Int4 → software dot product
    ///
    /// Returns (distance_i32, is_hamming). For Hamming, distance is in bits.
    /// For dot-product, distance is the negative dot (lower = more similar).
    pub fn embed_distance(&self, other: &CogRecord8K) -> (i32, bool) {
        match self.embedding_format() {
            EmbeddingFormat::Binary16K => {
                (self.embed.hamming(&other.embed) as i32, true)
            }
            EmbeddingFormat::Int8x1024 => {
                (-self.embed.int8_dot(&other.embed, 1024), false)
            }
            EmbeddingFormat::Int8x2048 => {
                (-self.embed.int8_dot(&other.embed, 2048), false)
            }
            EmbeddingFormat::Int4x1024 => {
                // Int4: decode pairs from bytes, compute dot
                (-int4_dot(&self.embed, &other.embed, 1024), false)
            }
            EmbeddingFormat::Int4x4096 => {
                (-int4_dot(&self.embed, &other.embed, 4096), false)
            }
        }
    }

    /// Cosine similarity on the embedding container.
    pub fn embed_cosine(&self, other: &CogRecord8K) -> f32 {
        match self.embedding_format() {
            EmbeddingFormat::Binary16K => {
                self.embed.similarity(&other.embed)
            }
            EmbeddingFormat::Int8x1024 => {
                self.embed.int8_cosine(&other.embed, 1024)
            }
            EmbeddingFormat::Int8x2048 => {
                self.embed.int8_cosine(&other.embed, 2048)
            }
            _ => {
                // Fallback: treat as binary
                self.embed.similarity(&other.embed)
            }
        }
    }

    // =========================================================================
    // GRAPH OPERATIONS (edges as rotation addresses)
    // =========================================================================

    /// XOR-bind edge: src ⊕ permute(rel, 1) ⊕ permute(tgt, 2).
    /// Stored in the INDEX container.
    pub fn make_edge(
        src: &WideContainer,
        rel: &WideContainer,
        tgt: &WideContainer,
    ) -> WideContainer {
        let r1 = rel.permute(1);
        let t2 = tgt.permute(2);
        src.xor(&r1).xor(&t2)
    }

    /// Recover target: edge ⊕ src ⊕ permute(rel, 1) → unpermute(result, 2).
    pub fn recover_target(
        edge: &WideContainer,
        src: &WideContainer,
        rel: &WideContainer,
    ) -> WideContainer {
        let r1 = rel.permute(1);
        edge.xor(src).xor(&r1).permute(-2)
    }

    /// Recover source: edge ⊕ permute(rel, 1) ⊕ permute(tgt, 2).
    pub fn recover_source(
        edge: &WideContainer,
        rel: &WideContainer,
        tgt: &WideContainer,
    ) -> WideContainer {
        let r1 = rel.permute(1);
        let t2 = tgt.permute(2);
        edge.xor(&r1).xor(&t2)
    }

    // =========================================================================
    // SERIALIZATION
    // =========================================================================

    /// Zero-copy byte view of the entire 8 KB record.
    pub fn as_bytes(&self) -> &[u8; Self::SIZE] {
        unsafe { &*(self as *const CogRecord8K as *const [u8; Self::SIZE]) }
    }

    /// Construct from an 8 KB byte slice.
    pub fn from_bytes(bytes: &[u8; Self::SIZE]) -> Self {
        let meta = WideContainer::from_bytes(bytes[0..WIDE_BYTES].try_into().unwrap());
        let cam = WideContainer::from_bytes(bytes[WIDE_BYTES..2*WIDE_BYTES].try_into().unwrap());
        let index = WideContainer::from_bytes(bytes[2*WIDE_BYTES..3*WIDE_BYTES].try_into().unwrap());
        let embed = WideContainer::from_bytes(bytes[3*WIDE_BYTES..4*WIDE_BYTES].try_into().unwrap());
        Self { meta, cam, index, embed }
    }

    // =========================================================================
    // UPGRADE FROM 2 KB COGRECORD
    // =========================================================================

    /// Promote a legacy 2 KB CogRecord into an 8 KB CogRecord8K.
    ///
    /// - Legacy meta → lower half of Container 0 (META)
    /// - Legacy content → lower half of Container 1 (CAM)
    /// - Containers 2 (INDEX) and 3 (EMBED) are zeroed.
    pub fn from_legacy(record: &crate::record::CogRecord) -> Self {
        Self {
            meta: WideContainer::from_container(&record.meta),
            cam: WideContainer::from_container(&record.content),
            index: WideContainer::zero(),
            embed: WideContainer::zero(),
        }
    }

    /// Downgrade to legacy 2 KB CogRecord (lossy — drops INDEX and EMBED).
    pub fn to_legacy(&self) -> crate::record::CogRecord {
        crate::record::CogRecord {
            meta: self.meta.to_container_lower(),
            content: self.cam.to_container_lower(),
        }
    }
}

impl Default for CogRecord8K {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for CogRecord8K {
    fn eq(&self, other: &Self) -> bool {
        self.meta == other.meta
            && self.cam == other.cam
            && self.index == other.index
            && self.embed == other.embed
    }
}
impl Eq for CogRecord8K {}

// =============================================================================
// INT4 DOT PRODUCT (software path)
// =============================================================================

/// Int4 dot product: each byte contains two int4 values (low nibble, high nibble).
fn int4_dot(a: &WideContainer, b: &WideContainer, dims: usize) -> i32 {
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    let mut accum = 0i32;
    for i in 0..dims / 2 {
        // Two int4 values per byte: low nibble and high nibble
        let a_lo = ((a_bytes[i] & 0x0F) as i8) - 8; // unsigned 0..15 → signed -8..7
        let a_hi = ((a_bytes[i] >> 4) as i8) - 8;
        let b_lo = ((b_bytes[i] & 0x0F) as i8) - 8;
        let b_hi = ((b_bytes[i] >> 4) as i8) - 8;
        accum += (a_lo as i32) * (b_lo as i32);
        accum += (a_hi as i32) * (b_hi as i32);
    }
    accum
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::wide_container::WIDE_BITS;

    #[test]
    fn test_cogrecord8k_size() {
        assert_eq!(RECORD8K_BITS, 65_536);
        assert_eq!(RECORD8K_BYTES, 8_192);
        assert_eq!(RECORD8K_WORDS, 1_024);
        assert_eq!(RECORD8K_AVX512_ITERS, 128);
        assert_eq!(CogRecord8K::SIZE, 8_192);
        assert_eq!(std::mem::size_of::<CogRecord8K>(), 8_192);
    }

    #[test]
    fn test_cogrecord8k_container_access() {
        let mut r = CogRecord8K::new();
        r.cam = WideContainer::random(42);
        assert_eq!(r.container(SLOT_CAM), &r.cam);
        assert_eq!(r.container(SLOT_META), &r.meta);
    }

    #[test]
    fn test_cogrecord8k_hamming_cam() {
        let mut a = CogRecord8K::new();
        let mut b = CogRecord8K::new();
        a.cam = WideContainer::random(1);
        b.cam = WideContainer::random(2);
        let d = a.hamming_cam(&b);
        assert!(d > 7800 && d < 8600, "cam hamming={d}, expected ~8192");
    }

    #[test]
    fn test_cogrecord8k_hamming_full() {
        let mut a = CogRecord8K::new();
        let mut b = CogRecord8K::new();
        for slot in 0..4 {
            *a.container_mut(slot) = WideContainer::random(slot as u64 * 10);
            *b.container_mut(slot) = WideContainer::random(slot as u64 * 10 + 1);
        }
        let full = a.hamming_full(&b);
        // 4 containers × ~8192 expected distance each = ~32768
        assert!(full > 31000 && full < 34500, "full hamming={full}, expected ~32768");
    }

    #[test]
    fn test_cogrecord8k_bytes_roundtrip() {
        let mut r = CogRecord8K::new();
        r.cam = WideContainer::random(42);
        r.meta = WideContainer::random(1);
        r.index = WideContainer::random(2);
        r.embed = WideContainer::random(3);
        let bytes = r.as_bytes();
        let r2 = CogRecord8K::from_bytes(bytes);
        assert_eq!(r, r2);
    }

    #[test]
    fn test_cogrecord8k_legacy_roundtrip() {
        let mut legacy = crate::record::CogRecord::default();
        legacy.content = Container::random(42);
        legacy.meta = Container::random(1);

        let upgraded = CogRecord8K::from_legacy(&legacy);
        let downgraded = upgraded.to_legacy();
        assert_eq!(legacy.meta, downgraded.meta);
        assert_eq!(legacy.content, downgraded.content);
    }

    #[test]
    fn test_cogrecord8k_edge_operations() {
        let src = WideContainer::random(10);
        let rel = WideContainer::random(20);
        let tgt = WideContainer::random(30);

        let edge = CogRecord8K::make_edge(&src, &rel, &tgt);
        let recovered = CogRecord8K::recover_target(&edge, &src, &rel);
        assert_eq!(recovered, tgt);

        let recovered_src = CogRecord8K::recover_source(&edge, &rel, &tgt);
        assert_eq!(recovered_src, src);
    }

    #[test]
    fn test_cogrecord8k_embedding_format() {
        let mut r = CogRecord8K::with_embedding_format(EmbeddingFormat::Int8x1024);
        assert_eq!(r.embedding_format(), EmbeddingFormat::Int8x1024);

        r.set_embedding_format(EmbeddingFormat::Binary16K);
        assert_eq!(r.embedding_format(), EmbeddingFormat::Binary16K);
    }

    #[test]
    fn test_cogrecord8k_embed_distance_binary() {
        let mut a = CogRecord8K::new();
        let mut b = CogRecord8K::new();
        a.embed = WideContainer::random(1);
        b.embed = WideContainer::random(2);
        // Default format is Binary16K
        let (dist, is_hamming) = a.embed_distance(&b);
        assert!(is_hamming);
        assert!(dist > 7800 && dist < 8600);
    }

    #[test]
    fn test_cogrecord8k_embed_distance_int8() {
        let mut a = CogRecord8K::with_embedding_format(EmbeddingFormat::Int8x1024);
        let mut b = CogRecord8K::with_embedding_format(EmbeddingFormat::Int8x1024);

        // Store known int8 embeddings
        let vals_a: Vec<i8> = (0..1024).map(|i| ((i % 127) as i8).wrapping_sub(64)).collect();
        let vals_b: Vec<i8> = (0..1024).map(|i| ((i % 127) as i8).wrapping_sub(64)).collect();
        a.embed.store_int8(&vals_a);
        b.embed.store_int8(&vals_b);

        let (dist, is_hamming) = a.embed_distance(&b);
        assert!(!is_hamming);
        // Same vector → dot product should be positive → negative dot is negative
        assert!(dist < 0, "same vector should have negative distance (high similarity)");
    }

    #[test]
    fn test_avx512_iteration_count() {
        // The key insight: 128 VPOPCNTDQ instructions for the full 8KB record
        // Same cost as 8 old containers (8 × 16 = 128)
        assert_eq!(RECORD8K_AVX512_ITERS, 128);

        // Per container: 32 iterations
        assert_eq!(crate::wide_container::WIDE_AVX512_ITERS, 32);

        // This means: 4× the data at the same VPOPCNTDQ throughput
        // as 8 legacy containers (which would be 8 × 16 = 128 iterations)
    }
}
