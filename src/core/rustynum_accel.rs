//! RustyNum-accelerated operations for Fingerprint and Container.
//!
//! Provides runtime-dispatched AVX-512 kernels when the `rustynum` feature
//! is enabled. Key advantages over ladybug's built-in SIMD:
//!
//! - **Runtime dispatch** (`is_x86_feature_detected!`) — same binary works
//!   on any x86_64 CPU vs compile-time `#[cfg(target_feature)]`
//! - **VNNI int8 dot product** — new capability for embedding containers
//! - **Optimized bundle** — ripple-carry > per-bit counting for large n
//! - **Zero-copy bridge** — `view_u64_as_bytes` reinterprets `[u64; N]` as `&[u8]`
//!   without allocation
//!
//! ## Usage
//!
//! ```rust,ignore
//! use ladybug::core::rustynum_accel::*;
//!
//! let fp = Fingerprint::from_content("hello");
//! let pc = fingerprint_popcount(&fp);   // VPOPCNTDQ when available
//! let hd = fingerprint_hamming(&fp, &fp); // VPOPCNTDQ when available
//! ```

use crate::core::Fingerprint;
use crate::FINGERPRINT_U64;
use ladybug_contract::Container;
use ladybug_contract::container::CONTAINER_WORDS;

// ────────────────────────────────────────────────────────────────
// Zero-copy reinterpretation: [u64] → [u8]
// ────────────────────────────────────────────────────────────────

/// Reinterpret a `&[u64]` slice as `&[u8]` without allocation.
///
/// # Safety justification
/// u64 is valid for any bit pattern, and u64 alignment >= u8 alignment.
/// The memory layout of `[u64; N]` is `N * 8` contiguous bytes.
#[inline]
pub fn view_u64_as_bytes(words: &[u64]) -> &[u8] {
    let ptr = words.as_ptr() as *const u8;
    let len = words.len() * 8;
    // SAFETY: u64 has alignment >= u8, and all bit patterns are valid u8.
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

// ────────────────────────────────────────────────────────────────
// Fingerprint operations (16384-bit = 2048 bytes = 256 u64 words)
// ────────────────────────────────────────────────────────────────

/// Popcount a Fingerprint using runtime-dispatched VPOPCNTDQ.
///
/// On AVX-512 VPOPCNTDQ hardware: 32 instructions (vs 256 scalar POPCNT).
#[inline]
pub fn fingerprint_popcount(fp: &Fingerprint) -> u32 {
    rustynum_core::simd::popcount(view_u64_as_bytes(fp.as_raw())) as u32
}

/// Hamming distance between two Fingerprints using runtime-dispatched VPOPCNTDQ.
///
/// On AVX-512 VPOPCNTDQ hardware: 32 XOR + 32 VPOPCNTDQ = 64 instructions.
#[inline]
pub fn fingerprint_hamming(a: &Fingerprint, b: &Fingerprint) -> u32 {
    rustynum_core::simd::hamming_distance(
        view_u64_as_bytes(a.as_raw()),
        view_u64_as_bytes(b.as_raw()),
    ) as u32
}

/// Similarity (0.0–1.0) between two Fingerprints.
#[inline]
pub fn fingerprint_similarity(a: &Fingerprint, b: &Fingerprint) -> f32 {
    1.0 - (fingerprint_hamming(a, b) as f32 / (FINGERPRINT_U64 * 64) as f32)
}

/// Signed i8 × i8 dot product on Fingerprint data (both interpreted as i8).
///
/// Uses AVX-512 VNNI (VPDPBUSD) when available: 32 instructions for 2048 bytes.
/// Useful for embedding similarity in CogRecord Container 3.
#[inline]
pub fn fingerprint_dot_i8(a: &Fingerprint, b: &Fingerprint) -> i64 {
    rustynum_core::simd::dot_i8(
        view_u64_as_bytes(a.as_raw()),
        view_u64_as_bytes(b.as_raw()),
    )
}

// ────────────────────────────────────────────────────────────────
// Container operations (16384-bit = 2048 bytes = 256 u64 words)
// ────────────────────────────────────────────────────────────────

/// Popcount a Container using runtime-dispatched VPOPCNTDQ.
///
/// On AVX-512 VPOPCNTDQ hardware: 16 instructions (vs 128 scalar POPCNT).
#[inline]
pub fn container_popcount(c: &Container) -> u32 {
    rustynum_core::simd::popcount(view_u64_as_bytes(&c.words)) as u32
}

/// Hamming distance between two Containers using runtime-dispatched VPOPCNTDQ.
#[inline]
pub fn container_hamming(a: &Container, b: &Container) -> u32 {
    rustynum_core::simd::hamming_distance(
        view_u64_as_bytes(&a.words),
        view_u64_as_bytes(&b.words),
    ) as u32
}

/// Container similarity (0.0–1.0).
#[inline]
pub fn container_similarity(a: &Container, b: &Container) -> f32 {
    1.0 - (container_hamming(a, b) as f32 / (CONTAINER_WORDS * 64) as f32)
}

/// Signed i8 × i8 dot product on Container data.
///
/// Interprets the container bytes as signed int8 values.
/// For embedding containers (CogRecord Container 3).
#[inline]
pub fn container_dot_i8(a: &Container, b: &Container) -> i64 {
    rustynum_core::simd::dot_i8(
        view_u64_as_bytes(&a.words),
        view_u64_as_bytes(&b.words),
    )
}

/// Bundle multiple Containers using rustynum's optimized majority-vote algorithm.
///
/// Zero-copy input: uses `view_u64_as_bytes` to reinterpret Container words
/// as byte slices without allocation. The `bundle_byte_slices` function
/// processes the slices directly — no intermediate NumArrayU8 wrapping.
pub fn container_bundle(items: &[&Container]) -> Container {
    if items.is_empty() {
        return Container::zero();
    }
    if items.len() == 1 {
        return items[0].clone();
    }

    // Zero-copy: view each Container's words as &[u8] (no .to_vec())
    let slices: Vec<&[u8]> = items
        .iter()
        .map(|c| view_u64_as_bytes(&c.words))
        .collect();

    let result_bytes = rustynum_rs::NumArrayU8::try_bundle_byte_slices(&slices)
        .expect("bundle_byte_slices: all slices same length");

    // Convert back to Container
    let mut container = Container::zero();
    debug_assert_eq!(result_bytes.len(), CONTAINER_WORDS * 8);

    // Write bytes back into container words
    for (i, chunk) in result_bytes.chunks_exact(8).enumerate() {
        container.words[i] = u64::from_ne_bytes(chunk.try_into().unwrap());
    }

    container
}

// ────────────────────────────────────────────────────────────────
// Raw slice operations (for arbitrary-length byte arrays)
// ────────────────────────────────────────────────────────────────

/// Popcount on any `&[u64]` slice (zero-copy). Works for any container size.
#[inline]
pub fn slice_popcount(data: &[u64]) -> u64 {
    rustynum_core::simd::popcount(view_u64_as_bytes(data))
}

/// Hamming distance on any two `&[u64]` slices (zero-copy).
#[inline]
pub fn slice_hamming(a: &[u64], b: &[u64]) -> u64 {
    debug_assert_eq!(a.len(), b.len());
    rustynum_core::simd::hamming_distance(view_u64_as_bytes(a), view_u64_as_bytes(b))
}

/// Signed i8 × i8 dot product on any two `&[u64]` slices.
#[inline]
pub fn slice_dot_i8(a: &[u64], b: &[u64]) -> i64 {
    debug_assert_eq!(a.len(), b.len());
    rustynum_core::simd::dot_i8(view_u64_as_bytes(a), view_u64_as_bytes(b))
}

// ────────────────────────────────────────────────────────────────
// Fingerprint-level convenience functions
// ────────────────────────────────────────────────────────────────

/// Compute Hamming distance between two fingerprints.
///
/// Delegates to rustynum's runtime-dispatched SIMD (AVX-512 → AVX2 → scalar).
#[inline]
pub fn hamming_distance(a: &Fingerprint, b: &Fingerprint) -> u32 {
    fingerprint_hamming(a, b)
}

/// Pure-scalar reference Hamming distance (no SIMD intrinsics).
///
/// Used by proof tests to verify SIMD-dispatched results match.
#[inline]
pub fn hamming_scalar(a: &Fingerprint, b: &Fingerprint) -> u32 {
    let a_raw = a.as_raw();
    let b_raw = b.as_raw();
    let mut total = 0u32;
    for i in 0..a_raw.len() {
        total += (a_raw[i] ^ b_raw[i]).count_ones();
    }
    total
}

/// Batch Hamming distance computation (parallel when `parallel` feature is on)
#[cfg(feature = "parallel")]
pub fn batch_hamming(query: &Fingerprint, corpus: &[Fingerprint]) -> Vec<u32> {
    use rayon::prelude::*;
    corpus
        .par_iter()
        .map(|fp| hamming_distance(query, fp))
        .collect()
}

/// Non-parallel batch Hamming
#[cfg(not(feature = "parallel"))]
pub fn batch_hamming(query: &Fingerprint, corpus: &[Fingerprint]) -> Vec<u32> {
    corpus
        .iter()
        .map(|fp| hamming_distance(query, fp))
        .collect()
}

/// Hamming search engine with pre-allocated buffers
pub struct HammingEngine {
    corpus: Vec<Fingerprint>,
    #[cfg(feature = "parallel")]
    thread_pool: rayon::ThreadPool,
}

impl HammingEngine {
    /// Create new engine
    pub fn new() -> Self {
        Self {
            corpus: Vec::new(),
            #[cfg(feature = "parallel")]
            thread_pool: rayon::ThreadPoolBuilder::new()
                .num_threads(num_cpus::get())
                .build()
                .unwrap(),
        }
    }

    /// Index corpus
    pub fn index(&mut self, corpus: Vec<Fingerprint>) {
        self.corpus = corpus;
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &Fingerprint, k: usize) -> Vec<(usize, u32, f32)> {
        let distances = batch_hamming(query, &self.corpus);

        let mut indexed: Vec<(usize, u32)> = distances.into_iter().enumerate().collect();

        let k = k.min(indexed.len());
        indexed.select_nth_unstable_by_key(k.saturating_sub(1), |&(_, d)| d);
        indexed.truncate(k);
        indexed.sort_by_key(|&(_, d)| d);

        indexed
            .into_iter()
            .map(|(idx, dist)| {
                let similarity = 1.0 - (dist as f32 / crate::FINGERPRINT_BITS as f32);
                (idx, dist, similarity)
            })
            .collect()
    }

    /// Search with threshold
    pub fn search_threshold(
        &self,
        query: &Fingerprint,
        threshold: f32,
        limit: usize,
    ) -> Vec<(usize, u32, f32)> {
        let max_distance = ((1.0 - threshold) * crate::FINGERPRINT_BITS as f32) as u32;
        let mut results = self.search(query, limit);
        results.retain(|&(_, dist, _)| dist <= max_distance);
        results
    }

    /// Corpus size
    pub fn len(&self) -> usize {
        self.corpus.len()
    }

    pub fn is_empty(&self) -> bool {
        self.corpus.is_empty()
    }
}

impl Default for HammingEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Detect SIMD capability at runtime
pub fn simd_level() -> &'static str {
    "rustynum-runtime-dispatch"
}

// ────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fingerprint_popcount_zero() {
        let fp = Fingerprint::zero();
        assert_eq!(fingerprint_popcount(&fp), 0);
    }

    #[test]
    fn test_fingerprint_popcount_ones() {
        let fp = Fingerprint::ones();
        assert_eq!(fingerprint_popcount(&fp), (FINGERPRINT_U64 * 64) as u32);
    }

    #[test]
    fn test_fingerprint_popcount_matches_scalar() {
        let fp = Fingerprint::from_content("test popcount");
        let accel = fingerprint_popcount(&fp);
        let scalar = fp.popcount();
        assert_eq!(accel, scalar);
    }

    #[test]
    fn test_fingerprint_hamming_identical() {
        let fp = Fingerprint::from_content("identical");
        assert_eq!(fingerprint_hamming(&fp, &fp), 0);
    }

    #[test]
    fn test_fingerprint_hamming_matches_scalar() {
        let a = Fingerprint::from_content("alpha");
        let b = Fingerprint::from_content("beta");
        let accel = fingerprint_hamming(&a, &b);
        let scalar = a.hamming(&b);
        assert_eq!(accel, scalar);
    }

    #[test]
    fn test_fingerprint_dot_i8_self() {
        let fp = Fingerprint::from_content("dot self");
        let dot = fingerprint_dot_i8(&fp, &fp);
        // Self-dot should always be non-negative (sum of squares)
        assert!(dot >= 0, "Self dot product should be >= 0, got {}", dot);
    }

    #[test]
    fn test_container_popcount_zero() {
        let c = Container::zero();
        assert_eq!(container_popcount(&c), 0);
    }

    #[test]
    fn test_container_popcount_ones() {
        let c = Container::ones();
        assert_eq!(container_popcount(&c), (CONTAINER_WORDS * 64) as u32);
    }

    #[test]
    fn test_container_popcount_matches_scalar() {
        let c = Container::random(42);
        let accel = container_popcount(&c);
        let scalar = c.popcount();
        assert_eq!(accel, scalar);
    }

    #[test]
    fn test_container_hamming_identical() {
        let c = Container::random(42);
        assert_eq!(container_hamming(&c, &c), 0);
    }

    #[test]
    fn test_container_hamming_matches_scalar() {
        let a = Container::random(1);
        let b = Container::random(2);
        let accel = container_hamming(&a, &b);
        let scalar = a.hamming(&b);
        assert_eq!(accel, scalar);
    }

    #[test]
    fn test_container_bundle_unanimous() {
        let a = Container::random(100);
        let result = container_bundle(&[&a, &a, &a]);
        assert_eq!(result, a);
    }

    #[test]
    fn test_container_bundle_majority() {
        let a = Container::ones();
        let b = Container::ones();
        let c = Container::zero();
        let result = container_bundle(&[&a, &b, &c]);
        assert_eq!(result, Container::ones());
    }

    #[test]
    fn test_container_bundle_matches_contract() {
        let a = Container::random(10);
        let b = Container::random(20);
        let c = Container::random(30);
        let accel = container_bundle(&[&a, &b, &c]);
        let contract = Container::bundle(&[&a, &b, &c]);
        assert_eq!(accel, contract);
    }

    #[test]
    fn test_slice_popcount() {
        let data: [u64; 4] = [u64::MAX; 4];
        assert_eq!(slice_popcount(&data), 256);
    }

    #[test]
    fn test_slice_hamming() {
        let a: [u64; 4] = [u64::MAX; 4];
        let b: [u64; 4] = [0; 4];
        assert_eq!(slice_hamming(&a, &b), 256);
    }

    #[test]
    fn test_view_u64_as_bytes_roundtrip() {
        let words: [u64; 2] = [0x0102030405060708, 0x090A0B0C0D0E0F10];
        let bytes = view_u64_as_bytes(&words);
        assert_eq!(bytes.len(), 16);
        // Verify first word's bytes (native endian)
        let first_word = u64::from_ne_bytes(bytes[0..8].try_into().unwrap());
        assert_eq!(first_word, 0x0102030405060708);
    }
}
