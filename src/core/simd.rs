//! SIMD-accelerated Hamming distance computation.
//!
//! All SIMD dispatch is handled by rustynum-core at runtime.
//! This module is a thin wrapper — ladybug-rs NEVER reimplements SIMD.
//!
//! Dispatch path:
//!   AVX-512 VPOPCNTDQ → AVX2 Harley-Seal → scalar POPCNT
//!   (one binary, all CPUs, runtime CPUID detection)

use crate::FINGERPRINT_U64;
use crate::core::Fingerprint;

/// Compute Hamming distance between two fingerprints.
///
/// Delegates to rustynum's runtime-dispatched SIMD (AVX-512 → AVX2 → scalar).
#[inline]
pub fn hamming_distance(a: &Fingerprint, b: &Fingerprint) -> u32 {
    crate::core::rustynum_accel::fingerprint_hamming(a, b)
}

/// Scalar reference implementation (for tests only).
#[inline]
pub fn hamming_scalar(a: &Fingerprint, b: &Fingerprint) -> u32 {
    let a_data = a.as_raw();
    let b_data = b.as_raw();

    let mut total = 0u32;
    for i in 0..FINGERPRINT_U64 {
        total += (a_data[i] ^ b_data[i]).count_ones();
    }
    total
}

/// Batch Hamming distance computation (parallel)
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

        // Find top-k by distance
        let mut indexed: Vec<(usize, u32)> = distances.into_iter().enumerate().collect();

        // Partial sort for top-k
        let k = k.min(indexed.len());
        indexed.select_nth_unstable_by_key(k.saturating_sub(1), |&(_, d)| d);
        indexed.truncate(k);
        indexed.sort_by_key(|&(_, d)| d);

        // Convert to (index, distance, similarity)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_self_is_zero() {
        let fp = Fingerprint::from_content("test");
        assert_eq!(hamming_distance(&fp, &fp), 0);
    }

    #[test]
    fn test_hamming_inverse() {
        let a = Fingerprint::zero();
        let b = Fingerprint::ones();
        // All bits differ: ones() now sets exactly FINGERPRINT_BITS (16384)
        // since 16384 % 64 == 0, no padding contamination
        assert_eq!(hamming_distance(&a, &b), crate::FINGERPRINT_BITS as u32);
    }

    #[test]
    fn test_hamming_symmetry() {
        let a = Fingerprint::from_content("hello");
        let b = Fingerprint::from_content("world");
        assert_eq!(hamming_distance(&a, &b), hamming_distance(&b, &a));
    }

    #[test]
    fn test_scalar_matches_simd() {
        let a = Fingerprint::from_content("test_a");
        let b = Fingerprint::from_content("test_b");

        let scalar = hamming_scalar(&a, &b);
        let simd = hamming_distance(&a, &b);

        assert_eq!(scalar, simd);
    }
}
