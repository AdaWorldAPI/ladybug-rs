//! SIMD-accelerated Hamming distance computation.
//!
//! Automatically selects the best implementation:
//! - AVX-512 VPOPCNTDQ (Intel Ice Lake+, AMD Zen 4+)
//! - AVX2 + manual popcount
//! - NEON + CNT (ARM)
//! - Scalar fallback

use crate::core::Fingerprint;
use crate::FINGERPRINT_U64;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Compute Hamming distance between two fingerprints.
/// 
/// Automatically dispatches to the best SIMD implementation available.
#[inline]
pub fn hamming_distance(a: &Fingerprint, b: &Fingerprint) -> u32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512vpopcntdq"))]
    unsafe { return hamming_avx512(a, b); }
    
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512vpopcntdq")))]
    unsafe { return hamming_avx2(a, b); }
    
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    unsafe { return hamming_neon(a, b); }
    
    // Scalar fallback
    hamming_scalar(a, b)
}

/// Scalar implementation (works everywhere)
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

/// AVX-512 with VPOPCNTDQ instruction (fastest)
#[cfg(all(target_arch = "x86_64", target_feature = "avx512vpopcntdq"))]
#[target_feature(enable = "avx512f", enable = "avx512vpopcntdq")]
unsafe fn hamming_avx512(a: &Fingerprint, b: &Fingerprint) -> u32 {
    let a_ptr = a.as_raw().as_ptr();
    let b_ptr = b.as_raw().as_ptr();
    
    let mut sum = _mm512_setzero_si512();
    
    // Process 8 u64 at a time (512 bits)
    let mut i = 0;
    while i + 8 <= FINGERPRINT_U64 {
        let va = _mm512_loadu_si512(a_ptr.add(i) as *const __m512i);
        let vb = _mm512_loadu_si512(b_ptr.add(i) as *const __m512i);
        let xor = _mm512_xor_si512(va, vb);
        let popcnt = _mm512_popcnt_epi64(xor);
        sum = _mm512_add_epi64(sum, popcnt);
        i += 8;
    }
    
    // Horizontal sum
    let mut total = _mm512_reduce_add_epi64(sum) as u32;
    
    // Handle remaining (157 % 8 = 5 remaining)
    while i < FINGERPRINT_U64 {
        total += (*a_ptr.add(i) ^ *b_ptr.add(i)).count_ones();
        i += 1;
    }
    
    total
}

/// AVX2 implementation (fallback for older x86_64)
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn hamming_avx2(a: &Fingerprint, b: &Fingerprint) -> u32 {
    let a_ptr = a.as_raw().as_ptr();
    let b_ptr = b.as_raw().as_ptr();
    
    // Lookup table for 4-bit popcount
    let lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    );
    let low_mask = _mm256_set1_epi8(0x0f);
    
    let mut total_sum = _mm256_setzero_si256();
    
    // Process 4 u64 at a time (256 bits)
    let mut i = 0;
    while i + 4 <= FINGERPRINT_U64 {
        let va = _mm256_loadu_si256(a_ptr.add(i) as *const __m256i);
        let vb = _mm256_loadu_si256(b_ptr.add(i) as *const __m256i);
        let xor = _mm256_xor_si256(va, vb);
        
        // Popcount via lookup table
        let lo = _mm256_and_si256(xor, low_mask);
        let hi = _mm256_and_si256(_mm256_srli_epi16(xor, 4), low_mask);
        let popcnt_lo = _mm256_shuffle_epi8(lookup, lo);
        let popcnt_hi = _mm256_shuffle_epi8(lookup, hi);
        let popcnt = _mm256_add_epi8(popcnt_lo, popcnt_hi);
        
        // Sum bytes
        let sad = _mm256_sad_epu8(popcnt, _mm256_setzero_si256());
        total_sum = _mm256_add_epi64(total_sum, sad);
        
        i += 4;
    }
    
    // Horizontal sum
    let sum_lo = _mm256_extracti128_si256(total_sum, 0);
    let sum_hi = _mm256_extracti128_si256(total_sum, 1);
    let sum128 = _mm_add_epi64(sum_lo, sum_hi);
    let mut total = (_mm_extract_epi64(sum128, 0) + _mm_extract_epi64(sum128, 1)) as u32;
    
    // Handle remaining
    while i < FINGERPRINT_U64 {
        total += (*a_ptr.add(i) ^ *b_ptr.add(i)).count_ones();
        i += 1;
    }
    
    total
}

/// ARM NEON implementation
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon")]
unsafe fn hamming_neon(a: &Fingerprint, b: &Fingerprint) -> u32 {
    let a_ptr = a.as_raw().as_ptr() as *const u8;
    let b_ptr = b.as_raw().as_ptr() as *const u8;
    
    let mut sum = vdupq_n_u64(0);
    
    // Process 16 bytes at a time
    let mut i = 0;
    let byte_len = FINGERPRINT_U64 * 8;
    while i + 16 <= byte_len {
        let va = vld1q_u8(a_ptr.add(i));
        let vb = vld1q_u8(b_ptr.add(i));
        let xor = veorq_u8(va, vb);
        let cnt = vcntq_u8(xor);  // Count bits per byte
        
        // Sum to 64-bit
        let sum16 = vpaddlq_u8(cnt);   // u8 -> u16
        let sum32 = vpaddlq_u16(sum16); // u16 -> u32
        let sum64 = vpaddlq_u32(sum32); // u32 -> u64
        sum = vaddq_u64(sum, sum64);
        
        i += 16;
    }
    
    // Horizontal sum
    let mut total = (vgetq_lane_u64(sum, 0) + vgetq_lane_u64(sum, 1)) as u32;
    
    // Handle remaining bytes
    while i < byte_len {
        total += (*a_ptr.add(i) ^ *b_ptr.add(i)).count_ones();
        i += 1;
    }
    
    total
}

/// Batch Hamming distance computation (parallel)
#[cfg(feature = "parallel")]
pub fn batch_hamming(
    query: &Fingerprint,
    corpus: &[Fingerprint],
) -> Vec<u32> {
    use rayon::prelude::*;
    
    corpus
        .par_iter()
        .map(|fp| hamming_distance(query, fp))
        .collect()
}

/// Non-parallel batch Hamming
#[cfg(not(feature = "parallel"))]
pub fn batch_hamming(
    query: &Fingerprint,
    corpus: &[Fingerprint],
) -> Vec<u32> {
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
        let mut indexed: Vec<(usize, u32)> = distances
            .into_iter()
            .enumerate()
            .collect();
        
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
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512vpopcntdq"))]
    return "avx512-vpopcnt";
    
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512vpopcntdq")))]
    return "avx2";
    
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    return "neon";
    
    #[allow(unreachable_code)]
    "scalar"
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
