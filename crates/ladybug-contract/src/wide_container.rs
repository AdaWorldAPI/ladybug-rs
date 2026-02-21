//! 16,384-bit WideContainer — the 2 KB SIMD unit for CogRecord8K.
//!
//! A WideContainer is 256 × u64 = 16,384 bits = 2 KB.
//! Exactly fits in L1 cache. A full VPOPCNTDQ sweep is 32 instructions
//! (256 words / 8 words per zmm register = 32 iterations).
//!
//! Container 3 of CogRecord8K supports dual distance metrics on the same bits:
//! - Binary Hamming fingerprint → VPOPCNTDQ (popcount on XOR)
//! - Int8 embedding dot-product → VPDPBUSD (AVX-512 VNNI)
//!
//! Same container, same memory, two distance metrics, both hardware-accelerated.

/// Total bits in a wide container.
pub const WIDE_BITS: usize = 16_384;
/// Number of u64 words in a wide container.
pub const WIDE_WORDS: usize = WIDE_BITS / 64; // 256
/// Number of bytes in a wide container.
pub const WIDE_BYTES: usize = WIDE_WORDS * 8; // 2048
/// AVX-512 iteration count (8 words per 512-bit register).
pub const WIDE_AVX512_ITERS: usize = WIDE_WORDS / 8; // 32
/// Expected Hamming distance for two random wide containers.
pub const WIDE_EXPECTED_DISTANCE: u32 = (WIDE_BITS / 2) as u32; // 8192
/// Sigma for random wide container Hamming distance.
pub const WIDE_SIGMA: f64 = 64.0; // sqrt(16384/4) = 64

// =============================================================================
// EMBEDDING FORMAT DESCRIPTORS
// =============================================================================

/// How Container 3 (embedding store) interprets its 16,384 bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum EmbeddingFormat {
    /// 16,384-bit binary hash — pure Hamming search via VPOPCNTDQ.
    /// Best for: binary fingerprints, SimHash, locality-sensitive hashing.
    Binary16K = 0,

    /// 1024-dimensional int8 embedding = 8,192 bits (half container).
    /// Remaining 8,192 bits available for metadata/padding.
    /// Distance: VPDPBUSD (VNNI) for dot-product.
    Int8x1024 = 1,

    /// 2048-dimensional int8 embedding = 16,384 bits (full container).
    /// Distance: VPDPBUSD (VNNI) for dot-product.
    Int8x2048 = 2,

    /// 1024-dimensional int4 embedding = 4,096 bits (quarter container).
    /// Remaining 12,288 bits for metadata/auxiliary data.
    Int4x1024 = 3,

    /// 4096-dimensional int4 embedding = 16,384 bits (full container).
    Int4x4096 = 4,
}

impl EmbeddingFormat {
    /// Number of embedding dimensions.
    pub fn dimensions(self) -> usize {
        match self {
            EmbeddingFormat::Binary16K => WIDE_BITS,
            EmbeddingFormat::Int8x1024 => 1024,
            EmbeddingFormat::Int8x2048 => 2048,
            EmbeddingFormat::Int4x1024 => 1024,
            EmbeddingFormat::Int4x4096 => 4096,
        }
    }

    /// Bits per dimension.
    pub fn bits_per_dim(self) -> usize {
        match self {
            EmbeddingFormat::Binary16K => 1,
            EmbeddingFormat::Int8x1024 | EmbeddingFormat::Int8x2048 => 8,
            EmbeddingFormat::Int4x1024 | EmbeddingFormat::Int4x4096 => 4,
        }
    }

    /// Total bits used by the embedding data.
    pub fn used_bits(self) -> usize {
        self.dimensions() * self.bits_per_dim()
    }

    /// Decode from u8.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Binary16K),
            1 => Some(Self::Int8x1024),
            2 => Some(Self::Int8x2048),
            3 => Some(Self::Int4x1024),
            4 => Some(Self::Int4x4096),
            _ => None,
        }
    }
}

// =============================================================================
// WIDE CONTAINER
// =============================================================================

/// 16,384-bit binary container for CogRecord8K operations.
///
/// Cache-line aligned for SIMD; `#[repr(C)]` for binary stability.
/// Each wide container is exactly 2 KB and fits in L1 cache.
#[derive(Clone, PartialEq, Eq)]
#[repr(C, align(64))]
pub struct WideContainer {
    pub words: [u64; WIDE_WORDS],
}

impl WideContainer {
    /// All-zero wide container.
    #[inline]
    pub fn zero() -> Self {
        Self {
            words: [0u64; WIDE_WORDS],
        }
    }

    /// All-ones wide container.
    #[inline]
    pub fn ones() -> Self {
        Self {
            words: [u64::MAX; WIDE_WORDS],
        }
    }

    /// Deterministic pseudo-random wide container (SplitMix64 + xorshift64).
    pub fn random(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9e3779b97f4a7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        let mut state = (z ^ (z >> 31)) | 1;
        let mut words = [0u64; WIDE_WORDS];
        for w in words.iter_mut() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *w = state;
        }
        Self { words }
    }

    /// XOR binding — the fundamental associative operation.
    #[inline]
    pub fn xor(&self, other: &WideContainer) -> WideContainer {
        let mut result = WideContainer::zero();
        for i in 0..WIDE_WORDS {
            result.words[i] = self.words[i] ^ other.words[i];
        }
        result
    }

    /// Hamming distance (number of differing bits).
    /// Full sweep: 32 VPOPCNTDQ instructions on AVX-512.
    #[inline]
    pub fn hamming(&self, other: &WideContainer) -> u32 {
        let mut dist = 0u32;
        for i in 0..WIDE_WORDS {
            dist += (self.words[i] ^ other.words[i]).count_ones();
        }
        dist
    }

    /// Cosine-like similarity: 1.0 − hamming / WIDE_BITS.
    #[inline]
    pub fn similarity(&self, other: &WideContainer) -> f32 {
        1.0 - (self.hamming(other) as f32 / WIDE_BITS as f32)
    }

    /// Population count (number of set bits).
    #[inline]
    pub fn popcount(&self) -> u32 {
        let mut count = 0u32;
        for &w in &self.words {
            count += w.count_ones();
        }
        count
    }

    /// True if all bits are zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    /// Majority-vote bundle (element-wise voting).
    pub fn bundle(items: &[&WideContainer]) -> WideContainer {
        if items.is_empty() {
            return WideContainer::zero();
        }
        if items.len() == 1 {
            return items[0].clone();
        }
        let threshold = items.len() / 2;
        let even = items.len() % 2 == 0;
        let mut result = WideContainer::zero();
        for word in 0..WIDE_WORDS {
            let mut out = 0u64;
            for bit in 0..64 {
                let mask = 1u64 << bit;
                let count = items
                    .iter()
                    .filter(|item| item.words[word] & mask != 0)
                    .count();
                if count > threshold
                    || (even && count == threshold && items[0].words[word] & mask != 0)
                {
                    out |= mask;
                }
            }
            result.words[word] = out;
        }
        result
    }

    /// Circular bit rotation by `positions` bits.
    pub fn permute(&self, positions: i32) -> WideContainer {
        if positions == 0 {
            return self.clone();
        }
        let total_bits = WIDE_BITS as i32;
        let shift = ((positions % total_bits) + total_bits) % total_bits;
        if shift == 0 {
            return self.clone();
        }
        let word_shift = (shift / 64) as usize;
        let bit_shift = (shift % 64) as u32;
        let mut result = WideContainer::zero();
        if bit_shift == 0 {
            for i in 0..WIDE_WORDS {
                let src = (i + WIDE_WORDS - word_shift) % WIDE_WORDS;
                result.words[i] = self.words[src];
            }
        } else {
            let complement = 64 - bit_shift;
            for i in 0..WIDE_WORDS {
                let lo_src = (i + WIDE_WORDS - word_shift) % WIDE_WORDS;
                let hi_src = (lo_src + WIDE_WORDS - 1) % WIDE_WORDS;
                result.words[i] =
                    (self.words[lo_src] << bit_shift) | (self.words[hi_src] >> complement);
            }
        }
        result
    }

    /// Get a single bit.
    #[inline]
    pub fn get_bit(&self, idx: usize) -> bool {
        debug_assert!(idx < WIDE_BITS);
        let word = idx / 64;
        let bit = idx % 64;
        self.words[word] & (1u64 << bit) != 0
    }

    /// Set a single bit.
    #[inline]
    pub fn set_bit(&mut self, idx: usize, val: bool) {
        debug_assert!(idx < WIDE_BITS);
        let word = idx / 64;
        let bit = idx % 64;
        if val {
            self.words[word] |= 1u64 << bit;
        } else {
            self.words[word] &= !(1u64 << bit);
        }
    }

    /// Zero-copy byte view (little-endian).
    #[inline]
    pub fn as_bytes(&self) -> &[u8; WIDE_BYTES] {
        unsafe { &*(self.words.as_ptr() as *const [u8; WIDE_BYTES]) }
    }

    /// Construct from byte slice (little-endian).
    pub fn from_bytes(bytes: &[u8; WIDE_BYTES]) -> Self {
        let mut words = [0u64; WIDE_WORDS];
        for (i, chunk) in bytes.chunks_exact(8).enumerate() {
            words[i] = u64::from_le_bytes(chunk.try_into().unwrap());
        }
        Self { words }
    }

    // =========================================================================
    // INT8 EMBEDDING OPERATIONS (Container 3 dual-metric support)
    // =========================================================================

    /// Interpret the container as an int8 embedding and compute dot product.
    ///
    /// On AVX-512 VNNI hardware this maps to VPDPBUSD:
    /// native int8 multiply-accumulate at 512 bits wide.
    ///
    /// For int8×1024D: uses first 1024 bytes (words 0..128).
    /// For int8×2048D: uses all 2048 bytes (words 0..256).
    pub fn int8_dot(&self, other: &WideContainer, dims: usize) -> i32 {
        debug_assert!(dims <= WIDE_BYTES); // max 2048 int8 values
        let a_bytes = self.as_bytes();
        let b_bytes = other.as_bytes();
        let mut accum = 0i32;
        for i in 0..dims {
            let a_val = a_bytes[i] as i8 as i32;
            let b_val = b_bytes[i] as i8 as i32;
            accum += a_val * b_val;
        }
        accum
    }

    /// Store an int8 embedding into this container.
    ///
    /// `values` must have length <= 2048 (WIDE_BYTES).
    /// Remaining bytes are zeroed.
    pub fn store_int8(&mut self, values: &[i8]) {
        debug_assert!(values.len() <= WIDE_BYTES);
        *self = WideContainer::zero();
        let bytes = unsafe {
            std::slice::from_raw_parts_mut(
                self.words.as_mut_ptr() as *mut u8,
                WIDE_BYTES,
            )
        };
        for (i, &v) in values.iter().enumerate() {
            bytes[i] = v as u8;
        }
    }

    /// Read int8 values from this container.
    pub fn read_int8(&self, dims: usize) -> Vec<i8> {
        let bytes = self.as_bytes();
        bytes[..dims].iter().map(|&b| b as i8).collect()
    }

    /// Cosine similarity for int8 embeddings (normalized dot product).
    pub fn int8_cosine(&self, other: &WideContainer, dims: usize) -> f32 {
        let dot = self.int8_dot(other, dims) as f64;
        let norm_a = self.int8_dot(self, dims) as f64;
        let norm_b = other.int8_dot(other, dims) as f64;
        let denom = (norm_a * norm_b).sqrt();
        if denom == 0.0 { 0.0 } else { (dot / denom) as f32 }
    }

    // =========================================================================
    // UPSCALE FROM 8192-BIT CONTAINER
    // =========================================================================

    /// Promote an 8192-bit Container into the lower half of a WideContainer.
    pub fn from_container(c: &crate::container::Container) -> WideContainer {
        let mut wide = WideContainer::zero();
        wide.words[..crate::container::CONTAINER_WORDS]
            .copy_from_slice(&c.words);
        wide
    }

    /// Promote an 8192-bit Container, duplicating into both halves.
    pub fn from_container_replicated(c: &crate::container::Container) -> WideContainer {
        let mut wide = WideContainer::zero();
        let n = crate::container::CONTAINER_WORDS;
        wide.words[..n].copy_from_slice(&c.words);
        wide.words[n..].copy_from_slice(&c.words);
        wide
    }

    /// Extract the lower 8192 bits as a Container.
    pub fn to_container_lower(&self) -> crate::container::Container {
        let n = crate::container::CONTAINER_WORDS;
        let mut words = [0u64; crate::container::CONTAINER_WORDS];
        words.copy_from_slice(&self.words[..n]);
        crate::container::Container { words }
    }

    /// Extract the upper 8192 bits as a Container.
    pub fn to_container_upper(&self) -> crate::container::Container {
        let n = crate::container::CONTAINER_WORDS;
        let mut words = [0u64; crate::container::CONTAINER_WORDS];
        words.copy_from_slice(&self.words[n..]);
        crate::container::Container { words }
    }

    // =========================================================================
    // PARTIAL HAMMING (for HDR early-exit on wide containers)
    // =========================================================================

    /// Hamming distance on a word sub-range [start..start+count].
    #[inline]
    pub fn hamming_partial(&self, other: &WideContainer, start: usize, count: usize) -> u32 {
        let mut dist = 0u32;
        let end = (start + count).min(WIDE_WORDS);
        for i in start..end {
            dist += (self.words[i] ^ other.words[i]).count_ones();
        }
        dist
    }
}

impl std::fmt::Debug for WideContainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let pc = self.popcount();
        write!(
            f,
            "WideContainer(16384-bit, popcount={}, words[0]={:#018x})",
            pc, self.words[0]
        )
    }
}

impl Default for WideContainer {
    fn default() -> Self {
        Self::zero()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wide_container_size() {
        assert_eq!(WIDE_BITS, 16_384);
        assert_eq!(WIDE_WORDS, 256);
        assert_eq!(WIDE_BYTES, 2048);
        assert_eq!(WIDE_AVX512_ITERS, 32);
        assert_eq!(std::mem::size_of::<WideContainer>(), 2048);
    }

    #[test]
    fn test_wide_hamming() {
        let a = WideContainer::random(42);
        let b = WideContainer::random(99);
        let d = a.hamming(&b);
        // Expected ~8192, sigma ~64
        assert!(d > 7800 && d < 8600, "hamming={d}, expected ~8192");
    }

    #[test]
    fn test_wide_xor_self() {
        let a = WideContainer::random(1);
        assert!(a.xor(&a).is_zero());
    }

    #[test]
    fn test_wide_popcount() {
        assert_eq!(WideContainer::zero().popcount(), 0);
        assert_eq!(WideContainer::ones().popcount(), WIDE_BITS as u32);
    }

    #[test]
    fn test_wide_permute_roundtrip() {
        let a = WideContainer::random(42);
        let rotated = a.permute(200);
        let back = rotated.permute(-200);
        assert_eq!(a, back);
    }

    #[test]
    fn test_wide_bytes_roundtrip() {
        let a = WideContainer::random(42);
        let bytes = a.as_bytes();
        let b = WideContainer::from_bytes(bytes);
        assert_eq!(a, b);
    }

    #[test]
    fn test_wide_from_container() {
        let c = crate::container::Container::random(42);
        let wide = WideContainer::from_container(&c);
        let back = wide.to_container_lower();
        assert_eq!(c, back);
        // Upper half should be zero
        assert!(wide.to_container_upper().is_zero());
    }

    #[test]
    fn test_wide_from_container_replicated() {
        let c = crate::container::Container::random(42);
        let wide = WideContainer::from_container_replicated(&c);
        let lower = wide.to_container_lower();
        let upper = wide.to_container_upper();
        assert_eq!(c, lower);
        assert_eq!(c, upper);
    }

    #[test]
    fn test_int8_dot_product() {
        let mut a = WideContainer::zero();
        let mut b = WideContainer::zero();
        // Store simple int8 vectors
        let vals_a: Vec<i8> = vec![1, 2, 3, 4, 5];
        let vals_b: Vec<i8> = vec![5, 4, 3, 2, 1];
        a.store_int8(&vals_a);
        b.store_int8(&vals_b);
        // dot = 1*5 + 2*4 + 3*3 + 4*2 + 5*1 = 5+8+9+8+5 = 35
        assert_eq!(a.int8_dot(&b, 5), 35);
    }

    #[test]
    fn test_int8_roundtrip() {
        let vals: Vec<i8> = (-50..50).collect();
        let mut c = WideContainer::zero();
        c.store_int8(&vals);
        let read = c.read_int8(100);
        assert_eq!(vals, read);
    }

    #[test]
    fn test_embedding_format_sizes() {
        assert_eq!(EmbeddingFormat::Binary16K.used_bits(), 16_384);
        assert_eq!(EmbeddingFormat::Int8x1024.used_bits(), 8_192);
        assert_eq!(EmbeddingFormat::Int8x2048.used_bits(), 16_384);
        assert_eq!(EmbeddingFormat::Int4x1024.used_bits(), 4_096);
        assert_eq!(EmbeddingFormat::Int4x4096.used_bits(), 16_384);
    }

    #[test]
    fn test_wide_bundle() {
        let a = WideContainer::random(1);
        let b = WideContainer::random(2);
        let c = WideContainer::random(3);
        let bundled = WideContainer::bundle(&[&a, &b, &c]);
        // Bundle should be closer to inputs than to random
        let random = WideContainer::random(999);
        let d_a = bundled.hamming(&a);
        let d_rand = bundled.hamming(&random);
        assert!(d_a < d_rand, "bundle should be closer to input a");
    }

    #[test]
    fn test_partial_hamming() {
        let a = WideContainer::random(42);
        let b = WideContainer::random(99);
        // Partial on full range should equal full hamming
        let partial = a.hamming_partial(&b, 0, WIDE_WORDS);
        let full = a.hamming(&b);
        assert_eq!(partial, full);
    }
}
