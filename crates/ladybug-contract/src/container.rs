//! 8192-bit Container — the atomic unit of LadybugDB.
//!
//! A Container is a 1 KB bit-vector: 128 × u64 = 8,192 bits.
//! All cognitive operations (XOR bind, Hamming distance, majority-vote bundle)
//! operate at this granularity.

/// Total bits in a single container.
pub const CONTAINER_BITS: usize = 8_192;
/// Number of u64 words in a container.
pub const CONTAINER_WORDS: usize = CONTAINER_BITS / 64; // 128
/// Number of bytes in a container.
pub const CONTAINER_BYTES: usize = CONTAINER_WORDS * 8; // 1024
/// AVX-512 iteration count (8 words per 512-bit register).
pub const CONTAINER_AVX512_ITERS: usize = CONTAINER_WORDS / 8; // 16
/// Maximum content containers in a CogRecord.
pub const MAX_CONTAINERS: usize = 255;
/// Expected Hamming distance for two random containers.
pub const EXPECTED_DISTANCE: u32 = (CONTAINER_BITS / 2) as u32; // 4096
/// Standard deviation of Hamming distance for random containers.
pub const SIGMA: f64 = 45.254833995939045;
/// Integer approximation of sigma.
pub const SIGMA_APPROX: u32 = 45;

/// 8,192-bit binary container for HDC/VSA operations.
///
/// Cache-line aligned for SIMD; `#[repr(C)]` for binary stability.
#[derive(Clone, PartialEq, Eq)]
#[repr(C, align(64))]
pub struct Container {
    pub words: [u64; CONTAINER_WORDS],
}

impl Container {
    /// All-zero container.
    #[inline]
    pub fn zero() -> Self {
        Self {
            words: [0u64; CONTAINER_WORDS],
        }
    }

    /// All-ones container.
    #[inline]
    pub fn ones() -> Self {
        Self {
            words: [u64::MAX; CONTAINER_WORDS],
        }
    }

    /// Deterministic pseudo-random container (SplitMix64 + xorshift64).
    pub fn random(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9e3779b97f4a7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        let mut state = (z ^ (z >> 31)) | 1;
        let mut words = [0u64; CONTAINER_WORDS];
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
    pub fn xor(&self, other: &Container) -> Container {
        let mut result = Container::zero();
        for i in 0..CONTAINER_WORDS {
            result.words[i] = self.words[i] ^ other.words[i];
        }
        result
    }

    /// Hamming distance (number of differing bits).
    #[inline]
    pub fn hamming(&self, other: &Container) -> u32 {
        let mut dist = 0u32;
        for i in 0..CONTAINER_WORDS {
            dist += (self.words[i] ^ other.words[i]).count_ones();
        }
        dist
    }

    /// Cosine-like similarity: 1.0 − hamming / CONTAINER_BITS.
    #[inline]
    pub fn similarity(&self, other: &Container) -> f32 {
        1.0 - (self.hamming(other) as f32 / CONTAINER_BITS as f32)
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
    pub fn bundle(items: &[&Container]) -> Container {
        if items.is_empty() {
            return Container::zero();
        }
        if items.len() == 1 {
            return items[0].clone();
        }
        let threshold = items.len() / 2;
        let even = items.len().is_multiple_of(2);
        let mut result = Container::zero();
        for word in 0..CONTAINER_WORDS {
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

    /// Weighted majority-vote bundle.
    pub fn bundle_weighted(items: &[(&Container, f32)]) -> Container {
        if items.is_empty() {
            return Container::zero();
        }
        let total_weight: f32 = items.iter().map(|(_, w)| w).sum();
        let half = total_weight / 2.0;
        let mut result = Container::zero();
        for word in 0..CONTAINER_WORDS {
            let mut out = 0u64;
            for bit in 0..64 {
                let mask = 1u64 << bit;
                let weighted_count: f32 = items
                    .iter()
                    .filter(|(c, _)| c.words[word] & mask != 0)
                    .map(|(_, w)| w)
                    .sum();
                if weighted_count > half {
                    out |= mask;
                }
            }
            result.words[word] = out;
        }
        result
    }

    /// Circular bit rotation by `positions` bits.
    pub fn permute(&self, positions: i32) -> Container {
        if positions == 0 {
            return self.clone();
        }
        let total_bits = CONTAINER_BITS as i32;
        let shift = ((positions % total_bits) + total_bits) % total_bits;
        if shift == 0 {
            return self.clone();
        }
        let word_shift = (shift / 64) as usize;
        let bit_shift = (shift % 64) as u32;
        let mut result = Container::zero();
        if bit_shift == 0 {
            for i in 0..CONTAINER_WORDS {
                let src = (i + CONTAINER_WORDS - word_shift) % CONTAINER_WORDS;
                result.words[i] = self.words[src];
            }
        } else {
            let complement = 64 - bit_shift;
            for i in 0..CONTAINER_WORDS {
                let lo_src = (i + CONTAINER_WORDS - word_shift) % CONTAINER_WORDS;
                let hi_src = (lo_src + CONTAINER_WORDS - 1) % CONTAINER_WORDS;
                result.words[i] =
                    (self.words[lo_src] << bit_shift) | (self.words[hi_src] >> complement);
            }
        }
        result
    }

    /// Get a single bit.
    #[inline]
    pub fn get_bit(&self, idx: usize) -> bool {
        debug_assert!(idx < CONTAINER_BITS);
        let word = idx / 64;
        let bit = idx % 64;
        self.words[word] & (1u64 << bit) != 0
    }

    /// Set a single bit.
    #[inline]
    pub fn set_bit(&mut self, idx: usize, val: bool) {
        debug_assert!(idx < CONTAINER_BITS);
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
    pub fn as_bytes(&self) -> &[u8; CONTAINER_BYTES] {
        // SAFETY: Container is repr(C), [u64; 128] has same layout as [u8; 1024]
        unsafe { &*(self.words.as_ptr() as *const [u8; CONTAINER_BYTES]) }
    }

    /// Construct from byte slice (little-endian).
    pub fn from_bytes(bytes: &[u8; CONTAINER_BYTES]) -> Self {
        let mut words = [0u64; CONTAINER_WORDS];
        for (i, chunk) in bytes.chunks_exact(8).enumerate() {
            words[i] = u64::from_le_bytes(chunk.try_into().unwrap());
        }
        Self { words }
    }

    /// Zero-cost borrow a `[u64; 128]` as `&Container`.
    ///
    /// # Safety
    /// Caller must guarantee 64-byte alignment.
    #[inline(always)]
    pub fn view(words: &[u64; CONTAINER_WORDS]) -> &Container {
        assert!(
            (words.as_ptr() as usize).is_multiple_of(64),
            "Container::view requires 64-byte aligned input"
        );
        unsafe { &*(words.as_ptr() as *const Container) }
    }

    /// Zero-cost mutable borrow.
    ///
    /// # Safety
    /// Caller must guarantee 64-byte alignment.
    #[inline(always)]
    pub fn view_mut(words: &mut [u64; CONTAINER_WORDS]) -> &mut Container {
        assert!(
            (words.as_ptr() as usize).is_multiple_of(64),
            "Container::view_mut requires 64-byte aligned input"
        );
        unsafe { &mut *(words.as_mut_ptr() as *mut Container) }
    }
}

impl std::fmt::Debug for Container {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let pc = self.popcount();
        write!(
            f,
            "Container(popcount={}, words[0]={:#018x})",
            pc, self.words[0]
        )
    }
}

impl Default for Container {
    fn default() -> Self {
        Self::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_container_hamming_distance() {
        let a = Container::random(42);
        let b = Container::random(99);
        let d = a.hamming(&b);
        assert!(d > 3500 && d < 4700, "hamming={d}, expected ~4096");
    }

    #[test]
    fn test_zero_popcount() {
        assert_eq!(Container::zero().popcount(), 0);
    }

    #[test]
    fn test_ones_popcount() {
        assert_eq!(Container::ones().popcount(), CONTAINER_BITS as u32);
    }

    #[test]
    fn test_xor_self_is_zero() {
        let a = Container::random(1);
        assert!(a.xor(&a).is_zero());
    }

    #[test]
    fn test_bundle_single() {
        let a = Container::random(7);
        let bundled = Container::bundle(&[&a]);
        assert_eq!(bundled, a);
    }

    #[test]
    fn test_permute_roundtrip() {
        let a = Container::random(42);
        let rotated = a.permute(100);
        let back = rotated.permute(-100);
        assert_eq!(a, back);
    }

    #[test]
    fn test_get_set_bit() {
        let mut c = Container::zero();
        assert!(!c.get_bit(1000));
        c.set_bit(1000, true);
        assert!(c.get_bit(1000));
        c.set_bit(1000, false);
        assert!(!c.get_bit(1000));
    }

    #[test]
    fn test_bytes_roundtrip() {
        let a = Container::random(42);
        let bytes = a.as_bytes();
        let b = Container::from_bytes(bytes);
        assert_eq!(a, b);
    }
}
