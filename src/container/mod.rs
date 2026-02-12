//! Container-based cognitive record architecture.
//!
//! Every record is built from aligned 8,192-bit containers (128 × u64, 1 KB).
//! Container 0 is always metadata. Containers 1..N hold content whose
//! interpretation is determined by the geometry field in metadata.
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────────┐
//! │  Container 0  (1 KB)  METADATA: identity, NARS, edges, rung  │
//! ├───────────────────────────────────────────────────────────────┤
//! │  Container 1+ (1 KB)  CONTENT: CAM / XYZ / Bridge / Tree     │
//! └───────────────────────────────────────────────────────────────┘
//! ```

pub mod geometry;
pub mod meta;
pub mod record;
pub mod cache;
pub mod search;
pub mod semiring;
pub mod delta;
pub mod spine;
pub mod insert;
pub mod adjacency;
pub mod graph;
pub mod dn_redis;
pub mod traversal;
pub mod migrate;
#[cfg(test)]
pub mod tests;

// Re-export primary types
pub use geometry::ContainerGeometry;
pub use record::CogRecord;
pub use meta::{MetaView, MetaViewMut};
pub use cache::ContainerCache;

// ============================================================================
// CONTAINER CONSTANTS
// ============================================================================

/// Bits per container (2^13)
pub const CONTAINER_BITS: usize = 8_192;

/// u64 words per container
pub const CONTAINER_WORDS: usize = CONTAINER_BITS / 64; // 128

/// Bytes per container
pub const CONTAINER_BYTES: usize = CONTAINER_WORDS * 8; // 1024

/// AVX-512 iterations per container (128 / 8 = 16, zero remainder)
pub const CONTAINER_AVX512_ITERS: usize = CONTAINER_WORDS / 8; // 16

/// Maximum containers per record (u8 max)
pub const MAX_CONTAINERS: usize = 255;

/// Expected Hamming distance between random containers = n/2
pub const EXPECTED_DISTANCE: u32 = (CONTAINER_BITS / 2) as u32; // 4096

/// Standard deviation: sqrt(n/4) ≈ 45.25
pub const SIGMA: f64 = 45.254833995939045;

/// Integer-approximate sigma
pub const SIGMA_APPROX: u32 = 45;

// ============================================================================
// CONTAINER TYPE
// ============================================================================

/// A single 8K-bit container. Cache-line aligned.
#[derive(Clone, PartialEq, Eq)]
#[repr(C, align(64))]
pub struct Container {
    pub words: [u64; CONTAINER_WORDS],
}

impl Container {
    /// All-zero container.
    #[inline]
    pub fn zero() -> Self {
        Self { words: [0u64; CONTAINER_WORDS] }
    }

    /// All-ones container.
    #[inline]
    pub fn ones() -> Self {
        Self { words: [u64::MAX; CONTAINER_WORDS] }
    }

    /// Deterministic pseudo-random container from seed (xorshift64).
    pub fn random(seed: u64) -> Self {
        // SplitMix64 pre-mixer to ensure distinct seeds produce distinct sequences
        let mut z = seed.wrapping_add(0x9e3779b97f4a7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        let mut state = (z ^ (z >> 31)) | 1;
        let mut words = [0u64; CONTAINER_WORDS];
        for w in words.iter_mut() {
            // xorshift64
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *w = state;
        }
        Self { words }
    }

    /// XOR binding: `self ⊕ other`
    #[inline]
    pub fn xor(&self, other: &Container) -> Container {
        let mut result = Container::zero();
        for i in 0..CONTAINER_WORDS {
            result.words[i] = self.words[i] ^ other.words[i];
        }
        result
    }

    /// Hamming distance between two containers.
    #[inline]
    pub fn hamming(&self, other: &Container) -> u32 {
        let mut dist = 0u32;
        for i in 0..CONTAINER_WORDS {
            dist += (self.words[i] ^ other.words[i]).count_ones();
        }
        dist
    }

    /// Similarity: 1.0 - hamming / CONTAINER_BITS
    #[inline]
    pub fn similarity(&self, other: &Container) -> f32 {
        1.0 - (self.hamming(other) as f32 / CONTAINER_BITS as f32)
    }

    /// Total set bits.
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

    /// Majority-vote bundle with even-count tie-breaker (first item's bit).
    pub fn bundle(items: &[&Container]) -> Container {
        if items.is_empty() {
            return Container::zero();
        }
        if items.len() == 1 {
            return items[0].clone();
        }

        let threshold = items.len() / 2;
        let even = items.len() % 2 == 0;
        let mut result = Container::zero();

        for word in 0..CONTAINER_WORDS {
            let mut out = 0u64;
            for bit in 0..64 {
                let mask = 1u64 << bit;
                let count = items.iter()
                    .filter(|item| item.words[word] & mask != 0)
                    .count();
                if count > threshold
                    || (even && count == threshold
                        && items[0].words[word] & mask != 0)
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
                let weighted_count: f32 = items.iter()
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

    /// Bit rotation for sequence encoding (circular shift by `positions` bits).
    pub fn permute(&self, positions: i32) -> Container {
        if positions == 0 {
            return self.clone();
        }

        let total_bits = CONTAINER_BITS as i32;
        // Normalize to positive rotation amount
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
                result.words[i] = (self.words[lo_src] << bit_shift)
                    | (self.words[hi_src] >> complement);
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

    /// Zero-copy byte view.
    #[inline]
    pub fn as_bytes(&self) -> &[u8; CONTAINER_BYTES] {
        // SAFETY: Container is repr(C), [u64; 128] has same layout as [u8; 1024]
        unsafe { &*(self.words.as_ptr() as *const [u8; CONTAINER_BYTES]) }
    }

    /// Construct from byte slice.
    pub fn from_bytes(bytes: &[u8; CONTAINER_BYTES]) -> Self {
        let mut words = [0u64; CONTAINER_WORDS];
        for (i, chunk) in bytes.chunks_exact(8).enumerate() {
            words[i] = u64::from_le_bytes(chunk.try_into().unwrap());
        }
        Self { words }
    }
}

impl std::fmt::Debug for Container {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let pc = self.popcount();
        write!(f, "Container(popcount={}, words[0]={:#018x})", pc, self.words[0])
    }
}

impl Default for Container {
    fn default() -> Self {
        Self::zero()
    }
}

// ============================================================================
// CONVERSIONS: Container <-> Fingerprint
// ============================================================================

impl From<&crate::core::Fingerprint> for Container {
    /// Take the first 128 words of a 256-word Fingerprint.
    fn from(fp: &crate::core::Fingerprint) -> Self {
        let mut c = Container::zero();
        c.words.copy_from_slice(&fp.as_raw()[..CONTAINER_WORDS]);
        c
    }
}

impl From<&Container> for crate::core::Fingerprint {
    /// Promote a Container to a Fingerprint (zero-extend to 256 words).
    fn from(c: &Container) -> Self {
        let mut data = [0u64; crate::FINGERPRINT_U64];
        data[..CONTAINER_WORDS].copy_from_slice(&c.words);
        crate::core::Fingerprint::from_raw(data)
    }
}
