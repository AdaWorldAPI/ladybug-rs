//! 10,000-bit VSA fingerprint.

use std::hash::{Hash, Hasher};
use std::fmt;

use crate::{FINGERPRINT_U64, FINGERPRINT_BITS, Error, Result};

/// 10,000-bit binary fingerprint for VSA operations.
/// 
/// Stored as 157 u64 values (157 × 64 = 10,048 bits, 48 padding).
/// Aligned to 64 bytes for SIMD operations.
#[repr(align(64))]
#[derive(Clone)]
pub struct Fingerprint {
    data: [u64; FINGERPRINT_U64],
}

impl Fingerprint {
    /// Create from raw u64 array
    pub fn from_raw(data: [u64; FINGERPRINT_U64]) -> Self {
        Self { data }
    }
    
    /// Create from byte slice
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != FINGERPRINT_U64 * 8 {
            return Err(Error::InvalidFingerprint {
                expected: FINGERPRINT_U64 * 8,
                got: bytes.len(),
            });
        }
        
        let mut data = [0u64; FINGERPRINT_U64];
        for (i, chunk) in bytes.chunks_exact(8).enumerate() {
            data[i] = u64::from_le_bytes(chunk.try_into().unwrap());
        }
        Ok(Self { data })
    }
    
    /// Create from content string (SHA-256 + LFSR expansion)
    pub fn from_content(content: &str) -> Self {
        use std::collections::hash_map::DefaultHasher;
        
        // Hash content to get seed
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        let mut state = hasher.finish();
        
        // LFSR expansion to 10K bits
        let mut data = [0u64; FINGERPRINT_U64];
        for word in &mut data {
            let mut val = 0u64;
            for bit in 0..64 {
                // LFSR tap: x^64 + x^62 + x^61 + x^1 + 1
                let feedback = (state ^ (state >> 2) ^ (state >> 3) ^ (state >> 63)) & 1;
                state = (state >> 1) | (feedback << 63);
                val |= (state & 1) << bit;
            }
            *word = val;
        }
        
        Self { data }
    }
    
    /// Create random fingerprint
    pub fn random() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        Self::from_content(&format!("random_{}", seed))
    }
    
    /// Create zero fingerprint
    pub fn zero() -> Self {
        Self { data: [0u64; FINGERPRINT_U64] }
    }
    
    /// Create all-ones fingerprint
    pub fn ones() -> Self {
        Self { data: [u64::MAX; FINGERPRINT_U64] }
    }
    
    /// Get raw data
    #[inline]
    pub fn as_raw(&self) -> &[u64; FINGERPRINT_U64] {
        &self.data
    }
    
    /// Get as byte slice
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const u8,
                FINGERPRINT_U64 * 8
            )
        }
    }
    
    /// Count set bits (popcount)
    pub fn popcount(&self) -> u32 {
        self.data.iter().map(|x| x.count_ones()).sum()
    }
    
    /// Get bit at position
    #[inline]
    pub fn get_bit(&self, pos: usize) -> bool {
        debug_assert!(pos < FINGERPRINT_BITS);
        let word = pos / 64;
        let bit = pos % 64;
        (self.data[word] >> bit) & 1 == 1
    }
    
    /// Set bit at position
    #[inline]
    pub fn set_bit(&mut self, pos: usize, value: bool) {
        debug_assert!(pos < FINGERPRINT_BITS);
        let word = pos / 64;
        let bit = pos % 64;
        if value {
            self.data[word] |= 1 << bit;
        } else {
            self.data[word] &= !(1 << bit);
        }
    }
    
    /// Hamming distance to another fingerprint
    #[inline]
    pub fn hamming(&self, other: &Fingerprint) -> u32 {
        super::simd::hamming_distance(self, other)
    }
    
    /// Similarity (0.0 - 1.0)
    #[inline]
    pub fn similarity(&self, other: &Fingerprint) -> f32 {
        1.0 - (self.hamming(other) as f32 / FINGERPRINT_BITS as f32)
    }
    
    // === VSA Operations ===
    
    /// XOR bind (creates compound representation)
    pub fn bind(&self, other: &Fingerprint) -> Fingerprint {
        let mut result = [0u64; FINGERPRINT_U64];
        for i in 0..FINGERPRINT_U64 {
            result[i] = self.data[i] ^ other.data[i];
        }
        Fingerprint { data: result }
    }
    
    /// XOR unbind (recovers component)
    /// Note: unbind(bind(a, b), a) ≈ b
    #[inline]
    pub fn unbind(&self, other: &Fingerprint) -> Fingerprint {
        self.bind(other)  // XOR is its own inverse
    }
    
    /// Permute (rotate bits for sequence encoding)
    pub fn permute(&self, positions: i32) -> Fingerprint {
        let mut result = Self::zero();
        let total_bits = FINGERPRINT_BITS;
        let shift = positions.rem_euclid(total_bits as i32) as usize;
        
        for i in 0..total_bits {
            let new_pos = (i + shift) % total_bits;
            if self.get_bit(i) {
                result.set_bit(new_pos, true);
            }
        }
        result
    }
    
    /// Inverse permute
    pub fn unpermute(&self, positions: i32) -> Fingerprint {
        self.permute(-positions)
    }
}

impl PartialEq for Fingerprint {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl Eq for Fingerprint {}

impl Hash for Fingerprint {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.data.hash(state);
    }
}

impl fmt::Debug for Fingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fingerprint({} bits set)", self.popcount())
    }
}

impl Default for Fingerprint {
    fn default() -> Self {
        Self::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_from_content_deterministic() {
        let fp1 = Fingerprint::from_content("hello");
        let fp2 = Fingerprint::from_content("hello");
        assert_eq!(fp1, fp2);
    }
    
    #[test]
    fn test_different_content_different_fp() {
        let fp1 = Fingerprint::from_content("hello");
        let fp2 = Fingerprint::from_content("world");
        assert_ne!(fp1, fp2);
    }
    
    #[test]
    fn test_bind_unbind() {
        let a = Fingerprint::from_content("color_red");
        let b = Fingerprint::from_content("object_apple");
        
        // Bind creates compound
        let bound = a.bind(&b);
        
        // Unbind recovers approximately
        let recovered = bound.unbind(&a);
        
        // Should be identical to b (XOR is exact inverse)
        assert_eq!(recovered, b);
    }
    
    #[test]
    fn test_similarity_range() {
        let fp1 = Fingerprint::random();
        let fp2 = Fingerprint::random();
        
        let sim = fp1.similarity(&fp2);
        assert!(sim >= 0.0 && sim <= 1.0);
        
        // Self-similarity should be 1.0
        assert_eq!(fp1.similarity(&fp1), 1.0);
    }
}

    /// Create orthogonal fingerprint for index (deterministic, well-separated)
    /// 
    /// Uses the index as a seed to generate a fingerprint that is approximately
    /// orthogonal (50% similarity) to fingerprints generated with other indices.
    pub fn orthogonal(index: usize) -> Self {
        let seed = (index as u64).wrapping_mul(0x9E3779B97F4A7C15);
        Self::from_content(&format!("__orthogonal_basis_{:016x}", seed))
    }
    
    /// Convert to owned byte vector
    pub fn to_bytes(&self) -> Vec<u8> {
        self.as_bytes().to_vec()
    }
