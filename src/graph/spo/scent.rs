//! NibbleScent — 48-byte nibble histogram for per-axis type discrimination.
//!
//! Replaces the 5-byte XOR-fold scent for SPO records.
//! Each axis (X, Y, Z) gets a 16-bin histogram of nibble (4-bit) frequencies.
//! Different content types have distinct nibble frequency profiles.

use super::sparse::SparseContainer;

/// 48-byte nibble histogram: 16 bins × 3 axes.
///
/// Stored in meta words 12-17 (6 × u64 = 48 bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct NibbleScent {
    pub x_hist: [u8; 16],
    pub y_hist: [u8; 16],
    pub z_hist: [u8; 16],
}

impl NibbleScent {
    pub const SIZE: usize = 48;
    pub const WORD_COUNT: usize = 6; // 48 bytes / 8 bytes per u64

    /// Zero scent (empty record).
    pub fn zero() -> Self {
        Self {
            x_hist: [0; 16],
            y_hist: [0; 16],
            z_hist: [0; 16],
        }
    }

    /// Compute scent from three sparse axes.
    pub fn from_axes(
        x: &SparseContainer,
        y: &SparseContainer,
        z: &SparseContainer,
    ) -> Self {
        Self {
            x_hist: nibble_histogram(&x.words),
            y_hist: nibble_histogram(&y.words),
            z_hist: nibble_histogram(&z.words),
        }
    }

    /// L1 distance between two scents (sum of absolute bin differences).
    pub fn distance(&self, other: &NibbleScent) -> u32 {
        let mut d = 0u32;
        for i in 0..16 {
            d += (self.x_hist[i] as i32 - other.x_hist[i] as i32).unsigned_abs();
            d += (self.y_hist[i] as i32 - other.y_hist[i] as i32).unsigned_abs();
            d += (self.z_hist[i] as i32 - other.z_hist[i] as i32).unsigned_abs();
        }
        d
    }

    /// Per-axis L1 distances for selective filtering.
    pub fn axis_distances(&self, other: &NibbleScent) -> (u32, u32, u32) {
        let mut dx = 0u32;
        let mut dy = 0u32;
        let mut dz = 0u32;
        for i in 0..16 {
            dx += (self.x_hist[i] as i32 - other.x_hist[i] as i32).unsigned_abs();
            dy += (self.y_hist[i] as i32 - other.y_hist[i] as i32).unsigned_abs();
            dz += (self.z_hist[i] as i32 - other.z_hist[i] as i32).unsigned_abs();
        }
        (dx, dy, dz)
    }

    /// Pack into 6 u64 words (for meta container W12-W17).
    pub fn to_words(&self) -> [u64; 6] {
        let mut words = [0u64; 6];
        // X hist → W12-W13
        words[0] = u64::from_le_bytes(self.x_hist[0..8].try_into().unwrap());
        words[1] = u64::from_le_bytes(self.x_hist[8..16].try_into().unwrap());
        // Y hist → W14-W15
        words[2] = u64::from_le_bytes(self.y_hist[0..8].try_into().unwrap());
        words[3] = u64::from_le_bytes(self.y_hist[8..16].try_into().unwrap());
        // Z hist → W16-W17
        words[4] = u64::from_le_bytes(self.z_hist[0..8].try_into().unwrap());
        words[5] = u64::from_le_bytes(self.z_hist[8..16].try_into().unwrap());
        words
    }

    /// Unpack from 6 u64 words.
    pub fn from_words(words: &[u64; 6]) -> Self {
        let mut s = Self::zero();
        s.x_hist[0..8].copy_from_slice(&words[0].to_le_bytes());
        s.x_hist[8..16].copy_from_slice(&words[1].to_le_bytes());
        s.y_hist[0..8].copy_from_slice(&words[2].to_le_bytes());
        s.y_hist[8..16].copy_from_slice(&words[3].to_le_bytes());
        s.z_hist[0..8].copy_from_slice(&words[4].to_le_bytes());
        s.z_hist[8..16].copy_from_slice(&words[5].to_le_bytes());
        s
    }
}

impl Default for NibbleScent {
    fn default() -> Self {
        Self::zero()
    }
}

/// Compute nibble histogram for a set of u64 words.
///
/// Each u64 has 16 nibbles. Count frequency of each nibble value (0x0-0xF).
fn nibble_histogram(words: &[u64]) -> [u8; 16] {
    let mut hist = [0u32; 16]; // use u32 internally to avoid overflow
    for &w in words {
        let mut val = w;
        for _ in 0..16 {
            hist[(val & 0xF) as usize] += 1;
            val >>= 4;
        }
    }
    // Saturate to u8
    let mut result = [0u8; 16];
    for i in 0..16 {
        result[i] = hist[i].min(255) as u8;
    }
    result
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ladybug_contract::container::Container;

    #[test]
    fn test_scent_size() {
        assert_eq!(std::mem::size_of::<NibbleScent>(), 48);
    }

    #[test]
    fn test_scent_word_roundtrip() {
        let x = SparseContainer::from_dense(&Container::random(1));
        let y = SparseContainer::from_dense(&Container::random(2));
        let z = SparseContainer::from_dense(&Container::random(3));
        let scent = NibbleScent::from_axes(&x, &y, &z);
        let words = scent.to_words();
        let restored = NibbleScent::from_words(&words);
        assert_eq!(scent, restored);
    }

    #[test]
    fn test_scent_zero_for_empty() {
        let empty = SparseContainer::zero();
        let scent = NibbleScent::from_axes(&empty, &empty, &empty);
        assert_eq!(scent, NibbleScent::zero());
    }

    #[test]
    fn test_scent_self_distance_zero() {
        let x = SparseContainer::from_dense(&Container::random(42));
        let y = SparseContainer::from_dense(&Container::random(43));
        let z = SparseContainer::from_dense(&Container::random(44));
        let scent = NibbleScent::from_axes(&x, &y, &z);
        assert_eq!(scent.distance(&scent), 0);
    }

    #[test]
    fn test_scent_different_content_different_scent() {
        let x1 = SparseContainer::from_dense(&Container::random(1));
        let y1 = SparseContainer::from_dense(&Container::random(2));
        let z1 = SparseContainer::from_dense(&Container::random(3));

        let x2 = SparseContainer::from_dense(&Container::random(100));
        let y2 = SparseContainer::from_dense(&Container::random(200));
        let z2 = SparseContainer::from_dense(&Container::random(300));

        let s1 = NibbleScent::from_axes(&x1, &y1, &z1);
        let s2 = NibbleScent::from_axes(&x2, &y2, &z2);

        // Random containers will have similar nibble distributions
        // but not identical
        let dist = s1.distance(&s2);
        assert!(dist > 0, "Different content should produce different scents");
    }

    #[test]
    fn test_nibble_histogram_uniform_random() {
        // A random u64 should have roughly uniform nibble distribution
        let words = vec![0xDEAD_BEEF_CAFE_BABEu64];
        let hist = nibble_histogram(&words);
        // 16 nibbles total, 16 bins → average 1 per bin
        let total: u32 = hist.iter().map(|&h| h as u32).sum();
        assert_eq!(total, 16); // 1 word × 16 nibbles
    }

    #[test]
    fn test_nibble_histogram_all_zeros() {
        let words = vec![0u64; 10];
        let hist = nibble_histogram(&words);
        // All nibbles are 0x0
        assert_eq!(hist[0], 160); // 10 words × 16 nibbles per word = 160
        for i in 1..16 {
            assert_eq!(hist[i], 0);
        }
    }
}
