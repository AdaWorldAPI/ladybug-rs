//! Crystal4K: Compressed 3 × 10Kbit holographic coordinate.
//!
//! Compresses 5×5×5 × 10Kbit (156KB) → 3 × 10Kbit (4KB) via axis projections.
//! This is NOT a lossy hash - it's a coordinate system for 2^1,250,000 space.
//!
//! ```text
//! 125 cells × 10Kbit = 1.25Mbit = 156KB
//!                ↓ XOR-fold per axis
//! 3 projections × 10Kbit = 30Kbit = 3.75KB ≈ 4KB
//!
//! Compression: 41:1
//! ```

use super::field::{FIELD_SIZE, QuorumField};
use crate::FINGERPRINT_U64;
use crate::core::Fingerprint;

/// 4KB crystal: holographic coordinate in 2^1.25M space.
///
/// Three orthogonal projections encode the full field structure.
/// Like holographic boundary encoding bulk information.
#[repr(C, align(64))]
#[derive(Clone, PartialEq, Eq)]
pub struct Crystal4K {
    /// X-axis projection (fold Y,Z)
    pub x: [u64; FINGERPRINT_U64],

    /// Y-axis projection (fold X,Z)
    pub y: [u64; FINGERPRINT_U64],

    /// Z-axis projection (fold X,Y)
    pub z: [u64; FINGERPRINT_U64],
}

impl Crystal4K {
    /// Create from three fingerprints
    pub fn new(x: Fingerprint, y: Fingerprint, z: Fingerprint) -> Self {
        Self {
            x: *x.as_raw(),
            y: *y.as_raw(),
            z: *z.as_raw(),
        }
    }

    /// Create zero crystal
    pub fn zero() -> Self {
        Self {
            x: [0u64; FINGERPRINT_U64],
            y: [0u64; FINGERPRINT_U64],
            z: [0u64; FINGERPRINT_U64],
        }
    }

    /// Compress QuorumField → Crystal4K
    ///
    /// ```text
    /// 156KB → 4KB via XOR-fold along each axis
    /// ```
    pub fn from_field(field: &QuorumField) -> Self {
        Self {
            x: *field.project_x().as_raw(),
            y: *field.project_y().as_raw(),
            z: *field.project_z().as_raw(),
        }
    }

    /// Get X projection as Fingerprint
    pub fn x_fp(&self) -> Fingerprint {
        Fingerprint::from_raw(self.x)
    }

    /// Get Y projection as Fingerprint
    pub fn y_fp(&self) -> Fingerprint {
        Fingerprint::from_raw(self.y)
    }

    /// Get Z projection as Fingerprint
    pub fn z_fp(&self) -> Fingerprint {
        Fingerprint::from_raw(self.z)
    }

    /// XOR-bind all three projections → unified signature
    ///
    /// This 10Kbit vector is a compact identifier for the crystal.
    pub fn signature(&self) -> Fingerprint {
        let mut result = [0u64; FINGERPRINT_U64];
        for i in 0..FINGERPRINT_U64 {
            result[i] = self.x[i] ^ self.y[i] ^ self.z[i];
        }
        Fingerprint::from_raw(result)
    }

    /// Hamming distance to another crystal (sum of axis distances)
    pub fn distance(&self, other: &Crystal4K) -> u32 {
        let mut total = 0u32;

        for i in 0..FINGERPRINT_U64 {
            total += (self.x[i] ^ other.x[i]).count_ones();
            total += (self.y[i] ^ other.y[i]).count_ones();
            total += (self.z[i] ^ other.z[i]).count_ones();
        }

        total
    }

    /// Similarity (0.0 - 1.0)
    pub fn similarity(&self, other: &Crystal4K) -> f32 {
        let max_bits = 3 * crate::FINGERPRINT_BITS;
        1.0 - (self.distance(other) as f32 / max_bits as f32)
    }

    /// XOR with another crystal (element-wise)
    pub fn xor(&self, other: &Crystal4K) -> Crystal4K {
        let mut result = Crystal4K::zero();
        for i in 0..FINGERPRINT_U64 {
            result.x[i] = self.x[i] ^ other.x[i];
            result.y[i] = self.y[i] ^ other.y[i];
            result.z[i] = self.z[i] ^ other.z[i];
        }
        result
    }

    /// Expand crystal back to approximate field.
    ///
    /// Uses position binding to reconstruct each cell.
    /// This is approximate - the original field had more information.
    pub fn expand(&self) -> QuorumField {
        let mut field = QuorumField::default_threshold();

        for x in 0..FIELD_SIZE {
            for y in 0..FIELD_SIZE {
                for z in 0..FIELD_SIZE {
                    // Position vectors (deterministic seeds)
                    let px = Fingerprint::from_content(&format!("pos_x_{}", x));
                    let py = Fingerprint::from_content(&format!("pos_y_{}", y));
                    let pz = Fingerprint::from_content(&format!("pos_z_{}", z));

                    // Reconstruct: bind projections with positions
                    let x_contribution = self.x_fp().bind(&px);
                    let y_contribution = self.y_fp().bind(&py);
                    let z_contribution = self.z_fp().bind(&pz);

                    // Combine (XOR all contributions)
                    let cell = x_contribution.bind(&y_contribution).bind(&z_contribution);

                    field.set(x, y, z, &cell);
                }
            }
        }

        field
    }

    /// Expand with quorum cleaning (more stable reconstruction)
    pub fn expand_clean(&self, settle_steps: usize) -> QuorumField {
        let mut field = self.expand();
        field.settle(settle_steps);
        field
    }

    /// Total popcount across all projections
    pub fn popcount(&self) -> u32 {
        let mut count = 0u32;
        for i in 0..FINGERPRINT_U64 {
            count += self.x[i].count_ones();
            count += self.y[i].count_ones();
            count += self.z[i].count_ones();
        }
        count
    }

    /// Size in bytes
    pub const fn size_bytes() -> usize {
        3 * FINGERPRINT_U64 * 8
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(Self::size_bytes());

        for &word in &self.x {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        for &word in &self.y {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        for &word in &self.z {
            bytes.extend_from_slice(&word.to_le_bytes());
        }

        bytes
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != Self::size_bytes() {
            return None;
        }

        let mut crystal = Crystal4K::zero();
        let word_size = 8;

        for i in 0..FINGERPRINT_U64 {
            let offset = i * word_size;
            crystal.x[i] = u64::from_le_bytes(bytes[offset..offset + 8].try_into().ok()?);
        }

        let base_y = FINGERPRINT_U64 * word_size;
        for i in 0..FINGERPRINT_U64 {
            let offset = base_y + i * word_size;
            crystal.y[i] = u64::from_le_bytes(bytes[offset..offset + 8].try_into().ok()?);
        }

        let base_z = 2 * FINGERPRINT_U64 * word_size;
        for i in 0..FINGERPRINT_U64 {
            let offset = base_z + i * word_size;
            crystal.z[i] = u64::from_le_bytes(bytes[offset..offset + 8].try_into().ok()?);
        }

        Some(crystal)
    }
}

impl Default for Crystal4K {
    fn default() -> Self {
        Self::zero()
    }
}

impl std::fmt::Debug for Crystal4K {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Crystal4K {{ x: {} bits, y: {} bits, z: {} bits }}",
            self.x.iter().map(|w| w.count_ones()).sum::<u32>(),
            self.y.iter().map(|w| w.count_ones()).sum::<u32>(),
            self.z.iter().map(|w| w.count_ones()).sum::<u32>(),
        )
    }
}

impl std::hash::Hash for Crystal4K {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.x.hash(state);
        self.y.hash(state);
        self.z.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size() {
        assert_eq!(Crystal4K::size_bytes(), 3 * crate::FINGERPRINT_U64 * 8);
        // 6144 bytes = 6KB
    }

    #[test]
    fn test_from_field() {
        let mut field = QuorumField::default();
        let fp = Fingerprint::from_content("test");
        field.inject(&fp);

        let crystal = Crystal4K::from_field(&field);

        // With single non-zero cell, projections capture it
        assert!(crystal.popcount() > 0);
    }

    #[test]
    fn test_xor_self_is_zero() {
        let fp = Fingerprint::from_content("test");
        let crystal = Crystal4K::new(fp.clone(), fp.clone(), fp.clone());

        let zero = crystal.xor(&crystal);
        assert_eq!(zero.popcount(), 0);
    }

    #[test]
    fn test_serialization() {
        let crystal = Crystal4K::new(
            Fingerprint::from_content("x"),
            Fingerprint::from_content("y"),
            Fingerprint::from_content("z"),
        );

        let bytes = crystal.to_bytes();
        let restored = Crystal4K::from_bytes(&bytes).unwrap();

        assert_eq!(crystal, restored);
    }

    #[test]
    fn test_similarity_self() {
        let crystal = Crystal4K::new(
            Fingerprint::from_content("x"),
            Fingerprint::from_content("y"),
            Fingerprint::from_content("z"),
        );

        assert!((crystal.similarity(&crystal) - 1.0).abs() < 0.0001);
    }
}
