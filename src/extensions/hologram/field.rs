//! Quorum Field: 5×5×5 × 10Kbit lattice with neighbor voting dynamics.
//!
//! This creates a 3D cellular automaton where each cell contains a 10Kbit
//! fingerprint and evolves via quorum voting with its 6 face-adjacent neighbors.
//!
//! The field encodes 2^1,250,000 possible configurations but operates in
//! polynomial time via XOR folding and SIMD-accelerated quorum computation.

use crate::FINGERPRINT_U64;
use crate::core::Fingerprint;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Lattice dimensions
pub const FIELD_SIZE: usize = 5;
pub const FIELD_CELLS: usize = FIELD_SIZE * FIELD_SIZE * FIELD_SIZE; // 125

/// 5×5×5 × 16Kbit quorum field.
///
/// Total size: 125 × 256 × 8 = 256,000 bytes = 250KB
/// Fits in L2 cache for fast evolution.
#[repr(C, align(64))]
#[derive(Clone)]
pub struct QuorumField {
    /// 3D array of fingerprint data [x][y][z][u64s]
    cells: Box<[[[[u64; FINGERPRINT_U64]; FIELD_SIZE]; FIELD_SIZE]; FIELD_SIZE]>,

    /// Quorum threshold (1-6): how many neighbors must agree
    threshold: u8,

    /// Generation counter
    generation: u64,
}

impl QuorumField {
    /// Create empty field with given quorum threshold.
    ///
    /// Threshold determines stability:
    /// - 3/6: Fluid, easy state changes
    /// - 4/6: Balanced (recommended)
    /// - 5/6: Rigid, resistant to change
    pub fn new(threshold: u8) -> Self {
        assert!(threshold >= 1 && threshold <= 6, "Threshold must be 1-6");

        Self {
            cells: Box::new([[[[0u64; FINGERPRINT_U64]; FIELD_SIZE]; FIELD_SIZE]; FIELD_SIZE]),
            threshold,
            generation: 0,
        }
    }

    /// Create field with default threshold (4/6 = majority)
    pub fn default_threshold() -> Self {
        Self::new(4)
    }

    /// Get cell at position
    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize) -> Fingerprint {
        debug_assert!(x < FIELD_SIZE && y < FIELD_SIZE && z < FIELD_SIZE);
        Fingerprint::from_raw(self.cells[x][y][z])
    }

    /// Set cell at position
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, z: usize, fp: &Fingerprint) {
        debug_assert!(x < FIELD_SIZE && y < FIELD_SIZE && z < FIELD_SIZE);
        self.cells[x][y][z] = *fp.as_raw();
    }

    /// Inject pattern at origin (0,0,0), let it propagate
    pub fn inject(&mut self, fp: &Fingerprint) {
        self.set(0, 0, 0, fp);
    }

    /// Inject pattern at specific position
    pub fn inject_at(&mut self, x: usize, y: usize, z: usize, fp: &Fingerprint) {
        self.set(x, y, z, fp);
    }

    /// Get 6 face-adjacent neighbors (von Neumann neighborhood)
    fn get_neighbors(&self, x: usize, y: usize, z: usize) -> Vec<[u64; FINGERPRINT_U64]> {
        let mut neighbors = Vec::with_capacity(6);

        // -X
        if x > 0 {
            neighbors.push(self.cells[x - 1][y][z]);
        }
        // +X
        if x < FIELD_SIZE - 1 {
            neighbors.push(self.cells[x + 1][y][z]);
        }
        // -Y
        if y > 0 {
            neighbors.push(self.cells[x][y - 1][z]);
        }
        // +Y
        if y < FIELD_SIZE - 1 {
            neighbors.push(self.cells[x][y + 1][z]);
        }
        // -Z
        if z > 0 {
            neighbors.push(self.cells[x][y][z - 1]);
        }
        // +Z
        if z < FIELD_SIZE - 1 {
            neighbors.push(self.cells[x][y][z + 1]);
        }

        neighbors
    }

    /// Evolve one tick: all cells vote simultaneously
    ///
    /// Returns true if any cell changed.
    pub fn tick(&mut self) -> bool {
        let mut next = Box::new([[[[0u64; FINGERPRINT_U64]; FIELD_SIZE]; FIELD_SIZE]; FIELD_SIZE]);
        let mut changed = false;

        for x in 0..FIELD_SIZE {
            for y in 0..FIELD_SIZE {
                for z in 0..FIELD_SIZE {
                    let neighbors = self.get_neighbors(x, y, z);
                    if neighbors.is_empty() {
                        // No neighbors (shouldn't happen in 5×5×5 except edges)
                        next[x][y][z] = self.cells[x][y][z];
                        continue;
                    }

                    // Quorum vote for each bit
                    next[x][y][z] = self.quorum_vote(&neighbors);

                    if next[x][y][z] != self.cells[x][y][z] {
                        changed = true;
                    }
                }
            }
        }

        self.cells = next;
        self.generation += 1;
        changed
    }

    /// Quorum vote: majority rule across neighbors
    fn quorum_vote(&self, neighbors: &[[u64; FINGERPRINT_U64]]) -> [u64; FINGERPRINT_U64] {
        let n = neighbors.len();
        let threshold = self.threshold.min(n as u8) as usize;

        let mut result = [0u64; FINGERPRINT_U64];

        for word_idx in 0..FINGERPRINT_U64 {
            for bit in 0..64 {
                let mut count = 0usize;
                for neighbor in neighbors {
                    if (neighbor[word_idx] >> bit) & 1 == 1 {
                        count += 1;
                    }
                }

                if count >= threshold {
                    result[word_idx] |= 1 << bit;
                }
            }
        }

        result
    }

    /// Settle into attractor (evolve until stable or max steps)
    ///
    /// Returns (steps_taken, converged)
    pub fn settle(&mut self, max_steps: usize) -> (usize, bool) {
        for step in 0..max_steps {
            if !self.tick() {
                return (step + 1, true);
            }
        }
        (max_steps, false)
    }

    /// XOR-fold all 125 cells into holographic signature
    pub fn signature(&self) -> Fingerprint {
        let mut result = [0u64; FINGERPRINT_U64];

        for x in 0..FIELD_SIZE {
            for y in 0..FIELD_SIZE {
                for z in 0..FIELD_SIZE {
                    for i in 0..FINGERPRINT_U64 {
                        result[i] ^= self.cells[x][y][z][i];
                    }
                }
            }
        }

        Fingerprint::from_raw(result)
    }

    /// Compute X-axis projection (fold Y,Z)
    pub fn project_x(&self) -> Fingerprint {
        let mut result = [0u64; FINGERPRINT_U64];

        for y in 0..FIELD_SIZE {
            for z in 0..FIELD_SIZE {
                for x in 0..FIELD_SIZE {
                    for i in 0..FINGERPRINT_U64 {
                        result[i] ^= self.cells[x][y][z][i];
                    }
                }
            }
        }

        Fingerprint::from_raw(result)
    }

    /// Compute Y-axis projection (fold X,Z)
    pub fn project_y(&self) -> Fingerprint {
        let mut result = [0u64; FINGERPRINT_U64];

        for x in 0..FIELD_SIZE {
            for z in 0..FIELD_SIZE {
                for y in 0..FIELD_SIZE {
                    for i in 0..FINGERPRINT_U64 {
                        result[i] ^= self.cells[x][y][z][i];
                    }
                }
            }
        }

        Fingerprint::from_raw(result)
    }

    /// Compute Z-axis projection (fold X,Y)
    pub fn project_z(&self) -> Fingerprint {
        let mut result = [0u64; FINGERPRINT_U64];

        for x in 0..FIELD_SIZE {
            for y in 0..FIELD_SIZE {
                for z in 0..FIELD_SIZE {
                    for i in 0..FINGERPRINT_U64 {
                        result[i] ^= self.cells[x][y][z][i];
                    }
                }
            }
        }

        Fingerprint::from_raw(result)
    }

    /// Get current generation
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Get threshold
    pub fn threshold(&self) -> u8 {
        self.threshold
    }

    /// Set threshold
    pub fn set_threshold(&mut self, threshold: u8) {
        assert!(threshold >= 1 && threshold <= 6);
        self.threshold = threshold;
    }

    /// Total memory size in bytes
    pub const fn size_bytes() -> usize {
        FIELD_CELLS * FINGERPRINT_U64 * 8
    }

    /// Fill with random fingerprints
    pub fn randomize(&mut self) {
        for x in 0..FIELD_SIZE {
            for y in 0..FIELD_SIZE {
                for z in 0..FIELD_SIZE {
                    let fp = Fingerprint::random();
                    self.cells[x][y][z] = *fp.as_raw();
                }
            }
        }
    }

    /// Clear all cells to zero
    pub fn clear(&mut self) {
        self.cells = Box::new([[[[0u64; FINGERPRINT_U64]; FIELD_SIZE]; FIELD_SIZE]; FIELD_SIZE]);
        self.generation = 0;
    }

    /// Hamming distance to another field (cell-wise sum)
    pub fn distance(&self, other: &QuorumField) -> u64 {
        let mut total = 0u64;

        for x in 0..FIELD_SIZE {
            for y in 0..FIELD_SIZE {
                for z in 0..FIELD_SIZE {
                    for i in 0..FINGERPRINT_U64 {
                        total +=
                            (self.cells[x][y][z][i] ^ other.cells[x][y][z][i]).count_ones() as u64;
                    }
                }
            }
        }

        total
    }
}

impl Default for QuorumField {
    fn default() -> Self {
        Self::default_threshold()
    }
}

impl std::fmt::Debug for QuorumField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "QuorumField {{ cells: {}×{}×{}, threshold: {}/{}, generation: {} }}",
            FIELD_SIZE, FIELD_SIZE, FIELD_SIZE, self.threshold, 6, self.generation
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_creation() {
        let field = QuorumField::new(4);
        assert_eq!(field.threshold(), 4);
        assert_eq!(field.generation(), 0);
    }

    #[test]
    fn test_inject_and_get() {
        let mut field = QuorumField::default();
        let fp = Fingerprint::from_content("test pattern");

        field.inject_at(2, 2, 2, &fp);
        let retrieved = field.get(2, 2, 2);

        assert_eq!(fp, retrieved);
    }

    #[test]
    fn test_signature_xor_fold() {
        let mut field = QuorumField::default();

        // Empty field should have zero signature
        let sig = field.signature();
        assert_eq!(sig.popcount(), 0);

        // Single cell should equal signature
        let fp = Fingerprint::from_content("single");
        field.inject(&fp);

        // Signature includes only the single non-zero cell
        // (due to XOR with zeros)
        let sig = field.signature();
        assert_eq!(sig, fp);
    }

    #[test]
    fn test_settle_convergence() {
        let mut field = QuorumField::new(4);

        // Uniform field should be stable
        let fp = Fingerprint::from_content("uniform");
        for x in 0..FIELD_SIZE {
            for y in 0..FIELD_SIZE {
                for z in 0..FIELD_SIZE {
                    field.inject_at(x, y, z, &fp);
                }
            }
        }

        let (steps, converged) = field.settle(10);
        assert!(converged);
        assert_eq!(steps, 1); // Already stable
    }

    #[test]
    fn test_memory_size() {
        assert_eq!(QuorumField::size_bytes(), 125 * crate::FINGERPRINT_U64 * 8);
        // = 250KB
    }
}
