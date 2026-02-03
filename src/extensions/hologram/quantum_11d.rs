//! 11D Quantum Crystal
//!
//! An 11-dimensional hypercube of sparse fingerprints for quantum-like computation.
//!
//! # Why 11D?
//!
//! ```text
//! 7D Crystal (5^7):
//!   - 78,125 cells
//!   - 2,186 neighbors per cell (3^7 - 1)
//!   - 21 interference directions
//!
//! 11D Crystal (3^11):
//!   - 177,147 cells
//!   - 177,146 neighbors per cell (3^11 - 1)
//!   - 55 interference directions
//!   - Extreme entanglement density
//! ```
//!
//! 11D is at the edge of practical computation. Even with prime=3,
//! we get 177,147 cells. With prime=5, we'd have 48,828,125 cells.

use std::collections::HashMap;

use crate::core::Fingerprint;

// =============================================================================
// PHASE TAG (128-bit quantum phase for 11D)
// =============================================================================

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct PhaseTag11D {
    bits: [u64; 2],
}

impl PhaseTag11D {
    pub fn zero() -> Self {
        Self { bits: [0, 0] }
    }

    pub fn pi() -> Self {
        Self { bits: [u64::MAX, u64::MAX] }
    }

    pub fn from_seed(seed: u64) -> Self {
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15);
        let mut bits = [0u64; 2];
        for word in &mut bits {
            state = state.wrapping_mul(0x5851F42D4C957F2D).wrapping_add(1);
            *word = state;
        }
        Self { bits }
    }

    pub fn from_angle(angle: f32) -> Self {
        let num_bits = ((angle.clamp(0.0, 1.0) * 128.0) as u32).min(128);
        let mut bits = [0u64; 2];
        for i in 0..num_bits as usize {
            bits[i / 64] |= 1 << (i % 64);
        }
        Self { bits }
    }

    pub fn hamming(&self, other: &PhaseTag11D) -> u32 {
        (self.bits[0] ^ other.bits[0]).count_ones()
            + (self.bits[1] ^ other.bits[1]).count_ones()
    }

    pub fn cos_angle_to(&self, other: &PhaseTag11D) -> f32 {
        let h = self.hamming(other) as f32;
        1.0 - 2.0 * h / 128.0
    }

    pub fn combine(&self, other: &PhaseTag11D) -> PhaseTag11D {
        PhaseTag11D {
            bits: [self.bits[0] ^ other.bits[0], self.bits[1] ^ other.bits[1]],
        }
    }

    pub fn negate(&self) -> PhaseTag11D {
        PhaseTag11D {
            bits: [!self.bits[0], !self.bits[1]],
        }
    }
}

// =============================================================================
// QUANTUM CELL
// =============================================================================

#[derive(Clone)]
pub struct QuantumCell11D {
    pub amplitude: Fingerprint,
    pub phase: PhaseTag11D,
}

impl QuantumCell11D {
    pub fn new(amplitude: Fingerprint, phase: PhaseTag11D) -> Self {
        Self { amplitude, phase }
    }

    pub fn from_fingerprint(fp: Fingerprint) -> Self {
        Self {
            amplitude: fp,
            phase: PhaseTag11D::zero(),
        }
    }

    pub fn interference_to(&self, other: &QuantumCell11D) -> f32 {
        let similarity = self.amplitude.similarity(&other.amplitude);
        let phase_cos = self.phase.cos_angle_to(&other.phase);
        similarity * phase_cos
    }

    pub fn probability(&self) -> f32 {
        self.amplitude.popcount() as f32 / crate::FINGERPRINT_BITS as f32
    }
}

// =============================================================================
// 11D COORDINATE
// =============================================================================

/// 11D coordinate using array for compactness
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Coord11D {
    pub axes: [usize; 11],
}

impl Coord11D {
    pub fn new(axes: [usize; 11]) -> Self {
        Self { axes }
    }

    pub fn from_components(
        x0: usize, x1: usize, x2: usize, x3: usize, x4: usize,
        x5: usize, x6: usize, x7: usize, x8: usize, x9: usize, x10: usize
    ) -> Self {
        Self { axes: [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10] }
    }

    pub fn to_index(&self, size: usize) -> usize {
        let mut idx = 0;
        let mut multiplier = 1;
        for i in (0..11).rev() {
            idx += self.axes[i] * multiplier;
            multiplier *= size;
        }
        idx
    }

    pub fn from_index(mut idx: usize, size: usize) -> Self {
        let mut axes = [0usize; 11];
        for i in (0..11).rev() {
            axes[i] = idx % size;
            idx /= size;
        }
        Self { axes }
    }

    pub fn manhattan(&self, other: &Coord11D) -> usize {
        self.axes.iter()
            .zip(other.axes.iter())
            .map(|(a, b)| (*a as isize - *b as isize).unsigned_abs())
            .sum()
    }

    pub fn chebyshev(&self, other: &Coord11D) -> usize {
        self.axes.iter()
            .zip(other.axes.iter())
            .map(|(a, b)| (*a as isize - *b as isize).unsigned_abs())
            .max()
            .unwrap_or(0)
    }

    /// Number of neighbors in 11D (3^11 - 1 = 177,146)
    pub const NEIGHBORS: usize = 177_146;
}

// =============================================================================
// 11D QUANTUM CRYSTAL
// =============================================================================

pub struct Crystal11D {
    size: usize,
    cells: HashMap<usize, QuantumCell11D>,
    quantum_threshold: f32,
    stats: Crystal11DStats,
}

#[derive(Debug, Clone, Default)]
pub struct Crystal11DStats {
    pub total_cells: usize,
    pub active_cells: usize,
    pub total_bits_set: u64,
    pub interference_events: u64,
}

impl Crystal11D {
    pub fn new(size: usize) -> Self {
        let total_cells = size.pow(11);
        Self {
            size,
            cells: HashMap::new(),
            quantum_threshold: std::f32::consts::FRAC_PI_4,
            stats: Crystal11DStats {
                total_cells,
                ..Default::default()
            },
        }
    }

    /// 3^11 = 177,147 cells (practical maximum for 11D)
    pub fn prime_3_11d() -> Self {
        Self::new(3)
    }

    /// 2^11 = 2,048 cells (tiny, for testing)
    pub fn binary_11d() -> Self {
        Self::new(2)
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.quantum_threshold = threshold;
        self
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn total_cells(&self) -> usize {
        self.stats.total_cells
    }

    pub fn active_cells(&self) -> usize {
        self.cells.len()
    }

    pub fn get(&self, coord: &Coord11D) -> Option<&Fingerprint> {
        let idx = coord.to_index(self.size);
        self.cells.get(&idx).map(|c| &c.amplitude)
    }

    pub fn get_quantum(&self, coord: &Coord11D) -> Option<&QuantumCell11D> {
        let idx = coord.to_index(self.size);
        self.cells.get(&idx)
    }

    pub fn set(&mut self, coord: &Coord11D, fp: Fingerprint) {
        self.set_quantum(coord, QuantumCell11D::from_fingerprint(fp));
    }

    pub fn set_quantum(&mut self, coord: &Coord11D, cell: QuantumCell11D) {
        let idx = coord.to_index(self.size);
        self.stats.total_bits_set += cell.amplitude.popcount() as u64;
        self.cells.insert(idx, cell);
        self.stats.active_cells = self.cells.len();
    }

    pub fn clear(&mut self, coord: &Coord11D) -> Option<QuantumCell11D> {
        let idx = coord.to_index(self.size);
        let removed = self.cells.remove(&idx);
        self.stats.active_cells = self.cells.len();
        removed
    }

    /// Populate with entangled pairs
    pub fn populate_entangled(&mut self, density: f32) {
        let target = (self.stats.total_cells as f32 * density) as usize;
        let mut rng_state = 0xDEADBEEFu64;

        for _ in 0..target {
            rng_state = rng_state.wrapping_mul(0x5DEECE66D).wrapping_add(0xB);
            let idx = (rng_state as usize) % self.stats.total_cells;
            let coord = Coord11D::from_index(idx, self.size);

            let mut fp = Fingerprint::zero();
            fp.set_bit((rng_state >> 17) as usize % crate::FINGERPRINT_BITS, true);
            fp.set_bit((rng_state >> 33) as usize % crate::FINGERPRINT_BITS, true);

            let phase = PhaseTag11D::from_seed(rng_state);
            self.set_quantum(&coord, QuantumCell11D::new(fp, phase));
        }
    }

    /// Bell test (sampled, since full neighbor scan is too expensive)
    pub fn bell_test(&self, samples: usize) -> BellTestResult11D {
        if self.cells.is_empty() {
            return BellTestResult11D::empty();
        }

        let indices: Vec<_> = self.cells.keys().copied().collect();
        let n_pairs = indices.len().min(samples);

        let mut e_ab = 0.0f32;
        let mut e_ab_prime = 0.0f32;
        let mut e_a_prime_b = 0.0f32;
        let mut e_a_prime_b_prime = 0.0f32;

        for i in 0..n_pairs {
            let j = (i + 1) % indices.len();

            let cell_a = self.cells.get(&indices[i]).unwrap();
            let cell_b = self.cells.get(&indices[j]).unwrap();

            let a = &cell_a.amplitude;
            let a_prime = cell_a.amplitude.permute(11);
            let b = &cell_b.amplitude;
            let b_prime = cell_b.amplitude.permute(13);

            let corr = |x: &Fingerprint, y: &Fingerprint| -> f32 {
                2.0 * x.similarity(y) - 1.0
            };

            e_ab += corr(a, b);
            e_ab_prime += corr(a, &b_prime);
            e_a_prime_b += corr(&a_prime, b);
            e_a_prime_b_prime += corr(&a_prime, &b_prime);
        }

        let n = n_pairs.max(1) as f32;
        e_ab /= n;
        e_ab_prime /= n;
        e_a_prime_b /= n;
        e_a_prime_b_prime /= n;

        let s = e_ab - e_ab_prime + e_a_prime_b + e_a_prime_b_prime;

        BellTestResult11D {
            s_value: s,
            is_quantum: s.abs() > 2.0,
            samples: n_pairs,
        }
    }
}

// =============================================================================
// BELL TEST RESULT
// =============================================================================

#[derive(Debug, Clone)]
pub struct BellTestResult11D {
    pub s_value: f32,
    pub is_quantum: bool,
    pub samples: usize,
}

impl BellTestResult11D {
    pub fn empty() -> Self {
        Self {
            s_value: 0.0,
            is_quantum: false,
            samples: 0,
        }
    }
}

// =============================================================================
// PRIME SWEET SPOTS FOR 11D
// =============================================================================

pub mod prime_sweet_spots {
    #[derive(Debug, Clone, Copy)]
    pub struct PrimeDimension11D {
        pub prime: usize,
        pub cells: u64,
        pub neighbors: usize,
        pub estimated_us: u64,
        pub fits_17min: bool,
        pub note: &'static str,
    }

    impl PrimeDimension11D {
        pub const fn new(prime: usize, note: &'static str) -> Self {
            let cells = const_pow(prime as u64, 11);
            let neighbors = const_pow(3, 11) as usize - 1; // 177,146
            let estimated_us = (prime as u64) * 11 * 10 * 100 / 1000 + cells / 10;
            let fits_17min = estimated_us < 17 * 60 * 1_000_000;

            Self {
                prime,
                cells,
                neighbors,
                estimated_us,
                fits_17min,
                note,
            }
        }
    }

    const fn const_pow(base: u64, exp: u32) -> u64 {
        let mut result = 1u64;
        let mut i = 0;
        while i < exp {
            result *= base;
            i += 1;
        }
        result
    }

    /// 2^11 = 2,048 cells (testing only)
    pub const BINARY: PrimeDimension11D = PrimeDimension11D::new(2, "testing");

    /// 3^11 = 177,147 cells (practical maximum)
    pub const PRIME_3: PrimeDimension11D = PrimeDimension11D::new(3, "practical max");

    /// 5^11 = 48,828,125 cells (theoretical, exceeds memory)
    pub const PRIME_5: PrimeDimension11D = PrimeDimension11D::new(5, "theoretical");

    /// All 11D configurations
    pub const ALL: [PrimeDimension11D; 3] = [BINARY, PRIME_3, PRIME_5];

    /// Practical configurations that fit in 17 minutes
    pub fn practical() -> Vec<PrimeDimension11D> {
        ALL.iter().filter(|p| p.fits_17min).copied().collect()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coord11d_roundtrip() {
        let size = 3;
        let coord = Coord11D::from_components(0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1);
        let idx = coord.to_index(size);
        let back = Coord11D::from_index(idx, size);
        assert_eq!(coord, back);
    }

    #[test]
    fn test_crystal11d_creation() {
        let crystal = Crystal11D::binary_11d();
        assert_eq!(crystal.size(), 2);
        assert_eq!(crystal.total_cells(), 2048);
    }

    #[test]
    fn test_prime_sweet_spots_11d() {
        use prime_sweet_spots::*;

        assert_eq!(BINARY.cells, 2048);
        assert_eq!(PRIME_3.cells, 177147);
        assert!(PRIME_3.fits_17min);
    }

    #[test]
    fn test_11d_bell_test() {
        let mut crystal = Crystal11D::binary_11d();
        crystal.populate_entangled(0.1);

        let result = crystal.bell_test(50);
        println!("11D Bell test: S = {:.3}, quantum = {}", result.s_value, result.is_quantum);
    }
}
