//! 13D Quantum Crystal
//!
//! A 13-dimensional hypercube - deep into theoretical territory.
//!
//! # Scale
//!
//! ```text
//! 13D Crystal characteristics:
//!   - Neighbors per cell: 3^13 - 1 = 1,594,322
//!   - 78 interference directions (13 choose 2)
//!   - 2^13 = 8,192 cells (binary, minimal practical)
//!   - 3^13 = 1,594,323 cells (exceeds practical limits)
//! ```
//!
//! 13D is primarily for theoretical exploration and verification
//! of quantum-like behavior at extreme dimensionality.

use std::collections::HashMap;

use crate::core::Fingerprint;

// =============================================================================
// PHASE TAG (128-bit quantum phase for 13D)
// =============================================================================

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct PhaseTag13D {
    bits: [u64; 2],
}

impl PhaseTag13D {
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

    pub fn hamming(&self, other: &PhaseTag13D) -> u32 {
        (self.bits[0] ^ other.bits[0]).count_ones()
            + (self.bits[1] ^ other.bits[1]).count_ones()
    }

    pub fn cos_angle_to(&self, other: &PhaseTag13D) -> f32 {
        let h = self.hamming(other) as f32;
        1.0 - 2.0 * h / 128.0
    }

    pub fn combine(&self, other: &PhaseTag13D) -> PhaseTag13D {
        PhaseTag13D {
            bits: [self.bits[0] ^ other.bits[0], self.bits[1] ^ other.bits[1]],
        }
    }

    pub fn negate(&self) -> PhaseTag13D {
        PhaseTag13D {
            bits: [!self.bits[0], !self.bits[1]],
        }
    }
}

// =============================================================================
// QUANTUM CELL
// =============================================================================

#[derive(Clone)]
pub struct QuantumCell13D {
    pub amplitude: Fingerprint,
    pub phase: PhaseTag13D,
}

impl QuantumCell13D {
    pub fn new(amplitude: Fingerprint, phase: PhaseTag13D) -> Self {
        Self { amplitude, phase }
    }

    pub fn from_fingerprint(fp: Fingerprint) -> Self {
        Self {
            amplitude: fp,
            phase: PhaseTag13D::zero(),
        }
    }

    pub fn interference_to(&self, other: &QuantumCell13D) -> f32 {
        let similarity = self.amplitude.similarity(&other.amplitude);
        let phase_cos = self.phase.cos_angle_to(&other.phase);
        similarity * phase_cos
    }

    pub fn probability(&self) -> f32 {
        self.amplitude.popcount() as f32 / crate::FINGERPRINT_BITS as f32
    }
}

// =============================================================================
// 13D COORDINATE
// =============================================================================

/// 13D coordinate using array
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Coord13D {
    pub axes: [usize; 13],
}

impl Coord13D {
    pub fn new(axes: [usize; 13]) -> Self {
        Self { axes }
    }

    pub fn origin() -> Self {
        Self { axes: [0; 13] }
    }

    pub fn to_index(&self, size: usize) -> usize {
        let mut idx = 0;
        let mut multiplier = 1;
        for i in (0..13).rev() {
            idx += self.axes[i] * multiplier;
            multiplier *= size;
        }
        idx
    }

    pub fn from_index(mut idx: usize, size: usize) -> Self {
        let mut axes = [0usize; 13];
        for i in (0..13).rev() {
            axes[i] = idx % size;
            idx /= size;
        }
        Self { axes }
    }

    pub fn manhattan(&self, other: &Coord13D) -> usize {
        self.axes.iter()
            .zip(other.axes.iter())
            .map(|(a, b)| (*a as isize - *b as isize).unsigned_abs())
            .sum()
    }

    pub fn chebyshev(&self, other: &Coord13D) -> usize {
        self.axes.iter()
            .zip(other.axes.iter())
            .map(|(a, b)| (*a as isize - *b as isize).unsigned_abs())
            .max()
            .unwrap_or(0)
    }

    /// Number of neighbors in 13D (3^13 - 1 = 1,594,322)
    pub const NEIGHBORS: usize = 1_594_322;

    /// Number of interference directions (13 choose 2 = 78)
    pub const INTERFERENCE_DIRECTIONS: usize = 78;
}

// =============================================================================
// 13D QUANTUM CRYSTAL
// =============================================================================

pub struct Crystal13D {
    size: usize,
    cells: HashMap<usize, QuantumCell13D>,
    quantum_threshold: f32,
    stats: Crystal13DStats,
}

#[derive(Debug, Clone, Default)]
pub struct Crystal13DStats {
    pub total_cells: usize,
    pub active_cells: usize,
    pub total_bits_set: u64,
}

impl Crystal13D {
    pub fn new(size: usize) -> Self {
        // For 13D, we need to be careful about overflow
        let total_cells = (size as u64).pow(13) as usize;
        Self {
            size,
            cells: HashMap::new(),
            quantum_threshold: std::f32::consts::FRAC_PI_4,
            stats: Crystal13DStats {
                total_cells,
                ..Default::default()
            },
        }
    }

    /// 2^13 = 8,192 cells (practical for 13D)
    pub fn binary_13d() -> Self {
        Self::new(2)
    }

    /// 3^13 = 1,594,323 cells (theoretical maximum)
    pub fn prime_3_13d() -> Self {
        Self::new(3)
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

    pub fn get(&self, coord: &Coord13D) -> Option<&Fingerprint> {
        let idx = coord.to_index(self.size);
        self.cells.get(&idx).map(|c| &c.amplitude)
    }

    pub fn get_quantum(&self, coord: &Coord13D) -> Option<&QuantumCell13D> {
        let idx = coord.to_index(self.size);
        self.cells.get(&idx)
    }

    pub fn set(&mut self, coord: &Coord13D, fp: Fingerprint) {
        self.set_quantum(coord, QuantumCell13D::from_fingerprint(fp));
    }

    pub fn set_quantum(&mut self, coord: &Coord13D, cell: QuantumCell13D) {
        let idx = coord.to_index(self.size);
        self.stats.total_bits_set += cell.amplitude.popcount() as u64;
        self.cells.insert(idx, cell);
        self.stats.active_cells = self.cells.len();
    }

    pub fn clear(&mut self, coord: &Coord13D) -> Option<QuantumCell13D> {
        let idx = coord.to_index(self.size);
        let removed = self.cells.remove(&idx);
        self.stats.active_cells = self.cells.len();
        removed
    }

    /// Populate with entangled pairs
    pub fn populate_entangled(&mut self, density: f32) {
        let target = (self.stats.total_cells as f32 * density) as usize;
        let mut rng_state = 0xCAFEBABEu64;

        for _ in 0..target {
            rng_state = rng_state.wrapping_mul(0x5DEECE66D).wrapping_add(0xB);
            let idx = (rng_state as usize) % self.stats.total_cells;
            let coord = Coord13D::from_index(idx, self.size);

            let mut fp = Fingerprint::zero();
            fp.set_bit((rng_state >> 17) as usize % crate::FINGERPRINT_BITS, true);
            fp.set_bit((rng_state >> 33) as usize % crate::FINGERPRINT_BITS, true);

            let phase = PhaseTag13D::from_seed(rng_state);
            self.set_quantum(&coord, QuantumCell13D::new(fp, phase));
        }
    }

    /// Bell test (heavily sampled due to extreme neighbor count)
    pub fn bell_test(&self, samples: usize) -> BellTestResult13D {
        if self.cells.is_empty() {
            return BellTestResult13D::empty();
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
            let a_prime = cell_a.amplitude.permute(13);
            let b = &cell_b.amplitude;
            let b_prime = cell_b.amplitude.permute(17);

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

        BellTestResult13D {
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
pub struct BellTestResult13D {
    pub s_value: f32,
    pub is_quantum: bool,
    pub samples: usize,
}

impl BellTestResult13D {
    pub fn empty() -> Self {
        Self {
            s_value: 0.0,
            is_quantum: false,
            samples: 0,
        }
    }
}

// =============================================================================
// PRIME SWEET SPOTS FOR 13D
// =============================================================================

pub mod prime_sweet_spots {
    #[derive(Debug, Clone, Copy)]
    pub struct PrimeDimension13D {
        pub prime: usize,
        pub cells: u64,
        pub neighbors: usize,
        pub fits_17min: bool,
        pub note: &'static str,
    }

    impl PrimeDimension13D {
        pub const fn new(prime: usize, note: &'static str) -> Self {
            let cells = const_pow(prime as u64, 13);
            let neighbors = const_pow(3, 13) as usize - 1; // 1,594,322
            // At 13D, even small sizes explode
            let fits_17min = cells < 100_000;

            Self {
                prime,
                cells,
                neighbors,
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

    /// 2^13 = 8,192 cells (only practical option)
    pub const BINARY: PrimeDimension13D = PrimeDimension13D::new(2, "practical");

    /// 3^13 = 1,594,323 cells (theoretical)
    pub const PRIME_3: PrimeDimension13D = PrimeDimension13D::new(3, "theoretical");

    /// 5^13 = 1,220,703,125 cells (impossible)
    pub const PRIME_5: PrimeDimension13D = PrimeDimension13D::new(5, "impossible");

    /// All 13D configurations
    pub const ALL: [PrimeDimension13D; 3] = [BINARY, PRIME_3, PRIME_5];

    pub fn print_info() -> String {
        let mut s = String::new();
        s.push_str("13D Quantum Crystal Configurations:\n");
        s.push_str("══════════════════════════════════════════════════\n");
        s.push_str("  Neighbors per cell: 1,594,322 (3^13 - 1)\n");
        s.push_str("  Interference directions: 78\n\n");
        for p in ALL.iter() {
            s.push_str(&format!(
                "  {}^13 = {} cells [{}]\n",
                p.prime, p.cells, p.note
            ));
        }
        s
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coord13d_roundtrip() {
        let size = 2;
        let coord = Coord13D::new([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]);
        let idx = coord.to_index(size);
        let back = Coord13D::from_index(idx, size);
        assert_eq!(coord, back);
    }

    #[test]
    fn test_crystal13d_creation() {
        let crystal = Crystal13D::binary_13d();
        assert_eq!(crystal.size(), 2);
        assert_eq!(crystal.total_cells(), 8192);
    }

    #[test]
    fn test_prime_sweet_spots_13d() {
        use prime_sweet_spots::*;

        assert_eq!(BINARY.cells, 8192);
        assert_eq!(PRIME_3.cells, 1594323);
        assert!(BINARY.fits_17min);
        assert!(!PRIME_3.fits_17min);
    }

    #[test]
    fn test_13d_bell_test() {
        let mut crystal = Crystal13D::binary_13d();
        crystal.populate_entangled(0.05);

        let result = crystal.bell_test(30);
        println!("13D Bell test: S = {:.3}, quantum = {}", result.s_value, result.is_quantum);
    }
}
