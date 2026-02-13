//! 17D Quantum Crystal
//!
//! A 17-dimensional hypercube - the theoretical frontier.
//!
//! # Scale
//!
//! ```text
//! 17D Crystal characteristics:
//!   - Neighbors per cell: 3^17 - 1 = 129,140,162
//!   - 136 interference directions (17 choose 2)
//!   - 2^17 = 131,072 cells (minimal practical)
//!   - 3^17 = 129,140,163 cells (matches neighbor count!)
//! ```
//!
//! # The 17-Minute Connection
//!
//! Within a 17-minute compute budget:
//! - 2^17 Bell test: ~1-5 minutes (practical)
//! - The 17 dimensions × 17 minutes creates a natural resonance
//!
//! 17D is the edge of what can be meaningfully computed,
//! chosen because 17 is prime and creates interesting
//! symmetries with the time budget.

use std::collections::HashMap;

use crate::core::Fingerprint;

// =============================================================================
// PHASE TAG (128-bit quantum phase for 17D)
// =============================================================================

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct PhaseTag17D {
    bits: [u64; 2],
}

impl PhaseTag17D {
    pub fn zero() -> Self {
        Self { bits: [0, 0] }
    }

    pub fn pi() -> Self {
        Self {
            bits: [u64::MAX, u64::MAX],
        }
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

    pub fn hamming(&self, other: &PhaseTag17D) -> u32 {
        (self.bits[0] ^ other.bits[0]).count_ones() + (self.bits[1] ^ other.bits[1]).count_ones()
    }

    pub fn cos_angle_to(&self, other: &PhaseTag17D) -> f32 {
        let h = self.hamming(other) as f32;
        1.0 - 2.0 * h / 128.0
    }

    pub fn combine(&self, other: &PhaseTag17D) -> PhaseTag17D {
        PhaseTag17D {
            bits: [self.bits[0] ^ other.bits[0], self.bits[1] ^ other.bits[1]],
        }
    }

    pub fn negate(&self) -> PhaseTag17D {
        PhaseTag17D {
            bits: [!self.bits[0], !self.bits[1]],
        }
    }
}

// =============================================================================
// QUANTUM CELL
// =============================================================================

#[derive(Clone)]
pub struct QuantumCell17D {
    pub amplitude: Fingerprint,
    pub phase: PhaseTag17D,
}

impl QuantumCell17D {
    pub fn new(amplitude: Fingerprint, phase: PhaseTag17D) -> Self {
        Self { amplitude, phase }
    }

    pub fn from_fingerprint(fp: Fingerprint) -> Self {
        Self {
            amplitude: fp,
            phase: PhaseTag17D::zero(),
        }
    }

    pub fn interference_to(&self, other: &QuantumCell17D) -> f32 {
        let similarity = self.amplitude.similarity(&other.amplitude);
        let phase_cos = self.phase.cos_angle_to(&other.phase);
        similarity * phase_cos
    }

    pub fn probability(&self) -> f32 {
        self.amplitude.popcount() as f32 / crate::FINGERPRINT_BITS as f32
    }
}

// =============================================================================
// 17D COORDINATE
// =============================================================================

/// 17D coordinate using array
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Coord17D {
    pub axes: [usize; 17],
}

impl Coord17D {
    pub fn new(axes: [usize; 17]) -> Self {
        Self { axes }
    }

    pub fn origin() -> Self {
        Self { axes: [0; 17] }
    }

    pub fn to_index(&self, size: usize) -> usize {
        let mut idx = 0usize;
        let mut multiplier = 1usize;
        for i in (0..17).rev() {
            idx = idx.wrapping_add(self.axes[i].wrapping_mul(multiplier));
            multiplier = multiplier.wrapping_mul(size);
        }
        idx
    }

    pub fn from_index(mut idx: usize, size: usize) -> Self {
        let mut axes = [0usize; 17];
        for i in (0..17).rev() {
            axes[i] = idx % size;
            idx /= size;
        }
        Self { axes }
    }

    pub fn manhattan(&self, other: &Coord17D) -> usize {
        self.axes
            .iter()
            .zip(other.axes.iter())
            .map(|(a, b)| (*a as isize - *b as isize).unsigned_abs())
            .sum()
    }

    pub fn chebyshev(&self, other: &Coord17D) -> usize {
        self.axes
            .iter()
            .zip(other.axes.iter())
            .map(|(a, b)| (*a as isize - *b as isize).unsigned_abs())
            .max()
            .unwrap_or(0)
    }

    /// Number of neighbors in 17D (3^17 - 1 = 129,140,162)
    pub const NEIGHBORS: usize = 129_140_162;

    /// Number of interference directions (17 choose 2 = 136)
    pub const INTERFERENCE_DIRECTIONS: usize = 136;
}

// =============================================================================
// 17D QUANTUM CRYSTAL
// =============================================================================

pub struct Crystal17D {
    size: usize,
    cells: HashMap<usize, QuantumCell17D>,
    quantum_threshold: f32,
    stats: Crystal17DStats,
}

#[derive(Debug, Clone, Default)]
pub struct Crystal17DStats {
    pub total_cells: u64, // Use u64 for extreme sizes
    pub active_cells: usize,
    pub total_bits_set: u64,
}

impl Crystal17D {
    pub fn new(size: usize) -> Self {
        let total_cells = (size as u64).pow(17);
        Self {
            size,
            cells: HashMap::new(),
            quantum_threshold: std::f32::consts::FRAC_PI_4,
            stats: Crystal17DStats {
                total_cells,
                ..Default::default()
            },
        }
    }

    /// 2^17 = 131,072 cells (the practical 17D configuration)
    pub fn binary_17d() -> Self {
        Self::new(2)
    }

    /// Small test configuration
    pub fn tiny_17d() -> Self {
        // Even size=2 gives 131K cells, so we just use binary
        Self::new(2)
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.quantum_threshold = threshold;
        self
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn total_cells(&self) -> u64 {
        self.stats.total_cells
    }

    pub fn active_cells(&self) -> usize {
        self.cells.len()
    }

    pub fn get(&self, coord: &Coord17D) -> Option<&Fingerprint> {
        let idx = coord.to_index(self.size);
        self.cells.get(&idx).map(|c| &c.amplitude)
    }

    pub fn get_quantum(&self, coord: &Coord17D) -> Option<&QuantumCell17D> {
        let idx = coord.to_index(self.size);
        self.cells.get(&idx)
    }

    pub fn set(&mut self, coord: &Coord17D, fp: Fingerprint) {
        self.set_quantum(coord, QuantumCell17D::from_fingerprint(fp));
    }

    pub fn set_quantum(&mut self, coord: &Coord17D, cell: QuantumCell17D) {
        let idx = coord.to_index(self.size);
        self.stats.total_bits_set += cell.amplitude.popcount() as u64;
        self.cells.insert(idx, cell);
        self.stats.active_cells = self.cells.len();
    }

    pub fn clear(&mut self, coord: &Coord17D) -> Option<QuantumCell17D> {
        let idx = coord.to_index(self.size);
        let removed = self.cells.remove(&idx);
        self.stats.active_cells = self.cells.len();
        removed
    }

    /// Populate with entangled pairs
    pub fn populate_entangled(&mut self, density: f32) {
        let target = ((self.stats.total_cells as f64) * (density as f64)) as usize;
        let mut rng_state = 0x17171717u64; // 17-themed seed

        for _ in 0..target {
            rng_state = rng_state.wrapping_mul(0x5DEECE66D).wrapping_add(0xB);
            let idx = (rng_state as usize) % (self.stats.total_cells as usize);
            let coord = Coord17D::from_index(idx, self.size);

            let mut fp = Fingerprint::zero();
            fp.set_bit((rng_state >> 17) as usize % crate::FINGERPRINT_BITS, true);
            fp.set_bit((rng_state >> 33) as usize % crate::FINGERPRINT_BITS, true);

            let phase = PhaseTag17D::from_seed(rng_state);
            self.set_quantum(&coord, QuantumCell17D::new(fp, phase));
        }
    }

    /// Bell test (heavily sampled)
    pub fn bell_test(&self, samples: usize) -> BellTestResult17D {
        if self.cells.is_empty() {
            return BellTestResult17D::empty();
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
            let a_prime = cell_a.amplitude.permute(17);
            let b = &cell_b.amplitude;
            let b_prime = cell_b.amplitude.permute(19);

            let corr = |x: &Fingerprint, y: &Fingerprint| -> f32 { 2.0 * x.similarity(y) - 1.0 };

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

        BellTestResult17D {
            s_value: s,
            is_quantum: s.abs() > 2.0,
            samples: n_pairs,
            dimension_resonance: self.compute_dimension_resonance(),
        }
    }

    /// Compute 17-dimensional resonance metric
    fn compute_dimension_resonance(&self) -> f32 {
        // 17 dimensions, 17 minutes, 17 as prime
        // This is a theoretical metric exploring dimensional symmetry
        let active = self.active_cells() as f32;
        let total = self.stats.total_cells as f32;
        let density = active / total;

        // Resonance peaks when density matches 1/17
        let target_density = 1.0 / 17.0;
        let resonance = 1.0 - (density - target_density).abs() * 17.0;
        resonance.max(0.0)
    }
}

// =============================================================================
// BELL TEST RESULT
// =============================================================================

#[derive(Debug, Clone)]
pub struct BellTestResult17D {
    pub s_value: f32,
    pub is_quantum: bool,
    pub samples: usize,
    pub dimension_resonance: f32,
}

impl BellTestResult17D {
    pub fn empty() -> Self {
        Self {
            s_value: 0.0,
            is_quantum: false,
            samples: 0,
            dimension_resonance: 0.0,
        }
    }
}

// =============================================================================
// PRIME SWEET SPOTS FOR 17D
// =============================================================================

pub mod prime_sweet_spots {
    #[derive(Debug, Clone, Copy)]
    pub struct PrimeDimension17D {
        pub prime: usize,
        pub cells: u64,
        pub neighbors: usize,
        pub fits_17min: bool,
        pub note: &'static str,
    }

    impl PrimeDimension17D {
        pub const fn new(prime: usize, note: &'static str) -> Self {
            let cells = const_pow(prime as u64, 17);
            let neighbors = const_pow(3, 17) as usize - 1; // 129,140,162
            // Only binary fits in any reasonable time
            let fits_17min = prime == 2;

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

    /// 2^17 = 131,072 cells (only practical 17D)
    pub const BINARY: PrimeDimension17D = PrimeDimension17D::new(2, "practical");

    /// 3^17 = 129,140,163 cells (theoretical only)
    pub const PRIME_3: PrimeDimension17D = PrimeDimension17D::new(3, "theoretical");

    /// All 17D configurations
    pub const ALL: [PrimeDimension17D; 2] = [BINARY, PRIME_3];

    /// Time slots within 17 minutes for 17D
    pub fn time_slots_17min() -> Vec<(&'static str, &'static str)> {
        vec![
            ("0-5 min", "2^17 sparse (1% density)"),
            ("5-10 min", "2^17 medium (5% density)"),
            ("10-17 min", "2^17 dense (10% density)"),
        ]
    }

    /// The 17×17 resonance table
    pub fn resonance_table() -> String {
        let mut s = String::new();
        s.push_str("╔══════════════════════════════════════════════════════════╗\n");
        s.push_str("║           17D QUANTUM CRYSTAL - 17 MINUTE BUDGET         ║\n");
        s.push_str("╠══════════════════════════════════════════════════════════╣\n");
        s.push_str("║  Dimension: 17 (prime)                                   ║\n");
        s.push_str("║  Time budget: 17 minutes (prime)                         ║\n");
        s.push_str("║  Neighbors: 129,140,162 (3^17 - 1)                       ║\n");
        s.push_str("║  Interference directions: 136 (17 choose 2)              ║\n");
        s.push_str("╠══════════════════════════════════════════════════════════╣\n");
        s.push_str("║  Config     │    Cells    │  Est. Time  │  Practical    ║\n");
        s.push_str("╠═════════════╪═════════════╪═════════════╪═══════════════╣\n");
        s.push_str("║  2^17       │    131,072  │   1-5 min   │     YES       ║\n");
        s.push_str("║  3^17       │ 129,140,163 │   hours+    │     NO        ║\n");
        s.push_str("╚══════════════════════════════════════════════════════════╝\n");
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
    fn test_coord17d_roundtrip() {
        let size = 2;
        let mut axes = [0usize; 17];
        axes[0] = 1;
        axes[8] = 1;
        axes[16] = 1;
        let coord = Coord17D::new(axes);
        let idx = coord.to_index(size);
        let back = Coord17D::from_index(idx, size);
        assert_eq!(coord, back);
    }

    #[test]
    fn test_crystal17d_creation() {
        let crystal = Crystal17D::binary_17d();
        assert_eq!(crystal.size(), 2);
        assert_eq!(crystal.total_cells(), 131072);
    }

    #[test]
    fn test_prime_sweet_spots_17d() {
        use prime_sweet_spots::*;

        assert_eq!(BINARY.cells, 131072);
        assert_eq!(PRIME_3.cells, 129140163);
        assert!(BINARY.fits_17min);
        assert!(!PRIME_3.fits_17min);
    }

    #[test]
    fn test_17d_bell_test() {
        let mut crystal = Crystal17D::tiny_17d();
        crystal.populate_entangled(0.01);

        let result = crystal.bell_test(20);
        println!(
            "17D Bell test: S = {:.3}, quantum = {}, resonance = {:.3}",
            result.s_value, result.is_quantum, result.dimension_resonance
        );
    }

    #[test]
    fn test_resonance_table() {
        let table = prime_sweet_spots::resonance_table();
        println!("{}", table);
        assert!(table.contains("17D"));
    }
}
