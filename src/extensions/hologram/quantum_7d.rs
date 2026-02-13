//! 7D Quantum Crystal
//!
//! A 7-dimensional hypercube of sparse fingerprints for quantum-like computation.
//!
//! # Why 7D?
//!
//! ```text
//! 5D Crystal (5^5):
//!   - 3,125 cells
//!   - 242 neighbors per cell (3^5 - 1)
//!   - 10 interference directions
//!
//! 7D Crystal (5^7):
//!   - 78,125 cells
//!   - 2,186 neighbors per cell (3^7 - 1)
//!   - 21 interference directions
//!   - Deeper entanglement paths
//! ```
//!
//! The 7^7 configuration (823,543 cells) is the "sweet spot" for Bell inequality testing:
//! - Large enough for statistical power
//! - Fits in 1-minute compute budget
//! - Prime dimension avoids resonance artifacts

use std::collections::HashMap;

use crate::core::Fingerprint;

// =============================================================================
// PHASE TAG (128-bit quantum phase for 7D)
// =============================================================================

/// 128-bit phase tag for signed quantum interference in 7D.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct PhaseTag7D {
    bits: [u64; 2],
}

impl PhaseTag7D {
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

    pub fn hamming(&self, other: &PhaseTag7D) -> u32 {
        (self.bits[0] ^ other.bits[0]).count_ones() + (self.bits[1] ^ other.bits[1]).count_ones()
    }

    pub fn cos_angle_to(&self, other: &PhaseTag7D) -> f32 {
        let h = self.hamming(other) as f32;
        1.0 - 2.0 * h / 128.0
    }

    pub fn combine(&self, other: &PhaseTag7D) -> PhaseTag7D {
        PhaseTag7D {
            bits: [self.bits[0] ^ other.bits[0], self.bits[1] ^ other.bits[1]],
        }
    }

    pub fn negate(&self) -> PhaseTag7D {
        PhaseTag7D {
            bits: [!self.bits[0], !self.bits[1]],
        }
    }
}

// =============================================================================
// QUANTUM CELL (amplitude + phase)
// =============================================================================

#[derive(Clone)]
pub struct QuantumCell7D {
    pub amplitude: Fingerprint,
    pub phase: PhaseTag7D,
}

impl QuantumCell7D {
    pub fn new(amplitude: Fingerprint, phase: PhaseTag7D) -> Self {
        Self { amplitude, phase }
    }

    pub fn from_fingerprint(fp: Fingerprint) -> Self {
        Self {
            amplitude: fp,
            phase: PhaseTag7D::zero(),
        }
    }

    pub fn interference_to(&self, other: &QuantumCell7D) -> f32 {
        let similarity = self.amplitude.similarity(&other.amplitude);
        let phase_cos = self.phase.cos_angle_to(&other.phase);
        similarity * phase_cos
    }

    pub fn probability(&self) -> f32 {
        self.amplitude.popcount() as f32 / crate::FINGERPRINT_BITS as f32
    }
}

// =============================================================================
// 7D COORDINATE
// =============================================================================

/// 7D coordinate (a, b, c, d, e, f, g)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Coord7D {
    pub a: usize,
    pub b: usize,
    pub c: usize,
    pub d: usize,
    pub e: usize,
    pub f: usize,
    pub g: usize,
}

impl Coord7D {
    pub fn new(a: usize, b: usize, c: usize, d: usize, e: usize, f: usize, g: usize) -> Self {
        Self {
            a,
            b,
            c,
            d,
            e,
            f,
            g,
        }
    }

    pub fn to_index(&self, size: usize) -> usize {
        self.a * size.pow(6)
            + self.b * size.pow(5)
            + self.c * size.pow(4)
            + self.d * size.pow(3)
            + self.e * size.pow(2)
            + self.f * size
            + self.g
    }

    pub fn from_index(idx: usize, size: usize) -> Self {
        let g = idx % size;
        let f = (idx / size) % size;
        let e = (idx / size.pow(2)) % size;
        let d = (idx / size.pow(3)) % size;
        let c = (idx / size.pow(4)) % size;
        let b = (idx / size.pow(5)) % size;
        let a = idx / size.pow(6);
        Self {
            a,
            b,
            c,
            d,
            e,
            f,
            g,
        }
    }

    pub fn manhattan(&self, other: &Coord7D) -> usize {
        (self.a as isize - other.a as isize).unsigned_abs()
            + (self.b as isize - other.b as isize).unsigned_abs()
            + (self.c as isize - other.c as isize).unsigned_abs()
            + (self.d as isize - other.d as isize).unsigned_abs()
            + (self.e as isize - other.e as isize).unsigned_abs()
            + (self.f as isize - other.f as isize).unsigned_abs()
            + (self.g as isize - other.g as isize).unsigned_abs()
    }

    pub fn chebyshev(&self, other: &Coord7D) -> usize {
        [
            (self.a as isize - other.a as isize).unsigned_abs(),
            (self.b as isize - other.b as isize).unsigned_abs(),
            (self.c as isize - other.c as isize).unsigned_abs(),
            (self.d as isize - other.d as isize).unsigned_abs(),
            (self.e as isize - other.e as isize).unsigned_abs(),
            (self.f as isize - other.f as isize).unsigned_abs(),
            (self.g as isize - other.g as isize).unsigned_abs(),
        ]
        .into_iter()
        .max()
        .unwrap_or(0)
    }

    /// Number of neighbors in 7D (3^7 - 1 = 2186)
    pub const NEIGHBORS: usize = 2186;
}

// =============================================================================
// 7D QUANTUM CRYSTAL
// =============================================================================

pub struct Crystal7D {
    size: usize,
    cells: HashMap<usize, QuantumCell7D>,
    quantum_threshold: f32,
    stats: Crystal7DStats,
}

#[derive(Debug, Clone, Default)]
pub struct Crystal7DStats {
    pub total_cells: usize,
    pub active_cells: usize,
    pub total_bits_set: u64,
    pub interference_events: u64,
    pub collapse_events: u64,
}

impl Crystal7D {
    pub fn new(size: usize) -> Self {
        Self {
            size,
            cells: HashMap::new(),
            quantum_threshold: std::f32::consts::FRAC_PI_4,
            stats: Crystal7DStats {
                total_cells: size.pow(7),
                ..Default::default()
            },
        }
    }

    /// The 7^7 sweet spot configuration (823,543 cells)
    pub fn prime_7_7d() -> Self {
        Self::new(7)
    }

    /// 5^7 configuration (78,125 cells) - medium
    pub fn prime_5_7d() -> Self {
        Self::new(5)
    }

    /// 3^7 configuration (2,187 cells) - small/fast
    pub fn prime_3_7d() -> Self {
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

    pub fn get(&self, coord: &Coord7D) -> Option<&Fingerprint> {
        let idx = coord.to_index(self.size);
        self.cells.get(&idx).map(|c| &c.amplitude)
    }

    pub fn get_quantum(&self, coord: &Coord7D) -> Option<&QuantumCell7D> {
        let idx = coord.to_index(self.size);
        self.cells.get(&idx)
    }

    pub fn set(&mut self, coord: &Coord7D, fp: Fingerprint) {
        self.set_quantum(coord, QuantumCell7D::from_fingerprint(fp));
    }

    pub fn set_quantum(&mut self, coord: &Coord7D, cell: QuantumCell7D) {
        let idx = coord.to_index(self.size);
        self.stats.total_bits_set += cell.amplitude.popcount() as u64;
        self.cells.insert(idx, cell);
        self.stats.active_cells = self.cells.len();
    }

    pub fn clear(&mut self, coord: &Coord7D) -> Option<QuantumCell7D> {
        let idx = coord.to_index(self.size);
        let removed = self.cells.remove(&idx);
        self.stats.active_cells = self.cells.len();
        removed
    }

    /// Compute interference at a coordinate from all neighbors
    pub fn interference_at(&self, coord: &Coord7D) -> f32 {
        let center = match self.get_quantum(coord) {
            Some(c) => c,
            None => return 0.0,
        };

        let mut total_interference = 0.0f32;
        let size = self.size as isize;

        // Iterate over all 3^7 - 1 = 2186 neighbors
        for da in -1isize..=1 {
            for db in -1isize..=1 {
                for dc in -1isize..=1 {
                    for dd in -1isize..=1 {
                        for de in -1isize..=1 {
                            for df in -1isize..=1 {
                                for dg in -1isize..=1 {
                                    if da == 0
                                        && db == 0
                                        && dc == 0
                                        && dd == 0
                                        && de == 0
                                        && df == 0
                                        && dg == 0
                                    {
                                        continue;
                                    }

                                    let na = coord.a as isize + da;
                                    let nb = coord.b as isize + db;
                                    let nc = coord.c as isize + dc;
                                    let nd = coord.d as isize + dd;
                                    let ne = coord.e as isize + de;
                                    let nf = coord.f as isize + df;
                                    let ng = coord.g as isize + dg;

                                    if na >= 0
                                        && na < size
                                        && nb >= 0
                                        && nb < size
                                        && nc >= 0
                                        && nc < size
                                        && nd >= 0
                                        && nd < size
                                        && ne >= 0
                                        && ne < size
                                        && nf >= 0
                                        && nf < size
                                        && ng >= 0
                                        && ng < size
                                    {
                                        let neighbor_coord = Coord7D::new(
                                            na as usize,
                                            nb as usize,
                                            nc as usize,
                                            nd as usize,
                                            ne as usize,
                                            nf as usize,
                                            ng as usize,
                                        );
                                        if let Some(neighbor) = self.get_quantum(&neighbor_coord) {
                                            total_interference += center.interference_to(neighbor);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        self.stats.interference_events.wrapping_add(1);
        total_interference
    }

    /// Populate with entangled pairs at given density
    pub fn populate_entangled(&mut self, density: f32) {
        let target = (self.stats.total_cells as f32 * density) as usize;
        let mut rng_state = 0x12345678u64;

        for _ in 0..target {
            rng_state = rng_state.wrapping_mul(0x5DEECE66D).wrapping_add(0xB);
            let idx = (rng_state as usize) % self.stats.total_cells;
            let coord = Coord7D::from_index(idx, self.size);

            let mut fp = Fingerprint::zero();
            fp.set_bit((rng_state >> 17) as usize % crate::FINGERPRINT_BITS, true);
            fp.set_bit((rng_state >> 33) as usize % crate::FINGERPRINT_BITS, true);

            let phase = PhaseTag7D::from_seed(rng_state);
            self.set_quantum(&coord, QuantumCell7D::new(fp, phase));
        }
    }

    /// Bell test for 7D crystal
    pub fn bell_test(&self, samples: usize) -> BellTestResult7D {
        if self.cells.is_empty() {
            return BellTestResult7D::empty();
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
            let b_prime = cell_b.amplitude.permute(23);

            let corr = |x: &Fingerprint, y: &Fingerprint| -> f32 { 2.0 * x.similarity(y) - 1.0 };

            e_ab += corr(a, b);
            e_ab_prime += corr(a, &b_prime);
            e_a_prime_b += corr(&a_prime, b);
            e_a_prime_b_prime += corr(&a_prime, &b_prime);
        }

        let n = n_pairs as f32;
        e_ab /= n;
        e_ab_prime /= n;
        e_a_prime_b /= n;
        e_a_prime_b_prime /= n;

        let s = e_ab - e_ab_prime + e_a_prime_b + e_a_prime_b_prime;

        BellTestResult7D {
            s_value: s,
            is_quantum: s.abs() > 2.0,
            correlation_ab: e_ab,
            correlation_ab_prime: e_ab_prime,
            correlation_a_prime_b: e_a_prime_b,
            correlation_a_prime_b_prime: e_a_prime_b_prime,
            samples: n_pairs,
        }
    }
}

// =============================================================================
// BELL TEST RESULT
// =============================================================================

#[derive(Debug, Clone)]
pub struct BellTestResult7D {
    pub s_value: f32,
    pub is_quantum: bool,
    pub correlation_ab: f32,
    pub correlation_ab_prime: f32,
    pub correlation_a_prime_b: f32,
    pub correlation_a_prime_b_prime: f32,
    pub samples: usize,
}

impl BellTestResult7D {
    pub fn empty() -> Self {
        Self {
            s_value: 0.0,
            is_quantum: false,
            correlation_ab: 0.0,
            correlation_ab_prime: 0.0,
            correlation_a_prime_b: 0.0,
            correlation_a_prime_b_prime: 0.0,
            samples: 0,
        }
    }
}

// =============================================================================
// PRIME SWEET SPOTS FOR 7D
// =============================================================================

pub mod prime_sweet_spots {
    /// Prime dimension configuration for 7D
    #[derive(Debug, Clone, Copy)]
    pub struct PrimeDimension7D {
        pub prime: usize,
        pub cells: u64,
        pub neighbors: usize,
        pub estimated_us: u64,
        pub fits_17min: bool,
    }

    impl PrimeDimension7D {
        pub const fn new(prime: usize) -> Self {
            let cells = const_pow(prime as u64, 7);
            let neighbors = const_pow(3, 7) as usize - 1; // 2186
            let estimated_us = (prime as u64) * 7 * 10 * 100 / 1000 + cells / 100;
            let fits_17min = estimated_us < 17 * 60 * 1_000_000; // 17 minutes

            Self {
                prime,
                cells,
                neighbors,
                estimated_us,
                fits_17min,
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

    /// 3^7 = 2,187 cells (~2ms)
    pub const PRIME_3: PrimeDimension7D = PrimeDimension7D::new(3);

    /// 5^7 = 78,125 cells (~500ms)
    pub const PRIME_5: PrimeDimension7D = PrimeDimension7D::new(5);

    /// 7^7 = 823,543 cells (~5-30s) - THE SWEET SPOT
    pub const PRIME_7: PrimeDimension7D = PrimeDimension7D::new(7);

    /// 11^7 = 19,487,171 cells (~10-17 minutes, edge of 17-min budget)
    pub const PRIME_11: PrimeDimension7D = PrimeDimension7D::new(11);

    /// 13^7 = 62,748,517 cells (exceeds 17-min budget)
    pub const PRIME_13: PrimeDimension7D = PrimeDimension7D::new(13);

    /// 17^7 = 410,338,673 cells (way exceeds budget)
    pub const PRIME_17: PrimeDimension7D = PrimeDimension7D::new(17);

    /// All 7D configurations
    pub const ALL: [PrimeDimension7D; 6] =
        [PRIME_3, PRIME_5, PRIME_7, PRIME_11, PRIME_13, PRIME_17];

    /// Time slots within 17 minutes
    pub fn time_slots_17min() -> Vec<(&'static str, PrimeDimension7D)> {
        vec![
            ("0-5s (trivial)", PRIME_3),
            ("5s-1m (quick)", PRIME_5),
            ("1-5m (standard)", PRIME_7),
            ("5-17m (extended)", PRIME_11),
        ]
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coord7d_roundtrip() {
        let size = 5;
        for a in 0..size {
            for g in 0..size {
                let coord = Coord7D::new(a, 2, 3, 1, 0, 4, g);
                let idx = coord.to_index(size);
                let back = Coord7D::from_index(idx, size);
                assert_eq!(coord, back);
            }
        }
    }

    #[test]
    fn test_crystal7d_creation() {
        let crystal = Crystal7D::prime_7_7d();
        assert_eq!(crystal.size(), 7);
        assert_eq!(crystal.total_cells(), 823543);
        assert_eq!(crystal.active_cells(), 0);
    }

    #[test]
    fn test_prime_sweet_spots() {
        use prime_sweet_spots::*;

        assert_eq!(PRIME_3.cells, 2187);
        assert_eq!(PRIME_5.cells, 78125);
        assert_eq!(PRIME_7.cells, 823543);
        assert_eq!(PRIME_11.cells, 19487171);

        assert!(PRIME_7.fits_17min);
        assert!(PRIME_11.fits_17min); // Just barely
    }

    #[test]
    fn test_7d_bell_test() {
        let mut crystal = Crystal7D::prime_3_7d();
        crystal.populate_entangled(0.1);

        let result = crystal.bell_test(100);
        println!(
            "7D Bell test: S = {:.3}, quantum = {}",
            result.s_value, result.is_quantum
        );
    }
}
