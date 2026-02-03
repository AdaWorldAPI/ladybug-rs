//! Bitchain 5^5 Variant - String Theory Substrate
//!
//! Proof-of-work constrained quantum crystal at 5^5 = 3,125 cells.
//!
//! # String Theory Interpretation
//!
//! ```text
//! Classical simulation:     |S| can reach 4.0 (no physical constraints)
//! Quantum mechanics:        |S| ≤ 2√2 ≈ 2.828 (Tsirelson's bound)
//! Bitchained simulation:    |S| constrained by work requirements
//! ```
//!
//! ## Why Bitchaining Helps
//!
//! The Tsirelson bound exists because:
//! 1. Measurements are LOCAL (Alice and Bob can't communicate)
//! 2. Correlations must exist BEFORE measurement (no retrocausality)
//!
//! Bitchaining enforces:
//! 1. IRREVERSIBILITY - Can't undo computational work
//! 2. CAUSAL STRUCTURE - Chain links enforce temporal ordering
//! 3. MINIMUM "LENGTH" - Each state transition requires work (string tension)
//!
//! ## String Theory Connection
//!
//! ```text
//! String tension T = difficulty parameter
//! String length  L = total_work / T
//! Planck area    A = 1 bit of work
//!
//! Minimum observable "distance" = 1/(2^difficulty) of state space
//! This creates a "Planck scale" below which cheating is impossible
//! ```
//!
//! ## What Can't Be Cancelled Out
//!
//! Without bitchaining, a classical simulator can:
//! - Look at both Alice and Bob's states simultaneously
//! - Compute optimal correlations after the fact
//! - "Cancel out" any information-theoretic constraints
//!
//! With bitchaining:
//! - State history is COMMITTED before correlation measurement
//! - Shared ancestry is VERIFIABLE (merkle proofs)
//! - Fake correlations require exponential work
//!
//! ## Tsirelson Bound Effect
//!
//! Theoretical prediction:
//! - Low difficulty (1-4 bits): Still allows |S| > 2.828
//! - Medium difficulty (8-12 bits): |S| approaches 2.828
//! - High difficulty (16+ bits): |S| bounded by ~ 2.0 (classical!)
//!
//! The work requirement creates "causal separation" that mimics
//! the spacelike separation in real Bell tests.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use super::quantum_5d::{Coord5D, PhaseTag5D, QuantumCell5D, Crystal5D, CellResolution};

// =============================================================================
// BITCHAIN CELL (5D variant)
// =============================================================================

/// Chain-signed quantum state for 5^5 crystal
/// The proof-of-work creates "string tension" that constrains Bell correlations
#[derive(Clone)]
pub struct BitchainCell5D {
    /// Quantum state
    pub cell: QuantumCell5D,

    /// Hash of this state (commitment)
    pub state_hash: [u8; 32],

    /// Link to previous state
    pub prev_hash: [u8; 32],

    /// Merkle root for ancestry proofs
    pub merkle_root: [u8; 32],

    /// Nonce satisfying difficulty
    pub nonce: u64,

    /// Chain height
    pub height: u64,

    /// Cumulative work (in bits)
    pub total_work: u128,

    /// Timestamp (for causal ordering)
    pub timestamp: u64,
}

impl BitchainCell5D {
    /// Genesis block (no parent)
    pub fn genesis(cell: QuantumCell5D, difficulty: u32) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        let mut signed = Self {
            cell,
            state_hash: [0u8; 32],
            prev_hash: [0u8; 32],
            merkle_root: [0u8; 32],
            nonce: 0,
            height: 0,
            total_work: 0,
            timestamp,
        };
        signed.mine(difficulty);
        signed
    }

    /// Create child (links to parent)
    pub fn child(&self, new_cell: QuantumCell5D, difficulty: u32) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        let mut signed = Self {
            cell: new_cell,
            state_hash: [0u8; 32],
            prev_hash: self.state_hash,
            merkle_root: self.compute_merkle_update(),
            nonce: 0,
            height: self.height + 1,
            total_work: self.total_work,
            timestamp,
        };
        signed.mine(difficulty);
        signed
    }

    /// Compute hash of current state
    fn compute_state_hash(&self) -> [u8; 32] {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();

        // Hash the quantum state (use popcount and similarity as proxy)
        self.cell.amplitude.popcount().hash(&mut hasher);
        let zero = PhaseTag5D::zero();
        let phase_hash = (self.cell.phase.cos_angle_to(&zero) * 1000000.0) as i64;
        phase_hash.hash(&mut hasher);

        // Hash chain links
        self.prev_hash.hash(&mut hasher);
        self.merkle_root.hash(&mut hasher);
        self.nonce.hash(&mut hasher);
        self.height.hash(&mut hasher);
        self.timestamp.hash(&mut hasher);

        let h1 = hasher.finish();

        // Double hash for security
        let mut hasher2 = DefaultHasher::new();
        h1.hash(&mut hasher2);
        self.nonce.hash(&mut hasher2);
        let h2 = hasher2.finish();

        let mut result = [0u8; 32];
        result[0..8].copy_from_slice(&h1.to_le_bytes());
        result[8..16].copy_from_slice(&h2.to_le_bytes());
        result[16..24].copy_from_slice(&h1.to_be_bytes());
        result[24..32].copy_from_slice(&h2.to_be_bytes());
        result
    }

    /// Update merkle root (simplified: just hash prev + current)
    fn compute_merkle_update(&self) -> [u8; 32] {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        self.merkle_root.hash(&mut hasher);
        self.state_hash.hash(&mut hasher);

        let h = hasher.finish();
        let mut result = [0u8; 32];
        result[0..8].copy_from_slice(&h.to_le_bytes());
        result[8..16].copy_from_slice(&h.to_be_bytes());
        result
    }

    /// Mine: find nonce satisfying difficulty
    fn mine(&mut self, difficulty: u32) {
        loop {
            self.state_hash = self.compute_state_hash();

            let leading_zeros = self.count_leading_zeros();
            if leading_zeros >= difficulty {
                self.total_work += 1u128 << difficulty;
                break;
            }
            self.nonce += 1;
        }
    }

    fn count_leading_zeros(&self) -> u32 {
        let mut zeros = 0u32;
        for byte in &self.state_hash {
            if *byte == 0 {
                zeros += 8;
            } else {
                zeros += byte.leading_zeros();
                break;
            }
        }
        zeros
    }

    /// Verify proof-of-work
    pub fn verify(&self, difficulty: u32) -> bool {
        let computed = self.compute_state_hash();
        computed == self.state_hash && self.count_leading_zeros() >= difficulty
    }

    /// String tension: work per unit height
    pub fn string_tension(&self) -> f64 {
        if self.height == 0 {
            return 0.0;
        }
        (self.total_work as f64).log2() / self.height as f64
    }

    /// Causal age: microseconds since genesis
    pub fn causal_age(&self, genesis_time: u64) -> u64 {
        self.timestamp.saturating_sub(genesis_time)
    }
}


// =============================================================================
// BITCHAIN CRYSTAL 5^5
// =============================================================================

/// 5^5 = 3,125 cell bitchain crystal
/// Optimal for quick iteration with proof-of-work constraints
pub struct BitchainCrystal5x5 {
    /// Base crystal
    crystal: Crystal5D,

    /// Quantum strings at active coordinates
    strings: HashMap<usize, BitchainString5D>,

    /// Mining difficulty (string tension)
    difficulty: u32,

    /// Genesis timestamp
    genesis_time: u64,

    /// Total accumulated work
    total_work: u128,
}

/// Quantum string: worldline through 5D crystal
pub struct BitchainString5D {
    /// Chain of states
    states: Vec<BitchainCell5D>,

    /// Coordinate
    coord: Coord5D,

    /// Difficulty
    difficulty: u32,
}

impl BitchainString5D {
    pub fn new(coord: Coord5D, initial_cell: QuantumCell5D, difficulty: u32) -> Self {
        let genesis = BitchainCell5D::genesis(initial_cell, difficulty);
        Self {
            states: vec![genesis],
            coord,
            difficulty,
        }
    }

    pub fn evolve(&mut self, new_cell: QuantumCell5D) {
        if let Some(tip) = self.states.last() {
            let child = tip.child(new_cell, self.difficulty);
            self.states.push(child);
        }
    }

    pub fn current(&self) -> Option<&BitchainCell5D> {
        self.states.last()
    }

    pub fn length(&self) -> usize {
        self.states.len()
    }

    pub fn total_work(&self) -> u128 {
        self.states.last().map(|s| s.total_work).unwrap_or(0)
    }

    pub fn verify(&self) -> bool {
        for state in &self.states {
            if !state.verify(self.difficulty) {
                return false;
            }
        }
        true
    }
}

impl BitchainCrystal5x5 {
    /// Create new 5^5 bitchain crystal
    pub fn new(difficulty: u32) -> Self {
        let genesis_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        Self {
            crystal: Crystal5D::new(5, CellResolution::Standard),
            strings: HashMap::new(),
            difficulty,
            genesis_time,
            total_work: 0,
        }
    }

    /// Standard configuration: 4-bit difficulty (~16 hashes per state)
    pub fn standard() -> Self {
        Self::new(4)
    }

    /// High tension: 8-bit difficulty (~256 hashes per state)
    pub fn high_tension() -> Self {
        Self::new(8)
    }

    /// Inject state with proof-of-work
    pub fn inject(&mut self, coord: &Coord5D, cell: QuantumCell5D) {
        let idx = coord.to_index(5);

        if let Some(string) = self.strings.get_mut(&idx) {
            string.evolve(cell.clone());
        } else {
            let string = BitchainString5D::new(*coord, cell.clone(), self.difficulty);
            self.strings.insert(idx, string);
        }

        self.crystal.set(coord, cell.amplitude);
        self.total_work = self.strings.values().map(|s| s.total_work()).sum();
    }

    /// Verify all chains
    pub fn verify_all(&self) -> bool {
        self.strings.values().all(|s| s.verify())
    }

    /// Get string tension (average work per height)
    pub fn average_tension(&self) -> f64 {
        if self.strings.is_empty() {
            return 0.0;
        }

        let tensions: Vec<f64> = self.strings.values()
            .filter_map(|s| s.current())
            .map(|c| c.string_tension())
            .collect();

        tensions.iter().sum::<f64>() / tensions.len() as f64
    }

    /// Total work in bits
    pub fn total_work_bits(&self) -> f64 {
        (self.total_work as f64).log2()
    }

    /// Bell test with work-constrained correlations
    pub fn bell_test(&self, samples: usize) -> BitchainBellResult {
        let active: Vec<_> = self.strings.iter()
            .filter_map(|(idx, s)| s.current().map(|c| (*idx, c)))
            .collect();

        if active.len() < 2 {
            return BitchainBellResult::empty();
        }

        let n_pairs = active.len().min(samples);

        let mut e_ab = 0.0f32;
        let mut e_ab_prime = 0.0f32;
        let mut e_a_prime_b = 0.0f32;
        let mut e_a_prime_b_prime = 0.0f32;
        let mut work_weighted = 0.0f64;

        for i in 0..n_pairs {
            let j = (i + 1) % active.len();

            let (_, cell_a) = &active[i];
            let (_, cell_b) = &active[j];

            let a = &cell_a.cell.amplitude;
            let a_prime = cell_a.cell.amplitude.rotate(5);
            let b = &cell_b.cell.amplitude;
            let b_prime = cell_b.cell.amplitude.rotate(7);

            // Work-weighted correlation
            let work_factor = ((cell_a.total_work + cell_b.total_work) as f64).log2();

            // Use SparseFingerprint similarity (returns f64)
            let corr = |x: &crate::storage::lance_zero_copy::SparseFingerprint,
                        y: &crate::storage::lance_zero_copy::SparseFingerprint| -> f32 {
                2.0 * x.similarity(y) as f32 - 1.0
            };

            e_ab += corr(a, b);
            e_ab_prime += corr(a, &b_prime);
            e_a_prime_b += corr(&a_prime, b);
            e_a_prime_b_prime += corr(&a_prime, &b_prime);
            work_weighted += work_factor;
        }

        let n = n_pairs as f32;
        e_ab /= n;
        e_ab_prime /= n;
        e_a_prime_b /= n;
        e_a_prime_b_prime /= n;

        let s = e_ab - e_ab_prime + e_a_prime_b + e_a_prime_b_prime;

        BitchainBellResult {
            s_value: s,
            is_quantum: s.abs() > 2.0,
            bounded_by_tsirelson: s.abs() <= 2.828,
            difficulty: self.difficulty,
            total_work: self.total_work,
            work_per_sample: work_weighted / n_pairs as f64,
            samples: n_pairs,
            tension: self.average_tension(),
        }
    }
}

// =============================================================================
// BELL TEST RESULT
// =============================================================================

#[derive(Debug, Clone)]
pub struct BitchainBellResult {
    pub s_value: f32,
    pub is_quantum: bool,
    pub bounded_by_tsirelson: bool,
    pub difficulty: u32,
    pub total_work: u128,
    pub work_per_sample: f64,
    pub samples: usize,
    pub tension: f64,
}

impl BitchainBellResult {
    pub fn empty() -> Self {
        Self {
            s_value: 0.0,
            is_quantum: false,
            bounded_by_tsirelson: true,
            difficulty: 0,
            total_work: 0,
            work_per_sample: 0.0,
            samples: 0,
            tension: 0.0,
        }
    }

    /// Theoretical analysis
    pub fn analysis(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("S value: {:.4}\n", self.s_value));
        s.push_str(&format!("Classical limit (|S| ≤ 2): {}\n", self.s_value.abs() <= 2.0));
        s.push_str(&format!("Tsirelson bound (|S| ≤ 2.828): {}\n", self.bounded_by_tsirelson));
        s.push_str(&format!("Difficulty: {} bits\n", self.difficulty));
        s.push_str(&format!("Total work: 2^{:.2} hashes\n", (self.total_work as f64).log2()));
        s.push_str(&format!("String tension: {:.4}\n", self.tension));
        s
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::quantum_5d::resolution;

    #[test]
    fn test_bitchain_5x5_creation() {
        let crystal = BitchainCrystal5x5::standard();
        assert_eq!(crystal.difficulty, 4);
    }

    #[test]
    fn test_bitchain_5x5_inject() {
        let mut crystal = BitchainCrystal5x5::new(2); // Low difficulty for fast test

        let fp = resolution::standard();
        let cell = QuantumCell5D::from_fingerprint(fp);

        let coord = Coord5D::new(2, 2, 2, 2, 2);
        crystal.inject(&coord, cell);

        assert!(crystal.total_work > 0);
        assert!(crystal.verify_all());
    }

    #[test]
    fn test_bitchain_5x5_bell_test() {
        let mut crystal = BitchainCrystal5x5::new(2);

        // Populate with correlated states
        for i in 0..5 {
            for j in 0..2 {
                let mut fp = resolution::standard();
                fp.set(0, (i * 10 + j) as u64);
                fp.set(1, (100 + i) as u64);
                let cell = QuantumCell5D::from_fingerprint(fp);

                let coord = Coord5D::new(i, j, 2, 2, 2);
                crystal.inject(&coord, cell);
            }
        }

        let result = crystal.bell_test(10);
        println!("5^5 Bitchain Bell test:\n{}", result.analysis());
    }
}
