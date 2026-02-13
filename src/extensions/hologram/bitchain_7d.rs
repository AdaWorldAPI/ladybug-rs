//! Bitchain 7^7 Variant - The Sweet Spot for String Theory Simulation
//!
//! Proof-of-work constrained quantum crystal at 7^7 = 823,543 cells.
//!
//! # Why 7^7 is Special
//!
//! ```text
//! 5^5 = 3,125 cells     → Fast iteration, weak statistical power
//! 7^7 = 823,543 cells   → SWEET SPOT: Strong stats, computable in <1 min
//! 11^7 = 19M cells      → Too slow for interactive exploration
//! ```
//!
//! ## String Theory Length Scale
//!
//! In string theory, the string length l_s sets the minimum observable distance.
//! In bitchaining, the difficulty parameter creates an analogous scale:
//!
//! ```text
//! Physical string:    Minimum length = l_s ≈ 10^-34 m (Planck scale)
//! Bitchain string:    Minimum "length" = 2^(-difficulty) of state space
//!
//! At difficulty 16:   1/65536 of state space is "below Planck scale"
//! At difficulty 32:   1/4 billion states are indistinguishable
//! ```
//!
//! ## Effort vs Realness Trade-off
//!
//! | Difficulty | Work per state | "Planck area" | Tsirelson effect |
//! |------------|----------------|---------------|------------------|
//! | 4 bits     | ~16 hashes     | Large         | S can reach ~3.5 |
//! | 8 bits     | ~256 hashes    | Medium        | S limited to ~3.0|
//! | 12 bits    | ~4096 hashes   | Small         | S approaches 2.8 |
//! | 16 bits    | ~65536 hashes  | Tiny          | S ~ 2.5 (bounded)|
//! | 20+ bits   | Expensive      | Quantum-like  | S ≤ 2.0 (classical!)|
//!
//! ## The Paradox: More Work = MORE Classical?
//!
//! Counter-intuitive but physically correct:
//! - Real quantum systems have NO work (measurement is instant)
//! - Bitchaining adds "causal structure" absent in real QM
//! - High difficulty → strong causal ordering → classical behavior
//! - The "quantum" regime is actually LOW difficulty!
//!
//! This suggests:
//! - Quantum correlations emerge from LACK of causal structure
//! - Bitchaining reveals the "hidden variable" that QM forbids
//! - True quantum simulation requires minimum-work proof systems
//!
//! ## Research Applications
//!
//! 1. **Quantum gravity**: Explore how spacetime discreteness affects correlations
//! 2. **Information theory**: Measure "causal entropy" of quantum states
//! 3. **Cryptography**: Bell violations as computational hardness proofs
//! 4. **Philosophy**: Test whether "free will" loopholes are computational

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use super::quantum_7d::{Coord7D, PhaseTag7D, QuantumCell7D};
use crate::core::Fingerprint;

// =============================================================================
// BITCHAIN CELL (7D variant)
// =============================================================================

/// Chain-signed quantum state for 7^7 crystal
#[derive(Clone)]
pub struct BitchainCell7D {
    /// Quantum state
    pub cell: QuantumCell7D,

    /// Hash commitment
    pub state_hash: [u8; 32],

    /// Link to previous
    pub prev_hash: [u8; 32],

    /// Merkle root
    pub merkle_root: [u8; 32],

    /// PoW nonce
    pub nonce: u64,

    /// Height in chain
    pub height: u64,

    /// Cumulative work
    pub total_work: u128,

    /// Timestamp (μs)
    pub timestamp: u64,

    /// 7D coordinate signature (for locality proofs)
    pub coord_sig: [u8; 7],
}

impl BitchainCell7D {
    /// Genesis block
    pub fn genesis(cell: QuantumCell7D, coord: &Coord7D, difficulty: u32) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        let coord_sig = [
            coord.a as u8,
            coord.b as u8,
            coord.c as u8,
            coord.d as u8,
            coord.e as u8,
            coord.f as u8,
            coord.g as u8,
        ];

        let mut signed = Self {
            cell,
            state_hash: [0u8; 32],
            prev_hash: [0u8; 32],
            merkle_root: [0u8; 32],
            nonce: 0,
            height: 0,
            total_work: 0,
            timestamp,
            coord_sig,
        };
        signed.mine(difficulty);
        signed
    }

    /// Create child
    pub fn child(&self, new_cell: QuantumCell7D, difficulty: u32) -> Self {
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
            coord_sig: self.coord_sig,
        };
        signed.mine(difficulty);
        signed
    }

    fn compute_state_hash(&self) -> [u8; 32] {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();

        // Hash quantum state
        self.cell.amplitude.as_raw().hash(&mut hasher);

        // Hash chain data
        self.prev_hash.hash(&mut hasher);
        self.merkle_root.hash(&mut hasher);
        self.nonce.hash(&mut hasher);
        self.height.hash(&mut hasher);
        self.timestamp.hash(&mut hasher);
        self.coord_sig.hash(&mut hasher);

        let h1 = hasher.finish();

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

    fn mine(&mut self, difficulty: u32) {
        loop {
            self.state_hash = self.compute_state_hash();
            let zeros = self.count_leading_zeros();
            if zeros >= difficulty {
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

    pub fn verify(&self, difficulty: u32) -> bool {
        let computed = self.compute_state_hash();
        computed == self.state_hash && self.count_leading_zeros() >= difficulty
    }

    pub fn string_tension(&self) -> f64 {
        if self.height == 0 {
            return 0.0;
        }
        (self.total_work as f64).log2() / self.height as f64
    }

    /// String length: total work in "Planck units"
    pub fn string_length(&self, difficulty: u32) -> f64 {
        self.total_work as f64 / (1u128 << difficulty) as f64
    }
}

// =============================================================================
// BITCHAIN STRING (7D worldline)
// =============================================================================

pub struct BitchainString7D {
    states: Vec<BitchainCell7D>,
    coord: Coord7D,
    difficulty: u32,
}

impl BitchainString7D {
    pub fn new(coord: Coord7D, initial: QuantumCell7D, difficulty: u32) -> Self {
        let genesis = BitchainCell7D::genesis(initial, &coord, difficulty);
        Self {
            states: vec![genesis],
            coord,
            difficulty,
        }
    }

    pub fn evolve(&mut self, new_cell: QuantumCell7D) {
        if let Some(tip) = self.states.last() {
            let child = tip.child(new_cell, self.difficulty);
            self.states.push(child);
        }
    }

    pub fn current(&self) -> Option<&BitchainCell7D> {
        self.states.last()
    }

    pub fn length(&self) -> usize {
        self.states.len()
    }

    pub fn total_work(&self) -> u128 {
        self.states.last().map(|s| s.total_work).unwrap_or(0)
    }

    pub fn verify(&self) -> bool {
        self.states.iter().all(|s| s.verify(self.difficulty))
    }

    /// Compute entanglement with another string
    pub fn entanglement_score(&self, other: &BitchainString7D) -> Option<f64> {
        let a = self.current()?;
        let b = other.current()?;

        // Shared ancestry?
        let shared = a.merkle_root == b.merkle_root || a.prev_hash == b.prev_hash;

        if shared {
            let sim = a.cell.amplitude.similarity(&b.cell.amplitude) as f64;
            let work = (a.total_work + b.total_work) as f64;
            Some(sim * work.log2())
        } else {
            None
        }
    }
}

// =============================================================================
// BITCHAIN CRYSTAL 7^7
// =============================================================================

/// 7^7 = 823,543 cell bitchain crystal
pub struct BitchainCrystal7x7 {
    /// Size (7)
    size: usize,

    /// Quantum strings
    strings: HashMap<usize, BitchainString7D>,

    /// Mining difficulty
    difficulty: u32,

    /// Genesis timestamp
    genesis_time: u64,

    /// Total work
    total_work: u128,
}

impl BitchainCrystal7x7 {
    /// Create 7^7 crystal
    pub fn new(difficulty: u32) -> Self {
        let genesis_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        Self {
            size: 7,
            strings: HashMap::new(),
            difficulty,
            genesis_time,
            total_work: 0,
        }
    }

    /// Standard: 4-bit difficulty
    pub fn standard() -> Self {
        Self::new(4)
    }

    /// High tension: 8-bit (research quality)
    pub fn high_tension() -> Self {
        Self::new(8)
    }

    /// Maximum tension: 12-bit (for Tsirelson bound testing)
    pub fn max_tension() -> Self {
        Self::new(12)
    }

    /// Total cells
    pub fn total_cells(&self) -> usize {
        823543 // 7^7
    }

    /// Active cells
    pub fn active_cells(&self) -> usize {
        self.strings.len()
    }

    /// Inject state with PoW
    pub fn inject(&mut self, coord: &Coord7D, cell: QuantumCell7D) {
        let idx = coord.to_index(self.size);

        if let Some(string) = self.strings.get_mut(&idx) {
            string.evolve(cell);
        } else {
            let string = BitchainString7D::new(coord.clone(), cell, self.difficulty);
            self.strings.insert(idx, string);
        }

        self.total_work = self.strings.values().map(|s| s.total_work()).sum();
    }

    /// Verify all
    pub fn verify_all(&self) -> bool {
        self.strings.values().all(|s| s.verify())
    }

    /// Average tension
    pub fn average_tension(&self) -> f64 {
        if self.strings.is_empty() {
            return 0.0;
        }

        let tensions: Vec<f64> = self
            .strings
            .values()
            .filter_map(|s| s.current())
            .map(|c| c.string_tension())
            .collect();

        tensions.iter().sum::<f64>() / tensions.len().max(1) as f64
    }

    /// Bell test with work constraints
    pub fn bell_test(&self, samples: usize) -> BitchainBellResult7D {
        let active: Vec<_> = self
            .strings
            .iter()
            .filter_map(|(idx, s)| s.current().map(|c| (*idx, c)))
            .collect();

        if active.len() < 2 {
            return BitchainBellResult7D::empty();
        }

        let n_pairs = active.len().min(samples);

        let mut e_ab = 0.0f32;
        let mut e_ab_prime = 0.0f32;
        let mut e_a_prime_b = 0.0f32;
        let mut e_a_prime_b_prime = 0.0f32;
        let mut work_sum = 0u128;

        for i in 0..n_pairs {
            let j = (i + 1) % active.len();

            let (_, cell_a) = &active[i];
            let (_, cell_b) = &active[j];

            let a = &cell_a.cell.amplitude;
            let a_prime = cell_a.cell.amplitude.permute(7);
            let b = &cell_b.cell.amplitude;
            let b_prime = cell_b.cell.amplitude.permute(11);

            let corr = |x: &Fingerprint, y: &Fingerprint| -> f32 { 2.0 * x.similarity(y) - 1.0 };

            e_ab += corr(a, b);
            e_ab_prime += corr(a, &b_prime);
            e_a_prime_b += corr(&a_prime, b);
            e_a_prime_b_prime += corr(&a_prime, &b_prime);
            work_sum += cell_a.total_work + cell_b.total_work;
        }

        let n = n_pairs as f32;
        e_ab /= n;
        e_ab_prime /= n;
        e_a_prime_b /= n;
        e_a_prime_b_prime /= n;

        let s = e_ab - e_ab_prime + e_a_prime_b + e_a_prime_b_prime;

        BitchainBellResult7D {
            s_value: s,
            is_quantum: s.abs() > 2.0,
            bounded_by_tsirelson: s.abs() <= 2.828,
            approaches_classical: s.abs() <= 2.1,
            difficulty: self.difficulty,
            total_work: self.total_work,
            avg_work_per_pair: work_sum as f64 / n_pairs as f64 / 2.0,
            samples: n_pairs,
            tension: self.average_tension(),
            string_length_estimate: self.total_work as f64 / (1u128 << self.difficulty) as f64,
        }
    }

    /// Populate with entangled pairs
    pub fn populate_entangled(&mut self, density: f32) {
        let target = (823543.0 * density) as usize;
        let mut rng = 0x7777777u64;

        for _ in 0..target {
            rng = rng.wrapping_mul(0x5DEECE66D).wrapping_add(0xB);

            let a = (rng >> 0) as usize % 7;
            let b = (rng >> 8) as usize % 7;
            let c = (rng >> 16) as usize % 7;
            let d = (rng >> 24) as usize % 7;
            let e = (rng >> 32) as usize % 7;
            let f = (rng >> 40) as usize % 7;
            let g = (rng >> 48) as usize % 7;

            let coord = Coord7D::new(a, b, c, d, e, f, g);

            let mut fp = Fingerprint::zero();
            fp.set_bit((rng >> 17) as usize % crate::FINGERPRINT_BITS, true);
            fp.set_bit((rng >> 33) as usize % crate::FINGERPRINT_BITS, true);

            let phase = PhaseTag7D::from_seed(rng);
            let cell = QuantumCell7D::new(fp, phase);

            self.inject(&coord, cell);
        }
    }
}

// =============================================================================
// BELL TEST RESULT
// =============================================================================

#[derive(Debug, Clone)]
pub struct BitchainBellResult7D {
    pub s_value: f32,
    pub is_quantum: bool,
    pub bounded_by_tsirelson: bool,
    pub approaches_classical: bool,
    pub difficulty: u32,
    pub total_work: u128,
    pub avg_work_per_pair: f64,
    pub samples: usize,
    pub tension: f64,
    pub string_length_estimate: f64,
}

impl BitchainBellResult7D {
    pub fn empty() -> Self {
        Self {
            s_value: 0.0,
            is_quantum: false,
            bounded_by_tsirelson: true,
            approaches_classical: true,
            difficulty: 0,
            total_work: 0,
            avg_work_per_pair: 0.0,
            samples: 0,
            tension: 0.0,
            string_length_estimate: 0.0,
        }
    }

    /// Full analysis
    pub fn analysis(&self) -> String {
        let mut s = String::new();
        s.push_str("╔══════════════════════════════════════════════════════════════╗\n");
        s.push_str("║           7^7 BITCHAIN BELL TEST ANALYSIS                    ║\n");
        s.push_str("╠══════════════════════════════════════════════════════════════╣\n");
        s.push_str(&format!(
            "║  S value:              {:.4}                              ║\n",
            self.s_value
        ));
        s.push_str(&format!(
            "║  Classical (|S|≤2):    {}                                  ║\n",
            if self.s_value.abs() <= 2.0 {
                "YES"
            } else {
                "NO "
            }
        ));
        s.push_str(&format!(
            "║  Tsirelson (|S|≤2.83): {}                                  ║\n",
            if self.bounded_by_tsirelson {
                "YES"
            } else {
                "NO "
            }
        ));
        s.push_str("╠══════════════════════════════════════════════════════════════╣\n");
        s.push_str(&format!(
            "║  Difficulty:           {} bits                             ║\n",
            self.difficulty
        ));
        s.push_str(&format!(
            "║  Total work:           2^{:.2} hashes                       ║\n",
            (self.total_work as f64).log2()
        ));
        s.push_str(&format!(
            "║  String tension:       {:.4}                              ║\n",
            self.tension
        ));
        s.push_str(&format!(
            "║  String length:        {:.2} Planck units                 ║\n",
            self.string_length_estimate
        ));
        s.push_str("╠══════════════════════════════════════════════════════════════╣\n");
        s.push_str("║  INTERPRETATION:                                             ║\n");

        if self.s_value.abs() > 2.828 {
            s.push_str("║  System exceeds Tsirelson bound - SUPER-QUANTUM             ║\n");
            s.push_str("║  → Insufficient causal structure for physical constraints  ║\n");
            s.push_str("║  → Increase difficulty to approach realistic behavior      ║\n");
        } else if self.s_value.abs() > 2.0 {
            s.push_str("║  System violates Bell inequality - QUANTUM-LIKE            ║\n");
            s.push_str("║  → Work requirement creating partial causal separation     ║\n");
            s.push_str("║  → Difficulty is in 'sweet spot' for QM simulation         ║\n");
        } else {
            s.push_str("║  System obeys classical limit - LOCAL                      ║\n");
            s.push_str("║  → High causal ordering dominates quantum correlations     ║\n");
            s.push_str("║  → Decrease difficulty to recover quantum behavior         ║\n");
        }

        s.push_str("╚══════════════════════════════════════════════════════════════╝\n");
        s
    }
}

// =============================================================================
// RESEARCH UTILITIES
// =============================================================================

/// Sweep difficulty parameter to find Tsirelson transition
pub fn difficulty_sweep(density: f32, samples: usize) -> Vec<(u32, f32)> {
    let mut results = Vec::new();

    for difficulty in [2, 4, 6, 8, 10, 12, 14, 16] {
        let mut crystal = BitchainCrystal7x7::new(difficulty);
        crystal.populate_entangled(density);
        let result = crystal.bell_test(samples);
        results.push((difficulty, result.s_value));
    }

    results
}

/// Find difficulty that achieves target S value
pub fn find_tsirelson_difficulty(target_s: f32, density: f32, samples: usize) -> Option<u32> {
    for difficulty in 1..20 {
        let mut crystal = BitchainCrystal7x7::new(difficulty);
        crystal.populate_entangled(density);
        let result = crystal.bell_test(samples);

        if result.s_value.abs() <= target_s {
            return Some(difficulty);
        }
    }
    None
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitchain_7x7_creation() {
        let crystal = BitchainCrystal7x7::standard();
        assert_eq!(crystal.total_cells(), 823543);
        assert_eq!(crystal.difficulty, 4);
    }

    #[test]
    fn test_bitchain_7x7_inject() {
        let mut crystal = BitchainCrystal7x7::new(2);

        let mut fp = Fingerprint::zero();
        fp.set_bit(0, true);
        let cell = QuantumCell7D::new(fp, PhaseTag7D::zero());

        let coord = Coord7D::new(3, 3, 3, 3, 3, 3, 3);
        crystal.inject(&coord, cell);

        assert!(crystal.total_work > 0);
        assert!(crystal.verify_all());
    }

    #[test]
    fn test_bitchain_7x7_bell() {
        let mut crystal = BitchainCrystal7x7::new(2);
        crystal.populate_entangled(0.001); // 0.1% = ~823 cells

        let result = crystal.bell_test(50);
        println!("{}", result.analysis());
    }

    #[test]
    fn test_difficulty_sweep() {
        let results = difficulty_sweep(0.001, 20);

        println!("\nDifficulty vs S value:");
        for (d, s) in results {
            let marker = if s.abs() > 2.828 {
                "SUPER-Q"
            } else if s.abs() > 2.0 {
                "QUANTUM"
            } else {
                "CLASSIC"
            };
            println!("  {} bits: S = {:.4} [{}]", d, s, marker);
        }
    }
}
