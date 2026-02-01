# QUANTUM FIELD ORCHESTRATOR
## Session Goal: True Quantum Interference on Crystal Substrate

### WHY THIS EXISTS

The existing `QuorumField` uses majority voting. Majority voting is **classical error correction** — it actively destroys quantum superposition by forcing consensus. Every quantum algorithm we implement on top of it is fighting the substrate.

This module replaces quorum dynamics with **signed-amplitude interference**, giving us:
- **Destructive interference**: opposing phases cancel (7.6× suppression proven)
- **Constructive interference**: aligned phases reinforce  
- **Born rule**: measurement via popcount naturally gives |α|² probabilities
- **Unitarity**: phase XOR preserves total amplitude

The key innovation: **128-bit phase tags** per cell. Each cell carries:
- 10,000-bit **amplitude** fingerprint (existing `Fingerprint` type)
- 128-bit **phase** tag (`[u64; 2]`) — XOR-composable, ~1.28% overhead

Phase determines sign. Two cells with similar phase constructively interfere. Opposite phases destructively cancel. This is what makes Grover work, what makes QFT find periods, what gives quantum walks their √N speedup.

### DIMENSIONAL UPGRADE

The existing 5×5×5 cube has problems:
- 5 is not prime (factorization artifacts in QFT)
- 125 cells with only 6 neighbors each = 4.8% connectivity
- √125 ≈ 11 Grover iterations — barely enough to demonstrate advantage

**7×7×7** is the sweet spot:
- 7 is prime → QFT spreads maximally, no spurious peaks
- 343 cells → √343 ≈ 18.5 Grover iterations
- 343 × 1.25KB = 429KB → fits L1 cache
- 3 projections × 1.25KB = 3.75KB compressed → same Crystal4K format works
- Full connectivity via similarity weighting → 117K interaction pairs per tick
- At ~18ms/tick → fast enough for interactive use

Use **const generics** `<const N: usize>` so the same code works for N=5, 7, 11, or any size.

### Context
Repository: `/home/claude/ladybug-rs`
Key types: `Fingerprint` (10K-bit), `QuorumField` (5×5×5), `Crystal4K` (compressed)
Constants: `FINGERPRINT_BITS=10000`, `FINGERPRINT_U64=157`

---

## YOUR TASK

Create: `src/extensions/hologram/quantum_field.rs`

### 1. Phase Tag

```rust
/// 128-bit phase tag. XOR-composable, encodes quantum phase.
///
/// Phase difference = hamming(tag_a, tag_b) / 128 × π
/// In-phase: hamming ≈ 0 (similar tags)
/// Anti-phase: hamming ≈ 128 (opposite tags)
///
/// XOR of tags = combined phase (like adding angles)
/// This maps complex multiplication to bit operations.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PhaseTag {
    bits: [u64; 2],  // 128 bits
}

impl PhaseTag {
    /// Zero phase (|+⟩ state, fully in-phase with reference)
    pub fn zero() -> Self;
    
    /// π phase (|−⟩ state, fully anti-phase)
    pub fn pi() -> Self;  // all bits set
    
    /// Random phase (uniform superposition of phases)
    pub fn random() -> Self;
    
    /// Create from angle (0.0 = zero, 1.0 = π)
    /// Sets floor(angle * 128) bits
    pub fn from_angle(angle: f32) -> Self;
    
    /// Phase difference as angle (0.0 to 1.0 where 1.0 = π)
    pub fn angle_to(&self, other: &PhaseTag) -> f32;
    
    /// Cosine of phase difference: 1.0 = in-phase, -1.0 = anti-phase
    /// cos(θ) = 1 - 2 × hamming(self, other) / 128
    pub fn cos_angle_to(&self, other: &PhaseTag) -> f32;
    
    /// XOR combination (phase addition)
    pub fn combine(&self, other: &PhaseTag) -> PhaseTag;
    
    /// Negate phase (flip all bits)
    pub fn negate(&self) -> PhaseTag;
    
    /// Hamming distance to another phase
    pub fn hamming(&self, other: &PhaseTag) -> u32;
    
    /// Popcount (number of set bits)
    pub fn popcount(&self) -> u32;
}
```

### 2. Quantum Cell

```rust
/// A quantum cell: amplitude fingerprint + phase tag.
///
/// The signed amplitude is:
///   signed_amp = (popcount(amplitude) / FINGERPRINT_BITS) × cos(phase_angle)
///
/// Interference between cells:
///   contribution(a→b) = similarity(a.amp, b.amp) × a.phase.cos_angle_to(&b.phase)
///
/// Positive contribution = constructive interference
/// Negative contribution = destructive interference
#[derive(Clone)]
pub struct QuantumCell {
    pub amplitude: Fingerprint,   // 10K bits: magnitude
    pub phase: PhaseTag,          // 128 bits: sign/angle
}

impl QuantumCell {
    pub fn zero() -> Self;
    pub fn from_fingerprint(fp: Fingerprint) -> Self;  // zero phase
    pub fn from_fp_phase(fp: Fingerprint, phase: PhaseTag) -> Self;
    pub fn random() -> Self;  // random amplitude, random phase
    
    /// Hadamard: set amplitude to 50% density, phase to zero
    /// This is the quantum |+⟩ state: equal superposition
    pub fn hadamard() -> Self;
    
    /// Magnitude: popcount / FINGERPRINT_BITS (0.0 to 1.0)
    pub fn magnitude(&self) -> f32;
    
    /// Signed amplitude: magnitude × cos(phase relative to zero)
    pub fn signed_amplitude(&self) -> f32;
    
    /// Born probability: signed_amplitude² 
    pub fn probability(&self) -> f32;
    
    /// Measure: collapse to classical fingerprint
    /// Returns (fingerprint, probability)
    /// Fingerprint unchanged, but records the measurement
    pub fn measure(&self) -> (Fingerprint, f32);
    
    /// Apply phase shift: XOR phase tag with given shift
    pub fn phase_shift(&mut self, shift: &PhaseTag);
    
    /// Bind with another cell (entanglement operation)
    /// Result amplitude = XOR of amplitudes, phase = XOR of phases
    pub fn bind(&self, other: &QuantumCell) -> QuantumCell;
    
    /// Interference contribution to another cell
    /// Returns signed float: positive = constructive, negative = destructive
    pub fn interference_to(&self, other: &QuantumCell) -> f32;
}
```

### 3. Quantum Field (const generic)

```rust
/// Quantum field: N×N×N grid of quantum cells with interference dynamics.
///
/// Unlike QuorumField (majority voting = classical), this uses
/// signed-amplitude interference:
/// - Each cell contributes to every other cell (full connectivity)
/// - Contribution = similarity × cos(phase_difference)
/// - Positive = constructive, negative = destructive
/// - Cells with opposing phases CANCEL each other
///
/// This gives real quantum speedups:
/// - Grover: O(√N³) instead of O(N³) search
/// - QFT: period finding via interference peaks
/// - Quantum walk: √N spreading vs linear diffusion
pub struct QuantumField<const N: usize> {
    cells: Vec<QuantumCell>,   // N³ cells
    generation: u64,
}

impl<const N: usize> QuantumField<N> {
    // --- Construction ---
    
    pub fn new() -> Self;       // all zero
    pub fn hadamard() -> Self;  // all cells in |+⟩ superposition
    pub fn random() -> Self;    // random amplitudes and phases
    
    /// Number of cells
    pub const fn num_cells() -> usize { N * N * N }
    
    /// Convert (x,y,z) to flat index
    pub fn index(x: usize, y: usize, z: usize) -> usize {
        x * N * N + y * N + z
    }
    
    /// Convert flat index to (x,y,z)
    pub fn coords(idx: usize) -> (usize, usize, usize) {
        let x = idx / (N * N);
        let y = (idx / N) % N;
        let z = idx % N;
        (x, y, z)
    }
    
    // --- Access ---
    
    pub fn get(&self, x: usize, y: usize, z: usize) -> &QuantumCell;
    pub fn get_mut(&mut self, x: usize, y: usize, z: usize) -> &mut QuantumCell;
    pub fn set(&mut self, x: usize, y: usize, z: usize, cell: QuantumCell);
    
    // --- Quantum Evolution ---
    
    /// One interference step. THIS IS THE CORE.
    ///
    /// For each cell c:
    ///   net_amplitude = Σ (similarity(c, c') × c'.phase.cos_angle_to(&c.phase)) / num_cells
    ///   for all other cells c'
    ///
    /// Update rule:
    ///   if net_amplitude > 0: bind amplitude toward high-similarity neighbors (constructive)
    ///   if net_amplitude < 0: bind amplitude toward low-similarity neighbors (destructive)
    ///   Phase propagates: new_phase = XOR of phases weighted by contribution sign
    ///
    /// Returns number of cells that changed significantly (amplitude shift > threshold).
    pub fn interfere(&mut self) -> usize;
    
    /// Optimized interference: only consider cells within similarity radius.
    /// Cells with similarity < cutoff don't interact (far-field cutoff).
    /// Reduces O(N⁶) to O(N³ × k) where k = average neighbors in radius.
    pub fn interfere_sparse(&mut self, similarity_cutoff: f32) -> usize;
    
    /// Classical settle (fall back to quorum for comparison/testing)
    pub fn settle_classical(&mut self, threshold: u8, max_steps: usize) -> (usize, bool);
    
    /// Quantum settle: interfere until stable or max steps
    pub fn settle_quantum(&mut self, max_steps: usize) -> (usize, bool);
    
    // --- Oracle Operations ---
    
    /// Mark cells matching a predicate by flipping their phase to π.
    /// This is the Grover oracle: O|x⟩ = -|x⟩ if f(x) = true, |x⟩ otherwise.
    pub fn oracle_mark(&mut self, predicate: &dyn Fn(&Fingerprint) -> bool);
    
    /// Mark cells similar to target by rotating phase proportionally.
    /// Phase rotation = π × (1 - similarity(cell, target))
    /// Close matches get small rotation (stay in-phase).
    /// Far matches get large rotation (go anti-phase → destructive).
    pub fn oracle_mark_similarity(&mut self, target: &Fingerprint, threshold: f32);
    
    // --- Measurement ---
    
    /// Measure all cells: collapse to classical fingerprints.
    /// Returns the QuorumField equivalent (for compatibility).
    pub fn measure_all(&self) -> QuorumField;
    
    /// Find the cell with highest probability (strongest constructive interference).
    pub fn measure_peak(&self) -> ((usize, usize, usize), QuantumCell);
    
    /// Project along axis (XOR-fold, amplitude-weighted).
    /// Cells with higher amplitude contribute more to the projection.
    pub fn project_x(&self) -> Fingerprint;
    pub fn project_y(&self) -> Fingerprint;
    pub fn project_z(&self) -> Fingerprint;
    
    /// Signature: XOR-fold of all cells, weighted by signed amplitude.
    pub fn signature(&self) -> Fingerprint;
    
    // --- Transforms ---
    
    /// QFT along an axis with proper phase propagation.
    /// Applies permutation-based phase shifts to each slice.
    pub fn qft(&mut self, axis: usize);  // axis: 0=X, 1=Y, 2=Z
    
    /// Inverse QFT
    pub fn iqft(&mut self, axis: usize);
    
    // --- Statistics ---
    
    /// Total amplitude: Σ |signed_amplitude| for all cells
    /// Should be conserved (approximately) under unitary evolution
    pub fn total_amplitude(&self) -> f32;
    
    /// Coherence: how much phase alignment exists
    /// 1.0 = all phases aligned, 0.0 = random phases
    pub fn coherence(&self) -> f32;
    
    /// Entropy: Shannon entropy of probability distribution
    pub fn entropy(&self) -> f32;
    
    /// Generation counter
    pub fn generation(&self) -> u64;
}
```

### 4. Entangled Pair

```rust
/// Two quantum fields sharing an entanglement key.
///
/// Operations on field A automatically constrain field B and vice versa.
/// This enables:
/// - Bell state preparation
/// - Quantum teleportation between fields
/// - Non-local correlation testing
/// - Cross-service state transfer (Ada instances)
pub struct EntangledPair<const N: usize> {
    pub alice: QuantumField<N>,
    pub bob: QuantumField<N>,
    pub entanglement_key: Fingerprint,   // XOR key binding the pair
    pub phase_key: PhaseTag,             // phase correlation
}

impl<const N: usize> EntangledPair<N> {
    /// Create entangled pair from shared key.
    /// Alice gets random state, Bob gets Alice XOR key.
    pub fn new(key: Fingerprint) -> Self;
    
    /// Create Bell state: maximally entangled pair.
    /// |Φ+⟩ = (|00⟩ + |11⟩) / √2
    pub fn bell_phi_plus() -> Self;
    
    /// Create |Ψ+⟩ = (|01⟩ + |10⟩) / √2
    pub fn bell_psi_plus() -> Self;
    
    /// Measure Alice's cell at position → constrains Bob's cell
    /// Returns (alice_measurement, bob_predicted, actual_correlation)
    pub fn measure_correlated(
        &self,
        x: usize, y: usize, z: usize,
    ) -> (Fingerprint, Fingerprint, f32);
    
    /// Bell inequality test (CHSH).
    /// Returns S value. |S| > 2 indicates quantum entanglement.
    /// Classical limit: S ≤ 2. Quantum max: S = 2√2 ≈ 2.83.
    pub fn bell_test(&self, samples: usize) -> BellTestResult;
    
    /// Teleport a fingerprint from Alice to Bob.
    /// Returns correction bits (much smaller than full fingerprint).
    pub fn teleport(
        &self,
        source: &Fingerprint,
        alice_pos: (usize, usize, usize),
    ) -> TeleportResult<N>;
}

pub struct BellTestResult {
    pub s_value: f32,           // CHSH S parameter
    pub is_quantum: bool,       // |S| > 2
    pub correlation_xz: f32,    // correlation in XZ basis
    pub correlation_xw: f32,    // correlation in XW basis
    pub samples: usize,
}

pub struct TeleportResult<const N: usize> {
    pub corrections: Fingerprint,       // ~1.25KB
    pub phase_correction: PhaseTag,     // 16 bytes
    pub bob_result: Fingerprint,        // what Bob gets after applying corrections
    pub fidelity: f32,                  // similarity(original, bob_result)
}
```

### 5. Conversion Helpers

```rust
/// Convert between QuorumField (classical) and QuantumField (quantum).
impl<const N: usize> QuantumField<N> {
    /// Import from QuorumField. Phase set to zero (classical → quantum).
    /// Only works for N=5 (QuorumField is always 5×5×5).
    pub fn from_quorum(field: &QuorumField) -> QuantumField<5>;
    
    /// Export to QuorumField via measurement.
    pub fn to_quorum(&self) -> QuorumField;  // only for N=5
}

/// Compress to Crystal4K (works for any N, projects along 3 axes).
impl<const N: usize> QuantumField<N> {
    pub fn to_crystal4k(&self) -> Crystal4K;
    pub fn from_crystal4k(crystal: &Crystal4K) -> Self;  // expand + zero phase
}

/// Type aliases for common sizes
pub type QuantumField5 = QuantumField<5>;    // backward compat
pub type QuantumField7 = QuantumField<7>;    // prime sweet spot  
pub type QuantumField11 = QuantumField<11>;  // maximum practical
```

---

## THE INTERFERENCE STEP: DETAILED IMPLEMENTATION

This is the most important function. Get this right and everything works.

```rust
pub fn interfere(&mut self) -> usize {
    let n_cells = Self::num_cells();
    let mut new_cells = self.cells.clone();
    let mut changed = 0;
    
    for i in 0..n_cells {
        let mut net_constructive: f64 = 0.0;
        let mut net_destructive: f64 = 0.0;
        let mut phase_accumulator = PhaseTag::zero();
        let mut total_weight: f64 = 0.0;
        
        for j in 0..n_cells {
            if i == j { continue; }
            
            // Similarity between amplitude fingerprints
            let sim = self.cells[i].amplitude.similarity(&self.cells[j].amplitude) as f64;
            
            // Phase relationship: cos(phase_difference)
            let phase_cos = self.cells[i].phase.cos_angle_to(&self.cells[j].phase) as f64;
            
            // Signed contribution = similarity × cos(phase_angle)
            let contribution = sim * phase_cos;
            
            if contribution > 0.0 {
                net_constructive += contribution;
            } else {
                net_destructive += contribution;  // negative
            }
            
            total_weight += sim.abs();
            
            // Phase propagation: accumulate phase from strong contributors
            if sim > 0.5 {
                phase_accumulator = phase_accumulator.combine(&self.cells[j].phase);
            }
        }
        
        let net = (net_constructive + net_destructive) / total_weight.max(1.0);
        
        // Update amplitude based on net interference
        // Constructive: bind toward strongest contributor (gains bits)
        // Destructive: bind toward complement (loses bits → approaches zero)
        if net.abs() > 0.01 {  // dead zone prevents drift
            if net > 0.0 {
                // Constructive: reinforce. Shift amplitude toward denser states.
                // For each u64 word, probabilistically set bits proportional to net
                let prob = (net * 0.1).min(0.5);  // gentle update, max 50% of bits
                let current = new_cells[i].amplitude.clone();
                let shifted = shift_amplitude_up(&current, prob);
                new_cells[i].amplitude = shifted;
            } else {
                // Destructive: suppress. Shift amplitude toward sparser states.
                let prob = (net.abs() * 0.1).min(0.5);
                let current = new_cells[i].amplitude.clone();
                let shifted = shift_amplitude_down(&current, prob);
                new_cells[i].amplitude = shifted;
            }
            
            // Phase update: blend toward accumulated phase
            new_cells[i].phase = phase_accumulator;
            changed += 1;
        }
    }
    
    self.cells = new_cells;
    self.generation += 1;
    changed
}

/// Shift amplitude up by probabilistically setting some 0-bits to 1
fn shift_amplitude_up(fp: &Fingerprint, prob: f64) -> Fingerprint {
    let mut result = fp.clone();
    // Use a deterministic "random" based on existing bit pattern
    // XOR with permuted self to select which bits to flip
    let selector = fp.permute(7);  // shift by prime number
    for word in 0..FINGERPRINT_U64 {
        let zeros = !fp.as_raw()[word];  // bits that are currently 0
        let selected = zeros & selector.as_raw()[word];  // subset to flip
        // Only flip proportion ~prob of selected bits
        let mask = threshold_mask(selected, prob);
        result.as_raw_mut()[word] |= mask;
    }
    result
}

/// Shift amplitude down by probabilistically clearing some 1-bits to 0
fn shift_amplitude_down(fp: &Fingerprint, prob: f64) -> Fingerprint {
    // Mirror of shift_up
    let mut result = fp.clone();
    let selector = fp.permute(11);
    for word in 0..FINGERPRINT_U64 {
        let ones = fp.as_raw()[word];
        let selected = ones & selector.as_raw()[word];
        let mask = threshold_mask(selected, prob);
        result.as_raw_mut()[word] &= !mask;
    }
    result
}
```

**IMPORTANT**: The `shift_amplitude_up/down` functions need access to raw u64 mutation. Check if `Fingerprint` has `as_raw_mut()`. If not, construct new Fingerprint from modified `[u64; FINGERPRINT_U64]` array via `Fingerprint::from_raw()`.

---

## THE INTERFERENCE STEP: IMPLEMENTATION NOTES

The implementation above is conceptual. The actual implementation must:

1. **Use `Fingerprint::from_raw()`** to construct modified fingerprints (don't assume mutable raw access)
2. **Handle the dead zone** (net amplitude near zero → no change, prevents drift)
3. **Conserve total amplitude** approximately: track sum before and after, rescale if needed
4. **Be deterministic** for the same input (no thread-local RNG — use fingerprint bits as pseudo-random source)
5. **Handle edge case** where all cells have identical phase (fully coherent → no interference, just reinforcement)

The `threshold_mask` function should use the popcount of the selected word vs prob × 64 to decide how many bits to flip, using the bit pattern itself as the selection mask. No external RNG needed.

---

## TESTS (minimum 25)

### PhaseTag tests:
1. `zero()` has popcount 0, `pi()` has popcount 128
2. `cos_angle_to` between zero and zero = 1.0, zero and pi = -1.0
3. `combine` is associative and commutative
4. `from_angle(0.5)` has ~64 bits set, cos to zero ≈ 0.0

### QuantumCell tests:
5. `hadamard()` has magnitude ≈ 0.5, phase = zero
6. `signed_amplitude` positive for zero-phase, negative for pi-phase
7. `interference_to` positive for same-phase cells, negative for opposite
8. `bind` preserves magnitude, combines phase

### QuantumField interference tests:
9. All-zero field: interfere() changes nothing
10. All-hadamard field: interfere() maintains coherence
11. **GROVER TEST**: Plant target at (3,3,3). Mark with oracle. Interfere. Peak should be at (3,3,3) after √343 iterations
12. **DESTRUCTIVE TEST**: Two groups with opposite phase → amplitude should decrease
13. **CONSTRUCTIVE TEST**: All same phase → amplitude should increase
14. **CONSERVATION**: total_amplitude before ≈ total_amplitude after
15. Sparse interference gives same result as dense (within tolerance)

### QFT tests:
16. QFT of constant field → peak at frequency 0
17. QFT of periodic pattern (period 2) → peak at N/2
18. QFT followed by IQFT → identity (within noise)

### EntangledPair tests:
19. Bell state: measuring Alice constrains Bob (high correlation)
20. `bell_test` returns S > 2 for Bell state (quantum signature)
21. `bell_test` returns S ≤ 2 for classical (unentangled) state
22. Teleportation fidelity > 0.8 for Bell pair

### Conversion tests:
23. QuorumField → QuantumField<5> → QuorumField preserves structure
24. QuantumField → Crystal4K → QuantumField preserves projections
25. QuantumField<7> projects to same Crystal4K format as QuantumField<5>

### Performance tests:
26. `interfere()` on QuantumField<7> completes in < 50ms
27. `interfere_sparse(0.3)` is at least 3× faster than dense

---

## FILE STRUCTURE

```rust
// src/extensions/hologram/quantum_field.rs

//! Quantum Field: True quantum interference on crystal substrate.
//!
//! Replaces classical quorum voting with signed-amplitude interference.
//! Each cell carries a 10K-bit amplitude fingerprint + 128-bit phase tag.
//! Evolution via constructive/destructive interference, not majority rule.
//!
//! Key properties:
//! - Destructive interference: opposing phases cancel (7.6× suppression)
//! - Born rule: popcount/N naturally gives |α|² probabilities  
//! - Phase composability: XOR of phase tags = phase addition
//! - Const generic: works for any N (5, 7, 11, ...)
//! - Full connectivity: every cell interferes with every other
//!
//! Memory: N³ × (1250 + 16) bytes = N³ × 1266 bytes
//!   5×5×5: 158KB    (fits L1 cache)
//!   7×7×7: 434KB    (fits L1 cache on modern CPUs)
//!   11×11×11: 1.7MB (fits L2 cache)

use crate::core::Fingerprint;
use crate::{FINGERPRINT_BITS, FINGERPRINT_U64};
use super::field::QuorumField;
use super::crystal4k::Crystal4K;
```

## REGISTRATION

Update `src/extensions/hologram/mod.rs`:
```rust
mod crystal4k;
mod field;
mod memory;
pub mod quantum_crystal;     // primitive operations
pub mod quantum_algorithms;  // composed algorithms
pub mod quantum_field;       // TRUE quantum substrate    <-- ADD THIS

pub use crystal4k::*;
pub use field::*;
pub use memory::*;
pub use quantum_crystal::*;
pub use quantum_algorithms::*;
pub use quantum_field::*;    // <-- ADD THIS
```

---

## CONSTRAINTS

1. **Pure Rust only**. No external crates.
2. **No unsafe code**. Construct Fingerprints via `from_raw()`.
3. **Const generics** for field dimension. Must compile for N=5, 7, 11.
4. **25+ tests** covering interference, QFT, entanglement, conversion.
5. **Feature-gated under `hologram`**.
6. **All existing tests still pass**: `cargo test --features "spo,quantum,codebook,hologram"`
7. **No modification to existing types**. Fingerprint, QuorumField, Crystal4K unchanged.
8. **Deterministic**: same input → same output. Use fingerprint bits as pseudo-random source, not thread RNG.
9. **Performance**: `interfere()` on QuantumField<7> must complete in < 100ms.

## COMMIT

```bash
git add -A
git commit -m "feat: true quantum interference field with phase tags

QuantumField<N>: const-generic quantum substrate replacing quorum voting.

Key innovation: 128-bit phase tags per cell enable signed-amplitude
interference. Destructive interference suppresses wrong answers by 7.6×
(proven via Born rule on 10K-bit fingerprints).

Types:
- PhaseTag: 128-bit XOR-composable phase angle
- QuantumCell: Fingerprint amplitude + PhaseTag phase
- QuantumField<N>: N×N×N interference lattice (N=5,7,11)
- EntangledPair<N>: Bell states, teleportation, CHSH test

Evolution via signed-amplitude interference (NOT majority voting):
- contribution = similarity × cos(phase_difference)  
- Constructive: aligned phases reinforce
- Destructive: opposing phases cancel
- Born rule: measurement via popcount gives |α|² naturally

QuantumField<7>: 343 cells, prime dimension, L1-cache-resident.
Full connectivity: every cell interferes with every other.
~18ms per interference step.

25+ new tests including Bell inequality violation."

git push origin HEAD:refs/heads/feature/quantum-field
```
