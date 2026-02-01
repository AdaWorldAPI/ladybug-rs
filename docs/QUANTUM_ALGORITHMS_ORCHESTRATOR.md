# QUANTUM ALGORITHMS ORCHESTRATOR
## Session Goal: Complete Quantum Algorithm Suite on Crystal Substrate

### Prerequisites
This file depends on `quantum_crystal.rs` (from QUANTUM_CRYSTAL_ORCHESTRATOR.md) being implemented first. That module provides the 9 primitive operations (entanglement, QFT, phase kickback, decoherence, surface code, quantum walk, adiabatic evolution, density matrix, teleportation).

This module composes those primitives into complete quantum algorithms.

### Context
You are working on `ladybug-rs` at `/home/claude/ladybug-rs`.

Key APIs available (after quantum_crystal.rs exists):
- `QuorumField`: 5×5×5 lattice, get/set/tick/settle/project_x/y/z/signature/set_threshold/randomize/clear
- `Crystal4K`: 41:1 compressed crystal, from_field/expand/expand_clean/xor/distance/similarity
- `CrystalMemory`: 43K crystal store, route/infer/learn
- `Fingerprint`: 10K-bit vectors, bind/unbind/permute/hamming/similarity/popcount/get_bit/set_bit/not/and/or/from_content/random/zero/ones/orthogonal
- `QuantumOp` trait: apply(&self, &Fingerprint) -> Fingerprint + name()
- Existing operators: IdentityOp, NotOp, BindOp, PermuteOp, ProjectOp, HadamardOp, MeasureOp, TimeEvolutionOp
- From quantum_crystal.rs: entangle_cells, qft_axis, phase_kickback, CoherentCrystal, surface_code_correct, quantum_walk_step, adiabatic_evolve, MixedCell, MixedField, teleport_prepare/receive, Axis, Face
- `bundle_fingerprints(fps: &[Fingerprint]) -> Fingerprint` (majority vote)
- `FINGERPRINT_BITS` = 10000, `FINGERPRINT_U64` = 157, `FIELD_SIZE` = 5

### YOUR TASK

Create a new file: `src/extensions/hologram/quantum_algorithms.rs`

This file implements 13 quantum algorithms as crystal-native operations. Each builds on the primitives from quantum_ops.rs and quantum_crystal.rs.

---

## THE 13 ALGORITHMS

### 1. Grover's Search
```rust
/// Grover's algorithm on the crystal: find a target fingerprint in a field
/// in O(√N) steps instead of O(N).
///
/// Uses quantum_walk_step for diffusion and oracle marking for phase inversion.
/// On a 125-cell crystal, optimal iterations = ⌊π/4 × √125⌋ ≈ 8.
pub struct GroverSearch {
    pub oracle: Fingerprint,    // the target we're searching for
    pub threshold: f32,         // similarity threshold to declare "found"
}

impl GroverSearch {
    pub fn new(target: Fingerprint, threshold: f32) -> Self;

    /// Run Grover's on a field. Returns (found_position, similarity, iterations).
    /// Algorithm:
    /// 1. Initialize field with uniform superposition (Hadamard each cell)
    /// 2. For each iteration:
    ///    a. Oracle: XOR-mark cells similar to target (similarity > threshold → bind with marker)
    ///    b. Diffusion: quantum_walk_step (interference amplifies marked cells)
    /// 3. Measure: find cell with highest similarity to target
    pub fn search(&self, field: &mut QuorumField, max_iter: usize) -> GroverResult;
}

pub struct GroverResult {
    pub position: (usize, usize, usize),
    pub similarity: f32,
    pub iterations: usize,
    pub found: bool,
}
```
**Test**: Plant a known fingerprint at random position. Run Grover. Verify it finds the correct position. Compare iteration count to classical linear scan. Verify O(√N) behavior: run on crystal with target and count iterations vs checking all 125 cells.

### 2. Shor's Period Finding
```rust
/// Shor's period-finding subroutine on crystal.
/// Uses QFT to detect periodicity in a fingerprint sequence.
///
/// Given a function f: index → Fingerprint that is periodic (f(x) = f(x+r)),
/// find the period r.
pub struct ShorPeriodFinder;

impl ShorPeriodFinder {
    /// Find the period of a function mapped onto the crystal.
    /// 1. Fill field along X axis with f(0), f(1), ..., f(4)
    /// 2. Apply qft_axis(X)
    /// 3. Measure: peaks in QFT output reveal period
    /// Returns estimated period (1-5 for a 5-wide crystal).
    pub fn find_period(
        f: &dyn Fn(usize) -> Fingerprint,
        field: &mut QuorumField,
    ) -> PeriodResult;
}

pub struct PeriodResult {
    pub estimated_period: usize,
    pub confidence: f32,       // how peaked the QFT output is
    pub qft_signature: Fingerprint,
}
```
**Test**: Create function with period 2 (f(0)=A, f(1)=B, f(2)=A, f(3)=B, f(4)=A). Verify estimated_period=2. Test period 1 (constant), period 5 (all different). Verify confidence is highest for period-2 case.

### 3. Deutsch-Jozsa Oracle Classification
```rust
/// Deutsch-Jozsa on crystal: determine if a function is constant or balanced.
/// A function f: Fingerprint → {0,1} is:
///   - constant: returns same value for all inputs
///   - balanced: returns 0 for exactly half, 1 for half
///
/// Classical: need N/2+1 queries worst case.
/// Quantum: 1 query via superposition + interference.
pub struct DeutschJozsa;

impl DeutschJozsa {
    /// Classify an oracle function.
    /// 1. Fill all 125 cells with Hadamard (superposition)
    /// 2. Apply oracle: for each cell, if f(cell) = 1, bind with phase_marker
    /// 3. Apply QFT across all three axes
    /// 4. Measure signature: if all-zero → constant, else → balanced
    pub fn classify(
        oracle: &dyn Fn(&Fingerprint) -> bool,
        field: &mut QuorumField,
    ) -> OracleClass;
}

pub enum OracleClass {
    Constant,
    Balanced,
    Unknown(f32),  // confidence that it's balanced (0.0 = likely constant, 1.0 = likely balanced)
}
```
**Test**: Constant oracle (always true) → Constant. Balanced oracle (popcount > 5000 → true) → Balanced. Verify single "query" (one field evolution) suffices.

### 4. Bernstein-Vazirani Hidden String
```rust
/// Bernstein-Vazirani: extract a hidden bitstring s from an oracle f(x) = s·x mod 2.
/// In Hamming space: f(x) = popcount(x AND s) mod 2.
///
/// Classical: need FINGERPRINT_BITS queries.
/// Quantum on crystal: single pass.
pub struct BernsteinVazirani;

impl BernsteinVazirani {
    /// Extract the hidden string.
    /// 1. Fill field with Hadamard states
    /// 2. Apply oracle: mark cells where f(cell) = 1
    /// 3. Apply QFT
    /// 4. Measure: result IS the hidden string s
    pub fn extract(
        oracle: &dyn Fn(&Fingerprint) -> bool,
        field: &mut QuorumField,
    ) -> Fingerprint;  // the recovered hidden string
}
```
**Test**: Choose random hidden string s. Create oracle f(x) = popcount(x AND s) % 2. Extract. Verify recovered string has high similarity (>0.7) to s. Not exact due to 125-cell approximation.

### 5. Simon's Period Finding (XOR variant)
```rust
/// Simon's algorithm: find hidden period s where f(x) = f(x ⊕ s).
/// In fingerprint space: XOR-period detection.
///
/// Different from Shor's: this finds XOR-shift symmetry, not additive period.
pub struct SimonPeriod;

impl SimonPeriod {
    /// Find hidden XOR period.
    /// 1. Fill field with pairs: for each cell, compute f(cell)
    /// 2. Look for cells where f(cell_a) == f(cell_b) (high similarity)
    /// 3. Hidden period s = cell_a XOR cell_b for matched pairs
    /// 4. Bundle all recovered s values for consensus
    pub fn find_xor_period(
        f: &dyn Fn(&Fingerprint) -> Fingerprint,
        field: &mut QuorumField,
    ) -> SimonResult;
}

pub struct SimonResult {
    pub hidden_period: Fingerprint,
    pub confidence: f32,
    pub matched_pairs: usize,
}
```
**Test**: Choose random s. Define f(x) = if popcount(x) % 2 == 0 { x } else { x.bind(&s) }. Run Simon's. Verify hidden_period similar to s.

### 6. QAOA (Quantum Approximate Optimization)
```rust
/// QAOA on crystal: solve combinatorial optimization by alternating
/// problem Hamiltonian (diagonal in computational basis) and
/// mixer Hamiltonian (off-diagonal transitions).
///
/// Problem: maximize similarity to a target fingerprint across the field.
/// Mixer: quantum_walk_step (off-diagonal mixing).
/// Problem: bind cells similar to target (diagonal phase marking).
pub struct Qaoa {
    pub target: Fingerprint,
    pub layers: usize,          // p parameter: alternation depth
    pub gamma: Vec<f32>,        // problem angles (one per layer)
    pub beta: Vec<f32>,         // mixer angles (one per layer)
}

impl Qaoa {
    pub fn new(target: Fingerprint, layers: usize) -> Self;

    /// Run QAOA.
    /// For each layer p:
    ///   1. Problem phase: for each cell, if similarity(cell, target) > gamma[p],
    ///      XOR cell with target (rotate toward solution)
    ///   2. Mixer phase: apply quantum_walk_step beta[p] times
    ///      (beta controls mixing strength)
    /// Returns best cell found and its similarity.
    pub fn optimize(&self, field: &mut QuorumField) -> QaoaResult;

    /// Recursive QAOA: start with layers=1, increase until improvement < threshold.
    pub fn optimize_recursive(
        target: &Fingerprint,
        field: &mut QuorumField,
        max_layers: usize,
    ) -> QaoaResult;
}

pub struct QaoaResult {
    pub best_position: (usize, usize, usize),
    pub best_similarity: f32,
    pub layers_used: usize,
    pub field_energy: f32,  // average similarity across all cells
}
```
**Test**: Create random target. Run QAOA with 1, 3, 5 layers. Verify similarity improves with more layers. Compare to raw settle() — QAOA should find better solutions for complex targets. Test recursive variant converges.

### 7. VQE (Variational Quantum Eigensolver)
```rust
/// VQE on crystal: find the ground state (lowest energy configuration) of a
/// Hamiltonian defined by a target fingerprint.
///
/// Energy = average Hamming distance from all cells to target.
/// Variational parameter: quorum threshold (1-6).
/// Quantum circuit: settle at given threshold.
/// Classical optimizer: grid search over threshold.
pub struct Vqe {
    pub hamiltonian: Fingerprint,  // defines the energy landscape
}

impl Vqe {
    pub fn new(hamiltonian: Fingerprint) -> Self;

    /// Run VQE.
    /// For each threshold in 1..=6:
    ///   1. Reset field to initial state
    ///   2. Settle at this threshold
    ///   3. Compute energy = Σ hamming(cell, hamiltonian) for all cells
    /// Return configuration with lowest energy.
    pub fn find_ground_state(
        &self,
        initial: &QuorumField,
    ) -> VqeResult;

    /// Extended VQE: also vary initial injection patterns.
    pub fn find_ground_state_extended(
        &self,
        seeds: &[Fingerprint],
    ) -> VqeResult;
}

pub struct VqeResult {
    pub ground_state: Crystal4K,
    pub energy: f32,
    pub optimal_threshold: u8,
    pub iterations: usize,
}
```
**Test**: Create Hamiltonian. Run VQE. Verify energy is minimized. Compare to random field energy — VQE should be significantly lower. Test extended variant with multiple seeds finds better minimum.

### 8. Quantum Counting
```rust
/// Quantum Counting: estimate how many cells in the field match a predicate
/// without inspecting each cell individually.
///
/// Combines Grover oracle marking with QFT-based estimation.
pub struct QuantumCounting;

impl QuantumCounting {
    /// Estimate count of cells matching predicate.
    /// 1. Run Grover iterations, recording field signature at each step
    /// 2. Apply QFT to the signature sequence
    /// 3. Peak position in QFT ↔ fraction of matching cells
    /// 4. Count estimate = 125 × (1 - cos²(π × peak / sequence_length))
    pub fn count(
        predicate: &dyn Fn(&Fingerprint) -> bool,
        field: &mut QuorumField,
        grover_iterations: usize,
    ) -> CountResult;
}

pub struct CountResult {
    pub estimated_count: usize,
    pub actual_count: usize,   // for verification: direct classical count
    pub relative_error: f32,
}
```
**Test**: Plant exactly K matching cells (K=10, 30, 60). Run counting. Verify estimated_count is within 30% of K. Verify it uses fewer queries than exhaustive inspection.

### 9. QSVT (Quantum Singular Value Transformation)
```rust
/// QSVT framework: apply a polynomial transformation to the singular values
/// of a crystal-encoded matrix.
///
/// The crystal's X and Z projections define a "matrix" M:
///   M = project_x() ⊗ project_z()^T (outer product in Hamming space)
/// Singular values ≈ per-bit correlations between X and Z projections.
///
/// QSVT applies polynomial f(σ) to each singular value without decomposition.
pub struct Qsvt {
    pub polynomial_coefficients: Vec<f32>,  // coefficients of the polynomial
}

impl Qsvt {
    pub fn new(coefficients: Vec<f32>) -> Self;

    /// Apply polynomial transformation to crystal's singular values.
    /// 1. Extract X, Z projections
    /// 2. For each u64 word: compute correlation between X[i] and Z[i]
    /// 3. Apply polynomial: result[i] = Σ coeff[k] × correlation^k
    /// 4. Reconstruct transformed projections
    /// Returns transformed Crystal4K.
    pub fn transform(&self, crystal: &Crystal4K) -> Crystal4K;

    /// Convenience: threshold transformation (step function at σ_min).
    /// Useful for rank estimation: zero out weak singular values.
    pub fn threshold(crystal: &Crystal4K, sigma_min: f32) -> Crystal4K;

    /// Convenience: inversion (1/σ for σ > cutoff).
    /// This is the core of HHL.
    pub fn invert(crystal: &Crystal4K, cutoff: f32) -> Crystal4K;
}
```
**Test**: Create crystal with known structure. Apply identity polynomial [0, 1] → no change. Apply threshold → weaker correlations zeroed. Apply quadratic → verify expected transformation.

### 10. HHL (Linear System Solver)
```rust
/// HHL on crystal: solve Ax = b where:
///   A is encoded in crystal structure (field dynamics)
///   b is the input fingerprint
///   x is the solution fingerprint
///
/// Uses phase estimation + QSVT inversion.
pub struct Hhl;

impl Hhl {
    /// Solve the linear system encoded by the field dynamics.
    /// 1. Inject b into field at (0,0,0)
    /// 2. Apply phase_kickback to extract eigenvalues of field dynamics
    /// 3. Apply QSVT inversion to eigenvalues
    /// 4. Reconstruct solution from inverted eigenvalues
    ///
    /// In crystal terms: b goes in, field dynamics act as matrix A,
    /// inverse dynamics applied via adiabatic evolution, x comes out.
    pub fn solve(
        field: &mut QuorumField,
        b: &Fingerprint,
        cutoff: f32,  // regularization: don't invert eigenvalues below this
    ) -> HhlResult;
}

pub struct HhlResult {
    pub solution: Fingerprint,
    pub residual: f32,        // ||Ax - b|| / ||b|| estimated via field dynamics
    pub condition_number: f32, // ratio of largest to smallest eigenvalue
}
```
**Test**: Create field with known dynamics (simple rotation). Inject b. Solve. Apply field dynamics to solution. Verify result is close to b (residual < 0.3). Test with ill-conditioned system (high condition number) → verify regularization works.

### 11. Boson Sampling Analogue
```rust
/// Boson Sampling on crystal: simulate the permanent of a matrix,
/// which is #P-hard classically.
///
/// In crystal terms: inject N identical fingerprints (bosons) at N input ports
/// (cells on one face). Let them interfere through the crystal.
/// Output distribution at opposite face encodes the permanent.
pub struct BosonSampler {
    pub n_bosons: usize,       // number of identical particles
    pub input_face: Face,      // which face to inject
    pub output_face: Face,     // which face to measure
}

impl BosonSampler {
    pub fn new(n_bosons: usize) -> Self;

    /// Run boson sampling.
    /// 1. Inject identical fingerprint at n_bosons cells on input face
    /// 2. Run quantum_walk_step for depth steps (through the crystal)
    /// 3. Measure popcount distribution on output face
    /// 4. Distribution encodes |permanent|² of the transfer matrix
    /// Returns output distribution as popcount array.
    pub fn sample(
        &self,
        boson: &Fingerprint,
        field: &mut QuorumField,
        depth: usize,
    ) -> BosonResult;
}

pub struct BosonResult {
    pub output_distribution: Vec<(usize, usize, usize, u32)>, // (x,y,z, popcount)
    pub bunching_ratio: f32,  // how clustered the outputs are (bosons bunch)
    pub total_popcount: u64,
}
```
**Test**: Inject 3 identical bosons. Run sampling. Verify output shows bunching (bosons prefer same output modes) — bunching_ratio > random baseline. Verify different boson fingerprints give different distributions.

### 12. Quantum Simulation (Trotter Steps)
```rust
/// Quantum simulation via Trotterization.
/// Simulate time evolution under a Hamiltonian H = H1 + H2 + ... + Hk
/// by alternating small steps of each term.
///
/// In crystal terms: each Hi is a fingerprint mask. Evolution under Hi
/// means XOR-rotating cells toward Hi. Trotter formula alternates them.
pub struct TrotterSimulator {
    pub terms: Vec<Fingerprint>,    // Hamiltonian terms H1, H2, ...
    pub weights: Vec<f32>,          // coupling strengths
    pub dt: f32,                    // time step size
}

impl TrotterSimulator {
    pub fn new(terms: Vec<Fingerprint>, weights: Vec<f32>, dt: f32) -> Self;

    /// First-order Trotter: e^(-iHt) ≈ Π e^(-iH_k t)
    /// Apply each term sequentially for dt time.
    pub fn evolve_first_order(
        &self,
        field: &mut QuorumField,
        total_time: f32,
    ) -> SimulationResult;

    /// Second-order Trotter: e^(-iHt) ≈ e^(-iH_1 t/2) ... e^(-iH_k t) ... e^(-iH_1 t/2)
    /// More accurate: error ∝ dt³ instead of dt².
    pub fn evolve_second_order(
        &self,
        field: &mut QuorumField,
        total_time: f32,
    ) -> SimulationResult;
}

pub struct SimulationResult {
    pub final_state: Crystal4K,
    pub steps_taken: usize,
    pub energy_history: Vec<f32>,  // energy at each time step
}
```
**Implementation detail**: For each Trotter step, "evolving under term Hi with weight w for time dt" means:
- For each cell: compute projection onto Hi via AND(cell, Hi)
- fraction = projection.popcount() / Hi.popcount()
- If fraction × w × dt > threshold: XOR cell with Hi (rotate toward it)

**Test**: Single-term Hamiltonian → should be identical to TimeEvolutionOp. Two-term with commuting terms → first and second order should agree. Two-term with non-commuting terms → second order should be more accurate (closer to exact).

### 13. QNN (Quantum Neural Network)
```rust
/// Quantum Neural Network: parameterized crystal circuits for learning.
///
/// Each layer is: (1) entangle pairs, (2) apply parameterized rotations, (3) measure.
/// Parameters are learned by classical optimization of output similarity to target.
pub struct QuantumNeuralNet {
    pub layers: Vec<QnnLayer>,
}

pub struct QnnLayer {
    pub rotation_keys: Vec<Fingerprint>,   // one per cell (parameterized via bind)
    pub entangle_pairs: Vec<((usize,usize,usize), (usize,usize,usize))>,
}

impl QuantumNeuralNet {
    pub fn new(n_layers: usize) -> Self;
    pub fn random(n_layers: usize) -> Self;

    /// Forward pass: apply all layers to field.
    pub fn forward(&self, field: &mut QuorumField) -> Crystal4K;

    /// Compute loss: Hamming distance from output to target.
    pub fn loss(&self, field: &mut QuorumField, target: &Crystal4K) -> f32;

    /// Train: parameter-shift rule.
    /// For each parameter (rotation key), shift it left/right,
    /// compute loss difference → gradient.
    /// Update parameter by XOR-shifting toward gradient direction.
    /// Returns loss history.
    pub fn train(
        &mut self,
        inputs: &[QuorumField],
        targets: &[Crystal4K],
        epochs: usize,
        learning_rate: f32,
    ) -> Vec<f32>;
}
```
**Test**: Create identity task (output should match input). Train QNN for 10 epochs. Verify loss decreases. Create XOR task (output = input XOR key). Train. Verify QNN learns the key. Verify untrained QNN has high loss.

---

## CONTRACTED QUANTUM EIGENSOLVER (bonus — fold into VQE)

This is a VQE variant where only 2-body reduced density matrices are used. In crystal terms:
```rust
// Inside Vqe impl:
/// Contracted VQE: optimize using only pairwise cell correlations
/// instead of full field energy. More efficient for large systems.
pub fn find_ground_state_contracted(
    &self,
    initial: &QuorumField,
) -> VqeResult;
```
**Implementation**: Instead of computing energy over all 125 cells, sample pairs of adjacent cells, compute pairwise Hamming distance, use that as energy estimate. Reduces measurement cost from O(N) to O(edges).

---

## FILE STRUCTURE

```rust
// src/extensions/hologram/quantum_algorithms.rs

//! Quantum Algorithms: Complete algorithm suite on crystal substrate.
//!
//! 13 algorithms composed from quantum_crystal.rs primitives:
//!
//! Foundational:
//!   1. Grover's search (O(√N) concept finding)
//!   2. Shor's period finding (cognitive pattern periodicity)
//!   3. Deutsch-Jozsa (oracle classification)
//!   4. Bernstein-Vazirani (hidden structure extraction)
//!   5. Simon's algorithm (XOR-period detection)
//!
//! Optimization:
//!   6. QAOA (combinatorial optimization with mixing)
//!   7. VQE + Contracted VQE (ground state via threshold variation)
//!
//! Advanced:
//!   8. Quantum Counting (match count estimation)
//!   9. QSVT (polynomial singular value transformation)
//!  10. HHL (linear system solving)
//!  11. Boson Sampling (permanent computation analogue)
//!  12. Trotter Simulation (multi-Hamiltonian evolution)
//!  13. QNN (parameterized crystal circuits for learning)
//!
//! Every algorithm operates natively on the 5×5×5 × 10K-bit substrate.
//! No qubit simulation. Hamming space IS the computational space.

use crate::core::Fingerprint;
use crate::{FINGERPRINT_BITS, FINGERPRINT_U64};
use super::field::{QuorumField, FIELD_SIZE};
use super::crystal4k::Crystal4K;
use super::quantum_crystal::*;  // primitives from orchestrator 1
```

## REGISTRATION

Update `src/extensions/hologram/mod.rs`:
```rust
//! Hologram Extension - 4KB Holographic Crystals with Quorum ECC
//! 5×5×5 quorum fields, quantum crystal operations, complete algorithm suite

mod crystal4k;
mod field;
mod memory;
pub mod quantum_crystal;
pub mod quantum_algorithms;   // <-- ADD THIS

pub use crystal4k::*;
pub use field::*;
pub use memory::*;
pub use quantum_crystal::*;
pub use quantum_algorithms::*;  // <-- ADD THIS
```

---

## CONSTRAINTS

1. **Pure Rust only**. No external crates.
2. **No unsafe code**.
3. **Every algorithm must have at least 2 tests**. Target: 30+ new tests.
4. **Depends on quantum_crystal.rs** existing and working.
5. **Feature-gated under `hologram`**.
6. **All tests must pass**: `cargo test --features "spo,quantum,codebook,hologram"`
7. **Respect existing API** — don't modify existing files except mod.rs.
8. **Use deterministic seeds** in tests where randomness is involved (from_content("test_seed_N")).

## QUALITY CHECKS

After implementation:
1. `cargo test --features "spo,quantum,codebook,hologram"` — ALL tests pass
2. `cargo clippy --features "spo,quantum,codebook,hologram"` — no warnings
3. Verify each algorithm is documented with `///` doc comments
4. Verify no algorithm takes more than 5 seconds in tests (use small parameters)

## COMMIT

```bash
git add -A
git commit -m "feat: complete quantum algorithm suite on crystal substrate

13 quantum algorithms native to 5×5×5 Hamming lattice:
- Grover search: O(√125) concept finding on crystal
- Shor period finding: QFT-based cognitive periodicity detection
- Deutsch-Jozsa: single-query oracle classification
- Bernstein-Vazirani: hidden structure extraction
- Simon: XOR-period discovery
- QAOA + Recursive QAOA: combinatorial optimization
- VQE + Contracted VQE: ground state via threshold variation
- Quantum Counting: match estimation
- QSVT: polynomial singular value transformation
- HHL: linear system solver via phase estimation + inversion
- Boson Sampling: interference-based permanent analogue
- Trotter Simulation: first & second order multi-Hamiltonian
- QNN: parameterized crystal circuits with gradient training

Every algorithm composes primitives from quantum_crystal.rs.
No qubit simulation — native Hamming space operations.

~600 lines, 30+ new tests."

git push origin HEAD:refs/heads/feature/quantum-algorithms
```
