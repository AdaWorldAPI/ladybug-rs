# QUANTUM CRYSTAL ORCHESTRATOR
## Session Goal: Complete Quantum Operation Set on 5×5×5 Crystal Lattice

### Context
You are working on `ladybug-rs`, a pure-Rust cognitive database with 10,000-bit binary fingerprints. The repository is at `/home/claude/ladybug-rs` (also https://github.com/AdaWorldAPI/ladybug-rs).

The project already has:
- **quantum_ops.rs** (1,086 lines): QuantumOp trait, Identity, Not, Bind, Permute, Project, Hadamard, Measure, TimeEvolution, NARS/ACT-R/RL/Causal/Qualia/Rung cognitive operators, ComposedOp, SumOp, TensorOp, commutator, expectation, variance
- **hologram/field.rs**: `QuorumField` — 5×5×5 lattice of 10K-bit fingerprints, quorum voting, settle(), project_x/y/z
- **hologram/crystal4k.rs**: `Crystal4K` — 41:1 compression via XOR axis projection, expand(), signature(), distance()
- **hologram/memory.rs**: `CrystalMemory` — 43K crystals, route → expand → settle → compress, Hebbian learning

Feature flags: `quantum`, `hologram` (both in `all` feature set)
Test command: `cargo test --features "spo,quantum,codebook,hologram"`
Current test count: ~222 tests

### YOUR TASK

Create a new file: `src/extensions/hologram/quantum_crystal.rs`

This file implements 9 quantum operations that work on the QuorumField/Crystal4K lattice. Each operation must:
1. Operate on `QuorumField` (5×5×5 × 10K-bit cells)
2. Be expressible as `Crystal4K` → `Crystal4K` transformations
3. Have thorough tests proving correctness
4. Use only existing types from `crate::core::Fingerprint`, `super::field::QuorumField`, `super::crystal4k::Crystal4K`

---

## THE 9 OPERATIONS TO IMPLEMENT

### 1. Spatial Entanglement (CNOT on lattice)
```rust
/// Entangle two cells: if control cell's projection onto `basis` has popcount > threshold,
/// XOR target cell with `gate_mask`. This creates correlated cell pairs where
/// measuring one constrains the other.
pub fn entangle_cells(
    field: &mut QuorumField,
    control: (usize, usize, usize),  // control cell coordinates
    target: (usize, usize, usize),   // target cell coordinates  
    basis: &Fingerprint,             // projection basis for control
    gate_mask: &Fingerprint,         // XOR mask applied to target
    threshold: u32,                  // popcount threshold (default: 5000 = half)
)
```
**Test**: Entangle (0,0,0) with (4,4,4). Modify control. Verify target correlation changes. Verify non-entangled cells are unaffected. Verify self-inverse property (apply twice = identity).

### 2. Quantum Fourier Transform along axis
```rust
/// QFT along one axis of the crystal. Converts spatial patterns to frequency patterns.
/// Each slice perpendicular to the axis gets permuted by 2^position before XOR-fold.
/// Reversible: apply twice with inverse permutation direction.
pub fn qft_axis(field: &mut QuorumField, axis: Axis) -> Fingerprint
// where Axis is:
pub enum Axis { X, Y, Z }
```
**Implementation**: For axis X, iterate x=0..5. For each x, extract the 5×5 YZ plane, permute all 25 cells by `2^x` positions, then XOR-fold the plane into a single fingerprint. Return the XOR of all 5 folded planes.
**Test**: Inject a periodic pattern (same fingerprint at x=0,2,4; different at x=1,3). QFT should show peak at the period-2 frequency. Verify inverse QFT recovers original.

### 3. Phase Kickback
```rust
/// Apply an operator to the field and extract the eigenvalue as Hamming distance
/// between input-face and output-face projections.
/// Returns (eigenvalue_estimate, kicked_field).
pub fn phase_kickback(
    field: &mut QuorumField,
    operator: &dyn Fn(&Fingerprint) -> Fingerprint,
    axis: Axis,
) -> (f32, Fingerprint)
```
**Implementation**: Save projection along axis before. Apply operator to every cell. Compute projection after. Eigenvalue = Hamming distance / FINGERPRINT_BITS. Return the delta fingerprint (before XOR after).
**Test**: Use identity operator → eigenvalue ≈ 0. Use NOT operator → eigenvalue ≈ 1.0. Use BindOp with known key → eigenvalue between 0 and 1.

### 4. Coherence Tracking & Decoherence
```rust
/// A crystal with coherence tracking. Coherence decreases each tick
/// proportional to field change. Below threshold, forces collapse.
pub struct CoherentCrystal {
    pub field: QuorumField,
    pub coherence: f32,           // 1.0 = fully coherent, 0.0 = fully decohered
    pub decoherence_rate: f32,    // how fast coherence drops per changed-bit-fraction
    pub collapse_threshold: f32,  // below this, force collapse to nearest attractor
}

impl CoherentCrystal {
    pub fn new(field: QuorumField) -> Self;
    pub fn tick(&mut self) -> bool;  // returns true if collapsed
    pub fn inject(&mut self, pos: (usize,usize,usize), fp: &Fingerprint);
    pub fn coherence(&self) -> f32;
    /// Force collapse: each cell snaps to nearest in codebook (if provided)
    /// or to majority-vote of neighbors
    pub fn collapse(&mut self);
}
```
**Test**: Create field with mixed states. Tick repeatedly. Verify coherence decreases. Verify collapse happens at threshold. Verify post-collapse field is more stable (fewer changes per tick).

### 5. Surface Code Error Correction (5×5 face)
```rust
/// Detect and correct single-cell errors on a face of the crystal
/// using parity checks with adjacent cells (surface code).
/// Returns number of corrections applied.
pub fn surface_code_correct(field: &mut QuorumField, face: Face) -> usize
// where Face is:
pub enum Face { XY0, XY4, XZ0, XZ4, YZ0, YZ4 }  // 6 faces of the cube
```
**Implementation**: For each cell on the face, compute XOR of all adjacent cells (syndrome). If syndrome popcount > threshold, the center cell is likely corrupted. Correct by replacing with majority-vote of neighbors. Only correct cells where syndrome is strong (popcount > 7000 out of 10000).
**Test**: Create uniform face. Corrupt one cell. Run correction. Verify corrupted cell is restored. Verify uncorrupted cells unchanged. Verify double-error detection (detect but don't correct).

### 6. Quantum Walk Step
```rust
/// One step of a quantum walk on the lattice.
/// Unlike classical settle (majority vote), this uses interference:
/// each cell receives popcount-weighted contributions from neighbors,
/// with constructive/destructive interference.
/// Returns number of cells that changed.
pub fn quantum_walk_step(field: &mut QuorumField) -> usize
```
**Implementation**: For each cell, compute weighted sum: for each of 6 neighbors, weight = neighbor.popcount() / FINGERPRINT_BITS. For each bit position, sum weights of neighbors that have bit=1. Set bit to 1 if sum > π/4 (≈0.785). This is NOT majority vote — it's interference with a quantum-threshold.
**Test**: Place a "walker" fingerprint at (2,2,2) in an otherwise empty field. Run quantum_walk_step. Verify spread is different from classical settle (test both, compare). Verify walker at center has non-zero overlap with original after 3 steps.

### 7. Adiabatic Evolution
```rust
/// Slowly evolve the field from easy ground state to complex target
/// by linearly interpolating the quorum threshold.
/// Returns (final_field_signature, steps_taken, converged).
pub fn adiabatic_evolve(
    field: &mut QuorumField,
    start_threshold: u8,    // typically 1 (fluid)
    end_threshold: u8,      // typically 4-5 (rigid)
    total_steps: usize,     // how many ticks for the interpolation
) -> (Fingerprint, usize, bool)
```
**Implementation**: For each step, set threshold = start + (end-start) * step/total_steps (rounded). Tick. If field hasn't changed for 3 consecutive ticks at the same threshold, advance. Return signature of final state.
**Test**: Start with random field, threshold 1. Adiabatically tighten to 5. Verify final state is stable. Compare to direct settle at threshold 5 (adiabatic should find better attractor).

### 8. Density Matrix (Mixed State Cells)  
```rust
/// A cell that holds multiple possible states with probabilities.
/// Enables reasoning under uncertainty.
pub struct MixedCell {
    pub states: Vec<(Fingerprint, f32)>,  // (state, probability)
}

impl MixedCell {
    pub fn pure(fp: Fingerprint) -> Self;
    pub fn mixed(states: Vec<(Fingerprint, f32)>) -> Self;
    pub fn measure(&self) -> Fingerprint;  // collapse: pick state proportional to probability
    pub fn entropy(&self) -> f32;           // Shannon entropy of the mixture
    pub fn expected(&self) -> Fingerprint;  // weighted bundle of all states
    pub fn purity(&self) -> f32;            // 1.0 = pure, 0.0 = maximally mixed
}

/// Field with mixed-state cells
pub struct MixedField {
    cells: [[[MixedCell; 5]; 5]; 5],
}

impl MixedField {
    pub fn from_pure(field: &QuorumField) -> Self;
    pub fn collapse_all(&self) -> QuorumField;  // measure every cell
    pub fn tick(&mut self);  // evolve: each cell's states interact with neighbor states
}
```
**Test**: Create MixedCell with 2 states (60%/40%). Measure 1000 times, verify ratio. Create MixedField, evolve, verify entropy decreases (uncertainty resolves).

### 9. State Teleportation
```rust
/// Teleport a cell's state from source crystal to target crystal
/// using a pre-shared entangled crystal pair.
/// Returns correction bits (much smaller than full cell).
pub fn teleport_prepare(
    source_cell: &Fingerprint,
    shared_half_a: &Fingerprint,  // Alice's half of entangled pair
) -> TeleportPacket

pub fn teleport_receive(
    packet: &TeleportPacket,
    shared_half_b: &Fingerprint,  // Bob's half of entangled pair
) -> Fingerprint

pub struct TeleportPacket {
    /// Correction bits: XOR of (source ⊕ shared_half_a) 
    /// This is ~1.25KB but compresses well (typically <500 bytes with RLE)
    pub corrections: Fingerprint,
}

/// Teleport entire crystal (all 125 cells)
pub fn teleport_crystal(
    source: &Crystal4K,
    shared_a: &Crystal4K,
    shared_b: &Crystal4K,
) -> Crystal4K
```
**Implementation**: Alice computes corrections = source XOR shared_a. Sends corrections (Fingerprint = 1.25KB). Bob computes result = corrections XOR shared_b. If shared_a and shared_b were created as entangled pair (shared_a XOR shared_b = known_key), then result = source XOR known_key. Bob unbinds known_key to recover source.
**Test**: Create entangled pair. Teleport a fingerprint. Verify received == original. Teleport a full crystal. Verify all 3 axes match. Verify packet size < full crystal size.

---

## FILE STRUCTURE

```rust
// src/extensions/hologram/quantum_crystal.rs

//! Quantum Crystal: Complete quantum operation set on 5×5×5 Hamming lattice.
//!
//! 9 operations mapping quantum computing primitives onto the crystal:
//! 1. Spatial entanglement (CNOT between cells)
//! 2. Quantum Fourier Transform (along crystal axes)
//! 3. Phase kickback (eigenvalue extraction)
//! 4. Coherence tracking & decoherence
//! 5. Surface code error correction
//! 6. Quantum walk (interference-based evolution)
//! 7. Adiabatic evolution (threshold interpolation)
//! 8. Density matrix (mixed-state cells)
//! 9. State teleportation (correction-bit transfer)
//!
//! Key insight: the 5×5×5 × 10K-bit lattice IS the register file.
//! Quorum dynamics ARE the gates. Projections ARE the measurements.
//! No simulation of qubits — direct operation on the native substrate.

use crate::core::Fingerprint;
use crate::FINGERPRINT_U64;
use crate::FINGERPRINT_BITS;
use super::field::{QuorumField, FIELD_SIZE};
use super::crystal4k::Crystal4K;
```

## REGISTRATION

After creating the file, update `src/extensions/hologram/mod.rs`:
```rust
//! Hologram Extension - 4KB Holographic Crystals with Quorum ECC
//! 5×5×5 quorum fields, any 2-of-3 copies can reconstruct
//! Quantum crystal operations for complete quantum gate set

mod crystal4k;
mod field;
mod memory;
pub mod quantum_crystal;   // <-- ADD THIS

pub use crystal4k::*;
pub use field::*;
pub use memory::*;
pub use quantum_crystal::*;  // <-- ADD THIS
```

---

## CONSTRAINTS

1. **Pure Rust only**. No external crates beyond what's already in Cargo.toml.
2. **No unsafe code**. All operations through safe Fingerprint/QuorumField APIs.
3. **Every operation must have at least 2 tests**. Target: 20+ new tests total.
4. **Use `super::field::FIELD_SIZE`** (which is 5), never hardcode 5.
5. **Feature-gated under `hologram`** — the file is inside the hologram module which already has the gate.
6. **All tests must pass**: `cargo test --features "spo,quantum,codebook,hologram"`
7. **Respect existing API**: use `field.get(x,y,z)`, `field.set(x,y,z, &fp)`, `field.tick()`, `field.settle()`, `field.project_x/y/z()`, `field.signature()`, `field.set_threshold()`, `crystal.signature()`, `crystal.xor()`, `crystal.from_field()`, `fp.bind()`, `fp.permute()`, `fp.hamming()`, `fp.similarity()`, `fp.popcount()`, `fp.get_bit()`, `fp.set_bit()`, `fp.not()`, `fp.and()`, `fp.or()`, `Fingerprint::from_content()`, `Fingerprint::random()`, `Fingerprint::zero()`, `Fingerprint::orthogonal()`

## QUALITY CHECKS

After implementation:
1. `cargo test --features "spo,quantum,codebook,hologram"` — ALL tests pass
2. `cargo clippy --features "spo,quantum,codebook,hologram"` — no warnings
3. Verify each operation is documented with `///` doc comments
4. Verify the module is properly exported and usable from outside

## COMMIT

When done:
```bash
git add -A
git commit -m "feat: complete quantum operation set on 5×5×5 crystal lattice

Implements 9 quantum operations on the holographic crystal:
- Spatial entanglement (CNOT between lattice cells)
- Quantum Fourier Transform along crystal axes
- Phase kickback for eigenvalue extraction
- Coherence tracking with natural decoherence
- Surface code error correction on 5×5 faces
- Quantum walk with interference dynamics
- Adiabatic evolution via threshold interpolation
- Density matrix mixed-state cells
- State teleportation via correction-bit packets

Each operation maps directly onto the 5×5×5 × 10K-bit substrate
without simulating qubits — native Hamming-space quantum analogues.

~390 lines, 20+ new tests."
```

Then push:
```bash
git push origin HEAD:refs/heads/feature/quantum-crystal
```
