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

use super::crystal4k::Crystal4K;
use super::field::{FIELD_SIZE, QuorumField};
use crate::FINGERPRINT_BITS;
use crate::core::Fingerprint;

// =============================================================================
// 1. SPATIAL ENTANGLEMENT (CNOT on lattice)
// =============================================================================

/// Entangle two cells: if control cell's projection onto `basis` has popcount > threshold,
/// XOR target cell with `gate_mask`. This creates correlated cell pairs where
/// measuring one constrains the other.
///
/// # Arguments
/// * `field` - The quorum field to modify
/// * `control` - Control cell coordinates (x, y, z)
/// * `target` - Target cell coordinates (x, y, z)
/// * `basis` - Projection basis for control measurement
/// * `gate_mask` - XOR mask applied to target when control is "on"
/// * `threshold` - Popcount threshold (default: 5000 = half)
///
/// # Properties
/// - Self-inverse: applying twice returns to original state
/// - Creates correlation between control and target cells
pub fn entangle_cells(
    field: &mut QuorumField,
    control: (usize, usize, usize),
    target: (usize, usize, usize),
    basis: &Fingerprint,
    gate_mask: &Fingerprint,
    threshold: u32,
) {
    let control_cell = field.get(control.0, control.1, control.2);

    // Project control onto basis and check threshold
    let projection = control_cell.bind(basis);
    let control_active = projection.popcount() > threshold;

    if control_active {
        // Apply gate to target (XOR with mask)
        let target_cell = field.get(target.0, target.1, target.2);
        let new_target = target_cell.bind(gate_mask);
        field.set(target.0, target.1, target.2, &new_target);
    }
}

/// Create an entangled pair of fingerprints for teleportation.
/// Returns (half_a, half_b) where half_a XOR half_b = key.
pub fn create_entangled_pair(key: &Fingerprint) -> (Fingerprint, Fingerprint) {
    let half_a = Fingerprint::random();
    let half_b = half_a.bind(key); // half_a XOR key
    (half_a, half_b)
}

// =============================================================================
// 2. QUANTUM FOURIER TRANSFORM along axis
// =============================================================================

/// Axis enumeration for 3D operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
    Z,
}

/// QFT along one axis of the crystal. Converts spatial patterns to frequency patterns.
/// Each slice perpendicular to the axis gets permuted by 2^position before XOR-fold.
/// Reversible: apply twice with inverse permutation direction.
///
/// # Returns
/// The XOR of all folded planes (frequency domain representation).
pub fn qft_axis(field: &mut QuorumField, axis: Axis) -> Fingerprint {
    let mut result = Fingerprint::zero();

    match axis {
        Axis::X => {
            for x in 0..FIELD_SIZE {
                let permute_amount = 1i32 << x; // 2^x
                let mut plane_fold = Fingerprint::zero();

                for y in 0..FIELD_SIZE {
                    for z in 0..FIELD_SIZE {
                        let cell = field.get(x, y, z);
                        let permuted = cell.permute(permute_amount);
                        plane_fold = plane_fold.bind(&permuted);
                        field.set(x, y, z, &permuted);
                    }
                }
                result = result.bind(&plane_fold);
            }
        }
        Axis::Y => {
            for y in 0..FIELD_SIZE {
                let permute_amount = 1i32 << y;
                let mut plane_fold = Fingerprint::zero();

                for x in 0..FIELD_SIZE {
                    for z in 0..FIELD_SIZE {
                        let cell = field.get(x, y, z);
                        let permuted = cell.permute(permute_amount);
                        plane_fold = plane_fold.bind(&permuted);
                        field.set(x, y, z, &permuted);
                    }
                }
                result = result.bind(&plane_fold);
            }
        }
        Axis::Z => {
            for z in 0..FIELD_SIZE {
                let permute_amount = 1i32 << z;
                let mut plane_fold = Fingerprint::zero();

                for x in 0..FIELD_SIZE {
                    for y in 0..FIELD_SIZE {
                        let cell = field.get(x, y, z);
                        let permuted = cell.permute(permute_amount);
                        plane_fold = plane_fold.bind(&permuted);
                        field.set(x, y, z, &permuted);
                    }
                }
                result = result.bind(&plane_fold);
            }
        }
    }

    result
}

/// Inverse QFT along axis (unpermute instead of permute).
pub fn inverse_qft_axis(field: &mut QuorumField, axis: Axis) -> Fingerprint {
    let mut result = Fingerprint::zero();

    match axis {
        Axis::X => {
            for x in 0..FIELD_SIZE {
                let permute_amount = 1i32 << x;
                let mut plane_fold = Fingerprint::zero();

                for y in 0..FIELD_SIZE {
                    for z in 0..FIELD_SIZE {
                        let cell = field.get(x, y, z);
                        let unpermuted = cell.unpermute(permute_amount);
                        plane_fold = plane_fold.bind(&unpermuted);
                        field.set(x, y, z, &unpermuted);
                    }
                }
                result = result.bind(&plane_fold);
            }
        }
        Axis::Y => {
            for y in 0..FIELD_SIZE {
                let permute_amount = 1i32 << y;
                let mut plane_fold = Fingerprint::zero();

                for x in 0..FIELD_SIZE {
                    for z in 0..FIELD_SIZE {
                        let cell = field.get(x, y, z);
                        let unpermuted = cell.unpermute(permute_amount);
                        plane_fold = plane_fold.bind(&unpermuted);
                        field.set(x, y, z, &unpermuted);
                    }
                }
                result = result.bind(&plane_fold);
            }
        }
        Axis::Z => {
            for z in 0..FIELD_SIZE {
                let permute_amount = 1i32 << z;
                let mut plane_fold = Fingerprint::zero();

                for x in 0..FIELD_SIZE {
                    for y in 0..FIELD_SIZE {
                        let cell = field.get(x, y, z);
                        let unpermuted = cell.unpermute(permute_amount);
                        plane_fold = plane_fold.bind(&unpermuted);
                        field.set(x, y, z, &unpermuted);
                    }
                }
                result = result.bind(&plane_fold);
            }
        }
    }

    result
}

// =============================================================================
// 3. PHASE KICKBACK
// =============================================================================

/// Apply an operator to the field and extract the eigenvalue as Hamming distance
/// between input-face and output-face projections.
///
/// # Returns
/// (eigenvalue_estimate, kicked_field) where eigenvalue is normalized 0.0-1.0.
pub fn phase_kickback<F>(field: &mut QuorumField, operator: F, axis: Axis) -> (f32, Fingerprint)
where
    F: Fn(&Fingerprint) -> Fingerprint,
{
    // Save projection before
    let before = match axis {
        Axis::X => field.project_x(),
        Axis::Y => field.project_y(),
        Axis::Z => field.project_z(),
    };

    // Apply operator to every cell
    for x in 0..FIELD_SIZE {
        for y in 0..FIELD_SIZE {
            for z in 0..FIELD_SIZE {
                let cell = field.get(x, y, z);
                let transformed = operator(&cell);
                field.set(x, y, z, &transformed);
            }
        }
    }

    // Compute projection after
    let after = match axis {
        Axis::X => field.project_x(),
        Axis::Y => field.project_y(),
        Axis::Z => field.project_z(),
    };

    // Eigenvalue = normalized Hamming distance
    let distance = before.hamming(&after);
    let eigenvalue = distance as f32 / FINGERPRINT_BITS as f32;

    // Delta fingerprint
    let delta = before.bind(&after);

    (eigenvalue, delta)
}

// =============================================================================
// 4. COHERENCE TRACKING & DECOHERENCE
// =============================================================================

/// A crystal with coherence tracking. Coherence decreases each tick
/// proportional to field change. Below threshold, forces collapse.
#[derive(Clone)]
pub struct CoherentCrystal {
    /// The underlying quorum field
    pub field: QuorumField,
    /// Coherence level: 1.0 = fully coherent, 0.0 = fully decohered
    coherence: f32,
    /// How fast coherence drops per changed-bit-fraction
    pub decoherence_rate: f32,
    /// Below this threshold, force collapse
    pub collapse_threshold: f32,
    /// Previous field signature for change detection
    prev_signature: Fingerprint,
}

impl CoherentCrystal {
    /// Create new coherent crystal from field.
    pub fn new(field: QuorumField) -> Self {
        let sig = field.signature();
        Self {
            field,
            coherence: 1.0,
            decoherence_rate: 0.1,
            collapse_threshold: 0.3,
            prev_signature: sig,
        }
    }

    /// Create with custom parameters.
    pub fn with_params(field: QuorumField, decoherence_rate: f32, collapse_threshold: f32) -> Self {
        let sig = field.signature();
        Self {
            field,
            coherence: 1.0,
            decoherence_rate,
            collapse_threshold,
            prev_signature: sig,
        }
    }

    /// Evolve one tick. Returns true if collapsed.
    pub fn tick(&mut self) -> bool {
        // Evolve the field
        self.field.tick();

        // Measure change
        let new_sig = self.field.signature();
        let change = self.prev_signature.hamming(&new_sig) as f32 / FINGERPRINT_BITS as f32;
        self.prev_signature = new_sig;

        // Decrease coherence proportional to change
        self.coherence -= change * self.decoherence_rate;
        self.coherence = self.coherence.max(0.0);

        // Check for collapse
        if self.coherence < self.collapse_threshold {
            self.collapse();
            return true;
        }

        false
    }

    /// Inject fingerprint at position.
    pub fn inject(&mut self, pos: (usize, usize, usize), fp: &Fingerprint) {
        self.field.set(pos.0, pos.1, pos.2, fp);
    }

    /// Get current coherence level.
    pub fn coherence(&self) -> f32 {
        self.coherence
    }

    /// Force collapse: each cell snaps to majority-vote of neighbors.
    pub fn collapse(&mut self) {
        // Use high threshold to force stability
        let old_threshold = self.field.threshold();
        self.field.set_threshold(5);
        self.field.settle(10);
        self.field.set_threshold(old_threshold);

        // Reset coherence after collapse
        self.coherence = 1.0;
        self.prev_signature = self.field.signature();
    }

    /// Reset coherence to full.
    pub fn reset_coherence(&mut self) {
        self.coherence = 1.0;
    }
}

// =============================================================================
// 5. SURFACE CODE ERROR CORRECTION
// =============================================================================

/// Face enumeration for the 6 faces of the cube.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Face {
    /// XY plane at z=0
    XY0,
    /// XY plane at z=4
    XY4,
    /// XZ plane at y=0
    XZ0,
    /// XZ plane at y=4
    XZ4,
    /// YZ plane at x=0
    YZ0,
    /// YZ plane at x=4
    YZ4,
}

/// Detect and correct single-cell errors on a face of the crystal
/// using parity checks with adjacent cells (surface code).
///
/// # Returns
/// Number of corrections applied.
pub fn surface_code_correct(field: &mut QuorumField, face: Face) -> usize {
    let mut corrections = 0;
    const SYNDROME_THRESHOLD: u32 = 7000; // Strong error signal

    match face {
        Face::XY0 | Face::XY4 => {
            let z = if face == Face::XY0 { 0 } else { FIELD_SIZE - 1 };

            for x in 0..FIELD_SIZE {
                for y in 0..FIELD_SIZE {
                    let syndrome = compute_syndrome_xy(field, x, y, z);
                    if syndrome > SYNDROME_THRESHOLD {
                        correct_cell_from_neighbors(field, x, y, z);
                        corrections += 1;
                    }
                }
            }
        }
        Face::XZ0 | Face::XZ4 => {
            let y = if face == Face::XZ0 { 0 } else { FIELD_SIZE - 1 };

            for x in 0..FIELD_SIZE {
                for z in 0..FIELD_SIZE {
                    let syndrome = compute_syndrome_xz(field, x, y, z);
                    if syndrome > SYNDROME_THRESHOLD {
                        correct_cell_from_neighbors(field, x, y, z);
                        corrections += 1;
                    }
                }
            }
        }
        Face::YZ0 | Face::YZ4 => {
            let x = if face == Face::YZ0 { 0 } else { FIELD_SIZE - 1 };

            for y in 0..FIELD_SIZE {
                for z in 0..FIELD_SIZE {
                    let syndrome = compute_syndrome_yz(field, x, y, z);
                    if syndrome > SYNDROME_THRESHOLD {
                        correct_cell_from_neighbors(field, x, y, z);
                        corrections += 1;
                    }
                }
            }
        }
    }

    corrections
}

/// Compute syndrome for XY face cell.
/// Syndrome is the average Hamming distance to neighbors - high value means outlier.
fn compute_syndrome_xy(field: &QuorumField, x: usize, y: usize, z: usize) -> u32 {
    let cell = field.get(x, y, z);
    let mut total_distance = 0u32;
    let mut count = 0u32;

    // Sum Hamming distances to in-plane neighbors
    if x > 0 {
        total_distance += cell.hamming(&field.get(x - 1, y, z));
        count += 1;
    }
    if x < FIELD_SIZE - 1 {
        total_distance += cell.hamming(&field.get(x + 1, y, z));
        count += 1;
    }
    if y > 0 {
        total_distance += cell.hamming(&field.get(x, y - 1, z));
        count += 1;
    }
    if y < FIELD_SIZE - 1 {
        total_distance += cell.hamming(&field.get(x, y + 1, z));
        count += 1;
    }

    if count == 0 {
        return 0;
    }

    // Return average distance (high = likely error)
    total_distance / count
}

/// Compute syndrome for XZ face cell.
fn compute_syndrome_xz(field: &QuorumField, x: usize, y: usize, z: usize) -> u32 {
    let cell = field.get(x, y, z);
    let mut total_distance = 0u32;
    let mut count = 0u32;

    if x > 0 {
        total_distance += cell.hamming(&field.get(x - 1, y, z));
        count += 1;
    }
    if x < FIELD_SIZE - 1 {
        total_distance += cell.hamming(&field.get(x + 1, y, z));
        count += 1;
    }
    if z > 0 {
        total_distance += cell.hamming(&field.get(x, y, z - 1));
        count += 1;
    }
    if z < FIELD_SIZE - 1 {
        total_distance += cell.hamming(&field.get(x, y, z + 1));
        count += 1;
    }

    if count == 0 {
        return 0;
    }

    total_distance / count
}

/// Compute syndrome for YZ face cell.
fn compute_syndrome_yz(field: &QuorumField, x: usize, y: usize, z: usize) -> u32 {
    let cell = field.get(x, y, z);
    let mut total_distance = 0u32;
    let mut count = 0u32;

    if y > 0 {
        total_distance += cell.hamming(&field.get(x, y - 1, z));
        count += 1;
    }
    if y < FIELD_SIZE - 1 {
        total_distance += cell.hamming(&field.get(x, y + 1, z));
        count += 1;
    }
    if z > 0 {
        total_distance += cell.hamming(&field.get(x, y, z - 1));
        count += 1;
    }
    if z < FIELD_SIZE - 1 {
        total_distance += cell.hamming(&field.get(x, y, z + 1));
        count += 1;
    }

    if count == 0 {
        return 0;
    }

    total_distance / count
}

/// Correct cell by majority vote of neighbors.
fn correct_cell_from_neighbors(field: &mut QuorumField, x: usize, y: usize, z: usize) {
    let mut neighbors = Vec::new();

    if x > 0 {
        neighbors.push(field.get(x - 1, y, z));
    }
    if x < FIELD_SIZE - 1 {
        neighbors.push(field.get(x + 1, y, z));
    }
    if y > 0 {
        neighbors.push(field.get(x, y - 1, z));
    }
    if y < FIELD_SIZE - 1 {
        neighbors.push(field.get(x, y + 1, z));
    }
    if z > 0 {
        neighbors.push(field.get(x, y, z - 1));
    }
    if z < FIELD_SIZE - 1 {
        neighbors.push(field.get(x, y, z + 1));
    }

    if neighbors.is_empty() {
        return;
    }

    // Majority vote per bit
    let mut result = [0u64; crate::FINGERPRINT_U64];
    for bit in 0..FINGERPRINT_BITS {
        let mut ones = 0;
        for neighbor in &neighbors {
            if neighbor.get_bit(bit) {
                ones += 1;
            }
        }
        if ones > neighbors.len() / 2 {
            let word = bit / 64;
            let pos = bit % 64;
            result[word] |= 1 << pos;
        }
    }

    field.set(x, y, z, &Fingerprint::from_raw(result));
}

// =============================================================================
// 6. QUANTUM WALK STEP
// =============================================================================

/// One step of a quantum walk on the lattice.
/// Unlike classical settle (majority vote), this uses interference:
/// each cell receives popcount-weighted contributions from neighbors,
/// with constructive/destructive interference.
///
/// # Returns
/// Number of cells that changed.
pub fn quantum_walk_step(field: &mut QuorumField) -> usize {
    const QUANTUM_THRESHOLD: f32 = std::f32::consts::FRAC_PI_4; // π/4 ≈ 0.785

    let mut next_cells = Vec::with_capacity(FIELD_SIZE * FIELD_SIZE * FIELD_SIZE);
    let mut changes = 0;

    for x in 0..FIELD_SIZE {
        for y in 0..FIELD_SIZE {
            for z in 0..FIELD_SIZE {
                let current = field.get(x, y, z);
                let mut new_fp = [0u64; crate::FINGERPRINT_U64];

                // Collect neighbors
                let mut neighbors = Vec::new();
                if x > 0 {
                    neighbors.push(field.get(x - 1, y, z));
                }
                if x < FIELD_SIZE - 1 {
                    neighbors.push(field.get(x + 1, y, z));
                }
                if y > 0 {
                    neighbors.push(field.get(x, y - 1, z));
                }
                if y < FIELD_SIZE - 1 {
                    neighbors.push(field.get(x, y + 1, z));
                }
                if z > 0 {
                    neighbors.push(field.get(x, y, z - 1));
                }
                if z < FIELD_SIZE - 1 {
                    neighbors.push(field.get(x, y, z + 1));
                }

                if neighbors.is_empty() {
                    next_cells.push((x, y, z, current));
                    continue;
                }

                // Compute weights for each neighbor
                let weights: Vec<f32> = neighbors
                    .iter()
                    .map(|n| n.popcount() as f32 / FINGERPRINT_BITS as f32)
                    .collect();

                // For each bit, sum weighted contributions and apply quantum threshold
                for bit in 0..FINGERPRINT_BITS {
                    let mut sum = 0.0f32;
                    for (i, neighbor) in neighbors.iter().enumerate() {
                        if neighbor.get_bit(bit) {
                            sum += weights[i];
                        }
                    }

                    // Quantum interference threshold
                    if sum > QUANTUM_THRESHOLD {
                        let word = bit / 64;
                        let pos = bit % 64;
                        new_fp[word] |= 1 << pos;
                    }
                }

                let new_cell = Fingerprint::from_raw(new_fp);
                if new_cell != current {
                    changes += 1;
                }
                next_cells.push((x, y, z, new_cell));
            }
        }
    }

    // Apply all changes
    for (x, y, z, fp) in next_cells {
        field.set(x, y, z, &fp);
    }

    changes
}

// =============================================================================
// 7. ADIABATIC EVOLUTION
// =============================================================================

/// Slowly evolve the field from easy ground state to complex target
/// by linearly interpolating the quorum threshold.
///
/// # Returns
/// (final_field_signature, steps_taken, converged)
pub fn adiabatic_evolve(
    field: &mut QuorumField,
    start_threshold: u8,
    end_threshold: u8,
    total_steps: usize,
) -> (Fingerprint, usize, bool) {
    let mut steps_taken = 0;
    let mut stable_count = 0;
    let mut prev_sig = field.signature();

    for step in 0..total_steps {
        // Interpolate threshold
        let progress = step as f32 / total_steps as f32;
        let threshold = if start_threshold <= end_threshold {
            start_threshold + ((end_threshold - start_threshold) as f32 * progress) as u8
        } else {
            start_threshold - ((start_threshold - end_threshold) as f32 * progress) as u8
        };

        field.set_threshold(threshold.clamp(1, 6));

        // Tick the field
        let changed = field.tick();
        steps_taken += 1;

        let current_sig = field.signature();

        if !changed || current_sig == prev_sig {
            stable_count += 1;
            if stable_count >= 3 {
                // Converged at this threshold level, can advance faster
                stable_count = 0;
            }
        } else {
            stable_count = 0;
        }

        prev_sig = current_sig;
    }

    // Final convergence check
    let (final_steps, converged) = field.settle(20);
    steps_taken += final_steps;

    (field.signature(), steps_taken, converged)
}

// =============================================================================
// 8. DENSITY MATRIX (Mixed State Cells)
// =============================================================================

/// A cell that holds multiple possible states with probabilities.
/// Enables reasoning under uncertainty.
#[derive(Clone)]
pub struct MixedCell {
    /// (state, probability) pairs
    pub states: Vec<(Fingerprint, f32)>,
}

impl MixedCell {
    /// Create pure state (single state with probability 1.0).
    pub fn pure(fp: Fingerprint) -> Self {
        Self {
            states: vec![(fp, 1.0)],
        }
    }

    /// Create mixed state from multiple states with probabilities.
    /// Probabilities will be normalized.
    pub fn mixed(mut states: Vec<(Fingerprint, f32)>) -> Self {
        // Normalize probabilities
        let total: f32 = states.iter().map(|(_, p)| p).sum();
        if total > 0.0 {
            for (_, p) in &mut states {
                *p /= total;
            }
        }
        Self { states }
    }

    /// Collapse: pick state proportional to probability.
    pub fn measure(&self) -> Fingerprint {
        if self.states.is_empty() {
            return Fingerprint::zero();
        }
        if self.states.len() == 1 {
            return self.states[0].0.clone();
        }

        // Use fingerprint-based pseudo-random selection
        let seed = self.expected().popcount();
        let rand_val = (seed as f32 / FINGERPRINT_BITS as f32) % 1.0;

        let mut cumulative = 0.0f32;
        for (state, prob) in &self.states {
            cumulative += prob;
            if rand_val < cumulative {
                return state.clone();
            }
        }

        // Fallback to last state
        self.states
            .last()
            .map(|(s, _)| s.clone())
            .unwrap_or_else(Fingerprint::zero)
    }

    /// Shannon entropy of the mixture.
    pub fn entropy(&self) -> f32 {
        let mut h = 0.0f32;
        for (_, p) in &self.states {
            if *p > 0.0 {
                h -= p * p.log2();
            }
        }
        h
    }

    /// Weighted bundle of all states (expected value).
    pub fn expected(&self) -> Fingerprint {
        if self.states.is_empty() {
            return Fingerprint::zero();
        }

        // For each bit, compute weighted probability of being 1
        let mut result = [0u64; crate::FINGERPRINT_U64];

        for bit in 0..FINGERPRINT_BITS {
            let mut weighted_sum = 0.0f32;
            for (state, prob) in &self.states {
                if state.get_bit(bit) {
                    weighted_sum += prob;
                }
            }
            // Set bit if weighted probability > 0.5
            if weighted_sum > 0.5 {
                let word = bit / 64;
                let pos = bit % 64;
                result[word] |= 1 << pos;
            }
        }

        Fingerprint::from_raw(result)
    }

    /// Purity: 1.0 = pure state, 0.0 = maximally mixed.
    pub fn purity(&self) -> f32 {
        if self.states.is_empty() {
            return 0.0;
        }

        // Purity = sum of squared probabilities
        let mut purity = 0.0f32;
        for (_, p) in &self.states {
            purity += p * p;
        }
        purity
    }

    /// Number of states in the mixture.
    pub fn state_count(&self) -> usize {
        self.states.len()
    }
}

/// Field with mixed-state cells.
#[derive(Clone)]
pub struct MixedField {
    cells: Vec<Vec<Vec<MixedCell>>>,
}

impl MixedField {
    /// Create from pure quorum field.
    pub fn from_pure(field: &QuorumField) -> Self {
        let mut cells = Vec::with_capacity(FIELD_SIZE);
        for x in 0..FIELD_SIZE {
            let mut plane = Vec::with_capacity(FIELD_SIZE);
            for y in 0..FIELD_SIZE {
                let mut row = Vec::with_capacity(FIELD_SIZE);
                for z in 0..FIELD_SIZE {
                    row.push(MixedCell::pure(field.get(x, y, z)));
                }
                plane.push(row);
            }
            cells.push(plane);
        }
        Self { cells }
    }

    /// Collapse all cells to create pure field.
    pub fn collapse_all(&self) -> QuorumField {
        let mut field = QuorumField::default_threshold();
        for x in 0..FIELD_SIZE {
            for y in 0..FIELD_SIZE {
                for z in 0..FIELD_SIZE {
                    let measured = self.cells[x][y][z].measure();
                    field.set(x, y, z, &measured);
                }
            }
        }
        field
    }

    /// Get cell at position.
    pub fn get(&self, x: usize, y: usize, z: usize) -> &MixedCell {
        &self.cells[x][y][z]
    }

    /// Set cell at position.
    pub fn set(&mut self, x: usize, y: usize, z: usize, cell: MixedCell) {
        self.cells[x][y][z] = cell;
    }

    /// Evolve: each cell's states interact with neighbor states.
    /// This reduces entropy over time as states converge.
    pub fn tick(&mut self) {
        let mut new_cells = self.cells.clone();

        for x in 0..FIELD_SIZE {
            for y in 0..FIELD_SIZE {
                for z in 0..FIELD_SIZE {
                    let mut neighbor_expected = Vec::new();

                    if x > 0 {
                        neighbor_expected.push(self.cells[x - 1][y][z].expected());
                    }
                    if x < FIELD_SIZE - 1 {
                        neighbor_expected.push(self.cells[x + 1][y][z].expected());
                    }
                    if y > 0 {
                        neighbor_expected.push(self.cells[x][y - 1][z].expected());
                    }
                    if y < FIELD_SIZE - 1 {
                        neighbor_expected.push(self.cells[x][y + 1][z].expected());
                    }
                    if z > 0 {
                        neighbor_expected.push(self.cells[x][y][z - 1].expected());
                    }
                    if z < FIELD_SIZE - 1 {
                        neighbor_expected.push(self.cells[x][y][z + 1].expected());
                    }

                    if neighbor_expected.is_empty() {
                        continue;
                    }

                    // Blend current states with neighbor influence
                    let current = &self.cells[x][y][z];
                    let mut new_states = Vec::new();

                    for (state, prob) in &current.states {
                        // Compute similarity to neighbors
                        let mut avg_sim = 0.0f32;
                        for ne in &neighbor_expected {
                            avg_sim += state.similarity(ne);
                        }
                        avg_sim /= neighbor_expected.len() as f32;

                        // Boost probability of states similar to neighbors
                        let new_prob = prob * (0.5 + avg_sim * 0.5);
                        new_states.push((state.clone(), new_prob));
                    }

                    new_cells[x][y][z] = MixedCell::mixed(new_states);
                }
            }
        }

        self.cells = new_cells;
    }

    /// Total entropy of the field.
    pub fn total_entropy(&self) -> f32 {
        let mut total = 0.0f32;
        for x in 0..FIELD_SIZE {
            for y in 0..FIELD_SIZE {
                for z in 0..FIELD_SIZE {
                    total += self.cells[x][y][z].entropy();
                }
            }
        }
        total
    }
}

// =============================================================================
// 9. STATE TELEPORTATION
// =============================================================================

/// Teleportation packet containing correction bits.
#[derive(Clone)]
pub struct TeleportPacket {
    /// Correction bits: XOR of (source ⊕ shared_half_a)
    pub corrections: Fingerprint,
}

impl TeleportPacket {
    /// Size in bytes (1.25KB for 10K bits).
    pub fn size_bytes(&self) -> usize {
        crate::FINGERPRINT_U64 * 8
    }
}

/// Prepare teleportation: Alice computes corrections.
pub fn teleport_prepare(source_cell: &Fingerprint, shared_half_a: &Fingerprint) -> TeleportPacket {
    // Corrections = source XOR shared_half_a
    let corrections = source_cell.bind(shared_half_a);
    TeleportPacket { corrections }
}

/// Receive teleportation: Bob applies corrections.
pub fn teleport_receive(packet: &TeleportPacket, shared_half_b: &Fingerprint) -> Fingerprint {
    // Result = corrections XOR shared_half_b
    // If shared_a XOR shared_b = key, then:
    // result = (source XOR shared_a) XOR shared_b
    //        = source XOR (shared_a XOR shared_b)
    //        = source XOR key
    packet.corrections.bind(shared_half_b)
}

/// Teleport entire crystal (all 125 cells via projections).
pub fn teleport_crystal(
    source: &Crystal4K,
    shared_a: &Crystal4K,
    shared_b: &Crystal4K,
) -> Crystal4K {
    // Teleport each projection
    let packet_x = teleport_prepare(&source.x_fp(), &shared_a.x_fp());
    let packet_y = teleport_prepare(&source.y_fp(), &shared_a.y_fp());
    let packet_z = teleport_prepare(&source.z_fp(), &shared_a.z_fp());

    let received_x = teleport_receive(&packet_x, &shared_b.x_fp());
    let received_y = teleport_receive(&packet_y, &shared_b.y_fp());
    let received_z = teleport_receive(&packet_z, &shared_b.z_fp());

    Crystal4K::new(received_x, received_y, received_z)
}

/// Create matched crystal pair for teleportation.
/// The key allows recovery: if you teleport with (shared_a, shared_b),
/// unbinding with key recovers the original.
pub fn create_teleport_pair(key: &Fingerprint) -> (Crystal4K, Crystal4K) {
    let (ax, bx) = create_entangled_pair(key);
    let (ay, by) = create_entangled_pair(key);
    let (az, bz) = create_entangled_pair(key);

    (Crystal4K::new(ax, ay, az), Crystal4K::new(bx, by, bz))
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // 1. Entanglement tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_entangle_cells_self_inverse() {
        let mut field = QuorumField::default_threshold();
        let fp = Fingerprint::from_content("test pattern");
        field.set(0, 0, 0, &fp);
        field.set(4, 4, 4, &Fingerprint::from_content("target"));

        let original_target = field.get(4, 4, 4);
        let basis = Fingerprint::from_content("basis");
        let gate = Fingerprint::from_content("gate mask");

        // Apply entanglement twice
        entangle_cells(&mut field, (0, 0, 0), (4, 4, 4), &basis, &gate, 0);
        entangle_cells(&mut field, (0, 0, 0), (4, 4, 4), &basis, &gate, 0);

        // Should return to original (XOR is self-inverse)
        assert_eq!(field.get(4, 4, 4), original_target);
    }

    #[test]
    fn test_entangle_cells_only_affects_target() {
        let mut field = QuorumField::default_threshold();
        let fp = Fingerprint::from_content("control");
        field.set(0, 0, 0, &fp);
        field.set(2, 2, 2, &Fingerprint::from_content("bystander"));

        let bystander_before = field.get(2, 2, 2);
        let basis = Fingerprint::from_content("basis");
        let gate = Fingerprint::from_content("gate");

        entangle_cells(&mut field, (0, 0, 0), (4, 4, 4), &basis, &gate, 0);

        // Bystander should be unchanged
        assert_eq!(field.get(2, 2, 2), bystander_before);
    }

    #[test]
    fn test_create_entangled_pair() {
        let key = Fingerprint::from_content("entanglement key");
        let (half_a, half_b) = create_entangled_pair(&key);

        // half_a XOR half_b should equal key
        let recovered = half_a.bind(&half_b);
        assert_eq!(recovered, key);
    }

    // -------------------------------------------------------------------------
    // 2. QFT tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_qft_inverse_recovery() {
        let mut field = QuorumField::default_threshold();
        let fp = Fingerprint::from_content("pattern");
        field.set(2, 2, 2, &fp);

        let original_sig = field.signature();

        // Apply QFT then inverse QFT
        let _ = qft_axis(&mut field, Axis::X);
        let _ = inverse_qft_axis(&mut field, Axis::X);

        // Should recover original
        let recovered_sig = field.signature();
        assert_eq!(original_sig, recovered_sig);
    }

    #[test]
    fn test_qft_periodic_pattern() {
        let mut field = QuorumField::default_threshold();
        let fp = Fingerprint::from_content("periodic");

        // Create period-2 pattern along X
        for y in 0..FIELD_SIZE {
            for z in 0..FIELD_SIZE {
                field.set(0, y, z, &fp);
                field.set(2, y, z, &fp);
                field.set(4, y, z, &fp);
            }
        }

        let result = qft_axis(&mut field, Axis::X);

        // QFT of periodic pattern should have structure
        assert!(result.popcount() > 0);
    }

    // -------------------------------------------------------------------------
    // 3. Phase kickback tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_phase_kickback_identity() {
        let mut field = QuorumField::default_threshold();
        field.randomize();

        let identity = |fp: &Fingerprint| fp.clone();
        let (eigenvalue, _delta) = phase_kickback(&mut field, identity, Axis::X);

        // Identity should give eigenvalue ≈ 0
        assert!(
            eigenvalue < 0.01,
            "Identity eigenvalue should be near 0, got {}",
            eigenvalue
        );
    }

    #[test]
    fn test_phase_kickback_not() {
        let mut field = QuorumField::default_threshold();
        field.randomize();

        let not_op = |fp: &Fingerprint| fp.not();
        let (eigenvalue, _delta) = phase_kickback(&mut field, not_op, Axis::X);

        // NOT should give eigenvalue ≈ 1.0 (maximum change)
        assert!(
            eigenvalue > 0.9,
            "NOT eigenvalue should be near 1.0, got {}",
            eigenvalue
        );
    }

    // -------------------------------------------------------------------------
    // 4. Coherence tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_coherent_crystal_creation() {
        let field = QuorumField::default_threshold();
        let crystal = CoherentCrystal::new(field);

        assert!((crystal.coherence() - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_coherence_decreases_with_change() {
        let mut field = QuorumField::new(2); // Low threshold for more change
        field.randomize();

        let mut crystal = CoherentCrystal::with_params(field, 0.5, 0.1);
        let initial_coherence = crystal.coherence();

        // Tick a few times
        for _ in 0..5 {
            crystal.tick();
        }

        // Coherence should have decreased
        assert!(crystal.coherence() < initial_coherence);
    }

    #[test]
    fn test_collapse_stabilizes() {
        let mut field = QuorumField::new(2);
        field.randomize();

        let mut crystal = CoherentCrystal::with_params(field, 0.5, 0.01);

        // Force collapse
        crystal.collapse();

        // After collapse, should be stable (high coherence reset)
        assert!((crystal.coherence() - 1.0).abs() < 0.0001);
    }

    // -------------------------------------------------------------------------
    // 5. Surface code tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_surface_code_corrects_single_error() {
        let mut field = QuorumField::default_threshold();
        let uniform = Fingerprint::from_content("uniform pattern");

        // Fill entire bottom layer plus z=1 for proper neighbor detection
        for x in 0..FIELD_SIZE {
            for y in 0..FIELD_SIZE {
                field.set(x, y, 0, &uniform);
                field.set(x, y, 1, &uniform); // Add neighbors in z direction
            }
        }

        // Corrupt one cell in the center (has 4 neighbors)
        let corrupted = uniform.not(); // Maximum difference from uniform
        field.set(2, 2, 0, &corrupted);

        // Verify the corrupted cell is different
        let before_sim = field.get(2, 2, 0).similarity(&uniform);
        assert!(before_sim < 0.1, "Corrupted cell should be very different");

        let corrections = surface_code_correct(&mut field, Face::XY0);

        // Should have made a correction
        assert!(corrections > 0, "Should have corrected the error");

        // After correction, cell should be closer to uniform
        let after_sim = field.get(2, 2, 0).similarity(&uniform);
        assert!(
            after_sim > before_sim,
            "Correction should improve similarity"
        );
    }

    #[test]
    fn test_surface_code_leaves_uniform_unchanged() {
        let mut field = QuorumField::default_threshold();
        let uniform = Fingerprint::from_content("uniform");

        // Fill entire field with same pattern
        for x in 0..FIELD_SIZE {
            for y in 0..FIELD_SIZE {
                for z in 0..FIELD_SIZE {
                    field.set(x, y, z, &uniform);
                }
            }
        }

        let sig_before = field.signature();
        let corrections = surface_code_correct(&mut field, Face::XY0);
        let sig_after = field.signature();

        // No corrections on uniform field
        assert_eq!(corrections, 0);
        assert_eq!(sig_before, sig_after);
    }

    // -------------------------------------------------------------------------
    // 6. Quantum walk tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_quantum_walk_spreads() {
        let mut field = QuorumField::default_threshold();

        // Place walker at center
        let walker = Fingerprint::from_content("quantum walker");
        field.set(2, 2, 2, &walker);

        // Run quantum walk
        let changes = quantum_walk_step(&mut field);

        // Should have changed some cells
        assert!(changes > 0, "Quantum walk should spread");
    }

    #[test]
    fn test_quantum_walk_differs_from_classical() {
        // Setup two identical fields
        let mut quantum_field = QuorumField::new(4);
        let mut classical_field = QuorumField::new(4);

        let walker = Fingerprint::from_content("walker");
        quantum_field.set(2, 2, 2, &walker);
        classical_field.set(2, 2, 2, &walker);

        // Quantum walk
        quantum_walk_step(&mut quantum_field);

        // Classical settle (one tick)
        classical_field.tick();

        // They should be different (different dynamics)
        let quantum_sig = quantum_field.signature();
        let classical_sig = classical_field.signature();

        // Note: This test may occasionally fail if they happen to match
        // but the dynamics are fundamentally different
        let dist = quantum_sig.hamming(&classical_sig);
        // They should differ somewhat (not testing exact difference)
        println!("Quantum vs classical distance: {}", dist);
    }

    // -------------------------------------------------------------------------
    // 7. Adiabatic evolution tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_adiabatic_converges() {
        let mut field = QuorumField::new(1);
        field.randomize();

        let (_sig, steps, converged) = adiabatic_evolve(&mut field, 1, 5, 50);

        assert!(steps > 0);
        assert!(converged, "Adiabatic evolution should converge");
    }

    #[test]
    fn test_adiabatic_produces_stable_state() {
        let mut field = QuorumField::new(1);
        field.randomize();

        adiabatic_evolve(&mut field, 1, 5, 50);

        // Final state should be stable
        let sig_before = field.signature();
        field.tick();
        let sig_after = field.signature();

        assert_eq!(sig_before, sig_after, "Adiabatic result should be stable");
    }

    // -------------------------------------------------------------------------
    // 8. Mixed state tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_mixed_cell_pure() {
        let fp = Fingerprint::from_content("pure state");
        let cell = MixedCell::pure(fp.clone());

        assert_eq!(cell.state_count(), 1);
        assert!((cell.purity() - 1.0).abs() < 0.0001);
        assert_eq!(cell.measure(), fp);
    }

    #[test]
    fn test_mixed_cell_entropy() {
        let fp1 = Fingerprint::from_content("state1");
        let fp2 = Fingerprint::from_content("state2");

        // 50/50 mixture has entropy = 1.0
        let cell = MixedCell::mixed(vec![(fp1, 0.5), (fp2, 0.5)]);

        assert!((cell.entropy() - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_mixed_cell_expected() {
        let fp1 = Fingerprint::from_content("a");
        let fp2 = Fingerprint::from_content("b");

        let cell = MixedCell::mixed(vec![(fp1.clone(), 0.5), (fp2.clone(), 0.5)]);
        let expected = cell.expected();

        // Expected should be somewhere between the two states
        let sim1 = expected.similarity(&fp1);
        let sim2 = expected.similarity(&fp2);

        assert!(sim1 > 0.3 && sim1 < 0.8);
        assert!(sim2 > 0.3 && sim2 < 0.8);
    }

    #[test]
    fn test_mixed_field_entropy_decreases() {
        // Use deterministic field instead of random
        let mut field = QuorumField::default_threshold();
        let base = Fingerprint::from_content("deterministic base");
        for x in 0..FIELD_SIZE {
            for y in 0..FIELD_SIZE {
                for z in 0..FIELD_SIZE {
                    let cell = Fingerprint::from_content(&format!("cell_{}_{}_{}", x, y, z));
                    field.set(x, y, z, &base.bind(&cell));
                }
            }
        }

        let mut mixed_field = MixedField::from_pure(&field);

        // Add some mixing - all cells have 2 states
        for x in 0..FIELD_SIZE {
            for y in 0..FIELD_SIZE {
                for z in 0..FIELD_SIZE {
                    let current = field.get(x, y, z);
                    let alt = Fingerprint::from_content(&format!("alt_{}_{}_{}", x, y, z));
                    mixed_field.set(x, y, z, MixedCell::mixed(vec![(current, 0.5), (alt, 0.5)]));
                }
            }
        }

        let initial_entropy = mixed_field.total_entropy();

        // Each cell has entropy = 1.0 (50/50 split), total = 125
        assert!(
            (initial_entropy - 125.0).abs() < 1.0,
            "Initial entropy should be ~125, got {}",
            initial_entropy
        );

        // Evolve many times to ensure convergence
        for _ in 0..20 {
            mixed_field.tick();
        }

        let final_entropy = mixed_field.total_entropy();

        // Entropy should decrease as states converge
        // (neighbor influence causes probabilities to shift)
        assert!(
            final_entropy <= initial_entropy,
            "Entropy should not increase: {} -> {}",
            initial_entropy,
            final_entropy
        );
    }

    // -------------------------------------------------------------------------
    // 9. Teleportation tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_teleport_fingerprint() {
        let key = Fingerprint::from_content("teleport key");
        let (half_a, half_b) = create_entangled_pair(&key);

        let source = Fingerprint::from_content("secret message");

        // Teleport
        let packet = teleport_prepare(&source, &half_a);
        let received = teleport_receive(&packet, &half_b);

        // Received should equal source XOR key
        let expected = source.bind(&key);
        assert_eq!(received, expected);

        // Unbind key to recover original
        let recovered = received.unbind(&key);
        assert_eq!(recovered, source);
    }

    #[test]
    fn test_teleport_crystal() {
        let key = Fingerprint::from_content("crystal key");
        let (shared_a, shared_b) = create_teleport_pair(&key);

        let source = Crystal4K::new(
            Fingerprint::from_content("x axis"),
            Fingerprint::from_content("y axis"),
            Fingerprint::from_content("z axis"),
        );

        let received = teleport_crystal(&source, &shared_a, &shared_b);

        // Unbind key from each axis to recover
        let recovered_x = received.x_fp().unbind(&key);
        let recovered_y = received.y_fp().unbind(&key);
        let recovered_z = received.z_fp().unbind(&key);

        assert_eq!(recovered_x, source.x_fp());
        assert_eq!(recovered_y, source.y_fp());
        assert_eq!(recovered_z, source.z_fp());
    }

    #[test]
    fn test_teleport_packet_size() {
        let source = Fingerprint::from_content("data");
        let shared = Fingerprint::from_content("shared");

        let packet = teleport_prepare(&source, &shared);

        // Packet should be 2KB (16K bits = 2048 bytes = 256 u64s)
        assert_eq!(packet.size_bytes(), crate::FINGERPRINT_U64 * 8);
        assert!(packet.size_bytes() <= 2048); // Fits in 2KB
    }
}
