//! Quantum-Inspired Operators + Tree Addressing
//!
//! The 4096 CAM operations as mathematical operators on fingerprint space.
//! Non-commutative, linear, with eigenvalue semantics.
//!
//! Tree addressing: 256-way branching for Neo4j-style hierarchical navigation.
//!
//! Key insight: Fingerprints ARE wave functions (superposition of basis states).
//! Operators transform them, measurements collapse them.

use crate::core::Fingerprint;

// =============================================================================
// TREE ADDRESSING (256-way branching)
// =============================================================================

/// Tree address: hierarchical path through 256-way tree
///
/// Format: [depth][b0][b1][b2]...[bn]
/// - depth: 0-255 (max 255 levels)
/// - each b_i: 0-255 (256-way branching)
///
/// Example: /concepts/animals/mammals/cats
///          = [4][0x01][0x03][0x0F][0x42]
///          
/// This maps to Distinguished Names (DN) like LDAP/Neo4j paths
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TreeAddr {
    /// Path components (first byte is depth, rest are branches)
    path: Vec<u8>,
}

impl TreeAddr {
    /// Root address
    pub fn root() -> Self {
        Self { path: vec![0] }
    }

    /// Create from path components
    pub fn from_path(components: &[u8]) -> Self {
        let mut path = vec![components.len() as u8];
        path.extend_from_slice(components);
        Self { path }
    }

    /// Parse from string path like "/concepts/animals/mammals"
    pub fn from_string(s: &str) -> Self {
        let components: Vec<u8> = s
            .split('/')
            .filter(|s| !s.is_empty())
            .map(|s| hash_component(s))
            .collect();
        Self::from_path(&components)
    }

    /// Get depth
    pub fn depth(&self) -> u8 {
        self.path.get(0).copied().unwrap_or(0)
    }

    /// Get component at level
    pub fn component(&self, level: usize) -> Option<u8> {
        self.path.get(level + 1).copied()
    }

    /// Get all components
    pub fn components(&self) -> &[u8] {
        if self.path.len() > 1 {
            &self.path[1..]
        } else {
            &[]
        }
    }

    /// Descend to child
    pub fn child(&self, branch: u8) -> Self {
        let mut path = self.path.clone();
        path[0] = path[0].saturating_add(1);
        path.push(branch);
        Self { path }
    }

    /// Ascend to parent
    pub fn parent(&self) -> Option<Self> {
        if self.depth() == 0 {
            return None;
        }
        let mut path = self.path.clone();
        path[0] = path[0].saturating_sub(1);
        path.pop();
        Some(Self { path })
    }

    /// Get ancestor at level
    pub fn ancestor(&self, level: u8) -> Option<Self> {
        if level >= self.depth() {
            return None;
        }
        let mut path = vec![level];
        path.extend_from_slice(&self.path[1..=(level as usize)]);
        Some(Self { path })
    }

    /// Check if this is ancestor of other
    pub fn is_ancestor_of(&self, other: &TreeAddr) -> bool {
        if self.depth() >= other.depth() {
            return false;
        }
        self.components() == &other.components()[..self.depth() as usize]
    }

    /// Common ancestor with another address
    pub fn common_ancestor(&self, other: &TreeAddr) -> Self {
        let mut common = vec![];
        for (a, b) in self.components().iter().zip(other.components()) {
            if a == b {
                common.push(*a);
            } else {
                break;
            }
        }
        Self::from_path(&common)
    }

    /// Convert to fingerprint (deterministic mapping)
    pub fn to_fingerprint(&self) -> Fingerprint {
        Fingerprint::from_content(&format!("TREE_ADDR::{:?}", self.path))
    }

    /// Encode as u64 (for shallow trees, max depth 7)
    pub fn to_u64(&self) -> u64 {
        let mut result = 0u64;
        for (i, &byte) in self.path.iter().take(8).enumerate() {
            result |= (byte as u64) << (i * 8);
        }
        result
    }

    /// Decode from u64
    pub fn from_u64(value: u64) -> Self {
        let depth = (value & 0xFF) as u8;
        let mut path = vec![depth];
        for i in 1..=depth.min(7) {
            path.push(((value >> (i * 8)) & 0xFF) as u8);
        }
        Self { path }
    }
}

/// Hash a path component to 0-255
fn hash_component(s: &str) -> u8 {
    let mut hash = 0u32;
    for byte in s.bytes() {
        hash = hash.wrapping_mul(31).wrapping_add(byte as u32);
    }
    (hash % 256) as u8
}

/// Well-known tree branches (like LDAP OIDs)
pub mod tree_branches {
    pub const CONCEPTS: u8 = 0x01;
    pub const ENTITIES: u8 = 0x02;
    pub const EVENTS: u8 = 0x03;
    pub const RELATIONS: u8 = 0x04;
    pub const TEMPLATES: u8 = 0x05;
    pub const MEMORIES: u8 = 0x06;
    pub const GOALS: u8 = 0x07;
    pub const BELIEFS: u8 = 0x08;
    pub const RULES: u8 = 0x09;
    pub const SCHEMAS: u8 = 0x0A;

    // NSM primes at 0x10-0x4F (65 primes)
    pub const NSM_I: u8 = 0x10;
    pub const NSM_YOU: u8 = 0x11;
    pub const NSM_SOMEONE: u8 = 0x12;
    // ... etc

    // Cognitive frameworks at 0x80-0x8F
    pub const NARS: u8 = 0x80;
    pub const ACTR: u8 = 0x81;
    pub const RL: u8 = 0x82;
    pub const CAUSALITY: u8 = 0x83;
    pub const QUALIA: u8 = 0x84;
    pub const RUNG: u8 = 0x85;

    // User-defined at 0xF0-0xFF
    pub const USER_BASE: u8 = 0xF0;
}

// =============================================================================
// QUANTUM-INSPIRED OPERATORS
// =============================================================================

/// Operator trait - mathematical transformation on fingerprint space
///
/// Properties:
/// - Linearity: Â(α|ψ⟩ + β|φ⟩) = αÂ|ψ⟩ + βÂ|φ⟩
/// - Hermitian operators have real eigenvalues
/// - Non-commutative: ÂB̂ ≠ B̂Â in general
pub trait QuantumOp: Send + Sync {
    /// Apply operator to state, return new state
    fn apply(&self, state: &Fingerprint) -> Fingerprint;

    /// Check if this operator is Hermitian (self-adjoint)
    fn is_hermitian(&self) -> bool {
        false
    }

    /// Get the adjoint (dagger) operator
    fn adjoint(&self) -> Box<dyn QuantumOp>;

    /// Commutator [Â, B̂] = ÂB̂ - B̂Â
    fn commutator(&self, other: &dyn QuantumOp, state: &Fingerprint) -> Fingerprint {
        let ab = other.apply(&self.apply(state));
        let ba = self.apply(&other.apply(state));
        // Commutator as XOR difference
        ab.bind(&ba)
    }

    /// Check if operators commute on given state
    fn commutes_with(&self, other: &dyn QuantumOp, state: &Fingerprint) -> bool {
        let comm = self.commutator(other, state);
        comm.density() < 0.01 // Nearly zero = commutes
    }

    /// Expectation value ⟨ψ|Â|ψ⟩ (as similarity)
    fn expectation(&self, state: &Fingerprint) -> f32 {
        let result = self.apply(state);
        state.similarity(&result)
    }

    /// Variance ⟨Â²⟩ - ⟨Â⟩²
    fn variance(&self, state: &Fingerprint) -> f32 {
        let exp = self.expectation(state);
        let result = self.apply(state);
        let exp_sq = self.expectation(&result);
        (exp_sq - exp * exp).abs()
    }
}

// =============================================================================
// CORE QUANTUM OPERATORS
// =============================================================================

/// Identity operator - does nothing
pub struct IdentityOp;

impl QuantumOp for IdentityOp {
    fn apply(&self, state: &Fingerprint) -> Fingerprint {
        state.clone()
    }

    fn is_hermitian(&self) -> bool {
        true
    }

    fn adjoint(&self) -> Box<dyn QuantumOp> {
        Box::new(IdentityOp)
    }
}

/// NOT operator - flips all bits (Pauli X on each qubit)
pub struct NotOp;

impl QuantumOp for NotOp {
    fn apply(&self, state: &Fingerprint) -> Fingerprint {
        state.not()
    }

    fn is_hermitian(&self) -> bool {
        true
    }

    fn adjoint(&self) -> Box<dyn QuantumOp> {
        Box::new(NotOp) // Self-inverse
    }
}

/// Bind operator - XOR with fixed fingerprint
/// This is like a controlled rotation in quantum computing
pub struct BindOp {
    operand: Fingerprint,
}

impl BindOp {
    pub fn new(operand: Fingerprint) -> Self {
        Self { operand }
    }
}

impl QuantumOp for BindOp {
    fn apply(&self, state: &Fingerprint) -> Fingerprint {
        state.bind(&self.operand)
    }

    fn is_hermitian(&self) -> bool {
        true
    } // XOR is self-inverse

    fn adjoint(&self) -> Box<dyn QuantumOp> {
        Box::new(BindOp {
            operand: self.operand.clone(),
        })
    }
}

/// Permutation operator - cyclic rotation of bits
/// Analogous to phase shift in quantum mechanics
pub struct PermuteOp {
    shift: i32,
}

impl PermuteOp {
    pub fn new(shift: i32) -> Self {
        Self { shift }
    }
}

impl QuantumOp for PermuteOp {
    fn apply(&self, state: &Fingerprint) -> Fingerprint {
        state.permute(self.shift)
    }

    fn is_hermitian(&self) -> bool {
        false
    } // Unless shift = 0

    fn adjoint(&self) -> Box<dyn QuantumOp> {
        Box::new(PermuteOp { shift: -self.shift })
    }
}

/// Projection operator - projects onto subspace defined by mask
/// |ψ⟩ → |mask⟩⟨mask|ψ⟩
pub struct ProjectOp {
    mask: Fingerprint,
}

impl ProjectOp {
    pub fn new(mask: Fingerprint) -> Self {
        Self { mask }
    }
}

impl QuantumOp for ProjectOp {
    fn apply(&self, state: &Fingerprint) -> Fingerprint {
        // Project state onto mask subspace
        state.and(&self.mask)
    }

    fn is_hermitian(&self) -> bool {
        true
    } // P² = P

    fn adjoint(&self) -> Box<dyn QuantumOp> {
        Box::new(ProjectOp {
            mask: self.mask.clone(),
        })
    }
}

/// Hadamard-like operator - creates superposition
/// For binary: H maps 0→(0+1), 1→(0-1)
/// For fingerprints: randomizes bits based on seed
pub struct HadamardOp {
    seed: u64,
}

impl HadamardOp {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }
}

impl QuantumOp for HadamardOp {
    fn apply(&self, state: &Fingerprint) -> Fingerprint {
        // Create pseudo-random superposition
        let mut result = state.clone();
        let mut rng = self.seed;

        for bit in 0..crate::FINGERPRINT_BITS {
            // Simple LCG PRNG
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);

            if (rng >> 32) & 1 == 1 {
                // Flip bit with 50% probability
                result.set_bit(bit, !state.get_bit(bit));
            }
        }

        result
    }

    fn is_hermitian(&self) -> bool {
        true
    } // H² = I

    fn adjoint(&self) -> Box<dyn QuantumOp> {
        Box::new(HadamardOp { seed: self.seed })
    }
}

/// Measurement operator - collapses to nearest eigenstate (codebook entry)
pub struct MeasureOp {
    /// Eigenstates (codebook entries)
    eigenstates: Vec<Fingerprint>,
}

impl MeasureOp {
    pub fn new(eigenstates: Vec<Fingerprint>) -> Self {
        Self { eigenstates }
    }

    /// Find eigenstate closest to given state
    pub fn measure(&self, state: &Fingerprint) -> (usize, f32, Fingerprint) {
        let mut best_idx = 0;
        let mut best_sim = 0.0f32;

        for (i, eigenstate) in self.eigenstates.iter().enumerate() {
            let sim = state.similarity(eigenstate);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }

        (best_idx, best_sim, self.eigenstates[best_idx].clone())
    }
}

impl QuantumOp for MeasureOp {
    fn apply(&self, state: &Fingerprint) -> Fingerprint {
        let (_, _, collapsed) = self.measure(state);
        collapsed
    }

    fn is_hermitian(&self) -> bool {
        true
    }

    fn adjoint(&self) -> Box<dyn QuantumOp> {
        Box::new(MeasureOp {
            eigenstates: self.eigenstates.clone(),
        })
    }
}

/// Time evolution operator - exp(-iĤt/ℏ) analogue
/// Implements decay/diffusion over fingerprint space
pub struct TimeEvolutionOp {
    /// Hamiltonian fingerprint (energy landscape)
    hamiltonian: Fingerprint,
    /// Time step
    time: f32,
}

impl TimeEvolutionOp {
    pub fn new(hamiltonian: Fingerprint, time: f32) -> Self {
        Self { hamiltonian, time }
    }
}

impl QuantumOp for TimeEvolutionOp {
    fn apply(&self, state: &Fingerprint) -> Fingerprint {
        // Evolve state based on Hamiltonian
        // Higher energy states decay faster

        let energy = 1.0 - state.similarity(&self.hamiltonian);
        let decay = (-energy * self.time).exp();

        // Interpolate toward Hamiltonian ground state
        if decay > 0.5 {
            state.clone()
        } else {
            // Gradual approach to ground state
            let blend_factor = 1.0 - decay;
            blend_fingerprints(state, &self.hamiltonian, blend_factor)
        }
    }

    fn is_hermitian(&self) -> bool {
        false
    }

    fn adjoint(&self) -> Box<dyn QuantumOp> {
        Box::new(TimeEvolutionOp {
            hamiltonian: self.hamiltonian.clone(),
            time: -self.time,
        })
    }
}

// =============================================================================
// COGNITIVE OPERATORS (derived from 4096 CAM ops)
// =============================================================================

/// NARS inference as quantum operator
pub struct NarsInferenceOp {
    /// Inference type: 0=deduction, 1=induction, 2=abduction
    inference_type: u8,
    /// Second premise (first is input state)
    premise: Fingerprint,
    /// Truth value weights
    truth: (f32, f32),
}

impl NarsInferenceOp {
    pub fn deduction(premise: Fingerprint, truth: (f32, f32)) -> Self {
        Self {
            inference_type: 0,
            premise,
            truth,
        }
    }

    pub fn induction(premise: Fingerprint, truth: (f32, f32)) -> Self {
        Self {
            inference_type: 1,
            premise,
            truth,
        }
    }

    pub fn abduction(premise: Fingerprint, truth: (f32, f32)) -> Self {
        Self {
            inference_type: 2,
            premise,
            truth,
        }
    }
}

impl QuantumOp for NarsInferenceOp {
    fn apply(&self, state: &Fingerprint) -> Fingerprint {
        // Inference as superposition of premises weighted by truth
        let (f, c) = self.truth;

        // Conclusion = weighted blend of state and premise
        let weight = f * c;
        let conclusion = blend_fingerprints(state, &self.premise, weight);

        // Add noise inversely proportional to confidence
        if c < 0.99 {
            let noise = Fingerprint::from_content(&format!("NOISE_{}", c));
            blend_fingerprints(&conclusion, &noise, 1.0 - c)
        } else {
            conclusion
        }
    }

    fn is_hermitian(&self) -> bool {
        false
    }

    fn adjoint(&self) -> Box<dyn QuantumOp> {
        // Adjoint reverses inference direction
        Box::new(NarsInferenceOp {
            inference_type: match self.inference_type {
                0 => 2, // Deduction ↔ Abduction
                2 => 0,
                _ => 1,
            },
            premise: self.premise.clone(),
            truth: self.truth,
        })
    }
}

/// ACT-R retrieval as measurement operator
pub struct ActrRetrievalOp {
    /// Declarative memory chunks
    chunks: Vec<Fingerprint>,
    /// Activation levels
    activations: Vec<f32>,
    /// Retrieval threshold
    threshold: f32,
}

impl ActrRetrievalOp {
    pub fn new(chunks: Vec<Fingerprint>, activations: Vec<f32>, threshold: f32) -> Self {
        Self {
            chunks,
            activations,
            threshold,
        }
    }
}

impl QuantumOp for ActrRetrievalOp {
    fn apply(&self, state: &Fingerprint) -> Fingerprint {
        // Find best matching chunk above threshold
        let mut best_chunk = state.clone();
        let mut best_match = 0.0f32;

        for (i, chunk) in self.chunks.iter().enumerate() {
            let activation = self.activations.get(i).copied().unwrap_or(0.0);
            let similarity = state.similarity(chunk);
            let match_strength = similarity * (1.0 + activation);

            if match_strength > best_match && match_strength > self.threshold {
                best_match = match_strength;
                best_chunk = chunk.clone();
            }
        }

        best_chunk
    }

    fn is_hermitian(&self) -> bool {
        true
    }

    fn adjoint(&self) -> Box<dyn QuantumOp> {
        Box::new(ActrRetrievalOp {
            chunks: self.chunks.clone(),
            activations: self.activations.clone(),
            threshold: self.threshold,
        })
    }
}

/// RL value operator - transforms state by value function
pub struct RlValueOp {
    /// Q-table: (state_hash, action_hash) -> value
    q_values: std::collections::HashMap<u64, f32>,
    /// Temperature for softmax
    temperature: f32,
}

impl RlValueOp {
    pub fn new(temperature: f32) -> Self {
        Self {
            q_values: std::collections::HashMap::new(),
            temperature,
        }
    }

    pub fn set_q(&mut self, state: &Fingerprint, action: &Fingerprint, value: f32) {
        let key = fp_hash(state) ^ fp_hash(action);
        self.q_values.insert(key, value);
    }

    pub fn get_q(&self, state: &Fingerprint, action: &Fingerprint) -> f32 {
        let key = fp_hash(state) ^ fp_hash(action);
        self.q_values.get(&key).copied().unwrap_or(0.0)
    }
}

impl QuantumOp for RlValueOp {
    fn apply(&self, state: &Fingerprint) -> Fingerprint {
        // Weight state by its value
        let state_hash = fp_hash(state);
        let value = self.q_values.get(&state_hash).copied().unwrap_or(0.0);

        // High-value states get amplified, low-value states get suppressed
        if value > 0.0 {
            state.clone()
        } else {
            // Suppress by mixing with zero
            blend_fingerprints(state, &Fingerprint::zero(), (-value).min(1.0))
        }
    }

    fn is_hermitian(&self) -> bool {
        true
    }

    fn adjoint(&self) -> Box<dyn QuantumOp> {
        Box::new(RlValueOp {
            q_values: self.q_values.clone(),
            temperature: self.temperature,
        })
    }
}

/// Causal do-operator
pub struct CausalDoOp {
    /// Variable to intervene on
    variable: Fingerprint,
    /// Value to set
    value: Fingerprint,
}

impl CausalDoOp {
    pub fn new(variable: Fingerprint, value: Fingerprint) -> Self {
        Self { variable, value }
    }
}

impl QuantumOp for CausalDoOp {
    fn apply(&self, state: &Fingerprint) -> Fingerprint {
        // Cut incoming edges to variable, set to value
        // Implemented as: remove variable component, add value component

        // Unbind variable from state
        let without_var = state.bind(&self.variable);

        // Bind in new value
        without_var.bind(&self.value)
    }

    fn is_hermitian(&self) -> bool {
        false
    }

    fn adjoint(&self) -> Box<dyn QuantumOp> {
        // Reverse intervention: restore variable
        Box::new(CausalDoOp {
            variable: self.value.clone(),
            value: self.variable.clone(),
        })
    }
}

/// Qualia shift operator - moves state in 8D affect space
pub struct QualiaShiftOp {
    /// Shift in each channel
    shifts: [f32; 8],
}

impl QualiaShiftOp {
    pub fn new(shifts: [f32; 8]) -> Self {
        Self { shifts }
    }

    pub fn arousal(delta: f32) -> Self {
        let mut shifts = [0.0; 8];
        shifts[0] = delta;
        Self { shifts }
    }

    pub fn valence(delta: f32) -> Self {
        let mut shifts = [0.0; 8];
        shifts[1] = delta;
        Self { shifts }
    }
}

impl QuantumOp for QualiaShiftOp {
    fn apply(&self, state: &Fingerprint) -> Fingerprint {
        // Shift state in qualia space by permuting bits
        let mut result = state.clone();

        for (i, &shift) in self.shifts.iter().enumerate() {
            if shift.abs() > 0.01 {
                let shift_amount = (shift * 100.0) as i32;
                let channel_mask = channel_fingerprint(i);

                // Extract channel bits, shift, reinsert
                let channel_bits = state.and(&channel_mask);
                let shifted = channel_bits.permute(shift_amount);

                // Blend shifted channel back
                result = blend_fingerprints(&result, &shifted, shift.abs().min(1.0));
            }
        }

        result
    }

    fn is_hermitian(&self) -> bool {
        false
    }

    fn adjoint(&self) -> Box<dyn QuantumOp> {
        let neg_shifts: [f32; 8] = std::array::from_fn(|i| -self.shifts[i]);
        Box::new(QualiaShiftOp { shifts: neg_shifts })
    }
}

/// Rung ladder operator - ascend/descend abstraction levels
pub struct RungLadderOp {
    /// Direction: +1 = ascend, -1 = descend
    direction: i8,
}

impl RungLadderOp {
    pub fn ascend() -> Self {
        Self { direction: 1 }
    }

    pub fn descend() -> Self {
        Self { direction: -1 }
    }
}

impl QuantumOp for RungLadderOp {
    fn apply(&self, state: &Fingerprint) -> Fingerprint {
        // Ascend: increase abstraction (bundle similar concepts)
        // Descend: decrease abstraction (add specific details)

        if self.direction > 0 {
            // Ascend: smooth out details
            let smoothed = smooth_fingerprint(state);
            smoothed
        } else {
            // Descend: add noise (specificity)
            let noise = Fingerprint::from_content(&format!("DETAIL_{}", fp_hash(state)));
            blend_fingerprints(state, &noise, 0.3)
        }
    }

    fn is_hermitian(&self) -> bool {
        false
    }

    fn adjoint(&self) -> Box<dyn QuantumOp> {
        Box::new(RungLadderOp {
            direction: -self.direction,
        })
    }
}

// =============================================================================
// OPERATOR ALGEBRA
// =============================================================================

/// Compose two operators: (ÂB̂)|ψ⟩ = Â(B̂|ψ⟩)
pub struct ComposedOp {
    first: Box<dyn QuantumOp>,
    second: Box<dyn QuantumOp>,
}

impl ComposedOp {
    pub fn new(first: Box<dyn QuantumOp>, second: Box<dyn QuantumOp>) -> Self {
        Self { first, second }
    }
}

impl QuantumOp for ComposedOp {
    fn apply(&self, state: &Fingerprint) -> Fingerprint {
        let intermediate = self.first.apply(state);
        self.second.apply(&intermediate)
    }

    fn is_hermitian(&self) -> bool {
        // Composition of Hermitian operators is Hermitian iff they commute
        false
    }

    fn adjoint(&self) -> Box<dyn QuantumOp> {
        // (ÂB̂)† = B̂†Â†
        Box::new(ComposedOp {
            first: self.second.adjoint(),
            second: self.first.adjoint(),
        })
    }
}

/// Sum of operators: (Â + B̂)|ψ⟩ = Â|ψ⟩ + B̂|ψ⟩
pub struct SumOp {
    ops: Vec<Box<dyn QuantumOp>>,
    weights: Vec<f32>,
}

impl SumOp {
    pub fn new(ops: Vec<Box<dyn QuantumOp>>, weights: Vec<f32>) -> Self {
        Self { ops, weights }
    }
}

impl QuantumOp for SumOp {
    fn apply(&self, state: &Fingerprint) -> Fingerprint {
        // Weighted superposition of results
        let results: Vec<Fingerprint> = self.ops.iter().map(|op| op.apply(state)).collect();

        weighted_bundle(&results, &self.weights)
    }

    fn is_hermitian(&self) -> bool {
        self.ops.iter().all(|op| op.is_hermitian())
    }

    fn adjoint(&self) -> Box<dyn QuantumOp> {
        let adj_ops: Vec<_> = self.ops.iter().map(|op| op.adjoint()).collect();
        Box::new(SumOp {
            ops: adj_ops,
            weights: self.weights.clone(),
        })
    }
}

/// Tensor product: Â ⊗ B̂ (acts on different subspaces)
pub struct TensorOp {
    op_a: Box<dyn QuantumOp>,
    op_b: Box<dyn QuantumOp>,
    /// Bit ranges for each subspace
    range_a: (usize, usize),
    range_b: (usize, usize),
}

impl TensorOp {
    pub fn new(
        op_a: Box<dyn QuantumOp>,
        op_b: Box<dyn QuantumOp>,
        range_a: (usize, usize),
        range_b: (usize, usize),
    ) -> Self {
        Self {
            op_a,
            op_b,
            range_a,
            range_b,
        }
    }
}

impl QuantumOp for TensorOp {
    fn apply(&self, state: &Fingerprint) -> Fingerprint {
        // Extract subspaces, apply operators, recombine
        let mut result = state.clone();

        // Apply op_a to range_a
        let sub_a = extract_bits(state, self.range_a.0, self.range_a.1);
        let new_a = self.op_a.apply(&sub_a);
        result = insert_bits(&result, &new_a, self.range_a.0, self.range_a.1);

        // Apply op_b to range_b
        let sub_b = extract_bits(state, self.range_b.0, self.range_b.1);
        let new_b = self.op_b.apply(&sub_b);
        result = insert_bits(&result, &new_b, self.range_b.0, self.range_b.1);

        result
    }

    fn is_hermitian(&self) -> bool {
        self.op_a.is_hermitian() && self.op_b.is_hermitian()
    }

    fn adjoint(&self) -> Box<dyn QuantumOp> {
        Box::new(TensorOp {
            op_a: self.op_a.adjoint(),
            op_b: self.op_b.adjoint(),
            range_a: self.range_a,
            range_b: self.range_b,
        })
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

fn fp_hash(fp: &Fingerprint) -> u64 {
    let raw = fp.as_raw();
    let mut hash = 0u64;
    for &word in raw.iter() {
        hash ^= word;
    }
    hash
}

fn blend_fingerprints(a: &Fingerprint, b: &Fingerprint, weight: f32) -> Fingerprint {
    let threshold = (weight * crate::FINGERPRINT_BITS as f32) as usize;
    let mut result = a.clone();

    for bit in 0..crate::FINGERPRINT_BITS {
        if bit < threshold {
            result.set_bit(bit, b.get_bit(bit));
        }
    }

    result
}

fn weighted_bundle(fps: &[Fingerprint], weights: &[f32]) -> Fingerprint {
    if fps.is_empty() {
        return Fingerprint::zero();
    }

    let mut result = Fingerprint::zero();
    let total_weight: f32 = weights.iter().sum();

    for bit in 0..10000 {
        let mut weighted_sum = 0.0f32;
        for (fp, &w) in fps.iter().zip(weights) {
            if fp.get_bit(bit) {
                weighted_sum += w;
            }
        }
        if weighted_sum > total_weight / 2.0 {
            result.set_bit(bit, true);
        }
    }

    result
}

fn channel_fingerprint(channel: usize) -> Fingerprint {
    Fingerprint::from_content(&format!("QUALIA_CHANNEL_{}", channel))
}

fn smooth_fingerprint(fp: &Fingerprint) -> Fingerprint {
    // Average nearby bits (low-pass filter)
    let mut result = Fingerprint::zero();

    for bit in 0..crate::FINGERPRINT_BITS {
        let mut count = 0;
        for offset in -5i32..=5 {
            let idx = ((bit as i32 + offset + crate::FINGERPRINT_BITS as i32)
                % crate::FINGERPRINT_BITS as i32) as usize;
            if fp.get_bit(idx) {
                count += 1;
            }
        }
        if count > 5 {
            result.set_bit(bit, true);
        }
    }

    result
}

fn extract_bits(fp: &Fingerprint, start: usize, end: usize) -> Fingerprint {
    let mut result = Fingerprint::zero();
    for i in start..end.min(crate::FINGERPRINT_BITS) {
        result.set_bit(i - start, fp.get_bit(i));
    }
    result
}

fn insert_bits(fp: &Fingerprint, bits: &Fingerprint, start: usize, end: usize) -> Fingerprint {
    let mut result = fp.clone();
    for i in start..end.min(crate::FINGERPRINT_BITS) {
        result.set_bit(i, bits.get_bit(i - start));
    }
    result
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_addr_basic() {
        let root = TreeAddr::root();
        assert_eq!(root.depth(), 0);

        let child = root.child(0x01);
        assert_eq!(child.depth(), 1);
        assert_eq!(child.component(0), Some(0x01));

        let grandchild = child.child(0x02);
        assert_eq!(grandchild.depth(), 2);
    }

    #[test]
    fn test_tree_addr_from_string() {
        let addr = TreeAddr::from_string("/concepts/animals/cats");
        assert_eq!(addr.depth(), 3);
    }

    #[test]
    fn test_tree_addr_parent() {
        let addr = TreeAddr::from_path(&[0x01, 0x02, 0x03]);
        let parent = addr.parent().unwrap();
        assert_eq!(parent.depth(), 2);

        let grandparent = parent.parent().unwrap();
        assert_eq!(grandparent.depth(), 1);
    }

    #[test]
    fn test_identity_op() {
        let state = Fingerprint::from_content("test");
        let op = IdentityOp;

        let result = op.apply(&state);
        assert_eq!(state, result);
    }

    #[test]
    fn test_not_op_self_inverse() {
        let state = Fingerprint::from_content("test");
        let op = NotOp;

        let result = op.apply(&op.apply(&state));
        assert_eq!(state, result);
    }

    #[test]
    fn test_bind_op_self_inverse() {
        let state = Fingerprint::from_content("test");
        let operand = Fingerprint::from_content("key");
        let op = BindOp::new(operand);

        let result = op.apply(&op.apply(&state));
        assert_eq!(state, result);
    }

    #[test]
    fn test_permute_adjoint() {
        let state = Fingerprint::from_content("test");
        let op = PermuteOp::new(100);
        let adj = op.adjoint();

        let result = adj.apply(&op.apply(&state));
        assert_eq!(state, result);
    }

    #[test]
    fn test_composed_op() {
        let state = Fingerprint::from_content("test");
        let op1 = Box::new(PermuteOp::new(10));
        let op2 = Box::new(PermuteOp::new(20));

        let composed = ComposedOp::new(op1, op2);
        let result = composed.apply(&state);

        // Should equal permute by 30
        let direct = PermuteOp::new(30).apply(&state);
        assert_eq!(result, direct);
    }

    #[test]
    fn test_measurement_collapse() {
        let state = Fingerprint::from_content("superposition");
        let eigenstates = vec![
            Fingerprint::from_content("cat"),
            Fingerprint::from_content("dog"),
            Fingerprint::from_content("bird"),
        ];

        let measure = MeasureOp::new(eigenstates.clone());
        let collapsed = measure.apply(&state);

        // Should collapse to one of the eigenstates
        let is_eigenstate = eigenstates.iter().any(|e| e.similarity(&collapsed) > 0.99);
        assert!(is_eigenstate);
    }

    #[test]
    fn test_commutator() {
        let state = Fingerprint::from_content("test");

        // Identity commutes with everything
        let identity = IdentityOp;
        let permute = PermuteOp::new(10);

        assert!(identity.commutes_with(&permute, &state));
    }
}
