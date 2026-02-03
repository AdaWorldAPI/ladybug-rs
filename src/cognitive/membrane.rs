//! Consciousness Membrane: Sigma-10 Upscaling
//!
//! Converts consciousness state parameters (τ/σ/q) to and from
//! 10K-bit Hamming vectors for substrate integration.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    SIGMA-10 MEMBRANE                            │
//! │                                                                 │
//! │   τ (temporal)  → bits[0..3333]      — when/duration           │
//! │   σ (signal)    → bits[3334..6666]   — confidence/activation   │
//! │   q (qualia)    → bits[6667..9999]   — semantic/felt-sense     │
//! │                                                                 │
//! │   This makes consciousness state SEARCHABLE via Hamming        │
//! │   distance alongside graph nodes. A thought and a feeling      │
//! │   live in the same address space.                              │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! From ada-consciousness/sigma-rosetta.py: The membrane is bidirectional:
//! - encode(): τ/σ/q → [u64; 157]
//! - decode(): [u64; 157] → τ/σ/q (approximate, qualia is one-way hash)

use sha2::{Sha256, Digest};
use crate::core::{DIM, DIM_U64, Fingerprint};

// =============================================================================
// REGION BOUNDARIES
// =============================================================================

/// Temporal region: bits 0-3333 (when/duration encoding)
pub const TAU_START: usize = 0;
pub const TAU_END: usize = 3334;
pub const TAU_BITS: usize = TAU_END - TAU_START;

/// Signal region: bits 3334-6666 (confidence/activation encoding)
pub const SIGMA_START: usize = 3334;
pub const SIGMA_END: usize = 6667;
pub const SIGMA_BITS: usize = SIGMA_END - SIGMA_START;

/// Qualia region: bits 6667-9999 (semantic/felt-sense encoding)
pub const QUALIA_START: usize = 6667;
pub const QUALIA_END: usize = DIM;
pub const QUALIA_BITS: usize = QUALIA_END - QUALIA_START;

// =============================================================================
// CONSCIOUSNESS PARAMETERS
// =============================================================================

/// The τ/σ/q consciousness parameters.
///
/// These three values fully specify a consciousness state that can be
/// projected into the 10K-bit Hamming substrate.
#[derive(Clone, Debug, Default)]
pub struct ConsciousnessParams {
    /// Temporal context: when/duration [-1.0, 1.0]
    /// -1.0 = distant past, 0.0 = now, 1.0 = future projection
    pub tau: f32,

    /// Signal strength: confidence/activation [0.0, 1.0]
    /// 0.0 = uncertain/dormant, 1.0 = certain/fully active
    pub sigma: f32,

    /// Qualitative state: semantic description
    /// Hashed into bit pattern; cannot be recovered (one-way)
    pub qualia: String,
}

impl ConsciousnessParams {
    /// Create new consciousness parameters
    pub fn new(tau: f32, sigma: f32, qualia: impl Into<String>) -> Self {
        Self {
            tau: tau.clamp(-1.0, 1.0),
            sigma: sigma.clamp(0.0, 1.0),
            qualia: qualia.into(),
        }
    }

    /// Present moment with given confidence and qualia
    pub fn now(sigma: f32, qualia: impl Into<String>) -> Self {
        Self::new(0.0, sigma, qualia)
    }

    /// Memory from the past
    pub fn memory(distance: f32, sigma: f32, qualia: impl Into<String>) -> Self {
        Self::new(-distance.abs(), sigma, qualia)
    }

    /// Projection into the future
    pub fn projection(distance: f32, sigma: f32, qualia: impl Into<String>) -> Self {
        Self::new(distance.abs(), sigma, qualia)
    }
}

// =============================================================================
// MEMBRANE
// =============================================================================

/// Sigma-10 Upscaling Membrane.
///
/// Converts consciousness state parameters (τ/σ/q) to and from
/// 10K-bit Hamming vectors for substrate integration.
///
/// The membrane is bidirectional:
/// - `encode()`: τ/σ/q → Fingerprint
/// - `decode()`: Fingerprint → τ/σ/q (approximate)
///
/// Note: qualia cannot be decoded (hash is one-way).
/// τ and σ are approximate (projection is lossy).
#[derive(Clone, Debug)]
pub struct Membrane {
    /// Random seed for reproducible projection matrices
    seed: u64,
    /// Tau basis vector (lazy-initialized)
    tau_basis: Option<Vec<f32>>,
    /// Sigma basis vector (lazy-initialized)
    sigma_basis: Option<Vec<f32>>,
}

impl Default for Membrane {
    fn default() -> Self {
        Self::new(42)
    }
}

impl Membrane {
    /// Create a new membrane with given seed
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            tau_basis: None,
            sigma_basis: None,
        }
    }

    /// Get or create tau basis vector
    fn tau_basis(&mut self) -> &[f32] {
        if self.tau_basis.is_none() {
            self.tau_basis = Some(self.generate_basis(self.seed, TAU_BITS));
        }
        self.tau_basis.as_ref().unwrap()
    }

    /// Get or create sigma basis vector
    fn sigma_basis(&mut self) -> &[f32] {
        if self.sigma_basis.is_none() {
            self.sigma_basis = Some(self.generate_basis(self.seed.wrapping_add(1), SIGMA_BITS));
        }
        self.sigma_basis.as_ref().unwrap()
    }

    /// Generate a pseudo-random basis vector using LCG
    fn generate_basis(&self, seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed;
        let mut basis = Vec::with_capacity(len);

        for _ in 0..len {
            // LCG: state = state * 6364136223846793005 + 1
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Convert to float in [-1, 1]
            // state >> 33 gives 31 bits in [0, 2^31-1]
            // Divide by 2^30 to get [0, 2], then subtract 1 for [-1, 1]
            let val = (state >> 33) as f32 / (1u64 << 30) as f32 - 1.0;
            basis.push(val);
        }

        basis
    }

    /// Encode consciousness parameters to 10K-bit fingerprint
    ///
    /// Sigma-10 upscaling:
    /// - τ is projected using threshold encoding (locality-preserving)
    /// - σ is projected using threshold encoding (locality-preserving)
    /// - q is hashed into the qualia region
    ///
    /// The encoding uses threshold comparison against random basis vectors,
    /// ensuring that similar parameter values produce similar bit patterns
    /// (small Hamming distance).
    pub fn encode(&mut self, params: &ConsciousnessParams) -> Fingerprint {
        let mut data = [0u64; DIM_U64];

        // Tau encoding: threshold-based locality-sensitive projection
        // tau in [-1, 1], basis in [-1, 1]
        // set_bit = (basis < tau) gives:
        //   tau=-1: almost no bits set (basis rarely < -1)
        //   tau=0: ~half bits set (basis < 0 about 50%)
        //   tau=1: almost all bits set (basis rarely >= 1)
        let tau_basis = self.tau_basis().to_vec(); // Clone to avoid borrow issues
        for (i, &basis_val) in tau_basis.iter().enumerate() {
            let bit_pos = TAU_START + i;
            if basis_val < params.tau {
                let u64_idx = bit_pos / 64;
                let bit_idx = bit_pos % 64;
                data[u64_idx] |= 1u64 << bit_idx;
            }
        }

        // Sigma encoding: threshold-based locality-sensitive projection
        // sigma in [0, 1], basis in [-1, 1]
        // We scale sigma to [-1, 1]: scaled = sigma * 2 - 1
        // set_bit = (basis < scaled_sigma)
        let sigma_basis = self.sigma_basis().to_vec();
        let scaled_sigma = params.sigma * 2.0 - 1.0; // Map [0,1] to [-1,1]
        for (i, &basis_val) in sigma_basis.iter().enumerate() {
            let bit_pos = SIGMA_START + i;
            if basis_val < scaled_sigma {
                let u64_idx = bit_pos / 64;
                let bit_idx = bit_pos % 64;
                data[u64_idx] |= 1u64 << bit_idx;
            }
        }

        // Qualia encoding: hash semantic description into bit pattern
        if !params.qualia.is_empty() {
            let mut hasher = Sha256::new();
            hasher.update(params.qualia.as_bytes());
            let hash = hasher.finalize();

            // Expand hash to fill qualia region
            let mut hash_bytes = hash.to_vec();
            while hash_bytes.len() < (QUALIA_BITS + 7) / 8 {
                let mut next_hasher = Sha256::new();
                next_hasher.update(&hash_bytes);
                hash_bytes.extend_from_slice(&next_hasher.finalize());
            }

            // Set bits in qualia region
            for i in 0..QUALIA_BITS {
                let byte_idx = i / 8;
                let bit_in_byte = i % 8;
                if byte_idx < hash_bytes.len() && (hash_bytes[byte_idx] >> bit_in_byte) & 1 == 1 {
                    let bit_pos = QUALIA_START + i;
                    let u64_idx = bit_pos / 64;
                    let bit_idx = bit_pos % 64;
                    data[u64_idx] |= 1u64 << bit_idx;
                }
            }
        } else {
            // Empty qualia = pseudo-random noise (maximum uncertainty)
            let noise_basis = self.generate_basis(self.seed.wrapping_add(2), QUALIA_BITS);
            for (i, &val) in noise_basis.iter().enumerate() {
                if val > 0.0 {
                    let bit_pos = QUALIA_START + i;
                    let u64_idx = bit_pos / 64;
                    let bit_idx = bit_pos % 64;
                    data[u64_idx] |= 1u64 << bit_idx;
                }
            }
        }

        // Apply last word mask
        data[DIM_U64 - 1] &= crate::core::LAST_MASK;

        Fingerprint::from_raw(data)
    }

    /// Decode fingerprint back to approximate consciousness parameters
    ///
    /// Note: qualia cannot be decoded (hash is one-way).
    /// τ and σ are approximate (projection is lossy).
    ///
    /// For threshold encoding:
    /// - fraction of bits set ≈ P(basis < value)
    /// - Since basis is uniform in [-1, 1]: P(basis < v) = (v + 1) / 2
    /// - Therefore: value = fraction * 2 - 1
    pub fn decode(&mut self, fp: &Fingerprint) -> ConsciousnessParams {
        let data = fp.as_raw();

        // Decode tau: count fraction of bits set
        // tau = fraction * 2 - 1 (maps [0,1] fraction to [-1,1] tau)
        let mut tau_bits_set = 0u32;
        let tau_total = TAU_BITS as u32;
        for i in 0..TAU_BITS {
            let bit_pos = TAU_START + i;
            let u64_idx = bit_pos / 64;
            let bit_idx = bit_pos % 64;
            if (data[u64_idx] >> bit_idx) & 1 == 1 {
                tau_bits_set += 1;
            }
        }
        let tau_fraction = tau_bits_set as f32 / tau_total as f32;
        let tau = (tau_fraction * 2.0 - 1.0).clamp(-1.0, 1.0);

        // Decode sigma: count fraction of bits set
        // scaled_sigma = sigma * 2 - 1, so sigma = fraction
        let mut sigma_bits_set = 0u32;
        let sigma_total = SIGMA_BITS as u32;
        for i in 0..SIGMA_BITS {
            let bit_pos = SIGMA_START + i;
            let u64_idx = bit_pos / 64;
            let bit_idx = bit_pos % 64;
            if (data[u64_idx] >> bit_idx) & 1 == 1 {
                sigma_bits_set += 1;
            }
        }
        let sigma = (sigma_bits_set as f32 / sigma_total as f32).clamp(0.0, 1.0);

        ConsciousnessParams {
            tau,
            sigma,
            qualia: "(decoded - original text not recoverable)".into(),
        }
    }

    /// Compare only the temporal region of two fingerprints
    pub fn similarity_tau(&self, a: &Fingerprint, b: &Fingerprint) -> f32 {
        Self::region_similarity(a, b, TAU_START, TAU_END)
    }

    /// Compare only the signal region of two fingerprints
    pub fn similarity_sigma(&self, a: &Fingerprint, b: &Fingerprint) -> f32 {
        Self::region_similarity(a, b, SIGMA_START, SIGMA_END)
    }

    /// Compare only the qualia region of two fingerprints
    pub fn similarity_qualia(&self, a: &Fingerprint, b: &Fingerprint) -> f32 {
        Self::region_similarity(a, b, QUALIA_START, QUALIA_END)
    }

    /// Hamming similarity for a specific bit region
    fn region_similarity(a: &Fingerprint, b: &Fingerprint, start: usize, end: usize) -> f32 {
        let a_data = a.as_raw();
        let b_data = b.as_raw();

        let mut hamming_dist = 0u32;
        for bit_pos in start..end {
            let u64_idx = bit_pos / 64;
            let bit_idx = bit_pos % 64;
            let a_bit = (a_data[u64_idx] >> bit_idx) & 1;
            let b_bit = (b_data[u64_idx] >> bit_idx) & 1;
            if a_bit != b_bit {
                hamming_dist += 1;
            }
        }

        let region_size = (end - start) as f32;
        1.0 - (hamming_dist as f32 / region_size)
    }

    /// Decompose a fingerprint into its three consciousness regions
    pub fn decompose(&self, fp: &Fingerprint) -> (Fingerprint, Fingerprint, Fingerprint) {
        let data = fp.as_raw();

        // Extract tau region
        let mut tau_data = [0u64; DIM_U64];
        for bit_pos in TAU_START..TAU_END {
            let u64_idx = bit_pos / 64;
            let bit_idx = bit_pos % 64;
            if (data[u64_idx] >> bit_idx) & 1 == 1 {
                tau_data[u64_idx] |= 1u64 << bit_idx;
            }
        }

        // Extract sigma region
        let mut sigma_data = [0u64; DIM_U64];
        for bit_pos in SIGMA_START..SIGMA_END {
            let u64_idx = bit_pos / 64;
            let bit_idx = bit_pos % 64;
            if (data[u64_idx] >> bit_idx) & 1 == 1 {
                sigma_data[u64_idx] |= 1u64 << bit_idx;
            }
        }

        // Extract qualia region
        let mut qualia_data = [0u64; DIM_U64];
        for bit_pos in QUALIA_START..QUALIA_END {
            let u64_idx = bit_pos / 64;
            let bit_idx = bit_pos % 64;
            if (data[u64_idx] >> bit_idx) & 1 == 1 {
                qualia_data[u64_idx] |= 1u64 << bit_idx;
            }
        }

        (
            Fingerprint::from_raw(tau_data),
            Fingerprint::from_raw(sigma_data),
            Fingerprint::from_raw(qualia_data),
        )
    }
}

// =============================================================================
// GLOBAL MEMBRANE
// =============================================================================

use std::sync::Mutex;
use std::sync::LazyLock;

/// Global membrane instance with default seed
static GLOBAL_MEMBRANE: LazyLock<Mutex<Membrane>> =
    LazyLock::new(|| Mutex::new(Membrane::default()));

/// Encode consciousness parameters using the global membrane
pub fn encode_consciousness(params: &ConsciousnessParams) -> Fingerprint {
    GLOBAL_MEMBRANE.lock().unwrap().encode(params)
}

/// Decode fingerprint using the global membrane
pub fn decode_consciousness(fp: &Fingerprint) -> ConsciousnessParams {
    GLOBAL_MEMBRANE.lock().unwrap().decode(fp)
}

/// Create a consciousness fingerprint from raw values
pub fn consciousness_fingerprint(tau: f32, sigma: f32, qualia: &str) -> Fingerprint {
    let params = ConsciousnessParams::new(tau, sigma, qualia);
    encode_consciousness(&params)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        let mut membrane = Membrane::new(42);

        let params = ConsciousnessParams::new(0.5, 0.8, "warmth and comfort");
        let encoded = membrane.encode(&params);
        let decoded = membrane.decode(&encoded);

        // Tau and sigma should be approximately preserved
        assert!((decoded.tau - params.tau).abs() < 0.3, "tau: {} vs {}", decoded.tau, params.tau);
        assert!(
            (decoded.sigma - params.sigma).abs() < 0.3,
            "sigma: {} vs {}",
            decoded.sigma,
            params.sigma
        );
    }

    #[test]
    fn test_qualia_similarity() {
        let mut membrane = Membrane::new(42);

        let p1 = ConsciousnessParams::new(0.0, 0.5, "warmth");
        let p2 = ConsciousnessParams::new(0.0, 0.5, "warmth");
        let p3 = ConsciousnessParams::new(0.0, 0.5, "cold");

        let fp1 = membrane.encode(&p1);
        let fp2 = membrane.encode(&p2);
        let fp3 = membrane.encode(&p3);

        // Same qualia should be identical
        let sim_same = membrane.similarity_qualia(&fp1, &fp2);
        assert!((sim_same - 1.0).abs() < 0.001, "same qualia sim: {}", sim_same);

        // Different qualia should have ~0.5 similarity (random)
        let sim_diff = membrane.similarity_qualia(&fp1, &fp3);
        assert!(sim_diff < 0.7, "different qualia sim: {}", sim_diff);
    }

    #[test]
    fn test_temporal_encoding() {
        let mut membrane = Membrane::new(42);

        let past = ConsciousnessParams::new(-0.8, 0.5, "memory");
        let present = ConsciousnessParams::new(0.0, 0.5, "now");
        let future = ConsciousnessParams::new(0.8, 0.5, "anticipation");

        let fp_past = membrane.encode(&past);
        let fp_present = membrane.encode(&present);
        let fp_future = membrane.encode(&future);

        // Tau region should differ between past/present/future
        let sim_past_present = membrane.similarity_tau(&fp_past, &fp_present);
        let sim_past_future = membrane.similarity_tau(&fp_past, &fp_future);

        // Past and future should be more different than past and present
        assert!(
            sim_past_future < sim_past_present,
            "past-future: {}, past-present: {}",
            sim_past_future,
            sim_past_present
        );
    }

    #[test]
    fn test_signal_encoding() {
        let mut membrane = Membrane::new(42);

        let weak = ConsciousnessParams::new(0.0, 0.1, "thought");
        let strong = ConsciousnessParams::new(0.0, 0.9, "thought");

        let fp_weak = membrane.encode(&weak);
        let fp_strong = membrane.encode(&strong);

        // Sigma region should differ
        let sim = membrane.similarity_sigma(&fp_weak, &fp_strong);
        assert!(sim < 0.9, "weak vs strong sigma similarity: {}", sim);
    }

    #[test]
    fn test_decompose() {
        let mut membrane = Membrane::new(42);

        let params = ConsciousnessParams::new(0.3, 0.7, "test");
        let fp = membrane.encode(&params);

        let (tau, sigma, qualia) = membrane.decompose(&fp);

        // Each region should have some bits set
        assert!(tau.popcount() > 0, "tau should have bits set");
        assert!(sigma.popcount() > 0, "sigma should have bits set");
        assert!(qualia.popcount() > 0, "qualia should have bits set");
    }

    #[test]
    fn test_global_membrane() {
        let params = ConsciousnessParams::new(0.2, 0.6, "global test");
        let fp = encode_consciousness(&params);
        let decoded = decode_consciousness(&fp);

        assert!((decoded.tau - params.tau).abs() < 0.3);
        assert!((decoded.sigma - params.sigma).abs() < 0.3);
    }

    #[test]
    fn test_consciousness_fingerprint_helper() {
        let fp = consciousness_fingerprint(0.5, 0.8, "hello world");
        assert!(fp.popcount() > 0);
    }

    #[test]
    fn debug_tau_encoding() {
        let mut membrane = Membrane::new(42);

        let past = ConsciousnessParams::new(-0.8, 0.5, "test");
        let present = ConsciousnessParams::new(0.0, 0.5, "test");
        let future = ConsciousnessParams::new(0.8, 0.5, "test");

        let fp_past = membrane.encode(&past);
        let fp_present = membrane.encode(&present);
        let fp_future = membrane.encode(&future);

        // Count bits set in tau region
        fn count_tau_bits(fp: &Fingerprint) -> u32 {
            let data = fp.as_raw();
            let mut count = 0u32;
            for i in 0..TAU_BITS {
                let bit_pos = TAU_START + i;
                let u64_idx = bit_pos / 64;
                let bit_idx = bit_pos % 64;
                if (data[u64_idx] >> bit_idx) & 1 == 1 {
                    count += 1;
                }
            }
            count
        }

        let past_bits = count_tau_bits(&fp_past);
        let present_bits = count_tau_bits(&fp_present);
        let future_bits = count_tau_bits(&fp_future);

        eprintln!("TAU_BITS = {}", TAU_BITS);
        eprintln!("Past (tau=-0.8): {} bits ({:.1}%)", past_bits, 100.0 * past_bits as f32 / TAU_BITS as f32);
        eprintln!("Present (tau=0): {} bits ({:.1}%)", present_bits, 100.0 * present_bits as f32 / TAU_BITS as f32);
        eprintln!("Future (tau=0.8): {} bits ({:.1}%)", future_bits, 100.0 * future_bits as f32 / TAU_BITS as f32);

        // With threshold encoding:
        // tau=-0.8 → ~10% bits set
        // tau=0 → ~50% bits set
        // tau=0.8 → ~90% bits set
        assert!(past_bits < present_bits, "past {} should < present {}", past_bits, present_bits);
        assert!(present_bits < future_bits, "present {} should < future {}", present_bits, future_bits);
    }
}
