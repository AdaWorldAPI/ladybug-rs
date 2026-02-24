//! Query→NARS Feedback Loop — closes the learning circuit.
//!
//! This module bridges the gap between recognition results and NARS truth
//! value updates. Previously, query results didn't feed back to update
//! evidence counts or revise truth values. This module wires:
//!
//! ```text
//! hybrid_pipeline() → HybridScore (matches)
//!   → from_energy_conflict() → TruthValue (agreement_ratio → frequency)
//!   → from_learning_signal() → evidence history + hybrid weights
//!   → apply_feedback() → WideMetaViewMut (persists to container metadata)
//!
//! BF16 prefix decomposition (1+7+8):
//!   → sign flip count → NARS negation signal (class-level cause)
//!   → exponent shift → NARS abduction signal (magnitude-level cause)
//!   → mantissa noise → masked out (irrelevant for learning)
//! ```
//!
//! The 4096 CAM ops in ladybug-rs already include Cypher and NARS as part
//! of the unified cognitive vocabulary. The RL schema lives in LanceDB via
//! BindSpace → DataFusion. This module wires the awareness substrate from
//! rustynum's hybrid pipeline into that existing NARS infrastructure.

use crate::learning::cognitive_frameworks::{NarsInference, TruthValue};

// ============================================================================
// Feedback Signal — the bridge struct
// ============================================================================

/// A feedback signal produced from recognition results.
///
/// Carries everything needed to update NARS truth values, RL Q-values,
/// and hybrid weights in a single pass over the metadata.
#[derive(Clone, Debug)]
pub struct FeedbackSignal {
    /// NARS truth value derived from match quality.
    pub truth_value: TruthValue,
    /// Evidence counts: (positive, negative).
    pub evidence: (f32, f32),
    /// Confidence gain from this recognition cycle.
    pub confidence_delta: f32,
    /// Evidence decay rate (0.0 = no decay, 1.0 = full decay per cycle).
    pub decay_rate: f32,
    /// Per-group hybrid weights (32 slots, matching WideMetaView W144-W159).
    pub hybrid_weights: [f32; 32],
    /// BF16 prefix decomposition results for causal learning.
    pub bf16_causal: Bf16CausalSignal,
    /// RL reward signal derived from recognition quality.
    pub rl_reward: f32,
    /// Which RL action slot to update (0-15).
    pub rl_action: usize,
}

/// BF16 prefix decomposition for causal learning.
///
/// The 1+7+8 bit layout of BF16 (sign + exponent + mantissa) encodes
/// three distinct causal signals:
///
/// - **Sign flips** = class-level cause (polarity reversal → negation)
/// - **Exponent shifts** = attention-level cause (magnitude change → abduction)
/// - **Mantissa noise** = irrelevant noise (mask out)
///
/// This decomposition maps directly to NARS inference rules:
/// - sign_flip → NarsInference::negation() (the dimension flipped polarity)
/// - exp_shift → NarsInference::abduction() (the cause changed magnitude)
/// - mantissa_only → no update (noise, not signal)
#[derive(Clone, Debug, Default)]
pub struct Bf16CausalSignal {
    /// Number of dimensions where the sign flipped.
    pub sign_flips: usize,
    /// Number of exponent bits that changed (total across all dims).
    pub exponent_shifts: usize,
    /// Number of mantissa-only changes (noise).
    pub mantissa_noise: usize,
    /// Total BF16 dimensions evaluated.
    pub total_dims: usize,
    /// Ratio of crystallized dimensions (settled knowledge).
    pub crystallized_ratio: f32,
    /// Ratio of tensioned dimensions (contradictory evidence → needs revision).
    pub tension_ratio: f32,
    /// Inferred NARS operation from the BF16 decomposition.
    pub inferred_op: NarsInferenceType,
}

/// Which NARS inference rule the BF16 decomposition suggests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NarsInferenceType {
    /// High crystallization, low tension → revise (accumulate evidence).
    Revision,
    /// High sign flips → negation (polarity reversed).
    Negation,
    /// High exponent shifts → abduction (magnitude-level cause inference).
    Abduction,
    /// Mixed signals → comparison (find similarity measure).
    Comparison,
    /// Mostly noise → no update.
    NoOp,
}

impl Default for NarsInferenceType {
    fn default() -> Self {
        Self::NoOp
    }
}

// ============================================================================
// Conversion: EnergyConflict → TruthValue
// ============================================================================

/// Convert an energy/conflict decomposition into a NARS TruthValue.
///
/// The mapping:
/// - **frequency** = agreement_ratio = agreement / (agreement + conflict)
///   High agreement → high frequency (positive evidence dominates)
/// - **confidence** = information_density = (energy_a + energy_b) / (2 × total_bits)
///   High energy in both vectors → high confidence (informative match)
///
/// This closes the loop: K2 exact match → NARS truth value → can be revised.
pub fn truth_from_energy_conflict(
    conflict: u32,
    energy_a: u32,
    energy_b: u32,
    agreement: u32,
    total_bits: u32,
) -> TruthValue {
    // Frequency: ratio of agreement to total difference
    let total_signal = agreement + conflict;
    let frequency = if total_signal > 0 {
        agreement as f32 / total_signal as f32
    } else {
        0.5 // No information → unknown
    };

    // Confidence: based on information density (how much energy both vectors carry)
    let max_energy = 2.0 * total_bits as f32;
    let total_energy = (energy_a + energy_b) as f32;
    let raw_confidence = if max_energy > 0.0 {
        total_energy / max_energy
    } else {
        0.0
    };
    // Scale to useful NARS range (0.0 - 0.95)
    let confidence = (raw_confidence * 0.95).min(0.95);

    TruthValue::new(frequency.clamp(0.0, 1.0), confidence.clamp(0.0, 1.0))
}

/// Convert a Hamming distance ratio to a TruthValue.
///
/// Simpler version when full energy/conflict decomposition isn't available.
/// frequency = 1.0 - (distance / total_bits), confidence from match quality.
pub fn truth_from_hamming(distance: u32, total_bits: u32) -> TruthValue {
    let ratio = distance as f32 / total_bits as f32;
    // Frequency: 1.0 for exact match, 0.0 for maximally different
    let frequency = (1.0 - ratio).clamp(0.0, 1.0);
    // Confidence: high for very close or very far, low for ~50% (random)
    // Uses a V-shaped curve: |2*ratio - 1| maps 0→1, 0.5→0, 1→1
    let distinctiveness = (2.0 * ratio - 1.0).abs();
    let confidence = (distinctiveness * 0.9).clamp(0.0, 0.95);
    TruthValue::new(frequency, confidence)
}

// ============================================================================
// Conversion: BF16 structural diff → Bf16CausalSignal
// ============================================================================

/// Extract a causal learning signal from BF16 structural diff metrics.
///
/// The 1+7+8 prefix decomposition:
/// - sign_flips → negation signal (class-level polarity change)
/// - exponent_bits_changed → abduction signal (magnitude cause)
/// - mantissa_bits_changed → noise (ignore for learning)
pub fn causal_from_bf16_diff(
    sign_flips: usize,
    exponent_bits_changed: usize,
    mantissa_bits_changed: usize,
    total_dims: usize,
    crystallized_ratio: f32,
    tension_ratio: f32,
) -> Bf16CausalSignal {
    let inferred_op = if total_dims == 0 {
        NarsInferenceType::NoOp
    } else {
        let sign_ratio = sign_flips as f32 / total_dims as f32;
        let exp_ratio = exponent_bits_changed as f32 / (total_dims as f32 * 8.0); // 8 exp bits per dim

        if sign_ratio < 0.05 && crystallized_ratio > 0.7 {
            // Few sign flips, mostly settled → revise (accumulate positive evidence)
            NarsInferenceType::Revision
        } else if sign_ratio > 0.3 {
            // Many sign flips → polarity changed → negation
            NarsInferenceType::Negation
        } else if exp_ratio > 0.2 && sign_ratio < 0.1 {
            // Exponent changed but signs agree → magnitude-level cause → abduction
            NarsInferenceType::Abduction
        } else if tension_ratio > 0.3 {
            // Mixed signals → comparison
            NarsInferenceType::Comparison
        } else {
            // Mostly mantissa noise → no update
            NarsInferenceType::NoOp
        }
    };

    Bf16CausalSignal {
        sign_flips,
        exponent_shifts: exponent_bits_changed,
        mantissa_noise: mantissa_bits_changed,
        total_dims,
        crystallized_ratio,
        tension_ratio,
        inferred_op,
    }
}

// ============================================================================
// Build FeedbackSignal from recognition results
// ============================================================================

/// Build a complete FeedbackSignal from recognition pipeline outputs.
///
/// This is the main entry point. Given match results from the hybrid pipeline,
/// it produces a FeedbackSignal that can be applied to WideMetaView metadata.
///
/// Parameters:
/// - `hamming_distance`, `total_bits`: from K2 exact match
/// - `conflict`, `energy_a`, `energy_b`, `agreement`: from EnergyConflict
/// - `sign_flips`, `exponent_bits_changed`, `mantissa_bits_changed`: from BF16 structural diff
/// - `total_bf16_dims`: number of BF16 dimensions scored
/// - `crystallized_ratio`, `tension_ratio`: from awareness decomposition
/// - `attention_weights`: per-group attention from learning signal (32 f32)
/// - `current_decay`: current evidence decay rate from WideMetaView
#[allow(clippy::too_many_arguments)]
pub fn build_feedback(
    hamming_distance: u32,
    total_bits: u32,
    conflict: u32,
    energy_a: u32,
    energy_b: u32,
    agreement: u32,
    sign_flips: usize,
    exponent_bits_changed: usize,
    mantissa_bits_changed: usize,
    total_bf16_dims: usize,
    crystallized_ratio: f32,
    tension_ratio: f32,
    attention_weights: &[f32; 32],
    current_decay: f32,
) -> FeedbackSignal {
    // 1. NARS truth value from energy/conflict
    let truth_value = truth_from_energy_conflict(
        conflict, energy_a, energy_b, agreement, total_bits,
    );

    // 2. Evidence counts (using NARS horizon k=1.0)
    let k = 1.0;
    let pos_ev = truth_value.positive_evidence(k);
    let total_ev = truth_value.total_evidence(k);
    let neg_ev = total_ev - pos_ev;
    let evidence = (pos_ev, neg_ev);

    // 3. Confidence delta (how much confidence changed from this observation)
    let hamming_tv = truth_from_hamming(hamming_distance, total_bits);
    let confidence_delta = hamming_tv.c - 0.5; // centered on 0.5 = neutral

    // 4. BF16 causal signal
    let bf16_causal = causal_from_bf16_diff(
        sign_flips,
        exponent_bits_changed,
        mantissa_bits_changed,
        total_bf16_dims,
        crystallized_ratio,
        tension_ratio,
    );

    // 5. Decay rate: increase decay if tension is high (evidence is contradictory)
    let decay_rate = if tension_ratio > 0.3 {
        (current_decay + 0.05).min(0.99) // accelerate decay under contradiction
    } else if crystallized_ratio > 0.7 {
        (current_decay - 0.01).max(0.01) // slow decay when knowledge is settled
    } else {
        current_decay // maintain current rate
    };

    // 6. RL reward: high crystallization → positive reward, high tension → negative
    let rl_reward = crystallized_ratio - tension_ratio;
    // Action slot: based on dominant causal signal
    let rl_action = match bf16_causal.inferred_op {
        NarsInferenceType::Revision => 0,
        NarsInferenceType::Negation => 1,
        NarsInferenceType::Abduction => 2,
        NarsInferenceType::Comparison => 3,
        NarsInferenceType::NoOp => 4,
    };

    FeedbackSignal {
        truth_value,
        evidence,
        confidence_delta,
        decay_rate,
        hybrid_weights: *attention_weights,
        bf16_causal,
        rl_reward,
        rl_action,
    }
}

// ============================================================================
// Apply feedback to WideMetaView — the write-back
// ============================================================================

/// Apply a FeedbackSignal to a WideMetaView's raw word buffer.
///
/// This writes back to the container metadata, closing the feedback loop:
///
/// 1. **NARS truth values** (W4-W7 via legacy MetaView):
///    - Revises frequency/confidence using NARS inference rules
///    - Updates positive/negative evidence counts
///
/// 2. **Extended NARS** (W160-W175):
///    - Updates derived frequency/confidence from inference chain
///    - Shifts evidence history ring buffer
///    - Updates decay rate
///
/// 3. **Hybrid weights** (W144-W159):
///    - EMA update of 32 attention weights from awareness signal
///
/// 4. **RL Q-values** (W32-W39):
///    - Updates the Q-value for the selected action slot
///
/// The `words` parameter is the raw 256 × u64 buffer of a WideContainer.
/// This function operates at the word level to avoid pulling in the full
/// WideMetaViewMut type (which would create a circular dependency with
/// ladybug-contract). Instead, callers create a WideMetaViewMut from
/// these words after this function returns.
pub fn apply_feedback(
    words: &mut [u64; 256],
    signal: &FeedbackSignal,
    learning_rate: f32,
    cycle: u64,
) {
    // --- 1. NARS truth values (legacy W4-W7) ---

    // Read current truth value
    const W_NARS_BASE: usize = 4;
    let current_freq = f32::from_bits((words[W_NARS_BASE] & 0xFFFF_FFFF) as u32);
    let current_conf = f32::from_bits(((words[W_NARS_BASE] >> 32) & 0xFFFF_FFFF) as u32);
    let current_tv = TruthValue::new(
        current_freq.clamp(0.0, 1.0),
        current_conf.clamp(0.0, 1.0),
    );

    // Apply the appropriate NARS inference based on BF16 causal signal
    let revised_tv = match signal.bf16_causal.inferred_op {
        NarsInferenceType::Revision => {
            // Standard revision: accumulate evidence
            NarsInference::revision(current_tv, signal.truth_value)
        }
        NarsInferenceType::Negation => {
            // Negation + revision: the dimension polarity flipped
            let negated = NarsInference::negation(signal.truth_value);
            NarsInference::revision(current_tv, negated)
        }
        NarsInferenceType::Abduction => {
            // Abduction: magnitude changed → infer cause
            NarsInference::abduction(current_tv, signal.truth_value)
        }
        NarsInferenceType::Comparison => {
            // Comparison: mixed signals → intersection for similarity
            NarsInference::intersection(current_tv, signal.truth_value)
        }
        NarsInferenceType::NoOp => {
            // No update — keep current
            current_tv
        }
    };

    // Write revised truth value (fields are .f and .c)
    let freq_bits = revised_tv.f.to_bits() as u64;
    let conf_bits = revised_tv.c.to_bits() as u64;
    words[W_NARS_BASE] = freq_bits | (conf_bits << 32);

    // Write evidence counts
    let k = 1.0f32;
    let pos_ev = revised_tv.positive_evidence(k);
    let total_ev = revised_tv.total_evidence(k);
    let neg_ev = total_ev - pos_ev;
    let pos_bits = pos_ev.to_bits() as u64;
    let neg_bits = neg_ev.to_bits() as u64;
    words[W_NARS_BASE + 1] = pos_bits | (neg_bits << 32);

    // --- 2. Extended NARS (W160-W175) ---

    const W_EXT_NARS_BASE: usize = 160;

    // Update decay rate (W160 hi)
    let decay_bits = signal.decay_rate.to_bits() as u64;
    words[W_EXT_NARS_BASE] =
        (words[W_EXT_NARS_BASE] & !(!0u64 << 32 << 0)) // keep hi bits
        | (words[W_EXT_NARS_BASE] & 0xFFFF_FFFF)        // keep lo bits
        ;
    // Actually: W160 lo = horizon, W160 hi = decay
    words[W_EXT_NARS_BASE] =
        (words[W_EXT_NARS_BASE] & 0xFFFF_FFFF) | (decay_bits << 32);

    // Update derived frequency/confidence (W161)
    let derived_freq_bits = revised_tv.f.to_bits() as u64;
    let derived_conf_bits = revised_tv.c.to_bits() as u64;
    words[W_EXT_NARS_BASE + 1] = derived_freq_bits | (derived_conf_bits << 32);

    // Shift evidence history ring buffer (W162-W167, 8 slots)
    // Shift entries: [0]→[1], [1]→[2], ... [6]→[7], new→[0]
    for slot in (1..8).rev() {
        let src_word = W_EXT_NARS_BASE + 2 + (slot - 1) / 2;
        let src_shift = ((slot - 1) % 2) * 32;
        let val = f32::from_bits(((words[src_word] >> src_shift) & 0xFFFF_FFFF) as u32);

        let dst_word = W_EXT_NARS_BASE + 2 + slot / 2;
        let dst_shift = (slot % 2) * 32;
        let bits = val.to_bits() as u64;
        words[dst_word] = (words[dst_word] & !(0xFFFF_FFFF_u64 << dst_shift)) | (bits << dst_shift);
    }
    // Write new evidence to slot 0 (total evidence from this cycle)
    let new_evidence = signal.evidence.0 + signal.evidence.1;
    let ev_bits = new_evidence.to_bits() as u64;
    words[W_EXT_NARS_BASE + 2] =
        (words[W_EXT_NARS_BASE + 2] & !0xFFFF_FFFF) | ev_bits;

    // --- 3. Hybrid weights (W144-W159) ---

    const W_HYBRID_BASE: usize = 144;
    for i in 0..32 {
        let word = W_HYBRID_BASE + i / 2;
        let shift = (i % 2) * 32;
        let current = f32::from_bits(((words[word] >> shift) & 0xFFFF_FFFF) as u32);
        let target = signal.hybrid_weights[i];
        // EMA update
        let updated = current * (1.0 - learning_rate) + target * learning_rate;
        let bits = updated.to_bits() as u64;
        words[word] = (words[word] & !(0xFFFF_FFFF_u64 << shift)) | (bits << shift);
    }

    // --- 4. RL Q-value update (W32-W39) ---

    const W_RL_BASE: usize = 32;
    if signal.rl_action < 16 {
        let word_idx = W_RL_BASE + signal.rl_action / 2;
        let shift = (signal.rl_action % 2) * 32;
        let current_q = f32::from_bits(((words[word_idx] >> shift) & 0xFFFF_FFFF) as u32);

        // Q-learning update: Q(s,a) ← Q(s,a) + α * (reward - Q(s,a))
        let updated_q = current_q + learning_rate * (signal.rl_reward - current_q);
        let bits = updated_q.to_bits() as u64;
        words[word_idx] = (words[word_idx] & !(0xFFFF_FFFF_u64 << shift)) | (bits << shift);
    }

    // --- 5. Update modification timestamp (W2 hi 32 bits) ---
    // Truncate cycle to u32 for compact storage
    let ts = (cycle & 0xFFFF_FFFF) as u32;
    let ts_bits = ts as u64;
    words[2] = (words[2] & 0xFFFF_FFFF) | (ts_bits << 32);
}

// ============================================================================
// Convenience: apply NARS revision from Hamming distance alone
// ============================================================================

/// Quick feedback from just a Hamming distance (no BF16 decomposition).
///
/// Useful when the full hybrid pipeline isn't available (e.g., Container-level
/// sweep without BF16 scoring). Applies simple NARS revision with the
/// distance-derived truth value.
pub fn apply_hamming_feedback(
    words: &mut [u64; 256],
    distance: u32,
    total_bits: u32,
    learning_rate: f32,
    cycle: u64,
) {
    let tv = truth_from_hamming(distance, total_bits);
    let k = 1.0;
    let pos_ev = tv.positive_evidence(k);
    let total_ev = tv.total_evidence(k);
    let neg_ev = total_ev - pos_ev;

    let signal = FeedbackSignal {
        truth_value: tv,
        evidence: (pos_ev, neg_ev),
        confidence_delta: 0.0,
        decay_rate: 0.95,
        hybrid_weights: [0.5; 32],
        bf16_causal: Bf16CausalSignal {
            inferred_op: NarsInferenceType::Revision, // Hamming match = positive evidence
            ..Bf16CausalSignal::default()
        },
        rl_reward: 0.0,
        rl_action: 0,
    };

    apply_feedback(words, &signal, learning_rate, cycle);
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truth_from_energy_conflict_perfect_match() {
        // Perfect agreement: all bits agree, no conflict
        let tv = truth_from_energy_conflict(0, 8000, 8000, 8000, 16384);
        assert!(tv.f > 0.99, "perfect match frequency = {}", tv.f);
        assert!(tv.c > 0.4, "should have meaningful confidence = {}", tv.c);
    }

    #[test]
    fn test_truth_from_energy_conflict_anti_match() {
        // Anti-match: all bits conflict, no agreement
        let tv = truth_from_energy_conflict(8000, 8000, 8000, 0, 16384);
        assert!(tv.f < 0.01, "anti-match frequency = {}", tv.f);
    }

    #[test]
    fn test_truth_from_energy_conflict_balanced() {
        // 50/50 split
        let tv = truth_from_energy_conflict(4000, 8000, 8000, 4000, 16384);
        assert!((tv.f - 0.5).abs() < 0.01, "balanced should be ~0.5, got {}", tv.f);
    }

    #[test]
    fn test_truth_from_hamming_exact() {
        let tv = truth_from_hamming(0, 16384);
        assert!((tv.f - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_truth_from_hamming_random() {
        // ~50% distance = random noise
        let tv = truth_from_hamming(8192, 16384);
        assert!((tv.f - 0.5).abs() < 0.01);
        assert!(tv.c < 0.1, "random should have low confidence");
    }

    #[test]
    fn test_causal_from_bf16_diff_revision() {
        let signal = causal_from_bf16_diff(2, 10, 500, 1024, 0.85, 0.05);
        assert_eq!(signal.inferred_op, NarsInferenceType::Revision);
    }

    #[test]
    fn test_causal_from_bf16_diff_negation() {
        let signal = causal_from_bf16_diff(400, 50, 100, 1024, 0.1, 0.4);
        assert_eq!(signal.inferred_op, NarsInferenceType::Negation);
    }

    #[test]
    fn test_causal_from_bf16_diff_abduction() {
        let signal = causal_from_bf16_diff(10, 2000, 500, 1024, 0.3, 0.1);
        assert_eq!(signal.inferred_op, NarsInferenceType::Abduction);
    }

    #[test]
    fn test_causal_from_bf16_diff_comparison() {
        let signal = causal_from_bf16_diff(50, 500, 300, 1024, 0.2, 0.4);
        assert_eq!(signal.inferred_op, NarsInferenceType::Comparison);
    }

    #[test]
    fn test_causal_from_bf16_diff_noop() {
        let signal = causal_from_bf16_diff(10, 50, 5000, 1024, 0.1, 0.05);
        assert_eq!(signal.inferred_op, NarsInferenceType::NoOp);
    }

    #[test]
    fn test_build_feedback_integration() {
        let signal = build_feedback(
            1000,  // hamming_distance
            16384, // total_bits
            1000,  // conflict
            8000,  // energy_a
            8000,  // energy_b
            7000,  // agreement
            5,     // sign_flips
            20,    // exponent_bits_changed
            200,   // mantissa_bits_changed
            1024,  // total_bf16_dims
            0.8,   // crystallized_ratio
            0.05,  // tension_ratio
            &[0.5; 32], // attention_weights
            0.95,  // current_decay
        );

        assert!(signal.truth_value.f > 0.5);
        assert!(signal.truth_value.c > 0.0);
        assert_eq!(signal.bf16_causal.inferred_op, NarsInferenceType::Revision);
        assert!(signal.rl_reward > 0.0, "high crystallization should give positive reward");
    }

    #[test]
    fn test_apply_feedback_writes_nars() {
        let mut words = [0u64; 256];

        // Set initial NARS values
        let init_freq = 0.5f32;
        let init_conf = 0.1f32;
        words[4] = init_freq.to_bits() as u64 | ((init_conf.to_bits() as u64) << 32);

        let signal = build_feedback(
            500, 16384, 500, 8000, 8000, 7500,
            2, 10, 100, 1024, 0.9, 0.02,
            &[0.8; 32], 0.95,
        );

        apply_feedback(&mut words, &signal, 0.1, 42);

        // Check that NARS values were updated
        let new_freq = f32::from_bits((words[4] & 0xFFFF_FFFF) as u32);
        let new_conf = f32::from_bits(((words[4] >> 32) & 0xFFFF_FFFF) as u32);
        assert!(new_freq > 0.0 && new_freq <= 1.0, "frequency should be valid: {}", new_freq);
        assert!(new_conf >= 0.0 && new_conf <= 1.0, "confidence should be valid: {}", new_conf);
    }

    #[test]
    fn test_apply_feedback_writes_hybrid_weights() {
        let mut words = [0u64; 256];

        let signal = build_feedback(
            1000, 16384, 1000, 8000, 8000, 7000,
            5, 20, 200, 1024, 0.8, 0.05,
            &[0.9; 32], 0.95,
        );

        apply_feedback(&mut words, &signal, 0.5, 100);

        // Check hybrid weights were written (W144-W159)
        let w0 = f32::from_bits((words[144] & 0xFFFF_FFFF) as u32);
        assert!(w0 > 0.0, "hybrid weight should be non-zero: {}", w0);
    }

    #[test]
    fn test_apply_feedback_writes_rl_qvalue() {
        let mut words = [0u64; 256];

        let signal = build_feedback(
            500, 16384, 500, 8000, 8000, 7500,
            2, 10, 100, 1024, 0.9, 0.02,
            &[0.5; 32], 0.95,
        );

        apply_feedback(&mut words, &signal, 0.1, 1);

        // Check RL Q-value at action slot 0 (Revision)
        let q0 = f32::from_bits((words[32] & 0xFFFF_FFFF) as u32);
        assert!(q0 != 0.0, "Q-value should be updated: {}", q0);
    }

    #[test]
    fn test_apply_feedback_evidence_history() {
        let mut words = [0u64; 256];

        // Apply feedback twice to see history shift
        let signal1 = build_feedback(
            1000, 16384, 1000, 8000, 8000, 7000,
            5, 20, 200, 1024, 0.8, 0.05,
            &[0.5; 32], 0.95,
        );
        apply_feedback(&mut words, &signal1, 0.1, 1);

        let ev_slot0_after_first = f32::from_bits((words[162] & 0xFFFF_FFFF) as u32);
        assert!(ev_slot0_after_first > 0.0, "evidence history[0] should be non-zero");

        let signal2 = build_feedback(
            500, 16384, 500, 8000, 8000, 7500,
            2, 10, 100, 1024, 0.9, 0.02,
            &[0.5; 32], 0.95,
        );
        apply_feedback(&mut words, &signal2, 0.1, 2);

        // Slot 1 should now have the old slot 0 value (shifted)
        let ev_slot1_after_second = f32::from_bits(((words[162] >> 32) & 0xFFFF_FFFF) as u32);
        assert!(
            (ev_slot1_after_second - ev_slot0_after_first).abs() < 0.01,
            "evidence should shift: slot1={} should equal old slot0={}",
            ev_slot1_after_second, ev_slot0_after_first
        );
    }

    #[test]
    fn test_apply_hamming_feedback_simple() {
        let mut words = [0u64; 256];
        // Set initial NARS state (unknown: f=0.5, c=0.1)
        let init_freq = 0.5f32;
        let init_conf = 0.1f32;
        words[4] = init_freq.to_bits() as u64 | ((init_conf.to_bits() as u64) << 32);

        apply_hamming_feedback(&mut words, 1000, 16384, 0.1, 42);

        let freq = f32::from_bits((words[4] & 0xFFFF_FFFF) as u32);
        assert!(freq > 0.5, "close match should give frequency > 0.5: {}", freq);
    }

    #[test]
    fn test_nars_revision_accumulates_evidence() {
        let mut words = [0u64; 256];

        // Multiple revision cycles should increase confidence
        for cycle in 0..5 {
            let signal = build_feedback(
                1000, 16384, 1000, 8000, 8000, 7000,
                2, 10, 100, 1024, 0.85, 0.05,
                &[0.7; 32], 0.95,
            );
            apply_feedback(&mut words, &signal, 0.2, cycle);
        }

        let final_conf = f32::from_bits(((words[4] >> 32) & 0xFFFF_FFFF) as u32);
        assert!(
            final_conf > 0.1,
            "5 revision cycles should build confidence: {}",
            final_conf
        );
    }

    #[test]
    fn test_negation_signal_reduces_frequency() {
        let mut words = [0u64; 256];

        // Start with high frequency
        let high_freq = 0.9f32;
        let init_conf = 0.5f32;
        words[4] = high_freq.to_bits() as u64 | ((init_conf.to_bits() as u64) << 32);

        // Apply negation feedback (many sign flips)
        let signal = build_feedback(
            5000, 16384, 5000, 8000, 8000, 3000,
            400, 50, 100, 1024, 0.1, 0.4,
            &[0.3; 32], 0.95,
        );

        apply_feedback(&mut words, &signal, 0.3, 1);

        let new_freq = f32::from_bits((words[4] & 0xFFFF_FFFF) as u32);
        assert!(
            new_freq < high_freq,
            "negation should reduce frequency: {} → {}",
            high_freq, new_freq
        );
    }

    #[test]
    fn test_decay_rate_adapts_to_tension() {
        // High tension → faster decay
        let signal = build_feedback(
            3000, 16384, 3000, 8000, 8000, 5000,
            200, 500, 300, 1024, 0.2, 0.5,
            &[0.5; 32], 0.90,
        );
        assert!(signal.decay_rate > 0.90, "high tension should increase decay: {}", signal.decay_rate);

        // High crystallization → slower decay
        let signal2 = build_feedback(
            500, 16384, 500, 8000, 8000, 7500,
            2, 10, 100, 1024, 0.9, 0.02,
            &[0.5; 32], 0.90,
        );
        assert!(signal2.decay_rate < 0.90, "high crystallization should decrease decay: {}", signal2.decay_rate);
    }
}
