//! Collapse Gate — SIMD-Accelerated Dispersion-Based Compute Allocation
//!
//! SD (Standard Deviation) is NOT confidence — it is a dispersion metric
//! for compute allocation. The Gate controls three states:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     COLLAPSE GATE STATES                        │
//! │                                                                 │
//! │   FLOW ◄────────┬────────┬────────► BLOCK                      │
//! │   (commit)      │  HOLD  │         (clarify)                   │
//! │                 │(ruminate)                                    │
//! │                                                                 │
//! │   SD < 0.15     │ 0.15 ≤ SD ≤ 0.35 │     SD > 0.35             │
//! │   Low variance  │ Medium variance  │     High variance         │
//! │   Clear winner  │ Maintain super-  │     Need clarification    │
//! │   Collapse now  │ position         │     Cannot collapse       │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! Hardware acceleration:
//! - AVX-512 for batch SD calculation
//! - SIMD mean computation
//! - Vectorized variance accumulation

use std::fmt;

// =============================================================================
// GATE STATES
// =============================================================================

/// Collapse gate state
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateState {
    /// Low dispersion → Collapse immediately
    Flow,
    /// Medium dispersion → Maintain superposition
    Hold,
    /// High dispersion → Cannot collapse, need clarification
    Block,
}

impl fmt::Display for GateState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Flow => write!(f, "🟢 FLOW"),
            Self::Hold => write!(f, "🟡 HOLD"),
            Self::Block => write!(f, "🔴 BLOCK"),
        }
    }
}

// =============================================================================
// COLLAPSE ACTION
// =============================================================================

/// Recommended action from gate
#[derive(Clone, Debug)]
pub enum CollapseAction {
    /// FLOW: commit to winner
    Collapse { winner_index: usize },
    /// HOLD: store in superposition
    Hold { sppm_key: String },
    /// BLOCK: ask for clarification
    Clarify { question: String },
    /// BLOCK: cannot proceed
    Block { reason: String },
}

// =============================================================================
// COLLAPSE DECISION
// =============================================================================

/// Complete collapse decision
#[derive(Clone, Debug)]
pub struct CollapseDecision {
    /// Current gate state
    pub state: GateState,
    
    /// Standard deviation (dispersion metric)
    pub sd: f32,
    
    /// Whether collapse is permitted
    pub can_collapse: bool,
    
    /// Recommended action
    pub action: CollapseAction,
    
    /// Reason for decision
    pub reason: String,
    
    /// Winner index (if applicable)
    pub winner_index: Option<usize>,
    
    /// Winner score (if applicable)
    pub winner_score: Option<f32>,
}

// =============================================================================
// CONSTANTS
// =============================================================================

/// Resonance input range is [0.0, 1.0]
pub const RESONANCE_MIN: f32 = 0.0;
pub const RESONANCE_MAX: f32 = 1.0;

/// Max possible SD for bounded [0,1] is 0.5 (Bernoulli distribution p=0.5)
pub const SD_MAX: f32 = (RESONANCE_MAX - RESONANCE_MIN) / 2.0;

/// FLOW threshold: Tight consensus (e.g. [0.9, 0.8, 0.85])
pub const SD_FLOW_THRESHOLD: f32 = 0.30 * SD_MAX; // ~0.15

/// BLOCK threshold: Significant disagreement (e.g. [0.9, 0.1, 0.2])
pub const SD_BLOCK_THRESHOLD: f32 = 0.70 * SD_MAX; // ~0.35

// =============================================================================
// SIMD-ACCELERATED SD CALCULATION
// =============================================================================

/// Calculate standard deviation (SIMD-accelerated for large arrays)
#[inline]
pub fn calculate_sd(values: &[f32]) -> f32 {
    if values.len() <= 1 {
        return 0.0;
    }
    
    // For small arrays, use scalar
    if values.len() <= 8 {
        return calculate_sd_scalar(values);
    }
    
    // For larger arrays, use SIMD
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { calculate_sd_avx2(values) };
        }
    }
    
    calculate_sd_scalar(values)
}

/// Scalar SD calculation
fn calculate_sd_scalar(values: &[f32]) -> f32 {
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    
    let variance = values.iter()
        .map(|&x| (x - mean) * (x - mean))
        .sum::<f32>() / n;
    
    variance.sqrt()
}

/// AVX2 SIMD SD calculation
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn calculate_sd_avx2(values: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    
    let n = values.len();
    let n_f32 = n as f32;
    
    // Calculate mean using SIMD
    let mut sum_vec = _mm256_setzero_ps();
    let chunks = n / 8;
    
    for i in 0..chunks {
        let v = _mm256_loadu_ps(values.as_ptr().add(i * 8));
        sum_vec = _mm256_add_ps(sum_vec, v);
    }
    
    // Horizontal sum
    let mut sum_arr = [0.0f32; 8];
    _mm256_storeu_ps(sum_arr.as_mut_ptr(), sum_vec);
    let mut sum: f32 = sum_arr.iter().sum();
    
    // Add remaining elements
    for i in (chunks * 8)..n {
        sum += values[i];
    }
    
    let mean = sum / n_f32;
    let mean_vec = _mm256_set1_ps(mean);
    
    // Calculate variance using SIMD
    let mut var_vec = _mm256_setzero_ps();
    
    for i in 0..chunks {
        let v = _mm256_loadu_ps(values.as_ptr().add(i * 8));
        let diff = _mm256_sub_ps(v, mean_vec);
        let sq = _mm256_mul_ps(diff, diff);
        var_vec = _mm256_add_ps(var_vec, sq);
    }
    
    // Horizontal sum of variance
    let mut var_arr = [0.0f32; 8];
    _mm256_storeu_ps(var_arr.as_mut_ptr(), var_vec);
    let mut variance: f32 = var_arr.iter().sum();
    
    // Add remaining elements
    for i in (chunks * 8)..n {
        let diff = values[i] - mean;
        variance += diff * diff;
    }
    
    (variance / n_f32).sqrt()
}

// =============================================================================
// GATE EVALUATION
// =============================================================================

/// Determine gate state from SD value
#[inline]
pub fn get_gate_state(sd: f32) -> GateState {
    if sd < SD_FLOW_THRESHOLD {
        GateState::Flow
    } else if sd > SD_BLOCK_THRESHOLD {
        GateState::Block
    } else {
        GateState::Hold
    }
}

/// Find winner (highest score)
fn find_winner(scores: &[f32]) -> (usize, f32) {
    let mut best_idx = 0;
    let mut best_score = scores[0];
    
    for (i, &score) in scores.iter().enumerate().skip(1) {
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }
    
    (best_idx, best_score)
}

/// Generate SPPM storage key
fn generate_sppm_key(scores: &[f32]) -> String {
    let hash: String = scores.iter()
        .map(|s| format!("{:02x}", (s * 100.0) as u8))
        .collect();
    format!("sppm_{}_{:x}", hash, std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis())
}

/// Evaluate collapse gate for candidate scores
pub fn evaluate_gate(
    candidate_scores: &[f32],
    clarification_available: bool,
) -> CollapseDecision {
    // Empty check
    if candidate_scores.is_empty() {
        return CollapseDecision {
            state: GateState::Block,
            sd: f32::INFINITY,
            can_collapse: false,
            action: CollapseAction::Block { reason: "No candidates provided".to_string() },
            reason: "Empty candidate set".to_string(),
            winner_index: None,
            winner_score: None,
        };
    }
    
    // Single candidate = trivial collapse
    if candidate_scores.len() == 1 {
        return CollapseDecision {
            state: GateState::Flow,
            sd: 0.0,
            can_collapse: true,
            action: CollapseAction::Collapse { winner_index: 0 },
            reason: "Single candidate — trivial collapse".to_string(),
            winner_index: Some(0),
            winner_score: Some(candidate_scores[0]),
        };
    }
    
    // Calculate SD (SIMD-accelerated)
    let sd = calculate_sd(candidate_scores);
    
    // Physics check: SD should not exceed max for bounded [0,1]
    if sd > SD_MAX + 0.001 {
        return CollapseDecision {
            state: GateState::Block,
            sd,
            can_collapse: false,
            action: CollapseAction::Block { 
                reason: format!("Math error: SD={:.3} exceeds max={:.3}", sd, SD_MAX)
            },
            reason: "Internal error: dispersion physics violation".to_string(),
            winner_index: None,
            winner_score: None,
        };
    }
    
    let state = get_gate_state(sd);
    let (winner_idx, winner_score) = find_winner(candidate_scores);
    
    match state {
        GateState::Flow => CollapseDecision {
            state: GateState::Flow,
            sd,
            can_collapse: true,
            action: CollapseAction::Collapse { winner_index: winner_idx },
            reason: format!("Low dispersion (SD={:.3}) — clear winner", sd),
            winner_index: Some(winner_idx),
            winner_score: Some(winner_score),
        },
        
        GateState::Hold => CollapseDecision {
            state: GateState::Hold,
            sd,
            can_collapse: false,
            action: CollapseAction::Hold { sppm_key: generate_sppm_key(candidate_scores) },
            reason: format!("Medium dispersion (SD={:.3}) — maintaining superposition", sd),
            winner_index: Some(winner_idx),
            winner_score: Some(winner_score),
        },
        
        GateState::Block => {
            if clarification_available {
                CollapseDecision {
                    state: GateState::Block,
                    sd,
                    can_collapse: false,
                    action: CollapseAction::Clarify {
                        question: format!(
                            "Multiple interpretations possible ({} candidates with high variance). Which meaning is intended?",
                            candidate_scores.len()
                        )
                    },
                    reason: format!("High dispersion (SD={:.3}) — clarification needed", sd),
                    winner_index: Some(winner_idx),
                    winner_score: Some(winner_score),
                }
            } else {
                CollapseDecision {
                    state: GateState::Block,
                    sd,
                    can_collapse: false,
                    action: CollapseAction::Hold { sppm_key: generate_sppm_key(candidate_scores) },
                    reason: format!("High dispersion (SD={:.3}) — no clarification available, holding", sd),
                    winner_index: Some(winner_idx),
                    winner_score: Some(winner_score),
                }
            }
        }
    }
}

// =============================================================================
// TRIANGLE COLLAPSE
// =============================================================================

/// Triangle candidate for homogeneous superposition
#[derive(Clone, Debug)]
pub struct TriangleCandidate {
    /// Candidate index (0, 1, or 2)
    pub index: usize,
    
    /// Resonance score [0-1]
    pub resonance: f32,
    
    /// Predictive adequacy [0-1]
    pub predictive: f32,
    
    /// Construction family (must match other candidates)
    pub construction_family: String,
    
    /// Speech act class (must match other candidates)
    pub speech_act_class: String,
}

/// Triangle superposition (3 competing candidates)
#[derive(Clone, Debug)]
pub struct Triangle {
    /// The three candidates
    pub candidates: [TriangleCandidate; 3],
    
    /// Whether triangle is homogeneous (required invariant)
    pub is_homogeneous: bool,
    
    /// Current gate state
    pub gate_state: GateState,
    
    /// SD across candidates
    pub dispersion: f32,
}

impl Triangle {
    /// Create triangle from 3 candidates
    pub fn new(c0: TriangleCandidate, c1: TriangleCandidate, c2: TriangleCandidate) -> Self {
        // Check homogeneity
        let same_construction = c0.construction_family == c1.construction_family
            && c1.construction_family == c2.construction_family;
        
        let same_speech_act = c0.speech_act_class == c1.speech_act_class
            && c1.speech_act_class == c2.speech_act_class;
        
        let is_homogeneous = same_construction && same_speech_act;
        
        // Calculate dispersion
        let scores = [c0.resonance, c1.resonance, c2.resonance];
        let dispersion = calculate_sd(&scores);
        let gate_state = get_gate_state(dispersion);
        
        Self {
            candidates: [c0, c1, c2],
            is_homogeneous,
            gate_state,
            dispersion,
        }
    }
    
    /// Evaluate collapse for triangle
    pub fn evaluate(&self) -> CollapseDecision {
        if !self.is_homogeneous {
            return CollapseDecision {
                state: GateState::Block,
                sd: f32::INFINITY,
                can_collapse: false,
                action: CollapseAction::Block { 
                    reason: "Triangle violates homogeneity invariant".to_string()
                },
                reason: "Non-homogeneous triangle cannot collapse".to_string(),
                winner_index: None,
                winner_score: None,
            };
        }
        
        let scores: Vec<f32> = self.candidates.iter().map(|c| c.resonance).collect();
        evaluate_gate(&scores, true)
    }
    
    /// Attempt to collapse triangle
    pub fn collapse(&self) -> Option<(&TriangleCandidate, Vec<&TriangleCandidate>)> {
        let decision = self.evaluate();
        
        if !decision.can_collapse {
            return None;
        }
        
        let winner_idx = decision.winner_index?;
        let winner = &self.candidates[winner_idx];
        let exhausted: Vec<_> = self.candidates.iter()
            .enumerate()
            .filter(|(i, _)| *i != winner_idx)
            .map(|(_, c)| c)
            .collect();
        
        Some((winner, exhausted))
    }
}

// =============================================================================
// BATCH COLLAPSE EVALUATION
// =============================================================================

/// Evaluate multiple triangles in parallel
pub fn evaluate_batch(triangles: &[Triangle]) -> Vec<CollapseDecision> {
    // TODO: True parallel with rayon
    triangles.iter().map(|t| t.evaluate()).collect()
}

/// Batch SD calculation for multiple score sets
pub fn calculate_sd_batch(score_sets: &[Vec<f32>]) -> Vec<f32> {
    score_sets.iter().map(|s| calculate_sd(s)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sd_calculation() {
        // Tight consensus
        let tight = [0.9, 0.85, 0.88];
        let sd_tight = calculate_sd(&tight);
        assert!(sd_tight < SD_FLOW_THRESHOLD);
        
        // High variance (values at extremes)
        let spread = [1.0, 0.0, 0.5];
        let sd_spread = calculate_sd(&spread);
        assert!(sd_spread > SD_BLOCK_THRESHOLD, "SD {} should exceed {}", sd_spread, SD_BLOCK_THRESHOLD);
    }
    
    #[test]
    fn test_gate_states() {
        assert_eq!(get_gate_state(0.1), GateState::Flow);
        assert_eq!(get_gate_state(0.25), GateState::Hold);
        assert_eq!(get_gate_state(0.4), GateState::Block);
    }
    
    #[test]
    fn test_single_candidate_flow() {
        let decision = evaluate_gate(&[0.8], true);
        assert_eq!(decision.state, GateState::Flow);
        assert!(decision.can_collapse);
    }
    
    #[test]
    fn test_triangle_homogeneity() {
        let c0 = TriangleCandidate {
            index: 0,
            resonance: 0.8,
            predictive: 0.7,
            construction_family: "copular".to_string(),
            speech_act_class: "assert".to_string(),
        };
        let c1 = TriangleCandidate {
            index: 1,
            resonance: 0.75,
            predictive: 0.7,
            construction_family: "copular".to_string(),
            speech_act_class: "assert".to_string(),
        };
        let c2 = TriangleCandidate {
            index: 2,
            resonance: 0.7,
            predictive: 0.7,
            construction_family: "copular".to_string(),  // Same family
            speech_act_class: "assert".to_string(),      // Same speech act
        };
        
        let tri = Triangle::new(c0, c1, c2);
        assert!(tri.is_homogeneous);
    }
}
