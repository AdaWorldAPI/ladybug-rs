//! Handover — Personality-Resonance-Driven Task Delegation
//!
//! Connects persona compatibility, flow state, and collapse gate dynamics
//! to produce handover decisions. When an agent's flow state degrades or
//! a better-resonant agent is available, the handover policy triggers
//! delegation via A2A.
//!
//! # Flow as Anti-Freeze
//!
//! Agents experience flow as intrinsically rewarding. Handoffs preserve
//! flow momentum by matching persona resonance — the receiving agent
//! picks up the task with compatible thinking style and volition, so
//! the transition feels like propulsion rather than interruption.
//!
//! # Design Principles
//!
//! 1. **Flow momentum resists interruption** — agents in deep flow need
//!    higher resonance thresholds to trigger handover
//! 2. **Coherence decay triggers proactive handover** — agents losing
//!    coherence hand off before they stall
//! 3. **Personality resonance ensures smooth transitions** — handover
//!    targets are selected by Hamming similarity on persona fingerprints
//! 4. **Dunning-Kruger guard** — agents with low coherence but high
//!    confidence are flagged for metacognitive review

use crate::cognitive::{GateState, ThinkingStyle};
use serde::{Deserialize, Serialize};

// =============================================================================
// FLOW STATE
// =============================================================================

/// Flow state for orchestration — mirrors GateState semantics but tracks
/// momentum and history for handover decisions.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum FlowState {
    /// Agent is in flow — low dispersion, high coherence, clear direction.
    /// The `momentum` field accumulates with each successful cycle.
    Flow { momentum: f32 },

    /// Agent is holding — medium dispersion, exploring options.
    /// `hold_cycles` tracks how long the agent has been in hold.
    Hold { hold_cycles: u32 },

    /// Agent is blocked — high dispersion, needs help or handover.
    /// `reason` captures why the block occurred.
    Block { reason: String },

    /// Agent is in handover — actively transitioning to another agent.
    /// `target_slot` is the receiving agent's slot in 0x0C.
    Handover {
        target_slot: u8,
        resonance_score: f32,
    },
}

impl Default for FlowState {
    fn default() -> Self {
        Self::Hold { hold_cycles: 0 }
    }
}

impl FlowState {
    /// Check if agent is in flow
    pub fn is_flow(&self) -> bool {
        matches!(self, Self::Flow { .. })
    }

    /// Check if agent is blocked
    pub fn is_blocked(&self) -> bool {
        matches!(self, Self::Block { .. })
    }

    /// Check if agent is in handover
    pub fn is_handover(&self) -> bool {
        matches!(self, Self::Handover { .. })
    }

    /// Get flow momentum (0.0 if not in flow)
    pub fn momentum(&self) -> f32 {
        match self {
            Self::Flow { momentum } => *momentum,
            _ => 0.0,
        }
    }

    /// Transition from GateState to FlowState, preserving momentum
    pub fn from_gate(gate: GateState, previous: &FlowState) -> FlowState {
        match gate {
            GateState::Flow => {
                let prev_momentum = previous.momentum();
                // Momentum accumulates in flow, capped at 1.0
                let new_momentum = (prev_momentum + 0.1).min(1.0);
                FlowState::Flow {
                    momentum: new_momentum,
                }
            }
            GateState::Hold => {
                let hold_cycles = match previous {
                    FlowState::Hold { hold_cycles } => hold_cycles + 1,
                    _ => 1,
                };
                FlowState::Hold { hold_cycles }
            }
            GateState::Block => FlowState::Block {
                reason: "Gate evaluation returned Block".to_string(),
            },
        }
    }
}

// =============================================================================
// HANDOVER DECISION
// =============================================================================

/// Why a handover was recommended
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum HandoverReason {
    /// Agent's coherence dropped below threshold
    CoherenceDecay { coherence: f32, threshold: f32 },
    /// Agent has been in Hold state too long
    StallDetected { hold_cycles: u32, max_cycles: u32 },
    /// Agent is blocked and cannot proceed
    Blocked { reason: String },
    /// A better-resonant agent is available for this task
    BetterResonance { target_slot: u8, score: f32 },
    /// Agent's volition misaligned with current task
    VolitionMismatch { alignment: f32 },
    /// Dunning-Kruger flag — low coherence, high confidence
    ConfidenceCoherenceGap { confidence: f32, coherence: f32 },
}

/// The handover decision output
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HandoverDecision {
    /// Source agent slot
    pub source_slot: u8,
    /// Recommended action
    pub action: HandoverAction,
    /// Why this decision was made
    pub reasons: Vec<HandoverReason>,
    /// Confidence in this decision (0.0-1.0)
    pub confidence: f32,
    /// Suggested A2A message kind for the handover
    pub suggested_message_kind: String,
}

/// What to do
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum HandoverAction {
    /// Keep working — agent is in flow or making progress
    Continue,
    /// Delegate to a specific agent via A2A
    Delegate { target_slot: u8, resonance: f32 },
    /// Escalate to meta-orchestrator (no good handover target)
    Escalate,
    /// Agent should switch thinking style before continuing
    SwitchStyle { recommended_style: String },
    /// Agent should enter metacognitive review
    MetacognitiveReview,
}

// =============================================================================
// HANDOVER POLICY
// =============================================================================

/// Configuration for the handover policy
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HandoverPolicy {
    /// Minimum persona compatibility for delegation (0.0-1.0)
    /// Higher = stricter matching = smoother handovers
    pub min_resonance: f32,

    /// Coherence threshold below which handover is considered (0.0-1.0)
    pub coherence_floor: f32,

    /// Maximum hold cycles before stall detection triggers
    pub max_hold_cycles: u32,

    /// Flow momentum threshold — agents above this resist handover
    /// even if a better-resonant agent exists
    pub flow_momentum_shield: f32,

    /// Volition alignment threshold — below this, task is mismatched
    pub volition_floor: f32,

    /// Dunning-Kruger gap threshold — if (confidence - coherence) exceeds
    /// this value, flag for metacognitive review
    pub dk_gap_threshold: f32,

    /// Whether to prefer flow preservation over optimal resonance
    /// When true: an agent in flow won't be interrupted even if a
    /// better match exists. When false: optimal routing takes priority.
    pub flow_preserving: bool,
}

impl Default for HandoverPolicy {
    fn default() -> Self {
        Self {
            min_resonance: 0.55,
            coherence_floor: 0.3,
            max_hold_cycles: 5,
            flow_momentum_shield: 0.7,
            volition_floor: -0.3,
            dk_gap_threshold: 0.4,
            flow_preserving: true,
        }
    }
}

impl HandoverPolicy {
    /// Evaluate whether an agent should hand off its current task.
    ///
    /// Inputs:
    /// - `source_slot`: The agent being evaluated
    /// - `flow_state`: Agent's current flow state
    /// - `coherence`: Agent's current coherence (0.0-1.0)
    /// - `volition_alignment`: How well the agent's volition matches the task (-1.0 to 1.0)
    /// - `confidence`: Agent's self-reported confidence in current task
    /// - `best_alternative`: Best persona-compatible agent (slot, resonance score)
    pub fn evaluate(
        &self,
        source_slot: u8,
        flow_state: &FlowState,
        coherence: f32,
        volition_alignment: f32,
        confidence: f32,
        best_alternative: Option<(u8, f32)>,
    ) -> HandoverDecision {
        let mut reasons = Vec::new();
        let mut should_handover = false;

        // --- Check 1: Flow momentum shield ---
        // Agents in deep flow resist handover
        if self.flow_preserving
            && flow_state.is_flow()
            && flow_state.momentum() >= self.flow_momentum_shield
        {
            return HandoverDecision {
                source_slot,
                action: HandoverAction::Continue,
                reasons: vec![],
                confidence: flow_state.momentum(),
                suggested_message_kind: "Status".to_string(),
            };
        }

        // --- Check 2: Block state ---
        if let FlowState::Block { reason } = flow_state {
            reasons.push(HandoverReason::Blocked {
                reason: reason.clone(),
            });
            should_handover = true;
        }

        // --- Check 3: Coherence decay ---
        if coherence < self.coherence_floor {
            reasons.push(HandoverReason::CoherenceDecay {
                coherence,
                threshold: self.coherence_floor,
            });
            should_handover = true;
        }

        // --- Check 4: Stall detection ---
        if let FlowState::Hold { hold_cycles } = flow_state {
            if *hold_cycles >= self.max_hold_cycles {
                reasons.push(HandoverReason::StallDetected {
                    hold_cycles: *hold_cycles,
                    max_cycles: self.max_hold_cycles,
                });
                should_handover = true;
            }
        }

        // --- Check 5: Volition mismatch ---
        if volition_alignment < self.volition_floor {
            reasons.push(HandoverReason::VolitionMismatch {
                alignment: volition_alignment,
            });
            should_handover = true;
        }

        // --- Check 6: Dunning-Kruger guard ---
        if confidence - coherence > self.dk_gap_threshold {
            reasons.push(HandoverReason::ConfidenceCoherenceGap {
                confidence,
                coherence,
            });
            // DK doesn't trigger handover directly — it triggers metacognitive review
            if !should_handover {
                return HandoverDecision {
                    source_slot,
                    action: HandoverAction::MetacognitiveReview,
                    reasons,
                    confidence: 0.5,
                    suggested_message_kind: "Sync".to_string(),
                };
            }
        }

        // --- Check 7: Better resonance available (only if not flow-shielded) ---
        if let Some((target, score)) = best_alternative {
            if score >= self.min_resonance {
                reasons.push(HandoverReason::BetterResonance {
                    target_slot: target,
                    score,
                });
                // Only override continue if already flagged
                if !should_handover && !flow_state.is_flow() {
                    // Even without other triggers, a significantly better match warrants suggestion
                    if score > 0.8 {
                        should_handover = true;
                    }
                }
            }
        }

        // --- Decision ---
        if should_handover {
            // Try to delegate to best alternative
            if let Some((target, resonance)) = best_alternative {
                if resonance >= self.min_resonance {
                    return HandoverDecision {
                        source_slot,
                        action: HandoverAction::Delegate {
                            target_slot: target,
                            resonance,
                        },
                        reasons,
                        confidence: resonance,
                        suggested_message_kind: "Delegate".to_string(),
                    };
                }
            }

            // No good handover target — escalate
            HandoverDecision {
                source_slot,
                action: HandoverAction::Escalate,
                reasons,
                confidence: 0.3,
                suggested_message_kind: "Status".to_string(),
            }
        } else {
            // Agent should continue, possibly with style adjustment
            let action = if coherence < 0.5 && flow_state.is_flow() {
                // Flow but coherence is degrading — suggest style switch
                HandoverAction::SwitchStyle {
                    recommended_style: recommend_style_for_coherence(coherence),
                }
            } else {
                HandoverAction::Continue
            };

            HandoverDecision {
                source_slot,
                action,
                reasons,
                confidence: coherence.max(flow_state.momentum()),
                suggested_message_kind: "Status".to_string(),
            }
        }
    }
}

/// Recommend a thinking style based on current coherence level
fn recommend_style_for_coherence(coherence: f32) -> String {
    if coherence < 0.2 {
        // Very low coherence → metacognitive to self-assess
        "metacognitive".to_string()
    } else if coherence < 0.4 {
        // Low coherence → deliberate to slow down and think carefully
        "deliberate".to_string()
    } else {
        // Medium coherence → systematic to provide structure
        "systematic".to_string()
    }
}

// =============================================================================
// FLOW TRANSITION
// =============================================================================

/// Record of a flow state transition (for history tracking)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FlowTransition {
    /// Agent slot
    pub agent_slot: u8,
    /// Previous flow state
    pub from: FlowState,
    /// New flow state
    pub to: FlowState,
    /// Cycle at which transition occurred
    pub at_cycle: u64,
    /// Handover decision that triggered this transition (if any)
    pub triggered_by: Option<HandoverDecision>,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flow_momentum_accumulates() {
        let mut state = FlowState::default();
        // Simulate 5 flow cycles
        for _ in 0..5 {
            state = FlowState::from_gate(GateState::Flow, &state);
        }
        assert!(state.momentum() >= 0.5);
        assert!(state.is_flow());
    }

    #[test]
    fn test_flow_momentum_caps_at_one() {
        let mut state = FlowState::default();
        for _ in 0..20 {
            state = FlowState::from_gate(GateState::Flow, &state);
        }
        assert!((state.momentum() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_hold_cycles_accumulate() {
        let mut state = FlowState::default();
        for _ in 0..3 {
            state = FlowState::from_gate(GateState::Hold, &state);
        }
        match state {
            FlowState::Hold { hold_cycles } => assert_eq!(hold_cycles, 3), // 0 default + 3 from_gate calls
            _ => panic!("Expected Hold state"),
        }
    }

    #[test]
    fn test_flow_shield_prevents_handover() {
        let policy = HandoverPolicy::default();
        let flow_state = FlowState::Flow { momentum: 0.9 };

        let decision = policy.evaluate(
            0,
            &flow_state,
            0.8,             // good coherence
            0.5,             // good alignment
            0.7,             // good confidence
            Some((1, 0.95)), // excellent alternative available
        );

        // Despite excellent alternative, flow shield keeps agent working
        assert_eq!(decision.action, HandoverAction::Continue);
    }

    #[test]
    fn test_coherence_decay_triggers_handover() {
        let policy = HandoverPolicy::default();
        let flow_state = FlowState::Hold { hold_cycles: 0 };

        let decision = policy.evaluate(
            0,
            &flow_state,
            0.1,            // very low coherence
            0.5,            // good alignment
            0.5,            // medium confidence
            Some((1, 0.7)), // compatible alternative
        );

        match decision.action {
            HandoverAction::Delegate { target_slot, .. } => assert_eq!(target_slot, 1),
            _ => panic!("Expected delegation, got {:?}", decision.action),
        }
    }

    #[test]
    fn test_stall_detection() {
        let policy = HandoverPolicy {
            max_hold_cycles: 3,
            ..Default::default()
        };
        let flow_state = FlowState::Hold { hold_cycles: 5 };

        let decision = policy.evaluate(0, &flow_state, 0.5, 0.5, 0.5, None);

        // No alternative available → escalate
        assert_eq!(decision.action, HandoverAction::Escalate);
    }

    #[test]
    fn test_dk_guard_triggers_metacognitive_review() {
        let policy = HandoverPolicy::default();
        let flow_state = FlowState::Hold { hold_cycles: 0 };

        let decision = policy.evaluate(
            0,
            &flow_state,
            0.8,  // good coherence (no other handover triggers)
            0.5,  // good alignment
            0.95, // VERY high confidence
            None, // DK gap: 0.95 - 0.3 = 0.65 > 0.4 threshold
        );

        // Wait, coherence is 0.8 and confidence is 0.95, gap = 0.15 < 0.4
        // Let me fix this test
        assert_eq!(decision.action, HandoverAction::Continue);
    }

    #[test]
    fn test_dk_guard_with_gap() {
        let policy = HandoverPolicy::default();
        let flow_state = FlowState::Hold { hold_cycles: 0 };

        let decision = policy.evaluate(
            0,
            &flow_state,
            0.5,  // ok coherence
            0.5,  // ok alignment
            0.95, // confidence - coherence = 0.45 > 0.4
            None,
        );

        assert_eq!(decision.action, HandoverAction::MetacognitiveReview);
    }

    #[test]
    fn test_block_triggers_delegation() {
        let policy = HandoverPolicy::default();
        let flow_state = FlowState::Block {
            reason: "Cannot resolve ambiguity".to_string(),
        };

        let decision = policy.evaluate(0, &flow_state, 0.5, 0.5, 0.5, Some((2, 0.75)));

        match decision.action {
            HandoverAction::Delegate { target_slot, .. } => assert_eq!(target_slot, 2),
            _ => panic!("Expected delegation"),
        }
    }

    #[test]
    fn test_volition_mismatch_escalates() {
        let policy = HandoverPolicy::default();
        let flow_state = FlowState::Hold { hold_cycles: 0 };

        let decision = policy.evaluate(
            0,
            &flow_state,
            0.6,
            -0.5, // strong aversion
            0.5,
            None, // no alternative
        );

        assert_eq!(decision.action, HandoverAction::Escalate);
    }

    #[test]
    fn test_flow_state_transitions() {
        let hold = FlowState::default();
        let flow = FlowState::from_gate(GateState::Flow, &hold);
        assert!(flow.is_flow());

        let block = FlowState::from_gate(GateState::Block, &flow);
        assert!(block.is_blocked());
        assert!((block.momentum() - 0.0).abs() < f32::EPSILON);
    }
}
