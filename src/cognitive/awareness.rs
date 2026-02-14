//! Awareness Blackboard — Borrow-Safe Cognitive State
//!
//! The blood-brain barrier between read-only computation (grey matter)
//! and mutable state updates (white matter).
//!
//! Grey matter produces owned values → Blackboard accumulates them
//! White matter reads Clone'd snapshots → Never borrows grey matter
//!
//! ```text
//! ┌────────────── GREY MATTER (&self) ──────────────┐
//! │  GrammarEngine.parse()     → CausalityFlow      │
//! │  CognitiveFabric.process() → CognitiveState     │
//! │  NarsInference.apply()     → TruthValue          │
//! └──────────────────────┬───────────────────────────┘
//!                        │ owned values (not references)
//!                        ▼
//! ┌──────────── AWARENESS BLACKBOARD ───────────────┐
//! │  Accumulates evidence, evaluates collapse gate   │
//! │  Provides Clone'd snapshots to white matter      │
//! └──────────────────────┬───────────────────────────┘
//!                        │ Clone'd state
//!                        ▼
//! ┌────────────── WHITE MATTER (&mut self) ──────────┐
//! │  CausalEngine.store_intervention()               │
//! │  BindSpace.write_at()                            │
//! └──────────────────────────────────────────────────┘
//! ```

use crate::cognitive::collapse_gate::{CollapseDecision, GateState, evaluate_gate};
use crate::cognitive::style::ThinkingStyle;
use crate::core::Fingerprint;
use crate::nars::TruthValue;

// =============================================================================
// AWARENESS SNAPSHOT — cheaply cloneable, no borrows
// =============================================================================

/// Snapshot of awareness state — cheaply cloneable, no borrows held
#[derive(Clone, Debug)]
pub struct AwarenessSnapshot {
    /// Current thinking style
    pub style: ThinkingStyle,
    /// Last gate evaluation
    pub gate: GateState,
    /// Coherence metric (0.0 = chaos, 1.0 = crystallized)
    pub coherence: f32,
    /// Accumulated evidence this cycle
    pub evidence: Vec<(Fingerprint, TruthValue)>,
    /// Processing cycle counter
    pub cycle: u64,
    /// Superposed fingerprint (bundle of all evidence)
    pub superposition: Fingerprint,
}

// =============================================================================
// CORTEX RESULT — what process() returns
// =============================================================================

/// Result of a cortex processing cycle
#[derive(Clone, Debug)]
pub enum CortexResult {
    /// Gate=FLOW: evidence collapsed, committed to winner
    Committed(AwarenessSnapshot),
    /// Gate=HOLD: maintaining superposition, accumulating more evidence
    Superposed(AwarenessSnapshot),
    /// Gate=BLOCK: high dispersion, style switch suggested
    Blocked {
        snapshot: AwarenessSnapshot,
        suggest_style: ThinkingStyle,
    },
}

impl CortexResult {
    /// Get the snapshot regardless of result type
    pub fn snapshot(&self) -> &AwarenessSnapshot {
        match self {
            Self::Committed(s) | Self::Superposed(s) | Self::Blocked { snapshot: s, .. } => s,
        }
    }

    /// Was this a FLOW collapse?
    pub fn is_committed(&self) -> bool {
        matches!(self, Self::Committed(_))
    }

    /// Was this a BLOCK?
    pub fn is_blocked(&self) -> bool {
        matches!(self, Self::Blocked { .. })
    }

    /// Gate state
    pub fn gate(&self) -> GateState {
        self.snapshot().gate
    }
}

// =============================================================================
// AWARENESS BLACKBOARD
// =============================================================================

/// The awareness blackboard — owns all cognitive state between phases.
///
/// This is the borrow boundary. Grey matter results arrive as owned values.
/// White matter reads Clone'd snapshots. No simultaneous borrows cross this.
pub struct AwarenessBlackboard {
    /// Current thinking style
    style: ThinkingStyle,

    /// Processing cycle counter
    cycle: u64,

    /// Accumulated evidence from grey matter (owned, not borrowed)
    evidence_buffer: Vec<(Fingerprint, TruthValue)>,

    /// Collapse gate history (recent decisions)
    collapse_history: Vec<CollapseDecision>,

    /// Superposed fingerprint (bundled from all evidence this cycle)
    superposition: Fingerprint,

    /// Last gate evaluation
    last_gate: GateState,

    /// Coherence metric
    coherence: f32,

    /// How many consecutive HOLD cycles
    hold_streak: usize,

    /// How many consecutive BLOCK cycles
    block_streak: usize,
}

impl AwarenessBlackboard {
    /// Create new blackboard with default style
    pub fn new() -> Self {
        Self {
            style: ThinkingStyle::Deliberate,
            cycle: 0,
            evidence_buffer: Vec::new(),
            collapse_history: Vec::new(),
            superposition: Fingerprint::zero(),
            last_gate: GateState::Hold,
            coherence: 0.0,
            hold_streak: 0,
            block_streak: 0,
        }
    }

    /// Create with specific thinking style
    pub fn with_style(style: ThinkingStyle) -> Self {
        let mut bb = Self::new();
        bb.style = style;
        bb
    }

    // -------------------------------------------------------------------------
    // GREY MATTER DEPOSITS (takes ownership, no borrows held after)
    // -------------------------------------------------------------------------

    /// Grey matter deposits evidence here (takes ownership, no borrows held)
    pub fn deposit_evidence(&mut self, fp: Fingerprint, tv: TruthValue) {
        // Bundle into superposition via XOR (VSA bundling)
        self.superposition = self.superposition.bind(&fp);
        self.evidence_buffer.push((fp, tv));
    }

    /// Deposit multiple evidence items at once
    pub fn deposit_batch(&mut self, evidence: Vec<(Fingerprint, TruthValue)>) {
        for (fp, tv) in evidence {
            self.deposit_evidence(fp, tv);
        }
    }

    // -------------------------------------------------------------------------
    // COLLAPSE GATE EVALUATION
    // -------------------------------------------------------------------------

    /// Evaluate collapse gate on accumulated evidence
    pub fn evaluate(&mut self) -> GateState {
        if self.evidence_buffer.is_empty() {
            self.last_gate = GateState::Hold;
            self.hold_streak += 1;
            self.block_streak = 0;
            return GateState::Hold;
        }

        // Extract confidence scores from truth values
        let scores: Vec<f32> = self
            .evidence_buffer
            .iter()
            .map(|(_, tv)| tv.expectation())
            .collect();

        // Evaluate via collapse gate
        let decision = evaluate_gate(&scores, true);
        let state = decision.state;

        // Update coherence from decision SD
        self.coherence = if decision.sd.is_finite() {
            (1.0 - decision.sd / 0.5).max(0.0)
        } else {
            0.0
        };

        // Track streaks
        match state {
            GateState::Flow => {
                self.hold_streak = 0;
                self.block_streak = 0;
            }
            GateState::Hold => {
                self.hold_streak += 1;
                self.block_streak = 0;
            }
            GateState::Block => {
                self.hold_streak = 0;
                self.block_streak += 1;
            }
        }

        self.collapse_history.push(decision);
        self.last_gate = state;
        state
    }

    // -------------------------------------------------------------------------
    // SNAPSHOTS (Clone, no borrow)
    // -------------------------------------------------------------------------

    /// Snapshot for white matter consumers (Clone, no borrow held)
    pub fn snapshot(&self) -> AwarenessSnapshot {
        AwarenessSnapshot {
            style: self.style,
            gate: self.last_gate,
            coherence: self.coherence,
            evidence: self.evidence_buffer.clone(),
            cycle: self.cycle,
            superposition: self.superposition.clone(),
        }
    }

    // -------------------------------------------------------------------------
    // CYCLE MANAGEMENT
    // -------------------------------------------------------------------------

    /// Advance to next cycle (clears evidence buffer, keeps state)
    pub fn next_cycle(&mut self) {
        self.cycle += 1;
        self.evidence_buffer.clear();
        self.superposition = Fingerprint::zero();
    }

    /// Get current cycle
    pub fn cycle(&self) -> u64 {
        self.cycle
    }

    /// Get current style
    pub fn style(&self) -> ThinkingStyle {
        self.style
    }

    /// Set thinking style
    pub fn set_style(&mut self, style: ThinkingStyle) {
        self.style = style;
    }

    /// Get last gate state
    pub fn gate(&self) -> GateState {
        self.last_gate
    }

    /// Get coherence metric
    pub fn coherence(&self) -> f32 {
        self.coherence
    }

    /// How many consecutive HOLD cycles?
    pub fn hold_streak(&self) -> usize {
        self.hold_streak
    }

    /// How many consecutive BLOCK cycles?
    pub fn block_streak(&self) -> usize {
        self.block_streak
    }

    /// Get last N collapse decisions
    pub fn recent_collapses(&self, n: usize) -> &[CollapseDecision] {
        let start = self.collapse_history.len().saturating_sub(n);
        &self.collapse_history[start..]
    }

    /// Suggest a style switch when blocked.
    ///
    /// Follows the fallback sequence from agent card spec:
    /// Analytical → Systematic → Exploratory → Creative → Metacognitive
    pub fn suggest_style_switch(&self) -> ThinkingStyle {
        match self.style {
            ThinkingStyle::Analytical => ThinkingStyle::Systematic,
            ThinkingStyle::Systematic => ThinkingStyle::Exploratory,
            ThinkingStyle::Exploratory => ThinkingStyle::Creative,
            ThinkingStyle::Creative => ThinkingStyle::Divergent,
            ThinkingStyle::Convergent => ThinkingStyle::Analytical,
            ThinkingStyle::Focused => ThinkingStyle::Diffuse,
            ThinkingStyle::Diffuse => ThinkingStyle::Peripheral,
            ThinkingStyle::Deliberate => ThinkingStyle::Intuitive,
            ThinkingStyle::Intuitive => ThinkingStyle::Analytical,
            _ => ThinkingStyle::Metacognitive,
        }
    }
}

impl Default for AwarenessBlackboard {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blackboard_creation() {
        let bb = AwarenessBlackboard::new();
        assert_eq!(bb.cycle(), 0);
        assert_eq!(bb.gate(), GateState::Hold);
        assert_eq!(bb.style(), ThinkingStyle::Deliberate);
    }

    #[test]
    fn test_deposit_and_evaluate_flow() {
        let mut bb = AwarenessBlackboard::new();

        // Deposit evidence with tight consensus (should → FLOW)
        bb.deposit_evidence(
            Fingerprint::from_content("evidence_a"),
            TruthValue::new(0.9, 0.9),
        );
        bb.deposit_evidence(
            Fingerprint::from_content("evidence_b"),
            TruthValue::new(0.88, 0.85),
        );
        bb.deposit_evidence(
            Fingerprint::from_content("evidence_c"),
            TruthValue::new(0.91, 0.87),
        );

        let gate = bb.evaluate();
        // All expectations are close (~0.86-0.87), should be low SD → FLOW
        assert_eq!(gate, GateState::Flow, "Tight consensus should collapse to FLOW");
    }

    #[test]
    fn test_deposit_and_evaluate_block() {
        let mut bb = AwarenessBlackboard::new();

        // Deposit evidence with high dispersion (should → BLOCK)
        bb.deposit_evidence(
            Fingerprint::from_content("strong_yes"),
            TruthValue::new(1.0, 0.99),
        );
        bb.deposit_evidence(
            Fingerprint::from_content("strong_no"),
            TruthValue::new(0.0, 0.99),
        );

        let gate = bb.evaluate();
        assert_eq!(gate, GateState::Block, "Contradictory evidence should BLOCK");
    }

    #[test]
    fn test_snapshot_is_independent() {
        let mut bb = AwarenessBlackboard::new();
        bb.deposit_evidence(
            Fingerprint::from_content("data"),
            TruthValue::new(0.7, 0.8),
        );

        let snap = bb.snapshot();

        // Mutate blackboard after snapshot
        bb.next_cycle();
        assert_eq!(bb.cycle(), 1);

        // Snapshot should still reflect old state
        assert_eq!(snap.cycle, 0);
        assert_eq!(snap.evidence.len(), 1);
    }

    #[test]
    fn test_cycle_clears_evidence() {
        let mut bb = AwarenessBlackboard::new();
        bb.deposit_evidence(
            Fingerprint::from_content("item"),
            TruthValue::new(0.5, 0.5),
        );
        assert_eq!(bb.snapshot().evidence.len(), 1);

        bb.next_cycle();
        assert_eq!(bb.snapshot().evidence.len(), 0);
        assert_eq!(bb.cycle(), 1);
    }

    #[test]
    fn test_style_switch_suggestion() {
        let bb = AwarenessBlackboard::with_style(ThinkingStyle::Analytical);
        assert_eq!(bb.suggest_style_switch(), ThinkingStyle::Systematic);

        let bb = AwarenessBlackboard::with_style(ThinkingStyle::Systematic);
        assert_eq!(bb.suggest_style_switch(), ThinkingStyle::Exploratory);
    }

    #[test]
    fn test_hold_streak_tracking() {
        let mut bb = AwarenessBlackboard::new();

        // Empty evidence → HOLD
        bb.evaluate();
        assert_eq!(bb.hold_streak(), 1);
        bb.evaluate();
        assert_eq!(bb.hold_streak(), 2);

        // Tight evidence → FLOW resets streak
        bb.deposit_evidence(
            Fingerprint::from_content("x"),
            TruthValue::new(0.9, 0.9),
        );
        bb.deposit_evidence(
            Fingerprint::from_content("y"),
            TruthValue::new(0.88, 0.88),
        );
        bb.evaluate();
        assert_eq!(bb.hold_streak(), 0);
    }
}
