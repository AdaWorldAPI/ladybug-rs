//! Cortex — XOR Delta Layer Coordinator
//!
//! Replaces procedural grey/white phase separation with XOR delta layers.
//! Ground truth (BindSpace) is `&self` forever — never mutated during processing.
//! Each step produces an ephemeral XOR delta layer (Photoshop/SharePoint model).
//! CollapseGate decides: FLOW → flatten delta, HOLD → keep floating, BLOCK → discard.
//!
//! ```text
//! ┌─────────────────────────────────────┐
//! │  Ground Truth (&self forever)       │  BindSpace
//! ├─────────────────────────────────────┤
//! │  Delta: CognitiveFabric results     │  cycle-scoped
//! ├─────────────────────────────────────┤
//! │  Delta: NARS inference              │  cycle-scoped
//! ├─────────────────────────────────────┤
//! │  Delta: Collapsed superposition     │  committed on FLOW
//! └─────────────────────────────────────┘
//!
//! CollapseGate evaluates the layer stack:
//!   FLOW  → flatten delta into ground truth (ice-cake)
//!   HOLD  → keep delta floating (superpose)
//!   BLOCK → discard delta, switch style
//!
//! Conflict detection: popcount(delta_a AND delta_b) > 0
//! Merge: delta_a ⊕ delta_b = combined diff (XOR is associative)
//! Undo: apply delta again (XOR is self-inverse)
//! ```
//!
//! # Why XOR delta layers?
//!
//! The borrow problem dissolves. Ground truth is `&self` forever — immutable,
//! read by everyone, no locks. Each writer gets an ephemeral delta that only
//! they `&mut`. Readers reconstruct any view with one XOR: O(1), SIMD-friendly.
//!
//! # 10-Layer Integration
//!
//! - L4 (Routing) = thinking style templates, resonance-gated selection
//! - L9 (Validation) = NARS + Brier + Socratic sieve
//! - L10 (Crystallization) = commits to BindSpace via delta layers
//! - CognitiveFabric delegates all 10-layer processing
//!
//! NARS rules are selected by resonance, not hardcoded dispatch.
//! The gestalt superposition resonates with rule fingerprints.
//! Rules that cross the effective threshold fire. fan_out caps how many.

use crate::cognitive::awareness::{AwarenessBlackboard, AwarenessSnapshot, CortexResult};
use crate::cognitive::collapse_gate::GateState;
use crate::cognitive::fabric::{CognitiveFabric, CognitiveState};
use crate::cognitive::style::ThinkingStyle;
use crate::core::Fingerprint;
use crate::nars::inference::{apply_rule, INFERENCE_RULES};
use crate::nars::TruthValue;
use crate::storage::bind_space::{Addr, BindSpace, FINGERPRINT_WORDS, PREFIX_BLACKBOARD};

// =============================================================================
// DELTA LAYER — Ephemeral XOR diff (Photoshop layer / SharePoint co-edit)
// =============================================================================

/// An ephemeral XOR delta layer.
///
/// Ground truth is never mutated during processing. Each writer produces
/// a delta layer containing only the bits that changed.
///
/// - Read: `ground ⊕ delta = writer's view`
/// - Write: only mutates YOUR delta layer (owned `&mut`)
/// - Merge: `delta_a ⊕ delta_b = combined diff`
/// - Conflict: `popcount(delta_a AND delta_b) > 0` = overlapping writes
/// - Undo: apply delta again (XOR is self-inverse)
#[derive(Clone, Debug)]
pub struct DeltaLayer {
    /// Sparse deltas: only addresses that changed
    pub deltas: Vec<(Addr, [u64; FINGERPRINT_WORDS])>,
    /// Processing cycle
    pub cycle: u64,
    /// Gate state when layer was produced
    pub gate: GateState,
    /// Source of this delta
    pub source: DeltaSource,
}

/// Who produced this delta layer
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeltaSource {
    /// CognitiveFabric 10-layer processing
    Fabric,
    /// NARS inference chain
    Inference,
    /// Collapsed output (ready to commit)
    Collapse,
    /// Meta/sieve observation
    Meta,
}

impl DeltaLayer {
    /// Create empty layer
    pub fn new(cycle: u64, source: DeltaSource) -> Self {
        Self {
            deltas: Vec::new(),
            cycle,
            gate: GateState::Hold,
            source,
        }
    }

    /// Write a fingerprint delta at an address (only mutates this layer)
    pub fn write(&mut self, addr: Addr, fingerprint: [u64; FINGERPRINT_WORDS]) {
        self.deltas.push((addr, fingerprint));
    }

    /// Check if this layer conflicts with another.
    ///
    /// Conflict = overlapping writes at the same address.
    /// Returns popcount of (delta_a AND delta_b) — 0 means no conflict.
    pub fn conflicts_with(&self, other: &DeltaLayer) -> u32 {
        let mut total_overlap = 0u32;
        for (addr_a, fp_a) in &self.deltas {
            for (addr_b, fp_b) in &other.deltas {
                if addr_a == addr_b {
                    for i in 0..FINGERPRINT_WORDS {
                        total_overlap += (fp_a[i] & fp_b[i]).count_ones();
                    }
                }
            }
        }
        total_overlap
    }

    /// Merge another layer into this one via XOR composition.
    ///
    /// `delta_a ⊕ delta_b = combined diff`
    pub fn merge(&mut self, other: &DeltaLayer) {
        for (addr, fp) in &other.deltas {
            if let Some(existing) = self.deltas.iter_mut().find(|(a, _)| a == addr) {
                // Same address: XOR the deltas together
                for i in 0..FINGERPRINT_WORDS {
                    existing.1[i] ^= fp[i];
                }
            } else {
                self.deltas.push((*addr, *fp));
            }
        }
    }

    /// Flatten this layer into BindSpace (commit to ground truth).
    ///
    /// Returns number of addresses written.
    pub fn commit_to(&self, bind_space: &mut BindSpace) -> usize {
        let mut written = 0;
        for (addr, fp) in &self.deltas {
            if bind_space.write_at(*addr, *fp) {
                written += 1;
            }
        }
        written
    }

    /// Is this layer empty (no deltas)?
    pub fn is_empty(&self) -> bool {
        self.deltas.is_empty()
    }

    /// Number of addresses touched
    pub fn len(&self) -> usize {
        self.deltas.len()
    }
}

// =============================================================================
// CORTEX — Thin layer-stack coordinator
// =============================================================================

/// The Cortex — coordinates XOR delta layers through the 10-layer stack.
///
/// Does NOT own CausalEngine or CausalSearch (no duplicated state).
/// Delegates all 10-layer processing to CognitiveFabric.
/// Produces ephemeral XOR delta layers instead of mutating shared state.
/// BindSpace is the single source of truth.
///
/// NARS inference rules are selected by resonance — the gestalt
/// superposition resonates with rule fingerprints. No hardcoded dispatch.
pub struct Cortex {
    /// CognitiveFabric: 10-layer stack + QuadTriangles + CollapseGate + Satisfaction
    fabric: CognitiveFabric,

    /// Layer stack evaluator (collapse gate on accumulated evidence)
    blackboard: AwarenessBlackboard,

    /// MetaCognition tracker for L9 Validation (Brier calibration)
    metacog: crate::cognitive::metacog::MetaCognition,

    /// Committed delta layers awaiting projection to BindSpace
    committed: Vec<DeltaLayer>,

    /// Slot counter for BindSpace addressing
    next_slot: u8,
}

impl Cortex {
    /// Create new Cortex with default style
    pub fn new(path: &str) -> Self {
        Self {
            fabric: CognitiveFabric::new(path),
            blackboard: AwarenessBlackboard::new(),
            metacog: crate::cognitive::metacog::MetaCognition::new(),
            committed: Vec::new(),
            next_slot: 0,
        }
    }

    /// Create with specific thinking style
    pub fn with_style(path: &str, style: ThinkingStyle) -> Self {
        Self {
            fabric: CognitiveFabric::with_style(path, style),
            blackboard: AwarenessBlackboard::with_style(style),
            metacog: crate::cognitive::metacog::MetaCognition::new(),
            committed: Vec::new(),
            next_slot: 0,
        }
    }

    // =========================================================================
    // MAIN PIPELINE — produces delta layers, not mutations
    // =========================================================================

    /// Process input through the full cognitive pipeline.
    ///
    /// Produces XOR delta layers instead of mutating state directly.
    /// Call `commit_to()` to flatten committed deltas into BindSpace.
    pub fn process(&mut self, input: &str) -> CortexResult {
        self.process_fp(&Fingerprint::from_content(input))
    }

    /// Process a fingerprint through the pipeline.
    ///
    /// NARS rules are selected by resonance — the gestalt superposition
    /// from the quad-triangle resonates with rule fingerprints. Rules that
    /// cross the effective threshold fire. No hardcoded "deduction" anywhere.
    pub fn process_fp(&mut self, input_fp: &Fingerprint) -> CortexResult {
        // --- Fabric: 10-layer wave processing (reads ground truth) ---
        let cognitive_state = self.fabric.process(input_fp);

        // --- NARS: resonance-gated inference (no hardcoded rule) ---
        let nars_tv = if let Some(last_collapse) = &cognitive_state.last_collapse {
            let premise = TruthValue::new(
                cognitive_state.coherence,
                last_collapse.sd.max(0.01).recip().min(1.0),
            );
            let evidence = TruthValue::new(
                input_fp.density(),
                cognitive_state.consciousness.coherence,
            );

            // Gestalt superposition resonates with rule fingerprints.
            // The resonance algebra selects which rules fire — no dispatch table.
            let gestalt = cognitive_state.triangles.fingerprint();
            let selected_rules = self.fabric.select_inference_rules(&gestalt);

            // Apply all selected rules, revise results together
            let mut merged = TruthValue::unknown();
            for rule_name in &selected_rules {
                if let Some(tv) = apply_rule(rule_name, &premise, &evidence) {
                    merged = merged.revision(&tv);
                }
            }
            merged
        } else {
            TruthValue::new(input_fp.density(), 0.5)
        };

        // --- L9 Validation: MetaCognition tracks calibration ---
        let _meta_assessment = self.metacog.assess(
            self.blackboard.gate(),
            &nars_tv,
        );

        // --- Blackboard: deposit evidence, evaluate collapse gate ---
        self.blackboard.next_cycle();
        self.blackboard.deposit_evidence(input_fp.clone(), nars_tv);
        let gate = self.blackboard.evaluate();

        // --- Layer stack decision ---
        match gate {
            GateState::Flow => {
                let snapshot = self.blackboard.snapshot();

                // Create delta layer from superposition and commit
                let mut layer = DeltaLayer::new(snapshot.cycle, DeltaSource::Collapse);
                let addr = Addr::new(PREFIX_BLACKBOARD, self.next_slot);
                layer.write(addr, fp_to_words(&snapshot.superposition));
                layer.gate = GateState::Flow;
                self.next_slot = self.next_slot.wrapping_add(1);
                self.committed.push(layer);

                CortexResult::Committed(snapshot)
            }
            GateState::Hold => {
                // Keep delta floating — accumulate more evidence
                CortexResult::Superposed(self.blackboard.snapshot())
            }
            GateState::Block => {
                let suggest = self.blackboard.suggest_style_switch();
                let snapshot = self.blackboard.snapshot();
                // Discard delta — evidence too dispersed
                CortexResult::Blocked {
                    snapshot,
                    suggest_style: suggest,
                }
            }
        }
    }

    // =========================================================================
    // DELTA LAYER PROJECTION
    // =========================================================================

    /// Flatten all committed deltas into BindSpace (the "ice-cake" operation).
    ///
    /// Returns number of addresses written.
    pub fn commit_to(&mut self, bind_space: &mut BindSpace) -> usize {
        let mut total = 0;
        for layer in self.committed.drain(..) {
            total += layer.commit_to(bind_space);
        }
        total
    }

    /// Get committed delta layers (for inspection)
    pub fn committed_layers(&self) -> &[DeltaLayer] {
        &self.committed
    }

    /// Take committed deltas (consumes them)
    pub fn take_committed(&mut self) -> Vec<DeltaLayer> {
        std::mem::take(&mut self.committed)
    }

    /// Check if two layers conflict (overlapping writes).
    ///
    /// Returns popcount of overlapping bits — 0 means safe to merge.
    pub fn check_conflict(a: &DeltaLayer, b: &DeltaLayer) -> u32 {
        a.conflicts_with(b)
    }

    // =========================================================================
    // NARS INFERENCE (stateless — just delegates to rules)
    // =========================================================================

    /// Apply NARS inference rule by name
    pub fn infer(
        &self,
        rule: &str,
        premise1: &TruthValue,
        premise2: &TruthValue,
    ) -> Option<TruthValue> {
        apply_rule(rule, premise1, premise2)
    }

    /// Get all available inference rule names
    pub fn inference_rules(&self) -> &[&str] {
        INFERENCE_RULES
    }

    // =========================================================================
    // STYLE & STATE
    // =========================================================================

    /// Get current thinking style
    pub fn style(&self) -> ThinkingStyle {
        self.fabric.style()
    }

    /// Set thinking style (propagates to fabric L4 and blackboard)
    pub fn set_style(&mut self, style: ThinkingStyle) {
        self.fabric.set_style(style);
        self.blackboard.set_style(style);
    }

    /// Get cognitive state from fabric (delegates to 10-layer stack)
    pub fn cognitive_state(&mut self, input: &Fingerprint) -> CognitiveState {
        self.fabric.process(input)
    }

    /// Get metacognition state (Brier calibration, meta-confidence)
    pub fn metacognition(&self) -> &crate::cognitive::metacog::MetaCognition {
        &self.metacog
    }

    /// Record a prediction outcome for calibration tracking.
    /// Call this when you know whether a previous prediction was correct.
    pub fn record_outcome(&mut self, predicted_confidence: f32, actual_outcome: f32) {
        self.metacog.record_outcome(predicted_confidence, actual_outcome);
    }

    /// Get awareness snapshot (Clone, no borrow)
    pub fn snapshot(&self) -> AwarenessSnapshot {
        self.blackboard.snapshot()
    }

    /// Is the cortex in FLOW state?
    pub fn is_flow(&self) -> bool {
        self.blackboard.gate() == GateState::Flow
    }

    /// Is the cortex blocked?
    pub fn is_blocked(&self) -> bool {
        self.blackboard.gate() == GateState::Block
    }

    /// Get hold streak (consecutive HOLD cycles)
    pub fn hold_streak(&self) -> usize {
        self.blackboard.hold_streak()
    }

    /// Get block streak (consecutive BLOCK cycles)
    pub fn block_streak(&self) -> usize {
        self.blackboard.block_streak()
    }

    /// Get recent collapse decisions
    pub fn recent_collapses(
        &self,
        n: usize,
    ) -> &[crate::cognitive::collapse_gate::CollapseDecision] {
        self.blackboard.recent_collapses(n)
    }
}

// =============================================================================
// HELPERS
// =============================================================================

/// Convert Fingerprint to raw word array for BindSpace compatibility
fn fp_to_words(fp: &Fingerprint) -> [u64; FINGERPRINT_WORDS] {
    let raw = fp.as_raw();
    let mut result = [0u64; FINGERPRINT_WORDS];
    result.copy_from_slice(&raw[..FINGERPRINT_WORDS]);
    result
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cortex_creation() {
        let cortex = Cortex::new("test");
        assert_eq!(cortex.style(), ThinkingStyle::Analytical);
    }

    #[test]
    fn test_cortex_with_style() {
        let cortex = Cortex::with_style("test", ThinkingStyle::Creative);
        assert_eq!(cortex.style(), ThinkingStyle::Creative);
    }

    #[test]
    fn test_delta_layer_write() {
        let mut layer = DeltaLayer::new(0, DeltaSource::Fabric);
        let addr = Addr::new(PREFIX_BLACKBOARD, 0);
        let fp = fp_to_words(&Fingerprint::from_content("test"));
        layer.write(addr, fp);
        assert_eq!(layer.len(), 1);
        assert!(!layer.is_empty());
    }

    #[test]
    fn test_delta_layer_merge_no_conflict() {
        let mut layer_a = DeltaLayer::new(0, DeltaSource::Fabric);
        let mut layer_b = DeltaLayer::new(0, DeltaSource::Inference);

        // Different addresses — no conflict
        layer_a.write(
            Addr::new(PREFIX_BLACKBOARD, 0),
            fp_to_words(&Fingerprint::from_content("a")),
        );
        layer_b.write(
            Addr::new(PREFIX_BLACKBOARD, 1),
            fp_to_words(&Fingerprint::from_content("b")),
        );

        assert_eq!(layer_a.conflicts_with(&layer_b), 0);
        layer_a.merge(&layer_b);
        assert_eq!(layer_a.len(), 2);
    }

    #[test]
    fn test_delta_layer_conflict_detection() {
        let mut layer_a = DeltaLayer::new(0, DeltaSource::Fabric);
        let mut layer_b = DeltaLayer::new(0, DeltaSource::Inference);

        let addr = Addr::new(PREFIX_BLACKBOARD, 0);
        let fp = fp_to_words(&Fingerprint::from_content("overlapping_data"));

        // Same address, same content — maximum conflict
        layer_a.write(addr, fp);
        layer_b.write(addr, fp);

        let overlap = layer_a.conflicts_with(&layer_b);
        assert!(overlap > 0, "Same addr + same data should conflict");
    }

    #[test]
    fn test_delta_xor_self_inverse() {
        let mut layer = DeltaLayer::new(0, DeltaSource::Fabric);
        let addr = Addr::new(PREFIX_BLACKBOARD, 0);
        let fp = fp_to_words(&Fingerprint::from_content("data"));

        layer.write(addr, fp);
        let clone = layer.clone();
        // Merge with self = XOR self-inverse → all zeros
        layer.merge(&clone);

        if let Some((_, merged_fp)) = layer.deltas.iter().find(|(a, _)| *a == addr) {
            for word in merged_fp {
                assert_eq!(*word, 0, "XOR with self should produce zero (self-inverse)");
            }
        }
    }

    #[test]
    fn test_cortex_process() {
        let mut cortex = Cortex::new("test");
        let result = cortex.process("Hello world");
        let _snap = result.snapshot();
    }

    #[test]
    fn test_cortex_process_fp() {
        let mut cortex = Cortex::new("test");
        let fp = Fingerprint::from_content("test input");
        let result = cortex.process_fp(&fp);
        let _snap = result.snapshot();
    }

    #[test]
    fn test_cortex_nars_inference() {
        let cortex = Cortex::new("test");
        let p1 = TruthValue::new(0.9, 0.9);
        let p2 = TruthValue::new(0.8, 0.8);

        let deduction = cortex.infer("deduction", &p1, &p2);
        assert!(deduction.is_some());

        // All rules should work
        for rule in cortex.inference_rules() {
            assert!(
                cortex.infer(rule, &p1, &p2).is_some(),
                "Rule {} should work",
                rule
            );
        }
    }

    #[test]
    fn test_cortex_style_switch() {
        let mut cortex = Cortex::with_style("test", ThinkingStyle::Analytical);
        assert_eq!(cortex.style(), ThinkingStyle::Analytical);
        cortex.set_style(ThinkingStyle::Creative);
        assert_eq!(cortex.style(), ThinkingStyle::Creative);
    }

    #[test]
    fn test_cortex_multiple_cycles() {
        let mut cortex = Cortex::new("test");
        let r1 = cortex.process("first");
        let r2 = cortex.process("second");
        let r3 = cortex.process("third");
        assert!(r1.snapshot().cycle <= r2.snapshot().cycle);
        assert!(r2.snapshot().cycle <= r3.snapshot().cycle);
    }

    #[test]
    fn test_cortex_commit_to_bindspace() {
        let mut cortex = Cortex::new("test");
        let mut bind_space = BindSpace::new();

        // Process several inputs to produce deltas
        for i in 0..5 {
            cortex.process(&format!("input_{}", i));
        }

        // Flatten committed deltas into BindSpace
        let written = cortex.commit_to(&mut bind_space);
        assert!(
            cortex.committed.is_empty(),
            "All deltas should be drained after commit"
        );
        let _ = written; // May be 0 if all were HOLD
    }

    #[test]
    fn test_cortex_full_pipeline() {
        let mut cortex = Cortex::with_style("test", ThinkingStyle::Analytical);

        // Exercises: 7-layer stack, NARS inference, CollapseGate, delta production
        let result = cortex.process("test stimulus");
        let snap = result.snapshot();
        assert!(!snap.evidence.is_empty(), "Should have input evidence");
    }

    #[test]
    fn test_delta_commit_writes_to_blackboard_surface() {
        let mut bind_space = BindSpace::new();

        let mut layer = DeltaLayer::new(1, DeltaSource::Collapse);
        let addr = Addr::new(PREFIX_BLACKBOARD, 42);
        let fp = fp_to_words(&Fingerprint::from_content("committed_result"));
        layer.write(addr, fp);
        layer.gate = GateState::Flow;

        let written = layer.commit_to(&mut bind_space);
        assert_eq!(written, 1, "Should write to blackboard surface (0x0E)");

        // Verify it landed in BindSpace
        let node = bind_space.read(addr);
        assert!(node.is_some(), "Written node should be readable");
    }
}
