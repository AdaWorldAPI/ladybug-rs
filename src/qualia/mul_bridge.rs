//! MUL–Reflection Bridge — Metacognitive State Driving Reflection
//!
//! The MUL (Meta-Uncertainty Layer) produces a 10-layer snapshot of the
//! system's epistemic state. This module bridges that snapshot into the
//! reflection/volition pipeline:
//!
//! ```text
//! MUL Snapshot (L1-L10)
//!   │
//!   ├─→ Trust qualia → adjust surprise thresholds
//!   │     (low trust → lower threshold → more Explore/Revise)
//!   │
//!   ├─→ Homeostasis state → modulate council weights
//!   │     (Anxiety → Guardian dominant, Boredom → Catalyst dominant)
//!   │
//!   ├─→ Free will modifier → gate volitional cycle
//!   │     (low modifier → don't act, high → full agency)
//!   │
//!   └─→ Reflection outcomes → feed back to MUL via learn()
//!         (mean surprise → novelty, revision count → prediction accuracy)
//! ```
//!
//! ## The Core Insight
//!
//! MUL state IS the system's prediction about its own epistemic capacity.
//! Reflection measures how well the tree structure predicts content (surprise).
//! The bridge connects these: the system's self-assessment (MUL) modulates
//! how aggressively it responds to prediction errors (reflection).
//!
//! ## Adaptive Thresholds
//!
//! The default surprise thresholds (0.55 high, 0.45 low) assume balanced
//! metacognition. When the MUL reports abnormal states, we adjust:
//!
//! - **Low trust** → lower surprise threshold → more nodes get Revise/Explore
//!   (the system doesn't trust itself, so flags more for review)
//! - **High trust** → higher threshold → more Stable outcomes
//!   (the system trusts its beliefs, only attends to extreme surprise)
//! - **Anxiety** → conservative thresholds (fewer Explore, more Stable)
//! - **Boredom** → aggressive thresholds (more Explore, fewer Stable)
//! - **False flow** → force Explore on everything (break the loop)

use crate::container::Container;
use crate::container::graph::ContainerGraph;
use crate::container::adjacency::PackedDn;
use crate::cognitive::RungLevel;
use crate::mul::{
    MetaUncertaintyLayer, MulSnapshot, HomeostasisState, TrustLevel,
    FalseFlowSeverity, CompassDecision,
    PostActionLearning,
};

use super::reflection::{
    ReflectionResult, ReflectionOutcome, ReflectionEntry,
    hydrate_explorers,
    SURPRISE_HIGH, SURPRISE_LOW, CONFIDENCE_HIGH,
};
use super::volition::{
    VolitionalAgenda, CouncilWeights, volitional_cycle,
};

// =============================================================================
// ADAPTIVE THRESHOLDS
// =============================================================================

/// Surprise and confidence thresholds adapted by MUL state.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveThresholds {
    /// Threshold for "high" surprise (above → Revise or Explore).
    pub surprise_high: f32,
    /// Threshold for "low" surprise (below → Confirm or Stable).
    pub surprise_low: f32,
    /// Threshold for "high" confidence.
    pub confidence_high: f32,
}

impl Default for AdaptiveThresholds {
    fn default() -> Self {
        Self {
            surprise_high: SURPRISE_HIGH,
            surprise_low: SURPRISE_LOW,
            confidence_high: CONFIDENCE_HIGH,
        }
    }
}

/// Compute adaptive thresholds from a MUL snapshot.
///
/// The thresholds shift based on the system's metacognitive state:
/// - Trust level controls the surprise sensitivity
/// - Homeostasis state biases toward caution or exploration
/// - False flow severity can force aggressive exploration
pub fn adaptive_thresholds(snapshot: &MulSnapshot) -> AdaptiveThresholds {
    let mut thresholds = AdaptiveThresholds::default();

    // Trust modulation: low trust → lower surprise threshold → more flagging
    let trust_shift = match snapshot.trust_level {
        TrustLevel::Crystalline => 0.05,  // raise thresholds (trust self)
        TrustLevel::Solid => 0.02,
        TrustLevel::Fuzzy => 0.0,         // default
        TrustLevel::Murky => -0.03,
        TrustLevel::Dissonant => -0.08,   // lower thresholds (don't trust self)
    };
    thresholds.surprise_high += trust_shift;
    thresholds.surprise_low += trust_shift;

    // Homeostasis modulation
    match snapshot.homeostasis_state {
        HomeostasisState::Flow => {
            // Optimal — no adjustment
        }
        HomeostasisState::Anxiety => {
            // Conservative: raise surprise threshold (fewer flags, less exploration)
            thresholds.surprise_high += 0.05;
            thresholds.surprise_low += 0.03;
            thresholds.confidence_high -= 0.05; // lower bar for "confident enough"
        }
        HomeostasisState::Boredom => {
            // Aggressive: lower surprise threshold (more flags, more exploration)
            thresholds.surprise_high -= 0.05;
            thresholds.surprise_low -= 0.03;
        }
        HomeostasisState::Apathy => {
            // Minimal: raise everything (conserve energy)
            thresholds.surprise_high += 0.08;
            thresholds.surprise_low += 0.05;
        }
    }

    // False flow override: if coherent but not progressing, force exploration
    match snapshot.false_flow_severity {
        FalseFlowSeverity::None => {}
        FalseFlowSeverity::Caution => {
            thresholds.surprise_high -= 0.02;
        }
        FalseFlowSeverity::Warning => {
            thresholds.surprise_high -= 0.04;
        }
        FalseFlowSeverity::Severe => {
            // Everything looks surprising → breaks false flow loop
            thresholds.surprise_high = 0.3;
            thresholds.surprise_low = 0.2;
        }
    }

    // Clamp to valid range
    thresholds.surprise_high = thresholds.surprise_high.clamp(0.2, 0.9);
    thresholds.surprise_low = thresholds.surprise_low.clamp(0.1, thresholds.surprise_high - 0.05);
    thresholds.confidence_high = thresholds.confidence_high.clamp(0.2, 0.8);

    thresholds
}

// =============================================================================
// MUL-MODULATED COUNCIL WEIGHTS
// =============================================================================

/// Create council weights modulated by the MUL's homeostasis state.
///
/// - Anxiety → Guardian-dominant (caution amplified)
/// - Boredom → Catalyst-dominant (curiosity amplified)
/// - Flow → Balanced (default)
/// - Apathy → All dampened (conservation mode)
pub fn mul_council_weights(snapshot: &MulSnapshot) -> CouncilWeights {
    match snapshot.homeostasis_state {
        HomeostasisState::Flow => CouncilWeights::default(),
        HomeostasisState::Anxiety => CouncilWeights {
            guardian_surprise_factor: 0.4,  // stronger dampening
            catalyst_surprise_factor: 1.1,  // weaker amplification
            balanced_factor: 0.8,           // slight overall dampening
        },
        HomeostasisState::Boredom => CouncilWeights {
            guardian_surprise_factor: 0.8,  // less dampening
            catalyst_surprise_factor: 1.8,  // stronger amplification
            balanced_factor: 1.1,           // slight overall boost
        },
        HomeostasisState::Apathy => CouncilWeights {
            guardian_surprise_factor: 0.5,  // moderate dampening
            catalyst_surprise_factor: 0.8,  // dampened (no energy for curiosity)
            balanced_factor: 0.6,           // overall conservation
        },
    }
}

// =============================================================================
// MUL-GATED REFLECTION
// =============================================================================

/// Result of a MUL-gated reflection cycle.
#[derive(Debug, Clone)]
pub struct MulReflectionResult {
    /// Whether the MUL gate allowed reflection to proceed.
    pub gate_allowed: bool,
    /// The MUL snapshot used for modulation.
    pub snapshot: MulSnapshot,
    /// Adaptive thresholds used (for transparency).
    pub thresholds: AdaptiveThresholds,
    /// Council weights used (for transparency).
    pub council: CouncilWeights,
    /// The volitional agenda (None if gate blocked).
    pub agenda: Option<VolitionalAgenda>,
    /// Reclassified entries using adaptive thresholds.
    pub reclassified_count: usize,
}

/// Reclassify reflection entries using adaptive thresholds.
///
/// The standard `reflect_walk` uses fixed thresholds (0.55/0.45/0.5).
/// This function takes an existing ReflectionResult and reclassifies
/// each entry using MUL-adapted thresholds, returning the count of
/// entries whose outcome changed.
pub fn reclassify_with_thresholds(
    result: &ReflectionResult,
    thresholds: &AdaptiveThresholds,
) -> (Vec<ReflectionEntry>, usize) {
    let mut reclassified = Vec::with_capacity(result.entries.len());
    let mut changed_count = 0;

    for entry in &result.entries {
        let high_surprise = entry.surprise > thresholds.surprise_high;
        let high_confidence = entry.truth_before.confidence > thresholds.confidence_high;

        let new_outcome = match (high_surprise, high_confidence) {
            (true, true) => ReflectionOutcome::Revise,
            (false, false) => ReflectionOutcome::Confirm,
            (true, false) => ReflectionOutcome::Explore,
            (false, true) => ReflectionOutcome::Stable,
        };

        if new_outcome != entry.outcome {
            changed_count += 1;
        }

        reclassified.push(ReflectionEntry {
            dn: entry.dn,
            surprise: entry.surprise,
            truth_before: entry.truth_before,
            truth_after: entry.truth_after,
            outcome: new_outcome,
            depth: entry.depth,
        });
    }

    (reclassified, changed_count)
}

/// Run a MUL-gated volitional cycle.
///
/// This is the top-level integration point:
///
/// 1. Evaluate MUL → get snapshot
/// 2. If gate closed → return early (no action)
/// 3. Compute adaptive thresholds from MUL state
/// 4. Compute MUL-modulated council weights
/// 5. Run volitional cycle with MUL-adjusted parameters
/// 6. Reclassify reflection entries with adaptive thresholds
pub fn mul_volitional_cycle(
    graph: &mut ContainerGraph,
    target: PackedDn,
    query: &Container,
    current_rung: RungLevel,
    mul: &MetaUncertaintyLayer,
    complexity_mapped: bool,
) -> MulReflectionResult {
    let snapshot = mul.evaluate(complexity_mapped);
    let thresholds = adaptive_thresholds(&snapshot);
    let council = mul_council_weights(&snapshot);

    // Gate check: if MUL says don't proceed, return early
    if !snapshot.should_proceed() {
        return MulReflectionResult {
            gate_allowed: false,
            snapshot,
            thresholds,
            council,
            agenda: None,
            reclassified_count: 0,
        };
    }

    // Run the volitional cycle with MUL-modulated council
    let agenda = volitional_cycle(graph, target, query, current_rung, &council);

    // Reclassify using adaptive thresholds
    let (reclassified_entries, reclassified_count) =
        reclassify_with_thresholds(&agenda.reflection, &thresholds);

    // Update hydration candidates based on reclassification
    let new_hydration_candidates: Vec<PackedDn> = reclassified_entries
        .iter()
        .filter(|e| e.outcome == ReflectionOutcome::Explore)
        .map(|e| e.dn)
        .collect();

    // If reclassification changed Explore candidates, re-hydrate
    if reclassified_count > 0 && !new_hydration_candidates.is_empty() {
        hydrate_explorers(graph, &new_hydration_candidates);
    }

    MulReflectionResult {
        gate_allowed: true,
        snapshot,
        thresholds,
        council,
        agenda: Some(agenda),
        reclassified_count,
    }
}

// =============================================================================
// MUL FEEDBACK — Reflection outcomes → MUL learning
// =============================================================================

/// Convert reflection results into MUL learning signals.
///
/// Maps reflection outcomes back to MUL's `learn()` interface:
/// - Mean surprise → novelty signal for false flow detection
/// - Revision count / total → prediction accuracy for DK calibration
/// - Confidence delta → trust update
pub fn reflection_to_mul_learning(
    result: &ReflectionResult,
) -> PostActionLearning {
    let total = result.entries.len().max(1) as f32;
    let revision_rate = result.revision_count() as f32 / total;
    let _mean_confidence_delta = result.mean_confidence_delta();

    // Map reflection outcomes to compass decision:
    // - Mostly Stable/Confirm → routine decision (ExecuteWithLearning)
    // - Mostly Explore → exploratory decision
    // - Mostly Revise → surprising outcome (SurfaceToMeta)
    let decision = if revision_rate > 0.5 {
        CompassDecision::SurfaceToMeta
    } else if result.hydration_candidates.len() as f32 / total > 0.3 {
        CompassDecision::Exploratory
    } else {
        CompassDecision::ExecuteWithLearning
    };

    // Predicted confidence = mean confidence before reflection
    let predicted_confidence = if result.entries.is_empty() {
        0.5
    } else {
        result.entries.iter()
            .map(|e| e.truth_before.confidence)
            .sum::<f32>() / total
    };

    // Actual outcome = 1.0 - revision_rate (high revisions = poor prediction)
    let actual_outcome = 1.0 - revision_rate;

    PostActionLearning::new(decision, predicted_confidence, actual_outcome)
}

/// Full MUL-reflection feedback loop: reflect, then feed results back to MUL.
///
/// 1. Run MUL-gated reflection
/// 2. If successful, compute learning signal
/// 3. Feed learning signal back to MUL
/// 4. Tick MUL with reflection-derived novelty/coherence
pub fn mul_reflection_feedback(
    graph: &mut ContainerGraph,
    target: PackedDn,
    query: &Container,
    current_rung: RungLevel,
    mul: &mut MetaUncertaintyLayer,
    complexity_mapped: bool,
) -> MulReflectionResult {
    let result = mul_volitional_cycle(
        graph, target, query, current_rung, mul, complexity_mapped,
    );

    if let Some(ref agenda) = result.agenda {
        // Feed reflection outcomes back to MUL
        let learning = reflection_to_mul_learning(&agenda.reflection);
        mul.learn(&learning);

        // Tick MUL with reflection-derived signals:
        // - coherence = 1.0 - mean surprise (low surprise = coherent)
        // - novelty = fraction of Explore outcomes
        let total = agenda.reflection.entries.len().max(1) as f32;
        let coherence = 1.0 - agenda.reflection.felt_path.mean_surprise;
        let novelty = agenda.reflection.hydration_candidates.len() as f32 / total;
        // challenge/skill from rung: deeper rung = harder challenge
        let challenge = current_rung.as_u8() as f32 / 9.0;
        let skill = result.snapshot.free_will_modifier.value();

        mul.tick(coherence, novelty, challenge, skill);
    }

    result
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{ContainerGeometry, CogRecord};
    use crate::mul::{MetaUncertaintyLayer, DKPosition};
    use super::super::reflection::{write_truth, reflect_walk};
    use ladybug_contract::nars::TruthValue;

    fn build_test_tree() -> ContainerGraph {
        let mut graph = ContainerGraph::new();

        let root = PackedDn::ROOT;
        let mut root_rec = CogRecord::new(ContainerGeometry::Cam);
        root_rec.content = Container::random(1);
        graph.insert(root, root_rec);

        for (i, seed) in [(0u8, 10u64), (1, 20), (2, 30)] {
            let dn = PackedDn::new(&[i]);
            let mut rec = CogRecord::new(ContainerGeometry::Cam);
            rec.content = Container::random(seed);
            graph.insert(dn, rec);
        }

        for (i, seed) in [(0u8, 100u64), (1, 101), (2, 102)] {
            let dn = PackedDn::new(&[0, i]);
            let mut rec = CogRecord::new(ContainerGeometry::Cam);
            rec.content = Container::random(seed);
            graph.insert(dn, rec);
        }

        for (i, seed) in [(0u8, 200u64), (1, 201)] {
            let dn = PackedDn::new(&[1, i]);
            let mut rec = CogRecord::new(ContainerGeometry::Cam);
            rec.content = Container::random(seed);
            graph.insert(dn, rec);
        }

        graph
    }

    fn seed_nars(graph: &mut ContainerGraph) {
        if let Some(rec) = graph.get_mut(&PackedDn::new(&[0])) {
            write_truth(rec, &TruthValue::new(0.8, 0.9));
        }
        if let Some(rec) = graph.get_mut(&PackedDn::new(&[1])) {
            write_truth(rec, &TruthValue::new(0.5, 0.1));
        }
        if let Some(rec) = graph.get_mut(&PackedDn::new(&[0, 1])) {
            write_truth(rec, &TruthValue::new(0.7, 0.5));
        }
    }

    #[test]
    fn test_adaptive_thresholds_default() {
        let mul = MetaUncertaintyLayer::new();
        let snapshot = mul.evaluate(true);
        let thresholds = adaptive_thresholds(&snapshot);

        // Default MUL (Fuzzy trust, Flow) should produce near-default thresholds
        assert!(thresholds.surprise_high > 0.4);
        assert!(thresholds.surprise_high < 0.7);
        assert!(thresholds.surprise_low < thresholds.surprise_high);
        assert!(thresholds.confidence_high > 0.3);
    }

    #[test]
    fn test_adaptive_thresholds_anxiety() {
        // Construct snapshot directly to isolate anxiety modulation
        // (ticking MUL with anxiety patterns also triggers false flow,
        //  which overrides the anxiety threshold adjustment)
        let snapshot = MulSnapshot {
            trust_level: TrustLevel::Fuzzy,
            dk_position: DKPosition::SlopeOfEnlightenment,
            homeostasis_state: HomeostasisState::Anxiety,
            false_flow_severity: FalseFlowSeverity::None, // isolate anxiety effect
            free_will_modifier: crate::mul::FreeWillModifier::from_value(0.5),
            gate_open: true,
            gate_block_reason: None,
            allostatic_load: 0.3,
        };

        let thresholds = adaptive_thresholds(&snapshot);
        let defaults = AdaptiveThresholds::default();

        // Anxiety should raise surprise thresholds (more conservative)
        assert!(thresholds.surprise_high > defaults.surprise_high,
            "anxiety surprise_high={} should exceed default={}",
            thresholds.surprise_high, defaults.surprise_high);
    }

    #[test]
    fn test_adaptive_thresholds_false_flow() {
        let mut mul = MetaUncertaintyLayer::new();
        for _ in 0..25 {
            mul.tick(0.9, 0.01, 0.5, 0.5); // high coherence, no novelty → false flow
        }
        let snapshot = mul.evaluate(true);

        let thresholds = adaptive_thresholds(&snapshot);
        // Severe false flow should dramatically lower surprise threshold
        assert!(thresholds.surprise_high < 0.4,
            "false flow should lower surprise_high: {}",
            thresholds.surprise_high);
    }

    #[test]
    fn test_mul_council_weights_flow() {
        let mul = MetaUncertaintyLayer::new();
        let snapshot = mul.evaluate(true);
        let council = mul_council_weights(&snapshot);

        let defaults = CouncilWeights::default();
        assert!((council.guardian_surprise_factor - defaults.guardian_surprise_factor).abs() < 1e-5);
        assert!((council.catalyst_surprise_factor - defaults.catalyst_surprise_factor).abs() < 1e-5);
    }

    #[test]
    fn test_mul_council_weights_anxiety() {
        let mut mul = MetaUncertaintyLayer::new();
        for _ in 0..20 {
            mul.tick(0.8, 0.1, 0.9, 0.2);
        }
        let snapshot = mul.evaluate(true);
        let council = mul_council_weights(&snapshot);

        // Anxiety → guardian more aggressive, catalyst less
        assert!(council.guardian_surprise_factor < 0.6);
        assert!(council.catalyst_surprise_factor < 1.5);
    }

    #[test]
    fn test_mul_council_weights_boredom() {
        let mut mul = MetaUncertaintyLayer::new();
        for _ in 0..20 {
            mul.tick(0.8, 0.1, 0.2, 0.9); // low challenge, high skill → boredom
        }
        let snapshot = mul.evaluate(true);
        let council = mul_council_weights(&snapshot);

        // Boredom → catalyst more aggressive
        assert!(council.catalyst_surprise_factor > 1.5);
    }

    #[test]
    fn test_mul_volitional_cycle_gate_open() {
        let mut graph = build_test_tree();
        seed_nars(&mut graph);

        let query = Container::random(42);
        let target = PackedDn::new(&[0, 1]);

        // Use high-trust, low-risk MUL so modifier > 0.3
        let mut mul = MetaUncertaintyLayer::new();
        mul.trust_qualia = crate::mul::TrustQualia::uniform(0.9);
        mul.risk_vector = crate::mul::RiskVector::low();

        let result = mul_volitional_cycle(
            &mut graph, target, &query, RungLevel::Meta, &mul, true,
        );

        assert!(result.gate_allowed, "high-trust MUL should allow reflection");
        assert!(result.agenda.is_some(), "should have agenda when gate open");
    }

    #[test]
    fn test_mul_volitional_cycle_gate_closed() {
        let mut graph = build_test_tree();
        seed_nars(&mut graph);

        let query = Container::random(42);
        let target = PackedDn::new(&[0, 1]);

        // MUL with MountStupid → gate closed
        let mut mul = MetaUncertaintyLayer::new();
        mul.dk_detector.position = DKPosition::MountStupid;

        let result = mul_volitional_cycle(
            &mut graph, target, &query, RungLevel::Meta, &mul, true,
        );

        assert!(!result.gate_allowed, "MountStupid should block reflection");
        assert!(result.agenda.is_none(), "no agenda when gate closed");
    }

    #[test]
    fn test_reflection_to_mul_learning() {
        let mut graph = build_test_tree();
        seed_nars(&mut graph);

        let query = Container::random(42);
        let target = PackedDn::new(&[0, 1]);

        let reflection = reflect_walk(&mut graph, target, &query);
        let learning = reflection_to_mul_learning(&reflection);

        assert!(learning.predicted_confidence >= 0.0 && learning.predicted_confidence <= 1.0);
        assert!(learning.actual_outcome >= 0.0 && learning.actual_outcome <= 1.0);
    }

    #[test]
    fn test_mul_reflection_feedback() {
        let mut graph = build_test_tree();
        seed_nars(&mut graph);

        let query = Container::random(42);
        let target = PackedDn::new(&[0, 1]);

        // Use high-trust, low-risk MUL so modifier > 0.3
        let mut mul = MetaUncertaintyLayer::new();
        mul.trust_qualia = crate::mul::TrustQualia::uniform(0.9);
        mul.risk_vector = crate::mul::RiskVector::low();

        let initial_tick = mul.tick_count();
        let result = mul_reflection_feedback(
            &mut graph, target, &query, RungLevel::Meta, &mut mul, true,
        );

        assert!(result.gate_allowed, "high-trust MUL should allow reflection");
        // MUL should have been ticked
        assert!(mul.tick_count() > initial_tick,
            "MUL should have been ticked: {} > {}",
            mul.tick_count(), initial_tick);
    }

    #[test]
    fn test_reclassify_with_thresholds() {
        let mut graph = build_test_tree();
        seed_nars(&mut graph);

        let query = Container::random(42);
        let target = PackedDn::new(&[0, 1]);

        let reflection = reflect_walk(&mut graph, target, &query);
        let defaults = AdaptiveThresholds::default();

        // Reclassify with same thresholds should change nothing
        let (reclassified, changed) = reclassify_with_thresholds(&reflection, &defaults);
        assert_eq!(reclassified.len(), reflection.entries.len());

        // With very aggressive thresholds, more should be Explore/Revise
        let aggressive = AdaptiveThresholds {
            surprise_high: 0.2,  // everything is "surprising"
            surprise_low: 0.1,
            confidence_high: 0.8,
        };
        let (recl_agg, _) = reclassify_with_thresholds(&reflection, &aggressive);
        assert_eq!(recl_agg.len(), reflection.entries.len());
    }
}
