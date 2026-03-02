//! Causal Trajectory Hydration: Bridge BNN Instrumentation to Ladybug NARS
//!
//! Sits between two already-built layers:
//!
//! 1. **Foveal layer** — resonator's converged SPO factorizations with high-confidence
//!    NARS truth values (99.8th percentile)
//! 2. **Context layer** — typed halo from cross-plane vote: 6 partial binding types
//!    extracted from the σ₂–σ₃ noise floor for FREE via bitwise AND/NOT
//!
//! The engine runs NARS on the delta between fovea and context across resonator
//! iterations to produce causality trajectories that grow the DN tree (Sigma Graph).
//!
//! # Architecture
//!
//! ```text
//! SpoDistanceResult (spo_harvest)
//!   → TrajectoryHydrator.feed()
//!     → CausalTrajectory (rustynum-bnn)
//!       → record_iteration() [RIF diff + EWM correction + BPReLU arrow + halo transitions]
//!     → finalize()
//!       → CausalChain [growth path]
//!       → SigmaEdge [DN tree growth instructions]
//!       → NarsCausalStatement [typed truth values]
//!   → gestalt_from_trajectory()
//!     → GestaltState [Crystallizing | Contested | Dissolving | Epiphany]
//!   → harvest_sigma_edges()
//!     → Vec<HydratedEdge> [SigmaEdge + canonical TruthValue]
//! ```
//!
//! # Cost
//!
//! ~15μs per resonator iteration on top of existing ~50μs. The 30% overhead buys
//! full causal trajectory recording, NARS truth grounded in convergence dynamics,
//! DN mutation guidance, and warm-start capability.

use rustynum_bnn::causal_trajectory::{
    CausalArrow, CausalChain, CausalDirection, CausalLink, CausalRelation, CausalSaliency,
    CausalTrajectory, DominantPlane, EwmCorrection, EwmTier, HaloTransition,
    NarsCausalStatement, NarsTruth, ResonatorSnapshot, RifDiff, SigmaEdge, SigmaNode,
};
use rustynum_bnn::{GrowthPath, HaloType, InferenceMode, MutationOp};
use rustynum_core::{CollapseGate, SigmaGate, SignificanceLevel};

use crate::nars::TruthValue;
use super::gestalt::GestaltState;
use super::spo_harvest::{Plane, SpoDistanceResult, TypedHalo};
use super::shift_detector::SpoShiftDetector;

// =============================================================================
// HYDRATED EDGE — SigmaEdge enriched with canonical TruthValue
// =============================================================================

/// A Sigma Graph edge enriched with ladybug's canonical `TruthValue`.
///
/// Wraps rustynum-bnn's `SigmaEdge` (which uses `NarsTruth`) and adds
/// the canonical `crate::nars::TruthValue` for use in ladybug's NARS system.
#[derive(Clone, Debug)]
pub struct HydratedEdge {
    /// The underlying BNN sigma edge (source, target, relation, BNN truth, iter).
    pub edge: SigmaEdge,
    /// Canonical ladybug truth value — use this for NARS inference.
    pub truth: TruthValue,
    /// Growth path that produced this edge (how the factorization converged).
    pub growth_path: Option<GrowthPath>,
    /// Gestalt state at the time of edge creation.
    pub gestalt: GestaltState,
}

/// A causal statement enriched with canonical TruthValue.
#[derive(Clone, Debug)]
pub struct HydratedStatement {
    /// The underlying BNN causal statement.
    pub statement: NarsCausalStatement,
    /// Canonical ladybug truth value.
    pub truth: TruthValue,
}

// =============================================================================
// TRAJECTORY HYDRATOR — main integration struct
// =============================================================================

/// Bridges rustynum-bnn's `CausalTrajectory` with ladybug's NARS system.
///
/// Wraps the BNN trajectory with SPO-harvest-specific logic:
/// - Feeds `SpoDistanceResult` into the trajectory pipeline
/// - Converts BNN `NarsTruth` to canonical `crate::nars::TruthValue`
/// - Computes `GestaltState` from trajectory saliency
/// - Produces `HydratedEdge` with growth path classification
///
/// # Usage
///
/// ```ignore
/// let mut hydrator = TrajectoryHydrator::new(sigma_gate);
///
/// // Feed resonator snapshots (typically 5-20 per factorization)
/// hydrator.record_snapshot(snapshot);
///
/// // After convergence, finalize and extract results
/// let result = hydrator.finalize();
/// let edges = result.hydrated_edges();
/// let gestalt = result.gestalt();
/// ```
pub struct TrajectoryHydrator {
    /// Inner BNN causal trajectory (owns snapshots, diffs, arrows, etc.).
    trajectory: CausalTrajectory,
    /// σ-gate for significance classification.
    gate: SigmaGate,
    /// SPO shift detector for distributional tracking.
    shift_detector: SpoShiftDetector,
    /// Accumulated harvest truth (revised across all searches in this trajectory).
    accumulated_truth: TruthValue,
    /// Count of distance results fed into this hydrator.
    feed_count: u64,
    /// Per-plane similarity EMA for tilt detection.
    plane_ema: [f32; 3],
}

impl TrajectoryHydrator {
    /// Create a new trajectory hydrator with the given σ-gate.
    pub fn new(gate: SigmaGate) -> Self {
        Self {
            trajectory: CausalTrajectory::with_sigma_gate(gate),
            gate,
            shift_detector: SpoShiftDetector::new(gate),
            accumulated_truth: TruthValue::unknown(),
            feed_count: 0,
            plane_ema: [0.0; 3],
        }
    }

    /// Record a resonator snapshot into the trajectory.
    ///
    /// This triggers all BNN instrumentation:
    /// - RIF diff (XOR with iter-2)
    /// - EWM correction (per-word L1 delta)
    /// - BPReLU causal arrow (forward/backward asymmetry)
    /// - Halo transition detection (promotion/demotion)
    /// - NARS statement generation from transitions
    pub fn record_snapshot(&mut self, snapshot: ResonatorSnapshot) {
        self.trajectory.record_iteration(snapshot);
    }

    /// Feed an SPO distance result into the shift detector and accumulate truth.
    ///
    /// Call this for every distance computation during a search batch.
    /// The shift detector tracks distributional migration; the truth value
    /// accumulates via NARS revision.
    pub fn feed_distance(&mut self, result: &SpoDistanceResult) {
        // Track distributional shift across σ-stripes.
        self.shift_detector.record(result);

        // Accumulate truth via NARS revision.
        let harvest_truth = harvest_to_truth(result);
        self.accumulated_truth = self.accumulated_truth.revision(&harvest_truth);

        // Update per-plane EMA (α = 0.1).
        const ALPHA: f32 = 0.1;
        self.plane_ema[0] += ALPHA * (result.s_p_similarity - self.plane_ema[0]);
        self.plane_ema[1] += ALPHA * (result.p_o_similarity - self.plane_ema[1]);
        self.plane_ema[2] += ALPHA * (result.s_o_similarity - self.plane_ema[2]);

        self.feed_count += 1;
    }

    /// Advance the shift detector's time window.
    ///
    /// Call this after processing a batch of searches.
    pub fn advance_window(&mut self) {
        self.shift_detector.advance_window();
    }

    /// Finalize the trajectory and produce the hydrated result.
    ///
    /// This:
    /// 1. Finalizes the BNN trajectory (causal chain, sigma edges)
    /// 2. Computes gestalt state from trajectory saliency
    /// 3. Enriches all edges with canonical TruthValue and growth path
    /// 4. Applies shift detector bias to gate decision
    pub fn finalize(mut self) -> HydratedTrajectory {
        // Finalize the BNN trajectory.
        self.trajectory.finalize();

        // Compute gestalt state from saliency.
        let gestalt = self.compute_gestalt();

        // Compute growth path from causal chain.
        let chain = CausalChain::from_rif_diffs(&self.trajectory.rif_diffs);
        let growth_path = infer_growth_path_from_chain(&chain);

        // Gate decision: combine trajectory gate with shift detector bias.
        let trajectory_gate = self.trajectory.gate_decision();
        let shift_bias = self.shift_detector.gate_bias();
        let final_gate = combine_gates(trajectory_gate, shift_bias);

        // Build hydrated edges.
        let hydrated_edges: Vec<HydratedEdge> = self
            .trajectory
            .sigma_edges
            .iter()
            .map(|edge| HydratedEdge {
                edge: edge.clone(),
                truth: nars_to_truth(&edge.truth),
                growth_path: Some(growth_path),
                gestalt: gestalt.clone(),
            })
            .collect();

        // Build hydrated statements.
        let hydrated_statements: Vec<HydratedStatement> = self
            .trajectory
            .nars_statements
            .iter()
            .map(|stmt| HydratedStatement {
                statement: stmt.clone(),
                truth: nars_to_truth(&stmt.truth),
            })
            .collect();

        HydratedTrajectory {
            trajectory: self.trajectory,
            gestalt,
            growth_path,
            gate: final_gate,
            accumulated_truth: self.accumulated_truth,
            hydrated_edges,
            hydrated_statements,
            chain,
            plane_ema: self.plane_ema,
            feed_count: self.feed_count,
        }
    }

    /// Compute gestalt state from the trajectory's EWM corrections.
    fn compute_gestalt(&self) -> GestaltState {
        if self.trajectory.ewm_corrections.len() < 2 {
            // Not enough data — default to Crystallizing.
            return GestaltState::Crystallizing;
        }

        let saliency = CausalSaliency::from_ewm_window(&self.trajectory.ewm_corrections);
        GestaltState::from_saliency_counts(
            &saliency.crystallizing_count,
            &saliency.dissolving_count,
            &saliency.contested_count,
        )
    }

    /// Number of recorded resonator iterations.
    pub fn iteration_count(&self) -> usize {
        self.trajectory.len()
    }

    /// Number of distance results fed.
    pub fn feed_count(&self) -> u64 {
        self.feed_count
    }

    /// Current accumulated truth value.
    pub fn accumulated_truth(&self) -> &TruthValue {
        &self.accumulated_truth
    }

    /// Current shift detector bias.
    pub fn shift_bias(&self) -> Option<CollapseGate> {
        self.shift_detector.gate_bias()
    }
}

// =============================================================================
// HYDRATED TRAJECTORY — finalized result
// =============================================================================

/// The finalized output of trajectory hydration.
///
/// Contains the full BNN trajectory plus ladybug-specific enrichments:
/// canonical TruthValues, GestaltState, growth path, and gate decision.
pub struct HydratedTrajectory {
    /// The finalized BNN trajectory (snapshots, diffs, arrows, statements, edges).
    pub trajectory: CausalTrajectory,
    /// Gestalt state derived from EWM saliency.
    pub gestalt: GestaltState,
    /// Growth path classification (which planes converged in what order).
    pub growth_path: GrowthPath,
    /// Final gate decision (combining trajectory + shift detector).
    pub gate: CollapseGate,
    /// Accumulated truth from all fed distance results.
    pub accumulated_truth: TruthValue,
    /// Sigma edges enriched with canonical TruthValue.
    pub hydrated_edges: Vec<HydratedEdge>,
    /// NARS statements enriched with canonical TruthValue.
    pub hydrated_statements: Vec<HydratedStatement>,
    /// Causal chain extracted from RIF diffs.
    pub chain: CausalChain,
    /// Per-plane similarity EMA at finalization.
    pub plane_ema: [f32; 3],
    /// Total distance results fed.
    pub feed_count: u64,
}

impl HydratedTrajectory {
    /// Root cause plane (first to converge in the causal chain).
    pub fn root_cause(&self) -> Option<DominantPlane> {
        self.chain.root_cause()
    }

    /// Causal chain depth (number of cause→effect links).
    pub fn chain_depth(&self) -> usize {
        self.chain.depth()
    }

    /// Number of NARS causal statements produced.
    pub fn statement_count(&self) -> usize {
        self.hydrated_statements.len()
    }

    /// Number of Sigma Graph edges to create.
    pub fn edge_count(&self) -> usize {
        self.hydrated_edges.len()
    }

    /// Extract edges filtered by causal relation type.
    pub fn edges_by_relation(&self, relation: CausalRelation) -> Vec<&HydratedEdge> {
        self.hydrated_edges
            .iter()
            .filter(|e| e.edge.relation == relation)
            .collect()
    }

    /// DN mutation guidance: which plane is weakest (most contested)?
    ///
    /// Uses the EWM saliency to identify the plane with the most contested
    /// dimensions, then maps to a conservative `MutationOp`.
    pub fn mutation_target(&self) -> MutationOp {
        if self.trajectory.ewm_corrections.len() < 2 {
            return MutationOp::MutateO; // Default: mutate object (safest)
        }

        let saliency = CausalSaliency::from_ewm_window(&self.trajectory.ewm_corrections);
        let s = saliency.contested_count[0];
        let p = saliency.contested_count[1];
        let o = saliency.contested_count[2];

        if s >= p && s >= o {
            MutationOp::MutateS
        } else if p >= o {
            MutationOp::MutateP
        } else {
            MutationOp::MutateO
        }
    }

    /// Radical mutation: returns a 2-slot mutation op based on growth path.
    ///
    /// For Catalyst mode (DN speciation): mutate the two planes that did NOT
    /// converge first, preserving the root cause plane.
    pub fn radical_mutation(&self) -> MutationOp {
        match self.root_cause() {
            Some(DominantPlane::S) => MutationOp::MutatePO,
            Some(DominantPlane::P) => MutationOp::MutateSO,
            Some(DominantPlane::O) => MutationOp::MutateSP,
            None => MutationOp::MutatePO, // Default: preserve subject
        }
    }

    /// Per-plane tilt report from EMA similarities.
    ///
    /// Returns `(s_tilt, p_tilt, o_tilt)` where negative = underperforming,
    /// positive = overperforming relative to mean.
    pub fn plane_tilt(&self) -> (f32, f32, f32) {
        let mean = (self.plane_ema[0] + self.plane_ema[1] + self.plane_ema[2]) / 3.0;
        (
            self.plane_ema[0] - mean,
            self.plane_ema[1] - mean,
            self.plane_ema[2] - mean,
        )
    }

    /// Convergence speed: iterations used / typical max (20).
    ///
    /// Fast convergence (< 0.3) → high confidence.
    /// Slow convergence (> 0.7) → low confidence, hard problem.
    pub fn convergence_speed(&self) -> f32 {
        const MAX_ITER: f32 = 20.0;
        (self.trajectory.len() as f32 / MAX_ITER).min(1.0)
    }

    /// Overall confidence derived from convergence speed + gestalt.
    pub fn overall_confidence(&self) -> f32 {
        let speed_confidence = 1.0 - self.convergence_speed();
        let gestalt_multiplier = match &self.gestalt {
            GestaltState::Crystallizing => 1.0,
            GestaltState::Epiphany => 0.9,
            GestaltState::Contested => 0.5,
            GestaltState::Dissolving => 0.3,
        };
        speed_confidence * gestalt_multiplier
    }
}

// =============================================================================
// CONVERSION FUNCTIONS
// =============================================================================

/// Convert BNN `NarsTruth` to canonical ladybug `TruthValue`.
///
/// Prefer using `TruthValue::from(nars)` or `nars.into()` directly —
/// the `From` impls in `crate::nars::truth` are the canonical bridge.
pub fn nars_to_truth(nars: &NarsTruth) -> TruthValue {
    TruthValue::from(nars)
}

/// Convert canonical ladybug `TruthValue` to BNN `NarsTruth`.
///
/// Prefer using `NarsTruth::from(truth)` or `truth.into()` directly.
pub fn truth_to_nars(truth: &TruthValue) -> NarsTruth {
    NarsTruth::from(truth)
}

/// Extract canonical `TruthValue` from an SPO distance harvest result.
///
/// - **Frequency**: core_count / total (fraction of full SPO matches)
/// - **Confidence**: 1 - normalized_entropy (distribution concentration)
fn harvest_to_truth(result: &SpoDistanceResult) -> TruthValue {
    let total = result.halo.total() as f32;
    if total == 0.0 {
        return TruthValue::unknown();
    }

    let frequency = result.halo.core_count as f32 / total;
    let entropy = result.halo.entropy();
    let max_entropy = (7.0_f32).ln(); // ln(7) for 7 halo types
    let confidence = if max_entropy > 0.0 {
        (1.0 - entropy / max_entropy).clamp(0.0, 1.0)
    } else {
        0.0
    };

    TruthValue::new(frequency.clamp(0.0, 1.0), confidence)
}

// =============================================================================
// GROWTH PATH INFERENCE
// =============================================================================

/// Infer the growth path from a causal chain (RIF diff analysis).
///
/// The causal chain records which plane stabilized first (root cause).
/// Combined with the second link, this determines which of the 6 growth
/// paths through the B₃ lattice the factorization took.
fn infer_growth_path_from_chain(chain: &CausalChain) -> GrowthPath {
    if chain.links.is_empty() {
        return GrowthPath::SubjectFirst; // Default: subject-first
    }

    let first = &chain.links[0];

    // The root cause plane stabilized first.
    // The effect plane tells us which pair formed.
    match (first.cause_plane, first.effect_plane) {
        // S stabilized first
        (DominantPlane::S, DominantPlane::P) => GrowthPath::SubjectFirst,   // S→SP→Core
        (DominantPlane::S, DominantPlane::O) => GrowthPath::SubjectObject,  // S→SO→Core

        // P stabilized first
        (DominantPlane::P, DominantPlane::S) => GrowthPath::ActionFirst,    // P→SP→Core
        (DominantPlane::P, DominantPlane::O) => GrowthPath::ActionObject,   // P→PO→Core

        // O stabilized first
        (DominantPlane::O, DominantPlane::P) => GrowthPath::ObjectAction,   // O→PO→Core
        (DominantPlane::O, DominantPlane::S) => GrowthPath::ObjectSubject,  // O→SO→Core

        // Same plane cause and effect shouldn't happen, but handle gracefully
        _ => GrowthPath::SubjectFirst,
    }
}

// =============================================================================
// GATE COMBINATION
// =============================================================================

/// Combine trajectory gate decision with shift detector bias.
///
/// The shift detector provides a bias based on distributional migration:
/// - `TowardNoise` → bias HOLD (ground is moving)
/// - `TowardFoveal` → bias FLOW (world is clarifying)
/// - `Bimodal` → bias HOLD (world is splitting)
/// - `Stable` → no bias
///
/// The combination follows a conservative merge: the more restrictive gate wins,
/// except when shift detector says FLOW and trajectory says HOLD — in that case,
/// the clarifying shift softens the hold.
fn combine_gates(trajectory: CollapseGate, shift_bias: Option<CollapseGate>) -> CollapseGate {
    match (trajectory, shift_bias) {
        // Block always wins.
        (CollapseGate::Block, _) | (_, Some(CollapseGate::Block)) => CollapseGate::Block,

        // Both agree → use that.
        (gate, None) => gate,
        (gate, Some(bias)) if gate == bias => gate,

        // Shift says Flow but trajectory says Hold → soften to Flow.
        // Rationale: distributional shift toward foveal means world is
        // clarifying, so the hold may be overly conservative.
        (CollapseGate::Hold, Some(CollapseGate::Flow)) => CollapseGate::Flow,

        // Shift says Hold but trajectory says Flow → conservative wins.
        (CollapseGate::Flow, Some(CollapseGate::Hold)) => CollapseGate::Hold,

        // Default: take the more conservative option.
        (a, Some(_)) => a,
    }
}

// =============================================================================
// DOMINANT PLANE ↔ SPO PLANE CONVERSIONS
// =============================================================================

/// Convert `DominantPlane` (BNN type) to `Plane` (SPO harvest type).
pub fn dominant_to_plane(dp: DominantPlane) -> Plane {
    match dp {
        DominantPlane::S => Plane::X,
        DominantPlane::P => Plane::Y,
        DominantPlane::O => Plane::Z,
    }
}

/// Convert `Plane` (SPO harvest type) to `DominantPlane` (BNN type).
pub fn plane_to_dominant(p: Plane) -> DominantPlane {
    match p {
        Plane::X => DominantPlane::S,
        Plane::Y => DominantPlane::P,
        Plane::Z => DominantPlane::O,
    }
}

// =============================================================================
// CAUSAL DIRECTION HELPERS
// =============================================================================

/// Check if a causal direction is forward.
pub fn is_forward(dir: &CausalDirection) -> bool {
    matches!(dir, CausalDirection::Forward(_))
}

/// Check if a causal direction is backward.
pub fn is_backward(dir: &CausalDirection) -> bool {
    matches!(dir, CausalDirection::Backward(_))
}

/// Extract the magnitude from a causal direction.
pub fn direction_magnitude(dir: &CausalDirection) -> f32 {
    match dir {
        CausalDirection::Forward(m) | CausalDirection::Backward(m) | CausalDirection::Contested(m) => *m,
        CausalDirection::Symmetric => 0.0,
    }
}

/// Map a causal arrow's per-plane directions to a NARS inference rule.
///
/// The BPReLU forward/backward asymmetry maps to NARS inference types:
/// - Forward → Deduction (commitment drove context)
/// - Backward → Abduction (context overrode commitment)
/// - Symmetric → Induction (parallel discovery)
/// - Contested → Comparison (competing hypotheses)
pub fn arrow_to_inference(arrow: &CausalArrow) -> InferenceMode {
    match arrow.overall {
        CausalDirection::Forward(_) => InferenceMode::Forward,
        CausalDirection::Backward(_) => InferenceMode::Backward,
        CausalDirection::Symmetric => InferenceMode::Abduction,
        CausalDirection::Contested(_) => InferenceMode::Analogy,
    }
}

/// Apply NARS inference rule to two truth values based on causal direction.
///
/// This is the key bridge: BPReLU asymmetry determines WHICH NARS inference
/// rule combines evidence from consecutive resonator iterations.
pub fn apply_causal_inference(
    prev_truth: &TruthValue,
    curr_truth: &TruthValue,
    direction: &CausalDirection,
) -> TruthValue {
    match direction {
        CausalDirection::Forward(_) => prev_truth.deduction(curr_truth),
        CausalDirection::Backward(_) => prev_truth.abduction(curr_truth),
        CausalDirection::Symmetric => prev_truth.induction(curr_truth),
        CausalDirection::Contested(_) => prev_truth.comparison(curr_truth),
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nars_truth_conversion_roundtrip() {
        let original = TruthValue::new(0.8, 0.9);
        let nars = truth_to_nars(&original);
        let back = nars_to_truth(&nars);

        assert!((back.frequency - original.frequency).abs() < 0.001);
        assert!((back.confidence - original.confidence).abs() < 0.001);
    }

    #[test]
    fn test_nars_truth_clamping() {
        // NarsTruth doesn't clamp internally, but our conversion does.
        let nars = NarsTruth::new(1.5, -0.1);
        let truth = nars_to_truth(&nars);
        assert!(truth.frequency <= 1.0);
        assert!(truth.confidence >= 0.0);
    }

    #[test]
    fn test_growth_path_from_chain() {
        let chain = CausalChain {
            links: vec![CausalLink {
                cause_plane: DominantPlane::S,
                effect_plane: DominantPlane::P,
                confidence: 0.8,
                from_iter: 0,
                to_iter: 3,
            }],
        };

        let path = infer_growth_path_from_chain(&chain);
        assert_eq!(path, GrowthPath::SubjectFirst);
    }

    #[test]
    fn test_growth_path_object_first() {
        let chain = CausalChain {
            links: vec![CausalLink {
                cause_plane: DominantPlane::O,
                effect_plane: DominantPlane::P,
                confidence: 0.7,
                from_iter: 1,
                to_iter: 5,
            }],
        };

        let path = infer_growth_path_from_chain(&chain);
        assert_eq!(path, GrowthPath::ObjectAction);
    }

    #[test]
    fn test_growth_path_empty_chain() {
        let chain = CausalChain::new();
        let path = infer_growth_path_from_chain(&chain);
        assert_eq!(path, GrowthPath::SubjectFirst);
    }

    #[test]
    fn test_combine_gates_block_wins() {
        assert_eq!(
            combine_gates(CollapseGate::Block, Some(CollapseGate::Flow)),
            CollapseGate::Block
        );
        assert_eq!(
            combine_gates(CollapseGate::Flow, Some(CollapseGate::Block)),
            CollapseGate::Block
        );
    }

    #[test]
    fn test_combine_gates_shift_softens_hold() {
        assert_eq!(
            combine_gates(CollapseGate::Hold, Some(CollapseGate::Flow)),
            CollapseGate::Flow
        );
    }

    #[test]
    fn test_combine_gates_conservative_hold() {
        assert_eq!(
            combine_gates(CollapseGate::Flow, Some(CollapseGate::Hold)),
            CollapseGate::Hold
        );
    }

    #[test]
    fn test_combine_gates_no_bias() {
        assert_eq!(combine_gates(CollapseGate::Flow, None), CollapseGate::Flow);
        assert_eq!(combine_gates(CollapseGate::Hold, None), CollapseGate::Hold);
    }

    #[test]
    fn test_plane_conversions() {
        assert_eq!(dominant_to_plane(DominantPlane::S), Plane::X);
        assert_eq!(dominant_to_plane(DominantPlane::P), Plane::Y);
        assert_eq!(dominant_to_plane(DominantPlane::O), Plane::Z);

        assert_eq!(plane_to_dominant(Plane::X), DominantPlane::S);
        assert_eq!(plane_to_dominant(Plane::Y), DominantPlane::P);
        assert_eq!(plane_to_dominant(Plane::Z), DominantPlane::O);
    }

    #[test]
    fn test_direction_magnitude() {
        assert!((direction_magnitude(&CausalDirection::Forward(0.8)) - 0.8).abs() < 0.001);
        assert!((direction_magnitude(&CausalDirection::Backward(0.5)) - 0.5).abs() < 0.001);
        assert!((direction_magnitude(&CausalDirection::Symmetric) - 0.0).abs() < 0.001);
        assert!((direction_magnitude(&CausalDirection::Contested(0.3)) - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_causal_inference_forward_deduction() {
        let prev = TruthValue::new(0.9, 0.9);
        let curr = TruthValue::new(0.8, 0.8);

        let result = apply_causal_inference(&prev, &curr, &CausalDirection::Forward(0.7));

        // Deduction: f = f1*f2, c = f1*f2*c1*c2
        assert!(result.frequency < prev.frequency, "Deduction should reduce frequency");
        assert!(result.confidence < prev.confidence, "Deduction should reduce confidence");
    }

    #[test]
    fn test_causal_inference_backward_abduction() {
        let prev = TruthValue::new(0.9, 0.9);
        let curr = TruthValue::new(0.8, 0.8);

        let result = apply_causal_inference(&prev, &curr, &CausalDirection::Backward(0.6));

        // Abduction: f = f1, c = f2*c1*c2
        assert!((result.frequency - prev.frequency).abs() < 0.01);
    }

    #[test]
    fn test_arrow_to_inference_mapping() {
        let arrow_fwd = CausalArrow {
            iter: 1,
            s_direction: CausalDirection::Forward(0.5),
            p_direction: CausalDirection::Forward(0.5),
            o_direction: CausalDirection::Forward(0.5),
            forward_magnitude: [0.5; 3],
            backward_magnitude: [0.2; 3],
            overall: CausalDirection::Forward(0.5),
        };
        assert_eq!(arrow_to_inference(&arrow_fwd), InferenceMode::Forward);

        let arrow_bwd = CausalArrow {
            iter: 2,
            s_direction: CausalDirection::Backward(0.6),
            p_direction: CausalDirection::Backward(0.6),
            o_direction: CausalDirection::Backward(0.6),
            forward_magnitude: [0.2; 3],
            backward_magnitude: [0.6; 3],
            overall: CausalDirection::Backward(0.6),
        };
        assert_eq!(arrow_to_inference(&arrow_bwd), InferenceMode::Backward);
    }

    #[test]
    fn test_hydrator_creation() {
        let gate = SigmaGate::new(16_384);
        let hydrator = TrajectoryHydrator::new(gate);

        assert_eq!(hydrator.iteration_count(), 0);
        assert_eq!(hydrator.feed_count(), 0);
        assert!(hydrator.shift_bias().is_none());
    }

    #[test]
    fn test_harvest_to_truth_unknown() {
        // Empty halo should produce unknown truth.
        let halo = TypedHalo {
            core_count: 0,
            sp_count: 0,
            so_count: 0,
            po_count: 0,
            s_count: 0,
            p_count: 0,
            o_count: 0,
        };

        let result = SpoDistanceResult {
            similarity: 0.0,
            s_p_similarity: 0.0,
            p_o_similarity: 0.0,
            s_o_similarity: 0.0,
            halo,
            x_dist: 8192,
            y_dist: 8192,
            z_dist: 8192,
            x_sigma: SignificanceLevel::Noise,
            y_sigma: SignificanceLevel::Noise,
            z_sigma: SignificanceLevel::Noise,
        };

        let truth = harvest_to_truth(&result);
        assert!((truth.frequency - 0.5).abs() < 0.01);
        assert!((truth.confidence - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_harvest_to_truth_full_core() {
        // All core should produce high frequency, high confidence.
        let halo = TypedHalo {
            core_count: 100,
            sp_count: 0,
            so_count: 0,
            po_count: 0,
            s_count: 0,
            p_count: 0,
            o_count: 0,
        };

        let result = SpoDistanceResult {
            similarity: 1.0,
            s_p_similarity: 1.0,
            p_o_similarity: 1.0,
            s_o_similarity: 1.0,
            halo,
            x_dist: 0,
            y_dist: 0,
            z_dist: 0,
            x_sigma: SignificanceLevel::Discovery,
            y_sigma: SignificanceLevel::Discovery,
            z_sigma: SignificanceLevel::Discovery,
        };

        let truth = harvest_to_truth(&result);
        assert!(truth.frequency > 0.9, "Full core should have high frequency");
        assert!(truth.confidence > 0.9, "Single-type distribution should have high confidence");
    }
}
