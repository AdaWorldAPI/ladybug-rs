//! Reflection — NARS Introspection via Felt Walk + Free Energy Semiring
//!
//! Reflection is the system looking at itself. It ties together:
//!
//! 1. **Felt traversal** walks the DN tree computing surprise (free energy)
//! 2. **NARS truth values** (W4-W7 in Container 0) encode belief state
//! 3. **SpineCache** provides borrow/mut semantics for simultaneous read+update
//! 4. **Delta encoding** tracks hydration steps as a reversible Markov chain
//! 5. **Cross-hydration** projects nodes into sibling contexts
//!
//! ## The Core Insight
//!
//! ```text
//! surprise (high) + confidence (high) = REVISION needed
//!   → belief doesn't match context, update via TruthValue::revision()
//!
//! surprise (low)  + confidence (low)  = CONFIRMATION
//!   → context matches, increase confidence via positive evidence
//!
//! surprise (high) + confidence (low)  = EXPLORATION
//!   → novel territory, begin hydration from siblings
//!
//! surprise (low)  + confidence (high) = STABLE BELIEF
//!   → well-predicted, well-grounded — no action needed
//! ```
//!
//! ## Reflection as Borrow+Mut
//!
//! The spine IS the structural prediction (borrowed reference from a joined
//! blackboard). Children ARE the mutable beliefs. Reflection = reading the
//! spine while updating children's NARS truth values. The dirty flag handles
//! invalidation — no lock needed because XOR is commutative and associative.
//!
//! ## Hydration as Reversible Markov Chain
//!
//! Adjacent containers under the same spine inherit semantic richness through
//! bind/unbind chains. Each step is an XOR delta (bind = forward, unbind =
//! reverse). The popcount of each delta IS the energy of the transition.
//! `chain_encode()` from delta.rs stores the chain compactly.
//!
//! ## Free Energy Semiring
//!
//! Surprise plugs into the DnSemiring trait for graph-wide propagation via
//! existing `container_mxv()`. MinSurprise finds the path of least resistance;
//! MaxSurprise finds where active inference is most needed.

use crate::container::{Container, CogRecord, CONTAINER_BITS};
use crate::container::graph::ContainerGraph;
use crate::container::adjacency::PackedDn;
use crate::container::delta::{chain_encode, delta_encode};
use crate::container::traversal::DnSemiring;
use ladybug_contract::nars::TruthValue;
use super::felt_traversal::{FeltPath, felt_walk};

// =============================================================================
// VERB CONSTANT
// =============================================================================

/// Verb ID for "reflection" edges — marks nodes that have been reflected on.
pub const VERB_REFLECTION: u8 = 0xFB;

// =============================================================================
// NARS BRIDGE — Read/write TruthValue from Container 0 metadata
// =============================================================================

/// Read the NARS truth value from a CogRecord's Container 0 metadata.
///
/// Extracts frequency and confidence from W4 (next to DN address in W0).
/// Returns `TruthValue::unknown()` if both are zero (uninitialized).
#[inline]
pub fn read_truth(record: &CogRecord) -> TruthValue {
    let meta = record.meta_view();
    let freq = meta.nars_frequency();
    let conf = meta.nars_confidence();
    if freq == 0.0 && conf == 0.0 {
        return TruthValue::unknown();
    }
    TruthValue::new(freq.clamp(0.0, 1.0), conf.clamp(0.0, 1.0))
}

/// Write a NARS truth value into a CogRecord's Container 0 metadata.
///
/// Updates W4 (frequency, confidence) and W5 (positive, negative evidence).
pub fn write_truth(record: &mut CogRecord, tv: &TruthValue) {
    let mut meta = record.meta_view_mut();
    meta.set_nars_frequency(tv.frequency);
    meta.set_nars_confidence(tv.confidence);
    let (pos, neg) = tv.to_evidence();
    meta.set_nars_positive_evidence(pos);
    meta.set_nars_negative_evidence(neg);
}

// =============================================================================
// REFLECTION OUTCOME — 2×2 classification of (surprise, confidence)
// =============================================================================

/// The outcome of reflecting on a single node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReflectionOutcome {
    /// High surprise + high confidence: belief contradicts context. Revise.
    Revise,
    /// Low surprise + low confidence: context confirms weak belief. Boost.
    Confirm,
    /// High surprise + low confidence: novel territory. Hydrate from siblings.
    Explore,
    /// Low surprise + high confidence: well-predicted. No action.
    Stable,
}

/// Threshold for "high" surprise (matches FeltPath::is_surprising()).
pub const SURPRISE_HIGH: f32 = 0.55;
/// Threshold for "low" surprise (matches FeltPath::is_predicted()).
pub const SURPRISE_LOW: f32 = 0.45;
/// Threshold for "high" confidence.
pub const CONFIDENCE_HIGH: f32 = 0.5;

/// Classify (surprise, confidence) into a ReflectionOutcome.
fn classify(surprise: f32, confidence: f32) -> ReflectionOutcome {
    let high_surprise = surprise > SURPRISE_HIGH;
    let high_confidence = confidence > CONFIDENCE_HIGH;
    match (high_surprise, high_confidence) {
        (true, true) => ReflectionOutcome::Revise,
        (false, false) => ReflectionOutcome::Confirm,
        (true, false) => ReflectionOutcome::Explore,
        (false, true) => ReflectionOutcome::Stable,
    }
}

// =============================================================================
// REFLECTION ENTRY + RESULT
// =============================================================================

/// Result of reflecting on a single node during the felt walk.
#[derive(Debug, Clone)]
pub struct ReflectionEntry {
    /// The DN of the node that was reflected on.
    pub dn: PackedDn,
    /// Surprise (free energy) at this node.
    pub surprise: f32,
    /// NARS truth value BEFORE reflection.
    pub truth_before: TruthValue,
    /// NARS truth value AFTER reflection.
    pub truth_after: TruthValue,
    /// What action was taken.
    pub outcome: ReflectionOutcome,
    /// Depth in the tree.
    pub depth: u8,
}

/// The full result of a reflection walk.
#[derive(Debug, Clone)]
pub struct ReflectionResult {
    /// Per-node reflection entries along the path.
    pub entries: Vec<ReflectionEntry>,
    /// The underlying felt path (surprise + sibling data).
    pub felt_path: FeltPath,
    /// Nodes flagged for hydration (Explore outcome).
    pub hydration_candidates: Vec<PackedDn>,
}

impl ReflectionResult {
    /// How many nodes were revised.
    pub fn revision_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.outcome == ReflectionOutcome::Revise)
            .count()
    }

    /// How many nodes were confirmed.
    pub fn confirmation_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.outcome == ReflectionOutcome::Confirm)
            .count()
    }

    /// Mean confidence change across all reflected nodes.
    pub fn mean_confidence_delta(&self) -> f32 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let total: f32 = self
            .entries
            .iter()
            .map(|e| e.truth_after.confidence - e.truth_before.confidence)
            .sum();
        total / self.entries.len() as f32
    }
}

// =============================================================================
// HYDRATION CHAIN — Reversible Markov Chain via Bind/Unbind
// =============================================================================

/// A single step in the hydration chain.
#[derive(Debug, Clone)]
pub struct HydrationStep {
    /// DN of the sibling whose context was projected.
    pub sibling_dn: PackedDn,
    /// The hydrated content after this step.
    pub hydrated: Container,
    /// XOR delta from previous state to this state.
    pub delta: Container,
    /// Information content = popcount(delta) = energy of this transition.
    pub info_content: u32,
}

/// A reversible chain of context projections (hydration steps).
///
/// Each step is an XOR delta from the node's content to a sibling's context.
/// The chain IS a reversible Markov chain: bind = forward (XOR),
/// unbind = reverse (XOR again, since XOR is its own inverse).
#[derive(Debug, Clone)]
pub struct HydrationChain {
    /// The original node content (chain anchor).
    pub origin: Container,
    /// The DN of the original node.
    pub origin_dn: PackedDn,
    /// Ordered hydration steps through sibling contexts.
    pub steps: Vec<HydrationStep>,
}

impl HydrationChain {
    /// Build a hydration chain for a node by projecting through its siblings.
    ///
    /// For each sibling under the same parent:
    /// 1. Compute context_delta = extract_perspective(current, sibling)
    /// 2. Hydrate: cross_hydrate(current, context_delta)
    /// 3. Record the delta and its info content (popcount)
    pub fn build(graph: &ContainerGraph, dn: PackedDn) -> Option<Self> {
        let record = graph.get(&dn)?;
        let origin = record.content.clone();
        let parent_dn = dn.parent()?;
        let siblings = graph.children_of(&parent_dn);

        let mut steps = Vec::new();
        let mut current = origin.clone();

        for &sib_dn in siblings {
            if sib_dn == dn {
                continue;
            }
            if let Some(sib_fp) = graph.fingerprint(&sib_dn) {
                // extract_perspective = XOR between the two contexts
                let context_delta = CogRecord::extract_perspective(&current, sib_fp);
                // cross_hydrate = XOR with the context delta
                let hydrated = CogRecord::cross_hydrate(&current, &context_delta);
                let (delta, info) = delta_encode(&current, &hydrated);
                steps.push(HydrationStep {
                    sibling_dn: sib_dn,
                    hydrated: hydrated.clone(),
                    delta,
                    info_content: info,
                });
                current = hydrated;
            }
        }

        Some(Self {
            origin,
            origin_dn: dn,
            steps,
        })
    }

    /// Replay the chain forward from origin (bind direction).
    pub fn replay_forward(&self) -> Container {
        let mut current = self.origin.clone();
        for step in &self.steps {
            current = current.xor(&step.delta);
        }
        current
    }

    /// Replay backward from end state (unbind direction).
    /// XOR is self-inverse: applying the same deltas in reverse recovers origin.
    pub fn replay_backward(&self) -> Container {
        let mut current = self.replay_forward();
        for step in self.steps.iter().rev() {
            current = current.xor(&step.delta);
        }
        current
    }

    /// Total information content (energy) of the hydration chain.
    pub fn total_energy(&self) -> u32 {
        self.steps.iter().map(|s| s.info_content).sum()
    }

    /// Mean information content per step.
    pub fn mean_energy(&self) -> f32 {
        if self.steps.is_empty() {
            return 0.0;
        }
        self.total_energy() as f32 / self.steps.len() as f32
    }

    /// Convert to delta encoding format for compact storage.
    pub fn as_delta_chain(&self) -> (Container, Vec<(Container, u32)>) {
        let mut containers = vec![self.origin.clone()];
        for step in &self.steps {
            containers.push(step.hydrated.clone());
        }
        chain_encode(&containers)
    }
}

// =============================================================================
// FREE ENERGY SEMIRING
// =============================================================================

/// How to combine free energies at graph junctions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnergyStrategy {
    /// Take minimum surprise — path of least resistance.
    MinSurprise,
    /// Take maximum surprise — path of greatest free energy.
    MaxSurprise,
}

/// Free energy semiring for graph-wide surprise computation.
///
/// Plugs into existing `container_mxv()` / `container_multi_hop()`.
pub struct FreeEnergySemiring {
    pub strategy: EnergyStrategy,
}

impl DnSemiring for FreeEnergySemiring {
    type Value = f32;

    fn zero(&self) -> f32 {
        match self.strategy {
            EnergyStrategy::MinSurprise => f32::INFINITY,
            EnergyStrategy::MaxSurprise => f32::NEG_INFINITY,
        }
    }

    fn multiply(
        &self,
        _verb: u8,
        _weight_hint: u8,
        input: &f32,
        src_fp: &Container,
        dst_fp: Option<&Container>,
    ) -> f32 {
        match dst_fp {
            Some(dst) => {
                let edge_surprise = src_fp.hamming(dst) as f32 / CONTAINER_BITS as f32;
                if input.is_finite() {
                    input + edge_surprise
                } else {
                    edge_surprise
                }
            }
            None => self.zero(),
        }
    }

    fn add(&self, a: &f32, b: &f32) -> f32 {
        match self.strategy {
            EnergyStrategy::MinSurprise => a.min(*b),
            EnergyStrategy::MaxSurprise => a.max(*b),
        }
    }

    fn is_zero(&self, val: &f32) -> bool {
        !val.is_finite()
    }

    fn name(&self) -> &'static str {
        match self.strategy {
            EnergyStrategy::MinSurprise => "FreeEnergy(MinSurprise)",
            EnergyStrategy::MaxSurprise => "FreeEnergy(MaxSurprise)",
        }
    }
}

// =============================================================================
// CORE REFLECTION FUNCTIONS
// =============================================================================

/// Walk the DN tree reflecting on each node: compare surprise with NARS
/// confidence, classify the outcome, and update truth values in-place.
///
/// Two-phase approach satisfies the borrow checker:
/// 1. `felt_walk()` — immutable borrow, collect all surprise data
/// 2. Iterate choices, read NARS, classify, mutate
pub fn reflect_walk(
    graph: &mut ContainerGraph,
    target: PackedDn,
    query: &Container,
) -> ReflectionResult {
    // Phase 1: felt walk (read-only) to gather surprise data
    let felt_path = felt_walk(graph, target, query);

    let mut entries = Vec::with_capacity(felt_path.choices.len());
    let mut hydration_candidates = Vec::new();

    // Phase 2: reflect on each felt choice's chosen child
    for choice in &felt_path.choices {
        let dn = choice.chosen_dn;

        let truth_before = match graph.get(&dn) {
            Some(record) => read_truth(record),
            None => continue,
        };

        let outcome = classify(choice.surprise, truth_before.confidence);

        let truth_after = match outcome {
            ReflectionOutcome::Revise => {
                // High surprise = contradicts belief. Create observation
                // with inverted surprise as frequency.
                let obs_freq = 1.0 - choice.surprise;
                let observation = TruthValue::new(obs_freq.clamp(0.0, 1.0), 0.5);
                truth_before.revision(&observation)
            }
            ReflectionOutcome::Confirm => {
                // Low surprise confirms weak belief. Small confidence boost.
                let confirmation = TruthValue::new(truth_before.frequency, 0.3);
                truth_before.revision(&confirmation)
            }
            ReflectionOutcome::Explore => {
                // Mark for hydration — truth unchanged until hydration completes.
                hydration_candidates.push(dn);
                truth_before
            }
            ReflectionOutcome::Stable => {
                // No change needed.
                truth_before
            }
        };

        // Write updated truth back (skip Stable and Explore)
        if outcome != ReflectionOutcome::Stable && outcome != ReflectionOutcome::Explore {
            if let Some(record) = graph.get_mut(&dn) {
                write_truth(record, &truth_after);
            }
        }

        entries.push(ReflectionEntry {
            dn,
            surprise: choice.surprise,
            truth_before,
            truth_after,
            outcome,
            depth: choice.depth,
        });
    }

    ReflectionResult {
        entries,
        felt_path,
        hydration_candidates,
    }
}

/// Hydrate exploration candidates: for each Explore node, build a
/// HydrationChain from adjacent siblings and initialize truth value
/// based on chain energy.
pub fn hydrate_explorers(
    graph: &mut ContainerGraph,
    candidates: &[PackedDn],
) -> Vec<HydrationChain> {
    let mut chains = Vec::new();

    for &dn in candidates {
        if let Some(chain) = HydrationChain::build(graph, dn) {
            // Initialize truth based on hydration energy:
            // Low energy = similar siblings = higher initial confidence
            // High energy = diverse siblings = lower initial confidence
            let mean_energy = chain.mean_energy();
            let normalized = mean_energy / CONTAINER_BITS as f32;
            let initial_conf = (1.0 - normalized).clamp(0.1, 0.5);
            let initial_tv = TruthValue::new(0.5, initial_conf);

            if let Some(record) = graph.get_mut(&dn) {
                write_truth(record, &initial_tv);
            }

            chains.push(chain);
        }
    }

    chains
}

/// Full reflection cycle: walk + reflect + hydrate.
///
/// Top-level entry point:
/// 1. Felt walk toward target, computing surprise
/// 2. Reflect at each node (compare surprise with NARS confidence)
/// 3. Hydrate any exploration candidates from their siblings
pub fn reflect_and_hydrate(
    graph: &mut ContainerGraph,
    target: PackedDn,
    query: &Container,
) -> (ReflectionResult, Vec<HydrationChain>) {
    let result = reflect_walk(graph, target, query);
    let chains = hydrate_explorers(graph, &result.hydration_candidates);
    (result, chains)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::ContainerGeometry;
    use crate::container::delta::chain_decode;

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
    fn test_read_write_truth_roundtrip() {
        let mut rec = CogRecord::new(ContainerGeometry::Cam);
        let tv = TruthValue::new(0.85, 0.72);
        write_truth(&mut rec, &tv);
        let read_back = read_truth(&rec);
        assert!((read_back.frequency - 0.85).abs() < 1e-6);
        assert!((read_back.confidence - 0.72).abs() < 1e-6);
    }

    #[test]
    fn test_read_truth_uninitialized() {
        let rec = CogRecord::new(ContainerGeometry::Cam);
        let tv = read_truth(&rec);
        assert!((tv.frequency - 0.5).abs() < 1e-6, "default freq={}", tv.frequency);
        assert!((tv.confidence - 0.0).abs() < 1e-6, "default conf={}", tv.confidence);
    }

    #[test]
    fn test_classify_outcomes() {
        assert_eq!(classify(0.7, 0.8), ReflectionOutcome::Revise);
        assert_eq!(classify(0.3, 0.2), ReflectionOutcome::Confirm);
        assert_eq!(classify(0.7, 0.2), ReflectionOutcome::Explore);
        assert_eq!(classify(0.3, 0.8), ReflectionOutcome::Stable);
    }

    #[test]
    fn test_reflect_walk_basic() {
        let mut graph = build_test_tree();
        seed_nars(&mut graph);
        let query = Container::random(42);
        let target = PackedDn::new(&[0, 1]);

        let result = reflect_walk(&mut graph, target, &query);

        assert!(!result.entries.is_empty(), "should have reflection entries");
        for entry in &result.entries {
            assert!(entry.surprise >= 0.0 && entry.surprise <= 1.0);
            assert!(entry.truth_after.frequency >= 0.0 && entry.truth_after.frequency <= 1.0);
            assert!(entry.truth_after.confidence >= 0.0 && entry.truth_after.confidence <= 1.0);
        }
    }

    #[test]
    fn test_reflect_walk_updates_nars() {
        let mut graph = build_test_tree();
        // Set high confidence on /0
        if let Some(rec) = graph.get_mut(&PackedDn::new(&[0])) {
            write_truth(rec, &TruthValue::new(0.9, 0.9));
        }

        let query = Container::random(42);
        let target = PackedDn::new(&[0, 1]);
        let result = reflect_walk(&mut graph, target, &query);

        assert!(!result.entries.is_empty(), "should have reflected on path nodes");
        // At least some entries should exist
        for entry in &result.entries {
            assert!(matches!(
                entry.outcome,
                ReflectionOutcome::Revise
                    | ReflectionOutcome::Confirm
                    | ReflectionOutcome::Explore
                    | ReflectionOutcome::Stable
            ));
        }
    }

    #[test]
    fn test_hydration_chain_build() {
        let graph = build_test_tree();
        let dn = PackedDn::new(&[0, 1]); // has siblings /0/0 and /0/2

        let chain = HydrationChain::build(&graph, dn);
        assert!(chain.is_some(), "should build chain for node with siblings");

        let chain = chain.unwrap();
        assert_eq!(chain.steps.len(), 2, "should have 2 sibling steps");

        for step in &chain.steps {
            assert!(step.info_content > 0, "info_content should be nonzero");
        }
    }

    #[test]
    fn test_hydration_chain_reversible() {
        let graph = build_test_tree();
        let dn = PackedDn::new(&[0, 1]);

        let chain = HydrationChain::build(&graph, dn).unwrap();

        // Forward then backward should recover origin
        let recovered = chain.replay_backward();
        assert_eq!(
            recovered, chain.origin,
            "unbind should recover original (XOR is self-inverse)"
        );
    }

    #[test]
    fn test_hydration_chain_delta_encoding() {
        let graph = build_test_tree();
        let dn = PackedDn::new(&[0, 0]);

        let chain = HydrationChain::build(&graph, dn).unwrap();
        let (first, deltas) = chain.as_delta_chain();

        let decoded = chain_decode(&first, &deltas);
        assert_eq!(decoded.len(), chain.steps.len() + 1);
        assert_eq!(decoded[0], chain.origin);
    }

    #[test]
    fn test_free_energy_semiring_min() {
        let semiring = FreeEnergySemiring {
            strategy: EnergyStrategy::MinSurprise,
        };

        assert_eq!(semiring.name(), "FreeEnergy(MinSurprise)");
        assert!(semiring.is_zero(&semiring.zero()));

        let a = Container::random(1);
        let b = Container::random(2);
        let result = semiring.multiply(0, 0, &0.3, &a, Some(&b));
        assert!(result > 0.3, "should accumulate surprise");

        let combined = semiring.add(&0.5, &0.3);
        assert!((combined - 0.3).abs() < 1e-6, "should take min");
    }

    #[test]
    fn test_free_energy_semiring_max() {
        let semiring = FreeEnergySemiring {
            strategy: EnergyStrategy::MaxSurprise,
        };

        let combined = semiring.add(&0.5, &0.3);
        assert!((combined - 0.5).abs() < 1e-6, "should take max");
    }

    #[test]
    fn test_reflect_and_hydrate_full_cycle() {
        let mut graph = build_test_tree();
        seed_nars(&mut graph);

        let query = Container::random(42);
        let target = PackedDn::new(&[0, 1]);

        let (result, chains) = reflect_and_hydrate(&mut graph, target, &query);

        assert!(!result.entries.is_empty());
        assert_eq!(chains.len(), result.hydration_candidates.len());

        for entry in &result.entries {
            if let Some(rec) = graph.get(&entry.dn) {
                let tv = read_truth(rec);
                assert!(tv.frequency >= 0.0 && tv.frequency <= 1.0);
                assert!(tv.confidence >= 0.0 && tv.confidence <= 1.0);
            }
        }
    }

    #[test]
    fn test_hydration_energy() {
        let graph = build_test_tree();
        let dn = PackedDn::new(&[0, 1]);
        let chain = HydrationChain::build(&graph, dn).unwrap();

        let total = chain.total_energy();
        let mean = chain.mean_energy();

        assert!(total > 0, "total energy should be positive");
        assert!(mean > 0.0, "mean energy should be positive");
        assert!(
            mean <= CONTAINER_BITS as f32,
            "mean energy <= max bits"
        );
    }

    #[test]
    fn test_reflection_result_stats() {
        let mut graph = build_test_tree();
        seed_nars(&mut graph);

        let query = Container::random(42);
        let target = PackedDn::new(&[0, 1]);
        let result = reflect_walk(&mut graph, target, &query);

        let _ = result.revision_count();
        let _ = result.confirmation_count();
        let mcd = result.mean_confidence_delta();
        assert!(mcd.is_finite(), "mean confidence delta should be finite");
    }
}
