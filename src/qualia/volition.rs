//! Volition — Self-Directed Action Selection via Free Energy + Ghost Resonance
//!
//! Volition closes the loop. The system now has:
//!
//! 1. **Felt traversal**: WHERE surprise is (free energy landscape)
//! 2. **Reflection**: WHAT kind of attention each node needs (Revise/Confirm/Explore/Stable)
//! 3. **Hydration**: HOW to gather information from siblings
//! 4. **Council**: WHO decides (Guardian/Catalyst/Balanced consensus)
//! 5. **Rung system**: WHEN to go deeper (elevation triggers)
//!
//! What was missing: **WHY** — the integration of all signals into a ranked
//! priority queue of volitional acts. Volition is the system choosing its
//! own next action.
//!
//! ## The Volitional Act
//!
//! Each candidate node from `reflect_walk()` has:
//! - **Free energy** (surprise): urgency signal — how much does reality disagree?
//! - **Ghost intensity** (sibling bundle resonance): felt context — how relevant
//!   is the uncollapsed superposition field at this branch?
//! - **NARS confidence**: certainty signal — how committed are current beliefs?
//! - **Rung accessibility**: depth gate — is this node accessible at current rung?
//!
//! The volition score combines these into a single priority:
//!
//! ```text
//! volition = free_energy × ghost_intensity × (1 - confidence) × rung_weight
//!          = urgency    × felt_relevance   × uncertainty       × accessibility
//! ```
//!
//! High volition = "I must attend to this NOW." Low volition = "This can wait."
//!
//! ## Council Modulation
//!
//! The council modulates the raw volition score:
//! - **Guardian** dampens high-surprise candidates (caution)
//! - **Catalyst** amplifies high-surprise candidates (curiosity)
//! - **Balanced** leaves scores unchanged
//!
//! The final council-weighted score determines which bucket-list candidates
//! the system actually pursues.

use crate::container::Container;
use crate::container::graph::ContainerGraph;
use crate::container::adjacency::PackedDn;
use crate::cognitive::RungLevel;

use super::reflection::{
    ReflectionResult, ReflectionOutcome, HydrationChain,
    read_truth, reflect_walk, hydrate_explorers,
};
use super::felt_traversal::node_free_energy;

// =============================================================================
// VERB CONSTANT
// =============================================================================

/// Verb ID for "volition" edges — marks nodes chosen by volitional act.
pub const VERB_VOLITION: u8 = 0xFA;

// =============================================================================
// VOLITIONAL ACT
// =============================================================================

/// A single candidate for volitional action.
#[derive(Debug, Clone)]
pub struct VolitionalAct {
    /// The DN of the candidate node.
    pub dn: PackedDn,
    /// The reflection outcome that flagged this node.
    pub outcome: ReflectionOutcome,
    /// Free energy (surprise) at this node — urgency.
    pub free_energy: f32,
    /// Sibling bundle resonance to the query — ghost intensity / felt context.
    pub ghost_intensity: f32,
    /// NARS confidence before reflection — current certainty.
    pub confidence: f32,
    /// Rung accessibility weight (0.0 = inaccessible, 1.0 = fully accessible).
    pub rung_weight: f32,
    /// Raw volition score (before council modulation).
    pub raw_score: f32,
    /// Council-modulated scores: [guardian, catalyst, balanced].
    pub council_scores: [f32; 3],
    /// Final consensus score (council majority).
    pub consensus_score: f32,
    /// Depth in the DN tree.
    pub depth: u8,
}

/// Priority queue of volitional acts, sorted by consensus score.
#[derive(Debug, Clone)]
pub struct VolitionalAgenda {
    /// All candidate acts, sorted highest consensus_score first.
    pub acts: Vec<VolitionalAct>,
    /// The reflection result that generated these candidates.
    pub reflection: ReflectionResult,
    /// Hydration chains for Explore candidates.
    pub chains: Vec<HydrationChain>,
    /// Total volitional energy = sum of all consensus scores.
    pub total_energy: f32,
    /// Entropy of the score distribution (higher = more indecisive).
    pub decisiveness: f32,
}

impl VolitionalAgenda {
    /// The top-priority act (highest consensus score).
    pub fn top(&self) -> Option<&VolitionalAct> {
        self.acts.first()
    }

    /// How many acts are above a given threshold.
    pub fn urgent_count(&self, threshold: f32) -> usize {
        self.acts.iter().filter(|a| a.consensus_score > threshold).count()
    }

    /// Split acts by outcome type.
    pub fn by_outcome(&self, outcome: ReflectionOutcome) -> Vec<&VolitionalAct> {
        self.acts.iter().filter(|a| a.outcome == outcome).collect()
    }

    /// The exploration candidates (Explore outcome, highest energy first).
    pub fn explorations(&self) -> Vec<&VolitionalAct> {
        self.by_outcome(ReflectionOutcome::Explore)
    }

    /// The revision candidates (Revise outcome — beliefs that need updating).
    pub fn revisions(&self) -> Vec<&VolitionalAct> {
        self.by_outcome(ReflectionOutcome::Revise)
    }
}

// =============================================================================
// COUNCIL MODULATION
// =============================================================================

/// Council modulation weights for volitional scoring.
///
/// Guardian dampens surprise (caution), Catalyst amplifies (curiosity).
#[derive(Debug, Clone)]
pub struct CouncilWeights {
    /// Guardian: how much to dampen high-surprise candidates.
    /// 0.0 = full damping, 1.0 = no effect, >1.0 = amplify (unusual for guardian).
    pub guardian_surprise_factor: f32,
    /// Catalyst: how much to amplify high-surprise candidates.
    pub catalyst_surprise_factor: f32,
    /// Balanced: neutral factor (typically 1.0).
    pub balanced_factor: f32,
}

impl Default for CouncilWeights {
    fn default() -> Self {
        Self {
            guardian_surprise_factor: 0.6,  // Guardian dampens surprise by 40%
            catalyst_surprise_factor: 1.5,  // Catalyst amplifies surprise by 50%
            balanced_factor: 1.0,           // Balanced is neutral
        }
    }
}

impl CouncilWeights {
    /// Modulate a raw volition score through each archetype.
    /// Returns [guardian_score, catalyst_score, balanced_score].
    fn modulate(&self, raw_score: f32, surprise: f32) -> [f32; 3] {
        // Guardian: more cautious with surprising things
        let guardian = raw_score * (1.0 - surprise * (1.0 - self.guardian_surprise_factor));
        // Catalyst: more excited by surprising things
        let catalyst = raw_score * (1.0 + surprise * (self.catalyst_surprise_factor - 1.0));
        // Balanced: unmoved
        let balanced = raw_score * self.balanced_factor;

        [guardian.max(0.0), catalyst.max(0.0), balanced.max(0.0)]
    }

    /// Council consensus: median of the three scores.
    /// This is the "majority vote" in score space — not bit-level but
    /// the same principle: the moderate voice prevails.
    fn consensus(scores: &[f32; 3]) -> f32 {
        let mut sorted = *scores;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted[1] // median = the "majority" position
    }
}

// =============================================================================
// RUNG ACCESSIBILITY
// =============================================================================

/// Compute rung accessibility weight for a node at a given depth.
///
/// Deeper nodes require higher rungs to access. If the current rung
/// is too shallow, the weight drops toward zero.
fn rung_accessibility(current_rung: RungLevel, node_depth: u8) -> f32 {
    let rung = current_rung.as_u8() as f32;
    let depth = node_depth as f32;

    if depth <= 2.0 {
        // Shallow nodes always accessible
        1.0
    } else if rung >= depth {
        // Rung at or above depth: full access
        1.0
    } else {
        // Rung below depth: diminishing access
        // At rung 0, depth 7 → 0.0/7.0 = 0.0
        // At rung 3, depth 5 → 3.0/5.0 = 0.6
        (rung / depth).clamp(0.0, 1.0)
    }
}

// =============================================================================
// VOLITION COMPUTATION
// =============================================================================

/// Compute the volitional agenda from a reflection result.
///
/// Takes the output of `reflect_walk()` and scores each reflected node
/// by combining free energy, ghost intensity, confidence, and rung weight.
/// Then modulates through the council and sorts by consensus priority.
pub fn compute_agenda(
    graph: &mut ContainerGraph,
    reflection: ReflectionResult,
    _query: &Container,
    current_rung: RungLevel,
    council: &CouncilWeights,
) -> VolitionalAgenda {
    let mut acts = Vec::with_capacity(reflection.entries.len());

    for entry in &reflection.entries {
        // Ghost intensity: how much does the sibling superposition
        // at this branch resonate with the query?
        let ghost_intensity = reflection
            .felt_path
            .choices
            .iter()
            .find(|c| c.chosen_dn == entry.dn)
            .map(|c| c.bundle_resonance)
            .unwrap_or(0.0);

        // Rung accessibility
        let rung_weight = rung_accessibility(current_rung, entry.depth);

        // Raw volition score: urgency × felt_context × uncertainty × accessibility
        let uncertainty = 1.0 - entry.truth_before.confidence;
        let raw_score = entry.surprise * ghost_intensity * uncertainty * rung_weight;

        // Council modulation
        let council_scores = council.modulate(raw_score, entry.surprise);
        let consensus_score = CouncilWeights::consensus(&council_scores);

        acts.push(VolitionalAct {
            dn: entry.dn,
            outcome: entry.outcome,
            free_energy: entry.surprise,
            ghost_intensity,
            confidence: entry.truth_before.confidence,
            rung_weight,
            raw_score,
            council_scores,
            consensus_score,
            depth: entry.depth,
        });
    }

    // Sort by consensus score descending (highest priority first)
    acts.sort_by(|a, b| b.consensus_score.partial_cmp(&a.consensus_score)
        .unwrap_or(std::cmp::Ordering::Equal));

    // Compute agenda-level metrics
    let total_energy: f32 = acts.iter().map(|a| a.consensus_score).sum();

    // Decisiveness: 1.0 - normalized entropy of score distribution
    // High decisiveness = one clear winner. Low = many similar scores.
    let decisiveness = if acts.len() < 2 {
        1.0
    } else {
        let max_score = acts.first().map(|a| a.consensus_score).unwrap_or(0.0);
        let second_score = acts.get(1).map(|a| a.consensus_score).unwrap_or(0.0);
        if max_score <= 0.0 {
            0.0
        } else {
            (1.0 - second_score / max_score).clamp(0.0, 1.0)
        }
    };

    // Hydrate exploration candidates
    let chains = hydrate_explorers(graph, &reflection.hydration_candidates);

    VolitionalAgenda {
        acts,
        reflection,
        chains,
        total_energy,
        decisiveness,
    }
}

/// Full volitional cycle: reflect → score → rank → hydrate.
///
/// Top-level entry point. Given a target and query:
/// 1. Walk the tree computing surprise (felt traversal)
/// 2. Reflect on each node (compare surprise with NARS confidence)
/// 3. Score each candidate (free energy × ghost × uncertainty × rung)
/// 4. Modulate through council (guardian dampens, catalyst amplifies)
/// 5. Rank by consensus and hydrate explorers
pub fn volitional_cycle(
    graph: &mut ContainerGraph,
    target: PackedDn,
    query: &Container,
    current_rung: RungLevel,
    council: &CouncilWeights,
) -> VolitionalAgenda {
    let reflection = reflect_walk(graph, target, query);
    compute_agenda(graph, reflection, query, current_rung, council)
}

/// Focused volition: only score nodes of a specific outcome type.
///
/// Useful when you want to find the most urgent revision candidate
/// without considering exploration nodes, or vice versa.
pub fn focused_volition(
    agenda: &VolitionalAgenda,
    focus: ReflectionOutcome,
) -> Vec<&VolitionalAct> {
    let focused: Vec<&VolitionalAct> = agenda.acts
        .iter()
        .filter(|a| a.outcome == focus)
        .collect();
    // Already sorted by consensus_score from compute_agenda
    focused
}

/// Compute the volitional gradient across the free energy landscape.
///
/// For each node in the graph rooted at `root`, compute the local free
/// energy and ghost intensity, producing a map of "where volition pulls."
/// This is the spatial derivative of the volition field — the system's
/// attentional gravity map.
pub fn volitional_gradient(
    graph: &ContainerGraph,
    root: PackedDn,
    current_rung: RungLevel,
) -> Vec<(PackedDn, f32)> {
    let mut gradient = Vec::new();

    fn walk(
        graph: &ContainerGraph,
        dn: PackedDn,
        depth: u8,
        current_rung: RungLevel,
        gradient: &mut Vec<(PackedDn, f32)>,
    ) {
        let rung_w = rung_accessibility(current_rung, depth);
        if rung_w <= 0.01 {
            return; // Inaccessible at current rung
        }

        let fe = node_free_energy(graph, dn).unwrap_or(0.0);
        let tv = graph.get(&dn)
            .map(|r| read_truth(r))
            .unwrap_or_else(|| ladybug_contract::nars::TruthValue::unknown());

        let uncertainty = 1.0 - tv.confidence;
        let score = fe * uncertainty * rung_w;

        if score > 0.001 {
            gradient.push((dn, score));
        }

        // Recurse into children
        for &child in graph.children_of(&dn) {
            walk(graph, child, depth + 1, current_rung, gradient);
        }
    }

    walk(graph, root, 0, current_rung, &mut gradient);

    // Sort by score descending
    gradient.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    gradient
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::{ContainerGeometry, CogRecord};
    use super::super::reflection::write_truth;
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
    fn test_rung_accessibility() {
        // Shallow nodes always accessible
        assert_eq!(rung_accessibility(RungLevel::Surface, 0), 1.0);
        assert_eq!(rung_accessibility(RungLevel::Surface, 2), 1.0);

        // Rung matches depth
        assert_eq!(rung_accessibility(RungLevel::Structural, 5), 1.0);

        // Rung above depth
        assert_eq!(rung_accessibility(RungLevel::Meta, 3), 1.0);

        // Rung below depth — diminished access
        let w = rung_accessibility(RungLevel::Analogical, 6); // rung 3, depth 6
        assert!(w > 0.0 && w < 1.0, "should be partial: {}", w);
        assert!((w - 0.5).abs() < 0.01, "3/6 = 0.5, got {}", w);
    }

    #[test]
    fn test_council_modulation() {
        let council = CouncilWeights::default();

        // High surprise candidate
        let scores = council.modulate(0.8, 0.9); // raw=0.8, surprise=0.9
        assert!(scores[0] < scores[2], "guardian should dampen: {:?}", scores);
        assert!(scores[1] > scores[2], "catalyst should amplify: {:?}", scores);

        // Low surprise candidate
        let scores_low = council.modulate(0.8, 0.1);
        assert!((scores_low[0] - scores_low[2]).abs() < 0.1,
            "low surprise: guardian ~= balanced: {:?}", scores_low);
    }

    #[test]
    fn test_council_consensus_median() {
        let scores = [0.3, 0.9, 0.6];
        let c = CouncilWeights::consensus(&scores);
        assert!((c - 0.6).abs() < 1e-6, "median of [0.3, 0.9, 0.6] = 0.6, got {}", c);
    }

    #[test]
    fn test_volitional_cycle_basic() {
        let mut graph = build_test_tree();
        seed_nars(&mut graph);

        let query = Container::random(42);
        let target = PackedDn::new(&[0, 1]);
        let council = CouncilWeights::default();

        let agenda = volitional_cycle(
            &mut graph, target, &query,
            RungLevel::Meta, &council,
        );

        assert!(!agenda.acts.is_empty(), "should have volitional acts");
        assert!(agenda.total_energy >= 0.0, "total energy should be non-negative");
        assert!(agenda.decisiveness >= 0.0 && agenda.decisiveness <= 1.0,
            "decisiveness in [0,1]: {}", agenda.decisiveness);

        // Acts should be sorted by consensus_score descending
        for w in agenda.acts.windows(2) {
            assert!(w[0].consensus_score >= w[1].consensus_score,
                "should be sorted: {} >= {}", w[0].consensus_score, w[1].consensus_score);
        }
    }

    #[test]
    fn test_volitional_cycle_rung_gating() {
        let mut graph = build_test_tree();
        seed_nars(&mut graph);

        let query = Container::random(42);
        let target = PackedDn::new(&[0, 1]);
        let council = CouncilWeights::default();

        // With high rung: full access
        let high_rung = volitional_cycle(
            &mut graph, target, &query,
            RungLevel::Transcendent, &council,
        );

        // With low rung: diminished access to deep nodes
        let low_rung = volitional_cycle(
            &mut graph, target, &query,
            RungLevel::Surface, &council,
        );

        // Both should have acts (shallow nodes always accessible)
        assert!(!high_rung.acts.is_empty());
        assert!(!low_rung.acts.is_empty());

        // Deep nodes should have lower rung_weight at low rung
        for act in &low_rung.acts {
            if act.depth > 2 {
                assert!(act.rung_weight < 1.0,
                    "deep node at low rung should have reduced weight: depth={}, weight={}",
                    act.depth, act.rung_weight);
            }
        }
    }

    #[test]
    fn test_focused_volition() {
        let mut graph = build_test_tree();
        seed_nars(&mut graph);

        let query = Container::random(42);
        let target = PackedDn::new(&[0, 1]);
        let council = CouncilWeights::default();

        let agenda = volitional_cycle(
            &mut graph, target, &query,
            RungLevel::Meta, &council,
        );

        let explores = focused_volition(&agenda, ReflectionOutcome::Explore);
        for act in &explores {
            assert_eq!(act.outcome, ReflectionOutcome::Explore);
        }
    }

    #[test]
    fn test_volitional_gradient() {
        let graph = build_test_tree();
        let gradient = volitional_gradient(
            &graph, PackedDn::ROOT, RungLevel::Meta,
        );

        // Should find some nodes with positive gradient
        // (random containers will have varying free energy)
        for (dn, score) in &gradient {
            assert!(*score > 0.0, "gradient entries should be positive");
        }

        // Should be sorted descending
        for w in gradient.windows(2) {
            assert!(w[0].1 >= w[1].1, "gradient should be sorted");
        }
    }

    #[test]
    fn test_agenda_helpers() {
        let mut graph = build_test_tree();
        seed_nars(&mut graph);

        let query = Container::random(42);
        let target = PackedDn::new(&[0, 1]);
        let council = CouncilWeights::default();

        let agenda = volitional_cycle(
            &mut graph, target, &query,
            RungLevel::Meta, &council,
        );

        // Test helper methods
        let _ = agenda.top();
        let _ = agenda.urgent_count(0.5);
        let _ = agenda.explorations();
        let _ = agenda.revisions();
    }
}
