//! Dream–Reflection Bridge — Connecting Ghost Resonance to Dream Consolidation
//!
//! This module bridges two existing systems:
//!
//! 1. **Reflection** (`qualia::reflection`) identifies nodes needing attention:
//!    - `Explore` nodes have sibling bundles (ghost vectors) waiting to be used
//!    - `Revise` nodes have high surprise against their current beliefs
//!    - Hydration chains carry context deltas between siblings
//!
//! 2. **Dream consolidation** (`learning::dream`) operates on CogRecord batches:
//!    prune low-confidence, merge similar, generate creative recombinations.
//!
//! ## The Missing Link
//!
//! Ghost vectors from sibling bundles (felt traversal) should surface during
//! dream consolidation. High-echo ghosts become inputs to the dream pipeline,
//! and dream-produced novels become hydration context for Explore nodes.
//!
//! ```text
//! VolitionalAgenda → ghost vectors (high resonance siblings)
//!                  ↓
//! DreamConsolidation(ghost_records + session_records)
//!                  ↓
//! Dream novels → inject as hydration context for Explore nodes
//! ```
//!
//! ## Ghost Harvesting
//!
//! During felt traversal, each branch records a sibling bundle (XOR-fold of
//! all siblings). High-resonance bundles indicate branches where the unchosen
//! paths are contextually relevant — these are the "lingering ghosts" that
//! should influence dream processing.
//!
//! Ghost harvesting extracts these high-resonance bundles as CogRecords
//! suitable for dream consolidation input.

use crate::container::{Container, CogRecord, ContainerGeometry, CONTAINER_BITS};
use crate::container::graph::ContainerGraph;
use crate::container::adjacency::PackedDn;
use crate::learning::dream::{DreamConfig, consolidate_with_config};

use super::reflection::{
    ReflectionOutcome, write_truth,
};
use super::volition::VolitionalAgenda;
use super::felt_traversal::FeltPath;
use ladybug_contract::nars::TruthValue;

// =============================================================================
// CONSTANTS
// =============================================================================

/// Minimum bundle resonance for a ghost to be harvested.
/// Ghosts below this threshold are too weakly resonant to influence dreams.
pub const GHOST_RESONANCE_THRESHOLD: f32 = 0.4;

/// NARS confidence assigned to ghost records (uncertain, contextual).
pub const GHOST_INITIAL_CONFIDENCE: f32 = 0.25;

/// NARS frequency assigned to ghost records (speculative).
pub const GHOST_INITIAL_FREQUENCY: f32 = 0.5;

/// NARS confidence assigned to dream-produced records injected as hydration context.
pub const DREAM_INJECT_CONFIDENCE: f32 = 0.3;

/// Verb ID for "dream ghost" edges — marks nodes sourced from dream consolidation.
pub const VERB_DREAM_GHOST: u8 = 0xF9;

// =============================================================================
// GHOST RECORD — A sibling bundle packaged for dream input
// =============================================================================

/// A ghost record harvested from a felt path's sibling bundles.
#[derive(Debug, Clone)]
pub struct GhostRecord {
    /// The branch DN where this ghost was observed.
    pub branch_dn: PackedDn,
    /// The sibling bundle (XOR-fold of all siblings at this branch).
    pub bundle: Container,
    /// Resonance of the query against this bundle.
    pub resonance: f32,
    /// Depth in the tree.
    pub depth: u8,
}

// =============================================================================
// GHOST HARVESTING — Extract high-resonance ghosts from felt paths
// =============================================================================

/// Harvest ghost records from a felt path.
///
/// Selects sibling bundles whose resonance exceeds the threshold,
/// packages them as CogRecords with low confidence (speculative),
/// ready for dream consolidation input.
pub fn harvest_ghosts(felt_path: &FeltPath, threshold: f32) -> Vec<GhostRecord> {
    felt_path
        .choices
        .iter()
        .filter(|choice| choice.bundle_resonance >= threshold)
        .map(|choice| GhostRecord {
            branch_dn: choice.branch_dn,
            bundle: choice.sibling_bundle.clone(),
            resonance: choice.bundle_resonance,
            depth: choice.depth,
        })
        .collect()
}

/// Convert ghost records into CogRecords suitable for dream consolidation.
///
/// Each ghost becomes a CogRecord with:
/// - Content = sibling bundle (the ghost vector)
/// - Geometry = Cam (compatible with standard containers)
/// - Confidence = low (speculative, from uncollapsed context)
/// - Frequency = 0.5 (neutral — neither confirmed nor denied)
pub fn ghosts_to_records(ghosts: &[GhostRecord]) -> Vec<CogRecord> {
    ghosts
        .iter()
        .map(|ghost| {
            let mut record = CogRecord::new(ContainerGeometry::Cam);
            record.content = ghost.bundle.clone();
            record.meta_view_mut().set_nars_confidence(GHOST_INITIAL_CONFIDENCE);
            record.meta_view_mut().set_nars_frequency(GHOST_INITIAL_FREQUENCY);
            record
        })
        .collect()
}

// =============================================================================
// DREAM-REFLECTION INTEGRATION
// =============================================================================

/// Configuration for dream-reflection integration.
#[derive(Debug, Clone)]
pub struct DreamReflectionConfig {
    /// Ghost resonance threshold for harvesting.
    pub ghost_threshold: f32,
    /// Dream consolidation config.
    pub dream_config: DreamConfig,
    /// Maximum number of dream novels to inject per Explore node.
    pub max_inject_per_node: usize,
    /// Minimum similarity between dream novel and Explore node for injection.
    pub inject_similarity_threshold: f32,
}

impl Default for DreamReflectionConfig {
    fn default() -> Self {
        Self {
            ghost_threshold: GHOST_RESONANCE_THRESHOLD,
            dream_config: DreamConfig {
                prune_confidence_threshold: 0.15, // lower than default — ghosts start low
                recombination_count: 8,           // more novels from ghost context
                ..DreamConfig::default()
            },
            max_inject_per_node: 3,
            inject_similarity_threshold: 0.35,
        }
    }
}

/// Result of dream-reflection integration.
#[derive(Debug, Clone)]
pub struct DreamReflectionResult {
    /// Ghost records harvested from the felt path.
    pub ghosts_harvested: usize,
    /// Total records fed to dream consolidation (session + ghosts).
    pub dream_input_count: usize,
    /// Records produced by dream consolidation.
    pub dream_output_count: usize,
    /// Novel dream records injected as hydration context.
    pub injections: Vec<DreamInjection>,
}

/// A single dream injection: a dream-produced novel matched to an Explore node.
#[derive(Debug, Clone)]
pub struct DreamInjection {
    /// The Explore node that received this context.
    pub target_dn: PackedDn,
    /// The dream-produced novel content.
    pub novel_content: Container,
    /// Similarity between the novel and the Explore node's content.
    pub similarity: f32,
}

/// Run dream consolidation on session records enriched with ghost context,
/// then inject dream novels into Explore nodes as hydration context.
///
/// This is the full dream-reflection integration cycle:
///
/// 1. Harvest high-resonance ghosts from the volitional agenda's felt path
/// 2. Combine ghost records with session records
/// 3. Run dream consolidation on the combined set
/// 4. Match dream-produced novels against Explore candidates
/// 5. Inject matching novels as hydration context (update content + NARS)
pub fn dream_reflection_cycle(
    graph: &mut ContainerGraph,
    agenda: &VolitionalAgenda,
    session_records: &[CogRecord],
    config: &DreamReflectionConfig,
) -> DreamReflectionResult {
    // Step 1: Harvest ghosts from the felt path
    let ghosts = harvest_ghosts(&agenda.reflection.felt_path, config.ghost_threshold);
    let ghosts_harvested = ghosts.len();
    let ghost_records = ghosts_to_records(&ghosts);

    // Step 2: Combine session records with ghost records
    let mut dream_input: Vec<CogRecord> = session_records.to_vec();
    dream_input.extend(ghost_records);
    let dream_input_count = dream_input.len();

    // Step 3: Run dream consolidation
    let dream_output = consolidate_with_config(&dream_input, &config.dream_config);
    let dream_output_count = dream_output.len();

    // Step 4: Identify dream-produced novels (records not in original input)
    // Novels are records whose content differs significantly from all inputs.
    let novels: Vec<&CogRecord> = dream_output
        .iter()
        .filter(|output| {
            // A record is "novel" if it doesn't closely match any input
            !dream_input.iter().any(|input| {
                input.content.hamming(&output.content) < (CONTAINER_BITS as u32 / 8)
            })
        })
        .collect();

    // Step 5: Match novels against Explore candidates and inject
    let explore_dns: Vec<PackedDn> = agenda
        .acts
        .iter()
        .filter(|act| act.outcome == ReflectionOutcome::Explore)
        .map(|act| act.dn)
        .collect();

    let mut injections = Vec::new();

    for &explore_dn in &explore_dns {
        let explore_fp = match graph.fingerprint(&explore_dn) {
            Some(fp) => fp.clone(),
            None => continue,
        };

        // Find the best-matching novels for this Explore node
        let mut matches: Vec<(&CogRecord, f32)> = novels
            .iter()
            .map(|novel| {
                let sim = explore_fp.similarity(&novel.content);
                (*novel, sim)
            })
            .filter(|(_, sim)| *sim >= config.inject_similarity_threshold)
            .collect();

        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        matches.truncate(config.max_inject_per_node);

        for (novel, similarity) in matches {
            // Inject: XOR the dream novel into the Explore node's content
            // This is cross-hydration with dream context
            if let Some(record) = graph.get_mut(&explore_dn) {
                let hydrated = record.content.xor(&novel.content);
                record.content = hydrated;

                // Set initial truth value: low confidence, neutral frequency
                write_truth(record, &TruthValue::new(GHOST_INITIAL_FREQUENCY, DREAM_INJECT_CONFIDENCE));
            }

            // Add a dream-ghost edge for provenance tracking
            graph.add_edge(&explore_dn, VERB_DREAM_GHOST, 0);

            injections.push(DreamInjection {
                target_dn: explore_dn,
                novel_content: novel.content.clone(),
                similarity,
            });
        }
    }

    DreamReflectionResult {
        ghosts_harvested,
        dream_input_count,
        dream_output_count,
        injections,
    }
}

/// Lightweight variant: harvest ghosts and consolidate, but don't inject.
///
/// Returns the consolidated dream output for external processing.
pub fn dream_consolidate_with_ghosts(
    felt_path: &FeltPath,
    session_records: &[CogRecord],
    config: &DreamReflectionConfig,
) -> Vec<CogRecord> {
    let ghosts = harvest_ghosts(felt_path, config.ghost_threshold);
    let ghost_records = ghosts_to_records(&ghosts);

    let mut dream_input: Vec<CogRecord> = session_records.to_vec();
    dream_input.extend(ghost_records);

    consolidate_with_config(&dream_input, &config.dream_config)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::graph::ContainerGraph;
    use crate::cognitive::RungLevel;
    use super::super::volition::{CouncilWeights, volitional_cycle};
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

    fn make_session_records(count: usize) -> Vec<CogRecord> {
        (0..count)
            .map(|i| {
                let mut rec = CogRecord::new(ContainerGeometry::Cam);
                rec.content = Container::random(1000 + i as u64);
                rec.meta_view_mut().set_nars_confidence(0.6);
                rec.meta_view_mut().set_nars_frequency(0.5);
                rec
            })
            .collect()
    }

    #[test]
    fn test_harvest_ghosts_empty_path() {
        let path = FeltPath {
            choices: vec![],
            target: PackedDn::ROOT,
            total_surprise: 0.0,
            mean_surprise: 0.0,
            path_context: Container::zero(),
        };
        let ghosts = harvest_ghosts(&path, GHOST_RESONANCE_THRESHOLD);
        assert!(ghosts.is_empty());
    }

    #[test]
    fn test_harvest_ghosts_from_felt_path() {
        let graph = build_test_tree();
        let query = Container::random(42);
        let target = PackedDn::new(&[0, 1]);

        let path = super::super::felt_traversal::felt_walk(&graph, target, &query);

        // With threshold 0.0, should harvest all branch bundles
        let all_ghosts = harvest_ghosts(&path, 0.0);
        assert_eq!(all_ghosts.len(), path.choices.len());

        // With very high threshold, should harvest none or few
        let strict_ghosts = harvest_ghosts(&path, 0.99);
        assert!(strict_ghosts.len() <= all_ghosts.len());
    }

    #[test]
    fn test_ghosts_to_records() {
        let ghosts = vec![
            GhostRecord {
                branch_dn: PackedDn::ROOT,
                bundle: Container::random(42),
                resonance: 0.7,
                depth: 0,
            },
            GhostRecord {
                branch_dn: PackedDn::new(&[0]),
                bundle: Container::random(43),
                resonance: 0.5,
                depth: 1,
            },
        ];

        let records = ghosts_to_records(&ghosts);
        assert_eq!(records.len(), 2);

        for rec in &records {
            let conf = rec.meta_view().nars_confidence();
            assert!((conf - GHOST_INITIAL_CONFIDENCE).abs() < 1e-5);
            let freq = rec.meta_view().nars_frequency();
            assert!((freq - GHOST_INITIAL_FREQUENCY).abs() < 1e-5);
        }
    }

    #[test]
    fn test_dream_consolidate_with_ghosts() {
        let graph = build_test_tree();
        let query = Container::random(42);
        let target = PackedDn::new(&[0, 1]);

        let path = super::super::felt_traversal::felt_walk(&graph, target, &query);
        let session = make_session_records(5);
        let config = DreamReflectionConfig::default();

        let output = dream_consolidate_with_ghosts(&path, &session, &config);
        // Should produce some records (at least surviving session records)
        assert!(!output.is_empty(), "dream should produce output");
    }

    #[test]
    fn test_dream_reflection_cycle() {
        let mut graph = build_test_tree();
        seed_nars(&mut graph);

        let query = Container::random(42);
        let target = PackedDn::new(&[0, 1]);
        let council = CouncilWeights::default();

        let agenda = volitional_cycle(
            &mut graph, target, &query,
            RungLevel::Meta, &council,
        );

        let session = make_session_records(10);
        let config = DreamReflectionConfig::default();

        let result = dream_reflection_cycle(&mut graph, &agenda, &session, &config);

        assert!(result.dream_input_count >= 10, "input should include session records");
        assert!(result.dream_output_count > 0, "should have dream output");
    }

    #[test]
    fn test_dream_reflection_config_defaults() {
        let config = DreamReflectionConfig::default();
        assert!((config.ghost_threshold - GHOST_RESONANCE_THRESHOLD).abs() < 1e-5);
        assert_eq!(config.max_inject_per_node, 3);
        assert!(config.inject_similarity_threshold > 0.0);
    }

    #[test]
    fn test_ghost_record_structure() {
        let ghost = GhostRecord {
            branch_dn: PackedDn::new(&[1, 2]),
            bundle: Container::random(999),
            resonance: 0.65,
            depth: 2,
        };
        assert_eq!(ghost.depth, 2);
        assert!(ghost.resonance > 0.5);
        assert!(!ghost.bundle.is_zero());
    }
}
