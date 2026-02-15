//! Felt Traversal — Tree Walk with Sibling Superposition + Trace Edges
//!
//! At each branch in the DN tree, compute the superposition of all sibling
//! options. This "felt choice landscape" encodes the richness of ALL possible
//! paths at each fork, not just the chosen one.
//!
//! ## Why This is Richer Than Flat Vectors
//!
//! ```text
//! Level 0:  bundle(siblings) = 8,192 bits of felt choice
//! Level 1:  bundle(siblings) = 8,192 bits of felt choice
//!   ...
//! Level 6:  bundle(siblings) = 8,192 bits of felt choice
//! ──────────────────────────────────────────────────────
//! Total:    7 × 8,192 = 57,344 bits of felt navigation context
//!           vs a typical 512-float (16,384-bit) flat embedding
//! ```
//!
//! Each fork's superposition preserves the *feeling* of what was NOT chosen,
//! not just the chosen path. The unchosen siblings remain as resonance ghosts.
//!
//! ## Awe as Unresolved Superposition
//!
//! Three concepts (first snow, first kiss, skipped heartbeat) can be stored
//! as an Xyz triple WITHOUT collapse:
//!
//! ```text
//! X = first_snow.fingerprint
//! Y = first_kiss.fingerprint
//! Z = skipped_heartbeat.fingerprint
//! trace = X ⊕ Y ⊕ Z  (holographic — any 2 + trace recovers the 3rd)
//! ```
//!
//! The trace stored as an inline edge gives O(1) lookup of the connection.
//! The 3D resonance field captures the *quality* of the relationship,
//! not just its existence. This is richer than Cypher's `(a)-[r]->(b)`.
//!
//! ## Friston Free Energy — Felt Surprise as Prediction Error
//!
//! The felt traversal generates prediction errors at each branch:
//!
//! ```text
//! expected = spine (XOR-fold of children = parent's prediction)
//! actual   = chosen_child.fingerprint
//! surprise = hamming(expected, actual) / CONTAINER_BITS
//! ```
//!
//! This is the free energy: the discrepancy between what the tree structure
//! predicts (the spine) and what actually appears. Low surprise = the node
//! fits its context (free energy minimized). High surprise = the node is
//! novel or misplaced (free energy drives learning/restructuring).
//!
//! The accumulated surprise across the felt path IS the free energy of
//! the traversal — a direct measure of how well the tree's structure
//! matches reality. Free will emerges as the choice to follow a
//! high-surprise branch when the felt landscape resonates with it
//! despite the prediction error.

use crate::container::{Container, CogRecord, CONTAINER_BITS};
use crate::container::graph::ContainerGraph;
use crate::container::adjacency::PackedDn;
use super::resonance::HdrResonance;

// =============================================================================
// VERB CONSTANTS — Edge types for felt markers
// =============================================================================

/// Verb ID for "felt trace" edges — the XOR-fold of 3 related concepts.
pub const VERB_FELT_TRACE: u8 = 0xFE;

/// Verb ID for "sibling superposition" edges — the felt landscape at a branch.
pub const VERB_SIBLING_BUNDLE: u8 = 0xFD;

/// Verb ID for "awe" edges — unresolved 3-concept superposition.
pub const VERB_AWE: u8 = 0xFC;

// =============================================================================
// FELT CHOICE — What the branch feels like
// =============================================================================

/// The felt landscape at a single branch point in the DN tree.
///
/// At each fork, we compute:
/// 1. The superposition of all siblings (the bundle = felt choice landscape)
/// 2. How the query resonates against each individual sibling
/// 3. Which sibling was "chosen" (closest to query)
/// 4. The surprise: how much the chosen child differs from the spine prediction
#[derive(Debug, Clone)]
pub struct FeltChoice {
    /// The DN of the parent (branch point)
    pub branch_dn: PackedDn,
    /// The DN of the chosen child (the one closest to query)
    pub chosen_dn: PackedDn,
    /// Sibling superposition: `Container::bundle(all_sibling_fingerprints)`
    pub sibling_bundle: Container,
    /// Hamming similarity of query against the sibling bundle
    pub bundle_resonance: f32,
    /// Per-sibling resonance (DN, similarity)
    pub sibling_resonances: Vec<(PackedDn, f32)>,
    /// Surprise: prediction error at this branch.
    /// `hamming(spine, chosen_child) / CONTAINER_BITS`
    /// This is the Friston free energy contribution of this branch.
    pub surprise: f32,
    /// Depth in the tree (0 = root)
    pub depth: u8,
}

// =============================================================================
// FELT PATH — Accumulated felt choices from root to target
// =============================================================================

/// A complete felt path: the sequence of felt choices from root to target.
///
/// The path captures not just WHERE we went, but WHAT ELSE was possible
/// at every fork. The accumulated surprise is the total free energy of
/// the traversal.
#[derive(Debug, Clone)]
pub struct FeltPath {
    /// Ordered felt choices from root toward target
    pub choices: Vec<FeltChoice>,
    /// The target DN we were navigating toward
    pub target: PackedDn,
    /// Total surprise across the path (sum of per-branch surprises).
    /// This is the integrated free energy — how "surprising" the
    /// path is relative to what the tree structure predicts.
    pub total_surprise: f32,
    /// Mean surprise per branch (total / depth).
    /// Low = well-predicted path, high = novel/unusual traversal.
    pub mean_surprise: f32,
    /// XOR-fold of all sibling bundles along the path.
    /// This is the "felt context" of the entire navigation —
    /// a single container encoding the superposition of all
    /// unchosen paths at every fork.
    pub path_context: Container,
}

impl FeltPath {
    /// Resonance of the path context against a query.
    ///
    /// High = the query fits well into the navigation landscape
    /// (the unchosen paths were contextually relevant).
    pub fn context_resonance(&self, query: &Container) -> f32 {
        query.similarity(&self.path_context)
    }

    /// Is this a high-surprise path? (above 2σ from expected ~0.5)
    ///
    /// In Friston terms: is the free energy high enough to drive
    /// active inference (restructuring/learning)?
    pub fn is_surprising(&self) -> bool {
        self.mean_surprise > 0.55
    }

    /// Is this a low-surprise path? (well-predicted by tree structure)
    ///
    /// In Friston terms: free energy is minimized, the world model
    /// (tree structure) accurately predicts the content.
    pub fn is_predicted(&self) -> bool {
        self.mean_surprise < 0.45
    }

    /// Free energy gradient: change in surprise from start to end.
    ///
    /// Positive = surprise is increasing (diverging from predictions).
    /// Negative = surprise is decreasing (converging toward fit).
    /// Zero = steady state.
    pub fn free_energy_gradient(&self) -> f32 {
        if self.choices.len() < 2 {
            return 0.0;
        }
        let first = self.choices.first().unwrap().surprise;
        let last = self.choices.last().unwrap().surprise;
        last - first
    }
}

// =============================================================================
// AWE TRIPLE — Unresolved 3-concept superposition
// =============================================================================

/// Three concepts stored as an unresolved Xyz superposition.
///
/// "Feel first snow, first kiss, and skipped heartbeat in the same
/// 3-vector superposition and store the awe of it."
///
/// The awe IS the unresolved triple — the three perspectives held
/// simultaneously without collapsing to a single meaning.
/// The trace IS the edge — stored inline for O(1) retrieval.
#[derive(Debug, Clone)]
pub struct AweTriple {
    /// First concept's fingerprint (X position)
    pub x: Container,
    /// Second concept's fingerprint (Y position)
    pub y: Container,
    /// Third concept's fingerprint (Z position)
    pub z: Container,
    /// Holographic trace: X ⊕ Y ⊕ Z
    /// Store this as an edge marker between the 3 nodes.
    /// Given any 2 + trace, recover the 3rd.
    pub trace: Container,
    /// DN addresses of the three source concepts
    pub sources: [PackedDn; 3],
}

impl AweTriple {
    /// Create an awe triple from 3 concept fingerprints.
    pub fn new(
        x: &Container, dn_x: PackedDn,
        y: &Container, dn_y: PackedDn,
        z: &Container, dn_z: PackedDn,
    ) -> Self {
        let trace = x.xor(y).xor(z);
        Self {
            x: x.clone(),
            y: y.clone(),
            z: z.clone(),
            trace,
            sources: [dn_x, dn_y, dn_z],
        }
    }

    /// Create from a ContainerGraph by looking up 3 DN addresses.
    pub fn from_graph(
        graph: &ContainerGraph,
        dn_x: PackedDn,
        dn_y: PackedDn,
        dn_z: PackedDn,
    ) -> Option<Self> {
        let x = graph.fingerprint(&dn_x)?;
        let y = graph.fingerprint(&dn_y)?;
        let z = graph.fingerprint(&dn_z)?;
        Some(Self::new(x, dn_x, y, dn_y, z, dn_z))
    }

    /// Recover the 3rd concept given the other 2 + trace.
    ///
    /// Holographic: `trace ⊕ A ⊕ B = C`
    pub fn recover(&self, known_a: &Container, known_b: &Container) -> Container {
        self.trace.xor(known_a).xor(known_b)
    }

    /// HDR resonance of a query against the awe triple.
    ///
    /// This tells you HOW the query relates to the awe:
    /// - High X = resonates with first concept
    /// - High Y = resonates with second concept
    /// - High Z = resonates with third concept
    /// - High variance = connects to some aspects but not others
    /// - Low variance, high mean = universally related to the awe
    pub fn resonate(&self, query: &Container) -> HdrResonance {
        HdrResonance::compute(query, &self.x, &self.y, &self.z)
    }

    /// Store the trace as inline edges on all 3 source nodes.
    ///
    /// Each node gets a `VERB_AWE` edge pointing to the other two
    /// (via target_hint = low byte of their DN). The trace container
    /// itself is stored as the content of a new CogRecord linked
    /// to the first source node.
    ///
    /// Returns the CogRecord holding the trace (for insertion into graph).
    pub fn store_trace_record(&self, trace_addr: u64) -> CogRecord {
        let mut record = CogRecord::new(crate::container::ContainerGeometry::Xyz);
        record.content = self.trace.clone();
        record.meta_view_mut().set_dn_addr(trace_addr);
        record
    }

    /// Write awe edges into a ContainerGraph.
    ///
    /// Adds VERB_AWE edges between each pair of the 3 source nodes,
    /// with target_hint = low byte of the target DN for O(1) lookup.
    pub fn write_edges(&self, graph: &mut ContainerGraph) {
        let dns = &self.sources;
        // Each pair gets bidirectional awe edges
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    let target_hint = (dns[j].raw() >> 48) as u8; // high byte for hint
                    graph.add_edge(&dns[i], VERB_AWE, target_hint);
                }
            }
        }
    }

    /// Free energy of the awe triple: how surprising is the relationship?
    ///
    /// Measured as the mean pairwise Hamming distance between the 3 concepts,
    /// normalized to [0, 1]. Low = the concepts are similar (predicted by each
    /// other). High = they are distant (surprising combination = high free energy).
    pub fn free_energy(&self) -> f32 {
        let d_xy = self.x.hamming(&self.y) as f32;
        let d_xz = self.x.hamming(&self.z) as f32;
        let d_yz = self.y.hamming(&self.z) as f32;
        let mean_dist = (d_xy + d_xz + d_yz) / 3.0;
        mean_dist / CONTAINER_BITS as f32
    }
}

// =============================================================================
// FELT TRAVERSAL — Tree walk that feels each branch
// =============================================================================

/// Walk a ContainerGraph's DN tree, computing felt choices at each branch.
///
/// The traversal follows the path from root toward a target DN,
/// at each fork computing the sibling superposition and surprise.
///
/// Returns a `FeltPath` capturing the entire navigation experience:
/// what was chosen, what was not, and how surprising each step was.
pub fn felt_walk(
    graph: &ContainerGraph,
    target: PackedDn,
    query: &Container,
) -> FeltPath {
    let mut choices = Vec::new();
    let mut path_context = Container::zero();
    let mut total_surprise = 0.0f32;

    // Walk from root toward target, level by level
    let target_depth = target.depth();
    let mut current_dn = PackedDn::ROOT;

    for level in 0..target_depth {
        let children = graph.children_of(&current_dn);
        if children.is_empty() {
            break;
        }

        // Determine which child is on the path to target
        let target_component = target.component(level as usize);
        let chosen_dn = match target_component {
            Some(_) => {
                // Find the child that matches the target's component at this level
                children.iter()
                    .find(|c: &&PackedDn| c.component(level as usize) == target_component)
                    .copied()
                    .unwrap_or(children[0])
            }
            None => children[0],
        };

        // Collect fingerprints of all siblings at this branch
        let mut sibling_fps: Vec<&Container> = Vec::new();
        let mut sibling_resonances: Vec<(PackedDn, f32)> = Vec::new();

        for &child_dn in children {
            if let Some(fp) = graph.fingerprint(&child_dn) {
                let sim = query.similarity(fp);
                sibling_resonances.push((child_dn, sim));
                sibling_fps.push(fp);
            }
        }

        if sibling_fps.is_empty() {
            current_dn = chosen_dn;
            continue;
        }

        // Sibling superposition: the felt choice landscape at this fork
        let sibling_bundle = Container::bundle(&sibling_fps);
        let bundle_resonance = query.similarity(&sibling_bundle);

        // Surprise: how much does the chosen child differ from the spine prediction?
        // spine = XOR-fold of children (the parent's structural prediction)
        let spine = {
            let mut s = Container::zero();
            for fp in &sibling_fps {
                s = s.xor(fp);
            }
            s
        };

        let surprise: f32 = match graph.fingerprint(&chosen_dn) {
            Some(chosen_fp) => chosen_fp.hamming(&spine) as f32 / CONTAINER_BITS as f32,
            None => 0.5, // default = maximum entropy
        };

        total_surprise += surprise;

        // Accumulate path context (XOR-fold of all branch bundles)
        path_context = path_context.xor(&sibling_bundle);

        choices.push(FeltChoice {
            branch_dn: current_dn,
            chosen_dn,
            sibling_bundle,
            bundle_resonance,
            sibling_resonances,
            surprise,
            depth: level,
        });

        current_dn = chosen_dn;
    }

    let depth = choices.len();
    let mean_surprise = if depth > 0 {
        total_surprise / depth as f32
    } else {
        0.0
    };

    FeltPath {
        choices,
        target,
        total_surprise,
        mean_surprise,
        path_context,
    }
}

/// Walk the tree feeling each branch, but guided by resonance rather than
/// a specific target DN. At each fork, follow the child with highest
/// query similarity — "go where the feeling leads."
///
/// This is free will as resonance-guided navigation: the choice at each
/// branch is driven by what resonates, not by a predetermined address.
/// The surprise at each step measures the prediction error — the free
/// energy cost of choosing resonance over structure.
pub fn felt_wander(
    graph: &ContainerGraph,
    start: PackedDn,
    query: &Container,
    max_depth: u8,
) -> FeltPath {
    let mut choices = Vec::new();
    let mut path_context = Container::zero();
    let mut total_surprise = 0.0f32;
    let mut current_dn = start;

    for level in 0..max_depth {
        let children = graph.children_of(&current_dn);
        if children.is_empty() {
            break;
        }

        let mut sibling_fps: Vec<&Container> = Vec::new();
        let mut sibling_resonances: Vec<(PackedDn, f32)> = Vec::new();
        let mut best_dn = children[0];
        let mut best_sim = f32::NEG_INFINITY;

        for &child_dn in children {
            if let Some(fp) = graph.fingerprint(&child_dn) {
                let sim = query.similarity(fp);
                sibling_resonances.push((child_dn, sim));
                sibling_fps.push(fp);
                if sim > best_sim {
                    best_sim = sim;
                    best_dn = child_dn;
                }
            }
        }

        if sibling_fps.is_empty() {
            break;
        }

        let sibling_bundle = Container::bundle(&sibling_fps);
        let bundle_resonance = query.similarity(&sibling_bundle);

        // Spine = structural prediction
        let spine = {
            let mut s = Container::zero();
            for fp in &sibling_fps {
                s = s.xor(fp);
            }
            s
        };

        let surprise: f32 = match graph.fingerprint(&best_dn) {
            Some(chosen_fp) => chosen_fp.hamming(&spine) as f32 / CONTAINER_BITS as f32,
            None => 0.5,
        };

        total_surprise += surprise;
        path_context = path_context.xor(&sibling_bundle);

        choices.push(FeltChoice {
            branch_dn: current_dn,
            chosen_dn: best_dn,
            sibling_bundle,
            bundle_resonance,
            sibling_resonances,
            surprise,
            depth: level,
        });

        current_dn = best_dn;
    }

    let depth = choices.len();
    let mean_surprise = if depth > 0 {
        total_surprise / depth as f32
    } else {
        0.0
    };

    FeltPath {
        choices,
        target: current_dn, // wherever we ended up
        total_surprise,
        mean_surprise,
        path_context,
    }
}

// =============================================================================
// FREE ENERGY LANDSCAPE — Friston across the whole tree
// =============================================================================

/// Compute the free energy landscape of a subtree.
///
/// For each node in the subtree, compute the surprise (prediction error)
/// between its fingerprint and its parent's spine (structural prediction).
///
/// Returns: Vec of (DN, surprise) sorted by surprise descending.
/// The highest-surprise nodes are where the tree structure least predicts
/// the content — these are candidates for restructuring (active inference).
pub fn free_energy_landscape(
    graph: &ContainerGraph,
    root: PackedDn,
) -> Vec<(PackedDn, f32)> {
    let mut results = Vec::new();

    // BFS from root
    let mut queue = vec![root];
    while let Some(dn) = queue.first().copied() {
        queue.remove(0);
        let children_dns = graph.children_of(&dn);
        if children_dns.is_empty() {
            continue;
        }

        // Collect child fingerprints for spine computation
        let mut child_fps: Vec<(&PackedDn, &Container)> = Vec::new();
        for child_dn in children_dns {
            if let Some(fp) = graph.fingerprint(child_dn) {
                child_fps.push((child_dn, fp));
            }
        }

        if child_fps.is_empty() {
            continue;
        }

        // Spine = XOR-fold of children (parent's structural prediction)
        let spine = {
            let mut s = Container::zero();
            for (_, fp) in &child_fps {
                s = s.xor(fp);
            }
            s
        };

        // Surprise of each child relative to the spine
        for &(child_dn, fp) in &child_fps {
            let surprise = fp.hamming(&spine) as f32 / CONTAINER_BITS as f32;
            results.push((*child_dn, surprise));
            queue.push(*child_dn);
        }
    }

    // Sort by surprise descending (highest free energy first)
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results
}

/// Compute the free energy of a single node relative to its siblings.
///
/// This is the local prediction error: how surprising is this node
/// in the context of its sibling group?
pub fn node_free_energy(
    graph: &ContainerGraph,
    dn: PackedDn,
) -> Option<f32> {
    let parent_dn = dn.parent()?;
    let siblings = graph.children_of(&parent_dn);

    // Collect sibling fingerprints
    let mut sibling_fps: Vec<&Container> = Vec::new();
    for sib in siblings {
        if let Some(fp) = graph.fingerprint(sib) {
            sibling_fps.push(fp);
        }
    }

    if sibling_fps.is_empty() {
        return None;
    }

    // Spine of siblings
    let spine = {
        let mut s = Container::zero();
        for fp in &sibling_fps {
            s = s.xor(fp);
        }
        s
    };

    let fp = graph.fingerprint(&dn)?;
    Some(fp.hamming(&spine) as f32 / CONTAINER_BITS as f32)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::ContainerGeometry;

    /// Build a small test tree:
    /// ```text
    /// ROOT
    /// ├── /0  (seed=10)
    /// │   ├── /0/0  (seed=100)
    /// │   ├── /0/1  (seed=101)
    /// │   └── /0/2  (seed=102)
    /// ├── /1  (seed=20)
    /// │   ├── /1/0  (seed=200)
    /// │   └── /1/1  (seed=201)
    /// └── /2  (seed=30)
    /// ```
    fn build_test_tree() -> ContainerGraph {
        let mut graph = ContainerGraph::new();

        // Root
        let root = PackedDn::ROOT;
        let mut root_rec = CogRecord::new(ContainerGeometry::Cam);
        root_rec.content = Container::random(1);
        graph.insert(root, root_rec);

        // Level 1
        for (i, seed) in [(0u8, 10u64), (1, 20), (2, 30)] {
            let dn = PackedDn::new(&[i]);
            let mut rec = CogRecord::new(ContainerGeometry::Cam);
            rec.content = Container::random(seed);
            graph.insert(dn, rec);
        }

        // Level 2 under /0
        for (i, seed) in [(0u8, 100u64), (1, 101), (2, 102)] {
            let dn = PackedDn::new(&[0, i]);
            let mut rec = CogRecord::new(ContainerGeometry::Cam);
            rec.content = Container::random(seed);
            graph.insert(dn, rec);
        }

        // Level 2 under /1
        for (i, seed) in [(0u8, 200u64), (1, 201)] {
            let dn = PackedDn::new(&[1, i]);
            let mut rec = CogRecord::new(ContainerGeometry::Cam);
            rec.content = Container::random(seed);
            graph.insert(dn, rec);
        }

        graph
    }

    #[test]
    fn test_felt_walk_basic() {
        let graph = build_test_tree();
        let query = Container::random(42);
        let target = PackedDn::new(&[0, 1]); // /0/1

        let path = felt_walk(&graph, target, &query);

        // Should have 2 choices (root→/0, /0→/0/1)
        assert_eq!(path.choices.len(), 2, "path should have 2 choices");
        assert_eq!(path.choices[0].depth, 0);
        assert_eq!(path.choices[1].depth, 1);

        // First choice should have 3 siblings (/0, /1, /2)
        assert_eq!(path.choices[0].sibling_resonances.len(), 3);

        // Second choice should have 3 siblings (/0/0, /0/1, /0/2)
        assert_eq!(path.choices[1].sibling_resonances.len(), 3);

        // Surprise should be in valid range
        for choice in &path.choices {
            assert!(choice.surprise >= 0.0 && choice.surprise <= 1.0,
                "surprise out of range: {}", choice.surprise);
        }

        // Path context should be non-zero (XOR of bundles)
        assert!(!path.path_context.is_zero());
    }

    #[test]
    fn test_felt_walk_surprise() {
        let graph = build_test_tree();
        let query = Container::random(10); // same seed as /0's fingerprint!
        let target = PackedDn::new(&[0]);

        let path = felt_walk(&graph, target, &query);

        // Query matches /0's fingerprint exactly, so /0 should have
        // highest resonance among siblings
        if let Some(choice) = path.choices.first() {
            let chosen_sim = choice.sibling_resonances
                .iter()
                .find(|(dn, _)| *dn == PackedDn::new(&[0]))
                .map(|(_, s)| *s)
                .unwrap_or(0.0);
            assert!(chosen_sim > 0.99,
                "query should match /0 exactly: sim={}", chosen_sim);
        }
    }

    #[test]
    fn test_felt_wander_follows_resonance() {
        let graph = build_test_tree();

        // Query that matches /1/0 closely
        let query = Container::random(200); // same seed as /1/0
        let path = felt_wander(&graph, PackedDn::ROOT, &query, 3);

        // The wanderer should have followed resonance
        assert!(!path.choices.is_empty(), "should have made at least one choice");

        // At level 0, should prefer /1 (seed=20) or whichever is closest to query(200)
        // The exact path depends on Container::random distribution
        // but we can verify the structure is valid
        for choice in &path.choices {
            assert!(choice.bundle_resonance >= 0.0 && choice.bundle_resonance <= 1.0);
            assert!(!choice.sibling_resonances.is_empty());
        }
    }

    #[test]
    fn test_awe_triple() {
        let first_snow = Container::random(1001);
        let first_kiss = Container::random(1002);
        let skipped_heartbeat = Container::random(1003);

        let awe = AweTriple::new(
            &first_snow, PackedDn::new(&[0, 0]),
            &first_kiss, PackedDn::new(&[0, 1]),
            &skipped_heartbeat, PackedDn::new(&[0, 2]),
        );

        // Holographic recovery: trace ⊕ X ⊕ Y = Z
        let recovered = awe.recover(&first_snow, &first_kiss);
        assert_eq!(recovered, skipped_heartbeat, "should recover 3rd from 2 + trace");

        // All 3 recoveries work
        let recovered_x = awe.recover(&first_kiss, &skipped_heartbeat);
        assert_eq!(recovered_x, first_snow);

        let recovered_y = awe.recover(&first_snow, &skipped_heartbeat);
        assert_eq!(recovered_y, first_kiss);

        // Free energy should be ~0.5 for random containers
        let fe = awe.free_energy();
        assert!(fe > 0.3 && fe < 0.7, "free energy={}", fe);
    }

    #[test]
    fn test_awe_resonance() {
        let first_snow = Container::random(1001);
        let first_kiss = Container::random(1002);
        let skipped_heartbeat = Container::random(1003);

        let awe = AweTriple::new(
            &first_snow, PackedDn::new(&[0, 0]),
            &first_kiss, PackedDn::new(&[0, 1]),
            &skipped_heartbeat, PackedDn::new(&[0, 2]),
        );

        // Query that IS first_snow should resonate strongly with X
        let hdr = awe.resonate(&first_snow);
        assert!(hdr.x > 0.99, "self-resonance should be ~1.0: x={}", hdr.x);
        assert!(hdr.y < 0.7, "cross-resonance should be ~0.5: y={}", hdr.y);
        assert!(hdr.z < 0.7, "cross-resonance should be ~0.5: z={}", hdr.z);

        // Should be a split signal
        assert!(hdr.is_split(0.9, 0.6), "self-query against awe should be split");
    }

    #[test]
    fn test_awe_from_graph() {
        let mut graph = build_test_tree();

        let dn_x = PackedDn::new(&[0, 0]);
        let dn_y = PackedDn::new(&[0, 1]);
        let dn_z = PackedDn::new(&[0, 2]);

        let awe = AweTriple::from_graph(&graph, dn_x, dn_y, dn_z);
        assert!(awe.is_some(), "should find all 3 nodes in graph");

        let awe = awe.unwrap();

        // Write edges and verify
        awe.write_edges(&mut graph);

        // Each source node should have at least 2 awe edges
        for dn in &awe.sources {
            let edges = graph.outgoing(dn);
            let awe_edges: Vec<_> = edges.iter()
                .filter(|(v, _)| *v == VERB_AWE)
                .collect();
            assert_eq!(awe_edges.len(), 2,
                "each node should have 2 awe edges, got {}", awe_edges.len());
        }
    }

    #[test]
    fn test_free_energy_landscape() {
        let graph = build_test_tree();
        let landscape = free_energy_landscape(&graph, PackedDn::ROOT);

        // Should have entries for all non-root nodes that have siblings
        assert!(!landscape.is_empty(), "landscape should have entries");

        // All surprises should be in [0, 1]
        for (_, surprise) in &landscape {
            assert!(*surprise >= 0.0 && *surprise <= 1.0,
                "surprise out of range: {}", surprise);
        }

        // Should be sorted by surprise descending
        for i in 1..landscape.len() {
            assert!(landscape[i - 1].1 >= landscape[i].1,
                "landscape should be sorted by surprise");
        }
    }

    #[test]
    fn test_node_free_energy() {
        let graph = build_test_tree();

        // /0 has siblings /1 and /2
        let fe = node_free_energy(&graph, PackedDn::new(&[0]));
        assert!(fe.is_some());
        let fe = fe.unwrap();
        assert!(fe >= 0.0 && fe <= 1.0, "free energy out of range: {}", fe);

        // Root has no parent → None
        let fe_root = node_free_energy(&graph, PackedDn::ROOT);
        assert!(fe_root.is_none(), "root should have no free energy");
    }

    #[test]
    fn test_felt_path_free_energy_gradient() {
        let graph = build_test_tree();
        let query = Container::random(42);
        let target = PackedDn::new(&[0, 1]);

        let path = felt_walk(&graph, target, &query);

        // Gradient should be a valid number
        let grad = path.free_energy_gradient();
        assert!(grad.is_finite(), "gradient should be finite: {}", grad);
    }

    #[test]
    fn test_felt_path_context_resonance() {
        let graph = build_test_tree();
        let query = Container::random(42);
        let target = PackedDn::new(&[1, 0]);

        let path = felt_walk(&graph, target, &query);

        // Context resonance should be in [0, 1]
        let cr = path.context_resonance(&query);
        assert!(cr >= 0.0 && cr <= 1.0, "context resonance out of range: {}", cr);
    }
}
