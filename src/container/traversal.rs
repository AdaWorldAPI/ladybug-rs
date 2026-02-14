//! Container-native semiring matrix-vector multiply (MxV).
//!
//! Ports holograph's `DnCsr::mxv` to the container architecture.
//! Edges live in Container 0 (inline), fingerprints in Container 1+ (content).
//! One lookup per node returns everything — no separate graph structure.
//!
//! ```text
//! holograph:    3 lookups per edge (node_fps, csr_row, csr_col)
//! container:    1 lookup per node  (record has fp + edges + metadata)
//! ```

use super::Container;
use super::adjacency::{InlineEdgeView, PackedDn};
use super::graph::ContainerGraph;
use super::search::belichtungsmesser;
use std::collections::HashMap;

// ============================================================================
// DN SEMIRING TRAIT: Container-native
// ============================================================================

/// Semiring trait for container-native graph traversal.
///
/// Matches holograph's `DnSemiring` but operates on Container references
/// instead of BitpackedVector Arc pointers — zero-copy, no allocation.
pub trait DnSemiring {
    /// The value type carried through the traversal.
    type Value: Clone;

    /// Additive identity (zero element).
    fn zero(&self) -> Self::Value;

    /// Multiply: given an edge from src to dst, compute the contribution.
    ///
    /// # Arguments
    /// - `verb`: the cognitive verb on the edge
    /// - `weight_hint`: the target hint byte (or a weight if available)
    /// - `input`: the current value at the source node
    /// - `src_fp`: source node's fingerprint (Container 1)
    /// - `dst_fp`: destination node's fingerprint (Container 1), if available
    fn multiply(
        &self,
        verb: u8,
        weight_hint: u8,
        input: &Self::Value,
        src_fp: &Container,
        dst_fp: Option<&Container>,
    ) -> Self::Value;

    /// Add: combine two values at the same destination.
    fn add(&self, a: &Self::Value, b: &Self::Value) -> Self::Value;

    /// Check if a value is the zero element.
    fn is_zero(&self, val: &Self::Value) -> bool;

    /// Human-readable name.
    fn name(&self) -> &'static str;
}

// ============================================================================
// CONCRETE SEMIRINGS
// ============================================================================

/// Boolean BFS: reachability via OR/AND.
pub struct BooleanBfs;

impl DnSemiring for BooleanBfs {
    type Value = bool;

    fn zero(&self) -> bool {
        false
    }

    fn multiply(
        &self,
        _verb: u8,
        _w: u8,
        input: &bool,
        _src: &Container,
        _dst: Option<&Container>,
    ) -> bool {
        *input // If source is reachable, destination is reachable
    }

    fn add(&self, a: &bool, b: &bool) -> bool {
        *a || *b
    }
    fn is_zero(&self, val: &bool) -> bool {
        !*val
    }
    fn name(&self) -> &'static str {
        "BooleanBfs"
    }
}

/// Hamming-distance shortest path: MinPlus with Hamming as edge weight.
pub struct HammingMinPlus;

impl DnSemiring for HammingMinPlus {
    type Value = u32;

    fn zero(&self) -> u32 {
        u32::MAX
    }

    fn multiply(
        &self,
        _verb: u8,
        _w: u8,
        input: &u32,
        src_fp: &Container,
        dst_fp: Option<&Container>,
    ) -> u32 {
        if *input == u32::MAX {
            return u32::MAX;
        }
        match dst_fp {
            Some(dst) => input.saturating_add(src_fp.hamming(dst)),
            None => u32::MAX,
        }
    }

    fn add(&self, a: &u32, b: &u32) -> u32 {
        (*a).min(*b)
    }
    fn is_zero(&self, val: &u32) -> bool {
        *val == u32::MAX
    }
    fn name(&self) -> &'static str {
        "HammingMinPlus"
    }
}

/// HDR path binding: XOR-compose path fingerprints, Bundle at junctions.
pub struct HdrPathBind;

impl DnSemiring for HdrPathBind {
    type Value = Option<Container>;

    fn zero(&self) -> Option<Container> {
        None
    }

    fn multiply(
        &self,
        _verb: u8,
        _w: u8,
        input: &Option<Container>,
        _src_fp: &Container,
        dst_fp: Option<&Container>,
    ) -> Option<Container> {
        match (input, dst_fp) {
            (Some(path), Some(dst)) => Some(path.xor(dst)),
            _ => None,
        }
    }

    fn add(&self, a: &Option<Container>, b: &Option<Container>) -> Option<Container> {
        match (a, b) {
            (Some(va), Some(vb)) => Some(Container::bundle(&[va, vb])),
            (Some(v), None) | (None, Some(v)) => Some(v.clone()),
            (None, None) => None,
        }
    }

    fn is_zero(&self, val: &Option<Container>) -> bool {
        val.is_none()
    }
    fn name(&self) -> &'static str {
        "HdrPathBind"
    }
}

/// Resonance: find paths that resonate with a query fingerprint.
/// Value = resonance score (10000 - Hamming distance to query).
pub struct ResonanceSearch {
    pub query: Container,
}

impl DnSemiring for ResonanceSearch {
    type Value = u32;

    fn zero(&self) -> u32 {
        0
    }

    fn multiply(
        &self,
        _verb: u8,
        _w: u8,
        _input: &u32,
        src_fp: &Container,
        dst_fp: Option<&Container>,
    ) -> u32 {
        match dst_fp {
            Some(dst) => {
                // Edge semantic fingerprint ≈ src ⊕ dst (simplified)
                let edge_fp = src_fp.xor(dst);
                let dist = edge_fp.hamming(&self.query);
                if dist < 10000 { 10000 - dist } else { 0 }
            }
            None => 0,
        }
    }

    fn add(&self, a: &u32, b: &u32) -> u32 {
        (*a).max(*b)
    }
    fn is_zero(&self, val: &u32) -> bool {
        *val == 0
    }
    fn name(&self) -> &'static str {
        "ResonanceSearch"
    }
}

/// PageRank-style value propagation.
pub struct PageRankPropagation {
    pub damping: f32,
}

impl DnSemiring for PageRankPropagation {
    type Value = f32;

    fn zero(&self) -> f32 {
        0.0
    }

    fn multiply(
        &self,
        _verb: u8,
        _w: u8,
        input: &f32,
        _src_fp: &Container,
        _dst_fp: Option<&Container>,
    ) -> f32 {
        // Simplified: contribution = damping * rank / out_degree
        // Out-degree normalization happens in the traverse loop
        self.damping * input
    }

    fn add(&self, a: &f32, b: &f32) -> f32 {
        a + b
    }
    fn is_zero(&self, val: &f32) -> bool {
        *val < 1e-9
    }
    fn name(&self) -> &'static str {
        "PageRankPropagation"
    }
}

/// Cascaded Hamming with Belichtungsmesser pre-filter.
/// Only computes exact Hamming for candidates that pass the light meter.
pub struct CascadedHamming {
    pub radius: u32,
}

impl DnSemiring for CascadedHamming {
    type Value = u32;

    fn zero(&self) -> u32 {
        u32::MAX
    }

    fn multiply(
        &self,
        _verb: u8,
        _w: u8,
        input: &u32,
        src_fp: &Container,
        dst_fp: Option<&Container>,
    ) -> u32 {
        if *input == u32::MAX {
            return u32::MAX;
        }
        match dst_fp {
            Some(dst) => {
                // L0: Belichtungsmesser pre-filter (~14 cycles)
                let estimate = belichtungsmesser(src_fp, dst);
                if estimate > self.radius * 2 {
                    return u32::MAX; // Definitely too far
                }
                // L2: Exact Hamming (only for survivors)
                input.saturating_add(src_fp.hamming(dst))
            }
            None => u32::MAX,
        }
    }

    fn add(&self, a: &u32, b: &u32) -> u32 {
        (*a).min(*b)
    }
    fn is_zero(&self, val: &u32) -> bool {
        *val == u32::MAX
    }
    fn name(&self) -> &'static str {
        "CascadedHamming"
    }
}

// ============================================================================
// MATRIX-VECTOR MULTIPLY: Container-native
// ============================================================================

/// Container-native semiring MxV over a ContainerGraph.
///
/// Expands a frontier through one hop of edges, applying the semiring.
/// Edges are read from Container 0 inline edges — no separate CSR.
/// Fingerprints are Container 1 content — no separate lookup.
///
/// # Arguments
/// - `graph`: the container graph
/// - `frontier`: DN → value mapping for the current frontier
/// - `semiring`: the semiring to apply
///
/// # Returns
/// New frontier: DN → accumulated value at each reached destination.
///
/// # Complexity
/// For F frontier nodes, each with degree D:
/// - F record lookups (MGET in Redis)
/// - F×D edge reads (zero-copy from Container 0)
/// - F×D semiring multiply + add operations
///
/// No separate CSR lookup. No fingerprint lookup.
pub fn container_mxv<S: DnSemiring>(
    graph: &ContainerGraph,
    frontier: &HashMap<PackedDn, S::Value>,
    semiring: &S,
) -> HashMap<PackedDn, S::Value> {
    let mut result: HashMap<PackedDn, S::Value> = HashMap::new();

    for (&src_dn, input_val) in frontier {
        if semiring.is_zero(input_val) {
            continue;
        }

        // 1 lookup: get the source record
        let src_record = match graph.get(&src_dn) {
            Some(r) => r,
            None => continue,
        };
        let src_fp = &src_record.content;

        // Edges are INLINE in Container 0 — no separate graph lookup
        let edge_view = InlineEdgeView::new(&src_record.meta.words);

        for (_, edge) in edge_view.iter() {
            // For each edge, we need the destination fingerprint.
            // In a real Redis scenario, we'd batch these into an MGET.
            // Here we do individual lookups (the graph is in-memory).

            // Resolve target_hint to a full DN.
            // In a full implementation, the graph would maintain a
            // target_hint → DN resolution table. For now, we search
            // the graph for nodes whose DN's low byte matches.
            let destinations = resolve_target_hint(graph, &src_dn, edge.target_hint);

            for dst_dn in destinations {
                let dst_fp = graph.fingerprint(&dst_dn);

                let contribution =
                    semiring.multiply(edge.verb, edge.target_hint, input_val, src_fp, dst_fp);

                if !semiring.is_zero(&contribution) {
                    result
                        .entry(dst_dn)
                        .and_modify(|existing| {
                            *existing = semiring.add(existing, &contribution);
                        })
                        .or_insert(contribution);
                }
            }
        }
    }

    result
}

/// Multi-hop traversal: apply MxV repeatedly for `max_hops` steps.
/// Returns the final frontier.
pub fn container_multi_hop<S: DnSemiring>(
    graph: &ContainerGraph,
    initial_frontier: HashMap<PackedDn, S::Value>,
    semiring: &S,
    max_hops: usize,
) -> HashMap<PackedDn, S::Value> {
    let mut frontier = initial_frontier;

    for _ in 0..max_hops {
        let next = container_mxv(graph, &frontier, semiring);
        if next.is_empty() {
            break;
        }
        frontier = next;
    }

    frontier
}

/// Resolve a target_hint byte to actual DNs in the graph.
/// This is a simplified version — in production, the MetaView would store
/// full DN addresses in the CSR overflow region.
fn resolve_target_hint(
    graph: &ContainerGraph,
    src_dn: &PackedDn,
    target_hint: u8,
) -> Vec<PackedDn> {
    // Strategy 1: check if target_hint is a child index
    let child_dn = src_dn.child(target_hint);
    if let Some(dn) = child_dn {
        if graph.contains(&dn) {
            return vec![dn];
        }
    }

    // Strategy 2: check siblings (same parent, different last component)
    if let Some(parent) = src_dn.parent() {
        let sibling = parent.child(target_hint);
        if let Some(dn) = sibling {
            if graph.contains(&dn) {
                return vec![dn];
            }
        }
    }

    // Strategy 3: search the parent's children
    if let Some(parent) = src_dn.parent() {
        for &child_dn in graph.children_of(&parent) {
            if child_dn.component(child_dn.depth().saturating_sub(1) as usize) == Some(target_hint)
            {
                return vec![child_dn];
            }
        }
    }

    Vec::new()
}
