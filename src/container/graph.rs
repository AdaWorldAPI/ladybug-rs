//! Container-native BlasGraph: DN-keyed graph over Container 0 inline edges.
//!
//! Replaces holograph's `DnNodeStore + DnCsr` with a single HashMap of CogRecords.
//! The graph IS the metadata — no separate adjacency structure.
//!
//! ```text
//! DnNodeStore (HashMap<PackedDn, NodeSlot>)  → records HashMap<PackedDn, CogRecord>
//!   ↓ fingerprint                             → record.content[0]
//!   ↓ label, rung, NARS                       → record.meta (Container 0)
//! DnCsr (row_ptrs, col_dns, edges)            → Container 0 words 16-31 + 96-111
//!   ↓ outgoing(src)                           → AdjacencyView::inline().iter()
//! DeltaDnMatrix (main + δ+ + δ-)             → SpineCache dirty bitmap
//! ```
//!
//! One Redis GET returns both the vector AND its edges.

use std::collections::HashMap;
use super::Container;
use super::record::CogRecord;
use super::meta::MetaViewMut;
use super::adjacency::{
    PackedDn, InlineEdge, InlineEdgeView, InlineEdgeViewMut,
    AdjacencyView, CsrOverflowView, EdgeDescriptor, CsrOverflowViewMut,
};
use super::search::belichtungsmesser;

// ============================================================================
// CONTAINER GRAPH
// ============================================================================

/// A DN-keyed graph where every node is a CogRecord.
///
/// The graph structure (edges) lives inside each record's Container 0 metadata.
/// No separate adjacency matrix or edge table.
pub struct ContainerGraph {
    /// DN → CogRecord mapping. One GET per node returns fingerprint + edges + metadata.
    records: HashMap<PackedDn, CogRecord>,

    /// DN → children index (maintained on insert for O(1) child enumeration).
    children: HashMap<PackedDn, Vec<PackedDn>>,
}

impl ContainerGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
            children: HashMap::new(),
        }
    }

    /// Number of nodes.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.records.len()
    }

    /// Insert a node. Sets the DN address in Container 0 metadata.
    pub fn insert(&mut self, dn: PackedDn, mut record: CogRecord) {
        // Stamp the DN into Container 0
        {
            let mut meta = MetaViewMut::new(&mut record.meta.words);
            meta.set_dn_addr(dn.raw());
        }

        // Maintain children index
        if let Some(parent) = dn.parent() {
            self.children.entry(parent).or_default().push(dn);
        }

        self.records.insert(dn, record);
    }

    /// Get a node's record.
    #[inline]
    pub fn get(&self, dn: &PackedDn) -> Option<&CogRecord> {
        self.records.get(dn)
    }

    /// Get a mutable reference to a node's record.
    #[inline]
    pub fn get_mut(&mut self, dn: &PackedDn) -> Option<&mut CogRecord> {
        self.records.get_mut(dn)
    }

    /// Check if a node exists.
    #[inline]
    pub fn contains(&self, dn: &PackedDn) -> bool {
        self.records.contains_key(dn)
    }

    /// All DN addresses in the graph.
    pub fn dns(&self) -> impl Iterator<Item = &PackedDn> {
        self.records.keys()
    }

    /// All (dn, record) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&PackedDn, &CogRecord)> {
        self.records.iter()
    }

    /// Direct children of a DN (from the children index).
    pub fn children_of(&self, dn: &PackedDn) -> &[PackedDn] {
        self.children.get(dn).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Walk from a DN to root, collecting all ancestors.
    pub fn walk_to_root(&self, dn: PackedDn) -> Vec<PackedDn> {
        let mut path = Vec::new();
        let mut current = dn;
        while let Some(parent) = current.parent() {
            if self.records.contains_key(&parent) {
                path.push(parent);
            }
            current = parent;
        }
        path
    }

    // ========================================================================
    // EDGE OPERATIONS: Read and write edges via Container 0
    // ========================================================================

    /// Get the adjacency view for a node.
    pub fn adjacency(&self, dn: &PackedDn) -> Option<AdjacencyView<'_>> {
        self.records.get(dn).map(|r| AdjacencyView::new(&r.meta.words))
    }

    /// Get inline edges of a node.
    pub fn inline_edges(&self, dn: &PackedDn) -> Option<InlineEdgeView<'_>> {
        self.records.get(dn).map(|r| InlineEdgeView::new(&r.meta.words))
    }

    /// Get CSR overflow edges of a node.
    pub fn overflow_edges(&self, dn: &PackedDn) -> Option<CsrOverflowView<'_>> {
        self.records.get(dn).map(|r| CsrOverflowView::new(&r.meta.words))
    }

    /// Add an inline edge from `src` to `dst` with a verb.
    /// Returns the slot index, or None if `src` doesn't exist or is full.
    pub fn add_edge(&mut self, src: &PackedDn, verb: u8, target_hint: u8) -> Option<usize> {
        let record = self.records.get_mut(src)?;
        let mut view = InlineEdgeViewMut::new(&mut record.meta.words);
        let slot = view.add(InlineEdge { verb, target_hint })?;

        // Update out-degree in graph metrics
        let edge_count = view.count() as u32;
        let mut meta = MetaViewMut::new(&mut record.meta.words);
        meta.set_out_degree(edge_count);

        Some(slot)
    }

    /// Remove an inline edge.
    pub fn remove_edge(&mut self, src: &PackedDn, verb: u8, target_hint: u8) -> bool {
        if let Some(record) = self.records.get_mut(src) {
            let mut view = InlineEdgeViewMut::new(&mut record.meta.words);
            let removed = view.remove(verb, target_hint);

            if removed {
                let edge_count = view.count() as u32;
                let mut meta = MetaViewMut::new(&mut record.meta.words);
                meta.set_out_degree(edge_count);
            }

            removed
        } else {
            false
        }
    }

    /// Add an edge to the CSR overflow region (for high-degree nodes).
    pub fn add_overflow_edge(
        &mut self,
        src: &PackedDn,
        edge: EdgeDescriptor,
    ) -> Option<usize> {
        let record = self.records.get_mut(src)?;
        let mut view = CsrOverflowViewMut::new(&mut record.meta.words);
        view.push_edge(edge)
    }

    // ========================================================================
    // FINGERPRINT OPERATIONS: Content containers
    // ========================================================================

    /// Get the primary fingerprint (content container 0) of a node.
    pub fn fingerprint(&self, dn: &PackedDn) -> Option<&Container> {
        self.records.get(dn)
            .and_then(|r| r.content.first())
    }

    /// Hamming distance between two nodes' fingerprints.
    pub fn hamming(&self, a: &PackedDn, b: &PackedDn) -> Option<u32> {
        let fa = self.fingerprint(a)?;
        let fb = self.fingerprint(b)?;
        Some(fa.hamming(fb))
    }

    /// Belichtungsmesser estimate between two nodes' fingerprints.
    pub fn quick_distance(&self, a: &PackedDn, b: &PackedDn) -> Option<u32> {
        let fa = self.fingerprint(a)?;
        let fb = self.fingerprint(b)?;
        Some(belichtungsmesser(fa, fb))
    }

    // ========================================================================
    // GRAPH QUERIES
    // ========================================================================

    /// All outgoing neighbors of a node (from inline edges).
    /// Returns Vec of (verb, target_hint) pairs.
    pub fn outgoing(&self, src: &PackedDn) -> Vec<(u8, u8)> {
        match self.inline_edges(src) {
            Some(view) => view.iter().map(|(_, e)| (e.verb, e.target_hint)).collect(),
            None => Vec::new(),
        }
    }

    /// Degree of a node (inline + overflow).
    pub fn degree(&self, dn: &PackedDn) -> usize {
        match self.adjacency(dn) {
            Some(view) => view.total_edges(),
            None => 0,
        }
    }

    /// Find the k nearest nodes to a query fingerprint using Belichtungsmesser.
    /// Returns Vec of (dn, estimated_distance) sorted by distance.
    pub fn nearest_k(&self, query: &Container, k: usize) -> Vec<(PackedDn, u32)> {
        let mut results: Vec<(PackedDn, u32)> = self.records.iter()
            .filter_map(|(dn, record)| {
                record.content.first()
                    .map(|fp| (*dn, belichtungsmesser(fp, query)))
            })
            .collect();

        results.sort_by_key(|(_, d)| *d);
        results.truncate(k);
        results
    }

    /// MGET pattern: get multiple records at once (simulates Redis MGET).
    pub fn mget(&self, dns: &[PackedDn]) -> Vec<Option<&CogRecord>> {
        dns.iter().map(|dn| self.records.get(dn)).collect()
    }

    /// Subtree: all descendants of a DN.
    pub fn subtree(&self, root: &PackedDn) -> Vec<PackedDn> {
        let mut result = Vec::new();
        let mut stack = vec![*root];
        while let Some(dn) = stack.pop() {
            if dn != *root {
                result.push(dn);
            }
            if let Some(children) = self.children.get(&dn) {
                stack.extend(children);
            }
        }
        result
    }
}
