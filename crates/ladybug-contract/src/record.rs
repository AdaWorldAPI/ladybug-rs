//! CogRecord — metadata container + N content containers.

use std::ops::Range;

use crate::container::Container;
use crate::geometry::ContainerGeometry;
use crate::meta::{MetaView, MetaViewMut, W_REPR_BASE};

/// A cognitive record: one metadata container + N content containers.
///
/// The metadata (Container 0) stores identity, NARS truth values,
/// graph edges, RL Q-values, qualia channels, etc.
/// The content containers hold the actual fingerprint data,
/// interpreted according to the record's [`ContainerGeometry`].
#[derive(Clone, Debug)]
pub struct CogRecord {
    /// Container 0: always metadata.
    pub meta: Container,
    /// Containers 1..N: content (geometry determines interpretation).
    pub content: Vec<Container>,
}

impl CogRecord {
    /// Create a new record with the given geometry and default content containers.
    pub fn new(geometry: ContainerGeometry) -> Self {
        let n = geometry.default_content_count();
        let mut meta = Container::zero();

        {
            let mut view = MetaViewMut::new(&mut meta.words);
            view.init(geometry, n as u8);
        }

        Self {
            meta,
            content: vec![Container::zero(); n],
        }
    }

    /// Total containers including meta.
    #[inline]
    pub fn container_count(&self) -> u8 {
        (1 + self.content.len()).min(255) as u8
    }

    /// Geometry, read from metadata word 1.
    pub fn geometry(&self) -> ContainerGeometry {
        MetaView::new(&self.meta.words).geometry()
    }

    /// Read-only metadata view.
    #[inline]
    pub fn meta_view(&self) -> MetaView<'_> {
        MetaView::new(&self.meta.words)
    }

    /// Mutable metadata view.
    #[inline]
    pub fn meta_view_mut(&mut self) -> MetaViewMut<'_> {
        MetaViewMut::new(&mut self.meta.words)
    }

    /// Bounds-checked access to a content container.
    #[inline]
    pub fn content_container(&self, idx: usize) -> Option<&Container> {
        self.content.get(idx)
    }

    /// The summary container (always content[0], the searchable fingerprint).
    #[inline]
    pub fn summary(&self) -> &Container {
        &self.content[0]
    }

    /// Hamming distance against the summary container only.
    #[inline]
    pub fn hamming_to(&self, query: &Container) -> u32 {
        self.content[0].hamming(query)
    }

    /// XOR-fold of specified content containers (spine computation).
    pub fn spine(containers: &[&Container]) -> Container {
        let mut result = Container::zero();
        for c in containers {
            result = result.xor(c);
        }
        result
    }

    /// Semantic difference: content[a] ⊕ content[b].
    pub fn delta(&self, a: usize, b: usize) -> Option<Container> {
        let ca = self.content.get(a)?;
        let cb = self.content.get(b)?;
        Some(ca.xor(cb))
    }

    // ========================================================================
    // XYZ HOLOGRAPHIC OPERATIONS
    // ========================================================================

    /// For Xyz geometry: store a trace by XOR-folding X, Y, Z.
    pub fn xyz_trace(&self) -> Option<Container> {
        if self.content.len() < 3 {
            return None;
        }
        Some(
            self.content[0]
                .xor(&self.content[1])
                .xor(&self.content[2]),
        )
    }

    /// Probe: given 2 of 3 dimensions + trace, recover the third.
    pub fn xyz_probe(known: &[&Container; 2], trace: &Container) -> Container {
        trace.xor(known[0]).xor(known[1])
    }

    // ========================================================================
    // TREE GEOMETRY OPERATIONS
    // ========================================================================

    /// Branching factor (stored in meta word 80).
    pub fn branching_factor(&self) -> usize {
        let k = (self.meta.words[W_REPR_BASE] & 0xFF) as usize;
        if k == 0 {
            2
        } else {
            k
        } // default to binary tree
    }

    /// Children of node at container index `i`.
    pub fn tree_children(&self, i: usize) -> Range<usize> {
        let k = self.branching_factor();
        let start = k * i + 1;
        let end = (k * (i + 1) + 1).min(self.content.len());
        start..end
    }

    /// Parent of node at container index `i`.
    pub fn tree_parent(&self, i: usize) -> Option<usize> {
        if i == 0 {
            None
        } else {
            Some((i - 1) / self.branching_factor())
        }
    }

    /// Spine of a subtree (XOR of all direct children).
    pub fn subtree_spine(&self, root_idx: usize) -> Container {
        let children = self.tree_children(root_idx);
        let mut spine = Container::zero();
        for child_idx in children {
            if let Some(child) = self.content.get(child_idx) {
                spine = spine.xor(child);
            }
        }
        spine
    }

    /// Cross-hydrate: project a node to a different context.
    pub fn cross_hydrate(source: &Container, context_delta: &Container) -> Container {
        source.xor(context_delta)
    }

    /// Extract the perspective difference between two branches.
    pub fn extract_perspective(branch_a: &Container, branch_b: &Container) -> Container {
        branch_a.xor(branch_b)
    }

    // ========================================================================
    // CHUNKED GEOMETRY OPERATIONS
    // ========================================================================

    /// Recompute summary (content[0]) from all chunks (content[1..]).
    pub fn recompute_summary(&mut self) {
        if self.content.len() <= 1 {
            return;
        }
        let chunk_refs: Vec<&Container> = self.content[1..].iter().collect();
        self.content[0] = Container::bundle(&chunk_refs);
    }

    /// Append a new chunk and update summary.
    pub fn append_chunk(&mut self, chunk: Container) {
        self.content.push(chunk);
        self.recompute_summary();
        let count = self.container_count();
        MetaViewMut::new(&mut self.meta.words).set_container_count(count);
    }

    /// Hierarchical search: summary first, then individual chunks.
    pub fn search_chunks(&self, query: &Container, threshold: u32) -> Vec<(usize, u32)> {
        if self.content.is_empty() {
            return vec![];
        }

        // Level 0: quick check on summary
        let summary_dist = self.content[0].hamming(query);
        if summary_dist > threshold * 2 {
            return vec![];
        }

        // Level 1: scan individual chunks
        let mut hits = Vec::new();
        for (i, chunk) in self.content[1..].iter().enumerate() {
            let dist = chunk.hamming(query);
            if dist <= threshold {
                hits.push((i + 1, dist)); // +1 because content[0] is summary
            }
        }
        hits.sort_by_key(|&(_, d)| d);
        hits
    }

    // ========================================================================
    // DELTA ENCODING
    // ========================================================================

    /// Delta-encode content[b] as XOR difference from content[a].
    pub fn delta_encode(&self, a: usize, b: usize) -> Option<(Container, u32)> {
        let ca = self.content.get(a)?;
        let cb = self.content.get(b)?;
        let delta = ca.xor(cb);
        let info = delta.popcount();
        Some((delta, info))
    }

    /// Recover container from base + delta. XOR is its own inverse.
    pub fn delta_decode(base: &Container, delta: &Container) -> Container {
        base.xor(delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cogrecord_meta_roundtrip() {
        let mut r = CogRecord::new(ContainerGeometry::Cam);
        {
            let mut m = r.meta_view_mut();
            m.set_dn_addr(0xDEADBEEF);
            m.set_nars_frequency(0.85);
            m.set_nars_confidence(0.72);
            m.set_rung_level(4);
            m.set_gate_state(0);
        }
        let m = r.meta_view();
        assert_eq!(m.dn_addr(), 0xDEADBEEF);
        assert!((m.nars_frequency() - 0.85).abs() < 1e-6);
        assert!((m.nars_confidence() - 0.72).abs() < 1e-6);
        assert_eq!(m.rung_level(), 4);
        assert_eq!(m.gate_state(), 0);
    }

    #[test]
    fn test_cogrecord_xyz_roundtrip() {
        let mut r = CogRecord::new(ContainerGeometry::Xyz);
        r.content[0] = Container::random(1);
        r.content[1] = Container::random(2);
        r.content[2] = Container::random(3);

        let trace = r.xyz_trace().unwrap();
        let recovered =
            CogRecord::xyz_probe(&[&r.content[0], &r.content[1]], &trace);
        assert_eq!(recovered, r.content[2]);
    }

    #[test]
    fn test_layer_marker_roundtrip() {
        let mut r = CogRecord::new(ContainerGeometry::Cam);
        {
            // Direct word manipulation for layer marker (no set_layer_marker in contract)
            // Layer 0 starts at byte offset 0 within W12-W15
            let byte_offset = 0;
            let word_idx = crate::meta::W_LAYER_BASE + byte_offset / 8;
            let packed: u64 = 255 | (200 << 8) | (0x0001 << 16) | (42u64 << 32);
            r.meta.words[word_idx] = packed;
        }
        let (strength, frequency, recency, flags) = r.meta_view().layer_marker(0);
        assert_eq!(strength, 255);
        assert_eq!(frequency, 200);
        assert_eq!(recency, 0x0001);
        assert_eq!(flags, 42);
    }
}
