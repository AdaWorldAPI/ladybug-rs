//! CogRecord — fixed 2 KB record: metadata container + content container.
//!
//! The **holy grail** layout: every record is exactly `[Container; 2]` = 2 KB.
//! Container 0 is always metadata. Container 1 is always content.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │  meta    (1 KB)  identity, NARS truth, edges, rung, RL     │
//! ├─────────────────────────────────────────────────────────────┤
//! │  content (1 KB)  searchable fingerprint (Hamming / SIMD)   │
//! └─────────────────────────────────────────────────────────────┘
//!       = 2 KB = 1 Fingerprint = 1 DN tree node = 1 Redis value
//! ```
//!
//! For multi-container geometries (Xyz, Chunked, Tree), records are
//! linked through the DN tree. Each linked record is still 2 KB.

use std::ops::Range;

use crate::container::Container;
use crate::geometry::ContainerGeometry;
use crate::meta::{MetaView, MetaViewMut, W_REPR_BASE};

/// Fixed-size cognitive record: 8,192-bit metadata + 8,192-bit content = 2 KB.
///
/// This is the atomic unit of storage, search, and transfer:
/// - DN tree value = 1 CogRecord = 2 KB
/// - Redis value = 1 CogRecord = 2 KB
/// - Fingerprint = 1 CogRecord (reinterpretable)
/// - SIMD scan unit = 2 × 16 AVX-512 iterations = 32 iterations
///
/// # Stack Allocated
///
/// No heap allocation. `[Container; 2]` lives entirely on the stack.
/// This enables zero-copy mmap, SIMD-aligned access, and fixed-size I/O.
#[derive(Clone, Debug)]
#[repr(C, align(64))]
pub struct CogRecord {
    /// Container 0: always metadata (identity, NARS, edges, RL, qualia).
    pub meta: Container,
    /// Container 1: always content (searchable fingerprint).
    pub content: Container,
}

impl CogRecord {
    /// Byte size of a CogRecord (2 × 1 KB = 2048 bytes).
    pub const SIZE: usize = 2 * crate::container::CONTAINER_BYTES;

    /// Create a new record with the given geometry.
    pub fn new(geometry: ContainerGeometry) -> Self {
        let mut meta = Container::zero();
        {
            let mut view = MetaViewMut::new(&mut meta.words);
            view.init(geometry, 1); // always 1 content container per record
        }
        Self {
            meta,
            content: Container::zero(),
        }
    }

    /// Total containers (always 2: meta + content).
    #[inline]
    pub fn container_count(&self) -> u8 {
        2
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

    /// The content container (the searchable fingerprint).
    #[inline]
    pub fn summary(&self) -> &Container {
        &self.content
    }

    /// Hamming distance against the content container.
    #[inline]
    pub fn hamming_to(&self, query: &Container) -> u32 {
        self.content.hamming(query)
    }

    /// Semantic difference between this record's content and another's.
    pub fn delta(&self, other: &CogRecord) -> Container {
        self.content.xor(&other.content)
    }

    /// XOR-fold of multiple content containers (spine computation).
    pub fn spine(records: &[&CogRecord]) -> Container {
        let mut result = Container::zero();
        for r in records {
            result = result.xor(&r.content);
        }
        result
    }

    /// Zero-copy byte view of the entire 2 KB record.
    #[inline]
    pub fn as_bytes(&self) -> &[u8; Self::SIZE] {
        // SAFETY: CogRecord is #[repr(C, align(64))] with two Containers.
        // Total size = 2 × 1024 = 2048 bytes.
        unsafe { &*(self as *const CogRecord as *const [u8; Self::SIZE]) }
    }

    /// Construct from a 2 KB byte slice.
    pub fn from_bytes(bytes: &[u8; Self::SIZE]) -> Self {
        let meta_bytes: &[u8; 1024] = bytes[..1024].try_into().unwrap();
        let content_bytes: &[u8; 1024] = bytes[1024..].try_into().unwrap();
        Self {
            meta: Container::from_bytes(meta_bytes),
            content: Container::from_bytes(content_bytes),
        }
    }

    /// Cross-hydrate: project content to a different context.
    pub fn cross_hydrate(source: &Container, context_delta: &Container) -> Container {
        source.xor(context_delta)
    }

    /// Extract the perspective difference between two contexts.
    pub fn extract_perspective(branch_a: &Container, branch_b: &Container) -> Container {
        branch_a.xor(branch_b)
    }

    // ========================================================================
    // LINKED RECORD OPERATIONS (for multi-container geometries)
    // ========================================================================
    // When a concept needs more than 1 content container (Xyz, Chunked, Tree),
    // the additional containers live in separate CogRecords linked via DN tree.
    // These methods operate on slices of linked records.

    /// For Xyz geometry (3 linked records): XOR-fold of X, Y, Z content.
    pub fn xyz_trace(records: &[&CogRecord; 3]) -> Container {
        records[0]
            .content
            .xor(&records[1].content)
            .xor(&records[2].content)
    }

    /// Probe: given 2 of 3 dimensions + trace, recover the third.
    pub fn xyz_probe(known: &[&Container; 2], trace: &Container) -> Container {
        trace.xor(known[0]).xor(known[1])
    }

    /// Branching factor for tree geometry (stored in meta word 80).
    pub fn branching_factor(&self) -> usize {
        let k = (self.meta.words[W_REPR_BASE] & 0xFF) as usize;
        if k == 0 { 2 } else { k }
    }

    /// Children indices of node at index `i` in a linked record tree.
    pub fn tree_children(&self, i: usize, total_records: usize) -> Range<usize> {
        let k = self.branching_factor();
        let start = k * i + 1;
        let end = (k * (i + 1) + 1).min(total_records);
        start..end
    }

    /// Parent index of node at index `i`.
    pub fn tree_parent(&self, i: usize) -> Option<usize> {
        if i == 0 { None } else { Some((i - 1) / self.branching_factor()) }
    }
}

impl Default for CogRecord {
    fn default() -> Self {
        Self::new(ContainerGeometry::Cam)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cogrecord_size() {
        assert_eq!(CogRecord::SIZE, 2048);
        assert_eq!(std::mem::size_of::<CogRecord>(), 2048);
    }

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
    fn test_cogrecord_bytes_roundtrip() {
        let mut r = CogRecord::new(ContainerGeometry::Cam);
        r.content = Container::random(42);
        r.meta_view_mut().set_dn_addr(0x1234);

        let bytes = r.as_bytes();
        let r2 = CogRecord::from_bytes(bytes);
        assert_eq!(r.meta, r2.meta);
        assert_eq!(r.content, r2.content);
    }

    #[test]
    fn test_cogrecord_xyz_linked() {
        let mut r0 = CogRecord::new(ContainerGeometry::Xyz);
        let mut r1 = CogRecord::new(ContainerGeometry::Xyz);
        let mut r2 = CogRecord::new(ContainerGeometry::Xyz);
        r0.content = Container::random(1); // X
        r1.content = Container::random(2); // Y
        r2.content = Container::random(3); // Z

        let trace = CogRecord::xyz_trace(&[&r0, &r1, &r2]);
        let recovered = CogRecord::xyz_probe(&[&r0.content, &r1.content], &trace);
        assert_eq!(recovered, r2.content);
    }

    #[test]
    fn test_cogrecord_spine() {
        let mut r0 = CogRecord::new(ContainerGeometry::Cam);
        let mut r1 = CogRecord::new(ContainerGeometry::Cam);
        r0.content = Container::random(10);
        r1.content = Container::random(20);

        let s = CogRecord::spine(&[&r0, &r1]);
        assert_eq!(s, r0.content.xor(&r1.content));
    }
}
