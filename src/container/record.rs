//! CogRecord: fixed 2 KB record — metadata container + content container.
//!
//! Re-exports the canonical CogRecord from ladybug_contract and adds
//! database-layer operations (search_chunks, delta encoding, etc.)
//! that operate on linked record slices for multi-container geometries.

use super::Container;
#[allow(unused_imports)]
use super::geometry::ContainerGeometry;
#[allow(unused_imports)]
use super::meta::{MetaView, MetaViewMut, W_REPR_BASE};

// Re-export the canonical CogRecord from contract
pub use ladybug_contract::record::CogRecord;

// ============================================================================
// LINKED RECORD OPERATIONS (database layer)
// ============================================================================
// These extend CogRecord with operations on linked record slices,
// which are database-level concerns (not substrate contract types).

/// Recompute summary from multiple linked records (bundle their contents).
pub fn recompute_summary_linked(records: &[&CogRecord]) -> Container {
    if records.is_empty() {
        return Container::zero();
    }
    let refs: Vec<&Container> = records.iter().map(|r| &r.content).collect();
    Container::bundle(&refs)
}

/// Hierarchical search across linked records: summary first, then individual.
pub fn search_linked_records(
    records: &[&CogRecord],
    query: &Container,
    threshold: u32,
) -> Vec<(usize, u32)> {
    let mut hits = Vec::new();
    for (i, record) in records.iter().enumerate() {
        let dist = record.content.hamming(query);
        if dist <= threshold {
            hits.push((i, dist));
        }
    }
    hits.sort_by_key(|&(_, d)| d);
    hits
}

/// Delta-encode: XOR difference between two linked records' content.
pub fn delta_encode_linked(a: &CogRecord, b: &CogRecord) -> (Container, u32) {
    let delta = a.content.xor(&b.content);
    let info = delta.popcount();
    (delta, info)
}

/// Recover content from base + delta. XOR is its own inverse.
pub fn delta_decode(base: &Container, delta: &Container) -> Container {
    base.xor(delta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cogrecord_local_re_export() {
        // Verify the re-exported CogRecord is the contract type
        let r = CogRecord::new(ContainerGeometry::Cam);
        assert_eq!(r.container_count(), 2);
        assert_eq!(std::mem::size_of::<CogRecord>(), 2048);
    }

    #[test]
    fn test_linked_search() {
        let mut r0 = CogRecord::new(ContainerGeometry::Cam);
        let mut r1 = CogRecord::new(ContainerGeometry::Cam);
        r0.content = Container::random(1);
        r1.content = Container::random(2);

        let query = Container::random(1); // should match r0 exactly
        let hits = search_linked_records(&[&r0, &r1], &query, 100);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].0, 0); // r0 is closest
        assert_eq!(hits[0].1, 0); // exact match
    }

    #[test]
    fn test_delta_encode_linked() {
        let mut r0 = CogRecord::new(ContainerGeometry::Cam);
        let mut r1 = CogRecord::new(ContainerGeometry::Cam);
        r0.content = Container::random(10);
        r1.content = Container::random(20);

        let (delta, _) = delta_encode_linked(&r0, &r1);
        let recovered = delta_decode(&r0.content, &delta);
        assert_eq!(recovered, r1.content);
    }
}
