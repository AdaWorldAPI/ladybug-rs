//! Unified Spectator — semantic search across all execution history.
//!
//! Wraps BindSpace reads and TemporalStore fork/diff to provide
//! contract-level queries over enriched execution steps.

use std::sync::Arc;

use parking_lot::RwLock;

use crate::storage::bind_space::{Addr, BindSpace, FINGERPRINT_WORDS, TOTAL_ADDRESSES};
use crate::storage::temporal::{TemporalStore, Version, VersionDiff, WhatIfBranch};

/// Spectator over unified execution history stored in BindSpace.
///
/// Steps ingested via Flight or PG poll are fingerprinted and stored
/// in BindSpace. The spectator searches across them by scanning
/// populated addresses and comparing Hamming similarity.
pub struct UnifiedSpectator {
    bind_space: Arc<RwLock<BindSpace>>,
    temporal: Arc<TemporalStore>,
}

impl UnifiedSpectator {
    pub fn new(bind_space: Arc<RwLock<BindSpace>>, temporal: Arc<TemporalStore>) -> Self {
        Self {
            bind_space,
            temporal,
        }
    }

    /// Search across ALL execution history for similar outputs.
    ///
    /// Scans all populated BindSpace addresses, computes Hamming similarity,
    /// and returns the top `limit` results above `threshold`.
    pub fn resonate_history(
        &self,
        query_fingerprint: &[u64; FINGERPRINT_WORDS],
        threshold: f32,
        limit: usize,
    ) -> Vec<(Addr, f32)> {
        let space = self.bind_space.read();
        let mut results: Vec<(Addr, f32)> = Vec::new();

        let total_bits = (FINGERPRINT_WORDS * 64) as f32;

        for raw in 0..TOTAL_ADDRESSES {
            let addr = Addr(raw as u16);
            if let Some(node) = space.read(addr) {
                // Compute Hamming similarity: 1.0 - (distance / total_bits)
                let distance: u32 = node
                    .fingerprint
                    .iter()
                    .zip(query_fingerprint.iter())
                    .map(|(a, b)| (a ^ b).count_ones())
                    .sum();
                let similarity = 1.0 - (distance as f32 / total_bits);
                if similarity >= threshold {
                    results.push((addr, similarity));
                }
            }
        }

        // Sort by similarity descending, take top `limit`
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    /// Fork the temporal store at a specific version for what-if analysis.
    pub fn fork_at_version(&self, version: Version) -> WhatIfBranch {
        self.temporal.fork(version)
    }

    /// Fork at the current version.
    pub fn fork_current(&self) -> WhatIfBranch {
        let version = self.temporal.current_version();
        self.temporal.fork(version)
    }

    /// Diff two versions of the temporal store.
    pub fn diff_versions(&self, from: Version, to: Version) -> VersionDiff {
        self.temporal.diff(from, to)
    }

    /// Diff a what-if branch against its base.
    pub fn diff_branch(&self, branch: &WhatIfBranch) -> VersionDiff {
        branch.diff_from_base(&self.temporal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Fingerprint;

    #[test]
    fn test_spectator_resonate() {
        let bind_space = Arc::new(RwLock::new(BindSpace::new()));
        let temporal = Arc::new(TemporalStore::new());
        let spectator = UnifiedSpectator::new(bind_space.clone(), temporal);

        // Write a fingerprint into BindSpace
        let fp = Fingerprint::from_content("test execution output");
        {
            let mut space = bind_space.write();
            space.write(*fp.as_raw());
        }

        // Search for it — exact match should have similarity ~1.0
        let results = spectator.resonate_history(fp.as_raw(), 0.5, 10);
        assert!(
            !results.is_empty(),
            "Should find the fingerprint we just wrote"
        );
        // The first result should be an exact match (similarity ≈ 1.0)
        assert!(
            results[0].1 > 0.99,
            "Exact match should have similarity > 0.99, got {}",
            results[0].1
        );
    }

    #[test]
    fn test_spectator_fork_diff() {
        let bind_space = Arc::new(RwLock::new(BindSpace::new()));
        let temporal = Arc::new(TemporalStore::new());
        let spectator = UnifiedSpectator::new(bind_space, temporal);

        let branch = spectator.fork_current();
        let diff = spectator.diff_branch(&branch);
        assert!(diff.is_empty(), "Fresh fork should have no diff");
    }
}
