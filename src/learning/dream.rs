//! Dream Consolidation — offline pruning, merging, and creative recombination.
//!
//! Inspired by the memory-consolidation role of sleep, this module operates
//! on a batch of `CogRecord`s (captured during a cognitive session) and
//! produces a reduced, strengthened set:
//!
//! 1. **Prune** low-confidence records (NARS confidence below threshold).
//! 2. **Merge** records with similar content fingerprints (Hamming distance
//!    below threshold) using majority-vote bundling.
//! 3. **Permute-XOR-Bind** surviving records in novel pairs to create
//!    creative recombinations (dream-like novel associations).
//!
//! The result is a new `Vec<CogRecord>` ready to be re-ingested or committed
//! to long-term storage.

use crate::container::Container;
use crate::container::record::CogRecord;

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Tuning parameters for dream consolidation.
#[derive(Clone, Debug)]
pub struct DreamConfig {
    /// NARS confidence threshold below which a record is pruned.
    pub prune_confidence_threshold: f32,

    /// Hamming distance threshold below which two records are considered
    /// similar enough to merge.
    pub merge_hamming_threshold: u32,

    /// Number of novel recombinations to generate from surviving records.
    /// Set to 0 to disable creative recombination.
    pub recombination_count: usize,

    /// Permutation offset used when creating novel bindings.
    pub permute_offset: i32,

    /// Maximum number of output records (0 = unlimited).
    pub max_output: usize,
}

impl Default for DreamConfig {
    fn default() -> Self {
        Self {
            prune_confidence_threshold: 0.2,
            merge_hamming_threshold: 1500, // ~18% of 8192 bits
            recombination_count: 5,
            permute_offset: 7,
            max_output: 0,
        }
    }
}

// =============================================================================
// PUBLIC API
// =============================================================================

/// Consolidate a batch of CogRecords using default configuration.
///
/// Returns a new, smaller set of records with:
/// - Low-confidence records removed
/// - Similar records merged
/// - Novel creative bindings appended
pub fn consolidate(records: &[CogRecord]) -> Vec<CogRecord> {
    consolidate_with_config(records, &DreamConfig::default())
}

/// Consolidate with custom configuration.
pub fn consolidate_with_config(records: &[CogRecord], config: &DreamConfig) -> Vec<CogRecord> {
    if records.is_empty() {
        return Vec::new();
    }

    // Phase 1: Prune low-confidence records.
    let survivors = prune(records, config.prune_confidence_threshold);

    if survivors.is_empty() {
        return Vec::new();
    }

    // Phase 2: Merge similar fingerprints.
    let merged = merge_similar(&survivors, config.merge_hamming_threshold);

    // Phase 3: Generate novel recombinations.
    let mut output = merged;
    if config.recombination_count > 0 && output.len() >= 2 {
        let novels = recombine(&output, config.recombination_count, config.permute_offset);
        output.extend(novels);
    }

    // Phase 4: Cap output size if requested.
    if config.max_output > 0 && output.len() > config.max_output {
        output.truncate(config.max_output);
    }

    output
}

// =============================================================================
// PHASE 1: PRUNING
// =============================================================================

/// Remove records whose NARS confidence falls below the threshold.
///
/// Records with zero or NaN confidence are always pruned.
fn prune(records: &[CogRecord], threshold: f32) -> Vec<CogRecord> {
    records
        .iter()
        .filter(|r| {
            let conf = r.meta_view().nars_confidence();
            conf.is_finite() && conf >= threshold
        })
        .cloned()
        .collect()
}

// =============================================================================
// PHASE 2: MERGING
// =============================================================================

/// Merge records whose content fingerprints are within `threshold` Hamming
/// distance.
///
/// Uses a greedy clustering approach:
/// - Iterate through records in order.
/// - For each record, find the first existing cluster whose centroid is within
///   threshold distance.
/// - If found, bundle the record into that cluster's centroid.
/// - Otherwise, start a new cluster.
///
/// The resulting records carry the merged content and the metadata of the
/// cluster's first (seed) record.
fn merge_similar(records: &[CogRecord], threshold: u32) -> Vec<CogRecord> {
    if records.is_empty() {
        return Vec::new();
    }

    struct Cluster {
        /// The accumulated content (majority-vote bundle of members).
        centroid: Container,
        /// The seed record (provides metadata for the merged output).
        seed: CogRecord,
        /// Number of records merged into this cluster.
        count: usize,
        /// All member contents (needed for majority-vote rebundling).
        members: Vec<Container>,
    }

    let mut clusters: Vec<Cluster> = Vec::new();

    for record in records {
        let mut merged = false;

        for cluster in clusters.iter_mut() {
            let dist = cluster.centroid.hamming(&record.content);
            if dist <= threshold {
                // Add to this cluster.
                cluster.members.push(record.content.clone());
                cluster.count += 1;

                // Rebundle the centroid via majority vote.
                let refs: Vec<&Container> = cluster.members.iter().collect();
                cluster.centroid = Container::bundle(&refs);

                merged = true;
                break;
            }
        }

        if !merged {
            clusters.push(Cluster {
                centroid: record.content.clone(),
                seed: record.clone(),
                count: 1,
                members: vec![record.content.clone()],
            });
        }
    }

    // Build output: each cluster becomes one record.
    clusters
        .into_iter()
        .map(|cluster| {
            let mut out = cluster.seed;
            out.content = cluster.centroid;
            // Update confidence proportional to cluster size (capped at 1.0).
            let original_conf = out.meta_view().nars_confidence();
            let boosted = (original_conf + 0.05 * (cluster.count as f32 - 1.0)).min(1.0);
            if boosted.is_finite() {
                out.meta_view_mut().set_nars_confidence(boosted);
            }
            out
        })
        .collect()
}

// =============================================================================
// PHASE 3: CREATIVE RECOMBINATION
// =============================================================================

/// Generate novel records by permute-XOR-binding pairs of existing records.
///
/// For each novel combination:
///   novel.content = permute(A.content, offset) XOR B.content
///
/// This creates fingerprints that are "between" A and B in Hamming space but
/// occupy novel regions — analogous to dream-state recombination.
fn recombine(
    records: &[CogRecord],
    count: usize,
    permute_offset: i32,
) -> Vec<CogRecord> {
    let n = records.len();
    if n < 2 {
        return Vec::new();
    }

    let mut novels = Vec::with_capacity(count);
    let mut pair_idx: usize = 0;

    for _ in 0..count {
        // Select pairs by stepping through the triangle of (i, j) combinations.
        let i = pair_idx % n;
        let j = (pair_idx / n + 1) % n;
        let j = if j == i { (j + 1) % n } else { j };
        pair_idx = pair_idx.wrapping_add(1);

        if i == j {
            continue;
        }

        // Permute A, then XOR-bind with B.
        let permuted_a = records[i].content.permute(permute_offset);
        let novel_content = permuted_a.xor(&records[j].content);

        // Build the novel record with averaged metadata from A.
        let mut novel = CogRecord::new(records[i].geometry());

        // Set content.
        novel.content = novel_content;

        // Set reduced confidence (dream-created records start uncertain).
        let avg_conf = (records[i].meta_view().nars_confidence()
            + records[j].meta_view().nars_confidence())
            / 2.0;
        let dream_conf = (avg_conf * 0.5).max(0.1);
        if dream_conf.is_finite() {
            novel.meta_view_mut().set_nars_confidence(dream_conf);
        }

        // Set averaged frequency.
        let avg_freq = (records[i].meta_view().nars_frequency()
            + records[j].meta_view().nars_frequency())
            / 2.0;
        if avg_freq.is_finite() {
            novel.meta_view_mut().set_nars_frequency(avg_freq);
        }

        novels.push(novel);
    }

    novels
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::container::ContainerGeometry;

    /// Helper: create a CogRecord with specific content and confidence.
    fn make_record(seed: u64, confidence: f32) -> CogRecord {
        let mut r = CogRecord::new(ContainerGeometry::Cam);
        r.content = Container::random(seed);
        r.meta_view_mut().set_nars_confidence(confidence);
        r.meta_view_mut().set_nars_frequency(0.5);
        r
    }

    #[test]
    fn test_empty_input() {
        let result = consolidate(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_prune_low_confidence() {
        let records = vec![
            make_record(1, 0.9),
            make_record(2, 0.1), // below default threshold (0.2)
            make_record(3, 0.5),
        ];
        let result = consolidate(&records);
        // Record 2 should be pruned (conf 0.1 < threshold 0.2).
        // But we should still have records.
        assert!(result.len() >= 2, "expected >= 2, got {}", result.len());
    }

    #[test]
    fn test_all_pruned() {
        let records = vec![
            make_record(1, 0.01),
            make_record(2, 0.05),
        ];
        let result = consolidate(&records);
        assert!(result.is_empty());
    }

    #[test]
    fn test_merge_identical() {
        // Two identical records should merge into one.
        let r1 = make_record(42, 0.8);
        let r2 = make_record(42, 0.7); // same seed = same content
        let records = vec![r1, r2];
        let config = DreamConfig {
            recombination_count: 0, // disable recombination for this test
            ..Default::default()
        };
        let result = consolidate_with_config(&records, &config);
        assert_eq!(result.len(), 1, "identical records should merge");
    }

    #[test]
    fn test_no_merge_distant() {
        // Records with very different content should not merge.
        let records = vec![
            make_record(1, 0.8),
            make_record(2, 0.8),
        ];
        let config = DreamConfig {
            merge_hamming_threshold: 100, // very tight threshold
            recombination_count: 0,
            ..Default::default()
        };
        let result = consolidate_with_config(&records, &config);
        assert_eq!(result.len(), 2, "distant records should not merge");
    }

    #[test]
    fn test_recombination_produces_novels() {
        let records = vec![
            make_record(10, 0.8),
            make_record(20, 0.8),
            make_record(30, 0.8),
        ];
        let config = DreamConfig {
            recombination_count: 3,
            merge_hamming_threshold: 0, // disable merging
            ..Default::default()
        };
        let result = consolidate_with_config(&records, &config);
        // 3 original + up to 3 novels.
        assert!(result.len() > 3, "expected novels, got len={}", result.len());
    }

    #[test]
    fn test_novel_content_is_different() {
        let records = vec![
            make_record(100, 0.9),
            make_record(200, 0.9),
        ];
        let config = DreamConfig {
            recombination_count: 1,
            merge_hamming_threshold: 0,
            ..Default::default()
        };
        let result = consolidate_with_config(&records, &config);
        assert!(result.len() >= 3);
        // The novel record should differ from both originals.
        let novel = &result[2];
        assert_ne!(novel.content, records[0].content);
        assert_ne!(novel.content, records[1].content);
    }

    #[test]
    fn test_max_output_cap() {
        let records: Vec<_> = (0..20).map(|i| make_record(i, 0.8)).collect();
        let config = DreamConfig {
            max_output: 5,
            recombination_count: 0,
            ..Default::default()
        };
        let result = consolidate_with_config(&records, &config);
        assert!(result.len() <= 5);
    }

    #[test]
    fn test_dream_confidence_reduced() {
        let records = vec![
            make_record(100, 0.9),
            make_record(200, 0.9),
        ];
        let config = DreamConfig {
            recombination_count: 1,
            merge_hamming_threshold: 0,
            ..Default::default()
        };
        let result = consolidate_with_config(&records, &config);
        if result.len() >= 3 {
            // Dream-created record should have lower confidence.
            let dream_conf = result[2].meta_view().nars_confidence();
            assert!(
                dream_conf < 0.9,
                "dream confidence ({}) should be less than original (0.9)",
                dream_conf
            );
        }
    }
}
