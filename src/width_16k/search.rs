//! Schema-filtered search over 16K cognitive records.
//!
//! Distance = popcount(XOR(a[0..224], b[0..224])) over resonance words only.
//! Metadata predicates (ANI level, NARS truth, node kind, bloom) are checked
//! BEFORE computing full distance — a few cycles vs full popcount.

use super::schema::SchemaSidecar;
use super::{RESONANCE_WORDS, VECTOR_WORDS};

/// Block mask: which of the 14 resonance blocks participate in distance.
/// Default = all 14 blocks. Can exclude blocks for partial/multi-resolution search.
#[derive(Clone, Copy, Debug)]
pub struct BlockMask {
    /// Bitmask: bit i = include block i in distance computation (0..13)
    pub mask: u16,
}

impl Default for BlockMask {
    fn default() -> Self {
        Self { mask: 0x3FFF } // all 14 resonance blocks
    }
}

impl BlockMask {
    /// All resonance blocks
    pub fn all() -> Self {
        Self::default()
    }

    /// Single block
    pub fn single(block: usize) -> Self {
        Self {
            mask: 1u16 << block,
        }
    }

    /// Check if a block is included
    pub fn includes(&self, block: usize) -> bool {
        block < 14 && (self.mask & (1u16 << block)) != 0
    }
}

/// Compute Hamming distance over resonance words only (words 0-223).
pub fn resonance_distance(a: &[u64; VECTOR_WORDS], b: &[u64; VECTOR_WORDS]) -> u32 {
    let mut dist = 0u32;
    for i in 0..RESONANCE_WORDS {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}

/// Compute Hamming distance over selected blocks only.
pub fn block_masked_distance(
    a: &[u64; VECTOR_WORDS],
    b: &[u64; VECTOR_WORDS],
    mask: BlockMask,
) -> u32 {
    let mut dist = 0u32;
    for block in 0..14 {
        if mask.includes(block) {
            let start = block * 16;
            let end = start + 16;
            for i in start..end {
                dist += (a[i] ^ b[i]).count_ones();
            }
        }
    }
    dist
}

/// Quick distance estimate using belichtungsmesser sample points.
/// Returns estimated distance scaled to full resonance width.
pub fn sample_distance(a: &[u64; VECTOR_WORDS], b: &[u64; VECTOR_WORDS]) -> u32 {
    let mut sample_dist = 0u32;
    for &p in &super::SAMPLE_POINTS {
        sample_dist += (a[p] ^ b[p]).count_ones();
    }
    // Scale: 7 sample words -> 224 resonance words
    sample_dist * 32 // 224/7 = 32
}

/// Schema predicate for filtering before distance computation.
#[derive(Clone, Debug, Default)]
pub struct SchemaQuery {
    /// Minimum ANI level (any level >= threshold passes)
    pub min_ani_level: Option<u16>,
    /// Required node kind
    pub node_kind: Option<u8>,
    /// Bloom pre-filter: candidate must might-contain this ID
    pub bloom_contains: Option<u64>,
    /// Maximum allowed distance (after computing)
    pub max_distance: Option<u32>,
    /// Block mask for distance computation
    pub block_mask: BlockMask,
}

impl SchemaQuery {
    /// Check if a record passes the schema predicates (fast, no distance).
    pub fn matches_schema(&self, words: &[u64; VECTOR_WORDS]) -> bool {
        let schema = SchemaSidecar::read_from_words(words);

        if let Some(min) = self.min_ani_level {
            let levels = schema.ani_levels;
            let max_level = [
                levels.reactive,
                levels.memory,
                levels.analogy,
                levels.planning,
                levels.meta,
                levels.social,
                levels.creative,
                levels.r#abstract,
            ]
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
            if max_level < min {
                return false;
            }
        }

        if let Some(kind) = self.node_kind {
            if schema.identity.node_type.kind != kind {
                return false;
            }
        }

        if let Some(id) = self.bloom_contains {
            if !schema.neighbors.might_contain(id) {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::width_16k::schema::NodeKind;

    #[test]
    fn test_resonance_distance_self() {
        let v = [0x5555555555555555u64; VECTOR_WORDS];
        assert_eq!(resonance_distance(&v, &v), 0);
    }

    #[test]
    fn test_resonance_distance_complement() {
        let a = [0u64; VECTOR_WORDS];
        let b = [u64::MAX; VECTOR_WORDS];
        // 224 words * 64 bits = 14336 set bits
        assert_eq!(resonance_distance(&a, &b), 14336);
    }

    #[test]
    fn test_block_mask() {
        let m = BlockMask::single(5);
        assert!(m.includes(5));
        assert!(!m.includes(0));
        assert!(!m.includes(14)); // out of range
    }

    #[test]
    fn test_sample_distance_estimate() {
        let a = [0u64; VECTOR_WORDS];
        let b = [u64::MAX; VECTOR_WORDS];
        let est = sample_distance(&a, &b);
        let exact = resonance_distance(&a, &b);
        // Estimate should be in the right ballpark
        assert!((est as i64 - exact as i64).unsigned_abs() < exact as u64 / 4);
    }

    #[test]
    fn test_schema_query_node_kind() {
        let mut words = [0u64; VECTOR_WORDS];
        let mut s = SchemaSidecar::default();
        s.identity.node_type.kind = NodeKind::Concept as u8;
        s.write_to_words(&mut words);

        let q = SchemaQuery {
            node_kind: Some(NodeKind::Concept as u8),
            ..Default::default()
        };
        assert!(q.matches_schema(&words));

        let q2 = SchemaQuery {
            node_kind: Some(NodeKind::Event as u8),
            ..Default::default()
        };
        assert!(!q2.matches_schema(&words));
    }
}
