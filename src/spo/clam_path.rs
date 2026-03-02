//! ClamPath — CLAM tree path as B-tree key for lineage/structural/causal queries.
//!
//! Each ClamPath encodes the root-to-leaf traversal of a CLAM binary tree.
//! Each bit is a bipolar split decision: 0 = left pole, 1 = right pole.
//! MSB = root split, depth controls how many bits are valid.
//!
//! One u16 encodes three query types simultaneously:
//! - **Address**: Domain.Node.branch.twig.leaf — O(1) structural lookup
//! - **Lineage**: ancestor→parent→self→child — range scan = phylogeny
//! - **Causality**: cause→mediator→effect — range scan = causal chain
//!
//! Depth of a CLAM tree on 1024 items ≈ 10-12 levels, so u16 has room to spare.

/// CLAM path as B-tree key.
///
/// Each bit = one bipolar split decision (left pole = 0, right pole = 1).
/// MSB = root split, LSB-aligned toward leaf. Depth tracks valid bits.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClamPath {
    /// Bitpacked path (MSB = root split).
    pub bits: u16,
    /// How many bits are valid (= tree depth for this leaf).
    pub depth: u8,
}

impl ClamPath {
    /// Maximum supported tree depth (16 bits available).
    pub const MAX_DEPTH: u8 = 16;

    /// Root path (depth 0, no splits yet).
    pub const ROOT: ClamPath = ClamPath { bits: 0, depth: 0 };

    /// Construct from CLAM tree traversal (sequence of bipolar split decisions).
    ///
    /// Each element: `true` = went right, `false` = went left.
    /// Panics if path length exceeds 16.
    pub fn from_tree_traversal(cluster_path: &[bool]) -> Self {
        assert!(
            cluster_path.len() <= Self::MAX_DEPTH as usize,
            "CLAM path too deep: {} > {}",
            cluster_path.len(),
            Self::MAX_DEPTH
        );
        let mut bits: u16 = 0;
        for (i, &went_right) in cluster_path.iter().enumerate() {
            if went_right {
                bits |= 1 << (15 - i); // MSB = root
            }
        }
        Self {
            bits,
            depth: cluster_path.len() as u8,
        }
    }

    /// Construct from raw bits and depth.
    ///
    /// Masks off any bits beyond the valid depth.
    pub fn from_raw(bits: u16, depth: u8) -> Self {
        let depth = depth.min(Self::MAX_DEPTH);
        let mask = if depth == 0 {
            0u16
        } else if depth >= 16 {
            !0u16
        } else {
            !0u16 << (16 - depth)
        };
        Self {
            bits: bits & mask,
            depth,
        }
    }

    /// Decode the path back into a sequence of split decisions.
    pub fn to_traversal(&self) -> Vec<bool> {
        (0..self.depth as usize)
            .map(|i| (self.bits >> (15 - i)) & 1 == 1)
            .collect()
    }

    /// B-tree range query: everything in this subtree.
    ///
    /// Returns `(lo, hi)` — all paths that share this prefix.
    /// Semantically: "everything descended from this ancestor"
    /// = "everything downstream of this cause".
    pub fn subtree_range(&self) -> (u16, u16) {
        if self.depth == 0 {
            return (0, u16::MAX);
        }
        let shift = 16u32.saturating_sub(self.depth as u32);
        let lo = self.bits & (!0u16).checked_shl(shift).unwrap_or(0);
        let hi = lo | (1u16.checked_shl(shift).unwrap_or(0)).wrapping_sub(1);
        (lo, hi)
    }

    /// Common ancestor depth between two paths.
    ///
    /// = how many split decisions they share from the root.
    /// XOR the bits, count leading zeros (shared prefix length).
    pub fn common_ancestor_depth(&self, other: &ClamPath) -> u8 {
        let xor = self.bits ^ other.bits;
        let shared = xor.leading_zeros() as u8;
        shared.min(self.depth).min(other.depth)
    }

    /// Lineage distance: total splits apart (symmetric).
    ///
    /// Distance = (self.depth - ancestor) + (other.depth - ancestor).
    pub fn lineage_distance(&self, other: &ClamPath) -> u8 {
        let ancestor = self.common_ancestor_depth(other);
        (self.depth - ancestor) + (other.depth - ancestor)
    }

    /// Sibling path: flip the last split decision.
    ///
    /// "What's on the other side of the last bipolar split?"
    /// = counterfactual query: "what if the last decision went the other way?"
    pub fn siblings(&self) -> ClamPath {
        if self.depth == 0 {
            return *self; // root has no sibling
        }
        let flip_bit = 1u16 << (16 - self.depth as u32);
        ClamPath {
            bits: self.bits ^ flip_bit,
            depth: self.depth,
        }
    }

    /// Parent path: remove the last split decision.
    pub fn parent(&self) -> ClamPath {
        if self.depth == 0 {
            return *self;
        }
        let new_depth = self.depth - 1;
        let mask = if new_depth == 0 {
            0u16
        } else {
            !0u16 << (16 - new_depth)
        };
        ClamPath {
            bits: self.bits & mask,
            depth: new_depth,
        }
    }

    /// Left child: append a 0 (left pole) split.
    pub fn left_child(&self) -> Option<ClamPath> {
        if self.depth >= Self::MAX_DEPTH {
            return None;
        }
        // Next bit position is already 0 in the masked bits
        Some(ClamPath {
            bits: self.bits, // no change — the new bit is 0
            depth: self.depth + 1,
        })
    }

    /// Right child: append a 1 (right pole) split.
    pub fn right_child(&self) -> Option<ClamPath> {
        if self.depth >= Self::MAX_DEPTH {
            return None;
        }
        let set_bit = 1u16 << (15 - self.depth as u32);
        Some(ClamPath {
            bits: self.bits | set_bit,
            depth: self.depth + 1,
        })
    }

    /// Check if `other` is a descendant of `self` (self is ancestor of other).
    pub fn is_ancestor_of(&self, other: &ClamPath) -> bool {
        if self.depth > other.depth {
            return false;
        }
        self.common_ancestor_depth(other) == self.depth
    }

    /// Adaptive search hint based on depth (proxy for Local Fractal Dimension).
    ///
    /// Shallow = dense cluster (low LFD) → broad sweep.
    /// Deep = sparse region (high LFD) → precise search.
    pub fn density_hint(&self) -> DensityHint {
        match self.depth {
            0..=6 => DensityHint::Dense,
            7..=10 => DensityHint::Moderate,
            _ => DensityHint::Sparse,
        }
    }

    // =========================================================================
    // COGRECORD8K INDEX CONTAINER ENCODING
    // =========================================================================

    /// Pack into a u64 word for storage in CogRecord8K's index container.
    ///
    /// Layout: bits 0..15 = path bits, bits 16..23 = depth, bits 24..63 = reserved.
    #[inline]
    pub fn to_word(&self) -> u64 {
        (self.bits as u64) | ((self.depth as u64) << 16)
    }

    /// Unpack from a u64 word (index container word[0]).
    #[inline]
    pub fn from_word(word: u64) -> Self {
        let bits = word as u16;
        let depth = ((word >> 16) & 0xFF) as u8;
        Self::from_raw(bits, depth)
    }

    /// Pack into big-endian bytes (for B-tree column storage in Lance).
    ///
    /// 3 bytes: [bits_hi, bits_lo, depth].
    pub fn to_bytes(&self) -> [u8; 3] {
        let [hi, lo] = self.bits.to_be_bytes();
        [hi, lo, self.depth]
    }

    /// Unpack from big-endian bytes.
    pub fn from_bytes(bytes: [u8; 3]) -> Self {
        let bits = u16::from_be_bytes([bytes[0], bytes[1]]);
        Self::from_raw(bits, bytes[2])
    }

    /// The u16 value for LanceDB B-tree indexing.
    ///
    /// This preserves prefix ordering: a subtree_range() scan on
    /// this value produces correct lineage traversals.
    pub fn btree_key(&self) -> u16 {
        self.bits
    }
}

impl std::fmt::Debug for ClamPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let path: String = (0..self.depth as usize)
            .map(|i| if (self.bits >> (15 - i)) & 1 == 1 { 'R' } else { 'L' })
            .collect();
        write!(f, "ClamPath({}, d={})", path, self.depth)
    }
}

impl std::fmt::Display for ClamPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let path: String = (0..self.depth as usize)
            .map(|i| if (self.bits >> (15 - i)) & 1 == 1 { '1' } else { '0' })
            .collect();
        write!(f, "{}", path)
    }
}

/// Depth-based density hint for adaptive search strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DensityHint {
    /// Shallow path = dense cluster = low LFD. Broad sweep, many expected hits.
    Dense,
    /// Medium depth = moderate density. Standard search.
    Moderate,
    /// Deep path = sparse region = high LFD. Precise search, few expected hits.
    Sparse,
}

// =============================================================================
// COGRECORD8K INTEGRATION
// =============================================================================

/// Extension trait for CogRecord8K to read/write ClamPath in the INDEX container.
///
/// The ClamPath occupies word[0] of the index container.
/// Remaining words (1..255) are available for edge adjacency, spine cache, etc.
pub trait ClamPathExt {
    /// Store a ClamPath into the INDEX container (word 0).
    fn set_clam_path(&mut self, path: ClamPath);
    /// Read the ClamPath from the INDEX container (word 0).
    fn clam_path(&self) -> ClamPath;
}

impl ClamPathExt for ladybug_contract::CogRecord8K {
    fn set_clam_path(&mut self, path: ClamPath) {
        self.index.words[0] = path.to_word();
    }

    fn clam_path(&self) -> ClamPath {
        ClamPath::from_word(self.index.words[0])
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_traversal_roundtrip() {
        let path = vec![true, false, true, true, false];
        let cp = ClamPath::from_tree_traversal(&path);
        assert_eq!(cp.depth, 5);
        assert_eq!(cp.to_traversal(), path);
    }

    #[test]
    fn test_empty_path() {
        let cp = ClamPath::from_tree_traversal(&[]);
        assert_eq!(cp.depth, 0);
        assert_eq!(cp.bits, 0);
        assert_eq!(cp, ClamPath::ROOT);
    }

    #[test]
    fn test_full_depth_path() {
        let path: Vec<bool> = (0..16).map(|i| i % 2 == 0).collect();
        let cp = ClamPath::from_tree_traversal(&path);
        assert_eq!(cp.depth, 16);
        assert_eq!(cp.to_traversal(), path);
    }

    #[test]
    fn test_subtree_range_root() {
        let root = ClamPath::ROOT;
        assert_eq!(root.subtree_range(), (0, u16::MAX));
    }

    #[test]
    fn test_subtree_range_depth_1() {
        // Left child at depth 1: path = [false] → bits = 0b0000...
        let left = ClamPath::from_tree_traversal(&[false]);
        let (lo, hi) = left.subtree_range();
        assert_eq!(lo, 0);
        assert_eq!(hi, 0x7FFF); // lower half

        // Right child at depth 1: path = [true] → bits = 0b1000...
        let right = ClamPath::from_tree_traversal(&[true]);
        let (lo, hi) = right.subtree_range();
        assert_eq!(lo, 0x8000);
        assert_eq!(hi, 0xFFFF); // upper half
    }

    #[test]
    fn test_subtree_range_contains_descendants() {
        let parent = ClamPath::from_tree_traversal(&[true, false]); // depth 2
        let child = ClamPath::from_tree_traversal(&[true, false, true]); // depth 3
        let (lo, hi) = parent.subtree_range();
        assert!(child.bits >= lo && child.bits <= hi);
    }

    #[test]
    fn test_common_ancestor_identical() {
        let a = ClamPath::from_tree_traversal(&[true, false, true]);
        assert_eq!(a.common_ancestor_depth(&a), 3);
    }

    #[test]
    fn test_common_ancestor_diverge_at_root() {
        let a = ClamPath::from_tree_traversal(&[true, false]);
        let b = ClamPath::from_tree_traversal(&[false, true]);
        assert_eq!(a.common_ancestor_depth(&b), 0);
    }

    #[test]
    fn test_common_ancestor_partial_overlap() {
        let a = ClamPath::from_tree_traversal(&[true, false, true]);
        let b = ClamPath::from_tree_traversal(&[true, false, false]);
        assert_eq!(a.common_ancestor_depth(&b), 2); // share [true, false]
    }

    #[test]
    fn test_lineage_distance() {
        let a = ClamPath::from_tree_traversal(&[true, false, true]); // d=3
        let b = ClamPath::from_tree_traversal(&[true, false, false]); // d=3, ancestor d=2
        assert_eq!(a.lineage_distance(&b), 2); // (3-2) + (3-2) = 2
    }

    #[test]
    fn test_lineage_distance_ancestor_descendant() {
        let ancestor = ClamPath::from_tree_traversal(&[true, false]); // d=2
        let descendant = ClamPath::from_tree_traversal(&[true, false, true, true]); // d=4
        // ancestor depth = 2, so distance = (2-2) + (4-2) = 2
        assert_eq!(ancestor.lineage_distance(&descendant), 2);
    }

    #[test]
    fn test_siblings() {
        let a = ClamPath::from_tree_traversal(&[true, false, true]);
        let b = a.siblings();
        // Sibling flips last bit: [true, false, false]
        assert_eq!(b.to_traversal(), vec![true, false, false]);
        assert_eq!(b.depth, 3);
        // And back
        assert_eq!(b.siblings(), a);
    }

    #[test]
    fn test_parent() {
        let child = ClamPath::from_tree_traversal(&[true, false, true]);
        let parent = child.parent();
        assert_eq!(parent.depth, 2);
        assert_eq!(parent.to_traversal(), vec![true, false]);
    }

    #[test]
    fn test_parent_of_root() {
        assert_eq!(ClamPath::ROOT.parent(), ClamPath::ROOT);
    }

    #[test]
    fn test_children() {
        let parent = ClamPath::from_tree_traversal(&[true, false]);
        let left = parent.left_child().unwrap();
        let right = parent.right_child().unwrap();
        assert_eq!(left.to_traversal(), vec![true, false, false]);
        assert_eq!(right.to_traversal(), vec![true, false, true]);
        assert_eq!(left.parent(), parent);
        assert_eq!(right.parent(), parent);
    }

    #[test]
    fn test_is_ancestor_of() {
        let ancestor = ClamPath::from_tree_traversal(&[true, false]);
        let descendant = ClamPath::from_tree_traversal(&[true, false, true, true]);
        let unrelated = ClamPath::from_tree_traversal(&[false, true]);
        assert!(ancestor.is_ancestor_of(&descendant));
        assert!(!descendant.is_ancestor_of(&ancestor));
        assert!(!ancestor.is_ancestor_of(&unrelated));
        assert!(ClamPath::ROOT.is_ancestor_of(&descendant));
    }

    #[test]
    fn test_word_roundtrip() {
        let cp = ClamPath::from_tree_traversal(&[true, false, true, true, false]);
        let word = cp.to_word();
        let cp2 = ClamPath::from_word(word);
        assert_eq!(cp, cp2);
    }

    #[test]
    fn test_bytes_roundtrip() {
        let cp = ClamPath::from_tree_traversal(&[false, true, true, false, true]);
        let bytes = cp.to_bytes();
        let cp2 = ClamPath::from_bytes(bytes);
        assert_eq!(cp, cp2);
    }

    #[test]
    fn test_cogrecord8k_integration() {
        let mut record = ladybug_contract::CogRecord8K::new();
        let path = ClamPath::from_tree_traversal(&[true, false, true, true]);
        record.set_clam_path(path);
        assert_eq!(record.clam_path(), path);
    }

    #[test]
    fn test_from_raw_masks_invalid_bits() {
        // depth=4 means only top 4 bits should survive
        let cp = ClamPath::from_raw(0xFFFF, 4);
        assert_eq!(cp.bits, 0xF000);
        assert_eq!(cp.depth, 4);
    }

    #[test]
    fn test_btree_key_ordering() {
        // B-tree key preserves prefix ordering: left subtree < right subtree
        let left = ClamPath::from_tree_traversal(&[false, true, false]);
        let right = ClamPath::from_tree_traversal(&[true, false, true]);
        assert!(left.btree_key() < right.btree_key());
    }

    #[test]
    fn test_density_hint() {
        let shallow = ClamPath::from_tree_traversal(&[true, false]);
        let medium = ClamPath::from_tree_traversal(&[true; 8]);
        let deep = ClamPath::from_tree_traversal(&[true; 14]);
        assert_eq!(shallow.density_hint(), DensityHint::Dense);
        assert_eq!(medium.density_hint(), DensityHint::Moderate);
        assert_eq!(deep.density_hint(), DensityHint::Sparse);
    }

    #[test]
    fn test_debug_display() {
        let cp = ClamPath::from_tree_traversal(&[true, false, true]);
        assert_eq!(format!("{:?}", cp), "ClamPath(RLR, d=3)");
        assert_eq!(format!("{}", cp), "101");
    }
}
