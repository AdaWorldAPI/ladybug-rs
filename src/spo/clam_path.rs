//! ClamPath + MerkleRoot — CLAM tree path as B-tree key with content-addressed identity.
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
//!
//! ## Eineindeutigkeit Resolution
//!
//! CogRecord8K word[0] packs ClamPath(24 bits) + MerkleRoot(40 bits) = 64 bits:
//!
//! ```text
//! ┌────────────────────┬────────────────────────────────────┐
//! │  ClamPath (24 bits) │  MerkleRoot (40 bits)             │
//! │  HOW you got here   │  WHAT lives here                  │
//! │  navigation/lineage │  identity/canonical address        │
//! └────────────────────┴────────────────────────────────────┘
//! ```
//!
//! ClamPath alone is path-dependent (same concept gets different addresses).
//! MerkleRoot alone has no lineage or subtree queries.
//! Together: navigate via ClamPath, resolve identity via MerkleRoot.

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

    // =========================================================================
    // U24 PACKING (for combined ClamPath + MerkleRoot in word[0])
    // =========================================================================

    /// Pack bits (u16) + depth (u8) into 24 bits.
    #[inline]
    pub fn to_u24(&self) -> u32 {
        ((self.depth as u32) << 16) | (self.bits as u32)
    }

    /// Unpack from 24 bits.
    #[inline]
    pub fn from_u24(packed: u32) -> Self {
        let bits = (packed & 0xFFFF) as u16;
        let depth = ((packed >> 16) & 0xFF) as u8;
        Self::from_raw(bits, depth)
    }

    /// Pack ClamPath (24 bits) + MerkleRoot (40 bits) into a single u64.
    ///
    /// This is the canonical encoding for CogRecord8K index.words[0].
    /// ClamPath in upper 24 bits, MerkleRoot in lower 40 bits.
    #[inline]
    pub fn pack_with_merkle(&self, root: MerkleRoot) -> u64 {
        let clam_bits = self.to_u24() as u64;
        let merkle_bits = root.0 & MerkleRoot::MASK;
        (clam_bits << 40) | merkle_bits
    }

    /// Unpack ClamPath + MerkleRoot from word[0].
    #[inline]
    pub fn unpack_with_merkle(word0: u64) -> (ClamPath, MerkleRoot) {
        let clam_bits = (word0 >> 40) as u32;
        let merkle_bits = word0 & MerkleRoot::MASK;
        (ClamPath::from_u24(clam_bits), MerkleRoot(merkle_bits))
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
// MERKLE ROOT — CONTENT-ADDRESSED CANONICAL IDENTITY
// =============================================================================

/// Truncated Merkle root for content-addressed identity.
///
/// 40 bits = ~1 trillion collision space. Sufficient for BindSpace.
/// Derived deterministically from three-plane binary fingerprints via blake3.
///
/// Same concept = same fingerprint = same root = same Redis key.
/// Self-healing: if Redis evicts the key, recompute from fingerprints.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct MerkleRoot(pub u64); // only lower 40 bits used

impl MerkleRoot {
    /// Bitmask for 40-bit truncation.
    pub const MASK: u64 = 0xFF_FFFF_FFFF;

    /// Zero root (placeholder / uninitialized).
    pub const ZERO: MerkleRoot = MerkleRoot(0);

    /// Derive from three-plane binary fingerprints (S, P, O).
    ///
    /// Each plane is a 16,384-bit (2,048-byte) binary fingerprint.
    /// Hash each plane separately with blake3, then combine:
    /// `blake3(S_hash || P_hash || O_hash)` — order is canonical (S < P < O).
    pub fn from_planes(
        s_binary: &[u8; 2048],
        p_binary: &[u8; 2048],
        o_binary: &[u8; 2048],
    ) -> Self {
        let s_hash = blake3::hash(s_binary);
        let p_hash = blake3::hash(p_binary);
        let o_hash = blake3::hash(o_binary);

        let mut hasher = blake3::Hasher::new();
        hasher.update(s_hash.as_bytes());
        hasher.update(p_hash.as_bytes());
        hasher.update(o_hash.as_bytes());
        let root = hasher.finalize();

        Self::from_hash_bytes(root.as_bytes())
    }

    /// Derive from single composite fingerprint (backward compat).
    ///
    /// Used when planes aren't separated yet.
    pub fn from_fingerprint(fp: &[u8; 2048]) -> Self {
        let hash = blake3::hash(fp);
        Self::from_hash_bytes(hash.as_bytes())
    }

    /// Extract 40-bit truncated root from hash bytes.
    #[inline]
    fn from_hash_bytes(bytes: &[u8; 32]) -> Self {
        let val = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], 0, 0, 0,
        ]);
        MerkleRoot(val & Self::MASK)
    }

    /// The Redis key for this concept's canonical address.
    pub fn redis_key(&self) -> String {
        format!("ada:bind:{:010x}", self.0)
    }

    /// Check if this is an uninitialized / zero root.
    pub fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl std::fmt::Debug for MerkleRoot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MerkleRoot({:010x})", self.0)
    }
}

impl std::fmt::Display for MerkleRoot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:010x}", self.0)
    }
}

// =============================================================================
// COGRECORD8K INTEGRATION
// =============================================================================

/// Extension trait for CogRecord8K to read/write ClamPath + MerkleRoot in the INDEX container.
///
/// word[0] = ClamPath(24 bits) + MerkleRoot(40 bits).
/// Remaining words (1..255) are available for edge adjacency, spine cache, etc.
pub trait ClamPathExt {
    /// Store ClamPath + MerkleRoot into INDEX container word[0].
    fn set_clam_path_merkle(&mut self, path: ClamPath, root: MerkleRoot);
    /// Read ClamPath + MerkleRoot from INDEX container word[0].
    fn clam_path_merkle(&self) -> (ClamPath, MerkleRoot);

    /// Store ClamPath only (MerkleRoot set to zero). Backward compat.
    fn set_clam_path(&mut self, path: ClamPath);
    /// Read ClamPath only (ignores MerkleRoot). Backward compat.
    fn clam_path(&self) -> ClamPath;
    /// Read MerkleRoot only (ignores ClamPath).
    fn merkle_root(&self) -> MerkleRoot;
}

impl ClamPathExt for ladybug_contract::CogRecord8K {
    fn set_clam_path_merkle(&mut self, path: ClamPath, root: MerkleRoot) {
        self.index.words[0] = path.pack_with_merkle(root);
    }

    fn clam_path_merkle(&self) -> (ClamPath, MerkleRoot) {
        ClamPath::unpack_with_merkle(self.index.words[0])
    }

    fn set_clam_path(&mut self, path: ClamPath) {
        self.set_clam_path_merkle(path, MerkleRoot::ZERO);
    }

    fn clam_path(&self) -> ClamPath {
        self.clam_path_merkle().0
    }

    fn merkle_root(&self) -> MerkleRoot {
        self.clam_path_merkle().1
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

    // =========================================================================
    // MerkleRoot tests
    // =========================================================================

    #[test]
    fn test_u24_roundtrip() {
        let cp = ClamPath::from_tree_traversal(&[true, false, true, true, false]);
        let packed = cp.to_u24();
        let cp2 = ClamPath::from_u24(packed);
        assert_eq!(cp, cp2);
    }

    #[test]
    fn test_merkle_root_from_fingerprint() {
        let fp = [0xABu8; 2048];
        let root = MerkleRoot::from_fingerprint(&fp);
        assert_ne!(root, MerkleRoot::ZERO);
        assert_eq!(root.0 & !MerkleRoot::MASK, 0, "upper bits must be zero");
    }

    #[test]
    fn test_merkle_root_deterministic() {
        let fp = [0x42u8; 2048];
        let r1 = MerkleRoot::from_fingerprint(&fp);
        let r2 = MerkleRoot::from_fingerprint(&fp);
        assert_eq!(r1, r2, "same content must produce same root");
    }

    #[test]
    fn test_merkle_root_from_planes() {
        let s = [0x11u8; 2048];
        let p = [0x22u8; 2048];
        let o = [0x33u8; 2048];
        let root = MerkleRoot::from_planes(&s, &p, &o);
        assert_ne!(root, MerkleRoot::ZERO);

        // Same planes → same root
        let root2 = MerkleRoot::from_planes(&s, &p, &o);
        assert_eq!(root, root2);

        // Different planes → different root
        let o2 = [0x44u8; 2048];
        let root3 = MerkleRoot::from_planes(&s, &p, &o2);
        assert_ne!(root, root3);
    }

    #[test]
    fn test_merkle_root_redis_key() {
        let root = MerkleRoot(0x12_3456_789A);
        assert_eq!(root.redis_key(), "ada:bind:123456789a");
    }

    #[test]
    fn test_pack_with_merkle_roundtrip() {
        let cp = ClamPath::from_tree_traversal(&[true, false, true, true]);
        let root = MerkleRoot(0xAB_CDEF_0123);

        let packed = cp.pack_with_merkle(root);
        let (cp2, root2) = ClamPath::unpack_with_merkle(packed);
        assert_eq!(cp, cp2);
        assert_eq!(root, root2);
    }

    #[test]
    fn test_pack_with_merkle_no_overlap() {
        // ClamPath in upper 24 bits must not corrupt MerkleRoot in lower 40
        let cp = ClamPath::from_raw(0xFFFF, 16); // all bits set
        let root = MerkleRoot(MerkleRoot::MASK); // all 40 bits set

        let packed = cp.pack_with_merkle(root);
        let (cp2, root2) = ClamPath::unpack_with_merkle(packed);
        assert_eq!(cp, cp2);
        assert_eq!(root, root2);
    }

    #[test]
    fn test_cogrecord8k_merkle_integration() {
        let mut record = ladybug_contract::CogRecord8K::new();
        let path = ClamPath::from_tree_traversal(&[true, false, true]);
        let fp = [0x77u8; 2048];
        let root = MerkleRoot::from_fingerprint(&fp);

        record.set_clam_path_merkle(path, root);
        let (p2, r2) = record.clam_path_merkle();
        assert_eq!(p2, path);
        assert_eq!(r2, root);

        // Backward compat: clam_path() still works
        assert_eq!(record.clam_path(), path);
        assert_eq!(record.merkle_root(), root);
    }

    #[test]
    fn test_cogrecord8k_clam_path_only_compat() {
        // set_clam_path (no merkle) → MerkleRoot reads as ZERO
        let mut record = ladybug_contract::CogRecord8K::new();
        let path = ClamPath::from_tree_traversal(&[false, true]);
        record.set_clam_path(path);
        assert_eq!(record.clam_path(), path);
        assert_eq!(record.merkle_root(), MerkleRoot::ZERO);
    }
}
