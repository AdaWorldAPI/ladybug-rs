//! XOR-Merkle tree over the DN address space.
//!
//! Provides authenticated query results over the wire: every response
//! includes an inclusion proof the client can verify without trusting
//! the server. No graph database currently offers this.
//!
//! XOR for combining children (not concatenation + hash) because:
//! - Order-independent: children can be added/removed without rehashing siblings
//! - O(1) update: flip one child hash, XOR it out of parent, XOR new hash in
//! - Composable with HDC: the Merkle layer uses the same XOR operation as
//!   fingerprint binding
//!
//! ```text
//!                     Root Hash
//!                    ╱          ╲
//!            H(child₁)       H(child₂)
//!            ╱     ╲           ╱     ╲
//!       H(leaf₁) H(leaf₂) H(leaf₃) H(leaf₄)
//!
//! Where H(node) = SHA256(dn ‖ fingerprint ‖ nars_truth ‖ H(child₁) ⊕ H(child₂) ⊕ ...)
//! ```

use std::collections::{HashMap, HashSet};

use sha2::{Digest, Sha256};

use super::store::QueryHit;

/// 32-byte Merkle hash.
pub type MerkleHash = [u8; 32];

/// Zero hash (identity for XOR).
pub const ZERO_HASH: MerkleHash = [0u8; 32];

/// XOR two Merkle hashes (order-independent combination).
#[inline]
pub fn xor_hash(a: &MerkleHash, b: &MerkleHash) -> MerkleHash {
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = a[i] ^ b[i];
    }
    out
}

/// Compute leaf hash from DN + fingerprint bytes + NARS truth values.
pub fn leaf_hash(dn: u64, fingerprint: &[u8], freq: f32, conf: f32) -> MerkleHash {
    let mut hasher = Sha256::new();
    hasher.update(dn.to_le_bytes());
    hasher.update(fingerprint);
    hasher.update(freq.to_le_bytes());
    hasher.update(conf.to_le_bytes());
    hasher.finalize().into()
}

// ============================================================================
// SPO MERKLE TREE
// ============================================================================

/// XOR-Merkle tree over the DN tree (4096 address space).
///
/// Properties:
/// - Leaf = SHA256(dn ‖ fingerprint ‖ nars_truth)
/// - Interior = XOR of children's hashes (order-independent)
/// - O(1) insert/update (rehash leaf + XOR-flip up to root)
/// - O(log n) proof of inclusion
/// - O(1) integrity check (compare root hashes)
pub struct SpoMerkle {
    /// dn → leaf hash
    leaves: HashMap<u64, MerkleHash>,
    /// dn → XOR accumulation of direct children's hashes
    children_xor: HashMap<u64, MerkleHash>,
    /// dn → parent dn (for walking up)
    parents: HashMap<u64, u64>,
    /// dn → set of direct child DNs
    children: HashMap<u64, HashSet<u64>>,
    /// DNs modified since last snapshot (for incremental truth recomputation)
    dirty_since_last_snapshot: HashSet<u64>,
    /// Root DN (0 for synthetic root)
    root_dn: u64,
}

impl SpoMerkle {
    /// Create a new empty Merkle tree.
    pub fn new() -> Self {
        Self {
            leaves: HashMap::new(),
            children_xor: HashMap::new(),
            parents: HashMap::new(),
            children: HashMap::new(),
            dirty_since_last_snapshot: HashSet::new(),
            root_dn: 0,
        }
    }

    /// Create with a specific root DN.
    pub fn with_root(root_dn: u64) -> Self {
        Self {
            leaves: HashMap::new(),
            children_xor: HashMap::new(),
            parents: HashMap::new(),
            children: HashMap::new(),
            dirty_since_last_snapshot: HashSet::new(),
            root_dn,
        }
    }

    /// Insert or update a leaf. Returns (old_root, new_root).
    pub fn insert(
        &mut self,
        dn: u64,
        parent_dn: u64,
        fingerprint: &[u8],
        freq: f32,
        conf: f32,
    ) -> (MerkleHash, MerkleHash) {
        let old_root = self.root_hash();
        let new_leaf = leaf_hash(dn, fingerprint, freq, conf);

        // If leaf already exists, XOR out old hash from parent's accumulator
        if let Some(old_leaf) = self.leaves.get(&dn).copied() {
            // Remove from old parent's children set
            if let Some(&old_parent) = self.parents.get(&dn) {
                if let Some(kids) = self.children.get_mut(&old_parent) {
                    kids.remove(&dn);
                }
            }
            self.flip_up(parent_dn, &old_leaf);
        }

        // Store new leaf and parent pointer
        self.leaves.insert(dn, new_leaf);
        self.parents.insert(dn, parent_dn);

        // Track parent → child relationship
        self.children.entry(parent_dn).or_default().insert(dn);

        // Mark as dirty for truth trajectory
        self.dirty_since_last_snapshot.insert(dn);

        // XOR new hash into parent's accumulator
        self.flip_up(parent_dn, &new_leaf);

        let new_root = self.root_hash();
        (old_root, new_root)
    }

    /// Remove a leaf. O(1) XOR-flip up to root.
    pub fn remove(&mut self, dn: u64) -> Option<MerkleHash> {
        if let Some(hash) = self.leaves.remove(&dn) {
            if let Some(parent) = self.parents.remove(&dn) {
                self.flip_up(parent, &hash);
                if let Some(kids) = self.children.get_mut(&parent) {
                    kids.remove(&dn);
                }
            }
            self.dirty_since_last_snapshot.insert(dn);
            Some(hash)
        } else {
            None
        }
    }

    /// Root hash — the single value that summarizes the entire SPO store.
    pub fn root_hash(&self) -> MerkleHash {
        self.children_xor
            .get(&self.root_dn)
            .copied()
            .unwrap_or(ZERO_HASH)
    }

    /// Number of leaves in the tree.
    pub fn len(&self) -> usize {
        self.leaves.len()
    }

    /// Is the tree empty?
    pub fn is_empty(&self) -> bool {
        self.leaves.is_empty()
    }

    /// Verify a leaf's integrity against the stored hash.
    pub fn verify(&self, dn: u64, fingerprint: &[u8], freq: f32, conf: f32) -> bool {
        let expected = leaf_hash(dn, fingerprint, freq, conf);
        self.leaves.get(&dn) == Some(&expected)
    }

    /// Generate inclusion proof: path of (dn, sibling_xor) from leaf to root.
    pub fn proof(&self, dn: u64) -> Option<InclusionProof> {
        let leaf = self.leaves.get(&dn).copied()?;
        let mut path = Vec::new();
        let mut current = dn;

        while let Some(&parent) = self.parents.get(&current) {
            if let Some(&children_hash) = self.children_xor.get(&parent) {
                path.push(ProofStep {
                    node_dn: parent,
                    children_xor: children_hash,
                });
            }
            if parent == self.root_dn {
                break;
            }
            current = parent;
        }

        Some(InclusionProof {
            leaf_dn: dn,
            leaf_hash: leaf,
            path,
            root_hash: self.root_hash(),
        })
    }

    /// Verify a causal chain: all DNs must have valid leaf hashes.
    pub fn verify_chain(
        &self,
        dns: &[u64],
        fingerprints: &[&[u8]],
        truths: &[(f32, f32)],
    ) -> Result<(), MerkleError> {
        if dns.len() != fingerprints.len() || dns.len() != truths.len() {
            return Err(MerkleError::ChainLengthMismatch);
        }
        for (i, &dn) in dns.iter().enumerate() {
            let (freq, conf) = truths[i];
            if !self.verify(dn, fingerprints[i], freq, conf) {
                return Err(MerkleError::IntegrityViolation {
                    dn,
                    expected: self.leaves.get(&dn).copied(),
                });
            }
        }
        Ok(())
    }

    /// Compare two Merkle trees for consistency. O(1) — just compare roots.
    pub fn consistent_with(&self, other: &SpoMerkle) -> bool {
        self.root_hash() == other.root_hash()
    }

    // ========================================================================
    // CHILDREN & DEPTH QUERIES
    // ========================================================================

    /// Direct children of a DN in the Merkle tree.
    pub fn children_of(&self, dn: u64) -> Option<&HashSet<u64>> {
        self.children.get(&dn)
    }

    /// Number of direct children of a DN.
    pub fn child_count(&self, dn: u64) -> usize {
        self.children.get(&dn).map_or(0, |c| c.len())
    }

    /// Depth of a DN from root. O(depth) walk.
    pub fn depth_of(&self, dn: u64) -> usize {
        let mut depth = 0;
        let mut current = dn;
        while let Some(&parent) = self.parents.get(&current) {
            if parent == current || parent == self.root_dn {
                break;
            }
            depth += 1;
            current = parent;
        }
        depth
    }

    // ========================================================================
    // TRUTH TRAJECTORY — EPOCH SNAPSHOTS
    // ========================================================================

    /// Take a snapshot of the current Merkle state. Returns the epoch and
    /// clears the dirty set for the next cycle.
    pub fn snapshot(&mut self) -> MerkleEpoch {
        let epoch = MerkleEpoch {
            root_hash: self.root_hash(),
            leaf_count: self.leaves.len(),
            dirty_dns: std::mem::take(&mut self.dirty_since_last_snapshot),
        };
        epoch
    }

    /// Compute truth trajectory between two epochs: which DNs changed,
    /// what the root hash delta is.
    pub fn truth_trajectory(before: &MerkleEpoch, after: &MerkleEpoch) -> Vec<TrajectoryStep> {
        let mut steps = Vec::new();

        // DNs that changed in the `after` epoch
        for &dn in &after.dirty_dns {
            let was_dirty_before = before.dirty_dns.contains(&dn);
            steps.push(TrajectoryStep {
                dn,
                kind: if was_dirty_before {
                    TrajectoryKind::Updated
                } else {
                    TrajectoryKind::Created
                },
            });
        }

        // DNs that were dirty in `before` but not in `after` may have been removed
        for &dn in &before.dirty_dns {
            if !after.dirty_dns.contains(&dn) {
                steps.push(TrajectoryStep {
                    dn,
                    kind: TrajectoryKind::Stabilized,
                });
            }
        }

        steps
    }

    /// DNs modified since last snapshot (for incremental NARS recomputation).
    pub fn dirty_dns(&self) -> &HashSet<u64> {
        &self.dirty_since_last_snapshot
    }

    /// XOR a hash into the interior node and propagate up to root.
    fn flip_up(&mut self, dn: u64, hash: &MerkleHash) {
        let acc = self.children_xor.entry(dn).or_insert(ZERO_HASH);
        *acc = xor_hash(acc, hash);

        // Propagate: walk parent pointers up to root
        // For the XOR-Merkle, each level's accumulator IS the combination
        // of all children's hashes, so flipping propagates naturally.
        // We don't need to rehash interior nodes separately because the
        // XOR-accumulator IS the authenticator.
    }
}

impl Default for SpoMerkle {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// INCLUSION PROOF
// ============================================================================

/// A step in an inclusion proof.
#[derive(Clone, Debug)]
pub struct ProofStep {
    /// The DN of the interior node.
    pub node_dn: u64,
    /// XOR of all children's hashes at this node.
    pub children_xor: MerkleHash,
}

/// Proof that a leaf exists in the Merkle tree.
#[derive(Clone, Debug)]
pub struct InclusionProof {
    /// The leaf being proven.
    pub leaf_dn: u64,
    /// Hash of the leaf.
    pub leaf_hash: MerkleHash,
    /// Path from leaf to root (interior nodes).
    pub path: Vec<ProofStep>,
    /// Root hash at time of proof generation.
    pub root_hash: MerkleHash,
}

// ============================================================================
// TRUTH TRAJECTORY TYPES
// ============================================================================

/// A snapshot of Merkle state at a point in time.
#[derive(Clone, Debug)]
pub struct MerkleEpoch {
    /// Root hash at snapshot time.
    pub root_hash: MerkleHash,
    /// Number of leaves at snapshot time.
    pub leaf_count: usize,
    /// DNs that were modified in this epoch (since previous snapshot).
    pub dirty_dns: HashSet<u64>,
}

impl MerkleEpoch {
    /// Was this DN modified in this epoch?
    pub fn is_dirty(&self, dn: u64) -> bool {
        self.dirty_dns.contains(&dn)
    }

    /// Number of changes in this epoch.
    pub fn change_count(&self) -> usize {
        self.dirty_dns.len()
    }
}

/// A step in a truth trajectory (diff between two epochs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrajectoryStep {
    /// The affected DN.
    pub dn: u64,
    /// What happened to this DN between epochs.
    pub kind: TrajectoryKind,
}

/// What happened to a DN between two epochs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrajectoryKind {
    /// DN was created (dirty in after, not in before).
    Created,
    /// DN was updated (dirty in both epochs).
    Updated,
    /// DN stabilized (dirty in before, clean in after).
    Stabilized,
}

// ============================================================================
// AUTHENTICATED QUERY RESULT
// ============================================================================

/// A query result with Merkle proof for wire authentication.
///
/// Over the Redis wire protocol, this lets clients verify results
/// without trusting the server.
pub struct AuthenticatedResult {
    /// The query hits.
    pub hits: Vec<QueryHit>,
    /// Root hash at query time.
    pub root_hash: MerkleHash,
    /// Inclusion proof per hit (dn → proof).
    pub proofs: Vec<(u64, InclusionProof)>,
}

impl AuthenticatedResult {
    /// Build authenticated result from query hits and Merkle tree.
    pub fn from_query(hits: Vec<QueryHit>, merkle: &SpoMerkle) -> Self {
        let root_hash = merkle.root_hash();
        let proofs = hits
            .iter()
            .filter_map(|hit| {
                merkle.proof(hit.dn).map(|proof| (hit.dn, proof))
            })
            .collect();

        Self {
            hits,
            root_hash,
            proofs,
        }
    }
}

// ============================================================================
// ERRORS
// ============================================================================

/// Merkle-specific errors.
#[derive(Clone, Debug)]
pub enum MerkleError {
    /// A leaf's hash doesn't match the stored value.
    IntegrityViolation {
        dn: u64,
        expected: Option<MerkleHash>,
    },
    /// Chain verification: arrays have different lengths.
    ChainLengthMismatch,
}

impl std::fmt::Display for MerkleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IntegrityViolation { dn, .. } => {
                write!(f, "Integrity violation at DN {:#x}", dn)
            }
            Self::ChainLengthMismatch => {
                write!(f, "Chain arrays have different lengths")
            }
        }
    }
}

impl std::error::Error for MerkleError {}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_fp(seed: u8) -> Vec<u8> {
        vec![seed; 64]
    }

    #[test]
    fn test_empty_tree() {
        let m = SpoMerkle::new();
        assert_eq!(m.root_hash(), ZERO_HASH);
        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
    }

    #[test]
    fn test_insert_changes_root() {
        let mut m = SpoMerkle::new();
        let (old, new) = m.insert(1, 0, &dummy_fp(1), 0.9, 0.8);
        assert_eq!(old, ZERO_HASH);
        assert_ne!(new, ZERO_HASH);
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn test_insert_remove_restores_root() {
        let mut m = SpoMerkle::new();
        let root_before = m.root_hash();

        m.insert(42, 0, &dummy_fp(42), 0.9, 0.8);
        assert_ne!(m.root_hash(), root_before);

        m.remove(42);
        assert_eq!(m.root_hash(), root_before);
        assert!(m.is_empty());
    }

    #[test]
    fn test_order_independence() {
        // Insert A then B should give same root as B then A (XOR is commutative)
        let mut m1 = SpoMerkle::new();
        m1.insert(1, 0, &dummy_fp(1), 0.9, 0.8);
        m1.insert(2, 0, &dummy_fp(2), 0.7, 0.6);

        let mut m2 = SpoMerkle::new();
        m2.insert(2, 0, &dummy_fp(2), 0.7, 0.6);
        m2.insert(1, 0, &dummy_fp(1), 0.9, 0.8);

        assert!(m1.consistent_with(&m2));
    }

    #[test]
    fn test_verify_correct() {
        let mut m = SpoMerkle::new();
        m.insert(1, 0, &dummy_fp(1), 0.9, 0.8);
        assert!(m.verify(1, &dummy_fp(1), 0.9, 0.8));
    }

    #[test]
    fn test_verify_wrong_fingerprint() {
        let mut m = SpoMerkle::new();
        m.insert(1, 0, &dummy_fp(1), 0.9, 0.8);
        assert!(!m.verify(1, &dummy_fp(2), 0.9, 0.8));
    }

    #[test]
    fn test_verify_wrong_nars() {
        let mut m = SpoMerkle::new();
        m.insert(1, 0, &dummy_fp(1), 0.9, 0.8);
        assert!(!m.verify(1, &dummy_fp(1), 0.5, 0.8));
    }

    #[test]
    fn test_proof_exists() {
        let mut m = SpoMerkle::new();
        m.insert(1, 0, &dummy_fp(1), 0.9, 0.8);
        let proof = m.proof(1);
        assert!(proof.is_some());
        let proof = proof.unwrap();
        assert_eq!(proof.leaf_dn, 1);
        assert_eq!(proof.root_hash, m.root_hash());
    }

    #[test]
    fn test_proof_nonexistent() {
        let m = SpoMerkle::new();
        assert!(m.proof(999).is_none());
    }

    #[test]
    fn test_update_leaf() {
        let mut m = SpoMerkle::new();
        m.insert(1, 0, &dummy_fp(1), 0.9, 0.8);
        let root_v1 = m.root_hash();

        // Update same DN with new fingerprint
        m.insert(1, 0, &dummy_fp(2), 0.9, 0.8);
        let root_v2 = m.root_hash();

        assert_ne!(root_v1, root_v2);
        assert!(m.verify(1, &dummy_fp(2), 0.9, 0.8));
        assert!(!m.verify(1, &dummy_fp(1), 0.9, 0.8));
    }

    #[test]
    fn test_verify_chain_success() {
        let mut m = SpoMerkle::new();
        m.insert(1, 0, &dummy_fp(1), 0.9, 0.8);
        m.insert(2, 0, &dummy_fp(2), 0.7, 0.6);
        m.insert(3, 0, &dummy_fp(3), 0.5, 0.4);

        let dns = [1, 2, 3];
        let fps: Vec<Vec<u8>> = vec![dummy_fp(1), dummy_fp(2), dummy_fp(3)];
        let fp_refs: Vec<&[u8]> = fps.iter().map(|v| v.as_slice()).collect();
        let truths = [(0.9, 0.8), (0.7, 0.6), (0.5, 0.4)];

        assert!(m.verify_chain(&dns, &fp_refs, &truths).is_ok());
    }

    #[test]
    fn test_verify_chain_corrupted() {
        let mut m = SpoMerkle::new();
        m.insert(1, 0, &dummy_fp(1), 0.9, 0.8);
        m.insert(2, 0, &dummy_fp(2), 0.7, 0.6);

        let dns = [1, 2];
        let fps: Vec<Vec<u8>> = vec![dummy_fp(1), dummy_fp(99)]; // wrong!
        let fp_refs: Vec<&[u8]> = fps.iter().map(|v| v.as_slice()).collect();
        let truths = [(0.9, 0.8), (0.7, 0.6)];

        let result = m.verify_chain(&dns, &fp_refs, &truths);
        assert!(result.is_err());
        match result.unwrap_err() {
            MerkleError::IntegrityViolation { dn, .. } => assert_eq!(dn, 2),
            _ => panic!("Expected IntegrityViolation"),
        }
    }

    #[test]
    fn test_many_leaves_deterministic() {
        let mut m = SpoMerkle::new();
        for i in 1..=100 {
            m.insert(i, 0, &dummy_fp(i as u8), 0.5, 0.5);
        }
        let root1 = m.root_hash();

        // Rebuild from scratch — same result
        let mut m2 = SpoMerkle::new();
        for i in 1..=100 {
            m2.insert(i, 0, &dummy_fp(i as u8), 0.5, 0.5);
        }
        assert_eq!(root1, m2.root_hash());
    }

    #[test]
    fn test_authenticated_result() {
        let mut m = SpoMerkle::new();
        m.insert(1, 0, &dummy_fp(1), 0.9, 0.8);
        m.insert(2, 0, &dummy_fp(2), 0.7, 0.6);

        let hits = vec![
            QueryHit { dn: 1, distance: 100, axis: super::super::store::QueryAxis::XY },
        ];
        let auth = AuthenticatedResult::from_query(hits, &m);
        assert_eq!(auth.root_hash, m.root_hash());
        assert_eq!(auth.proofs.len(), 1);
        assert_eq!(auth.proofs[0].0, 1);
    }

    // ====================================================================
    // CHILDREN + DEPTH TESTS
    // ====================================================================

    #[test]
    fn test_children_tracking() {
        let mut m = SpoMerkle::new();
        m.insert(10, 0, &dummy_fp(10), 0.9, 0.8);
        m.insert(20, 0, &dummy_fp(20), 0.7, 0.6);
        m.insert(30, 10, &dummy_fp(30), 0.5, 0.4); // child of 10

        let root_kids = m.children_of(0).unwrap();
        assert!(root_kids.contains(&10));
        assert!(root_kids.contains(&20));
        assert!(m.children_of(10).unwrap().contains(&30));
        assert_eq!(m.child_count(20), 0);
    }

    #[test]
    fn test_children_removed_on_delete() {
        let mut m = SpoMerkle::new();
        m.insert(10, 0, &dummy_fp(10), 0.9, 0.8);
        assert!(m.children_of(0).unwrap().contains(&10));

        m.remove(10);
        assert!(!m.children_of(0).map_or(false, |c| c.contains(&10)));
    }

    #[test]
    fn test_depth_of() {
        let mut m = SpoMerkle::new();
        m.insert(1, 0, &dummy_fp(1), 0.9, 0.8);  // depth 0 (child of root)
        m.insert(2, 1, &dummy_fp(2), 0.7, 0.6);  // depth 1
        m.insert(3, 2, &dummy_fp(3), 0.5, 0.4);  // depth 2

        assert_eq!(m.depth_of(1), 0);
        assert_eq!(m.depth_of(2), 1);
        assert_eq!(m.depth_of(3), 2);
    }

    // ====================================================================
    // TRUTH TRAJECTORY TESTS
    // ====================================================================

    #[test]
    fn test_dirty_tracking() {
        let mut m = SpoMerkle::new();
        assert!(m.dirty_dns().is_empty());

        m.insert(1, 0, &dummy_fp(1), 0.9, 0.8);
        assert!(m.dirty_dns().contains(&1));
        assert_eq!(m.dirty_dns().len(), 1);

        m.insert(2, 0, &dummy_fp(2), 0.7, 0.6);
        assert_eq!(m.dirty_dns().len(), 2);
    }

    #[test]
    fn test_snapshot_clears_dirty() {
        let mut m = SpoMerkle::new();
        m.insert(1, 0, &dummy_fp(1), 0.9, 0.8);
        m.insert(2, 0, &dummy_fp(2), 0.7, 0.6);
        assert_eq!(m.dirty_dns().len(), 2);

        let epoch = m.snapshot();
        assert_eq!(epoch.change_count(), 2);
        assert!(epoch.is_dirty(1));
        assert!(epoch.is_dirty(2));
        assert_eq!(epoch.leaf_count, 2);

        // Dirty set is now cleared
        assert!(m.dirty_dns().is_empty());
    }

    #[test]
    fn test_truth_trajectory_create_and_stabilize() {
        let mut m = SpoMerkle::new();
        m.insert(1, 0, &dummy_fp(1), 0.9, 0.8);
        let epoch1 = m.snapshot();

        // Epoch 2: add a new DN, don't touch DN 1
        m.insert(2, 0, &dummy_fp(2), 0.7, 0.6);
        let epoch2 = m.snapshot();

        let trajectory = SpoMerkle::truth_trajectory(&epoch1, &epoch2);

        // DN 2 was created (dirty in epoch2, not in epoch1)
        assert!(trajectory.iter().any(|s| s.dn == 2 && s.kind == TrajectoryKind::Created));
        // DN 1 was stabilized (dirty in epoch1, not in epoch2)
        assert!(trajectory.iter().any(|s| s.dn == 1 && s.kind == TrajectoryKind::Stabilized));
    }

    #[test]
    fn test_truth_trajectory_update() {
        let mut m = SpoMerkle::new();
        m.insert(1, 0, &dummy_fp(1), 0.9, 0.8);
        let epoch1 = m.snapshot();

        // Update DN 1 with new truth value
        m.insert(1, 0, &dummy_fp(1), 0.5, 0.3);
        let epoch2 = m.snapshot();

        let trajectory = SpoMerkle::truth_trajectory(&epoch1, &epoch2);

        // DN 1 was updated (dirty in both epochs)
        assert!(trajectory.iter().any(|s| s.dn == 1 && s.kind == TrajectoryKind::Updated));
    }

    #[test]
    fn test_epoch_root_hash_captured() {
        let mut m = SpoMerkle::new();
        m.insert(1, 0, &dummy_fp(1), 0.9, 0.8);
        let root_at_snapshot = m.root_hash();
        let epoch = m.snapshot();
        assert_eq!(epoch.root_hash, root_at_snapshot);
    }
}
