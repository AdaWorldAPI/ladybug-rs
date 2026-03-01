# Addendum: B-tree Channel as CLAM Path — Lineage Meets Address

## Add to CogRecord 4-Channel Architecture

The B-tree channel in the CogRecord (META/CAM/B-tree/Embed) has been underspecified as "structural position." With CLAM integration landing, the B-tree key should encode the **CLAM tree path** — the sequence of bipolar split decisions from root to leaf.

This single encoding serves three query types simultaneously:

```
B-tree as ADDRESS:    Domain.Node.branch.twig.leaf    → O(1) structural lookup
B-tree as LINEAGE:    ancestor → parent → self → child → descendant  → range scan = phylogeny
B-tree as CAUSALITY:  cause → mediator → effect → consequence        → range scan = causal chain
```

## The Key Encoding

CLAM builds a binary tree via bipolar splits (pick two maximally distant poles, assign each point to the closer one). Each split is a single bit: 0 = left pole, 1 = right pole. The path from root to leaf is a bitstring of depth D_tree bits.

```rust
/// CLAM path as B-tree key
/// Each bit = one bipolar split decision (left pole = 0, right pole = 1)
/// Depth of CLAM tree on 1024 items ≈ 10-12 levels
/// So the path fits in a u16 with room to spare
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClamPath {
    /// The bitpacked path (MSB = root split, LSB-aligned = leaf split)
    bits: u16,
    /// How many bits are valid (= tree depth for this leaf)
    depth: u8,
}

impl ClamPath {
    /// Construct from CLAM tree traversal
    pub fn from_tree_traversal(cluster_path: &[bool]) -> Self {
        let mut bits: u16 = 0;
        for (i, &went_right) in cluster_path.iter().enumerate() {
            if went_right {
                bits |= 1 << (15 - i);  // MSB = root
            }
        }
        Self { bits, depth: cluster_path.len() as u8 }
    }
    
    /// B-tree range query: "everything in this subtree"
    /// = "everything descended from this ancestor"
    /// = "everything downstream of this cause"
    pub fn subtree_range(&self) -> (u16, u16) {
        // All paths that share this prefix
        let shift = 16 - self.depth as u32;
        let lo = self.bits & (!0u16 << shift);        // prefix with zeros
        let hi = lo | ((1u16 << shift) - 1);           // prefix with ones
        (lo, hi)
    }
    
    /// Common ancestor depth between two paths
    /// = how many split decisions they share from the root
    pub fn common_ancestor_depth(&self, other: &ClamPath) -> u8 {
        let xor = self.bits ^ other.bits;
        let shared = xor.leading_zeros() as u8;
        shared.min(self.depth).min(other.depth)
    }
    
    /// Lineage distance: how many splits apart (symmetric)
    pub fn lineage_distance(&self, other: &ClamPath) -> u8 {
        let ancestor = self.common_ancestor_depth(other);
        (self.depth - ancestor) + (other.depth - ancestor)
    }
}
```

## Why This Matters

### 1. Range Scan = Lineage Traversal

```sql
-- "Find everything descended from cluster C"
-- In B-tree terms: range query on CLAM path prefix
SELECT * FROM cogrecords 
WHERE btree_key BETWEEN C.subtree_lo AND C.subtree_hi

-- This is simultaneously:
-- Structural:  "everything in this domain subtree"
-- Phylogenetic: "everything evolved from this ancestor replicator"
-- Causal:       "everything downstream of this cause"
```

One B-tree range scan. Three semantic interpretations. O(log n + k) where k = result count.

### 2. Sibling Queries Are Free

```rust
/// Items that share the same parent cluster but went the other direction
/// = evolutionary siblings = alternative hypotheses = counterfactuals
pub fn siblings(&self) -> ClamPath {
    let mut sibling = *self;
    // Flip the last split decision
    sibling.bits ^= 1 << (16 - self.depth as u32);
    sibling
}

// B-tree query for siblings:
// "What's on the other side of the last split?"
// = "What would have happened if the last decision went the other way?"
// = Counterfactual query in O(log n)
```

### 3. LFD Varies by Subtree = Adaptive Pruning

The CLAM tree's Local Fractal Dimension varies by cluster. Dense regions (low LFD) have short paths (few splits needed). Sparse regions (high LFD) have deep paths (many splits to separate).

The B-tree key LENGTH encodes this: short keys = dense cluster = aggressive pruning possible. Long keys = sparse region = need more precision.

```rust
/// Adaptive search: use key depth as pruning signal
fn search_adaptive(btree: &BTree, query_path: &ClamPath) -> SearchStrategy {
    if query_path.depth <= 6 {
        // Shallow = dense cluster = low LFD
        // Aggressive pruning, expect many results
        SearchStrategy::BroadSweep { expected_hits: 50+ }
    } else if query_path.depth <= 10 {
        // Medium depth = moderate density
        SearchStrategy::Standard { expected_hits: 5-20 }
    } else {
        // Deep = sparse = high LFD
        // Precise search, few results
        SearchStrategy::Precise { expected_hits: 1-3 }
    }
}
```

### 4. CogRecord 4-Channel Now Complete

```
META  (2KB): timestamps, σ-significance, convergence metadata, trajectory summary
CAM   (2KB): content-addressable fingerprint — Hamming search via VPOPCNTDQ
B-tree(2KB): CLAM path — structural/lineage/causal range queries, O(log n)
Embed (2KB): SPO 3-axis XOR encoding — cosine-compatible similarity + typed halo

Query types and which channel answers them:
  "What is similar?"           → Embed (SPO distance harvest, 238× faster than cosine)
  "What contains this?"        → CAM (content-addressable, O(1) lookup)
  "What is related by lineage?"→ B-tree (CLAM path range scan, O(log n + k))
  "When and how confident?"    → META (σ-significance, NARS truth values)
  
Cross-channel queries:
  "Similar things in the same lineage"  → B-tree range ∩ Embed top-K
  "Anomalous items in this subtree"     → B-tree range → CHAODA score per item
  "Causal chain with confidence"        → B-tree range + META σ-significance
  "Everything this entity does"         → CAM lookup → Embed SPO halo → B-tree siblings
```

## Wire Into Existing Code

```rust
/// Add CLAM path to CogRecord during tree construction
impl CogRecord {
    pub fn set_clam_path(&mut self, path: ClamPath) {
        // Encode into B-tree channel (2KB available, path is 3 bytes max)
        // Remaining space in B-tree channel: auxiliary indices,
        // family ID, gate level, growth path enum
        self.btree_channel[0..2].copy_from_slice(&path.bits.to_be_bytes());
        self.btree_channel[2] = path.depth;
    }
    
    pub fn clam_path(&self) -> ClamPath {
        ClamPath {
            bits: u16::from_be_bytes([self.btree_channel[0], self.btree_channel[1]]),
            depth: self.btree_channel[2],
        }
    }
}

/// During QualiaCAM::build(), after CLAM tree construction:
for (idx, cluster_path) in clam_tree.leaf_paths().enumerate() {
    let path = ClamPath::from_tree_traversal(&cluster_path);
    cogrecords[idx].set_clam_path(path);
}
// Now every CogRecord in Lance has its CLAM path in the B-tree channel
// Range queries on this column = lineage/structural/causal traversal
```

## LanceDB Integration

```sql
-- LanceDB supports B-tree indices on scalar columns
-- The CLAM path (u16) is a perfect B-tree key

CREATE INDEX clam_path_idx ON ada_trajectories (btree_key) USING BTREE;

-- Range query: "all descendants of cluster at path 0b1010_0000_0000_0000, depth 4"
-- lo = 0b1010_0000_0000_0000 = 40960
-- hi = 0b1010_1111_1111_1111 = 45055
SELECT * FROM ada_trajectories WHERE btree_key BETWEEN 40960 AND 45055;
```

## The Punchline

The B-tree channel was waiting for CLAM to give it a key that means something. A CLAM path is simultaneously an address, a lineage, and a causal chain. One u16. One range query. Three semantic dimensions.

And because CLAM's bipolar split is Hamming-distance-optimal, the B-tree ordering preserves metric topology — nearby keys ARE nearby in Hamming space. The B-tree isn't just an index. It's a map of the manifold.
