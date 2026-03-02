# Addendum 04a: Merkle Root for Eineindeutigkeit Resolution

## Add to: `src/spo/clam_path.rs` (Phase 5, currently building)

## The Problem

Three queries converge on the same concept — semantic, explicit, and relational
all land in same σ-bands. That's Eineindeutigkeit (unique determination).
But WHERE does the concept live?

Without content-addressing, the address depends on which query found it first.
Three paths → three ClamPath addresses → three Redis keys → **duplication**.

## The Solution: ClamPath + MerkleRoot in word[0]

```
word[0] of CogRecord8K (u64, 64 bits):
┌────────────────────┬────────────────────────────────────┐
│  ClamPath (24 bits) │  MerkleRoot (40 bits)             │
│  HOW you got here   │  WHAT lives here                  │
│  navigation/lineage │  identity/canonical address        │
└────────────────────┴────────────────────────────────────┘

ClamPath  = structural path through B-tree (depth + split decisions)
MerkleRoot = truncated blake3 of content fingerprint → canonical identity
```

## Why Both

```
ClamPath alone:  path-dependent → same concept gets different addresses
MerkleRoot alone: no lineage, no subtree range queries, no causality chains
Together:         navigate via ClamPath, resolve identity via MerkleRoot
```

## MerkleRoot Derivation

The Merkle root is derived from the three-plane binary fingerprints.
This is deterministic: same concept = same fingerprint = same root.

```rust
use blake3;

/// Truncated Merkle root for content-addressed identity.
/// 40 bits = ~1 trillion collision space. Sufficient for BindSpace.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct MerkleRoot(pub u64); // only lower 40 bits used

impl MerkleRoot {
    /// Derive from three-plane binary fingerprints.
    /// The canonical address IS the content hash.
    pub fn from_planes(
        s_binary: &[u8; 2048],  // 16384-bit S-plane
        p_binary: &[u8; 2048],  // 16384-bit P-plane  
        o_binary: &[u8; 2048],  // 16384-bit O-plane
    ) -> Self {
        // Hash each plane separately, then combine.
        // This is a Merkle tree with 3 leaves.
        let s_hash = blake3::hash(s_binary);
        let p_hash = blake3::hash(p_binary);
        let o_hash = blake3::hash(o_binary);

        // Combine: hash(S || P || O) — order matters (S < P < O is canonical)
        let mut hasher = blake3::Hasher::new();
        hasher.update(s_hash.as_bytes());
        hasher.update(p_hash.as_bytes());
        hasher.update(o_hash.as_bytes());
        let root = hasher.finalize();

        // Truncate to 40 bits
        let bytes = root.as_bytes();
        let val = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], 0, 0, 0,
        ]);
        MerkleRoot(val & 0xFF_FFFF_FFFF) // mask to 40 bits
    }

    /// Derive from single composite fingerprint (backward compat).
    /// Used when planes aren't separated yet.
    pub fn from_fingerprint(fp: &[u8; 2048]) -> Self {
        let hash = blake3::hash(fp);
        let bytes = hash.as_bytes();
        let val = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], 0, 0, 0,
        ]);
        MerkleRoot(val & 0xFF_FFFF_FFFF)
    }

    /// The Redis key for this concept's canonical address.
    pub fn redis_key(&self) -> String {
        format!("ada:bind:{:010x}", self.0)
    }
}
```

## Packing into word[0]

```rust
impl ClamPath {
    /// Pack ClamPath (24 bits) + MerkleRoot (40 bits) into a single u64.
    pub fn pack_with_merkle(&self, root: MerkleRoot) -> u64 {
        let clam_bits = self.to_u24() as u64;       // 24 bits
        let merkle_bits = root.0 & 0xFF_FFFF_FFFF;  // 40 bits
        (clam_bits << 40) | merkle_bits
    }

    /// Unpack from word[0].
    pub fn unpack_with_merkle(word0: u64) -> (ClamPath, MerkleRoot) {
        let clam_bits = (word0 >> 40) as u32;        // upper 24 bits
        let merkle_bits = word0 & 0xFF_FFFF_FFFF;    // lower 40 bits
        (ClamPath::from_u24(clam_bits), MerkleRoot(merkle_bits))
    }
}
```

## Self-Healing Redis Address

This is the key insight for distributed operation:

```
1. Concept arrives → three-plane fingerprints computed
2. MerkleRoot derived deterministically: blake3(S || P || O)
3. Redis key = ada:bind:{merkle_root_hex}
4. ClamPath = structural location in B-tree

If Redis evicts the key:
  - Recompute MerkleRoot from fingerprints (deterministic)
  - Key is reconstructed: same content → same hash → same address
  - Tree heals itself

If two paths find the same concept:
  - Both compute same MerkleRoot (content-addressed)
  - Both write to same Redis key
  - Eineindeutigkeit: last-write-wins is fine because content is identical
  - ClamPaths may differ (different navigation routes) — that's OK,
    ClamPath is HOW, MerkleRoot is WHAT

If concept is modified (evidence updates the soaking register):
  - MerkleRoot changes (new content → new hash)
  - Old key naturally expires (TTL or eviction)
  - New key created at new address
  - No dangling references: anyone who had the old root will miss,
    recompute from current fingerprints, find the new address
```

## Merkle Tree for Subtree Integrity (Future)

When the hive goes multi-writer, extend the per-concept MerkleRoot
to a full Merkle tree over the CLAM subtree:

```
              root_hash
             /         \
      left_hash      right_hash
      /     \         /     \
   leaf_0  leaf_1  leaf_2  leaf_3
   (concept fingerprints)
```

Each CLAM split level has a hash that summarizes its children.
To verify a subtree after network partition:

```rust
/// Verify subtree integrity after reconnection.
/// Compare roots — if they match, entire subtree is consistent.
/// If they diverge, walk down to find the first differing leaf.
pub fn verify_subtree(
    local_root: MerkleRoot,
    remote_root: MerkleRoot,
    clam_path: &ClamPath,
) -> SubtreeStatus {
    if local_root == remote_root {
        SubtreeStatus::Consistent
    } else {
        // Walk the Merkle tree to find divergence point
        SubtreeStatus::Diverged { at_depth: /* compare level by level */ }
    }
}
```

But this is ice cake 21. For now: per-concept MerkleRoot in word[0] is sufficient.

## Dependency

```toml
# Cargo.toml
blake3 = "1"
```

Blake3 is fast (~1GB/s on modern CPUs), deterministic, and the truncated
40-bit output is sufficient for ~1 trillion address space.

## Integration with Existing ClamPath

The existing `04_btree_clam_path_lineage.md` spec defines ClamPath as u16 (16 bits)
with depth in a separate u8. Repack as u24 to fit alongside MerkleRoot in word[0]:

```rust
impl ClamPath {
    /// Pack bits (u16) + depth (u8) into 24 bits.
    pub fn to_u24(&self) -> u32 {
        ((self.depth as u32) << 16) | (self.bits as u32)
    }

    /// Unpack from 24 bits.
    pub fn from_u24(packed: u32) -> Self {
        ClamPath {
            bits: (packed & 0xFFFF) as u16,
            depth: ((packed >> 16) & 0xFF) as u8,
        }
    }
}
```

## Summary

| Component | Bits | Purpose |
|-----------|------|---------|
| ClamPath.bits | 16 | B-tree split decisions (navigation) |
| ClamPath.depth | 8 | Valid bit count (confidence/resolution) |
| MerkleRoot | 40 | Content-addressed canonical identity |
| **Total** | **64** | **= word[0] of CogRecord8K** |

ClamPath = HOW you got here.
MerkleRoot = WHAT lives here.
Redis key = WHERE it's stored.
Self-healing = WHY it works distributed.
