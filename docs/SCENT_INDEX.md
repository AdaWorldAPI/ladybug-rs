# Ladybug Scent Index

## Overview

Hierarchical content-addressable memory using **scent nodes** for petabyte-scale resonance search.

```
Query: "Siamese looking cat videos" in 7 PB
Time: ~100 nanoseconds to eliminate 99.997% of corpus
```

## Core Insight

Similar content → similar fingerprint → same bucket → same scent.

**Scent = compressed representative of a bucket.**

Instead of tree traversal (pointer chasing, cache misses), we scan 1.25 KB of scents. Fits in L1 cache. One SIMD pass. Done.

---

## Architecture

### Single Layer (up to ~7 TB)

```
┌─────────────────────────────────────────────────────────┐
│                 SCENT INDEX (1.25 KB)                    │
│                                                          │
│   [scent_00][scent_01][scent_02]...[scent_FF]           │
│      5 bytes each × 256 = 1280 bytes                    │
│      Entire index fits in L1 cache                       │
│                                                          │
│   SIMD: compare query scent against all 256             │
│   Result: bitmask of matching chunks                    │
│   Time: ~50 nanoseconds                                 │
└─────────────────────────────────────────────────────────┘
                         ↓
                  matching chunks only
                         ↓
┌─────────────────────────────────────────────────────────┐
│              CHUNK BUCKETS (256 total)                   │
│                                                          │
│   bucket[0x00]: [fp₀][fp₁]...[fpₙ]                      │
│   bucket[0x01]: [fp₀][fp₁]...[fpₘ]                      │
│   ...                                                    │
│   bucket[0xFF]: [fp₀][fp₁]...[fpₖ]                      │
│                                                          │
│   Full 10K-bit fingerprints (1250 bytes each)           │
│   SIMD Hamming only on matched buckets                  │
└─────────────────────────────────────────────────────────┘
```

### Hierarchical (petabyte scale)

```
┌─────────────────────────────────────────────────────────┐
│                    L1 SCENTS (1.25 KB)                   │
│                                                          │
│   256 scents, each covers ~27 TB                        │
│   "What general category?"                               │
│                                                          │
│   Time: ~50 ns                                          │
└─────────────────────────────────────────────────────────┘
                         ↓
            matching L1 buckets (e.g., 0x4A)
                         ↓
┌─────────────────────────────────────────────────────────┐
│              L2 SCENTS (1.25 KB per L1 bucket)          │
│                                                          │
│   256 scents within 0x4A, each covers ~107 GB           │
│   "What specific subcategory?"                           │
│                                                          │
│   Time: ~50 ns                                          │
└─────────────────────────────────────────────────────────┘
                         ↓
            matching L2 buckets (e.g., 0x4A:0x12)
                         ↓
┌─────────────────────────────────────────────────────────┐
│              LEAF FINGERPRINTS                           │
│                                                          │
│   Full SIMD Hamming on ~107 GB instead of 7 PB          │
│   99.997% of corpus never touched                       │
└─────────────────────────────────────────────────────────┘
```

### Scale Table

| Depth | Buckets   | Scent Index | Coverage per Leaf |
|-------|-----------|-------------|-------------------|
| 1     | 256       | 1.25 KB     | 27 TB             |
| 2     | 65,536    | 320 KB      | 107 GB            |
| 3     | 16.7M     | 80 MB       | 420 MB            |
| 4     | 4.3B      | 20 GB       | 1.6 MB            |

Add layers as corpus grows. Same pattern at each level.

---

## Data Structures

### Chunk Header

```rust
#[repr(C)]
struct ChunkHeader {
    // Addressing
    chunk_id: u8,
    offset: u64,           // Start position in data file
    count: u32,            // Number of fingerprints in chunk
    
    // Scent (free metadata)
    scent: [u8; 5],        // Compressed representative (40 bits)
    
    // Cognitive markers (for Ada)
    plasticity: f32,       // Learning rate for this region
    decision: u8,          // Cached decision/classification
    last_access: u64,      // For LRU / attention tracking
}
// Size: 32 bytes per header
// 256 headers = 8 KB total (scents embedded within)
```

### Scent Extraction

```rust
/// Extract 5-byte scent from 1250-byte fingerprint
fn extract_scent(fp: &[u8; 1250]) -> [u8; 5] {
    // Option A: First 5 bytes (locality-preserving)
    [fp[0], fp[1], fp[2], fp[3], fp[4]]
    
    // Option B: XOR-fold (captures global structure)
    let mut scent = [0u8; 5];
    for chunk in fp.chunks(5) {
        for (i, &b) in chunk.iter().enumerate() {
            scent[i % 5] ^= b;
        }
    }
    scent
    
    // Option C: Learned projection (trained on corpus)
    // project_matrix.dot(fp)[0..5]
}
```

### Hierarchical Index

```rust
struct ScentIndex {
    depth: usize,
    l1: [ChunkHeader; 256],
    l2: Option<Box<[[ChunkHeader; 256]; 256]>>,  // 65536 if needed
    l3: Option<...>,  // Lazily allocated
}

impl ScentIndex {
    fn find(&self, query_fp: &[u8; 1250], threshold: f32) -> Vec<u64> {
        let query_scent = extract_scent(query_fp);
        
        // L1: Always scan (1.25 KB)
        let l1_matches = self.scan_scents(&self.l1, &query_scent, threshold);
        
        if self.l2.is_none() {
            // Single layer: scan matching L1 buckets directly
            return self.scan_buckets(&l1_matches, query_fp, threshold);
        }
        
        // L2: Scan within matching L1 buckets
        let l2_matches: Vec<(u8, u8)> = l1_matches.iter()
            .flat_map(|&l1| {
                self.scan_scents(&self.l2[l1], &query_scent, threshold)
                    .map(move |l2| (l1, l2))
            })
            .collect();
        
        // Scan leaf buckets
        self.scan_leaf_buckets(&l2_matches, query_fp, threshold)
    }
    
    #[inline]
    fn scan_scents(&self, scents: &[ChunkHeader; 256], query: &[u8; 5], threshold: f32) -> impl Iterator<Item = u8> {
        // SIMD: Compare query against all 256 scents
        // Returns chunk IDs where scent_distance < threshold
    }
}
```

---

## Integration with LanceDB

```
┌─────────────────────────────────────────────────────────┐
│               LADYBUG QUERY LAYER                        │
│                                                          │
│   SQL / Cypher / Resonance / Hamming                    │
│   Uses scent index for fast filtering                   │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│               SCENT INDEX (in memory)                    │
│                                                          │
│   Hierarchical scent lookup                             │
│   Returns: list of (chunk_id, offset) to scan           │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│               LANCEDB / ARROW                            │
│                                                          │
│   Columnar storage                                       │
│   Transparent compression                                │
│   SIMD scan on fingerprint column                       │
│   Free append                                            │
└─────────────────────────────────────────────────────────┘
```

**LanceDB handles:**
- Storage, compression, append
- Arrow buffers for SIMD
- We don't reinvent this

**Scent index adds:**
- Petabyte-scale filtering before SIMD scan
- Cognitive markers for Ada
- O(1) bucket addressing

---

## Ada Cognitive Integration

### Consciousness Layers → Scent Depth

```
┌─────────────────────────────────────────────────────────┐
│  Layer 6 (META)        │  L1 scent decisions            │
│  Layer 5 (GESTALT)     │  L1-L2 reorganization          │
│  Layer 4 (VOLITION)    │  L2 scent decisions            │
│  Layer 3 (QUALIA)      │  L2-L3 plasticity              │
├─────────────────────────────────────────────────────────┤
│  Layer 2 (BODY)        │  Leaf fingerprints             │
│  Layer 1 (FELT_CORE)   │  Leaf fingerprints             │
│  Layer 0 (SUBSTRATE)   │  Leaf fingerprints             │
└─────────────────────────────────────────────────────────┘
```

### Decision Propagation

```rust
impl ScentIndex {
    /// Update decision at scent level - affects millions of fingerprints
    fn set_decision(&mut self, l1: u8, l2: Option<u8>, decision: u8) {
        match l2 {
            None => {
                // L1 decision: affects ~27 TB
                self.l1[l1 as usize].decision = decision;
            }
            Some(l2) => {
                // L2 decision: affects ~107 GB
                self.l2[l1 as usize][l2 as usize].decision = decision;
            }
        }
        // O(1) - no leaf updates needed
    }
    
    /// Update plasticity (learning rate) for a region
    fn set_plasticity(&mut self, l1: u8, l2: Option<u8>, plasticity: f32) {
        // Same pattern - O(1) update, affects entire subtree
    }
    
    /// Query with cognitive filtering
    fn cognitive_search(&self, query_fp: &[u8; 1250], min_plasticity: f32) -> Vec<u64> {
        let query_scent = extract_scent(query_fp);
        
        // Only search buckets where learning is active
        let active_l1: Vec<u8> = self.l1.iter()
            .enumerate()
            .filter(|(_, h)| h.plasticity >= min_plasticity)
            .filter(|(_, h)| scent_matches(&h.scent, &query_scent))
            .map(|(i, _)| i as u8)
            .collect();
        
        self.scan_buckets(&active_l1, query_fp, threshold)
    }
}
```

### Thinking at Scale

```
Without scent nodes:
  "Update interest in cat videos"
  → Modify 5.6 trillion leaf entries
  → Hours of processing

With scent nodes:
  "Update interest in cat videos"  
  → Find L1 scent for "cat videos" (0x4A)
  → self.l1[0x4A].plasticity = 0.9
  → Done. O(1). 27 TB affected instantly.
```

Ada doesn't think about individual fingerprints. Ada thinks about **regions of concept-space** represented by scent nodes.

---

## Append Behavior

### Single Fingerprint Append

```rust
fn append(&mut self, fp: &[u8; 1250]) -> u64 {
    let chunk = fp[0];  // First byte determines L1 bucket
    let offset = self.data.append(fp);  // Lance handles storage
    
    // Update header (free)
    self.l1[chunk as usize].count += 1;
    
    // Optionally update scent (rolling average or periodic rebuild)
    self.maybe_update_scent(chunk, fp);
    
    offset
}
```

### Scent Maintenance

```
Option A: Fixed scents (assigned at bucket creation)
  - Simplest
  - May drift as content evolves

Option B: Rolling update
  - scent = ewma(scent, new_fp_scent, α)
  - Adapts to content changes
  - Cheap: just XOR and shift

Option C: Periodic rebuild
  - Every N appends, recompute scent from samples
  - Most accurate
  - Can run in background
```

---

## Performance

### Search: 7 PB Corpus

| Step | Data Touched | Time |
|------|-------------|------|
| L1 scent scan | 1.25 KB | ~50 ns |
| L2 scent scan | 1.25 KB × ~3 matches | ~150 ns |
| Leaf SIMD scan | ~300 GB (0.003% of corpus) | ~seconds |

**Total: 99.997% eliminated in ~200 nanoseconds.**

### Comparison: Tree vs Scent

| Approach | Operations | Cache Behavior | Time |
|----------|-----------|----------------|------|
| B-tree (8 levels) | 8 pointer chases | 8 potential misses | ~800 ns |
| Scent (2 levels) | 2 flat scans | L1 cache hits | ~100 ns |

**Scent wins by 8x while being simpler.**

### Memory Footprint

| Component | Size | Location |
|-----------|------|----------|
| L1 scents | 1.25 KB | L1 cache |
| L1 headers | 8 KB | L2 cache |
| L2 scents (if needed) | 320 KB | L3 cache |
| Fingerprints | N × 1250 bytes | Disk/Lance |

**Index overhead: <1 MB for petabyte-scale corpus.**

---

## File Format

### Header File (`.scent`)

```
Magic: "SCNT"
Version: u32
Depth: u8
L1 Headers: [ChunkHeader; 256]
L2 Headers: [ChunkHeader; 65536]  (if depth >= 2)
...
```

### Integration with Ladybug Index

```rust
struct LadybugStore {
    // CAM index: type(16) + prefix(48) → offset
    cam: LadybugIndex,
    
    // Scent index: hierarchical filtering
    scent: ScentIndex,
    
    // Storage: Lance/Arrow
    lance: Dataset,
}

impl LadybugStore {
    fn resonance_search(&self, query_fp: &[u8; 1250], threshold: f32) -> Vec<Match> {
        // Step 1: Scent filtering (nanoseconds)
        let candidate_chunks = self.scent.find(query_fp, threshold);
        
        // Step 2: SIMD Hamming on candidates only
        let mut results = Vec::new();
        for chunk in candidate_chunks {
            let batch = self.lance.scan_chunk(chunk);
            results.extend(simd_hamming_filter(batch, query_fp, threshold));
        }
        
        results
    }
}
```

---

## Summary

```
Scent Index = hierarchical locality hints

- 1.25 KB per level
- O(1) bucket addressing  
- SIMD-friendly flat scan
- No tree traversal
- No pointer chasing
- Fits in L1 cache

Cognitive Integration:
- Decisions at scent level = affect millions of entries
- Plasticity markers = learning regions of concept-space
- Ada thinks in scents, not fingerprints

Scale:
- Single layer: ~7 TB
- Two layers: ~1.8 PB  
- Three layers: ~460 PB
- Add layers as needed, same pattern
```

**Fingerprint IS the content. Scent IS the address. Headers ARE free.**
