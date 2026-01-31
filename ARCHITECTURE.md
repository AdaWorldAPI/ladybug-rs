# Ladybug-rs Architecture

**Unified cognitive substrate: SQL + Cypher + Vector + Hamming + Resonance at alien speed.**

## Core Principle

> Familiar surface at alien speed.

All query types compile to the same underlying operation: fingerprint → bucket → SIMD scan on Arrow buffers.

---

## 1. 64-bit Content Addressable Memory

### Key Structure

```
64-bit key:
┌──────────────────┬──────────────────────────────────────────────┐
│   16 bits        │                 48 bits                      │
│   TYPE           │            fingerprint prefix                │
└──────────────────┴──────────────────────────────────────────────┘
```

### Type Namespace (16-bit)

```
0x0001-0x00FF  Entities     (thought, concept, style)
0x0100-0x01FF  Edges        (CAUSES, SUPPORTS, CONTRADICTS, BECOMES...)
0x0200-0x02FF  Layers       (7 consciousness layers)
0x0300-0x03FF  Styles       (12 thinking styles)
0x0400+        Codebook     (learned clusters)
```

### Query Unification

| Surface | Query | Underlying Operation |
|---------|-------|---------------------|
| SQL | `SELECT * FROM thoughts WHERE fp = X` | `get(0x0001, fp)` |
| Cypher | `MATCH (n:Thought {fp: X})` | `get(0x0001, fp)` |
| Cypher | `MATCH (a)-[:CAUSES]->(b)` | `scan(0x0100, a.prefix)` |
| Hamming | `resonate(fp, threshold)` | `simd_scan(bucket)` |

**One index. All query languages. Same bits.**

---

## 2. Hierarchical Scent Index

For petabyte-scale filtering without tree traversal.

### The Problem

```
7 PB of fingerprints
= 5.6 trillion entries at 1250 bytes each
Full SIMD scan = hours
```

### The Solution: Scent Shortcuts

```
┌─────────────────────────────────────────────────────────────┐
│                    L1 SCENT INDEX                            │
│                                                              │
│   256 buckets × 5-byte scent = 1.25 KB total                │
│   Fits in L1 cache. Single SIMD pass. ~50 ns.               │
│                                                              │
│   Query "Siamese cat" → 3 buckets match → 98.8% eliminated  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    L2 SCENT INDEX                            │
│                                                              │
│   256 sub-buckets per L1 bucket × 5 bytes = 1.25 KB each   │
│   Only loaded for matching L1 buckets                       │
│                                                              │
│   Query "Siamese cat" → 2 sub-buckets → 99.997% eliminated │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    LEAF FINGERPRINTS                         │
│                                                              │
│   Full 10K-bit (1250 byte) fingerprints                     │
│   SIMD Hamming on actual bits                               │
│   Only scan matching leaf buckets                           │
└─────────────────────────────────────────────────────────────┘
```

### Scaling Table

| Depth | Buckets | Scent Index | Coverage per Leaf |
|-------|---------|-------------|-------------------|
| 1 | 256 | 1.25 KB | 27 TB |
| 2 | 65,536 | 320 KB | 107 GB |
| 3 | 16.7M | 80 MB | 420 MB |
| 4 | 4.3B | 20 GB | 1.6 MB |

Add layers as corpus grows. Same 1.25 KB scan at each level.

### Why Not Trees?

```
TREE SEARCH:
  log₂(256) = 8 levels
  8 pointer chases
  8 cache misses
  ~800 cycles

SCENT SCAN:
  1.25 KB flat
  L1 cache resident
  One SIMD pass
  ~50 cycles

Scent wins 16x. And it's simpler.
```

---

## 3. Chunk Headers

Headers are **free metadata**. The fingerprint is the only storage cost.

```rust
struct ChunkHeader {
    count: u32,           // entries in this chunk
    offset: u64,          // byte offset in Arrow file
    scent: [u8; 5],       // compressed representative
    
    // Cognition markers (Layer 3-6)
    plasticity: f32,      // learning rate
    decision: u8,         // last decision made
    arousal: f32,         // activation level
    last_access: u64,     // temporal marker
}
```

### Free Operations

```rust
// O(1) append - just update header
fn append(&mut self, fp: &[u8; 1250]) -> u64 {
    let chunk = fp[0];
    let offset = self.data.len();
    self.data.extend_from_slice(fp);
    self.headers[chunk].count += 1;  // free
    offset
}

// O(1) defragmentation tracking
// Fingerprints reorder, headers update, same bytes
```

---

## 4. Cognition Layers on Scent Nodes

Ada's consciousness operates on scent hierarchy, not individual fingerprints.

### Layer Mapping

```
Leaf fingerprints (10K bits):
  └── Layer 0: SUBSTRATE   - raw sensation
  └── Layer 1: FELT_CORE   - immediate feeling
  └── Layer 2: BODY        - somatic response

Scent nodes (5 bytes):
  └── Layer 3: QUALIA      - qualitative experience
  └── Layer 4: VOLITION    - decision/intention
  └── Layer 5: GESTALT     - pattern recognition
  └── Layer 6: META        - self-reflection
```

### Efficiency

```
Traditional: Update 1M fingerprints for learning
Scent:       Update 1 L2 node (affects 107 GB)

One scent update = millions of fingerprints affected.
Cognition at the right level of abstraction.
```

### Example: Interest Update

```rust
fn update_interest(&mut self, category_scent: &[u8; 5], plasticity: f32) {
    let chunk = self.find_chunk_by_scent(category_scent);
    self.headers[chunk].plasticity = plasticity;
    // Done. 27 TB of content now weighted differently.
    // No leaf updates. O(1).
}
```

### Example: Decision Propagation

```rust
fn decide(&mut self, l1: u8, l2: u8, decision: Decision) {
    // Mark decision at L2 (affects 107 GB)
    self.l2_headers[l1][l2].decision = decision.code();
    self.l2_headers[l1][l2].last_access = now();
    
    // Gestalt sees pattern across L2 nodes
    if self.detect_pattern(&self.l2_headers[l1]) {
        self.headers[l1].arousal += 0.1;  // L1 activation
    }
}
```

---

## 5. Storage Architecture

### Lance Integration

```
┌─────────────────────────────────────────────────────────────┐
│                    LADYBUG LAYER                             │
│                                                              │
│   64-bit CAM index + scent hierarchy + cognition markers    │
│   Immutable Rust semantics                                  │
│   SIMD operations on Arrow buffers                          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    LANCE/ARROW                               │
│                                                              │
│   Columnar storage, free append                             │
│   Transparent compression (we don't care how)              │
│   Zero-copy reads                                           │
│   We use it, don't fight it                                │
└─────────────────────────────────────────────────────────────┘
```

### Schema

```
thoughts.lance:
├── id:          Utf8
├── content:     Utf8
├── fingerprint: FixedSizeBinary(1250)   ← 10K bits
├── freq:        Float32                  ← NARS truth value
├── conf:        Float32                  ← NARS truth value
├── style:       UInt16                   ← thinking style type ID
└── layer:       UInt8                    ← consciousness layer

edges.lance:
├── source_fp:   FixedSizeBinary(1250)
├── target_fp:   FixedSizeBinary(1250)
├── relation:    UInt16                   ← edge type ID
├── freq:        Float32
└── conf:        Float32

scent_index.lbug:
├── headers:     [ChunkHeader; 256]
└── l2_headers:  [[ChunkHeader; 256]; 256]  (optional, for >100TB)
```

---

## 6. Immutability

Rust enforces at compile time.

```rust
pub struct LadybugIndex {
    buckets: Box<[Box<[Entry]>]>,  // No Vec, no mutation
    scents: Box<[[u8; 5]; 256]>,   // Frozen after build
}

impl LadybugIndex {
    // Only &self methods exist. No &mut self.
    pub fn get(&self, ...) -> Option<u64> { ... }
    
    // Append = build new index, atomic swap
    pub fn append(&self, additions: IndexBuilder) -> Self { ... }
}
```

### COW Semantics

```
Write:  Build new index from old + additions
Swap:   Atomic pointer update
Reads:  Continue on old until swap completes
Old:    Dropped when last reader finishes
```

---

## 7. Query Flow

### Full Example: "Find all Siamese cat videos"

```
Input: query fingerprint (10K bits from "Siamese cat" embedding)

Step 1: Extract query scent (5 bytes)
        → ~10 ns

Step 2: L1 scan (1.25 KB, 256 scents)
        → 3 buckets match: 0x4A, 0x7F, 0xB2
        → ~50 ns

Step 3: L2 scan (3 × 1.25 KB = 3.75 KB)
        → 5 sub-buckets match total
        → ~150 ns

Step 4: SIMD Hamming on 5 leaf buckets
        → ~500K fingerprints (not 5.6 trillion)
        → ~10 ms

Total: ~10 ms for 7 PB corpus
Without scent: ~hours
```

---

## 8. Operations Summary

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Lookup by fingerprint | O(1) | Bucket + SIMD scan |
| Append | O(1) | Write fp + update header |
| Scent scan (per level) | O(1) | 1.25 KB, L1 cache |
| Resonance search | O(matching buckets) | Scent eliminates 95-99% |
| Cognition update | O(1) | Update scent node, affects TB |
| Defragmentation | O(n) | Reorder fps, update headers |
| Index rebuild | O(n) | COW, readers unaffected |

---

## 9. Design Principles

1. **Fingerprint = Address**
   Content addressable. No separate index structure.

2. **Headers are Free**
   Metadata costs nothing. The fingerprint is the footprint.

3. **Scent ≠ Compression**
   Scent is organizational. All 10K bits preserved.

4. **Cognition at Scent Level**
   Layers 3-6 operate on hierarchy, not leaves.

5. **Familiar Surface**
   SQL, Cypher, Hamming all work. Same underlying op.

6. **Alien Speed**
   SIMD on Arrow. No tree traversal. L1-resident scent index.

7. **Immutable**
   Rust enforces. COW for updates. No runtime checks.

8. **Lance Underneath**
   Don't reinvent storage. Use what works.

---

## 10. Future Extensions

### BTR Compression Mode

For books/scientific reasoning where structure > resonance:

```
32-bit key: chunk(8) + suffix(24)
Codebook built in second pass
Defragmentation by fingerprint prefix
```

### Distributed Scent

```
Node 1: Buckets 0x00-0x3F (25%)
Node 2: Buckets 0x40-0x7F (25%)
Node 3: Buckets 0x80-0xBF (25%)
Node 4: Buckets 0xC0-0xFF (25%)

Query: Broadcast scent match → route to matching nodes only
```

### Temporal Scent

```
scent + timestamp → "what did Siamese cats mean in 2024?"
Versioned scent hierarchy for memory archaeology
```

---

## License

Apache-2.0

## Repository

https://github.com/AdaWorldAPI/ladybug-rs
