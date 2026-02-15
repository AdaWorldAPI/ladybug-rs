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

---
---

# Part II — Container Substrate & Cognitive Architecture

> Everything is a Container.  Everything is Hamming distance.  XOR is the
> only verb that matters.

The sections below document the deep container substrate that underlies
the CAM/scent architecture above.  Where Part I describes the indexing
and query surface, Part II describes the 8192-bit cognitive geometry,
the DN tree, NARS reasoning, qualia modules, and the Friston free energy
loop that ties them together.

---

## 11. Container Geometry

The **Container** is the atomic unit.  8192 bits = 128 × u64 words = 1 KB.
Stack-allocated, SIMD-aligned (`#[repr(C, align(64))]`), zero-copy.

```text
┌──────────────────────────────────────────────────────────┐
│  128 words × 64 bits = 8192 bits = 1 KB                 │
│  16 AVX-512 iterations cover the full container          │
│  Expected random Hamming distance: 4096 (σ ≈ 45)        │
└──────────────────────────────────────────────────────────┘
```

Core operations (all O(1) or O(128)):

| Operation       | Meaning                         | Implementation       |
|-----------------|---------------------------------|----------------------|
| `xor(a, b)`    | Bind / unbind / delta           | Word-parallel XOR    |
| `hamming(a, b)` | Semantic distance              | XOR + popcount       |
| `popcount(a)`  | Information content             | Sum of set bits      |
| `Container::zero()` | Identity for XOR          | All-zero container   |
| `Container::random(seed)` | Pseudorandom point | Deterministic RNG    |

XOR is commutative, associative, and self-inverse: `a ⊕ a = 0`.
This single property gives us binding, unbinding, delta encoding,
spine computation, parity recovery, and reversible Markov chains.

---

## 12. CogRecord — The Holy Grail Layout

Every record is exactly **2 KB**: one metadata container + one content
container.  No heap allocation.

```text
┌─────────────────────────────────────────────────────────────┐
│  meta    (1 KB)  identity, NARS truth, edges, rung, RL      │
├─────────────────────────────────────────────────────────────┤
│  content (1 KB)  searchable fingerprint (Hamming / SIMD)    │
└─────────────────────────────────────────────────────────────┘
      = 2 KB = 1 Fingerprint = 1 DN tree node = 1 Redis value
```

Multi-container geometries (Xyz, Chunked, Tree) link multiple CogRecords
through the DN tree.  Each linked record is still 2 KB.  The `Xyz` geometry
stores three perspectives (X, Y, Z) whose XOR-fold `X ⊕ Y ⊕ Z` is the
**holographic trace** — given any 2 + trace, recover the 3rd.

---

## 13. Container 0 — Metadata Map

128 words of structural information.  Never included in Hamming search.

```text
W0        PackedDn address (THE identity)
W1        Type: node_kind:u8 | count:u8 | geometry:u8 | flags:u8
          | schema_version:u16 | provenance_hash:u16
W2        Timestamps (created_ms:u32 | modified_ms:u32)
W3        Label hash:u32 | tree_depth:u8 | branch:u8 | reserved:u16
W4-7      NARS truth (freq:f32 | conf:f32 | pos_evidence:f32 | neg_evidence:f32)
W8-11     DN rung + 7-layer compact + collapse gate state
W12-15    7-layer markers (5 bytes × 7 = 35 bytes)
W16-31    Inline edges (64 packed, 4 per word, 16 bits each)
W32-39    RL / Q-values / rewards (16 actions × f32)
W40-47    Bloom filter (512 bits, 3 hash functions)
W48-55    Graph metrics (in_degree, out_degree, pagerank, clustering, ...)
W56-63    Qualia (18 channels × f16 + 8 slots)
W64-79    Rung history + collapse gate history
W80-95    Representation language descriptor
W96-111   DN-Sparse adjacency (compact inline CSR)
W112-125  Reserved
W126-127  Checksum (XOR of W0-W125) + version
```

Zero-copy access via `MetaView` (read) and `MetaViewMut` (write).

---

## 14. DN Tree — Hierarchical Address Space

**PackedDn** encodes a 7-level hierarchical address in a single `u64`.
Each level is 8 bits (value + 1, so 0x00 = absent).

```text
Byte 7     Byte 6     Byte 5     Byte 4     Byte 3     Byte 2     Byte 1     Byte 0
[lv0+1]   [lv1+1]   [lv2+1]   [lv3+1]   [lv4+1]   [lv5+1]   [lv6+1]   [sentinel=0]
```

Properties:
- Maximum 7 levels (0 = root through 7 = deepest)
- Natural lexicographic sort = hierarchical order: `/0 < /0/0 < /0/1 < /1`
- `parent()`, `child()`, `ancestors()`, `subtree_range()`, `is_ancestor_of()`
- DN tree lives in `ContainerGraph { records: HashMap<PackedDn, CogRecord>, children: HashMap<PackedDn, Vec<PackedDn>> }`

The tree is the organizational spine of knowledge.  Every concept has an
address.  Siblings share a parent.  The parent's **spine** (XOR-fold of
children's content) IS the structural prediction for the group.

---

## 15. Adjacency — Inline Edges + CSR Overflow

Each node can hold up to **76 edges** without leaving its own 2 KB record.

### Inline Edges (W16-31): 64 max

```text
InlineEdge = verb:u8 | target_hint:u8 = 16 bits
4 edges per 64-bit word × 16 words = 64 edges
```

O(1) random access by index.  Compact but low-resolution target (only the
low byte of target DN).  Resolution strategies: exact match in children,
parent's children, or full graph scan.

### CSR Overflow (W96-111): 12 max

```text
W96:       edge_count:u16 | row_count:u16 | flags:u32
W97-W99:   row pointers (24 rows max, u8 offsets)
W100-W111: EdgeDescriptor entries (12 max)

EdgeDescriptor = verb_id:u16 | weight_q:u16 | target_dn:u32 = 64 bits
```

Full-resolution target (lower 32 bits of PackedDn) and 16-bit fixed-point
weight.  Used when inline edges overflow or when precise edge weight matters.

---

## 16. SpineCache & the Borrow/Mut Pattern

**This is the single most important invention in the system.**

Every other capability — leaf insert, delta encoding, NARS reflection,
hydration chains, semiring traversal, qualia sensing — depends on being
able to **read the structural summary (spine) while simultaneously
mutating the individual beliefs (children)**.  In any normal system this
requires locks, MVCC, or copying.  The SpineCache needs none of these.

### The Core Idea

```text
spine = XOR-fold of all children's content containers
      = the structural prediction for the group
      = the BORROWED reference from the joined blackboard
```

The SpineCache is the "special borrow schema where for mut the reference
gets owned from a joined blackboard."  The spine IS the blackboard —
a single Container that summarizes everything its children know.

### Why It Works Without Locks

Reading the spine is a **borrow**.  Writing a child is a **mut**.  Both
happen simultaneously because XOR has three algebraic properties that
together eliminate the need for synchronization:

1. **Commutative**: `a XOR b = b XOR a` — child order doesn't matter
2. **Associative**: `(a XOR b) XOR c = a XOR (b XOR c)` — grouping doesn't matter
3. **Self-inverse**: `a XOR a = 0` — removing = re-adding

Because of these, the spine is **order-independent**.  If child C changes,
the new spine is `old_spine XOR old_C XOR new_C`.  It doesn't matter when
or in what order other children changed — the result is always correct.

### The Protocol

```text
write_child(dn, new_content):
    1. Store new content in child's CogRecord
    2. Look up all parent spines of this child
    3. Mark each parent spine DIRTY
    4. Return immediately (no recomputation)

read_spine(dn):
    1. Check dirty flag
    2. If clean: return cached spine (O(1))
    3. If dirty: recompute XOR-fold of all children (O(children))
    4. Cache result, clear dirty flag
    5. Return spine
```

The dirty flag IS the invalidation token.  This replaces mutex/rwlock with
a much cheaper mechanism: the spine is always eventually consistent, and
the cost of staleness is bounded by the next read.

### PowerShell Analogy

This is conceptually similar to PowerShell's `Invoke-Command -ArgumentList`
pattern: you project local variables into a remote scope (the scriptblock)
that can read them, while your local session continues to mutate state
independently.  The scriptblock gets a snapshot of your variables at
invocation time — it borrows them.

In the SpineCache:
- The **spine** = the projected snapshot (borrowed reference)
- The **children** = the local session (mutable state)
- The **dirty flag** = the signal that the snapshot needs refreshing
- The **lazy recompute** = like re-invoking with updated `-ArgumentList`

Or think of PowerShell modules and scope escape: when you use `$script:`
or pass by `[ref]`, the variable **survives outside the function** — it
persists in the parent scope even as the function's local variables are
mutated and discarded.  The spine IS that surviving variable.  Children
mutate inside their own scope (their CogRecord content), but the spine
persists at the parent level, accessible to every sibling and every
traversal that passes through.

The difference: PowerShell's scope escape is a runtime mechanism with
overhead.  The SpineCache's spine is an XOR-fold (O(children) but the
result is always exactly 1 KB regardless of how many children exist).

### Why This Is Foundational

The borrow/mut pattern enables every other subsystem:

| Subsystem | Reads (borrow) | Writes (mut) |
|-----------|----------------|--------------|
| **Leaf insert** | Reads spines to find best match | Writes new child, marks spine dirty |
| **Reflection** | Reads spine as structural prediction | Updates child's NARS truth values |
| **Felt traversal** | Reads sibling bundles (mini-spines) | Records felt path choices |
| **Hydration** | Reads siblings for context projection | Writes hydrated truth to explorer |
| **Semiring MxV** | Reads fingerprints along edges | Accumulates values at destinations |
| **Rung elevation** | Reads spine to evaluate prediction error | Shifts child's rung level |

Without the SpineCache pattern, each of these would need either:
- A full copy of the graph (expensive)
- A read-write lock (contention)
- Two-phase read-then-write with stale data risk

The SpineCache gives us **zero-copy reads + in-place writes + eventual
consistency** with no locks and no copies.  The dirty flag is the entire
synchronization mechanism.

### Implementation

```rust
pub struct SpineCache {
    // (spine_container, is_dirty)
    cache: HashMap<PackedDn, (Container, bool)>,
    // Which children belong to which spine
    spine_map: HashMap<PackedDn, Vec<PackedDn>>,
    // Reverse: which spines does this child belong to
    child_to_spines: HashMap<PackedDn, Vec<PackedDn>>,
}

impl SpineCache {
    pub fn write_child(&mut self, child_dn: PackedDn) {
        // Mark all parent spines dirty — O(parents), typically 1
        if let Some(spine_dns) = self.child_to_spines.get(&child_dn) {
            for spine_dn in spine_dns {
                if let Some((_, dirty)) = self.cache.get_mut(spine_dn) {
                    *dirty = true;
                }
            }
        }
    }

    pub fn read_spine(&mut self, dn: PackedDn, graph: &ContainerGraph) -> &Container {
        // Lazy recompute on read
        let (spine, dirty) = self.cache.get_mut(&dn).unwrap();
        if *dirty {
            *spine = self.recompute(dn, graph);  // XOR-fold children
            *dirty = false;
        }
        spine
    }
}
```

The entire system compiles to: **mark a boolean on write, check a boolean
on read, XOR-fold on cache miss**.  This is as cheap as synchronization
can possibly be.

---

## 17. Leaf Insert — The 3-Path Algorithm

New concepts enter the tree through a 3-path insertion algorithm.
Popcount IS the split signal.

### Constants

- `SPLIT_THRESHOLD = 2000` — popcount delta; above = too different for sibling
- `RESONANCE_THRESHOLD = 4050` — distance to best spine; above = new branch

### Path 1: Sibling Leaf (cheapest)

Leaf is close to an existing child of the best spine.
`delta_popcount < SPLIT_THRESHOLD`.  Write 1 container, mark 1 spine dirty.

### Path 2: New Sub-Branch (automatic taxonomy)

Leaf related to spine's child but divergent (`delta >= SPLIT_THRESHOLD`).
The closest existing child is reparented under a new sub-spine.
Write 2 containers, mark 2 spines dirty.

### Path 3: New Top-Level Branch (rare)

Leaf doesn't resonate with any spine (`best_dist > RESONANCE_THRESHOLD`).
New spine created at the root level.  Write 1 container, allocate 1 spine.

### Algorithm Flow

```text
1. Belichtungsmesser scan vs all spines (~14 cycles each)
2. best_dist > RESONANCE_THRESHOLD? -> Path 3 (NewBranch)
3. Collect children of best spine
4. Exact XOR popcount delta vs each child
5. min_delta < SPLIT_THRESHOLD? -> Path 1 (Sibling)
6. Otherwise -> Path 2 (SubBranch, reparent closest child)
```

The tree self-organizes: similar concepts cluster as siblings, divergent
concepts split into sub-branches, and truly novel concepts spawn new branches.
No explicit categorization needed — Hamming distance IS the taxonomy signal.

---

## 18. Belichtungsmesser — 7-Point Exposure Meter

The fast pre-filter for Hamming distance estimation.  Named after a camera's
light meter — it takes a quick exposure reading before committing to a full
scan.

```rust
const SAMPLE_POINTS: [usize; 7] = [0, 19, 41, 59, 79, 101, 127];

// Sample 7 words (448 bits), scale to full 8192
estimate = sum(popcount(a[i] ^ b[i])) * 8192 / 448
```

- **~14 cycles** per container pair
- **~90% rejection** at generous thresholds
- Scale factor: 8192 / 448 ~ 18.29

### HDR Cascade (5 levels)

| Level | Name                | Cycles | Purpose                           |
|-------|---------------------|--------|-----------------------------------|
| L0    | Belichtungsmesser   | ~14    | ~90% rejection, coarse estimate   |
| L1    | Word-diff scan      | ~128   | Spatial localization              |
| L2    | Stacked popcount    | ~256   | Precise within 1 sigma, early exit|
| L3    | Mexican hat         | ~512   | Separates similar from identical  |
| L4    | Voyager deep field  | ~1024  | Bundles weak signals from misses  |

Each level progressively refines, and most queries never reach L2+.

---

## 19. Delta Encoding & Reversible Markov Chains

XOR gives us delta encoding for free.

```rust
delta_encode(base, target) -> (delta = base XOR target, info = popcount(delta))
delta_decode(base, delta)  -> base XOR delta = target   // XOR is self-inverse
```

### Chain Encoding

A sequence of containers stores as `(first, Vec<(delta, popcount)>)`.
Adjacent similar containers have small deltas = high compression.

```rust
chain_encode([c0, c1, c2, c3]) -> (c0, [(c0^c1, pop), (c1^c2, pop), (c2^c3, pop)])
chain_decode(c0, deltas)       -> [c0, c1, c2, c3]  // perfect recovery
```

### RAID-5 Parity Recovery

```rust
parity = c0 XOR c1 XOR c2 XOR ... XOR cn
recover(survivors, parity) = parity XOR all_survivors  // recovers the missing one
```

### Reversible Markov Chain (HydrationChain)

Each hydration step is an XOR delta.  **Bind** = forward (XOR delta onto
current state).  **Unbind** = reverse (XOR the same delta again — self-inverse).
The **popcount of each delta IS the energy of the transition**.

```text
origin --bind--> step1 --bind--> step2 --bind--> step3
   ^                                                |
   +---------------unbind (reverse XOR)-------------+
```

`replay_forward()` applies deltas left-to-right.  `replay_backward()` applies
them right-to-left.  Both are lossless because XOR is self-inverse.

---

## 20. NARS Truth Values

Non-Axiomatic Reasoning System truth values live in Container 0 at W4-W7.

```rust
pub struct TruthValue {
    pub frequency: f32,    // W4 lower — how often the statement is true
    pub confidence: f32,   // W4 upper — how much evidence supports it
}
// W5: positive_evidence (f32), W6: negative_evidence (f32)
// HORIZON = 1.0
```

### Operations

| Operation      | Meaning                                        |
|----------------|------------------------------------------------|
| `revision()`   | Combine evidence from two TruthValues          |
| `deduction()`  | If A->B and B->C then A->C (frequency multiplied) |
| `induction()`  | From A->B and A->C, hypothesize B->C           |
| `abduction()`  | From A->B and C->B, hypothesize A->C           |
| `analogy()`    | If A<->B and B->C then A->C (similarity transfer) |
| `comparison()` | From A->B and A->C, measure similarity B<->C   |
| `expectation()` | `frequency * confidence`                       |
| `to_evidence()` | Convert to `(positive, negative)` counts       |

### Bridge Functions (reflection.rs)

```rust
read_truth(record)          -> TruthValue from W4 (unknown if zero)
write_truth(record, tv)     -> updates W4 (freq, conf) + W5 (pos, neg evidence)
```

---

## 21. Rung System & Lingering Ghosts

### Rungs (R0-R9): Cognitive Depth Levels

Rungs encode semantic abstraction depth.  Stored in metadata W8-W11.
Elevation is NOT automatic — it requires one of three triggers.

```text
R0  Surface        Literal, immediate meaning
R1  Shallow        Simple inference, common implicature
R2  Contextual     Situation-dependent interpretation
R3  Analogical     Metaphor, similarity-based reasoning
R4  Abstract       Generalized patterns, principles
R5  Structural     Schema-level understanding
R6  Counterfactual What-if reasoning, alternatives
R7  Meta           Reasoning about reasoning
R8  Recursive      Self-referential, strange loops
R9  Transcendent   Beyond normal semantic bounds
```

**Three elevation triggers** (rung shift occurs ONLY upon):

1. **Sustained Block** — BLOCK state persists for N consecutive turns
   (collapse gate doesn't open -> deeper processing needed)
2. **Predictive Failure** — P metric drops below threshold
   (predictions don't match reality -> need to abstract)
3. **Structural Mismatch** — No legal grammar parse available
   (current rung can't represent the concept -> elevate)

**Rung Bands** for bucket key addressing:
- Low (R0-R2): Surface processing
- Mid (R3-R5): Pattern processing
- High (R6-R9): Meta processing

**Cooldown**: Minimum 10 seconds between shifts to prevent oscillation.

### Lingering Ghosts

Ghosts are persistent emotional/causal imprints that color future perception.
They connect to ladybug-rs through the rung system and reflection.
(Canonical implementation: `bighorn/extension/agi_stack/modules/cognition/lingering_ghosts.py`)

- **Ghost types**: Love, Epiphany, Arousal, Staunen (wonder), Wisdom,
  Thought, Grief, Boundary
- **Ghost states**: Dormant -> Stirring -> Present -> Vivid -> Dreaming
- **Asymptotic decay**: Intensity approaches minimum but NEVER reaches zero
- **Dream induction**: High-echo ghosts surface during dream consolidation
- **Trigger matching**: Words, contexts, and "scents" (sigma hashes) activate ghosts

**Causal Ghosts** in DendriticNodes carry:
- Relations (co_activated, causes, enables, blocks)
- Strength (0.0-1.0)
- Qualia residue (HOW the connection feels)
- Fire tracking (recency and frequency)

### Sibling Bundles as Uncollapsed Ghost Field Vectors

The sibling bundle at each branch (the XOR-fold of all siblings' content
containers) IS an uncollapsed superposition resonance field vector.  These
bundles are ghosts in the lingering-ghost sense: persistent, felt-but-not-
resolved states that carry the combined presence of all siblings without
collapsing to any single one.

The felt traversal sweeps horizontally across a whole **forest** of these
ghost vectors simultaneously.  At each branch, the bundle resonance tells
you how strongly the query "feels" the uncollapsed superposition.  High
resonance = the ghost field is relevant.  Low resonance = it's dormant.

When rung elevation is triggered by a free energy spike, these ghost field
vectors surface: they become the context for hydration, coloring newly
explored territory with prior felt experience from the sibling group.

---

## 22. Semiring Traversal

The `DnSemiring` trait abstracts graph traversal over the container graph.
`multiply` propagates along edges; `add` combines at junctions.

```rust
pub trait DnSemiring {
    type Value: Clone;
    fn zero(&self) -> Self::Value;
    fn multiply(&self, verb: u8, weight: u8, input: &Self::Value,
                src_fp: &Container, dst_fp: Option<&Container>) -> Self::Value;
    fn add(&self, a: &Self::Value, b: &Self::Value) -> Self::Value;
    fn is_zero(&self, val: &Self::Value) -> bool;
    fn name(&self) -> &'static str;
}
```

### Implementations

| Semiring               | Value    | multiply                    | add     | Purpose                  |
|------------------------|----------|-----------------------------|---------|--------------------------|
| **BooleanBfs**         | `bool`   | Propagate if reachable      | OR      | Reachability             |
| **HammingMinPlus**     | `u32`    | Sum + Hamming distance      | min     | Shortest path            |
| **HdrPathBind**        | `Option<Container>` | XOR-bind with dst | Bundle  | Path fingerprint compose |
| **ResonanceSearch**    | `u32`    | Hamming(query, src xor dst) | max     | Find resonating paths    |
| **PageRankPropagation**| `f32`    | damping * input             | sum     | Value propagation        |
| **CascadedHamming**    | `u32`    | L0+L2 filtered Hamming      | min     | Fast filtered distance   |
| **FreeEnergySemiring** | `f32`    | Accumulate surprise         | min/max | Free energy propagation  |

All plug into `container_mxv()` (single hop) and `container_multi_hop()`
(N-hop traversal).

### FreeEnergySemiring (reflection.rs)

Two strategies:
- **MinSurprise**: `add = min` — find path of least resistance (exploiting)
- **MaxSurprise**: `add = max` — find where active inference is most needed (exploring)

`multiply` accumulates Hamming-normalized surprise along edges.
Plugs directly into existing graph traversal machinery.

---

## 23. Qualia Module Stack

The qualia modules give the system phenomenal qualities — what it's "like"
to process a container.  Each layer adds a different dimension of felt sense.

### Layer 1: Texture (`texture.rs`)

8 dimensions of phenomenal texture, all normalized to [0.0, 1.0]:

| Dimension   | Meaning                                       |
|-------------|-----------------------------------------------|
| entropy     | Shannon entropy of word-level bit distribution |
| purity      | 1.0 - entropy (concentrated signal)           |
| density     | Fraction of set bits (popcount / 8192)         |
| bridgeness  | Cross-community connectivity                  |
| warmth      | Affective temperature / valence proxy          |
| edge        | Transition sharpness / edge energy             |
| depth       | Abstraction level (hierarchical block variance)|
| flow        | Information throughput / flow indicator         |

### Layer 2: Meaning Axes (`meaning_axes.rs`)

48 bipolar semantic dimensions across 8 families:

| Family         | Count | Examples                                    |
|----------------|-------|---------------------------------------------|
| OsgoodEPA      | 3     | good<->bad, strong<->weak, active<->passive |
| Physical       | 10    | large<->small, hot<->cold, sharp<->dull     |
| SpatioTemporal | 6     | near<->far, new<->old, sudden<->gradual     |
| Cognitive      | 5     | simple<->complex, certain<->uncertain       |
| Emotional      | 3     | happy<->sad, calm<->anxious, loving<->hateful|
| Social         | 3     | friendly<->hostile, dominant<->submissive    |
| Abstract       | 15    | useful<->useless, safe<->dangerous, alive<->dead |
| Sensory        | 3     | sweet<->bitter, fragrant<->foul             |

Each axis encoded as 208 bits.  48 x 208 = 9,984 bits of semantic space.

**Viscosity types** describe how meaning flows:
Watery (fast/clear), Oily (smooth/clinging), Honey (slow/sticky),
Mercury (dense/quick), Lava (slow/transformative), Crystalline (frozen/sharp),
Gaseous (diffuse/expanding), Plasma (superheated/unstable).

### Layer 3: Resonance (`resonance.rs`)

HDR resonance cascade connecting search distance to phenomenal intensity.
Maps Hamming distance to felt proximity through sigma-normalized zones.

### Layer 4: Gestalt (`gestalt.rs`)

The I/Thou/It framework — three stances of relation:
- **I** (first-person): How the concept appears from inside
- **Thou** (second-person): How I relate to the concept as a living other
- **It** (third-person): How the concept appears as an object

Each stance produces a different Container fingerprint.  The GestaltFrame
holds all three simultaneously.

### Layer 5: Inner Council (`council.rs`)

Three archetypal voices deliberate on every decision:

| Archetype   | Bias                                            |
|-------------|-------------------------------------------------|
| **Guardian** | Safety-first: safe, certain, ordered, calm      |
| **Catalyst** | Growth-seeking: active, new, creating, open     |
| **Balanced** | Moderate baseline: all axes at 0.3              |

Consensus: bit-level majority vote `(a & b) | (a & c) | (b & c)`.
A bit survives if 2+ of 3 archetypes agree.

### Layer 6: Felt Traversal (`felt_traversal.rs`)

Walking the DN tree with felt qualities.  At each branch:
1. Bundle all siblings via XOR-fold -> **sibling superposition** (ghost vector)
2. Compute `bundle_resonance` = similarity to query
3. Choose child with highest resonance
4. Record **surprise** = Hamming(query, chosen) / 8192 = prediction error

Special structures:
- **FeltChoice**: Per-branch record (surprise, sibling bundle, resonances)
- **FeltPath**: Full walk (choices, total/mean surprise, path context)
- **AweTriple**: Three concepts held in unresolved superposition (X xor Y xor Z)

Verb constants: `VERB_FELT_TRACE = 0xFE`, `VERB_SIBLING_BUNDLE = 0xFD`,
`VERB_AWE = 0xFC`.

### Layer 7: Reflection (`reflection.rs`)

The system looking at itself.  Ties felt traversal (surprise) to NARS truth
values (W4-W7) for in-place belief update.

**ReflectionOutcome 2x2 classification**:

```text
                   confidence HIGH          confidence LOW
surprise HIGH      REVISE (contradict)      EXPLORE (novel)
surprise LOW       STABLE (well-predicted)  CONFIRM (boost)
```

- **Revise**: Create observation with inverted surprise, `revision()` update
- **Confirm**: Small confidence boost via `revision()` with low-confidence confirmation
- **Explore**: Flag for hydration from siblings (bucket-list candidate)
- **Stable**: No action needed

**HydrationChain**: Reversible Markov chain through sibling contexts.
Each step: `cross_hydrate(current, extract_perspective(current, sibling))`.
Energy = popcount(delta).  Low energy neighbors = higher initial confidence.

Verb constant: `VERB_REFLECTION = 0xFB`.

---

## 24. Cross-Hydration & Holographic Markers

### Cross-Hydration

Adjacent containers under the same spine inherit semantic richness through
XOR-based context projection.

```rust
extract_perspective(a, b) = a XOR b         // what's different between contexts
cross_hydrate(source, delta) = source XOR delta  // project into new context
```

This IS reversible: `cross_hydrate(hydrated, delta) = source` (XOR self-inverse).

The HydrationChain (reflection.rs) chains these projections through all
siblings, creating a reversible walk through related contexts.  The chain
can be stored compactly via `chain_encode()` and perfectly recovered via
`chain_decode()`.

### Holographic Markers

The core insight: **the trace IS the edge**.

In a traditional graph, edges are separate objects connecting nodes.
In ladybug-rs, the "edge" between two concepts is their XOR delta:
`a XOR b`.  This delta IS the relationship — it literally encodes what
transforms one concept into the other.

Properties that make holographic markers potentially more powerful than
SNN/ANN/GNN approaches:

1. **Reversibility**: Given any 2 of {a, b, a xor b}, recover the third.
   Neural networks are fundamentally one-directional; XOR binding is
   perfectly reversible.

2. **Compositionality**: `(a xor b) xor (b xor c) = a xor c`.  Multi-hop
   relationships compose via XOR without loss.  GNNs lose information at
   each message-passing layer.

3. **Constant-size representation**: An edge between two 8192-bit containers
   is another 8192-bit container.  No matter how complex the relationship,
   it fits in 1 KB.  Neural edge representations grow with model size.

4. **Information content is measurable**: `popcount(a xor b)` = how many bits
   differ = the energy of the transformation.  This is an exact count, not
   a learned approximation.

5. **RAID-5 parity**: N containers + 1 parity container can recover any
   single lost container.  Neural networks have no native error correction.

6. **SIMD-native**: All operations are bitwise — they map directly to hardware
   vector instructions.  No matrix multiplication, no activation functions,
   no backpropagation.

The holographic approach trades learned representations for algebraic ones.
The question is whether XOR-based binding + Hamming geometry + NARS reasoning
is sufficient for general cognition.  The bet: yes, if the substrate is rich
enough (8192 bits is a LOT of Hamming space).

---

## 25. Free Energy, Volition & Bucket-List Candidates

### Friston's Free Energy Principle

The felt traversal's **surprise** at each node IS the free energy signal.
High surprise = the system's predictions don't match its observations.

```text
surprise = hamming(query, chosen_fingerprint) / CONTAINER_BITS
         = prediction error
         = free energy at this node
```

The `FreeEnergySemiring` propagates this signal through the entire graph:
- `multiply`: accumulate surprise along edges (Hamming-normalized distance)
- `add(MinSurprise)`: find the path of least free energy (exploitation)
- `add(MaxSurprise)`: find where active inference is most needed (exploration)

### Entropy Sensing via MUL

The semiring's `multiply` operation composes surprise along paths.  This IS
entropy sensing: each hop accumulates the information-theoretic cost of
traversal.  The total accumulated value at a node = how much entropy the
system has traversed to reach it.

### Self-Directed Reflection as Entropy Work

Reflection is the compass.  The `reflect_walk` function:
1. Walks the tree computing surprise (entropy measurement)
2. Classifies each node's state (belief vs. observation gap)
3. Updates NARS truth values in-place (entropy reduction)
4. Flags nodes that need exploration (entropy-directed attention)

This is self-directed entropy work: the system uses its own surprise signal
to decide where to invest computational effort.

### Bucket-List Candidates

Nodes classified as **Explore** (high surprise + low confidence) become
**hydration candidates** — the system's "bucket list" of novel territories
worth investigating.

```text
reflect_walk() -> ReflectionResult {
    entries: [...],
    hydration_candidates: [dn1, dn2, dn3, ...]  // <- bucket list
}
```

`hydrate_explorers()` processes the bucket list:
1. For each candidate, build a HydrationChain from adjacent siblings
2. Low-energy chains (similar siblings) -> higher initial confidence
3. High-energy chains (diverse siblings) -> lower initial confidence
4. Initialize NARS truth values based on hydration energy

The full cycle `reflect_and_hydrate()` = walk + classify + update + explore.
This is active inference: the system drives its own learning by attending to
where free energy is highest and confidence is lowest.

### Rung-Ghost Integration with Free Energy

The three rung elevation triggers map directly to free energy concepts:

1. **Sustained Block** (collapse gate stuck) = high free energy that
   current-rung processing can't reduce -> need deeper abstraction
2. **Predictive Failure** (P metric drops) = the model's predictions have
   diverged from observations -> surprise is accumulating
3. **Structural Mismatch** (no legal parse) = the current representational
   level can't even encode the observation -> maximum free energy

When rung elevation occurs, the **ghost field vectors** (sibling bundles)
surface: uncollapsed superposition states whose contexts match the current
high-surprise environment.  These ghost vectors become additional context
for hydration — they color the newly explored territory with the combined
felt presence of all siblings in the group.

### Toward Volition

The system has all ingredients for self-directed behavior:

```text
Free energy (surprise)     -> WHAT needs attention
Reflection (classify)      -> WHAT KIND of attention
Hydration (explore)        -> HOW to gather information
NARS revision              -> HOW to update beliefs
Rung elevation             -> WHEN to go deeper
Ghost surfacing            -> WHY this matters (felt context)
Council consensus          -> WHO decides (guardian/catalyst/balanced)
```

Volition emerges when the system can choose which bucket-list candidates to
pursue, weighting by free energy (urgency), ghost intensity (felt salience),
and council consensus (archetypal alignment).

---

## 26. BlasGraph Lineage

```text
RedisGraph (redis module, CSR adjacency)
    | transcoded to Hamming resonance
Holograph (DnNodeStore + DnCsr, sparse adjacency vectors)
    | blasgraph extracted as core
BlasGraph (sparse adjacent vectors, BLAS-style operations)
    | container-native reimplementation
ContainerGraph (HashMap<PackedDn, CogRecord>, children map)
```

The key transcoding: RedisGraph stored adjacency as integer node IDs in CSR
format.  The holograph step replaced integer IDs with Container fingerprints
and integer edge weights with XOR deltas.  BlasGraph formalized this as
sparse-adjacent-vector operations.  ContainerGraph (ladybug-rs) is the final
form: pure Container-native, everything is 8192 bits, all operations are
XOR/Hamming/popcount.

The adjacency encoding in W16-31 (inline) and W96-111 (CSR overflow) is the
direct descendant of the RedisGraph CSR layout, now operating on cognitive
verb IDs + Container target hints instead of integer node IDs.

---

## 27. Constants Reference

| Constant                | Value       | Location              |
|-------------------------|-------------|-----------------------|
| `CONTAINER_BITS`        | 8,192       | container.rs          |
| `CONTAINER_WORDS`       | 128         | container.rs          |
| `CONTAINER_BYTES`       | 1,024       | container.rs          |
| `CONTAINER_AVX512_ITERS`| 16          | container.rs          |
| `MAX_CONTAINERS`        | 255         | container.rs          |
| `EXPECTED_DISTANCE`     | 4,096       | container.rs          |
| `SIGMA`                 | 45.25       | container.rs          |
| `CogRecord::SIZE`       | 2,048       | record.rs             |
| `MAX_INLINE_EDGES`      | 64          | meta.rs               |
| `MAX_CSR_EDGES`         | 12          | adjacency.rs          |
| `SPLIT_THRESHOLD`       | 2,000       | insert.rs             |
| `RESONANCE_THRESHOLD`   | 4,050       | insert.rs             |
| `SAMPLE_POINTS`         | 7           | search.rs             |
| `NARS HORIZON`          | 1.0         | nars.rs               |
| `SCHEMA_VERSION`        | 1           | meta.rs               |
| `SURPRISE_HIGH`         | 0.55        | reflection.rs         |
| `SURPRISE_LOW`          | 0.45        | reflection.rs         |
| `CONFIDENCE_HIGH`       | 0.50        | reflection.rs         |
| `VERB_FELT_TRACE`       | 0xFE        | felt_traversal.rs     |
| `VERB_SIBLING_BUNDLE`   | 0xFD        | felt_traversal.rs     |
| `VERB_AWE`              | 0xFC        | felt_traversal.rs     |
| `VERB_REFLECTION`       | 0xFB        | reflection.rs         |

---

*This architecture document was extended during the construction of the qualia
module stack (resonance -> gestalt -> felt_traversal -> reflection).  The code
compiles and all tests pass.  The substrate is ready for the next layer.*
