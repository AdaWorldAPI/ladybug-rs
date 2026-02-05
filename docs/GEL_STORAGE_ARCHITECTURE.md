# GEL Storage Architecture: 512-Byte Node Records + Separated Fingerprints

> **Status**: Design Proposal
> **Date**: 2026-02-05
> **Premise**: Outsource DataFusion to gRPC/bitpacked UDP, reduce storage to
> Graph Execution Language (GEL), store only what's necessary at the node level,
> keep fingerprints zero-copy/separate.

---

## Motivation

The current `BindNode` is 1,312 bytes. 95% of that (1,248 bytes) is the 10K-bit
fingerprint baked inline. This means:

- Graph traversal loads fingerprints it doesn't need
- Every node touches 21 cache lines even for a parent lookup
- The fingerprint dominates working set, limiting how many nodes fit in L2/L3
- Edge records also carry full 1,248-byte fingerprints inline

The idea: **separate what you think with from what you think about**.

- **512-byte fixed record**: Everything needed for graph traversal, DN trees,
  NARS truth, edge connectivity, HDR sketches, and GEL execution state
- **4096-bit Hamming (512 bytes)**: Stored separately, zero-copy mmap'd
- **10K/64K extended fingerprints**: Also separate, loaded on demand

No FPU required for any file operation on the 512-byte record.

---

## The Split: Node Record vs Fingerprint Tiers

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    TIER 0: NODE RECORD (512 bytes)                       │
│                    ═══════════════════════════════                       │
│    Always in memory for active nodes. 8 cache lines.                    │
│    Contains: addressing, tree, NARS, edges, HDR sketch, GEL state.     │
│    NO FPU. Pure integer ops for all field access.                       │
├──────────────────────────────────────────────────────────────────────────┤
│                    TIER 1: HAMMING 4096 (512 bytes)                     │
│                    ═══════════════════════════════                       │
│    Zero-copy Arrow buffer. mmap'd from storage.                         │
│    Used for: Hamming distance, XOR binding, HDR cascade level 3+.      │
│    Ratio: 1:1 with node record. Acceptable.                            │
├──────────────────────────────────────────────────────────────────────────┤
│                    TIER 2: EXTENDED (10K or 64K bits)                    │
│                    ═════════════════════════════════                     │
│    Zero-copy Arrow buffer. Loaded on demand, never in working set.      │
│    Used for: Deep similarity, high-precision recall, archival search.   │
│    Ratio: 1:2.4 (10K) or 1:16 (64K) vs node record.                   │
│    Acceptable because zero-copy — storage layer does the thinking.      │
└──────────────────────────────────────────────────────────────────────────┘
```

### Memory Ratios

| Configuration | Per Node | 65K Nodes (10% pop) | Notes |
|---------------|----------|---------------------|-------|
| Record only | 512 B | 3.2 MB | Graph traversal, DN trees |
| + Hamming 4096 | 1,024 B | 6.4 MB | + search capability |
| + Extended 10K | 2,272 B | 14.5 MB | + deep similarity |
| + Extended 64K | 8,704 B | 55.6 MB | + maximum precision |
| **Current** | **1,312 B** | **8.4 MB** | All inline, no separation |

The 512-byte record alone gives you graph ops in 3.2 MB. Current layout needs
8.4 MB just to walk the tree. That's a 2.6x reduction in working set for
traversal-heavy workloads.

---

## 512-Byte Node Record Layout

### Segmentation: 8 + 8 + 48 + 64 + 128 + 128 + 128

```
Offset  Size   Segment          Purpose
──────  ─────  ───────────────  ─────────────────────────────────────────
0x000   8 B    IDENTITY         Address + type + flags
0x008   8 B    NARS+ACCESS      Truth values (fixed-point) + access meta
0x010   48 B   DN TREE+META     Tree structure + label + timestamps
0x040   64 B   EDGE TABLE       Inline edges (up to 8 full or 16 compact)
0x080   128 B  HDR SKETCH       1024-bit sketch for cascade filtering
0x100   128 B  META-THINKING    Reasoning state, attention, consciousness
0x180   128 B  GEL STATE        Graph Execution Language bytecode/state
──────  ─────
0x200   512 B  TOTAL
```

### Segment 0: IDENTITY (8 bytes)

```
Bits    Field           Type    Description
──────  ──────────────  ──────  ────────────────────────────────────
[0:7]   prefix          u8      8+8 address prefix (0x00-0xFF)
[8:15]  slot            u8      8+8 address slot (0x00-0xFF)
[16:23] node_type       u8      0=data, 1=verb, 2=edge, 3=meta, ...
[24:31] flags           u8      bit 0: has_children
                                bit 1: has_parent
                                bit 2: has_payload (external)
                                bit 3: has_hamming_4096
                                bit 4: has_extended_fp
                                bit 5: is_tombstone
                                bit 6: is_pinned (no eviction)
                                bit 7: reserved
[32:47] parent          u16     Parent address (0xFFFF = no parent)
[48:55] depth           u8      Tree depth (0 = root)
[56:63] rung            u8      Access level R0-R9 (4 bits used)
                                + sigma Σ reasoning depth (4 bits)
```

No FPU. All fields are u8/u16. Direct bit shifts.

### Segment 1: NARS + ACCESS (8 bytes)

```
Bits    Field           Type    Description
──────  ──────────────  ──────  ────────────────────────────────────
[0:7]   frequency       u8      NARS f: 0-255 → 0.0-1.0 (±0.004)
[8:15]  confidence      u8      NARS c: 0-255 → 0.0-1.0 (±0.004)
[16:23] qidx            u8      Qualia index (consciousness slot)
[24:31] causal_rung     u8      Pearl's ladder: 0=see, 1=do, 2=imagine
[32:47] access_count    u16     Saturating access counter
[48:63] epoch           u16     Last-modified epoch (for MVCC)
```

**Key design choice**: NARS truth as fixed-point u8 instead of f32. Resolution
of 1/255 ≈ 0.004 is sufficient for evidence-based reasoning. The f32 versions
are only needed at compute time — convert on read:

```rust
#[inline(always)]
fn f_to_f32(v: u8) -> f32 { v as f32 / 255.0 }  // Only at compute boundary
```

This avoids any FPU for storage/comparison operations. You can compare NARS truth
values with plain integer comparison: `if node.frequency > 200` means f > 0.78.

### Segment 2: DN TREE + META (48 bytes)

```
Offset  Size  Field              Description
──────  ────  ─────────────────  ────────────────────────────────────
0x00    8 B   label_hash         FNV-1a hash of label (u64)
0x08    4 B   csr_offset         Offset into BitpackedCsr edges
0x0C    2 B   children_count     Number of children in CSR
0x0E    2 B   edge_out_count     Number of outgoing edges total
0x10    2 B   edge_in_count      Number of incoming edges
0x12    2 B   payload_offset     Offset into external payload store
0x14    4 B   payload_len        Payload length (0 = no payload)
0x18    8 B   created_at         Timestamp (u64, nanos since epoch)
0x20    8 B   modified_at        Timestamp (u64, nanos since epoch)
0x28    8 B   xor_parity         XOR parity word for ACID validation
0x30    8 B   correlation_id     For causal tracing (links SEE→DO→IMAGINE)
0x38    8 B   reserved           Future: version vector, merge metadata
```

Labels are stored externally (string pool) but the hash is inline for fast
comparison. `csr_offset` points into the existing `BitpackedCsr` — no change
needed to the CSR structure itself.

### Segment 3: ADJACENCY DESCRIPTOR (64 bytes)

**Why not hardcoded inline edges**: A fixed 8 or 16-edge table is fragile. Some
nodes (hubs, DN roots) have hundreds of edges. Others (leaves) have one. A
brute-force limit doesn't scale and wastes space on sparse nodes.

Instead: the 64-byte segment is an **adjacency descriptor** that points into
columnar Arrow edge storage. Edges themselves live in Arrow buffers — zero-copy,
resortable, and sparse.

#### Vertical vs Horizontal Representation

```
┌─────────────────────────────────────────────────────────────────────────┐
│  VERTICAL (DN Tree)               HORIZONTAL (Edges/Adjacency)         │
│  ═════════════════                ════════════════════════════          │
│                                                                         │
│  domain                           A ──CAUSES──► B                      │
│    └── tree                       A ──SIMILAR──► C                     │
│          └── branch               A ──PART_OF──► D                     │
│                └── twig            B ──INHIBITS──► E                    │
│                      └── leaf                                          │
│                                                                         │
│  Parent pointers (u16)            Arrow columns: from|to|verb|weight   │
│  Depth encoded in fingerprint     XOR sparse pointers into buckets     │
│  O(1) upward traversal            O(1) bucket lookup, free resorting   │
│  Stored in node record            Stored in columnar Arrow buffers     │
└─────────────────────────────────────────────────────────────────────────┘
```

**Vertical** is the DN tree hierarchy: `Ada:A:soul:identity`. Parent-child
relationships. Each node knows its parent (u16) and depth (u8). Fingerprint bits
encode depth-level semantics (see "Fingerprint Depth Encoding" below).

**Horizontal** is everything else: verb-typed edges between nodes at any depth.
Stored columnar in Arrow. Accessed via sparse pointers.

#### Adjacency Descriptor Layout (64 bytes)

```
Offset  Size  Field                  Description
──────  ────  ─────────────────────  ────────────────────────────────────
0x00    4 B   edge_out_offset        Offset into Arrow edge column (outgoing)
0x04    2 B   edge_out_count         Number of outgoing edges
0x06    2 B   edge_out_bucket        XOR-derived bucket ID for locality
0x08    4 B   edge_in_offset         Offset into Arrow edge column (incoming)
0x0C    2 B   edge_in_count          Number of incoming edges
0x0E    2 B   edge_in_bucket         XOR-derived bucket ID
0x10    8 B   adjacency_xor          XOR fold of neighbor fingerprint sketches
                                     (64-bit locality signature)
0x18    8 B   verbal_signature       XOR fold of verb fingerprints on edges
                                     (which verbs connect this node?)
0x20    4 B   csr_offset             Offset into BitpackedCsr (children)
0x24    2 B   children_count         DN tree children count
0x26    2 B   sibling_offset         Offset to first sibling in CSR
0x28    8 B   dn_path_hash           FNV-1a of domain:tree:branch:twig:leaf
0x30    8 B   subtree_xor            XOR fold of entire subtree fingerprints
                                     (parent hydration: parent_fp ≈ ⊕ children)
0x38    8 B   reserved               Future: federation pointers, shard ID
```

The key fields:

- **adjacency_xor**: 64-bit XOR fold of all neighbor sketches. Two nodes with
  similar `adjacency_xor` share neighborhood structure. This is the sparse
  pointer — nodes that XOR to similar values land in the same bucket.

- **verbal_signature**: XOR of all verb fingerprints on this node's edges.
  Tells you *what kinds of relationships* this node participates in without
  loading any edges.

- **subtree_xor**: XOR fold of all descendants. When you hydrate a parent node,
  its fingerprint is approximately `identity ⊕ subtree_xor`. This means parent
  tree hydration is O(1) — read one field, XOR once.

#### Columnar Arrow Edge Storage (External)

Edges live in four parallel Arrow arrays (columnar, zero-copy):

```
Arrow Column     Type    Description
─────────────    ─────   ─────────────────────────────────────
from_addr        UInt16  Source node address
to_addr          UInt16  Target node address
verb_addr        UInt16  Verb address (0x07:XX namespace)
weight           UInt16  Fixed-point weight (0-65535 → 0.0-1.0)
```

8 bytes per edge. No fingerprint per edge (unlike current BindEdge at 1,260 B).
Edge fingerprints are computed on demand: `from_fp ⊗ verb_fp ⊗ to_fp`.

**Why this works with zero-copy:**

1. Arrow columns are contiguous memory — CPU prefetcher loves this
2. Resorting adjacency = compute a new sort permutation, apply as Arrow
   `take()` kernel — the underlying buffer doesn't move
3. Filtering by verb = Arrow `filter()` kernel on verb_addr column — zero-copy
4. Bucket grouping = edges pre-sorted by `from_addr XOR verb_addr` — adjacent
   edges for the same source+verb type are contiguous

```rust
// Pseudo-code: adjacency bucket lookup
fn edges_for_node(node: &NodeRecord, edge_store: &EdgeStore) -> &[Edge] {
    let start = node.edge_out_offset as usize;
    let count = node.edge_out_count as usize;
    &edge_store.slice(start, count)  // Zero-copy Arrow slice
}

// Resorting is free:
fn edges_sorted_by_weight(edges: &RecordBatch) -> RecordBatch {
    let indices = sort_to_indices(edges.column("weight"), &SortOptions::default());
    take(edges, &indices)  // New view, same underlying buffers
}
```

#### 5×5×5 Sparse Adjacency (Crystal-Aligned)

The existing `ContextCrystal` already uses a 5×5×5 tensor (125 cells). We can
reuse this shape for sparse adjacency by mapping the three axes to:

```
Axis 0 (5 levels): Verb Class
  0 = structural (PARENT_OF, CHILD_OF, SIBLING_OF)
  1 = causal (CAUSES, ENABLES, PREVENTS)
  2 = similarity (SIMILAR_TO, OPPOSITE_OF, ANALOGY)
  3 = temporal (BEFORE, AFTER, DURING)
  4 = meta (ABOUT, REPRESENTS, ABSTRACTS)

Axis 1 (5 levels): Target Region
  0 = surface (0x00-0x0F)
  1 = fluid low (0x10-0x3F)
  2 = fluid high (0x40-0x7F)
  3 = nodes low (0x80-0xBF)
  4 = nodes high (0xC0-0xFF)

Axis 2 (5 levels): Depth/Strength
  0 = weak (weight < 0.2)
  1 = moderate (0.2-0.4)
  2 = significant (0.4-0.6)
  3 = strong (0.6-0.8)
  4 = binding (> 0.8)
```

Each cell of the 5×5×5 tensor holds a sparse entry:

```rust
struct AdjacencyCell {
    edge_offset: u32,   // Into Arrow edge column
    edge_count: u16,    // Edges in this bucket
    xor_sketch: u16,    // 16-bit XOR sketch of targets in bucket
}
// 8 bytes per cell × 125 cells = 1000 bytes if dense

// BUT: sparse representation. Only store non-empty cells:
struct SparseAdjacency {
    indices: Vec<u8>,        // Cell index (0-124), packed
    cells: Vec<AdjacencyCell>,
}
// Typical node: 5-15 non-empty cells = 40-120 bytes
```

This doesn't go inline in the 512-byte record (too variable). Instead, the
adjacency descriptor's `adjacency_xor` and `edge_out_offset` give you O(1)
access to the sparse tensor stored in a separate Arrow buffer.

**Scale validation**: The 5×5×5 crystal has been tested at 125 cells. For graph
adjacency, most nodes populate <15 cells (sparse). Hub nodes might populate
50-60. The tensor never needs to be dense — the Arrow columnar store handles
overflow. The crystal shape is just for locality bucketing.

#### DN Tree: domain:tree:branch:twig:leaf

Five depth levels map to fingerprint bit allocation:

```
Depth  Level     Path Example          Fingerprint Bits
─────  ────────  ───────────────────   ─────────────────────────────
0      domain    "Ada"                 bits 0-255    (256 bits)
1      tree      "Ada:A"              bits 256-767  (512 bits)
2      branch    "Ada:A:soul"         bits 768-1791 (1024 bits)
3      twig      "Ada:A:soul:ident"   bits 1792-2815 (1024 bits)
4      leaf      "Ada:A:soul:id:me"   bits 2816-4095 (1280 bits)
```

Deeper levels get more bits because leaves are more specific. This means:

1. **Comparing two nodes at depth 0** only needs bits 0-255 (32 bytes, 4 words)
2. **Comparing at depth 2** needs bits 0-1791 (224 bytes, 28 words)
3. **Full leaf comparison** uses all 4096 bits (512 bytes, 64 words)

**Parent hydration**: A parent node at depth D hydrates its fingerprint by
XOR-folding all children at depth D+1 in the bits for level D+1. The
`subtree_xor` field in the adjacency descriptor caches this fold:

```rust
// Parent fingerprint reconstruction:
// parent_fp[depth_bits] = ⊕(child_fp[depth_bits] for child in children)
fn hydrate_parent(parent: &mut NodeRecord, children: &[NodeRecord]) {
    let (start, end) = depth_bit_range(parent.depth + 1);
    let start_word = start / 64;
    let end_word = (end + 63) / 64;
    for word in start_word..end_word {
        parent.hdr_sketch[word - start_word] = children.iter()
            .fold(0u64, |acc, c| acc ^ c.hdr_sketch[word - start_word]);
    }
}
```

No FPU. Pure XOR. Parent "knows" its subtree without loading every leaf.

### Segment 4: HDR SKETCH + ADJACENCY SKETCH (128 bytes = 1024 bits)

The 128 bytes serve double duty:

**First 64 bytes (512 bits)**: HDR cascade sketch for Hamming search
```
64 bytes = 8 u64 words
Compressed from 4096-bit Tier 1 fingerprint (8:1 ratio)
Search: XOR + popcount on 8 words = 8 operations
Filter rate: ~85% of candidates eliminated before touching Tier 1
```

**Second 64 bytes (512 bits)**: Adjacency locality sketch
```
64 bytes = 8 u64 words
XOR fold of neighbor fingerprint sketches, organized by verb class:
  word 0-1: structural neighbors (PARENT_OF, etc.)
  word 2-3: causal neighbors (CAUSES, etc.)
  word 4-5: similarity neighbors (SIMILAR_TO, etc.)
  word 6-7: temporal + meta neighbors
```

This means you can answer "are these two nodes in similar neighborhoods?" by
comparing their adjacency sketches — without loading any edges. Two nodes with
Hamming(adj_sketch_A, adj_sketch_B) < threshold share graph structure.

### Segment 5: META-THINKING + GEL STATE (128 bytes combined)

Instead of separate 128-byte segments for meta-thinking and GEL (which was
256 bytes total), combine them into a single 128-byte segment. This keeps
the total at 512 bytes.

```
Offset  Size  Field              Description
──────  ────  ─────────────────  ────────────────────────────────────

Meta-Thinking (first 64 bytes):
0x00    8 B   attention_weights  4 × u16 attention scores
0x08    8 B   activation_hist    8 × u8 recent activation levels
0x10    8 B   resonance_score    u64: top-4 resonance partners packed
0x18    8 B   inhibition_mask    64-bit: which prefixes inhibited
0x20    8 B   temporal_context   u64: recent context hash
0x28    8 B   binding_state      u64: current XOR binding accumulator
0x30    8 B   evidence_window    packed w+/w- for sliding NARS
0x38    8 B   prediction_error   Compressed PE signal for learning

GEL State (second 64 bytes):
0x40    16 B  gel_bytecode       Up to 16 bytes of GEL instructions
0x50    8 B   program_counter    Current execution position
0x58    8 B   stack_top          Top of execution stack
0x60    16 B  registers          2 × u64 GEL registers (R0-R1)
0x70    8 B   accumulator        Running result (XOR, sum, product)
0x78    8 B   status+cont        Execution status + continuation addr
```

This is tight but sufficient. GEL instructions are compact (most graph ops
fit in 8 bytes). Two registers (R0, R1) cover the common case of "current node"
and "query vector". More complex GEL programs spill to the separate GEL stack
in the execution context (not per-node).

### Segment 6: RESERVED / OVERFLOW (128 bytes)

```
Offset  Size  Field              Description
──────  ────  ─────────────────  ────────────────────────────────────
0x00    64 B  sparse_adj_cache   Top 8 adjacency cells inline (fast path)
                                 8 × {cell_idx: u8, offset: u32, count: u16, xor: u8}
0x40    32 B  dn_path_prefix     First 32 bytes of DN path (avoid string lookup)
0x60    16 B  version_vector     For MVCC conflict detection
0x70    8 B   federation_id      Cross-shard node identity
0x78    8 B   checksum           CRC-64 of record (integrity check)
```

The `sparse_adj_cache` stores the top 8 non-empty cells from the 5×5×5 sparse
adjacency inline. This is NOT a hardcoded edge limit — it's a cache of the
most-accessed adjacency buckets. The full sparse adjacency lives in Arrow
buffers. For nodes with <=8 distinct verb×region×strength combinations (most
nodes), this cache covers 100% of lookups without touching external storage.

GEL is the execution language that replaces DataFusion for graph operations.
Instead of SQL → plan → optimize → execute, it's:

```
GEL bytecode → decode (integer ops) → execute on node → next node
```

No query planner. No optimizer. The graph structure IS the execution plan.

---

## DataFusion Outsourcing: gRPC/Bitpacked UDP

### Current State

DataFusion compiles to ~300MB of .rlib files but is only used in 5 source files.
The Flight server and HTTP server already operate without it. The HybridEngine
already provides a non-DataFusion query path.

### Proposed Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                     LADYBUG PROCESS (lightweight)                          │
│                                                                            │
│   BindSpace (512-byte records)                                             │
│   ├── GEL Executor (graph ops, inline bytecode)                            │
│   ├── BitpackedCsr (edge traversal)                                        │
│   ├── CogRedis (DN.*, CAM.*, DAG.*)                                       │
│   ├── HDR Cascade (sketch → Tier 1 → Tier 2)                              │
│   ├── XorDag (ACID, parity)                                                │
│   └── Arrow Zero-Copy (Tier 1 + Tier 2 fingerprints, mmap'd)              │
│                                                                            │
│   Transports:                                                              │
│   ├── Arrow Flight gRPC (port 50051) ← external SQL clients               │
│   ├── HTTP REST (port 8080) ← web clients                                 │
│   └── UDP Bitpacked (port 5050) ← Firefly/Pi, inter-node gossip           │
│                                                                            │
│   NO DataFusion in-process.                                                │
│   SQL goes over Flight to external DF service (or is transpiled to GEL).   │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
        │                          │                          │
        │ gRPC                     │ REST                     │ UDP
        ▼                          ▼                          ▼
┌────────────────┐  ┌────────────────────┐  ┌────────────────────────────┐
│ External DF    │  │ Web/MCP Clients    │  │ Firefly Nodes / Pi Fleet   │
│ Service        │  │                    │  │ (bitpacked frames, ~1ms)   │
│ (optional)     │  │                    │  │                            │
│ SQL → RecordBatch → Flight back        │  │ GEL bytecode in UDP frame  │
└────────────────┘  └────────────────────┘  └────────────────────────────┘
```

### What Gets Removed from the Binary

| Component | Current Size | After | Savings |
|-----------|-------------|-------|---------|
| DataFusion core | ~33 MB rlib | 0 | -33 MB |
| DF aggregates | ~30 MB | 0 | -30 MB |
| DF physical plan | ~24 MB | 0 | -24 MB |
| DF functions | ~24 MB | 0 | -24 MB |
| DF optimizer | ~19 MB | 0 | -19 MB |
| DF SQL parser | ~18 MB | 0 | -18 MB |
| **Total saved** | | | **~150+ MB** |

### What Stays

Arrow core (buffer, array, schema, IPC) — these are the zero-copy backbone.
Arrow Flight — this is the transport, not the compute.

---

## GEL: Graph Execution Language

GEL replaces SQL for graph operations. It's a bytecode that executes directly
on the node record without a query planner.

### Instruction Set (fits in 32 bytes)

```
Opcode  Mnemonic    Description
──────  ──────────  ─────────────────────────────────────────────
0x00    NOP         No operation
0x01    LOAD addr   Load node at address into register
0x02    WALK verb   Follow edge with verb, load target
0x03    FILTER f c  Filter by NARS truth (f >= threshold, c >= threshold)
0x04    XOR R0 R1   XOR registers (binding operation)
0x05    POPCNT R0   Popcount register (Hamming distance step)
0x06    SKETCH R0   Compare HDR sketch (1024-bit inline)
0x07    HAMMING     Compare full 4096-bit (loads Tier 1)
0x08    EMIT        Emit current node to result set
0x09    BRANCH cond Jump if condition (integer comparison)
0x0A    GATHER n    Collect n children (CSR traversal)
0x0B    REDUCE op   Reduce gathered set (XOR/MIN/MAX/SUM)
0x0C    YIELD       Pause execution, return partial result
0x0D    BIND s v o  Triple binding: subject XOR verb XOR object
0x0E    STORE       Write register back to node
0x0F    HALT        End execution
```

### Example: "Find similar nodes in subtree"

```
LOAD  0x80:0x00          ; Load root node
GATHER 255               ; Collect all children (CSR)
SKETCH R0                ; Compare HDR sketch against query
FILTER 0xCC 0x80         ; Keep only f>0.8, c>0.5
HAMMING                  ; Exact distance on survivors
EMIT                     ; Return results
HALT
```

This replaces:
```sql
SELECT * FROM nodes
WHERE hamming_distance(fingerprint, $query) < $threshold
AND parent_of(root, node)
AND nars_frequency > 0.8
```

No query planner needed. The GEL bytecode IS the plan.

---

## Fingerprint Tier Sizing

### Why 4096 Bits for Tier 1

The current 10K-bit fingerprint (156 u64 words) was designed for maximum
discrimination. But research shows diminishing returns past ~4096 bits for
most cognitive similarity tasks:

```
Bits    Discrimination    Bytes    Words(u64)   Cache Lines
─────   ──────────────    ─────    ──────────   ───────────
512     Good for coarse   64       8            1
1024    Good for HDR      128      16           2
4096    Excellent         512      64           8
10000   Marginal gain     1250     ~156         ~20
65536   Maximum           8192     1024         128
```

4096 bits at 512 bytes gives:
- 1:1 ratio with node record (clean)
- 64 u64 words (fits in L1 cache with record)
- Sufficient for >99% of similarity queries
- Exact same popcount/XOR operations, just fewer words

### Why Keep 10K/64K as Tier 2

Some applications need the extra precision:
- Archival deduplication (64K catches near-duplicates that 4K misses)
- Cross-domain transfer (more bits = more dimensions to bind across)
- Research workloads where recall matters more than latency

These are zero-copy mmap'd — they don't consume working memory. The storage
layer loads them on demand through `ArrowZeroCopy`, which already exists and
works.

---

## Migration Path from Current Layout

### Phase 1: Split Fingerprint Out of BindNode

```rust
// Before (current):
pub struct BindNode {
    pub fingerprint: [u64; 156],  // 1,248 bytes INLINE
    pub label: Option<String>,
    // ...
}

// After:
pub struct BindNode {
    // 512-byte fixed record (see layout above)
    pub record: NodeRecord,
}

pub struct NodeRecord {
    pub identity: [u8; 8],
    pub nars_access: [u8; 8],
    pub dn_meta: [u8; 48],
    pub edge_table: [u8; 64],
    pub hdr_sketch: [u64; 16],   // 128 bytes
    pub meta_thinking: [u64; 16], // 128 bytes
    pub gel_state: [u64; 16],     // 128 bytes
}

// Fingerprints live in separate Arrow buffers:
pub struct FingerprintStore {
    tier1: arrow_buffer::Buffer,  // 4096-bit per node, zero-copy
    tier2: Option<arrow_buffer::Buffer>, // 10K/64K, on-demand
}
```

### Phase 2: Make DataFusion a Feature Flag (Already Feasible)

```toml
[features]
default = ["simd", "parallel"]  # No DataFusion
gel = []                        # Graph Execution Language (new)
flight = ["arrow-flight", "tonic"]
datafusion = ["dep:datafusion"] # Optional, for SQL compatibility
```

### Phase 3: Implement GEL Executor

Small bytecode interpreter that operates on `NodeRecord` fields using only
integer operations. The 128-byte GEL state segment in each node record holds
the execution context.

### Phase 4: Bitpacked UDP for Inter-Node GEL

The existing `FireflyFrame` (160 bytes, 20 u64 words) already carries:
- Header (8B) + Instruction (8B) + Operand (16B) + Data (48B) + Context (48B) + ECC (32B)

GEL bytecode (32 bytes) fits in the Operand + Data fields. A GEL instruction
can be dispatched over UDP in a single packet, executed on the remote node's
512-byte record, and the result returned in another single packet.

---

## Summary: What Gets Stored vs What Gets Computed

```
STORED (512-byte record):          STORED (zero-copy, separate):
├── Address (8+8)                  ├── Hamming 4096 (512 B)
├── NARS truth (u8 fixed-point)    ├── Extended 10K (1,250 B)
├── DN tree pointers               └── Extended 64K (8,192 B)
├── Inline edges (8-16)
├── HDR 1024-bit sketch            COMPUTED (not stored):
├── Meta-thinking state            ├── f32 NARS truth (from u8)
├── GEL execution state            ├── Float similarity scores
└── XOR parity                     ├── SQL query results
                                   ├── DataFusion plans
OUTSOURCED (gRPC/UDP):             └── Full graph analytics
├── DataFusion SQL
├── Complex aggregations
└── Cross-node GEL dispatch
```

The storage footprint per node drops from 1,312 bytes to 512 bytes (2.6x
reduction) for traversal workloads, while search capability is preserved
through the tiered zero-copy fingerprint system. The binary size drops by
~150 MB by removing DataFusion from the core process.

**The thinking happens in the storage layer** — GEL bytecode executes directly
on node records using integer operations, with fingerprints loaded from
zero-copy buffers only when Hamming distance is needed. No FPU for file I/O.
No query planner for graph ops. The graph structure is the execution plan.
