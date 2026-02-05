# The 192×u64 Cognitive Record

**Date**: 2026-02-05
**Status**: Proposal (alternate to COMPOSITE_FINGERPRINT_SCHEMA.md)
**Core idea**: One 1,536-byte fixed-size record is the node, the edge row, the sparse adjacency vector, the NARS belief, the scent, AND the VSA fingerprint. Every cognitive operation reduces to SIMD on a lane of the same buffer. Zero-copy from storage through compute through transport.

---

## 0. Why 192

| Words | Bytes | AVX-512 iters | AVX2 iters | NEON iters | Wasted tail |
|-------|-------|---------------|------------|------------|-------------|
| 156   | 1,248 | 19 + 4 scalar | 39 + 0     | 78 + 0     | 4 words     |
| 157   | 1,256 | 19 + 5 scalar | 39 + 1     | 78 + 1     | 5 words     |
| 160   | 1,280 | 20 + 0        | 40 + 0     | 80 + 0     | 0           |
| **192** | **1,536** | **24 + 0** | **48 + 0** | **96 + 0** | **0**       |

192 = 2⁶ × 3. Divides cleanly by 8 (AVX-512), 4 (AVX2), 2 (NEON), 3 (lane split).
1,536 bytes = 24 cache lines at 64B. No partial lines.

192 × 64 = 12,288 bits. That's 2,288 bits more than the 10,000-bit fingerprint target. Those extra bits aren't wasted — they become the graph structure, the NARS state, the DN address, and the scent, all inside the same SIMD-friendly buffer. The record IS the node.

---

## 1. Lane Layout

```
192 u64 words = 3 lanes of 64 u64 each
Each lane = 512 bytes = 8 AVX-512 iterations = exactly 8 cache lines

┌─────────────────────────────────────────────────────────────────────────┐
│                    THE 192-WORD COGNITIVE RECORD                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Lane H (words 0-63):    HEADER + GRAPH STRUCTURE     512 bytes        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ words 0-7:    Composite key + DN + metadata       64 bytes     │   │
│  │ words 8-15:   Adjacency-OUT bitvector             64 bytes     │   │
│  │ words 16-23:  Adjacency-IN bitvector              64 bytes     │   │
│  │ words 24-31:  Verb-OUT mask (512 verb slots)      64 bytes     │   │
│  │ words 32-39:  Verb-IN mask (512 verb slots)       64 bytes     │   │
│  │ words 40-47:  NARS belief + scent (expanded)      64 bytes     │   │
│  │ words 48-55:  Edge weight sketch                  64 bytes     │   │
│  │ words 56-63:  Parity + ECC + spare                64 bytes     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Lane F (words 64-191):  SEMANTIC FINGERPRINT         1024 bytes       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 128 u64 words = 8,192 bits semantic payload                    │   │
│  │ Pure VSA: XOR bind, Hamming distance, Mexican hat              │   │
│  │ Exactly 16 AVX-512 iterations (128 / 8)                        │   │
│  │ Exactly 32 AVX2 iterations (128 / 4)                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Every lane boundary is 64-byte aligned (512 bytes = 8 cache lines)    │
│  Total: 64 + 128 = 192 words = 1,536 bytes                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why 8,192 bits for the fingerprint instead of 10,000

8,192 = 2¹³. Exactly 128 u64 words. This matters:

1. **Power-of-two popcount range**: max Hamming distance = 8,192. All popcount thresholds and similarity calculations use bit shifts instead of division. `similarity = 1.0 - (dist >> 13 as f32)` vs the current `dist as f32 / 10000.0`.

2. **XOR-fold scent**: fold 128 words → 1 word is exactly 7 XOR rounds (128→64→32→16→8→4→2→1). Current 156-word fold has an awkward remainder.

3. **ECC**: Hamming(13,8192) parity is 13 bits. Fits in one u16. The ECC bits live in Lane H (word 56-63), not in the fingerprint itself, so the fingerprint is purely semantic.

4. **Capacity**: 8,192 bits is 82% of 10,000. For binary VSA with XOR binding, information-theoretic capacity scales as `bits / ln(bits)` ≈ 908 vs 1,085 for 10K bits. This is a 16% capacity reduction, but the structural metadata we gain (graph adjacency, NARS, verbs) more than compensates because those signals would otherwise require separate data structures and joins.

If 10K bits are non-negotiable, use 160 words (10,240 bits) for Lane F and 32 words for Lane H. Total = 192. The lane split changes but the principle holds. The rest of this document uses the 64/128 split because it's cleaner.

---

## 2. Header Lane (Words 0-63): Byte-Level Layout

### Words 0-7: Composite Key + DN + Metadata (64 bytes)

```
word 0 (bytes 0-7):
  ┌──────┬──────┬────────────┬──────┬───────┬───────┬───────┐
  │ 0:7  │ 8:15 │ 16:31      │ 32:39│ 40:47 │ 48:55 │ 56:63 │
  │prefix│ slot │ context_id │ rung │ sigma │ etype │ flags │
  │ u8   │ u8   │ u16        │ u8   │ u8    │ u8    │ u8    │
  └──────┴──────┴────────────┴──────┴───────┴───────┴───────┘

word 1 (bytes 8-15):
  ┌────────────────────────────────┬─────────────────────────┐
  │ 0:47                           │ 48:63                   │
  │ group48                        │ popcount                │
  │ u48 (frozen path hash T1-T4)  │ u16 (fp popcount)       │
  └────────────────────────────────┴─────────────────────────┘

word 2 (bytes 16-23):
  ┌─────────────────────────────────────────────────────────┐
  │ disambig: u64 (content hash)                            │
  └─────────────────────────────────────────────────────────┘

word 3 (bytes 24-31):
  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
  │ T1   │ T2   │ T3   │ T4   │ T5   │ T6   │ T7   │ T8   │
  │ u8   │ u8   │ u8   │ u8   │ u8   │ u8   │ u8   │ u8   │
  │frozen│frozen│frozen│frozen│ ephem│ ephem│ ephem│ ephem│
  │anchor│anchor│anchor│anchor│ leaf │ leaf │ leaf │ leaf │
  └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘

word 4 (bytes 32-39):
  ┌──────────────┬──────────────┬─────────────────────────────┐
  │ 0:39         │ 40:47        │ 48:63                       │
  │ scent        │ depth        │ edge_count_out              │
  │ [u8;5]       │ u8           │ u16                         │
  └──────────────┴──────────────┴─────────────────────────────┘

word 5 (bytes 40-47):
  ┌───────────────────────────────┬───────────────────────────┐
  │ 0:31                          │ 32:63                     │
  │ parent_addr: u32              │ edge_count_in: u16 +      │
  │ (prefix:slot:group16)         │ access_count: u16         │
  └───────────────────────────────┴───────────────────────────┘

word 6 (bytes 48-55):
  ┌─────────────────────────────────────────────────────────┐
  │ updated_at: u64 (microseconds since epoch)              │
  └─────────────────────────────────────────────────────────┘

word 7 (bytes 56-63):
  ┌─────────────────────────────────────────────────────────┐
  │ reserved / future use / per-record CRC32 + spare        │
  └─────────────────────────────────────────────────────────┘
```

**L0 filtering requires only word 0 + word 1 + word 4 = 24 bytes.**
In practice, the CPU loads the entire 64-byte cache line containing words 0-7, so all metadata is free once word 0 is touched.

### Words 8-15: Adjacency-OUT (64 bytes = 512 bits)

Each bit represents a "bucket." Bit `b` = 1 means this node has at least one outgoing edge to a node whose `bucket_id = b`.

**Bucket assignment**: `bucket_id = hash(target_node.dn_anchor) & 0x1FF` (9-bit hash → 512 buckets). Each bucket covers ~128 addresses on average (65,536 / 512).

This is **one row of a sparse boolean adjacency matrix** in GraphBLAS terms. The matrix is 65,536 × 65,536, but we coarsen it to 512 × 512 by bucketing. The coarsened matrix is dense within each row (512 bits), sparse across rows.

```
Node A adjacency-OUT:  [0,0,1,0,0,1,0,0, 0,0,0,1,0,0,0,0, ...]
                         │       │             │
                         │       │             └── edge to bucket 11
                         │       └── edge to bucket 5
                         └── edge to bucket 2
```

### Words 16-23: Adjacency-IN (64 bytes = 512 bits)

Same as adjacency-OUT, but for incoming edges. Bit `b` = 1 means at least one incoming edge from a node in bucket `b`.

### Words 24-31: Verb-OUT mask (64 bytes = 512 bits)

Bit `v` = 1 means this node has at least one outgoing edge of verb type `v`. The verb slot space (0x07:0x00 through 0x07:0xFF = 256 verbs) is doubled to 512 to encode direction:
- Bits 0-255: outgoing verb types
- Bits 256-511: incoming verb types (mirrored in Verb-IN)

But since we have a dedicated Verb-OUT word range, we use all 512 bits for fine-grained outgoing verb typing, supporting up to 512 distinct relation types.

### Words 32-39: Verb-IN mask (64 bytes = 512 bits)

Mirror of Verb-OUT for incoming edges.

### Words 40-47: NARS Belief + Expanded Scent (64 bytes)

```
words 40-41 (16 bytes):
  ┌────────────┬────────────┬────────────┬────────────┐
  │ nars_f     │ nars_c     │ expect     │ horizon    │
  │ f32        │ f32        │ f32        │ f32        │
  │ frequency  │ confidence │ expectation│ temporal   │
  └────────────┴────────────┴────────────┴────────────┘

words 42-47 (48 bytes):
  Expanded scent: 384 bits
  The 5-byte (40-bit) scent from core/scent.rs is XOR-expanded
  to 48 bytes for SIMD-friendly comparison.
  scent_distance becomes a 48-byte XOR + popcount = 6 u64 ops.
  This kills ~90% of candidates at L0 using only this sub-lane.
```

### Words 48-55: Edge Weight Sketch (64 bytes = 512 bits)

For each of the 512 adjacency buckets, stores a 1-bit "strong/weak" flag indicating whether the max edge weight to that bucket exceeds a threshold. This enables "find nodes connected by strong edges" as a single AND with the adjacency bitvector.

Alternatively, this space can store a 512-element 1-bit "NARS confidence above threshold" sketch for all neighbors, enabling NARS-filtered graph traversal.

### Words 56-63: Parity + ECC (64 bytes)

```
words 56-57: XOR parity of Lane F (128 words XOR-folded to 2 words)
words 58-59: XOR parity of Lane H words 8-55
words 60-63: ECC bits + CRC32 + spare capacity
```

---

## 3. The GraphBLAS Connection: Semiring Operations on Lanes

### The Adjacency Matrix Lives Inside the Records

Think of the full dataset as a matrix **M** where:
- Row `i` = node `i`'s adjacency-OUT lane (words 8-15, 512 bits)
- **M[i][b]** = 1 if node `i` has an outgoing edge to bucket `b`

Standard GraphBLAS: `y = M ⊕.⊗ x` (matrix-vector multiply with semiring)

In ladybug, **BFS** on the coarsened graph is:

```rust
/// One BFS step using (OR, AND) semiring on adjacency lanes.
/// frontier: 512-bit bitvector of "active" buckets
/// returns:  512-bit bitvector of "reachable" buckets
fn bfs_step(
    records: &[CognitiveRecord],  // all nodes
    frontier: &[u64; 8],          // 512 bits: which buckets are active
) -> [u64; 8] {
    let mut next = [0u64; 8];
    for record in records {
        // Check if this node is in any active bucket
        let in_frontier = (0..8).any(|i|
            record.adjacency_out()[i] & frontier[i] != 0
        );
        if in_frontier {
            // OR this node's outgoing adjacency into next frontier
            for i in 0..8 {
                next[i] |= record.adjacency_out()[i];
            }
        }
    }
    next
}
```

This is `O(n)` per BFS step with pure bitwise operations. No pointer chasing. No adjacency list lookups. The inner loop is 8 u64 OR operations = **one AVX-512 instruction** per node.

### SPARQL / GQL Pattern Matching

```sparql
SELECT ?x ?z WHERE {
    ?x :CAUSES ?y .
    ?y :INHIBITS ?z .
}
```

Translates to:

```rust
fn pattern_match(records: &[CognitiveRecord]) -> Vec<(usize, usize)> {
    let causes_bit = verb_slot("CAUSES");    // e.g., bit 0x00
    let inhibits_bit = verb_slot("INHIBITS"); // e.g., bit 0x08

    let mut results = Vec::new();

    // Step 1: Find all ?x that have outgoing CAUSES
    let x_candidates: Vec<usize> = records.iter().enumerate()
        .filter(|(_, r)| r.verb_out_bit(causes_bit))
        .map(|(i, _)| i)
        .collect();

    // Step 2: OR their adjacency-OUT → candidate ?y buckets
    let mut y_buckets = [0u64; 8];
    for &xi in &x_candidates {
        for i in 0..8 {
            y_buckets[i] |= records[xi].adjacency_out()[i];
        }
    }

    // Step 3: Find ?y nodes in those buckets that have outgoing INHIBITS
    let y_candidates: Vec<usize> = records.iter().enumerate()
        .filter(|(_, r)|
            r.in_bucket_set(&y_buckets) && r.verb_out_bit(inhibits_bit)
        )
        .map(|(i, _)| i)
        .collect();

    // Step 4: OR their adjacency-OUT → candidate ?z buckets
    // (then resolve exact edges from edge table for precision)
    // ...
    results
}
```

**Key point**: Steps 1-3 never touch the fingerprint lane or the edge table. They operate entirely on the 512-bit bitvectors in Lane H. This is the "90% filter at L0" for graph patterns.

Exact edge resolution (which specific nodes, not just buckets) happens only for the surviving candidates, using the separate edge table or the CSR.

### NARS Inference in Storage

```rust
/// NARS deduction push-down: for all nodes reachable via CAUSES,
/// compute deduced truth value.
fn nars_deduction_scan(
    source: &CognitiveRecord,
    records: &[CognitiveRecord],
) -> Vec<(usize, f32, f32)> {
    let (f1, c1) = source.nars_fc();

    records.iter().enumerate()
        .filter(|(_, r)| {
            // Bucket overlap check (is this node reachable?)
            (0..8).any(|i| source.adjacency_out()[i] & r.bucket_mask()[i] != 0)
        })
        .map(|(idx, r)| {
            let (f2, c2) = r.nars_fc();
            // NARS deduction: f = f1*f2, c = f1*f2*c1*c2
            let f = f1 * f2;
            let c = f1 * f2 * c1 * c2;
            (idx, f, c)
        })
        .collect()
}
```

This runs inside the storage layer. No roundtrip. The planner pushes the deduction as a custom DataFusion operator.

---

## 4. Arrow Schema: Three Columns, One Record

### Physical Layout

```rust
use arrow_schema::{DataType, Field, Schema};

fn cognitive_record_schema() -> Schema {
    Schema::new(vec![
        // === Extracted scalar columns (for DataFusion predicate pushdown) ===
        Field::new("key",          DataType::FixedSizeBinary(16), false),
        Field::new("context_id",   DataType::UInt16,              false),
        Field::new("rung",         DataType::UInt8,               false),
        Field::new("dn_anchor",    DataType::FixedSizeBinary(4),  false),
        Field::new("dn_leaf",      DataType::FixedSizeBinary(4),  false),
        Field::new("scent",        DataType::FixedSizeBinary(5),  false),
        Field::new("popcount",     DataType::UInt16,              false),
        Field::new("nars_f",       DataType::Float32,             false),
        Field::new("nars_c",       DataType::Float32,             false),
        Field::new("label",        DataType::Utf8,                true),
        Field::new("entity_type",  DataType::Dictionary(
            Box::new(DataType::UInt8),
            Box::new(DataType::Utf8),
        ), false),

        // === The cognitive record (SIMD lanes) ===
        Field::new("header",       DataType::FixedSizeBinary(512),  false),
        Field::new("fingerprint",  DataType::FixedSizeBinary(1024), false),
    ])
}
```

**Why both extracted scalars AND the full record?**

The extracted scalar columns are **projections** of bytes already inside the header lane. They exist so DataFusion's standard `FilterExec` can push down `WHERE context_id = 0 AND rung <= 5` without a custom operator. They're maintained at write time (one extra write per field, negligible cost).

The `header` and `fingerprint` columns are the SIMD-friendly lanes. Custom operators (`HdrCascadeExec`, `GraphTraversalExec`, `NarsInferenceExec`) read these directly.

**This is NOT redundant storage.** The scalars are < 40 bytes/row. The lanes are 1,536 bytes/row. The scalars add 2.6% overhead in exchange for full DataFusion pushdown compatibility.

### Edge Table

```rust
fn edge_schema() -> Schema {
    Schema::new(vec![
        Field::new("src_key",     DataType::FixedSizeBinary(16),   false),
        Field::new("dst_key",     DataType::FixedSizeBinary(16),   false),
        Field::new("verb_id",     DataType::UInt16,                false),
        Field::new("bound_fp",    DataType::FixedSizeBinary(1024), false),
        Field::new("weight",      DataType::Float32,               false),
        Field::new("nars_f",      DataType::Float32,               false),
        Field::new("nars_c",      DataType::Float32,               false),
        Field::new("context_id",  DataType::UInt16,                false),
    ])
}
```

`bound_fp` = `src.fingerprint XOR verb.fingerprint XOR dst.fingerprint` (128 u64 words = 1,024 bytes). This is the edge's semantic signature. Unbinding recovers any component:
```
edge.bound_fp ⊕ src.fingerprint ⊕ verb.fingerprint = dst.fingerprint
```

Sort order: `(src_key, verb_id, dst_key)`. This gives adjacency-list locality: all edges from a source are contiguous, grouped by verb type.

### Alignment

- `FixedSizeBinary(512)`: rows at offsets `n * 512`. Since 512 = 64 × 8, every row is 64-byte aligned. AVX-512 aligned loads.
- `FixedSizeBinary(1024)`: rows at offsets `n * 1024`. 1024 = 64 × 16. Same guarantee.
- Combined: the header starts at `n * 512` and the fingerprint starts at `n * 1024` in their respective column buffers. Both are independently SIMD-aligned. Zero padding waste.

---

## 5. Solving the 8+8 Blocking Problem

### The Problem Restated

`Addr(prefix, slot)` is a 16-bit direct array index. One address → one slot in the `BindSpace` array. If exploration context A and exploration context B both want to modify the same conceptual entity, they fight over the same slot.

### The Solution: Stable Anchor + Ephemeral Stack

The cognitive record has word 3:
```
│ T1 │ T2 │ T3 │ T4 │ T5 │ T6 │ T7 │ T8 │
│ frozen  anchor      │ ephemeral leaf       │
```

Plus word 0 has `context_id: u16`.

```
┌─────────────────────────────────────────────────────────────────┐
│              MULTI-CONTEXT RESOLUTION                            │
│                                                                  │
│  LAYER 1: BindSpace arrays (context_id = 0)                     │
│    └── 65,536 slots, O(1) direct index                          │
│    └── THE hot path. This never changes.                        │
│                                                                  │
│  LAYER 2: Arrow "overlay" table (context_id > 0)                │
│    └── Same schema as above                                      │
│    └── Sorted by (dn_anchor, context_id, key)                   │
│    └── Each exploration context gets its own rows                │
│    └── TTL-governed: auto-expire after configurable timeout      │
│                                                                  │
│  RESOLUTION FLOW:                                                │
│                                                                  │
│    Query("give me node X"):                                      │
│      1. Read BindSpace[X] → base (context_id=0)                 │
│      2. If caller has context_id C:                              │
│         Scan overlay WHERE dn_anchor=X.anchor AND context_id=C  │
│      3. If overlay exists: merge (overlay wins for leaf fields)  │
│      4. If not: use base                                         │
│                                                                  │
│    Explore("try leaf path T5:T6:T7:T8 in context 7"):           │
│      1. Clone base record                                        │
│      2. Set T5-T8 to new leaf path, set context_id=7            │
│      3. Insert into overlay table (NOT into BindSpace)           │
│      4. Caller gets back the modified view                       │
│                                                                  │
│    Promote("context 7 wins, commit to base"):                    │
│      1. Read overlay record for context_id=7                     │
│      2. Write fingerprint + leaf into BindSpace[X]               │
│      3. Delete all overlay rows for this anchor                  │
│      4. Update adjacency bitvectors if edges changed             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Impact on sort adjacency:**

Primary sort: `(T1, T2, T3, T4, context_id, T5, T6, T7, T8)`

This means:
1. All contexts for the same frozen anchor are physically adjacent.
2. Base (context_id=0) is always first.
3. A subtree scan `WHERE T1=x AND T2=y` returns all contexts for all nodes in that subtree, contiguously.
4. Adding `AND context_id = 0` to any query restricts to base view. Since it's the first context_id in sort order, this is a prefix scan.

---

## 6. Sort Adjacency and XOR Delta Compression

### Primary Sort Key

```
(T1, T2, T3, T4, bucket_id, context_id, disambig)
  │    │    │    │     │          │          │
  │    │    │    │     │          │          └── uniqueness tiebreaker
  │    │    │    │     │          └── base before exploration
  │    │    │    │     └── similarity clustering
  │    │    │    └── frozen path (twig)
  │    │    └── frozen path (branch)
  │    └── frozen path (tree)
  └── frozen path (domain)
```

`bucket_id` is derived from the fingerprint's scent: `bucket_id = scent[0] as u16 | ((scent[1] as u16) << 8)`. Scent-similar fingerprints sort adjacent.

### What Sort Adjacency Buys

**Within a (T1, T2, T3, T4, bucket_id) group**, fingerprints are semantically similar. XOR delta between adjacent rows is sparse.

```
Row 0 fingerprint (128 words):    [a0, a1, a2, ..., a127]
Row 1 fingerprint:                [b0, b1, b2, ..., b127]
XOR delta:                        [a0^b0, a1^b1, ..., a127^b127]
                                   ~~~~~~ mostly zero ~~~~~~~~~
```

**Sparse delta encoding:**
```
┌──────────────────────────────┬────────────────────────────────┐
│ bitmask: [u64; 2] (128 bits)│ non-zero words (variable)      │
│ bit i = 1 if word i changed │ only changed words stored      │
└──────────────────────────────┴────────────────────────────────┘
```

If 80% of words match (same-bucket similarity), delta = 16 bytes mask + 25 × 8 bytes = 216 bytes. vs 1,024 bytes verbatim. **79% compression.**

**Delta decode at ingress:**
```
read delta → XOR with previous row → full fingerprint in Arrow buffer
```
This happens once, at the storage→buffer boundary. All operators downstream see fixed-size 1,024-byte fingerprints. Zero-copy from there.

For Lane H (512 bytes), deltas are even more sparse within a sorted group: only T5-T8, context_id, access_count, and timestamps differ. The adjacency bitvectors are often identical for nodes in the same subtree.

---

## 7. DataFusion Execution Pipeline

### Query: "DN prefix + query fingerprint → top-k nodes + edges with resonance"

```sql
SELECT
    n.key, n.label, n.nars_f, n.nars_c,
    hdr_distance(n.fingerprint, ?query_fp) AS distance,
    -- edges via graph lane (approximate)
    has_verb(n.header, 'CAUSES') AS has_causes
FROM cognitive_records n
WHERE n.dn_anchor >= X'80420000'
  AND n.dn_anchor <  X'80430000'
  AND n.context_id = 0
  AND n.rung <= 5
ORDER BY distance ASC
LIMIT 20
```

### Physical Plan

```
TopK(k=20, sort=distance ASC)
  └── HdrCascadeExec(query_fp, threshold=2000)
       │
       │  LANE-AWARE CASCADE:
       │
       │  L0: Read words 40-47 of header (expanded scent, 48 bytes)
       │      XOR + popcount vs query scent.
       │      If scent_distance > 8 → SKIP.
       │      Cost: 6 u64 XOR + popcount.
       │      Kills ~90% of candidates.
       │
       │  L1: Read word 1 of header (popcount field, bytes 48-49)
       │      |row.popcount - query.popcount| > tolerance → SKIP.
       │      Cost: 1 u16 subtract.
       │      Kills ~50% of L0 survivors.
       │
       │  L2: Read first 8 words of fingerprint (64 bytes = 1 cache line)
       │      Partial Hamming on 512 bits.
       │      Extrapolate: if partial > threshold * (512/8192) → SKIP.
       │      Cost: 8 u64 XOR + popcount = 1 AVX-512 instruction.
       │      Kills ~80% of L1 survivors.
       │
       │  L3: Full Hamming on fingerprint lane (128 words = 16 AVX-512 iters)
       │      Exact distance computation.
       │      Only ~0.2% of original input reaches this level.
       │
       │  L4: Mexican hat scoring on L3 survivors.
       │      Excite / inhibit / ignore classification.
       │
       │  Output: RecordBatch with appended _distance: UInt16 column
       │
       └── FilterExec(context_id = 0 AND rung <= 5)
            └── RangeFilterExec(dn_anchor range)
                 └── ParquetExec("cognitive_records",
                       projection=[key, label, nars_f, nars_c,
                                   dn_anchor, context_id, rung,
                                   header, fingerprint])
```

### Custom Physical Operators

**HdrCascadeExec**: Consumes `header` and `fingerprint` columns. Produces `_distance` column. Operates in streaming mode (no buffering). Lane H gives L0+L1 for free. Lane F gives L2+L3.

**GraphTraversalExec**: Consumes `header` column only (specifically words 8-39: adjacency + verb masks). Produces `_reachable` boolean column. Implements BFS/DFS on the coarsened 512-bucket graph.

```
GraphTraversalExec(src_key=X, verb="CAUSES", max_depth=3)
  └── Scan cognitive_records (projection=[key, header])
```

This never touches the fingerprint lane. 512 bytes/row instead of 1,536.

**NarsInferenceExec**: Reads NARS fields from header (words 40-41) plus adjacency (words 8-23). Computes deduction/induction/abduction on the fly. Produces `_inferred_f` and `_inferred_c` columns.

**Combined query** (semantic + structural):

```
TopK(k=10)
  └── HdrCascadeExec(query_fp, threshold=2000)
       └── GraphTraversalExec(src_key=X, verb="CAUSES", max_depth=2)
            └── FilterExec(context_id = 0)
                 └── ParquetExec(projection=[key, header, fingerprint])
```

GraphTraversalExec first narrows to structurally reachable nodes (using only the header lane = 512 bytes/row). Then HdrCascadeExec runs semantic search on survivors (1,024 bytes/row fingerprint lane). The fingerprint lane is only loaded for reachable nodes.

### Avoiding Materialization

The HdrCascadeExec optimizer rule rewrites `hdr_distance(n.fingerprint, ?query_fp) AS distance` in the SELECT to reference the pre-computed `_distance` column:

```rust
// Before optimization:
// Project: hdr_distance(fingerprint, literal) AS distance
// After:
// Project: _distance AS distance
```

Similarly, `has_verb(n.header, 'CAUSES')` is rewritten to a direct bit check on the header buffer, not a UDF call:

```rust
// has_verb(header, 'CAUSES') →
// header.word(24 + CAUSES_BIT/64).bit(CAUSES_BIT % 64)
```

---

## 8. The Semantic Kernel Thinking-in-Storage Model

```
┌─────────────────────────────────────────────────────────────────┐
│           SEMANTIC KERNEL → STORAGE OPERATOR MAPPING            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  SK Planner                                                      │
│    │                                                             │
│    ├── Skill: "resonate"                                         │
│    │    └── HdrCascadeExec on fingerprint lane                   │
│    │                                                             │
│    ├── Skill: "traverse"                                         │
│    │    └── GraphTraversalExec on header lane (adjacency)        │
│    │                                                             │
│    ├── Skill: "infer"                                            │
│    │    └── NarsInferenceExec on header lane (NARS + adjacency)  │
│    │                                                             │
│    ├── Skill: "bind"                                             │
│    │    └── XOR on fingerprint lane (in-place, zero-copy)        │
│    │                                                             │
│    ├── Skill: "counterfactual"                                   │
│    │    └── Clone record → new context_id → modify in overlay    │
│    │                                                             │
│    └── Skill: "compose" (chained)                                │
│         └── GraphTraversalExec → HdrCascadeExec → NarsExec       │
│             (pipeline, no intermediate materialization)           │
│                                                                  │
│  Memory:                                                         │
│    └── Arrow tables with DN-sorted cognitive records             │
│    └── BindSpace arrays for O(1) base view                       │
│    └── Overlay table for exploration contexts                    │
│                                                                  │
│  The storage layer IS the thinking substrate.                    │
│  Every cognitive operation is a SIMD pass on a lane.             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Concrete Rust Types

```rust
/// 192 × u64 cognitive record.
/// Lane H (words 0-63): header + graph structure.
/// Lane F (words 64-191): semantic fingerprint.
#[repr(C, align(64))]
pub struct CognitiveRecord {
    data: [u64; 192],
}

impl CognitiveRecord {
    // === Lane accessors (zero-copy slices) ===

    #[inline(always)]
    pub fn header(&self) -> &[u64; 64] {
        // SAFETY: data[0..64] is always valid
        unsafe { &*(self.data.as_ptr() as *const [u64; 64]) }
    }

    #[inline(always)]
    pub fn fingerprint(&self) -> &[u64; 128] {
        // SAFETY: data[64..192] is always valid
        unsafe { &*(self.data.as_ptr().add(64) as *const [u64; 128]) }
    }

    // === Sub-lane accessors ===

    #[inline(always)]
    pub fn adjacency_out(&self) -> &[u64; 8] {
        unsafe { &*(self.data.as_ptr().add(8) as *const [u64; 8]) }
    }

    #[inline(always)]
    pub fn adjacency_in(&self) -> &[u64; 8] {
        unsafe { &*(self.data.as_ptr().add(16) as *const [u64; 8]) }
    }

    #[inline(always)]
    pub fn verb_out(&self) -> &[u64; 8] {
        unsafe { &*(self.data.as_ptr().add(24) as *const [u64; 8]) }
    }

    #[inline(always)]
    pub fn verb_in(&self) -> &[u64; 8] {
        unsafe { &*(self.data.as_ptr().add(32) as *const [u64; 8]) }
    }

    // === Field accessors (bit manipulation on header) ===

    #[inline(always)]
    pub fn prefix(&self) -> u8 {
        self.data[0] as u8
    }

    #[inline(always)]
    pub fn slot(&self) -> u8 {
        (self.data[0] >> 8) as u8
    }

    #[inline(always)]
    pub fn context_id(&self) -> u16 {
        (self.data[0] >> 16) as u16
    }

    #[inline(always)]
    pub fn rung(&self) -> u8 {
        (self.data[0] >> 32) as u8
    }

    #[inline(always)]
    pub fn popcount(&self) -> u16 {
        (self.data[1] >> 48) as u16
    }

    #[inline(always)]
    pub fn dn_anchor(&self) -> [u8; 4] {
        let w = self.data[3].to_le_bytes();
        [w[0], w[1], w[2], w[3]]
    }

    #[inline(always)]
    pub fn dn_leaf(&self) -> [u8; 4] {
        let w = self.data[3].to_le_bytes();
        [w[4], w[5], w[6], w[7]]
    }

    #[inline(always)]
    pub fn nars_fc(&self) -> (f32, f32) {
        let f = f32::from_bits(self.data[40] as u32);
        let c = f32::from_bits((self.data[40] >> 32) as u32);
        (f, c)
    }

    #[inline(always)]
    pub fn scent_expanded(&self) -> &[u64; 6] {
        unsafe { &*(self.data.as_ptr().add(42) as *const [u64; 6]) }
    }

    // === GraphBLAS-style operations ===

    /// Check if this node has an edge to bucket b (outgoing)
    #[inline(always)]
    pub fn has_edge_to_bucket(&self, b: u16) -> bool {
        let word = (b / 64) as usize;
        let bit = (b % 64) as u64;
        self.adjacency_out()[word] & (1u64 << bit) != 0
    }

    /// Check if this node has outgoing verb v
    #[inline(always)]
    pub fn has_verb_out(&self, v: u16) -> bool {
        let word = (v / 64) as usize;
        let bit = (v % 64) as u64;
        self.verb_out()[word] & (1u64 << bit) != 0
    }

    /// Check if this node is in any bucket marked in the frontier set
    #[inline(always)]
    pub fn in_bucket_set(&self, frontier: &[u64; 8], bucket_id: u16) -> bool {
        let word = (bucket_id / 64) as usize;
        let bit = (bucket_id % 64) as u64;
        frontier[word] & (1u64 << bit) != 0
    }

    // === VSA operations on fingerprint lane ===

    /// Hamming distance (fingerprint lane only)
    pub fn hamming_fp(&self, other: &CognitiveRecord) -> u32 {
        let a = self.fingerprint();
        let b = other.fingerprint();
        let mut dist = 0u32;
        for i in 0..128 {
            dist += (a[i] ^ b[i]).count_ones();
        }
        dist
    }

    /// XOR bind (produces new record with bound fingerprint)
    pub fn bind_fp(&self, other: &CognitiveRecord) -> [u64; 128] {
        let a = self.fingerprint();
        let b = other.fingerprint();
        let mut result = [0u64; 128];
        for i in 0..128 {
            result[i] = a[i] ^ b[i];
        }
        result
    }

    /// Full-record Hamming (header lane + fingerprint lane)
    /// Useful for "structurally AND semantically similar" queries
    pub fn hamming_full(&self, other: &CognitiveRecord) -> u32 {
        let mut dist = 0u32;
        // Header lane (words 8-55, skip key/metadata at 0-7 and parity at 56-63)
        for i in 8..56 {
            dist += (self.data[i] ^ other.data[i]).count_ones();
        }
        // Fingerprint lane (words 64-191)
        for i in 64..192 {
            dist += (self.data[i] ^ other.data[i]).count_ones();
        }
        dist
    }
}
```

---

## 10. How It All Fits: Node, Edge, DN Tree, GQL, SPARQL, NARS, Scent, Semantic Kernel

| Capability | Where in the record | Operation | Zero-copy? |
|------------|---------------------|-----------|------------|
| **Node identity** | word 0-2 (key) | Direct array index | Yes |
| **DN tree traversal** | word 3 (T1-T8) | Sorted prefix scan | Yes |
| **DN parent** | word 5 (parent_addr) | O(1) lookup | Yes |
| **DN subtree** | words 0-3 sort key | Range scan on (T1,T2,T3,T4) | Yes |
| **Graph edge (approximate)** | words 8-15 (adjacency-OUT) | Bit check / OR / AND | Yes |
| **Graph edge (exact)** | Edge table join on src_key | Sorted merge join | Yes |
| **Edge direction** | words 8-15 (OUT) vs 16-23 (IN) | Lane select | Yes |
| **Verb filtering** | words 24-31 (verb-OUT) | Bit check | Yes |
| **GQL MATCH** | adjacency + verb lanes | Semiring BFS | Yes |
| **SPARQL pattern** | adjacency + verb lanes | Same as GQL | Yes |
| **SQL filter** | Extracted scalar columns | DataFusion FilterExec | Yes |
| **NARS inference** | words 40-41 (f, c, expect) | Arithmetic on 4 floats | Yes |
| **NARS deduction chain** | NARS fields + adjacency | Custom operator | Yes |
| **Scent L0 filter** | words 42-47 (expanded scent) | 48-byte XOR+popcount | Yes |
| **Popcount L1 filter** | word 1 bits 48-63 | u16 subtract | Yes |
| **HDR cascade L2-L3** | words 64-191 (fingerprint) | SIMD XOR+popcount | Yes |
| **Mexican hat** | After L3 | Float arithmetic on distance | Yes |
| **VSA bind/unbind** | words 64-191 | XOR on 128 words | Yes |
| **Semantic Kernel skill** | Mapped to physical operator | Pipeline composition | Yes |
| **Exploration context** | word 0 (context_id) + overlay | Extra rows, not slots | Yes |
| **ECC / integrity** | words 56-63 (parity) | XOR fold + verify | Yes |
| **Compression** | Sort-adjacent XOR deltas | Sparse encoding at storage boundary | Decode once |

---

## 11. Migration Path from Current Codebase

### What Changes

| Current | New | Migration |
|---------|-----|-----------|
| `FINGERPRINT_WORDS = 156` | `FP_WORDS = 128` (fingerprint lane) | Reindex. Existing 156-word fps: truncate last 28 words or pad to 128. |
| `BindNode` struct | `CognitiveRecord` struct | `BindNode` becomes a view over words 0-7 + 64-191. |
| `BindEdge` struct | Edge table + adjacency lanes | Edge data splits: exact edges in table, approximate in bitvectors. |
| `BitpackedCsr` | Adjacency bitvectors in Lane H | CSR still exists for exact edge lookup; bitvectors are the fast path. |
| `FingerprintBuffer` (lance_zero_copy) | Column buffer for `FixedSizeBinary(1024)` | Same concept, different size. `as_ptr()` still works. |
| `FingerprintTableProvider` | `CognitiveRecordProvider` | Expose header + fingerprint columns instead of flat bindspace. |
| `hdr_cascade.rs` WORDS=156 | WORDS=128 | Change constant. SIMD loops simplify (no remainder). |
| `cognitive_udfs.rs` FP_BYTES=1250 | FP_BYTES=1024 | Change constant. All UDFs work unchanged. |

### What Doesn't Change

- `Addr(u16)` — still works for O(1) BindSpace lookup
- `BindSpace` arrays — still the hot path for context_id=0
- DN path parsing — still `"Domain:tree:branch:twig"` strings
- CogRedis command syntax — unchanged
- Flight server protocol — unchanged (just different payload sizes)
- ACID transactions (XorDag) — unchanged
- All existing tests — still pass (with updated constants)

### Phase 1: MVP

```
1. Define CognitiveRecord struct with lane accessors
2. Change FP constants: WORDS=128, BYTES=1024
3. Update FingerprintBuffer to use 1024-byte fingerprints
4. Add adjacency bitvectors (initially zero — populated as edges are added)
5. Update FingerprintTableProvider to expose new schema
6. Update HdrCascadeExec constants
```

### Phase 2: Graph Lanes

```
7. Populate adjacency-OUT/IN bitvectors on edge insert
8. Populate verb-OUT/IN masks on edge insert
9. Implement GraphTraversalExec using adjacency lanes
10. Add bucket_id to sort key
```

### Phase 3: Cognitive Operators

```
11. Implement NarsInferenceExec
12. Add expanded scent to header lane
13. Implement context_id overlay table
14. Wire Semantic Kernel skills to physical operators
```

---

## 12. Tradeoffs

| Dimension | Impact |
|-----------|--------|
| **Fingerprint capacity** | 8,192 bits vs 10,000. 18% fewer bits. For VSA with ~50% density, capacity drops ~16%. Acceptable for most tasks; use 10,240-bit variant if needed. |
| **Memory per node** | 1,536 bytes (record) + ~40 bytes (scalar columns) = ~1,576 bytes vs current ~1,248 bytes (fingerprint only). 26% increase. For 32K nodes: 50MB → 49MB (the extracted columns are small). |
| **L0 filter speed** | Expanded scent (48 bytes) is 9.6× larger than current scent (5 bytes). But it enables SIMD (6 u64 XOR+popcount vs byte-by-byte loop). Net: ~2× faster on AVX2. |
| **Graph traversal** | New capability. Currently requires edge list scan. With adjacency bitvectors: O(1) per bucket check, O(n/512) for BFS step. |
| **SPARQL/GQL** | New capability. Pattern matching via bitvector semiring. Exact resolution via edge table for final results. |
| **NARS in storage** | New capability. Currently requires application-layer roundtrip. |
| **Engineering complexity** | Higher than Schema A from previous proposal. But the conceptual model is simpler: one record type, three operations (SIMD on header lane, SIMD on fingerprint lane, scalar on extracted columns). |
| **Compression** | Better within sorted groups (adjacency lanes are highly redundant for co-located nodes). Worse for random access (1,536 bytes vs 1,280 bytes cold read). |
| **Zero-copy** | Fully preserved. Every operator reads from Arrow column buffers via pointer arithmetic. No deserialization, no copies, no allocations in the hot path. |
