# Composite Fingerprint Schema Design

**Date**: 2026-02-05
**Status**: Proposal
**Scope**: Physical Arrow schema, 8+8 non-blocking, DataFusion pipeline, sort-adjacency

---

## 0. Constants Decision: 156 vs 157 u64 Words

**Answer: Use 160 u64 words (10,240 bits).**

Neither 156 nor 157 is correct. Here's why:

| Words | Bits   | SIMD tail (mod 8) | Problem |
|-------|--------|---------------------|---------|
| 156   | 9,984  | 4 remainder         | Misses ceil(10000/64)=157. Data loss. |
| 157   | 10,048 | 5 remainder         | 5-word scalar tail on every AVX-512 pass. |
| **160** | **10,240** | **0 remainder** | **Perfectly divides by 8 (AVX-512), 4 (AVX2), 2 (NEON).** |

160 u64 = 1,280 bytes. Cost: 24 extra bytes per fingerprint (1.9% overhead).
Benefit: zero scalar tail in *every* SIMD loop. The remainder loops in `simd.rs:60-76` and `hdr_cascade.rs:165-178` disappear entirely.

The extra 240 bits (10,240 - 10,000) serve as ECC/parity space. You already want Hamming ECC bits; they fit here.

```rust
pub const FP_WORDS: usize = 160;  // 10,240 bits, SIMD-clean
pub const FP_BYTES: usize = 1280; // 160 * 8
pub const FP_DATA_BITS: usize = 10_000; // semantic payload
pub const FP_ECC_BITS: usize = 240;     // Hamming parity + spare
```

Every SIMD function becomes:
```rust
// AVX-512: exactly 20 iterations, no remainder
for i in 0..20 { /* 8 words per iter */ }
// AVX2: exactly 40 iterations, no remainder
for i in 0..40 { /* 4 words per iter */ }
```

**All three schemas below use FP_WORDS=160.**

---

## 1. Three Candidate Schemas

### Schema A: "Wide Columnar" (One Column Per Concern)

Every field is its own Arrow column. Maximum DataFusion pushdown. Maximum sort flexibility.

```
RecordBatch "nodes"
┌───────────────────────────────────────────────────────────────────┐
│ Column              │ Arrow Type                  │ Bytes/row     │
├─────────────────────┼─────────────────────────────┼───────────────┤
│ key                 │ FixedSizeBinary(16)         │ 16            │
│   ├─ prefix    [0]  │   u8  (T1: domain)          │               │
│   ├─ slot      [1]  │   u8  (T2: tree)            │               │
│   ├─ group_hi  [2..7]│  u48 (T3+T4 frozen hash)   │               │
│   └─ disambig  [8..15]│ u64 (content hash)         │               │
│                     │                             │               │
│ dn_anchor           │ FixedSizeBinary(4)          │ 4             │
│   ├─ t1        [0]  │   u8  (domain)              │               │
│   ├─ t2        [1]  │   u8  (tree)                │               │
│   ├─ t3        [2]  │   u8  (branch)              │               │
│   └─ t4        [3]  │   u8  (twig)                │               │
│                     │                             │               │
│ context_id          │ UInt16                      │ 2             │
│ dn_leaf             │ FixedSizeBinary(4)          │ 4             │
│   ├─ t5        [0]  │   u8                        │               │
│   ├─ t6        [1]  │   u8                        │               │
│   ├─ t7        [2]  │   u8                        │               │
│   └─ t8        [3]  │   u8                        │               │
│                     │                             │               │
│ fingerprint         │ FixedSizeBinary(1280)       │ 1280          │
│ scent               │ FixedSizeBinary(5)          │ 5             │
│ popcount            │ UInt16                      │ 2             │
│ label               │ Utf8                        │ variable      │
│ rung                │ UInt8                       │ 1             │
│ sigma               │ UInt8                       │ 1             │
│ nars_f              │ Float32                     │ 4             │
│ nars_c              │ Float32                     │ 4             │
│ parent_key          │ FixedSizeBinary(16)         │ 16            │
│ entity_type         │ Dictionary(UInt8, Utf8)     │ 1 + dict      │
│ verb_mask           │ FixedSizeBinary(32)         │ 32  (256 bits)│
│ edge_count_out      │ UInt16                      │ 2             │
│ edge_count_in       │ UInt16                      │ 2             │
│ access_count        │ UInt32                      │ 4             │
│ updated_at          │ Timestamp(Microsecond, UTC) │ 8             │
├─────────────────────┼─────────────────────────────┼───────────────┤
│ TOTAL (fixed cols)  │                             │ ~1388 + var   │
└─────────────────────────────────────────────────────────────────────┘

RecordBatch "edges"
┌───────────────────────────────────────────────────────────────────┐
│ Column              │ Arrow Type                  │ Bytes/row     │
├─────────────────────┼─────────────────────────────┼───────────────┤
│ src_key             │ FixedSizeBinary(16)         │ 16            │
│ dst_key             │ FixedSizeBinary(16)         │ 16            │
│ verb                │ Dictionary(UInt8, Utf8)     │ 1 + dict      │
│ verb_addr           │ UInt16                      │ 2             │
│ bound_fp            │ FixedSizeBinary(1280)       │ 1280          │
│ weight              │ Float32                     │ 4             │
│ context_id          │ UInt16                      │ 2             │
│ nars_f              │ Float32                     │ 4             │
│ nars_c              │ Float32                     │ 4             │
├─────────────────────┼─────────────────────────────┼───────────────┤
│ TOTAL               │                             │ ~1329 + dict  │
└─────────────────────────────────────────────────────────────────────┘
```

**Key design points:**

- `key` is `FixedSizeBinary(16)`: first 2 bytes are prefix:slot (the current 8+8 Addr), bytes 2-7 are a 48-bit group hash (from the DN frozen path T1:T2:T3:T4), bytes 8-15 are a 64-bit content disambiguator. Sorting on `key` gives DN-locality for free.
- `dn_anchor` is the frozen T1-T4 path, separate from key, for direct equality filtering. 4 bytes, fits in a register.
- `context_id` solves the 8+8 blocking problem (see Section 3).
- `dn_leaf` is ephemeral T5-T8, separate column so it can be updated independently.
- `fingerprint` is `FixedSizeBinary(1280)` not `FixedSizeList<UInt64>` because FixedSizeBinary has zero overhead per element (no offsets array, no child array). The bytes are directly SIMD-accessible after a single `as_ptr()`.
- `scent` is pre-computed at write time. 5 bytes. Pushed down to L0 filter.
- `popcount` is pre-computed. Enables "popcount range" filters without touching the fingerprint column at all.
- `verb_mask` is a 256-bit bitvector: bit i=1 means "this node has at least one edge via verb slot i". Enables "has outgoing CAUSES edge?" as a bitwise AND on a 32-byte column, no edge join needed.
- `entity_type` uses Dictionary encoding. ~20 distinct types across 32K+ nodes. Dictionary saves ~15 bytes/row vs Utf8.
- `nars_f`, `nars_c` are top-level columns, not buried in a payload blob. This lets DataFusion push `WHERE nars_c > 0.5` directly.

**Alignment:**

`FixedSizeBinary(1280)` rows start at arbitrary byte offsets in Arrow buffers. For SIMD, the HDR cascade operator casts the pointer to `*const u64` — this requires 8-byte alignment, which Arrow guarantees (all buffers are 64-byte aligned, and 1280 is divisible by 64). No padding needed.

---

### Schema B: "Hot/Cold Split" (Routing Key + Centroid-Residual)

Separate the data into two batches: a narrow "routing" batch that fits in L2 cache, and a wide "payload" batch loaded only for survivors.

```
RecordBatch "routing" (hot — ~40 bytes/row, fits L2 for 100K nodes)
┌───────────────────────────────────────────────────────────────────┐
│ Column              │ Arrow Type                  │ Bytes/row     │
├─────────────────────┼─────────────────────────────┼───────────────┤
│ key                 │ FixedSizeBinary(16)         │ 16            │
│ dn_anchor           │ FixedSizeBinary(4)          │ 4             │
│ context_id          │ UInt16                      │ 2             │
│ scent               │ FixedSizeBinary(5)          │ 5             │
│ popcount            │ UInt16                      │ 2             │
│ bucket_id           │ UInt16                      │ 2             │
│ centroid_dist       │ UInt16                      │ 2             │
│ verb_mask           │ FixedSizeBinary(32)         │ 32            │
│ entity_type         │ Dictionary(UInt8, Utf8)     │ 1             │
│ rung                │ UInt8                       │ 1             │
├─────────────────────┼─────────────────────────────┼───────────────┤
│ TOTAL               │                             │ ~67           │
└─────────────────────────────────────────────────────────────────────┘

RecordBatch "payload" (cold — loaded after filter)
┌───────────────────────────────────────────────────────────────────┐
│ Column              │ Arrow Type                  │ Bytes/row     │
├─────────────────────┼─────────────────────────────┼───────────────┤
│ key                 │ FixedSizeBinary(16)         │ 16            │
│ fingerprint         │ FixedSizeBinary(1280)       │ 1280          │
│ dn_leaf             │ FixedSizeBinary(4)          │ 4             │
│ label               │ Utf8                        │ variable      │
│ nars_f              │ Float32                     │ 4             │
│ nars_c              │ Float32                     │ 4             │
│ parent_key          │ FixedSizeBinary(16)         │ 16            │
│ sigma               │ UInt8                       │ 1             │
│ payload             │ Binary                      │ variable      │
├─────────────────────┼─────────────────────────────┼───────────────┤
│ TOTAL               │                             │ ~1325 + var   │
└─────────────────────────────────────────────────────────────────────┘

Array "centroids" (256 cluster centers, persistent)
┌───────────────────────────────────────────────────────────────────┐
│ 256 × FixedSizeBinary(1280)  → 320 KB total (fits L2/L3)         │
│ Each centroid is the XOR-majority of its bucket members            │
└───────────────────────────────────────────────────────────────────┘
```

**How it works:**

1. On ingest, each fingerprint is assigned to the nearest centroid (by Hamming distance). `bucket_id` stores which centroid. `centroid_dist` stores the distance to that centroid.
2. Query arrives. Compute query distance to all 256 centroids (256 × 1280-byte comparisons = 320KB, fits L2). Sort centroids by distance.
3. For each centroid, use triangle inequality: if `|query_dist_to_centroid - centroid_dist| > threshold`, skip the row entirely. This is the "radius check" — a single u16 comparison per row.
4. Survivors (typically <10% of total) are looked up in the payload batch by key join.

**Sort key:** `(bucket_id, dn_anchor, key)`. This clusters same-centroid rows together, enabling:
- Streaming prefix scan: "give me all rows in bucket 7 with dn_anchor = [0x80, 0x42, 0x01, 0x00]"
- XOR delta compression within a bucket (adjacent fingerprints are similar, XOR residuals are sparse)

---

### Schema C: "Unified Envelope" (Composite Fixed-Size Record)

Single fixed-size row for maximum cache friendliness. No joins. Everything in one pass.

```
RecordBatch "bind_records"
┌───────────────────────────────────────────────────────────────────┐
│ Single record: FixedSizeBinary(1344) — 1 cache line overshoot     │
│                                                                   │
│ Offset  Bytes  Field            Encoding                          │
│ ──────  ─────  ───────────────  ──────────────────────────────── │
│ 0       2      prefix:slot      u8:u8 (the 8+8 addr)             │
│ 2       6      group48          u48 (frozen path hash T1-T4)      │
│ 8       8      disambig64       u64 (content hash)                │
│ 16      4      dn_anchor        [u8;4] (T1,T2,T3,T4 raw)         │
│ 20      2      context_id       u16                               │
│ 22      4      dn_leaf          [u8;4] (T5,T6,T7,T8)             │
│ 26      1      rung             u8                                │
│ 27      1      sigma            u8                                │
│ 28      4      nars_f           f32                               │
│ 32      4      nars_c           f32                               │
│ 36      2      popcount         u16                               │
│ 38      5      scent            [u8;5]                            │
│ 43      1      entity_type_id   u8 (index into external dict)     │
│ 44      32     verb_mask        [u8;32] (256-bit bitmask)         │
│ 76      2      edge_count_out   u16                               │
│ 78      2      access_count_hi  u16 (top 16 bits)                 │
│ 80      1200   fingerprint[0..149] first 150 u64 words (9600 bits)│
│ 1280    80     fingerprint[150..159] last 10 u64 words (640 bits) │
│ 1360    4      padding/checksum  u32 (CRC32 or zero)              │
│ ──────  ─────                                                     │
│ TOTAL   1364   (round to 1408 for 64B alignment? See below)       │
└───────────────────────────────────────────────────────────────────┘
```

Actually, this approach has a fatal problem: DataFusion can't push predicates into sub-fields of a single FixedSizeBinary blob. Every filter requires a custom physical operator. **Schema C is included for completeness but is NOT recommended.**

---

## 2. Byte Layouts and Alignment Details

### The 16-byte Composite Key

```
┌─────────────────────────────────────────────────────────────┐
│ byte  0: prefix   (u8)   ─── current Addr high byte        │
│ byte  1: slot     (u8)   ─── current Addr low byte         │
│ bytes 2-7: group48 (6B)  ─── hash(T1:T2:T3:T4)            │
│ bytes 8-15: disambig (u64) ── hash(full DN path + content) │
└─────────────────────────────────────────────────────────────┘
```

**Why 16 bytes?** `FixedSizeBinary(16)` fits in a single SSE register. Key comparison is one `_mm_cmpeq_epi8` + `_mm_movemask_epi8`. Arrow binary sort on this key is a single memcmp(16).

**Construction from DN path:**
```rust
fn make_key(path: &str, content_hash: u64) -> [u8; 16] {
    let segments: Vec<&str> = path.split(':').collect();
    let addr = dn_path_to_addr(path);

    // T1:T2:T3:T4 frozen hash (first 4 segments)
    let frozen = segments[..4.min(segments.len())].join(":");
    let group48 = hash48(&frozen);

    let mut key = [0u8; 16];
    key[0] = addr.prefix();
    key[1] = addr.slot();
    key[2..8].copy_from_slice(&group48);
    key[8..16].copy_from_slice(&content_hash.to_le_bytes());
    key
}
```

**Parent/child/sibling/subtree operations on this key:**

| Operation | How | Complexity |
|-----------|-----|------------|
| Parent | Truncate DN path string, recompute key. Or: follow `parent_key` column. | O(1) |
| Children | Filter `WHERE parent_key = my_key`. Sorted data = single range scan. | O(log n + k) |
| Siblings | Filter `WHERE parent_key = my_parent_key AND key != my_key`. | O(log n + k) |
| Subtree (T1:T2:T3) | Range scan: `WHERE dn_anchor[0..3] = [t1,t2,t3]`. Sorted = prefix scan. | O(log n + k) |
| DN prefix range | `WHERE key >= [prefix, 0, 0..0] AND key < [prefix+1, 0, 0..0]` | O(log n + k) |

### Fingerprint Column: FixedSizeBinary(1280) vs FixedSizeList<UInt64, 160>

| | FixedSizeBinary(1280) | FixedSizeList<UInt64>(160) |
|---|---|---|
| Overhead/row | 0 bytes | 0 bytes (no offsets for fixed-size) |
| Buffer count | 1 values buffer | 1 values buffer (inner UInt64) |
| SIMD access | `buf.as_ptr() as *const u64` | `inner.values().as_ptr()` |
| DataFusion filter pushdown | Opaque blob — needs UDF | Can use element-wise ops |
| Zero-copy to BindSpace | `std::slice::from_raw_parts` | Direct pointer cast |
| IPC serialization | 1 buffer, trivial | 1 buffer + child array metadata |

**Recommendation: `FixedSizeBinary(1280)`**. The fingerprint is always consumed as a whole (Hamming distance, XOR bind). Element-wise column access on individual u64 words is never needed. The simpler wire format and fewer indirections win.

### Alignment Guarantee

Arrow specification: all buffers are allocated with 64-byte alignment. `FixedSizeBinary(1280)` stores rows contiguously. Row N starts at offset `N * 1280`. Since 1280 = 64 × 20, every row starts at a 64-byte boundary. AVX-512 aligned loads (`_mm512_load_si512`) work without penalty.

---

## 3. Solving the 8+8 Blocking Problem

### The Problem

The current `Addr(u16)` assigns one prefix:slot per entity. In storage, `BindSpace::read(addr)` returns a single `Option<&BindNode>`. If two exploration contexts want to read the same conceptual entity with different leaf decorations (T5-T8), they collide: the second context overwrites the first.

### The Solution: (dn_anchor, context_id, dn_leaf) Triple

```
Storage key = (dn_anchor, context_id)
                │             │
                │             └── u16: which exploration frame
                └── [u8;4]: frozen T1:T2:T3:T4 identity
```

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CONTEXT OVERLAY MODEL                            │
│                                                                         │
│  dn_anchor = [0x80, 0x42, 0x01, 0x00]  (frozen identity: "Ada:A:soul") │
│                                                                         │
│  context_id=0 (base)    │ dn_leaf=[00,00,00,00] │ fp = base_fp         │
│  context_id=1 (explore) │ dn_leaf=[01,03,00,00] │ fp = base_fp ^ delta │
│  context_id=2 (explore) │ dn_leaf=[01,05,00,00] │ fp = base_fp ^ delta │
│  context_id=3 (commit)  │ dn_leaf=[01,03,00,00] │ fp = resolved_fp     │
│                                                                         │
│  Base row (context_id=0) is always present.                             │
│  Exploration rows are ephemeral (TTL or explicit delete).               │
│  When exploration resolves, winner is promoted to context_id=0.         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Concrete Arrow representation:**

```rust
// Primary sort key for "nodes" RecordBatch:
// (dn_anchor, context_id, key)
//
// This means: all contexts for the same anchor are adjacent.
// Base (context_id=0) comes first. Exploratory contexts follow.
```

**How queries use this:**

```sql
-- "Give me the base view" (blocking-compatible, same as today)
SELECT * FROM nodes WHERE context_id = 0

-- "Give me exploration context 7's view"
-- Uses COALESCE: if context 7 has an override, use it; else fall back to base
SELECT
    base.key,
    COALESCE(ctx.dn_leaf, base.dn_leaf) AS dn_leaf,
    COALESCE(ctx.fingerprint, base.fingerprint) AS fingerprint,
    base.dn_anchor
FROM nodes base
LEFT JOIN nodes ctx
    ON base.dn_anchor = ctx.dn_anchor
    AND ctx.context_id = 7
WHERE base.context_id = 0

-- "How many concurrent explorations exist for anchor X?"
SELECT COUNT(DISTINCT context_id)
FROM nodes
WHERE dn_anchor = X'80420100'
```

**Impact on BindSpace in-memory model:**

The current `BindSpace` arrays don't change. They represent `context_id=0`. Exploration contexts live as extra rows in the Arrow-backed storage (DataFusion tables). The hot path (`BindSpace::read`) stays O(1) array lookup. Exploration contexts are queried through DataFusion SQL, which is slightly slower but that's acceptable for exploration (not the critical path).

```
┌─────────────────────────────────────────────────┐
│ HOT PATH (context_id=0):  BindSpace arrays      │
│   └── O(1) array[prefix][slot]                  │
│                                                  │
│ EXPLORATION (context_id>0): Arrow RecordBatch    │
│   └── O(log n) sort-based range scan             │
│   └── Non-blocking: each context is a row        │
│   └── TTL: auto-expire after N seconds           │
│                                                  │
│ PROMOTION: exploration → base                    │
│   └── Write winner's fp/leaf into BindSpace      │
│   └── Delete exploration rows                    │
└─────────────────────────────────────────────────┘
```

**Impact on index keys and sorting adjacency:**

Primary sort order becomes: `(dn_anchor, context_id, key)`. This means:
1. All versions of an entity are physically adjacent on disk/in buffer.
2. Base is always first (context_id=0 sorts before exploration contexts).
3. Range scan `WHERE dn_anchor = X` returns base + all explorations in one sequential read.
4. Within-anchor XOR deltas are small (explorations differ by leaf context, not identity).

---

## 4. Sort Adjacency and XOR Delta Compression

### Primary Sort Key

```
Sort order: (dn_anchor[4], bucket_id[2], context_id[2], key[16])
             ──────────── ──────────── ──────────── ────────────
             locality     similarity   versioning   uniqueness
```

**Why this order:**

1. **dn_anchor first**: DN tree traversal (`subtree(T1:T2:T3)`) becomes a prefix range scan. All nodes under `Ada:A:soul` are contiguous.

2. **bucket_id second**: Within a subtree, nodes are clustered by fingerprint similarity (same Hamming-centroid bucket). Adjacent fingerprints are similar → XOR deltas are sparse.

3. **context_id third**: Base views cluster before exploration views. A query that only wants `context_id=0` reads a contiguous prefix and stops.

4. **key last**: Tiebreaker for uniqueness.

### How This Enables XOR Delta Compression

Within a `(dn_anchor, bucket_id)` group, adjacent fingerprints share high similarity. Store the first fingerprint verbatim, then store XOR deltas for subsequent rows:

```
Row 0: fp_0                           (1280 bytes, verbatim)
Row 1: fp_1 XOR fp_0                  (sparse — most words are 0)
Row 2: fp_2 XOR fp_1                  (sparse)
...
```

Sparse encoding of a 1280-byte XOR delta where ~80% of words are zero:

```
┌────────────┬────────────────────────────┐
│ mask[20B]  │ non-zero words (variable)  │
│ 160 bits   │ only set words             │
│ bit i = 1  │                            │
│ if word i  │                            │
│ differs    │                            │
└────────────┴────────────────────────────┘
```

If 80% of 160 words are zero, the delta is: 20 bytes mask + 32 × 8 bytes = 276 bytes (78% compression).

**When to compress:** Only for cold storage / IPC transport. In-memory compute buffers stay fixed-size `FixedSizeBinary(1280)`. Decode happens at ingress (once), then all operators work on fixed-size SIMD-friendly buffers.

### Decode Pipeline

```
Transport (compressed) → Ingress decode → Arrow Buffer (fixed 1280B/row) → Operators
                                          ↑
                                          This is the zero-copy boundary.
                                          Everything after this is pointer math.
```

---

## 5. DataFusion Execution Pipeline

### Example Query: "Given DN prefix + query fingerprint, return top-k nodes and edges with resonance"

```sql
SELECT
    n.key,
    n.label,
    n.fingerprint,
    hamming(n.fingerprint, ?query_fp) AS distance,
    similarity(n.fingerprint, ?query_fp) AS sim,
    n.nars_f,
    n.nars_c
FROM nodes n
WHERE n.dn_anchor >= X'80420000'    -- subtree prefix scan
  AND n.dn_anchor <  X'80430000'
  AND n.context_id = 0              -- base view only
  AND n.rung <= 5                   -- access control
  AND scent_distance(n.scent, ?query_scent) <= 8  -- L0 filter (5 bytes)
  AND hamming(n.fingerprint, ?query_fp) < 2000     -- L3 exact filter
ORDER BY distance ASC
LIMIT 20
```

### Logical Plan

```
Limit(20)
  └── Sort(distance ASC)
       └── Project(key, label, fingerprint, hamming(...) AS distance, similarity(...) AS sim, nars_f, nars_c)
            └── Filter(hamming(...) < 2000)
                 └── Filter(scent_distance(...) <= 8)
                      └── Filter(rung <= 5 AND context_id = 0)
                           └── Filter(dn_anchor range)
                                └── TableScan("nodes")
```

### Physical Plan (with Custom HDR Cascade Operator)

```
TopK(k=20, sort=distance ASC)                        ← avoids full sort
  └── ProjectionExec(distance = hamming UDF)
       └── HdrCascadeExec                             ← CUSTOM PHYSICAL OPERATOR
            │
            │  Input: RecordBatch stream (post dn_anchor + context + rung filter)
            │
            │  Internally:
            │  ┌──────────────────────────────────────────────────────────┐
            │  │ L0: scent_distance(row.scent, query_scent) > 8 → SKIP  │
            │  │     Cost: 5-byte XOR+popcount. Kills ~90%.              │
            │  │                                                         │
            │  │ L1: popcount_diff = |row.popcount - query_popcount|     │
            │  │     if popcount_diff > threshold → SKIP                 │
            │  │     Cost: 1 u16 subtract. Kills ~50% of L0 survivors.  │
            │  │                                                         │
            │  │ L2: sketch_4bit_sum(row.fp, query.fp) > threshold → SKIP│
            │  │     Cost: 78-byte scan. Kills ~80% of L1 survivors.    │
            │  │                                                         │
            │  │ L3: hamming_full(row.fp, query.fp) against threshold    │
            │  │     Cost: 1280-byte SIMD. Only ~0.1% of input reaches. │
            │  │                                                         │
            │  │ L4: Mexican hat discrimination                          │
            │  │     Excite/inhibit scoring on L3 survivors.             │
            │  └──────────────────────────────────────────────────────────┘
            │
            │  Output: RecordBatch of survivors with `distance` column appended
            │
            └── FilterExec(rung <= 5 AND context_id = 0)
                 └── SortPreservingRepartitionExec
                      └── RangeFilterExec(dn_anchor range scan)
                           └── ParquetScan("nodes", projection=[...])
```

### Where Each Concern Lives

| Concern | Where | Why |
|---------|-------|-----|
| DN prefix range | `RangeFilterExec` or Parquet row group pruning | Sorted data → skip entire row groups |
| `context_id = 0` | `FilterExec` (standard DataFusion) | Trivial column predicate |
| `rung <= 5` | `FilterExec` (standard DataFusion) | Trivial column predicate |
| Scent L0 filter | Inside `HdrCascadeExec` | Needs access to query fingerprint |
| Popcount L1 filter | Inside `HdrCascadeExec` | Needs pre-computed popcount column |
| 4-bit sketch L2 | Inside `HdrCascadeExec` | Needs fingerprint column bytes |
| Full Hamming L3 | Inside `HdrCascadeExec` | SIMD on fingerprint column |
| Mexican hat L4 | Inside `HdrCascadeExec` | Scoring on L3 distance |
| `hamming()` UDF for output | `ProjectionExec` | Reuses distance already computed in HDR |
| Top-k | `TopKExec` (DataFusion built-in) | Heap-based partial sort |

### Avoiding Accidental Materialization

**Problem:** If `hamming()` is used in both `WHERE` and `SELECT`, DataFusion may compute it twice — and worse, may materialize intermediate batches.

**Solution:** The `HdrCascadeExec` operator appends a `distance: UInt32` column to its output batches. The projection above it reads this pre-computed column instead of calling the UDF again. Implementation:

```rust
impl ExecutionPlan for HdrCascadeExec {
    fn schema(&self) -> SchemaRef {
        // Output schema = input schema + "distance" UInt32 column
        let mut fields = self.input.schema().fields().to_vec();
        fields.push(Arc::new(Field::new("_hdr_distance", DataType::UInt32, false)));
        Arc::new(Schema::new(fields))
    }

    fn execute(&self, partition: usize, ctx: Arc<TaskContext>) -> Result<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, ctx)?;
        Ok(Box::pin(HdrCascadeStream {
            input: input_stream,
            query_fp: self.query_fp.clone(),
            query_scent: self.query_scent,
            query_popcount: self.query_popcount,
            threshold: self.threshold,
            mexican_hat: self.mexican_hat,
        }))
    }
}
```

The `ProjectionExec` above is rewritten by a custom optimizer rule to reference `_hdr_distance` instead of calling `hamming()`:

```rust
// Optimizer rule: HdrDistanceRewrite
// Before: hamming(n.fingerprint, ?query_fp) AS distance
// After:  n._hdr_distance AS distance
```

### XOR Delta Decode: Where It Happens

```
                              Ingress boundary
                                    │
  Parquet (compressed)  ───────────►│──────────► Arrow Buffer (1280B fixed rows)
  IPC (XOR delta encoded) ─────────►│              │
  Flight (XOR delta encoded) ──────►│              │
                                    │              ▼
                                    │    HdrCascadeExec reads fixed-size rows
                                    │    via as_ptr() — zero-copy from here on
```

Delta decode happens once, at the boundary between transport/storage and the in-process Arrow buffer pool. After decode, every operator sees `FixedSizeBinary(1280)` with guaranteed alignment.

### Edge Query: "Given node X, find top-k edges by resonance"

```sql
SELECT
    e.dst_key,
    e.verb,
    hamming(e.bound_fp, ?query_fp) AS edge_distance,
    -- Unbind: recover destination from edge + source + verb
    xor_bind(xor_bind(e.bound_fp, n_src.fingerprint), v.fingerprint) AS recovered_fp
FROM edges e
JOIN nodes n_src ON e.src_key = n_src.key AND n_src.context_id = 0
JOIN nodes v ON e.verb_addr = v.key
WHERE e.src_key = ?source_key
ORDER BY edge_distance ASC
LIMIT 10
```

The `xor_bind` UDF chains to implement unbind: `edge ⊕ source ⊕ verb = target`. This is computed *in the projection*, after the join, on only the top-k candidates. No materialization of all unbinds.

---

## 6. Recommendation

### Pick: Schema A (Wide Columnar)

**Schema A is the recommendation.** Here's why:

| Criterion | Schema A (Wide) | Schema B (Hot/Cold) | Schema C (Envelope) |
|-----------|-----------------|---------------------|---------------------|
| DataFusion pushdown | Full (every field is a column) | Partial (hot batch only) | None (opaque blob) |
| Cache behavior | Good (column pruning skips unused cols) | Best (hot batch fits L2) | Mixed (always loads 1.3KB) |
| Engineering effort | Low (standard Arrow) | Medium (two-batch join) | High (custom everything) |
| Sort adjacency | Natural (sort on any column) | Natural (within each batch) | Possible but manual |
| Context overlay | Clean (context_id is a column) | Clean | Awkward (inside blob) |
| Upgrade path | Add columns freely | Need to update both batches | Redefine blob format |
| Zero-copy to BindSpace | `as_ptr()` on fingerprint col | Join first, then `as_ptr()` | Offset arithmetic |

**Schema B** is better if you have >1M nodes and the L2 cache pressure matters. It's the upgrade path from Schema A: extract the hot columns into a separate batch when profiling shows column-pruned scans are still too wide. But for the current 32K-65K address space, Schema A's column pruning is sufficient.

**Schema C** is an anti-pattern in Arrow/DataFusion. Don't do it.

### Tradeoffs

| Dimension | Schema A Impact |
|-----------|----------------|
| **Memory** | ~1.4KB/node × 32K nodes = ~45MB for full node table. Fits in RAM trivially. Column pruning means a "key + scent + popcount" scan touches only ~25 bytes/row = 800KB. |
| **CPU** | HDR cascade kills 99.9% of candidates before full Hamming. Full Hamming on survivors: 20 AVX-512 iterations (vs 19.5 + scalar today). |
| **Recall** | Identical to current — no approximation. HDR cascade is exact filtering, not approximate. |
| **Precision** | Mexican hat discrimination improves precision over raw Hamming by suppressing near-duplicates (inhibition zone). |
| **Engineering** | Requires: (1) fix FP_WORDS to 160, (2) add `HdrCascadeExec` as DataFusion physical operator, (3) add `context_id` + `dn_leaf` columns, (4) add optimizer rule for distance reuse. |

### MVP Layout (Minimum Viable Schema)

Start with these columns only:

```rust
// MVP — enough for resonance search + DN traversal + non-blocking contexts
let schema = Schema::new(vec![
    Field::new("key",         DataType::FixedSizeBinary(16), false),
    Field::new("dn_anchor",   DataType::FixedSizeBinary(4),  false),
    Field::new("context_id",  DataType::UInt16,              false),
    Field::new("fingerprint", DataType::FixedSizeBinary(1280), false),
    Field::new("scent",       DataType::FixedSizeBinary(5),  false),
    Field::new("popcount",    DataType::UInt16,              false),
    Field::new("rung",        DataType::UInt8,               false),
    Field::new("parent_key",  DataType::FixedSizeBinary(16), true),
    Field::new("label",       DataType::Utf8,                true),
]);
```

9 columns, ~1310 bytes/row fixed + label. Supports: HDR cascade search, DN subtree traversal, context overlay, access control.

### Upgrade Path

```
MVP (9 cols)
  │
  ├── +nars_f, +nars_c, +sigma       (NARS reasoning signals)
  ├── +verb_mask                       (edge existence bitmap)
  ├── +entity_type (Dictionary)        (semantic typing)
  ├── +dn_leaf                         (ephemeral exploration state)
  ├── +edge_count_out, +edge_count_in  (degree stats)
  │
  ▼
Full Schema A (17+ cols)
  │
  ├── If >1M nodes: extract hot cols → Schema B hot/cold split
  ├── If centroid radius calibration needed: add bucket_id + centroid_dist
  │
  ▼
Schema B (hot/cold, centroid-residual)
```

Each step is additive. No schema migrations break existing queries.
