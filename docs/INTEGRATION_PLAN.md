# Integration Plan: CogRecord + Lance + CogRedis

> **Date**: 2026-02-06
> **Status**: Design specification
> **Constraint**: Many nodes, max 32 edges per node, zero copy, zero GPU, zero LLM

---

## 1. The Thesis

Cognition IS the storage layout. A single `[u64; 256]` CogRecord is:
- The computation unit (Hamming, XOR bind, superposition)
- The storage unit (Lance `FixedSizeBinary(2048)`)
- The transport unit (Flight Arrow batch)
- The addressing unit (8+8 prefix:slot → array index)

No deserialization. No ETL. No load step. Read the record, compute on it, write it back.

---

## 2. Addressing: Three Layers

```
Layer 1: 8+8 = 16 bits (BindSpace position)
  prefix:slot → array index → 3-5 cycles
  ┌────────┬────────┐
  │ prefix │  slot  │  = Addr(u16)
  │ 8 bits │ 8 bits │
  └────────┴────────┘
  Surfaces: 0x00-0x0F (4,096 CAM operations)
  Fluid:    0x10-0x7F (28,672 working memory)
  Nodes:    0x80-0xFF (32,768 DN tree nodes)

Layer 2: 16+48 = 64 bits (Content-Addressable Key)
  ┌──────────┬──────────────────────────────┐
  │ type_id  │  48-bit fingerprint prefix   │  = Key(u64)
  │ 16 bits  │  48 bits                     │
  └──────────┴──────────────────────────────┘
  type_id: THOUGHT=0x0001, CONCEPT=0x0002, EDGE_CAUSES=0x0100, ...
  48 bits: first 6 bytes of fingerprint → collision-resistant lookup
  Used by: CAM index, Lance row offset mapping

Layer 3: 16,384 bits (CogRecord payload)
  ┌──────────────────────────────────────────┐
  │ [u64; 256] = 2,048 bytes = 16,384 bits  │
  │                                          │
  │ DENSE:  10,048 bits (fingerprint)        │
  │ STRUCT:  6,336 bits (metadata + edges)   │
  └──────────────────────────────────────────┘
```

The 48-bit layer bridges BindSpace addresses and Lance rows:
- BindSpace: `Addr(0x8042)` → `BindNode` (in-memory, O(1))
- CAM Key:   `Key(0x0001_ABCDEF123456)` → Lance row offset (persistent)
- CogRecord: the actual 2KB payload at both locations

---

## 3. CogRecord Layout: 256 x u64

### Compartment Map

```
 Word    Bytes    Bits      Content
 ─────── ──────── ───────── ─────────────────────────────────────────
 COMPARTMENT 0: FINGERPRINT (dense, SIMD-aligned)
 [0..156]  0-1255  0-10047   10,048-bit fingerprint (10,000 effective)
                              ├── Can encapsulate compressed 1024D
                              ├── Binary quantized Jina embedding
                              ├── OR LFSR-expanded content hash
                              └── 48 padding bits (bits 10000-10047)

 COMPARTMENT 1: ADDRESS + KEY
 [157]     1256    10048     ┌── prefix:   u8  (8+8 BindSpace addr)
                             ├── slot:     u8
                             ├── type_id:  u16 (CAM type namespace)
                             ├── fp48:     [u8; 3] (first 24 bits of fp prefix,
                             │             remaining 24 in word 0-2)
                             └── flags:    u8  (dirty, frozen, tombstone)

 COMPARTMENT 2: DN TREE
 [158]     1264    10112     ┌── parent:   u16 (Addr of parent, 0=root)
                             ├── depth:    u8  (0=root, max 255)
                             ├── rung:     u8  (R0=public..R9=soul)
                             ├── sigma:    u8  (reasoning depth)
                             ├── node_type: u8  (domain=0,tree=1,branch=2,
                             │                   twig=3,leaf=4)
                             └── reserved: u16

 COMPARTMENT 3: TRUTH + QUALIA (NARS)
 [159]     1272    10176     ┌── freq:     u16 (truth frequency, 0-65535)
                             ├── conf:     u16 (confidence)
                             ├── evid_pos: u8  (positive evidence count)
                             ├── evid_neg: u8  (negative evidence count)
                             ├── qidx:     u8  (qualia index 0-255)
                             └── priority:  u8  (scheduling priority)

 COMPARTMENT 4: THINKING STYLE
 [160-161]  1280   10240     7 x f16 thinking style axes (14 bytes):
                              ├── analytical..intuitive
                              ├── focused..diffuse
                              ├── convergent..divergent
                              ├── concrete..abstract
                              ├── sequential..holistic
                              ├── verbal..spatial
                              └── deliberate..automatic
                              + 2 bytes reserved

 COMPARTMENT 5: TEMPORAL
 [162-163]  1296   10368     ┌── created_at:  u64 (microseconds since epoch)
                             └── updated_at:  u64

 COMPARTMENT 6: VERSION / PARITY
 [164]     1312    10496     ┌── version:    u32 (monotonic, for XorDag)
                             ├── parity_gen: u16 (parity block generation)
                             └── access_cnt: u16 (for TOCTOU detection)

 COMPARTMENT 7: EMBEDDING SUMMARY
 [165-180]  1320   10560     1,024 bits = compressed centroid
                              ├── Binary quantized 1024D (1 bit per dim)
                              ├── Used for fast pre-filter before full
                              │   Hamming on compartment 0
                              └── Lance ANN can index this column separately

 COMPARTMENT 8: EDGES (max 32)
 [181]     1448    11584     edge_count: u8 | edge_generation: u8 |
                             reserved: [u8; 6]

 [182-213]  1456   11648     edges[32] x 8 bytes each:
                              ┌── target: u16 (Addr of target node)
                              ├── verb:   u16 (Addr of verb in surface 0x07)
                              ├── weight: u8  (0-255, quantized)
                              ├── amp:    u8  (amplification 0-255)
                              └── flags:  u16 (bidirectional, temporal, etc.)

 COMPARTMENT 9: FREE / FUTURE
 [214-255]  1712   13696     42 x u64 = 2,688 bits FREE
                              Candidates:
                              ├── Content payload hash (256 bits)
                              ├── Cognitive gate state (collapse/superposition)
                              ├── Learning rate / plasticity params
                              ├── Attention weights (for transformer-like ops)
                              └── User-defined extensions
```

### Striped Read Patterns

```
Operation                  Compartments Read    Bytes    Cache Lines
──────────────────────────────────────────────────────────────────────
Hamming distance           [0..156]             1256     ~20
Edge traversal             [181..213] + [157]    264     ~5
DN tree walk               [158] + [181..213]    264     ~5
Truth revision (NARS)      [159]                   8     1
Full cognitive step        ALL                  2048     32
Vector ANN pre-filter      [165-180]             128     2
Superposition check        [0..156] + [159]     1264     ~20
```

In Lance columnar storage, each compartment CAN be a separate column for
stripe-read efficiency. But for zero-copy BindSpace ↔ Lance, storing as
one `FixedSizeBinary(2048)` is simpler — the record IS the cognitive unit,
you always want the whole thing for a cognitive step.

**Hybrid approach**: Store as ONE column for hot-path access. Maintain
derived materialized columns (fingerprint-only, edges-only) for Lance
ANN indexing and DataFusion SQL queries.

---

## 4. Fingerprint Compartment: 10K Bits, Two Modes

The 10,048-bit fingerprint serves dual purpose:

### Mode A: Content-Hashed (current, LFSR)

```
content string → DefaultHasher → 64-bit seed → LFSR expansion → 10K bits
```

- Deterministic (same content → same fingerprint)
- Quasi-random distribution (good Hamming spread)
- No semantic similarity — "cat" and "kitten" are ~5000 bits apart
- Used for: exact identity, codebook lookup, XOR binding

### Mode B: Embedding-Compressed (Jina hydration)

```
content → Jina API → 1024 x f32 → median threshold → bit expansion → 10K bits
```

- Each of 1024 dimensions maps to ~10 bits (strength-proportional)
- Semantically similar content → similar fingerprints (low Hamming distance)
- Requires context (what embedding model, what domain)
- Used for: semantic search, resonance, clustering

### The Hydration Pipeline

```
Phase 1: IMMEDIATE (on DN.SET)
  content → LFSR fingerprint (Mode A)
  Record written to BindSpace immediately
  User gets response in <1ms

Phase 2: DEFERRED (async hydration)
  JinaCache checks for cached embedding
  ├── Cache hit (90%+): use cached 1024D
  └── Cache miss: call Jina API, cache result
  1024D → binary expansion → overwrite fingerprint (Mode B)
  Orthogonal cleaning: project_out(existing codebook vectors)
  Record updated in-place in BindSpace
  Flush to Lance on next sync

Phase 3: CONTEXTUAL (on query)
  If node queried before hydration completes:
    Use Mode A fingerprint (content hash)
    Flag response as "pre-semantic"
  After hydration:
    Hamming distances become semantic distances
    Resonance search finds related concepts
```

### Why Not Random

The 10K-bit fingerprint should NOT be random because:
- Random fingerprints are ~5000 Hamming distance from everything (by birthday paradox)
- That means no resonance, no learning signal, no clustering
- Random is only useful for orthogonal basis vectors (codebook symbols)

The LFSR fingerprint (Mode A) is pseudo-random from a CONTENT seed.
Two different contents produce quasi-orthogonal fingerprints.
But the same content always produces the same fingerprint.

After Jina hydration (Mode B), the fingerprint becomes SEMANTIC:
similar content → similar fingerprints → Hamming distance encodes meaning.

The transition from Mode A to Mode B is the "hydration" —
the record goes from "I know WHAT this is" to "I know what this MEANS."

---

## 5. Edge Storage: Co-located, Bounded

### Why Co-located Works

With max 32 edges per node:
- 32 × 8 bytes = 256 bytes = 32 u64s = compartment 8
- Every cognitive step reads the full record anyway (for fingerprint)
- Edges arrive free — no second lookup, no join, no CSR
- Write: one record write updates both node data AND its edges

### Edge Format: 8 bytes each

```
┌────────┬────────┬────────┬────────┐
│ target │  verb  │weight│ amp  │ flags  │
│  u16   │  u16   │  u8  │  u8  │  u16   │
└────────┴────────┴────────┴────────┘
  8 bytes per edge × 32 max = 256 bytes
```

- `target`: Addr of destination node (0x0000-0xFFFF)
- `verb`: Addr of verb in surface layer 0x07 (CAUSES=0x0700, BECOMES=0x0701, ...)
- `weight`: quantized 0-255 (0=no connection, 255=certain)
- `amp`: amplification factor 0-255 (attention-like scaling)
- `flags`: bidirectional(1), temporal(2), inhibitory(4), learned(8), ...

### Semantic Constraint

> "If I put more than 32 edges in a node it's either process data or meaningless noise"

This is a hard cognitive boundary, not a technical limitation.
A concept that relates to 33+ things has lost coherence.
The correct response is to split (create child nodes in DN tree)
or prune (drop lowest-weight edges).

### Edge Lifecycle

```
1. DN.SET creates node → edge_count = 0
2. Resonance search finds similar nodes → auto-link top-K (K ≤ 32)
3. User adds explicit edge → DN.LINK source verb target
4. Learning adjusts weights → Hebbian: co-activated edges strengthen
5. Pruning removes noise → weight < threshold → edge removed
6. Split if saturated → promote to branch with children
```

---

## 6. CAM 4096: Operations ARE Addresses

The surface layer (0x00-0x0F) contains 4,096 pre-defined operations.
Each operation is itself a BindNode with a fingerprint derived from its name.

```
Surface 0x00: Lance ops    (VECTOR_SEARCH=0x0000, TRAVERSE=0x0001, ...)
Surface 0x01: SQL ops      (SELECT=0x0100, INSERT=0x0101, ...)
Surface 0x02: Cypher ops   (MATCH=0x0200, CREATE=0x0201, ...)
Surface 0x03: Hamming ops  (DISTANCE=0x0300, BIND=0x0310, CLEAN=0x0350, ...)
Surface 0x04: NARS ops     (DEDUCE=0x0400, ABDUCT=0x0401, ...)
Surface 0x05: Causal ops   (OBSERVE=0x0500, INTERVENE=0x0501, ...)
Surface 0x06: Meta ops     (REFLECT=0x0600, ...)
Surface 0x07: Verbs        (CAUSES=0x0700, BECOMES=0x0701, ...)
Surface 0x08: Concepts     (...)
Surface 0x09: Qualia       (...)
Surface 0x0A: Memory       (...)
Surface 0x0B: Learning     (...)
Surface 0x0C-0x0F: Reserved/User
```

### Operations as CogRecords

Every CAM operation is stored as a CogRecord in BindSpace:
- Its fingerprint is `Fingerprint::from_content("CAUSES")` (LFSR, Mode A)
- This means operations can be XOR-bound with data nodes
- `node.fingerprint ^ verb.fingerprint` = "this node, in the context of CAUSES"
- That bound fingerprint can be searched via Hamming

### Integration with Edges

An edge's `verb: u16` field is a surface address:
```
edge.verb = 0x0700  →  BindSpace[0x07][0x00]  →  "CAUSES" BindNode
```

To get the verb's fingerprint for XOR binding:
```rust
let verb_fp = bind_space.read(Addr(edge.verb)).unwrap().fingerprint;
let bound = xor(source.fingerprint, verb_fp);
// bound encodes "source IN-THE-CONTEXT-OF verb"
// Search for nodes near bound → finds targets of that relationship
```

---

## 7. Lance Column Mapping

### Primary Storage: One Column

```
nodes.lance schema:
  ├── addr:       UInt16       (BindSpace address, partition key)
  ├── record:     FixedSizeBinary(2048)  (the entire CogRecord)
  └── dn_path:    Utf8         (human-readable path, for SQL convenience)
```

That's it. The CogRecord IS the row. Lance stores it, versions it (ACID),
and serves it back as raw bytes. Zero copy: Arrow `FixedSizeBinary` buffer
→ pointer cast → `&[u64; 256]`.

### Derived Columns (materialized for indexing)

```
nodes_fingerprint.lance:
  ├── addr:        UInt16
  └── fingerprint: FixedSizeBinary(1256)  (compartment 0 only)
  → Lance ANN index (IVF-PQ or custom Hamming)

nodes_embedding.lance:
  ├── addr:        UInt16
  └── embedding:   FixedSizeBinary(128)   (compartment 7 only, 1024 bits)
  → Lance ANN index for vector similarity pre-filter

nodes_edges.lance:
  ├── source_addr: UInt16
  ├── target_addr: UInt16
  ├── verb_addr:   UInt16
  ├── weight:      UInt8
  └── amp:         UInt8
  → Derived from compartment 8, exploded into rows for DataFusion SQL
  → Rebuilt from CogRecord on flush (source of truth is the record)
```

### Write Path

```
User: DN.SET Ada:A:soul:identity "I think therefore I am" RUNG 5

CogRedis:
  1. path_to_addr("Ada:A:soul:identity") → Addr(0x83A2)
  2. Create parent chain: Ada → Ada:A → Ada:A:soul → Ada:A:soul:identity
  3. For each node in chain:
     a. Allocate CogRecord [u64; 256] = zeroed
     b. Set compartment 0: LFSR fingerprint from content
     c. Set compartment 1: addr, type_id=THOUGHT
     d. Set compartment 2: parent, depth, rung
     e. Set compartment 3: truth <1.0, 0.5> (new, low confidence)
     f. Set compartment 8: parent edge (CHILD_OF verb)
     g. Write to BindSpace[addr]

Async hydration:
  4. JinaCache.get_fingerprint("I think therefore I am")
     → Cache miss → Jina API → 1024 x f32
  5. Binary expand 1024D → overwrite compartment 0
  6. Compute embedding summary → write compartment 7
  7. Orthogonal clean: project_out(codebook)
  8. Resonance scan: find top-K Hamming neighbors
     → Auto-link edges in compartment 8

Lance flush:
  9. Serialize CogRecord → FixedSizeBinary(2048)
  10. Append to nodes.lance (ACID, versioned)
  11. Rebuild derived columns if needed
```

### Read Path

```
User: DN.GET Ada:A:soul:identity

CogRedis:
  1. path_to_addr("Ada:A:soul:identity") → Addr(0x83A2)
  2. BindSpace[0x83][0xA2] → &CogRecord (zero copy)
  3. Return fingerprint + edges + truth + metadata

If BindSpace cold (after restart):
  1. Lance scan: addr = 0x83A2
  2. FixedSizeBinary(2048) → &[u8; 2048]
  3. pointer cast → &[u64; 256]  (zero copy)
  4. Write into BindSpace[0x83][0xA2]
  5. Continue as above
```

---

## 8. Implementation Order

### Phase 1: CogRecord struct (Rust)

Define the `CogRecord` as `#[repr(C, align(64))] struct CogRecord([u64; 256])`
with accessor methods for each compartment. This replaces `BindNode`.

```rust
#[repr(C, align(64))]
pub struct CogRecord {
    data: [u64; 256],
}

impl CogRecord {
    // Compartment 0: fingerprint
    pub fn fingerprint(&self) -> &[u64; 157] { ... }
    pub fn set_fingerprint(&mut self, fp: &Fingerprint) { ... }

    // Compartment 2: DN tree
    pub fn parent(&self) -> Option<Addr> { ... }
    pub fn depth(&self) -> u8 { ... }

    // Compartment 8: edges
    pub fn edge_count(&self) -> u8 { ... }
    pub fn edge(&self, i: u8) -> Option<Edge> { ... }
    pub fn add_edge(&mut self, target: Addr, verb: Addr, weight: u8) -> bool { ... }
}
```

Files touched:
- NEW: `src/core/cog_record.rs`
- MODIFY: `src/storage/bind_space.rs` (replace `BindNode` with `CogRecord`)

### Phase 2: BindSpace migration

Replace `BindNode` with `CogRecord` in BindSpace storage arrays.
All existing accessors (fingerprint, parent, depth, etc.) become
forwarding calls to `CogRecord` methods.

Files touched:
- `src/storage/bind_space.rs`
- `src/storage/cog_redis.rs` (update DN.SET/GET to use CogRecord)
- `src/storage/xor_dag.rs` (update ACID to use CogRecord)

### Phase 3: Lance persistence

Store CogRecords as `FixedSizeBinary(2048)` in Lance.
Flush path: BindSpace → serialize → Lance append.
Rehydrate path: Lance scan → deserialize → BindSpace.

Files touched:
- `src/storage/lance.rs` (new schema: addr + record + dn_path)
- `src/storage/database.rs` (flush/rehydrate methods)

### Phase 4: Hydration pipeline

Wire Jina embedding → fingerprint conversion as async post-write step.
Orthogonal cleaning via existing `project_out()` in spo.rs.
Auto-edge creation via Hamming resonance scan.

Files touched:
- `src/storage/cog_redis.rs` (async hydration after DN.SET)
- `src/extensions/spo/jina_cache.rs` (integrate with CogRecord)
- `src/search/hdr_cascade.rs` (resonance scan for auto-edges)

### Phase 5: Derived columns + DataFusion

Create materialized Lance columns for fingerprint-only and edges-only.
Register as DataFusion tables for SQL query support.

Files touched:
- `src/storage/lance.rs` (derived column rebuild)
- `src/query/datafusion.rs` (register derived tables)

---

## 9. What We Do NOT Build

- **Iceberg**: Lance already provides ACID, versioning, time-travel
- **Separate edge table** (for hot path): edges live in CogRecord
- **CSR**: unnecessary with 32 max edges co-located in record
- **Random projection matrices**: Jina provides the embedding, we binarize
- **GPU inference**: CPU Hamming + SIMD is the computation model
- **LLM tokenizer**: content → Jina API (deferred) OR content → LFSR hash (immediate)

---

## 10. Invariants

1. One CogRecord = one cognitive unit. No partial reads for cognition.
2. Max 32 edges per node. Overflow means split or prune, never expand.
3. BindSpace is authoritative. Lance is durable. Flight is transport.
4. Fingerprint is LFSR until hydrated, then semantic. Never random.
5. All verbs are surface addresses. Edges reference verbs by Addr.
6. Zero copy from Lance → Arrow → BindSpace → Hamming. No deserialization.
7. O(1) addressing. No HashMap. No B-tree. No join. Array index only.
