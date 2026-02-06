# Integration Plan: CogRecord + Lance + CogRedis

> **Date**: 2026-02-06 (rev 2)
> **Status**: Design specification
> **Reference**: [holograph/docs/06_METADATA_REVIEW.md](https://github.com/AdaWorldAPI/holograph/blob/main/docs/06_METADATA_REVIEW.md)
> **Constraint**: Many nodes, max 32 edges per node, zero copy, zero GPU, zero LLM

---

## 1. The Thesis

Cognition IS the storage layout. A single `[u64; 256]` CogRecord is:
- The computation unit (Hamming, XOR bind, superposition)
- The storage unit (Lance `FixedSizeBinary(2048)`)
- The transport unit (Flight Arrow batch)
- The addressing unit (8+8 prefix:slot → array index)

No deserialization. No ETL. No load step. Read the record, compute on it, write it back.

**1024D embeddings do not exist at runtime.** They are transient calibration
artifacts cached in JinaCache during fingerprint generation. Once the binary
fingerprint is written, only Hamming distance operates. No floats. No ANN.
No cosine. No GPU.

---

## 2. Why 256 Words (from holograph)

256 × u64 = 16,384 bits = 2^14 bits. This is not arbitrary:

| Property | Value | Why it matters |
|----------|-------|----------------|
| σ = √(n/4) | **64.0 exactly** | Integer thresholds, no FP math in distance gates |
| AVX-512 iterations | **32 exactly** (full), **26** (semantic) | Zero remainder, no cleanup loop |
| BindSpace total | 65,536 × 2KB = **128 MiB** | Fits in L3 cache |
| Cache lines per record | 32 | Exact, no partial lines |

The surplus beyond 10K bits: 16,384 − 10,000 = **6,384 bits for metadata**.
But with holograph's layout, words 0-207 (13,312 bits) carry semantic content,
and words 208-255 (3,072 bits) carry structured metadata. The 3,312 extra
semantic bits (beyond 10K) give 33% more VSA binding/bundling capacity.

---

## 3. Addressing: Three Layers

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
  │ [u64; 256] = 2,048 bytes = 2^14 bits    │
  │                                          │
  │ Words 0-207:   SEMANTIC (13,312 bits)    │
  │ Words 208-255: METADATA (3,072 bits)     │
  └──────────────────────────────────────────┘
```

---

## 4. CogRecord Layout: 256 × u64

**Aligned with holograph `06_METADATA_REVIEW.md`.**

### Semantic Zone: Words 0-207 (Blocks 0-12)

```
Words 0-207 = 13,312 bits = 26 AVX-512 iterations (zero remainder)

  Content: Pure VSA binary fingerprint
  Operations: XOR bind, Hamming distance, majority bundle
  Generation: Jina 1024D → median threshold → bit expansion → 10K+ bits
              OR: content → LFSR hash → 10K bits (immediate, pre-calibration)

  Distance is computed ONLY on these words.
  Hamming(A[0..207], B[0..207]) = semantic similarity.
  Metadata words (208-255) are orthogonal to distance.
```

Jina 1024D embeddings are used ONLY during fingerprint generation.
They are cached transiently in `JinaCache` for deduplication/batching,
then discarded. The binary fingerprint IS the final representation.
No float vectors exist at query time.

### Metadata Zone: Words 208-255 (Blocks 13-15)

#### Block 13: ANI + NARS + Identity (Words 208-223)

| Word | Field | Bits | Type | Purpose |
|------|-------|------|------|---------|
| 208 | ANI level | 0-7 | u8 | Consciousness tier (0-255) |
| 208 | Layer mask | 8-15 | u8 | Active 7-layer bitmap |
| 208 | Peak activation | 16-31 | u16 | Max layer firing |
| 208 | L1-L4 confidence | 32-63 | 4×u8 | Per-layer confidence |
| 209 | L5-L7 confidence | 0-23 | 3×u8 | Upper layer confidence |
| 209 | Processing cycle | 24-39 | u16 | Current cycle count |
| 209 | Consciousness flags | 40-47 | u8 | dormant, active, etc. |
| 209 | Tau quantized | 48-55 | u8 | Temporal context window |
| 210 | NARS frequency | 0-15 | u16 | Truth freq (0-65535 → 0.0-1.0) |
| 210 | NARS confidence | 16-31 | u16 | Truth conf |
| 210 | Positive evidence | 32-47 | u16 | Evidence count |
| 210 | Negative evidence | 48-63 | u16 | Evidence count |
| 211 | Membrane sigma | 0-15 | u16 | Reasoning depth |
| 211 | Processing mode | 16-23 | u8 | Execution strategy |
| 211 | Reserved | 24-31 | u8 | — |
| 211 | Tau hash | 32-63 | u32 | Temporal context hash |
| 212 | Valence | 0-15 | u16 | Emotional tone |
| 212 | Arousal | 16-31 | u16 | Energy/activation |
| 212 | Dominance | 32-47 | u16 | Control/agency |
| 212 | Novelty | 48-63 | u16 | Surprise |
| 213 | Certainty | 0-15 | u16 | Epistemic confidence |
| 213 | Urgency | 16-31 | u16 | Temporal pressure |
| 213 | Depth | 32-47 | u16 | Conceptual complexity |
| 213 | Salience | 48-63 | u16 | Attention weight |
| 214 | Program counter | 0-15 | u16 | GEL execution position |
| 214 | Stack depth | 16-23 | u8 | Call stack height |
| 214 | Execution flags | 24-31 | u8 | Branch, trap, etc. |
| 214 | Current verb | 32-39 | u8 | Active operation opcode |
| 214 | GEL phase | 40-47 | u8 | Compile/route/execute |
| 214 | Reserved | 48-63 | u16 | — |
| 215 | Integration state | 0-15 | u16 | Kernel binding degree |
| 215 | Kernel mode | 16-23 | u8 | Assembly/inference/learn |
| 215 | Kernel epoch | 24-31 | u8 | Version counter |
| 215 | Reserved | 32-63 | u32 | — |
| 216 | Parent address | 0-15 | u16 | Addr of parent (0=root) |
| 216 | Depth | 16-23 | u8 | DN tree depth |
| 216 | Rung | 24-31 | u8 | Access tier R0-R9 |
| 216 | Sigma | 32-39 | u8 | Reasoning depth Σ |
| 216 | Node type | 40-47 | u8 | domain/tree/branch/twig/leaf |
| 216 | Flags | 48-63 | u16 | Immutable, pinned, overflow |
| 217 | Label hash | 0-31 | u32 | Hash of human-readable label |
| 217 | Access count | 32-47 | u16 | LRU/frequency counter |
| 217 | TTL remaining | 48-63 | u16 | Quantized expiry |
| 218 | Created timestamp | 0-31 | u32 | Unix epoch (136-year range) |
| 218 | Last access delta | 32-47 | u16 | Time since last read |
| 218 | Reserved | 48-63 | u16 | — |
| 219-222 | Edges 0-15 | 4×u64 | 4 edges per word | verb:u8 + target:u8 × 4 |
| 223 | Overflow count | 0-7 | u8 | Inline edges used |
| 223 | Overflow flag | 8-15 | u8 | >32 edges → Lance table |
| 223 | Overflow ptr | 16-31 | u16 | Lance edge table row |
| 223 | Schema version | 56-63 | u8 | v0=legacy, v1=current |

#### Block 14: RL + Temporal (Words 224-239)

| Word | Field | Bits | Type | Purpose |
|------|-------|------|------|---------|
| 224-225 | Q-values 0-7 | 128 | 8×u16 | Action expected rewards |
| 226 | Recent rewards | 64 | 4×u16 | Ring buffer: last 4 |
| 227 | Reward trend | 0-15 | u16 | Slope of recent rewards |
| 227 | Action count | 16-31 | u16 | Total actions executed |
| 227 | Epsilon | 32-47 | u16 | Exploration rate |
| 227 | Reserved | 48-63 | u16 | — |
| 228 | Policy fingerprint | 0-63 | u64 | Condensed policy hash |
| 229 | State-action cache | 0-63 | u64 | Memoized binding key |
| 230 | TD error | 0-31 | u32 | Temporal difference sum |
| 230 | Gamma | 32-47 | u16 | Discount factor |
| 230 | Alpha | 48-63 | u16 | Learning rate |
| 231 | RL routing score | 0-31 | u32 | Action selection cache |
| 231 | Reserved | 32-63 | u32 | — |

#### Edge Extension + Graph (Words 232-255)

| Words | Content |
|-------|---------|
| 232-235 | Edges 16-31: extended slots (4 words × 4 edges = 16 edges) |
| 236-239 | Reserved |
| 240 | Edge metadata: count(u8), overflow(u8), ptr(u16), version(u16) |
| 241 | Degree: in(u16), out(u16), bidirectional(u16) |
| 242-243 | CSR offsets (reserved for bulk graph ops) |
| 244-247 | Neighbor Bloom filter: 256 bits, ~1% FP at 20 neighbors |
| 248 | Degree metrics: total(u16), in(u16), out(u16) |
| 249 | Centrality: PageRank(u32), HITS auth(u16), hub(u16) |
| 250 | Clustering: cluster_id(u16), community(u16), betweenness(u16), closeness(u16) |
| 251 | Local: clustering_coeff(u16), triangle_count(u16) |
| 252 | Path: eccentricity(u16), katz(u16) |
| 253 | Temporal: recent_degree(u16), growth_rate(u16) |
| 254 | Reserved |
| 255 | Checksum(u32) + version(u8) |

### Edge Encoding

Holograph uses 16-bit edges: `verb:u8 + target:u8`.
4 edges per u64 word. 16 edges in words 219-222, 16 more in 232-235 = **32 total**.

**Open question**: target is u8 in holograph (addresses 256 nodes).
If full 65K addressing needed, target becomes u16 → 2 edges per word
→ 32 edges need 16 words instead of 8. This eats into RL block.
Resolution options:
- Use u8 relative addressing within DN subtree (most edges are local)
- Use u16 and reduce RL to 4 words (still enough for 4 Q-values + basics)
- Keep u8 + overflow to Lance for cross-subtree edges

---

## 5. The Hydration Pipeline

1024D embeddings exist ONLY in phase 2. They are never stored in the CogRecord.

```
Phase 1: IMMEDIATE (on DN.SET, <1ms)
  content → LFSR fingerprint → words[0..207]
  Record written to BindSpace immediately
  Pre-semantic: Hamming distances are content-hash based

Phase 2: CALIBRATION (async, deferred)
  JinaCache checks for cached 1024D
  ├── Hit (90%+): use cached float vector
  └── Miss: Jina API call, cache result
  1024D → median threshold → bit expansion → overwrite words[0..207]
  Orthogonal cleaning: project_out(codebook)
  Float vector discarded. Only binary fingerprint remains.
  Post-semantic: Hamming distances encode meaning

Phase 3: STEADY STATE
  All operations are Hamming on binary fingerprints.
  No floats. No 1024D. No ANN index. No cosine.
  Resonance = low Hamming distance.
  Noise = high Hamming distance (orthogonal).
```

---

## 6. XOR Delta Chains (from holograph)

The XOR property of the `[u64; 256]` layout enables sparse updates:

```rust
// Delta = base ⊕ modified (only changed words are non-zero)
fn record_delta(base: &[u64; 256], modified: &[u64; 256]) -> [u64; 256] {
    let mut delta = [0u64; 256];
    for i in 0..256 { delta[i] = base[i] ^ modified[i]; }
    delta
}

// Recovery: current = base ⊕ delta
fn apply_delta(base: &[u64; 256], delta: &[u64; 256]) -> [u64; 256] {
    let mut result = [0u64; 256];
    for i in 0..256 { result[i] = base[i] ^ delta[i]; }
    result
}
```

**Metadata updates are sparse deltas.** Updating NARS truth (word 210) produces
a delta where only word 210 is non-zero. The other 255 words are 0.

**DN tree compression.** Parent-child nodes share semantic content:
```
Node A (Ada):                full 256-word record (base)
Node B (Ada:A):              A ⊕ delta_B (depth change, context shift)
Node C (Ada:A:soul):         B ⊕ delta_C (further specialization)
```

Each delta is sparse. Storage cost ∝ difference, not record size.

---

## 7. Lance Column Mapping

### Primary Storage

```
nodes.lance:
  ├── addr:     UInt16                  (BindSpace address, partition key)
  ├── record:   FixedSizeBinary(2048)   (the entire CogRecord)
  └── dn_path:  Utf8                    (human-readable, for SQL convenience)
```

No embedding column. No float vectors. No ANN index.
Lance stores binary records. Hamming search runs on the `record` column.

### Write Path

```
DN.SET Ada:A:soul:identity "I think therefore I am" RUNG 5
  ↓
CogRedis:
  1. path_to_addr() → Addr(0x83A2)
  2. Allocate CogRecord [u64; 256] = zeroed
  3. Write words[0..207]: LFSR fingerprint from content
  4. Write word 216: parent|depth|rung|sigma|node_type
  5. Write word 210: NARS truth <1.0, 0.5> (new belief, low confidence)
  6. Write words 219+: parent edge (CHILD_OF verb)
  7. Store in BindSpace[addr]
  ↓
Async calibration:
  8. JinaCache → 1024D → binary expansion → overwrite words[0..207]
  9. project_out(codebook) → orthogonal cleaning
  10. Resonance scan → auto-link top-K edges in words 219-235
  11. Discard 1024D float vector
  ↓
Flush:
  12. CogRecord → FixedSizeBinary(2048) → nodes.lance (ACID append)
```

### Read Path

```
DN.GET Ada:A:soul:identity
  → Addr(0x83A2) → BindSpace[0x83][0xA2] → &CogRecord (zero copy)

After restart (cold):
  → Lance scan: addr = 0x83A2
  → FixedSizeBinary(2048) → pointer cast → &[u64; 256] (zero copy)
  → Write into BindSpace → continue
```

---

## 8. CAM 4096: Operations ARE Addresses

The surface layer (0x00-0x0F) holds 4,096 pre-defined operations.
Each is a CogRecord with a fingerprint derived from its name.

```
Surface 0x00: Lance ops     Surface 0x08: Concepts
Surface 0x01: SQL ops       Surface 0x09: Qualia
Surface 0x02: Cypher ops    Surface 0x0A: Memory
Surface 0x03: Hamming ops   Surface 0x0B: Learning
Surface 0x04: NARS ops      Surface 0x0C-0x0F: Reserved
Surface 0x05: Causal ops
Surface 0x06: Meta ops
Surface 0x07: Verbs (CAUSES=0x0700, BECOMES=0x0701, ...)
```

An edge's `verb` field is a surface address:
```
edge.verb = 0x07  →  BindSpace[0x07][verb_slot]  →  verb's fingerprint
XOR(source.fingerprint, verb.fingerprint) = "source IN-CONTEXT-OF verb"
```

---

## 9. Implementation Order

### Phase 1: CogRecord struct

`#[repr(C, align(64))] struct CogRecord([u64; 256])` with accessor methods
per compartment. Port from holograph `width_16k/schema.rs` SchemaSidecar.

Files: NEW `src/core/cog_record.rs`

### Phase 2: BindSpace migration

Replace `BindNode` (struct with named fields) with `CogRecord` (256 u64 array).
All existing accessors become word/bit extractions.

Files: `bind_space.rs`, `cog_redis.rs`, `xor_dag.rs`

### Phase 3: Lance persistence

Store CogRecords as `FixedSizeBinary(2048)`. Drop `NodeRecord`/`EdgeRecord`
structs. Drop `vector_search()`. Only `hamming_search()` and predicate scan.

Files: `lance.rs`, `database.rs`

### Phase 4: Hydration pipeline

Wire JinaCache → fingerprint generation → orthogonal cleaning → auto-edges.
1024D is transient, discarded after binary conversion.

Files: `cog_redis.rs`, `jina_cache.rs`, `hdr_cascade.rs`

### Phase 5: DataFusion integration

Register Lance `nodes.lance` as DataFusion table. Predicate pushdown
reads metadata words (208-255) without touching semantic words (0-207).

Files: `datafusion.rs`

---

## 10. What We Do NOT Build

- **Embedding columns or ANN indexes** — no floats at query time
- **Iceberg** — Lance has ACID, versioning, time-travel
- **Separate edge table** (hot path) — edges co-located in record
- **CSR** (for hot path) — 32 edges inline, no indirection needed
- **GPU inference** — CPU Hamming + SIMD is the computation model
- **Cosine / L2 distance** — Hamming only, integer arithmetic only

---

## 11. Invariants

1. **One CogRecord = one cognitive unit.** No partial reads for cognition.
2. **Max 32 edges per node.** Overflow means split or prune, never expand.
3. **BindSpace is authoritative. Lance is durable. Flight is transport.**
4. **Fingerprint is LFSR until calibrated, then semantic. Never random.**
5. **All verbs are surface addresses.** Edges reference verbs by Addr.
6. **Zero copy from Lance → Arrow → BindSpace → Hamming.** No deserialization.
7. **O(1) addressing.** No HashMap. No B-tree. No join. Array index only.
8. **No floats at query time.** 1024D exists only during calibration, then discarded.
9. **σ = 64 exactly.** Integer thresholds. No FP in distance gates.
10. **32 AVX-512 iterations.** Zero remainder. No cleanup loops.
