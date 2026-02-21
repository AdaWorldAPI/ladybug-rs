# CogRecord 65536: The 8 KB Cognitive Record

**Container width doubles. Record quadruples. Cost stays the same.**
*VPOPCNTDQ × VNNI: Two distance metrics, one memory layout.*

ladybug-rs Architecture Decision Record
February 21, 2026 — Jan Hübener / Claude Architecture Audit

---

## 0. The Insight

Current: `Container = [u64; 128]` = 8,192 bits = 1 KB.
Current: `CogRecord = [Container; 2]` = 16,384 bits = 2 KB.
Current: `Fingerprint = [u64; 256]` = 16,384 bits = 2 KB.

Container is **half-width Fingerprint**. Every `From<&Fingerprint> for Container` truncates. Every `From<&Container> for Fingerprint` zero-extends. This is wasted information.

**New: Container = Fingerprint = `[u64; 256]` = 16,384 bits = 2 KB.**

One type. Full width. No truncation. No zero-extension. The conversion functions become identity operations.

Then: `CogRecord = [Container; 4]` = 65,536 bits = 8 KB.

With AVX-512 `VPOPCNTDQ` (confirmed present on this hardware), a full 65,536-bit sweep costs 128 instructions — the same throughput class as the current 8,192-bit sweep's 16 instructions. The pipeline fills identically; only the iteration count changes.

---

## 1. Hardware Confirmation

```
CPU flags (confirmed via /proc/cpuinfo):
  avx512_vpopcntdq  ✓  — 512-bit popcount (8 × u64 per instruction)
  avx512_vnni        ✓  — int8 dot product (VPDPBUSD, 64 MACs per instruction)
  avx512f/bw/dq/vl   ✓  — foundation + byte/word/doubleword + vector length
  avx512_vbmi2       ✓  — byte/word permute
  avx512_bitalg      ✓  — bit manipulation
  popcnt             ✓  — scalar fallback
```

### Cost Model

| Operation | Current (8192 bits) | New (16384 bits per container) | New (65536 bits full record) |
|-----------|--------------------|-----------------------------|----------------------------|
| `VPOPCNTDQ` iterations | 16 | 32 | 128 |
| Throughput (Ice Lake+) | 1 cycle/instruction | 1 cycle/instruction | 1 cycle/instruction |
| Wall time @ 3 GHz | ~5 ns | ~11 ns | ~43 ns |
| L1 fit (32 KB typical) | 1 KB ✓ | 2 KB ✓ | 8 KB ✓ (4 records) |
| `VPDPBUSD` for 1024D int8 | n/a | 16 instructions | n/a (container 3 only) |

**Key fact**: 43 ns per full-record Hamming search. For comparison, a single Redis `GET` is ~50,000 ns over loopback. The 65536-bit record is still 1000× faster than the cheapest network operation.

---

## 2. The 4-Container Layout

```
CogRecord (65,536 bits = 8 KB):

┌─────────────────────────────────────────────────────────────────────┐
│  Container 0  (2 KB)  META: codebook identity + DN + NARS + edges  │
├─────────────────────────────────────────────────────────────────────┤
│  Container 1  (2 KB)  CAM: content-addressable fingerprint         │
├─────────────────────────────────────────────────────────────────────┤
│  Container 2  (2 KB)  STRUCTURE: B-tree index / positional hash    │
├─────────────────────────────────────────────────────────────────────┤
│  Container 3  (2 KB)  EMBEDDING: quantized dense vector            │
└─────────────────────────────────────────────────────────────────────┘
```

### Container 0 — META (codebook identity)

256 words × 8 bytes = 2048 bytes. Double the current metadata budget.

**What the extra space solves:**

| Problem | Current (128 words) | New (256 words) |
|---------|--------------------|-----------------| 
| Layer markers: 3 bytes × 10 layers = 30 | W12-W15 (32 bytes, tight) | W12-W19 (64 bytes, room for 20+ layers) |
| Inline edges: 4 per word × 16 words = 64 | W16-W31 | W16-W63 (192 words = 768 edges) |
| Qualia: 18 channels × f16 | W56-W63 (8 words) | W80-W95 (16 words, 36 channels) |
| Brier calibration | nowhere (D5 from audit) | W96-W99 (4 words: Brier f32, count u32, last_update, style_idx) |
| Granger temporal | nowhere (Tactic #12 gap) | W100-W107 (8 words: causal effect sizes for 4 predecessors) |
| TD-learning Q-values | W32-W39 (8 words, 4 styles) | W108-W123 (16 words: all 12 ThinkingStyles × f32 Q-value + count) |
| Reserved/future | W112-W125 (14 words) | W124-W253 (130 words — massive expansion room) |
| Checksum + version | W126-W127 | W254-W255 |

**The layer marker overflow (D5) is now solved structurally.** 64 bytes for layer markers means 3 bytes × 21 layers with room to spare. No encoding change needed.

**The codebook identity thesis**: Container 0 IS the node. Not "describes" the node — IS it. The DN path, NARS truth, edges, qualia, RL state, calibration history — all packed into one 16,384-bit block. The codebook entry for this identity is derived from its Hamming fingerprint, not stored separately. Phase-shifting between identities = XOR between two Container 0's.

### Container 1 — CAM (content-addressable memory)

Pure 16,384-bit CAM fingerprint. This is the searchable content. `belichtungsmesser()` operates here. Hamming-based nearest neighbor operates here. The XOR algebra (BIND/UNBIND/BUNDLE) operates here.

**Identical to current Fingerprint.** No conversion needed. `Fingerprint` type can alias `Container` directly.

Statistical properties at 16,384 bits:
- Expected Hamming distance (random pair): 8,192
- σ = √(16384/4) ≈ 64.0
- Berry-Esseen noise floor: < 0.003 (improved from < 0.004 at 8192 bits)
- Codebook separation: 144 verbs stay well-separated (expected pairwise distance = 8192 ± 64)

### Container 2 — STRUCTURE (B-tree / positional hash)

The DN tree position, structural relationships, and graph topology encoded as a searchable binary vector. This enables:

- **Structural similarity**: two nodes at similar tree positions have similar Container 2's
- **B-tree navigation**: split decisions encoded as bit prefixes
- **Hashtag zone**: every label, property key, relationship type that applies to this node is XOR-bundled into Container 2. "Does node X have label Y?" = one Hamming query against Container 2.

This is the "hashtag everything" container. Edge = node. NARS = edge. Verb = rotation address. All content-addressable via Container 2.

### Container 3 — EMBEDDING (dual-metric dense vector)

16,384 bits = 2,048 bytes. Two operating modes:

**Mode A — Binary (Hamming via VPOPCNTDQ)**:
16,384-bit binary hash. Pure Hamming search. Same hardware path as Containers 0-2. Used for: binary LSH projections, SimHash of text, binary CLIP embeddings.

**Mode B — Quantized int8 (dot product via VNNI)**:
2,048 bytes at int8 = 2,048 dimensions. Or 1,024 dimensions at int8 with 1,024 bytes metadata (norms, offsets, quantization params).

The VNNI path (`VPDPBUSD`):
```
// 1024D int8 dot product in 16 instructions:
// Each VPDPBUSD: 64 int8 multiply-accumulates per 512-bit register
// 1024 / 64 = 16 iterations
// Wall time: ~16 cycles ≈ 5 ns at 3 GHz
```

**Both metrics use the same memory.** A flag in Container 0 metadata (W1 byte 3) indicates the distance metric for Container 3. The query engine dispatches:

```rust
match meta.embedding_metric() {
    EmbeddingMetric::Hamming => container3.hamming(query3),      // VPOPCNTDQ
    EmbeddingMetric::DotInt8 => container3.dot_int8(query3),     // VPDPBUSD
    EmbeddingMetric::CosineInt8 => container3.cosine_int8(query3), // VPDPBUSD + norm
}
```

### Embedding Format Table

| Format | Dimensions | Bits Used | Container Fill | Distance Instruction |
|--------|-----------|-----------|----------------|---------------------|
| Binary hash (1-bit) | 16,384 | 16,384 | 100% | `VPOPCNTDQ` (128 iter) |
| int4 quantized | 4,096 | 16,384 | 100% | Custom (shift+mask+accumulate) |
| int4 quantized | 1,024 | 4,096 | 25% | Custom (+ room for metadata) |
| int8 quantized | 2,048 | 16,384 | 100% | `VPDPBUSD` (32 iter) |
| int8 quantized | 1,024 | 8,192 | 50% | `VPDPBUSD` (16 iter) |
| f16 (half float) | 1,024 | 16,384 | 100% | `VCVTPH2PS` + `VFMADD` |
| Jina v3 (1024D, int8) | 1,024 | 8,192 | 50% | `VPDPBUSD` (16 iter) |
| CLIP ViT-L (768D, int8) | 768 | 6,144 | 37.5% | `VPDPBUSD` (12 iter) |
| Cohere v3 (1024D, int8) | 1,024 | 8,192 | 50% | `VPDPBUSD` (16 iter) |

---

## 3. Migration Path: Constants-First

The entire change starts at one file: `crates/ladybug-contract/src/container.rs`.

### Phase 0: Change Constants (1 hour)

```rust
// BEFORE:
pub const CONTAINER_BITS: usize = 8_192;
pub const CONTAINER_WORDS: usize = CONTAINER_BITS / 64;  // 128
pub const CONTAINER_BYTES: usize = CONTAINER_WORDS * 8;  // 1024
pub const CONTAINER_AVX512_ITERS: usize = CONTAINER_WORDS / 8;  // 16

// AFTER:
pub const CONTAINER_BITS: usize = 16_384;
pub const CONTAINER_WORDS: usize = CONTAINER_BITS / 64;  // 256
pub const CONTAINER_BYTES: usize = CONTAINER_WORDS * 8;  // 2048
pub const CONTAINER_AVX512_ITERS: usize = CONTAINER_WORDS / 8;  // 32
```

Because the code uses `CONTAINER_WORDS` everywhere (not hardcoded `128`), **most code compiles immediately**. The 17 files importing these constants just get wider containers.

### Phase 0.1: Statistical Constants

```rust
// BEFORE:
pub const EXPECTED_DISTANCE: u32 = 4096;  // 8192/2
pub const SIGMA: f64 = 45.254833995939045;  // √(8192/4)
pub const SIGMA_APPROX: u32 = 45;

// AFTER:
pub const EXPECTED_DISTANCE: u32 = 8192;  // 16384/2
pub const SIGMA: f64 = 64.0;  // √(16384/4) = √4096 = 64.0 exactly
pub const SIGMA_APPROX: u32 = 64;
```

Clean numbers. σ = 64 exactly. This simplifies all threshold math.

### Phase 0.2: CogRecord → 4 Containers

```rust
// BEFORE (ladybug-contract/src/record.rs):
pub struct CogRecord {
    pub meta: Container,     // 1 KB
    pub content: Container,  // 1 KB
}  // = 2 KB

// AFTER:
pub struct CogRecord {
    pub meta: Container,       // 2 KB — identity, NARS, edges, qualia, RL
    pub cam: Container,        // 2 KB — content-addressable fingerprint
    pub structure: Container,  // 2 KB — B-tree index, hashtag zone
    pub embedding: Container,  // 2 KB — quantized dense vector
}  // = 8 KB

impl CogRecord {
    pub const SIZE: usize = 4 * CONTAINER_BYTES;  // 8192 bytes
    pub const CONTAINERS: usize = 4;
}
```

### Phase 0.3: Rename `content` → `cam` Globally

```bash
# ~30 occurrences across 15 files
sed -i 's/\.content/\.cam/g' src/**/*.rs
# Then fix false positives (doc comments, string literals)
```

### Phase 0.4: Fingerprint = Container

```rust
// BEFORE: Fingerprint is [u64; 256], Container is [u64; 128] — different types
// Convert with truncation/zero-extension

// AFTER: Same width. Type alias or newtype with zero-cost conversion.
pub type Fingerprint = Container;  // simplest
// OR: keep Fingerprint as separate type with Deref<Target=Container>
```

Decision: **Keep both types.** `Fingerprint` emphasizes "this is searchable content." `Container` emphasizes "this is a SIMD-aligned memory block." They're the same bits, but different semantic roles. The `From` impls become identity operations:

```rust
impl From<&Fingerprint> for Container {
    fn from(fp: &Fingerprint) -> Self {
        // Was: copy first 128 words, discard rest
        // Now: copy all 256 words — no truncation
        let mut c = Container::zero();
        c.words.copy_from_slice(fp.as_raw());
        c
    }
}
```

---

## 4. Hardcoded Values That Need Manual Update

The constants-first approach handles most code, but these 10 locations use hardcoded values:

| File | Line | What | Change |
|------|------|------|--------|
| `container/search.rs` | SAMPLE_POINTS | `[0,19,41,59,79,101,127]` | `[0,37,83,119,157,203,255]` — same prime-spaced pattern, doubled |
| `container/meta.rs` | Word offsets | All `W_*` constants | Expand layout to 256 words (see §2 Container 0) |
| `container/meta.rs` | `as_bytes()` | `[u8; CONTAINER_BYTES]` | Automatic (uses const) ✓ |
| `container/record.rs` | `SIZE` | `2 * CONTAINER_BYTES` | `4 * CONTAINER_BYTES` |
| `container/record.rs` | `from_bytes()` | Slices at `1024` | Slices at `CONTAINER_BYTES` |
| `container/migrate.rs` | `[u64; 256]` | Old 16K migration | Remove — Container IS 256 words now |
| `core/fingerprint.rs` | `FINGERPRINT_U64 = 256` | Same as CONTAINER_WORDS now | Alias or verify equal |
| `cam_graph.rs` (project) | `WORD_BITS: 8192, WORD_LANES: 128` | `WORD_BITS: 16384, WORD_LANES: 256` |
| `query/fingerprint_table.rs` | `FIXED_SIZE_BINARY(2048)` | `FIXED_SIZE_BINARY(8192)` (full CogRecord) |
| `storage/lance_zero_copy` | `MORSEL_SIZE: 2048` | `MORSEL_SIZE: 8192` |

---

## 5. MetaView Layout v2 (256 Words)

```
┌─────────────────────────────────────────────────────────────────────┐
│  W0         PackedDn address (THE identity)                         │
│  W1         Type: node_kind(u8) | geometry(u8) | flags(u8)         │
│             | embedding_metric(u8) | schema_version(u16)           │
│             | provenance_hash(u16)                                  │
│  W2         Timestamps (created_ms:32 | modified_ms:32)            │
│  W3         Label hash(u32) | tree_depth(u8) | branch(u8) | rsvd   │
├─────────────────────────────────────────────────────────────────────┤
│  W4-W7      NARS (freq:f32 | conf:f32 | pos_ev:f32 | neg_ev:f32)  │
│  W8-W11     DN rung + 10-layer compact + collapse gate state       │
│  W12-W19    10-Layer markers (3 bytes × 10 = 30 bytes, 34 spare)   │
│             *** OVERFLOW SOLVED: 64 bytes for 30 bytes of markers  │
│  W20-W23    SPOQ phase state (S-confidence, P-verb-idx, O-target,  │
│             Q-qualia-hash) — 4 words for 4 phases                  │
├─────────────────────────────────────────────────────────────────────┤
│  W24-W63    Inline edges (160 packed, 4 per word × 40 words)       │
│             *** 2.5× more edges than current 64                     │
├─────────────────────────────────────────────────────────────────────┤
│  W64-W79    Qualia (36 channels × f16 + 8 spare slots)             │
│             *** Double from 18 to 36 channels                       │
│  W80-W95    Bloom filter (1024 bits — double current 512)           │
│  W96-W99    Brier calibration (score:f32, count:u32,               │
│             last_update:u32, best_style:u8 + pad)                  │
│  W100-W107  Granger temporal (4 × CausalEffect: effect:f32,        │
│             lag:u16, confidence:f16)                                │
│  W108-W123  TD-learning (12 ThinkingStyles × (Q:f32 + visits:u32)) │
├─────────────────────────────────────────────────────────────────────┤
│  W124-W143  Graph metrics (full precision, expanded)               │
│  W144-W159  RL / reward history (16 words, 8 episodes)             │
│  W160-W191  Rung history + collapse gate history (expanded)        │
│  W192-W223  Representation language descriptor (expanded)          │
│  W224-W239  DN-Sparse adjacency (compact inline CSR)               │
│  W240-W251  Reserved for future expansion                          │
│  W252-W253  Container 3 metadata (embedding format, norm, offset)  │
│  W254-W255  Checksum + version                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. The Dual-Metric Container 3

### 6.1 VPOPCNTDQ Path (Hamming on binary fingerprints)

Identical to Containers 0-2. Hardware doesn't care about semantics:

```rust
// 32 VPOPCNTDQ instructions per 16384-bit container
fn hamming_avx512(a: &Container, b: &Container) -> u32 {
    // Compiles to: vpxorq + vpopcntq, 32 iterations
    let mut dist = 0u32;
    for i in 0..CONTAINER_WORDS {
        dist += (a.words[i] ^ b.words[i]).count_ones();
    }
    dist
}
```

### 6.2 VNNI Path (int8 dot product on quantized embeddings)

```rust
// 16 VPDPBUSD instructions for 1024D int8
fn dot_int8_avx512(a: &Container, b: &Container) -> i32 {
    // Interpret [u64; 256] as [i8; 2048]
    let a_bytes = a.as_bytes();  // &[u8; 2048]
    let b_bytes = b.as_bytes();
    
    // VPDPBUSD: unsigned × signed → i32 accumulate
    // 64 bytes per register × 32 iterations = 2048 bytes
    // But for 1024D: first 1024 bytes only → 16 iterations
    let dims = meta.embedding_dims();  // from W252
    let mut acc = 0i32;
    for i in 0..dims {
        acc += (a_bytes[i] as i8 as i32) * (b_bytes[i] as i8 as i32);
    }
    acc
    // With VNNI intrinsics, this compiles to VPDPBUSD
}
```

### 6.3 Distance Dispatch

```rust
pub enum EmbeddingMetric {
    Hamming,      // VPOPCNTDQ — binary fingerprint
    DotInt8,      // VPDPBUSD — int8 dot product
    CosineInt8,   // VPDPBUSD + norm from W252
    L2Int8,       // VPDPBUSD + expansion for L2
}

impl CogRecord {
    pub fn embedding_distance(&self, other: &CogRecord) -> f32 {
        match self.meta_view().embedding_metric() {
            EmbeddingMetric::Hamming => {
                self.embedding.hamming(&other.embedding) as f32
            }
            EmbeddingMetric::DotInt8 => {
                let dot = self.embedding.dot_int8(&other.embedding);
                -(dot as f32)  // negate for distance (higher dot = closer)
            }
            EmbeddingMetric::CosineInt8 => {
                let dot = self.embedding.dot_int8(&other.embedding) as f32;
                let norm_a = self.meta_view().embedding_norm();  // from W252
                let norm_b = other.meta_view().embedding_norm();
                1.0 - dot / (norm_a * norm_b)
            }
            EmbeddingMetric::L2Int8 => {
                // ||a-b||² = ||a||² + ||b||² - 2·dot(a,b)
                let dot = self.embedding.dot_int8(&other.embedding) as f32;
                let norm_a_sq = self.meta_view().embedding_norm_sq();
                let norm_b_sq = other.meta_view().embedding_norm_sq();
                (norm_a_sq + norm_b_sq - 2.0 * dot).sqrt()
            }
        }
    }
}
```

---

## 7. Neo4j-rs: Cypher on 4 Containers

Cypher compiles to container-specific operations:

| Cypher | Container | Operation | Instructions |
|--------|-----------|-----------|-------------|
| `MATCH (a {name: 'Ada'})` | C0 meta | DN lookup | 1 |
| `MATCH (a)-[:CAUSES]->(b)` | C2 structure | Belichtungsmesser + Hamming | ~32 SIMD |
| `WHERE a.confidence > 0.8` | C0 meta | NARS read (W4-W7) | 1 float compare |
| `WHERE similarity(a, b) > 0.9` | C1 cam | Full Hamming | 32 VPOPCNTDQ |
| `WHERE embedding_sim(a, b) > 0.8` | C3 embedding | VNNI dot product | 16 VPDPBUSD |
| `RETURN a` | All | `&CogRecord` zero-copy | 0 (borrow) |
| `CREATE (n:Person)` | C0+C2 | Meta init + label hashtag | 2 writes |
| `SET n.embedding = $vec` | C3 | Write quantized vector | 1 bulk write |

**New capability**: `embedding_sim()` function in Cypher that dispatches to Container 3 VNNI. This means a single Cypher query can combine structural graph matching (Container 2), content similarity (Container 1), AND embedding similarity (Container 3) in one traversal — no external vector database needed.

---

## 8. Cross-Reference: What This Solves

### From SPOQ Audit — Discrepancies Resolved

| Discrepancy | Resolution |
|-------------|-----------|
| **D5**: Layer marker overflow (50 bytes in 32) | 64 bytes at W12-W19. Solved permanently. |
| **D1**: Xyz "24,576 bits" misleading | Each linked CogRecord is now 65,536 bits. Xyz = 3 linked × 65,536 = 196,608 bits. |

### From SPOQ Audit — Expansion Opportunities Enabled

| Opportunity | How 8 KB Records Enable It |
|------------|---------------------------|
| **§4.1 Granger Causality** | W100-W107: 4 causal effect slots in metadata. No external storage. |
| **§4.3 Brier Calibration** | W96-W99: Brier score + count + style tracking. CollapseGate reads it. |
| **§4.6 TD-Learning Feedback** | W108-W123: All 12 ThinkingStyles × (Q-value + visit count). Full RL state in-record. |
| **§4.4 Cross-Domain Fusion** | Container 2 (structure) handles hashtag-based fusion. Container 3 holds embedding fusion via VNNI. |

### From 34 Tactics × Reasoning Ladder

| Tactic | New Container | Improvement |
|--------|--------------|------------|
| **#12 Temporal Context** (Granger) | C0 W100-W107 | Causal effect sizes stored per-node |
| **#10 MetaCognition** (Brier) | C0 W96-W99 | Calibration tracking in metadata |
| **#14 Multimodal CoT** | C3 embedding | Dense vector + CAM in same record |
| **#25 Hyperdimensional Matching** | C1 cam (16384 bits) | σ=64 exactly, better separation |
| **#15 Latent Space Introspection** | C3 embedding | Direct access to quantized embedding |
| **#23 Adaptive Meta-Prompting** | C0 W108-W123 | Full TD-learning state inline |
| **#4 Reverse Causality** | C2 structure | Verb rotation on structural container |

---

## 9. What Stays the Same

The following are width-independent and compile unchanged:

- **All RISC operations**: BIND (XOR), BUNDLE (majority), MATCH (Hamming), PERMUTE (rotate), STORE — they loop over `CONTAINER_WORDS`, not `128`.
- **SPO Crystal**: Role seeds are expanded to Container width via `Container::random(seed)`. Same algebra. Wider vectors.
- **NARS truth values**: Still in meta words 4-7. Position unchanged.
- **CollapseGate**: Still reads NARS + layer satisfaction. Format unchanged.
- **Blackboard pattern**: Still `&self` grey matter / `&mut self` white matter. Width doesn't affect borrow semantics.
- **Cognitive kernel**: 10-layer stack. Same layer IDs, same flow. Just reads wider containers.
- **Codebook**: 4096 entries × wider Container. Same seed expansion, same lookup logic.

---

## 10. On-Disk Format (LanceDB)

```
BEFORE:
  Column: fingerprint  FixedSizeBinary(2048)   -- 1 CogRecord = 2 KB

AFTER:
  Column: record       FixedSizeBinary(8192)   -- 1 CogRecord = 8 KB
  -- OR decomposed:
  Column: meta         FixedSizeBinary(2048)   -- Container 0
  Column: cam          FixedSizeBinary(2048)   -- Container 1
  Column: structure    FixedSizeBinary(2048)   -- Container 2
  Column: embedding    FixedSizeBinary(2048)   -- Container 3
```

The decomposed form enables column-specific queries: scan only Container 1 for content search, only Container 3 for embedding search. Arrow columnar storage handles this naturally — each column is a separate page, and `VPOPCNTDQ` operates on one column at a time.

4× storage increase (2 KB → 8 KB per record). For 1M records: 2 GB → 8 GB. Fits in RAM. For 10M records: 80 GB — still feasible on server hardware. The embedding container can be lazy-loaded (zeroed by default, populated on demand).

---

## 11. Redis Format

```
BEFORE:
  Key: dn:path:to:node
  Value: 2048 bytes (1 CogRecord)

AFTER:
  Key: dn:path:to:node
  Value: 8192 bytes (1 CogRecord)
  -- OR hash fields:
  Key: dn:path:to:node
  Field: meta       → 2048 bytes
  Field: cam        → 2048 bytes
  Field: structure  → 2048 bytes
  Field: embedding  → 2048 bytes
```

Hash fields enable partial reads (e.g., read only `cam` for content search). Upstash Redis has 1 MB per value limit — 8 KB per record is well within bounds.

---

## 12. The Formula (Updated)

```
CogRecord = Container[4] = 4 × 16,384 bits = 65,536 bits = 8 KB

Container 0: WHO I AM    (identity, truth, edges, calibration, RL)
Container 1: WHAT I MEAN (content-addressable fingerprint)
Container 2: WHERE I SIT (structural position, hashtag topology)
Container 3: HOW I LOOK  (dense embedding for external similarity)

Distance metrics:
  C0-C2: Hamming via VPOPCNTDQ  (binary, 32 instructions per container)
  C3:    Hamming OR dot-product  (binary via VPOPCNTDQ, int8 via VPDPBUSD)

One node. Four perspectives. Two hardware accelerators. Zero serialization.

cargo build --release --features full → one binary

The only graph database where nodes are 65,536-bit SIMD vectors
with dual-metric hardware acceleration in the same memory layout.
```

---

## 13. Implementation Priority

| Phase | Action | Est. | Blocks |
|-------|--------|------|--------|
| 0 | Change `CONTAINER_BITS` to 16384 in contract crate | 1 hr | Everything |
| 0.1 | Update SIGMA, EXPECTED_DISTANCE, sample points | 30 min | Search |
| 0.2 | Add `structure` + `embedding` fields to CogRecord | 1 hr | Record ops |
| 0.3 | Rename `content` → `cam` | 1 hr | Cosmetic |
| 1 | Expand MetaView layout to 256 words | 1 day | Gap D5, Brier, Granger, TD |
| 2 | Add `EmbeddingMetric` enum + `dot_int8()` method | 1 day | Container 3 |
| 3 | Update LanceDB schema (column-per-container) | 2 days | Storage |
| 4 | Update Redis format (hash fields) | 1 day | Storage |
| 5 | `belichtungsmesser_16k()` — 7 points in 256-word space | 30 min | Search |
| 6 | neo4j-rs: add `embedding_sim()` Cypher function | 2 days | Query |
| 7 | Benchmark: Hamming throughput at 16K, VNNI throughput | 1 day | Validation |

**Phase 0 is the critical path.** One constant change, then `cargo check` reveals all downstream impacts. The compiler does the audit.

---

*"The container doubled. The record quadrupled. The cost stayed the same. Because the hardware was waiting."*

*VPOPCNTDQ doesn't care about your data model. It counts bits. Give it more bits.*
