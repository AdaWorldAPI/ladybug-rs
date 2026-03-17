# Metadata Schema Reference

> **Generated**: 2026-03-17
> **Source of truth**: `src/container/meta.rs`, `src/width_16k/schema.rs`
> **Schema version**: 1 (`meta.rs:93`)

---

## Overview

Ladybug-rs uses a two-tier metadata system:

1. **MetaView** (128 words = 1,024 bytes) — full-resolution canonical metadata in Container 0
2. **SchemaSidecar** (32 words = 256 bytes) — compact summary in words 224-255

Container 0 is **never searched** by Hamming distance. It holds structural information only.
Content containers (1+) hold fingerprints for SIMD operations.

---

## MetaView — Container 0 Layout (W0-W127)

Source: `src/container/meta.rs:1-93`

```text
Word(s)    Offset   Content
─────────  ───────  ─────────────────────────────────────────────────
W0         0        PackedDn address (THE identity, u64)
W1         8        node_kind:u8 | container_count:u8 | geometry:u8
                    | flags:u8 | schema_version:u16 | provenance_hash:u16
W2         16       Timestamps: created_ms:u32 | modified_ms:u32
W3         24       label_hash:u32 | tree_depth:u8 | branch:u8 | reserved:u16
W4-7       32       NARS: freq:f32 | conf:f32 | pos_evidence:f32 | neg_evidence:f32
W8-11      64       DN rung + 7-Layer compact + collapse gate
W12-15     96       7-Layer markers (5 bytes x 7 = 35 bytes)
W16-31     128      Inline edges (64 packed, 4 per word)
W32-39     256      RL / Q-values / rewards
W40-47     320      Bloom filter (512 bits)
W48-55     384      Graph metrics (full precision f64)
W56-63     448      Qualia (18 channels x f16 + 8 slots)
W64-79     512      Rung history + collapse gate history
W80-95     640      Representation language descriptor
W96-111    768      DN-Sparse adjacency (compact inline CSR)
W112-125   896      Reserved
W126-127   1008     Checksum (CRC32:u32 | parity:u32) + schema version
```

### Word Constants (`meta.rs:35-93`)

| Constant          | Value | Purpose                                   |
|-------------------|-------|-------------------------------------------|
| `W_DN_ADDR`       | 0     | PackedDn address                          |
| `W_TYPE`          | 1     | Record type + geometry                    |
| `W_TIME`          | 2     | Timestamps                                |
| `W_LABEL`         | 3     | Label hash + tree metadata                |
| `W_NARS_BASE`     | 4     | NARS truth values (4 words)               |
| `W_DN_RUNG`       | 8     | DN rung + 7-layer compact                 |
| `W_LAYER_BASE`    | 12    | 7-layer markers                           |
| `W_EDGE_BASE`     | 16    | Inline edges start                        |
| `W_EDGE_END`      | 31    | Inline edges end                          |
| `W_RL_BASE`       | 32    | Reinforcement learning data               |
| `W_BLOOM_BASE`    | 40    | Bloom filter (512 bits)                   |
| `W_GRAPH_BASE`    | 48    | Graph metrics                             |
| `W_QUALIA_BASE`   | 56    | Qualia channels                           |
| `W_RUNG_HIST`     | 64    | Rung + collapse gate history              |
| `W_REPR_BASE`     | 80    | Representation language descriptor        |
| `W_ADJ_BASE`      | 96    | DN-Sparse adjacency (inline CSR)         |
| `W_RESERVED`      | 112   | Reserved                                  |
| `W_CHECKSUM`      | 126   | Checksum + version                        |
| `MAX_INLINE_EDGES` | 64   | Maximum edges stored inline               |
| `SCHEMA_VERSION`  | 1     | Current schema version                    |

### MetaView API (`meta.rs:99+`)

```rust
pub struct MetaView<'a> { /* zero-copy borrow of &'a [u64; 128] */ }
pub struct MetaViewMut<'a> { /* mutable borrow */ }
```

Both provide zero-copy access to the word layout above. No allocation on read.

---

## SchemaSidecar — Compact Summary (W224-W255)

Source: `src/width_16k/schema.rs:1-299`

The SchemaSidecar packs identity, reasoning, learning, and topology into the
upper 32 words of the metadata container. It is a compressed snapshot useful
for quick deserialization without parsing the full MetaView.

### Block 14: Identity + Reasoning + Learning (W224-W239)

```text
[224]     depth:u8 | rung:u8 | qidx:u16 | access_count:u32
[225]     ttl:u16 | sigma_q:u16 | node_type:u32
[226]     label_hash:u64
[227]     edge_type:u32 | version:u8 | reserved:u24
[228-229] ANI levels: 8 x u16 = 128 bits
[230]     NARS truth:u32 | budget_lo:u32 (priority + durability)
[231]     budget_hi:u32 (quality + reserved) | reserved:u32
[232-233] Q-values: 16 x i8 = 128 bits
[234-235] Rewards: 8 x i16 = 128 bits
[236-237] STDP: 8 x u16 = 128 bits
[238-239] Hebbian: 8 x u16 = 128 bits
```

### Block 15: Graph Topology + Edges (W240-W255)

```text
[240-243] DN address: 32 x u8 = 256 bits
[244-247] Neighbor bloom: 4 x u64 = 256 bits (3-hash bloom filter)
[248]     Graph metrics: packed u64
[249-255] Inline edges: 7 words = up to 28 edges at 16 bits each
```

### Key Structs

#### NodeIdentity (`schema.rs:46-65`)

```rust
pub struct NodeIdentity {
    pub depth: u8,           // Tree depth (0 = root)
    pub rung: u8,            // Pearl's causal rung: 0=SEE, 1=DO, 2=IMAGINE
    pub qidx: u16,           // Quantization index (codebook entry)
    pub access_count: u32,   // LRU/frequency tracking
    pub ttl: u16,            // Time-to-live in ticks (0 = permanent)
    pub sigma_q: u16,        // Uncertainty: sigma * 1000 as u16
    pub node_type: NodeTypeMarker,
    pub label_hash: u64,
    pub edge_type: EdgeTypeMarker,
}
```

#### AniLevels (`schema.rs:78-89`) — 8 cognitive reasoning levels

```rust
pub struct AniLevels {
    pub reactive: u16,   // Layer 0: stimulus-response
    pub memory: u16,     // Layer 1: episodic recall
    pub analogy: u16,    // Layer 2: structural mapping
    pub planning: u16,   // Layer 3: multi-step lookahead
    pub meta: u16,       // Layer 4: self-reflection
    pub social: u16,     // Layer 5: theory of mind
    pub creative: u16,   // Layer 6: generative novelty
    pub abstract: u16,   // Layer 7: formal reasoning
}
```

Packed as `u128` (16 bits each). `dominant()` returns the index of the highest level.

#### NarsTruth (`schema.rs:137-185`) — NARS truth value

```rust
pub struct NarsTruth {
    pub frequency: u16,   // Quantized 0.0-1.0 as 0-65535
    pub confidence: u16,  // Quantized 0.0-0.9999 as 0-65535
}
```

Methods: `from_floats()`, `f()`, `c()`, `revision()`, `deduction()`, `pack()/unpack()`.
Packed as `u32` (frequency in low 16 bits, confidence in high 16 bits).

#### NarsBudget (`schema.rs:188-222`) — NARS resource allocation

```rust
pub struct NarsBudget {
    pub priority: u16,
    pub durability: u16,
    pub quality: u16,
    pub _reserved: u16,
}
```

Packed as `u64`.

#### EdgeTypeMarker (`schema.rs:225-258`)

```rust
pub struct EdgeTypeMarker {
    pub verb_id: u8,      // Cognitive verb identifier
    pub direction: u8,    // Edge direction
    pub weight: u8,       // Edge weight
    pub flags: u8,        // Bit 0: temporal, Bit 1: causal, Bit 2: hierarchical
}
```

#### NodeTypeMarker (`schema.rs:280-298`)

```rust
pub struct NodeTypeMarker {
    pub kind: u8,         // See NodeKind enum
    pub subtype: u8,
    pub provenance: u16,
}
```

#### NodeKind (`schema.rs:262-276`)

```rust
pub enum NodeKind {
    Entity = 0,
    Concept = 1,
    Event = 2,
    Rule = 3,
    Goal = 4,
    Query = 5,
    Hypothesis = 6,
    Observation = 7,
}
```

---

## Auxiliary Metadata Types

### EnvelopeMetadata (`src/contract/types.rs`)

Wire-format metadata for cross-runtime data envelopes:

```rust
pub struct EnvelopeMetadata {
    pub agent_id: Option<String>,
    pub confidence: Option<f64>,
    pub epoch: Option<i64>,
    pub version: Option<String>,
}
```

Part of the `DataEnvelope` struct shared across ada-n8n, crewai-rust, and ladybug-rs.

### DocumentMeta (`src/storage/corpus.rs`)

Metadata for scent-indexed training corpora. Arrow columns:
- `chunk_id: u64`
- `doc_id: string`
- `text: string`
- `fingerprint: binary[48]` (384-bit scent)
- `position: u32`
- `metadata: json`

### Unified Execution Contract (`src/contract/types.rs`)

```rust
pub struct UnifiedStep {
    pub step_id: String,
    pub execution_id: String,
    pub step_type: String,    // "n8n.*" | "crew.*" | "lb.*" | "core.*"
    pub runtime: String,
    pub name: String,
    pub status: StepStatus,   // Pending | Running | Completed | Failed | Skipped
    pub input: Value,
    pub output: Value,
    pub error: Option<String>,
    pub started_at: DateTime<Utc>,
    pub finished_at: Option<DateTime<Utc>>,
    pub sequence: i32,
    pub reasoning: Option<String>,
    pub confidence: Option<f64>,
    pub alternatives: Option<Value>,
}
```

---

## Invariants

1. **Container 0 = metadata ONLY** — never included in Hamming search
2. **Schema version** in `W127` — currently version 1
3. **Checksum**: CRC32 of content (W126 bits 0-31) + XOR parity of W0-W125 (W126 bits 32-63)
4. **Hot/Cold separation**: cold path metadata NEVER modifies hot path state
5. **Zero-copy access**: MetaView borrows `&[u64; 128]` directly — no allocation
6. **64-byte alignment**: cache-line aligned for SIMD safety

---

## Record Geometry

```text
Full record: 2,048 bytes = 256 x u64
├── Container 0 (metadata):  1,024 bytes = 128 x u64 (W0-W127)
└── Container 1+ (content):  1,024 bytes = 128 x u64 (fingerprints)

16K-bit upgrade record: 16,384 bytes = 2,048 x u64
├── Container 0 (metadata):  1,024 bytes = 128 x u64 (W0-W127)
├── SchemaSidecar:            256 bytes =  32 x u64 (W224-W255)
└── Content containers:       Variable
```
