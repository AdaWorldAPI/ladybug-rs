# CLAUDE.md — Ladybug-RS

> **Last Updated**: 2026-02-27
> **Branch**: claude/check-ladybug-rs-access-yVeJG
> **Status**: All rustynum crates BindSpace-native, Lance persistence wired, Stable 1.93

---

## Single Binary Integration (JITSON)

ladybug-rs is the **host binary** for the single-binary architecture.
Vendor features link crewai-rust and n8n-rs as library crates — no HTTP proxy.

### Vendor Features (Cargo.toml — now uncommented)

```toml
vendor-n8n = ["dep:n8n-core", "dep:n8n-workflow", "dep:n8n-arrow", "dep:n8n-grpc", "dep:n8n-hamming"]
vendor-crewai = ["dep:crewai-vendor", "crewai"]
```

### Dispatch Pattern (src/bin/server.rs)

```
dispatch_crew_post() / dispatch_crew_get()
  #[cfg(feature = "vendor-crewai")] → in-process call
  #[cfg(not(...))] → HTTP proxy to localhost:8090

dispatch_n8n_post() / dispatch_n8n_get()
  #[cfg(feature = "vendor-n8n")] → in-process call
  #[cfg(not(...))] → HTTP proxy to localhost:8091
```

### JITSON Integration Roadmap (n8n-rs side)

| Step | Task | Status |
|------|------|--------|
| J.1 | jitson as dep in n8n-contract | **Done** |
| J.3 | CompiledStyle: ThinkingStyle → ScanKernel | **Done** |
| J.4 | WorkflowHotPath: compile routing tables | **Done** |
| J.5 | Kernel cache in n8n-core | **Done** |
| J.6 | activate → compile → cache → execute | **Done** |

### Remaining Work

| Task | Status |
|------|--------|
| Expose handle_request_body from crewai-rust server module | Needed for vendor-crewai |
| Expose handle_api_post/get from n8n-grpc | Needed for vendor-n8n |
| Implement `SubstrateView` for `BindSpace` (crewai-rust bridge) | **DONE** (substrate_bridge.rs) |
| Wire `BindBridge` into server startup for zero-serde awareness hydration | Planned |
| BERT embedding model for blood-brain barrier inbound translation | Planned |
| Migrate server.rs to async (axum) | Future — cleaner vendor bridge |

### BindSpace ↔ Blackboard Bridge (2026-02-26)

crewai-rust now defines a `SubstrateView` trait that ladybug-rs will implement
for BindSpace. This enables zero-serde data flow:

```
BindSpace.hydrate() → AwarenessFrame TypedSlot → Blackboard
Blackboard → NarsSemanticState → writeback_nars() → BindSpace meta words
Blackboard → XOR delta → flush_deltas() → BindSpace fingerprints
```

Three-tier awareness: Core (BindSpace ↔ Blackboard) → Blood-Brain Barrier
(MarkovBarrier XOR budget) → External (LLM APIs via n8n-rs workflows).

LLMs are in the loop, NOT source of truth. NARS truth gate filters all
external insights before they enter BindSpace.

---

# ⚠️ STOP. READ THIS FIRST. DO NOT SKIP. ⚠️

## CRITICAL: Understand What Already Exists BEFORE Touching Anything

**Every session MUST read this section completely before making ANY changes.**

### The Storage Architecture You MUST NOT Break

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHAT WORKS NOW (default features)                        │
│                    ════════════════════════════════════                     │
│                                                                             │
│  CogRedis ──────────────► BindSpace (8+8 addressing, O(1) lookup)          │
│      │                         │                                           │
│      │ DN.SET/GET              │ 65,536 addresses                          │
│      │ CAM.* (4096 ops)        │ 3-5 cycles per lookup                     │
│      │ DAG.* (ACID)            │ No HashMap, pure arrays                   │
│      │                         │                                           │
│  UnifiedEngine ◄───────────────┤                                           │
│      │                         │                                           │
│      ├── XorDag ───────────────┤ ACID transactions + XOR parity            │
│      ├── MVCC ─────────────────┤ Concurrent writes                         │
│      ├── BitpackedCSR ─────────┤ Zero-copy edge traversal                  │
│      └── ArrowZeroCopy ◄───────┘ Pure Arrow buffers (NO lance crate!)      │
│              │                                                             │
│              │  src/storage/lance_zero_copy/mod.rs                         │
│              │  ════════════════════════════════════                       │
│              │  This is the Arrow integration layer.                       │
│              │  It does NOT depend on the lance crate.                     │
│              │  It uses arrow_array, arrow_buffer directly.                │
│              │  IT ALREADY WORKS. DO NOT REWRITE IT.                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    OPTIONAL (lancedb feature, currently broken)             │
│                    ════════════════════════════════════════════             │
│                                                                             │
│  src/storage/lance.rs ─── Direct Lance Dataset API                         │
│  src/storage/database.rs ── Abstraction over LanceStore                    │
│                                                                             │
│  Status: API mismatch. Cargo.toml says lance="1.0" but vendor has 2.1.0.   │
│  To fix: Add patch to use vendor, then update API calls.                   │
│  NOT CRITICAL: ArrowZeroCopy works without this.                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```


---

## CLAM Hardening — Academic Foundation

**READ**: `docs/CLAM_HARDENING.md` before any search/index work.

CLAM (URI-ABD/clam, MIT, pure Rust) provides the mathematical proof that
ladybug-rs's fingerprint-based search works. Key concepts:

| ladybug-rs | CLAM equivalent | Formal guarantee |
|------------|-----------------|-----------------|
| Scent hierarchy | CLAM Tree (bipolar split) | O(k · 2^LFD · log n) |
| HDR cascade filtering | d_min/d_max bounds | Triangle inequality: provable |
| XOR-fold scent | Cluster center + radius | Metric ball containment |
| Mexican hat thresholds | Cluster radius/span | Adaptive, data-driven |
| Full fingerprints | panCAKES XOR-diff | 5-70x compression, search on compressed |

Papers: CHESS (1908.08551), CHAODA (2103.11774), CAKES (2309.05491), panCAKES (2409.12161)

### Feature Flags - Know What They Enable

```toml
# Cargo.toml
default = ["simd", "parallel"]     # ← Core functionality

# THESE WORK:
simd = []                          # AVX-512/AVX2 Hamming distance
parallel = ["rayon"]               # Parallel processing
flight = ["arrow-flight", ...]     # Arrow Flight gRPC server

# THIS IS BROKEN (API mismatch):
lancedb = ["lance"]                # Direct Lance Dataset API
                                   # lance.rs needs updating for 2.1 API
```

### Vendor Directory Structure

```
vendor/
├── lance/           # Lance 2.1.0-beta.0 (NOT currently used by Cargo.toml!)
│   └── rust/lance/  # The actual crate
└── lancedb/         # LanceDB (NOT currently used)

# Cargo.toml currently pulls from crates.io:
lance = { version = "1.0", optional = true }  # ← MISMATCH with vendor

# To use vendor instead, add to Cargo.toml:
[patch.crates-io]
lance = { path = "vendor/lance/rust/lance" }
```

### Dockerfile Build Features

```dockerfile
ARG FEATURES="simd,parallel,flight"  # ← Current production build
# lancedb NOT included because lance.rs has API issues
```

---

## What Each Storage Module Does

| Module | Purpose | Dependencies | Status |
|--------|---------|--------------|--------|
| `bind_space.rs` | 8+8 addressing, O(1) arrays, rustynum-native ops | rustynum-{core,holo,clam} | ✅ Working |
| `cog_redis.rs` | Redis syntax (DN.*, CAM.*, DAG.*) | bind_space | ✅ Working |
| `unified_engine.rs` | ACID + CSR + MVCC + ArrowZeroCopy | All below | ✅ Working |
| `xor_dag.rs` | ACID transactions, XOR parity | bind_space | ✅ Working |
| `lance_zero_copy/` | **Pure Arrow buffers** + write-through | arrow_* only | ✅ Working |
| `lance_persistence.rs` | Lance cold-tier (flush_aged, hydrate) | lance, arrow | ✅ Working |
| `lance.rs` | Direct Lance Dataset API | lance crate | ❌ API mismatch |
| `database.rs` | Abstraction over lance.rs | lance.rs | ❌ Blocked |

### The Key Insight

**`lance_zero_copy/` is NOT the same as `lance.rs`.**

- `lance_zero_copy/` = Pure Arrow integration, NO lance crate dependency
- `lance.rs` = Direct Lance Dataset API, REQUIRES lance crate

The owner already built the Arrow zero-copy layer. It feeds data back to BindSpace through `ArrowZeroCopy`, `FingerprintBuffer`, `LanceView`, and `ZeroCopyBubbler`. This is the production path.

---

## Role in the Four-Level Architecture

Ladybug-RS is **Level 2 — Awareness** (temporal process).

> **Canonical cross-repo architecture:** [ada-docs/architecture/FOUR_LEVEL_ARCHITECTURE.md](https://github.com/AdaWorldAPI/ada-docs/blob/main/architecture/FOUR_LEVEL_ARCHITECTURE.md)

Ladybug-RS owns the temporal dimension: the 10-layer cognitive stack processed
in 7 waves, HDR resonance (3D triangle: Guardian/Catalyst/Balanced), FocusMask,
12 ThinkingStyles with FieldModulation, CollapseGate (SD-based FLOW/HOLD/BLOCK),
and the AwarenessBlackboard (grey matter → gate → white matter).

**Resonance is selection, not thought.** HDR resonance selects *which* atoms
activate on the flow. It does not reason. Reasoning happens in graph edges
(neo4j-rs, Level 3). Thinking styles as JIT workflows are composed by
crewai-rust/n8n-rs (Level 4).

---

## Project Identity

**Ladybug-RS** is a pure-Rust cognitive substrate implementing:
- 8+8 address model (65,536 addresses, no FPU required)
- Redis syntax with cognitive semantics
- Universal bind space where all query languages hit same addresses
- 4096 CAM operations translated to LanceDB ops

**Repository**: https://github.com/AdaWorldAPI/ladybug-rs

---

## The 8+8 Address Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PREFIX (8-bit) : SLOT (8-bit)                          │
├─────────────────┬───────────────────────────────────────────────────────────┤
│  0x00-0x0F:XX   │  SURFACE (16 prefixes × 256 = 4,096)                      │
│                 │  0x00: Lance      0x04: NARS       0x08: Concepts         │
│                 │  0x01: SQL        0x05: Causal     0x09: Qualia           │
│                 │  0x02: Cypher     0x06: Meta       0x0A: Memory           │
│                 │  0x03: GraphQL    0x07: Verbs      0x0B: Learning         │
│                 │  0x0C: Agents  0x0D: Thinking  0x0E: Blackboard  0x0F: A2A │
├─────────────────┼───────────────────────────────────────────────────────────┤
│  0x10-0x7F:XX   │  FLUID (112 prefixes × 256 = 28,672)                      │
│                 │  Edges + Context selector + Working memory                │
│                 │  TTL governed, promote/demote                             │
├─────────────────┼───────────────────────────────────────────────────────────┤
│  0x80-0xFF:XX   │  NODES (128 prefixes × 256 = 32,768)                      │
│                 │  THE UNIVERSAL BIND SPACE                                 │
│                 │  All query languages hit the same addresses               │
└─────────────────┴───────────────────────────────────────────────────────────┘
```

**Critical**: The 16-bit address is NOT a hash. It's direct array indexing.
```rust
let prefix = (addr >> 8) as u8;
let slot = (addr & 0xFF) as u8;
// 3-5 cycles. No HashMap. No FPU. Works on embedded/WASM.
```

---

## Current State

**Codebase**: ~40K lines of Rust
**Last updated**: 2026-02-27
**Rust**: 1.93 stable (edition 2024) — ALL repos on stable, zero nightly
**DataFusion**: 51 (DF 52 upgrade path documented)
**Arrow**: 57.x / arrow-flight 57 / tonic 0.14
**rustynum**: ALL 5 crates wired (rs, core, arrow, holo, clam)

### ✅ Completed

| Feature | Location | Status |
|---------|----------|--------|
| 8+8 addressing (prefix:slot) | bind_space.rs | ✓ Working |
| Universal BindSpace O(1) indexing | bind_space.rs | ✓ Working |
| 4096 CAM operations (16×256) | cam_ops.rs | ✓ Working |
| CogRedis command executor | cog_redis.rs | ✓ Working |
| DN.* tree commands | cog_redis.rs | ✓ Working |
| DAG.* ACID transactions | cog_redis.rs | ✓ Working |
| UnifiedEngine | unified_engine.rs | ✓ Working |
| ArrowZeroCopy | lance_zero_copy/ | ✓ Working |
| HDR Cascade Search | hdr_cascade.rs | ✓ Working |
| Arrow Flight Server | flight/server.rs | ✓ Working |
| Flight Streaming (DoGet) | flight/server.rs | ✓ Working |
| MCP Actions (DoAction) | flight/actions.rs | ✓ Working |
| HTTP server with CogRedis | bin/server.rs | ✓ Working |
| Flight gRPC binary | bin/flight_server.rs | ✓ Working |
| rustynum-holo BindSpace-native | bind_space.rs | ✓ Working |
| rustynum-clam BindSpace-native | bind_space.rs | ✓ Working |
| lance_zero_copy write-through | bind_space.rs | ✓ Working |
| Hot→cold tier flush (flush_aged) | lance_persistence.rs | ✓ Working |
| Parallel bulk ops (split_at_mut) | bind_space.rs | ✓ Working |
| SubstrateView (crewai bridge) | substrate_bridge.rs | ✓ Working |
| 10-layer cognitive stack | layer_stack.rs | ✓ Working |
| Tier 4 dispatch (n8n/crewai) | server.rs | ✓ Working |
| Stable Rust 1.93 (all repos) | Cargo.toml | ✓ Working |

### ❌ Known Issues

| Issue | Location | Status |
|-------|----------|--------|
| lance.rs API mismatch | storage/lance.rs | Vendor=2.1, Cargo=1.0 |
| 10 test failures | Various | See test section below |

### Recent Commits

```
3610a9e Wire rustynum-holo, rustynum-clam, lance_zero_copy natively through BindSpace
f3f83fd Wire Lance persistence + BindSpace parallel ops + age-based hot→cold flush
1c7f274 Revert server/orchestration dispatch — keep rustynum stable integration
afd5b09 Make rustynum/n8n-rs/crewai-rust obligatory — remove all feature gates
a3d9307 P0: Wire rustynum_accel into simd.rs, bind_space.rs, hdr_cascade.rs
dc446d9 Correct P2 TODO: ALL AVX-512 intrinsics stable 1.93.1, ZERO gaps
```

---

## Key Files

```
src/bin/
├── server.rs           # HTTP server (port 8080)
└── flight_server.rs    # Arrow Flight gRPC (port 50051)

src/storage/
├── bind_space.rs       # Universal DTO (8+8 addressing) ← THE CORE
│                       # Now includes: rustynum-holo phase/carrier/focus ops,
│                       # rustynum-clam batch_hamming/clam_top_k,
│                       # lance_zero_copy write-through, parallel_into_slices,
│                       # age-based hot→cold flush, NARS revision
├── cog_redis.rs        # Redis syntax adapter ← DN.*, CAM.*, DAG.*
├── unified_engine.rs   # Composes all storage features
├── xor_dag.rs          # ACID + XOR parity
├── lance_zero_copy/    # Pure Arrow integration (NO lance crate!) ← WORKS
│   └── mod.rs          # ArrowZeroCopy, FingerprintBuffer, LanceView,
│                       # LanceWriteThrough, UnifiedStorage, StorageBackend,
│                       # AdjacencyIndex, WAL, MVCC, Transactions
├── lance_persistence.rs # Lance Dataset cold-tier persistence ← WORKS
│                       # flush_aged(), persist_nodes(), hydrate_nodes()
├── lance.rs            # Direct Lance Dataset API (BROKEN - API mismatch)
└── database.rs         # Over lance.rs (BLOCKED)

src/flight/
├── mod.rs              # Module exports
├── server.rs           # LadybugFlightService
├── actions.rs          # MCP action handlers
└── capabilities.rs     # Transport negotiation

src/search/
├── hdr_cascade.rs      # HDR filtering (~7ns per candidate)
├── cognitive.rs        # NARS + Qualia + SPO
└── causal.rs           # SEE/DO/IMAGINE

src/learning/
├── cam_ops.rs          # 4096 CAM operations
├── quantum_ops.rs      # Quantum-style operators
└── causal_ops.rs       # Pearl's 3 rungs
```

---

## Testing

```bash
# Default features (what works)
cargo test

# With all working features
cargo test --features "simd,parallel,flight"

# With experimental features (some failures expected)
cargo test --features "simd,parallel,codebook,hologram,quantum,spo"

# Check compilation only
cargo check --features "simd,parallel,flight"
```

### 🔴 Test Failures (10 total with experimental features)

| Test | Error | Root Cause |
|------|-------|------------|
| `collapse_gate::test_sd_calculation` | threshold | Algorithm logic |
| `quantum_ops::test_permute_adjoint` | not inverse | Permute logic |
| `cypher::test_variable_length` | ParseFloatError | Tokenizer |
| `causal_ops::test_store_query_correlation` | empty | Query issue |
| `causal::test_correlation_store` | empty | Query issue |
| `context_crystal::test_temporal_flow` | popcount=0 | Insert not persisting |
| `nsm_substrate::test_codebook_initialization` | primes<60 | Init issue |
| `nsm_substrate::test_learning` | vocab<65 | Learning issue |
| `jina_cache::test_cache_hit_rate` | off-by-one | Trivial fix |
| `crystal_lm::test_serialize` | None unwrap | Validation strict |

---

## DataFusion 51 → 52 Upgrade Path

Currently on DF 51 to avoid liblzma conflict. For DF 52:

```toml
# In Cargo.toml, change:
datafusion = "52"

# And uncomment:
[patch.crates-io]
datafusion = { git = "https://github.com/AdaWorldAPI/datafusion", branch = "liblzma-fix" }
```

---

## Lance Vendor Integration (When Ready)

To use the vendored Lance 2.1.0-beta.0:

```toml
# Add to Cargo.toml:
[patch.crates-io]
lance = { path = "vendor/lance/rust/lance" }
```

Then update `src/storage/lance.rs` for the 2.1 API:
- `Dataset::query()` method changed
- Schema types moved to `lance::datatypes::Schema`
- RecordBatchReader trait requirements changed

---

## Two-Layer Architecture: Addressing vs Compute

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LAYER 1: ADDRESSING                               │
│                              (always int8)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  prefix:slot (u8:u8) → array index → 3-5 cycles                            │
│  Works everywhere: embedded, WASM, Raspberry Pi, phone                     │
│  NO runtime detection needed                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       LAYER 2: COMPUTE (adaptive)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  AVX-512 (Railway, modern Xeon): ~2ns per comparison                       │
│  AVX2 (most laptops): ~4ns per comparison                                  │
│  Fallback (WASM, ARM): ~50ns per comparison                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## MCP Agent Guidance

When context window > 60%, spawn continuation with state:

```yaml
handover:
  current_task: "Description"
  files_modified: [...]
  decisions: [...]
  next_steps: [...]
  blockers: []
```

---

## Extended Documentation

The following documents provide deep-dive coverage of specific topics:

### Storage Layer Hardening

| Document | Purpose |
|----------|---------|
| [`docs/STORAGE_CONTRACTS.md`](docs/STORAGE_CONTRACTS.md) | **9 race conditions** identified in storage stack with root cause analysis |
| [`docs/REWIRING_GUIDE.md`](docs/REWIRING_GUIDE.md) | **Copy-paste ready fixes** for each race condition |
| [`docs/BACKUP_AND_SCHEMA.md`](docs/BACKUP_AND_SCHEMA.md) | XOR diff versioning, S3 integration, schema migrations |
| [`docs/DELTA_ENCODING_FORMATS.md`](docs/DELTA_ENCODING_FORMATS.md) | Multi-format delta encoding with prefix envelope headers |

### Delta Encoding (Prefix Envelope)

Magic bytes in prefix envelope determine format:
```
Prefix FF:FF              → Sparse Bitpacked (2^16 addr space)
Prefix FF:FF + FF:FF      → Float32/32-bit Hamming Delta
Prefix FF:FF + FF:FF + FF:FF → Non-Sparse 48-bit / 10000D XOR
```

### Critical Race Conditions (Summary)

| # | Location | Severity | Issue |
|---|----------|----------|-------|
| 1 | `hardening.rs:LruTracker` | HIGH | Duplicate entries in order queue |
| 2 | `hardening.rs:WriteAheadLog` | CRITICAL | Write-behind (not write-ahead) |
| 3 | `resilient.rs:WriteBuffer` | HIGH | ID allocated before buffered |
| 4 | `resilient.rs:DependencyGraph` | MEDIUM | Partial map updates |
| 5 | `xor_dag.rs:commit` | HIGH | TOCTOU in parity update |
| 6 | `xor_dag.rs:EpochGuard` | MEDIUM | Work item orphan on epoch advance |
| 7 | `snapshots.rs:TieredStorage` | MEDIUM | Eviction races with writes |
| 8 | `snapshots.rs:SnapshotChain` | LOW | Chain length race |
| 9 | `temporal.rs:commit` | HIGH | Serializable conflict detection gap |

**Before touching storage code, read `docs/STORAGE_CONTRACTS.md`.**

### Backup Strategy (Summary)

```
PRIMARY:   Redis (Railway)  - XOR deltas, <1ms latency
SECONDARY: PostgreSQL       - Schema metadata, versions
ARCHIVE:   S3 (via Lance)   - Full Parquet snapshots, 11 9s durability
```

**Before changing backup logic, read `docs/BACKUP_AND_SCHEMA.md`.**

---

## Composition Model — XOR Binding vs Parallel Slicing vs Address Space

### READ THIS BEFORE TOUCHING STORAGE CODE

ladybug-rs uses three composition mechanisms. They are NOT interchangeable.
Each operates at a different level. Using the wrong one breaks the architecture.

### Decision Matrix

| Question | Answer → Mechanism |
|----------|-------------------|
| "Are these regions accessed **independently**?" | YES → **Parallel Slicing** |
| "Must both signals **coexist** in search?" | YES → **XOR Binding** |
| "Need to **recover one signal from another**?" | YES → **XOR Binding** (self-inverse) |
| "Are they **functionally unrelated**?" | YES → **Parallel Slicing** |
| "Is memory allocation a bottleneck?" | YES → **XOR Binding** (1 container vs 2) |
| "Do you need **tree topology**?" | YES → **Address Space** + DnIndex |

### Mechanism 1: Parallel Slicing (Independent Regions)

Used when regions are **functionally independent** and both accessed
in **different query types**. No semantic relationship between them.

```
BindNode.fingerprint[256 words]
├── [0..128]   = META container  (graph structure, NARS, timestamps)
└── [128..256] = CONTENT container (semantic search, HDC fingerprint)
```

Access pattern:
```rust
let meta = node.meta_words();       // &[u64; 128] — borrows [0..128]
let content = node.content_words(); // &[u64; 128] — borrows [128..256]
// No borrow conflict: non-overlapping slices
```

**When to use**: Metadata vs content. Graph topology vs search fingerprint.
Any case where you need fast direct access to two unrelated concerns.

### Mechanism 2: XOR Binding (Reversible Superposition)

Used when signals are **semantically related** and should **coexist in
the same container** for query flexibility.

```rust
// Bind: SPO structure + semantic axes → composite container
let composite = spo_container.xor(&axis_container);

// Unbind: recover either signal by XOR with the other
let recovered_spo = composite.xor(&axis_container);  // = spo_container
let recovered_axes = composite.xor(&spo_container);  // = axis_container
```

**Why it works**: XOR is self-inverse and associative.
`A ⊕ B ⊕ B = A` always. Both signals live in one container,
either queryable via unbind. Half the storage cost.

**When to use**: SPO triples + semantic axes. Edge encoding
(from ⊕ permute(verb) ⊕ permute(to)). Any case where you need
two signals in one container and want to query by either.

### Mechanism 3: Address Space (Tree Hierarchy)

Used for **DN tree topology**. Parent-child relationships are
NOT encoded in fingerprints — they're encoded in Addr assignments.

```rust
let parent_addr = Addr::new(0x80, 0x01);
let child_addr = Addr::new(0x81, 0x02);
dn_index.register(child_packed_dn, child_addr);
// Relationship is in DnIndex, not in fingerprint content
```

**When to use**: Any tree/graph topology. Reparenting doesn't
require recomputing fingerprints. Upward traversal via PackedDn
is O(1) bit masking.

---

## CollapseGate — Dispersion-Based Compute Allocation

CollapseGate is **NOT** a storage mechanism. It's a **decision gate**
that controls when superposition collapses to commitment.

### Three Gate States

| State | SD Range | Action | Semantics |
|-------|----------|--------|-----------|
| **FLOW** | SD < 0.15 | Collapse immediately | Clear winner → commit |
| **HOLD** | 0.15 ≤ SD ≤ 0.35 | Maintain superposition | Ruminate → store in SPPM |
| **BLOCK** | SD > 0.35 | Cannot collapse | Need clarification → ask user |

SD measures **dispersion across candidate scores**, not confidence.
FLOW = fast path. HOLD = exploratory. BLOCK = expensive.

### CollapseGate in the Writethrough Pattern

This is how the Blackboard borrow-mut scheme works without breaking
ownership or zero-copy:

```
Step 1: Agent READS AwarenessFrame from Blackboard (immutable borrow)
        ├─ No copy: TypedSlot returns &AwarenessFrame
        └─ The frame is owned by the Blackboard

Step 2: NARS driver runs pure inference → produces NEW NarsSemanticState
        ├─ Pure function: (&AwarenessFrame, &axes) → NarsSemanticState
        └─ New allocation — no borrow conflict with Step 1

Step 3: SPO driver runs → produces NEW Vec<SpoTriple>
        └─ Same pattern: pure function, new allocation

Step 4: CollapseGate evaluates the candidates
        ├─ FLOW: low dispersion → safe to commit
        ├─ HOLD: high dispersion → buffer in SPPM, don't commit yet
        └─ BLOCK: extreme dispersion → cannot proceed

Step 5 (FLOW): Agent WRITES new TypedSlots to Blackboard
        ├─ bb.put_typed("awareness:nars", nars_state, ...)
        ├─ bb.put_typed("awareness:spo_triples", triples, ...)
        ├─ Different keys from Step 1 → no borrow conflict
        └─ TypedSlot is Box<dyn Any> → moved, not copied

Step 6: BindSpace reads the new TypedSlots
        ├─ Computes XOR delta: old ⊕ new = sparse diff
        ├─ XorDelta stores only changed words (typically 1-2 of 256)
        ├─ Applies delta to storage: words[i] ^= delta.values[vi]
        └─ Marks address dirty in 65K-bit bitset
```

**Why this is zero-copy**:
- Step 1: `&AwarenessFrame` reference, no copy
- Steps 2-3: New allocations (driver outputs), not copies of input
- Step 5: `Box<dyn Any>` moved into Blackboard, no copy
- Step 6: XOR delta is sparse (1-2 words), not a 2KB container copy

**Why borrow-mut is safe**:
- Reads and writes use DIFFERENT slot keys
- Each phase owns its output slots exclusively
- Phase discipline ensures one writer at a time
- No mutable reference to input survives past the pure function call

**Why CollapseGate is critical**:
- Without it, every awareness update would immediately commit
- HOLD state allows multiple turns of evidence to accumulate
- BLOCK state prevents nonsensical commits when evidence is contradictory
- This maps to NARS: FLOW = crystallized, HOLD = uncertain, BLOCK = tensioned

---

## XOR Delta Caching (width_16k/xor_bubble.rs)

The XOR writethrough to storage uses sparse deltas:

```rust
pub struct XorDelta {
    pub bitmap: [u64; 4],   // 256-bit mask (which words changed)
    pub values: Vec<u64>,   // Only the non-zero XOR values
}

// Compute: old ⊕ new
let delta = XorDelta::compute(&old_words, &new_words);

// Apply: mutate storage in-place
delta.apply(&mut storage_words);
```

A typical awareness update changes 1-2 words of 256. The delta is
~16 bytes instead of 2048 bytes. This is why the writethrough is
effectively zero-copy at the storage level.

---

## Ownership Map — Who Owns What

| Component | Owner | Location | Sacred? |
|-----------|-------|----------|---------|
| BindSpace (8+8 addressing) | ladybug-rs | `src/storage/bind_space.rs` | YES |
| CogRedis (Redis syntax) | ladybug-rs | `src/storage/cog_redis.rs` | YES |
| ArrowZeroCopy | ladybug-rs | `src/storage/lance_zero_copy/` | YES |
| LancePersistence | ladybug-rs | `src/storage/lance_persistence.rs` | YES |
| CollapseGate | ladybug-rs | `src/cognitive/collapse_gate.rs` | YES |
| 10-Layer Stack | ladybug-rs | `src/cognitive/layer_stack.rs` | YES |
| WideMetaView | ladybug-contract | `src/wide_meta.rs` | YES |
| Container XOR/Hamming | ladybug-contract | `src/container.rs` | YES |
| StorageBackend trait | lance_zero_copy | `src/storage/lance_zero_copy/mod.rs` | YES |
| SIMD dispatch | rustynum | `rustynum-core/src/simd.rs` | YES |
| simd_compat (stable wrappers) | rustynum | `rustynum-core/src/simd_compat.rs` | YES |
| K0/K1/K2 pipeline | rustynum | `rustynum-core/src/kernels.rs` | YES |
| Phase/Carrier/Focus ops | rustynum | `rustynum-holo/src/{phase,carrier,focus}.rs` | YES |
| CogRecordView<'a> | rustynum | `rustynum-arrow/src/arrow_bridge.rs` | YES |
| CascadeIndices | rustynum | `rustynum-arrow/src/indexed_cascade.rs` | YES |
| ClamTree / HammingSIMD | rustynum | `rustynum-clam/src/tree.rs` | YES |
| bundle_byte_slices | rustynum | `rustynum-rs/src/num_array/hdc.rs` | YES |
| Blackboard | crewai-rust | `src/blackboard/view.rs` | YES |
| TypedSlots | crewai-rust | `src/blackboard/typed_slot.rs` | YES |
| SubstrateView trait | crewai-rust | `src/blackboard/bind_bridge.rs` | YES |
| Drivers (NARS, SPO) | crewai-rust | `src/drivers/` | YES |

**"Sacred" means: do not modify the public interface without consensus
across all repos that depend on it.**

---

## The Rustynum Acceleration Contract (HARD-WON — 6 REWRITES)

> **After 1.5M lines of code and 6 rewrites, this lesson was learned:
> Arrow is the zero-copy backbone. Lance is the cold tier.
> rustynum is the acceleration layer. NEVER copy Arrow buffers into owned memory.**

### Why This Matters

Everything goes through **BindSpace**. BindSpace needs:
1. **rustynum** — SIMD kernels, Fingerprint types, DeltaLayer, CollapseGate
2. **Lance/Arrow** — mmap'd zero-copy buffers for cold-tier persistence
3. **split_at_mut / parallel_into_slices** — lock-free parallel writes to disjoint regions

The dilemma: wiring rustynum's acceleration into the Lance data path
WITHOUT breaking the zero-copy chain from Arrow mmap → BindSpace → compute.

### The Zero-Copy Chain (DO NOT BREAK)

```
Lance (cold tier, disk)
    │ mmap
    ▼
Arrow Buffer (64-byte aligned, zero-copy)
    │ FingerprintBuffer.get(i) → &[u64; 256] directly into mmap
    ▼
BindSpace (8+8 addressing, O(1) lookup)
    │ Container bytes = same pointer as Arrow buffer
    ▼
rustynum-core SIMD kernels
    │ select_hamming_fn()(container_a, container_b) → u32
    │ k0_probe() → k1_stats() → k2_exact() cascade
    ▼
Result (no allocation needed for distance computation)
```

**Where it USED TO break (P1 debt — NOW FIXED in rustynum-arrow):**

```
Arrow Buffer → record_batch_to_cogrecords() → .to_vec() PER ROW  ← FIXED
CogRecord[] → CascadeIndices::build() → extend_from_slice()       ← FIXED
```

**Fix applied:** `CogRecordView<'a>` (commit b709737) borrows Arrow's buffer.
`CascadeIndices::build_from_arrow()` builds directly from RecordBatch.
**Remaining:** Wire CogRecordView into `hydrate_nodes()` for the full zero-copy path.

### The Acceleration Stack (What rustynum Provides)

| rustynum Crate | Acceleration | For BindSpace | Status |
|---------------|-------------|---------------|--------|
| **rustynum-core** | Hamming VPOPCNTDQ, K0/K1/K2, BF16 awareness | Primary search, CollapseGate awareness | ✅ BindSpace-native |
| **rustynum-rs** | NumArray, bundle_byte_slices, HDC ops | Binary bundle/bind, NARS truth | ✅ BindSpace-native |
| **rustynum-arrow** | CogRecordView<'a>, CascadeIndices, Arrow bridge | Zero-copy Lance hydration | ✅ Dep wired |
| **rustynum-holo** | Phase bind, Wasserstein, carrier, focus, Gabor | Spatial navigation, focus-of-attention | ✅ BindSpace-native |
| **rustynum-clam** | CLAM tree, HammingSIMD, triangle-inequality | Batch Hamming, top-k nearest | ✅ BindSpace-native |
| **rustyblas** | 138 GFLOPS GEMM, INT8 VNNI, BF16 mixed-precision | Batch similarity, SimHash projection | Indirect (via core) |
| **rustymkl** | VML (exp, ln, sigmoid), LAPACK, FFT | NARS truth scoring, spectral analysis | Indirect (via core) |
| **jitson** | JSON config → native AVX-512 scan kernels | Per-query compiled scans on Arrow buffers | Available |

### Parallel Write Patterns (CANONICAL — From rustynum)

**For contiguous byte slices (GEMM rows, Arrow columns):**
```rust
// split_at_mut — zero synchronization, each thread owns its rows
parallel_into_slices(output_buf, rows_per_thread * n, |offset, chunk| {
    // chunk is exclusively owned by this thread — no locks
});
```

**For binary vector state (fingerprints, CogRecords):**
```rust
// XOR Delta Layers — ground truth is &self forever
let mut stack = LayerStack::new(ground_truth, num_agents);
stack.writer_mut(agent_id).write(&ground, &desired);
// Conflict: stack.evaluate(threshold) → Flow/Hold/Block
// Commit: stack.commit() — XOR all deltas into ground truth
```

**NEVER use `Arc<Mutex>` for parallel writes. NEVER use `&self → &mut` casts.**

### What ladybug MUST NOT Duplicate

ladybug-rs has `src/core/simd.rs` with its own Hamming implementation.
rustynum-core has `simd.rs` with runtime-dispatched Hamming (AVX-512/AVX2/scalar).

**Rule:** ladybug-rs should call rustynum's SIMD, not maintain a duplicate.
The vendored rustynum at `vendor/rustynum/` provides the canonical dispatch.
Delete ladybug's simd.rs when the stable-Rust blocker is resolved.

### portable_simd → std::arch Port (DONE — 2026-02-27)

The whole stack is pinned to **stable Rust 1.93, Arrow 57, DataFusion 51**.
Nightly is NEVER used — it changes dependencies across ALL repos.

**COMPLETED**: 870 `portable_simd` occurrences replaced with `std::arch` wrappers
in `rustynum-core/src/simd_compat.rs` (~1300 lines). All 622 tests pass on
stable 1.93.1 (commit `ebbf554`). Only remaining `portable_simd`:
- `rustynum-archive/` — sacred, don't touch
- `simd_avx2.rs` — non-default AVX2 fallback path
- Doc comments in `simd_compat.rs`

### Cross-Repo References

- `rustynum/CLAUDE.md` § 12 "The Lance Zero-Copy Contract"
- `PLAN-RUSTYNUM-INTEGRATION.md` — 5-phase integration roadmap
- `docs/LANCE_HARVEST.md` — why Lance is cold-tier only
- `docs/BINDSPACE_UNIFICATION.md:1122` — `Buffer::from_slice_ref()` warning
- `crewai-rust/CLAUDE.md` § "Storage Strategy"

---

## OPEN TODOs — Wiring Checklist (SESSION-DURABLE)

> **READ THIS EVERY SESSION.** Do NOT invent new code. Wire EXISTING.
> Mark items DONE with date. If you skip an item, explain why.

### P0 — Stop Duplicating rustynum

- [x] **Wire rustynum_accel into core search** — **DONE 2026-02-27**
  - `src/core/rustynum_accel.rs` has `fingerprint_hamming()`, `container_hamming()`, `slice_hamming()`
  - NOW wired into: `src/storage/bind_space.rs` (hamming, similarity, popcount, dot_i8, bundle,
    bind, nearest, cascade_search), `src/search/hdr_cascade.rs`

- [x] **Fix .to_vec() in rustynum_accel.rs** — **DONE 2026-02-27**
  - Added `bundle_byte_slices(&[&[u8]]) → Vec<u8>` to rustynum-rs/hdc.rs
  - Updated container_bundle to use zero-copy slices (no .to_vec())

- [ ] **Delete src/core/simd.rs** (348 lines) — Still duplicates rustynum's hamming_distance
  - Replace all call sites with `rustynum_core::simd::hamming_distance()` (has scalar fallback)
  - Note: rustynum's select_hamming_fn() does runtime dispatch (AVX-512 > AVX2 > scalar)

### P1 — Zero-Copy Pipeline

- [x] **CogRecordView<'a>** exists in rustynum-arrow (commit b709737) — **DONE**
  - 32 bytes per view vs 8192 per owned CogRecord
  - Available via `rustynum_arrow::cogrecord_views(batch)` — borrows Arrow buffers

- [x] **CascadeIndices::build_from_arrow()** exists (rustynum-arrow) — **DONE**
  - Eliminates temp allocs — builds directly from Arrow RecordBatch

- [ ] **Wire CogRecordView into BindSpace hydration path**
  - `lance_persistence.rs:hydrate_nodes()` currently copies full records
  - Should use CogRecordView<'a> to borrow from Arrow batch → write to BindSpace

### P2 — Toolchain: Ship on Stable 1.93

- [x] **portable_simd → std::arch port** — **DONE 2026-02-27** (commit ebbf554)
  - 870 occurrences replaced across rustynum-core
  - simd_compat.rs (~1300 lines) provides stable std::arch wrappers
  - ALL 622 tests pass on stable Rust 1.93.1
  - Only remaining portable_simd: archive crate (sacred, don't touch),
    simd_avx2.rs (non-default AVX2 fallback), doc comments in simd_compat.rs

### P3 — BindSpace-Native rustynum Integration (NEW — 2026-02-27)

All rustynum crates now permeate through BindSpace via transparent address
delegation. Each method takes `Addr`, looks up `&[u8]` fingerprint data
(zero-copy via `view_u64_as_bytes`), and passes the reference directly
to the rustynum function. No duplication.

**Cargo.toml deps (all OBLIGATORY, not behind feature gates):**
```toml
rustynum-rs = { path = "../rustynum/rustynum-rs" }
rustynum-core = { path = "../rustynum/rustynum-core", features = ["avx512"] }
rustynum-arrow = { path = "../rustynum/rustynum-arrow", features = ["arrow"] }
rustynum-holo = { path = "../rustynum/rustynum-holo", features = ["avx512"] }
rustynum-clam = { path = "../rustynum/rustynum-clam", features = ["avx512"] }
```

**BindSpace methods wired (bind_space.rs ~2050-2350):**

| Category | Methods | rustynum Crate | Zero-Copy? |
|----------|---------|---------------|------------|
| **SIMD Core** | `hamming()`, `similarity()`, `popcount()`, `dot_i8()` | rustynum-core | ✅ `view_u64_as_bytes` |
| **Binary HDC** | `bundle()`, `bind()`, `nearest()`, `cascade_search()` | rustynum-core | ✅ direct `&[u8]` |
| **Phase-space** | `phase_bind()`, `phase_unbind()`, `wasserstein()`, `circular_distance()` | rustynum-holo | ✅ `fp_bytes()` |
| **Phase util** | `phase_histogram()`, `phase_bundle()`, `sort_phase()`, `histogram_distance()` | rustynum-holo | ✅ `fp_bytes()` |
| **Carrier** | `carrier_distance()`, `carrier_correlation()`, `carrier_spectrum()`, `spectral_distance()` | rustynum-holo | ✅ `as_i8()` reinterpret |
| **Focus** | `focus_xor()`, `focus_read()`, `focus_hamming()`, `focus_l1()` | rustynum-holo | ✅ read path, copy-back on mut |
| **Focus bind** | `focus_bind_phase()`, `focus_unbind_phase()`, `focus_xor_auto()` | rustynum-holo | ✅ copy-back on mut |
| **Batch search** | `batch_hamming()`, `clam_top_k()` | rustynum-clam | ✅ SIMD Hamming via core |
| **Lance bridge** | `lance_write_through()`, `unified_storage()`, `arrow_zero_copy()` | lance_zero_copy | ✅ hot cache + mmap |
| **Lance export** | `adjacency_index()`, `to_fingerprint_buffer()` | lance_zero_copy | ✅ ownership transfer |

**Hot→Cold tier management (bind_space.rs + lance_persistence.rs):**

| Method | Location | Purpose |
|--------|----------|---------|
| `BindNode.updated_at` | bind_space.rs | Epoch millis timestamp per node |
| `collect_aged(threshold_secs)` | bind_space.rs | Find nodes older than threshold |
| `evict_aged(threshold_secs)` | bind_space.rs | Remove aged nodes after persist |
| `flush_aged(space, secs)` | lance_persistence.rs | Append to Lance → evict from hot |
| `nodes_to_batch()` | lance_persistence.rs | Convert (Addr, BindNode) → Arrow RecordBatch |
| `persist_nodes()` | lance_persistence.rs | Full snapshot to Lance |
| `hydrate_nodes()` | lance_persistence.rs | Restore from Lance → BindSpace |

**Parallel bulk operations (bind_space.rs ~1860-1977):**

| Method | Pattern | Purpose |
|--------|---------|---------|
| `parallel_into_node_slices(f)` | `split_at_mut` | Lock-free parallel write to disjoint chunks |
| `bulk_write_nodes(items)` | Sequential | Batch insert pre-parsed nodes |
| `parallel_hydrate_nodes(items)` | `split_at_mut` per chunk | Zero-sync Lance → BindSpace hydration |

### NEXT STEPS (prioritized)

| Priority | Task | Effort | Blocked by |
|----------|------|--------|------------|
| **N1** | Wire `BindBridge` into server startup for zero-serde awareness hydration | Medium | Nothing |
| **N2** | Delete `src/core/simd.rs` — replace call sites with `rustynum_core::simd` | Small | Nothing |
| **N3** | Wire CogRecordView<'a> into `hydrate_nodes()` for zero-copy Lance → BindSpace | Medium | Nothing |
| **N4** | Expose `handle_request_body` from crewai-rust server module | Small | crewai-rust change |
| **N5** | Expose `handle_api_post/get` from n8n-grpc | Small | n8n-rs change |
| **N6** | Add rustynum-holo `Overlay`/`DeltaLayer`/`LayerStack` to BindSpace | Medium | Nothing |
| **N7** | Wire rustynum-clam `ClamTree::build()` for indexed sub-linear search in BindSpace | Medium | Nothing |
| **N8** | Wire rustynum-holo `LodIndex`/`lod_knn_search()` for hierarchical LOD pruning | Medium | N7 |
| **N9** | Periodic `flush_aged()` timer in server.rs (every 5-30 min) | Small | Nothing |
| **N10** | BERT embedding model for blood-brain barrier inbound translation | Large | External model |

### DONE

- [x] 2026-02-27: Added Miri job to ci-master.yml (5 min timeout)
- [x] 2026-02-27: Documented Rustynum Acceleration Contract
- [x] 2026-02-27: vendor/rustynum submodule wired
- [x] 2026-02-27: Arrow zero-copy working via cascade_scan_4ch + arrow_to_flat_bytes
- [x] 2026-02-27: portable_simd → std::arch (870 occurrences, 622 tests pass stable 1.93)
- [x] 2026-02-27: bundle_byte_slices zero-copy entry point (rustynum-rs commit f3d65d0)
- [x] 2026-02-27: rustynum made OBLIGATORY in ladybug-rs (not behind feature gate)
- [x] 2026-02-27: BindSpace parallel ops (parallel_into_node_slices, bulk_write_nodes, parallel_hydrate_nodes)
- [x] 2026-02-27: Age-based hot→cold flush (updated_at, collect_aged, evict_aged, flush_aged → Lance)
- [x] 2026-02-27: rustynum-accel wired into BindSpace (hamming, popcount, dot_i8, bundle, bind, nearest, cascade_search)
- [x] 2026-02-27: rustynum-holo wired into BindSpace (phase bind/unbind, wasserstein, circular, carrier, focus, spatial)
- [x] 2026-02-27: rustynum-clam wired into BindSpace (batch_hamming, clam_top_k)
- [x] 2026-02-27: lance_zero_copy wired into BindSpace (write-through, unified storage, adjacency index)
- [x] 2026-02-27: SubstrateView for BindSpace (crewai-rust bridge) implemented
- [x] 2026-02-27: 10-layer cognitive stack verified intact (L1-L10 with 7-wave processing)
- [x] 2026-02-27: Tier 4 (n8n-rs/crewai-rust) dispatch verified intact (dispatch_crew/n8n_post/get)

---

## Contact

**GitHub**: https://github.com/AdaWorldAPI/ladybug-rs

---

**LADYBUG: Where all queries become one.**
