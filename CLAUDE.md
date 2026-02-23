# CLAUDE.md — Ladybug-RS

> **Last Updated**: 2026-02-04
> **Branch**: claude/code-review-X0tu2
> **Status**: Flight + CogRedis wired, Arrow zero-copy WORKING

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
| `bind_space.rs` | 8+8 addressing, O(1) arrays | None | ✅ Working |
| `cog_redis.rs` | Redis syntax (DN.*, CAM.*, DAG.*) | bind_space | ✅ Working |
| `unified_engine.rs` | ACID + CSR + MVCC + ArrowZeroCopy | All below | ✅ Working |
| `xor_dag.rs` | ACID transactions, XOR parity | bind_space | ✅ Working |
| `lance_zero_copy/` | **Pure Arrow buffers** | arrow_* only | ✅ Working |
| `lance.rs` | Direct Lance Dataset API | lance crate | ❌ API mismatch |
| `database.rs` | Abstraction over lance.rs | lance.rs | ❌ Blocked |

### The Key Insight

**`lance_zero_copy/` is NOT the same as `lance.rs`.**

- `lance_zero_copy/` = Pure Arrow integration, NO lance crate dependency
- `lance.rs` = Direct Lance Dataset API, REQUIRES lance crate

The owner already built the Arrow zero-copy layer. It feeds data back to BindSpace through `ArrowZeroCopy`, `FingerprintBuffer`, `LanceView`, and `ZeroCopyBubbler`. This is the production path.

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
│                 │  0x0C-0x0F: Reserved                                      │
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
**Last updated**: 2026-02-04
**Rust**: 1.93 (edition 2024)
**DataFusion**: 51 (DF 52 upgrade path documented)
**Arrow**: 57.x / arrow-flight 57 / tonic 0.14

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

### ❌ Known Issues

| Issue | Location | Status |
|-------|----------|--------|
| lance.rs API mismatch | storage/lance.rs | Vendor=2.1, Cargo=1.0 |
| 10 test failures | Various | See test section below |

### Recent Commits

```
f3f455f feat: Wire Flight gRPC, CogRedis DN commands, and fix deployment features
15fb4d9 feat: Add unified storage engine with ACID, CSR, DAG, work stealing
db22a5e fix: Replace JSON with Arrow IPC as default serialization format
a11a0f4 feat: Add XOR DAG storage with ACID transactions and parity protection
4a4af54 feat: Wire DN tree as CogRedis addresses for Redis-syntax tree traversal
```

---

## Key Files

```
src/bin/
├── server.rs           # HTTP server (port 8080)
└── flight_server.rs    # Arrow Flight gRPC (port 50051)

src/storage/
├── bind_space.rs       # Universal DTO (8+8 addressing) ← THE CORE
├── cog_redis.rs        # Redis syntax adapter ← DN.*, CAM.*, DAG.*
├── unified_engine.rs   # Composes all storage features
├── xor_dag.rs          # ACID + XOR parity
├── lance_zero_copy/    # Pure Arrow integration (NO lance crate!) ← WORKS
│   └── mod.rs          # ArrowZeroCopy, FingerprintBuffer, LanceView
├── lance.rs            # Direct Lance API (BROKEN - API mismatch)
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

## Contact

**GitHub**: https://github.com/AdaWorldAPI/ladybug-rs

---

**🦔 LADYBUG: Where all queries become one.**
