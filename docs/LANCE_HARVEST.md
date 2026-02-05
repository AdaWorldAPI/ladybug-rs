# Lance / LanceDB Architecture Harvest for Ladybug-RS

> **Date**: 2026-02-05
> **Branch**: claude/code-review-SMMuY
> **Repos scanned**:
> - [AdaWorldAPI/ladybug-rs](https://github.com/AdaWorldAPI/ladybug-rs) (core)
> - [AdaWorldAPI/lance](https://github.com/AdaWorldAPI/lance) (vendor, v2.1.0-beta.0)
> - [AdaWorldAPI/lancedb](https://github.com/AdaWorldAPI/lancedb) (vendor, v0.24.1)

---

## Invariants (do not break these)

| # | Rule |
|---|------|
| I1 | `Addr(u16)` = `(prefix << 8) \| slot` is the primary key everywhere |
| I2 | BindSpace `[Option<BindNode>; 256]` arrays stay O(1), no HashMap |
| I3 | SoA canonical storage: structure(512B) + fingerprint(1536B) as separate FixedSizeBinary columns |
| I4 | DataFusion `TableProvider` / `ExecutionPlan` stay custom (not replaced by Lance scanner) |
| I5 | Arrow Flight server stays as-is; Lance is a persistence backend, not a transport |
| I6 | No float in the CogRecord 256-word path; fixed-point only |
| I7 | XOR delta compression continues to own the hot-tier versioning story |

---

## Deliverable 1 — API Surface Map

### What Lance/LanceDB offers vs. what ladybug-rs already has

| Capability | Lance/LanceDB module | Ladybug-RS equivalent | Harvest? |
|---|---|---|---|
| **Columnar persistence** | `lance::Dataset::write()`, `InsertBuilder` | None (BindSpace is in-memory only) | **YES** |
| **Fragment management** | `lance::dataset::fragment` (`FileFragment`, `Fragment`, deletion vectors) | None | **YES** |
| **Versioned manifests** | `lance::io::commit` (`Manifest`, `Transaction`, `CommitHandler`) | `xor_dag.rs` (XOR parity, epoch) | **YES** (complement, not replace) |
| **Object-store I/O** | `lance-io::ObjectStore` (local, S3, GCS, Azure) | None (local filesystem only) | **YES** |
| **Columnar encoding** | `lance-encoding` (Plain, Bitpacking, RLE, Dictionary, FSST, Zstd) | None | **YES** (Bitpacking for structure, Plain for fingerprint) |
| **Page/metadata cache** | `lance::session::Session` (LRU, 6 GiB index + 1 GiB metadata defaults) | None | **YES** |
| **Scalar indices** | `lance-index` (BTree, Bitmap, Inverted) | HDR cascade + popcount pre-filter | **MAYBE** (BTree on `addr` for cold scan) |
| **Vector indices** | `lance-index` (IVF-PQ, IVF-HNSW-SQ, etc.) | HDR cascade (scent L0 → popcount L1 → sketch L2 → Hamming L3) | **NO** (HDR is purpose-built for 10K-bit binary fingerprints) |
| **Compaction** | `lance::dataset::optimize` (`compact_files`, deletion materialization) | None | **YES** |
| **DataFusion scanner** | `lance::dataset::scanner::Scanner` | `FingerprintTableProvider` + `BindSpaceScan` | **NO** (we keep our custom exec plans) |
| **Table CRUD** | `lancedb::table` (`add`, `delete`, `update`, `merge_insert`) | CogRedis DN.SET/GET, link/unlink | **NO** (different addressing model) |
| **Embedding pipeline** | `lancedb::embeddings` (`EmbeddingFunction`, `EmbeddingRegistry`) | None | **NO** (no float; fingerprints are VSA, not embeddings) |
| **Remote/cloud client** | `lancedb::remote` (HTTP, retry, namespaces) | Arrow Flight server | **NO** (Flight is our transport) |
| **Hybrid search / reranking** | `lancedb::rerankers` (RRF) | HDR cascade + DataFusion UDFs | **NO** |
| **Schema evolution** | `lance::dataset::schema_evolution` (`ColumnAlteration`, `NewColumnTransform`) | None | **LATER** (useful for CogRecord migrations) |

### Modules to harvest (ranked by value)

| Priority | Lance module | Harvest target | Why |
|---|---|---|---|
| **P0** | `lance-io::ObjectStore` + `lance-io::scheduler` | `src/storage/persistence/io.rs` | S3/GCS/local abstraction; ladybug-rs has zero persistence today |
| **P0** | `lance::Dataset::write()` + `InsertBuilder` | `src/storage/persistence/writer.rs` | Write SoA columns (addr, structure, fingerprint) to Lance format on disk |
| **P0** | `lance::dataset::fragment` | `src/storage/persistence/fragment.rs` | Fragment metadata, deletion vectors, row tracking |
| **P1** | `lance::io::commit` + `Manifest` | `src/storage/persistence/manifest.rs` | Versioned manifest for cold tier; complement XorDag for hot tier |
| **P1** | `lance-encoding` (Bitpacking, Plain) | Encoding config only | Bitpacking for structure column (lots of small integers), Plain for fingerprint (high entropy) |
| **P1** | `lance::session::Session` | `src/storage/persistence/cache.rs` | LRU page cache for cold-tier reads; avoids re-reading fragments |
| **P2** | `lance::dataset::optimize::compact_files` | `src/storage/persistence/compaction.rs` | Background compaction of cold fragments; deletion materialization |
| **P2** | `lance-index::scalar::BTree` | Optional cold-tier index | BTree on `addr(u16)` for point lookups into cold fragments |
| **P3** | `lance::dataset::schema_evolution` | Future migration tool | Add/drop columns when CogRecord schema evolves |

---

## Deliverable 2 — Minimal Integration Design

### 2-Tier Hot/Cold Architecture

```
                         WRITE PATH
                         ══════════
                              │
            DN.SET / link / write_in_txn
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    HOT TIER (in-memory)                      │
│                                                              │
│   BindSpace  ──────────────────────────────────────────┐    │
│   [Option<BindNode>; 256] × 256 prefixes               │    │
│   O(1) read/write, 3-5 cycles                          │    │
│                                                         │    │
│   XorDag ── ACID + XOR parity ── epoch versioning      │    │
│   MVCC   ── concurrent reads                           │    │
│   CSR    ── zero-copy edge traversal                   │    │
│                                                         │    │
│   ArrowZeroCopy ── FingerprintBuffer (Arrow Buffer)    │    │
│                                                         │    │
│   ┌────────────────────────────────────────────────┐   │    │
│   │ CogColumns (SoA canonical)                     │   │    │
│   │   addr:        Arc<[u16]>       ← column 0     │   │    │
│   │   structure:   Arc<[u8]>  n×512  (64B-aligned) │   │    │
│   │   fingerprint: Arc<[u8]>  n×1536 (64B-aligned) │   │    │
│   │   parent_addr: Arc<[u16]>                      │   │    │
│   │   first_child: Arc<[u16]>                      │   │    │
│   │   context_id:  Arc<[u16]>                      │   │    │
│   │   rung:        Arc<[u8]>                       │   │    │
│   │   ...                                          │   │    │
│   └─────────────────────┬──────────────────────────┘   │    │
│                         │                               │    │
└─────────────────────────┼───────────────────────────────┘    │
                          │                                    │
              FLUSH (background, periodic or on commit)        │
                          │                                    │
                          ▼                                    │
┌─────────────────────────────────────────────────────────────┐
│                    COLD TIER (Lance format on disk/S3)       │
│                                                              │
│   LancePersistence {                                        │
│     dataset: lance::Dataset,        ← versioned             │
│     session: lance::Session,        ← shared cache          │
│     io: lance_io::ObjectStore,      ← local/S3/GCS         │
│   }                                                          │
│                                                              │
│   On-disk layout:                                            │
│     cogrecords.lance/                                        │
│       _versions/                                             │
│         000000000000000001.manifest  (protobuf)              │
│         000000000000000002.manifest                          │
│       data/                                                  │
│         frag-0.lance   (fragment 0: rows 0..N)               │
│         frag-1.lance   (fragment 1: rows N..M)               │
│                                                              │
│   Arrow schema on disk:                                      │
│     addr:          UInt16                                    │
│     structure:     FixedSizeBinary(512)                      │
│     fingerprint:   FixedSizeBinary(1536)                     │
│     parent_addr:   UInt16                                    │
│     first_child:   UInt16                                    │
│     next_sibling:  UInt16                                    │
│     prev_sibling:  UInt16                                    │
│     context_id:    UInt16                                    │
│     rung:          UInt8                                     │
│     depth:         UInt8                                     │
│     access_count:  UInt16                                    │
│     updated_at:    UInt64                                    │
│     label:         Utf8  (nullable)                          │
│                                                              │
│   Encoding strategy:                                         │
│     structure    → Bitpacking + Zstd  (many small ints)      │
│     fingerprint  → Plain (high entropy, incompressible)      │
│     addr/parent  → Bitpacking (16-bit, dense)                │
│     label        → Dictionary + FSST                         │
│                                                              │
│   Deletion vectors: RoaringBitmap per fragment               │
│   Versioning: Manifest per commit (Lance native)             │
│                                                              │
└──────────────────────────┬──────────────────────────────────┘
                           │
               READ PATH (cold miss)
               ═══════════
                           │
            BindSpace.read(addr) returns None
                           │
                           ▼
            LancePersistence.load_addr(addr)
              → Scanner.filter("addr = {addr}")
              → RecordBatch (1 row)
              → Hydrate into BindNode
              → Insert into BindSpace HOT tier
              → Return &BindNode
```

### Key Design Decisions

**D1: Lance as append-only journal, XorDag stays for hot-tier ACID.**

Lance's `Transaction` and `Manifest` handle cold-tier versioning. XorDag continues to own the hot-tier write path with epoch-based conflict detection. On flush, a Lance `Append` transaction writes dirty BindNodes to a new fragment. The XorDag parity blocks are *not* stored in Lance — they live in hot memory and are recomputed on startup from the Lance snapshot.

**D2: Flush is a background operation, never blocks reads.**

A dedicated flush task periodically (or on explicit commit) snapshots dirty addresses from BindSpace, converts to SoA `RecordBatch`, and calls `InsertBuilder::execute_uncommitted()` followed by `CommitBuilder::execute()`. Reads continue hitting hot-tier arrays. If BindSpace evicts a node (LRU or TTL), the cold tier has the last-flushed version.

**D3: Cold reads go through Lance Scanner with `addr` filter.**

When `BindSpace.read(addr)` returns `None` (address not in HOT tier), the persistence layer does:
```rust
dataset.scan()
    .project(&["addr", "structure", "fingerprint", "parent_addr", ...])
    .filter(&format!("addr = {}", addr.0))
    .try_into_stream().await
```
This returns at most one row. The row is hydrated into a `BindNode` and inserted into hot tier. Future reads hit the array.

**D4: Bulk load on startup reads the latest Lance manifest.**

On server boot, `LancePersistence::load_latest()` reads the full dataset and populates BindSpace. For 65K records at 2KB each, this is ~128 MB — a single sequential read, ~200ms on SSD, ~2s from S3.

**D5: Fragment strategy: one fragment per flush epoch.**

Each flush creates one fragment containing only the dirty addresses. Over time, compaction merges small fragments. Target: 65K rows per fragment (the entire address space fits in one fragment). Compaction runs when fragment count exceeds a threshold (e.g., 64 fragments → compact to 1).

**D6: The `lance` crate is used directly (not `lancedb`).**

LanceDB adds table management, embedding pipelines, and a remote client — none of which ladybug-rs needs. The core `lance` crate provides `Dataset`, `Fragment`, `Manifest`, `Session`, and `ObjectStore`, which is exactly the persistence layer we need. This avoids pulling in LanceDB's HTTP client, embedding registry, and query builders.

### New Module Structure

```
src/storage/
├── bind_space.rs           # (existing) HOT tier, unchanged
├── unified_engine.rs       # (existing) add flush_to_cold() method
├── xor_dag.rs              # (existing) HOT tier ACID, unchanged
├── lance_zero_copy/        # (existing) Arrow buffer bridge, unchanged
│   └── mod.rs
├── persistence/            # NEW — cold tier via Lance
│   ├── mod.rs              # LancePersistence struct
│   ├── writer.rs           # flush_dirty() → Lance Append transaction
│   ├── reader.rs           # load_addr() / load_all() from Lance
│   ├── manifest.rs         # Manifest helpers, version listing
│   ├── cache.rs            # Session wrapper, cache configuration
│   └── compaction.rs       # Background compaction task
├── lance.rs                # (existing, DEPRECATED) replaced by persistence/
└── database.rs             # (existing, DEPRECATED) replaced by persistence/
```

### Data Flow: Flush Dirty Addresses

```rust
// In persistence/writer.rs

pub async fn flush_dirty(
    bind_space: &BindSpace,
    dirty_addrs: &[Addr],
    dataset: &mut Dataset,
) -> Result<u64> {
    // 1. Collect dirty nodes into SoA columns
    let n = dirty_addrs.len();
    let mut addrs = Vec::with_capacity(n);
    let mut structures = Vec::with_capacity(n * 512);
    let mut fingerprints = Vec::with_capacity(n * 1536);
    let mut parent_addrs = Vec::with_capacity(n);
    // ... other columns

    for &addr in dirty_addrs {
        if let Some(node) = bind_space.read(addr) {
            addrs.push(addr.0);
            // Pack C0-C7 (structure, 512 bytes)
            structures.extend_from_slice(&node_to_structure_bytes(node));
            // Pack C8-C31 (fingerprint, 1536 bytes)
            fingerprints.extend_from_slice(&node_to_fingerprint_bytes(node));
            parent_addrs.push(node.parent.map(|a| a.0).unwrap_or(0));
            // ...
        }
    }

    // 2. Build RecordBatch (zero-copy where possible)
    let batch = RecordBatch::try_new(cog_schema(), vec![
        Arc::new(UInt16Array::from(addrs)),
        Arc::new(FixedSizeBinaryArray::try_from_iter(
            structures.chunks(512).map(|c| c.to_vec())
        )?),
        Arc::new(FixedSizeBinaryArray::try_from_iter(
            fingerprints.chunks(1536).map(|c| c.to_vec())
        )?),
        Arc::new(UInt16Array::from(parent_addrs)),
        // ...
    ])?;

    // 3. Append as new fragment (uncommitted)
    let txn = InsertBuilder::new(dataset)
        .execute_uncommitted(vec![batch])
        .await?;

    // 4. Commit atomically
    let committed = CommitBuilder::new(dataset)
        .execute(txn)
        .await?;

    Ok(committed.manifest().version)
}
```

### Data Flow: Cold Read

```rust
// In persistence/reader.rs

pub async fn load_addr(
    dataset: &Dataset,
    addr: Addr,
) -> Result<Option<BindNode>> {
    let stream = dataset.scan()
        .project(&["addr", "structure", "fingerprint",
                    "parent_addr", "first_child", "next_sibling",
                    "prev_sibling", "context_id", "rung", "depth",
                    "access_count", "updated_at", "label"])
        .filter(&format!("addr = {}", addr.0))?
        .with_batch_size(1)
        .try_into_stream()
        .await?;

    use futures::StreamExt;
    let mut stream = stream;
    if let Some(batch) = stream.next().await {
        let batch = batch?;
        if batch.num_rows() > 0 {
            return Ok(Some(bind_node_from_batch(&batch, 0)?));
        }
    }
    Ok(None)
}

pub async fn load_all(dataset: &Dataset) -> Result<Vec<(Addr, BindNode)>> {
    // Full scan for startup hydration
    let stream = dataset.scan()
        .with_batch_size(8192)
        .try_into_stream()
        .await?;

    use futures::StreamExt;
    let mut results = Vec::with_capacity(65536);
    let mut stream = stream;
    while let Some(batch) = stream.next().await {
        let batch = batch?;
        for i in 0..batch.num_rows() {
            let addr = Addr(batch.column_by_name("addr")
                .unwrap().as_any()
                .downcast_ref::<UInt16Array>().unwrap()
                .value(i));
            let node = bind_node_from_batch(&batch, i)?;
            results.push((addr, node));
        }
    }
    Ok(results)
}
```

---

## Deliverable 3 — Concrete First PR Plan

### PR Title: `feat: Add Lance cold-tier persistence for BindSpace`

### File-by-file plan

| # | File | Action | Lines (est.) | Description |
|---|------|--------|-------------|-------------|
| 1 | `Cargo.toml` | **Edit** | +10 | Add `[patch.crates-io] lance = { path = "vendor/lance/rust/lance" }`; change `lance = "1.0"` → `lance = "2.1"`; add `persistence` feature flag |
| 2 | `src/storage/persistence/mod.rs` | **Create** | ~80 | `LancePersistence` struct: `dataset: Arc<RwLock<Dataset>>`, `session: Arc<Session>`, `io: Arc<ObjectStore>`; `open()`, `create()`, `flush()`, `load()` |
| 3 | `src/storage/persistence/schema.rs` | **Create** | ~60 | `cog_schema() -> Schema` returning the SoA Arrow schema (addr, structure, fingerprint, tree columns, metadata columns) |
| 4 | `src/storage/persistence/writer.rs` | **Create** | ~120 | `flush_dirty(bind_space, dirty_addrs, dataset)` as shown above; `node_to_structure_bytes()`, `node_to_fingerprint_bytes()` |
| 5 | `src/storage/persistence/reader.rs` | **Create** | ~100 | `load_addr(dataset, addr)`, `load_all(dataset)`, `bind_node_from_batch()` |
| 6 | `src/storage/persistence/cache.rs` | **Create** | ~40 | `create_session(index_cache_mb, metadata_cache_mb) -> Session` wrapper |
| 7 | `src/storage/persistence/manifest.rs` | **Create** | ~50 | `list_versions(dataset)`, `checkout_version(dataset, v)` helpers |
| 8 | `src/storage/persistence/compaction.rs` | **Create** | ~60 | `compact_if_needed(dataset, max_fragments)` background task |
| 9 | `src/storage/mod.rs` | **Edit** | +5 | Add `pub mod persistence;` |
| 10 | `src/storage/unified_engine.rs` | **Edit** | +30 | Add `persistence: Option<Arc<LancePersistence>>` field; add `flush_to_cold()` and `load_from_cold(addr)` methods |
| 11 | `src/storage/bind_space.rs` | **Edit** | +10 | Add `dirty: HashSet<Addr>` field; mark addresses as dirty on `write()` / `write_at()` |
| 12 | **Tests** | **Create** | ~150 | `tests/persistence_test.rs` — roundtrip: write to BindSpace → flush → clear hot → read from cold → verify equality |

**Total**: ~715 new lines + ~55 edited lines

### Dependency chain (build order)

```
schema.rs          (no deps)
    ↓
cache.rs           (depends on lance::Session)
    ↓
writer.rs          (depends on schema.rs, lance::InsertBuilder)
reader.rs          (depends on schema.rs, lance::Scanner)
    ↓
manifest.rs        (depends on lance::Dataset)
compaction.rs      (depends on lance::optimize)
    ↓
mod.rs             (composes all above)
    ↓
unified_engine.rs  (add persistence field)
bind_space.rs      (add dirty tracking)
```

### What this PR does NOT touch

- `lance_zero_copy/mod.rs` — stays as-is; FingerprintBuffer continues to work for in-memory Arrow buffers
- `lance.rs` — not modified; deprecated in favor of `persistence/`
- `database.rs` — not modified; deprecated in favor of `persistence/`
- `fingerprint_table.rs` — TableProvider stays unchanged; still reads from BindSpace
- `flight/` — Flight server stays unchanged; still serves from BindSpace
- `cog_redis.rs` — CogRedis stays unchanged; still writes to BindSpace
- `xor_dag.rs` — XorDag stays unchanged; still owns hot-tier ACID

### Feature flag gating

```toml
[features]
persistence = ["lance"]   # NEW
lancedb = ["lance"]       # EXISTING (broken, unchanged)
```

All new code in `src/storage/persistence/` is gated behind `#[cfg(feature = "persistence")]`. Default features remain `["simd", "parallel"]`. The Dockerfile adds `persistence` when cold storage is desired:

```dockerfile
ARG FEATURES="simd,parallel,flight,persistence"
```

---

## Deliverable 4 — Risk & Compatibility Checklist

### R1: Lance crate version mismatch (CRITICAL)

| Item | Detail |
|---|---|
| **Risk** | `Cargo.toml` says `lance = "1.0"` but vendor has `2.1.0-beta.0`. Adding `[patch.crates-io]` is required. |
| **Mitigation** | First line of the PR: add `[patch.crates-io] lance = { path = "vendor/lance/rust/lance" }`. Verify `cargo check --features persistence` compiles. |
| **Status** | Blocks everything. Must be resolved first. |

### R2: Lance 2.1 API differences from 1.0

| Item | Detail |
|---|---|
| **Risk** | `Dataset::write()` signature changed; `Scanner` API changed; `WriteParams` fields differ. |
| **Mitigation** | The new `persistence/` module writes against the 2.1 API directly. The old `lance.rs` (which targets 1.0) is left as-is and deprecated — it already doesn't compile. |
| **Impact** | None on existing code; only new code uses 2.1 API. |

### R3: Arrow version alignment

| Item | Detail |
|---|---|
| **Risk** | Ladybug-rs uses Arrow 57. Lance 2.1.0-beta.0 must also use Arrow 57 (or compatible). |
| **Mitigation** | Check `vendor/lance/rust/lance/Cargo.toml` for Arrow version. If mismatch, patch lance to match. The AdaWorldAPI fork may already be aligned. |
| **Verification** | `cargo tree -i arrow-schema --features persistence` must show a single version. |

### R4: DataFusion version conflict

| Item | Detail |
|---|---|
| **Risk** | Lance 2.1 may pull DataFusion 52+. Ladybug-rs is on DataFusion 51 (liblzma conflict). |
| **Mitigation** | Option A: Lance's DataFusion integration is optional; disable it via feature flags and use Lance only for I/O + encoding (no Lance scanner). Option B: Use the existing `[patch.crates-io] datafusion` fork that fixes liblzma. |
| **Verification** | `cargo tree -i datafusion --features persistence` must show a single version. |

### R5: Build time increase

| Item | Detail |
|---|---|
| **Risk** | Lance 2.1 is a large crate (~100K lines). Full rebuild could add 2-5 minutes. |
| **Mitigation** | Feature-gate behind `persistence`. CI caches `target/` between runs. Only build with `persistence` when deploying to production. |

### R6: Binary size increase

| Item | Detail |
|---|---|
| **Risk** | Lance + protobuf (prost) + object_store adds ~5-10 MB to release binary. |
| **Mitigation** | Acceptable for server binary. Not included in WASM/embedded builds (feature-gated). `lto = "fat"` in release profile already strips dead code. |

### R7: Addr(u16) as primary key vs Lance row IDs

| Item | Detail |
|---|---|
| **Risk** | Lance uses its own `_rowid` (u32 or u64) as internal primary key. We use `addr` (u16). These could collide if Lance reassigns row IDs during compaction. |
| **Mitigation** | Never rely on Lance `_rowid`. Always filter by `addr` column. Enable `stable_row_ids` in WriteParams as defense-in-depth. The `addr` column has at most 65,536 unique values — duplicates from multiple flushes are resolved by taking the latest fragment's version. |
| **Alternative** | Add a `version: u64` column; on read, `SELECT * WHERE addr = X ORDER BY version DESC LIMIT 1`. |

### R8: FixedSizeBinary(512) and FixedSizeBinary(1536) encoding

| Item | Detail |
|---|---|
| **Risk** | Lance's default encoding for FixedSizeBinary may not be optimal. Structure column (512B, many small integers XOR'd) compresses well; fingerprint column (1536B, high entropy) does not. |
| **Mitigation** | Set per-column encoding in WriteParams: Bitpacking + Zstd for structure, Plain (no compression) for fingerprint. Benchmark both with `criterion`. |

### R9: Concurrent flush + read race

| Item | Detail |
|---|---|
| **Risk** | If a flush commits a new manifest while a cold read is using the old manifest, the reader may see stale data. |
| **Mitigation** | Lance handles this via MVCC manifests — readers hold a reference to their manifest version. New commits don't invalidate existing readers. This is a built-in Lance guarantee. |

### R10: Object store latency for S3/GCS

| Item | Detail |
|---|---|
| **Risk** | Cold reads from S3 add 50-200ms latency per request. This is unacceptable for the 3-5 cycle hot path. |
| **Mitigation** | Cold reads are ONLY for cache misses (address not in BindSpace). On startup, `load_all()` hydrates the full 128 MB dataset in one sequential read (~2s from S3). During normal operation, hot-tier serves all reads. S3 latency only matters for cold starts and evicted addresses. |
| **Future** | The `Session` cache layer (LRU, configurable GiB) caches Lance pages. Repeated cold reads to the same fragment hit the page cache, not the network. |

### R11: Existing `lance.rs` and `database.rs` deprecation

| Item | Detail |
|---|---|
| **Risk** | The existing `lance.rs` (String IDs, Utf8 keys, float embeddings) and `database.rs` modules use a fundamentally different data model than BindSpace (Addr, u16 keys, integer fingerprints). |
| **Mitigation** | Mark both as `#[deprecated]`. Do not modify them. The new `persistence/` module is the replacement. Remove `lance.rs` and `database.rs` in a follow-up PR once `persistence/` is stable. |

### R12: Lance sub-crate dependency tree

| Item | Detail |
|---|---|
| **Risk** | `lance` (the main crate) pulls in `lance-file`, `lance-io`, `lance-encoding`, `lance-table`, `lance-index`, `lance-core`, `lance-linalg`, and others. Some of these pull in `tantivy` (full-text search), `roaring`, `zstd`, etc. |
| **Mitigation** | We only need: `lance-core`, `lance-io`, `lance-file`, `lance-table`, `lance-encoding`. Consider depending on sub-crates directly instead of the umbrella `lance` crate to minimize dependency tree. Test with `cargo tree --features persistence \| wc -l`. |
| **Alternative** | If the dependency tree is too large, use only `lance-io` + `lance-file` for raw I/O and write our own minimal manifest format using protobuf. This loses compaction and versioning but keeps the binary small. |

---

## Summary Matrix

| Deliverable | Status | Key takeaway |
|---|---|---|
| **D1: API Surface Map** | Complete | 9 modules to harvest (P0-P3), 6 modules rejected with reasons |
| **D2: Integration Design** | Complete | 2-tier hot/cold; Lance as append-only journal; BindSpace stays authoritative |
| **D3: First PR Plan** | Complete | 12 files, ~715 new lines, feature-gated behind `persistence` |
| **D4: Risk Checklist** | Complete | 12 risks identified; R1 (version mismatch) is the only blocker |

---

## Appendix A: Lance 2.1 API Quick Reference

```rust
// Open existing dataset
let dataset = Dataset::open("path/to/data.lance").await?;

// Write new dataset
let dataset = Dataset::write(
    reader,                    // impl RecordBatchReader
    "path/to/data.lance",
    Some(WriteParams {
        mode: WriteMode::Append,
        max_rows_per_file: 65536,
        ..Default::default()
    }),
).await?;

// Scan with filter
let stream = dataset.scan()
    .project(&["addr", "structure", "fingerprint"])
    .filter("addr = 32834")?
    .with_batch_size(1)
    .try_into_stream()
    .await?;

// Compact fragments
compact_files(&mut dataset, CompactionOptions {
    target_rows_per_fragment: 65536,
    materialize_deletions: true,
    ..Default::default()
}, None).await?;

// Version listing
let versions = dataset.list_versions().await?;
let old = dataset.checkout_version(versions[0].version).await?;

// Session with caching
let session = Session::new(
    256 * 1024 * 1024,     // 256 MB index cache
    64 * 1024 * 1024,      // 64 MB metadata cache
    Arc::new(ObjectStoreRegistry::default()),
);
```

## Appendix B: Current lance.rs vs. New persistence/ Comparison

| Aspect | `lance.rs` (current, broken) | `persistence/` (proposed) |
|---|---|---|
| Primary key | `id: String` (Utf8) | `addr: u16` (UInt16) |
| Fingerprint | `FixedSizeBinary(1250)` | `FixedSizeBinary(1536)` (CogRecord C8-C31) |
| Structure | Not stored | `FixedSizeBinary(512)` (CogRecord C0-C7) |
| Embeddings | `FixedSizeList<Float32>(1024)` | Not stored (no float in CogRecord) |
| Tree pointers | Not stored | `parent_addr`, `first_child`, `next_sibling`, `prev_sibling` (UInt16) |
| Lance version | 1.0 (crates.io, won't compile) | 2.1.0-beta.0 (vendor) |
| API style | `Dataset::query().nearest_to()` (1.0) | `Dataset::scan().filter()` (2.1) |
| Data model | Document-oriented (id → record) | Address-oriented (addr → CogRecord columns) |
