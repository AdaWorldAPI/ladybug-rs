# Session Handover: CogRecord Refactoring

> **Date**: 2026-02-06
> **Branch**: `claude/code-review-SMMuY`
> **Last commit**: `a28a78a` — integration plan rev 2

---

## What Was Accomplished

### 1. Lance 2.1 API rewrite (`00e0f19`)

Ported `src/storage/lance.rs` from Lance 1.0 to 2.1.0-beta.0 API:
- `Dataset::write()` takes `RecordBatchIterator` (not `&[RecordBatch]`)
- `WriteParams`/`WriteMode` moved to `lance::dataset::write`
- Scanner: `dataset.scan().nearest()` replaces removed `dataset.query()`
- Added `From<lance::Error> for Error` in `src/lib.rs`
- Fixed `database.rs` (`Thought.fingerprint` is `Fingerprint`, not `Option`)
- Fixed `datafusion.rs` lance-gated code
- **Verified**: `cargo check` and `cargo check --features lancedb` both 0 errors
- Backup of old code: `src/storage/lance_v1.rs`, `docs/status_quo/`

### 2. Integration plan (`a28a78a`)

`docs/INTEGRATION_PLAN.md` — aligned with holograph reference design.

---

## What Needs To Be Done: Refactor to 16K CogRecord

### The Design (settled, do not re-debate)

The 256×u64 CogRecord layout from [holograph](https://github.com/AdaWorldAPI/holograph):

```
Words 0-207:   SEMANTIC fingerprint (13,312 bits, 26 AVX-512 iterations)
Words 208-255: METADATA (ANI, NARS, qualia, GEL, DN tree, RL, edges, bloom, graph metrics)
```

**Read holograph docs in this order before touching code:**
1. `docs/00_PROMPT_FOR_LADYBUG_SESSION.md` — bootstrap context
2. `docs/01_THE_256_WORD_SOLUTION.md` — why 256 words (sigma=64, SIMD perfect)
3. `docs/06_METADATA_REVIEW.md` — complete bit-level layout (46KB, read ALL of it)
4. `docs/07_COMPRESSION_AND_RESONANCE.md` — XOR deltas, dimensional sparsity

**The holograph `src/width_16k/` module has working Rust implementations:**
- `schema.rs` — SchemaSidecar with pack/unpack for all metadata blocks
- `search.rs` — 16K Hamming search with bloom acceleration
- `xor_bubble.rs` — XOR delta chains
- `compat.rs` — 10K→16K conversion (depends on holograph's bitpack module)

### Key Constraints (non-negotiable)

1. **Max 32 edges per node** — co-located in the record, NOT separate table
2. **No floats at query time** — 1024D embeddings are transient calibration only
3. **Hamming distance only** — no cosine, no L2, no ANN index
4. **O(1) addressing** — 8+8 prefix:slot array indexing stays
5. **Zero copy** — Lance `FixedSizeBinary(2048)` → pointer cast → `&[u64; 256]`
6. **No CSR needed** — edges co-located, 32 max, graph walk reads full record anyway
7. **DO NOT import holograph code** — refactor existing ladybug-rs code to match the patterns

### Refactoring Steps

**Phase 1: CogRecord struct** — `src/core/cog_record.rs` (NEW)
```rust
#[repr(C, align(64))]
pub struct CogRecord {
    data: [u64; 256],
}
```
With accessor methods for each compartment. Use holograph `schema.rs` as reference
for the SchemaSidecar pattern (pack/unpack into words 208-255). Do NOT copy the file.
Write it fresh using ladybug-rs conventions.

**Phase 2: Expand Fingerprint** — `src/core/fingerprint.rs`
- Change `data: [u64; 157]` to `data: [u64; 256]` (or make CogRecord the container)
- Hamming distance computed on words 0-207 only (semantic zone)
- Update `FINGERPRINT_U64`, `FINGERPRINT_BITS` constants
- Update `src/core/mod.rs`: `DIM`, `DIM_U64`, `LAST_MASK`

**Phase 3: Replace BindNode** — `src/storage/bind_space.rs`
- Current: `BindNode { fingerprint: [u64; 156], label, qidx, parent, depth, rung, sigma, ... }`
- Target: `CogRecord` where label_hash, qidx, parent, depth, rung, sigma are bitpacked in words 208-255
- Storage: `Vec<Box<[Option<CogRecord>; 256]>>` chunks (same pattern, bigger unit)

**Phase 4: Update lance.rs schema**
- Drop `NodeRecord`, `EdgeRecord` structs
- Drop `vector_search()` (no float vectors)
- Single column: `FixedSizeBinary(2048)` for the full CogRecord
- Keep `hamming_search()` but operate on words 0-207

**Phase 5: Wire CogRedis** — `src/storage/cog_redis.rs`
- DN.SET writes CogRecord instead of BindNode
- Edge commands write into words 219-235 of the record
- Metadata commands update specific words via XOR delta

---

## Codebase State

```
cargo check                     → 0 errors ✓
cargo check --features lancedb  → 0 errors ✓
git status                      → clean
```

### Files that will change during refactoring

| File | Change |
|------|--------|
| `src/core/fingerprint.rs` | Expand to 256 words, Hamming on 0-207 |
| `src/core/mod.rs` | Update DIM constants |
| `src/core/simd.rs` | Update Hamming for 208 words (26 AVX-512 iter) |
| `src/storage/bind_space.rs` | Replace BindNode with CogRecord |
| `src/storage/cog_redis.rs` | Update DN.*/CAM.*/DAG.* for CogRecord |
| `src/storage/xor_dag.rs` | XOR delta on CogRecord |
| `src/storage/lance.rs` | FixedSizeBinary(2048), drop float schemas |
| `src/storage/unified_engine.rs` | Update ArrowZeroCopy for 2KB records |

### Files that should NOT change

| File | Why |
|------|-----|
| `src/flight/` | Already works, transport layer is format-agnostic |
| `src/search/hdr_cascade.rs` | Already operates on fingerprints, will adapt to wider ones |
| `src/learning/cam_ops.rs` | CAM operations are addresses, layout-independent |

---

## Vendor/Lance State

- `vendor/lance/` — git submodule, AdaWorldAPI fork, at commit `8fdbd5aa`
- Fork is on pre-2.0.0 development line (Cargo.toml says `2.1.0-beta.0`)
- Upstream has `v2.0.0` (stable) and `v3.0.0-beta.2` (active development)
- `[patch.crates-io]` in Cargo.toml routes 14 lance sub-crates to vendor
- Compiles cleanly with `--features lancedb`

### To rebase fork onto v2.0.0 (when ready):
```bash
cd vendor/lance
git fetch upstream
git checkout -b ladybug/main v2.0.0
# make ladybug-specific changes
git push origin ladybug/main
cd ../..
git add vendor/lance
git commit -m "vendor: rebase lance fork onto v2.0.0"
```

---

## Critical Reminders

- **Read holograph docs BEFORE coding.** Every question you might have is answered there.
- **Refactor, don't import.** Write CogRecord fresh in ladybug-rs style.
- **Edges are co-located.** verb:u8 + target:u8 = 16 bits per edge, 4 per u64 word.
- **No 1024D at runtime.** Jina embeddings are cached transiently during calibration only.
- **sigma = 64.** Integer thresholds. No floating point in distance gates.
- **32 AVX-512 iterations for full record, 26 for semantic zone.** Zero remainder.
