# BindSpace Unification Plan

> **Date**: 2026-02-13
> **Status**: PLAN — awaiting implementation
> **Branch for work**: Create fresh branch from `claude/code-review-SMMuY`
> **Rollback**: `git revert` the implementation commit(s); all consolidation files remain untouched on the parent branch

---

## 0. Purpose of This Document

A future session (with no token pressure) should be able to read THIS FILE ONLY and
execute the refactor. Every decision, every file path, every rationale is here.

**The problem**: The "Algorithm Debt Consolidation" (commit `b704a68`) built a
parallel storage universe (DnSpineCache + bridges + duplicate UDFs) instead of
extending the canonical BindSpace model. This creates:

- Two stores for the same tree data (BindSpace vs DnSpineCache)
- Two incompatible fingerprint widths (256 vs 128 words) with lossy conversion
- Two address hashes (`DefaultHasher` on strings vs SplitMix64 on packed u64)
- Three edge representations (BindSpace CSR, CogRedis Vec, Container inline)
- Dead bridge code that nobody calls
- An unprotected write path that bypasses XorDag/WAL
- Structural impossibility of zero-copy between DnSpineCache and ArrowZeroCopy

**The fix**: Harvest the good ideas. Embed them into BindSpace. Delete the parallel universe.

---

## 1. What to Harvest (Do Not Lose These)

### 1a. Worth keeping as-is (already in `src/container/`)

| File | What | Why it's good |
|------|------|---------------|
| `adjacency.rs` — `PackedDn` | 7-level hierarchical address in u64 | Compact, sortable, parent/child O(1) |
| `adjacency.rs` — `InlineEdgeView` | Zero-copy edge reads over Container words | Good format for metadata-embedded edges |
| `search.rs` — all functions | belichtungsmesser, cascade, MexicanHat, hamming_early_exit | Production-ready search over 8K containers |
| `semiring.rs` | Tropical/Boolean/MinPlus semiring MxV | Good algebra, just needs canonical data source |
| `spine.rs` — `SpineCache` | Arena allocator for 1KB Containers | Useful if Container views need scratch space |
| `meta.rs` — `MetaView`/`MetaViewMut` | Structured word-level access to Container 0 | Clean abstraction over layout |
| `geometry.rs` | Container geometry encoding | Good metadata model |

### 1b. Worth keeping as concepts (rewrite into BindSpace)

| Concept | Source file | What to extract |
|---------|-------------|-----------------|
| DN index (PackedDn → Addr) | `dn_spine_cache.rs` | The HashMap mapping, NOT the SpineCache storage |
| Plasticity algorithm | `plasticity.rs` | Prune/consolidate/rename logic, rewritten to operate on Addr |
| cascade_filter UDF | `container_udfs.rs` | 5-level cascade is new; merge into cognitive_udfs.rs at 256-word width |
| belichtung UDF | `container_udfs.rs` | 7-point estimator UDF; merge into cognitive_udfs.rs |
| word_diff UDF | `container_udfs.rs` | Word-level diff; merge into cognitive_udfs.rs |
| mexican_hat UDF | `container_udfs.rs` | Wavelet response UDF; merge into cognitive_udfs.rs |
| DnTree SQL table | `dn_tree_provider.rs` | Concept of querying tree via SQL; rewrite to read from BindSpace |
| Spine/leaf distinction | `dn_spine_cache.rs` | `is_spine` flag per node; add to BindNode |
| Dirty tracking | `dn_spine_cache.rs` | Track modified addresses; add to BindSpace |

### 1c. Delete (parallel implementations, dead code)

| File | Why delete |
|------|-----------|
| `dn_spine_cache.rs` | Parallel store; replaced by DnIndex in BindSpace |
| `addr_bridge.rs` | Incompatible hash; replaced by DnIndex |
| `cog_redis_bridge.rs` | Dead code; CogRedis already talks to BindSpace |
| `csr_bridge.rs` | Dead code; BindSpace already has BitpackedCsr |
| `container_udfs.rs` | Duplicate UDFs at wrong width; merge useful ones into cognitive_udfs |
| `dn_tree_provider.rs` | Copies truncated data; rewrite to read BindSpace directly |

---

## 2. The Options

### Option A: BindSpace Lens (RECOMMENDED)

**Core idea**: BindSpace stays canonical. Container becomes a zero-copy VIEW.
DnSpineCache is replaced by a lightweight DnIndex. All surfaces keep working.

```
BindSpace.nodes[65536]
    │
    ├── BindNode.fingerprint: [u64; 256]    ← canonical data
    │       │
    │       ├── [0..128]  → ContainerRef<'_>  (zero-copy borrow, meta view)
    │       └── [128..256] → ContainerRef<'_>  (zero-copy borrow, content view)
    │
    ├── BindNode.is_spine: bool              ← NEW (from DnSpineCache concept)
    │
    ├── DnIndex                              ← NEW (pure addressing, no storage)
    │       ├── dn_to_addr: HashMap<PackedDn, Addr>
    │       ├── addr_to_dn: Vec<Option<PackedDn>>   (indexed by Addr.0)
    │       └── spine_addrs: BitVec                   (which addrs are spines)
    │
    ├── BitpackedCsr                         ← existing (canonical edges)
    │
    └── dirty: BitVec                        ← NEW (65536 bits, one per addr)
```

**The key insight**: `Container` is `#[repr(C, align(64))]` with `words: [u64; 128]`.
A `&[u64; 128]` can be transmuted to `&Container` at zero cost:

```rust
impl Container {
    /// Zero-copy view into any 128-word slice.
    /// SAFETY: Container is #[repr(C)] — identical layout to [u64; 128].
    pub fn view(words: &[u64; 128]) -> &Container {
        unsafe { &*(words.as_ptr() as *const Container) }
    }
}

impl BindNode {
    /// View the first 128 words as a Container (meta).
    pub fn meta_container(&self) -> &Container {
        Container::view(self.fingerprint[..128].try_into().unwrap())
    }

    /// View the second 128 words as a Container (content).
    pub fn content_container(&self) -> &Container {
        Container::view(self.fingerprint[128..].try_into().unwrap())
    }
}
```

Now every Container search function (belichtungsmesser, hamming_early_exit,
cascade, MexicanHat) works on BindSpace data with ZERO COPIES.

ArrowZeroCopy already handles 256-word fingerprints. FingerprintBuffer can
expose ContainerRef views the same way.

**Pros**:
- True zero-copy everywhere: BindSpace → Container → Arrow
- ONE canonical store, ONE address space, ONE edge representation
- All 8 surfaces (Redis, SQL, Flight, HTTP, Search, ACID, Arrow, Learning) unaffected
- Minimal BindNode changes (add `is_spine: bool`)
- XorDag parity continues to cover everything
- Container search functions get free 256-word versions (just call on both halves)
- Rollback = remove DnIndex field + revert BindNode.is_spine

**Cons**:
- "Meta" vs "content" interpretation of first/second 128 words is NEW semantic
  meaning on EXISTING data. Must audit all write sites.
- Lifetime annotations: `ContainerRef<'a>` can't outlive the BindSpace borrow.
  Callers that need owned Containers must explicitly `.clone()`.
- Container-specific features (inline edges in words 16-31) now live
  inside BindNode.fingerprint[16..32] — may conflict with existing data
  that uses those words as pure fingerprint bits.

**Migration risk**: LOW. Existing BindNodes have all 256 words as fingerprint.
The new interpretation (meta + content) only applies to nodes written through
the new API. Old nodes continue to work — their "meta half" just has no
structured meaning (fine; MetaView reads zeros as defaults).

---

### Option B: CogRecord Retrofit (Matryoshka)

**Core idea**: Formalize BindNode.fingerprint as a CogRecord (meta Container + content Container).
All writes MUST go through structured API that separates meta from content.

```rust
impl BindNode {
    pub fn set_meta(&mut self, meta: &Container) {
        self.fingerprint[..128].copy_from_slice(&meta.words);
    }
    pub fn set_content(&mut self, content: &Container) {
        self.fingerprint[128..].copy_from_slice(&content.words);
    }
}
```

**Pros**:
- Clean semantic separation
- All Container features (inline edges, MetaView, geometry) work natively
- Full 256 bits used meaningfully (not half-wasted)
- ArrowZeroCopy sees full record

**Cons**:
- EVERY existing write site must be audited and migrated
  - `BindSpace::write_at(addr, fingerprint)` currently writes 256 words as one unit
  - `XorDag::write()` stages 256-word writes
  - `CogRedis::dn_set()` passes full fingerprint to write_dn_path
  - `UnifiedEngine::write()` takes `[u64; 256]`
  - All these must split into meta + content
- Existing similarity/hamming on full 256 words NOW includes metadata bits
  - Need new `content_hamming()` that only compares [128..256]
  - Or deprecate full-width hamming
- **Migration**: existing data has content in [0..128] (the first words).
  Under Matryoshka, content is [128..256]. Data must be shuffled or
  all existing nodes become semantically wrong.

**Migration risk**: HIGH. Semantic inversion of existing data layout.

---

### Option C: Container-First Rewrite (Radical)

**Core idea**: Container (128 words) replaces Fingerprint (256 words) as the
canonical unit. BindSpace stores `meta: Container` + `content: Container` per slot
instead of `fingerprint: [u64; 256]`.

```rust
pub struct BindNode {
    pub meta: Container,
    pub content: Container,
    pub label: Option<String>,
    // ...
}
```

**Pros**:
- Cleanest architecture: one width everywhere
- Container operations are native, not borrowed views
- SpineCache becomes the backing store for BindSpace (arena of Containers)
- All existing Container code (search, adjacency, semiring) works directly

**Cons**:
- **Massive rewrite**: Every file that touches `BindNode.fingerprint` breaks
  - bind_space.rs (>500 lines of fingerprint handling)
  - xor_dag.rs (parity over `[u64; 256]`)
  - unified_engine.rs (all write/read methods)
  - cog_redis.rs (all DN/CAM/DAG commands)
  - hardening.rs, resilient.rs (WAL entries, buffers)
  - cognitive_udfs.rs (all UDFs assume 256-word FixedSizeBinary(2048))
  - fingerprint_table.rs (schema is FixedSizeBinary(2048))
  - lance_zero_copy/mod.rs (FingerprintBuffer hardcoded to 256 words)
  - flight/server.rs, flight/actions.rs (serialize fingerprints)
- XorDag parity must be rewritten for 2×128 blocks
- ArrowZeroCopy/FingerprintBuffer must change to 128-word buffers
- All existing data becomes invalid (width change)
- Estimate: ~5000 lines touched, ~200 test changes

**Migration risk**: EXTREME. Essentially a new codebase.

---

### Option D: Matryoshka Vendor Import

**Core idea**: Keep BindSpace AND DnSpineCache, but make DnSpineCache a
read-through cache that stores pointers into BindSpace (not its own copies).

```rust
pub struct DnSpineCache {
    bind_space: Arc<RwLock<BindSpace>>,  // reads/writes go HERE
    dn_to_addr: HashMap<PackedDn, Addr>, // index only
    // NO SpineCache, NO Container storage
}

impl DnSpineCache {
    pub fn read_content(&self, dn: PackedDn) -> Option<&Container> {
        let addr = self.dn_to_addr.get(&dn)?;
        let bs = self.bind_space.read();
        let node = bs.read(*addr)?;
        Some(Container::view(&node.fingerprint[128..256]))
    }
}
```

**Pros**:
- DnSpineCache API stays similar (less rewriting of callers)
- Data lives in BindSpace (single source of truth)
- XorDag covers all writes

**Cons**:
- Lifetime problem: `read_content()` can't return `&Container` borrowed from
  a `RwLockReadGuard` that's dropped at end of method. Requires either:
  - Returning owned Container (defeats zero-copy)
  - Passing closure: `with_content(dn, |c| {...})` (ugly API)
  - Parking_lot `RwLockReadGuard::map()` (leaks lock abstraction)
- DnSpineCache still exists as a separate struct with its own API
- Two ways to access the same data (via BindSpace or via DnSpineCache)
- Write coordination still needed (who calls XorDag?)

**Migration risk**: MEDIUM. API gymnastics around lifetimes.

---

### Option E: Start Over from BindSpace (Nuclear)

**Core idea**: Delete ALL consolidation code. Add the harvested ideas
directly to BindSpace as methods.

Same as Option A but without keeping DnIndex as a separate struct — instead,
add `dn_paths: HashMap<PackedDn, Addr>` directly to `BindSpace`.

**Pros**:
- Simplest possible result
- No new structs, no new modules

**Cons**:
- BindSpace struct becomes very large (already ~30 fields)
- DN-specific logic pollutes general-purpose BindSpace
- Harder to test DN features in isolation
- Loss of modularity

**Migration risk**: LOW, but poor separation of concerns.

---

## 3. Recommended Approach: Option A (BindSpace Lens)

Option A is the only approach that achieves:
1. Zero-copy from BindSpace through Container through Arrow
2. Single canonical store
3. Minimal migration risk
4. All 8 surfaces continue working
5. XorDag protection maintained

The key additions to BindSpace are small:
- `BindNode.is_spine: bool` (1 byte per node)
- `DnIndex` struct (pure addressing, ~3 fields)
- `Container::view()` (one unsafe transmute, zero runtime cost)
- `BindSpace.dirty: BitVec` (8KB for 65536 bits)

Everything else is deletion or reorganization.

---

## 4. Implementation Steps (For Next Session)

### Phase 0: Branch Setup (Do First)

```bash
git checkout claude/code-review-SMMuY
git checkout -b claude/bindspace-unification-<session-id>
```

All work happens on the new branch. The consolidation code stays
intact on the parent branch for reference.

---

### Phase 1: Add `Container::view()` — The Zero-Copy Lens

**File**: `src/container/mod.rs`

Add this method to `Container`:

```rust
/// Zero-cost transmute: borrow a &[u64; 128] as &Container.
/// Container is #[repr(C, align(64))], same layout as [u64; 128].
#[inline(always)]
pub fn view(words: &[u64; 128]) -> &Container {
    // SAFETY: Container is #[repr(C)] with single field `words: [u64; 128]`.
    // The layout is identical. Alignment is >= that of [u64; 128].
    unsafe { &*(words.as_ptr() as *const Container) }
}

/// Mutable zero-cost transmute.
#[inline(always)]
pub fn view_mut(words: &mut [u64; 128]) -> &mut Container {
    unsafe { &mut *(words.as_mut_ptr() as *mut Container) }
}
```

**Why first**: Everything else depends on this. Once this exists,
BindNode can expose Container views without copying.

**Test**: `assert_eq!(Container::view(&c.words), &c)` for any Container c.

---

### Phase 2: Extend BindNode — Meta/Content Views + is_spine

**File**: `src/storage/bind_space.rs`

Add to BindNode struct:

```rust
pub struct BindNode {
    pub fingerprint: [u64; FINGERPRINT_WORDS],  // unchanged
    // ... existing fields ...

    /// Whether this node is a spine (cluster centroid) in the DN tree.
    pub is_spine: bool,  // NEW
}
```

Add methods:

```rust
impl BindNode {
    /// Zero-copy Container view of first 128 words.
    /// Interpretation: metadata (rung encoding, inline edges, geometry).
    pub fn meta_container(&self) -> &Container {
        Container::view(
            <&[u64; 128]>::try_from(&self.fingerprint[..128]).unwrap()
        )
    }

    /// Zero-copy Container view of second 128 words.
    /// Interpretation: semantic content fingerprint.
    pub fn content_container(&self) -> &Container {
        Container::view(
            <&[u64; 128]>::try_from(&self.fingerprint[128..]).unwrap()
        )
    }

    /// Mutable meta container view.
    pub fn meta_container_mut(&mut self) -> &mut Container {
        Container::view_mut(
            <&mut [u64; 128]>::try_from(&mut self.fingerprint[..128]).unwrap()
        )
    }

    /// Mutable content container view.
    pub fn content_container_mut(&mut self) -> &mut Container {
        Container::view_mut(
            <&mut [u64; 128]>::try_from(&mut self.fingerprint[128..]).unwrap()
        )
    }
}
```

**Import**: Add `use crate::container::Container;` to bind_space.rs.

**Backward compat**: ALL existing code that reads `node.fingerprint` directly
continues to work. The views are additive. `is_spine` defaults to `false`.

---

### Phase 3: Add DnIndex to BindSpace

**File**: `src/storage/bind_space.rs` (or new file `src/storage/dn_index.rs`)

```rust
use crate::container::adjacency::PackedDn;

/// Bidirectional DN ↔ Addr index. Pure addressing — no data storage.
pub struct DnIndex {
    dn_to_addr: HashMap<PackedDn, Addr>,
    addr_to_dn: Vec<Option<PackedDn>>,  // indexed by Addr.0, 65536 entries
}

impl DnIndex {
    pub fn new() -> Self {
        Self {
            dn_to_addr: HashMap::new(),
            addr_to_dn: vec![None; TOTAL_ADDRESSES],
        }
    }

    pub fn register(&mut self, dn: PackedDn, addr: Addr) {
        self.dn_to_addr.insert(dn, addr);
        self.addr_to_dn[addr.0 as usize] = Some(dn);
    }

    pub fn addr_for(&self, dn: PackedDn) -> Option<Addr> {
        self.dn_to_addr.get(&dn).copied()
    }

    pub fn dn_for(&self, addr: Addr) -> Option<PackedDn> {
        self.addr_to_dn[addr.0 as usize]
    }

    pub fn len(&self) -> usize {
        self.dn_to_addr.len()
    }
}
```

Add to BindSpace struct:

```rust
pub struct BindSpace {
    // ... existing fields ...
    pub dn_index: DnIndex,                    // NEW
    pub dirty: bitvec::vec::BitVec,           // NEW (65536 bits)
}
```

Wire `write_dn_path()` to automatically register in DnIndex:

```rust
// Inside write_dn_path(), after creating a node:
let packed = /* convert string path to PackedDn */;
self.dn_index.register(packed, addr);
```

**Address resolution**: Use BindSpace's existing `dn_path_to_addr()` (DefaultHasher).
Remove the SplitMix64 variant from addr_bridge.rs — ONE hash function, period.

To convert string path → PackedDn, add a helper:

```rust
impl PackedDn {
    /// Create from colon-separated path string.
    /// "Ada:A:soul" → PackedDn::new(&[hash("Ada"), hash("A"), hash("soul")])
    pub fn from_path(path: &str) -> Self {
        let components: Vec<u8> = path.split(':')
            .take(Self::MAX_DEPTH)
            .map(|seg| {
                // Deterministic 8-bit hash of segment
                let mut h = 0u8;
                for &b in seg.as_bytes() {
                    h = h.wrapping_mul(31).wrapping_add(b);
                }
                h
            })
            .collect();
        Self::new(&components)
    }
}
```

---

### Phase 4: Add Dirty Tracking to BindSpace

**File**: `src/storage/bind_space.rs`

```rust
/// Mark address as dirty (modified since last flush).
pub fn mark_dirty(&mut self, addr: Addr) {
    self.dirty.set(addr.0 as usize, true);
}

/// Get all dirty addresses.
pub fn dirty_addrs(&self) -> impl Iterator<Item = Addr> + '_ {
    self.dirty.iter_ones().map(|i| Addr(i as u16))
}

/// Clear dirty bits.
pub fn clear_dirty(&mut self) {
    self.dirty.fill(false);
}
```

Wire into `write_at()`:

```rust
pub fn write_at(&mut self, addr: Addr, fp: [u64; FINGERPRINT_WORDS]) {
    // ... existing write logic ...
    self.mark_dirty(addr);
}
```

---

### Phase 5: Merge Useful UDFs into cognitive_udfs.rs

**File**: `src/query/cognitive_udfs.rs`

Add these NEW UDFs (they don't exist yet at 256-word width):

1. `belichtung(a, b)` → 7-point estimator on FULL 256-word fingerprint
   - Adapt: sample 7 words from 256 (not 128). Indices: 0, 37, 73, 109, 146, 182, 219.
2. `cascade_filter(query, candidate, threshold)` → 5-level cascade on 256 words
3. `word_diff(a, b)` → count differing words (0-256)
4. `mexican_hat(distance)` → wavelet response (distance-only, width-independent)
5. `bundle(a, b)` → majority-vote bundle on 256 words

**Do NOT add**: `container_hamming`, `container_similarity`, `container_popcount`,
`container_xor` — these duplicate `hamming`, `similarity`, `popcount`, `xor_bind`
which already exist at 256-word width.

**Delete**: `src/query/container_udfs.rs` after merging.

---

### Phase 6: Rewrite DnTreeTableProvider to Read from BindSpace

**File**: `src/query/dn_tree_provider.rs` (rewrite in place)

Change from:
```rust
pub struct DnTreeTableProvider {
    dn_cache: Arc<RwLock<DnSpineCache>>,  // OLD: reads from parallel store
}
```

To:
```rust
pub struct DnTreeTableProvider {
    bind_space: Arc<RwLock<BindSpace>>,  // NEW: reads from canonical store
}
```

Schema changes:
- `meta`: FixedSizeBinary(1024) → borrow `node.meta_container().as_bytes()` via `Container::view()`
- `content`: FixedSizeBinary(1024) → borrow `node.content_container().as_bytes()`
- OR change to FixedSizeBinary(2048) and expose full fingerprint (simpler, no split)

Iterate BindSpace's DnIndex instead of DnSpineCache:

```rust
for (dn, addr) in bind_space.dn_index.iter() {
    if let Some(node) = bind_space.read(addr) {
        // Build Arrow arrays from node fields
        // Use Container::view() for zero-copy meta/content
    }
}
```

**Critical**: For zero-copy into Arrow, use `Buffer::from_slice_ref()` instead of `.to_vec()`:

```rust
use arrow::buffer::Buffer;
// Instead of: meta_values.push(meta.as_bytes().to_vec())
// Use: Buffer::from_slice_ref(node.meta_container().as_bytes())
```

---

### Phase 7: Rewrite Plasticity to Operate on BindSpace Addrs

**File**: `src/storage/plasticity.rs` (NEW file, replaces `src/container/plasticity.rs`)

The algorithm is the same. The data source changes:

```rust
pub fn post_insert_plasticity(
    bind_space: &mut BindSpace,
    changed_addr: Addr,
    codebook: &[(&str, &[u64; FINGERPRINT_WORDS])],
    prune_threshold: u32,
    consolidation_threshold: u32,
) -> PlasticityOps {
    let spine_fp = bind_space.read(changed_addr)?.fingerprint;
    let children = bind_space.children_raw(changed_addr);
    // ... same prune/consolidate/rename logic using Addr, not PackedDn ...
}
```

Plasticity now operates on the same data that XorDag protects.
Writes go through `bind_space.write_at()` → dirty tracking → XorDag eligible.

---

### Phase 8: Clean Up container/mod.rs Exports

**File**: `src/container/mod.rs`

Remove module declarations for deleted files:

```rust
// DELETE these lines:
pub mod dn_spine_cache;
pub mod plasticity;
pub mod addr_bridge;
pub mod cog_redis_bridge;
pub mod csr_bridge;
```

Keep everything else (geometry, meta, record, cache, search, semiring,
delta, spine, insert, adjacency, graph, dn_redis, traversal, migrate).

---

### Phase 9: Update query/mod.rs

**File**: `src/query/mod.rs`

Remove `container_udfs` module declaration. Keep `dn_tree_provider`
(it's been rewritten, not deleted).

---

### Phase 10: Run Tests, Fix, Verify

```bash
# All default tests must pass
cargo test

# With features
cargo test --features "simd,parallel,flight"

# Clippy clean
cargo clippy --features "simd,parallel,flight" -- -D warnings

# Verify test count is >= 567 minus deleted tests plus new tests
```

The 35 tests from the consolidation will be deleted (they test deleted code).
Replace with ~20 new tests that verify:
- `Container::view()` is truly zero-copy (pointer equality)
- `BindNode.meta_container()` / `content_container()` work
- `DnIndex` register/lookup round-trips
- Dirty tracking fires on write
- DnTreeTableProvider reads from BindSpace correctly
- Merged UDFs produce correct results at 256-word width
- Plasticity on BindSpace Addrs produces same results as on DnSpineCache

---

## 5. Files — Complete Action List

### Delete (7 files)

| File | Lines | Reason |
|------|-------|--------|
| `src/container/dn_spine_cache.rs` | 582 | Parallel store → replaced by DnIndex |
| `src/container/plasticity.rs` | 194 | Rewritten as `src/storage/plasticity.rs` |
| `src/container/addr_bridge.rs` | 164 | Incompatible hash → replaced by DnIndex |
| `src/container/cog_redis_bridge.rs` | 134 | Dead code → CogRedis uses BindSpace directly |
| `src/container/csr_bridge.rs` | 102 | Dead code → BindSpace has CSR directly |
| `src/query/container_udfs.rs` | 813 | Duplicate UDFs → merged into cognitive_udfs.rs |
| `src/query/dn_tree_provider.rs` | 391 | Rewritten in place (reads BindSpace, not DnSpineCache) |

### Modify (5 files)

| File | Changes |
|------|---------|
| `src/container/mod.rs` | Remove 5 module declarations |
| `src/storage/bind_space.rs` | Add `is_spine`, DnIndex, dirty tracking, Container views |
| `src/query/cognitive_udfs.rs` | Add belichtung, cascade_filter, word_diff, mexican_hat, bundle UDFs |
| `src/query/dn_tree_provider.rs` | Rewrite to read from BindSpace (keep file, change contents) |
| `src/query/mod.rs` | Remove container_udfs module |

### Create (2 files)

| File | Purpose |
|------|---------|
| `src/storage/dn_index.rs` | DnIndex struct (if not inlined into bind_space.rs) |
| `src/storage/plasticity.rs` | Plasticity operating on BindSpace Addrs |

### Keep Unchanged

Everything else. Specifically:
- `src/container/adjacency.rs` (PackedDn, InlineEdgeView)
- `src/container/search.rs` (all search functions)
- `src/container/semiring.rs`
- `src/container/spine.rs` (SpineCache — still useful for scratch)
- `src/container/meta.rs`
- `src/container/geometry.rs`
- `src/container/graph.rs`
- `src/container/dn_redis.rs`
- `src/container/traversal.rs`
- All storage/ files (unified_engine, xor_dag, cog_redis, hardening, resilient, lance_zero_copy)
- All flight/ files
- All query/ files except noted above
- All bin/ files

---

## 6. Zero-Copy Verification Checklist

After implementation, verify these zero-copy paths:

| Path | How to verify |
|------|---------------|
| BindNode → Container | `std::ptr::eq(node.meta_container().words.as_ptr(), node.fingerprint.as_ptr())` |
| BindNode → Arrow | DnTreeTableProvider uses `Buffer::from_slice_ref()`, no `.to_vec()` |
| Arrow → FingerprintBuffer | Already works (256-word FingerprintBuffer unchanged) |
| FingerprintBuffer → Container | `Container::view(&fp_buffer.get(i)?[..128])` |
| BindSpace → XorDag | Already works (XorDag reads `[u64; 256]` from BindSpace) |

If any path involves `.to_vec()`, `.clone()`, or `Container::from_bytes()`,
it's not zero-copy. Fix it.

---

## 7. What This Does NOT Fix (Out of Scope)

These are pre-existing issues documented in `docs/STORAGE_CONTRACTS.md`:

1. XorDag TOCTOU race (parity computed after lock release)
2. Write-behind WAL in hardening.rs
3. UnifiedEngine bypass write path
4. LRU tracker race in hardening.rs
5. WriteBuffer orphan in resilient.rs
6. DependencyGraph partial update in resilient.rs

These should be fixed AFTER the unification, in a separate PR.
The unification makes them EASIER to fix because there's now one
write path instead of two.

---

## 8. Rollback Strategy

The implementation happens on a NEW branch. If it fails:

```bash
git checkout claude/code-review-SMMuY   # back to consolidation branch
# All consolidation code is intact, nothing lost
```

If it succeeds but needs partial revert:

```bash
git revert <commit>   # surgical revert of any phase
```

Each phase should be a SEPARATE COMMIT so individual phases can be
reverted without affecting others:

1. `feat: add Container::view() zero-copy lens`
2. `feat: extend BindNode with meta/content views and is_spine`
3. `feat: add DnIndex to BindSpace for PackedDn addressing`
4. `feat: add dirty tracking to BindSpace`
5. `feat: merge container UDFs into cognitive_udfs.rs`
6. `refactor: rewrite DnTreeTableProvider to read from BindSpace`
7. `refactor: rewrite plasticity to operate on BindSpace Addrs`
8. `cleanup: remove parallel DnSpineCache, bridges, and duplicate UDFs`

Phase 8 (deletion) happens LAST. Everything before it is additive.
If any phase breaks, stop and revert just that phase.

---

## 9. Success Criteria

The refactor is DONE when:

- [ ] `cargo test` passes with >= 550 tests (some deleted, some new)
- [ ] `cargo clippy` has 0 new warnings
- [ ] Zero `.to_vec()` calls in dn_tree_provider.rs
- [ ] Zero `Container::from_bytes()` calls in UDF hot paths
- [ ] `DnSpineCache` struct no longer exists
- [ ] `addr_bridge.rs` no longer exists
- [ ] `container_udfs.rs` no longer exists
- [ ] `cog_redis_bridge.rs` and `csr_bridge.rs` no longer exist
- [ ] All Container operations available through `BindNode.meta_container()` / `content_container()`
- [ ] DnIndex has bidirectional PackedDn ↔ Addr lookup in BindSpace
- [ ] Dirty tracking covers all `write_at()` calls
- [ ] One hash function for DN → Addr (`dn_path_to_addr`, DefaultHasher)

---

## 10. Reference: The 8 Surfaces That Touch BindSpace

Every surface must continue to work after unification:

| # | Surface | Entry point | What it reads/writes | Impact |
|---|---------|-------------|---------------------|--------|
| 1 | Redis | `cog_redis.rs` → `bind_space` | DN.*, CAM.*, DAG.* | None (already direct) |
| 2 | SQL | `fingerprint_table.rs` → `bind_space` | SELECT from bindspace | None (unchanged) |
| 3 | SQL-Tree | `dn_tree_provider.rs` → `bind_space` | SELECT from dn_tree | Rewritten (Phase 6) |
| 4 | Flight | `flight/server.rs` → `bind_space` | DoGet, DoAction | None (reads BindSpace) |
| 5 | HTTP | `bin/server.rs` → `cog_redis` → `bind_space` | REST endpoints | None (pass-through) |
| 6 | Search | `hdr_cascade.rs` → `bind_space` | HDR filtering | None (reads BindSpace) |
| 7 | ACID | `xor_dag.rs` → `bind_space` | Transactions + parity | None (unchanged) |
| 8 | Arrow | `lance_zero_copy/` → (should feed) `bind_space` | Zero-copy buffers | Enabled (Container::view) |
| 9 | Learning | `cam_ops.rs` → `bind_space` | 4096 CAM operations | None (reads BindSpace) |

Surfaces 1-2, 4-7, 9 require ZERO changes. Surface 3 is rewritten.
Surface 8 gets new capability (Container views into Arrow buffers).

---

## 11. The Holy Grail: End-to-End Zero-Copy External API

The deeper goal is not just internal zero-copy between BindSpace and Container.
It's **zero-copy from storage through DataFusion through Arrow Flight to the
external client**. No serialization. No intermediate buffers. The client gets
a pointer into the same memory that BindSpace owns.

### The Current Copy Chain (What's Broken)

```
BindSpace.nodes[addr].fingerprint      ← canonical memory
    │
    ▼  copy #1: BindNode → Vec<u8> (.to_vec())
Vec<u8>
    │
    ▼  copy #2: Vec<u8> → FixedSizeBinaryArray (try_from_iter)
Arrow RecordBatch
    │
    ▼  copy #3: RecordBatch → IPC serialization (Arrow Flight)
Bytes on wire
    │
    ▼  copy #4: IPC deserialization (client side)
Client RecordBatch
```

Four copies. Every query. Every row.

### The Zero-Copy Chain (What Should Exist)

```
Arrow Buffer (mmap or owned)           ← ONE allocation
    │
    ├── FingerprintBuffer.get(i)       ← zero-copy: pointer into Buffer
    │       │
    │       ├── &[u64; 256]            ← zero-copy: raw fingerprint ref
    │       │       │
    │       │       ├── Container::view(&fp[..128])   ← zero-copy: meta lens
    │       │       └── Container::view(&fp[128..])   ← zero-copy: content lens
    │       │
    │       └── UInt64Array (shares same Buffer)      ← zero-copy: Arrow view
    │
    ├── FixedSizeListArray             ← zero-copy: wraps same Buffer
    │       │
    │       └── RecordBatch            ← zero-copy: column refs into Buffer
    │               │
    │               └── Arrow Flight IPC    ← zero-copy: Buffer → FlightData
    │                       │
    │                       └── Client      ← mmap or single copy at network edge
```

ONE allocation. Everything else is a typed view into it.

### How to Achieve This

#### Step 1: Arrow Buffer as Canonical Storage

Instead of `BindSpace` owning `Vec<Box<[Option<BindNode>; 256]>>` (heap-allocated
arrays of heap-allocated structs), the fingerprint data should live in a single
contiguous Arrow `Buffer`:

```rust
pub struct FingerprintStore {
    /// Contiguous Arrow buffer: 65536 × 256 × 8 = 128 MB
    /// Each fingerprint is 256 u64s = 2048 bytes.
    buffer: Buffer,

    /// Occupancy bitmap: which slots are populated
    occupied: BooleanBuffer,  // 65536 bits = 8 KB
}

impl FingerprintStore {
    /// Zero-copy access to fingerprint at address.
    pub fn get(&self, addr: Addr) -> Option<&[u64; FINGERPRINT_WORDS]> {
        if !self.occupied.value(addr.0 as usize) {
            return None;
        }
        let offset = (addr.0 as usize) * FINGERPRINT_WORDS;
        unsafe {
            let ptr = (self.buffer.as_ptr() as *const u64).add(offset);
            Some(&*(ptr as *const [u64; FINGERPRINT_WORDS]))
        }
    }

    /// Zero-copy Container view of meta half.
    pub fn meta(&self, addr: Addr) -> Option<&Container> {
        let fp = self.get(addr)?;
        Some(Container::view(<&[u64; 128]>::try_from(&fp[..128]).unwrap()))
    }

    /// Zero-copy Container view of content half.
    pub fn content(&self, addr: Addr) -> Option<&Container> {
        let fp = self.get(addr)?;
        Some(Container::view(<&[u64; 128]>::try_from(&fp[128..]).unwrap()))
    }

    /// The entire buffer as an Arrow array — ZERO COPY.
    /// This is what DataFusion and Flight read.
    pub fn as_arrow_array(&self) -> FixedSizeListArray {
        let u64_array = UInt64Array::from(
            ArrayData::builder(DataType::UInt64)
                .len(65536 * FINGERPRINT_WORDS)
                .add_buffer(self.buffer.clone())  // Arc bump, no copy
                .build().unwrap()
        );
        let field = Arc::new(Field::new("item", DataType::UInt64, false));
        FixedSizeListArray::new(field, FINGERPRINT_WORDS as i32, Arc::new(u64_array), None)
    }
}
```

The key: `self.buffer.clone()` on an Arrow `Buffer` is an **Arc increment**, not
a data copy. The same bytes back the BindSpace lookup AND the Arrow array AND
the Flight response.

#### Step 2: TableProvider Returns Buffer-Backed Batches

The rewritten `DnTreeTableProvider` and `FingerprintTableProvider` should
return `RecordBatch`es whose columns share the FingerprintStore buffer:

```rust
// In FingerprintTableProvider::execute():
let fp_array = store.as_arrow_array();  // zero-copy: shares Buffer
let batch = RecordBatch::try_new(schema, vec![
    address_array,
    Arc::new(fp_array),  // NO .to_vec(), NO FixedSizeBinaryArray::try_from_iter()
    label_array,
    // ...
])?;
```

DataFusion passes this batch through its query engine. When it reaches
Arrow Flight serialization, the IPC writer reads directly from the Buffer.

#### Step 3: Flight DoGet Streams Buffer-Backed Batches

Arrow Flight IPC already supports zero-copy when the RecordBatch columns
reference Arrow Buffers (not copied Vecs). The existing FlightDataEncoder
in `arrow-flight` crate handles this:

```rust
// flight/server.rs — DoGet handler
async fn do_get(&self, ticket: Ticket) -> Result<FlightDataStream> {
    let batch = self.execute_query(ticket)?;
    // batch.columns() share the FingerprintStore buffer
    // FlightDataEncoder writes IPC format directly from buffer memory
    Ok(FlightDataEncoderBuilder::new()
        .build(futures::stream::once(async move { Ok(batch) })))
}
```

On the client side, `FlightClient::do_get()` deserializes into a RecordBatch
whose buffers may be mmap'd or single-copy from the network. This is the
minimum possible copy count for a network API.

#### Step 4: External Client Gets FingerprintRef

For Rust clients consuming Flight results:

```rust
// Client code:
let batch = flight_client.do_get(ticket).await?;
let fp_column = batch.column(1)
    .as_any()
    .downcast_ref::<FixedSizeListArray>()
    .unwrap();

// Zero-copy: get the underlying buffer
let fp_buffer = FingerprintBuffer::from_fixed_size_list(fp_column).unwrap();

// Zero-copy: access individual fingerprints
let fp: &[u64; 256] = fp_buffer.get(0).unwrap();

// Zero-copy: view as Container
let meta = Container::view(&fp[..128]);
let content = Container::view(&fp[128..]);

// Compute distance without any allocation
let dist = meta.hamming(other_meta);
```

**This is the holy grail**: storage → query → flight → client, all sharing
the same buffer. The Container is never serialized or deserialized — it's
a typed lens over Arrow memory.

### Why This Requires BindSpace Unification First

You CANNOT achieve end-to-end zero-copy if:
- DnSpineCache stores its own copy (it does today)
- UDFs deserialize FixedSizeBinary → Container::from_bytes() (they do today)
- TableProviders call .to_vec() on every row (they do today)
- Two incompatible widths exist (128 vs 256, they do today)

The unification (Phases 1-8 above) eliminates all four blockers.
Then FingerprintStore (this section) becomes a natural evolution
of BindSpace's internal storage, not a separate system.

### Implementation Order

FingerprintStore is Phase 11 — AFTER the unification:

1. Phases 1-8: Unify on BindSpace (this plan)
2. Phase 9: Verify zero-copy from BindNode through Container::view()
3. Phase 10: Replace BindNode.fingerprint arrays with Buffer-backed storage
4. Phase 11: Wire FingerprintStore into TableProviders and Flight
5. Phase 12: Add mmap backing for persistence (via lance_zero_copy LanceView)

The unification is prerequisite infrastructure. Without it,
FingerprintStore would just be yet another parallel copy.

### DataFusion Usage: Doing It Right

DataFusion's query engine is designed for zero-copy. The correct patterns:

**TableProvider scan()**: Return RecordBatches with Buffer-backed columns.
Do NOT materialize data into Vecs. Use `ArrayData::builder().add_buffer()`
with the canonical Buffer.

**UDFs**: Accept `ArrayRef` columns, not individual scalars. Process entire
arrays at once using Arrow compute kernels. This is both zero-copy AND
vectorized:

```rust
// WRONG: scalar-at-a-time, copies on every call
fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
    let a = scalar_to_container(&args.args[0])?;  // copy
    let b = scalar_to_container(&args.args[1])?;  // copy
    Ok(ColumnarValue::Scalar(...))
}

// RIGHT: array-at-a-time, zero-copy
fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
    match (&args.args[0], &args.args[1]) {
        (ColumnarValue::Array(a), ColumnarValue::Array(b)) => {
            // Get underlying buffers (zero-copy)
            let a_buf = FingerprintBuffer::from_fixed_size_list(a.as_fixed_size_list())?;
            let b_buf = FingerprintBuffer::from_fixed_size_list(b.as_fixed_size_list())?;

            // Vectorized computation, no per-element allocation
            let mut results = UInt32Builder::with_capacity(a_buf.len());
            for (fa, fb) in a_buf.iter().zip(b_buf.iter()) {
                results.append_value(hamming_256(fa, fb));
            }
            Ok(ColumnarValue::Array(Arc::new(results.finish())))
        }
        // ... scalar fallback for single values ...
    }
}
```

This is how DataFusion is MEANT to be used. The current UDFs (both
cognitive_udfs.rs and container_udfs.rs) only handle scalars, which
forces DataFusion to call the UDF once per row instead of once per batch.

### The LanceDB Connection

Lance datasets store data in Arrow IPC format on disk. When opened with mmap:

```rust
let dataset = lance::dataset::Dataset::open("path").await?;
let batch = dataset.scan().try_into_batch().await?;
// batch columns are mmap'd Arrow Buffers — zero-copy from disk
```

If BindSpace's FingerprintStore IS an Arrow Buffer, then:
- Loading from Lance = swapping the buffer pointer (mmap'd file)
- Saving to Lance = writing the buffer to disk (already Arrow format)
- No serialization. No deserialization. The file IS the memory.

This is why the lance_zero_copy module exists. It just needs to be
connected to BindSpace instead of being a standalone parallel system.

The full chain becomes:

```
Lance file on disk (Arrow IPC format)
    ↓ mmap
Arrow Buffer (OS page-faults on access)
    ↓ FingerprintStore wraps it
BindSpace (all lookups hit mmap'd memory)
    ↓ Container::view()
Container operations (zero-copy lens)
    ↓ TableProvider returns Buffer-backed batch
DataFusion query engine (columnar processing)
    ↓ FlightDataEncoder
Arrow Flight (IPC on wire)
    ↓ client deserialize
Client RecordBatch (single copy at network edge)
```

Disk to client: ONE copy (at the network boundary). Everything else
is pointer arithmetic and typed views.

THAT is the architecture worth building. And it starts with unifying
on BindSpace so there's one buffer to rule them all.

---

*This document is the complete specification. A new session should
read this file, create the branch, and execute phases 0-8 in order.
Phases 9-12 (FingerprintStore + end-to-end zero-copy) follow as the
natural next step once unification is complete.*
