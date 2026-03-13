# SWAP_LANCE_TO_LANCEDB.md

## Switch from raw lance to lancedb SDK. Get versioning for free.

**Repo:** ladybug-rs
**Read first:** CLAUDE.md, then this file

---

## WHY

We depend on `lance = "2.0"` directly. Raw columnar read/write. No versioning.
Every write replaces. No history. No time travel. No compaction.

`lancedb = "0.26"` uses the SAME lance 2.0.0 underneath but adds:
- Automatic versioning on every write (MVCC for free)
- Time travel: `table.checkout(version_n)` reads any previous state
- Zero-copy between versions (only changed pages are new)
- Compaction: merge small versions into larger ones
- Schema evolution: add columns without rewriting
- Built-in ANN index management

This maps directly to the architecture:
- `mind.merge(held)` → lancedb creates a new version automatically
- `Seal::Staunen` → version changed → something is different
- SpineCache dirty → compaction trigger
- "show me what Ada knew before" → `table.checkout(old_version)`

## THE SWAP

### Step 1: Cargo.toml

```
REMOVE:
  lance = { version = "2.0", optional = true, default-features = false }

REMOVE feature definition:
  lancedb = ["lance"]

ADD:
  lancedb = { version = "0.26", optional = true }

CHANGE feature definition:
  lancedb = ["dep:lancedb"]
```

Also remove the commented-out vendor lance paths (lines 316-329):
```
# lance = { path = "vendor/lance/rust/lance", ... }
# lance-core = { path = ... }
# ... etc (14 commented lines)
```

### Step 2: Fix imports

Search for all `use lance::` and `lance::` in src/:

```bash
grep -rn "use lance::\|lance::" src/ --include="*.rs" | grep -v "lance_parser\|lance_graph\|lance-zero"
```

The lancedb SDK re-exports lance types via:
```rust
use lancedb::arrow::*;           // Arrow types
use lancedb::connection::*;      // Database connection  
use lancedb::table::*;           // Table operations
use lancedb::query::*;           // Query builder
use lancedb::index::*;           // Index management
```

For any raw lance types still needed:
```rust
use lancedb::lance::*;           // Re-exported lance internals
```

### Step 3: CI workflow

The protoc requirement remains (lancedb uses lance-encoding internally).
Add to .github/workflows/:

```yaml
- name: Install protoc
  run: sudo apt-get install -y protobuf-compiler
```

### Step 4: storage/ refactor

Read ALL files in src/storage/ that use lance. Refactor to use lancedb SDK:

```rust
// BEFORE (raw lance):
use lance::Dataset;
let ds = Dataset::open(path).await?;
let batch = ds.scan().try_into_stream().await?;

// AFTER (lancedb SDK):
use lancedb::connect;
let db = connect(path).execute().await?;
let table = db.open_table("spo_nodes").execute().await?;
let results = table.query().execute().await?;
// Versioning happens automatically on writes.
```

### Step 5: Use versioning where the architecture needs it

```rust
// mind.merge() writes a new version:
table.add(new_record_batches).execute().await?;
// lancedb automatically creates version N+1

// Seal check via version comparison:
let current_version = table.version().await?;
if current_version != expected_version {
    Seal::Staunen  // something changed underneath us
} else {
    Seal::Wisdom
}

// Time travel for "what did we know before?":
let old_table = table.checkout(version_n).await?;
let old_state = old_table.query()
    .filter("merkle_root = X")
    .execute().await?;

// Compaction (crystallize accumulated small writes):
table.optimize()
    .compact_files()  // merge small versions
    .cleanup_old_versions()  // prune ancient history
    .execute().await?;
```

### Step 6: Verify

```bash
cargo check --no-default-features --features "simd,lancedb"
cargo check --no-default-features --features "simd"  # without lancedb still works
```

Both must pass. lancedb is optional.

---

## VERSION ROADMAP

```
NOW:       lancedb 0.26.2 (crates.io) → lance 2.0.0
           Stable. Matches our current lance dep exactly.

NEXT:      lancedb 0.27+ (git tag) → lance 3.0.0-rc
           When stable: bump. Lance 3 moved to lance-format/lance.
           Will need import path changes for lance-* subcrates.

FUTURE:    lancedb 0.27+ on crates.io → lance 3.0 stable
           Just change version string in Cargo.toml.
```

## NOT IN SCOPE

```
× Don't touch lance-graph (separate repo, separate sessions)
× Don't add lancedb to lance-graph upstream PRs (it has its own dep)
× Don't refactor Node/Plane/Mask in this session (that's Phase B)
× Don't touch crewai/n8n deps
```
