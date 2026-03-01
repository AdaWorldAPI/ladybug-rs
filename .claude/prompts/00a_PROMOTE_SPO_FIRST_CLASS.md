# Phase 0: Promote SPO to First-Class Module

**Priority: Do this BEFORE any new work. Clean the house first.**

SPO, BNN, and CLAM are no longer experimental. They're default-on, unconditional dependencies, core to everything ladybug-rs does. The `extensions/spo/` location with `#[cfg(feature = "spo")]` gates is technical debt from when these were experimental. Remove it.

## The Move

### 1. Create `src/spo/` as top-level module

```
src/spo/
  mod.rs                    ← new, declares all submodules (no feature gate)
  spo.rs                    ← from extensions/spo/spo.rs (53KB — the SPO encoding)
  gestalt.rs                ← from extensions/spo/gestalt.rs (34KB — BundlingProposal)
  spo_harvest.rs            ← from extensions/spo/spo_harvest.rs (34KB — cosine replacement)
  shift_detector.rs         ← from extensions/spo/shift_detector.rs (10KB — stripe migration)
  jina_api.rs               ← from extensions/spo/jina_api.rs
  jina_cache.rs             ← from extensions/spo/jina_cache.rs
  clam_path.rs              ← NEW (Phase 5 — create here, not in extensions/)
  causal_trajectory.rs      ← NEW (Phase 6 — create here, not in extensions/)
```

### 2. Move other spo-gated extension modules up

These are all `#[cfg(feature = "spo")]` in `extensions/mod.rs` but `spo` is default-on:

```
src/extensions/context_crystal.rs      → src/spo/context_crystal.rs
src/extensions/meta_resonance.rs       → src/spo/meta_resonance.rs
src/extensions/nsm_substrate.rs        → src/spo/nsm_substrate.rs
src/extensions/codebook_training.rs    → src/spo/codebook_training.rs
src/extensions/deepnsm_integration.rs  → src/spo/deepnsm_integration.rs
src/extensions/cognitive_codebook.rs   → src/spo/cognitive_codebook.rs
src/extensions/crystal_lm.rs          → src/spo/crystal_lm.rs
src/extensions/sentence_crystal.rs    → src/spo/sentence_crystal.rs
```

### 3. Update `src/lib.rs`

```rust
// Replace:
//   pub mod extensions;
// With:
pub mod spo;           // Subject-Predicate-Object — core cognitive substrate

// Keep extensions/ ONLY for actually optional things (codebook, hologram, compress)
#[cfg(feature = "codebook")]
pub mod codebook;      // moved from extensions::codebook
#[cfg(feature = "hologram")]  
pub mod hologram;      // moved from extensions::hologram
#[cfg(feature = "compress")]
pub mod compress;      // moved from extensions::compress
```

Or simpler: just keep `extensions/` for the 3 actually-optional ones and remove `spo` from it entirely.

### 4. Update all `crate::extensions::spo::` paths

```bash
# Find all references
grep -rn "extensions::spo" --include="*.rs" src/

# Replace all
sed -i 's/crate::extensions::spo::/crate::spo::/g' src/**/*.rs
sed -i 's/super::spo::/crate::spo::/g' src/extensions/*.rs  # for the files that moved

# Also update any use statements
sed -i 's/use crate::extensions::spo/use crate::spo/g' src/**/*.rs
```

### 5. Update `extensions/mod.rs`

Remove ALL `#[cfg(feature = "spo")]` entries. What remains:

```rust
//! Optional Extensions for LadybugDB
//!
//! Enable via Cargo features: `codebook`, `hologram`, `compress`

#[cfg(feature = "codebook")]
pub mod codebook;

#[cfg(feature = "hologram")]
pub mod hologram;

#[cfg(feature = "compress")]
pub mod compress;
```

### 6. Update `Cargo.toml` features

```toml
[features]
default = ["simd", "parallel", "lancedb", "crewai"]
# NOTE: spo is NO LONGER a feature — it's always-on, unconditional
# Remove "spo" from default feature list and from [features] section entirely

spo_jina = ["reqwest"]  # Only the Jina HTTP client needs reqwest, rename from "spo"
```

### 7. Remove `src/extensions/spo/` directory

After all files are moved and paths updated:

```bash
rm -rf src/extensions/spo/
```

### 8. Verify

```bash
cargo test --workspace
cargo clippy --workspace -- -D warnings
grep -rn "extensions::spo" --include="*.rs" src/  # must return ZERO
grep -rn "cfg.*feature.*spo" --include="*.rs" src/  # must return ZERO (except spo_jina for reqwest)
```

## Why This Matters

Every `#[cfg(feature = "spo")]` gate is a lie — spo is always on. Every `crate::extensions::spo::` path says "this is optional" when it's the core cognitive substrate. The new work (clam_path.rs, causal_trajectory.rs) should not be born into a deprecated location. Clean the house, then build.

## What Stays in `extensions/`

Only things that are genuinely optional and feature-gated:
- `codebook/` — codebook training extensions
- `hologram/` — hologram extensions  
- `compress/` — compression extensions

Everything else moves to `src/spo/` or its natural top-level home.
