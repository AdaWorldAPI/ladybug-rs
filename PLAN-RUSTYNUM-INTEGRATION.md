# Ladybug-RS × RustyNum Integration Plan

## Pre-Implementation Findings

### Q1: Can they compile into the same binary?

**Yes, with one blocker that requires a fix first.**

| Property | ladybug-rs | rustynum | Compatible? |
|---|---|---|---|
| Edition | 2024 | 2021 | Yes (2024 builds 2021 deps) |
| Rust version | 1.88+ (stable) | nightly (`portable_simd`) | **BLOCKER** |
| Arrow | 57 | 57 | Yes (exact match) |
| DataFusion | 51 | 51 (optional) | Yes |
| Lance | 2.0 | 2.0 (optional) | Yes |
| tokio | 1.49 | 1.x (optional) | Yes |
| rand | 0.9 | 0.8 (oracle only) | Minor conflict |
| SIMD approach | `std::arch` intrinsics | `portable_simd` feature | Different layers |
| Alignment | 64-byte (`repr(align(64))`) | 64-byte (alloc with ALIGNMENT=64) | Yes |
| External deps | arrow/datafusion/lance/rayon | zero (core+blas+mkl+holo) | No conflicts |

**The blocker**: rustynum uses `#![feature(portable_simd)]` in 5 crates (core, rs, blas, mkl, archive). This requires nightly Rust. Ladybug-rs compiles on stable 1.93.

**Fix**: Replace `portable_simd` with `std::arch` intrinsics (same approach ladybug already uses). The rustynum SIMD code already uses explicit AVX-512/AVX2 intrinsics in many hot paths — `portable_simd` is used mainly for convenience types (`f32x16`, `u8x64`). These can be rewritten as `__m512`/`__m256` operations, which work on stable Rust.

**Alternative**: Add `rust-toolchain.toml` to ladybug-rs specifying nightly. Faster to ship but forces all ladybug contributors to nightly.

**Recommendation**: Phase 0 (below) rewrites rustynum to stable `std::arch`. This is a one-time cost that unblocks all downstream users.

### Q2: Does BindSpace + Blackboard borrow-mut work with zero-copy?

**Yes, they operate at different levels and compose naturally.**

| Property | BindSpace | Blackboard |
|---|---|---|
| Purpose | 65K-address cognitive memory | SIMD-aligned computation arena |
| Granularity | Per-fingerprint (2KB each) | Per-named-buffer (arbitrary size) |
| Addressing | `Addr(u16)` → array index (3-5 cycles) | String name → HashMap → raw ptr |
| Ownership | Owns `BindNode` structs (fingerprint + metadata) | Owns raw byte allocations |
| Borrow model | `&mut self` for write, `&self` for read | Split-borrow: multiple `&mut [T]` from `&self` |
| Thread safety | `Send` (single-owner) | `Send` (single-owner, unsafe interior mut) |
| Memory layout | `[u64; 256]` per fingerprint + metadata fields | 64-byte aligned contiguous buffers |

**How they compose (zero-copy path):**

```
BindSpace (owns fingerprints)
    │
    ├── node.fingerprint: [u64; 256]  ← 2KB, 64-byte aligned (repr(align(64)))
    │
    │   For batch BLAS/VML operations:
    │
    ├── Blackboard::alloc_u8("query", 2048)    ← allocate scratch buffer
    ├── copy query fingerprint bytes into Blackboard buffer (one memcpy)
    │
    ├── rustyblas::int8_gemm() / rustymkl::vsexp()  ← operate on Blackboard buffers
    │
    └── read results back from Blackboard  ← pointer read, no copy
```

**Key insight**: BindSpace fingerprints are `[u64; 256]` = 2048 bytes = exactly the CogRecord container size in rustynum. The Blackboard's `alloc_u8("containers", N * 2048)` can hold a batch of fingerprints for SIMD bulk operations, then results are read back.

**The split-borrow is critical for GEMM-style ops**: `borrow_3_mut_f32("A", "B", "C")` gives three non-aliasing mutable slices from a shared `&self`, which Rust's normal borrow checker can't do with a single struct. This is exactly what's needed for batch Hamming (query in A, corpus chunk in B, distances in C).

**No architectural conflict.** The two systems serve different purposes:
- BindSpace = persistent cognitive addressing (owns the data)
- Blackboard = transient SIMD compute scratch (borrows/copies for computation)

For truly zero-copy batch operations, a thin adapter can provide `&[u8]` views of BindSpace fingerprint ranges directly to rustynum functions that take slices (most of rustyblas level-1 and rustynum-holo phase ops). No Blackboard needed for slice-based APIs.

---

## Phase 0 — Stable Rust Port (prerequisite)

**Goal**: Make rustynum compile on stable Rust so it can be a dependency of ladybug-rs.

### 0.1 Replace `portable_simd` with `std::arch` intrinsics

Files requiring changes:
- `rustynum-core/src/lib.rs` — remove `#![feature(portable_simd)]`
- `rustynum-core/src/simd.rs` — rewrite `Simd<f32, 16>` → `__m512` with `_mm512_*`
- `rustynum-rs/src/lib.rs` — remove feature gate
- `rustynum-rs/src/simd_ops.rs` — rewrite portable SIMD ops to `std::arch`
- `rustyblas/src/lib.rs` — remove feature gate
- `rustyblas/src/level1.rs` — sdot/ddot/saxpy etc. already use raw intrinsics in hot paths
- `rustyblas/src/level3.rs` — microkernels already use `_mm512_*` intrinsics
- `rustymkl/src/lib.rs` — remove feature gate
- `rustymkl/src/vml.rs` — vsexp/vsln etc. use `_mm512_*` (already stable-compatible)
- `rustynum-archive/src/lib.rs` — remove feature gate

**Estimate**: The actual SIMD kernels already use `std::arch` intrinsics. `portable_simd` is used for:
1. Type aliases (`f32x16`, `u8x64`) — replace with `__m512`, `__m512i`
2. Convenience ops (`.reduce_sum()`) — replace with `_mm512_reduce_add_ps()`
3. `Simd::from_slice()` — replace with `_mm512_loadu_ps()`

Most of rustyblas/rustymkl hot paths are already `std::arch`. This is primarily a cleanup.

### 0.2 Add path dependencies to ladybug-rs

```toml
# In ladybug-rs/Cargo.toml [dependencies]
rustynum-core = { path = "../rustynum/rustynum-core", default-features = false, features = ["avx512"], optional = true }
rustyblas     = { path = "../rustynum/rustyblas", default-features = false, features = ["avx512"], optional = true }
rustymkl      = { path = "../rustynum/rustymkl", default-features = false, features = ["avx512"], optional = true }
rustynum-rs   = { path = "../rustynum/rustynum-rs", optional = true }
rustynum-holo = { path = "../rustynum/rustynum-holo", default-features = false, features = ["avx512"], optional = true }

# Feature gate
[features]
rustynum = ["rustynum-core", "rustyblas", "rustymkl", "rustynum-rs", "rustynum-holo"]
full = [..., "rustynum"]
```

### 0.3 Verify compilation

```bash
cd ladybug-rs
cargo check --features rustynum   # must pass on stable
cargo test --features rustynum     # must pass
```

---

## Phase 1 — Drop-In HDC Acceleration

**Goal**: Replace ladybug's scalar/per-bit loops with rustynum's SIMD-vectorized equivalents.

### 1.1 Bundle acceleration (highest impact)

**Current** (`src/core/vsa.rs:55-93`): Bit-by-bit counting loop — O(N × 16384) with branch per bit.
```rust
// Current: 16384 iterations × N items × branch per bit
let mut counts = [0u32; FINGERPRINT_U64 * 64];
for item in items {
    for (word_idx, &word) in item.as_raw().iter().enumerate() {
        for bit in 0..64 {
            if (word >> bit) & 1 == 1 {
                counts[word_idx * 64 + bit] += 1;
            }
        }
    }
}
```

**Replace with**: rustynum-rs `CogRecord::bundle()` which uses SIMD ripple-carry majority voting. Expected 17× speedup.

**Implementation**: Add `fn bundle_simd(items: &[Fingerprint]) -> Fingerprint` in `src/core/vsa.rs` gated on `#[cfg(feature = "rustynum")]`, delegating to rustynum's bundle. The existing scalar path remains as fallback.

### 1.2 Bind/XOR acceleration

**Current** (`src/core/fingerprint.rs:151-157`): Scalar loop over 256 words.
```rust
for i in 0..FINGERPRINT_U64 {
    result[i] = self.data[i] ^ other.data[i];
}
```

**Replace with**: rustynum-core SIMD XOR (processes 64 bytes per instruction on AVX-512 = 4 iterations instead of 256). Expected 8-16× speedup.

### 1.3 Permute acceleration

**Current** (`src/core/fingerprint.rs:167-191`): Bit-by-bit rotation — O(16384) with get_bit/set_bit per position.

**Replace with**: Word-level rotation with carry (32 iterations on AVX-512). Expected 50-100× speedup.

### 1.4 Popcount acceleration

**Current** (`src/core/fingerprint.rs:110-112`): `iter().map(|x| x.count_ones()).sum()` — good but not SIMD-vectorized.

**Replace with**: rustynum VPOPCNTDQ path (same as ladybug's `simd.rs` but unified). The ladybug `simd.rs` AVX-512 Hamming is already excellent — the win here is unification, not speedup.

---

## Phase 2 — HDR Cascade Pre-Stage

**Goal**: Add rustynum's INT8 quantization and prefilter as an optional cascade stage.

### 2.1 INT8 sketch stage for HDR cascade

**Current** (`src/search/hdr_cascade.rs`): 4-level cascade (1-bit → 4-bit → 8-bit → full popcount), all scalar.

**Add**: INT8 quantized pre-stage using rustyblas `int8_gemm_i32`.

```
New cascade:
  L-1: INT8 batch distance (VNNI vpdpbusd, 64 MACs/instruction)  ← NEW
  L0:  1-bit sketch (existing)
  L1:  4-bit sketch (existing)
  L2:  8-bit sketch (existing)
  L3:  Full popcount (existing)
```

**How**: Quantize fingerprint chunks to i8 vectors, compute batch dot products via VNNI. Candidates below threshold skip to L0. This gives 4× throughput improvement for the initial filtering stage.

### 2.2 Batch Hamming via Blackboard

**Current** (`src/core/simd.rs:197-213`): Per-pair Hamming distance, parallelized with rayon.

**Replace with**: Blackboard-based batch processing. Allocate corpus chunk in Blackboard, run rustynum's parallel_for_chunks() with SIMD Hamming. Avoids rayon overhead for small batches and exploits cache locality for large batches.

---

## Phase 3 — Statistics & VML

**Goal**: Replace manual math with SIMD-accelerated transcendentals and statistics.

### 3.1 VML for truth value computation

**Target**: `src/nars/truth.rs` — NARS truth value functions (frequency × confidence).

Currently scalar `f32` operations. Batch truth evaluation across many edges can use rustymkl VML:
- `vsexp()` for exponential decay
- `vsln()` for log-evidence
- `vssqrt()` for confidence intervals

### 3.2 Statistics for temporal search

**Target**: `src/search/temporal.rs` — autocorrelation, cross-correlation, variance.

Replace manual variance/stddev with rustynum-rs statistics (SIMD-accelerated mean, std, variance).

### 3.3 VML for spectroscopy

**Target**: `src/container/spectroscopy/` — frequency analysis.

Replace scalar log/sqrt/sin/cos with rustymkl VML batch operations. 16-wide f32 SIMD instead of one-at-a-time.

---

## Phase 4 — Holographic Unification

**Goal**: Connect ladybug's hologram extensions to rustynum-holo's principled implementations.

### 4.1 Phase-space ops for quantum_field.rs

**Target**: `src/extensions/hologram/quantum_field.rs`

Replace ladybug's PhaseTag-based operations with rustynum-holo's phase-space primitives:
- `phase_bind_i8()` / `phase_unbind_i8()` — reversible ADD/SUB mod 256
- `wasserstein_sorted_i8()` — Earth Mover's distance (new capability, not in ladybug)
- `circular_distance_i8()` — wrap-around distance for unsorted vectors

### 4.2 Carrier waveform for embedding encoding

Ladybug's fingerprint→embedding pipeline can use rustynum-holo carrier encoding:
- `carrier_encode()` — frequency-domain concept encoding with VNNI acceleration
- `carrier_decode()` — demodulation via dot product
- Fibonacci spacing avoids harmonic interference, enables ~16-item bundling

### 4.3 Focus gating for scent extraction

Replace ladybug's 5-byte "flavor" extraction (`src/core/scent.rs`) with rustynum-holo's principled focus-of-attention:
- 3D geometric attention (8×8×32 volume)
- 48-bit masks for non-overlapping concept allocation
- `focus_xor()`, `focus_read()` — gated operations

### 4.4 Gabor wavelets for hologram extensions

Rustynum-holo's Gabor wavelet system subsumes phase+carrier+focus+5D projection into spatially-localized frequency encoding. This is the eventual target for ladybug's hologram extension modules.

### 4.5 Organic X-Trans for write pipeline

Ladybug's separate write → clean → learn pipeline can be replaced by rustynum-oracle's organic model where write=clean=learn in one pass, using X-Trans Fibonacci sampling.

---

## Phase 5 — Foundations (LAPACK/FFT/GEMM)

**Goal**: Use rustymkl for any dense linear algebra ladybug needs.

### 5.1 LAPACK QR for orthogonalization

Any Gram-Schmidt operations in learning paths can use `sgeqrf`/`dgeqrf` from rustymkl.

### 5.2 FFT for spectroscopy

`fft_f32`/`fft_f64` from rustymkl replaces any DFT needs in spectroscopy or frequency analysis.

### 5.3 GEMM for dense matrix operations

If ladybug ever needs dense matrix multiply (e.g., batch embedding transforms), rustyblas `sgemm` delivers 115 GFLOPS at 1024×1024.

---

## Implementation Order & Dependencies

```
Phase 0.1  (stable port)          ← PREREQUISITE for everything
    │
Phase 0.2  (Cargo.toml wiring)
    │
    ├── Phase 1.1 (bundle)        ← highest user-visible impact
    ├── Phase 1.2 (bind/xor)      ← simple, high frequency
    ├── Phase 1.3 (permute)       ← moderate impact
    └── Phase 1.4 (popcount)      ← unification, not speedup
         │
         ├── Phase 2.1 (INT8 cascade)  ← search throughput
         ├── Phase 2.2 (batch hamming)  ← memory efficiency
         │
         ├── Phase 3.1 (VML truth)      ← NARS acceleration
         ├── Phase 3.2 (temporal stats) ← search quality
         └── Phase 3.3 (VML spectro)    ← analysis speed
              │
              ├── Phase 4.1-4.5 (holographic) ← deep integration
              └── Phase 5.1-5.3 (foundations)  ← as-needed
```

## Testing Strategy

Each phase must:
1. Run all existing ladybug-rs tests (`cargo test`) — no regressions
2. Add `#[cfg(feature = "rustynum")]` + `#[cfg(not(feature = "rustynum"))]` dual paths
3. Add comparative benchmarks (existing vs rustynum) in `benches/`
4. Verify SIMD correctness: scalar fallback must produce identical results

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `portable_simd` removal breaks rustynum tests | Medium | High | Run full rustynum test suite after port |
| Fingerprint size mismatch (16K vs 2048-byte CogRecord) | Low | Medium | Adapt at boundary: FP = 2 × CogRecord containers |
| Nightly-only users of rustynum break | Low | Low | Keep nightly feature gate as optional |
| Arrow version drift | Low | Medium | Both at 57 now; pin together |
| rand 0.8 vs 0.9 conflict | Low | Low | Update rustynum-oracle to rand 0.9 |
