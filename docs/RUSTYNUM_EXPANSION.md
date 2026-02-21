# Rustynum Expansion Plan

**What to build. What to skip. In what order.**
*Scoped to CogRecord 65536 + ladybug-rs + "PyTorch in Rust" ambitions*

---

## 0. The Framing

PyTorch has ~2,000 ops. You have ~80. Closing that gap naively would take years and miss the point.

Your actual compute workloads fall into 3 tiers:

| Tier | What | Math involved | Status |
|------|------|---------------|--------|
| **A** | CogRecord 65536 operations | XOR, Hamming, bundle, int8 dot, cascade search | **DONE** (hdc.rs + int8_gemm.rs) |
| **B** | Cognitive kernel + NARS + RL | Small f32 vector ops, matrix-vector, softmax | **Mostly done** (operations.rs, statistics.rs, BLAS L1-L2) |
| **C** | PyTorch-equivalent tensor engine | Autograd, broadcasting, NN layers, optimizers | **Not started** |

Tier A is your competitive moat. Nobody else has VNNI-accelerated HDC with 3-stage cascade filters.

Tier B needs minor fills. Tier C is the long game — and the place where strategic choices matter.

---

## 1. TIER A Expansions: CogRecord 65536 Engine (1-2 weeks)

These are small, targeted additions that complete the CogRecord picture.

### 1.1 Arrow Zero-Copy Bridge (NEW CRATE: `rustynum-arrow`)

**Why**: LanceDB stores CogRecords as `FixedSizeBinary(8192)`. DataFusion scans return Arrow `RecordBatch`. You need zero-cost conversion from Arrow buffers to `NumArrayU8` / `Container` without copying 8KB per record.

```rust
// rustynum-arrow/src/lib.rs
use arrow::array::FixedSizeBinaryArray;

/// Zero-copy view: Arrow binary column → slice of NumArrayU8 references
pub fn arrow_to_containers(col: &FixedSizeBinaryArray) -> &[u8] {
    // Arrow FixedSizeBinary stores values contiguously
    // Same layout as packed database for hamming_search_adaptive
    col.value_data()
}

/// Scan an Arrow column with cascade Hamming filter
pub fn cascade_scan_arrow(
    query: &NumArrayU8,
    column: &FixedSizeBinaryArray,
    threshold: u64,
) -> Vec<(usize, u64)> {
    let db = NumArrayU8::from_borrowed(column.value_data());
    let vec_len = column.value_length() as usize;
    let count = column.len();
    query.hamming_search_adaptive(&db, vec_len, count, threshold)
}
```

**Effort**: ~200 LOC. **Impact**: Eliminates the copy between LanceDB scan and Container operations. For 1M records × 8KB = 8GB, this saves 8GB of allocation.

### 1.2 Quantization Pipeline (expand `int8_gemm.rs`)

**Why**: Container 3 needs to ingest Jina/CLIP f32 embeddings and quantize to int8 for storage. You have `quantize_f32_to_i8()` but need the full pipeline.

**Add**:
```rust
// rustyblas/src/quantize.rs (~300 LOC)

/// Quantize 1024D f32 embedding to int8 with calibration stats
pub fn quantize_embedding(
    embedding: &[f32],           // 1024 floats from Jina/CLIP
    container_bytes: &mut [u8],  // 2048 bytes (Container 3)
) -> QuantMeta {
    // 1. Symmetric quantization: scale = max(|x|) / 127
    // 2. Write int8 values to first 1024 bytes
    // 3. Write QuantMeta (scale, zero_point, norm) to last 1024 bytes
    // 4. Return meta for storage in Container 0 W252-W253
}

/// Dequantize for exact comparison when cascade filter passes
pub fn dequantize_embedding(
    container_bytes: &[u8],
    meta: &QuantMeta,
    out: &mut [f32],
);

/// Per-channel quantization for batch of embeddings
pub fn quantize_batch(
    embeddings: &[f32],  // N × D flat
    n: usize, d: usize,
) -> (Vec<u8>, Vec<QuantMeta>);
```

**Effort**: ~300 LOC. **Impact**: Completes the Jina→Container 3 pipeline.

### 1.3 Batch Container Operations (expand `hdc.rs`)

**Why**: Cognitive kernel processes BindSpace of N records at once. Current API is single-pair. Need batch BIND, batch BUNDLE, batch PERMUTE.

**Add**:
```rust
// rustynum-rs/src/num_array/hdc.rs additions (~200 LOC)

/// Batch XOR-bind: result[i] = a[i] XOR b[i] for N containers
/// Uses parallel_for_chunks for large N
pub fn bind_batch(
    a: &NumArrayU8,  // N × vec_len contiguous
    b: &NumArrayU8,
    vec_len: usize,
    count: usize,
) -> NumArrayU8;

/// Batch permute: result[i] = permute(v[i], k) for N containers
pub fn permute_batch(
    vectors: &NumArrayU8,
    vec_len: usize,
    count: usize,
    k: usize,
) -> NumArrayU8;

/// Top-K Hamming: return the K nearest containers, heap-selected
pub fn hamming_top_k(
    query: &NumArrayU8,
    database: &NumArrayU8,
    vec_len: usize,
    count: usize,
    k: usize,
) -> Vec<(usize, u64)>;

/// Top-K Cosine: return the K most similar int8 embeddings
pub fn cosine_top_k(
    query: &NumArrayU8,
    database: &NumArrayU8,
    vec_len: usize,
    count: usize,
    k: usize,
) -> Vec<(usize, f64)>;
```

**Effort**: ~200 LOC. **Impact**: Batch ops for cognitive kernel and neo4j-rs traversal.

---

## 2. TIER B Expansions: Cognitive Compute (2-4 weeks)

These complete the math that NARS, RL, and Granger computations need.

### 2.1 Broadcasting (CRITICAL — enables everything else)

**Why**: Without broadcasting, every binary op requires exact shape match. PyTorch's magic is that `[1024, 1] + [1, 768]` auto-expands to `[1024, 768]`. Every NN layer depends on this.

**What to build**:
```rust
// rustynum-rs/src/num_array/broadcast.rs (~400 LOC)

/// Compute broadcast-compatible output shape
pub fn broadcast_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, ShapeError>;

/// Iterator that yields (a_index, b_index) pairs for broadcast element access
/// Zero-copy: doesn't expand tensors in memory, just computes virtual indices
pub struct BroadcastIter { ... }

/// Element-wise binary op with broadcasting
pub fn broadcast_binary_op<F>(a: &NumArray, b: &NumArray, f: F) -> NumArray
where F: Fn(f32, f32) -> f32;
```

**Effort**: ~400 LOC. **Impact**: Unlocks every future NN layer and complex tensor operation. Without this, nothing in Tier C works properly.

### 2.2 View / Stride-based Tensor (CRITICAL — zero-copy reshape)

**Why**: Current `reshape()` copies data. A `view()` just changes shape+strides metadata. For 8KB CogRecords, copying on every reshape wastes bandwidth.

**What to build**:
```rust
// Modify NumArray internals (~300 LOC refactor)

pub struct NumArray<T> {
    data: Arc<Vec<T>>,    // shared backing store (was Vec<T>)
    shape: Vec<usize>,
    strides: Vec<usize>,  // byte strides, not element strides
    offset: usize,        // start offset into data
}

impl NumArray<T> {
    /// Zero-copy reshape (returns view, not copy)
    pub fn view(&self, new_shape: &[usize]) -> Result<Self, ShapeError>;
    
    /// True if memory is contiguous (no gaps from slicing)
    pub fn is_contiguous(&self) -> bool;
    
    /// Make contiguous (copy if needed, no-op if already contiguous)
    pub fn contiguous(&self) -> Self;
}
```

**Effort**: ~300 LOC refactor of array_struct.rs. **Impact**: Eliminates copies in reshape/transpose/slice chains. Essential for CogRecord field access (Container 0 slice → MetaView → no copy).

### 2.3 LAPACK Fills (SVD + Eigendecomposition)

**Why**: Granger causality needs eigendecomposition for VAR model fitting. SVD is needed for dimensionality reduction and pseudoinverse.

**What to add to `rustymkl/src/lapack.rs`**:
```rust
/// SVD: A = U Σ V^T
pub fn sgesdd(a: &mut [f32], m: usize, n: usize, 
              u: &mut [f32], s: &mut [f32], vt: &mut [f32]);
pub fn dgesdd(/* same for f64 */);

/// Symmetric eigendecomposition: A = V Λ V^T  
pub fn ssyev(a: &mut [f32], n: usize, w: &mut [f32]);
pub fn dsyev(/* same for f64 */);

/// Solve linear system: A X = B
pub fn sgesv(a: &mut [f32], b: &mut [f32], n: usize, nrhs: usize);
pub fn dgesv(/* same for f64 */);

/// Matrix inverse via LU
pub fn sgetri(a: &mut [f32], n: usize, ipiv: &[i32]);
pub fn dgetri(/* same for f64 */);
```

**Effort**: ~400 LOC (follows existing LAPACK patterns). **Impact**: Enables Granger causality, PCA on embeddings, and proper least-squares for NARS parameter estimation.

### 2.4 Activation Functions

**Why**: If you ever want to run inference on a small local model (even just an MLP for reward prediction in the TD-learning loop), you need activations.

**What to add to `rustynum-rs/src/num_array/activations.rs`** (~200 LOC):
```rust
pub fn relu(&self) -> Self;          // max(0, x)
pub fn leaky_relu(&self, alpha: f32) -> Self;
pub fn gelu(&self) -> Self;          // x × Φ(x)
pub fn silu(&self) -> Self;          // x × σ(x), aka swish
pub fn tanh(&self) -> Self;
pub fn relu6(&self) -> Self;         // min(max(0,x), 6)
pub fn mish(&self) -> Self;          // x × tanh(softplus(x))
```

Plus VML-accelerated versions in `rustymkl/src/vml.rs` for `tanh` (which needs `vsTanh`).

**Effort**: ~200 LOC. **Impact**: Small but necessary. Every inference path needs at least relu + gelu.

### 2.5 Constructors + Random

**Add to `constructors.rs`** (~150 LOC):
```rust
pub fn rand(shape: Vec<usize>) -> Self;       // uniform [0, 1)
pub fn randn(shape: Vec<usize>) -> Self;      // normal(0, 1) via Box-Muller
pub fn full(shape: Vec<usize>, val: T) -> Self;
pub fn eye(n: usize) -> Self;                 // identity matrix
pub fn rand_like(&self) -> Self;
pub fn zeros_like(&self) -> Self;
pub fn ones_like(&self) -> Self;
```

**Effort**: ~150 LOC. **Impact**: Quality of life. Every test, every initialization, every random projection uses these.

---

## 3. TIER C Expansions: PyTorch Equivalent (3-6 months)

This is the long game. These are ordered by impact:dependency ratio.

### 3.1 Autograd Engine (THE big one)

Without autograd, you can't train models. With autograd, everything changes.

**Architecture**:
```rust
// rustynum-autograd/src/lib.rs (~2000 LOC)

/// A Tensor with optional gradient tracking
pub struct Variable {
    data: NumArrayF32,
    grad: Option<NumArrayF32>,
    grad_fn: Option<Arc<dyn Function>>,
    requires_grad: bool,
}

/// Trait for differentiable operations
pub trait Function: Send + Sync {
    fn forward(&self, inputs: &[&Variable]) -> Variable;
    fn backward(&self, grad_output: &NumArrayF32) -> Vec<NumArrayF32>;
}

/// Topological sort + backward pass
pub fn backward(loss: &Variable) {
    // Build reverse topological order via grad_fn graph
    // Call backward() on each Function
    // Accumulate gradients
}
```

**Key design decision**: Do you need a full dynamic graph (PyTorch-style) or is a static graph (TensorFlow-style) sufficient? For ladybug-rs cognitive kernel, static is probably enough — the cognitive cycle graph doesn't change shape at runtime. Static is MUCH simpler to implement (~800 LOC vs ~2000 LOC).

**Effort**: ~2000 LOC (dynamic), ~800 LOC (static). **Impact**: Enables training. Without this, rustynum is inference-only.

**My recommendation**: Start with **eager-mode, tape-based autograd**. Record operations as they execute (like PyTorch), replay tape backward. Simpler than a full graph compiler, sufficient for your workloads.

### 3.2 NN Layer Library

**Depends on**: Autograd + Broadcasting

```rust
// rustynum-nn/src/lib.rs

pub struct Linear { weight: Variable, bias: Option<Variable> }
pub struct LayerNorm { weight: Variable, bias: Variable, eps: f32 }
pub struct Embedding { weight: Variable }  // lookup table
pub struct Dropout { p: f32, training: bool }
```

**Priority layers for ladybug-rs**:
1. `Linear` — reward prediction MLP, NARS parameter estimation
2. `Embedding` — codebook lookup (4096 entries × Container width)
3. `LayerNorm` — any transformer-style processing
4. `MultiheadAttention` — if you ever run local attention over CogRecords

**Conv2d is NOT needed** unless you add vision processing. Skip it.

**Effort**: ~1000 LOC for {Linear, Embedding, LayerNorm, Dropout}. **Impact**: Enables small local models.

### 3.3 Optimizers

**Depends on**: Autograd

```rust
// rustynum-optim/src/lib.rs (~300 LOC)

pub trait Optimizer {
    fn step(&mut self);
    fn zero_grad(&mut self);
}

pub struct Adam { params: Vec<Variable>, lr: f32, betas: (f32, f32), eps: f32 }
pub struct SGD { params: Vec<Variable>, lr: f32, momentum: f32 }
```

**Effort**: ~300 LOC for Adam + SGD. **Impact**: Closes the training loop. Combined with autograd + Linear, you can train an MLP.

### 3.4 Batched Matrix Multiply (bmm)

**Why**: MultiheadAttention needs `bmm` for parallel attention heads. Batch size × heads × seq_len × d_k.

```rust
// rustyblas/src/level3.rs addition (~200 LOC)

/// Batched GEMM: C[i] = A[i] × B[i] for i in 0..batch
pub fn sgemm_batched(
    batch: usize,
    m: usize, n: usize, k: usize,
    a: &[f32], b: &[f32], c: &mut [f32],
);
```

**Effort**: ~200 LOC (loop over sgemm with offset computation, or call cblas_sgemm_batch if MKL available). **Impact**: Enables multi-head attention.

### 3.5 Einsum (powerful but can wait)

Einsum is a general tensor contraction. It subsumes dot, matmul, bmm, trace, transpose, outer product. Very expressive but complex to implement efficiently.

**Recommendation**: Skip for now. Use explicit matmul/bmm. Add einsum when you have a concrete need that isn't expressible as a sequence of existing ops.

---

## 4. What NOT to Build

| Skip | Why |
|------|-----|
| Conv2d/Conv3d | No vision workload in ladybug-rs |
| RNN/LSTM/GRU | Sequential models, not your architecture |
| Distributed training (DDP) | Single machine for now |
| CUDA backend | Intel GPU (Xe2) via Level Zero is your path, not NVIDIA |
| ONNX import/export | Not needed — you're not importing PyTorch models |
| DataLoader / Dataset | Use LanceDB/DataFusion for data pipeline |
| TorchScript / JIT compiler | Overkill for your workloads |
| torch.compile / Triton | GPU kernel generation — wrong hardware target |

---

## 5. The Intel-Specific Stack You Should Build Instead

Instead of chasing CUDA-centric PyTorch features, lean into what your hardware does that NVIDIA can't:

### 5.1 AMX Tile GEMM (if Sapphire Rapids+)

`compute.rs` detects `amx_tile`, `amx_int8`, `amx_bf16`. If present, AMX does 16×32 × 32×16 → 16×16 tile multiply in ONE instruction. This is **16× faster than VNNI** for large GEMM.

```rust
// rustyblas/src/amx_gemm.rs (~500 LOC)
// Tile configuration, load, dpbusd, store
// Falls back to VNNI int8_gemm on pre-AMX hardware
```

**Impact**: If your deployment target is Sapphire Rapids or newer, AMX makes int8 GEMM near-GPU speed.

### 5.2 Intel Xe2 GPU Offload

`compute.rs` detects GPU. For GEMM > 2048×2048, dispatch to GPU via Level Zero API:

```rust
// rustynum-gpu/src/lib.rs (~1000 LOC)
// Level Zero: discover device, create context, compile SPIR-V kernel
// Async dispatch: CPU does Container ops while GPU does large GEMM
```

**Impact**: Offloads massive embedding batch operations to iGPU while CPU handles Hamming/cascade.

### 5.3 NPU Offload (Meteor Lake+)

`compute.rs` detects NPU. Intel NPU excels at sustained int8 inference. Perfect for:
- Running quantized reward prediction model continuously
- Embedding generation for new Container 3 content
- NARS evidence accumulation at low power

**Impact**: Frees CPU entirely for Container operations while NPU handles ML inference.

---

## 6. Expansion Summary & Priority Matrix

| # | Expansion | LOC | Time | Blocks | Tier |
|---|-----------|-----|------|--------|------|
| **1** | Arrow zero-copy bridge | 200 | 2 days | LanceDB integration | A |
| **2** | Quantization pipeline (f32→int8→Container 3) | 300 | 2 days | Embedding ingest | A |
| **3** | Batch Container ops (bind_batch, top_k) | 200 | 2 days | Cognitive kernel perf | A |
| **4** | Broadcasting | 400 | 3 days | All Tier C | B |
| **5** | View/stride-based tensor (Arc refcount) | 300 | 3 days | All Tier C | B |
| **6** | LAPACK: SVD + eigen + solve + inverse | 400 | 3 days | Granger causality | B |
| **7** | Activation functions (relu, gelu, silu, tanh) | 200 | 1 day | NN layers | B |
| **8** | Constructors (rand, randn, eye, full) | 150 | 1 day | Quality of life | B |
| **9** | Autograd (tape-based, eager mode) | 1500 | 2 weeks | Training | C |
| **10** | NN layers (Linear, Embedding, LayerNorm) | 1000 | 1 week | Local models | C |
| **11** | Optimizers (Adam, SGD) | 300 | 2 days | Training loop | C |
| **12** | Batched GEMM | 200 | 2 days | Multi-head attention | C |
| **13** | AMX tile GEMM | 500 | 1 week | Sapphire Rapids+ perf | C |

**Total new code: ~5,650 LOC across 13 modules.**

---

## 7. Recommended Crate Structure

```
rustynum/                        (workspace)
├── rustynum-core/               ✓ EXISTS (1,747 LOC)
│   ├── blackboard.rs            ✓ Zero-copy arena
│   ├── compute.rs               ✓ Capability detection
│   ├── prefilter.rs             ✓ INT8 cascade
│   ├── simd.rs                  ✓ SIMD primitives
│   ├── parallel.rs              ✓ Scoped parallelism
│   └── layout.rs                ✓ Row/column major
│
├── rustynum-rs/                 ✓ EXISTS (9,934 LOC)
│   ├── num_array/
│   │   ├── array_struct.rs      ✓ → REFACTOR: Arc<Vec<T>> + strides (#5)
│   │   ├── broadcast.rs         ★ NEW (#4)
│   │   ├── hdc.rs               ✓ → EXPAND: batch ops, top_k (#3)
│   │   ├── constructors.rs      ✓ → EXPAND: rand, eye, full (#8)
│   │   ├── activations.rs       ★ NEW (#7)
│   │   └── ... (existing)
│   └── simd_ops/                ✓ SIMD trait dispatch
│
├── rustyblas/                   ✓ EXISTS (3,147 LOC)
│   ├── level1-3.rs              ✓ Full BLAS
│   ├── int8_gemm.rs             ✓ VNNI path
│   ├── bf16_gemm.rs             ✓ BF16 path
│   ├── quantize.rs              ★ NEW (#2)
│   └── batched_gemm.rs          ★ NEW (#12)
│
├── rustymkl/                    ✓ EXISTS (1,427 LOC)
│   ├── lapack.rs                ✓ → EXPAND: SVD, eigen, solve (#6)
│   ├── fft.rs                   ✓ FFT/IFFT
│   └── vml.rs                   ✓ Vectorized math
│
├── rustynum-arrow/              ★ NEW CRATE (#1)
│   └── lib.rs                   Zero-copy Arrow ↔ NumArray bridge
│
├── rustynum-autograd/           ★ NEW CRATE (#9)
│   ├── variable.rs              Tensor + gradient tracking
│   ├── tape.rs                  Operation recording
│   ├── backward.rs              Reverse-mode AD
│   └── ops/                     Differentiable op implementations
│
├── rustynum-nn/                 ★ NEW CRATE (#10)
│   ├── linear.rs                Linear layer
│   ├── embedding.rs             Embedding table
│   ├── norm.rs                  LayerNorm, BatchNorm
│   └── attention.rs             Multi-head attention (future)
│
├── rustynum-optim/              ★ NEW CRATE (#11)
│   ├── adam.rs                  Adam / AdamW
│   └── sgd.rs                   SGD + momentum
│
└── bindings/
    └── python/                  ✓ EXISTS (PyO3 bindings)
```

---

## 8. The Punchline

You don't need to clone PyTorch. You need to be **PyTorch for HDC/VSA + quantized inference on Intel SIMD**.

That's a much smaller, more focused target:

```
PyTorch:      ~2M LOC, 2000+ ops, CUDA-first, training-first
Rustynum:     ~17K LOC, 80 ops, AVX-512-first, inference-first + HDC

Target:       ~23K LOC, 130 ops, Intel SIMD-first, HDC-native, training-capable

The ops you ADD should be the ops CogRecord 65536 EXECUTES.
Everything else is premature generalization.
```

Build Tier A (items 1-3) this week. Build Tier B (items 4-8) next two weeks. Start Tier C (autograd) when you need to train a model, not before.
