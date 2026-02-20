# SPO 3D: Three-Axis Content-Addressable Graph

**Status:** Contract-ready. Implementation pending.
**Date:** 2026-02-20
**Crate:** `ladybug-rs` → `src/graph/spo/`
**Contract:** `crates/ladybug-contract/src/` (geometry, scent, spo_record extensions)

---

## 1. PROBLEM

CogRecord stores one content Container (1KB). Querying "who knows Ada?" requires scanning ALL records and testing each content fingerprint. No structural axis separation means forward, reverse, and relation queries all hit the same data.

The existing `ContainerGeometry::Xyz` links 3 CogRecords via DN tree (what/where/how). This works but requires 3 separate Redis GETs and DN tree traversal to reconstitute.

## 2. SOLUTION: SPO Geometry

A new `ContainerGeometry::Spo` that uses **sparse containers** within a single 2KB CogRecord envelope. Three axes — Subject (X), Predicate (Y), Object (Z) — encoded as bitmap + non-zero words, co-located in one record.

```text
┌──────────────────────────────────────────────────────────┐
│ CogRecord (ContainerGeometry::Spo)                       │
│                                                          │
│  meta: Container (1024 bytes)                            │
│    W0       DN address                                   │
│    W1       type | geometry=Spo(6) | flags               │
│    W2-3     timestamps, labels                           │
│    W4-7     NARS truth (freq, conf, pos_ev, neg_ev)      │
│    W8-11    DN tree (parent, child, next_sib, prev_sib)  │
│    W12-17   Scent (48 bytes: 3×16 nibble histograms)     │
│    W18-33   Inline edge index (64 slots)                 │
│    W34-39   Sparse axis descriptors (bitmap offsets)      │
│    W40-47   Bloom filter                                 │
│    W48-55   Graph metrics                                │
│    W56-63   Qualia                                       │
│    W64-79   Rung/RL history                              │
│    W80-95   Representation descriptor                    │
│    W96-111  Adjacency CSR                                │
│    W112-125 Reserved                                     │
│    W126-127 Checksum + version                           │
│                                                          │
│  content: Container (1024 bytes) — packed sparse axes    │
│    [0..2]    X bitmap (128 bits = 2 u64)                 │
│    [2..N]    X non-zero words                            │
│    [N..N+2]  Y bitmap                                    │
│    [N+2..M]  Y non-zero words                            │
│    [M..M+2]  Z bitmap                                    │
│    [M+2..K]  Z non-zero words                            │
│    [K..128]  padding / overflow                          │
│                                                          │
│  Total: 2048 bytes (same as Cam geometry)                │
└──────────────────────────────────────────────────────────┘
```

### Why Sparse Containers

At 30% density (typical for real-world content):
- Dense axis: 128 words = 1024 bytes
- Sparse axis: 2 words bitmap + ~38 non-zero words = 320 bytes
- Three sparse axes: 960 bytes ← fits in one content Container

Three axes in one record. One Redis GET. Same 2KB envelope.

## 3. KEY INSIGHT: Z→X CAUSAL CHAIN CORRELATION

When Record A's Z axis (Object) resonates with Record B's X axis (Subject), a causal link exists:

```
Record A:  X(Jan)   → Y(KNOWS)   → Z(Rust)
Record B:  X(Rust)  → Y(ENABLES) → Z(CAM)

hamming(A.z_dense, B.x_dense) ≈ 0  →  A causally feeds B
```

This is not a JOIN — it's a resonance test. The Hamming distance between Z₁ and X₂ IS the causal coherence score. The chain is valid iff each Z→X handoff resonates.

### Meta-Awareness Stacking (Piaget Development)

Each level's Object becomes the next level's Subject:

```
Level 0: X(body)        → Y(acts_on)       → Z(world)
Level 1: X(world)       → Y(represented)   → Z(symbols)
Level 2: X(symbols)     → Y(operate_on)    → Z(logic)
Level 3: X(logic)       → Y(reflects_on)   → Z(abstraction)
Level 4: X(abstraction) → Y(aware_of)      → Z(awareness)
```

The meta-record observing a chain gets its own scent. The system recognizes its own epiphanies by their nibble histogram signature. The BUNDLE of all meta-levels should CONVERGE back toward the original content — this is the testable tsunami prediction.

## 4. WHAT CHANGES

### Contract Crate (`crates/ladybug-contract/`)

| File | Change |
|------|--------|
| `geometry.rs` | Add `Spo = 6` variant |
| `container.rs` | Add `SparseAxes` packed encoding within Container |
| `scent.rs` (NEW) | 48-byte nibble histogram (`NibbleScent`) |
| `spo_record.rs` (NEW) | `SpoView` / `SpoViewMut` — zero-copy axis access |

### Implementation (`src/graph/spo/`)

| File | Purpose |
|------|---------|
| `mod.rs` | Module root, re-exports |
| `sparse.rs` | `SparseContainer` type + bitmap ops |
| `axes.rs` | X/Y/Z axis construction (build_node, build_edge) |
| `store.rs` | `SpoStore` with three-axis scanning |
| `chain.rs` | Causal chain discovery (Z→X correlation) |
| `tests.rs` | 6 ironclad tests |

### What DOES NOT Change

- `Container` type (128×u64, 8192 bits, 1KB)
- `CogRecord` struct (meta + content = 2KB)
- 5 RISC ops (BIND, BUNDLE, MATCH, PERMUTE, STORE/SCAN)
- Codebook (4096 entries, deterministic generation)
- Existing geometries (Cam, Xyz, Bridge, Extended, Chunked, Tree)
- MetaView word layout (W0-W127) — we use reserved words
- NARS truth value type and inference functions
- All existing tests (1,267+)

## 5. CONTRACTS

See: `CONTRACTS.md` in this directory.

## 6. SCHEMA

See: `SCHEMA.md` in this directory.

## 7. IMPLEMENTATION PHASES

### Phase 1: Contract Types (Day 1)
- Add `ContainerGeometry::Spo = 6`
- Add `NibbleScent` (48-byte histogram)
- Add `SparseAxes` (packed 3-axis encoding within Container)
- Add `SpoView` / `SpoViewMut` (zero-copy axis access)
- Tests: round-trip, packing invariants

### Phase 2: Sparse Container (Day 1-2)
- `SparseContainer` with bitmap + non-zero words
- `to_dense()` / `from_dense()` / `hamming_sparse()` / `bind_sparse()`
- Pack/unpack 3 sparse axes into one Container
- Tests: density invariants, hamming equivalence

### Phase 3: Axis Construction (Day 2-3)
- `build_node(dn, labels, properties) → CogRecord`
- `build_edge(dn, src_fp, verb, tgt_fp, nars) → CogRecord`
- Scent computation: nibble histogram per axis
- Tests: node round-trip, edge encoding

### Phase 4: SPO Store (Day 3-4)
- `SpoStore` wrapping `HashMap<u64, CogRecord>`
- `query_forward(src_fp, verb_fp) → Vec<(u64, u32)>` — scan X+Y, return Z matches
- `query_reverse(tgt_fp, verb_fp) → Vec<(u64, u32)>` — scan Z+Y, return X matches
- `query_relation(src_fp, tgt_fp) → Vec<(u64, u32)>` — scan X+Z, return Y matches
- Tests: forward, reverse, relation queries

### Phase 5: Causal Chain (Day 4-5)
- `causal_successors(record, radius) → Vec<(u64, u32)>` — Z→X scan
- `causal_predecessors(record, radius) → Vec<(u64, u32)>` — X→Z scan
- `chain_coherence(chain) → f32` — product of link coherences
- Meta-awareness record construction
- NARS truth propagation along chains
- Tests: chain coherence, meta convergence

### Phase 6: Lance Integration (Day 5+)
- Columnar schema with per-axis columns
- Sort key: (dn_prefix, scent_x, scent_y)
- XOR delta compression within sorted groups
- Production store replacing BTreeMap

## 8. DECISION LOG

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | `Spo = 6` in ContainerGeometry | Natural extension, doesn't break existing variants |
| 2 | Sparse axes packed in content Container | One Redis GET, same 2KB envelope |
| 3 | 48-byte nibble histogram replaces 5-byte XOR-fold for SPO | Per-axis type discrimination, no structure loss |
| 4 | Meta stays dense at W0-W127 | Identity/NARS/DN need fixed O(1) offsets |
| 5 | BTreeMap for POC, LanceDB for production | Prove correctness first, optimize second |
| 6 | Z→X Hamming distance = causal coherence | No explicit linking needed, geometry IS the test |
| 7 | Meta-awareness as recursive SPO records | Epiphanies stack as Z_{n} → X_{n+1} chains |
| 8 | Codebook slots 0-4095 unchanged | Instruction set is immutable |
