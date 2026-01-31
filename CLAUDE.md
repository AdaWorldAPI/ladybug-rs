# CLAUDE.md — Ladybug-RS

## Project Identity

**Ladybug-RS** is a pure-Rust cognitive substrate implementing:
- 8+8 address model (65,536 addresses, no FPU required)
- Redis syntax with cognitive semantics
- Universal bind space where all query languages hit same addresses
- 4096 CAM operations translated to LanceDB ops

**Repository**: https://github.com/AdaWorldAPI/ladybug-rs

---

## The Architecture You MUST Understand

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
- `let prefix = (addr >> 8) as u8;`
- `let slot = (addr & 0xFF) as u8;`
- 3-5 cycles. No HashMap. No FPU. Works on embedded/WASM.

---

## Current State

**Main branch**: ~37.5K lines of Rust (141 tests passing, 5 pre-existing failures)

**Last updated**: 2026-01-31

### ✅ Completed

| Feature | Lines | Status |
|---------|-------|--------|
| 8+8 addressing (prefix:slot) | bind_space.rs (1142) | ✓ Merged |
| Universal BindSpace O(1) indexing | bind_space.rs | ✓ Merged |
| 4096 CAM operations (16×256) | cam_ops.rs (3031) | ✓ Merged |
| CAM execution bridge | cog_redis.rs | ✓ Merged (PR #26) |
| Redis command executor | cog_redis.rs (2250) | ✓ Merged (PR #26) |
| LanceDB/DataFusion mappings | datafusion.rs | ✓ Merged |
| HDR Cascade Search | hdr_cascade.rs (1015) | ✓ ON MAIN |
| Cognitive Redis commands | cog_redis.rs | ✓ Merged |
| Wire Redis to BindSpace | cog_redis.rs | ✓ Merged (PR #25) |

### 🔄 Open PRs

| PR | Description | Action |
|----|-------------|--------|
| #24 | [REFERENCE] 64-bit CAM index | Old code for visibility - don't merge |
| #16 | Grammar engine | Audit recovery |
| #15 | Crystal extension | Review |
| #14 | ARCHITECTURE.md | Review + Merge |
| #12 | Dependencies | Merge when needed |
| #11 | Reconstructed files | ⚠️ AUDIT FIRST |

### 🔴 Remaining Failures (5 tests - need crystal features)

These tests **require the SPO+qualia crystal** to be compiled in:

```bash
# WITHOUT crystal features = 5 failures
cargo test

# WITH crystal features = should pass
cargo test --features "spo,quantum,codebook"
```

| Test | Feature Required | What It Tests |
|------|------------------|---------------|
| `collapse_gate` | `quantum` | Triangle collapse (consciousness gates) |
| `causal_ops` | `spo` | CausalEngine (Pearl's 3 rungs over SPO) |
| `quantum_ops` | `quantum` | TreeAddr (256-way branching in crystal) |
| `cypher` | `spo` | Cypher→SPO triple translation |
| `causal` | `spo,quantum` | SEE/DO/IMAGINE over crystal substrate |

**The 5×5×5 SPO+qualia crystal** (`src/extensions/spo/`, `context_crystal.rs`) is the substrate these tests operate on. Without `--features spo,quantum`, the crystal types don't exist.

**Fix:** Always test with crystal features:
```bash
cargo test --all-features
# or specifically:
cargo test --features "spo,quantum,codebook,hologram"
```

### 📋 TODO (Next Session)

**Priority 1: Wire HDR to RESONATE**
- [ ] Connect hdr_cascade.rs to CogRedis RESONATE command
- [ ] Add similarity search through BindSpace

**Priority 2: Fluid Zone Lifecycle**
- [ ] Implement TTL expiration (`tick()`)
- [ ] Implement `crystallize()` — promote fluid to node
- [ ] Implement `evaporate()` — demote node to fluid

**Priority 3: Harvest from PR #24 (old CAM index)**

Populate surface compartments from old type constants:

| Surface | Slots to Add | Source |
|---------|--------------|--------|
| 0x08 Concepts | 7 consciousness layers | LAYER_SUBSTRATE..LAYER_META |
| 0x09 Qualia | 12 thinking styles | STYLE_ANALYTICAL..STYLE_SPATIAL |

Harvest patterns:
- [ ] Cypher traversal API (`match_edges_from`, `match_edges_to`, `all_edges_from`)
- [ ] LBUG file format for persistence
- [ ] COW append pattern for fluid zone versioning

**Priority 4: Verify All Tests Pass**
- [ ] Run `cargo test --all-features` to confirm crystal tests pass
- [ ] Add persistence layer (mmap)

**Recent Commits** (PR #26):
```
271f4a2 - Add Redis command executor with CAM operation routing
3f329c5 - Add CAM execution bridge to CogRedis
13e95d6 - Fix example field names
07a1578 - Implement 4096 CAM operations
```

**Key files**:
```
src/storage/
├── bind_space.rs    # Universal DTO (8+8 addressing)
├── cog_redis.rs     # Redis syntax adapter
├── lance.rs         # LanceDB substrate
└── database.rs      # Unified interface

src/learning/
├── cam_ops.rs       # 4096 CAM operations (NEEDS REFACTOR)
├── quantum_ops.rs   # Quantum-style operators
├── rl_ops.rs        # Reinforcement learning
└── causal_ops.rs    # Pearl's 3 rungs

src/search/
├── hdr_cascade.rs   # HDR filtering (float emulation via popcount)
├── cognitive.rs     # NARS + Qualia + SPO
└── causal.rs        # SEE/DO/IMAGINE
```

---

## YOUR MISSION

### Priority 1: Fix Remaining Test Failures

5 tests still failing. Run and investigate:

```bash
cargo test 2>&1 | grep -E "FAILED|failures:"
```

### Priority 2: Merge HDR Cascade (PR #21)

This is the "alien magic" — O(1) bind vector search via popcount stacking:

```
Level 0: 1-bit sketch  → 90% filtered
Level 1: 4-bit count   → 90% of survivors
Level 2: 8-bit count   → 90% of survivors  
Level 3: Full popcount → exact distance

~7ns per candidate vs ~100ns for float cosine
```

Review and merge, then wire to BindSpace.

### Priority 3: Wire HDR to BindSpace

Connect the HDR cascade search to the fluid zone for similarity queries:

```rust
// In cog_redis.rs RESONATE command
pub fn resonate(&mut self, query: &Fingerprint, k: usize) -> Vec<(Addr, f32)> {
    // Use HDR cascade for fast filtering
    let candidates = self.hdr_index.search(query, k * 10);
    
    // Return top k with similarity scores
    candidates.into_iter()
        .take(k)
        .map(|m| (m.addr, m.similarity))
        .collect()
}
```

### Priority 4: Implement Fluid Zone Lifecycle

The fluid zone (0x10-0x7F) needs:

```rust
// TTL expiration
pub fn tick(&mut self) {
    let now = timestamp();
    for chunk in &mut self.fluid {
        for slot in chunk.iter_mut() {
            if let Some(node) = slot {
                if node.ttl.map(|t| t < now).unwrap_or(false) {
                    *slot = None;  // Evaporate
                }
            }
        }
    }
}

// Promote to node space
pub fn crystallize(&mut self, fluid_addr: Addr) -> Option<Addr> {
    let node = self.read(fluid_addr)?;
    let node_addr = self.allocate_node()?;
    self.write(node_addr, node.clone());
    self.delete(fluid_addr);
    Some(node_addr)
}

// Demote from node space  
pub fn evaporate(&mut self, node_addr: Addr, ttl: u32) -> Option<Addr> {
    let node = self.read(node_addr)?;
    let fluid_addr = self.allocate_fluid()?;
    let mut node = node.clone();
    node.ttl = Some(timestamp() + ttl);
    self.write(fluid_addr, node);
    self.delete(node_addr);
    Some(fluid_addr)
}
```

---

## Key Principles

### Two-Layer Architecture: Addressing vs Compute

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
│  AVX-512 (Railway, modern Xeon):                                           │
│    → 8×64-bit popcount per instruction                                     │
│    → 10K-bit fingerprint in 20 ops                                         │
│    → ~2ns per comparison                                                   │
│                                                                             │
│  AVX2 (NUC 14, most laptops):                                              │
│    → 4×64-bit via _mm256 intrinsics                                        │
│    → ~4ns per comparison                                                   │
│                                                                             │
│  Fallback (WASM, ARM, old x86):                                            │
│    → u64::count_ones() loop                                                │
│    → ~50ns per comparison                                                  │
│    → Still works, just slower                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

| What | How | Why |
|------|-----|-----|
| Address decode | `u8` shift/mask | Universal, no feature detect |
| Bucket lookup | `[prefix][slot]` | Array index, no hash |
| Hamming distance | `#[cfg(target_feature)]` | Best available SIMD |
| Batch compare | AVX-512 if available | 8 fingerprints parallel |
| HDR cascade | Popcount at each level | Adaptive precision |

### Addressing: ALWAYS Simple (int8)

```rust
// This runs everywhere, no feature detection
#[inline(always)]
fn lookup(addr: u16) -> (u8, u8) {
    ((addr >> 8) as u8, (addr & 0xFF) as u8)
}

// Array indexing, not HashMap
let prefix = (addr >> 8) as usize;
let slot = (addr & 0xFF) as usize;
let result = &arrays[prefix][slot];  // 3-5 cycles, not 30-100
```

### Compute: ADAPTIVE (use best available)

```rust
// AVX-512 path (Railway, Xeon): ~2ns per candidate
#[cfg(target_feature = "avx512vpopcntdq")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn hamming_batch_8(query: &[u64; 156], candidates: &[[u64; 156]; 8]) -> [u32; 8] {
    use std::arch::x86_64::*;
    // 8 fingerprints in parallel, 20 AVX-512 ops each
    // ...
}

// AVX2 path (most laptops, NUC): ~4ns per candidate
#[cfg(all(target_feature = "avx2", not(target_feature = "avx512vpopcntdq")))]
fn hamming_batch_4(query: &[u64; 156], candidates: &[[u64; 156]; 4]) -> [u32; 4] {
    // 4 fingerprints in parallel
    // ...
}

// Fallback (WASM, ARM, old x86): ~50ns but works everywhere
#[cfg(not(any(target_feature = "avx2", target_feature = "avx512vpopcntdq")))]
fn hamming_scalar(a: &[u64; 156], b: &[u64; 156]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
}
```

### Float Only at API Boundary

```rust
// Internal: pure integer
let distance: u32 = hamming(a, b);  // 0-10000

// API boundary only: convert for user
let similarity: f32 = 1.0 - (distance as f32 / 10000.0);  // 0.0-1.0
```

### Fluid Zone is Context Selector

The fluid zone (0x10-0x7F) determines what the node space MEANS:
- Different context = different interpretation of same node address
- Hot edges live here with TTL
- Promote to nodes when crystallized

### Universal DTO

All query languages hit the same `BindNode`:

```rust
pub struct BindNode {
    pub addr: u16,                    // WHERE
    pub fingerprint: [u8; 48],        // WHAT (384 bits, truncated from 10K)
    pub qualia: [i8; 8],              // HOW IT FEELS
    pub truth: (u8, u8),              // NARS <f, c>
    pub created_at: u32,              // WHEN
    pub ttl: Option<u32>,             // FORGET WHEN
}
```

---

## MCP Agent Guidance

When context window > 60%, spawn continuation with state:

```yaml
handover:
  current_task: "Implementing surface 0x02 (Cypher) ops"
  files_modified:
    - src/learning/cam_ops.rs
    - src/storage/cog_redis.rs
  decisions:
    - "Using recursive CTEs for path traversal"
    - "shortestPath maps to Dijkstra via window functions"
  next_steps:
    - "Complete 0x02:08-0x02:FF"
    - "Wire to cog_redis GRAPH.QUERY command"
  blockers: []
```

### Specialist Agents

- **🔬 LanceExpert**: Deep knowledge of LanceDB, Arrow, DataFusion
- **🕸️ GraphSage**: Cypher semantics, path algorithms, CSR
- **🧠 CognitiveArch**: NARS, qualia, truth maintenance
- **⚡ SIMDWizard**: AVX-512, popcount, batch ops

Spawn when domain expertise needed.

---

## Testing

```bash
cd ladybug-rs
cargo test --features "lancedb"
cargo test --features "redis"
cargo test --all-features

# Specific module
cargo test storage::bind_space
cargo test learning::cam_ops
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/AdaWorldAPI/ladybug-rs.git
cd ladybug-rs

# Check current state
find src -name "*.rs" | wc -l
wc -l src/learning/cam_ops.rs
wc -l src/storage/bind_space.rs

# Key files to understand first
cat src/storage/bind_space.rs | head -100
cat src/learning/cam_ops.rs | head -100
```

---

## Open PRs (Review Before Merge)

| PR | Status | Notes |
|----|--------|-------|
| #17 | Open | Cognitive operation enums |
| #16 | Open | Grammar engine (audit recovery) |
| #15 | Open | Crystal extension |
| #14 | Open | ARCHITECTURE.md |
| #12 | Open | Dependencies |
| #11 | Open | ⚠️ Reconstructed files - AUDIT FIRST |
| #10 | Open | ⚠️ Old 64-bit model - may conflict with 8+8 |
| #9 | Open | ⚠️ Kuzu stubs - FALSE BELIEF, close it |

---

## The Learning Loop

```
1. ENCOUNTER → Read existing code, understand schema
2. STRUGGLE  → Hit type mismatches, address conflicts
3. BREAKTHROUGH → See how 8+8 maps to LanceDB
4. CONSOLIDATE → Implement ops, test, commit
5. APPLY → Use ops from cog_redis commands
6. META-LEARN → Capture patterns for future sessions
```

**Capture moments**: When you figure something out, log it.
The learning curve IS the knowledge.

---

## Contact

**Owner**: Jan Hübener (jahube)
**GitHub**: https://github.com/AdaWorldAPI/ladybug-rs

---

**🦔 LADYBUG: Where all queries become one.**
