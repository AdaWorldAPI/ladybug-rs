# Ladybug-RS Storage Layer Handover

## Session Summary

**Date**: 2026-01-30  
**Focus**: Universal Bind Space with 8-bit prefix architecture

---

## Architecture Overview

### The 8-bit Prefix : 8-bit Slot Model

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

TOTAL: 256 prefixes × 256 slots = 65,536 addresses
```

### Why 8+8?

| Operation          | HashMap (16-bit) | Array Index (8+8) |
|-------------------|------------------|-------------------|
| Hash compute      | ~20 cycles       | 0                 |
| Bucket lookup     | ~10-50 cycles    | 0                 |
| Cache miss risk   | High             | Low (predictable) |
| Branch prediction | Poor             | Perfect (3-way)   |
| **TOTAL**         | ~30-100 cycles   | **3-5 cycles**    |

**No AVX-512. No SIMD. No special CPU instructions.**  
Just shift, mask, array index. Works on embedded/WASM.

---

## The Key Insight

**The fluid zone (0x10-0x7F) is a CONTEXT SELECTOR.**

It defines what the node space (0x80-0xFF) means:
- Chunk context = Concepts → node space holds concepts
- Chunk context = Memories → node space holds memories
- Chunk context = Codebook → node space holds patterns

**The node space is the UNIVERSAL DTO.**

All query languages bind to the same 32K addresses:
```
GET 0x8042              (Redis)
MATCH (n) WHERE id(n) = 0x8042  (Cypher)
SELECT * WHERE addr = 0x8042    (SQL)
{ node(id: "0x8042") }          (GraphQL)
```

Same address. Same fingerprint. The data doesn't care what syntax asked for it.

---

## Open Pull Requests

### PR #18 - Cognitive Redis
**Branch**: `feature/cognitive-redis`  
**Status**: OPEN  
**Files**: `src/storage/cog_redis.rs`, `src/storage/mod.rs`

### PR #19 - Hot Edge Cache
**Branch**: `feature/hot-edge-cache`  
**Status**: OPEN (depends on #18)

### PR #20 - Universal Bind Space
**Branch**: `feature/universal-bind-space`  
**Status**: OPEN  
**Files**: `src/storage/bind_space.rs`, `src/storage/mod.rs`

---

## Merge Order

1. **PR #20** first (Universal Bind Space) - the foundation
2. **PR #18** second (Cognitive Redis) - uses bind_space types
3. **PR #19** third (Hot Cache) - enhancement to CogRedis

---

## Code Structure

```
src/storage/
├── mod.rs           # Module exports
├── bind_space.rs    # Universal DTO (PR #20)
├── cog_redis.rs     # Redis adapter (PR #18, #19)
├── lance.rs         # Vector storage
├── kuzu.rs          # Graph storage (stub)
└── database.rs      # Unified interface
```

### The Hot Path

```rust
pub fn read(&self, addr: Addr) -> Option<&BindNode> {
    let prefix = addr.prefix();
    let slot = addr.slot() as usize;
    
    match prefix {
        // Surface: 16 compartments (0x00-0x0F)
        p if p <= PREFIX_SURFACE_END => {
            self.surfaces.get(p as usize).and_then(|c| c[slot].as_ref())
        }
        // Fluid: 112 chunks (0x10-0x7F)
        p if p >= PREFIX_FLUID_START && p <= PREFIX_FLUID_END => {
            let chunk = (p - PREFIX_FLUID_START) as usize;
            self.fluid.get(chunk).and_then(|c| c[slot].as_ref())
        }
        // Nodes: 128 chunks (0x80-0xFF)
        p if p >= PREFIX_NODE_START => {
            let chunk = (p - PREFIX_NODE_START) as usize;
            self.nodes.get(chunk).and_then(|c| c[slot].as_ref())
        }
        _ => None,
    }
}
```

---

## What's NOT Done Yet

1. **Language Adapters** - `QueryAdapter` trait defined but no implementations
2. **Persistence** - Currently in-memory only
3. **Kuzu Integration** - `kuzu.rs` is still a stub
4. **AVX-512 Integration** - `avx_engine.rs` exists but not wired
5. **Tests** - Basic tests exist, need more coverage

---

## Quick Start

```bash
cd /home/claude/ladybug-rs
git fetch origin
git checkout feature/universal-bind-space

# Key files
cat src/storage/bind_space.rs    # The DTO
cat src/storage/cog_redis.rs     # Redis adapter
cat src/storage/mod.rs           # Exports
```

---

## Links

- PR #18: https://github.com/AdaWorldAPI/ladybug-rs/pull/18
- PR #19: https://github.com/AdaWorldAPI/ladybug-rs/pull/19
- PR #20: https://github.com/AdaWorldAPI/ladybug-rs/pull/20
