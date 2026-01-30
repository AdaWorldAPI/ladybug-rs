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

**Main branch**: ~35K lines of Rust

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

### 1. Refactor CAM Ops to 8+8 Schema

The `src/learning/cam_ops.rs` uses old 12-bit flat addressing. Refactor to:

```rust
/// Surface compartment (0x00-0x0F)
pub fn surface_prefix(compartment: SurfaceCompartment) -> u8 {
    compartment as u8  // 0x00-0x0F
}

/// Full 16-bit address
pub fn addr(prefix: u8, slot: u8) -> u16 {
    ((prefix as u16) << 8) | (slot as u16)
}

/// Decode address
pub fn decode(addr: u16) -> (u8, u8) {
    ((addr >> 8) as u8, (addr & 0xFF) as u8)
}
```

### 2. Implement Missing 4096 Operations

Each surface compartment (0x00-0x0F) has 256 slots. Map them to LanceDB ops:

```
SURFACE 0x00 (Lance) - 256 ops:
├── 0x00:00 VECTOR_SEARCH    → lance.search_vector(fp, k)
├── 0x00:01 VECTOR_INSERT    → lance.insert(table, batch)
├── 0x00:02 VECTOR_DELETE    → lance.delete(table, ids)
├── 0x00:03 VECTOR_UPDATE    → lance.update(table, batch)
├── 0x00:04 ANN_BUILD        → lance.create_index(table, IVF_PQ)
├── 0x00:05 ANN_QUERY        → lance.query_ann(fp, k, nprobes)
├── 0x00:06 SCAN_TABLE       → lance.scan(table, filter)
├── 0x00:07 SCAN_FRAGMENT    → lance.scan_fragment(table, frag_id)
...

SURFACE 0x01 (SQL) - 256 ops:
├── 0x01:00 SELECT           → datafusion.sql("SELECT ...")
├── 0x01:01 INSERT           → datafusion.sql("INSERT ...")
├── 0x01:02 UPDATE           → datafusion.sql("UPDATE ...")
├── 0x01:03 DELETE           → datafusion.sql("DELETE ...")
├── 0x01:04 JOIN             → datafusion.sql("... JOIN ...")
├── 0x01:05 AGGREGATE        → datafusion.sql("... GROUP BY ...")
├── 0x01:06 WINDOW           → datafusion.sql("... OVER ...")
├── 0x01:07 CTE              → datafusion.sql("WITH ... AS ...")
...

SURFACE 0x02 (Cypher) - 256 ops:
├── 0x02:00 MATCH            → cypher_to_sql("MATCH ...")
├── 0x02:01 CREATE_NODE      → cypher_to_sql("CREATE ...")
├── 0x02:02 CREATE_EDGE      → cypher_to_sql("CREATE ...-[...]->...")
├── 0x02:03 MERGE            → cypher_to_sql("MERGE ...")
├── 0x02:04 DELETE_NODE      → cypher_to_sql("DELETE ...")
├── 0x02:05 DELETE_EDGE      → cypher_to_sql("DELETE ...")
├── 0x02:06 SET_PROPERTY     → cypher_to_sql("SET ...")
├── 0x02:07 SHORTEST_PATH    → cypher_to_sql("shortestPath(...)")
...

SURFACE 0x07 (Verbs) - 256 ops:
├── 0x07:00 CAUSES           → bind(src, VERB_CAUSES, tgt)
├── 0x07:01 SUPPORTS         → bind(src, VERB_SUPPORTS, tgt)
├── 0x07:02 CONTRADICTS      → bind(src, VERB_CONTRADICTS, tgt)
├── 0x07:03 BECOMES          → bind(src, VERB_BECOMES, tgt)
├── 0x07:04 REFINES          → bind(src, VERB_REFINES, tgt)
├── 0x07:05 GROUNDS          → bind(src, VERB_GROUNDS, tgt)
├── 0x07:06 ABSTRACTS        → bind(src, VERB_ABSTRACTS, tgt)
├── 0x07:07 ENABLES          → bind(src, VERB_ENABLES, tgt)
...
```

### 3. Wire Redis Commands to Bind Space

In `src/storage/cog_redis.rs`, ensure all commands route through bind space:

```rust
pub fn execute(&mut self, cmd: &str) -> CogResult {
    let parsed = parse_redis_command(cmd)?;
    
    match parsed.op {
        "GET" => {
            let addr = self.resolve_key(&parsed.key)?;
            let node = self.bind_space.read(addr)?;
            CogResult::from_node(node)
        }
        "SET" => {
            let addr = self.resolve_or_allocate(&parsed.key)?;
            let node = BindNode::from_value(&parsed.value)?;
            self.bind_space.write(addr, node)?;
            CogResult::Ok
        }
        "BIND" => {
            let (a, b, verb) = parse_bind_args(&parsed.args)?;
            let edge_fp = a.bind(&verb).bind(&b);
            let addr = self.allocate_fluid()?;
            self.bind_space.write(addr, BindNode::edge(edge_fp))?;
            CogResult::Addr(addr)
        }
        // ... etc
    }
}
```

---

## Key Principles

### No FPU

All operations must work with pure integer arithmetic:
- Hamming distance via popcount
- Similarity via `1.0 - (dist as f32 / 10000.0)` ONLY at API boundary
- Internal: everything is u8, u16, u32, u64

### Array Indexing Over HashMap

```rust
// WRONG
let result = hashmap.get(&key);  // 30-100 cycles

// RIGHT
let prefix = (addr >> 8) as usize;
let slot = (addr & 0xFF) as usize;
let result = &arrays[prefix][slot];  // 3-5 cycles
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
