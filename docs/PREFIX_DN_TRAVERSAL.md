# Prefix/DN Traversal — Vertical & Horizontal Node/Edge Representation

> **Last Updated**: 2026-02-05
> **Status**: Core addressing working, orchestration prefixes wired

---

## The 8+8 Address Model

Every address in ladybug-rs is a 16-bit value split into **prefix** (high 8 bits) and **slot** (low 8 bits):

```
ADDRESS = PREFIX : SLOT
          (u8)    (u8)

Lookup:   3-5 CPU cycles (shift, mask, array index)
          No HashMap. No FPU. Works on embedded/WASM.
```

```rust
let prefix = (addr >> 8) as u8;
let slot   = (addr & 0xFF) as u8;
```

---

## Three Memory Zones

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SURFACE (0x00-0x0F)    16 prefixes x 256 slots = 4,096 addresses          │
│                                                                             │
│  Query language ops, verbs, concepts, meta-cognition                       │
│  0x00: Lance/Kuzu   0x04: NARS/ACT-R   0x08: Concepts   0x0C: Agents      │
│  0x01: SQL/CQL      0x05: Causal       0x09: Qualia     0x0D: Thinking    │
│  0x02: Cypher/GQL   0x06: Meta         0x0A: Memory     0x0E: Blackboard  │
│  0x03: GraphQL      0x07: Verbs        0x0B: Learning   0x0F: A2A         │
├─────────────────────────────────────────────────────────────────────────────┤
│  FLUID (0x10-0x7F)      112 prefixes x 256 slots = 28,672 addresses       │
│                                                                             │
│  Edges, working memory, context selectors, TTL-governed promote/demote     │
├─────────────────────────────────────────────────────────────────────────────┤
│  NODES (0x80-0xFF)      128 prefixes x 256 slots = 32,768 addresses       │
│                                                                             │
│  THE UNIVERSAL BIND SPACE                                                  │
│  All query languages hit the same addresses here                           │
│  DN tree nodes live in this zone                                           │
└─────────────────────────────────────────────────────────────────────────────┘

Total: 65,536 addresses (full u16 range)
```

### Slot Subdivision

Within a prefix, slots can be subdivided for dual-purpose use:

```
Slots 0x00-0x7F: Primary language/purpose
Slots 0x80-0xFF: Secondary language/purpose

Example: PREFIX_AGENTS (0x0C)
  0x0C:00-7F  Agent cards (128 agents)
  0x0C:80-FF  Agent persona fingerprints (1:1 with card slot)
```

Constant: `SLOT_SUBDIVISION = 0x80`

---

## BindNode — The Universal Container

Every address holds a `BindNode`:

```rust
pub struct BindNode {
    // Content
    pub fingerprint: [u64; 156],     // 10K-bit (9,984 bits) semantic vector
    pub label: Option<String>,        // Human-readable path or name
    pub payload: Option<Vec<u8>>,    // Binary data

    // DN TREE FIELDS (O(1) hierarchical navigation)
    pub parent: Option<Addr>,         // Single parent address (immediate)
    pub depth: u8,                    // Tree depth (0 = root)
    pub rung: u8,                     // Access level (R0-R9)
    pub sigma: u8,                    // Reasoning depth

    // Metadata
    pub qidx: u8,                    // Qualia index
    pub access_count: u32,           // Hot path tracking
}
```

---

## DN Path to Deterministic Address

DN (Distinguished Name) paths map deterministically to node-zone addresses:

```rust
pub fn dn_path_to_addr(path: &str) -> Addr {
    // Hash full path deterministically
    // Map to node address space (0x80-0xFF:XX)
    let hash = hash(path);
    let prefix = PREFIX_NODE_START + ((hash >> 8) as u8 & 0x7F);  // 0x80-0xFF
    let slot = (hash & 0xFF) as u8;
    Addr::new(prefix, slot)
}
```

**Example paths**:
```
"Ada:A:soul:identity"  → 0x8A47  (unique address)
"Ada:A:soul"           → 0x8C2F  (parent)
"Ada:A"                → 0x85B1  (grandparent)
"Ada"                  → 0x82E3  (root)
```

**O(1) parent path extraction** (no lookup needed):
```rust
pub fn dn_parent_path(path: &str) -> Option<&str> {
    path.rfind(':').map(|i| &path[..i])  // "Ada:A:soul" -> "Ada:A"
}
```

---

## Vertical Traversal (Parent-Child)

### Upward: Parent Chain — O(1) per hop

```rust
// Get immediate parent
pub fn parent(&self, addr: Addr) -> Option<Addr> {
    self.read(addr).and_then(|n| n.parent)  // Direct field access
}

// Get all ancestors (iterator, no allocation)
pub fn ancestors(&self, addr: Addr) -> impl Iterator<Item = Addr> + '_ {
    std::iter::successors(self.parent(addr), |&a| self.parent(a))
}
```

**Walk from leaf to root**:
```
Node 0x8A47 (Ada:A:soul:identity)
  → parent = 0x8C2F (Ada:A:soul)
    → parent = 0x85B1 (Ada:A)
      → parent = 0x82E3 (Ada)
        → parent = None (root)
```

### Downward: Children via BitpackedCSR — O(k)

```rust
// Zero-copy children slice (raw u16 addresses, no allocation)
pub fn children_raw(&self, addr: Addr) -> &[u16] {
    self.csr.as_ref().map(|c| c.children(addr)).unwrap_or(&[])
}
```

Children are stored in a **BitpackedCSR** (Compressed Sparse Row) for memory efficiency:

```rust
pub struct BitpackedCsr {
    offsets: Vec<u32>,   // 65K entries, one per address
    edges: Vec<u16>,     // Flat array of target addresses
    verbs: Vec<u16>,     // Parallel array of verb addresses
}
```

**Memory comparison** (~32K nodes, avg 2 children):
- Traditional `Vec<Vec<Addr>>`: ~2.5MB
- BitpackedCSR: ~256KB (10x more efficient)

### Auto-Linking via PARENT_OF

When a node is created with `write_dn_path()`, parent-child edges are auto-created:

```rust
pub fn write_dn_path(&mut self, path: &str, fingerprint: [...], rung: u8) -> Addr {
    // Segment-by-segment creation
    // For each segment, auto-link parent -> child via PARENT_OF verb
    if let Some(parent_addr) = current_parent {
        if let Some(parent_of) = self.verb("PARENT_OF") {
            self.link(parent_addr, parent_of, addr);
        }
    }
}
```

---

## Horizontal Traversal (Siblings and Edges)

### Sibling Discovery

Siblings share the same parent:

```rust
pub fn siblings(&self, addr: Addr) -> impl Iterator<Item = Addr> + '_ {
    let parent = self.parent(addr);           // O(1): parent field
    let parent_of = self.verb("PARENT_OF");   // O(1): surface lookup

    parent
        .into_iter()
        .flat_map(|p| {
            parent_of.into_iter()
                .flat_map(|verb| self.traverse(p, verb))  // O(k): k children
        })
        .filter(|&a| a != addr)                            // Exclude self
}
```

### BindEdge — Semantic Edge with Binding

```rust
pub struct BindEdge {
    pub from: Addr,                                      // Source node
    pub to: Addr,                                        // Target node
    pub verb: Addr,                                      // Relationship verb
    pub fingerprint: [u64; FINGERPRINT_WORDS],           // from XOR verb XOR to
    pub weight: f32,                                     // Edge weight
}
```

Edge fingerprints are computed via XOR binding:
```rust
edge.fingerprint[i] = from_fp[i] ^ verb_fp[i] ^ to_fp[i];
```

This is a holographic binding — the edge fingerprint resonates with all three components, enabling similarity search on relationships.

---

## Verb Prefix (0x07)

Tree-structural verbs occupy slots 0x20-0x27 in PREFIX_VERBS:

| Slot | Verb | Direction | Purpose |
|------|------|-----------|---------|
| 0x0720 | PARENT_OF | parent -> child | Downward edges |
| 0x0721 | CHILD_OF | child -> parent | Reverse navigation |
| 0x0722 | SIBLING_OF | sibling <-> sibling | Horizontal |
| 0x0723 | ANCESTOR_OF | transitive up | Multi-hop upward |
| 0x0724 | DESCENDANT_OF | transitive down | Multi-hop downward |
| 0x0725 | ROOT_OF | any -> root | Jump to root |
| 0x0726 | NEXT_SIBLING | ordered sibling | Arrow adjacency |
| 0x0727 | PREV_SIBLING | ordered sibling | Bidirectional |

---

## CogRedis DN.* Commands

The Redis-compatible interface for DN tree operations:

```
DN.GET <path>                      Get node at path
DN.SET <path> <value> [RUNG r]    Create with parent chain + rung
DN.PARENT <path>                   Get parent path (O(1) string op)
DN.CHILDREN <path>                 List children (CSR zero-copy)
DN.ANCESTORS <path>                All ancestors (O(1) per hop)
DN.SIBLINGS <path>                 Siblings (same parent)
DN.DEPTH <path>                    Tree depth (O(1) field)
DN.RUNG <path>                     Access rung R0-R9 (O(1) field)
DN.TREE <path> [DEPTH n]           BFS traversal to depth n
```

---

## CSR Rebuild and Lazy Evaluation

The CSR is rebuilt lazily after edge modifications:

```rust
pub fn link(&mut self, from: Addr, verb: Addr, to: Addr) -> usize {
    let mut edge = BindEdge::new(from, verb, to);
    // ... bind fingerprints ...
    self.edge_out[from.0 as usize].push(idx);
    self.edge_in[to.0 as usize].push(idx);
    self.csr_dirty = true;   // Mark dirty, don't rebuild yet
    self.edges.push(edge);
    idx
}

pub fn rebuild_csr(&mut self) {
    if self.csr_dirty || self.csr.is_none() {
        self.csr = Some(BitpackedCsr::build_from_edges(&self.edges));
        self.csr_dirty = false;
    }
}
```

This allows batch edge creation without per-edge rebuild overhead.

---

## Orchestration Prefixes (0x0C-0x0F)

The orchestration layer uses the reserved surface prefixes:

### 0x0C: Agent Registry
```
0x0C:00  Agent card "researcher"  → identity_fingerprint()
0x0C:01  Agent card "writer"      → identity_fingerprint()
0x0C:80  Persona fingerprint for slot 0 → persona.to_fingerprint()
0x0C:81  Persona fingerprint for slot 1 → persona.to_fingerprint()
```

### 0x0D: Thinking Templates
```
0x0D:00  Base style "analytical"  → modulation_fingerprint()
0x0D:01  Base style "convergent"  → modulation_fingerprint()
...
0x0D:0B  Base style "metacognitive"
0x0D:0C  Custom "deep_research"   → base + overrides
0x0D:0D  Custom "brainstorm"      → base + overrides
```

### 0x0E: Agent Blackboards
```
0x0E:00  Blackboard for agent slot 0 → state_fingerprint()
0x0E:01  Blackboard for agent slot 1 → state_fingerprint()
```

### 0x0F: A2A Channels
```
0x0F:XX  Channel hash(sender=2, receiver=5) → XOR-composed messages
0x0F:YY  Channel hash(sender=5, receiver=2) → different channel (asymmetric)
```

---

## Traversal Patterns

### Pattern 1: Walk root to leaf (vertical down)
```rust
let mut frontier = vec![root_addr];
while let Some(node) = frontier.pop() {
    for &child_raw in bind_space.children_raw(node) {
        frontier.push(Addr(child_raw));
    }
}
```

### Pattern 2: Walk leaf to root (vertical up)
```rust
let mut current = leaf_addr;
while let Some(parent) = bind_space.parent(current) {
    // Process parent
    current = parent;
}
```

### Pattern 3: Find siblings (horizontal)
```rust
let siblings = bind_space.siblings(addr);
for sibling in siblings {
    // Process sibling
}
```

### Pattern 4: N-hop traversal via verb
```rust
let results = bind_space.traverse_n_hops(start, verb, max_hops);
// Returns Vec<(hop_count, Addr)>
```

### Pattern 5: HDR similarity search in zone
```rust
// Find similar agents (search only 0x0C prefix)
let query_fp = agent.identity_fingerprint();
let results = hdr_index.search_in_prefix(PREFIX_AGENTS, &query_fp, threshold);
```

---

## Performance Characteristics

| Operation | Complexity | Method |
|-----------|-----------|--------|
| Read node | O(1) | Array index: prefix->chunk, slot->offset |
| Get parent | O(1) | Direct field access |
| Get ancestors | O(d) | Iterate parent chain (d = depth) |
| Get children | O(k) | CSR slice (k = child count) |
| Get siblings | O(k) | Parent + children traversal |
| DN path -> address | O(1) | Hash + modulo to node range |
| Parent path | O(1) | String rfind(':') |
| Edge creation | O(1) | Array push + mark dirty |
| CSR rebuild | O(E) | Once after batch edges |
| Hamming similarity | O(1) | SIMD: ~2ns (AVX-512), ~4ns (AVX2) |

---

## Relationship Between Zones

```
SURFACE (4,096 addresses)                    FLUID (28,672 addresses)
  Verbs at 0x07:XX                             BitpackedCSR edges
  PARENT_OF @ 0x0720  ────referenced by────►   edges[] / verbs[]
  Agents at 0x0C:XX                                   │
  Templates at 0x0D:XX                               contains
                                                      │
                                              NODES (32,768 addresses)
                                                DN Tree nodes
                                                parent pointer ──► O(1) upward
                                                depth field
                                                rung field (R0-R9)
                                                fingerprint (10K bits)
```
