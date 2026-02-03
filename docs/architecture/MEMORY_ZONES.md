# Memory Zones

Ladybug-RS divides the 65,536 address space into three zones with distinct behaviors.

## Zone Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PREFIX (8-bit) : SLOT (8-bit)                          │
├─────────────────┬───────────────────────────────────────────────────────────┤
│  0x00-0x0F:XX   │  SURFACE (16 prefixes × 256 = 4,096)                      │
│                 │  Query language mappings, CAM operations                  │
│                 │  Persistent, no TTL                                       │
├─────────────────┼───────────────────────────────────────────────────────────┤
│  0x10-0x7F:XX   │  FLUID (112 prefixes × 256 = 28,672)                      │
│                 │  Edges, context selector, working memory                  │
│                 │  TTL governed, promote/demote                             │
├─────────────────┼───────────────────────────────────────────────────────────┤
│  0x80-0xFF:XX   │  NODES (128 prefixes × 256 = 32,768)                      │
│                 │  THE UNIVERSAL BIND SPACE                                 │
│                 │  All query languages hit the same addresses               │
└─────────────────┴───────────────────────────────────────────────────────────┘
```

## Surface Zone (0x00-0x0F)

**Purpose**: Query language operations and CAM (Content-Addressable Memory) mappings.

| Prefix | Language | Operations |
|--------|----------|------------|
| 0x00 | Lance | Vector similarity, ANN search |
| 0x01 | SQL | SELECT, JOIN, aggregate |
| 0x02 | Cypher | MATCH, path traversal |
| 0x03 | GraphQL | Field resolution |
| 0x04 | NARS | Truth value propagation |
| 0x05 | Causal | SEE/DO/IMAGINE |
| 0x06 | Meta | Schema operations |
| 0x07 | Verbs | Action predicates |

**Behavior**:
- Persistent (no TTL)
- Read-heavy workload
- CAM operations routed here

## Fluid Zone (0x10-0x7F)

**Purpose**: Working memory for edges, context, and transient data.

**Behavior**:
- TTL governed
- `tick()` evicts expired entries
- `crystallize()` promotes to node zone
- Context selector determines interpretation

```rust
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
```

## Node Zone (0x80-0xFF)

**Purpose**: Universal bind space for persistent concepts.

**Behavior**:
- All query languages hit same addresses
- Permanent storage (no TTL)
- `evaporate()` demotes to fluid zone

**Key insight**: The node zone is the "ground truth" that all query interfaces share. A concept at address 0x8042 is the same whether accessed via SQL, Cypher, or NARS.

## Lifecycle Operations

### Crystallize (Fluid → Node)

```rust
pub fn crystallize(&mut self, fluid_addr: Addr) -> Option<Addr> {
    let node = self.read(fluid_addr)?;
    let node_addr = self.allocate_node()?;
    self.write(node_addr, node.clone());
    self.delete(fluid_addr);
    Some(node_addr)
}
```

### Evaporate (Node → Fluid)

```rust
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

## Memory Layout

Each zone is backed by contiguous arrays for O(1) access:

```rust
pub struct BindSpace {
    // Surface: 16 chunks × 256 slots = 4,096
    surface: Vec<[Option<BindNode>; 256]>,

    // Fluid: 112 chunks × 256 slots = 28,672
    fluid: Vec<[Option<BindNode>; 256]>,

    // Nodes: 128 chunks × 256 slots = 32,768
    nodes: Vec<[Option<BindNode>; 256]>,
}
```
