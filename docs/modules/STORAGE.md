# Storage Module

The storage module provides the core data structures for Ladybug-RS.

## Components

### BindSpace (`bind_space.rs`)

Universal O(1) indexed storage for 65,536 addresses.

```rust
pub struct BindSpace {
    surface: Vec<[Option<BindNode>; 256]>,  // 0x00-0x0F
    fluid: Vec<[Option<BindNode>; 256]>,    // 0x10-0x7F
    nodes: Vec<[Option<BindNode>; 256]>,    // 0x80-0xFF
}
```

**Key Operations:**

| Method | Complexity | Description |
|--------|------------|-------------|
| `read(addr)` | O(1) | Read node at address |
| `write(addr, node)` | O(1) | Write node to address |
| `delete(addr)` | O(1) | Remove node at address |
| `stats()` | O(1) | Get zone statistics |

**Address Decomposition:**

```rust
impl Addr {
    pub fn new(prefix: u8, slot: u8) -> Self {
        Self(((prefix as u16) << 8) | (slot as u16))
    }

    pub fn prefix(&self) -> u8 { (self.0 >> 8) as u8 }
    pub fn slot(&self) -> u8 { (self.0 & 0xFF) as u8 }
}
```

### BindNode

The universal data transfer object (DTO):

```rust
pub struct BindNode {
    pub fingerprint: [u64; FINGERPRINT_WORDS],  // 156 words = 9984 bits
    pub label: Option<String>,                   // Human-readable name
    pub qidx: u8,                                // Qualia index
    pub access_count: u32,                       // Usage tracking
    pub created_at: u32,                         // Timestamp
    pub ttl: Option<u32>,                        // Time-to-live (fluid zone)
}
```

### CogRedis (`cog_redis.rs`)

Redis-compatible command executor with cognitive extensions.

**Standard Commands:**
- `GET`, `SET`, `DEL`, `EXISTS`
- `KEYS`, `SCAN`
- `PING`, `INFO`

**Cognitive Extensions:**
- `RESONATE` - Similarity search
- `ENCODE` - Text to fingerprint
- `BIND` - Write with label
- `HAMMING` - Distance computation
- `XOR` - Holographic binding

**CAM Operation Routing:**

```rust
pub fn execute_cam(&mut self, prefix: u8, slot: u8, args: &[Fingerprint]) -> CamResult {
    match prefix {
        0x00 => self.execute_lance_op(slot, args),
        0x01 => self.execute_sql_op(slot, args),
        0x02 => self.execute_cypher_op(slot, args),
        0x03 => self.execute_graphql_op(slot, args),
        0x04 => self.execute_nars_op(slot, args),
        0x05 => self.execute_causal_op(slot, args),
        // ...
    }
}
```

### LanceDB Integration (`lance.rs`)

Arrow-native persistent storage via LanceDB.

**Features:**
- DataFusion SQL queries
- Vector similarity search
- Zero-copy Arrow integration
- Automatic schema inference

```rust
pub struct LanceStorage {
    db: lancedb::Database,
    table: Option<lancedb::Table>,
}

impl LanceStorage {
    pub async fn create_table(&mut self, name: &str, schema: SchemaRef) -> Result<()>;
    pub async fn insert(&self, batch: RecordBatch) -> Result<()>;
    pub async fn query(&self, sql: &str) -> Result<Vec<RecordBatch>>;
    pub async fn vector_search(&self, query: &[f32], k: usize) -> Result<Vec<(usize, f32)>>;
}
```

## Constants

```rust
pub const FINGERPRINT_WORDS: usize = 156;    // 156 * 64 = 9984 bits
pub const FINGERPRINT_BITS: usize = 9984;
pub const SURFACE_PREFIXES: usize = 16;       // 0x00-0x0F
pub const FLUID_PREFIXES: usize = 112;        // 0x10-0x7F
pub const NODE_PREFIXES: usize = 128;         // 0x80-0xFF
```

## Usage Example

```rust
use ladybug::storage::{BindSpace, Addr, BindNode};

// Create storage
let mut space = BindSpace::new();

// Write a node
let addr = Addr::new(0x80, 0x01);
let mut node = BindNode::new([0u64; 156]);
node.label = Some("my_concept".to_string());
space.write(addr, node);

// Read back
if let Some(n) = space.read(addr) {
    println!("Label: {:?}", n.label);
}

// Get stats
let stats = space.stats();
println!("Nodes: {}", stats.node_count);
```

## Thread Safety

BindSpace is wrapped in `parking_lot::RwLock` for concurrent access:

```rust
let bind_space = Arc::new(RwLock::new(BindSpace::new()));

// Read access (multiple readers)
let space = bind_space.read();
let node = space.read(addr);

// Write access (exclusive)
let mut space = bind_space.write();
space.write(addr, node);
```
