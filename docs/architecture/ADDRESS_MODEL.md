# 8+8 Address Model

The Ladybug-RS address model uses 16-bit addresses divided into prefix:slot (8:8).

## Address Structure

```
┌─────────────────────────────────────────────────────────┐
│                    16-bit Address                        │
├─────────────────────────┬───────────────────────────────┤
│   PREFIX (high byte)    │      SLOT (low byte)          │
│        0x00-0xFF        │         0x00-0xFF             │
└─────────────────────────┴───────────────────────────────┘
```

## Rust Implementation

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Addr(pub u16);

impl Addr {
    /// Create from prefix and slot
    #[inline(always)]
    pub fn new(prefix: u8, slot: u8) -> Self {
        Self(((prefix as u16) << 8) | (slot as u16))
    }

    /// Get prefix (high byte)
    #[inline(always)]
    pub fn prefix(&self) -> u8 {
        (self.0 >> 8) as u8
    }

    /// Get slot (low byte)
    #[inline(always)]
    pub fn slot(&self) -> u8 {
        (self.0 & 0xFF) as u8
    }
}
```

## Zone Mapping

| Zone | Prefix Range | Addresses | Purpose |
|------|-------------|-----------|---------|
| Surface | 0x00-0x0F | 4,096 | Query language operations |
| Fluid | 0x10-0x7F | 28,672 | Edges, context, TTL memory |
| Nodes | 0x80-0xFF | 32,768 | Persistent concepts |

## Surface Zone Prefixes

| Prefix | Query Language |
|--------|---------------|
| 0x00 | Lance (vector) |
| 0x01 | SQL |
| 0x02 | Cypher (graph) |
| 0x03 | GraphQL |
| 0x04 | NARS (logic) |
| 0x05 | Causal |
| 0x06 | Meta |
| 0x07 | Verbs |
| 0x08 | Concepts |
| 0x09 | Qualia |
| 0x0A | Memory |
| 0x0B | Learning |
| 0x0C-0x0F | Reserved |

## Performance

- **Address decode**: 3-5 CPU cycles
- **No HashMap**: Direct array indexing
- **No FPU**: Pure integer operations
- **Works everywhere**: Embedded, WASM, ARM

## Example

```rust
// Create address in node zone
let addr = Addr::new(0x80, 0x42);  // 0x8042

// Decompose
assert_eq!(addr.prefix(), 0x80);
assert_eq!(addr.slot(), 0x42);

// Direct array access
let prefix_idx = addr.prefix() as usize;
let slot_idx = addr.slot() as usize;
let node = &bind_space.chunks[prefix_idx][slot_idx];
```
