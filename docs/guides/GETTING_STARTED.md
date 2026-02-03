# Getting Started with Ladybug-RS

This guide covers building, running, and using Ladybug-RS.

## Prerequisites

- **Rust**: 1.70+ (`rustup update stable`)
- **System**: Linux, macOS, or Windows with WSL2
- **Optional**: AVX2/AVX-512 CPU for SIMD acceleration

## Building

### Basic Build

```bash
git clone https://github.com/AdaWorldAPI/ladybug-rs.git
cd ladybug-rs

# Debug build
cargo build

# Release build (optimized)
cargo build --release
```

### Feature Flags

```bash
# Full feature set
cargo build --release --features "simd,parallel,codebook,hologram,quantum"

# With Arrow Flight server
cargo build --release --features "flight"

# Minimal build
cargo build --release
```

| Feature | Description |
|---------|-------------|
| `simd` | AVX-512/AVX2/NEON SIMD acceleration |
| `parallel` | Rayon parallel processing |
| `codebook` | Codebook-based encoding |
| `hologram` | Holographic memory operations |
| `quantum` | Quantum-style operators |
| `flight` | Arrow Flight gRPC server |
| `lancedb` | LanceDB storage backend |

## Running Tests

```bash
# All tests
cargo test

# With features
cargo test --features "simd,parallel,codebook,hologram,quantum"

# Specific module
cargo test storage::bind_space
cargo test search::hdr_cascade
```

## Starting the Server

### Flight Server

```bash
# Build with flight feature
cargo build --release --features "flight"

# Run (default port 50051)
./target/release/ladybug-server

# Custom port
FLIGHT_PORT=9999 ./target/release/ladybug-server
```

### Redis-Compatible Server

```bash
./target/release/ladybug-redis --port 6379
```

## Quick Usage Examples

### Rust API

```rust
use ladybug::storage::{BindSpace, Addr};
use ladybug::core::Fingerprint;

// Create BindSpace
let mut space = BindSpace::new();

// Encode and bind
let fp = Fingerprint::from_content("hello world");
let addr = Addr::new(0x80, 0x01); // Node zone
space.write(addr, fp);

// Read back
if let Some(node) = space.read(addr) {
    println!("Fingerprint: {:?}", node.fingerprint);
}

// Search
use ladybug::search::HdrIndex;
let mut index = HdrIndex::new();
index.add(&fp);
let results = index.search(&query_fp, 10);
```

### Python via Flight

```python
import pyarrow.flight as flight

# Connect
client = flight.connect("grpc://localhost:50051")

# Encode action
action = flight.Action("encode", b'{"text": "hello world"}')
result = list(client.do_action(action))[0]
print(result.body.to_pybytes())

# Stream fingerprints
ticket = flight.Ticket(b"all")
reader = client.do_get(ticket)
table = reader.read_all()
print(table.to_pandas())
```

### Redis CLI

```bash
redis-cli -p 6379

# Basic operations
SET concept:hello <fingerprint_hex>
GET concept:hello

# Cognitive operations
RESONATE <query_hex> 10 2000
ENCODE "hello world"
HAMMING <fp1> <fp2>
```

## Architecture Overview

```
┌─────────────────────────────────────────────┐
│              Client (Claude/MCP)            │
└─────────────────────────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
┌───────────────┐         ┌───────────────┐
│ Arrow Flight  │         │ Redis Proto   │
│  (gRPC:50051) │         │   (:6379)     │
└───────────────┘         └───────────────┘
        │                           │
        └─────────────┬─────────────┘
                      ▼
        ┌─────────────────────────┐
        │       CogRedis          │
        │  (Command Executor)     │
        └─────────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
┌───────────────┐         ┌───────────────┐
│   BindSpace   │         │   HdrIndex    │
│ (O(1) Array)  │         │ (HDR Cascade) │
└───────────────┘         └───────────────┘
```

## Memory Zones

| Zone | Prefix | Addresses | Purpose |
|------|--------|-----------|---------|
| Surface | 0x00-0x0F | 4,096 | Query language ops |
| Fluid | 0x10-0x7F | 28,672 | Edges, TTL memory |
| Nodes | 0x80-0xFF | 32,768 | Persistent concepts |

## Next Steps

- [Architecture Overview](../architecture/OVERVIEW.md)
- [Flight API Reference](../api/FLIGHT_ENDPOINTS.md)
- [Redis Commands](../api/REDIS_COMMANDS.md)
- [MCP Actions](../api/MCP_ACTIONS.md)
