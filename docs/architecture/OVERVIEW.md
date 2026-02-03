# Architecture Overview

Ladybug-RS is a cognitive substrate that provides a unified interface for multiple query languages over a shared address space.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          External Clients                                   │
│                    (Claude, MCP Tools, Applications)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
            ┌───────────┐   ┌───────────┐   ┌───────────┐
            │  Arrow    │   │   Redis   │   │   HTTP    │
            │  Flight   │   │  Protocol │   │   REST    │
            │ (gRPC)    │   │  (:6379)  │   │  (:8080)  │
            └───────────┘   └───────────┘   └───────────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    ▼
            ┌─────────────────────────────────────────────────────────────────┐
            │                      CogRedis                                   │
            │              (Command Executor Layer)                           │
            ├─────────────────────────────────────────────────────────────────┤
            │  • Parses Redis-style commands                                  │
            │  • Routes to CAM operations by prefix                           │
            │  • Manages transactions and batching                            │
            └─────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
            ┌───────────┐   ┌───────────┐   ┌───────────┐
            │ BindSpace │   │ HdrIndex  │   │  LanceDB  │
            │  (O(1))   │   │ (Search)  │   │ (Persist) │
            └───────────┘   └───────────┘   └───────────┘
```

## Core Components

### 1. Protocol Layer

| Protocol | Port | Purpose |
|----------|------|---------|
| Arrow Flight | 50051 | Zero-copy streaming, MCP integration |
| Redis | 6379 | Cognitive Redis commands |
| HTTP | 8080 | REST API (optional) |

### 2. Command Executor (CogRedis)

The CogRedis layer translates commands into operations:

```rust
// Command routing
match command {
    "GET" => self.bind_space.read(addr),
    "SET" => self.bind_space.write(addr, fp),
    "RESONATE" => self.hdr_index.search(query, k),
    "ENCODE" => encode_to_fingerprint(text),
    // CAM operations by prefix
    _ => self.execute_cam_op(prefix, slot, args),
}
```

### 3. Storage Layer

**BindSpace** - O(1) array-indexed storage:
- 65,536 addresses (16-bit)
- Three zones: Surface, Fluid, Nodes
- Direct array access (3-5 cycles)

**HdrIndex** - HDR Cascade Search:
- Multi-level filtering
- ~7ns per candidate
- Binary fingerprint comparison

**LanceDB** - Persistent storage:
- Arrow-native format
- Vector indexing
- DataFusion SQL integration

### 4. Compute Layer

**SIMD Acceleration**:
- AVX-512 (8×64-bit popcount)
- AVX2 (4×64-bit)
- NEON (ARM)
- Scalar fallback

**Fingerprint Operations**:
- 10K-bit binary vectors
- Hamming distance
- XOR binding (holographic)

## Data Flow

### Read Path

```
Client Request
    │
    ▼
Protocol Parse (Flight/Redis/HTTP)
    │
    ▼
CogRedis.execute(command)
    │
    ▼
BindSpace.read(addr) ──► O(1) array lookup
    │
    ▼
Response Encode (Arrow IPC / RESP)
    │
    ▼
Client Response
```

### Search Path

```
Client Request (RESONATE query k)
    │
    ▼
Protocol Parse
    │
    ▼
CogRedis.resonate(query, k)
    │
    ▼
HdrIndex.search(query, k)
    │
    ├──► Level 0: 1-bit sketch (90% filtered)
    ├──► Level 1: 4-bit count (90% of survivors)
    ├──► Level 2: 8-bit count (90% of survivors)
    └──► Level 3: Full popcount (exact)
    │
    ▼
Top-k results with distances
    │
    ▼
Response (addresses + similarities)
```

### Write Path

```
Client Request (SET/BIND)
    │
    ▼
Protocol Parse
    │
    ▼
CogRedis.execute(SET addr fp)
    │
    ▼
BindSpace.write(addr, fp)
    │
    ├──► HdrIndex.add(fp) [if indexed]
    └──► WAL.append(entry) [if durable]
    │
    ▼
Response (OK)
```

## Feature Flags

| Flag | Purpose |
|------|---------|
| `simd` | SIMD-accelerated Hamming distance |
| `parallel` | Rayon parallel processing |
| `flight` | Arrow Flight gRPC server |
| `lancedb` | LanceDB persistence |
| `codebook` | Codebook-based encoding |
| `hologram` | Holographic memory ops |
| `quantum` | Quantum-style operators |

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Address decode | 3-5 ns | Shift + mask |
| BindSpace read | 5-10 ns | Array index |
| Hamming (AVX-512) | 2 ns | 8 fingerprints parallel |
| Hamming (AVX2) | 4 ns | 4 fingerprints parallel |
| Hamming (scalar) | 50 ns | Portable fallback |
| HDR search (per candidate) | 7 ns | Multi-level filter |
| Flight stream (per batch) | 100 µs | Arrow IPC encoding |

## Design Principles

1. **Integer-first**: No FPU for addressing, integers throughout
2. **Zero-copy**: Arrow IPC for data transfer
3. **Adaptive compute**: Best SIMD for platform
4. **Universal addressing**: All languages share bind space
5. **TTL-governed memory**: Fluid zone auto-expires
