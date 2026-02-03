# LadybugDB MCP Connector

Model Context Protocol (MCP) connector for LadybugDB, enabling AI assistants to interact with the cognitive database.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Claude / AI Client                            │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
       ┌──────────┐    ┌──────────┐    ┌──────────┐
       │ JSON-RPC │    │  Arrow   │    │ Streaming│
       │   MCP    │    │  Flight  │    │  gRPC    │
       └──────────┘    └──────────┘    └──────────┘
              │               │               │
              └───────────────┼───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LadybugDB Server                              │
├─────────────────────────────────────────────────────────────────┤
│  BindSpace  │  HDR Cascade  │  DataFusion  │  Sigma-10 Membrane │
└─────────────────────────────────────────────────────────────────┘
```

## Protocol Options

### 1. JSON-RPC MCP (Standard)

Compatible with existing MCP tooling. Best for:
- Simple queries
- Small payloads
- Maximum compatibility

```json
{
  "method": "tools/call",
  "params": {
    "name": "ladybug_encode",
    "arguments": {
      "text": "hello world",
      "style": "balanced"
    }
  }
}
```

### 2. Arrow Flight MCP (High Performance)

Zero-copy binary transfer. Best for:
- Bulk fingerprint operations
- Streaming search results
- 10-100x faster than JSON

```python
# Python client
import pyarrow.flight as flight

client = flight.connect("grpc://localhost:50051")

# Stream search results as they're found
ticket = flight.Ticket(b"topk:aabbccdd...:10")
reader = client.do_get(ticket)
for batch in reader:
    process_batch(batch)  # RecordBatch with fingerprints
```

### 3. Streaming gRPC (Real-time)

Bidirectional streaming for real-time applications:
- Live similarity monitoring
- Continuous fingerprint ingestion
- Real-time resonance detection

## MCP Tools

### `ladybug_encode`
Encode text/data to 10K-bit fingerprint via Sigma-10 membrane.

**Arguments:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| text | string | No* | Text to encode |
| data | string | No* | Hex-encoded binary data |
| style | string | No | "creative", "balanced", "precise" |

*One of `text` or `data` required.

**Returns:**
```json
{
  "fingerprint": "base64...",
  "bits_set": 4987,
  "encoding_style": "balanced"
}
```

### `ladybug_bind`
Bind fingerprint to BindSpace address.

**Arguments:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| address | integer | Yes | 16-bit address (0x0000-0xFFFF) |
| fingerprint | string | Yes | Base64-encoded fingerprint |
| label | string | No | Optional label |

### `ladybug_read`
Read node from BindSpace address.

**Arguments:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| address | integer | Yes | 16-bit address |

**Returns:**
```json
{
  "address": 32896,
  "fingerprint": "base64...",
  "label": "example",
  "zone": "node"
}
```

### `ladybug_resonate`
Find similar fingerprints via HDR cascade search.

**Arguments:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| query | string | Yes | Base64-encoded query fingerprint |
| k | integer | No | Number of results (default: 10) |
| threshold | integer | No | Max Hamming distance |

**Returns (streaming):**
```json
{
  "results": [
    {
      "address": 32897,
      "fingerprint": "base64...",
      "distance": 234,
      "similarity": 0.9766,
      "cascade_level": 2
    }
  ],
  "cascade_stats": {
    "l0_candidates": 10000,
    "l1_candidates": 1000,
    "l2_candidates": 100,
    "final_candidates": 10
  }
}
```

### `ladybug_hamming`
Compute Hamming distance between two fingerprints.

**Arguments:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| a | string | Yes | First fingerprint (base64) |
| b | string | Yes | Second fingerprint (base64) |

**Returns:**
```json
{
  "distance": 234,
  "similarity": 0.9766,
  "max_bits": 10000
}
```

### `ladybug_xor_bind`
XOR bind two fingerprints (holographic composition).

**Arguments:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| a | string | Yes | First fingerprint |
| b | string | Yes | Second fingerprint |

**Returns:**
```json
{
  "fingerprint": "base64...",
  "bits_set": 5012
}
```

### `ladybug_stats`
Get BindSpace statistics.

**Returns:**
```json
{
  "total_nodes": 1234,
  "surface_nodes": 100,
  "fluid_nodes": 500,
  "node_space_nodes": 634
}
```

## Arrow Flight Endpoints

### GetFlightInfo
Returns schema for fingerprint tables.

### DoGet
Stream fingerprints or search results.

**Ticket formats:**
- `all` - Stream all fingerprints
- `search:<query_hex>:<threshold>` - Similarity search
- `topk:<query_hex>:<k>` - Top-K search

### DoPut
Ingest fingerprints from Arrow RecordBatch (zero-copy).

### DoAction
Execute MCP tools via Flight.

**Action types:** `encode`, `bind`, `read`, `resonate`, `hamming`, `xor_bind`, `stats`

### ListActions
Returns available MCP tools.

## Performance Comparison

| Operation | JSON-RPC | Arrow Flight |
|-----------|----------|--------------|
| Single fingerprint encode | 5ms | 2ms |
| Bulk encode (1000) | 5000ms | 50ms |
| Search TopK=10 | 10ms | 3ms |
| Search TopK=1000 | 500ms | 15ms |
| Fingerprint transfer | 1KB (base64) | 1.25KB (binary, zero-copy) |

## Installation

### Server Setup

```bash
# Enable flight feature
cargo build --release --features "flight"

# Run with both JSON and Flight endpoints
./target/release/ladybug-server --http-port 8080 --flight-port 50051
```

### Python Client

```bash
pip install ladybugdb[flight]

# Or with Arrow Flight support
pip install pyarrow
```

## Usage Examples

### Python (JSON-RPC)

```python
from ladybugdb import LadybugDB

db = LadybugDB("http://localhost:8080")

# Encode
fp = db.fingerprint("hello world")
print(f"Popcount: {fp.popcount}")

# Search
results = db.topk("similar query", k=10)
for r in results:
    print(f"{r.id}: {r.similarity:.2%}")
```

### Python (Arrow Flight)

```python
from ladybugdb import LadybugDB

# Connect with Flight transport
db = LadybugDB("grpc://localhost:50051", transport="flight")

# Stream search results
for batch in db.search_stream("query", k=100):
    for row in batch:
        print(row["similarity"])
```

### TypeScript/Node.js

```typescript
import { LadybugMCP } from 'ladybug-mcp';

const mcp = new LadybugMCP({
  server: 'http://localhost:8080',
  transport: 'json'  // or 'flight'
});

// Use as MCP tool provider
const result = await mcp.call('ladybug_encode', {
  text: 'hello world',
  style: 'balanced'
});
```

## Related Repositories

- **ladybug-rs**: Core database implementation (this repo)
- **ladybug-mcp**: MCP server implementation
- **ladybug-vsa**: Python SDK

## License

Apache-2.0
