# Flight Module

Arrow Flight gRPC server for zero-copy MCP integration.

## Overview

The flight module provides high-performance streaming of fingerprints and search results using Apache Arrow Flight protocol.

```
src/flight/
├── mod.rs       # Module exports
├── server.rs    # LadybugFlightService (717 lines)
└── actions.rs   # MCP action handlers
```

## LadybugFlightService

Main service implementing Arrow Flight RPC:

```rust
pub struct LadybugFlightService {
    bind_space: Arc<RwLock<BindSpace>>,
    hdr_index: Arc<RwLock<HdrIndex>>,
}

impl FlightService for LadybugFlightService {
    // Stream types
    type DoGetStream = TonicStream<FlightData>;
    type DoPutStream = TonicStream<PutResult>;
    type DoActionStream = TonicStream<Result>;
    // ...
}
```

## Endpoints

### DoGet - Stream Fingerprints

Streams fingerprints based on ticket format:

| Ticket | Description |
|--------|-------------|
| `all` | All fingerprints (0x00-0xFF) |
| `surface` | Surface zone (0x00-0x0F) |
| `fluid` | Fluid zone (0x10-0x7F) |
| `nodes` | Node zone (0x80-0xFF) |
| `search:<hex>:<threshold>` | Similarity search |
| `topk:<hex>:<k>` | Top-k nearest |

### DoPut - Ingest Fingerprints

Accepts Arrow RecordBatch stream for bulk ingestion:

```rust
async fn do_put(&self, request: Request<Streaming<FlightData>>)
    -> Result<Response<Self::DoPutStream>, Status>
```

### DoAction - MCP Tools

Executes MCP-style actions:

| Action | Description |
|--------|-------------|
| `encode` | Text/data → fingerprint |
| `bind` | Store fingerprint at address |
| `read` | Read node from address |
| `resonate` | Similarity search |
| `hamming` | Compute distance |
| `xor_bind` | Holographic composition |
| `stats` | BindSpace statistics |

## Streaming Implementation

### stream_all_fingerprints

```rust
fn stream_all_fingerprints(
    bind_space: Arc<RwLock<BindSpace>>,
    prefix_range: RangeInclusive<u8>,
) -> impl Stream<Item = Result<FlightData, Status>>
```

Uses `stream::unfold` to iterate BindSpace by prefix, batching into RecordBatches of 1000 fingerprints.

### stream_search_results

```rust
fn stream_search_results(
    bind_space: Arc<RwLock<BindSpace>>,
    hdr_index: Arc<RwLock<HdrIndex>>,
    query: Vec<u8>,
    threshold: u32,
) -> impl Stream<Item = Result<FlightData, Status>>
```

**Key design**: All synchronous lock operations complete before async awaits to maintain Send safety.

```rust
// Synchronous: acquire locks, do search, build batch
let batch_result = {
    let hdr = hdr_index.read();
    let results = hdr.search(&query_fp, MAX_SEARCH_RESULTS);
    // ... build RecordBatch
};
// Locks released here

// Async: encode and stream
stream::once(async move {
    // FlightDataEncoderBuilder for Arrow IPC
}).flat_map(...)
```

## Schemas

### Fingerprint Schema

```rust
pub fn fingerprint_schema() -> SchemaRef {
    Schema::new(vec![
        Field::new("address", DataType::UInt16, false),
        Field::new("fingerprint", DataType::FixedSizeBinary(1248), false),
        Field::new("label", DataType::Utf8, true),
        Field::new("zone", DataType::Utf8, false),
        Field::new("distance", DataType::UInt32, true),
        Field::new("similarity", DataType::Float32, true),
    ])
}
```

### Search Result Schema

```rust
pub fn search_result_schema() -> SchemaRef {
    Schema::new(vec![
        Field::new("address", DataType::UInt16, false),
        Field::new("fingerprint", DataType::FixedSizeBinary(1248), false),
        Field::new("label", DataType::Utf8, true),
        Field::new("distance", DataType::UInt32, false),
        Field::new("similarity", DataType::Float32, false),
        Field::new("cascade_level", DataType::UInt8, false),
    ])
}
```

## Configuration

```rust
const BATCH_SIZE: usize = 1000;        // Fingerprints per batch
const MAX_SEARCH_RESULTS: usize = 10000; // Unbounded search limit
```

## Dependencies

```toml
[dependencies]
arrow-flight = { version = "57.2", features = ["flight-sql-experimental"] }
arrow-array = "48.0"
arrow-schema = "48.0"
arrow-ipc = "48.0"
tonic = "0.14"
prost = "0.14"
futures = "0.3"
parking_lot = "0.12"
```

## Usage

### Starting the Server

```rust
use ladybug::flight::LadybugFlightService;
use arrow_flight::flight_service_server::FlightServiceServer;

let bind_space = Arc::new(RwLock::new(BindSpace::new()));
let hdr_index = Arc::new(RwLock::new(HdrIndex::new()));

let service = LadybugFlightService::new(bind_space, hdr_index);

Server::builder()
    .add_service(FlightServiceServer::new(service))
    .serve("[::]:50051".parse()?)
    .await?;
```

### Python Client

```python
import pyarrow.flight as flight

client = flight.connect("grpc://localhost:50051")

# Stream all fingerprints
reader = client.do_get(flight.Ticket(b"all"))
table = reader.read_all()

# Similarity search
ticket = f"search:{query_hex}:2000".encode()
reader = client.do_get(flight.Ticket(ticket))

# MCP action
action = flight.Action("encode", b'{"text": "hello"}')
result = list(client.do_action(action))[0]
```

## Thread Safety

The service uses `parking_lot::RwLock` with careful lock scoping:

1. All lock acquisitions happen in synchronous blocks
2. Locks are released before any `.await` points
3. This ensures `Send` safety for async streams
