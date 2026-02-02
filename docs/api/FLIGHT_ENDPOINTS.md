# Arrow Flight API Endpoints

Ladybug-RS exposes an Arrow Flight gRPC service for zero-copy fingerprint streaming.

## Service: LadybugFlightService

**Default Port**: 50051 (configurable)

## Endpoints

### Handshake

Establishes connection and returns protocol version.

```
Request: HandshakeRequest
Response: HandshakeResponse { protocol_version: 1, payload: "ladybug-flight-v1" }
```

### ListFlights

Returns available flight descriptors.

```
Request: Criteria (empty)
Response: Stream<FlightInfo>
  - all: Total fingerprints across all zones
  - surface: Surface zone (0x00-0x0F)
  - fluid: Fluid zone (0x10-0x7F)
  - nodes: Node zone (0x80-0xFF)
```

### GetFlightInfo

Get metadata about a specific flight.

```
Request: FlightDescriptor { path: ["all" | "surface" | "fluid" | "nodes"] }
Response: FlightInfo { schema, total_records }
```

### DoGet

Stream fingerprints or search results.

#### Ticket Formats

| Ticket | Description |
|--------|-------------|
| `all` | Stream all fingerprints |
| `surface` | Stream surface zone (0x00-0x0F) |
| `fluid` | Stream fluid zone (0x10-0x7F) |
| `nodes` | Stream node zone (0x80-0xFF) |
| `search:<hex>:<threshold>` | Similarity search within Hamming threshold |
| `topk:<hex>:<k>` | Top-k nearest neighbors |

#### Response Schema (Fingerprints)

| Field | Type | Description |
|-------|------|-------------|
| address | UInt16 | 16-bit BindSpace address |
| fingerprint | FixedSizeBinary(1248) | 156 * 8 bytes |
| label | Utf8 (nullable) | Human-readable label |
| zone | Utf8 | "surface", "fluid", or "node" |
| distance | UInt32 (nullable) | Hamming distance |
| similarity | Float32 (nullable) | 0.0-1.0 similarity |

#### Response Schema (Search Results)

| Field | Type | Description |
|-------|------|-------------|
| address | UInt16 | 16-bit BindSpace address |
| fingerprint | FixedSizeBinary(1248) | 156 * 8 bytes |
| label | Utf8 (nullable) | Human-readable label |
| distance | UInt32 | Hamming distance |
| similarity | Float32 | 0.0-1.0 similarity |
| cascade_level | UInt8 | HDR cascade level (0-3) |

### DoPut

Ingest fingerprints into BindSpace.

```
Request: Stream<FlightData> with fingerprint schema
Response: PutResult { app_metadata: {"ingested": N} }
```

### DoAction

Execute MCP tool actions.

See [MCP_ACTIONS.md](MCP_ACTIONS.md) for full action reference.

### ListActions

Returns available MCP actions.

```
Response: Stream<ActionType>
  - encode: Encode text/data to fingerprint
  - bind: Bind fingerprint to address
  - read: Read node from address
  - resonate: Find similar fingerprints
  - hamming: Compute Hamming distance
  - xor_bind: XOR bind two fingerprints
  - stats: Get BindSpace statistics
```

## Example Usage

### Python (pyarrow)

```python
import pyarrow.flight as flight

client = flight.connect("grpc://localhost:50051")

# Stream all fingerprints
ticket = flight.Ticket(b"all")
reader = client.do_get(ticket)
for batch in reader:
    print(batch.to_pandas())

# Similarity search
query_hex = "0a1b2c3d..."  # 1248 bytes hex-encoded
ticket = flight.Ticket(f"search:{query_hex}:1000".encode())
reader = client.do_get(ticket)
for batch in reader:
    print(batch.to_pandas())
```

### Rust (arrow-flight)

```rust
use arrow_flight::flight_service_client::FlightServiceClient;

let mut client = FlightServiceClient::connect("http://localhost:50051").await?;

// Stream all
let ticket = Ticket { ticket: b"all".to_vec().into() };
let mut stream = client.do_get(ticket).await?.into_inner();
while let Some(data) = stream.next().await {
    // Process FlightData
}
```

## Performance

- **Streaming**: Zero-copy Arrow IPC encoding
- **Batch Size**: 1000 fingerprints per RecordBatch
- **Search**: HDR cascade ~7ns per candidate
- **Max Results**: 10,000 for unbounded search
