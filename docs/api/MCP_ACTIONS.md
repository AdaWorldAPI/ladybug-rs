# MCP Tool Actions

Ladybug-RS exposes cognitive operations via Arrow Flight DoAction for MCP (Model Context Protocol) integration.

## Action Format

```
Request:
  Action {
    type: "action_name",
    body: JSON bytes
  }

Response:
  Result { body: JSON bytes }
```

## Available Actions

### encode

Encode text or data to a 10K-bit fingerprint.

**Request:**
```json
{
  "text": "hello world",
  "style": "semantic"
}
```

Or binary data:
```json
{
  "data": "base64_encoded_bytes",
  "style": "binary"
}
```

**Response:**
```json
{
  "fingerprint": "hex_encoded_1248_bytes",
  "bits_set": 4992
}
```

**Styles:**
- `semantic` - Text semantic encoding (default)
- `binary` - Raw binary encoding
- `sparse` - Sparse distributed representation

---

### bind

Bind a fingerprint to a BindSpace address.

**Request:**
```json
{
  "address": 32768,
  "fingerprint": "hex_encoded_1248_bytes",
  "label": "my_concept"
}
```

**Response:**
```json
{
  "success": true,
  "address": 32768,
  "zone": "node"
}
```

---

### read

Read a node from a BindSpace address.

**Request:**
```json
{
  "address": 32768
}
```

**Response:**
```json
{
  "address": 32768,
  "fingerprint": "hex_encoded_1248_bytes",
  "label": "my_concept",
  "zone": "node",
  "access_count": 42,
  "created_at": 1706889600
}
```

Or if not found:
```json
{
  "error": "not_found",
  "address": 32768
}
```

---

### resonate

Find similar fingerprints using HDR cascade search.

**Request:**
```json
{
  "query": "hex_encoded_1248_bytes",
  "k": 10,
  "threshold": 2000
}
```

**Response:**
```json
{
  "results": [
    {
      "address": 32768,
      "distance": 450,
      "similarity": 0.955,
      "label": "similar_concept"
    }
  ],
  "count": 10,
  "search_time_ns": 7500
}
```

---

### hamming

Compute Hamming distance between two fingerprints.

**Request:**
```json
{
  "a": "hex_encoded_1248_bytes",
  "b": "hex_encoded_1248_bytes"
}
```

**Response:**
```json
{
  "distance": 1250,
  "similarity": 0.875,
  "max_distance": 9984
}
```

---

### xor_bind

XOR bind two fingerprints (holographic composition).

**Request:**
```json
{
  "a": "hex_encoded_1248_bytes",
  "b": "hex_encoded_1248_bytes"
}
```

**Response:**
```json
{
  "result": "hex_encoded_1248_bytes",
  "bits_set": 5100
}
```

---

### stats

Get BindSpace statistics.

**Request:**
```json
{}
```

**Response:**
```json
{
  "surface_count": 1024,
  "fluid_count": 8192,
  "node_count": 16384,
  "total_count": 25600,
  "surface_capacity": 4096,
  "fluid_capacity": 28672,
  "node_capacity": 32768
}
```

## Error Responses

All actions return errors in this format:

```json
{
  "error": "error_type",
  "message": "Human readable description"
}
```

**Error Types:**
- `invalid_request` - Malformed JSON or missing fields
- `invalid_address` - Address out of range
- `invalid_fingerprint` - Wrong fingerprint size
- `not_found` - Address not populated
- `internal` - Server error

## Usage with Claude/MCP

These actions are designed for MCP tool integration:

```typescript
// MCP tool definition
{
  name: "ladybug_resonate",
  description: "Find similar concepts in cognitive memory",
  inputSchema: {
    type: "object",
    properties: {
      query: { type: "string", description: "Hex-encoded fingerprint" },
      k: { type: "number", default: 10 },
      threshold: { type: "number", default: 2000 }
    }
  }
}
```
