# Redis Command Reference

Ladybug-RS implements Redis-compatible commands with cognitive extensions.

## Standard Commands

### GET

Read fingerprint at address.

```
GET <address>
```

**Response**: Hex-encoded fingerprint or `(nil)`

### SET

Write fingerprint to address.

```
SET <address> <fingerprint_hex>
```

**Response**: `OK`

### DEL

Delete node at address.

```
DEL <address>
```

**Response**: `(integer) 1` or `(integer) 0`

### EXISTS

Check if address is populated.

```
EXISTS <address>
```

**Response**: `(integer) 1` or `(integer) 0`

### KEYS

List addresses matching pattern.

```
KEYS <pattern>
```

**Pattern**: `*` (all), `80*` (prefix 0x80), etc.

### PING

Health check.

```
PING
```

**Response**: `PONG`

### INFO

Server statistics.

```
INFO
```

**Response**:
```
# Storage
surface_count:1024
fluid_count:8192
node_count:16384
total_addresses:65536

# Search
hdr_index_size:25600
```

## Cognitive Extensions

### ENCODE

Encode text to fingerprint.

```
ENCODE <text> [STYLE semantic|binary|sparse]
```

**Response**: Hex-encoded 1248-byte fingerprint

**Example**:
```
ENCODE "hello world"
ENCODE "hello world" STYLE semantic
```

### BIND

Write fingerprint with label.

```
BIND <address> <fingerprint_hex> [LABEL <name>]
```

**Response**: `OK`

**Example**:
```
BIND 32768 0a1b2c3d... LABEL "my_concept"
```

### RESONATE

Similarity search using HDR cascade.

```
RESONATE <query_hex> <k> [THRESHOLD <max_distance>]
```

**Response**: Array of `[address, distance, similarity]`

**Example**:
```
RESONATE 0a1b2c3d... 10 THRESHOLD 2000
```

### HAMMING

Compute Hamming distance between fingerprints.

```
HAMMING <fp1_hex> <fp2_hex>
```

**Response**: `(integer) <distance>`

### XOR

XOR-bind two fingerprints (holographic composition).

```
XOR <fp1_hex> <fp2_hex>
```

**Response**: Hex-encoded result fingerprint

### SIMILAR

Find fingerprints similar to address.

```
SIMILAR <address> <k> [THRESHOLD <max_distance>]
```

**Response**: Array of similar addresses with distances

## CAM Operations

Content-Addressable Memory operations routed by prefix.

### Surface Zone (0x00-0x0F)

#### 0x00: Lance Operations

```
CAM.LANCE.SEARCH <query_hex> <k>
CAM.LANCE.INSERT <address> <fingerprint>
CAM.LANCE.DELETE <address>
```

#### 0x01: SQL Operations

```
CAM.SQL.SELECT <fields> FROM <table> [WHERE ...]
CAM.SQL.INSERT INTO <table> VALUES (...)
CAM.SQL.UPDATE <table> SET ... WHERE ...
```

#### 0x02: Cypher Operations

```
CAM.CYPHER.MATCH (n)-[r]->(m) RETURN n,r,m
CAM.CYPHER.CREATE (n:Label {props})
CAM.CYPHER.PATH <start> <end> [MAX_DEPTH <n>]
```

#### 0x03: GraphQL Operations

```
CAM.GRAPHQL.QUERY { field { subfield } }
CAM.GRAPHQL.MUTATION { ... }
```

#### 0x04: NARS Operations

```
CAM.NARS.JUDGE <statement> <frequency> <confidence>
CAM.NARS.QUERY <statement>
CAM.NARS.REVISE <statement>
```

#### 0x05: Causal Operations

```
CAM.CAUSAL.SEE <fingerprint>
CAM.CAUSAL.DO <action> <fingerprint>
CAM.CAUSAL.IMAGINE <counterfactual>
```

## Transactions

### MULTI/EXEC

```
MULTI
SET 32768 0a1b2c...
SET 32769 0d1e2f...
EXEC
```

### WATCH

```
WATCH 32768
GET 32768
MULTI
SET 32768 <new_value>
EXEC
```

## Error Responses

```
-ERR unknown command 'FOO'
-ERR wrong number of arguments for 'GET' command
-ERR invalid address format
-ERR fingerprint must be 1248 bytes hex-encoded
-ERR address out of range
```

## Connection

### Default Ports

| Protocol | Port |
|----------|------|
| Redis | 6379 |
| Flight | 50051 |

### Authentication (if enabled)

```
AUTH <password>
```

## Examples

### Basic Usage

```bash
redis-cli -p 6379

# Encode and store
> ENCODE "artificial intelligence"
"0a1b2c3d4e5f..."

> BIND 32768 0a1b2c3d4e5f... LABEL "AI"
OK

# Search
> RESONATE 0a1b2c3d4e5f... 5
1) 1) (integer) 32768
   2) (integer) 0
   3) "1.0"
2) 1) (integer) 32800
   2) (integer) 1250
   3) "0.875"
```

### Python Client

```python
import redis

r = redis.Redis(host='localhost', port=6379)

# Encode
fp = r.execute_command('ENCODE', 'hello world')

# Bind
r.execute_command('BIND', 32768, fp, 'LABEL', 'greeting')

# Search
results = r.execute_command('RESONATE', fp, 10, 'THRESHOLD', 2000)
```
