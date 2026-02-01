# LadybugDB — Deployment & Consumer Guide

## Overview

LadybugDB is a pure-Rust cognitive database exposing 10,000-bit Hamming fingerprints via multiple query interfaces. This document covers deployment (Railway, Docker, Claude Code) and consumption (Python SDK, REST API, Redis protocol, SQL, Cypher).

---

## 1. Building

### Prerequisites

- Rust 1.75+ (`rustup install 1.78`)
- For Python bindings: `maturin` (`pip install maturin`)

### AVX-512 Build (Railway / Cloud Servers)

```bash
RUSTFLAGS="-C target-cpu=skylake-avx512 -C target-feature=+avx512f,+avx512vl,+avx512vpopcntdq,+avx512bw" \
  cargo build --release --bin ladybug-server
```

This enables hardware VPOPCNTDQ — **65M+ Hamming comparisons/sec** on a single core.

### AVX2 Build (Fallback)

```bash
RUSTFLAGS="-C target-cpu=haswell" cargo build --release --bin ladybug-server
```

### Native Build (Local Development Only)

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --bin ladybug-server
```

### Verify SIMD Level

```bash
curl http://localhost:8080/api/v1/simd
# → {"level":"avx512","avx512f":true,"avx512vpopcntdq":true,...}
```

---

## 2. Deployment

### 2a. Railway

**One-click**: Push this repo to a Railway project. Railway auto-detects the `Dockerfile` and `railway.toml`.

**Manual**:
```bash
# Install Railway CLI
npm i -g @railway/cli

# Link project
railway link

# Deploy
railway up
```

**Networking**:
- Public URL: `https://ladybugdb-production-XXXX.up.railway.app`
- Private (internal): `ladybugdb.railway.internal:8080` (for ada-unified/bighorn)
- The server auto-detects `RAILWAY_*` env vars and binds `0.0.0.0:8080`

**Environment Variables** (Railway dashboard):
```
PORT=8080                  # Railway convention (auto-set)
LADYBUG_DATA_DIR=/data     # Persistent volume mount
```

**Private Networking**: Other Railway services (ada-unified, bighorn) call:
```
http://ladybugdb.railway.internal:8080/api/v1/search/topk
```

### 2b. Docker (standalone)

```bash
# Build with AVX-512 (default)
docker build -t ladybugdb .

# Build with AVX2 fallback
docker build --build-arg SIMD=avx2 -t ladybugdb:avx2 .

# Run
docker run -d -p 8080:8080 --name ladybugdb ladybugdb

# Run with persistent data
docker run -d -p 8080:8080 -v $(pwd)/data:/data --name ladybugdb ladybugdb
```

### 2c. Claude Code Backend

Claude Code auto-handles port binding. Just run:
```bash
cargo build --release --bin ladybug-server
./target/release/ladybug-server
```

It detects `CLAUDE_*` env vars and binds `127.0.0.1:5432`. Override with:
```bash
LADYBUG_HOST=127.0.0.1 LADYBUG_PORT=5555 ./target/release/ladybug-server
```

### 2d. Pre-compiled Binary

If you don't want to compile:
```bash
# From another machine with same arch:
RUSTFLAGS="-C target-cpu=skylake-avx512" cargo build --release --bin ladybug-server
scp target/release/ladybug-server user@server:/usr/local/bin/
```

Binary is statically optimized (`lto=fat`, `codegen-units=1`, stripped). ~15-25MB.

---

## 3. REST API Reference

Base URL: `http://localhost:8080` (or your Railway URL)

### Health & Info

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (for Railway/k8s probes) |
| `/ready` | GET | Readiness check |
| `/` | GET | Server info, version, available endpoints |
| `/api/v1/info` | GET | SIMD level, CPU features, indexed count |
| `/api/v1/simd` | GET | Detailed SIMD capability report |

### Fingerprint Operations

**Create fingerprint** — `POST /api/v1/fingerprint`
```json
{"content": "hello world"}
```
Response:
```json
{"fingerprint": "base64...", "popcount": 5023, "density": 0.5023, "bits": 10000}
```

**Batch create** — `POST /api/v1/fingerprint/batch`
```json
{"contents": ["hello", "world", "test"]}
```

**Hamming distance** — `POST /api/v1/hamming`
```json
{"a": "hello", "b": "world"}
```
Response:
```json
{"distance": 4982, "similarity": 0.5018, "bits": 10000}
```
Values for `a` and `b` can be base64-encoded fingerprints or content strings.

**XOR Bind** — `POST /api/v1/bind`
```json
{"a": "base64_fp_a", "b": "base64_fp_b"}
```

**Majority Bundle** — `POST /api/v1/bundle`
```json
{"fingerprints": ["base64_fp1", "base64_fp2", "base64_fp3"]}
```

### Search

**Top-K** — `POST /api/v1/search/topk`
```json
{"query": "hello", "k": 10}
```
Response:
```json
{
  "results": [
    {"index": 0, "id": "node1", "distance": 42, "similarity": 0.9958, "metadata": {"type": "thought"}}
  ],
  "count": 1,
  "total_indexed": 100
}
```

**Threshold** — `POST /api/v1/search/threshold`
```json
{"query": "base64_fp", "max_distance": 2000, "limit": 50}
```

**Resonate** (content-based) — `POST /api/v1/search/resonate`
```json
{"content": "consciousness and awareness", "threshold": 0.7, "limit": 10}
```

### Index Management

**Add to index** — `POST /api/v1/index`
```json
{
  "id": "thought_001",
  "content": "The nature of consciousness",
  "metadata": {"type": "thought", "domain": "philosophy"}
}
```

**Count** — `GET /api/v1/index/count`

**Clear** — `DELETE /api/v1/index`

### NARS Inference

All NARS endpoints accept `{f1, c1, f2, c2}` (frequency/confidence pairs):

| Endpoint | Operation | Formula |
|----------|-----------|---------|
| `POST /api/v1/nars/deduction` | A→B, B→C ⊢ A→C | f=f1·f2, c=f1·f2·c1·c2 |
| `POST /api/v1/nars/induction` | A→B, A→C ⊢ B→C | Truth with positive evidence |
| `POST /api/v1/nars/abduction` | A→B, C→B ⊢ A→C | Truth with positive evidence |
| `POST /api/v1/nars/revision` | Combine independent evidence | Weighted merge |

```json
{"f1": 0.9, "c1": 0.8, "f2": 0.85, "c2": 0.75}
```

### SQL Endpoint

`POST /api/v1/sql`
```json
{"query": "SELECT * FROM nodes WHERE label = 'Thought'"}
```
DataFusion SQL with custom UDFs: `hamming(a, b)`, `similarity(a, b)`, `popcount(x)`, `xor_bind(a, b)`.

### Cypher Endpoint

`POST /api/v1/cypher`
```json
{"query": "MATCH (a:Thought)-[:CAUSES*1..5]->(b) RETURN b"}
```
Cypher is transpiled to recursive CTEs over the SQL engine.

### Redis Protocol

`POST /redis`
```
SET mykey "hello world"
GET mykey
KEYS *
PING
INFO
```

CogRedis extensions (coming): `RESONATE`, `BIND`, `CAUSE`, `DEDUCE`, `INTUIT`.

### LanceDB-Compatible API

These endpoints mirror the LanceDB Python API for drop-in replacement:

`POST /api/v1/lance/add`
```json
{"id": "doc1", "text": "hello world"}
```

`POST /api/v1/lance/search`
```json
{"query": "hello", "limit": 10}
```
Response matches LanceDB format: `[{"id": "doc1", "_distance": 42, "_similarity": 0.99, "text": "hello world"}]`

---

## 4. Python SDK

### Installation

```bash
# HTTP client (works with any running ladybug-server)
pip install requests
cp sdk/python/ladybugdb.py /your/project/

# Native bindings (compile from source)
cd ladybug-rs
pip install maturin
maturin develop --features python
```

### Quick Start — HTTP Client

```python
import ladybugdb

# Connect to server
client = ladybugdb.Client("http://localhost:8080")

# Create fingerprints
fp1 = client.fingerprint("consciousness is awareness")
fp2 = client.fingerprint("the nature of mind")

# Compare
result = client.hamming(fp1, fp2)
print(f"Distance: {result['distance']}, Similarity: {result['similarity']:.4f}")

# Index content
client.index(content="The hard problem of consciousness", id="t1", metadata={"domain": "philosophy"})
client.index(content="Neural correlates of awareness", id="t2", metadata={"domain": "neuroscience"})

# Search
results = client.topk("consciousness", k=5)
for r in results:
    print(f"  {r['id']}: sim={r['similarity']:.4f}")

# NARS inference
conclusion = client.deduction(f1=0.9, c1=0.8, f2=0.85, c2=0.75)
print(f"Deduction: f={conclusion['frequency']:.4f}, c={conclusion['confidence']:.4f}")
```

### LanceDB-Compatible Usage

```python
import ladybugdb

# Connect (same API as lancedb.connect)
db = ladybugdb.connect("http://localhost:8080")

# Create table with data
data = [
    {"id": "1", "text": "The cat sat on the mat"},
    {"id": "2", "text": "Dogs are loyal companions"},
    {"id": "3", "text": "Feline behavior is complex"},
]
table = db.create_table("animals", data=data)

# Search (same API as LanceDB)
results = table.search("cats and their behavior").limit(5).to_list()
for r in results:
    print(r)

# Count
print(f"Rows: {table.count_rows()}")
```

### Native Bindings (Maximum Performance)

```python
import ladybug  # compiled with maturin

# SIMD level
print(ladybug.simd_level())  # "avx512" or "avx2" or "scalar"

# Fingerprints
fp1 = ladybug.Fingerprint("hello world")
fp2 = ladybug.Fingerprint("hello earth")
print(fp1.hamming(fp2))       # Hamming distance
print(fp1.similarity(fp2))    # 0.0-1.0

# Batch operations (SIMD-accelerated)
query = ladybug.Fingerprint("search term")
candidates = [ladybug.Fingerprint(f"doc_{i}") for i in range(10000)]
distances = ladybug.batch_hamming(query, candidates)

# Top-k
results = ladybug.topk_hamming(query, candidates, 10)
# [(index, distance, similarity), ...]

# Bundle (majority vote)
bundled = ladybug.bundle([fp1, fp2])

# Direct byte operations (for integration with existing systems)
raw_bytes = fp1.to_bytes()  # 1256 bytes
fp3 = ladybug.Fingerprint.from_bytes(raw_bytes)

# NARS
tv = ladybug.TruthValue(frequency=0.9, confidence=0.8)
result = tv.deduction(ladybug.TruthValue(0.85, 0.75))
print(result)  # ⟨85%, 54%⟩
```

---

## 5. Integration with Ada Services

### From ada-unified (Python)

```python
# In ada-unified, replace LanceDB 10000D with LadybugDB
import ladybugdb

client = ladybugdb.Client("http://ladybugdb.railway.internal:8080")

# Index a memory
client.index(
    content=memory_text,
    id=f"mem_{uuid}",
    metadata={"qidx": str(qidx), "region": region, "sigma": sigma_level}
)

# Search memories
results = client.topk(query_text, k=20)
for r in results:
    mem_id = r["id"]
    similarity = r["similarity"]
    # ... process
```

### From bighorn (Python → Rust migration)

```python
# bighorn currently uses LanceDB with 10000D vectors
# Replace with ladybugdb bit-packed Hamming:

import ladybugdb
client = ladybugdb.Client("http://ladybugdb.railway.internal:8080")

# Old: lancedb search with float vectors
# New: ladybugdb search with 10K-bit fingerprints
results = client.resonate(content=thought_text, threshold=0.7, limit=10)
```

### From ada-consciousness (Node.js / MCP)

```javascript
// HTTP call from any service
const response = await fetch("http://ladybugdb.railway.internal:8080/api/v1/search/topk", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({query: "consciousness state", k: 10})
});
const results = await response.json();
```

---

## 6. Interface Comparison

| Interface | Syntax | Use Case |
|-----------|--------|----------|
| **REST API** | `POST /api/v1/...` | Any HTTP client, cross-service |
| **Python SDK** | `client.topk(...)` | Python consumers (ada-unified, bighorn) |
| **LanceDB API** | `table.search(...)` | Drop-in LanceDB replacement |
| **Native PyO3** | `ladybug.batch_hamming(...)` | Maximum performance, same-process |
| **SQL** | `SELECT * WHERE similarity > 0.7` | DataFusion queries with Hamming UDFs |
| **Cypher** | `MATCH (a)-[:CAUSES]->(b)` | Graph traversal, causal chains |
| **Redis** | `SET/GET/KEYS/RESONATE` | Key-value + cognitive extensions |
| **CogRedis** | `BIND/CAUSE/DEDUCE/INTUIT` | Full cognitive operations |

---

## 7. Performance Notes

- **10K-bit Hamming**: 65M+ comparisons/sec (AVX-512 single core)
- **Memory**: 1256 bytes per fingerprint (vs 40KB for 10000D float32)
- **Batch operations**: SIMD-accelerated, auto-tunes batch size to CPU cache
- **Buffer pool**: DuckDB-inspired clock-sweep eviction, adaptive prefetch
- **Storage**: Lance columnar format, zero-copy Arrow, IVF-PQ indices

### Fingerprint Size Comparison

| Format | Dimensions | Bytes/Vector | 1M vectors |
|--------|-----------|--------------|------------|
| Float32 10000D | 10,000 | 40,000 | 38 GB |
| **Hamming 10K-bit** | **10,000** | **1,256** | **1.2 GB** |
| Ratio | — | **32x smaller** | **32x smaller** |

---

## 8. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LADYBUG_HOST` | auto-detect | Bind address |
| `LADYBUG_PORT` | auto-detect | Bind port |
| `PORT` | — | Railway convention (overrides LADYBUG_PORT) |
| `LADYBUG_DATA_DIR` | `./data` | Persistent storage path |
| `RAILWAY_*` | — | Railway auto-detection |
| `CLAUDE_*` | — | Claude Code auto-detection |

### Auto-Detection Logic

| Environment | Host | Port |
|-------------|------|------|
| Railway | `0.0.0.0` | `8080` (or `$PORT`) |
| Claude Code | `127.0.0.1` | `5432` |
| Docker | `0.0.0.0` | `8080` |
| Local | `127.0.0.1` | `8080` |
