# Ladybug-RS Deployment Guide

Crystal Lake Cognitive Database - Deployment Documentation

## Table of Contents

1. [Quick Start](#quick-start)
2. [Docker Deployment](#docker-deployment)
3. [Railway Deployment](#railway-deployment)
4. [Claude Backend Deployment](#claude-backend-deployment)
5. [CPU Optimization (AVX-512/AVX2)](#cpu-optimization)
6. [Python SDK Usage](#python-sdk-usage)
7. [API Reference](#api-reference)
8. [Configuration](#configuration)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

- Rust 1.75+ (for building)
- Docker (for containerized deployment)
- Python 3.8+ (for SDK usage)

### Local Development

```bash
# Clone repository
git clone https://github.com/AdaWorldAPI/ladybug-rs.git
cd ladybug-rs

# Build with all features
cargo build --release --bin ladybug-server \
  --features "simd,parallel,codebook,hologram,quantum"

# Run server
./target/release/ladybug-server
# Server starts at http://127.0.0.1:5000

# Test health endpoint
curl http://localhost:5000/health
```

---

## Docker Deployment

### Basic Docker Build

```bash
# Build with auto-detected CPU features
docker build -t ladybug-rs .

# Build with specific CPU target
docker build -t ladybug-rs --build-arg TARGET_CPU=skylake .

# Build AVX-512 optimized version
docker build -t ladybug-rs:avx512 --target runtime-avx512 .
```

### Docker Run

```bash
# Run with default settings
docker run -p 8080:8080 ladybug-rs

# Run with custom port
docker run -p 3000:3000 -e PORT=3000 ladybug-rs

# Run with debug logging
docker run -p 8080:8080 -e RUST_LOG=debug ladybug-rs
```

### Docker Compose

```bash
# Start server (auto-detect CPU)
docker-compose up

# Start with AVX-512 optimization
docker-compose --profile avx512 up

# Development mode with hot reload
docker-compose --profile dev up

# Background mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## Railway Deployment

### One-Click Deploy

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template?template=https://github.com/AdaWorldAPI/ladybug-rs)

### Manual Railway Setup

1. Connect your GitHub repository to Railway
2. Railway auto-detects the `railway.json` or `railway.toml`
3. The server automatically binds to `0.0.0.0:8080`

### Railway Configuration

The following files configure Railway deployment:

**railway.json**
```json
{
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile",
    "buildArgs": {
      "TARGET_CPU": "skylake-avx512",
      "FEATURES": "simd,parallel,codebook,hologram,quantum"
    }
  },
  "deploy": {
    "startCommand": "./ladybug-server",
    "healthcheckPath": "/health"
  }
}
```

### Environment Variables

Set these in Railway dashboard or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | Server port (Railway sets this) |
| `HOST` | `0.0.0.0` | Bind address |
| `RUST_LOG` | `info` | Log level (debug/info/warn/error) |

---

## Claude Backend Deployment

For Claude Code backend integration:

### Automatic Detection

The server automatically detects Claude backend environment:
- Binds to `127.0.0.1:$PORT` (default 5000)
- Claude Code manages the port assignment

### Manual Configuration

```bash
# Set specific port
PORT=5123 ./target/release/ladybug-server

# Or via environment
export PORT=5123
export HOST=127.0.0.1
./target/release/ladybug-server
```

### Docker for Claude Backend

```bash
# Build
docker build -t ladybug-rs .

# Run with Claude-compatible settings
docker run \
  -p 5000:5000 \
  -e HOST=127.0.0.1 \
  -e PORT=5000 \
  ladybug-rs
```

---

## CPU Optimization

### AVX-512 (Railway, Modern Xeon, Claude Backend)

Best performance on modern servers:

```bash
# Compile with AVX-512
RUSTFLAGS="-C target-cpu=skylake-avx512 -C target-feature=+avx512f,+avx512vl" \
cargo build --release --bin ladybug-server --features "simd,parallel"

# Docker AVX-512 build
docker build -t ladybug-rs:avx512 \
  --build-arg TARGET_CPU=skylake-avx512 .
```

**Performance**: ~2ns per 10K-bit Hamming distance

### AVX2 (Most Laptops, NUC)

Fallback for older CPUs:

```bash
# Compile with AVX2
RUSTFLAGS="-C target-cpu=haswell" \
cargo build --release --bin ladybug-server --features "simd,parallel"

# Docker AVX2 build
docker build -t ladybug-rs:avx2 \
  --build-arg TARGET_CPU=haswell .
```

**Performance**: ~4ns per 10K-bit Hamming distance

### Generic (WASM, ARM, Old x86)

Maximum compatibility:

```bash
# Compile generic
RUSTFLAGS="-C target-cpu=x86-64" \
cargo build --release --bin ladybug-server

# Docker generic build
docker build -t ladybug-rs:generic \
  --build-arg TARGET_CPU=x86-64 .
```

**Performance**: ~50ns per 10K-bit Hamming distance

### Detecting CPU Features

```bash
# Check CPU capabilities
cat /proc/cpuinfo | grep -E "avx512|avx2"

# In Rust
#[cfg(target_arch = "x86_64")]
{
    if is_x86_feature_detected!("avx512f") {
        println!("AVX-512 available");
    }
}
```

---

## Python SDK Usage

### Installation

```bash
# From PyPI (zero dependencies)
pip install ladybug-vsa

# Or install from source
cd sdk/python
pip install -e .
```

### Basic Usage

```python
from ladybugdb import LadybugDB

# Connect to Railway production (default)
db = LadybugDB()

# Or specify URL
db = LadybugDB("http://localhost:8080")

# Health check
print(db.health())
# {"status": "healthy", "service": "ladybug-rs"}

# Server info
print(db.info())
# {"name": "LadybugDB", "version": "0.3.0", ...}

# Create fingerprint
fp = db.fingerprint("hello world")
print(f"Popcount: {fp.popcount}, Density: {fp.density:.3f}")
```

### Similarity & VSA Operations

```python
from ladybugdb import LadybugDB

db = LadybugDB()

# Create fingerprints
fp1 = db.fingerprint("hello world")
fp2 = db.fingerprint("hello there")

# Compute Hamming distance
result = db.hamming(fp1, fp2)
print(f"Distance: {result['distance']}, Similarity: {result['similarity']:.2%}")

# XOR Binding (role-filler pairs)
role = db.fingerprint("president")
filler = db.fingerprint("Lincoln")
bound = db.bind(role, filler)

# Unbind to query (XOR is self-inverse)
recovered = db.bind(bound, role)
print(db.similarity(recovered, filler))  # ~1.0

# Bundle (majority-vote superposition)
colors = [db.fingerprint(c) for c in ["red", "blue", "green"]]
color_concept = db.bundle(colors)
```

### LanceDB-Compatible Table API

```python
from ladybugdb import LadybugDB

db = LadybugDB()

# Create table with data
table = db.create_table("thoughts", data=[
    {"text": "The quick brown fox"},
    {"text": "Machine learning basics"},
    {"text": "Neural network architectures"}
])

# Fluent search
results = table.search("fast fox").limit(5).to_list()
for r in results:
    print(f"ID: {r.id}, Similarity: {r.similarity:.4f}")
```

### NARS Inference

```python
from ladybugdb import LadybugDB

db = LadybugDB()
nars = db.nars

# Deduction: A→B, B→C ⊢ A→C
# "Birds fly" (0.9, 0.8) + "Tweety is bird" (1.0, 0.9)
result = nars.deduction(f1=0.9, c1=0.8, f2=1.0, c2=0.9)
print(f"Tweety flies: {result}")  # <0.900, 0.648>

# Revision: combine evidence
result = nars.revision(f1=0.8, c1=0.6, f2=0.9, c2=0.7)
```

### Redis Protocol

```python
from ladybugdb import LadybugDB

db = LadybugDB()

# Redis-like commands
db.redis("SET 'hello world'")
result = db.redis("RESONATE 'hello' 10")
db.redis("SCAN 0 COUNT 100")
```

### SQL & Cypher Queries

```python
from ladybugdb import LadybugDB

db = LadybugDB()

# SQL query
result = db.sql("SELECT * FROM nodes WHERE confidence > 0.7")

# Cypher graph query
result = db.cypher("MATCH (a)-[:CAUSES]->(b) RETURN a, b")
```

---

## API Reference

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/info` | Server information |
| POST | `/redis` | Redis-like commands |
| POST | `/sql` | SQL queries |
| POST | `/cypher` | Cypher graph queries |
| POST | `/vectors/search` | Vector similarity search |
| POST | `/vectors/insert` | Insert vectors |
| POST | `/cam/{operation}` | CAM operations |
| POST | `/fingerprint` | Create fingerprint |

### Vector Search Request

```json
POST /vectors/search
{
  "query": "search text",
  "k": 10,
  "threshold": 0.5
}
```

### Vector Search Response

```json
{
  "success": true,
  "matches": [
    {
      "addr": "0x0100",
      "distance": 2500,
      "similarity": 0.75
    }
  ],
  "count": 1
}
```

### Vector Insert Request

```json
POST /vectors/insert
{
  "vectors": [
    {"content": "text to vectorize"},
    {"content": "another document"}
  ]
}
```

### Redis Command Request

```json
POST /redis
{
  "command": "SET hello_world 0.9"
}
```

### CAM Operation Request

```json
POST /cam/BIND
{
  "args": ["concept_a", "concept_b"]
}
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | Auto-detected | Bind address (0.0.0.0 for Railway, 127.0.0.1 local) |
| `PORT` | 8080 (Railway) / 5000 (local) | Server port |
| `RUST_LOG` | `info` | Log level |
| `RAILWAY_ENVIRONMENT` | - | Set by Railway (triggers Railway mode) |

### Feature Flags

Build with specific features:

```bash
cargo build --release --features "simd,parallel,codebook,hologram,quantum"
```

| Feature | Description |
|---------|-------------|
| `simd` | AVX-512/AVX2 SIMD operations |
| `parallel` | Parallel processing (rayon) |
| `codebook` | Cognitive codebook (NSM, NARS) |
| `hologram` | Holographic reduced representations |
| `quantum` | Quantum-inspired operators |
| `spo` | Subject-Predicate-Object triples |

---

## Troubleshooting

### Connection Refused

```bash
# Check if server is running
curl http://localhost:8080/health

# Check Docker container
docker ps
docker logs ladybug-rs

# Check port binding
netstat -tlnp | grep 8080
```

### Slow Performance

1. Check CPU features are detected:
   ```bash
   curl http://localhost:8080/info
   # Look for "avx512" or "avx2" in response
   ```

2. Ensure release build:
   ```bash
   cargo build --release
   ```

3. Use optimized Docker target:
   ```bash
   docker build --target runtime-avx512 -t ladybug-rs:fast .
   ```

### Railway Deployment Issues

1. Check `railway.json` or `railway.toml` exists
2. Verify Dockerfile builds locally
3. Check Railway logs in dashboard
4. Ensure health check passes (`/health` endpoint)

### Memory Issues

The server uses ~500MB-2GB depending on data:

```bash
# Docker memory limit
docker run -m 2g ladybug-rs

# Docker Compose limit (in docker-compose.yml)
deploy:
  resources:
    limits:
      memory: 2G
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Ladybug-RS Server                           │
├─────────────────────────────────────────────────────────────────┤
│  HTTP API Layer                                                  │
│  ├── /redis    (Redis-like commands)                            │
│  ├── /sql      (SQL via DataFusion)                             │
│  ├── /cypher   (Cypher graph queries)                           │
│  ├── /vectors  (LanceDB-compatible API)                         │
│  └── /cam      (4096 CAM operations)                            │
├─────────────────────────────────────────────────────────────────┤
│  Cognitive Engine                                                │
│  ├── CogRedis       (Command execution)                         │
│  ├── BindSpace      (8+8 address model)                         │
│  └── CAM Operations (4096 ops × 16 prefixes)                    │
├─────────────────────────────────────────────────────────────────┤
│  Storage Layer                                                   │
│  ├── Fingerprint (10K-bit VSA)                                  │
│  ├── QuorumField (5×5×5 lattice)                                │
│  ├── QuantumField (N×N×N with phase tags)                       │
│  └── Crystal4K   (holographic compression)                      │
├─────────────────────────────────────────────────────────────────┤
│  SIMD Layer (AVX-512 / AVX2 / Scalar)                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## License

Apache-2.0

## Support

- GitHub Issues: https://github.com/AdaWorldAPI/ladybug-rs/issues
- Documentation: https://github.com/AdaWorldAPI/ladybug-rs/wiki
