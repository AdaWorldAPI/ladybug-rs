# LadybugDB Python SDK (`ladybug-vsa`)

Production-ready Python client for LadybugDB cognitive database.

## Installation

```bash
# From PyPI (zero dependencies)
pip install ladybug-vsa

# Or install from source
cd sdk/python
pip install -e .
```

**Import as:**
```python
from ladybugdb import LadybugDB  # Module name is still ladybugdb
```

## Quick Start

```python
from ladybugdb import LadybugDB

# Connect to Railway production server
db = LadybugDB()

# Or specify URL
db = LadybugDB("http://localhost:8080")

# Check connection
print(db.info())
# {'name': 'LadybugDB', 'version': '0.3.0', 'fingerprint_bits': 10000, ...}
```

## Fingerprint Operations

10K-bit binary fingerprints for Vector Symbolic Architecture (VSA):

```python
# Create fingerprints
fp1 = db.fingerprint("hello world")
fp2 = db.fingerprint("hello there")

print(fp1)  # Fingerprint(popcount=4987, density=0.499)

# Compute similarity (Hamming distance)
result = db.hamming(fp1, fp2)
print(f"Distance: {result['distance']}")      # ~500
print(f"Similarity: {result['similarity']:.1%}")  # ~95%

# Or use convenience method
sim = db.similarity("hello", "hallo")
print(f"Similarity: {sim:.1%}")
```

## VSA Binding & Bundling

```python
# XOR Binding: create role-filler pairs
role = db.fingerprint("president")
filler = db.fingerprint("Lincoln")
bound = db.bind(role, filler)

# Unbind to query (XOR is self-inverse)
recovered = db.bind(bound, role)
print(db.similarity(recovered, filler))  # ~1.0

# Bundle: superposition via majority vote
colors = [db.fingerprint(c) for c in ["red", "blue", "green"]]
color_concept = db.bundle(colors)

# color_concept is similar to all inputs
for fp in colors:
    print(db.similarity(color_concept, fp))  # ~0.67 each
```

## NARS Inference

Non-Axiomatic Reasoning System for uncertain inference:

```python
# Access NARS engine
nars = db.nars

# Deduction: A→B, B→C ⊢ A→C
# "Birds fly" (0.9, 0.8) + "Tweety is a bird" (1.0, 0.9)
result = nars.deduction(f1=0.9, c1=0.8, f2=1.0, c2=0.9)
print(f"Tweety flies: {result}")  # <0.900, 0.648>

# Induction: A→B, A→C ⊢ B→C
result = nars.induction(f1=0.8, c1=0.7, f2=0.9, c2=0.8)

# Abduction: A→B, C→B ⊢ A→C
result = nars.abduction(f1=0.8, c1=0.7, f2=0.9, c2=0.8)

# Revision: combine evidence
result = nars.revision(f1=0.8, c1=0.6, f2=0.9, c2=0.7)
print(f"Combined: {result}")  # Higher confidence
```

## Search & Index

```python
# Index fingerprints
db.index(content="The quick brown fox", id="doc1")
db.index(content="The lazy dog", id="doc2")

# Top-K search
results = db.topk("quick fox", k=5)
for r in results:
    print(f"{r.id}: {r.similarity:.1%}")

# Fluent search builder
results = db.search("brown fox").limit(10).to_list()

# Threshold search
results = db.search("fox").threshold(0.8).to_list()
```

## LanceDB-Compatible API

Drop-in replacement for LanceDB:

```python
import ladybugdb

# Connect (same as lancedb.connect)
db = ladybugdb.connect()

# Create table with data
table = db.create_table("thoughts", data=[
    {"text": "The quick brown fox"},
    {"text": "jumps over the lazy dog"},
    {"text": "Machine learning is powerful"},
])

# Search
results = table.search("brown fox").limit(5).to_list()

# Add more data
table.add([{"text": "Neural networks learn patterns"}])
```

## SQL & Cypher Queries

```python
# SQL query
result = db.sql("SELECT * FROM nodes WHERE confidence > 0.7")

# Cypher graph query
result = db.cypher("MATCH (a)-[:CAUSES]->(b) RETURN a, b")
```

## Redis Protocol

```python
# Redis-like commands
db.redis("SET 'hello world'")
result = db.redis("RESONATE 'hello' 10")
db.redis("SCAN 0 COUNT 100")
```

## Error Handling

```python
from ladybugdb import LadybugDB, ConnectionError, APIError

try:
    db = LadybugDB("http://localhost:8080")
    result = db.fingerprint("test")
except ConnectionError as e:
    print(f"Cannot connect: {e}")
except APIError as e:
    print(f"API error {e.status}: {e}")
```

## API Reference

### LadybugDB

| Method | Description |
|--------|-------------|
| `health()` | Server health check |
| `info()` | Server information |
| `fingerprint(content)` | Create 10K-bit fingerprint |
| `hamming(a, b)` | Hamming distance |
| `similarity(a, b)` | Similarity score (0-1) |
| `bind(a, b)` | XOR binding |
| `bundle(fps)` | Majority-vote bundle |
| `topk(query, k)` | Top-K search |
| `resonate(content, threshold)` | Threshold search |
| `index(content, id, metadata)` | Add to index |
| `sql(query)` | SQL query |
| `cypher(query)` | Cypher query |
| `redis(command)` | Redis command |

### NARSEngine

| Method | Description |
|--------|-------------|
| `deduction(f1, c1, f2, c2)` | A→B, B→C ⊢ A→C |
| `induction(f1, c1, f2, c2)` | A→B, A→C ⊢ B→C |
| `abduction(f1, c1, f2, c2)` | A→B, C→B ⊢ A→C |
| `revision(f1, c1, f2, c2)` | Combine evidence |

## Production URL

Default connects to Railway production:
```
https://ladybug-rs-production.up.railway.app
```

## License

Apache-2.0
