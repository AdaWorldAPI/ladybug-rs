# LadybugDB Extensions

LadybugDB is designed as a **production-ready substrate** with optional extensions for specialized use cases.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LADYBUGDB CORE                                   │
│                                                                          │
│   SQL (DataFusion) + Cypher (Transpiler) + Vector (Lance) + Hamming     │
│   CAM Index (64-bit universal addressing)                               │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                      OPTIONAL EXTENSIONS                                 │
│                                                                          │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│   │   Crystal   │  │   Crystal   │  │    SPO      │  │  LangExtract │   │
│   │   Savant    │  │   Memory    │  │  Crystal    │  │   Grammar    │   │
│   │  (Codebook) │  │ (Holograph) │  │ (3D Graph)  │  │  (Triangle)  │   │
│   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Extension Summary

| Extension | Repo | Purpose | When to Use |
|-----------|------|---------|-------------|
| **Crystal Savant** | `crystal-savant` | Multi-pass CAM codebook with Hamming meta-resonance | Trained lookup tables, 176K ops/sec |
| **Crystal Memory** | `crystal-memory` | 4KB holographic crystals with quorum ECC | Fault-tolerant storage |
| **SPO Crystal** | `spo-crystal` | 3D content-addressable knowledge graph | O(1) triple queries (vs O(log N) Cypher) |
| **Crystal Compress** | `crystal-compress` | Semantic compression via 3D dictionary + BTR-RL | Large contexts → small indices |
| **LangExtract** | `langextract` | Grammar Triangle extraction (NSM + Causality + Qualia) | Structured info from unstructured text |

---

## Crystal Savant

**Multi-pass Codebook: Concept CAM with Hamming Meta-Resonance**

### Use Case
When you have a rich corpus (books, NARS patterns, qualia mappings) and want **zero-API-call lookups** at runtime.

### How It Works
```
Pass 1 (expensive, one-time):
  Corpus → Jina Embed → 10Kbit fingerprint → Cluster into 128 CAM slots

Pass 2 (cheap, runtime):
  New text → Hash fingerprint (NO API) → XOR scan against CAM
  Result: ~6µs per lookup, 176K lookups/sec, 157KB memory (L2 cache)
```

### Integration
```rust
use ladybug::extensions::crystal_savant::Codebook;

// Train once
let codebook = Codebook::train(&corpus_fingerprints, 128)?;
codebook.save("codebook.bin")?;

// Runtime: pure binary ops
let slot = codebook.lookup(&query_fingerprint); // 6µs
```

---

## Crystal Memory

**4KB Holographic Crystals: 5×5×5 quorum fields**

### Use Case
When you need **fault-tolerant storage** where partial data can reconstruct the whole.

### How It Works
```
3 copies × 10Kbit = 30Kbit per memory
5×5×5 = 125 cells for spatial addressing
Any 2-of-3 copies can reconstruct (quorum consensus)
4096 attractor basins in 170MB total
```

### Integration
```rust
use ladybug::extensions::crystal_memory::QuorumField;

let field = QuorumField::new();
field.store_quorum(&key, &value);  // 3-way redundant
let value = field.recall_quorum(&noisy_key)?;  // Tolerates corruption
```

---

## SPO Crystal

**3D Content-Addressable Knowledge Graph**

### Use Case
When Cypher's O(log N) is too slow and you want **O(1) triple lookups**.

### How It Works
```
Traditional:  MATCH (s)-[p]->(o) WHERE s.name = "Ada"
              → Index lookup + graph traversal

Crystal:      query(S="Ada", P="feels", O=?)
              → 3D address + orthogonal cleanup
              → O(1) resonance

Encoding:     S ⊕ ROLE_S ⊕ P ⊕ ROLE_P ⊕ O ⊕ ROLE_O
              → x-axis    → y-axis    → z-axis
```

### Integration
```rust
use ladybug::extensions::spo_crystal::SPOGrid;

let grid = SPOGrid::new(5, 5, 5);  // 125 cells
grid.store(subject, predicate, object, qualia)?;
let objects = grid.query(Some(subject), Some(predicate), None)?;  // O(1)
```

---

## Crystal Compress

**Semantic Compression via 3D Dictionary + BTR Procella RL**

### Use Case
When you have **huge contexts** (100K tokens) and need a **small queryable index**.

### How It Works
```
LangExtract chunks → Crystal Dictionary (125 centroids) → BTR-RL optimization
100K tokens → 125 prototypes + sparse pointers ≈ O(KB) index
Compression: 8-300x depending on architecture
```

### Integration
```rust
use ladybug::extensions::crystal_compress::{Compressor, BTRPolicy};

let compressor = Compressor::with_policy(BTRPolicy::balanced());
let index = compressor.compress(&chunks)?;  // KB-scale
let relevant = index.query(&query_embedding, 10)?;
```

---

## LangExtract Grammar

**Grammar Triangle: NSM + Causality + Qualia Field**

### Use Case
When you need **structured extraction** from unstructured text with semantic grounding.

### The Triangle
```
        CAUSALITY (flows, not causes)
              /\
             /  \
            /    \
    NSM <--⊕--> ICC (Qualia field)
  (65 primes)    (18D continuous)
        |
        ↓
   BIPOLAR VSA FIELD
   (holds superposition)
```

### NSM Primitives (subset)
`THINK`, `KNOW`, `WANT`, `FEEL`, `SEE`, `HEAR`, `SAY`, `DO`, `HAPPEN`, `MOVE`...

### Qualia Dimensions
`valence`, `arousal`, `tension`, `boundary`, `depth`, `velocity`...

### Integration
```python
from langextract.core.grammar_triangle import GrammarTriangleField

field = GrammarTriangleField()
projection = field.project(text)
# Returns: NSM activations + causality flow + qualia coordinates
```

---

## Feature Flags

Enable extensions in `Cargo.toml`:

```toml
[dependencies]
ladybug = { version = "0.1", features = ["crystal-savant", "spo-crystal"] }
```

Available features:
- `crystal-savant` - Multi-pass codebook
- `crystal-memory` - Holographic quorum storage
- `spo-crystal` - 3D knowledge graph
- `crystal-compress` - Semantic compression
- `langextract` - Grammar triangle (Python interop via PyO3)

---

## When NOT to Use Extensions

| If you need... | Use core ladybug-rs | Extension overkill |
|----------------|--------------------|--------------------|
| SQL queries | ✅ DataFusion | - |
| Graph traversal | ✅ Cypher transpiler | SPO Crystal (unless O(1) critical) |
| Vector search | ✅ Lance ANN | - |
| Hamming similarity | ✅ SIMD engine | Crystal Savant (unless pre-trained) |
| Simple CRUD | ✅ LanceStore | - |

**Rule of thumb**: Start with core. Add extensions only when you hit specific bottlenecks.
