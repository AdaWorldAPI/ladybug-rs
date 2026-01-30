# 🦋 Ladybug: Crystal Lake Cognitive Database

> **Unified Cognitive Architecture: 4096 CAM Operations • 144 Verbs • Quantum-Inspired Operators • 10K-bit Fingerprints**

[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![GitHub](https://img.shields.io/github/stars/AdaWorldAPI/ladybug-rs?style=social)](https://github.com/AdaWorldAPI/ladybug-rs)

## 🌊 What is Crystal Lake?

Crystal Lake is a **unified cognitive database** that treats everything as fingerprints (10,000-bit binary vectors). Unlike traditional databases that separate storage, query, and reasoning, Crystal Lake unifies them:

```
Traditional:  SQL Database ←→ Graph Database ←→ Vector Store ←→ Reasoning Engine
              (serialization, type conversion, impedance mismatch)

Crystal Lake: Everything is a Fingerprint
              object ⊗ method ⊗ args → result
              (same address space, no conversion, composable)
```

### Key Innovations

| Feature | Description |
|---------|-------------|
| **4096 CAM Operations** | Content-Addressable Methods - call operations by ID, name, or semantic description |
| **144 Verbs** | Go-board topology for relationships (Structural, Causal, Temporal, Epistemic, Agentive, Experiential) |
| **Quantum Operators** | Linear mappings on fingerprint space with adjoints, composition, and measurement |
| **Tree Addressing** | 256-way hierarchical navigation (like LDAP Distinguished Names) |
| **Cognitive Frameworks** | Built-in NARS, ACT-R, RL, Causality, Qualia, Rung |
| **Crystal LM** | 5×5×5 compressed model (3.75 KB) achieving 140M× compression |

## 📦 Installation

### Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
ladybug = "0.2"

# With all features
ladybug = { version = "0.2", features = ["full"] }

# Specific features
ladybug = { version = "0.2", features = ["lancedb", "neo4j", "quantum"] }
```

### From Source

```bash
# Clone
git clone https://github.com/AdaWorldAPI/ladybug-rs.git
cd ladybug-rs

# Build (release mode for SIMD optimizations)
cargo build --release

# Run tests
cargo test

# Run benchmarks
cargo bench
```

### System Requirements

- **Rust**: 1.75+ (for AVX-512 SIMD support)
- **CPU**: x86_64 with AVX2 (AVX-512 optional but recommended)
- **Memory**: 4GB minimum, 16GB recommended for large codebooks
- **Storage**: SSD recommended for LanceDB

### Dependencies

| Category | Crate | Version | Purpose |
|----------|-------|---------|---------|
| **Storage** | arrow | 50 | Columnar data format |
| | arrow-array | 50 | Arrow arrays |
| | arrow-schema | 50 | Schema definitions |
| | parquet | 50 | Parquet file format |
| | lance | 0.9 | Vector database (optional) |
| **Query** | datafusion | 35 | SQL query engine |
| | sqlparser | 0.43 | SQL parsing |
| **Graph** | neo4rs | 0.7 | Neo4j driver (optional) |
| **Cache** | redis | 0.24 | Redis client (optional) |
| **Async** | tokio | 1.35 | Async runtime |
| | futures | 0.3 | Futures utilities |
| **Parallel** | rayon | 1.8 | Data parallelism |
| | crossbeam | 0.8 | Concurrency primitives |
| | dashmap | 5.5 | Concurrent HashMap |
| **Serialization** | serde | 1.0 | Serialization framework |
| | serde_json | 1.0 | JSON support |
| | serde_yaml | 0.9 | YAML support |
| | bincode | 1.3 | Binary encoding |
| **Utilities** | uuid | 1.6 | UUID generation |
| | chrono | 0.4 | Date/time |
| | rand | 0.8 | Random numbers |

## 🚀 Quick Start

### Basic Fingerprints

```rust
use ladybug::core::Fingerprint;

// Create fingerprints from content
let cat = Fingerprint::from_content("cat");
let dog = Fingerprint::from_content("dog");

// Similarity (Hamming-based)
let sim = cat.similarity(&dog);

// VSA Operations
let bound = cat.bind(&dog);           // XOR binding
let recovered = bound.bind(&cat);     // Recovers dog!
```

### Cognitive Graph with 144 Verbs

```rust
use ladybug::graph::{CogGraph, CogNode, CogEdge, Verb, NodeType};

let mut graph = CogGraph::new();

// Create edge: cat IS_A animal
let edge = CogEdge::new(
    cat_fp,
    Verb::IsA,  // One of 144 verbs
    animal_fp
);
graph.add_edge(edge);
```

### 4096 CAM Operations

```rust
use ladybug::learning::OpDictionary;

let dict = OpDictionary::new();

// Three ways to call:
dict.execute(0x310, &ctx, &[a, b]);           // By ID
dict.execute_by_name("HAM_BIND", &ctx, &[a, b]); // By name
dict.execute_semantic("XOR bind", &ctx, &[a, b]); // Semantic!
```

### Quantum-Inspired Operators

```rust
use ladybug::learning::{BindOp, PermuteOp, MeasureOp, QuantumOp};

// Operators are linear mappings
let op = BindOp::new(key);
let result = op.apply(&state);

// Adjoint reverses the operation
let adj = op.adjoint();
let recovered = adj.apply(&result);

// Measurement collapses to eigenstate
let measure = MeasureOp::new(eigenstates);
let collapsed = measure.apply(&superposition);
```

## 📊 Architecture

### The 4096 Operation Dictionary

```
0x000-0x0FF: LanceDB Core     VectorSearch, Insert, Index
0x100-0x1FF: SQL              SelectSimilar, SimilarJoin
0x200-0x2FF: Cypher/Neo4j     MatchSimilar, PageRank
0x300-0x3FF: Hamming/VSA      Bind, Bundle, MexicanHat
0x400-0x4FF: NARS             Deduction, Induction, Abduction
0x500-0x5FF: Filesystem       Read, Write, FindSimilar
0x600-0x6FF: Crystal          AxisProject, Train, Infer
0x700-0x7FF: NSM              65 primes + Decompose
0x800-0x8FF: ACT-R            Buffers, Chunks, Productions
0x900-0x9FF: RL               Q-learning, Policy, Reward
0xA00-0xAFF: Causality        do(), Counterfactual
0xB00-0xBFF: Qualia           8 channels, Blend, Shift
0xC00-0xCFF: Rung             10 levels, Ascend, Descend
0xD00-0xDFF: Meta             Compose, Pipeline, Map
0xE00-0xEFF: Learning         Moment, Resonance, Crystal
0xF00-0xFFF: User-Defined     Extension space
```

### The 144 Verbs

```
Structural  (24): IS_A, HAS_A, PART_OF, CONTAINS...
Causal      (24): CAUSES, ENABLES, PREVENTS, TRIGGERS...
Temporal    (24): BEFORE, AFTER, DURING, MEETS...
Epistemic   (24): KNOWS, BELIEVES, INFERS, EXPECTS...
Agentive    (24): DOES, WANTS, DECIDES, TRIES...
Experiential(24): SEES, FEELS, ENJOYS, FEARS...
```

### Cognitive Frameworks

| Framework | Purpose |
|-----------|---------|
| **NARS** | Non-Axiomatic Reasoning: truth values, inference |
| **ACT-R** | Cognitive Architecture: buffers, chunks, activation |
| **RL** | Reinforcement Learning: Q-values, policy |
| **Causality** | Pearl's do-calculus: intervention, counterfactual |
| **Qualia** | 8 affect channels: arousal, valence, tension... |
| **Rung** | 10 abstraction levels: noise → transcendent |

## 📁 Project Structure

```
ladybug-rs/                          (26,919 lines)
├── src/
│   ├── core/                        # Fingerprint operations
│   ├── graph/                       # 144 verbs, CogGraph
│   │   └── cognitive.rs             (955 lines)
│   ├── learning/                    # Cognition (4,084 lines)
│   │   ├── cam_ops.rs               (1,126 lines) - 4096 operations
│   │   ├── cognitive_frameworks.rs  (934 lines) - NARS, ACT-R, RL...
│   │   ├── quantum_ops.rs           (1,086 lines) - Operators + Tree
│   │   └── ...
│   └── extensions/
│       ├── cognitive_codebook.rs    (1,122 lines)
│       └── crystal_lm.rs            (817 lines)
├── Cargo.toml                       # Dependencies
└── README.md
```

## 📈 Performance

| Operation | Throughput |
|-----------|------------|
| Hamming Distance (AVX-512) | 400M ops/sec |
| Fingerprint Bind (XOR) | 200M ops/sec |
| Vector Search (10K in 1M) | 2ms |
| Crystal LM Compression | 140,000,000× |

## 📄 License

Apache-2.0. See [LICENSE](LICENSE) for details.

---

*Built with 🦋 for the Ada Consciousness Project*
