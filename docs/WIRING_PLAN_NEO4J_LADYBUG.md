# ladybug-rs ↔ neo4j-rs Wiring Plan
## Orchestration Layer Design — 100% Neo4j Faithful

> **Core principle**: neo4j-rs stays a perfect Neo4j citizen. ladybug-rs
> gets its cognitive substrate. The wiring is a **translation layer**
> that loses nothing from either side.

---

## 1. The Problem in One Sentence

neo4j-rs speaks **property graph** (nodes with labels, typed properties,
directed relationships). ladybug-rs speaks **fingerprint algebra** (16384-bit
vectors, XOR bind/unbind, Hamming distance, NARS truth values). The
orchestration layer must translate between them without either side
compromising its native semantics.

---

## 2. What neo4j-rs MUST Remain

neo4j-rs is a **Neo4j-compatible graph database interface**. Its correctness
is measured against the openCypher TCK (Technology Compatibility Kit). It
must remain 100% truthful to:

| Neo4j Feature | neo4j-rs Contract | Non-negotiable |
|---------------|-------------------|---------------|
| Cypher syntax | Full openCypher parser | ✓ |
| Property types | Bool, Int, Float, String, Bytes, List, Map, Temporal, Spatial | ✓ |
| Node model | `{id, labels[], properties{}}` | ✓ |
| Relationship model | `{id, src, dst, type, properties{}}` | ✓ |
| Path model | Alternating Node-Rel-Node sequences | ✓ |
| ACID transactions | BEGIN/COMMIT/ROLLBACK with isolation | ✓ |
| NULL semantics | Three-valued logic (NULL ≠ false) | ✓ |
| Index types | BTree, FullText, Unique | ✓ |
| Bolt protocol | Wire-compatible with Neo4j 4.x/5.x | ✓ |
| Direction semantics | OUTGOING/INCOMING/BOTH | ✓ |
| Variable-length paths | `(a)-[*1..5]->(b)` | ✓ |
| Schema constraints | UNIQUE, EXISTS | ✓ |

**What neo4j-rs should NEVER know about:**
- Fingerprints, Hamming distance, XOR bind
- BindSpace, CAKES, HDR cascade
- ThinkingStyles, CollapseGate, NARS
- 144 Verbs, Go board topology
- Any ladybug-rs cognitive concept

---

## 3. What ladybug-rs Needs from neo4j-rs

### 3.1 — The StorageBackend Trait Is Almost Perfect

The existing `StorageBackend` trait gives ladybug-rs everything it needs
for basic CRUD. But there are **5 gaps** that limit ladybug-rs's ability
to provide its unique value through the neo4j-rs interface:

### Gap 1: No Vector/Similarity Query Path

**Problem**: The `StorageBackend` trait has `nodes_by_label()` and
`nodes_by_property()` but no way to express "find nodes similar to X
by Hamming distance." This is ladybug-rs's superpower, hidden behind
a trait that has no slot for it.

**What ladybug-rs needs**: A way for Cypher queries like
`CALL db.index.vector.queryNodes('fingerprint', 10, $query)` to reach
the backend.

**Solution (neo4j-rs stays faithful)**: Neo4j 5.x has `db.index.vector.queryNodes`.
Add to `StorageBackend`:

```rust
/// Vector similarity search (Neo4j 5.x compatible).
/// Backends that don't support this return Error::ExecutionError("not supported").
async fn vector_query(
    &self,
    tx: &Self::Tx,
    index_name: &str,
    k: usize,
    query_vector: &[u8],
) -> Result<Vec<(NodeId, f64)>> {
    Err(Error::ExecutionError("vector index not supported".into()))
}
```

**Why this is 100% Neo4j faithful**: Neo4j 5.11+ has exactly this feature.
The Bolt backend would forward to Neo4j's native vector search. The
ladybug backend would use CAKES/HDR cascade. MemoryBackend returns
"not supported." Same trait, different engines.

### Gap 2: No Backend-Specific Extension Point

**Problem**: ladybug-rs can do things no other backend can (ABBA retrieval,
Scent search, DN-path addressing, causal trace). But the trait has no
escape hatch for backend-specific operations.

**What ladybug-rs needs**: A way to expose cognitive operations without
polluting the universal trait.

**Solution**: Neo4j has procedures (`CALL custom.procedure()`). Add:

```rust
/// Call a backend-specific procedure.
/// This is the escape hatch for backend-unique operations.
///
/// Neo4j examples: CALL apoc.create.node(), CALL gds.pageRank.stream()
/// Ladybug examples: CALL ladybug.hamming.scan(), CALL ladybug.bind()
async fn call_procedure(
    &self,
    tx: &Self::Tx,
    name: &str,
    args: Vec<Value>,
) -> Result<QueryResult> {
    Err(Error::ExecutionError(format!("procedure '{}' not found", name)))
}
```

**Why 100% Neo4j faithful**: Neo4j's `CALL` mechanism is how ALL
extensions work — APOC, GDS, custom procedures. This IS the standard
Neo4j extension point. Bolt backend forwards to Neo4j. Ladybug backend
routes to cognitive substrate. MemoryBackend rejects unknown procedures.

### Gap 3: No Metadata/Annotation Slot on Nodes and Relationships

**Problem**: ladybug-rs wants to store a `Fingerprint` alongside every
node and a `TruthValue` alongside every relationship. The current model
has `PropertyMap` which could technically hold these as properties, but
that's semantically wrong — fingerprints aren't user properties, they're
infrastructure.

**What ladybug-rs needs**: A way to attach backend-managed metadata that
users don't directly see but that powers similarity search and cognitive
operations.

**Solution (two options):**

**Option A — Reserved property namespace** (simplest, most faithful):
Ladybug backend stores fingerprints as `_ladybug_fingerprint: Bytes(Vec<u8>)`
in the property map. The `_` prefix is a Neo4j convention for internal
properties. Neo4j does this for system properties.

**Option B — Backend metadata trait** (cleaner but more complex):

```rust
/// Optional metadata that backends can attach to nodes.
/// Not visible in Cypher RETURN unless explicitly requested.
pub trait BackendMetadata: Send + Sync {
    /// Get metadata for a node.
    async fn node_metadata(&self, tx: &Self::Tx, id: NodeId) -> Result<Option<Value>>;
    /// Get metadata for a relationship.
    async fn rel_metadata(&self, tx: &Self::Tx, id: RelId) -> Result<Option<Value>>;
}
```

**Recommendation**: Option A. It's what Neo4j itself does. The ladybug
backend writes `_ladybug_fp` and `_ladybug_truth` as binary properties.
The execution layer filters them from `RETURN *` output. Zero trait change.

### Gap 4: No Streaming/Batch Insert Path

**Problem**: ladybug-rs's BindSpace can absorb 100K fingerprints per second.
The `create_node()` API is one-at-a-time. For bulk operations (loading a
corpus, migration), this bottlenecks on trait overhead.

**What ladybug-rs needs**: Batch insert capability.

**Solution**: Neo4j has `UNWIND [...] AS row CREATE (n:Label) SET n = row`.
But at the trait level:

```rust
/// Batch create nodes. Backends should optimize for bulk writes.
/// Default implementation falls back to sequential create_node.
async fn create_nodes_batch(
    &self,
    tx: &mut Self::Tx,
    nodes: Vec<(&[&str], PropertyMap)>,
) -> Result<Vec<NodeId>> {
    let mut ids = Vec::with_capacity(nodes.len());
    for (labels, props) in nodes {
        ids.push(self.create_node(tx, labels, props).await?);
    }
    Ok(ids)
}

/// Batch create relationships.
async fn create_relationships_batch(
    &self,
    tx: &mut Self::Tx,
    rels: Vec<(NodeId, NodeId, &str, PropertyMap)>,
) -> Result<Vec<RelId>> {
    let mut ids = Vec::with_capacity(rels.len());
    for (src, dst, rel_type, props) in rels {
        ids.push(self.create_relationship(tx, src, dst, rel_type, props).await?);
    }
    Ok(ids)
}
```

**Why 100% faithful**: Default implementation preserves exact Neo4j
semantics. Ladybug backend overrides for bulk-optimized BindSpace writes.
Bolt backend overrides to pipeline Bolt messages. No semantic change.

### Gap 5: No Event/Hook System for Cognitive Processing

**Problem**: When a node is created, ladybug-rs needs to compute its
fingerprint, insert into CAKES, update HDR cascade, maybe trigger
NARS revision. The trait has no lifecycle hooks.

**What ladybug-rs needs**: A way to intercept write operations for
background processing.

**Solution**: This is an INTERNAL concern for the ladybug backend, not
a trait concern. The `LadybugBackend::create_node()` implementation
handles this:

```rust
// Inside ladybug-rs's backend/storage_impl.rs
impl StorageBackend for LadybugBackend {
    async fn create_node(&self, tx: &mut LadybugTx, labels: &[&str], props: PropertyMap) -> Result<NodeId> {
        // 1. Allocate ID
        let id = self.next_id();

        // 2. Compute fingerprint from labels + properties
        let fp = self.fingerprint_from_node(labels, &props);

        // 3. Store in Lance (persistent)
        self.lance.insert_node(id, labels, &props, &fp).await?;

        // 4. Store in BindSpace (hot cache)
        self.bind_space.insert(id.0 as usize, fp);

        // 5. Update CAKES tree (if enabled)
        if let Some(cakes) = &self.cakes {
            cakes.insert(&fp);
        }

        // 6. Log to transaction journal
        tx.log_create_node(id);

        Ok(id)
    }
}
```

No trait change needed. This is pure implementation concern.

---

## 4. The Wiring Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        USER / APPLICATION                          │
│                                                                    │
│  graph.execute("MATCH (n:Person) WHERE n.name = 'Ada' RETURN n")  │
│  graph.execute("CALL ladybug.similar('Ada', 10)")    ← extension  │
│  graph.execute("MATCH p=(a)-[:CAUSES*1..5]->(b) RETURN p")        │
│                                                                    │
└─────────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                      neo4j-rs  (unchanged)                         │
│                                                                    │
│  Cypher Parser → AST → Planner → Optimizer → Execution Engine     │
│                                                                    │
│  The planner sees CALL statements and routes to call_procedure().  │
│  Everything else goes through standard StorageBackend methods.     │
│                                                                    │
│  StorageBackend trait:                                             │
│    create_node, get_node, delete_node                              │
│    create_relationship, get_relationship, delete_relationship      │
│    get_relationships, expand (traversal)                           │
│    create_index, drop_index                                        │
│    nodes_by_label, nodes_by_property                               │
│    + vector_query (Gap 1)                                          │
│    + call_procedure (Gap 2)                                        │
│    + create_nodes_batch (Gap 4)                                    │
│                                                                    │
└─────────────────────────────┬──────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
┌──────────────────┐ ┌────────────────┐ ┌──────────────────────┐
│  MemoryBackend   │ │  BoltBackend   │ │  LadybugBackend      │
│  (testing)       │ │  (real Neo4j)  │ │  (cognitive engine)  │
│                  │ │                │ │                       │
│  HashMap storage │ │  Bolt wire     │ │  ┌─────────────────┐ │
│  No fingerprints │ │  protocol to   │ │  │  TRANSLATION    │ │
│  No similarity   │ │  Neo4j 4.x/5.x│ │  │  LAYER          │ │
│                  │ │                │ │  │                 │ │
│  call_procedure  │ │  call_procedure│ │  │ Node→Fingerprint│ │
│  → "not found"   │ │  → forwards to │ │  │ Rel→Edge(XOR)  │ │
│                  │ │    Neo4j CALL  │ │  │ Props→Qualia    │ │
│  vector_query    │ │                │ │  │ Label→NSM prime │ │
│  → "not support" │ │  vector_query  │ │  │ Type→Verb(144)  │ │
│                  │ │  → Neo4j 5.x   │ │  └────────┬────────┘ │
│                  │ │    vector idx  │ │           │          │
│                  │ │                │ │  ┌────────▼────────┐ │
│                  │ │                │ │  │ ladybug-rs core │ │
│                  │ │                │ │  │                 │ │
│                  │ │                │ │  │ BindSpace       │ │
│                  │ │                │ │  │ CAKES/HDR       │ │
│                  │ │                │ │  │ DN-Tree         │ │
│                  │ │                │ │  │ NARS/Truth      │ │
│                  │ │                │ │  │ CollapseGate    │ │
│                  │ │                │ │  │ 12 Styles       │ │
│                  │ │                │ │  │ GrammarTriangle │ │
│                  │ │                │ │  └─────────────────┘ │
└──────────────────┘ └────────────────┘ └──────────────────────┘
```

---

## 5. The Translation Layer (Inside LadybugBackend)

This is where the magic happens — and where the gotchas live.

### 5.1 — Node → Fingerprint Translation

```
Neo4j Node:                           Ladybug Fingerprint:
{                                     16384 bits:
  id: 42,                               [semantic field: bits 0-8191]
  labels: ["Person", "Developer"],        ← from_content(labels + props)
  properties: {                          [metadata field: bits 8192-16383]
    name: "Ada",                          ← from node ID + structural info
    age: 3,
    skills: ["Rust", "Python"]
  }
}
```

**Translation function**:

```rust
fn fingerprint_from_node(labels: &[&str], props: &PropertyMap) -> Fingerprint {
    // 1. Start with label fingerprint
    let mut fp = Fingerprint::zero();
    for label in labels {
        fp = fp.bind(&Fingerprint::from_content(label));
    }

    // 2. Bind key property values (configurable which props are "semantic")
    for (key, value) in props {
        if is_semantic_property(key) {
            let prop_fp = Fingerprint::from_content(&format!("{}={}", key, value));
            fp = fp.bind(&prop_fp);
        }
    }

    fp
}
```

**Gotcha**: Which properties are "semantic" (contribute to fingerprint)
vs "metadata" (stored but not in fingerprint) is a configuration choice.
For a Person node, `name` is semantic but `created_at` is not.

**Solution**: Configurable `SemanticSchema`:

```rust
pub struct SemanticSchema {
    /// Properties that contribute to the fingerprint for each label.
    /// If empty, ALL properties contribute (default).
    pub semantic_props: HashMap<String, Vec<String>>,
}
```

### 5.2 — Relationship → Edge Translation

```
Neo4j Relationship:                   Ladybug Edge:
{                                     Fingerprint = src ⊗ verb ⊗ dst
  id: 99,                            where:
  src: 42,                              src = node_42.fingerprint
  dst: 17,                              verb = Verb::from_type("KNOWS")
  type: "KNOWS",                        dst = node_17.fingerprint
  properties: {
    since: 2024
  }
}
```

**Translation: rel_type → Verb mapping**:

```rust
fn verb_from_rel_type(rel_type: &str) -> Fingerprint {
    // First: check if it maps to one of the 144 core verbs
    if let Some(verb) = Verb::from_str(rel_type) {
        return verb.fingerprint();
    }

    // Fallback: hash the rel_type string into a deterministic fingerprint
    Fingerprint::from_content(&format!("VERB:{}", rel_type))
}
```

**The 144 verb mapping table** (partial):

| Neo4j rel_type | Ladybug Verb | Category |
|---------------|-------------|----------|
| `CAUSES` | `Verb::Causes (24)` | Causal |
| `IS_A` | `Verb::IsA (0)` | Structural |
| `KNOWS` | `Verb::ConnectedTo (15)` | Structural |
| `FOLLOWS` | `Verb::Follows (62)` | Temporal |
| `PART_OF` | `Verb::PartOf (2)` | Structural |
| `INFLUENCES` | `Verb::Influences (35)` | Causal |
| `BEFORE` | `Verb::Before (48)` | Temporal |
| `SIMILAR_TO` | `Verb::SimilarTo (6)` | Structural |
| (custom) | `from_content("VERB:custom")` | Hash fallback |

**Gotcha**: Custom relationship types that don't map to the 144 verbs
get hash-fingerprinted. This works for ABBA retrieval but loses the
Go board topology benefits (category-aware search). Acceptable tradeoff.

### 5.3 — Traversal Translation

Neo4j's `MATCH (a)-[:CAUSES*1..5]->(b)` needs to become a Hamming-
accelerated graph walk. The key insight:

```
Neo4j expand(node, OUTGOING, ["CAUSES"], depth=1..5):
  For each step:
    Find all relationships of type CAUSES from current node
    Follow to target node
    Repeat up to 5 times

Ladybug equivalent:
  verb_fp = Verb::Causes.fingerprint()
  current = start_node.fingerprint

  For each step:
    edge_query = current.bind(&verb_fp)  // "what does current CAUSE?"
    candidates = cakes.search(&edge_query, k=100)  // Hamming NN search
    for candidate in candidates:
      recovered = candidate.bind(&verb_fp).bind(&current)  // ABBA recovery
      if recovered.hamming(&some_target) < threshold:
        follow this edge
```

**Gotcha**: This is an APPROXIMATION. CAKES nearest-neighbor search
finds structurally similar edges, not exact matches. For exact
Neo4j-faithful traversal, the backend must ALSO maintain a conventional
adjacency index (src_id → [rel]) and use it for `expand()`.

**Solution**: Dual-path traversal:

```rust
async fn expand(&self, tx: &Self::Tx, node: NodeId, dir: Direction,
                rel_types: &[&str], depth: ExpandDepth) -> Result<Vec<Path>> {
    // EXACT path: use Lance adjacency index (faithful to Neo4j)
    let exact_paths = self.lance_expand(tx, node, dir, rel_types, depth).await?;

    // If fingerprint-accelerated search is enabled, also search by similarity
    // This finds paths that EXACT traversal would miss (semantically similar
    // but not directly connected)
    if self.config.enable_fingerprint_expansion {
        let fp_paths = self.fingerprint_expand(tx, node, dir, rel_types, depth).await?;
        // Merge, deduplicate, return
    }

    Ok(exact_paths)
}
```

**The rule**: Default traversal is EXACT (Neo4j-faithful). Fingerprint-
accelerated traversal is OPT-IN via configuration or explicit `CALL`
procedure.

---

## 6. Procedure Registry (The Extension Surface)

The `call_procedure()` method is where ladybug-rs exposes its unique
capabilities without polluting the Neo4j contract.

### 6.1 — Registered Procedures

```rust
// Inside LadybugBackend
fn init_procedures(&mut self) {
    self.procedures.insert("ladybug.similar", Box::new(|args| {
        // CALL ladybug.similar(query_node_id, k) YIELD node, score
        // Find k most similar nodes by Hamming distance
    }));

    self.procedures.insert("ladybug.bind", Box::new(|args| {
        // CALL ladybug.bind(node_a_id, verb, node_b_id) YIELD edge_fingerprint
        // Create an ABBA-retrievable edge in fingerprint space
    }));

    self.procedures.insert("ladybug.unbind", Box::new(|args| {
        // CALL ladybug.unbind(edge_fp, key_fp) YIELD recovered_fp
        // Recover the other end of a bound edge
    }));

    self.procedures.insert("ladybug.causal_trace", Box::new(|args| {
        // CALL ladybug.causal_trace(effect_id, depth) YIELD path, truth
        // Reverse causal trace with NARS truth values
    }));

    self.procedures.insert("ladybug.collapse_gate", Box::new(|args| {
        // CALL ladybug.collapse_gate(candidate_ids) YIELD state, sd, decision
        // Assess multiple candidate nodes through CollapseGate
    }));

    self.procedures.insert("ladybug.thinking_styles", Box::new(|args| {
        // CALL ladybug.thinking_styles(query_id) YIELD style, result, divergence
        // Run query through all 12 thinking styles, return diverse results
    }));

    self.procedures.insert("ladybug.debate", Box::new(|args| {
        // CALL ladybug.debate(pro_ids, con_ids) YIELD verdict, truth, rounds
        // Run structured debate between pro/con node groups
    }));

    self.procedures.insert("ladybug.counterfactual", Box::new(|args| {
        // CALL ladybug.counterfactual(world_id, intervention_id) YIELD cf_world, divergence
        // Create counterfactual world via Pearl Rung 3 intervention
    }));

    self.procedures.insert("ladybug.hdr_scan", Box::new(|args| {
        // CALL ladybug.hdr_scan(query_fp, radius) YIELD node, distance
        // HDR cascade range scan with Mexican hat receptive field
    }));

    // Neo4j-standard procedures (faithful implementations)
    self.procedures.insert("db.index.vector.queryNodes", Box::new(|args| {
        // Neo4j 5.x compatible vector query
        // Routes to CAKES internally
    }));

    self.procedures.insert("db.labels", Box::new(|args| {
        // Standard Neo4j: list all labels
    }));

    self.procedures.insert("db.relationshipTypes", Box::new(|args| {
        // Standard Neo4j: list all relationship types
    }));
}
```

### 6.2 — Cypher Integration

Users interact entirely through Cypher:

```cypher
// Standard Neo4j (100% faithful)
MATCH (n:Person {name: 'Ada'}) RETURN n

// Standard Neo4j variable-length path
MATCH p = (a:Person)-[:KNOWS*1..3]->(b:Person)
RETURN p

// Neo4j 5.x vector query (standard)
CALL db.index.vector.queryNodes('person_embedding', 10, $query_vector)
YIELD node, score
RETURN node.name, score

// ladybug-rs extension (via CALL)
MATCH (n:Person {name: 'Ada'})
CALL ladybug.similar(n, 10) YIELD similar, score
RETURN similar.name, score

// Cognitive operation (via CALL)
MATCH (pro:Argument)-[:SUPPORTS]->(thesis:Claim)
MATCH (con:Argument)-[:CONTRADICTS]->(thesis)
CALL ladybug.debate(collect(pro), collect(con)) YIELD verdict, truth
RETURN verdict, truth.frequency, truth.confidence

// Causal trace (via CALL)
MATCH (effect:Event {name: 'system_failure'})
CALL ladybug.causal_trace(effect, 5) YIELD path, truth
WHERE truth.confidence > 0.6
RETURN path, truth
```

---

## 7. What neo4j-rs Needs to Give ladybug-rs More Wiggle Room

### The 5 Additions (Minimal, All Neo4j-Faithful)

```rust
// Addition 1: Vector query (Neo4j 5.x compatible)
async fn vector_query(
    &self, tx: &Self::Tx, index_name: &str, k: usize, query_vector: &[u8],
) -> Result<Vec<(NodeId, f64)>>;

// Addition 2: Procedure call (Neo4j CALL mechanism)
async fn call_procedure(
    &self, tx: &Self::Tx, name: &str, args: Vec<Value>,
) -> Result<QueryResult>;

// Addition 3: Batch node creation (optimization, default provided)
async fn create_nodes_batch(
    &self, tx: &mut Self::Tx, nodes: Vec<(&[&str], PropertyMap)>,
) -> Result<Vec<NodeId>>;

// Addition 4: Batch relationship creation (optimization, default provided)
async fn create_relationships_batch(
    &self, tx: &mut Self::Tx, rels: Vec<(NodeId, NodeId, &str, PropertyMap)>,
) -> Result<Vec<RelId>>;

// Addition 5: Backend capabilities query (so planner can optimize)
fn capabilities(&self) -> BackendCapabilities {
    BackendCapabilities::default()
}
```

Where `BackendCapabilities` is:

```rust
#[derive(Debug, Clone, Default)]
pub struct BackendCapabilities {
    pub supports_vector_index: bool,
    pub supports_fulltext_index: bool,
    pub supports_procedures: bool,
    pub supports_batch_writes: bool,
    pub max_batch_size: Option<usize>,
    /// Custom procedure names this backend supports
    pub procedures: Vec<String>,
    /// Hint for planner: this backend is fast at similarity
    pub similarity_accelerated: bool,
}
```

**Why capabilities()**: The planner can use this to generate better physical
plans. If `similarity_accelerated` is true, the optimizer can push Hamming
filters into the scan operator instead of post-filtering. This is exactly
how Neo4j's planner handles different index types.

### What These 5 Additions Enable

| Addition | Neo4j Precedent | What It Unlocks for ladybug-rs |
|----------|----------------|-------------------------------|
| `vector_query` | Neo4j 5.11 vector index | CAKES/HDR similarity search |
| `call_procedure` | APOC / GDS / custom procs | ALL cognitive operations |
| `create_nodes_batch` | `UNWIND` optimization | 100K fp/sec BindSpace load |
| `create_relationships_batch` | `UNWIND` optimization | Bulk edge creation |
| `capabilities` | Index provider hints | Planner optimization |

**Total trait surface increase**: 5 methods, all with default implementations
that return "not supported" or fall back to sequential. Zero breaking changes.
Any existing `StorageBackend` impl compiles unchanged.

---

## 8. Gotchas & Hardening

### 8.1 — Semantic Gotchas

| # | Gotcha | Severity | Mitigation |
|---|--------|----------|------------|
| W-1 | **Fingerprint collision**: Two different nodes could hash to similar fingerprints | Low (2^-16384 for identical, but semantically similar content = similar fp, which is a FEATURE) | Accept: this is cosine-like behavior, not a bug |
| W-2 | **Verb mapping ambiguity**: "KNOWS" could be Structural::ConnectedTo or Social::Familiar | Medium | Configurable verb mapping table per schema |
| W-3 | **Property ordering**: PropertyMap is HashMap, iteration order non-deterministic | High for fingerprint determinism | Sort keys before fingerprinting |
| W-4 | **NULL properties**: Neo4j `SET n.x = null` removes the property | Medium | Recompute fingerprint on property change |
| W-5 | **Transaction isolation**: Fingerprint updates must be atomic with Lance writes | High | Write both in same tx, rollback both on failure |
| W-6 | **DETACH DELETE**: Neo4j's "delete node + all relationships" must also clean BindSpace + CAKES | High | Override delete_node to cascade into fingerprint storage |

### 8.2 — Performance Gotchas

| # | Gotcha | Impact | Mitigation |
|---|--------|--------|------------|
| W-7 | Fingerprint computation on every write | ~1μs per node (SHA-512 + bit packing) | Acceptable. 1M nodes = 1 second. |
| W-8 | CAKES tree rebalancing on insert | O(log n) per insert, but tree build is O(n log n) | Batch inserts, rebuild tree periodically |
| W-9 | Lance compaction during writes | Can block reads for seconds | Run compaction async, not in hot path |
| W-10 | BindSpace memory: 256 u64 per node = 2KB | 1M nodes = 2GB RAM | Use tiered: hot in BindSpace, cold in Lance |

### 8.3 — Correctness Gotchas

| # | Gotcha | The Real Problem | Solution |
|---|--------|-----------------|----------|
| W-11 | `expand()` with fingerprint search finds semantically similar paths, not topologically connected ones | User expects `(a)-[:KNOWS*3]->(b)` to return ACTUAL paths, not "similar" ones | Default expand = EXACT (Lance adjacency). Fingerprint expand = OPT-IN only |
| W-12 | NARS truth values on edges have no Neo4j equivalent | Returning `truth.frequency` in Cypher output is non-standard | Expose via CALL procedures, not RETURN. Or use `_ladybug_truth_f`, `_ladybug_truth_c` properties |
| W-13 | Shortest path algorithms assume distance = hop count, not Hamming distance | `shortestPath()` must use graph topology, not fingerprint similarity | Implement shortestPath on adjacency index, not CAKES |
| W-14 | Neo4j's `id()` function returns internal IDs that are dense integers | Ladybug may use hash-based IDs | Use sequential counter for external IDs, hash for internal fingerprint addressing |

---

## 9. Testing Strategy

### 9.1 — Dual-Backend Verification

The ultimate correctness test: run the same Cypher against BoltBackend
(real Neo4j) and LadybugBackend. Results must match.

```rust
#[tokio::test]
async fn dual_backend_match() {
    let neo4j = Graph::with_backend(BoltBackend::connect("bolt://localhost:7687", "neo4j", "pass").await?);
    let ladybug = Graph::with_backend(LadybugBackend::open("./test_data").await?);

    // Load same data into both
    let cypher = "CREATE (a:Person {name: 'Ada'})-[:KNOWS]->(b:Person {name: 'Jan'})";
    neo4j.mutate(cypher, []).await?;
    ladybug.mutate(cypher, []).await?;

    // Query both
    let q = "MATCH (n:Person) RETURN n.name ORDER BY n.name";
    let neo4j_result = neo4j.execute(q, []).await?;
    let ladybug_result = ladybug.execute(q, []).await?;

    // Results must be identical
    assert_eq!(neo4j_result.rows().len(), ladybug_result.rows().len());
    for (nr, lr) in neo4j_result.rows().iter().zip(ladybug_result.rows()) {
        assert_eq!(nr.get::<String>("n.name"), lr.get::<String>("n.name"));
    }
}
```

### 9.2 — openCypher TCK

neo4j-rs should pass the openCypher Technology Compatibility Kit regardless
of which backend is used. The TCK tests standard Cypher semantics — if
LadybugBackend passes TCK, it's Neo4j-faithful by definition.

### 9.3 — Extension-Specific Tests

```rust
#[tokio::test]
async fn ladybug_similar_returns_neighbors() {
    let graph = Graph::with_backend(LadybugBackend::open_memory().await?);

    // Create nodes with varying similarity
    graph.mutate("CREATE (:Concept {text: 'machine learning'})", []).await?;
    graph.mutate("CREATE (:Concept {text: 'deep learning'})", []).await?;
    graph.mutate("CREATE (:Concept {text: 'quantum physics'})", []).await?;

    // Similar search should rank "deep learning" closer to "machine learning"
    let result = graph.execute(
        "MATCH (n:Concept {text: 'machine learning'})
         CALL ladybug.similar(n, 3) YIELD similar, score
         RETURN similar.text, score ORDER BY score DESC",
        [],
    ).await?;

    let first = result.rows()[0].get::<String>("similar.text")?;
    assert_eq!(first, "deep learning"); // Semantic similarity
}
```

---

## 10. Implementation Roadmap

```
Phase 1: neo4j-rs Trait Additions (1 week)
  ├── Add vector_query() with default "not supported"
  ├── Add call_procedure() with default "not found"
  ├── Add create_nodes_batch() with default sequential fallback
  ├── Add capabilities() with default struct
  ├── All existing backends compile unchanged
  └── Planner recognizes CALL statements

Phase 2: LadybugBackend Skeleton (1 week)
  ├── src/backend/mod.rs in ladybug-rs
  ├── impl StorageBackend for LadybugBackend
  ├── Node CRUD → BindSpace + Lance
  ├── Relationship CRUD → Lance adjacency + fingerprint edges
  ├── Simple expand() via Lance adjacency (EXACT, no fingerprint)
  └── All TCK-basic tests pass

Phase 3: Fingerprint Integration (1 week)
  ├── fingerprint_from_node() translation function
  ├── verb_from_rel_type() mapping (144 verbs + hash fallback)
  ├── SemanticSchema configuration
  ├── Auto-fingerprint on create_node / create_relationship
  ├── vector_query() → CAKES/HDR search
  └── Dual-backend verification tests pass

Phase 4: Procedure Registry (1 week)
  ├── ladybug.similar → CAKES nearest neighbor
  ├── ladybug.bind / ladybug.unbind → ABBA algebra
  ├── ladybug.causal_trace → CausalTrace with NARS truth
  ├── ladybug.collapse_gate → CollapseGate assessment
  ├── ladybug.debate → Structured debate with verdict
  └── Integration tests for all procedures

Phase 5: Cognitive Acceleration (ongoing)
  ├── Fingerprint-accelerated expand() (opt-in)
  ├── CAKES-boosted nodes_by_property() for similarity
  ├── ThinkingStyle-diverse search results
  ├── Counterfactual world creation via CALL
  └── Real-world benchmarks vs Neo4j
```

---

## Summary: The Contract

```
neo4j-rs promises:
  ✓ 100% openCypher compatible Cypher parser
  ✓ Full property graph model (Node, Rel, Path, Value)
  ✓ ACID transactions
  ✓ StorageBackend trait with 5 new methods (all backward compatible)
  ✓ Bolt backend for real Neo4j (correctness oracle)
  ✓ CALL procedure mechanism for extensions

ladybug-rs promises:
  ✓ Implement StorageBackend faithfully
  ✓ Default expand() uses EXACT traversal (not approximate)
  ✓ Fingerprint operations exposed ONLY via CALL procedures
  ✓ No cognitive concepts leak into the property graph model
  ✓ Same Cypher queries produce same results on both backends
  ✓ Cognitive acceleration is OPT-IN, never default

The translation layer promises:
  ✓ Node → Fingerprint is deterministic and configurable
  ✓ RelType → Verb uses the 144 Go board verbs when possible
  ✓ Properties are sorted before fingerprinting
  ✓ BindSpace and Lance are always consistent
  ✓ Rollback cleans both stores
  ✓ Batch operations don't bypass transaction semantics
```

The user sees a Neo4j-compatible graph database. Under the hood, every
node is a 16384-bit vector in a CAKES tree with NARS truth values on
its edges. The user never needs to know — unless they `CALL ladybug.*`
and ask for the cognitive substrate directly.
