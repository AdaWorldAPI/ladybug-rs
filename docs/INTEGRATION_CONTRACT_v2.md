# neo4j-rs ↔ ladybug-rs Integration Contract v2
## Addressing All 5 Gap Categories

> **Context**: This document responds to the gap analysis from the neo4j-rs
> session and provides the definitive trait surface with ladybug-rs translation
> table. Every addition has a Neo4j precedent and a ladybug-rs implementation path.

---

## 1. Gap Resolution Summary

| Gap | Category | Items | Resolution |
|-----|----------|-------|------------|
| **A** | ARCHITECTURE.md vs code | `connect()`, `execute_raw()` | Add both to trait |
| **B** | 8 missing operation categories | Rel props, DETACH DELETE, MERGE, shortest path, scans, batch, constraints, degree | Add 12 methods to trait |
| **C** | ladybug-rs exports | LadybugBackend, LadybugTx, LadybugConfig | Define in new `ladybug-rs/src/backend/` module |
| **D** | Transaction trait too thin | bookmark, database, timeout | Extend Transaction trait |
| **E** | No capability negotiation | Backend hints for planner | Add `capabilities()` method |

**Total trait change**: +17 methods on StorageBackend, +3 on Transaction, +1 capabilities struct. All with default implementations. Zero breaking change for existing backends.

---

## 2. The Complete StorageBackend Trait (v2)

Below is the full trait with every gap addressed. New methods marked with their gap letter.

```rust
#[async_trait]
pub trait StorageBackend: Send + Sync + 'static {
    type Tx: Transaction;

    // ========================================================================
    // Lifecycle  [Gap A: connect was missing]
    // ========================================================================

    /// Connect to the storage backend.
    /// Each backend type has its own config variant.
    async fn connect(config: BackendConfig) -> Result<Self> where Self: Sized;

    /// Shut down, flushing pending writes.
    async fn shutdown(&self) -> Result<()>;

    // ========================================================================
    // Transactions
    // ========================================================================

    async fn begin_tx(&self, mode: TxMode) -> Result<Self::Tx>;
    async fn commit_tx(&self, tx: Self::Tx) -> Result<()>;
    async fn rollback_tx(&self, tx: Self::Tx) -> Result<()>;

    // ========================================================================
    // Node CRUD  (unchanged from v1)
    // ========================================================================

    async fn create_node(
        &self, tx: &mut Self::Tx, labels: &[&str], props: PropertyMap,
    ) -> Result<NodeId>;

    async fn get_node(&self, tx: &Self::Tx, id: NodeId) -> Result<Option<Node>>;

    async fn delete_node(&self, tx: &mut Self::Tx, id: NodeId) -> Result<bool>;

    async fn set_node_property(
        &self, tx: &mut Self::Tx, id: NodeId, key: &str, val: Value,
    ) -> Result<()>;

    async fn remove_node_property(
        &self, tx: &mut Self::Tx, id: NodeId, key: &str,
    ) -> Result<()>;

    async fn add_label(&self, tx: &mut Self::Tx, id: NodeId, label: &str) -> Result<()>;
    async fn remove_label(&self, tx: &mut Self::Tx, id: NodeId, label: &str) -> Result<()>;

    // ========================================================================
    // Node CRUD additions  [Gap B: detach delete, merge]
    // ========================================================================

    /// DETACH DELETE: delete node and all its relationships in one call.
    /// Neo4j: `DETACH DELETE n`
    /// Default: get_relationships + delete each + delete_node (slow but correct).
    async fn detach_delete_node(&self, tx: &mut Self::Tx, id: NodeId) -> Result<bool> {
        // Delete all relationships first
        let rels = self.get_relationships(tx, id, Direction::Both, None).await?;
        for rel in &rels {
            self.delete_relationship(tx, rel.id).await?;
        }
        self.delete_node(tx, id).await
    }

    /// MERGE: atomic upsert — find-or-create a node matching labels + key props.
    /// Neo4j: `MERGE (n:Label {key: val}) ON CREATE SET ... ON MATCH SET ...`
    /// Returns (NodeId, created: bool).
    async fn merge_node(
        &self,
        tx: &mut Self::Tx,
        labels: &[&str],
        match_props: PropertyMap,       // Properties used to find existing
        on_create_props: PropertyMap,    // Additional props if creating
        on_match_props: PropertyMap,     // Props to set if found
    ) -> Result<(NodeId, bool)> {
        // Default: lookup + branch (TOCTOU risk without backend atomicity)
        let label = labels.first().map(|s| *s).unwrap_or("");
        for (key, val) in &match_props {
            let found = self.nodes_by_property(tx, label, key, val).await?;
            if let Some(node) = found.first() {
                // ON MATCH: set properties
                for (k, v) in on_match_props {
                    self.set_node_property(tx, node.id, &k, v).await?;
                }
                return Ok((node.id, false));
            }
        }
        // ON CREATE: create with match_props + on_create_props
        let mut all_props = match_props;
        all_props.extend(on_create_props);
        let id = self.create_node(tx, labels, all_props).await?;
        Ok((id, true))
    }

    // ========================================================================
    // Relationship CRUD  [Gap B: relationship property CRUD was missing]
    // ========================================================================

    async fn create_relationship(
        &self, tx: &mut Self::Tx, src: NodeId, dst: NodeId,
        rel_type: &str, props: PropertyMap,
    ) -> Result<RelId>;

    async fn get_relationship(&self, tx: &Self::Tx, id: RelId) -> Result<Option<Relationship>>;

    async fn delete_relationship(&self, tx: &mut Self::Tx, id: RelId) -> Result<bool>;

    /// SET r.prop = val  [Gap B: was missing entirely]
    async fn set_relationship_property(
        &self, tx: &mut Self::Tx, id: RelId, key: &str, val: Value,
    ) -> Result<()>;

    /// REMOVE r.prop  [Gap B: was missing entirely]
    async fn remove_relationship_property(
        &self, tx: &mut Self::Tx, id: RelId, key: &str,
    ) -> Result<()>;

    // ========================================================================
    // Traversal
    // ========================================================================

    async fn get_relationships(
        &self, tx: &Self::Tx, node: NodeId, dir: Direction, rel_type: Option<&str>,
    ) -> Result<Vec<Relationship>>;

    async fn expand(
        &self, tx: &Self::Tx, node: NodeId, dir: Direction,
        rel_types: &[&str], depth: ExpandDepth,
    ) -> Result<Vec<Path>>;

    /// shortestPath((a)-[*..maxDepth]->(b))  [Gap B: was missing]
    /// Returns None if no path exists within maxDepth.
    async fn shortest_path(
        &self,
        tx: &Self::Tx,
        src: NodeId,
        dst: NodeId,
        dir: Direction,
        rel_types: &[&str],
        max_depth: usize,
    ) -> Result<Option<Path>> {
        // Default: BFS via expand (correct but not optimized)
        let paths = self.expand(tx, src, dir, rel_types, ExpandDepth::Range { min: 1, max: max_depth }).await?;
        Ok(paths.into_iter()
            .filter(|p| p.end_node().map(|n| n.id) == Some(dst))
            .min_by_key(|p| p.length()))
    }

    /// allShortestPaths  [Gap B]
    async fn all_shortest_paths(
        &self,
        tx: &Self::Tx,
        src: NodeId,
        dst: NodeId,
        dir: Direction,
        rel_types: &[&str],
        max_depth: usize,
    ) -> Result<Vec<Path>> {
        // Default: find shortest, then find all of that length
        if let Some(shortest) = self.shortest_path(tx, src, dst, dir, rel_types, max_depth).await? {
            let len = shortest.length();
            let paths = self.expand(tx, src, dir, rel_types, ExpandDepth::Exact(len)).await?;
            Ok(paths.into_iter()
                .filter(|p| p.end_node().map(|n| n.id) == Some(dst))
                .collect())
        } else {
            Ok(vec![])
        }
    }

    /// Degree count without materializing relationships  [Gap B]
    /// Neo4j: `size((n)-->())` or `size((n)-[:TYPE]->())`
    async fn degree(
        &self,
        tx: &Self::Tx,
        node: NodeId,
        dir: Direction,
        rel_type: Option<&str>,
    ) -> Result<u64> {
        // Default: materialize and count
        Ok(self.get_relationships(tx, node, dir, rel_type).await?.len() as u64)
    }

    // ========================================================================
    // Scan  [Gap B: scan gaps]
    // ========================================================================

    /// All nodes (no label filter).  [Gap B: was missing]
    /// Neo4j: `MATCH (n) RETURN n`
    async fn all_nodes(&self, tx: &Self::Tx) -> Result<Vec<Node>>;

    /// All relationships of a given type (no start node filter).  [Gap B: was missing]
    /// Neo4j: `MATCH ()-[r:TYPE]->() RETURN r`
    async fn relationships_by_type(
        &self, tx: &Self::Tx, rel_type: &str,
    ) -> Result<Vec<Relationship>> {
        // Default: scan all nodes, collect relationships by type
        // Backends should override with index-backed scan
        let mut result = Vec::new();
        let nodes = self.all_nodes(tx).await?;
        for node in &nodes {
            let rels = self.get_relationships(tx, node.id, Direction::Outgoing, Some(rel_type)).await?;
            result.extend(rels);
        }
        Ok(result)
    }

    /// Range scan on property value.  [Gap B: was missing]
    /// Neo4j: `WHERE n.age > 25 AND n.age < 50`
    async fn nodes_by_property_range(
        &self,
        tx: &Self::Tx,
        label: &str,
        key: &str,
        min: Option<&Value>,  // None = no lower bound
        max: Option<&Value>,  // None = no upper bound
    ) -> Result<Vec<Node>> {
        // Default: scan by label, filter in memory
        let all = self.nodes_by_label(tx, label).await?;
        Ok(all.into_iter().filter(|n| {
            if let Some(val) = n.get(key) {
                let above_min = min.map_or(true, |m| val.neo4j_cmp(m).map_or(false, |o| o != std::cmp::Ordering::Less));
                let below_max = max.map_or(true, |m| val.neo4j_cmp(m).map_or(false, |o| o != std::cmp::Ordering::Greater));
                above_min && below_max
            } else {
                false
            }
        }).collect())
    }

    async fn nodes_by_label(&self, tx: &Self::Tx, label: &str) -> Result<Vec<Node>>;

    async fn nodes_by_property(
        &self, tx: &Self::Tx, label: &str, key: &str, value: &Value,
    ) -> Result<Vec<Node>>;

    // ========================================================================
    // Index + Constraints  [Gap B: constraints missing]
    // ========================================================================

    async fn create_index(
        &self, label: &str, property: &str, index_type: IndexType,
    ) -> Result<()>;

    async fn drop_index(&self, label: &str, property: &str) -> Result<()>;

    /// CREATE CONSTRAINT ... UNIQUE  [Gap B: was missing]
    async fn create_constraint(
        &self,
        label: &str,
        property: &str,
        constraint_type: ConstraintType,
    ) -> Result<()> {
        Err(Error::ExecutionError("constraints not supported".into()))
    }

    /// DROP CONSTRAINT  [Gap B]
    async fn drop_constraint(
        &self, label: &str, property: &str,
    ) -> Result<()> {
        Err(Error::ExecutionError("constraints not supported".into()))
    }

    // ========================================================================
    // Schema introspection
    // ========================================================================

    async fn node_count(&self, tx: &Self::Tx) -> Result<u64>;
    async fn relationship_count(&self, tx: &Self::Tx) -> Result<u64>;
    async fn labels(&self, tx: &Self::Tx) -> Result<Vec<String>>;
    async fn relationship_types(&self, tx: &Self::Tx) -> Result<Vec<String>>;

    // ========================================================================
    // Batch operations  [Gap B: performance]
    // ========================================================================

    /// Batch node creation (100-1000x for Lance columnar writes).
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

    /// Batch relationship creation.
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

    // ========================================================================
    // Escape hatches  [Gap A: execute_raw was missing]
    // ========================================================================

    /// Pass-through for backend-native queries.
    /// Bolt: forwards raw Cypher to Neo4j.
    /// Ladybug: forwards to DataFusion SQL engine.
    /// Memory: not supported.
    async fn execute_raw(
        &self,
        tx: &Self::Tx,
        query: &str,
        params: PropertyMap,
    ) -> Result<QueryResult> {
        Err(Error::ExecutionError("raw query execution not supported".into()))
    }

    /// Call a registered procedure.
    /// Neo4j: `CALL db.labels()`, `CALL apoc.create.node()`
    /// Ladybug: `CALL ladybug.similar()`, `CALL ladybug.bind()`
    async fn call_procedure(
        &self,
        tx: &Self::Tx,
        name: &str,
        args: Vec<Value>,
    ) -> Result<QueryResult> {
        Err(Error::ExecutionError(format!("procedure '{}' not found", name)))
    }

    /// Vector similarity search (Neo4j 5.x compatible).
    async fn vector_query(
        &self,
        tx: &Self::Tx,
        index_name: &str,
        k: usize,
        query_vector: &[u8],
    ) -> Result<Vec<(NodeId, f64)>> {
        Err(Error::ExecutionError("vector index not supported".into()))
    }

    // ========================================================================
    // Capability negotiation  [Gap E]
    // ========================================================================

    /// What this backend can do natively vs what needs emulation.
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities::default()
    }
}
```

### Constraint Type Enum

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintType {
    /// Node property uniqueness within a label
    Unique,
    /// Node property must exist (NOT NULL)
    Exists,
    /// Node key (UNIQUE + EXISTS)
    NodeKey,
}
```

### BackendCapabilities Struct [Gap E]

```rust
#[derive(Debug, Clone)]
pub struct BackendCapabilities {
    // --- Core ---
    pub supports_merge: bool,
    pub supports_detach_delete: bool,   // native batch vs emulated
    pub supports_constraints: bool,

    // --- Traversal ---
    pub supports_shortest_path: bool,   // native BFS vs emulated
    pub supports_all_shortest_paths: bool,

    // --- Scan ---
    pub supports_range_scan: bool,      // index-backed range vs memory filter
    pub supports_fulltext_index: bool,
    pub supports_vector_index: bool,

    // --- Performance ---
    pub supports_batch_writes: bool,
    pub max_batch_size: Option<usize>,
    pub supports_degree_count: bool,    // O(1) vs materialize-and-count

    // --- Extensions ---
    pub supports_procedures: bool,
    pub supported_procedures: Vec<String>,
    pub supports_execute_raw: bool,

    // --- Planner hints ---
    pub similarity_accelerated: bool,   // Hamming-accelerated search
    pub filter_pushdown: FilterPushdownLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterPushdownLevel {
    None,       // MemoryBackend: all filtering in execution engine
    Full,       // BoltBackend: Neo4j handles everything
    Selective,  // LadybugBackend: DataFusion handles some, BindSpace handles Hamming
}

impl Default for BackendCapabilities {
    fn default() -> Self {
        Self {
            supports_merge: false,
            supports_detach_delete: false,
            supports_constraints: false,
            supports_shortest_path: false,
            supports_all_shortest_paths: false,
            supports_range_scan: false,
            supports_fulltext_index: false,
            supports_vector_index: false,
            supports_batch_writes: false,
            max_batch_size: None,
            supports_degree_count: false,
            supports_procedures: false,
            supported_procedures: vec![],
            supports_execute_raw: false,
            similarity_accelerated: false,
            filter_pushdown: FilterPushdownLevel::None,
        }
    }
}
```

---

## 3. The Complete Transaction Trait (v2) [Gap D]

```rust
pub trait Transaction: Send + Sync {
    fn mode(&self) -> TxMode;
    fn id(&self) -> TxId;

    /// Causal consistency bookmark.  [Gap D]
    /// Bolt: Neo4j bookmark string.
    /// Ladybug: Lance version number.
    /// Memory: None.
    fn bookmark(&self) -> Option<String> { None }

    /// Which database this transaction targets.  [Gap D]
    /// Neo4j 4.0+: multi-database support.
    /// Ladybug: could map to different Lance datasets.
    fn database(&self) -> &str { "neo4j" }

    /// Transaction timeout.  [Gap D]
    fn timeout(&self) -> Option<std::time::Duration> { None }
}
```

---

## 4. What Each Backend Reports

### MemoryBackend

```rust
fn capabilities(&self) -> BackendCapabilities {
    BackendCapabilities {
        supports_merge: true,           // simple HashMap upsert
        supports_detach_delete: true,    // iterate + delete
        supports_constraints: false,     // no persistent schema
        supports_shortest_path: false,   // BFS emulation only
        supports_range_scan: false,      // memory scan (no index)
        supports_batch_writes: false,    // sequential create
        supports_degree_count: true,     // HashMap::len() is O(1)
        filter_pushdown: FilterPushdownLevel::None,
        ..Default::default()
    }
}
```

### BoltBackend

```rust
fn capabilities(&self) -> BackendCapabilities {
    BackendCapabilities {
        supports_merge: true,
        supports_detach_delete: true,
        supports_constraints: true,
        supports_shortest_path: true,
        supports_all_shortest_paths: true,
        supports_range_scan: true,
        supports_fulltext_index: true,
        supports_vector_index: self.neo4j_version >= Version(5, 11),
        supports_batch_writes: true,
        supports_degree_count: true,
        supports_procedures: true,
        supported_procedures: vec![/* discovered via CALL dbms.procedures() */],
        supports_execute_raw: true,
        filter_pushdown: FilterPushdownLevel::Full,
        ..Default::default()
    }
}
```

### LadybugBackend

```rust
fn capabilities(&self) -> BackendCapabilities {
    BackendCapabilities {
        supports_merge: true,           // Lance INSERT ON CONFLICT
        supports_detach_delete: true,    // Batch Lance DELETE
        supports_constraints: true,      // Lance unique index
        supports_shortest_path: true,    // HDR cascade + BFS
        supports_all_shortest_paths: true,
        supports_range_scan: true,       // DataFusion filter pushdown
        supports_fulltext_index: false,  // not yet
        supports_vector_index: true,     // CAKES + HDR = THE value prop
        supports_batch_writes: true,     // Lance columnar batch
        max_batch_size: Some(100_000),
        supports_degree_count: true,     // DN-Sparse CSR metadata
        supports_procedures: true,
        supported_procedures: vec![
            "ladybug.similar".into(),
            "ladybug.bind".into(),
            "ladybug.unbind".into(),
            "ladybug.causal_trace".into(),
            "ladybug.collapse_gate".into(),
            "ladybug.debate".into(),
            "ladybug.thinking_styles".into(),
            "ladybug.counterfactual".into(),
            "ladybug.hdr_scan".into(),
            "db.index.vector.queryNodes".into(),
        ],
        supports_execute_raw: true,      // DataFusion SQL
        similarity_accelerated: true,
        filter_pushdown: FilterPushdownLevel::Selective,
    }
}
```

---

## 5. ladybug-rs Translation Table [Gap C]

Every StorageBackend method → concrete ladybug-rs implementation path.

### Node Operations

| Trait Method | Lance Operation | BindSpace Operation | Fingerprint Operation |
|-------------|----------------|--------------------|-----------------------|
| `create_node` | INSERT into nodes table | `bind_space.insert(id, fp)` | `fingerprint_from_node(labels, props)` |
| `get_node` | SELECT by id | — (Lance is source of truth) | — |
| `delete_node` | DELETE WHERE id = ? | `bind_space.remove(id)` | — |
| `detach_delete_node` | DELETE nodes + edges WHERE src/dst = id | `bind_space.remove(id)` + remove edge fps | — |
| `set_node_property` | UPDATE nodes SET props = ? WHERE id = ? | Recompute fp if semantic prop | `fingerprint_from_node(labels, new_props)` |
| `remove_node_property` | UPDATE nodes SET props = ? WHERE id = ? | Recompute fp if semantic prop | `fingerprint_from_node(labels, new_props)` |
| `add_label` | UPDATE nodes SET labels = ? WHERE id = ? | Recompute fp (labels change fingerprint) | `fp = fp.bind(&from_content(new_label))` |
| `remove_label` | UPDATE nodes SET labels = ? WHERE id = ? | Recompute fp | `fp = fp.bind(&from_content(old_label))` (unbind) |
| `merge_node` | `INSERT ... ON CONFLICT UPDATE` | Insert or update fp | Compute fp for match |

### Relationship Operations

| Trait Method | Lance Operation | FingerprintGraph Operation | Verb Mapping |
|-------------|----------------|--------------------------|----|
| `create_relationship` | INSERT into edges table | `graph.add_edge(src_fp, verb_fp, dst_fp)` | `verb_from_rel_type(rel_type)` |
| `get_relationship` | SELECT by id | — | — |
| `delete_relationship` | DELETE WHERE id = ? | Remove from FingerprintGraph | — |
| `set_relationship_property` | UPDATE edges SET props WHERE id = ? | — (props don't affect edge fingerprint) | — |
| `remove_relationship_property` | UPDATE edges SET props WHERE id = ? | — | — |

### Traversal Operations

| Trait Method | Primary (EXACT) | Accelerated (OPT-IN) |
|-------------|-----------------|---------------------|
| `get_relationships` | Lance: `SELECT * FROM edges WHERE src_id = ? AND type = ?` | — |
| `expand` | Lance adjacency BFS/DFS | FingerprintGraph.traverse_n_hops() |
| `shortest_path` | Lance BFS with early termination | HDR cascade prunes 90% of candidates, then BFS on survivors |
| `all_shortest_paths` | Lance BFS (find length, then enumerate) | HDR cascade + enumerate at shortest length |
| `degree` | `SELECT COUNT(*) FROM edges WHERE src_id = ?` | DN-Sparse CSR: `adjacency[node].len()` (O(1)) |

### Scan Operations

| Trait Method | DataFusion SQL | Lance Native |
|-------------|---------------|-------------|
| `all_nodes` | `SELECT * FROM nodes` | Full table scan |
| `nodes_by_label` | `SELECT * FROM nodes WHERE labels @> ARRAY['L']` | Arrow filter on labels column |
| `nodes_by_property` | `SELECT * FROM nodes WHERE props->>'key' = ?` | JSON property filter pushdown |
| `nodes_by_property_range` | `SELECT * FROM nodes WHERE CAST(props->>'key' AS INT) BETWEEN ? AND ?` | DataFusion range filter |
| `relationships_by_type` | `SELECT * FROM edges WHERE type = ?` | Arrow filter on type column |

### Extension Operations

| Trait Method | ladybug-rs Implementation |
|-------------|--------------------------|
| `execute_raw` | DataFusion `SqlEngine.execute(query)` — SQL not Cypher |
| `call_procedure("ladybug.similar")` | `cakes.search(&query_fp, k)` → CAKES nearest neighbor |
| `call_procedure("ladybug.bind")` | `src_fp.bind(&verb_fp).bind(&dst_fp)` → ABBA edge |
| `call_procedure("ladybug.causal_trace")` | `CausalTrace::reverse_trace()` from search/causal.rs |
| `call_procedure("ladybug.collapse_gate")` | `calculate_sd(scores)` → `get_gate_state(sd)` |
| `call_procedure("ladybug.debate")` | `debate(&pro_args, &con_args, &config)` from orchestration/debate.rs |
| `vector_query` | `cakes.search(&query_fp, k)` → HDR cascade + CAKES |

### Index + Constraint Operations

| Trait Method | Lance Implementation |
|-------------|---------------------|
| `create_index(BTree)` | Lance scalar index on property column |
| `create_index(FullText)` | Not supported (return error) |
| `create_index(Unique)` | Lance scalar index + unique constraint |
| `create_index(Vector)` | Lance IVF-PQ index on fingerprint column |
| `create_constraint(Unique)` | Lance unique index on (label, property) |
| `create_constraint(Exists)` | NOT NULL constraint on property column |
| `create_constraint(NodeKey)` | Unique + Exists combined |

---

## 6. LadybugBackend + LadybugTx Definition [Gap C]

```rust
// ladybug-rs/src/backend/mod.rs

pub mod storage_impl;
pub mod tx_impl;
pub mod translate;
pub mod procedures;

pub use storage_impl::LadybugBackend;
pub use tx_impl::LadybugTx;

/// Configuration for the ladybug-rs storage backend.
#[derive(Debug, Clone)]
pub struct LadybugConfig {
    /// Path to Lance dataset directory.
    pub data_dir: PathBuf,

    /// BindSpace size (number of fingerprint slots).
    /// Default: 65536
    pub bind_space_capacity: usize,

    /// Enable CAKES tree for similarity search.
    /// Default: true
    pub enable_cakes: bool,

    /// Enable fingerprint-accelerated traversal (in addition to exact).
    /// Default: false (must be explicitly opted into)
    pub enable_fp_traversal: bool,

    /// Semantic schema: which properties contribute to fingerprints.
    /// None = all properties contribute (default).
    pub semantic_schema: Option<SemanticSchema>,

    /// DataFusion parallelism.
    /// Default: number of CPU cores
    pub datafusion_threads: Option<usize>,

    /// Lance cache size in MB.
    /// Default: 256
    pub cache_size_mb: usize,
}

impl Default for LadybugConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("./ladybug_data"),
            bind_space_capacity: 65536,
            enable_cakes: true,
            enable_fp_traversal: false,
            semantic_schema: None,
            datafusion_threads: None,
            cache_size_mb: 256,
        }
    }
}

/// Semantic schema: controls which properties affect fingerprints.
#[derive(Debug, Clone)]
pub struct SemanticSchema {
    /// label → list of property names that are "semantic".
    /// If a label is not in this map, ALL its properties are semantic.
    pub semantic_props: HashMap<String, Vec<String>>,
}
```

```rust
// ladybug-rs/src/backend/tx_impl.rs

pub struct LadybugTx {
    id: TxId,
    mode: TxMode,
    database: String,
    lance_version: u64,         // Snapshot version for read consistency
    timeout: Option<Duration>,
    journal: Vec<TxEntry>,      // Write-ahead log for rollback
}

enum TxEntry {
    CreateNode(NodeId),
    DeleteNode(NodeId, Node),   // Store deleted node for rollback
    CreateRel(RelId),
    DeleteRel(RelId, Relationship),
    SetProperty(NodeId, String, Option<Value>),  // key, old value
    // ... etc
}

impl Transaction for LadybugTx {
    fn mode(&self) -> TxMode { self.mode }
    fn id(&self) -> TxId { self.id }
    fn bookmark(&self) -> Option<String> {
        Some(format!("lance:v{}", self.lance_version))
    }
    fn database(&self) -> &str { &self.database }
    fn timeout(&self) -> Option<Duration> { self.timeout }
}
```

### Error Conversion

```rust
// ladybug-rs → neo4j-rs error mapping

impl From<ladybug::Error> for neo4j_rs::Error {
    fn from(e: ladybug::Error) -> Self {
        match e {
            ladybug::Error::LanceError(e) => neo4j_rs::Error::StorageError(e.to_string()),
            ladybug::Error::DataFusionError(e) => neo4j_rs::Error::ExecutionError(e.to_string()),
            ladybug::Error::NotFound(msg) => neo4j_rs::Error::NotFound(msg),
            ladybug::Error::InvalidFingerprint(msg) => neo4j_rs::Error::StorageError(msg),
            _ => neo4j_rs::Error::StorageError(e.to_string()),
        }
    }
}
```

---

## 7. How the Planner Uses Capabilities [Gap E]

The planner's job changes based on what the backend can do:

```rust
// neo4j-rs/src/planner/mod.rs

pub fn plan_with_capabilities(
    ast: &Statement,
    params: &PropertyMap,
    caps: &BackendCapabilities,
) -> Result<LogicalPlan> {
    let base_plan = plan(ast, params)?;
    optimize_for_backend(base_plan, caps)
}

fn optimize_for_backend(plan: LogicalPlan, caps: &BackendCapabilities) -> Result<LogicalPlan> {
    match caps.filter_pushdown {
        FilterPushdownLevel::Full => {
            // Bolt: push everything down, let Neo4j optimize
            Ok(plan)
        }
        FilterPushdownLevel::Selective => {
            // Ladybug: push label/property filters to DataFusion,
            // push similarity filters to CAKES,
            // keep complex path patterns in execution engine
            selective_pushdown(plan, caps)
        }
        FilterPushdownLevel::None => {
            // Memory: pull all data, filter in engine
            Ok(plan)
        }
    }
}

fn selective_pushdown(plan: LogicalPlan, caps: &BackendCapabilities) -> Result<LogicalPlan> {
    // Example: if the plan has a NodeScan + Filter(similarity > 0.8),
    // and caps.similarity_accelerated is true,
    // rewrite to VectorScan with threshold
    match plan {
        LogicalPlan::Filter { input, predicate } if is_similarity_predicate(&predicate) && caps.similarity_accelerated => {
            // Rewrite: pull similarity threshold from predicate,
            // replace NodeScan + Filter with VectorQuery
            rewrite_to_vector_query(*input, predicate)
        }
        _ => Ok(plan)
    }
}
```

---

## 8. Method Count Summary

### StorageBackend Trait

| Category | v1 Methods | v2 Additions | v2 Total |
|----------|-----------|-------------|---------|
| Lifecycle | 1 (shutdown) | +1 (connect) | 2 |
| Transactions | 3 | 0 | 3 |
| Node CRUD | 7 | +2 (detach_delete, merge) | 9 |
| Relationship CRUD | 3 | +2 (set_rel_prop, remove_rel_prop) | 5 |
| Traversal | 2 | +3 (shortest, all_shortest, degree) | 5 |
| Scan | 2 | +3 (all_nodes, rels_by_type, range) | 5 |
| Index | 2 | +2 (create/drop constraint) | 4 |
| Schema | 4 | 0 | 4 |
| Escape hatches | 0 | +3 (execute_raw, call_procedure, vector_query) | 3 |
| Capabilities | 0 | +1 (capabilities) | 1 |
| Batch | 0 | +2 (batch nodes, batch rels) | 2 |
| **TOTAL** | **24** | **+19** | **43** |

### Transaction Trait

| v1 | v2 Additions | v2 Total |
|----|-------------|---------|
| 2 (mode, id) | +3 (bookmark, database, timeout) | 5 |

### Breaking Changes: ZERO

Every new method has a default implementation. Existing MemoryBackend compiles unchanged. The only REQUIRED addition for new backends is `all_nodes()` (no default possible without scanning something).

Wait — `all_nodes()` has no default and is new. That's a breaking change for MemoryBackend. Fix:

```rust
// MemoryBackend addition (trivial):
async fn all_nodes(&self, _tx: &MemoryTx) -> Result<Vec<Node>> {
    Ok(self.nodes.read().await.values().cloned().collect())
}
```

For `connect()` — this IS an associated function, not a method. It's on the trait but implementations define it. No default possible. But MemoryBackend already has `new()`, which maps to `connect(BackendConfig::Memory)`.

Actually `connect()` should be a separate constructor, not a trait method, because trait methods with `Self: Sized` can't be called through `dyn StorageBackend`. Better pattern:

```rust
// Instead of connect() on trait, use associated functions per backend:
impl MemoryBackend {
    pub fn new() -> Self { ... }
}

impl BoltBackend {
    pub async fn connect(uri: &str, user: &str, pass: &str) -> Result<Self> { ... }
}

impl LadybugBackend {
    pub async fn open(config: LadybugConfig) -> Result<Self> { ... }
}

// In BackendConfig, keep for documentation but don't force on trait
```

---

## 9. Priority Implementation Order

### Must Have (blocks Cypher compliance)

```
1. set_relationship_property     — 10 lines in MemoryBackend
2. remove_relationship_property  — 10 lines in MemoryBackend
3. detach_delete_node            — default impl exists, override in Memory
4. all_nodes                     — 5 lines in MemoryBackend
5. create_constraint             — default returns error (acceptable)
6. drop_constraint               — default returns error (acceptable)
7. execute_raw                   — default returns error (acceptable)
8. connect per-backend           — rename existing constructors
```

### Should Have (faithful semantics)

```
9.  merge_node                   — default impl exists (TOCTOU risk ok for v1)
10. shortest_path                — default BFS via expand
11. all_shortest_paths           — default via shortest_path
12. nodes_by_property_range      — default scan-and-filter
13. relationships_by_type        — default scan-and-filter
14. bookmark on Transaction      — default None
15. database on Transaction      — default "neo4j"
16. timeout on Transaction       — default None
```

### Nice to Have (performance)

```
17. degree                       — default materializes (override in Ladybug for O(1))
18. create_nodes_batch           — default sequential (override in Ladybug for columnar)
19. create_relationships_batch   — default sequential
20. capabilities                 — default struct
21. call_procedure               — default "not found"
22. vector_query                 — default "not supported"
```

**Critical path**: Items 1-8 can all be done in one session. Items 9-16
use default implementations that work immediately. Items 17-22 are pure
optimization — ladybug-rs overrides the defaults for performance, but
correctness is never blocked.

---

## 10. The One-Page Contract

```
neo4j-rs GUARANTEES:
  • Full openCypher parser with MERGE, DETACH DELETE, SET r.prop, constraints
  • Planner calls capabilities() and adapts
  • Execution engine walks plan tree, calls StorageBackend methods
  • CALL statements route to call_procedure()
  • Vector queries route to vector_query()
  • Bolt backend passes TCK (correctness oracle)

StorageBackend GUARANTEES:
  • 43 methods, 19 with defaults (no breaking change)
  • Transaction with bookmark + database + timeout
  • BackendCapabilities for planner optimization hints
  • connect() as per-backend associated function (not trait method)

LadybugBackend GUARANTEES:
  • impl StorageBackend with full CRUD
  • Node → Fingerprint via configurable SemanticSchema
  • RelType → Verb via 144-verb table + hash fallback
  • Default expand() is EXACT (Lance adjacency)
  • Fingerprint traversal is OPT-IN only
  • All cognitive ops via CALL ladybug.* procedures
  • Batch writes use Lance columnar (100-1000x)
  • shortest_path uses HDR cascade pruning
  • degree uses DN-Sparse CSR (O(1))
  • vector_query uses CAKES
  • TruthValue + Fingerprint stored as _ladybug_* properties

The correctness test:
  Same Cypher → BoltBackend → Neo4j produces
  same results as → LadybugBackend → ladybug-rs
  for all openCypher TCK queries.
```
