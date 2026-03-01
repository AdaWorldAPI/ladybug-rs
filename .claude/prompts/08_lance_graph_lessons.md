Now I have comprehensive analysis from all research agents. Let me write up the findings and push them to the repos.

## What Rustynum and Ladybug-rs Can Learn from Lance-Graph

I've done a deep comparative analysis of all three codebases. Here are the key lessons:

---

### 1. Structured Multi-Phase Query Pipeline

**Lance-graph does this well:** It has a clean, well-separated pipeline:
```
Parse (nom) → Semantic Analysis → Logical Plan → Physical Plan → Execute
```

Each phase is its own module with clear inputs/outputs. The `LogicalOperator` enum is serializable, debuggable, and decoupled from execution.

**Rustynum gap:** No equivalent pipeline for its compute dispatch. The tiered dispatch (INT8 → BF16 → FP32 → GPU) is ad-hoc. A formalized `ComputePlan` intermediate representation — analogous to lance-graph's `LogicalOperator` — would make the dispatch inspectable, testable, and cacheable.

**Ladybug-rs gap:** The query layer (`src/query/`) has Cypher, SQL, and hybrid paths, but they're not unified through a common logical plan. The `cypher_to_sql` transpiler goes straight from Cypher text to SQL text, losing the opportunity for graph-aware optimization. Lance-graph's approach of going through a typed `LogicalOperator` tree before hitting DataFusion is cleaner and more extensible.

---

### 2. Error Handling with Location Tracking (snafu)

**Lance-graph does this well:** Every `GraphError` variant captures `snafu::Location` (file, line, column). Errors are structured with domain-specific variants (`ParseError`, `ConfigError`, `PlanError`, `ExecutionError`). External errors (Arrow, DataFusion, Lance) are properly wrapped with `From` impls.

```rust
// lance-graph: structured, located errors
GraphError::PlanError {
    message: "Unable to find mapping for 'KNOWS'".into(),
    location: snafu::Location::new(file!(), line!(), column!()),
}
```

**Rustynum gap:** `NumError` is a manual enum with no `std::error::Error` derive via `thiserror` or `snafu`. No location tracking. Worse, the **public API is panic-heavy** — `assert!()` on dimension mismatches instead of returning `Result`. Lance-graph's approach of structured errors with location tracking should be adopted.

**Ladybug-rs gap:** Uses `thiserror` (good), but has **fragmented error types** — `Error`, `DagError`, `UnifiedError`, `QueryError`, plus raw `String` errors. Conversions mostly go through `.to_string()`, losing structured context. Lance-graph's single unified `GraphError` with `From` impls for every external error is the better model. Also, ladybug-rs has **211 `.unwrap()` calls** — lance-graph uses `?` propagation consistently.

---

### 3. Builder Pattern with Validation

**Lance-graph does this well:** `GraphConfigBuilder` enforces invariants:
- Keys normalized to lowercase automatically
- `build()` calls `validate()` which checks for empty fields, non-normalized keys, duplicates
- The builder is the only recommended construction path

```rust
let config = GraphConfig::builder()
    .with_node_label("Person", "person_id")
    .with_relationship("KNOWS", "src_id", "dst_id")
    .build()?;  // Validates on build
```

**Rustynum gap:** `NumArray` construction doesn't validate. `new()` takes raw data with no shape validation — mismatched shapes cause panics later during operations. A builder that validates shape/stride consistency at construction time would prevent downstream panics.

**Ladybug-rs gap:** `BindSpace`, `CogRedis`, and `Container` are constructed directly without validation builders. The `FINGERPRINT_WORDS` vs `FINGERPRINT_U64` mismatch (156 vs 157) is exactly the kind of bug that a validated builder pattern prevents.

---

### 4. Catalog/Source Abstraction via Traits

**Lance-graph does this well:** The `GraphSourceCatalog` trait is minimal and clean:
```rust
pub trait GraphSourceCatalog: Send + Sync {
    fn node_source(&self, label: &str) -> Option<Arc<dyn TableSource>>;
    fn relationship_source(&self, rel_type: &str) -> Option<Arc<dyn TableSource>>;
}
```
With `InMemoryCatalog` for tests and `DirNamespace` for production. This makes the planner testable without real datasets.

**Rustynum gap:** No equivalent abstraction for compute backends. The BLAS dispatch hard-codes backend selection. A trait like `ComputeBackend` with `InMemoryBackend` (for tests) and `AvxBackend`/`MklBackend` (for production) would improve testability and portability.

**Ladybug-rs gap:** Has competing storage philosophies (BindSpace path vs Lance path vs lance_zero_copy) with no unified trait. Lance-graph's approach of a single `GraphSourceCatalog` trait that all backends implement would unify the three paths.

---

### 5. Test Fixture Organization

**Lance-graph does this well:**
- Dedicated `test_fixtures.rs` module with shared helpers (`person_schema()`, `make_catalog()`, `person_scan()`)
- Integration tests include **ASCII art diagrams** of test graph structures
- Tests cover the full pipeline (parse → plan → execute → assert on RecordBatch)
- 9,954 lines of integration tests across 11 focused test files

**Rustynum gap:** Tests are scattered inline with `#[cfg(test)]` modules. No shared test fixtures. Most tests use `.unwrap()` without testing error paths. The integration test file exists but is modest compared to the codebase size.

**Ladybug-rs gap:** Only 4 integration test files, all focused on mathematical proofs rather than system behavior. **Benchmark suite is empty** (3 placeholder files with no content). Lance-graph's Criterion benchmarks with parameterized sizes (100, 10K, 1M) and throughput metrics are the model to follow.

---

### 6. Two-Phase Planning with Context

**Lance-graph does this well:** The DataFusion planner uses a two-phase approach:
1. **Phase 1 (Analysis):** Walk the logical plan to assign unique relationship instance IDs, collect variable-to-label mappings, identify required datasets
2. **Phase 2 (Building):** Construct the physical plan using the analysis context

This avoids column name conflicts (e.g., two KNOWS relationships become `knows_1`, `knows_2`) and makes the planning deterministic.

**Ladybug-rs gap:** The DataFusion integration in `src/query/datafusion.rs` doesn't have this separation. Custom UDFs are registered but there's no analysis phase to detect conflicts. For Cypher queries that traverse multiple relationships of the same type, this will cause column ambiguity bugs.

---

### 7. Minimal, Targeted Feature Flags in Dependencies

**Lance-graph does this well:** DataFusion is pulled with only 7 specific feature flags:
```toml
datafusion = { version = "50.3", default-features = false, features = [
    "nested_expressions", "regex_expressions", "unicode_expressions",
    "crypto_expressions", "encoding_expressions", "datetime_expressions",
    "string_expressions",
] }
```

**Ladybug-rs gap:** `datafusion = "51"` pulls **all default features**. `tokio = { features = ["full"] }` enables everything. This bloats compile times and binary size. Following lance-graph's approach of `default-features = false` + explicit feature selection would help.

**Rustynum gap:** Less relevant (minimal deps), but `rustynum-rs` pulls in `ndarray` and `nalgebra` as dev-dependencies when they're only used in benchmarks — these should be behind a `bench` feature flag.

---

### 8. Case-Insensitive Identifier Handling

**Lance-graph does this well:** A dedicated `CaseInsensitiveStr` wrapper with custom `Eq`/`Hash`, plus normalized lowercase keys throughout `GraphConfig`, `GraphSourceCatalog`, and the semantic analyzer. This is tested thoroughly (15+ test cases for case variations).

**Ladybug-rs gap:** The CAM operations in `cam_ops.rs` use string-based lookups without case normalization. Query variables in the Cypher bridge are case-sensitive, creating inconsistency with how users expect Cypher to work.

---

### 9. Serializable Intermediate Representations

**Lance-graph does this well:** `LogicalOperator`, `CypherQuery` (AST), and all config types derive `Serialize`/`Deserialize`. This enables:
- Logging query plans for debugging
- Caching compiled plans
- Sending plans across process boundaries

**Rustynum gap:** `NumArray` doesn't implement `Serialize`/`Deserialize`. For ML inference pipelines, being able to serialize computation graphs is valuable.

**Ladybug-rs gap:** `Fingerprint` and `Container` do have serde support, but the query plan types don't. Making the query intermediate representation serializable would enable plan caching.

---

### Summary: Top 5 Actionable Items Per Repo

**Rustynum:**
1. Replace `assert!()`/`panic!()` with `Result<T, NumError>` in public APIs (learn from lance-graph's error discipline)
2. Add a `ComputeBackend` trait abstraction (learn from `GraphSourceCatalog`)
3. Add validated builder patterns for `NumArray` construction
4. Create shared test fixtures and add error-path tests
5. Make `LogicalOperator`-style intermediate repr for compute dispatch

**Ladybug-rs:**
1. Unify error types into a single `LadybugError` enum with `snafu::Location` tracking (eliminate 211 `.unwrap()` calls)
2. Add a logical plan intermediate representation for the query pipeline (learn from lance-graph's Parse → Semantic → Logical → Physical)
3. Implement `StorageBackend` trait to unify BindSpace/Lance/ZeroCopy paths (learn from `GraphSourceCatalog`)
4. Use `default-features = false` on DataFusion and Tokio
5. Fill in the empty benchmark suite using lance-graph's Criterion patterns (parameterized sizes, throughput metrics, HTML reports)