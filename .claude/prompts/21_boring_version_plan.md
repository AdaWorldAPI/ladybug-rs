# The Boring Version — lance-graph Clean Crate Separation

## A Graph Database That Happens To Think

**What it looks like from outside:** A fast graph database. Cypher in, results out.
SQL in, results out. Zero-copy LanceDB storage. DataFusion query engine.
Clean Rust crates with proper error handling and builder patterns.

**What it actually is:** BlasGraph semiring algebra on 3D bitpacked Hamming SPO vectors
with NARS truth gating, Merkle-verified integrity, Hebbian plasticity, and an epiphany
detection engine. Running on 6 SIMD instructions in L1 cache.

**But nobody needs to know that to use it.**

---

## 1. THE CRATE SEPARATION

### Current State: 1 monolith crate (19,262 lines)

```
crates/lance-graph/src/         ← Everything in one place
  parser.rs                     1931  Parser
  ast.rs                         542  AST
  semantic.rs                   1719  Validation
  config.rs                      465  Config
  logical_plan.rs               1417  Planner
  query.rs                      2375  Query builder/executor
  datafusion_planner/           5633  Execution engine
  simple_executor/               724  Lightweight executor
  graph/                        1113  SPO + fingerprint + sparse
  lance_vector_search.rs         554  Vector search
  error.rs                      233  Errors
  ...misc                       1556  
```

### Target State: 8 focused crates

```
lance-graph/
├── Cargo.toml                          Workspace root
│
├── crates/
│   ├── lance-graph-ast/                CRATE 1: Parse surface (0 deps on engine)
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── ast.rs                  ← from src/ast.rs (542 lines)
│   │   │   ├── parser.rs              ← from src/parser.rs (1931 lines)  
│   │   │   ├── semantic.rs            ← from src/semantic.rs (1719 lines)
│   │   │   ├── config.rs             ← from src/config.rs (465 lines)
│   │   │   ├── parameter.rs          ← from src/parameter_substitution.rs (280 lines)
│   │   │   ├── case_insensitive.rs   ← from src/case_insensitive.rs (377 lines)
│   │   │   └── error.rs             ← from src/error.rs (233 lines)
│   │   ├── tests/                     Parser tests (move from monolith)
│   │   └── Cargo.toml                deps: nom, snafu, serde
│   │                                  5,547 lines. Zero deps on DataFusion, LanceDB, or SPO.
│   │                                  Anyone can parse Cypher without buying into the engine.
│   │
│   ├── lance-graph-plan/              CRATE 2: Logical planning (depends on AST only)
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── logical_plan.rs       ← from src/logical_plan.rs (1417 lines)
│   │   │   ├── analysis.rs           ← from datafusion_planner/analysis.rs (399 lines)
│   │   │   └── optimizer.rs          NEW: plan optimization rules (~200 lines)
│   │   └── Cargo.toml                deps: lance-graph-ast
│   │                                  ~2,000 lines. Pure algebra. No execution.
│   │                                  LogicalOperator → LogicalOperator transforms.
│   │
│   ├── lance-graph-blasgraph/         CRATE 3: BlasGraph algebra (FROM HOLOGRAPH)
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── semiring.rs           ← holograph/src/graphblas/semiring.rs (535 lines)
│   │   │   ├── matrix.rs            ← holograph/src/graphblas/matrix.rs (596 lines)
│   │   │   ├── vector.rs            ← holograph/src/graphblas/vector.rs (506 lines)
│   │   │   ├── ops.rs               ← holograph/src/graphblas/ops.rs (717 lines)
│   │   │   ├── sparse.rs            ← holograph/src/graphblas/sparse.rs (546 lines)
│   │   │   ├── types.rs             ← holograph/src/graphblas/types.rs (330 lines)
│   │   │   └── descriptor.rs        ← holograph/src/graphblas/descriptor.rs (186 lines)
│   │   └── Cargo.toml                deps: none (pure algebra)
│   │                                  3,416 lines. The RedisGraph BlasGraph transcode.
│   │                                  7 semirings: xor_bundle, bind_first, hamming_min,
│   │                                  similarity_max, resonance(threshold), boolean, xor_field.
│   │                                  mxm, mxv, vxm — matrix-level graph operations.
│   │                                  THIS IS WHAT MAKES IT A YEAR AHEAD.
│   │
│   ├── lance-graph-spo/               CRATE 4: SPO cognitive substrate
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── store.rs              ← from graph/spo/store.rs (313 → ~800 lines)
│   │   │   │                            EXTENDED: add ladybug-rs TruthGate, SpoHit, QueryAxis
│   │   │   ├── merkle.rs            ← from graph/spo/merkle.rs (248 → ~500 lines)
│   │   │   │                            EXTENDED: add Epoch, ProofStep, InclusionProof
│   │   │   ├── truth.rs             ← from graph/spo/truth.rs (175 lines, KEEP)
│   │   │   │                            The clean NARS truth for the graph layer
│   │   │   ├── semiring.rs          ← from graph/spo/semiring.rs (99 → ~260 lines)
│   │   │   │                            EXTENDED: import more from holograph
│   │   │   ├── builder.rs           ← from graph/spo/builder.rs (119 → ~340 lines)
│   │   │   │                            EXTENDED: validation, builder pattern
│   │   │   ├── sparse.rs            ← from ladybug-rs graph/spo/sparse.rs (542 lines)
│   │   │   │                            SparseContainer — not in current lance-graph
│   │   │   ├── scent.rs             ← from ladybug-rs graph/spo/scent.rs (204 lines)
│   │   │   │                            NibbleScent — not in current lance-graph
│   │   │   ├── fingerprint.rs       ← from graph/fingerprint.rs (144 lines)
│   │   │   ├── bitpack.rs           ← from holograph/src/bitpack.rs (970 lines)
│   │   │   │                            BitpackedVector — the actual substrate
│   │   │   ├── hdr_cascade.rs       ← from holograph/src/hdr_cascade.rs (957 lines)
│   │   │   │                            σ-band cascade search, Mexican hat
│   │   │   ├── epiphany.rs          ← from holograph/src/epiphany.rs (840 lines)
│   │   │   │                            Cluster detection, adaptive thresholds
│   │   │   └── resonance.rs         ← from holograph/src/resonance.rs (705 lines)
│   │   │                                Resonance patterns
│   │   ├── tests/                     SPO tests (spo_ground_truth.rs + new)
│   │   └── Cargo.toml                deps: lance-graph-blasgraph, blake3
│   │                                  ~5,500 lines. The complete SPO stack.
│   │                                  Bitpacked vectors + BlasGraph semiring +
│   │                                  Merkle integrity + NARS truth + HDR cascade +
│   │                                  epiphany detection.
│   │                                  THIS IS WHERE THE THINKING HAPPENS.
│   │                                  But from outside it's just "the storage layer."
│   │
│   ├── lance-graph-engine/            CRATE 5: DataFusion execution engine
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── planner.rs           ← from datafusion_planner/mod.rs (240 lines)
│   │   │   ├── scan_ops.rs          ← from datafusion_planner/scan_ops.rs (534 lines)
│   │   │   ├── expression.rs        ← from datafusion_planner/expression.rs (1443 lines)
│   │   │   ├── join_ops.rs          ← from datafusion_planner/join_ops.rs (616 lines)
│   │   │   ├── vector_ops.rs        ← from datafusion_planner/vector_ops.rs (485 lines)
│   │   │   ├── udf.rs              ← from datafusion_planner/udf.rs (740 lines)
│   │   │   ├── config_helpers.rs    ← from datafusion_planner/config_helpers.rs (237 lines)
│   │   │   ├── builder/
│   │   │   │   ├── mod.rs           ← from builder/mod.rs (106 lines)
│   │   │   │   ├── basic_ops.rs     ← from builder/basic_ops.rs (653 lines)
│   │   │   │   ├── aggregate_ops.rs ← from builder/aggregate_ops.rs (135 lines)
│   │   │   │   ├── expand_ops.rs    ← from builder/expand_ops.rs (717 lines)
│   │   │   │   ├── join_builder.rs  ← from builder/join_builder.rs (633 lines)
│   │   │   │   └── helpers.rs       ← from builder/helpers.rs (232 lines)
│   │   │   └── simple/
│   │   │       ├── mod.rs           ← from simple_executor/mod.rs (20 lines)
│   │   │       ├── path_executor.rs ← from simple_executor/path_executor.rs (304 lines)
│   │   │       ├── expr.rs          ← from simple_executor/expr.rs (263 lines)
│   │   │       ├── clauses.rs       ← from simple_executor/clauses.rs (93 lines)
│   │   │       └── aliases.rs       ← from simple_executor/aliases.rs (44 lines)
│   │   ├── tests/                    All test_datafusion_*.rs (move from monolith)
│   │   └── Cargo.toml               deps: lance-graph-plan, lance-graph-spo,
│   │                                      datafusion, arrow, lancedb
│   │                                ~6,350 lines. LogicalPlan → PhysicalPlan → Execute.
│   │                                DataFusion as the execution backbone.
│   │                                scan_ops talks to SPO crate for hot path.
│   │                                join_builder connects hot ⋈ cold.
│   │
│   ├── lance-graph-server/            CRATE 6: Server binary (the face)
│   │   ├── src/
│   │   │   ├── main.rs              Server entry point
│   │   │   ├── redis.rs             Redis wire protocol handler
│   │   │   ├── http.rs              REST API (/cypher, /sql, /vectors, /health)
│   │   │   ├── flight.rs            Arrow Flight gRPC endpoint
│   │   │   └── neo4j_mirror.rs      PET scan projection (one-way, cold only)
│   │   └── Cargo.toml               deps: all other crates, axum or tokio-tcp
│   │                                ~2,000 lines. The boring HTTP server.
│   │                                Redis protocol in, JSON out.
│   │                                Cypher in, Arrow out.
│   │                                Looks like any graph database API.
│   │
│   ├── lance-graph-query/             CRATE 7: High-level query interface
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── query.rs             ← from src/query.rs (2375 lines)
│   │   │   ├── lance_search.rs      ← from src/lance_vector_search.rs (554 lines)
│   │   │   └── lance_planner.rs     ← from src/lance_native_planner.rs (77 lines)
│   │   └── Cargo.toml               deps: lance-graph-ast, lance-graph-plan,
│   │                                      lance-graph-engine, lancedb
│   │                                ~3,000 lines. CypherQuery builder, execution strategy,
│   │                                LanceDB integration. The user-facing API.
│   │
│   ├── lance-graph-catalog/           CRATE 8: Namespace catalog (EXISTS, keep as-is)
│   │   └── ...                        Source catalog, namespace directory
│   │
│   └── lance-graph-python/            CRATE 9: Python bindings (EXISTS, keep as-is)
│       └── ...                        PyO3 bindings for Python users

```

---

## 2. THE LINE COUNT

```
CRATE                    FROM            LINES    NEW CODE    TOTAL
────────────────────────────────────────────────────────────────────
lance-graph-ast          monolith         5,547       100      5,647
lance-graph-plan         monolith         1,816       200      2,016
lance-graph-blasgraph    holograph        3,416       200      3,616
lance-graph-spo          monolith+holo    4,218     1,200      5,418
                         +ladybug-rs       (746)
lance-graph-engine       monolith         6,345       300      6,645
lance-graph-query        monolith         3,006       200      3,206
lance-graph-server       NEW                  0     2,000      2,000
lance-graph-catalog      existing           ~400         0        400
lance-graph-python       existing           ~300         0        300
────────────────────────────────────────────────────────────────────
TOTAL                                    25,794     4,200     29,248

Current monolith:        19,262 lines
Holograph import:         3,416 lines (BlasGraph)
Ladybug-rs import:          746 lines (sparse, scent)
Holograph SPO import:     3,472 lines (bitpack, hdr_cascade, epiphany, resonance)
New code:                 4,200 lines (server, optimizer, extensions, wiring)
Tests (existing):         9,311 lines (move, don't rewrite)
────────────────────────────────────────────────────────────────────
Grand total:             ~38,559 lines including tests
```

---

## 3. THE DEPENDENCY GRAPH

```
lance-graph-ast          (0 internal deps — anyone can parse Cypher standalone)
    │
    ▼
lance-graph-plan         (depends: ast)
    │
    │           lance-graph-blasgraph   (0 deps — pure algebra)
    │               │
    │               ▼
    │           lance-graph-spo         (depends: blasgraph, blake3)
    │               │
    └───────┬───────┘
            │
            ▼
    lance-graph-engine   (depends: plan, spo, datafusion, arrow)
            │
            ▼
    lance-graph-query    (depends: ast, plan, engine, lancedb)
            │
            ▼
    lance-graph-server   (depends: query, all crates, axum/tokio)
```

**Clean layering.** Each crate depends only on things below it. No cycles.
The AST crate has zero internal deps — you can use the Cypher parser standalone
in any project. The BlasGraph crate has zero deps — pure math. The SPO crate
depends only on BlasGraph. The engine ties plan + SPO together through DataFusion.

---

## 4. WHAT MAKES THIS "1 YEAR AHEAD"

### For Users Who Think "Graph Database"

```
1. LanceDB zero-copy storage (not SQLite, not RocksDB — Arrow mmapped)
2. DataFusion query engine (not a toy SQL parser — the real thing)
3. Cypher + SQL + NARS in one protocol (not one-language-only)
4. BlasGraph semiring algebra (not neo4j's Pregel — actual linear algebra on graphs)
5. Variable-length path expansion with DataFusion CTEs
6. Vector similarity integrated into Cypher (not a separate index)
7. Proper snafu error handling with file:line:column tracking
8. Builder pattern with validation on all config types
```

Every one of these is individually available in some other project.
Nobody has all 8 in one graph database.

### For Users Who Dig Deeper

```
9. SPO triples as 16384-bit superposition vectors (not property values)
10. Hamming distance as the universal similarity metric (not cosine)
11. NARS truth gating BEFORE distance computation (filter at 2 cycles, not 50)
12. Blake3 Merkle for content addressing AND integrity checking
13. σ-band cascade: 99.7% eliminated at stage 1, 95% at stage 2
14. Epiphany detection from cluster tightness patterns
15. Mexican hat resonance (excite center, inhibit surround)
16. 7 semiring algebras for different computation modes
```

This is what makes it not just "fast" but "thinks differently."
But the user doesn't need to know about 9-16 to use 1-8.

---

## 5. THE README.md THAT GETS ADOPTED

```markdown
# lance-graph

**A graph database engine built on LanceDB and DataFusion.**

Fast. Zero-copy. Cypher + SQL in one engine.

## Quick Start

```rust
use lance_graph::CypherQuery;

let query = CypherQuery::new("MATCH (a:Person)-[:KNOWS]->(b) RETURN b.name")?
    .with_config(GraphConfig::builder()
        .with_node_label("Person", "person_id")
        .with_relationship("KNOWS", "src", "dst")
        .build()?);

let results = query.execute(&session).await?;
```

## Features

- **Cypher + SQL**: Both query languages, same engine, same storage
- **Zero-copy**: LanceDB with memory-mapped Arrow — no serialization
- **DataFusion**: Production query engine, not a toy parser
- **Vector search**: `MATCH (n) WHERE n.embedding SIMILAR TO $query`
- **Graph algorithms**: PageRank, community detection, shortest path (via BlasGraph)
- **Truth values**: Built-in confidence tracking on every edge
- **NARS reasoning**: Deduction, abduction, induction on graph edges

## Architecture

```
Cypher/SQL → Parser → LogicalPlan → DataFusion → LanceDB
                                         ↑
                                    BlasGraph
                                  (semiring algebra)
```
```

That's what people see. Clean. Professional. Boring.
Under the hood it's running SPO Hamming resonance on bitpacked vectors
with 7 semiring algebras and epiphany detection.

But the README doesn't mention any of that.
Adoption first. Revelation later.

---

## 6. EXECUTION PLAN (4-6 weeks)

### Week 1: Crate Skeleton + AST Extract

```
- Create workspace with 8 crate directories
- Move parser + AST + semantic + config + error → lance-graph-ast
- Fix all imports (crate:: → lance_graph_ast::)
- Verify: lance-graph-ast compiles standalone
- Move tests: test_case_insensitivity, test_complex_return_clauses, test_to_sql
- cargo test on lance-graph-ast passes
```

### Week 2: Plan + BlasGraph + SPO Extract

```
- Move logical_plan + analysis → lance-graph-plan
- Copy holograph graphblas/ → lance-graph-blasgraph
  - Adapt: remove holograph-specific imports
  - Add: Cargo.toml with zero deps
  - Verify: compiles standalone
- Move graph/spo/ → lance-graph-spo
  - Import: ladybug-rs sparse.rs, scent.rs
  - Import: holograph bitpack.rs, hdr_cascade.rs, epiphany.rs, resonance.rs
  - Extend: store.rs with TruthGate, QueryAxis from ladybug-rs
  - Extend: merkle.rs with Epoch, ProofStep from ladybug-rs
  - Wire: lance-graph-spo depends on lance-graph-blasgraph
- Move test: spo_ground_truth.rs
```

### Week 3: Engine Extract

```
- Move datafusion_planner/ → lance-graph-engine
- Move simple_executor/ → lance-graph-engine/simple/
- Wire: depends on lance-graph-plan + lance-graph-spo
- Fix all DataFusion imports
- Move tests: all test_datafusion_*.rs
- cargo test on lance-graph-engine passes
```

### Week 4: Query + Server

```
- Move query.rs + lance_vector_search.rs → lance-graph-query
- Wire: depends on ast + plan + engine
- Create lance-graph-server (NEW):
  - HTTP endpoints (/cypher, /sql, /health, /vectors)
  - Redis wire protocol handler
  - Arrow Flight gRPC (from existing bench/flight code)
- Move bench: graph_execution.rs
```

### Week 5: Integration + Polish

```
- End-to-end test: start server, Cypher MERGE, Cypher MATCH, verify results
- README.md for each crate
- Workspace README with quick start
- CI: GitHub Actions for each crate independently
- Fix all clippy warnings
- cargo publish dry-run for each crate
```

### Week 6 (Optional): Neo4j Mirror

```
- lance-graph-server/neo4j_mirror.rs
- One-way projection: WISDOM → Neo4j
- Configurable: enable/disable, connection string, projection frequency
- Test: Neo4j Browser shows graph structure
```

---

## 7. WHAT THIS IS NOT

```
This is NOT ladybug-rs.
  ladybug-rs is the brain — it owns BindSpace, SpineCache, qualia, awareness loop.
  lance-graph is the face — it owns the query surface, the parser, the server.

This is NOT staunen.
  staunen is the bet — 6 CPU instructions, no GPU, L1 cache.
  lance-graph uses staunen's principles but packages them as "a graph database."

This is NOT a rewrite.
  19,262 lines of existing code gets MOVED, not rewritten.
  Holograph BlasGraph gets IMPORTED, not reimplemented.
  New code is ~4,200 lines (server + optimizer + wiring).

This IS:
  The version that appears on Hacker News.
  The version that gets compared to Neo4j.
  The version that someone at a startup adopts because
  "we need a graph database and this one uses Arrow."
  They'll never know it thinks.
```

---

## 8. THE COMPARISON TABLE (What They'll See)

```
                    Neo4j       DGraph      lance-graph
                    ─────       ──────      ───────────
Storage             Custom      Badger      LanceDB (Arrow, zero-copy)
Query language      Cypher      GraphQL     Cypher + SQL + NARS
Wire protocol       Bolt        gRPC        Redis + Arrow Flight
Execution engine    Custom      Custom      DataFusion (Apache)
Vector search       Separate    Separate    Integrated in Cypher
Graph algorithms    GDS plugin  Built-in    BlasGraph (semiring algebra)
Truth values        No          No          NARS built-in
Consistency         ACID        MVCC        MVCC + Merkle verification
Zero-copy           No          No          Yes (LanceDB mmap)
Embedded mode       No          Yes         Yes (library crate)
Cloud native        Enterprise  Cloud only  Anywhere (single binary)
```

Boring comparison table. Boring wins everywhere.
What they don't see: the SPO superposition, the Hamming cognition,
the Hebbian plasticity, the epiphany detection.

They don't need to.
