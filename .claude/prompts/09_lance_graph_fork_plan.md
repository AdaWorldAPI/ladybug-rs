# Lance-Graph Fork Plan: Surgical Import with Luftschleuse Boundaries

## Four Distinct Actions

### Action 1: LLM Harvesting Engine → crewai-rust OSINT Agent
### Action 2: Keyword Validation → Grammar/NSM/Semantic Kernel Expansion  
### Action 3: Steal the Cypher Parser + DataFusion Error Handling
### Action 4: Fork Refactoring with BindSpace/Zero-Copy Preservation

---

## Action 1: LLM Harvesting Engine → crewai-rust

**What lance-graph has:**
```
python/python/knowledge_graph/
  extractors/
    base.py          (553b)   — ExtractionResult(entities, relationships) base class
    heuristic.py     (1.5KB)  — regex/pattern-based extraction fallback
    llm.py           (8.4KB)  — LLM-based entity/relationship extraction
  llm/
    prompts.py       (11.7KB) — structured prompts for extraction
    qa.py            (12.5KB) — question-answering over extracted graph
    semantic.py      (5.2KB)  — semantic analysis of extracted entities
  extraction.py      (2.4KB)  — orchestrator: text → entities → relationships
  embeddings.py      (3.8KB)  — embedding integration (cosine, not Hamming)
  config.py          (7.2KB)  — GraphConfig with node/edge type registry
  service.py         (6.3KB)  — FastAPI service wrapping the pipeline
  cli/ingest.py      (7.5KB)  — CLI for batch ingestion
```

**Where it belongs:** NOT in ladybug-rs. NOT in rustynum. This is a **crewai-rust** agent that:
1. Takes unstructured text (web pages, documents, feeds)
2. Extracts entities + relationships (LLM or heuristic)
3. Encodes them as SPO fingerprints via rustynum
4. Validates against NARS truth values in the existing graph
5. Inserts into LanceDB with causal structure at ingestion time

**The AI-war use case:** Point this agent at public data sources about AI capabilities, funding, partnerships. It extracts entities (companies, models, benchmarks, people) and relationships (trains, funds, competes-with, surpasses). Each extraction becomes an SPO triple in the Cypher graph. NARS validates: does this new edge agree with existing evidence? If confidence < threshold, flag for human review. If irreducible SPO term is non-zero, flag as emergent (something new that can't be explained by existing pairwise relationships).

**What to take from lance-graph:**
- `ExtractionResult` type (entities + relationships) — good abstraction
- `heuristic.py` patterns — useful as fallback when LLM unavailable
- `config.py` GraphConfig builder pattern — use lance-graph's builder, it's validated

**What NOT to take:**
- `llm.py` / `prompts.py` / `qa.py` — these are OpenAI-specific. crewai-rust should use any LLM backend via traits
- `embeddings.py` — cosine-based. We use Hamming. Replace entirely with rustynum-datafusion UDFs
- `service.py` — FastAPI. crewai-rust has its own server architecture

---

## Action 2: Keyword Validation → Grammar/NSM/Semantic Kernel Expansion

**The problem:** lance-graph's Cypher parser validates edge keywords against a static config:
```rust
// lance-graph: regex-style keyword matching
if !config.has_relationship_type("KNOWS") { return Err(PlanError) }
```

This is a string-equality check. It doesn't know that "KNOWS" ≈ "IS_ACQUAINTED_WITH" ≈ "HAS_MET".

**Three expansion paths (not mutually exclusive):**

### Path A: Grammar-based (ladybug-rs `src/grammar/`)
```
ladybug-rs already has:
  src/grammar/causality.rs  — CausalityFlow with agent/action/patient/reason
  src/grammar/verbs.rs      — 144 verbs at Go-board intersections
  src/grammar/sigma.rs      — Sigma state transition grammar
```
The 144 verbs ARE the edge types. When a Cypher query says `MATCH (a)-[:CAUSES]->(b)`, the grammar module can resolve `CAUSES` to the specific verb fingerprint and search by Hamming distance to the verb plane, not string equality.

### Path B: NSM-based (ladybug-rs `src/spo/deepnsm_integration.rs`)
65 semantic primes from the DeepNSM paper (arXiv:2505.11764). Every verb decomposes into primes: `KNOWS` = `THINK` + `SOMEONE` + `EXIST`. Edge validation becomes: does the query verb's NSM decomposition overlap with any stored verb's NSM decomposition? This is a Hamming distance on the prime fingerprint, not a string match.

### Path C: Semantic kernel loci (rustynum σ₃ codebook)
The 1024 σ₃-distinct centroids include verb-type centroids. Edge validation becomes: which codebook centroid is the query verb nearest to? Return all edges whose verb plane is within σ₁ of that centroid. This gives fuzzy matching with statistical guarantees.

**Implementation:** Replace lance-graph's `GraphConfig::has_relationship_type()` with:
```rust
pub enum EdgeResolution {
    Exact(String),                          // lance-graph style: "KNOWS" == "KNOWS"
    Grammar(VerbFingerprint),               // Path A: 144-verb lookup
    NSM(PrimeDecomposition),                // Path B: 65-prime decomposition
    SemanticKernel(CodebookIndex, f32),     // Path C: nearest σ₃ centroid + distance
}

pub fn resolve_edge_type(
    query_type: &str,
    config: &GraphConfig,
    bs: &BindSpace,
) -> Vec<(String, EdgeResolution, f32)> {
    // Try exact match first (fast path)
    // Fall back to grammar verb matching
    // Fall back to NSM prime decomposition
    // Fall back to semantic kernel loci
    // Return ranked matches with confidence
}
```

---

## Action 3: What to Steal from Lance-Graph (the hard, perfect thing)

### STEAL: Cypher Parser (66KB, production-grade)

```
crates/lance-graph/src/parser.rs       (66KB)  — nom-based Cypher parser
crates/lance-graph/src/ast.rs          (16KB)  — typed AST
crates/lance-graph/src/semantic.rs     (67KB)  — semantic analysis (variable scoping, type checking)
crates/lance-graph/src/logical_plan.rs (56KB)  — LogicalOperator enum
crates/lance-graph/src/parameter_substitution.rs (10KB) — $param handling
```
Total: ~215KB of production-tested Cypher infrastructure.

**Why steal it:** Ladybug's `src/query/cypher.rs` is a hand-rolled regex parser. It handles basic `MATCH (a)-[r]->(b)` but breaks on:
- Variable-length paths (`*1..3`)
- WITH chaining
- COLLECT / COUNT(DISTINCT)
- Nested WHERE clauses
- Parameter substitution
- Case-insensitive identifiers

Lance-graph's parser handles all of these with 280+ tests.

### STEAL: DataFusion Error Handling (2.6KB, but influences everything)

```
crates/lance-graph/src/error.rs        (2.6KB)  — GraphError with snafu::Location
```

**Why steal it:** Single unified error type with location tracking. Every error knows where it came from. Arrow/DataFusion/Lance errors wrapped with `From` impls. This replaces ladybug's fragmented Error/DagError/UnifiedError/QueryError mess.

### STEAL: DataFusion Planner (selective)

```
crates/lance-graph/src/datafusion_planner/
  analysis.rs       (15KB)  — two-phase planning (analysis → build)
  mod.rs            (8.5KB) — planner entry point
  config_helpers.rs (9.2KB) — config-to-DataFusion bridge
  udf.rs            (30KB)  — UDF registration patterns (NOT the UDF implementations)
  test_fixtures.rs  (1.7KB) — shared test helpers
```

**What to take:** The two-phase planning architecture, the UDF registration pattern, the test fixture pattern.

**What NOT to take:** The specific UDF implementations (cosine-based vector search). These get replaced by rustynum-datafusion's Hamming/SPO/NARS UDFs.

### STEAL: Expand Operation Pattern

```
crates/lance-graph/src/datafusion_planner/builder/expand_ops.rs (27KB)
```

Graph traversal as DataFusion joins. `(a)-[:KNOWS]->(b)` becomes `scan(Person) JOIN scan(KNOWS) ON a.id = KNOWS.source_id`. This is the single most important piece for making graph queries composable with DataFusion's optimizer.

### DO NOT STEAL:

```
crates/lance-graph/src/lance_vector_search.rs   (18KB)  — cosine-based, we use Hamming
crates/lance-graph/src/lance_native_planner.rs  (2.6KB) — Lance-specific planner, couples to Lance internals
crates/lance-graph/src/simple_executor/         (27KB)  — simple executor, we use DataFusion
crates/lance-graph-catalog/                     (6.7KB) — directory-based catalog, we use BindSpace
crates/lance-graph-python/                      (53KB)  — Python bindings, we have our own
```

---

## Action 4: Fork Refactoring — The Luftschleuse Boundary

This is the dangerous part. Lance-graph assumes:
- Immutable Arrow RecordBatches
- Lance dataset as storage (copy-on-write versioning)
- Cosine similarity as the distance metric
- No concurrent mutation

Ladybug-rs assumes:
- BindSpace with 8-bit prefix:8-bit address (mutable, O(1) lookup)
- XOR-DAG with ACID micro-transactions
- Hamming distance on binary fingerprints
- Write-through collapse gate owned XOR micro-copies (Luftschleuse)

### What is Luftschleuse?

The airlock between lance-graph's immutable Arrow world and ladybug's mutable BindSpace world. Every write goes through:

```
External query (Cypher/SQL)
    │
    ▼
Lance-graph Cypher parser → AST → Semantic → LogicalPlan
    │
    ▼
[LUFTSCHLEUSE — the airlock]
    │
    ├── READ path: LogicalPlan → DataFusion physical plan → Arrow RecordBatch
    │   (pure lance-graph, no mutation, zero-copy Arrow scan)
    │
    └── WRITE path: LogicalPlan → CollapseGate evaluation
        │
        ├── FLOW → XOR micro-copy into BindSpace
        │   (fingerprint written to prefix:addr, XOR-DAG edge created)
        │   (NARS truth value computed from 8-term factorization)
        │   (lance-graph metadata column updated via Lance append)
        │
        ├── HOLD → Buffer in staging area, don't commit
        │   (evidence insufficient, wait for more comparisons)
        │
        └── BLOCK → Reject write, log contradiction
            (new evidence contradicts existing high-confidence edges)
```

### The Meticulous Import Map

| lance-graph file | Import? | Modifications needed |
|-----------------|---------|---------------------|
| `parser.rs` (66KB) | **YES — verbatim** | None. Pure parser, no storage dependency. |
| `ast.rs` (16KB) | **YES — verbatim** | None. Pure data types. |
| `semantic.rs` (67KB) | **YES — with trait injection** | Replace `GraphSourceCatalog` with `LadybugCatalog` trait that wraps BindSpace. Variable resolution must check BindSpace prefixes, not Lance datasets. |
| `logical_plan.rs` (56KB) | **YES — extend** | Keep all existing operators. ADD: `CausalFactorize`, `NarsAccumulate`, `HaloExtract`, `EmitCausalEdge`, `CollapseGateEval`. |
| `error.rs` (2.6KB) | **YES — extend** | Keep `GraphError`. ADD: `BindSpaceError`, `NarsError`, `CollapseGateError` variants with `From` impls. |
| `parameter_substitution.rs` (10KB) | **YES — verbatim** | None. Pure transformation. |
| `case_insensitive.rs` (12KB) | **YES — verbatim** | None. Utility type. |
| `config.rs` (17KB) | **YES — adapt** | Replace node/edge label registry with BindSpace prefix registry. Node labels map to prefixes (0x00-0xFF). Edge types map to verb fingerprints + grammar verbs. |
| `datafusion_planner/analysis.rs` (15KB) | **YES — extend** | Keep two-phase pattern. Phase 1 must also scan BindSpace adjacency lists, not just Lance datasets. |
| `datafusion_planner/builder/expand_ops.rs` (27KB) | **YES — dual path** | Keep join-based expansion for Lance path. ADD: BindSpace adjacency expansion as alternative physical operator (faster for small hops, no join overhead). Optimizer chooses based on estimated cardinality. |
| `datafusion_planner/builder/basic_ops.rs` (23KB) | **YES — adapt** | Replace `scan` operators to also scan BindSpace prefixes, not just Lance tables. |
| `datafusion_planner/builder/join_builder.rs` (26KB) | **YES — verbatim** | Pure DataFusion join construction. |
| `datafusion_planner/expression.rs` (52KB) | **YES — extend** | Keep all expression handling. ADD: Hamming distance expressions, SPO interaction expressions, NARS truth expressions. |
| `datafusion_planner/udf.rs` (30KB) | **REPLACE** | Delete cosine/vector UDFs. Replace with rustynum-datafusion UDFs (hamming, spo_distance, spo_interaction, nars_revision, sigma_classify). |
| `datafusion_planner/vector_ops.rs` (17KB) | **REPLACE** | Delete entirely. Replaced by rustynum-datafusion Hamming search. |
| `datafusion_planner/scan_ops.rs` (21KB) | **ADAPT** | Keep Lance scan path. ADD: BindSpace scan as alternative TableProvider. |
| `lance_vector_search.rs` (18KB) | **DO NOT IMPORT** | Cosine-based. Incompatible. |
| `lance_native_planner.rs` (2.6KB) | **DO NOT IMPORT** | Couples to Lance internals we don't need. |
| `simple_executor/` (27KB) | **DO NOT IMPORT** | Simple in-memory executor. We use DataFusion. |
| `query.rs` (88KB) | **EVALUATE** | This is the integration glue. Likely needs heavy adaptation. Read carefully before deciding. |
| `config_helpers.rs` (9KB) | **ADAPT** | Bridge between config and DataFusion. Replace Lance table registration with BindSpace + Lance dual registration. |

### What MUST NOT Break

1. **BindSpace O(1) lookup** — prefix:addr remains the hot path. lance-graph's catalog abstraction sits ABOVE BindSpace, never replaces it.

2. **Zero-copy Arrow** — `lance_zero_copy.rs` in ladybug already bridges Arrow ↔ Fingerprint. Lance-graph's RecordBatch output must flow through this bridge, not create copies.

3. **XOR-DAG ACID** — Write-through to XOR-DAG must go through CollapseGate. lance-graph's immutable model does not understand mutable writes. The Luftschleuse handles this.

4. **Collapse Gate ownership** — Every write creates an XOR micro-copy owned by the gate state (FLOW/HOLD/BLOCK). Lance-graph's copy-on-write versioning is at the dataset level; our micro-copies are at the fingerprint level. These are different granularities. The Luftschleuse translates between them.

5. **NARS truth accumulation** — Every comparison that passes through the DataFusion planner must also update NARS truth values. This is a side effect that lance-graph doesn't know about. Inject it at the Luftschleuse.

### Fork Strategy

```
Step 1: Fork lance-format/lance-graph → AdaWorldAPI/lance-graph
Step 2: Create branch `ladybug-integration`
Step 3: Import ONLY the files marked YES/ADAPT above
Step 4: Create `src/luftschleuse.rs` — the airlock module
Step 5: Replace UDFs with rustynum-datafusion
Step 6: Add BindSpace as alternative TableProvider
Step 7: Add CausalOperator extensions to LogicalPlan
Step 8: Integration tests: existing lance-graph tests MUST still pass
Step 9: New tests: Cypher → BindSpace → NARS → XOR-DAG roundtrip
```

### Dependency Direction

```
rustynum-datafusion (UDFs)
    ↑ consumed by
lance-graph-fork (Cypher parser + DataFusion planner)
    ↑ consumed by
ladybug-rs (BindSpace + XOR-DAG + NARS + CollapseGate + Luftschleuse)
    ↑ consumed by
crewai-rust (OSINT agent using extraction pipeline)
```

No circular dependencies. Each layer depends only on layers below it. The Luftschleuse lives in ladybug-rs, not in the lance-graph fork. The fork stays as close to upstream as possible to pull future improvements.
