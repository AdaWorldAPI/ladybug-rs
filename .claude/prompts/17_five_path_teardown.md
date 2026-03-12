# 17_FIVE_PATH_TEARDOWN.md

## Brutally Honest Teardown of 5 Cypher + 3 SPO Implementations

**Date:** 2026-03-12  
**Repos:** ladybug-rs (164K LOC) × lance-graph (19K LOC) × holograph (34K LOC)  
**Goal:** One path. No duplication. Every good piece rescued.

---

## PART A: THE FIVE CYPHER PATHS

### P1 — src/query/cypher.rs (1560 lines)

```
WHAT IT IS:   Hand-rolled tokenizer + Cypher→SQL transpiler.
              24 types (CypherQuery, CypherParser, CypherTranspiler, AST nodes).
              Handles: MATCH, CREATE, WHERE, RETURN, ORDER BY, variable-length paths.
              Generates SQL with recursive CTEs for path traversal.

WHO CALLS IT: server.rs:1625 via cypher_to_sql() — returns SQL string to client.
              The /cypher endpoint is literally a transpilation service.
              THE SQL IS NEVER EXECUTED.

WHAT'S GOOD:
  ✓ Recursive CTE generation for variable-length paths (lines 1253-1361)
    This is genuinely useful. Translating (a)-[*1..5]->(b) into
    WITH RECURSIVE traverse AS (...) is non-trivial and correct here.
    ~110 lines of reusable logic.

  ✓ Clean AST types (lines 37-215)
    CypherQuery, MatchClause, Pattern, PatternElement, NodePattern,
    EdgePattern, WhereClause, Condition, Expr, Value.
    Well-structured but DUPLICATED by P3 and P5.

WHAT'S BAD:
  ✗ Hand-rolled tokenizer is fragile (no nom, no proper error recovery)
  ✗ 4 tests. Four. For 1560 lines.
  ✗ No semantic validation (accepts any label/relationship type)
  ✗ Only transpiles — never executes
  ✗ AST name collision: CypherQuery clashes with lance_parser::ast::CypherQuery
    (commented out in P3's mod.rs, line 26)
  ✗ Doesn't handle MERGE, SET, DELETE (only MATCH and CREATE)

VERDICT: DELETE.
  Rescue: recursive CTE generator (110 lines) → move to lance-graph's
  expand_ops.rs or create a cte_builder utility.
  Everything else is done better by P3/P5.
```

### P2 — src/cypher_bridge.rs (897 lines)

```
WHAT IT IS:   Cypher string → parse → CypherOp → execute against &mut BindSpace.
              The ONLY code that actually writes Cypher operations to BindSpace.
              5 types: CypherOp, NodeRef, WhereClause, CypherValue, CypherResult.
              3 public fns: parse_cypher(), execute_cypher(), empty().

WHO CALLS IT: NOBODY from server.rs. Completely disconnected.
              Has 6 tests, all passing. Functional but orphaned.

WHAT'S GOOD:
  ✓ ONLY execution engine that writes to BindSpace (lines 230-520)
    execute_merge_node(), execute_create_node(), execute_create_edge(),
    execute_set_property(), execute_match_return().
    These are THE write primitives. Irreplaceable.

  ✓ Proper MERGE semantics (upsert — create if not exists, update if exists)
    Lines 260-300. Scans by label+name, creates or updates.
    This is what Cypher MERGE actually means. P1 doesn't do this.

  ✓ find_node_by_label_and_name() (line 495)
    BindSpace scan filtered by label and property — the basic graph lookup.

  ✓ CypherResult with columns + rows (line 111)
    Clean result type that maps naturally to Arrow RecordBatch.

WHAT'S BAD:
  ✗ Parser is minimal (lines 139-227) — regex-style splitting, not nom
    Handles MERGE, CREATE, MATCH with WHERE/LIMIT/ORDER BY
    But no variable-length paths, no OPTIONAL MATCH, no WITH, no UNWIND
  ✗ MATCH does full node scan (line 378: bs.nodes_iter())
    No scent index, no Hamming pre-filter, no σ-band cascade
    Works for small BindSpaces, won't scale
  ✗ No error types — returns Result<_, String> everywhere
  ✗ Not integrated with DataFusion — returns CypherResult, not Arrow

VERDICT: KEEP THE EXECUTION ENGINE. REPLACE THE PARSER.
  Keep: execute_cypher() + all execute_* helper functions (lines 230-520)
  Keep: find_node_by_label_and_name(), evaluate_where() (lines 495-610)
  Keep: CypherResult type
  Replace: parse_cypher() → use P5's lance-graph parser instead
  Wire: server.rs /cypher → P5 parser → P2 execution engine for writes
  Extend: add Hamming pre-filter to MATCH (use hdr_cascade instead of nodes_iter)
```

### P3 — src/query/lance_parser/ (5532 lines)

```
WHAT IT IS:   Cypher parser stolen from lance-graph and adapted for ladybug-rs.
              7 files: ast.rs, parser.rs, semantic.rs, config.rs, error.rs,
              case_insensitive.rs, parameter_substitution.rs.
              nom combinator parser (proper, not hand-rolled).
              Semantic validation layer with GraphConfig.

WHO CALLS IT: NOBODY outside tests. Orphaned.
              parse_cypher_query() exported but never imported by server.rs
              or any other module.

EXACT DUPLICATION:
  parser.rs:   12 diff lines vs P5 (only import paths differ)
  ast.rs:      27 diff lines vs P5
  config.rs:   20 diff lines vs P5
  semantic.rs: 126 diff lines vs P5 (MOST DIVERGED — has ladybug-specific additions)

WHAT'S GOOD:
  ✓ The mod.rs regime boundary comment (lines 1-8) is architecturally correct:
    "This module sits OUTSIDE the door. It validates syntax, rejects garbage,
    and hands off clean resolved ASTs. It does NOT know about BindSpace."
    THIS IS THE RIGHT DESIGN. The bouncer SHOULD be blind to substrate.

  ✓ semantic.rs divergence contains ladybug-specific validation (126 lines different)
    Need to diff carefully — some of this might be valuable.

  ✓ parameter_substitution.rs (284 lines)
    Handles $param → value substitution.
    Needed for programmatic Cypher from Redis protocol.

  ✓ GraphConfig + GraphConfigBuilder (config.rs, 465 lines)
    Node/edge type registry with validation.
    Builder pattern with validate-on-build.
    This is the type namespace check the bouncer needs.

WHAT'S BAD:
  ✗ It's a COPY of P5 that has drifted.
    Any fix in one doesn't propagate to the other.
    The 126-line semantic.rs divergence means bugs fixed in one place aren't fixed in the other.
  ✗ No execution backend behind it — parse only.
  ✗ error.rs GraphError is separate from query/error.rs QueryError
    Two error types for the same concept.

VERDICT: DELETE. IMPORT P5 AS CARGO DEPENDENCY.
  Rescue: the 126 lines of semantic.rs divergence → PR to lance-graph
  Rescue: parameter_substitution.rs → if lance-graph doesn't have it, PR it there
  Rescue: mod.rs regime boundary comment → put in lance-graph's documentation
  Rescue: error.rs From<GraphError> for QueryError → keep in query/error.rs
```

### P4 — src/learning/cam_ops.rs CypherOp (in 4775-line file)

```
WHAT IT IS:   Comprehensive opcode enum mapping ALL Cypher operations to numbered
              instruction codes (0x200-0x2FF). Part of a larger OpDictionary system
              that maps 15 enum categories (Lance, SQL, Cypher, Hamming, NARS,
              Filesystem, Crystal, NSM, ACTR, RL, Causal, Qualia, Rung, Meta, Verb,
              Memory, User, Learn) to unified opcodes.

              CypherOp alone: ~80 opcodes across 8 groups
              (Match, Create, Update, Traversal, Aggregation, Graph Algorithms,
              Similarity, Projections)

WHO CALLS IT: Partially wired to OpDictionary (line 1766).
              Two concrete uses: CypherOp::MatchSimilar (line 1977),
              CypherOp::PageRank (line 2019). Rest of opcodes defined but unused.

WHAT'S GOOD:
  ✓ Comprehensive opcode space — every Cypher operation has a numeric code
    This is genuinely valuable as the dispatch table for the unified instruction set.
    Match patterns, Create/Update, Traversal, Aggregation, Graph algorithms,
    Similarity, Projections — all mapped.

  ✓ Graph algorithm opcodes (0x2A0-0x2BF) are UNIQUE
    PageRank, Betweenness, Closeness, DegreeCentrality,
    CommunityLouvain, CommunityLabelProp, WeaklyConnected,
    StronglyConnected, TriangleCount, LocalClustering.
    NOBODY ELSE HAS THESE. Not P1, P2, P3, or P5.
    These map to holograph's graphblas operations.

  ✓ Similarity opcodes (0x2C0-0x2DF) — Jaccard, Cosine, Euclidean, Overlap, KNN
    These map to the hot path Hamming operations.

  ✓ OpDictionary + OpContext + OpParam (lines 1716-1762)
    Framework for unified operation dispatch with typed parameters.

WHAT'S BAD:
  ✗ 78 out of 80 opcodes have no executor
    They're defined but there's no fn that takes a CypherOp and does anything.
  ✗ Sits in learning/cam_ops.rs — wrong location
    This is instruction set architecture, not learning.
  ✗ Duplicates P2's CypherOp enum but with completely different structure
    P2: CypherOp { MergeNode, CreateNode, CreateEdge, SetProperty, MatchReturn }
    P4: CypherOp { MatchNode=0x200, MatchEdge=0x201, ... PageRank=0x2A0, ... }
    Same name, incompatible types, both in scope.
  ✗ cam_ops.rs is 4775 lines — too large, mixing concerns
    15 different enum categories in one file

VERDICT: KEEP THE OPCODE ENUM. MOVE IT. WIRE IT.
  Keep: CypherOp enum + opcode numbering → move to src/instruction/ or src/opcode/
  Keep: Graph algorithm opcodes (UNIQUE — nobody else has them)
  Keep: OpDictionary framework
  Delete: the duplicate naming → rename to CypherInstruction to avoid clash with P2
  Wire: lance-graph AST node types → CypherInstruction dispatch → executor
```

### P5 — lance-graph fork (19,262 lines total)

```
WHAT IT IS:   Complete Cypher query engine: parser + semantic analysis + logical plan
              + DataFusion physical plan + execution. Forked from lance-graph upstream.
              Extended with SPO graph/spo/ module (from holograph lineage).

FILES UNIQUE TO P5 (not in P1-P4):
  logical_plan.rs     1417 lines  ← THE MOST IMPORTANT FILE
    LogicalOperator enum: Scan, Filter, Project, Join, Expand, Aggregate,
    Sort, Limit, Skip, Distinct, Union, Create, Delete, Merge, Set.
    LogicalPlanner takes CypherQuery AST → LogicalOperator tree.
    This is where Cypher becomes an execution plan.

  query.rs            2375 lines  ← Query builder + executor
    CypherQuery struct with config, parameters, execution strategy.
    CypherQueryBuilder for programmatic query construction.
    ExecutionStrategy enum: DataFusion, Simple, Hybrid.

  datafusion_planner/ 5633 lines  ← THE EXECUTION ENGINE
    builder/
      basic_ops.rs      653 lines  Scan, Filter, Project → DataFusion plans
      join_builder.rs   633 lines  ← THE JOIN WE NEED (hot ⋈ cold)
      expand_ops.rs     717 lines  Variable-length path expansion
      aggregate_ops.rs  135 lines  COUNT, SUM, AVG, etc.
      helpers.rs        232 lines  Column resolution, alias handling
      mod.rs            106 lines
    scan_ops.rs         534 lines  LanceDB table scanning
    expression.rs      1443 lines  Cypher expressions → DataFusion expressions
    join_ops.rs         616 lines  Join type resolution and optimization
    vector_ops.rs       485 lines  Vector similarity in DataFusion
    udf.rs              740 lines  Custom DataFusion UDFs
    analysis.rs         399 lines  Query analysis and optimization hints
    config_helpers.rs   237 lines  GraphConfig → DataFusion schema mapping
    mod.rs              240 lines
    test_fixtures.rs     55 lines

  simple_executor/      724 lines  ← Lightweight executor for simple queries
    path_executor.rs    304 lines  Path traversal without DataFusion overhead
    expr.rs             263 lines  Expression evaluation
    clauses.rs           93 lines  Clause handling
    aliases.rs           44 lines  Alias resolution

  lance_vector_search.rs 554 lines ← Vector search builder (float-based, MAY NEED ADAPTATION)

WHAT'S GOOD:
  ✓ Complete Cypher→DataFusion pipeline — this is what ladybug-rs is missing
  ✓ LogicalPlanner separates "what" from "how" (logical vs physical plan)
  ✓ join_builder.rs — THE piece needed for hot↔cold join on merkle_root
  ✓ expression.rs — translates Cypher WHERE to DataFusion predicates (1443 lines!)
  ✓ scan_ops.rs — knows how to scan LanceDB tables as DataFusion sources
  ✓ simple_executor — lightweight path for queries that don't need full DataFusion
  ✓ Proper error handling with snafu (GraphError with file/line/column)
  ✓ GraphConfig for type namespace validation

WHAT'S BAD:
  ✗ graph/spo/ is duplicated from ladybug-rs and DIVERGED
    store.rs: 313 vs 1188 lines (ladybug-rs has TruthGate, much more)
    merkle.rs: 248 vs 769 lines (ladybug-rs has Epoch, ProofStep, more)
    semiring.rs: 99 vs 260 lines (ladybug-rs has more semiring variants)
    lance-graph's copies are OLDER and LESS DEVELOPED.

  ✗ lance_vector_search.rs uses float vectors (f32)
    This needs adaptation for bitpacked Hamming
    The VectorSearch builder pattern is reusable, the distance metric is not

  ✗ udf.rs defines its own UDFs that partially overlap with
    ladybug-rs cognitive_udfs.rs (1415 lines, 12 UDFs)
    Need to merge, not duplicate

  ✗ Not currently importable as Cargo dependency by ladybug-rs
    Separate repo, separate workspace. Need to add as git dep or workspace member.

VERDICT: IMPORT AS DEPENDENCY. THE EXECUTION ENGINE.
  Import: parser.rs, ast.rs, semantic.rs, config.rs, error.rs → via Cargo dep
  Import: logical_plan.rs → the planner (irreplaceable)
  Import: datafusion_planner/ → the execution engine (irreplaceable)
  Import: simple_executor/ → lightweight path for simple queries
  Import: query.rs → CypherQuery builder

  Delete from lance-graph: graph/spo/ → replace with ladybug-rs dependency
  Adapt: lance_vector_search.rs → Hamming distance instead of float cosine
  Merge: udf.rs → combine with ladybug-rs cognitive_udfs.rs
```

---

## PART B: THE THREE SPO IMPLEMENTATIONS

### SPO-1: ladybug-rs src/spo/ (18 files, ~14K lines)

```
WHAT IT IS:   The primary SPO encoding and processing stack.
              The Hamming core. The hot path. The RISC brain.

KEY FILES:
  spo.rs              1568 lines  SPO encoding (3D bitpacked, XOR binding)
  gestalt.rs          1606 lines  BundlingProposal, TiltReport, PlaneCalibration
  spo_harvest.rs       973 lines  SPO distance computation (cosine replacement at 238x less cost)
  clam_path.rs         985 lines  CLAM tree path encoding + MerkleRoot
  causal_trajectory.rs 831 lines  Causal trajectory hydration (BNN instrumentation)
  codebook_hydration.rs 859 lines Codebook hydration from CLAM clusters
  cognitive_codebook.rs 1175 lines 13 types, codebook management
  sentence_crystal.rs  1280 lines Text → SPO encoding pipeline
  deepnsm_integration.rs 785 lines DeepNSM → prime weights → fingerprint

STATUS: ALIVE. The real thing. Recently active (PR 170, Mar 7).
        All tests pass. Recently hardened with Merkle, TruthGate, Epoch.

UNIQUE VALUE: This is THE implementation. Everything SPO starts here.
```

### SPO-2: ladybug-rs src/graph/spo/ (9 files, ~4K lines)

```
WHAT IT IS:   Graph-level SPO operations on BindSpace.
              SpoStore, SpoMerkle, semiring algebra, sparse containers.
              This is where SPO encoding (from src/spo/) meets graph structure (BindSpace).

KEY FILES:
  store.rs            1188 lines  SpoStore: insert, query_forward/reverse/relation/content
                                  TruthGate (NARS filter, ~2 cycles, from PR 170)
                                  SpoHit, QueryHit, QueryAxis
  merkle.rs            769 lines  SpoMerkle: blake3 integrity, InclusionProof, MerkleEpoch
                                  TrajectoryStep, TrajectoryKind, AuthenticatedResult
  sparse.rs            542 lines  SparseContainer: bitmap-indexed sparse storage
  semiring.rs          260 lines  SPO semiring algebra (XOR-Hamming variant)
  scent.rs             204 lines  NibbleScent: 3-axis scent for fast filtering
  builder.rs           340 lines  SpoStore builder
  tests.rs             349 lines  Test suite

STATUS: ALIVE. Recently hardened (PR 170 added TruthGate, MerkleEpoch).
        The bridge between SPO encoding and BindSpace graph structure.

UNIQUE VALUE:
  TruthGate — NARS truth filter BEFORE distance computation (~2 cycles vs ~50)
  MerkleEpoch — XOR dirty bitset snapshot for change detection
  SparseContainer — bitmap-indexed sparse storage (not in lance-graph)
  NibbleScent — 3-axis scent index (not in lance-graph)
  AuthenticatedResult — Merkle-verified query results
```

### SPO-3: lance-graph graph/spo/ (6 files, ~1K lines)

```
WHAT IT IS:   SPO operations for lance-graph's Cypher execution.
              Smaller, simpler versions of the same concepts as SPO-2.

KEY FILES:
  store.rs            313 lines   SpoStore: insert, query_forward/reverse/relation
                                  walk_chain_forward (path traversal — UNIQUE)
                                  query_forward_gated (truth-gated query — parallel development)
  merkle.rs           248 lines   MerkleRoot, ClamPath, BindNode, VerifyStatus, BindSpace
                                  ⚠ DEFINES ITS OWN BindSpace — name collision with ladybug-rs
  truth.rs            175 lines   TruthValue (f,c), TruthGate, NARS revision
                                  ⚠ PARALLEL to ladybug-rs nars/truth.rs but simpler
  semiring.rs          99 lines   BasicSemiring: XOR/AND/OR/MIN_PLUS/MAX_TIMES
  builder.rs          119 lines   SpoStoreBuilder

STATUS: STALE. Diverged from ladybug-rs. Simpler but less complete.
        Has its own BindSpace type which will collide.

UNIQUE VALUE:
  truth.rs — 175 lines of clean NARS truth for the cold path
    TruthValue::revision() — proper NARS evidence fusion
    TruthGate with 3 variants: MinFrequency, MinConfidence, MinBoth
    CLEANER than ladybug-rs nars/truth.rs (which is more complete but messier)
    
  walk_chain_forward() — walks SPO chains following edges
    Useful for path traversal in Cypher MATCH (a)-[:X*1..5]->(b)
    Not in ladybug-rs graph/spo/store.rs
    
  merkle.rs BindSpace — has verify_tree_integrity() and stamp_all()
    Different from ladybug-rs BindSpace (storage/bind_space.rs)
    The integrity check pattern is useful but the BindSpace name collision is not
```

### BONUS SPO: holograph (relevant subset, ~8K lines)

```
WHAT IT IS:   The BlasGraph origin. RedisGraph-transcoded algebra.
              Full GraphBLAS implementation adapted for Hamming vectors.

KEY FILES:
  graphblas/semiring.rs   535 lines  HdrSemiring: 7 variants
    xor_bundle, bind_first, hamming_min, similarity_max,
    resonance(threshold), boolean, xor_field
    Semiring trait with multiply/add/zero/one/absorbing
    THIS IS THE COMPLETE ALGEBRA. ladybug-rs has a subset.

  graphblas/ops.rs        717 lines  Matrix-level graph operations
    mxm (matrix-matrix multiply), mxv (matrix-vector), vxm
    apply, reduce, transpose, extract, assign
    WITH SEMIRING DISPATCH — different semiring = different computation

  graphblas/sparse.rs     546 lines  CSR/CSC sparse matrix formats
    GraphSparse: compressed sparse row for graph adjacency
    Efficient iteration, transpose, element-wise ops

  dn_sparse.rs           3180 lines  DN tree with sparse containers
    PackedDn: 7×8-bit packed DN address (same concept as ClamPath)
    hierarchical_fingerprint(), xor_bind_fingerprint()
    Complete DN tree traversal with sparse container storage

  hdr_cascade.rs          957 lines  HDR cascade search with Mexican hat
    MexicanHat: excite/inhibit with threshold
    QualityTracker: adaptive threshold based on result quality
    SearchResult with activation levels

  epiphany.rs             840 lines  Epiphany detection from SPO patterns
    classify(), activation(), is_significant()
    CentroidStats: cluster tightness, is_epiphany_cluster()
    AdaptiveThreshold: bayesian threshold adaptation

STATUS: FOUNDATIONAL. This is where the architecture originated.
        Some code has been ported to ladybug-rs, some hasn't.

UNIQUE VALUE:
  Full GraphBLAS algebra (7 semirings vs ladybug-rs's ~3)
  Matrix-level graph ops (mxm, mxv — not in ladybug-rs)
  Epiphany detection (not ported to ladybug-rs)
  AdaptiveThreshold (partially in ladybug-rs search/hdr_cascade.rs)
  PackedDn is more complete than ClamPath in some operations
```

---

## PART C: FILE-BY-FILE VERDICT TABLE

### Cypher Files

```
FILE                               LINES  VERDICT     ACTION                           WHY
─────────────────────────────────────────────────────────────────────────────────────────────────────

P1: query/cypher.rs                1560   DELETE      Rescue CTE generator (110 lines) P5 does everything better
                                                                                        Only 4 tests

P2: cypher_bridge.rs               897    KEEP+WIRE   Connect to server.rs             Only BindSpace executor
                                          KEEP        execute_cypher() (290 lines)     Irreplaceable write path
                                          KEEP        find_node, evaluate_where (115)  Working scan+filter
                                          REPLACE     parse_cypher() (90 lines)        Use P5 parser instead
                                          KEEP        CypherResult type                Clean result type

P3: lance_parser/ast.rs            532    DELETE      Use P5 directly                  12 diff lines from P5
P3: lance_parser/parser.rs         1930   DELETE      Use P5 directly                  12 diff lines from P5
P3: lance_parser/semantic.rs       1719   DELETE+PR   Diff 126 lines → PR to P5        Diverged additions
P3: lance_parser/config.rs         465    DELETE      Use P5 directly                  20 diff lines from P5
P3: lance_parser/case_insensitive  377    DELETE      Use P5 directly                  Identical
P3: lance_parser/param_subst.rs    284    DELETE+PR   PR to P5 if missing              Useful feature
P3: lance_parser/error.rs          195    PARTIAL     Keep From<GraphError> in query/   Error bridge needed
P3: lance_parser/mod.rs            30     DELETE      -                                 -

P4: learning/cam_ops.rs CypherOp   ~400   KEEP+MOVE  Move to src/instruction/          Opcode dispatch table
                                          RENAME     CypherOp → CypherInstruction      Avoid name clash with P2
                                          WIRE       AST → Instruction → Executor      Currently unexecutable

P5: lance-graph parser.rs          1931   IMPORT     Cargo dependency                  The real parser
P5: lance-graph ast.rs             542    IMPORT     Cargo dependency                  Canonical AST
P5: lance-graph semantic.rs        1719   IMPORT     Merge P3 divergence first         Has validation
P5: lance-graph config.rs          465    IMPORT     Cargo dependency                  Type registry
P5: lance-graph error.rs           233    IMPORT     Cargo dependency                  snafu errors
P5: lance-graph logical_plan.rs    1417   IMPORT     Cargo dependency                  THE PLANNER
P5: lance-graph query.rs           2375   IMPORT     Cargo dependency                  Query builder
P5: datafusion_planner/            5633   IMPORT     Cargo dependency                  THE ENGINE
P5: simple_executor/               724    IMPORT     Cargo dependency                  Light path
P5: lance_vector_search.rs         554    ADAPT      Hamming instead of float cosine   Rewrite distance metric
```

### SPO Files

```
FILE                               LINES  VERDICT     ACTION                           WHY
─────────────────────────────────────────────────────────────────────────────────────────────────────

SPO-1: ladybug-rs src/spo/        ~14K   KEEP ALL    The canonical implementation      Recently hardened
  spo.rs                           1568   KEEP        The encoding core                 Irreplaceable
  gestalt.rs                       1606   KEEP        Bundling/tilt/calibration         Unique
  spo_harvest.rs                   973    KEEP        238x cheaper than cosine          Unique
  clam_path.rs                     985    KEEP        CLAM + Merkle in word[0]          PR 170 work
  causal_trajectory.rs             831    KEEP        BNN instrumentation               Unique
  shift_detector.rs                305    KEEP        Stripe migration detection        Unique
  codebook_hydration.rs            859    KEEP        Codebook from CLAM clusters       Unique
  cognitive_codebook.rs            1175   KEEP        Codebook management               Unique
  sentence_crystal.rs              1280   KEEP        Text→SPO pipeline                 Unique
  deepnsm_integration.rs           785    KEEP        DeepNSM→primes→fingerprint        Unique
  meta_resonance.rs                450    KEEP        Meta-level resonance              Unique
  context_crystal.rs               604    KEEP        Context crystallization           Unique
  crystal_lm.rs                    857    KEEP        Crystal language model             Unique
  nsm_substrate.rs                 792    KEEP        NSM substrate bridge              Unique
  jina_api.rs                      226    KEEP        Jina API client                   Shared utility
  jina_cache.rs                    475    KEEP        Jina cache                        Shared utility

SPO-2: ladybug-rs src/graph/spo/  ~4K    KEEP ALL    The graph bridge                  Recently hardened
  store.rs                         1188   KEEP        SpoStore + TruthGate              PR 170
  merkle.rs                        769    KEEP        SpoMerkle + Epoch + Proof         PR 170
  sparse.rs                        542    KEEP        SparseContainer                   Unique
  semiring.rs                      260    KEEP+EXTEND Add holograph semirings           Subset of holograph
  scent.rs                         204    KEEP        NibbleScent                       Unique
  builder.rs                       340    KEEP        SpoStore builder                  Standard
  tests.rs                         349    KEEP        Test suite                        Needed

SPO-3: lance-graph graph/spo/     ~1K    MOSTLY DELETE  Older diverged copies          Replace with ladybug-rs dep
  store.rs                         313    DELETE       ladybug-rs has 1188-line version  3x more complete
  merkle.rs                        248    DELETE       ladybug-rs has 769-line version   ⚠ BindSpace name collision
  semiring.rs                      99     DELETE       ladybug-rs has 260-line version   Subset
  builder.rs                       119    DELETE       ladybug-rs has 340-line version   Less complete
  truth.rs                         175    RESCUE       Copy to ladybug-rs graph/spo/     Clean NARS truth for cold path
                                          OR          Merge into ladybug-rs nars/truth.rs
  mod.rs                           23     DELETE       -                                 -

HOLOGRAPH SPO-RELEVANT:           ~8K    SELECTIVE IMPORT
  graphblas/semiring.rs            535    IMPORT       Full algebra (7 semirings)        ladybug-rs has ~3
  graphblas/ops.rs                 717    IMPORT       mxm, mxv graph ops               Not in ladybug-rs
  graphblas/sparse.rs              546    EVALUATE     May overlap with graph/spo/sparse Compare first
  dn_sparse.rs                     3180   EVALUATE     PackedDn vs ClamPath              May be richer
  hdr_cascade.rs                   957    COMPARE      vs ladybug-rs search/hdr_cascade  Check for missed features
  epiphany.rs                      840    IMPORT       Epiphany detection                Not in ladybug-rs
```

---

## PART D: THE COMBINED PLAN

### What Gets Deleted (net reduction)

```
P1: query/cypher.rs                    -1560  (rescue 110 lines → P5)
P3: query/lance_parser/ (7 files)      -5532  (rescue 126 diff lines → PR to P5)
SPO-3: lance-graph graph/spo/ (5 of 6) -802   (rescue truth.rs 175 lines)
                                       ──────
                                       -7894 lines deleted
                                       +285 lines rescued elsewhere
```

### What Gets Imported (via Cargo dependency)

```
lance-graph as git dependency:
  Parser: parser.rs, ast.rs, semantic.rs, config.rs, error.rs     ~5K lines
  Planner: logical_plan.rs                                         1.4K lines
  Engine: datafusion_planner/                                      5.6K lines
  Builder: query.rs                                                2.4K lines
  Light: simple_executor/                                          0.7K lines
                                                                   ─────────
                                                                   ~15K lines IMPORTED (not copied)
```

### What Gets Written (new code)

```
Wire P2 to server.rs                    ~50 lines   Connect execute_cypher to /cypher
Wire P5 to server.rs                    ~200 lines  lance-graph parser → dispatch
Move P4 CypherOp to src/instruction/    ~100 lines  Rename + reorganize
Adapt lance_vector_search for Hamming   ~200 lines  Float→bitpacked distance
Merge lance-graph truth.rs              ~50 lines   Into graph/spo/ or nars/
Add wisdom_seal UDF                     ~50 lines   DataFusion UDF
metadata_provider.rs                    ~300 lines  Cold LanceDB → DataFusion
join_plan.rs                            ~500 lines  Hot ⋈ cold on merkle_root
Neo4j projection                        ~200 lines  WISDOM → Neo4j
                                        ──────────
                                        ~1650 lines new code
```

### Net Effect

```
Deleted:    -7,894 lines
Rescued:    +285 lines (moved, not new)
New code:   +1,650 lines
Imported:   ~15K lines (via Cargo dep, NOT in ladybug-rs tree)
──────────────────────────────────
Net change in ladybug-rs: -6,244 lines (codebase gets smaller)
Capability gain: Full Cypher execution, hot↔cold joins, Neo4j PET scan
Duplication eliminated: 3 redundant parsers, 1 redundant SPO implementation
```

---

## PART E: EXECUTION DEPENDENCY ORDER

```
STEP  TASK                                      BLOCKS    RISK
─────────────────────────────────────────────────────────────────
 1    Add lance-graph as Cargo git dep           Nothing   MED (workspace compat)
 2    Delete P1 (query/cypher.rs)                Step 1    LOW
 3    Delete P3 (query/lance_parser/)            Step 1    LOW  
 4    PR P3 semantic.rs divergence to lance-graph Step 1    LOW
 5    Delete SPO-3 from lance-graph              Step 1    LOW
      (replace with ladybug-rs Cargo dep)
 6    Wire P2 execute_cypher to server.rs        Step 2,3  LOW
 7    Wire P5 parser to server.rs /cypher        Step 1    MED
 8    Move P4 CypherOp to src/instruction/       Nothing   LOW
 9    Rescue lance-graph truth.rs                Step 5    LOW
10    Build metadata_provider.rs                 Step 7    MED
11    Build join_plan.rs using P5 join_builder   Step 10   HIGH
12    Adapt lance_vector_search for Hamming      Step 1    MED
13    Add wisdom_seal UDF                        Step 7    LOW
14    Build Neo4j projection                     Step 13   MED
15    Import holograph semirings to graph/spo/   Nothing   LOW
16    Import holograph epiphany.rs               Nothing   LOW
```

```
CRITICAL PATH:  1 → 7 → 10 → 11 (the join is the hard part)
PARALLEL:       2,3,4,5 (cleanup, can happen alongside anything)
PARALLEL:       6 (P2 wiring, independent of P5)
PARALLEL:       8,9,13,14,15,16 (can happen alongside the join work)
```

---

*"Five paths entered. One path leaves. The brain gets smaller and thinks better."*
