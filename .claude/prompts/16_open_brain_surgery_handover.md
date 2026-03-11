# 16_OPEN_BRAIN_SURGERY_HANDOVER.md

## Mapping the Disconnection. Reconnecting the Code.

**Jan Hübener — Ada Architecture — March 2026**
**Status:** Mid-surgery. Brain exposed. Reconnection map follows.

---

## 1. THE FIVE CYPHER PATHS (Found During Archaeology)

There are FIVE separate Cypher/query implementations. Not three. Five.

```
PATH   FILE(S)                          LINES   DOES WHAT                        CALLS BINDSPACE?
─────────────────────────────────────────────────────────────────────────────────────────────────────
P1     src/query/cypher.rs              1560    Parse + transpile → SQL string   NO (transpile only)
       └─ CypherParser, CypherTranspiler        Outputs SQL, doesn't execute
       └─ Called by server.rs:1625              Server returns SQL string to client
                                                THE /cypher ENDPOINT IS A STUB.

P2     src/cypher_bridge.rs             897     Parse + EXECUTE → BindSpace      YES (direct mutation)
       └─ parse_cypher() + execute_cypher()     MERGE, CREATE, MATCH, SET
       └─ Operates on &mut BindSpace            The only path that actually WRITES
       └─ NOBODY CALLS IT FROM SERVER.          Completely disconnected.

P3     src/query/lance_parser/          5532    Parse Cypher → AST only          NO (parse only)
       └─ parser.rs (1930), ast.rs (532)        Stolen from lance-graph fork
          semantic.rs (1719), config.rs (465)   Has parameter substitution
       └─ parse_cypher_query()                  lance-graph's parser, adapted
       └─ NOBODY CALLS IT OUTSIDE TESTS.        Orphaned.

P4     learning/cam_ops.rs (CypherOp)   ~300    Opcode enum for Cypher ops       INDIRECT (via DataFusion)
       └─ 0x200-0x2BF opcode range              Maps Cypher ops to SQL
       └─ MatchNode, Traverse, PageRank         Intended for LanceDB execution
       └─ PARTIALLY WIRED to OpDictionary       Has opcodes but missing executor.

P5     lance-graph (EXTERNAL REPO)      ~7000   Full Cypher → DataFusion         YES (via LanceDB tables)
       └─ crates/lance-graph/src/               Complete parser + planner + execution
          ast.rs, parser.rs, datafusion_planner/ Has join_builder, scan_ops, vector_ops
       └─ Also has graph/spo/ (duplicated)      SPO store + merkle + semiring + truth
       └─ THIS IS THE BOUNCER WE NEED.          But it's in a separate repo.
```

### The Intended Architecture (What Was Being Built)

```
                    ┌──────────────────────────────────────┐
                    │ P5: lance-graph (THE BOUNCER)         │
                    │     Cypher parser                     │
                    │     → DataFusion logical plan          │
                    │     → scan_ops (hot LanceDB table)     │
                    │     → join_builder (hot ⋈ cold)        │
                    │     → execution                        │
                    └──────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────┐
                    │ P2: cypher_bridge (BindSpace writer)   │
                    │     parse_cypher → CypherOp            │
                    │     execute_cypher(&mut BindSpace)      │
                    │     Handles MERGE, CREATE, SET          │
                    └──────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────┐
                    │ IMPLEMENTATION 1: The Hamming Core     │
                    │     src/spo/ (50K lines)               │
                    │     src/graph/spo/ (4K lines)          │
                    │     src/storage/bind_space.rs (3K)     │
                    │     Blake3 Merkle, NARS, SpineCache    │
                    └──────────────────────────────────────┘
```

### What Got Disconnected

```
P1 (cypher.rs):        Transpiles but never executes. Server stub returns SQL string.
                       Was supposed to be replaced by P5's planner.

P2 (cypher_bridge):    The ONLY code that writes Cypher to BindSpace.
                       897 lines. 28 tests. Working. NOBODY CALLS IT.
                       server.rs doesn't know it exists.

P3 (lance_parser):     Stolen from P5 (lance-graph) and partially adapted.
                       Parser works. No execution backend behind it.
                       semantic.rs has validation logic that P1 doesn't have.

P4 (cam_ops CypherOp): Opcode enum that was intended to be the unified dispatch.
                       Maps Cypher ops to numbered instructions (0x200+).
                       Has PageRank, BFS, community detection opcodes.
                       Missing: the executor that runs these opcodes.

P5 (lance-graph):      The real parser + DataFusion planner.
                       Has its OWN graph/spo/ which DUPLICATES ladybug-rs graph/spo/.
                       The duplication happened during the fork surgery.
```

---

## 2. THE SPO DUPLICATION MAP

### graph/spo/ exists in TWO repos with DIFFERENT content

```
FILE            LADYBUG-RS              LANCE-GRAPH             STATUS
                (src/graph/spo/)        (crates/.../graph/spo/)
──────────────────────────────────────────────────────────────────────────
builder.rs      340 lines               3823 bytes (~150 lines) DIVERGED
merkle.rs       769 lines               7738 bytes (~300 lines) DIVERGED (ladybug has PR 170 work)
mod.rs          29 lines                898 bytes (~35 lines)   Similar
semiring.rs     260 lines               2588 bytes (~100 lines) DIVERGED
store.rs        1188 lines              10373 bytes (~400 lines) DIVERGED (ladybug has TruthGate)
truth.rs        —                       4995 bytes (~200 lines) LANCE-GRAPH ONLY
scent.rs        204 lines               —                       LADYBUG ONLY
sparse.rs       542 lines               —                       LADYBUG ONLY
tests.rs        349 lines               —                       LADYBUG ONLY
case_insens.    98 lines                —                       LADYBUG ONLY
```

**Key insight:** lance-graph has `truth.rs` (NARS truth for the cold path) that ladybug-rs doesn't have. Ladybug-rs has `sparse.rs`, `scent.rs` (hot path optimizations) that lance-graph doesn't have. They were being developed in parallel and diverged.

### holograph graphblas: The BlasGraph Origin

```
holograph/src/graphblas/    (~100KB total)
  matrix.rs     18K bytes  — sparse matrix ops (CSR/CSC)
  ops.rs        18K bytes  — algebraic graph operations
  semiring.rs   17K bytes  — semiring definitions (min-plus, max-times, etc.)
  sparse.rs     15K bytes  — sparse storage formats
  vector.rs     14K bytes  — vector ops on sparse structure
  types.rs      7K bytes   — GrB types
  descriptor.rs 4K bytes   — operation descriptors
  mod.rs        2K bytes   — module root
```

This was transcoded from RedisGraph's BlasGraph. The SPO semiring in both ladybug-rs AND lance-graph derives from this. But holograph has the FULL GraphBLAS algebra (min-plus, max-times, boolean, etc.) while ladybug-rs/graph/spo/semiring.rs has only the XOR/Hamming subset.

---

## 3. THE COLD PATH WIRING (LanceDB metadata)

### What Exists

```
src/storage/lance.rs           934 lines  — LanceDB connection, schema definition
src/storage/lance_persistence.rs 1025 lines — persist BindSpace → LanceDB
src/storage/lance_v1.rs        823 lines  — LanceDB v1 API wrapper
src/storage/lance_zero_copy/   (directory) — zero-copy read experiments
src/storage/database.rs        398 lines  — async Database with LanceDB + DataFusion

src/query/datafusion.rs        777 lines  — DataFusion session/context setup
src/query/graph_provider.rs    968 lines  — BindSpace edges as DataFusion TableProvider
src/query/fingerprint_table.rs 585 lines  — fingerprints as DataFusion table
src/query/dn_tree_provider.rs  403 lines  — DN tree as DataFusion table
src/query/cognitive_udfs.rs    1415 lines — 12 custom DataFusion UDFs
src/query/scent_scan.rs        950 lines  — scent index DataFusion scan
```

### What's Connected

```
graph_provider.rs → BindSpace ✓ (exposes edges as Arrow table)
fingerprint_table.rs → BindSpace ✓ (exposes fingerprints as Arrow table)
dn_tree_provider.rs → BindSpace ✓ (exposes DN tree as Arrow table)
cognitive_udfs.rs → registered with DataFusion ✓

lance_persistence.rs → BindSpace ✓ (can persist to LanceDB)
database.rs → LanceDB + DataFusion ✓ (can run SQL on Lance tables)
```

### What's Disconnected

```
CRITICAL: server.rs does NOT use database.rs
  server.rs uses CogRedis directly (line 480)
  database.rs has async LanceDB + DataFusion — completely bypassed

CRITICAL: lance_persistence.rs writes TO LanceDB but nobody reads back
  The persist direction works (BindSpace → Lance)
  The hydrate direction (Lance → BindSpace on startup) is partial

CRITICAL: No hot↔cold join path exists in the code
  graph_provider.rs exposes BindSpace as DataFusion table
  lance.rs has the LanceDB table
  NOBODY creates the DataFusion join between them
  This is exactly where lance-graph's join_builder.rs belongs

CRITICAL: cognitive_udfs.rs has 12 UDFs registered but never called from server
  hamming_distance, xor_bind, majority_bundle, scent_match, etc.
  All registered. None reachable from the /cypher or /sql endpoints.
```

---

## 4. THE SERVER.RS SITUATION (3681 lines)

```
LINE    WHAT IT DOES                            STATUS
──────────────────────────────────────────────────────────────────
480     Uses CogRedis directly                  WRONG — should use Substrate
500     CogRedis::new()                         WRONG — should be RedisAdapter
522     Replaces bind_space (Lance hydration)   Partial — no schema validation
699     /cypher endpoint → cypher_to_sql()      STUB — transpiles, doesn't execute
1625    cypher_to_sql() call                    Returns SQL string to client (!!)
1665    /redis → CogRedis.execute_command()     Works but bypasses Substrate
1856+   /sci/* endpoints → CogRedis.bind_space() Direct BindSpace access, OK
3632    Persistence tick                        Works — lance_persistence on timer
3681    Total lines                             Needs rewire, not rewrite
```

### The server.rs Rewire (Updated From PHASE2)

The rewire is NOT "replace CogRedis with RedisAdapter." It's:

1. Keep CogRedis as the Redis protocol parser (it's the mouth)
2. Route CogRedis commands through Substrate (not flat HashMap)
3. Wire /cypher to ACTUALLY EXECUTE via cypher_bridge.rs
4. Wire /sql to ACTUALLY EXECUTE via database.rs (async DataFusion)
5. Add lance-graph's planner for queries that need hot↔cold join

---

## 5. THE RECONNECTION MAP (What Needs To Happen)

### Step 1: Unify the Cypher Paths (Eliminate P1, P3, Keep P2+P5)

```
KEEP:   P2 (cypher_bridge.rs)  — The only code that executes Cypher against BindSpace
        Already works. Already tested. Just needs to be called from server.rs.

KEEP:   P5 (lance-graph)       — The Cypher parser + DataFusion planner
        Import as Cargo dependency. Use its parser (better than P1/P3).
        Use its datafusion_planner/ for hot↔cold joins.

ADAPT:  P4 (cam_ops CypherOp)  — The opcode dispatch enum
        This becomes the instruction set for the bouncer.
        lance-graph's parser produces AST → map to CypherOp → execute.

DELETE: P1 (query/cypher.rs)   — 1560 lines of transpile-only code
        Replaced by lance-graph's parser (P5). No execution = no value.

DELETE: P3 (lance_parser/)     — 5532 lines of orphaned parser
        A partial copy of lance-graph's parser. The real thing is in P5.
        The error.rs From<GraphError> conversion stays (move to query/error.rs).
```

### Step 2: Wire The Bouncer (lance-graph parser → execution dispatch)

```
Inbound query (Redis wire protocol)
    │
    ▼
CogRedis protocol parser (existing, keep)
    │
    ├── Pure Redis commands → Substrate → BindSpace (existing path, works)
    │
    ├── GRAPH.QUERY → lance-graph parser (P5)
    │   │
    │   ├── Parse Cypher → AST
    │   ├── Validate against type namespace (read-only BindSpace borrow)
    │   │
    │   ├── Write operations (MERGE, CREATE, SET)
    │   │   └── → cypher_bridge.rs execute_cypher(&mut BindSpace) (P2)
    │   │
    │   ├── Hot-only reads (Hamming scan, SPO extraction)
    │   │   └── → graph_provider.rs (DataFusion on BindSpace Arrow table)
    │   │
    │   ├── Cold-only reads (label filter, property lookup)
    │   │   └── → LanceDB scan via database.rs
    │   │
    │   └── Hot+Cold reads (need both)
    │       └── → lance-graph's join_builder (DataFusion HashJoinExec)
    │           Left:  graph_provider.rs (BindSpace → Arrow)
    │           Right: LanceDB metadata table (Arrow)
    │           Join:  merkle_root
    │
    ├── SQL → DataFusion directly (database.rs, existing)
    │
    └── NARS → truth revision against BindSpace (src/nars/, existing)
```

### Step 3: Deduplicate graph/spo/ (Merge lance-graph's truth.rs into ladybug-rs)

```
FROM lance-graph:
  truth.rs → copy to ladybug-rs/src/graph/spo/truth.rs
  (NARS truth gates for cold path queries — ladybug-rs only has hot path)

FROM ladybug-rs (KEEP, already correct):
  store.rs — has TruthGate from PR 170, more recent
  merkle.rs — has ClamPath+MerkleRoot from PR 170, more recent
  sparse.rs — hot path optimization, lance-graph doesn't have
  scent.rs — scent index integration, lance-graph doesn't have

DELETE from lance-graph:
  graph/spo/builder.rs, merkle.rs, semiring.rs, store.rs
  → Replace with Cargo dependency on ladybug-rs
  → lance-graph should IMPORT ladybug-rs types, not duplicate them
```

### Step 4: Connect The Cold Path (LanceDB metadata joins)

```
EXISTING (works, just needs wiring):
  lance_persistence.rs   → BindSpace → LanceDB (persist direction)
  graph_provider.rs      → BindSpace → DataFusion Arrow table
  fingerprint_table.rs   → BindSpace fingerprints → DataFusion table
  dn_tree_provider.rs    → DN tree → DataFusion table
  database.rs            → async DataFusion + LanceDB session

MISSING (the gap):
  1. LanceDB metadata table schema definition
     (labels, properties, timestamps — the cold columns)
  2. MetadataTableProvider (cold LanceDB table → DataFusion)
  3. JoinPlan: graph_provider (hot) ⋈ metadata_provider (cold) ON merkle_root
  4. server.rs endpoint that executes JoinPlan

NEW CODE NEEDED:
  src/query/metadata_provider.rs (~300 lines)
    LanceDB metadata table → DataFusion TableProvider
    Schema: merkle_root, label, properties, created_at, source, causal_parent
    
  src/query/join_plan.rs (~500 lines)
    Import lance-graph's join_builder
    Connect: graph_provider (hot) + metadata_provider (cold)
    Execute: DataFusion HashJoinExec on merkle_root
    Return: merged Arrow RecordBatch

  Modify: server.rs /cypher endpoint (~200 lines)
    Replace cypher_to_sql stub with actual execution
    Route through bouncer → dispatch → execution → response
```

### Step 5: Blake3 Seal Wiring (Wisdom/Staunen in Query Path)

```
EXISTING:
  graph/spo/merkle.rs — MerkleRoot computation (blake3 of fingerprint)
  storage/bind_space.rs — SpineCache with dirty flags
  storage/xor_dag.rs — XOR parity for subtree verification

MISSING:
  1. On-read seal check in query path
     graph_provider.rs scans edges but doesn't check spine_hash
     Need: after scan, verify blake3 → tag WISDOM vs STAUNEN

  2. DataFusion UDF for seal status
     cognitive_udfs.rs has 12 UDFs but no wisdom_seal()
     Need: wisdom_seal(spine_hash, spine_dirty) → "WISDOM"|"STAUNEN"|"PENDING"
     ~50 lines

  3. NARS confidence modulation on seal status
     Existing: NARS revision in nars/truth.rs
     Missing: seal_status → confidence adjustment in query path
     ~100 lines

  4. Neo4j projection trigger
     When seal transitions PENDING→WISDOM, project to Neo4j
     When seal transitions anything→STAUNEN, update Neo4j label
     ~200 lines (using neo4j-rs bolt driver)
```

---

## 6. LINE-LEVEL TODO LIST

### DELETE (Reduce Surface Area)

```
src/query/cypher.rs                    1560 lines  DELETE (replaced by lance-graph P5)
src/query/lance_parser/ast.rs           532 lines  DELETE (lance-graph has the real one)
src/query/lance_parser/parser.rs       1930 lines  DELETE (lance-graph has the real one)
src/query/lance_parser/semantic.rs     1719 lines  DELETE (lance-graph has the real one)
src/query/lance_parser/config.rs        465 lines  DELETE (lance-graph has the real one)
src/query/lance_parser/case_insensitive 377 lines  DELETE (lance-graph has the real one)
src/query/lance_parser/param_subst.rs   284 lines  DELETE (lance-graph has the real one)
src/query/lance_parser/mod.rs            30 lines  DELETE
src/query/lance_parser/error.rs         195 lines  KEEP error.rs (move From<GraphError> to query/error.rs)
                                       ─────────
                                       7092 lines  DELETED (net code reduction)
```

### KEEP + CONNECT (Wire Existing Code Into Server)

```
src/cypher_bridge.rs                    897 lines  CONNECT to server.rs /cypher endpoint
  → server.rs handle_cypher() calls execute_cypher(&mut bs) instead of cypher_to_sql()
  → Handles all write operations (MERGE, CREATE, SET)
  → ~50 lines changed in server.rs

src/query/graph_provider.rs             968 lines  CONNECT to join path
  → Currently only used if you manually create EdgeTableProvider
  → Needs to be instantiated in the join plan builder
  → ~30 lines to wire

src/query/cognitive_udfs.rs            1415 lines  CONNECT to DataFusion session
  → UDFs registered but session not used by server
  → Wire database.rs session into server.rs
  → ~50 lines to wire

src/storage/database.rs                 398 lines  CONNECT to server.rs
  → Has async LanceDB + DataFusion — completely bypassed
  → server.rs should use this for SQL/Cypher execution
  → ~100 lines to wire
```

### NEW CODE (Minimal)

```
src/query/metadata_provider.rs          ~300 lines  Cold LanceDB metadata → DataFusion
src/query/join_plan.rs                  ~500 lines  Hot ⋈ Cold join via lance-graph
src/query/wisdom_seal_udf.rs            ~50 lines   DataFusion UDF for seal check
src/neo4j_projection.rs                 ~200 lines  Crystallized nodes → Neo4j
Modify: server.rs /cypher handler       ~200 lines  Wire bouncer + execution

Modify: Cargo.toml                       ~10 lines  Add lance-graph as dependency
Modify: src/query/mod.rs                 ~20 lines  Remove lance_parser, add new modules
Modify: src/query/error.rs               ~30 lines  Move GraphError conversion
                                        ─────────
                                        ~1310 lines NEW CODE
```

### IMPORT FROM LANCE-GRAPH (Cargo Dependency, Not Copy)

```
lance-graph as Cargo workspace member or git dependency:
  - Parser: parse_cypher_query() → CypherQuery AST
  - Planner: datafusion_planner/ → LogicalPlan → PhysicalPlan
  - Join: join_builder.rs → HashJoinExec setup
  - Config: GraphConfig → node/edge type registry
  - Error: GraphError with snafu location tracking

DO NOT COPY into ladybug-rs. Import. Use. Depend.
```

### IMPORT FROM HOLOGRAPH (For PageRank/Community Detection Only)

```
holograph graphblas as optional Cargo dependency:
  - src/graphblas/ops.rs → PageRank, BFS, community detection
  - src/graphblas/semiring.rs → additional semiring algebras
  - src/graphblas/sparse.rs → CSR/CSC sparse formats

These map to P4's CypherOp opcodes (0x2A0-0x2BF graph algorithms).
Only needed when someone runs Cypher graph algorithms.
Feature-gated: #[cfg(feature = "graph_algorithms")]
```

---

## 7. NET EFFECT

```
BEFORE:  5 Cypher paths, 3 disconnected, 7092 lines of duplicate/dead code
AFTER:   1 Cypher path (lance-graph parser → bouncer → dispatch → execute)
         cypher_bridge.rs handles writes
         DataFusion handles reads (hot, cold, or joined)
         ~1310 lines new code
         ~7092 lines deleted
         NET: -5782 lines (the codebase gets SMALLER)
```

```
BEFORE:  server.rs /cypher returns SQL string (stub)
         cypher_bridge.rs exists but nobody calls it
         graph_provider.rs exists but nobody instantiates it
         cognitive_udfs.rs registered but unreachable
         database.rs has DataFusion but server uses CogRedis

AFTER:   server.rs /cypher executes Cypher against BindSpace + LanceDB
         Writes go through cypher_bridge → BindSpace
         Hot reads go through graph_provider → DataFusion
         Cold reads go through metadata_provider → DataFusion
         Joined reads go through join_plan → DataFusion HashJoin
         UDFs (including wisdom_seal) available in all DataFusion paths
         Crystallized WISDOM nodes project to Neo4j for PET scan visualization
```

---

## 8. EXECUTION ORDER

```
PHASE   TASK                                    LOC     RISK    PREREQ
─────────────────────────────────────────────────────────────────────────
A       Delete P1 + P3 (7092 lines)             -7092   LOW     None
        Fix imports in query/error.rs            +30
        Verify cargo check passes

B       Wire cypher_bridge.rs to server.rs       +50    LOW     A
        /cypher calls execute_cypher()
        Test: MERGE + MATCH work via /cypher

C       Wire database.rs to server.rs            +100   MED     A
        /sql executes via DataFusion
        Register cognitive_udfs in session
        Test: SQL queries return results

D       Add Cargo dep on lance-graph              +10   MED     A
        Import parser, planner, join_builder
        Wire lance-graph parser as the bouncer
        Test: Cypher parse → AST → validate

E       Create metadata_provider.rs              +300   MED     C
        LanceDB metadata table schema
        DataFusion TableProvider impl
        Test: SELECT * FROM metadata WHERE label='X'

F       Create join_plan.rs                      +500   HIGH    D+E
        Hot (graph_provider) ⋈ Cold (metadata)
        HashJoinExec on merkle_root
        Test: Cypher query returning hot+cold columns

G       Add wisdom_seal UDF                      +50    LOW     C
        Register in DataFusion session
        Test: WHERE wisdom_seal() = 'STAUNEN'

H       Add Neo4j projection                    +200   MED     G
        Crystallize → project to Neo4j
        Test: WISDOM node visible in Neo4j Browser

I       Copy lance-graph truth.rs                +200   LOW     D
        Merge into ladybug-rs graph/spo/
        Wire NARS truth gates into cold path
```

**Critical path: A → B → C → D → E → F (the join is the hard part)**
**Parallel: G, H, I can happen alongside E/F**

Total: ~1310 new lines, ~7092 deleted, ~200 modified = net -5582 lines.
Timeline: 2-4 weeks depending on lance-graph Cargo dependency complexity.

---

*"The brain was always there. We just disconnected the nerves during surgery. Time to reconnect."*
