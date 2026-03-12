# 22_SISYPHUS_THEN_FORKS.md

## Phase 1: The Harvest (Sisyphus). Phase 2: The Forks (Versions).

**Jan Hübener — Ada Architecture — March 2026**

---

## PART A: SISYPHUS — Harvest Best, Delete Rest

### What This Is

The ugly work. No new architecture. No new repos. No new ideas.
Read 5 Cypher implementations. Read 3 SPO implementations. Read holograph BlasGraph.
Pick the best version of each function. Delete everything else.
Wire what survives into one path through server.rs.

This is NOT creative work. This is JANITORIAL work.
The prompts (15-19) document WHAT. This prompt documents HOW to actually do it.

### Pre-Requisites

```
MANDATORY READS (don't start without these):
  .claude/prompts/17_five_path_teardown.md       ← File-by-file verdicts
  .claude/prompts/17a_spo_rosetta_stone_addendum.md ← spo.rs analysis
  CLAUDE.md                                       ← All traps documented
```

### The Harvest Table

Every function that exists in multiple versions. Pick ONE. Delete the rest.

```
FUNCTION                  P1        P2         P3         P4         P5         HOLOGRAPH   VERDICT
                          cypher.rs bridge.rs  lance_p/   cam_ops    lance-gr   holograph
──────────────────────────────────────────────────────────────────────────────────────────────────────
CYPHER PARSING
──────────────────────────────────────────────────────────────────────────────────────────────────────
Parse MATCH               hand-roll ✓ regex    ✓ nom      -          ✓ nom      query/      P5 nom parser
                          tokenizer  split                            (=P3)     parser.rs   (BEST: proper combinator)

Parse CREATE              partial   ✓          ✓          -          ✓          partial     P5 nom parser
Parse MERGE               ✗         ✓ ONLY     ✓          -          ✓          ✗           P2 execution + P5 parse
Parse SET                 ✗         ✓ ONLY     ✓          -          ✓          ✗           P2 execution + P5 parse
Parse WHERE               basic     basic      ✓ full     -          ✓ full     basic       P5 (property refs, funcs)
Parse ORDER BY            ✓         ✓          ✓          -          ✓          ✗           P5
Parse LIMIT               ✓         ✓          ✓          -          ✓          ✗           P5
Parse variable-length     ✓ CTE     ✗          ✓          -          ✓          ✗           P5 + P1's CTE generator
Parse OPTIONAL MATCH      ✗         ✗          ✓          -          ✓          ✗           P5
Parse WITH/UNWIND         ✗         ✗          ✓          -          ✓          ✗           P5
Parse vector SIMILAR TO   ✗         ✗          ✓          -          ✓          ✗           P5
Parameter substitution    ✗         ✗          ✓          -          ✓          ✗           P5

CYPHER VALIDATION
──────────────────────────────────────────────────────────────────────────────────────────────────────
Semantic validation       ✗         ✗          ✓ 1719L    -          ✓ 1719L   ✗           P5 semantic.rs
GraphConfig builder       ✗         ✗          ✓ 465L     -          ✓ 465L    ✗           P5 config.rs
Type namespace check      ✗         ✗          ✓          -          ✓         ✗           P5
Error with location       ✗         strings    ✓ snafu    -          ✓ snafu   ✗           P5 error.rs

CYPHER EXECUTION
──────────────────────────────────────────────────────────────────────────────────────────────────────
Execute MERGE (write)     ✗         ✓ ONLY     ✗          -          ✗ (plan)  ✗           P2 (irreplaceable)
Execute CREATE node       ✗         ✓ ONLY     ✗          -          ✗ (plan)  ✗           P2 (irreplaceable)
Execute CREATE edge       ✗         ✓ ONLY     ✗          -          ✗ (plan)  ✗           P2 (irreplaceable)
Execute SET property      ✗         ✓ ONLY     ✗          -          ✗ (plan)  ✗           P2 (irreplaceable)
Execute MATCH (read)      ✗         ✓ O(N)scan ✗          -          ✓ DF      ✗           P2 write + P5 DF read
Find node by label+name   ✗         ✓          ✗          -          ✗         ✗           P2 (keep, improve)
Evaluate WHERE clause     ✗         ✓ basic    ✗          -          ✓ full    ✗           P5 for parse, P2 for eval

LOGICAL PLANNING
──────────────────────────────────────────────────────────────────────────────────────────────────────
LogicalOperator enum      ✗         ✗          ✗          -          ✓ 1417L   ✗           P5 (UNIQUE, irreplaceable)
LogicalPlanner            ✗         ✗          ✗          -          ✓         ✗           P5 (UNIQUE)
Plan optimization         ✗         ✗          ✗          -          ✓ partial ✗           P5 analysis.rs

DATAFUSION EXECUTION
──────────────────────────────────────────────────────────────────────────────────────────────────────
scan_ops.rs               ✗         ✗          ✗          -          ✓ 534L    ✗           P5 (UNIQUE)
expression.rs             ✗         ✗          ✗          -          ✓ 1443L   ✗           P5 (UNIQUE)
join_builder.rs           ✗         ✗          ✗          -          ✓ 633L    ✗           P5 (UNIQUE)
join_ops.rs               ✗         ✗          ✗          -          ✓ 616L    ✗           P5 (UNIQUE)
expand_ops.rs             ✗         ✗          ✗          -          ✓ 717L    ✗           P5 (UNIQUE)
vector_ops.rs             ✗         ✗          ✗          -          ✓ 485L    ✗           P5 (UNIQUE)
udf.rs                    ✗         ✗          ✗          -          ✓ 740L    ✗           P5 (merge w/ cognitive_udfs)
simple_executor           ✗         ✗          ✗          -          ✓ 724L    ✗           P5 (UNIQUE)

CTE GENERATION
──────────────────────────────────────────────────────────────────────────────────────────────────────
Recursive CTE for paths   ✓ 110L    ✗          ✗          -          ✗         ✗           P1 (UNIQUE to P1, rescue)

OPCODE DISPATCH
──────────────────────────────────────────────────────────────────────────────────────────────────────
Graph algo opcodes        ✗         ✗          ✗          ✓ UNIQUE   ✗         ✗           P4 (PageRank etc, UNIQUE)
Similarity opcodes        ✗         ✗          ✗          ✓          ✗         ✗           P4
Full CypherOp enum        ✗         ✗          ✗          ✓ 80 ops   ✗         ✗           P4 (rename, keep)

SPO STORE
──────────────────────────────────────────────────────────────────────────────────────────────────────
SpoStore insert           LB 1188   -          -          -          LG 313    -           ladybug-rs (3x more)
SpoStore query_forward    LB ✓+gate -          -          -          LG ✓+gate -           ladybug-rs (has TruthGate)
walk_chain_forward        ✗         -          -          -          LG ✓      -           lance-graph (UNIQUE)
SpoMerkle                 LB 769    -          -          -          LG 248    -           ladybug-rs (has Epoch etc)
SpoSemiring               LB 260    -          -          -          LG 99     HOLO 535    holograph (7 semirings, BEST)
SparseContainer           LB 542    -          -          -          ✗         -           ladybug-rs (UNIQUE)
NibbleScent               LB 204    -          -          -          ✗         -           ladybug-rs (UNIQUE)
TruthValue/TruthGate      spo 45L   -          -          -          LG 175    -           lance-graph truth.rs (cleanest)
                          nars 200L                                                         + nars/ for inference rules

BITPACKED VECTOR OPS
──────────────────────────────────────────────────────────────────────────────────────────────────────
BitpackedVector           ✗         -          -          -          ✗         HOLO 970    holograph (BEST)
HDR cascade               LB ~800   -          -          -          ✗         HOLO 957    holograph (more complete)
Mexican hat               LB ✓      -          -          -          ✗         HOLO ✓      compare, pick best
Epiphany detection        ✗         -          -          -          ✗         HOLO 840    holograph (UNIQUE)
Resonance patterns        ✗         -          -          -          ✗         HOLO 705    holograph (UNIQUE)
GraphBLAS semiring        ✗         -          -          -          ✗         HOLO 535    holograph (UNIQUE)
GraphBLAS matrix ops      ✗         -          -          -          ✗         HOLO 717    holograph (UNIQUE)
GraphBLAS sparse          ✗         -          -          -          ✗         HOLO 546    holograph (UNIQUE)

SPO CRYSTAL (spo.rs)
──────────────────────────────────────────────────────────────────────────────────────────────────────
SPOCrystal 5³ grid        spo.rs    -          -          -          ✗         ✗           spo.rs (UNIQUE, private)
OrthogonalCodebook        spo.rs    -          -          -          ✗         ✗           spo.rs (UNIQUE, private)
QuorumField               spo.rs    -          -          -          ✗         HOLO(diff)  spo.rs (UNIQUE, private)
CubicDistance + gradient   spo.rs    -          -          -          ✗         ✗           spo.rs (UNIQUE, private)
FieldCloseness            spo.rs    -          -          -          ✗         ✗           spo.rs (UNIQUE, private)
Qualia 4D binding (ROLE_Q) spo.rs   -          -          -          ✗         ✗           spo.rs (UNIQUE, private)
Iterative cleanup()       spo.rs    -          -          -          ✗         ✗           spo.rs (UNIQUE, private)
project_out (Gram-Schmidt) spo.rs   -          -          -          ✗         ✗           spo.rs (UNIQUE, private)
```

### Execution Order (Claude Code Session)

```
STEP 1: DELETIONS (fast, low risk, immediate LOC reduction)
  □ Delete src/query/cypher.rs (1560 lines)
    - Save lines 1253-1361 → src/query/cte_builder.rs (rescue CTE gen)
    - Remove from src/query/mod.rs
    - Remove cypher_to_sql from exports
  □ Delete src/query/lance_parser/ (5532 lines)
    - Save error.rs From<GraphError> → src/query/error.rs
    - Remove from src/query/mod.rs
  □ cargo check --no-default-features --features "simd"
  CHECKPOINT: -7092 lines. Verify compilation.

STEP 2: WIRE P2 TO SERVER (close the /cypher stub)
  □ server.rs handle_cypher: replace cypher_to_sql() with:
    let ops = cypher_bridge::parse_cypher(&query)?;
    let result = cypher_bridge::execute_cypher(&mut bs, &ops)?;
  □ Test: curl /cypher with MERGE → node in BindSpace
  □ Test: curl /cypher with MATCH → results from BindSpace
  CHECKPOINT: /cypher actually executes. First time ever from server.

STEP 3: UNLOCK spo.rs (make key types accessible)
  □ In src/spo/mod.rs: change `mod spo;` → `pub(crate) mod spo;`
  □ Create src/spo/crystal_api.rs (thin public facade):
    - pub fn insert_triple(crystal, s, p, o) → MerkleRoot
    - pub fn query_object(crystal, s, p) → Vec<Hit>
    - pub fn query_subject(crystal, p, o) → Vec<Hit>
    - pub fn resonate_spo(crystal, s?, p?, o?, threshold) → Vec<Hit>
  □ Add project_out() to core::Fingerprint (from spo.rs lines 116-140)
  □ cargo check
  CHECKPOINT: SPO Crystal accessible from outside src/spo/.

STEP 4: WIRE P2 MATCH THROUGH SPO (replace O(N) scan)
  □ In cypher_bridge.rs execute_match_return:
    Replace bs.nodes_iter() with crystal_api.resonate_spo()
  □ Test: MATCH query uses SPO spatial lookup, not full scan
  CHECKPOINT: Cypher reads are O(25) not O(N).

STEP 5: HARVEST lance-graph (P5) BEST PARTS into ladybug-rs
  This is the big one. NOT a Cargo dep. Native rewrite.
  
  □ Read lance-graph logical_plan.rs (1417 lines)
    Rewrite LogicalOperator enum natively in src/query/logical_plan.rs
    Use ladybug-rs types (Container, Fingerprint, CogRecord)
    Adopt snafu error pattern from lance-graph error.rs
    ~600 lines (simplified — we don't need all variants yet)
  
  □ Read lance-graph graph/spo/truth.rs (175 lines)
    Merge clean TruthGate 3-variant pattern into src/graph/spo/truth.rs
    ~175 lines (mostly copy + adapt imports)
  
  □ Read lance-graph graph/spo/store.rs walk_chain_forward()
    Add to ladybug-rs src/graph/spo/store.rs
    ~80 lines
    
  □ Read lance-graph config.rs GraphConfig + builder
    Adapt for ladybug-rs type namespace (0x0100-0x01FF)
    Create src/query/graph_config.rs
    ~300 lines
    
  □ cargo check
  CHECKPOINT: LogicalPlan, TruthGate, GraphConfig native in ladybug-rs.

STEP 6: HARVEST holograph BEST PARTS into ladybug-rs
  □ Copy holograph/src/graphblas/semiring.rs → src/graph/spo/semiring.rs
    Replace existing 260-line version with 535-line 7-semiring version
    Adapt imports
    ~535 lines (replaces 260)
    
  □ Evaluate holograph/src/epiphany.rs
    If not already covered by ladybug-rs search/:
    Copy → src/search/epiphany.rs
    ~840 lines
    
  □ Evaluate holograph/src/graphblas/ops.rs (mxm, mxv)
    If useful for batch SPO: copy → src/graph/spo/matrix_ops.rs
    ~717 lines
    
  □ cargo check
  CHECKPOINT: 7 semirings, epiphany detection, matrix ops available.

STEP 7: RENAME P4 CypherOp
  □ In src/learning/cam_ops.rs: rename CypherOp → CypherInstruction
  □ Update all references
  □ cargo check
  CHECKPOINT: No name clash between P2 CypherOp and P4 CypherInstruction.

STEP 8: CLOSE STALE PRs
  □ Close PRs #11-#33, #54 with "superceded by main"
  □ Evaluate #168, #169 against merged #170
  CHECKPOINT: Open PR count < 5.

TOTAL ESTIMATE: 3-5 Claude Code sessions, ~2 weeks.
```

---

## PART B: THE FIVE INVARIANT REPOS (After Sisyphus)

After the harvest, the code is clean enough to fork into focused repos.
Each repo owns ONE invariant. They compose into one binary via Cargo.

### Repo 1: rustynum — The Muscle
```
Invariant:  Every cognitive op compiles to SIMD on Arrow buffers.
Status:     EXISTS. CI needs fixing.
After Sisy: No change. Fix CI. Stable substrate.
```

### Repo 2: ladybug-rs — The Brain  
```
Invariant:  One vector per node. Zero floats in SPO. Hot path sovereign.
            n8n-core JIT + crewai-core persona are INSEPARABLE from the 10 layers.
Status:     EXISTS. Mid-surgery → clean after Sisyphus.
After Sisy: 7K less code. One Cypher path. spo.rs unlocked. Server works.
```

### Repo 3: lance-graph → The Face
```
Invariant:  Every query language compiles to fingerprint → bucket → SIMD scan.
Status:     EXISTS. Monolith → 8 crates (prompt 21).
After Sisy: Crate separation. Import holograph BlasGraph. Clean public API.
```

### Repo 4: staunen — The Bet
```
Invariant:  No GPU. 6 CPU instructions. L1 cache. That's all.
Status:     CREATED. Stubs only.
After Sisy: Fill stubs from harvested spo.rs + rustynum SIMD.
```

### Repo 5: erntefeld ✦ (NEW — The Memory)

```
Repository:    github.com/AdaWorldAPI/erntefeld
Tagline:       "What the brain remembers after it stops thinking"
```

**Invariant: The cold path is Long-Term Memory. It observes, harvests, persists. It never modifies cognition.**

```
What it is:
  The cold path LTM. neo4j-rs as the durable memory graph.
  OSINT harvesting agents that go out, collect, and bring back SPO triples.
  Chess position analysis as a cognitive benchmark.
  AI-war scenario modeling as strategic intelligence.
  
  The boring Cypher database that happens to be fed by a thinking brain.
  
  While ladybug-rs thinks in nanoseconds (hot, volatile, BindSpace),
  erntefeld remembers in perpetuity (cold, durable, Neo4j).
  
  While staunen computes in L1 cache,
  erntefeld stores on disk with indices and ACID transactions.
  
  While lance-graph serves Cypher to users,
  erntefeld runs Cypher AGAINST THE WORLD — scraping, parsing,
  harvesting intelligence and encoding it as SPO triples.

What it owns:
  - neo4j-rs core: bolt driver, Cypher execution, transaction management
  - LTM projection: receives WISDOM crystallizations from ladybug-rs
  - Staunen hydration: feeds STAUNEN-marked triples back for re-examination
    (one-way: sends SPO triples TO ladybug-rs hot path as NEW INPUT,
     NOT as modifications to existing state — the invariant holds)
  - OSINT harvesting: agents that collect from external sources
  - Chess engine: position → SPO encoding → strategic pattern library
  - AI-war modeling: scenario simulation → causal trajectory SPO

The one-way mirror FROM THE COLD SIDE:
  ladybug-rs → erntefeld:   WISDOM crystallizations (hot→cold projection)
  erntefeld → ladybug-rs:   NEW SPO triples for ingestion (not modification!)
                             "I found this in the wild. Here, learn it."
                             The brain decides what to do with it.
                             erntefeld never tells the brain WHAT to think.
                             It brings ingredients. The brain cooks.

Crate layout:
  erntefeld-core/           neo4j-rs wrapper, LTM graph schema, Cypher templates
  erntefeld-projection/     WISDOM receiver, cold metadata writer, seal mirror
  erntefeld-hydration/      STAUNEN feeder: query Neo4j for context → send to BindSpace
  erntefeld-harvest/        OSINT agent framework
    ├── sources/            Pluggable source adapters (web, API, feed, file)
    ├── extractors/         Entity + relationship extraction (SPO encoding)
    └── validators/         NARS truth assignment on harvested triples
  erntefeld-chess/          Chess position → SPO encoding → pattern library
    ├── position.rs         FEN → bitpacked SPO (pieces as subjects, squares as objects)
    ├── opening.rs          Opening book as causal trajectory graph
    ├── tactics.rs          Tactical patterns as SPO templates
    └── strategy.rs         Strategic concepts as high-rung SPO
  erntefeld-war/            AI-war scenario modeling
    ├── scenario.rs         Scenario → SPO causal graph
    ├── capability.rs       AI capability tracking (entities, benchmarks, funding)
    ├── trajectory.rs       Causal trajectory prediction
    └── intelligence.rs     Intelligence assessment as NARS truth aggregation
  erntefeld-bench/          LTM benchmarks: write throughput, query latency, graph depth
```

**Why "Erntefeld":**
German: *Ernte* (harvest) + *Feld* (field). The harvest field.
Where the brain's crystallized knowledge lands and where fresh
intelligence is gathered before being fed back.

**Why chess + AI-war specifically:**
- Chess: the canonical cognitive benchmark. Every position is an SPO triple
  (piece OCCUPIES square, piece ATTACKS square, piece DEFENDS square).
  Opening theory is a causal trajectory graph. Tactics are pattern recognition.
  Strategy is high-level SPO reasoning. If the architecture can't play decent
  chess from pure SPO encoding, the architecture doesn't work.

- AI-war: the practical application. Track AI companies, models, benchmarks,
  funding, partnerships, capabilities as SPO triples. NARS truth values
  track confidence. Causal trajectories predict who's ahead, who's falling
  behind, what capabilities are emerging. The harvesting agents are OSINT
  collectors that encode the AI landscape as a living knowledge graph.

**The erntefeld invariant in code:**

```rust
/// erntefeld NEVER writes to BindSpace hot path.
/// It sends NEW triples via the Redis protocol (same as any external client).
/// The brain (ladybug-rs) decides what to do with them.
/// 
/// erntefeld CAN read from Neo4j (its own cold store).
/// erntefeld CAN query ladybug-rs via Redis protocol (as a client).
/// erntefeld CANNOT access BindSpace directly.
/// erntefeld CANNOT modify SPO vectors.
/// erntefeld CANNOT change NARS truth values on existing nodes.
///
/// It harvests. It remembers. It suggests.
/// The brain thinks. The brain decides. The brain learns.
pub trait ColdPathInvariant {
    /// Receive crystallized WISDOM from hot path (one-way in)
    fn receive_wisdom(&mut self, projection: WisdomProjection);
    
    /// Send new SPO triples for hot path ingestion (one-way out)
    /// These arrive as NEW DATA, not as modifications.
    fn suggest_triple(&self, triple: HarvestedTriple) -> SuggestionResult;
    
    /// Query the cold store (Neo4j, internal to erntefeld)
    fn query_memory(&self, cypher: &str) -> Vec<ColdRecord>;
    
    // There is no fn modify_hot_path().
    // There is no fn override_nars_truth().
    // There is no fn write_to_bindspace().
}
```

---

## PART C: FORK DEPENDENCY AFTER SISYPHUS

```
AFTER SISYPHUS CLEANUP:

rustynum (The Muscle) ──────────────────────────────────┐
  SIMD substrate, no changes needed                      │
                                                         │
staunen (The Bet) ──────────────────────────────────────┤
  Fill stubs from harvested spo.rs + rustynum            │
  6 RISC instructions as clean crate                     │
                                                         ▼
ladybug-rs (The Brain) ────────────────────────────► lance-graph (The Face)
  Clean after Sisyphus                                Crate separation
  One Cypher path                                     Import BlasGraph
  spo.rs unlocked                                     Clean public API
  Server works                                        The boring version
       │
       │ WISDOM projections (hot→cold)
       │ NEW triples (cold→hot, as ingestion)
       ▼
erntefeld (The Memory)
  neo4j-rs LTM
  OSINT harvesting
  Chess engine
  AI-war modeling
  
COMPILATION ORDER:
  1. rustynum (standalone)
  2. staunen (depends: rustynum optional)
  3. ladybug-rs (depends: rustynum, staunen optional)
  4. lance-graph (depends: ladybug-rs types optional)
  5. erntefeld (depends: neo4j-rs, talks to ladybug-rs via Redis protocol)
```

---

## PART D: UPDATED A2A ORCHESTRATION FOR SISYPHUS PHASE

### Agent JANITOR — Delete + Rescue

```
READS: prompt 17 teardown, CLAUDE.md §3 (five Cypher paths)
DOES:
  Delete P1 (src/query/cypher.rs, 1560 lines)
    Rescue CTE generator → src/query/cte_builder.rs
  Delete P3 (src/query/lance_parser/, 5532 lines)
    Rescue error bridge → src/query/error.rs
  Rename P4 CypherOp → CypherInstruction in cam_ops.rs
  Close stale PRs #11-#33, #54
  cargo check after each deletion
EXIT GATE: -7092 lines deleted, cargo check passes
```

### Agent LOCKSMITH — Unlock spo.rs

```
READS: prompt 17a Rosetta Stone, src/spo/spo.rs (read ENTIRE file)
DOES:
  Change mod spo → pub(crate) mod spo in src/spo/mod.rs
  Create src/spo/crystal_api.rs (public facade, ~300 lines)
  Add project_out() to core::Fingerprint (~60 lines)
  Bridge spo.rs Fingerprint → core::Fingerprint
  cargo check
EXIT GATE: crystal_api compiles, project_out test passes
```

### Agent WIRER — Connect P2 to Server + SPO

```
READS: CLAUDE.md §4 (server uses CogRedis), src/cypher_bridge.rs, src/bin/server.rs
DEPENDS: JANITOR complete (P1/P3 deleted), LOCKSMITH complete (crystal_api exists)
DOES:
  Wire server.rs /cypher → cypher_bridge parse + execute
  Wire execute_match → crystal_api.resonate_spo (replace nodes_iter scan)
  Test: MERGE then MATCH via curl
  cargo check
EXIT GATE: curl /cypher with MATCH returns SPO-encoded results
```

### Agent HARVESTER — Import Best from lance-graph + holograph

```
READS: prompt 17 (lance-graph files), prompt 21 (boring version), holograph src/
DEPENDS: JANITOR complete (clean workspace)
DOES:
  Read + rewrite LogicalOperator natively → src/query/logical_plan.rs (~600 lines)
  Read + merge lance-graph truth.rs → src/graph/spo/truth.rs (~175 lines)
  Read + add walk_chain_forward → src/graph/spo/store.rs (~80 lines)
  Read + adapt GraphConfig → src/query/graph_config.rs (~300 lines)
  Read + replace semiring.rs with holograph 7-semiring version (~535 lines)
  Read + import epiphany.rs → src/search/epiphany.rs (~840 lines)
  Adopt snafu error pattern from lance-graph
  cargo check
EXIT GATE: LogicalPlan, 7 semirings, epiphany, GraphConfig all compile natively
```

### Agent SEALER — Merkle + Wisdom/Staunen

```
READS: prompt 19 (hot/cold invariant), src/graph/spo/merkle.rs
DEPENDS: WIRER complete (queries go through SPO path)
DOES:
  Add wisdom_seal() DataFusion UDF (~80 lines)
  Wire seal check into query path: WISDOM boost, STAUNEN penalize
  Wire Staunen propagation up DN tree via SpineCache dirty
  Register UDF in DataFusion session
  cargo check
EXIT GATE: queries return seal status, NARS confidence changes on access
```

### Dependency Graph

```
JANITOR ──────► LOCKSMITH ──────► WIRER ──────► SEALER
    │                                              
    └─────────► HARVESTER (parallel with LOCKSMITH+WIRER)
```

Critical path: JANITOR → LOCKSMITH → WIRER → SEALER
Parallel: HARVESTER runs alongside LOCKSMITH+WIRER

---

## PART E: POST-SISYPHUS VERSION PROMPTS

After Sisyphus, each repo gets its own A2A orchestration prompt:

```
REPO              PROMPT FILE                              WHAT IT DRIVES
──────────────────────────────────────────────────────────────────────────
staunen           staunen/.claude/prompts/01_fill_stubs.md   Fill 6 instruction modules
                                                              from spo.rs + rustynum SIMD

lance-graph       lance-graph/CRATE_SEPARATION_PLAN.md        8-crate separation
                  + .claude/prompts/01_ast_extract.md         Week-by-week execution

erntefeld         erntefeld/.claude/prompts/01_bootstrap.md   Neo4j schema, WISDOM receiver,
                                                              chess SPO encoding, first harvester

ladybug-rs        .claude/prompts/23_post_sisyphus.md         Awareness loop wiring (prompt 13),
                                                              EMPA integration (prompt 14),
                                                              thinking style routing (prompt 12)
```

Each prompt is self-contained. Each repo has its own CLAUDE.md guardrails.
Sessions don't need to understand the other repos to work on their own.

---

*"First the harvest. Then the forks. In that order."*
*"Sisyphus doesn't get to rest until the rock is at the top."*
