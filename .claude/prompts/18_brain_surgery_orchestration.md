# 18_BRAIN_SURGERY_ORCHESTRATION.md

## A2A Multi-Agent Session: Reconnect the RISC Brain

**Jan Hübener — Ada Architecture — March 2026**
**Execution mode:** Claude Code multi-agent, in-session orchestration
**Repos:** ladybug-rs (write), lance-graph (write), holograph (read-only), rustynum (read-only)

---

## 0. READ THESE FIRST (Mandatory Context)

Every agent in this session MUST read these files before writing any code.
They contain the complete archaeology, the architectural vision, the duplication
map, and the wiring gap analysis. DO NOT skip them. DO NOT summarize from memory.

```
MANDATORY READS (in order):

.claude/prompts/15_RISC_brain_convergence_vision.md
  → The glass-clear vision. 6 RISC instructions. Live plasticity.
  → One vector per SPO node. Zero floats. LanceDB zero-copy.
  → Neo4j as PET scan. Redis as universal mouth.

.claude/prompts/16_open_brain_surgery_handover.md
  → 5 Cypher paths found. Only P2 writes to BindSpace. Nobody calls it.
  → server.rs /cypher is a stub. 66K lines of architecture bypassed.
  → Complete disconnection map with reconnection plan.

.claude/prompts/17_five_path_teardown.md
  → File-by-file verdict for every Cypher and SPO implementation.
  → What to delete, what to keep, what to import.
  → Net effect: -7894 deleted, +1650 written, codebase shrinks.

.claude/prompts/17a_spo_rosetta_stone_addendum.md
  → spo.rs is PRIVATE (mod spo, not pub mod spo). The Rosetta Stone behind a locked door.
  → 5 text→fingerprint encoders, none called from Cypher.
  → 3 TruthValue implementations. P2 execute_match does O(N) scan not O(25) SPO lookup.
  → P4 opcode enum maps 1:1 to spo.rs methods. No executor exists.

REFERENCE READS (as needed per agent role):

.claude/prompts/00_SESSION_A_META.md          → Repo access, module placement
.claude/prompts/00a_PROMOTE_SPO_FIRST_CLASS.md → SPO promotion (DONE — already in src/spo/)
.claude/prompts/04_btree_clam_path_lineage.md  → ClamPath encoding spec
.claude/prompts/04a_merkle_eineindeutigkeit_addon.md → Merkle + ClamPath in word[0]
.claude/prompts/05_nars_causal_trajectory.md   → NARS × BNN instrumentation
.claude/prompts/08_lance_graph_lessons.md      → Error handling, builder patterns
.claude/prompts/09_lance_graph_fork_plan.md    → lance-graph fork actions
.claude/prompts/12_thinking_style_substrate_routing.md → 4-substrate routing
.claude/prompts/13_awareness_loop_architecture.md → 8-layer awareness loop
.claude/PHASE2_SERVER_REWIRE.md               → server.rs rewire spec
```

---

## 1. SITUATION

We're mid-surgery. The brain (ladybug-rs, 164K LOC) has five Cypher paths where
it should have one. The SPO Crystal (spo.rs, 1568 lines) is the architectural
blueprint but is private — nothing can call it. Three SPO implementations exist
across three repos. The server's /cypher endpoint returns a SQL string and calls
it a day.

The goal: collapse five paths to one, unlock spo.rs, wire the Cypher bouncer
(lance-graph) to the SPO execution engine (cypher_bridge.rs + spo.rs),
and make it all work through the Redis protocol.

---

## 2. AGENT ROLES

### Agent SURGEON — Code Deletion + Reorganization

```
RESPONSIBILITY:
  Delete dead code. Move files. Fix imports. Clean the workspace.
  This agent touches MANY files but writes MINIMAL new code.

TASKS (in order):

S1. Delete P1: src/query/cypher.rs (1560 lines)
    - Remove from src/query/mod.rs
    - Remove cypher_to_sql from pub exports
    - Save the recursive CTE generator (lines 1253-1361, ~110 lines)
      → park in src/query/cte_builder.rs for future use
    - Fix any compilation errors from removed imports
    
S2. Delete P3: src/query/lance_parser/ (entire directory, 5532 lines)
    - Remove from src/query/mod.rs
    - Move error.rs From<GraphError> conversion → src/query/error.rs
    - Verify: cargo check after deletion
    
S3. Clean stale PRs
    - Close PRs #11-#33, #54 with "superceded by main"
    - Evaluate #168, #169 against merged #170 — close if covered
    
S4. Fix CI
    - Run: cargo check --all-features
    - Fix compilation errors (likely from feature flag combinations)
    - Target: CI green on main

S5. Rename P4 CypherOp to avoid clash with P2
    - In src/learning/cam_ops.rs: rename CypherOp → CypherInstruction
    - Update all references (grep -rn "CypherOp" src/learning/)
    
GATES:
  S1 complete before any other agent starts (import paths change)
  S2 complete before BRIDGE agent starts
  S4 = EXIT GATE for this agent (CI must be green)
  
FILES TOUCHED: ~15 files modified, 7100+ lines deleted, ~150 lines new
```

### Agent LOCKSMITH — Unlock spo.rs + Port Core Types

```
RESPONSIBILITY:
  Make spo.rs accessible. Port unique concepts to public API.
  Bridge spo.rs Fingerprint → core::Fingerprint.

DEPENDS ON: SURGEON S1 + S2 complete (clean workspace)

TASKS:

L1. Add project_out() to core::Fingerprint
    - Copy Gram-Schmidt projection from spo.rs lines 116-140
    - Adapt: spo.rs Fingerprint uses [u64; N64], core uses [u64; FINGERPRINT_U64]
      → both are 256, so direct port
    - Add test: project_out(a, b) makes result quasi-orthogonal to b
    - ~60 lines in src/core/fingerprint.rs

L2. Create src/spo/crystal_api.rs — public facade over SPOCrystal
    - Import private types from super::spo
    - Expose: insert_triple, query_object, query_subject, query_predicate,
      resonate_spo, encode_triple
    - Use core::Fingerprint (not spo.rs's private one)
    - Use nars::TruthValue (not spo.rs's private one)
    - Map spo.rs Qualia → qualia/meaning_axes.rs AxisActivation
    - ~300 lines

L3. Add OrthogonalCodebook to public API
    - Either make spo.rs OrthogonalCodebook pub
    - Or: recreate as src/spo/codebook.rs using core::Fingerprint with project_out
    - Must maintain: add_orthogonal(), get(), resonate(), cleanup()
    - ~200 lines

L4. Add iterative cleanup() to codebook resonate path
    - The cleanup loop from spo.rs lines 291-310
    - Add as method on whatever codebook type L3 produces
    - ~50 lines

L5. Delete duplicate TruthValue from spo.rs
    - Replace all internal spo.rs uses with nars::TruthValue
    - NOTE: nars uses f32, spo.rs uses f64 — need to decide which
      (f32 is correct for production: matches Container word layout)
    - ~30 lines of changes

GATES:
  L1 must land before L2 (crystal_api needs project_out in core::Fingerprint)
  L2 = EXIT GATE (crystal_api.rs compiles and passes tests)
  
FILES TOUCHED: src/core/fingerprint.rs, src/spo/crystal_api.rs (NEW),
  src/spo/codebook.rs (NEW or modified), src/spo/spo.rs (internal changes)
LINES: ~640 new, ~100 deleted
```

### Agent BRIDGE — Wire P2 (cypher_bridge) to SPO Crystal

```
RESPONSIBILITY:
  Make cypher_bridge.rs execute_match use SPO encoding instead of full scan.
  Connect P2 writes + reads through the SPO Crystal API.

DEPENDS ON: LOCKSMITH L2 complete (crystal_api.rs exists)

TASKS:

B1. Wire execute_match_return to crystal_api
    - Replace: bs.nodes_iter() full scan (lines 378-420)
    - With: crystal.resonate_spo(s, p, o, threshold)
    - The WHERE clause extracts subject/predicate/object text
    - Encode via crystal_api's codebook
    - Return CypherResult with resonance scores
    - ~150 lines changed in cypher_bridge.rs

B2. Wire execute_merge_node through SPO encoding
    - Currently: writes a BindNode with label + properties
    - Should also: encode as SPO triple, insert into crystal, compute Merkle
    - ~80 lines

B3. Wire execute_create_edge through SPO chain insert
    - Currently: creates BindEdge(from, verb, to)
    - Should also: encode S→P→O as causal trajectory in crystal
    - Merkle seal on the new edge
    - ~80 lines

B4. Connect to server.rs /cypher endpoint
    - Replace: handle_cypher calls cypher_to_sql (stub, returns SQL string)
    - With: handle_cypher calls parse_cypher + execute_cypher against BindSpace
    - Return: CypherResult serialized as JSON (matching existing API shape)
    - ~100 lines in server.rs

B5. Add SPOCrystal to server DatabaseState
    - DatabaseState gets a crystal: SPOCrystal field
    - Initialize on startup alongside CogRedis/BindSpace
    - ~30 lines

GATES:
  B4 = EXIT GATE (curl /cypher with MATCH returns SPO-encoded results)
  
TEST: After B4, this must work:
  curl -X POST localhost:5000/cypher \
    -d '{"query": "MERGE (n:Person {name: \"Alice\"})"}' 
  → 200 OK, node created in BindSpace AND crystal
  
  curl -X POST localhost:5000/cypher \
    -d '{"query": "MATCH (a:Person)-[:LOVES]->(b) RETURN b"}'
  → 200 OK, results from SPO Crystal resonance (not full scan)

FILES TOUCHED: src/cypher_bridge.rs, src/bin/server.rs
LINES: ~440 new/changed
```

### Agent BOUNCER — Import lance-graph as Cypher Parser

```
RESPONSIBILITY:
  Add lance-graph as Cargo dependency. Wire its parser as the production
  Cypher parser. Set up the bouncer pattern (parse + validate before execution).

DEPENDS ON: SURGEON S1 + S2 complete (old parsers deleted)

TASKS:

N1. Add lance-graph as Cargo git dependency
    - In Cargo.toml: lance-graph = { git = "https://github.com/AdaWorldAPI/lance-graph", branch = "main" }
    - Resolve workspace compatibility issues
    - May need to feature-gate lance-graph's LanceDB dep to avoid version conflicts
    - ~20 lines in Cargo.toml, potentially more if workspace issues

N2. Create src/query/bouncer.rs — the Cypher validation gate
    - Import lance-graph::parse_cypher_query
    - Import lance-graph::GraphConfig for type namespace validation
    - Build GraphConfig from BindSpace type namespace (0x0100-0x01FF)
    - Validate before execution: reject unknown labels, unknown edge types
    - Route: write ops → cypher_bridge, read ops → DataFusion or crystal_api
    - ~400 lines

N3. Wire bouncer to server.rs
    - Replace B4's direct parse_cypher call with bouncer
    - Bouncer: parse (lance-graph) → validate → route → execute
    - Server sees: /cypher → bouncer → result (single entry point)
    - ~80 lines in server.rs

N4. Delete lance-graph graph/spo/ duplication
    - In lance-graph repo: remove crates/lance-graph/src/graph/spo/
    - Replace with: dependency on ladybug-rs types (or abstract trait)
    - Keep truth.rs — merge into ladybug-rs graph/spo/truth.rs
    - This is a PR to lance-graph, not ladybug-rs
    - ~200 lines deleted from lance-graph, ~175 lines added to ladybug-rs

N5. Wire lance-graph LogicalPlan for future hot↔cold joins
    - Don't build the join yet — just wire LogicalPlanner
    - Parse Cypher → LogicalOperator tree → log/inspect
    - This prepares for the DataFusion join work (future session)
    - ~150 lines

GATES:
  N1 must compile before anything else (Cargo dep resolution)
  N2 = EXIT GATE (bouncer validates and routes correctly)
  N4 is a separate PR to lance-graph repo
  
TEST: After N2:
  curl -X POST localhost:5000/cypher \
    -d '{"query": "MATCH (a)-[:NONEXISTENT]->(b)"}'
  → 400 error: "Unknown relationship type: NONEXISTENT"
  (Bouncer rejects before touching BindSpace)

FILES TOUCHED: Cargo.toml, src/query/bouncer.rs (NEW), src/query/mod.rs,
  src/bin/server.rs, lance-graph Cargo.toml + graph/spo/
LINES: ~650 new in ladybug-rs, ~200 deleted from lance-graph
```

### Agent SEAL — Blake3 Merkle + Wisdom/Staunen Wiring

```
RESPONSIBILITY:
  Wire Blake3 seal verification into the query path.
  Make Wisdom/Staunen observable via DataFusion UDF and Neo4j projection.

DEPENDS ON: BRIDGE B1 complete (queries go through SPO path)

TASKS:

K1. Add wisdom_seal() DataFusion UDF
    - Register in existing cognitive_udfs.rs or new src/query/seal_udf.rs
    - Input: spine_hash (Binary), spine_dirty (u8)
    - Output: Utf8 "WISDOM" | "STAUNEN" | "PENDING"
    - Logic: dirty=true → PENDING, recompute hash → match=WISDOM, mismatch=STAUNEN
    - ~80 lines

K2. Wire seal check into SPO query path
    - After crystal_api.resonate_spo returns results:
      for each hit, check spine_hash against children
    - Boost NARS confidence for WISDOM hits (+ε)
    - Penalize NARS confidence for STAUNEN hits (-ε)
    - This is live plasticity — reads change the substrate
    - ~120 lines in crystal_api.rs or graph/spo/store.rs

K3. Wire Staunen propagation up DN tree
    - When a seal breaks: mark parent spine dirty
    - Parent's next read will also detect break → cascade up
    - Use existing SpineCache dirty mechanism
    - ~100 lines

K4. Create src/neo4j_projection.rs — crystallize to Neo4j
    - On WISDOM crystallization: project node to Neo4j
    - Uses neo4j-rs bolt driver (already in Cargo.toml?)
    - Creates (:SPONode {merkle, label, seal, nars_conf, activation})
    - Creates relationship edges from causal trajectories
    - ~300 lines

K5. Register UDF in DataFusion session
    - Wire into database.rs DataFusion context
    - Test: SQL query with WHERE wisdom_seal(...) = 'STAUNEN'
    - ~30 lines

GATES:
  K1 + K2 = EXIT GATE (queries return results tagged with seal status)
  K4 is optional for this session (can be done in follow-up)

FILES TOUCHED: src/query/seal_udf.rs (NEW), src/query/cognitive_udfs.rs,
  src/neo4j_projection.rs (NEW), crystal_api or graph/spo/store.rs
LINES: ~630 new
```

---

## 3. DEPENDENCY GRAPH

```
        SURGEON
        ┌──────┐
        │S1 S2 │──────────────────────────────────────┐
        │S3 S4 │                                       │
        │  S5  │                                       │
        └──┬───┘                                       │
           │                                           │
     ┌─────▼─────┐                              ┌─────▼─────┐
     │ LOCKSMITH  │                              │  BOUNCER   │
     │ L1→L2→L3  │                              │ N1→N2→N3   │
     │  L4, L5   │                              │  N4, N5    │
     └─────┬─────┘                              └─────┬─────┘
           │                                           │
           └──────────┐                    ┌───────────┘
                      │                    │
                ┌─────▼────────────────────▼──┐
                │         BRIDGE               │
                │ B1→B2→B3→B4→B5               │
                └─────────────┬───────────────┘
                              │
                        ┌─────▼─────┐
                        │   SEAL    │
                        │ K1→K2→K3  │
                        │  K4, K5   │
                        └───────────┘
```

**SURGEON starts immediately.**
**LOCKSMITH and BOUNCER start in parallel after SURGEON S1+S2.**
**BRIDGE starts after LOCKSMITH L2.**
**SEAL starts after BRIDGE B1.**

Critical path: S1 → S2 → L1 → L2 → B1 → B4 → K1 → K2

---

## 4. BLACKBOARD STATE

Agents communicate via this shared state. Update after each completed task.

```yaml
# .claude/SURGERY_BLACKBOARD.md

session_id: "brain-surgery-2026-03"
started: "2026-03-12"

surgeon:
  S1_delete_P1: PENDING    # query/cypher.rs deleted, CTE rescued
  S2_delete_P3: PENDING    # lance_parser/ deleted, error bridge kept
  S3_stale_prs: PENDING    # PRs #11-#33 closed
  S4_ci_green: PENDING     # cargo check --all-features passes
  S5_rename_p4: PENDING    # CypherOp → CypherInstruction

locksmith:
  L1_project_out: PENDING  # Gram-Schmidt in core::Fingerprint
  L2_crystal_api: PENDING  # Public facade over SPOCrystal
  L3_codebook: PENDING     # OrthogonalCodebook public
  L4_cleanup: PENDING      # Iterative cleanup in resonate path
  L5_truthvalue: PENDING   # Delete duplicate, use nars::TruthValue

bridge:
  B1_match_spo: PENDING    # execute_match uses SPO lookup not scan
  B2_merge_spo: PENDING    # MERGE writes to crystal
  B3_edge_spo: PENDING     # CREATE edge uses chain insert
  B4_server_cypher: PENDING # /cypher executes, not transpiles
  B5_crystal_state: PENDING # SPOCrystal in DatabaseState

bouncer:
  N1_cargo_dep: PENDING    # lance-graph as git dependency
  N2_bouncer: PENDING      # Validation gate with type namespace
  N3_server_wire: PENDING  # Bouncer in server.rs /cypher path
  N4_dedup_spo3: PENDING   # Delete lance-graph graph/spo/ duplication
  N5_logical_plan: PENDING # Wire LogicalPlanner for future joins

seal:
  K1_udf: PENDING          # wisdom_seal() DataFusion UDF
  K2_query_seal: PENDING   # Seal check in SPO query path
  K3_propagate: PENDING    # Staunen propagation up DN tree
  K4_neo4j: PENDING        # Neo4j projection (optional this session)
  K5_register: PENDING     # UDF registered in DataFusion

blocking_issues: []
decisions_made: []
```

---

## 5. INVARIANTS (Every Agent Must Respect)

```
1. spo.rs (src/spo/spo.rs) is the REFERENCE SPECIFICATION.
   Do not rewrite it. Port from it. Unlock it. Bridge to it.
   If your code contradicts spo.rs's design, YOUR code is wrong.

2. The Redis protocol stays. CogRedis is the mouth, not a legacy wrapper.
   Any Redis client must be able to talk to ladybug-rs unchanged.

3. BindSpace is the single source of truth for in-memory state.
   SPOCrystal is an indexed VIEW over BindSpace, not a separate store.
   Both must agree at all times.

4. Zero floats in the SPO vector path.
   Fingerprints are [u64; 256]. NARS truth is packed binary.
   Floats belong in the awareness/qualia passthrough, not in SPO.

5. Blake3 Merkle is dual-use: content identity AND integrity seal.
   Don't split these into separate systems.

6. Staunen is a feature, not a bug.
   When a seal breaks, the system is HONESTLY SURPRISED.
   Never suppress Staunen. Never auto-heal broken seals.

7. Neo4j is the PET scan, not the database.
   LanceDB is the database. Neo4j gets PROJECTIONS of crystallized knowledge.
   Never write to Neo4j first — always LanceDB/BindSpace first, project after.

8. One Cypher path. ONE.
   P5 (lance-graph) parses. P2 (cypher_bridge) executes writes.
   crystal_api executes reads. That's it. No other path.

9. cargo check --all-features MUST pass after every task.
   No agent is done until their changes compile.

10. When in doubt, read spo.rs. It knows.
```

---

## 6. SUCCESS CRITERIA (When Surgery Is Complete)

```
□ CI green on ladybug-rs main
□ P1 (query/cypher.rs) deleted
□ P3 (query/lance_parser/) deleted
□ Net code reduction > 5000 lines
□ crystal_api.rs exists and is public
□ core::Fingerprint has project_out()
□ server.rs /cypher executes against BindSpace (not transpile stub)
□ cypher_bridge execute_match uses SPO lookup (not nodes_iter scan)
□ lance-graph imported as Cargo dependency
□ Bouncer validates type namespace before execution
□ wisdom_seal() UDF registered in DataFusion
□ SPO query results tagged with WISDOM/STAUNEN seal status

STRETCH GOALS:
□ Neo4j projection of WISDOM nodes (K4)
□ lance-graph graph/spo/ replaced with ladybug-rs dependency (N4)
□ LogicalPlanner wired for future hot↔cold joins (N5)
□ Open stale PRs closed (S3)
```

---

## 7. EXECUTION PROTOCOL

```
1. Every agent: read mandatory docs (Section 0) FIRST.
2. SURGEON executes S1, S2. Announces completion on blackboard.
3. LOCKSMITH and BOUNCER start in parallel.
4. All agents: after each task, run cargo check. Fix before proceeding.
5. BRIDGE waits for LOCKSMITH L2, then begins.
6. SEAL waits for BRIDGE B1, then begins.
7. Each agent updates blackboard after each task completion.
8. If blocking issue found: add to blackboard, tag which agent is blocked.
9. Session ends when all EXIT GATES pass (S4, L2, B4, N2, K2).
```

---

## 8. SPAWN TRIGGERS (When To Create Sub-Agents)

```
MANDATORY SPAWN:
  - Context window > 60% → spawn continuation agent with blackboard state
  - Compilation failure after 3 attempts → spawn fresh agent with error context
  - Cargo.toml dependency conflict → spawn specialist for workspace resolution

OPTIONAL SPAWN:
  - Need to read >3 files to understand a connection → spawn archaeologist
  - Uncertain about type compatibility → spawn type-checker agent
  - N4 (lance-graph dedup) is a separate repo → spawn dedicated PR agent
```

---

*"Five paths entered. One path leaves. The brain gets smaller and thinks better."*
*"Read spo.rs. It knows."*
