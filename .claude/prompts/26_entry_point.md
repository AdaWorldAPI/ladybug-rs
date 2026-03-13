# 26_ENTRY_POINT.md

## The Entry Point. Read This First. Do Everything From Here.

**Repo:** ladybug-rs
**Date:** 2026-03-12
**Status:** This is the orchestration root. All sessions start here.

---

## 0. WHERE YOU ARE

```
ladybug-rs is a 164K LOC Rust cognitive substrate.
It has 5 Cypher implementations where it should have 1.
Its core architectural reference (spo.rs) is private.
Its server endpoint for Cypher is a stub.
Its CI is broken.
It has 13 stale PRs.

But it also has a 10-layer qualia stack, NARS inference,
a working SPO encoding, a working Cypher executor (cypher_bridge.rs),
a working nom Cypher parser (lance_parser/), Blake3 Merkle integrity,
SpineCache borrow/mut, and the complete awareness loop designed in prompts.

The job: wire what exists. Delete what's dead. Refactor what's tangled.
Then implement prompt 25 (Node/Plane/Mask) as the production API.
```

---

## 1. MANDATORY READS (in this order, no skipping)

```
FIRST (understand what you're working with):
  CLAUDE.md                          Guardrails. Traps. Known issues. Read ALL of it.

THEN (understand the target architecture):
  .claude/prompts/25_node_plane_mask.md    THE object model. Three structures.
                                            Node, Plane, Mask. i8 accumulator as
                                            sole ground truth. Seven words.
                                            This is what ladybug-rs BECOMES.

THEN (understand what's broken and what to harvest):
  .claude/prompts/17_five_path_teardown.md     5 Cypher paths. File-by-file verdicts.
  .claude/prompts/17a_spo_rosetta_stone_addendum.md   spo.rs is private. 3 TruthValues.

THEN (understand the invariants):
  .claude/prompts/19_hot_cold_separation_constraint.md   One-way mirror. Never violated.

CONTEXT (read as needed for specific tasks):
  .claude/prompts/15_RISC_brain_convergence_vision.md    6 RISC instructions
  .claude/prompts/22_sisyphus_then_forks.md              Harvest table
  .claude/prompts/23_alpha_channel_risc_modifier.md      Alpha = .α modifier
  .claude/prompts/24_blake3_int8_bundle_encoding.md      BLAKE3 → i8 → (data, alpha)
```

---

## 2. THE CURRENT NAMES (DO NOT RENAME)

```
FILE/MODULE               WHAT IT IS                          KEEP/REFACTOR/DELETE
─────────────────────────────────────────────────────────────────────────────────
src/spo/spo.rs            Reference implementation (PRIVATE)  KEEP as reference
src/spo/mod.rs            SPO module root                     REFACTOR: pub(crate) mod spo
src/spo/gestalt.rs        Bundling/tilt/calibration           KEEP
src/spo/spo_harvest.rs    SPO distance (238x cheaper)         KEEP
src/spo/clam_path.rs      CLAM + Merkle in word[0]           KEEP
src/spo/causal_trajectory.rs  BNN instrumentation            KEEP
src/spo/shift_detector.rs    Stripe migration                KEEP
src/spo/codebook_*.rs     Codebook management                KEEP
src/spo/sentence_crystal.rs  Text→SPO pipeline               KEEP
src/spo/deepnsm_integration.rs  DeepNSM bridge              KEEP
src/spo/jina_api.rs       Jina client                        KEEP (optional)
src/spo/jina_cache.rs     Jina cache                         KEEP (optional)

src/query/lance_parser/   Production Cypher parser (nom)      KEEP — THIS IS THE PARSER
src/query/cypher.rs       Hand-rolled transpiler (P1)         DELETE
src/query/graph_provider.rs   BindSpace → DataFusion table   KEEP
src/query/cognitive_udfs.rs   12 DataFusion UDFs             KEEP
src/query/datafusion.rs   DataFusion session setup            KEEP
src/query/hybrid.rs       Hybrid query                        REFACTOR (imports P1 types)

src/cypher_bridge.rs      Cypher → BindSpace executor (P2)    REFACTOR to take P3 AST
src/bin/server.rs         HTTP server (3681 lines)            REFACTOR /cypher endpoint

src/graph/spo/store.rs    SpoStore + TruthGate                KEEP
src/graph/spo/merkle.rs   SpoMerkle + Epoch + Proof           KEEP
src/graph/spo/sparse.rs   SparseContainer                     KEEP
src/graph/spo/semiring.rs SPO semiring (extend with holograph) KEEP + EXTEND
src/graph/spo/scent.rs    NibbleScent                         KEEP

src/nars/                 NARS inference (7 files)             KEEP — canonical TruthValue
src/qualia/               10-layer qualia stack                KEEP
src/storage/              BindSpace + Substrate + etc          KEEP
src/orchestration/        crewAI integration (12 files)        KEEP
src/learning/cam_ops.rs   Opcode dispatch (CypherOp)          RENAME CypherOp→CypherInstruction
src/cognitive/            7-layer consciousness                KEEP
src/search/               HDR cascade, causal search           KEEP
src/core/fingerprint.rs   16384-bit Fingerprint                KEEP + ADD project_out

src/learning/cam_ops.rs CypherOp   Opcode enum (80 opcodes)   RENAME only, keep all opcodes
```

---

## 3. THE FOUR PHASES

### Phase A: Sisyphus Cleanup (Sessions 1-2)

**Goal:** Delete dead code. Fix the /cypher endpoint. Get things compiling.

```
A1. Delete src/query/cypher.rs (P1, 1560 lines)
    - Save CTE generator (lines 1253-1361) → src/query/cte_builder.rs
    - Remove from query/mod.rs
    - Fix hybrid.rs imports (it imports CypherParser from P1)
    - Comment out server.rs /cypher temporarily

A2. Refactor cypher_bridge.rs to take lance_parser AST directly
    - Remove: CypherOp enum, NodeRef, WhereClause, CypherValue, parse_cypher()
    - Change: execute_cypher(&mut BindSpace, &CypherQuery) using P3 ast types
    - Keep: ALL execution logic (execute_merge_node, execute_create_edge, etc)
    - Keep: find_node_by_label_and_name, evaluate_where
    - Rewrite evaluate_where to take lance_parser::BooleanExpression
    - Rewrite execute_merge to take lance_parser::NodePattern
    - Rewrite execute_match to take lance_parser::MatchClause

A3. Wire server.rs /cypher:
    parse_cypher_query(&query) → execute_cypher(&mut bs, &ast) → JSON response

A4. Rename cam_ops CypherOp → CypherInstruction

A5. cargo check passes
```

### Phase B: Unlock + Implement Node/Plane/Mask (Sessions 3-5)

**Goal:** Implement prompt 25 as the production API in src/spo/.

```
B1. Unlock spo.rs: mod spo → pub(crate) mod spo

B2. Add to core/fingerprint.rs:
    project_out() — Gram-Schmidt (from spo.rs lines 116-140)
    dot_bipolar() — matching bits - non-matching bits

B3. Create src/spo/node.rs — THE Node struct (prompt 25 §1)
    Node { s: Plane, p: Plane, o: Plane }
    Plane { acc: [i8; 16384], encounter_count: u32 }
    Mask { s: bool, p: bool, o: bool }
    8 Mask constants: SPO, SP_, S_O, _PO, S__, _P_, __O, ___

B4. Create src/spo/plane_ops.rs — Plane implementation (prompt 25 §2)
    bits() → derived from sign(acc)
    alpha() → derived from |acc| > threshold
    encounter() → BLAKE3 → i8 ±1 saturating
    reinforce() → BNN from Seal
    truth() → NARS from accumulator state
    distance() → alpha-normalized Hamming with penalty
    density() → alpha density

B5. Create src/spo/mind.rs — Mind implementation (prompt 25 §4)
    Mind { bind_space: &BindSpace, crystal: &SPOCrystal }
    open() → Mind (immutable borrow, zero copy)
    at(s, p, o) → &Node (Hebbian side-effect)
    hold(node, Mask) → HeldNode (owned micro-copy)
    merge(HeldNode) → Changed { seal, diffs, alpha_shift }

B6. Create src/spo/seal.rs — Seal types (prompt 25 §3)
    Seal::Wisdom | Seal::Staunen
    blake3_masked(data AND alpha)
    Address { bits, alpha, merkle }

B7. Create src/spo/gestalt.rs extension — VSA projection (prompt 25 §5)
    GestaltNode { s: ContinuousPlane, p: ContinuousPlane, o: ContinuousPlane }
    project_to_continuous() — i8 acc → f16 for qualia operations
    qualia(Mask) → QualiaState
    This is the awareness passthrough. Not hot path. Computed once per cycle.

B8. Wire into cypher_bridge.rs:
    execute_match uses mind.at() + node.resonate() instead of nodes_iter
    execute_merge uses mind.hold() + encounter + merge
    
B9. Wire into server.rs:
    Mind created from BindSpace on request
    /cypher flows through Mind API
    Response includes Changed { seal } metadata
```

### Phase C: Hardening (Sessions 6-7)

```
C1. Fix CI — cargo check --all-features or closest
C2. Close stale PRs #11-#33, #54
C3. Evaluate #168, #169 against #170
C4. .unwrap() audit on hot paths (storage/, spo/, server.rs)
C5. Race condition P0 fixes from TECHNICAL_DEBT.md
```

### Phase D: Integration (Sessions 8-10)

```
D1. Harvest holograph semiring.rs (7 semirings) → src/graph/spo/semiring.rs
D2. Harvest holograph epiphany.rs → src/search/epiphany.rs
D3. Harvest lance-graph truth.rs → src/graph/spo/truth.rs
D4. Harvest lance-graph walk_chain_forward → src/graph/spo/store.rs
D5. Wire awareness loop (prompt 13) using Mind + Node + gestalt
D6. Wire thinking style routing (prompt 12) using Mask as attention selector
```

---

## 4. AGENT ROLES FOR A2A ORCHESTRATION

### Agent JANITOR [Phase A, sessions 1-2]

```
READS:  CLAUDE.md, prompt 17, src/query/cypher.rs, src/cypher_bridge.rs,
        src/query/lance_parser/ast.rs, src/bin/server.rs
DOES:   Delete P1. Refactor P2 to take P3 AST. Wire server. Rename P4.
EXIT:   cargo check passes. /cypher calls lance_parser → cypher_bridge.
LINES:  -1560 deleted, ~400 refactored in cypher_bridge.rs, ~50 in server.rs
```

### Agent ARCHITECT [Phase B, sessions 3-5]

```
READS:  CLAUDE.md, prompt 25 (ENTIRE THING), src/spo/spo.rs (ENTIRE FILE),
        src/core/fingerprint.rs, src/storage/bind_space.rs
DOES:   Implement Node, Plane, Mask, Mind, Seal, HeldNode, GestaltNode.
        Create node.rs, plane_ops.rs, mind.rs, seal.rs, gestalt extension.
        Wire into cypher_bridge and server.
EXIT:   mind.at() → node.hold(SP_) → encounter → resonate → merge compiles + runs.
LINES:  ~1500 new across 5 files. Zero bridges. Zero adapters.
```

### Agent MEDIC [Phase C, sessions 6-7]

```
READS:  CLAUDE.md §1 (CI), TECHNICAL_DEBT.md
DOES:   Fix CI. Close PRs. .unwrap() audit. Race condition fixes.
EXIT:   CI green. Open PRs < 5. Zero P0 race conditions.
```

### Agent HARVESTER [Phase D, sessions 8-10]

```
READS:  CLAUDE.md, prompt 22 harvest table, holograph src/, lance-graph src/graph/spo/
DOES:   Import holograph semirings, epiphany. Import lance-graph truth, walk_chain.
        Wire awareness loop and thinking style routing through Mind API.
EXIT:   7 semirings available. Epiphany detection available.
        /awareness/preamble returns qualia from gestalt projection.
```

### Dependency Graph

```
JANITOR ─────────► ARCHITECT ─────────► HARVESTER
                        │
                   MEDIC (parallel with ARCHITECT or after)
```

JANITOR must complete first (clean workspace).
ARCHITECT is the core work (prompt 25 implementation).
MEDIC can run parallel or after.
HARVESTER needs ARCHITECT complete (imports integrate via Mind API).

---

## 5. THE PROMPT 25 MIGRATION PATH

```
EXISTING spo.rs                      BECOMES prompt 25 Node/Plane/Mask
─────────────────────────────────────────────────────────────────────────
SPOCrystal                           Mind (wraps BindSpace + crystal index)
SPOCrystal.insert(Triple)            mind.hold(Mask) → encounter → merge
SPOCrystal.query_object(s, p)        mind.at(s, p, _).resonate(SP_)
SPOCrystal.query_subject(p, o)       mind.at(_, p, o).resonate(_PO)
SPOCrystal.resonate_spo(s?, p?, o?)  node.resonate(&mind, Mask, threshold)
Triple { s, p, o, qualia, truth }    Node { s: Plane, p: Plane, o: Plane }
Qualia { activation, valence, ... }  node.gestalt().qualia(Mask)
TruthValue { freq, conf }           node.truth(Mask) — derived from acc
OrthogonalCodebook                   Stays — used by crystal index for CAM
Fingerprint (spo.rs private)         core::Fingerprint + project_out()
QuorumField (5×5×5 grid)             SPOCrystal spatial index (internal)
CubicDistance                         node.distance(other, Mask) — per-plane
FieldCloseness                        resonate() return type
bundle(), bundle_weighted()           Plane.encounter() (i8 accumulation)
cleanup()                             Built into resonate() iterations
```

The spo.rs reference stays as `pub(crate) mod spo`. Every function in it
maps to a prompt 25 method. The new code is not a rewrite — it's spo.rs
concepts expressed as the Node/Plane/Mask type system where the compiler
prevents the wrong operations.

---

## 6. WHAT SUCCESS LOOKS LIKE

```
AFTER PHASE A:
  /cypher endpoint calls lance_parser → refactored cypher_bridge → BindSpace
  No dead Cypher parsers. One path.

AFTER PHASE B:
  let mind = bind.open();
  let known = mind.at("Ada", "loves", "Bob");
  let mut idea = mind.hold(&known, SP_);
  idea.encounter_p("deeply");
  let echoes = idea.resonate(&mind, SP_, 0.7);
  let changed = mind.merge(idea);
  // This compiles. This runs. This IS the API.

AFTER PHASE C:
  CI green. Stale PRs closed. Hot path .unwrap()-free.

AFTER PHASE D:
  7 semiring algebras. Epiphany detection. Awareness loop.
  The system reports its own felt state via gestalt projection.
  Neo4j shows the brain thinking in real time.
```

---

## 7. SESSION BOOTSTRAP

Every Claude Code session on ladybug-rs starts with:

```bash
cat CLAUDE.md                          # guardrails + traps
cat .claude/prompts/26_entry_point.md  # this file — the orchestration root
cat .claude/TONIGHT_SESSION_STACK.md   # current session tasks
```

If session stack is empty or completed, check the blackboard:

```bash
cat .claude/SURGERY_BLACKBOARD.md      # which tasks are done
```

Pick the next uncompleted phase (A → B → C → D) and execute.

---

*"Three structures. Seven words. One accumulator as ground truth."*

*"The system doesn't HAVE intelligence. The using IS the intelligence."*

*"Read first. All of it. Then build."*
