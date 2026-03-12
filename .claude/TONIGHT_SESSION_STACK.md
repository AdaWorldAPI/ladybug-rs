# TONIGHT_SESSION_STACK.md (v2 — CORRECTED)

## ⚠ CORRECTION FROM v1

**v1 said delete lance_parser (P3, 5532 lines). WRONG.**
lance_parser IS the production Cypher parser. It's a correctly adapted
port of lance-graph's nom parser with proper ladybug-rs error type integration.
The "divergence" in semantic.rs is correct QueryError adaptation, not drift.

**Session 1 now deletes ONLY P1 (cypher.rs, 1560 lines).**

---

## ⚠ REPO FOCUS RULES

```
EVERY TASK IS TAGGED WITH A REPO.
DO NOT touch files in a different repo than the current tag.
When switching repos: STOP. Read that repo's CLAUDE.md FIRST.

Current repo names (DO NOT RENAME ANYTHING TONIGHT):
  ladybug-rs     ← main surgery target
  rustynum       ← READ ONLY from ladybug-rs sessions
```

---

## SESSION 1: Delete P1 Only [REPO: ladybug-rs]

**Read first:** `CLAUDE.md`, `src/query/cypher.rs` (skim — understand what's being removed)

```
1. Save CTE generator FIRST:
   cp src/query/cypher.rs /tmp/cypher_backup.rs
   Extract lines 1253-1361 → src/query/cte_builder.rs
   (Recursive CTE for variable-length paths — UNIQUE, neither P3 nor P5 has it)

2. Delete src/query/cypher.rs

3. Update src/query/mod.rs:
   Remove: pub mod cypher;
   Remove: pub use cypher::{CypherParser, CypherQuery, CypherTranspiler, cypher_to_sql};
   
4. Fix broken references:
   server.rs:1625 calls cypher_to_sql — this will break. COMMENT IT OUT for now:
     // TODO: wire lance_parser + cypher_bridge (session 2)
     http_error(501, "cypher_not_implemented", "being rewired", format)

   hybrid.rs:20 imports CypherParser — comment out or remove usage

5. cargo check --no-default-features --features "simd"

6. Rename in src/learning/cam_ops.rs:
   CypherOp → CypherInstruction
   Update all references within src/learning/
   
7. cargo check again
```

**-1560 lines deleted. lance_parser UNTOUCHED.**

**Commit:** `chore: delete P1 cypher.rs (hand-rolled transpiler) — rescue CTE builder`

---

## SESSION 2: Build AST→CypherOp Bridge + Wire Server [REPO: ladybug-rs]

**Read first:** 
- `src/query/lance_parser/ast.rs` (the AST types P3 produces)
- `src/cypher_bridge.rs` (the CypherOp types P2 consumes)

```
1. Create src/query/ast_bridge.rs (~100-150 lines):
   Convert lance_parser AST → cypher_bridge CypherOp
   
   use crate::query::lance_parser::ast::{CypherQuery as ParsedQuery, ...};
   use crate::cypher_bridge::{CypherOp, NodeRef, CypherValue};
   
   pub fn ast_to_ops(parsed: &ParsedQuery) -> Result<Vec<CypherOp>, String> {
       // Walk the AST, produce CypherOp list
       // MATCH clause → CypherOp::MatchReturn
       // CREATE clause → CypherOp::CreateNode / CreateEdge
       // MERGE → CypherOp::MergeNode
       // SET → CypherOp::SetProperty
   }

2. Wire server.rs /cypher endpoint:
   Replace the commented-out TODO from session 1:
   
   use ladybug::query::lance_parser::parse_cypher_query;
   use ladybug::query::ast_bridge::ast_to_ops;
   use ladybug::cypher_bridge::execute_cypher;
   
   match parse_cypher_query(&query) {
     Ok(parsed) => {
       // Validate via semantic analyzer (P3 has this!)
       match ast_to_ops(&parsed) {
         Ok(ops) => {
           let bs = db.cog_redis.bind_space_mut();
           match execute_cypher(bs, &ops) {
             Ok(result) => serialize result as JSON
             Err(e) => http_error(500, ...)
           }
         }
         Err(e) => http_error(400, "ast_conversion", &e, format)
       }
     }
     Err(e) => http_error(400, "cypher_parse", &e.to_string(), format)
   }

3. cargo check
4. Test: curl /cypher with MERGE → node created
5. Test: curl /cypher with MATCH → results returned
```

**~150 lines new. Full parse→validate→execute pipeline working.**

**Commit:** `feat: wire lance_parser → ast_bridge → cypher_bridge → server.rs /cypher`

---

## SESSION 3: Unlock spo.rs [REPO: ladybug-rs]

**Read first:** `src/spo/spo.rs` (ENTIRE FILE, 1568 lines. No skimming.)

```
1. src/spo/mod.rs: change `mod spo;` → `pub(crate) mod spo;`

2. Add to core/fingerprint.rs:
   project_out() — Gram-Schmidt from spo.rs lines 116-140
   dot_bipolar() — from spo.rs lines 109-115
   
3. Create src/spo/crystal_api.rs (~200 lines):
   Public facade wrapping private SPOCrystal.
   
4. Add to src/spo/mod.rs: pub mod crystal_api;

5. cargo check
```

**Commit:** `feat: unlock spo.rs — pub(crate), crystal_api facade, project_out`

---

## SESSION 4: Wire Crystal to Cypher Bridge [REPO: ladybug-rs]

**Read:** `src/cypher_bridge.rs` lines 370-420, `src/spo/crystal_api.rs`

```
1. Add CrystalQuery to server DatabaseState
2. In cypher_bridge execute_match: try crystal_api first, fall back to nodes_iter
3. cargo check
```

**Commit:** `feat: MATCH queries use SPO Crystal O(25) lookup with nodes_iter fallback`

---

## SESSION 5: CI Triage [REPO: ladybug-rs, then rustynum if needed]

```
⚠ REPO SWITCH possible. Read each CLAUDE.md before touching.
Focus: get cargo check green, not cargo test (tests can wait).
```

---

## SESSION 6: Close Stale PRs [REPO: ladybug-rs]

```
Close #11-#33, #54 with "superceded by main"
Evaluate #168, #169 against merged #170
Target: open PRs ≤ 5
```

---

## SESSION 7 (STRETCH): Harvest lance-graph truth.rs [REPO: ladybug-rs]

```
READ lance-graph truth.rs (175 lines). 
MERGE into ladybug-rs graph/spo/.
ADD walk_chain_forward from lance-graph store.rs.
```

---

## SESSION ORDER

```
SESSION   WHAT                    EST. TIME   PRIORITY
1         Delete P1 only          20-30 min   P0
2         Bridge P3→P2 + server   45-60 min   P0
3         Unlock spo.rs           45-60 min   P0
4         Crystal → Cypher        30-45 min   P1
5         CI triage               30-60 min   P1
6         Close stale PRs         15 min      P2
7         Harvest truth.rs        30 min      P2 (stretch)
```

**Minimum tonight:** Sessions 1-3 (delete P1, bridge P3→P2, unlock spo.rs).

---

*"Read first. All of it. Then decide."*
