# TONIGHT_SESSION_STACK.md (v3 — REFACTOR, NOT BRIDGE)

## 4 Hours. No Bridges. No Adapters. Refactor Properly.

---

## SESSION 1: Kill P1, Refactor P2 to Consume P3 AST Directly [REPO: ladybug-rs]

**Read COMPLETELY before writing:**
```
src/query/lance_parser/ast.rs        (532 lines — the AST types)
src/cypher_bridge.rs                 (897 lines — the executor)
src/query/cypher.rs                  (1560 lines — the thing being deleted)
src/bin/server.rs lines 1621-1655    (the /cypher endpoint)
```

**The refactor:**

P2's `execute_cypher()` takes `&[CypherOp]`. CypherOp is a flat enum with 5 variants
(MergeNode, CreateNode, CreateEdge, SetProperty, MatchReturn).

P3's parser produces `lance_parser::ast::CypherQuery` with rich tree types
(ReadingClause, MatchClause, GraphPattern, NodePattern, PathPattern, etc.).

**DO NOT create an ast_bridge.rs. Rewrite cypher_bridge.rs to take P3 types directly:**

```rust
// BEFORE (current cypher_bridge.rs):
pub fn execute_cypher(bs: &mut BindSpace, ops: &[CypherOp]) -> Result<CypherResult, String>

// AFTER (refactored):
pub fn execute_cypher(bs: &mut BindSpace, query: &lance_parser::ast::CypherQuery) -> Result<CypherResult, String>
```

The internal `CypherOp` enum DISAPPEARS. The execute functions work directly on the AST:

```rust
// BEFORE:
CypherOp::MergeNode { labels, properties } => execute_merge_node(bs, labels, properties, &mut result)?

// AFTER:
for clause in &query.reading_clauses {
    match clause {
        ReadingClause::Match(m) => execute_match(bs, m, &query.where_clause, &query.return_clause, &mut result)?,
        ReadingClause::Unwind(u) => execute_unwind(bs, u, &mut result)?,
    }
}
// CREATE/MERGE/SET handled from query.update_clauses (or however P3 structures writes)
```

Keep P2's execution LOGIC (find_node_by_label_and_name, evaluate_where, MERGE semantics).
Replace P2's TYPES (CypherOp, NodeRef, WhereClause, CypherValue) with P3's types.
Delete P2's parse_cypher() (P3's parse_cypher_query replaces it).
Keep P2's execute_* functions, rewrite their signatures to take P3 AST nodes.

**Steps:**

```
1. Delete src/query/cypher.rs
   - Save lines 1253-1361 to src/query/cte_builder.rs first
   - Remove from query/mod.rs

2. Rewrite src/cypher_bridge.rs:
   - Remove: CypherOp enum, NodeRef enum, WhereClause enum, CypherValue enum
   - Remove: parse_cypher() function
   - Change: execute_cypher signature to take &CypherQuery (P3 type)
   - Change: execute_merge_node to take &NodePattern (P3 type)
   - Change: execute_create_edge to take &PathPattern (P3 type)
   - Change: execute_match_return to take &MatchClause + &WhereClause (P3 types)
   - Change: evaluate_where to take &BooleanExpression (P3 type)
   - Keep: CypherResult (it's the output format, not input)
   - Keep: find_node_by_label_and_name (working scan logic)
   - Keep: MERGE upsert semantics
   
   Map P3's PropertyValue → to the value types execute_* needs
   Map P3's BooleanExpression → to what evaluate_where checks
   This is INLINE refactoring, not a bridge module.

3. Wire server.rs:
   use ladybug::query::lance_parser::parse_cypher_query;
   use ladybug::cypher_bridge::execute_cypher;
   
   match parse_cypher_query(&query) {
     Ok(ast) => match execute_cypher(&mut bs, &ast) {
       Ok(result) => serialize
       Err(e) => error
     }
     Err(e) => parse error
   }

4. Rename cam_ops CypherOp → CypherInstruction

5. cargo check --no-default-features --features "simd"
```

**Exit gate:** `/cypher` endpoint calls P3 parser → refactored P2 executor. Zero bridge types.

---

## SESSION 2: Unlock spo.rs Properly [REPO: ladybug-rs]

**Read COMPLETELY:** `src/spo/spo.rs` (all 1568 lines, every function, every type)

```
1. src/spo/mod.rs: `mod spo` → `pub(crate) mod spo`

2. In spo.rs itself: make key types pub(crate)
   pub(crate) struct SPOCrystal
   pub(crate) struct OrthogonalCodebook  
   pub(crate) fn bundle()
   pub(crate) fn bundle_weighted()
   Keep private: internal helpers, Fingerprint (use core::Fingerprint instead)

3. Add to core/fingerprint.rs:
   pub fn project_out(&self, other: &Fingerprint) -> Fingerprint
   pub fn dot_bipolar(&self, other: &Fingerprint) -> i64
   (Port from spo.rs, adapt to core::Fingerprint layout)

4. In spo.rs: replace internal Fingerprint usage with core::Fingerprint
   where signatures cross module boundary.
   Keep internal Fingerprint for private functions if easier —
   but crystal_api RETURNS core::Fingerprint.

5. Create src/spo/crystal_api.rs — NOT a facade. Direct re-export + helpers:
   pub use super::spo::{SPOCrystal, OrthogonalCodebook};
   
   Plus convenience constructors that use core types:
   pub fn new_crystal() -> SPOCrystal
   pub fn encode_and_insert(crystal: &mut SPOCrystal, s: &str, p: &str, o: &str)
   pub fn query_object(crystal: &SPOCrystal, s: &str, p: &str) -> Vec<QueryHit>
   
   These are THIN — they call spo.rs methods directly, not wrap them.

6. cargo check
```

**Exit gate:** `use crate::spo::crystal_api::SPOCrystal` works from server.rs.

---

## SESSION 3: Wire Crystal Into Cypher Execute [REPO: ladybug-rs]

**Read:** refactored cypher_bridge.rs (from session 1) + crystal_api.rs (from session 2)

```
1. Add SPOCrystal to DatabaseState in server.rs

2. In cypher_bridge execute_match (the read path):
   REPLACE bs.nodes_iter() full scan
   WITH crystal.resonate_spo() for queries that have S and/or P
   
   KEEP nodes_iter as fallback for label-only queries without SPO terms
   
   This is refactoring execute_match, not adding a layer.

3. In cypher_bridge execute_merge/create:
   AFTER writing to BindSpace, ALSO insert into SPOCrystal
   Both stores get the data. Crystal is the fast index.
   BindSpace is the source of truth.

4. cargo check + test with curl
```

**Exit gate:** MATCH with subject+predicate hits Crystal (O(25)), not scan (O(N)).

---

## SESSION 4: CI + Stale PRs [REPO: ladybug-rs, maybe rustynum]

```
1. cargo check --all-features (or closest working feature set)
   Fix what breaks. Don't add features.

2. Close PRs #11-#33, #54
   Evaluate #168, #169 against #170

3. If rustynum CI blocks ladybug-rs:
   ⚠ SWITCH REPO — read rustynum CLAUDE.md first
   Fix the minimum needed for cargo check to pass
```

---

## TIMING

```
SESSION   EST.        PRIORITY
1         90 min      P0 (the big refactor)
2         60 min      P0
3         45 min      P1
4         30 min      P2
TOTAL     ~4 hours
```

---

*"No bridges. No adapters. No facades. Refactor the types. Inline the logic."*
