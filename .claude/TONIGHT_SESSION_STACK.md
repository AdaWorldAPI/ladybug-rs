# TONIGHT_SESSION_STACK.md

## Claude Code Session Stack — Ordered, Repo-Tagged, Focus-Guarded

**Date:** 2026-03-12 evening
**Goal:** Sisyphus harvest + minimal wiring. No new architecture.

---

## ⚠ REPO FOCUS RULES

```
EVERY TASK IS TAGGED WITH A REPO. 
DO NOT touch files in a different repo than the current tag.
DO NOT import types from a repo you're not currently in.
When switching repos: STOP. Read that repo's CLAUDE.md FIRST. Then proceed.

Current repo names (DO NOT RENAME ANYTHING TONIGHT):
  ladybug-rs     ← The Brain (main surgery target)
  rustynum       ← The Muscle (READ ONLY from ladybug-rs sessions)
  lance-graph    ← The Face (reference only tonight, no modifications)
  hexagon        ← The Spark (was staunen/inception, stubs only)
  ikarus         ← The Scout (was erntefeld/monolith/obsint, empty)
```

---

## SESSION 1: Sisyphus Deletions [REPO: ladybug-rs]

**Read first:** `CLAUDE.md`, `.claude/prompts/17_five_path_teardown.md`

**What to do:**

```
1. Delete src/query/cypher.rs (1560 lines)
   - FIRST: save lines 1253-1361 → src/query/cte_builder.rs
     (the recursive CTE generator for variable-length paths)
   - Remove `pub mod cypher;` from src/query/mod.rs
   - Remove `pub use cypher::{CypherParser, CypherQuery, CypherTranspiler, cypher_to_sql};`
   - Remove any imports of these types elsewhere
   
2. Delete src/query/lance_parser/ (entire directory, 5532 lines)
   - FIRST: copy the From<GraphError> impl from lance_parser/error.rs
     → paste into src/query/error.rs
   - Remove `pub mod lance_parser;` from src/query/mod.rs
   - Remove all `pub use lance_parser::*` re-exports
   
3. Verify compilation:
   cargo check --no-default-features --features "simd"
   Fix any broken imports.
   
4. Rename in src/learning/cam_ops.rs:
   CypherOp → CypherInstruction (avoid name clash with cypher_bridge.rs)
   Update all references within src/learning/

5. Verify again:
   cargo check --no-default-features --features "simd"
```

**What NOT to do:**
```
× Don't touch src/cypher_bridge.rs (that's P2, we're KEEPING it)
× Don't touch src/spo/ (that's the Rosetta Stone)
× Don't modify rustynum (read-only)
× Don't start any new architecture work
× Don't modify server.rs yet (that's session 2)
```

**Exit gate:** `cargo check --no-default-features --features "simd"` passes. -7092 lines deleted.

**Commit message:** `chore: delete P1 (query/cypher.rs) + P3 (lance_parser/) — 5 Cypher paths → 3`

---

## SESSION 2: Wire P2 to Server [REPO: ladybug-rs]

**Read first:** `CLAUDE.md §4`, `src/cypher_bridge.rs`, `src/bin/server.rs` line 1621-1655

**What to do:**

```
1. In src/bin/server.rs, find handle_cypher (line ~1621):

   CURRENT:
     match ladybug::query::cypher_to_sql(&query) {  // ← THIS IS GONE (session 1)
       Ok(sql) => return sql as JSON
       
   REPLACE WITH:
     match ladybug::cypher_bridge::parse_cypher(&query) {
       Ok(ops) => {
         let mut db = state.write().unwrap();
         let bs = db.cog_redis.bind_space_mut();
         match ladybug::cypher_bridge::execute_cypher(bs, &ops) {
           Ok(result) => {
             // Serialize CypherResult as JSON
             let json = serde_json::json!({
               "columns": result.columns,
               "rows": result.rows,
               "status": "executed"
             });
             http_json(200, &json.to_string())
           }
           Err(e) => http_error(500, "cypher_execution_error", &e, format)
         }
       }
       Err(e) => http_error(400, "cypher_parse_error", &e, format)
     }

2. Add serde_json serialization for CypherResult if needed.
   CypherResult has columns: Vec<String> and rows: Vec<Vec<String>>.
   May need to derive Serialize or manual JSON construction.

3. Test:
   cargo build --bin ladybug-server
   (may fail on path deps — if so, test with cargo check)
```

**What NOT to do:**
```
× Don't rewrite cypher_bridge.rs (it works, just wire it)
× Don't add lance-graph dependency yet
× Don't touch SPO encoding path yet
× Don't modify CogRedis / RedisAdapter
```

**Exit gate:** `cargo check` passes with the new /cypher handler.

**Commit message:** `feat: wire cypher_bridge to server.rs — /cypher now executes against BindSpace`

---

## SESSION 3: Unlock spo.rs [REPO: ladybug-rs]

**Read first:** `.claude/prompts/17a_spo_rosetta_stone_addendum.md`, `src/spo/spo.rs` (ENTIRE FILE)

**What to do:**

```
1. In src/spo/mod.rs:
   Change: mod spo;
   To:     pub(crate) mod spo;

2. Add project_out() to src/core/fingerprint.rs:
   Copy the Gram-Schmidt projection from spo.rs lines 116-140.
   Adapt: spo.rs uses `data: [u64; N64]` where N64=256.
   core::Fingerprint uses `data: [u64; FINGERPRINT_U64]` where FINGERPRINT_U64=256.
   Same layout — direct port.

   /// Gram-Schmidt projection: remove component parallel to `other`.
   /// Makes self quasi-orthogonal to `other`.
   pub fn project_out(&self, other: &Fingerprint) -> Fingerprint {
       // dot_bipolar: count matching bits minus non-matching bits
       let dot = self.dot_bipolar(other);
       let norm = other.dot_bipolar(other);
       if norm == 0 { return self.clone(); }
       
       // For binary vectors: if overlap > 50%, flip those bits
       let mut result = self.clone();
       if dot > 0 {
           // Positive correlation → XOR to decorrelate
           let mask_strength = dot as f64 / norm as f64;
           if mask_strength > 0.5 {
               return self.xor(other);
           }
       }
       result
   }
   
   /// Bipolar dot product: matching bits - non-matching bits
   pub fn dot_bipolar(&self, other: &Fingerprint) -> i64 {
       let matching = self.hamming_similarity(other);  // fraction of matching bits
       let total = FINGERPRINT_BITS as i64;
       let matches = (matching * total as f64) as i64;
       matches - (total - matches)  // matches - mismatches
   }
   
   (Adjust to use existing methods. The exact implementation matters less
   than having the method signature available on core::Fingerprint.)

3. Create src/spo/crystal_api.rs (~200 lines, thin facade):

   //! Public API over the private SPOCrystal.
   //! This module bridges spo.rs's internal types to the rest of ladybug-rs.
   
   use super::spo::{SPOCrystal, Triple, Qualia};
   use crate::core::Fingerprint;
   
   pub struct CrystalQuery {
       crystal: SPOCrystal,
   }
   
   impl CrystalQuery {
       pub fn new() -> Self { ... }
       pub fn insert_triple(&mut self, s: &str, p: &str, o: &str) { ... }
       pub fn query_object(&self, s: &str, p: &str) -> Vec<(String, f64)> { ... }
       pub fn query_subject(&self, p: &str, o: &str) -> Vec<(String, f64)> { ... }
       pub fn resonate_spo(&self, s: Option<&str>, p: Option<&str>, 
                           o: Option<&str>, threshold: f64) -> Vec<(String, f64)> { ... }
   }
   
   Add to src/spo/mod.rs:
     pub mod crystal_api;

4. cargo check --no-default-features --features "simd"
```

**What NOT to do:**
```
× Don't rewrite spo.rs internals (wrap them, don't change them)
× Don't delete spo.rs's private Fingerprint type yet (just bridge)
× Don't wire crystal_api to server yet (that's session 4)
```

**Exit gate:** `crystal_api::CrystalQuery::new()` compiles. `project_out()` has a test.

**Commit message:** `feat: unlock spo.rs — pub(crate) + crystal_api facade + project_out on core::Fingerprint`

---

## SESSION 4: Wire Crystal to Cypher Bridge [REPO: ladybug-rs]

**Read first:** `src/cypher_bridge.rs` lines 370-420 (execute_match), `src/spo/crystal_api.rs`

**What to do:**

```
1. Add CrystalQuery to server DatabaseState:
   struct DatabaseState {
       cog_redis: CogRedis,
       crystal: spo::crystal_api::CrystalQuery,  // NEW
   }

2. In cypher_bridge.rs execute_match_return:
   Add crystal parameter, try SPO lookup first, fall back to nodes_iter:
   
   // Try SPO Crystal lookup first (O(25) spatial)
   if let Some(subject_name) = extract_subject_from_where(where_clause) {
       if let Some(rel_type) = &label { // label here is actually relationship type
           let hits = crystal.resonate_spo(
               Some(subject_name), Some(rel_type), None, 0.7
           );
           if !hits.is_empty() {
               // Convert to CypherResult and return
               ...
           }
       }
   }
   // Fall back to nodes_iter scan for queries Crystal can't handle
   for (addr, node) in bs.nodes_iter() { ... }

3. Wire in server.rs: pass crystal to execute_cypher calls.

4. cargo check
```

**What NOT to do:**
```
× Don't remove the nodes_iter fallback (Crystal doesn't handle all queries yet)
× Don't touch lance-graph
× Don't start bouncer work
```

**Exit gate:** MATCH query that hits Crystal returns in O(25), not O(N).

**Commit message:** `feat: wire crystal_api to cypher_bridge — SPO lookup for MATCH queries`

---

## SESSION 5: CI Triage [REPO: ladybug-rs, then rustynum]

**Read first:** `CLAUDE.md §1` (path deps), CI failure URLs

```
⚠ REPO SWITCH: This session touches TWO repos.
Start in ladybug-rs. Then switch to rustynum if needed.
Read EACH repo's CLAUDE.md before touching it.
```

**What to do:**

```
PART A [REPO: ladybug-rs]:
  1. Read the actual CI failure log:
     https://github.com/AdaWorldAPI/ladybug-rs/actions/runs/23019030859
  2. Identify the specific compilation errors
  3. If path dep issue: check if cargo check --no-default-features --features "simd" passes
  4. Fix what's fixable without touching sibling repos
  5. If CI needs rustynum fix → go to Part B

PART B [REPO: rustynum]:
  ⚠ REPO SWITCH — read rustynum CLAUDE.md first
  1. Read CI failure: https://github.com/AdaWorldAPI/rustynum/actions/runs/22624160086
  2. Most likely: PRs 91/92 deprecated panicking APIs, broke callers
  3. Fix: update callers to use try_* variants, OR restore deprecated fns
  4. cargo test (Tier 1 only)
```

**What NOT to do:**
```
× Don't modify rustynum from a ladybug-rs mindset
× Don't add features to either repo
× Don't fix "nice to have" warnings — only fix what blocks CI
```

**Exit gate:** CI green on at least one workflow per repo.

---

## SESSION 6: Close Stale PRs [REPO: ladybug-rs]

**Read first:** Nothing special. This is janitorial.

```
1. Close with "superceded by main":
   #11, #12, #14, #15, #16 (Jan 30)
   #29, #30, #31, #32, #33 (Jan 31)
   #54 (Feb 2)

2. Evaluate against merged #170:
   #168 "SPO Integrity & Query Hardening" — check if #170 covers it
   #169 "SPO stack hardening" — check if #170 covers it
   If covered → close. If gaps remain → rebase onto main.

3. Target: open PR count < 5
```

**Exit gate:** Open PRs ≤ 5.

---

## SESSION 7 (STRETCH): Harvest lance-graph truth.rs [REPO: ladybug-rs]

**Read first:** `.claude/prompts/17_five_path_teardown.md` SPO-3 section

```
⚠ This session READS lance-graph but WRITES to ladybug-rs only.
DO NOT modify lance-graph repo.
```

**What to do:**

```
1. Read lance-graph/crates/lance-graph/src/graph/spo/truth.rs (175 lines)
   Understand: TruthValue, TruthGate with 3 variants (MinFreq/MinConf/MinBoth)

2. Read ladybug-rs/src/nars/truth.rs
   Understand: existing TruthValue, inference rules

3. Read ladybug-rs/src/graph/spo/store.rs
   Understand: existing TruthGate from PR 170

4. Merge: lance-graph's clean 3-variant TruthGate pattern
   into ladybug-rs graph/spo/store.rs (or graph/spo/truth.rs — new file)
   
   Keep: ladybug-rs's richer inference rules
   Adopt: lance-graph's cleaner gate API

5. Read lance-graph/src/graph/spo/store.rs walk_chain_forward()
   Add to ladybug-rs src/graph/spo/store.rs

6. cargo check
```

**Exit gate:** TruthGate with 3 variants compiles. walk_chain_forward exists.

**Commit message:** `feat: merge lance-graph truth.rs — clean TruthGate + walk_chain_forward`

---

## SESSION ORDER + TIMING

```
SESSION   REPO          EST. TIME   DEPENDS ON    PRIORITY
──────────────────────────────────────────────────────────────
1         ladybug-rs    30-45 min   nothing       P0 (do first)
2         ladybug-rs    20-30 min   session 1     P0
3         ladybug-rs    45-60 min   session 1     P0
4         ladybug-rs    30-45 min   sessions 2+3  P1
5         lb-rs + rn    30-60 min   nothing       P1 (parallel OK)
6         ladybug-rs    15 min      nothing       P2 (parallel OK)
7         ladybug-rs    30-45 min   session 1     P2 (stretch)
```

**Minimum tonight:** Sessions 1-3 (delete dead code, wire server, unlock spo.rs).
**Full run:** Sessions 1-6 (everything except lance-graph harvest).
**With stretch:** All 7.

---

## BETWEEN SESSIONS

```
After each session:
  1. Verify cargo check passes
  2. Commit with descriptive message
  3. Push to main (or branch if preferred)
  4. If starting a new session on SAME repo: continue
  5. If switching repos: STOP. Read new repo's CLAUDE.md. Then start.
```

---

*"Delete first. Wire second. Unlock third. The rest follows."*
