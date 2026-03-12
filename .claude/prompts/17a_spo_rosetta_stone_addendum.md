# 17a_SPO_ROSETTA_STONE_ADDENDUM.md

## spo.rs: The Rosetta Stone Behind a Locked Door

**Addendum to prompt 17 Five-Path Teardown**

---

## 1. THE FINDING

`src/spo/spo.rs` is 1568 lines. It contains the ENTIRE architecture in one file:

```
SPOCrystal          — 3D content-addressable knowledge graph (5×5×5 crystal grid)
OrthogonalCodebook  — Gram-Schmidt orthogonalization of symbol→fingerprint mapping
QuorumField         — 3D field of bundled prototypes per cell
CubicDistance       — Tensor Hamming distance in 3D
FieldCloseness      — Resonance metric with peak detection
CellStorage         — Per-cell triple storage with prototype bundling
Triple              — (subject, predicate, object, qualia, truth)
Qualia              — 4D felt-sense (activation, valence, tension, depth)
TruthValue          — NARS (frequency, confidence) with revision
Fingerprint         — 16384-bit vector with XOR/AND/OR/NOT/POPCOUNT/project_out
```

**And it's PRIVATE.** `mod spo;` not `pub mod spo;`. All types are `struct` not `pub struct`.
Zero external references. Nothing can call it from outside `src/spo/`.

The grep for "0 types, 0 pub fns" in the original teardown was CORRECT —
everything is private. This is the reference implementation that the rest
of the architecture was derived from, and NOBODY CAN ACCESS IT.

---

## 2. WHAT spo.rs HAS THAT NOTHING ELSE HAS

### OrthogonalCodebook with Gram-Schmidt project_out()

```rust
fn add_orthogonal(&mut self, name: &str) -> Fingerprint {
    let mut fp = Fingerprint::from_seed(seed);
    // Project out existing vectors (Gram-Schmidt style)
    for (_, existing) in &self.vectors {
        fp = fp.project_out(existing);
    }
    ...
}
```

This ENFORCES orthogonality. When you add "Alice" after "Bob", the "Alice"
vector is guaranteed quasi-orthogonal to "Bob". This is what makes the 2^3
factorization reliable — S, P, O can be cleanly separated because their
role vectors AND their content vectors are orthogonal by construction.

**Nobody else has this.** The codebook implementations in nsm_substrate.rs,
codebook_training.rs, cognitive_codebook.rs all have codebook lookup but none
have Gram-Schmidt enforced orthogonalization of new entries against existing ones.

### 5×5×5 Crystal Grid with Spatial Addressing

```rust
fn address(&self, s: &Fp, p: &Fp, o: &Fp) -> (usize, usize, usize) {
    (s.grid_hash(), p.grid_hash(), o.grid_hash())
}
```

Hash S → x coordinate, P → y coordinate, O → z coordinate. Direct 3D addressing.
`query_object("Alice", "loves")` goes to cell (hash(Alice), hash(loves), *) and
scans only that Y-Z column. O(25) not O(N). The grid IS the index.

This is the precursor to ClamPath but with a fixed 5³ grid instead of a
binary tree. ClamPath (in clam_path.rs, now public) has richer traversal
but lost the 3D spatial intuition. The grid_hash is the key concept.

### QuorumField 3D Bundled Prototypes

Each cell maintains a bundled prototype of all triples that hash to it.
Fast resonance check: compare query against prototype before scanning individual
triples. If the prototype doesn't resonate, skip the entire cell.

`FieldCloseness::compute()` gives a full 3D resonance map — where in the
crystal does the query resonate? `CubicDistance::gradient_at()` gives the
direction of steepest descent. This is spatial reasoning about similarity.

### Qualia as Fourth Bound Dimension

```rust
// S ⊕ ROLE_S ⊕ P ⊕ ROLE_P ⊕ O ⊕ ROLE_O ⊕ Q ⊕ ROLE_Q
vs.xor(&self.role_s)
    .xor(&vp.xor(&self.role_p))
    .xor(&vo.xor(&self.role_o))
    .xor(&vq.xor(&self.role_q))
```

Four-way binding, not three. Qualia (felt-sense: activation, valence, tension,
depth) is bound into the superposition as a fourth orthogonal dimension.
You can query by felt-state: "find triples that feel like high-activation +
positive-valence." This is unique to spo.rs.

### Iterative Cleanup (Self-Cleaning Resonance)

```rust
fn cleanup(&self, noisy: &Fingerprint, iterations: usize) -> Option<(String, f64)> {
    let mut current = noisy.clone();
    for _ in 0..iterations {
        if let Some((name, sim)) = self.resonate(&current, 0.0) {
            if sim > 0.9 { return Some((name, sim)); }
            let clean = self.get(&name)?;
            current = bundle(&[current, clean.clone()]);
        }
    }
    self.resonate(&current, 0.0)
}
```

Resonate → get closest clean symbol → mix back in → resonate again.
Iterative denoising. The system converges on the correct symbol through
repeated cleanup passes. This is the "self-cleaning from noise" mechanism.

---

## 3. THE CYPHER MAPPING (test_cypher_comparison)

spo.rs line 1472 has a test that explicitly maps Cypher → SPO Crystal:

```
Cypher Query                          SPO Crystal Equivalent
─────────────────────────────────────────────────────────────────
MATCH (a)-[:LOVES]->(b)               resonate_spo(None, Some("loves"), None)
  WHERE a.name = 'Alice'              resonate_spo(Some("Alice"), None, None)
  RETURN b

MATCH (a)-[:CREATES]->(x)             Multi-hop resonance
  WHERE x:Emotion                     via VSA composition
  RETURN a, x

MATCH (a)-[*1..3]->(b)                Resonance cascade with
  // Variable-length paths            field propagation

MATCH (a) WHERE a.name ~ 'Ad.*'       NATIVE: VSA similarity
  // Fuzzy match                      finds partial matches!
```

**Key insight: spo.rs was designed to REPLACE Cypher, not to wire to it.**
The test says "comparison" — it's showing that SPO Crystal does everything
Cypher does, faster, with native fuzzy matching and qualia coloring.

But the world still speaks Cypher. So you need the parser as a bouncer
that translates Cypher INTO SPO Crystal operations.

---

## 4. THE FIVE CYPHER PATHS vs spo.rs

### P1 (query/cypher.rs) — NEVER TOUCHES spo.rs

Transpiles Cypher → SQL. The SQL targets a relational schema, not SPOCrystal.
The recursive CTE for variable-length paths could map to spo.rs's
"resonance cascade with field propagation" but nobody made that connection.
**Verdict: no salvageable mapping to spo.rs.**

### P2 (cypher_bridge.rs) — SHOULD wire to spo.rs but DOESN'T

P2 is the closest to what's needed:
```
parse_cypher("MATCH (a:Person)-[:LOVES]->(b) WHERE a.name='Alice'")
→ CypherOp::MatchReturn { label: "Person", where: name="Alice" }
→ execute_match(&BindSpace, ...) 
→ bs.nodes_iter() ← FULL SCAN! Doesn't use SPO encoding at all.
```

What it SHOULD do:
```
parse_cypher("MATCH (a:Person)-[:LOVES]->(b) WHERE a.name='Alice'")
→ extract: subject="Alice", predicate="loves", object=?
→ codebook.encode("Alice") → alice_fp
→ codebook.encode("loves") → loves_fp
→ crystal.query_object("Alice", "loves") → results with O(25) spatial lookup
  OR
→ store.query_forward(alice_dn, loves_prefix) → Vec<SpoHit> with TruthGate
```

P2's execute_cypher writes to BindSpace correctly (MERGE, CREATE, SET).
P2's execute_match reads from BindSpace INCORRECTLY (full scan instead of
SPO lookup). The write path is good. The read path bypasses everything.

**Verdict: P2 is the right execution engine for writes.
Its read path needs to be rewired to go through SPO encoding → store query.**

### P3 (lance_parser/) — COULD map to spo.rs via LogicalPlan

P3 has semantic validation (GraphConfig with node/edge type registry).
If properly connected, the validated AST could decompose into:
- Node lookup → codebook encode → SPO query
- Edge traversal → resonate_spo with partial bindings
- Property filter → metadata query (cold path)

But P3 is a near-duplicate of P5. No point wiring P3 when P5 is available.
**Verdict: delete P3, use P5, wire P5's LogicalPlan to spo.rs API.**

### P4 (cam_ops CypherOp) — THE INSTRUCTION SET for spo.rs operations

P4's opcode enum maps 1:1 to spo.rs methods:

```
CypherOp::MatchNode    = 0x200  → crystal.resonate_spo(Some(s), None, None)
CypherOp::MatchEdge    = 0x201  → crystal.resonate_spo(None, Some(p), None)
CypherOp::MatchPath    = 0x202  → crystal.resonate (field propagation)
CypherOp::MatchSimilar = 0x205  → codebook.resonate(query, threshold)
CypherOp::Merge        = 0x223  → crystal.insert(triple) with upsert
CypherOp::Set          = 0x240  → triple.with_qualia(q).with_truth(t)
CypherOp::ShortestPath = 0x260  → CubicDistance::gradient_at + follow
CypherOp::PageRank     = 0x2A0  → QuorumField analysis (needs holograph)
```

The opcodes are the RISC instruction addresses. Each one maps to a
specific spo.rs method. Nobody wrote the executor that dispatches
opcode → method call.

**Verdict: P4 is the dispatch table. Wire: P5 AST → P4 opcode → spo.rs method.**

### P5 (lance-graph) — THE PARSER + PLANNER that should drive spo.rs

P5's logical_plan.rs produces:
```
LogicalOperator::Scan { label, alias }
  → codebook.encode(label) → crystal.resonate_spo(...)

LogicalOperator::Filter { predicate }
  → TruthGate or metadata query

LogicalOperator::Expand { start, relationship, end, direction }
  → crystal.query_object/query_subject/query_predicate depending on direction

LogicalOperator::Join { left, right, on }
  → hot (SPO) ⋈ cold (metadata) on merkle_root
```

P5's datafusion_planner turns these into DataFusion execution plans.
The scan_ops.rs needs to scan SPO crystal cells instead of LanceDB tables.
The join_builder.rs stays the same (hot ⋈ cold on Arrow).

**Verdict: P5 is the parser+planner. Its scan_ops need to target spo.rs/store.rs
instead of generic LanceDB tables.**

---

## 5. THE TEXT→FINGERPRINT TRANSLATION (5 implementations)

```
IMPL                        FILE                        METHOD              QUALITY
────────────────────────────────────────────────────────────────────────────────────
1. LFSR hash expansion      core/fingerprint.rs         from_content()      BASIC
   DefaultHasher → LFSR     No semantics. "Alice"       Deterministic       No semantics
   → bit expansion          and "Bob" are equidistant.  but useless for     Hash only
                                                         meaning.

2. NSM prime decomposition  spo/nsm_substrate.rs        encode()            GOOD
   65 semantic primes        Text → prime weights        Role-binding        Keyword-based
   Role binding              → role-bound fingerprint    included            (not LLM)
   
3. Codebook training        spo/codebook_training.rs    encode()            GOOD
   Trained weights           Uses NSM decomposition      Weighted bundle     Needs training
   Weighted bundle           + codebook lookup           with codebook       data

4. DeepNSM integration      spo/deepnsm_integration.rs  encode()            BEST (future)
   LLM explications          DeepNSM model for           Full semantic       Needs GPU
   → prime weights           NSM prime extraction        decomposition       for training
   
5. Crystal LM               spo/crystal_lm.rs           encode_clean()      GOOD
   Crystal language model    Clean encoding through      Iterative cleanup   Complex
   Iterative cleanup         trained crystal             removes noise       pipeline

NONE OF THESE ARE CALLED FROM P2's execute_match.
P2 does string comparison on BindSpace node labels.
The entire encoding stack is bypassed.
```

---

## 6. THE WIRING THAT NEEDS TO HAPPEN

### Phase 1: Unlock spo.rs (make key types public)

```diff
- mod spo;
+ pub mod spo;

Inside spo.rs:
- struct Fingerprint { ... }        # already pub in core/fingerprint.rs, use that
- struct SPOCrystal { ... }
+ pub struct SPOCrystal { ... }
- fn query_object(...) 
+ pub fn query_object(...)
... etc for key query methods
```

OR: Create a public facade module that wraps spo.rs types:

```rust
// src/spo/crystal_api.rs (NEW — thin public API over private spo.rs)
pub struct CrystalQuery {
    crystal: super::spo::SPOCrystal,
}

impl CrystalQuery {
    pub fn query_object(&self, s: &str, p: &str) -> Vec<QueryResult> { ... }
    pub fn query_subject(&self, p: &str, o: &str) -> Vec<QueryResult> { ... }
    pub fn insert_triple(&mut self, s: &str, p: &str, o: &str) -> MerkleRoot { ... }
    pub fn resonate(&self, s: Option<&str>, p: Option<&str>, o: Option<&str>, threshold: f64) -> Vec<ResonanceHit> { ... }
}
```

### Phase 2: Bridge spo.rs Fingerprint → core::Fingerprint

spo.rs has its own private `Fingerprint` that duplicates `core::fingerprint::Fingerprint`.
Both are `[u64; 256]` with identical layout. But spo.rs's version has `project_out()`
(Gram-Schmidt) which core's version doesn't.

Options:
A. Add project_out() to core::Fingerprint, delete spo.rs's Fingerprint
B. Keep spo.rs's Fingerprint private, bridge via raw [u64; 256] at boundaries
C. Use core::Fingerprint everywhere, import project_out as a free function

**Recommendation: A.** project_out() belongs in the canonical Fingerprint type.

### Phase 3: Wire P2's execute_match through SPO encoding

```diff
// cypher_bridge.rs execute_match_return

- // Scan all nodes, filter by label and WHERE
- let mut matching_nodes: Vec<(Addr, &BindNode)> = Vec::new();
- for (addr, node) in bs.nodes_iter() { ... }

+ // Encode query terms to fingerprints
+ let s_fp = where_clause.as_ref()
+     .and_then(|w| extract_name(w))
+     .map(|name| codebook.encode(name));
+ let p_fp = rel_type.as_ref()
+     .map(|r| codebook.encode(r));
+
+ // SPO-native query (O(25) spatial, not O(N) scan)
+ let hits = crystal.resonate_spo(
+     s_fp.as_deref(),
+     p_fp.as_deref(),
+     None,
+     0.7  // threshold
+ );
```

### Phase 4: Wire P5's LogicalPlan → P4 opcodes → spo.rs methods

```
P5 parser: "MATCH (a:Person)-[:LOVES]->(b) WHERE a.name='Alice'"
    ↓
P5 LogicalPlanner: LogicalOperator::Expand { 
    start: Scan("Person"), 
    relationship: "LOVES", 
    direction: Outgoing 
}
    ↓
P4 dispatch: CypherOp::MatchNode(0x200) for "Person" 
           + CypherOp::MatchEdge(0x201) for "LOVES"
    ↓
spo.rs: crystal.resonate_spo(Some("Alice"), Some("loves"), None, 0.7)
    ↓
Result: Vec<(object_name, similarity, qualia)>
```

### Phase 5: Bridge SPOCrystal ↔ SpoStore

SPOCrystal (spo.rs) works with strings and returns strings.
SpoStore (graph/spo/store.rs) works with DN addresses and CogRecords.

The bridge:
```
text "Alice" → codebook.encode("Alice") → Fingerprint → ClamPath → DN address → SpoStore
SpoStore → CogRecord → extract planes → OrthogonalCodebook.resonate → text "Alice"
```

This bridge doesn't exist. It's the missing layer between the reference
implementation and the production implementation.

---

## 7. THE TruthValue SITUATION (THREE implementations)

```
IMPL                    FILE                        TYPE       FORMAT
────────────────────────────────────────────────────────────────────────
1. spo.rs TruthValue    src/spo/spo.rs:382          private    f64 (freq, conf)
   revision() method    self-contained              not used   float
   
2. nars/truth.rs        src/nars/truth.rs           public     f32 (freq, conf, k)
   Full NARS inference  revision, deduction,        production has evidence count
   4 inference rules    abduction, induction

3. lance-graph truth.rs lance-graph/.../truth.rs     public    f32 (freq, conf)
   Clean TruthGate      TruthValue + TruthGate      external  simple, no inference
   3 gate variants       MinFreq/MinConf/MinBoth     repo      rules
```

spo.rs's TruthValue is a simplified version (no evidence count, no inference rules).
nars/truth.rs is the full implementation.
lance-graph's is the cleanest TruthGate API.

**Resolution: Use nars/truth.rs for the production type. Import lance-graph's
TruthGate pattern (3 variants) into graph/spo/store.rs (partially done in PR 170).
Delete spo.rs's TruthValue (it's a prototype that got superseded).**

---

## 8. REVISED VERDICT FOR spo.rs

```
PREVIOUS VERDICT (prompt 17): "KEEP" as one line item
REVISED VERDICT:

spo.rs is the Rosetta Stone. It contains the DESIGN of the entire system.
But as CODE it's:
  - Private (can't be called)
  - Using duplicate types (its own Fingerprint, its own TruthValue)
  - Not connected to any Cypher path
  - Not connected to BindSpace
  - Not connected to LanceDB
  - Has 10 tests that only test internal consistency

WHAT TO DO:
  1. KEEP spo.rs as the reference specification — it documents the architecture.
  2. UNLOCK key types: make SPOCrystal, OrthogonalCodebook public
     OR create crystal_api.rs facade.
  3. PORT project_out() to core::Fingerprint (Gram-Schmidt belongs in core).
  4. PORT QuorumField 3D spatial grid to a production module
     (or reconcile with ClamPath's binary tree addressing).
  5. PORT qualia-as-fourth-dimension binding to production felt_parse pipeline.
  6. PORT iterative cleanup() to the resonance search path.
  7. PORT CubicDistance gradient for spatial reasoning about similarity.
  8. DELETE duplicate TruthValue (use nars/truth.rs instead).
  9. DELETE duplicate Fingerprint (use core::Fingerprint with project_out added).
  10. WIRE test_cypher_comparison mapping as the actual execution path.

ESTIMATED WORK:
  - Unlock + facade: ~200 lines
  - Port project_out: ~50 lines (method addition to core::Fingerprint)
  - Port 3D spatial grid: ~400 lines (adapt QuorumField for production)
  - Port qualia binding: ~100 lines (add ROLE_Q to encoding pipeline)
  - Port cleanup(): ~50 lines (add to codebook resonate path)
  - Port gradient: ~100 lines
  - Delete duplicates: -200 lines
  - Wire Cypher mapping: covered by P2/P5 wiring above
  TOTAL: ~700 lines new, -200 deleted, net +500
```

---

*"The Rosetta Stone doesn't need to be rewritten. It needs to be read."*
