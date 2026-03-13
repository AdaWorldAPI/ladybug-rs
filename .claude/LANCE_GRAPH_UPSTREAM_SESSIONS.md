# LANCE_GRAPH_UPSTREAM_SESSIONS.md

## Fix PR #146. Split. Rebase. Contribute BlasGraph.

**Repo:** AdaWorldAPI/lance-graph (fork of lance-format/lance-graph)
**Upstream:** lance-format/lance-graph
**PR #146:** open, CI failing, merge conflicts, maintainer asked us to fix

---

## SITUATION

```
PR #146 is a kitchen sink: dep bumps + error macros + graph/spo/ module + docs
11 commits, 2577 additions, mergeable: false, state: dirty

Upstream merged 3 PRs since we opened:
  #145  Unity catalog/delta lake          (Mar 3)
  #147  Remove Simple executor            (Mar 4) ← CONFLICT
  #148  Move benchmarks to separate crate (Mar 5) ← CONFLICT

Our fork: 10 ahead, 3 behind upstream main.
```

---

## SESSION 1: Sync Fork + Close #146

```bash
cd adaworld/lance-graph

# Sync fork to upstream
git remote add upstream https://github.com/lance-format/lance-graph.git
git fetch upstream
git checkout main
git merge upstream/main
# Resolve conflicts (Cargo.lock, workspace layout changes)
git push origin main

# Close #146 with comment
```

Comment on PR #146:
```
Closing this PR to split into focused contributions:
- PR A: dep bumps (arrow 57, datafusion 51, lance 2)
- PR B: graph/spo/ module (SPO triple store, TruthGate, semiring)
- PR C: BlasGraph algebra (from holograph — 7 semirings, matrix ops)

Sorry for the kitchen sink. Splitting for clean review.
```

**Exit gate:** Fork synced to upstream main. #146 closed.

---

## SESSION 2: PR A — Dep Bumps Only

```bash
git checkout -b feat/bump-arrow-57-datafusion-51-lance-2
```

**What to do:**
```
1. ONLY change Cargo.toml dep versions:
   crates/lance-graph/Cargo.toml
   crates/lance-graph-catalog/Cargo.toml
   crates/lance-graph-python/Cargo.toml
   
   arrow:      current → 57
   datafusion: current → 51
   lance:      current → 2.0

2. cargo update (regenerate Cargo.lock)

3. Fix any API breakages from dep bumps:
   - arrow 57: check RecordBatch API changes
   - datafusion 51: check SessionContext, LogicalPlan changes
   - lance 2: check table API, write params

4. cargo test --workspace (exclude python crate)
   Fix ALL test failures.

5. Check upstream CI requirements:
   - cargo fmt --check
   - cargo clippy -- -D warnings
   - cargo test

6. Push, open PR to lance-format/lance-graph
```

PR title: `feat: bump arrow 57, datafusion 51, lance 2`
PR body:
```
Align dependency matrix:
  arrow      → 57
  datafusion → 51
  lance      → 2.0

All tests pass. No API breakages.
Follows up on closed #146 (split into focused PRs).
```

**Exit gate:** Clean PR with ONLY dep bumps. CI green. No extra files.

---

## SESSION 3: PR B — graph/spo/ Module

```bash
git checkout main
git pull upstream main  # get PR A merged first, or branch from main
git checkout -b feat/spo-triple-store
```

**What to do:**
```
1. Add ONLY the graph/spo/ module:
   crates/lance-graph/src/graph/mod.rs
   crates/lance-graph/src/graph/fingerprint.rs
   crates/lance-graph/src/graph/sparse.rs
   crates/lance-graph/src/graph/spo/mod.rs
   crates/lance-graph/src/graph/spo/builder.rs
   crates/lance-graph/src/graph/spo/merkle.rs
   crates/lance-graph/src/graph/spo/semiring.rs
   crates/lance-graph/src/graph/spo/store.rs
   crates/lance-graph/src/graph/spo/truth.rs

2. Add test:
   crates/lance-graph/tests/spo_ground_truth.rs

3. Add `pub mod graph;` to lib.rs

4. REMOVE anything not relevant to upstream:
   - No SPARE_PARTS_SUMMARY.md
   - No #[track_caller] error macros (separate PR if wanted)
   - No ladybug-rs specific imports
   - No references to BindSpace, CogRedis, etc
   
5. Make sure graph/spo/ is SELF-CONTAINED:
   - Uses lance-graph's own error types (not ladybug's QueryError)
   - Uses standard blake3 crate (add to Cargo.toml)
   - No dependency on rustynum or ladybug-rs
   
6. Clean the code for upstream standards:
   - cargo fmt
   - cargo clippy -- -D warnings
   - Doc comments on all pub types and methods
   - Examples in doc comments where useful

7. cargo test (including spo_ground_truth.rs)

8. Push, open PR
```

PR title: `feat(graph): add SPO triple store with Merkle integrity, TruthGate, and semiring traversal`
PR body:
```
Add a content-addressable SPO (Subject-Predicate-Object) triple store:

- **SpoStore**: insert, query_forward, query_reverse, query_relation
- **SpoMerkle**: Blake3-based integrity with MerkleEpoch and inclusion proofs
- **TruthGate**: NARS-inspired confidence gating (MinFreq/MinConf/MinBoth)
- **SpoSemiring**: Algebraic traversal operations for graph algorithms
- **SpoBuilder**: Builder pattern for constructing stores
- **Fingerprint**: 16384-bit binary fingerprint with Hamming operations
- **SparseContainer**: Memory-efficient sparse vector storage

Ground truth test included (357 lines).

This enables knowledge-graph style operations on LanceDB with 
content-addressed nodes and confidence-weighted edges.
```

**Exit gate:** Clean PR, no ladybug-rs deps, all tests pass, upstream CI green.

---

## SESSION 4: PR C — BlasGraph Semiring Algebra (FROM holograph)

```bash
git checkout main
git pull upstream main
git checkout -b feat/blasgraph-semiring-algebra
```

**What to do:**
```
1. Port from holograph/src/graphblas/ to lance-graph:
   
   Create: crates/lance-graph/src/graph/blasgraph/
     mod.rs        ← from holograph graphblas/mod.rs (94 lines)
     semiring.rs   ← from holograph graphblas/semiring.rs (535 lines)
     matrix.rs     ← from holograph graphblas/matrix.rs (596 lines)
     vector.rs     ← from holograph graphblas/vector.rs (506 lines)
     ops.rs        ← from holograph graphblas/ops.rs (717 lines)
     sparse.rs     ← from holograph graphblas/sparse.rs (546 lines)
     types.rs      ← from holograph graphblas/types.rs (330 lines)
     descriptor.rs ← from holograph graphblas/descriptor.rs (186 lines)

2. CLEAN for upstream:
   - Remove holograph-specific imports
   - Remove any reference to ladybug-rs types
   - Use lance-graph error types
   - All pub types and methods get doc comments
   - cargo fmt + clippy clean
   
3. The 7 semirings to include:
   - XOR Bundle (bind/superpose)
   - Bind First (key-value association)
   - Hamming Min (nearest neighbor)
   - Similarity Max (most similar)
   - Resonance with threshold (sigma-gated)
   - Boolean (standard graph traversal)
   - XOR Field (algebraic field operations)
   
4. Matrix operations:
   - mxm (matrix × matrix — graph composition)
   - mxv (matrix × vector — graph query)
   - vxm (vector × matrix — reverse query)
   - element-wise add/mult

5. Write tests:
   - One test per semiring showing expected behavior
   - Matrix multiplication with at least 2 semirings
   - Sparse matrix efficiency test

6. Update graph/mod.rs: pub mod blasgraph;

7. Push, open PR
```

PR title: `feat(graph): add BlasGraph semiring algebra — 7 semirings, sparse matrix ops`
PR body:
```
Port GraphBLAS-inspired sparse matrix algebra to lance-graph.

7 semiring algebras for different graph computation modes:
- XOR Bundle, Bind First, Hamming Min, Similarity Max
- Resonance (threshold-gated), Boolean, XOR Field

Matrix operations: mxm, mxv, vxm, element-wise.
CSR sparse format for memory-efficient large graphs.

This enables algebraic graph algorithms (PageRank, community detection,
shortest path) as matrix operations on LanceDB-backed graphs,
replacing Pregel-style message passing with linear algebra.

Based on the RedisGraph BlasGraph approach, adapted for LanceDB 
and binary Hamming distance vectors.
```

**Exit gate:** Clean PR, holograph code adapted, all tests pass, no ladybug-rs deps.

---

## SUMMARY

```
SESSION   PR     WHAT                          LINES   DEPENDS ON
1         -      Sync fork, close #146         0       nothing
2         A      Dep bumps only                ~50     session 1
3         B      graph/spo/ module             ~1600   session 1 (or session 2 merged)
4         C      BlasGraph semiring algebra    ~3500   session 1 (or session 2 merged)
```

Sessions 3 and 4 can target main even if PR A isn't merged yet.
PR B and C are independent of each other.

**What beinan and the lance-format team get:**
- Arrow 57 / DataFusion 51 / Lance 2 alignment (PR A)
- SPO triple store with Merkle integrity and NARS confidence (PR B)
- BlasGraph algebra that no other graph database has in Rust (PR C)

**What we get:**
- Clean upstream relationship (not a kitchen sink PR)
- Our SPO and BlasGraph contributions in the official repo
- Upstream CI validates our code
- Community review catches bugs we missed
