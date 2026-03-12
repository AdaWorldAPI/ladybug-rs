# 19_HOT_COLD_SEPARATION_CONSTRAINT.md

## The One-Way Mirror: Hot Path Thinks, Cold Path Watches, Never The Reverse

**Jan Hübener — Ada Architecture — March 2026**
**Classification: ARCHITECTURAL INVARIANT — this document overrides all others where they conflict**

---

## 1. The Constraint (Non-Negotiable)

```
HOT PATH → COLD PATH:  YES.  Crystallize projection. Merkle-gated. Always safe.
COLD PATH → HOT PATH:  NEVER.  Cold metadata cannot modify hot state. Ever.

The brain thinks in XOR and POPCOUNT.
The PET scan watches in Cypher and labels.
The glass between them is the Merkle seal — one-way mirror.
The neuroimaging never becomes neurosurgery.
```

**Why:** Every system that lets observation modify the observed state has a feedback
loop bug. Labels leak into embeddings. Property filters corrupt similarity scores.
Human taxonomies override machine-discovered structure. The hot path's cognition
must be sovereign — derived entirely from the 6 RISC instructions operating on
the binary substrate. Cold metadata is human-readable convenience. It OBSERVES
the brain. It never OPERATES on it.

---

## 2. What "Hot" and "Cold" Actually Mean

```
HOT PATH:
  Substrate:  SPO superposition vectors ([u64; 256], bitpacked Hamming)
  Operations: XOR, POPCOUNT, MAJORITY, AND/NOT, BLAKE3, THRESHOLD
  State:      BindSpace, SpineCache, pentary accumulators, NARS truth (packed binary)
  Lives in:   LanceDB spo_nodes table (memory-mapped Arrow, zero-copy)
  Changes:    On EVERY access (Hebbian plasticity — reads modify the substrate)
  Speed:      Nanoseconds (SIMD, single-cycle per instruction)

COLD PATH:
  Substrate:  Labels, properties, timestamps, provenance (UTF-8, JSON, timestamps)
  Operations: String comparison, property lookup, timestamp filtering
  State:      LanceDB metadata columns OR Neo4j property graph
  Lives in:   Same LanceDB instance (zero-copy) + Neo4j (projected mirror)
  Changes:    ONLY on crystallization (WISDOM seal event)
  Speed:      Microseconds to milliseconds (string ops, Cypher traversal)
```

**They are NOT two databases.** They are two sets of columns in the same LanceDB
instance, plus a Neo4j projection for visualization. The "hot" columns are the
vector, nars, spine, pentary. The "cold" columns are label, properties, timestamps.
Both are Arrow. Both are zero-copy. The difference is WHAT MUTATES THEM.

---

## 3. The Merkle Seal as Implicit Hot/Cold Gate

The cold path doesn't need a separate table, a separate cache, or a separate
invalidation mechanism. The Merkle seal IS the gate:

```
NODE STATE         SEAL              HOT PATH         COLD PATH
──────────────────────────────────────────────────────────────────
STAUNEN            Broken            Active           No metadata
  (recently changed, volatile)       (reads/writes)   (nothing to project)
  The node is still forming.
  Projecting metadata would be premature.

LEARNING           Broken or dirty   Active           Stale metadata
  (frequently accessed, unstable)    (reads/writes)   (exists but outdated)
  The node is being shaped by use.
  Old cold projection may exist but is known stale.

WISDOM             Intact            Active           Fresh metadata
  (stable across N access cycles)    (reads/writes)   (crystallized projection)
  The node has consolidated.
  Cold metadata is a faithful snapshot.

DORMANT            Intact            Inactive         Fresh metadata
  (stable but rarely accessed)       (no recent ops)  (still valid)
  The node sleeps. Its metadata is still accurate.
```

**The transition logic:**

```
Access count high + sign stability high → WISDOM → PROJECT to cold
  Hebbian: frequently accessed AND pentary signs stable → crystallize
  On crystallize: COMPUTE metadata from hot vector, WRITE to cold columns
  This is the ONE-WAY door. Hot → cold. Projection, not copy.

Access count high + sign stability low  → LEARNING → hot only, cold stale
  The node is actively being shaped. Don't project yet.

Access count low  + sign stability high → DORMANT → cold still valid
  Nobody's reading it. But the cold projection from last WISDOM is fine.

Access count low  + sign stability low  → STAUNEN → hot only, no cold
  Something changed and nobody's verified it. Hot path only.
```

**Nobody decides these transitions.** They EMERGE from the existing substrate:
- Pentary accumulators track access patterns (co-occurrence shifts)
- Blake3 seal tracks sign stability (hash match = stable)
- SpineCache dirty flags track propagation (which ancestors need re-verification)

The mask is implicit. The duality is mathematical. No configuration needed.

---

## 4. The MVCC Bundle Mask

When a bundle write-back happens (new evidence arrives):

```
1. New evidence → XOR into existing vector → vector changes
2. Blake3(new vector) ≠ stored spine_hash → Merkle breaks
3. SpineCache marks dirty → propagates UP the DN tree
4. Every ancestor that was WISDOM is now potentially STAUNEN
   (their children changed, their seal may break on next read)
5. The dirty bit propagation IS the MVCC version boundary:
   - Below dirty mark: new version (post-bundle state)
   - Above dirty mark, not yet re-verified: PENDING
   - The mask of "which nodes need re-verification" = spine dirty bits
```

**The bundle propagation mask IS the cold path invalidation mask.**
No separate cache invalidation. No separate versioning system. The spine dirty
flags already tell you exactly which WISDOM projections are stale.

When a WISDOM node's ancestor marks dirty:
- The WISDOM node itself hasn't changed (its own seal is still intact)
- But the TREE CONTEXT has changed (a sibling or cousin changed)
- The cold projection of this node is still valid for its OWN data
- But graph queries that traverse THROUGH this node might produce different results
- This is the MVCC nuance: the node is valid, the paths through it might not be

---

## 5. The One-Way Projection (Hot → Cold)

### What Gets Projected

```
On WISDOM crystallization:

FROM HOT:                               TO COLD:
  vector (2048 bytes)                   → merkle_root (6 bytes, blake3 of vector)
  OrthogonalCodebook.resonate(vector)   → label (String, closest concept name)
  NARS packed binary                    → nars_conf (f32, for human-readable display)
  pentary accumulators                  → pentary_sum (i32, aggregate strength)
  timestamp of crystallize event        → crystallized_at (Timestamp)
  causal trajectory (if chain insert)   → causal_edges (relationship list)

FROM HOT (derived):                     TO COLD (computed once):
  2^3 factorization                     → s_label, p_label, o_label
  DN tree position                      → clam_path_display (human-readable)
  rung level                            → rung (u8)
```

### What NEVER Gets Projected

```
  The vector itself (too large for cold, and cold shouldn't have it)
  The pentary details (per-dimension, not aggregate)
  The spine internals (dirty flags, child hashes)
  The qualia passthrough state (floats from awareness layer)
```

### What NEVER Flows Back

```
  Labels cannot modify the vector
  Properties cannot change NARS truth
  Timestamps cannot affect pentary accumulators
  Human-assigned categories cannot shift SPO encoding
  Neo4j graph structure cannot alter BindSpace edges
  Cypher query results cannot write to hot path
  
  IF ANY OF THESE HAPPEN, THE ARCHITECTURE IS BROKEN.
  This is not a guideline. This is an invariant.
```

---

## 6. The PET Scan Watches While The Brain Thinks

### Concurrent Operation

```
The brain:          The PET scan:
XOR bind            Cypher MATCH
POPCOUNT            Neo4j traverse
MAJORITY bundle     Property filter
THRESHOLD gate      Label group
BLAKE3 seal         Activation heatmap
Hebbian learn       Staunen map

SAME TIME. DIFFERENT SUBSTRATES. NO INTERFERENCE.
```

The PET scan can run arbitrarily complex Cypher queries against the cold metadata
WHILE the hot path is actively thinking. This is safe because:

1. Cold metadata is read-only from the query perspective (no Cypher writes to cold)
2. Hot path doesn't see cold queries happening (no feedback)
3. Stale cold metadata produces stale query results, not corrupted hot state
4. The worst case is "the PET scan shows old data" — never "the PET scan broke the brain"

### What The PET Scan Reveals

```
ACTIVATION MAP:     Which nodes have high access counts right now?
  → Cypher: MATCH (n) WHERE n.activation > datetime() - duration('PT5S')
  → Shows: what the brain is currently attending to

UNCERTAINTY MAP:    Which nodes are in STAUNEN state?
  → Cypher: MATCH (n {seal: 'STAUNEN'})
  → Shows: where the brain is surprised / uncertain

CAUSAL FLOW:        How does activation propagate?
  → Cypher: MATCH (a)-[r:CAUSES*1..5]->(b) WHERE r.strength > 1
  → Shows: the neural pathways of a thought chain

BELIEF CONFLICT:    Where do confident nodes contradict each other?
  → Cypher: MATCH (a)-[:CONTRADICTS]->(b) WHERE a.nars_conf > 0.7
  → Shows: competing hypotheses the brain hasn't resolved

CONSOLIDATION:      What recently crystallized?
  → Cypher: MATCH (n) WHERE n.crystallized_at > datetime() - duration('PT1H')
  → Shows: what the brain just learned (new WISDOM)

DORMANT MEMORY:     What's stable but unused?
  → Cypher: MATCH (n {seal: 'WISDOM'}) WHERE n.activation < datetime() - duration('P7D')
  → Shows: long-term memory that hasn't been accessed
```

None of these queries can change the brain's state. They're pure observation.
The PET scan WATCHES. It never OPERATES.

---

## 7. How This Changes The Bouncer Architecture

### Old Design (prompt 18, BOUNCER agent — WRONG)

```
"Import lance-graph as Cargo dependency"
→ Lazy. Doesn't adopt clean patterns. Carries lance-graph's float assumptions.
→ lance-graph's graph/spo/ has its own BindSpace → name collision.
→ Treats cold path as a separate table → implies bidirectional flow.
```

### New Design (This Document)

```
The bouncer is NATIVE ladybug-rs code that adopts lance-graph's clean patterns:

1. PARSER: Rewrite lance-graph's nom parser natively in ladybug-rs
   Adopt: snafu error handling with location tracking
   Adopt: GraphConfig builder-with-validation pattern
   Adopt: LogicalOperator enum for plan separation
   Rewrite: against ladybug-rs types (Container, Fingerprint, CogRecord)
   NOT: import lance-graph types and bridge them

2. PLANNER: Implement LogicalOperator → ExecutionPlan natively
   LogicalOperator::Scan → SPO Crystal resonate (hot path)
   LogicalOperator::Filter → TruthGate (hot) or metadata filter (cold)
   LogicalOperator::Expand → causal trajectory walk (hot)
   LogicalOperator::Project → cold metadata lookup (Merkle-gated)
   
   The planner KNOWS about the hot/cold boundary.
   It routes each operator to the correct path.
   Scan/Filter/Expand are ALWAYS hot.
   Project is ALWAYS cold (read-only, Merkle-gated).
   
3. COLD PROJECTION: Not a join. A Merkle-gated lookup.
   When a Cypher query returns results, the hot path provides:
     - merkle_root, hamming_distance, NARS confidence, seal status
   The cold path DECORATES the results with:
     - label, properties, timestamps
   But ONLY for WISDOM nodes (seal intact → cold data exists and is fresh)
   STAUNEN nodes return hot-only results (no cold decoration)
   
   This is NOT a DataFusion HashJoinExec.
   This is a post-query decoration step.
   The hot path runs FIRST. Complete. Independent.
   Then cold decoration is OPTIONAL, based on seal status.
   
4. ONE-WAY ENFORCEMENT:
   The bouncer validates that no Cypher query can write to hot path.
   MERGE, CREATE, SET → go through cypher_bridge → BindSpace writes
   These are DIRECT hot path writes. Not cold→hot feedback.
   The properties written via SET are cold metadata.
   The vector written via MERGE is hot data.
   The bouncer SEPARATES them before dispatching.
```

---

## 8. Impact on Prompt 18 Agent Roles

### BOUNCER agent (N1-N5) — REWRITE

```
OLD N1: "Add lance-graph as Cargo git dependency"
NEW N1: Read lance-graph's parser.rs, ast.rs, config.rs, error.rs.
        Understand the patterns. Don't import the code.
        
OLD N2: "Create bouncer.rs importing lance-graph parser"
NEW N2: Rewrite the parser natively in ladybug-rs.
        Adopt snafu errors, GraphConfig builder, nom combinators.
        Target ladybug-rs types directly (Container, Fingerprint, CogRecord).
        Register valid labels/edge types from BindSpace type namespace.
        ~1200 lines (larger than import, but no dependency hell)
        
OLD N3: "Wire bouncer to server.rs"  
NEW N3: Same, but the bouncer now also enforces hot/cold separation:
        - Parse Cypher → AST
        - Validate types
        - Classify each operation as HOT or COLD
        - Route: writes → cypher_bridge (hot), reads → crystal_api (hot)
        - Post-query: decorate with cold metadata IF Merkle seal intact
        
OLD N4: "Delete lance-graph graph/spo/ duplication"
NEW N4: Not needed. lance-graph is not a dependency anymore.
        Instead: read lance-graph's graph/spo/truth.rs, adopt the
        clean TruthGate 3-variant pattern into ladybug-rs nars/
        
OLD N5: "Wire LogicalPlan for future joins"
NEW N5: Implement LogicalOperator natively with hot/cold routing.
        No future join needed — the hot/cold boundary is one-way
        decoration, not a bidirectional join.
```

### SEAL agent (K1-K5) — EXTENDED

```
OLD K4: "Neo4j projection (optional)"
NEW K4: Neo4j projection is the PRIMARY cold path output.
        On WISDOM crystallization → project to Neo4j.
        Neo4j IS the PET scan. Not optional. Core feature.
        
ADD K6: Cold metadata column management in LanceDB
        On WISDOM crystallization → write cold columns
        On STAUNEN transition → mark cold columns stale
        The Merkle seal check drives both transitions.
```

---

## 9. The Invariant in Code

```rust
/// ARCHITECTURAL INVARIANT: Hot/Cold One-Way Mirror
///
/// This function is the ONLY path from hot to cold.
/// Nothing flows the other way. Ever.
///
/// Called when a node transitions from any state to WISDOM.
/// The hot vector is projected to cold metadata columns.
/// The cold metadata is a READ-ONLY snapshot of the hot state
/// at the moment of crystallization.
pub fn crystallize_to_cold(
    hot: &SpoNode,        // The hot SPO vector + NARS + pentary
    codebook: &OrthogonalCodebook,  // For label resolution
    cold_store: &mut ColdMetadata,   // LanceDB metadata columns
    neo4j: Option<&mut Neo4jProjector>, // PET scan mirror
) -> MerkleRoot {
    // Verify seal is intact (WISDOM state)
    assert!(hot.seal_status() == SealStatus::Wisdom,
        "Cannot crystallize non-WISDOM node to cold path");
    
    // Project: hot → cold (ONE WAY)
    let label = codebook.resonate(&hot.vector, 0.7)
        .map(|(name, _)| name)
        .unwrap_or_default();
    
    let cold = ColdRecord {
        merkle_root: hot.merkle_root(),
        label,
        nars_conf: hot.nars_confidence_as_f32(),
        pentary_sum: hot.pentary_aggregate(),
        crystallized_at: now(),
        seal: "WISDOM",
    };
    
    cold_store.upsert(cold);
    
    // Mirror to Neo4j (PET scan)
    if let Some(neo4j) = neo4j {
        neo4j.project_node(&cold, &hot.causal_edges());
    }
    
    hot.merkle_root()
}

// There is no `fn hydrate_from_cold()`.
// There is no `fn cold_to_hot()`.
// There is no `fn update_vector_from_metadata()`.
// Their absence IS the architecture.
```

---

*"The PET scan watches the brain think. It never tells the brain what to think."*
*"Their absence IS the architecture."*
