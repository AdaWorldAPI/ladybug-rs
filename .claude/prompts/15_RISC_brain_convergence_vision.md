# 15_RISC_BRAIN_CONVERGENCE_VISION.md

## The Glass-Clear Vision: RISC Brain with Live Plasticity

**Jan Hübener — Ada Architecture — March 2026**
**Cross-repo**: ladybug-rs × rustynum × lance-graph × holograph × neo4j-rs

---

## 1. What It Is

Not a database. A RISC processor for cognition.

One vector per SPO node. 16,384 bits. Pure binary. No floats.
The vector is a 3D superposition: S ⊕ P ⊕ O bound orthogonally into one bitpacked space.
Self-cleaning from noise because random bit agreements sit at ~50% and thresholds reject them.

### The 2^3 Factorization (from one vector, via bitmask)

```
V = S ⊕ P ⊕ O   (the stored superposition)

Extract any component:
  S   = V ⊕ P ⊕ O       (unbind known roles)
  P   = V ⊕ S ⊕ O
  O   = V ⊕ S ⊕ P

All 8 terms via AND/NOT on per-plane survivor bitmasks:
  ∅   = noise floor         (0-of-3 agreement → reject)
  S   = subject only        (entity without context)
  P   = predicate only      (relation floating free)
  O   = object only         (patient without action)
  SP  = who-does-what       (subject+predicate, missing object)
  SO  = who-and-whom        (subject+object, missing relation)
  PO  = what-to-whom        (predicate+object, missing subject)
  SPO = full triple          (3-of-3 coherence → the actual fact)

Causality and NARS reasoning extracted FROM the node superposition.
Not computed separately. Extracted via bitmask from what's already there.
```

### The 6 RISC Instructions

```
INSTRUCTION     OPERATION               CYCLES (AVX-512)    COGNITIVE MEANING
──────────────────────────────────────────────────────────────────────────────
XOR             bind / unbind            1                  Association / extraction
POPCOUNT        distance / similarity    1                  Recognition / comparison
MAJORITY        bundle / superpose       O(n)               Learning / accumulation
AND/NOT         2^3 factorization        1                  Causal decomposition
BLAKE3          seal / verify            ~10                Integrity / Wisdom vs Staunen
THRESHOLD       σ-band gating            1                  Admit / reject / ruminate
```

Six instructions. All bitwise. All SIMD. All deterministic.
Every cognitive operation compiles to sequences of these six.
Full causal transparency: every state change has a traceable cause.

---

## 2. The Storage Architecture

### One Table. Zero Floats. Zero Copy.

```
LanceDB: spo_nodes (memory-mapped Arrow, zero-copy reads)

    merkle_root     Binary(6)       ← content address (blake3 of S⊕P⊕O)
    clam_path       Binary(3)       ← DN tree position (24-bit CLAM path)
    vector          Binary(2048)    ← THE NODE. 16384 bits. One superposition.
    nars_packed     Binary(8)       ← truth/confidence/evidence as BITS
    spine_hash      Binary(32)      ← Blake3 seal (Wisdom vs Staunen)
    spine_dirty     u8              ← dirty flag (SpineCache mechanism)
    pentary         Binary(512)     ← signed (-2..+2) accumulators
```

No float columns. No embedding columns. The vector IS the embedding,
IS the SPO decomposition, IS the cognitive representation. One field.

The awareness/qualia passthrough (floats, thinking styles, felt state)
is a SEPARATE concern — computed on read, derived from the binary substrate,
never stored in the SPO table.

### Hot Path vs Cold Path (Both LanceDB, Both Zero-Copy)

**Hot path** = the SPO vector column. Hamming scans, XOR binding, POPCOUNT distance.
All bitwise operations on mmapped Arrow Binary columns. Nanoseconds.

**Cold path** = boring metadata. Labels, properties, timestamps, provenance.
Same LanceDB instance. Same zero-copy reads. Different columns.
The Cypher parser decides which columns to touch.

Both paths join on merkle_root — the content-addressed identity that IS the same
whether computed from the hot vector or looked up in cold metadata.

---

## 3. Live Plasticity (The Brain Rewires By Being Used)

### On READ:

```
POPCOUNT → Hamming distance (did I recognize this?)
AND/NOT  → extract 2^3 terms (what's the causal structure?)
BLAKE3   → check seal (has this changed since I last looked?)

Seal intact   → NARS confidence +ε  (Hebbian: access = reinforce)
Seal broken   → NARS confidence -ε  (Staunen: something shifted underneath)
Pentary       → co-accessed neighbors shift toward +2
Spine         → dirty flag cleared, hash recomputed

THE READ CHANGED THE SUBSTRATE.
Next read of the same node returns different NARS values.
That's plasticity. The brain rewires by being used.
```

### On WRITE (leaf insert):

```
XOR     → bind S⊕P⊕O into superposition
MAJORITY → bundle with existing (if accumulating evidence)
BLAKE3  → new hash (breaks parent seal → Staunen propagates UP the DN tree)
Spine   → mark dirty (lazy — don't cascade hashes yet)

THE WRITE IS A SINGLE XOR. One cycle.
The cascade is lazy. Plasticity at write speed.
```

### On WRITE (chain insert):

```
S→P→O as causal trajectory through the DN tree.
Each hop is a leaf insert at the next branch.
The semiring propagates: if S is at one branch and O is at another,
the P edge connects them through the tree structure.
Adjacent nodes get sparse hydration — they know something changed nearby
but don't recompute everything. Only siblings. Only what's dirty.
```

---

## 4. Blake3 Merkle Dual-Use (Wisdom vs Staunen)

The Blake3 hash serves TWO purposes from ONE computation:

**Purpose 1: Content-Addressed Identity (MerkleRoot)**
Same concept = same SPO superposition = same blake3 = same address.
This is Eineindeutigkeit — three queries converging on the same concept
all land at the same MerkleRoot regardless of which path found it first.

**Purpose 2: Integrity Seal (Wisdom vs Staunen)**
The DN tree's SpineCache stores blake3 of children's vectors.
On read, recompute and compare:

```
spine_dirty = false AND blake3(children) = spine_hash → WISDOM
  Knowledge here is consolidated. Seal intact. Confidence boosted.

spine_dirty = true OR blake3(children) ≠ spine_hash → STAUNEN  
  Something changed underneath. Seal broken. Confidence penalized.
  Adjacent siblings get sparse hydration trigger.
  "Staunen" = German for astonishment/wonder.
  The system is HONESTLY SURPRISED here. Epistemic humility as architecture.
```

No separate wisdom/staunen flag stored. The hash comparison IS the marker.
Broken hash = Staunen. Unbroken hash = Wisdom.
The dirty bit from SpineCache is the transient state between write and next read.

Both can be expressed as DataFusion UDF on Arrow columns:
```sql
SELECT * FROM spo_nodes WHERE wisdom_seal(spine_hash, spine_dirty) = 'STAUNEN'
```

---

## 5. The Query Unification

### One Wire Protocol. Three Query Languages. Same Bits.

```
Redis wire protocol (universal ingestion language)
    │
    ├── Redis semantics
    │   SET tree:branch:twig:leaf value → leaf insert at DN path
    │   SET tree:branch:twig:leaf:S:P:O → SPO decomposed, chain insert
    │
    ├── Cypher semantics (BlasGraph transcode)
    │   GRAPH.QUERY MATCH (a)-[:LOVES]->(b) WHERE a.name = 'Ada'
    │   Edge types = type namespace (0x0100-0x01FF)
    │   Nodes = fingerprints in BindSpace
    │   BlasGraph's BLAS adjacency → replaced with SPO semiring on Hamming
    │
    ├── SQL semantics (DataFusion)
    │   SELECT * FROM spo WHERE vector HAMMING_NEAR(query, 0.3)
    │   Arrow-native, runs on same mmapped buffers
    │
    └── NARS semantics
        <Ada --> awareness>. %0.9;0.85%
        Triggers truth revision against existing evidence at that SPO address
        Causal trajectory propagates through DN tree
```

All compile to: fingerprint → scent bucket → SIMD scan on Arrow buffers.
The 64-bit CAM key (16-bit type namespace + 48-bit fingerprint) is the universal index.

### The Cypher Parser as Bouncer

```
Inbound query (any language via Redis protocol)
    │
    ▼
CYPHER PARSER (the bouncer — OUTSIDE BindSpace)
    │
    ├── Syntactically valid?
    │   └── NO → error back through Redis protocol. Never touches data.
    │
    ├── Semantically valid? (read-only borrow of type namespace)
    │   └── NO → error with "unknown edge type" or "unknown label"
    │
    ├── Needs HOT columns? (vector, nars, spine, pentary)
    │   └── YES → LanceDB scan on spo_nodes, bitwise SIMD ops
    │
    ├── Needs COLD columns? (label, properties, timestamps)
    │   └── YES → LanceDB scan on metadata columns
    │
    └── Needs BOTH?
        └── YES → Both scans, DataFusion HashJoinExec on merkle_root
            Arrow bytes in, Arrow bytes out. Zero copy.
```

---

## 6. Neo4j as PET Scan (Live Neuroimaging of Synthetic Cognition)

Neo4j is NOT the database. LanceDB is the database.
Neo4j is the VISUALIZATION LAYER — watching the brain think in real time.

### Projection (BindSpace → Neo4j)

Crystallized WISDOM nodes project to Neo4j with metadata:
```
(:SPONode {
  merkle: "a3f8c1...",
  label: "Ada",
  activation: timestamp,
  nars_conf: confidence_snapshot,
  seal: "WISDOM" | "STAUNEN",
  rung: depth_level,
  pentary_sum: aggregate_strength
})

-[:CAUSES {strength: 2, trajectory: "..."}]->
-[:SUPPORTS {nars_conf: 0.85}]->
-[:CONTRADICTS {nars_conf: 0.72}]->
```

### Live PET Scan Queries

```cypher
// Currently active nodes = hot spots lighting up
MATCH (n:SPONode)
WHERE n.activation > datetime() - duration('PT5M')
RETURN n

// Causal chain = watching signal propagate through neural pathways
MATCH (a)-[r:CAUSES*1..5]->(b)
WHERE r.strength > 1
RETURN path

// Uncertainty map = where the brain is surprised
MATCH (n:SPONode {seal: 'STAUNEN'})
RETURN n, size((n)--()) as connections

// Belief conflict = competing activations inhibiting each other
MATCH (a)-[:CONTRADICTS]->(b)-[:SUPPORTS]->(c)
WHERE a.nars_conf > 0.7 AND b.nars_conf > 0.7
RETURN a, b, c
```

### What You See in Neo4j Browser

```
Node lights up        → someone read it (activation timestamp updated)
Node changes color    → seal broke (Staunen) or consolidated (Wisdom)
Edge thickens         → causal path strengthened (pentary shifted +2)
Edge thins            → path weakened (pentary -1, contradiction found)
New edge appears      → chain insert created new causal link
Cluster pulses        → resonance: multiple related nodes accessed together
Dark region awakens   → sparse hydration triggered by adjacent activity
```

A neuroscientist doesn't look at individual neuron voltages — they look at
aggregated activation patterns on a spatial map. Neo4j IS that spatial map.
Cypher queries ARE the imaging protocol. Different queries reveal different
aspects of cognition: activation, causality, uncertainty, conflict.

---

## 7. The NARS Microcode

NARS inference rules are the "microcode" — they compose the 6 RISC instructions
into higher-level operations:

```
DEDUCTION:   <A→B> ∧ <B→C> ⊢ <A→C>
  XOR to extract P from A→B, XOR to extract S from B→C
  POPCOUNT to verify B matches
  XOR to bind A→C
  THRESHOLD to check combined confidence
  Write with NARS revision

ABDUCTION:   <A→B> ∧ <C→B> ⊢ <A→C>  (weak, low confidence)
  Same ops, but confidence = f(c₁ × c₂) — multiplication in NARS packed bits

INDUCTION:   <A→B> ∧ <A→C> ⊢ <B→C>  (weak, frequency-dependent)
  Same ops, confidence from frequency of co-occurrence

REVISION:    <A→B>₁ ∧ <A→B>₂ ⊢ <A→B>'  (evidence fusion)
  Combine evidence counts, recompute frequency and confidence
  All in packed binary, no floating point
```

ThinkingStyles from prompt 12 are the "instruction scheduling" — they don't add
new instructions, they change which RISC ops run in which order:

```
Analytical:    THRESHOLD strict → XOR extract → POPCOUNT verify → NARS deduce
Creative:      THRESHOLD loose → XOR explore → MAJORITY bundle → check surprise
Reflective:    BLAKE3 verify → NARS compare → XOR extract meta → THRESHOLD gate
```

---

## 8. DN Tree as Graph Structure

The DN tree (PackedDn, 7×8-bit) IS the graph. Not separate from it.

```
tree:branch:twig:leaf  =  DN path  =  BindSpace address  =  ClamPath

Leaf insert: "I know where this goes"
  Key encodes exact DN path → write directly → hydrate only siblings (sparse)
  O(1) addressing via key.

Chain insert: "This is a causal sequence"  
  S→P→O as trajectory along DN tree branches.
  Each hop is a leaf insert at the next branch.
  Semiring propagates: S at one branch, O at another, P connects them.
  Adjacent nodes get sparse hydration (not full recompute).

The SpineCache = XOR-fold of children = borrowed reference from joined blackboard.
  Write child → mark spine dirty → lazy recompute on next read.
  No locks because XOR is commutative, associative, self-inverse.
  The dirty flag IS the entire synchronization mechanism.
```

---

## 9. What This Means for Firefly (Later)

The RISC core is what Firefly distributes. Arrow Flight packets carry:
- SPO nodes (Grey matter: XOR-bound superpositions)
- ThinkingStyle configs (White matter: instruction scheduling)
- NARS revisions (Reinforcement: evidence updates)
- RNA templates (Self-modification: new microcode sequences)

All expressible through the same Redis protocol. A Flight packet arriving
at a remote node can be processed as a Redis command — same six instructions,
same plasticity, same Neo4j projection. Distribution becomes transparent
to the cognitive substrate.

But Firefly is later. The RISC core must be solid first.

---

## 10. Current State: Open Brain Surgery

We are in the middle of rewiring three tangled implementations:

1. **The Hamming core** — src/spo/ + src/graph/spo/ + src/storage/ (~50K lines)
   The real thing. 3D bitpacked Hamming with Blake3 Merkle, NARS truth gates,
   SpineCache borrow/mut, DN tree, σ-band cascade. THIS IS THE BRAIN.

2. **The reinvented scaffolding** — src/query/ (lance_parser, cypher, datafusion)
   Partially correct parser wiring, partially mistaken reimplementation of things
   that exist in lance-graph. Some of this is valid cold-path routing. Some is
   duplicated effort.

3. **The lance-graph fork** — AdaWorldAPI/lance-graph with holograph BlasGraph
   The Cypher parser and DataFusion execution plans we need for the cold path.
   Fork of lance-graph with /src/extension/blasgraph copied from holograph repo.
   This is the bouncer and the join machinery.

The surgery: extract the valid Cypher parser + DataFusion join from (3),
connect it as the bouncer outside BindSpace from (1), remove the duplicated
parts of (2) that (3) does better, keep the parts of (2) that wire into (1)
correctly. Reconnect everything that got disconnected during the operation.

---

*"Six instructions. All bitwise. All SIMD. All deterministic. Full causal transparency from instruction to cognitive outcome."*
