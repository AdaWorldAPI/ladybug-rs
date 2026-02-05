# The 64×256 Cognitive Record

**Date**: 2026-02-05
**Status**: Proposal — the "pay the tax once, think in storage forever" design
**Premise**: 256 u64 words = 2,048 bytes = 32 cache lines. The tax is already paid. Every bit that isn't routing should be pre-computed cognitive state that never needs recalculating.

---

## 0. The Argument

Current state: the system stores a fingerprint (156 u64) and metadata (label, rung, parent, etc.) separately. Every query that needs NARS truth, SPO structure, adjacency, or scent recomputes or joins at read time. This work is done over and over.

The 64×256 design says: **compute once at write time, store the result in a parseable fixed layout, never compute it again.** The record is a materialized cognitive view. Every reader just does pointer arithmetic.

```
256 u64 = 2,048 bytes = 32 × 64-byte cache lines

Each 8-word group (64 bytes) = 1 cache line = 1 AVX-512 register

SIMD iterations for full-record Hamming distance:
  AVX-512: 256 / 8  = 32 iterations, zero remainder
  AVX2:    256 / 4  = 64 iterations, zero remainder
  NEON:    256 / 2  = 128 iterations, zero remainder
  Scalar:  256 / 1  = 256 iterations, zero remainder

Everything divides. Nothing is wasted.
```

---

## 1. The 32-Compartment Layout

Every compartment is exactly 8 u64 words = 64 bytes = 1 cache line = 1 AVX-512 register width.

An operation that needs only compartment C3 reads exactly one cache line. No more.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         256 × u64 = 2,048 BYTES                             │
│                         32 COMPARTMENTS × 64 BYTES                          │
├────┬─────────────────────────────────────────────────────────────────────────┤
│    │                                                                         │
│ C0 │  KEY + DN TREE + ROUTING                                     64 bytes  │
│    │  The (8+8+48+64+128) routing bits + DN T1-T8 + context_id              │
│    │                                                                         │
│ C1 │  ADJACENCY-OUT                                               64 bytes  │
│    │  512-bit bitvector: outgoing edge bucket presence                       │
│    │                                                                         │
│ C2 │  ADJACENCY-IN                                                64 bytes  │
│    │  512-bit bitvector: incoming edge bucket presence                       │
│    │                                                                         │
│ C3 │  VERB-TYPE MASK                                              64 bytes  │
│    │  512 bits: which of the 40+ verb types this node participates in       │
│    │  bit layout: [0..255]=outgoing verb slots, [256..511]=incoming         │
│    │                                                                         │
│ C4 │  SPO SKETCH                                                  64 bytes  │
│    │  Pre-computed: which Subject/Predicate/Object roles this node fills    │
│    │  XOR-compressed triple signatures for O(1) role query                  │
│    │                                                                         │
│ C5 │  NARS BELIEF STATE (integer-only)                            64 bytes  │
│    │  Fixed-point truth, evidence counts, inference bitmap                   │
│    │  No float anywhere in this compartment                                 │
│    │                                                                         │
│ C6 │  SEMANTIC KERNEL MEMO                                        64 bytes  │
│    │  Pre-computed: skill relevance bits, planner state hash,               │
│    │  zone assignment, thinking style weights (all fixed-point)             │
│    │                                                                         │
│ C7 │  SCENT + POPCOUNT + PARITY                                  64 bytes  │
│    │  Expanded scent (48 bytes) + popcount (u16) + ECC + CRC               │
│    │                                                                         │
│ C8 │  FINGERPRINT words  0-7                                     64 bytes  │
│ C9 │  FINGERPRINT words  8-15                                    64 bytes  │
│C10 │  FINGERPRINT words 16-23                                    64 bytes  │
│C11 │  FINGERPRINT words 24-31                                    64 bytes  │
│C12 │  FINGERPRINT words 32-39                                    64 bytes  │
│C13 │  FINGERPRINT words 40-47                                    64 bytes  │
│C14 │  FINGERPRINT words 48-55                                    64 bytes  │
│C15 │  FINGERPRINT words 56-63                                    64 bytes  │
│C16 │  FINGERPRINT words 64-71                                    64 bytes  │
│C17 │  FINGERPRINT words 72-79                                    64 bytes  │
│C18 │  FINGERPRINT words 80-87                                    64 bytes  │
│C19 │  FINGERPRINT words 88-95                                    64 bytes  │
│C20 │  FINGERPRINT words 96-103                                   64 bytes  │
│C21 │  FINGERPRINT words 104-111                                  64 bytes  │
│C22 │  FINGERPRINT words 112-119                                  64 bytes  │
│C23 │  FINGERPRINT words 120-127                                  64 bytes  │
│C24 │  FINGERPRINT words 128-135                                  64 bytes  │
│C25 │  FINGERPRINT words 136-143                                  64 bytes  │
│C26 │  FINGERPRINT words 144-151                                  64 bytes  │
│C27 │  FINGERPRINT words 152-159                                  64 bytes  │
│C28 │  FINGERPRINT words 160-167                                  64 bytes  │
│C29 │  FINGERPRINT words 168-175                                  64 bytes  │
│C30 │  FINGERPRINT words 176-183                                  64 bytes  │
│C31 │  FINGERPRINT words 184-191                                  64 bytes  │
│    │                                                                         │
│    │  FINGERPRINT TOTAL: 192 words = 12,288 bits                            │
│    │  (covers 10,000 semantic + 2,288 ECC/spare)                            │
│    │                                                                         │
├────┴─────────────────────────────────────────────────────────────────────────┤
│ STRUCTURE COMPARTMENTS (C0-C7):   8 × 64 =   512 bytes  (25%)              │
│ FINGERPRINT COMPARTMENTS (C8-C31): 24 × 64 = 1,536 bytes (75%)              │
│ TOTAL:                             32 × 64 = 2,048 bytes (100%)             │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Compartment Byte Layouts (All Integer, No Float)

### C0: Key + DN Tree + Routing (words 0-7) — Addr Is Primary Citizen

The key insight: **Addr(u16) is the primary key everywhere.** When you `DN.SET domain:tree:branch:twig:leaf`, the path hashes to an Addr. That Addr IS the record's identity — in BindSpace arrays, in Arrow columns, in DataFusion predicates, in Redis GET/SET. No secondary lookup. No foreign key. The record lives at `words[addr.prefix()][addr.slot()]` and that's it.

The DN tree uses **left-child right-sibling (LCRS)** encoding — three u16 pointers embedded in the record that give O(1) traversal in every direction:

```
word 0:  PRIMARY KEY (Addr) + ROUTING
  ┌─────────┬─────────┬───────────────────────────────────────────────┐
  │ [0:7]   │ [8:15]  │ [16:63]                                      │
  │ prefix  │ slot    │ group48: hash(T1:T2:T3:T4)                   │
  │ u8      │ u8      │ u48 frozen path hash                         │
  └─────────┴─────────┴───────────────────────────────────────────────┘
  ↑                     ↑
  └── THIS IS Addr ──┘  └── Sort prefix for DN subtree locality

  Addr = (prefix << 8) | slot.  That's the record's identity.
  group48 = hash of the frozen T1:T2:T3:T4 path.
  Sort on (T1,T2,T3,T4,group48) clusters the DN subtree.

word 1:  ROUTING KEY continued (the 64+128 bits start here)
  ┌───────────────────────────────────────────────────────────────────┐
  │ [0:63]                                                            │
  │ disambig: u64 (content hash — the "64" from 8+8+48+64+128)       │
  └───────────────────────────────────────────────────────────────────┘

word 2:  ROUTING KEY continued (the 128-bit extension)
  ┌───────────────────────────────────────────────────────────────────┐
  │ [0:63]                                                            │
  │ ext_key_lo: u64 (first half of 128-bit extension)                 │
  └───────────────────────────────────────────────────────────────────┘

word 3:  ROUTING KEY continued
  ┌───────────────────────────────────────────────────────────────────┐
  │ [0:63]                                                            │
  │ ext_key_hi: u64 (second half of 128-bit extension)                │
  └───────────────────────────────────────────────────────────────────┘

word 4:  DN TIER SLOTS (T1-T8: the path decomposed into tiers)
  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
  │ T1   │ T2   │ T3   │ T4   │ T5   │ T6   │ T7   │ T8   │
  │ u8   │ u8   │ u8   │ u8   │ u8   │ u8   │ u8   │ u8   │
  │domain│tree  │branch│twig  │leaf  │sub   │sub   │sub   │
  │FROZEN│FROZEN│FROZEN│FROZEN│ ephemeral ──────────────── │
  └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘

  "Ada:A:soul:identity:core" →
    T1=hash("Ada")&0xFF, T2=hash("A")&0xFF, T3=hash("soul")&0xFF,
    T4=hash("identity")&0xFF, T5=hash("core")&0xFF, T6-T8=0x00

word 5:  O(1) DN TREE POINTERS (LCRS encoding)
  ┌────────────────┬────────────────┬──────────────────┬──────────────────┐
  │ [0:15]         │ [16:31]        │ [32:47]          │ [48:63]          │
  │ parent_addr    │ first_child    │ next_sibling     │ prev_sibling     │
  │ u16 (Addr)     │ u16 (Addr)     │ u16 (Addr)       │ u16 (Addr)       │
  │ 0xFFFF = none  │ 0xFFFF = none  │ 0xFFFF = none    │ 0xFFFF = none    │
  └────────────────┴────────────────┴──────────────────┴──────────────────┘

  FOUR Addr pointers. Every tree operation is O(1):
  ┌─────────────────────────────────────────────────────────────────┐
  │ Operation               │ How                    │ Cost         │
  ├─────────────────────────┼────────────────────────┼──────────────┤
  │ Go to parent            │ words[5] & 0xFFFF      │ 1 read       │
  │ Go to first child       │ words[5] >> 16 & 0xFFFF│ 1 read       │
  │ Go to next sibling      │ words[5] >> 32 & 0xFFFF│ 1 read       │
  │ Go to prev sibling      │ words[5] >> 48 & 0xFFFF│ 1 read       │
  │ Enumerate children      │ first_child, then      │ O(k) for k   │
  │                         │ follow next_sibling    │ children     │
  │ Insert child            │ new.next_sib = parent  │ 3 writes     │
  │                         │   .first_child         │              │
  │                         │ old_first.prev_sib =   │              │
  │                         │   new                  │              │
  │                         │ parent.first_child =   │              │
  │                         │   new                  │              │
  │ Delete node             │ relink prev/next sib,  │ 4 writes     │
  │                         │ update parent if first │              │
  │ Count children          │ word 6 child_count     │ 1 read       │
  │ Is leaf?                │ first_child == 0xFFFF  │ 1 compare    │
  │ Is root?                │ parent_addr == 0xFFFF  │ 1 compare    │
  └─────────────────────────┴────────────────────────┴──────────────┘

  No CSR rebuild. No edge table scan. No PARENT_OF verb lookup.
  The tree IS the data. The pointers are IN the record.

word 6:  CONTEXT + HIERARCHY + COUNTS
  ┌──────────┬──────┬───────┬───────┬──────────┬──────────┬──────────┐
  │ [0:15]   │[16:23]│[24:31]│[32:39]│ [40:47]  │ [48:55]  │ [56:63]  │
  │context_id│ rung  │ sigma │ depth │child_cnt │ out_deg  │ in_deg   │
  │ u16      │ u8    │ u8    │ u8    │ u8       │ u8       │ u8       │
  └──────────┴──────┴───────┴───────┴──────────┴──────────┴──────────┘

  context_id: 0 = base BindSpace, >0 = exploration overlay
  rung: R0-R9 access control
  sigma: reasoning depth Σ
  depth: tree depth (0=root, max 255)
  child_cnt: number of direct children (saturates at 255)
  out_deg: outgoing graph edges (not tree; saturates at 255)
  in_deg: incoming graph edges (saturates at 255)

word 7:  TIMESTAMP + FLAGS + ACCESS
  ┌─────────────────────────────────────┬──────────┬──────────────────┐
  │ [0:39]                              │ [40:55]  │ [56:63]          │
  │ updated_at: u40 (~34.8 yr at 1ms)   │access_cnt│ entity_type_id   │
  │                                     │ u16      │ u8               │
  └─────────────────────────────────────┴──────────┴──────────────────┘
```

**Every field is an integer.** No float in C0. `Addr` at bytes [0:1] of word 0. `context_id` at bytes [0:1] of word 6. `parent_addr` at bytes [0:1] of word 5. All fixed offsets, all pointer arithmetic.

### The Self-Organizing DN Tree

When you execute `DN.SET domain:tree:branch:twig:leaf`, the record self-hydrates:

```
Step 1: Path → Addr
  "Ada:A:soul:identity" → dn_path_to_addr() → Addr(0xA3, 0x7B)
  That IS the record. No lookup table. No secondary index.

Step 2: Path → Tier slots (T1-T8)
  T1 = hash("Ada")    & 0xFF   (domain — frozen)
  T2 = hash("A")      & 0xFF   (tree — frozen)
  T3 = hash("soul")   & 0xFF   (branch — frozen)
  T4 = hash("identity") & 0xFF (twig — frozen)
  T5-T8 = 0x00                 (no deeper levels)
  → Written into word 4.

Step 3: Parent chain auto-links
  parent_path = "Ada:A:soul" → dn_path_to_addr() → Addr(0xB1, 0x22)
  IF parent doesn't exist → create it first (recursive).
  THEN:
    self.parent_addr   = parent_addr         // word 5 [0:15]
    self.next_sibling  = parent.first_child  // word 5 [32:47]
    IF parent.first_child != 0xFFFF:
      old_first.prev_sibling = self_addr     // update old first child
    parent.first_child = self_addr           // word 5 [16:31] of PARENT
    parent.child_count += 1                  // word 6 [40:47] of PARENT
    self.depth         = parent.depth + 1    // word 6 [32:39]

Step 4: Context resonance markers
  C5 (NARS) inherits parent's belief if no override.
  C4 (SPO) gets role bits from parent's predicate context.
  C1/C2 (adjacency) get bucket bits from PARENT_OF edge.
  C3 (verb mask) gets PARENT_OF bit set.

Step 5: Sort key for Arrow
  sort_key = (T1, T2, T3, T4, scent[0..2], context_id, disambig)
  → Adjacent records in Arrow share DN subtree → XOR delta compresses.

Result: ONE write, ZERO subsequent computation.
  Any reader does pointer arithmetic on the Addr.
  parent() = 1 read. first_child() = 1 read. siblings() = chase list.
  No CSR. No edge table. No HashMap. No verb lookup for tree structure.
```

### Addr Flows Through Everything

```
                    DN.SET Ada:A:soul:identity
                              │
                              ▼
                     dn_path_to_addr()
                              │
                              ▼
                       Addr(0xA3, 0x7B)        ◄── PRIMARY KEY
                              │
              ┌───────────────┼───────────────────┐
              ▼               ▼                   ▼
         BindSpace        CogRedis            DataFusion
         array[0xA3]      DN.GET 0xA3:7B      WHERE addr = 0xA37B
          [0x7B]          → reads record       → UInt16 filter
           │               at array[0xA3][0x7B]  → Arrow column[row]
           │               → returns CogRecord   → same bytes
           ▼                                      │
       CogRecord                                  ▼
       words[0..255]                          Arrow RecordBatch
           │                                  addr: UInt16
           ├── word 0: prefix=0xA3 slot=0x7B  structure: FixedSizeBinary(512)
           ├── word 5: parent=0xB122           fingerprint: FixedSizeBinary(1536)
           │           first_child=0xFFFF
           │           next_sib=0xC455
           │           prev_sib=0xFFFF
           └── ...

  Redis path:   DN.GET Ada:A:soul:identity → addr lookup → array read
  SQL path:     SELECT * FROM nodes WHERE addr = 0xA37B → same array
  Flight path:  DoGet(addr=0xA37B) → same record, zero-copy via Arrow
  GQL path:     MATCH (n) WHERE n.addr = 0xA37B → same record

  ONE record. ONE address. ALL paths converge.
```

### Redis Self-Organization Example

```
> DN.SET Ada:A:soul:identity <fingerprint>

  Internal:
    addr = dn_path_to_addr("Ada:A:soul:identity") = 0xA37B
    parent_addr = dn_path_to_addr("Ada:A:soul") = 0xB122
    // Auto-create parent chain if needed:
    //   "Ada" → Addr(0x8F, 0x01) [depth=0, root]
    //   "Ada:A" → Addr(0x92, 0x44) [depth=1, first_child of 0x8F01]
    //   "Ada:A:soul" → Addr(0xB1, 0x22) [depth=2, first_child of 0x9244]

    record[0xA37B].words[5] = pack(
      parent_addr: 0xB122,
      first_child: 0xFFFF,   // leaf: no children yet
      next_sibling: <whatever B122.first_child was>,
      prev_sibling: 0xFFFF   // new head of sibling list
    );
    record[0xB122].words[5].first_child = 0xA37B;  // parent points to new child
    record[0xA37B].words[4] = pack(T1,T2,T3,T4,0,0,0,0);  // tier slots
    record[0xA37B].words[6].depth = 3;
    record[0xA37B].words[6].context_id = 0;  // base context

  Now:
> DN.CHILDREN Ada:A:soul
    addr = 0xB122
    child = record[0xB122].first_child   // 0xA37B — O(1)
    while child != 0xFFFF:
      emit child
      child = record[child].next_sibling  // chase the list
    // No CSR. No edge scan. No rebuild.

> DN.PARENT Ada:A:soul:identity
    addr = 0xA37B
    parent = record[0xA37B].parent_addr   // 0xB122 — O(1)

> DN.SIBLINGS Ada:A:soul:identity
    addr = 0xA37B
    sib = record[0xA37B].next_sibling     // chase forward
    sib = record[0xA37B].prev_sibling     // chase backward
    // Doubly-linked: can go either direction

> DN.ANCESTORS Ada:A:soul:identity
    addr = 0xA37B
    while addr.parent_addr != 0xFFFF:
      emit addr.parent_addr
      addr = addr.parent_addr
    // 0xB122 → 0x9244 → 0x8F01 → done (3 reads, O(depth))

> DN.ISLEAF Ada:A:soul:identity
    record[0xA37B].first_child == 0xFFFF   // true — O(1)
```

### How DN Tree Pointers (C0) Relate to Graph Adjacency (C1-C3)

The DN tree (C0 word 5) and the graph adjacency (C1-C3) are **separate structures that serve different purposes**:

```
DN Tree (C0 word 5):
  Structural hierarchy. "Ada:A:soul" is parent of "Ada:A:soul:identity".
  This is the namespace, the organizational skeleton.
  Pointers: parent, first_child, next_sibling, prev_sibling.
  Operations: path lookup, subtree enumeration, depth queries.

Graph Adjacency (C1-C3):
  Semantic relationships. "Ada:A:soul:identity" CAUSES "Ada:A:behavior:response".
  This is the knowledge graph, the causal/associative web.
  Bitvectors: 512 bits OUT (C1), 512 bits IN (C2), 512 verb bits (C3).
  Operations: BFS, pattern matching, GQL MATCH, SPARQL BGP.

They coexist:
  A node at Addr 0xA37B has BOTH:
    - A tree position (parent=0xB122, children via LCRS)
    - Graph edges (CAUSES → bucket 0x1A3, INHIBITS → bucket 0x0FF)

  Tree operations use C0 word 5 (O(1) pointer chase).
  Graph operations use C1-C3 (O(1) bitvector ops).
  Neither interferes with the other.

  When you add a child via DN.SET, ONLY C0 word 5 is updated.
  When you add a CAUSES edge, ONLY C1/C3 are updated.
  The tree doesn't pollute the graph. The graph doesn't pollute the tree.
```

### C1: Adjacency-OUT (words 8-15)

```
512 bits. Bit b = 1 ⟹ ∃ outgoing edge to a node in bucket b.
Bucket assignment: bucket_id = (target.prefix ^ target.slot ^ (group48 >> 40) as u8) as u16 & 0x1FF
512 buckets. 65,536 addresses / 512 = ~128 addresses per bucket.
```

This IS a row of a GraphBLAS-style boolean sparse matrix. The matrix is M ∈ {0,1}^{N×512} where N = number of nodes. Operations:

| GraphBLAS | On C1 lane | Cost |
|-----------|-----------|------|
| `mxv(M, x)` BFS step | `OR` all C1s where `x` bit is set | 8 u64 OR per node |
| `mxm(M, M)` 2-hop | OR of ORs | Same, two passes |
| `eWiseAdd(M[i], M[j])` | `OR` two C1 bitvectors | 1 AVX-512 instr |
| `eWiseMult(M[i], M[j])` | `AND` two C1 bitvectors | 1 AVX-512 instr |
| `apply(complement, M[i])` | `NOT` the bitvector | 1 AVX-512 instr |
| Jaccard(i,j) | `popcount(AND) / popcount(OR)` | 3 AVX-512 instr |

### C2: Adjacency-IN (words 16-23)

Mirror of C1 for incoming edges. Same layout. Enables reverse traversal.

### C3: Verb-Type Mask (words 24-31)

```
512 bits total.

Bits [0..255]:   outgoing verb presence
  bit 0x00 = has outgoing CAUSES edge
  bit 0x01 = has outgoing BECOMES edge
  ...
  bit 0x27 = has outgoing PREV_SIBLING edge
  bits 0x28..0xFF = future verb slots (216 spare)

Bits [256..511]: incoming verb presence (same encoding, reversed direction)
```

With 40 verbs currently defined, bits 0-39 and 256-295 are active. 432 spare bits for growth.

**GQL pattern `MATCH (a)-[:CAUSES]->(b)`**:
1. Filter: `a.verb_out_bit(0x00) == 1` → one bit read from C3.
2. Traverse: `a.adjacency_out AND b.bucket_mask` → one AND on C1.
No edge table lookup for existence check. Edge table only for exact resolution.

**SPARQL `?x :CAUSES ?y . ?y :INHIBITS ?z`**:
1. All nodes with outgoing CAUSES: `C3 & CAUSES_MASK != 0`
2. Their adjacency-OUT ORed: bucket set for candidate ?y
3. Filter ?y for outgoing INHIBITS: `C3 & INHIBITS_MASK != 0`
4. Their adjacency-OUT ORed: bucket set for candidate ?z
All bitwise. All integer.

### C4: SPO Sketch (words 32-39)

Pre-computed at write time. Stores compressed triple information so that "does this node appear as Subject / Predicate / Object?" is a bit check.

```
word 32: SPO role bitmap + triple count
  ┌─────────┬─────────┬──────────┬────────────────────────────────────┐
  │ [0:7]   │ [8:15]  │ [16:31]  │ [32:63]                           │
  │ roles   │ count   │ reserved │ spo_signature: u32                  │
  │ u8 bits │ u8      │          │ XOR-fold of all triple encodings   │
  │ bit0=S  │         │          │                                    │
  │ bit1=P  │         │          │                                    │
  │ bit2=O  │         │          │                                    │
  └─────────┴─────────┴──────────┴────────────────────────────────────┘

words 33-34: Subject sketch (128 bits)
  XOR-fold of all subject fingerprints this node participates in.
  XOR distance from query subject → rough SPO filtering.

words 35-36: Predicate sketch (128 bits)
  XOR-fold of all predicate/verb fingerprints.

words 37-38: Object sketch (128 bits)
  XOR-fold of all object fingerprints.

word 39: Qualia summary (integer-quantized)
  ┌────────────────┬────────────────┬────────────────┬────────────────┐
  │ [0:15]         │ [16:31]        │ [32:47]        │ [48:63]        │
  │ arousal_q      │ valence_q      │ tension_q      │ depth_q        │
  │ u16 (0..65535) │ u16            │ u16            │ u16            │
  │ 0=calm         │ 0=negative     │ 0=relaxed      │ 0=surface      │
  │ 65535=excited  │ 65535=positive │ 65535=tense    │ 65535=profound │
  └────────────────┴────────────────┴────────────────┴────────────────┘
```

**All integer.** Qualia values quantized to u16. 16-bit precision is ~0.0015% resolution — more than adequate for affect channels.

### C5: NARS Belief State — Integer Only (words 40-47)

This is the critical design: NARS truth values stored as fixed-point integers so that basic storage and retrieval never touches float.

```
word 40: Truth value (fixed-point)
  ┌────────────────────────────────┬────────────────────────────────┐
  │ [0:31]                         │ [32:63]                        │
  │ frequency_fp: u32              │ confidence_fp: u32             │
  │ fixed-point Q16.16             │ fixed-point Q16.16             │
  │ 0x00000000 = 0.0               │ 0x00000000 = 0.0              │
  │ 0x00010000 = 1.0               │ 0x00010000 = 1.0              │
  │ 0x00008000 = 0.5               │                               │
  └────────────────────────────────┴────────────────────────────────┘

  Fixed-point NARS operations (all integer):
    deduction:  f = (f1 * f2) >> 16
                c = (f1 * f2 * c1 >> 16 * c2 >> 16) >> 16
    revision:   convert to evidence, sum, convert back (all integer)

word 41: Evidence counts (integer)
  ┌────────────────────────────────┬────────────────────────────────┐
  │ [0:31]                         │ [32:63]                        │
  │ evidence_positive: u32         │ evidence_negative: u32         │
  └────────────────────────────────┴────────────────────────────────┘

word 42: Expectation + derived
  ┌────────────────────────────────┬────────────────────────────────┐
  │ [0:31]                         │ [32:63]                        │
  │ expectation_fp: u32            │ horizon_k: u32                 │
  │ Q16.16: c*(f-0.5)+0.5         │ evidential horizon parameter   │
  └────────────────────────────────┴────────────────────────────────┘

word 43: Inference history (which rules have fired on this node)
  ┌───────────────────────────────────────────────────────────────────┐
  │ 64 bits: inference bitmap                                         │
  │ bit 0: deduction applied    bit 8: revision applied               │
  │ bit 1: induction applied    bit 9: comparison applied             │
  │ bit 2: abduction applied    bit 10: analogy applied               │
  │ bit 3: negation applied     bits 11-63: reserved                  │
  └───────────────────────────────────────────────────────────────────┘

words 44-47: NARS evidence ring buffer (4 most recent revisions)
  Each word: { source_hash: u32, delta_evidence_pos: i16, delta_evidence_neg: i16 }
  Enables "where did this belief come from?" without touching history tables.
```

**The key invariant: you can read C5, perform NARS deduction/revision/expectation, and write back C5 without any float conversion.** Float is only needed for human-readable output (`f as f32 / 65536.0`).

Fixed-point NARS deduction in pure integer:
```rust
/// Q16.16 fixed-point multiply: (a * b) >> 16
#[inline(always)]
fn fp_mul(a: u32, b: u32) -> u32 {
    ((a as u64 * b as u64) >> 16) as u32
}

/// NARS deduction: f = f1*f2, c = f1*f2*c1*c2
#[inline(always)]
fn nars_deduction_fp(f1: u32, c1: u32, f2: u32, c2: u32) -> (u32, u32) {
    let f = fp_mul(f1, f2);
    let c = fp_mul(fp_mul(fp_mul(f1, f2), c1), c2);
    (f, c)
}
```

No float. No division. Just multiply and shift. Works on WASM, embedded, FPGA.

### C6: Semantic Kernel Memo (words 48-55)

Pre-computed thinking state. Written once when the node is classified. Read many times by every query that touches this node.

```
word 48: Skill relevance bitvector
  ┌───────────────────────────────────────────────────────────────────┐
  │ 64 bits: which semantic kernel skills are relevant to this node   │
  │ bit 0: resonate    bit 8: infer       bit 16: counterfact        │
  │ bit 1: traverse    bit 9: observe     bit 17: learn              │
  │ bit 2: bind        bit 10: intervene  bit 18: reflect            │
  │ bit 3: unbind      bit 11: imagine    bit 19: analogize          │
  │ bit 4: bundle      bit 12: deduce     bit 20: crystallize        │
  │ bit 5: permute     bit 13: induce     bits 21-63: CAM categories │
  │ bit 6: search      bit 14: abduct                                │
  │ bit 7: traverse    bit 15: revise                                 │
  └───────────────────────────────────────────────────────────────────┘

word 49: Zone assignment + planner state
  ┌────────────────┬────────────────┬────────────────────────────────┐
  │ [0:7]          │ [8:15]         │ [16:63]                        │
  │ zone_type      │ zone_prefix    │ planner_state_hash: u48        │
  │ u8 (S/F/N)    │ u8             │ hash of last planner output    │
  └────────────────┴────────────────┴────────────────────────────────┘

word 50: Thinking style weights (8 × u8 = 64 bits)
  ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
  │ S0   │ S1   │ S2   │ S3   │ S4   │ S5   │ S6   │ S7   │
  │ u8   │ u8   │ u8   │ u8   │ u8   │ u8   │ u8   │ u8   │
  │analyt│holist│diver │conver│assoc │formal│intuit│criti │
  └──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘
  0 = not relevant, 255 = maximally relevant

words 51-52: CAM category activation (128 bits)
  Bit i = 1 ⟹ this node has been processed by CAM category i.
  16 categories × 8 sub-categories = 128 tracked.
  Saves recomputation: if bit is set, the result is already in the node.

words 53-55: Reserved for per-node memo cache
  E.g., last resonance result hash, collapse gate state, etc.
```

**All integer.** Thinking style weights are u8 (256 levels). Zone assignment is u8. Planner state is a hash. No float.

### C7: Scent + Popcount + Parity (words 56-63)

```
words 56-61: Expanded scent (48 bytes = 384 bits)
  The 5-byte scent from core/scent.rs, XOR-expanded to 48 bytes.
  Each original byte becomes 8 bytes via LFSR expansion.
  This makes scent_distance a 48-byte XOR + popcount = 6 u64 ops.
  On AVX-512: nearly one instruction (load + XOR + popcount + hadd).

word 62: Popcount + ECC
  ┌────────────────────────────────┬────────────────────────────────┐
  │ [0:15]                         │ [16:31]                        │
  │ fp_popcount: u16               │ ecc_parity: u16                │
  │ popcount(C8..C31)              │ Hamming(14, 12288) parity bits │
  ├────────────────────────────────┼────────────────────────────────┤
  │ [32:47]                        │ [48:63]                        │
  │ structure_crc: u16             │ reserved: u16                  │
  │ CRC16 of C0-C6                │                                │
  └────────────────────────────────┴────────────────────────────────┘

word 63: XOR fold parity of fingerprint (C8-C31)
  192 words XOR-folded to 1 word.
  Quick integrity check: fold(fingerprint) should equal this word.
```

### C8-C31: Fingerprint (192 words = 12,288 bits)

```
192 u64 words of pure semantic VSA content.

  Bits 0-9999:     semantic payload (10,000 bits)
  Bits 10000-12287: ECC + structured spare (2,288 bits)
    — Hamming ECC uses 14 bits
    — Remaining 2,274 bits: future extensions or
      additional semantic capacity

The fingerprint is contiguous across C8-C31.
SIMD scan of the full fingerprint: 24 AVX-512 iterations.
Early-exit partial scan on C8 alone (1 AVX-512 iter, 512 bits)
gives a 512/12288 = 4.2% sample — enough for L2 sketch filtering.
```

---

## 3. No Float Rule — How NARS, Qualia, and Weights Work

### The Fixed-Point Convention

All "fractional" values in C0-C7 use Q16.16 fixed-point encoding in u32, or u16 with implicit /65535 scaling.

```rust
/// Q16.16 fixed-point: 16 bits integer + 16 bits fraction
/// Range: 0.0 to 65535.99998 (but we use 0.0 to 1.0 for NARS)
/// Resolution: 1/65536 ≈ 0.0000153

// Encode:
fn to_fp(f: f32) -> u32 { (f * 65536.0) as u32 }

// Decode (only for display/output, never in hot path):
fn from_fp(fp: u32) -> f32 { fp as f32 / 65536.0 }

// Multiply (the only operation NARS needs):
fn fp_mul(a: u32, b: u32) -> u32 { ((a as u64 * b as u64) >> 16) as u32 }

// Add:
fn fp_add(a: u32, b: u32) -> u32 { a.saturating_add(b) }
```

**NARS truth functions in pure integer:**

```rust
// Deduction: f=f1*f2, c=f1*f2*c1*c2
fn deduction(f1: u32, c1: u32, f2: u32, c2: u32) -> (u32, u32) {
    let f = fp_mul(f1, f2);
    let c = fp_mul(fp_mul(fp_mul(f1, f2), c1), c2);
    (f, c)
}

// Induction: f=f2, c=f1*c1*c2
fn induction(f1: u32, c1: u32, f2: u32, c2: u32) -> (u32, u32) {
    let c = fp_mul(fp_mul(f1, c1), c2);
    (f2, c)
}

// Revision: convert to evidence, sum, convert back
fn revision(f1: u32, c1: u32, f2: u32, c2: u32) -> (u32, u32) {
    // w = k * c / (1 - c) where k = 1.0 = 0x10000
    // We store evidence directly in C5 word 41, so revision is just:
    // pos_total = pos1 + pos2, neg_total = neg1 + neg2
    // Then convert back to f, c.
    // All integer addition.
    todo!() // but all u32 arithmetic
}
```

### Why No Float Matters

1. **Deterministic**: integer arithmetic is bit-exact across platforms. A record written on x86 reads identically on ARM/WASM.
2. **SIMD-friendly**: integer XOR/AND/OR/popcount are the native SIMD operations. No lane-type conversion.
3. **No NaN/Inf surprises**: fixed-point saturates, never produces NaN.
4. **Embedded/WASM**: works on targets without FPU. The "any CPU" promise from bind_space.rs is preserved.
5. **Comparison is free**: `frequency_fp > 0x8000` is "frequency > 0.5". No float comparison.

Float conversion happens **only** at the boundary with human-readable output or external APIs. Inside the storage layer, everything is integer.

---

## 4. How the Record Serves Each Query Language

### SQL (DataFusion FilterExec)

```sql
WHERE context_id = 0 AND rung <= 5 AND entity_type_id = 3
```
Reads: C0 word 5 bytes [0:1] for context_id, byte [2] for rung, C0 word 6 bytes [6:7] for entity_type_id. Three integer comparisons on one cache line.

### GQL / openCypher

```cypher
MATCH (a)-[:CAUSES]->(b)-[:INHIBITS]->(c) RETURN c
```

1. Scan C3 for all nodes where bit 0x00 (CAUSES out) is set. **1 bit check per node.**
2. OR their C1 (adjacency-OUT) → candidate bucket set for ?b. **1 AVX-512 OR per node.**
3. Scan candidate ?b nodes for bit 0x08 (INHIBITS out) in C3. **1 bit check.**
4. OR their C1 → candidate buckets for ?c. **1 AVX-512 OR.**
5. Resolve exact edges from edge table for final result.

Steps 1-4 read only C1 + C3 = 128 bytes per node. Never touches the 1,536-byte fingerprint.

### SPARQL

```sparql
SELECT ?x WHERE { ?x rdf:type :Concept . ?x :CAUSES ?y . }
```

Same bitvector pattern as GQL. RDF type check → C4 (SPO sketch, roles bit 0=Subject, check predicate sketch against `rdf:type` hash). CAUSES check → C3 bit 0x00.

### NARS Inference

```
Given: A→B <0.9, 0.8>, B→C <0.7, 0.6>
Compute: A→C via deduction
```

1. Read A's C5 word 40: f1=0xE666, c1=0xCCCC (fixed-point for 0.9, 0.8)
2. Read B's C5 word 40: f2=0xB333, c2=0x9999 (0.7, 0.6)
3. Deduction: f = fp_mul(0xE666, 0xB333) = 0xA28F (≈0.634), c = ... (≈0.272)
4. Write result into C's C5. Update inference bitmap word 43.

All integer. No float. Reads and writes only C5 (64 bytes).

### Scented Bucket Search (HDR Cascade)

1. **L0 (C7)**: Read expanded scent (48 bytes). XOR + popcount vs query scent. 6 u64 ops. Kills ~90%.
2. **L1 (C7)**: Read popcount (2 bytes, word 62). `|row.popcount - query.popcount| > tolerance` → skip. 1 integer subtract. Kills ~50% of survivors.
3. **L2 (C8)**: Read first fingerprint compartment (64 bytes = 512 bits). Partial Hamming. Extrapolate. 1 AVX-512 iteration. Kills ~80% of L1 survivors.
4. **L3 (C8-C31)**: Full Hamming on 192 words = 24 AVX-512 iterations. Only ~0.2% of original reaches here.
5. **L4**: Mexican hat discrimination on L3 survivors.

**Compartment access pattern:**
- L0+L1: C7 only (64 bytes)
- L2: C7 + C8 (128 bytes)
- L3: C7 + C8-C31 (1,600 bytes)

90% of candidates only load 64 bytes. 99% load at most 128 bytes.

### Semantic Kernel

The planner reads C6 to decide which skills are relevant to a node. If `skill_relevance & RESONATE_BIT`, it dispatches HdrCascadeExec. If `skill_relevance & TRAVERSE_BIT`, it dispatches GraphTraversalExec. The planner doesn't need to analyze the node's content — the answer was pre-computed at write time and stored in C6.

---

## 5. Sort Order and XOR Delta Compression

### Primary Sort Key

```
(T1, T2, T3, T4, scent[0..2], context_id, disambig)
 ──── DN locality ───  ─ scent ─  ─ version ─  ─ unique ─
```

T1-T4 gives DN subtree locality. `scent[0..2]` (first 3 bytes of raw scent) clusters semantically similar nodes within a subtree. `context_id` puts base before exploration.

### XOR Delta Within Sort-Adjacent Groups

Within a (T1, T2, T3, T4, scent[0..2]) group:

- **C0**: mostly identical (same DN anchor, same entity type). Delta ≈ 0.
- **C1-C3**: often identical (nodes in same subtree share neighbors). Delta ≈ 0.
- **C4**: similar (same SPO roles in subtree). Delta sparse.
- **C5**: similar (NARS beliefs cluster). Delta sparse.
- **C6**: often identical (same zone, same relevant skills). Delta ≈ 0.
- **C7**: similar (scent is what we sorted on!). Delta sparse.
- **C8-C31**: similar (scent-adjacent fingerprints have low Hamming). Delta sparse.

**Estimated compression** within a sorted group of 16 records:
- C0-C7 (512 bytes): ~90% zero → ~51 bytes per delta
- C8-C31 (1,536 bytes): ~75% zero → ~384 bytes per delta
- **Total: ~435 bytes per record vs 2,048 verbatim = 79% compression**

Decode at ingress: XOR with previous row. One pass. Then everything downstream is fixed-size, SIMD-aligned, zero-copy.

---

## 6. Arrow Schema — Addr Is Column Zero

```rust
fn cognitive_256_schema() -> Schema {
    Schema::new(vec![
        // ══════════════════════════════════════════════════════════════
        // PRIMARY KEY: Addr(u16). This is the record's identity.
        // Every Redis command, every SQL query, every Flight RPC, every
        // GQL pattern match resolves to this column first.
        // ══════════════════════════════════════════════════════════════
        Field::new("addr",         DataType::UInt16,               false),

        // Extracted scalars for DataFusion pushdown (< 30 bytes/row)
        Field::new("dn_anchor",    DataType::FixedSizeBinary(4),  false), // T1-T4
        Field::new("dn_leaf",      DataType::FixedSizeBinary(4),  false), // T5-T8
        Field::new("context_id",   DataType::UInt16,              false),
        Field::new("rung",         DataType::UInt8,               false),
        Field::new("depth",        DataType::UInt8,               false),
        Field::new("parent_addr",  DataType::UInt16,              false), // DN tree up
        Field::new("first_child",  DataType::UInt16,              false), // DN tree down
        Field::new("popcount",     DataType::UInt16,              false),
        Field::new("label",        DataType::Utf8,                true),

        // The compartmentalized record (SIMD-aligned bulk columns)
        Field::new("structure",    DataType::FixedSizeBinary(512),  false), // C0-C7
        Field::new("fingerprint",  DataType::FixedSizeBinary(1536), false), // C8-C31
    ])
}
```

**Why `addr: UInt16` is column zero:**

1. **BindSpace → Arrow**: `addr` is already the array index. Converting BindSpace to Arrow is `for i in 0..65536 { batch.addr[row] = i as u16; }`. No transformation.
2. **DataFusion filter pushdown**: `WHERE addr = 0xA37B` is a direct UInt16 equality filter — Arrow dictionary encoding makes this O(1).
3. **DN tree in DataFusion**: `SELECT * FROM nodes WHERE parent_addr = 0xB122` returns all children. No join. No edge table. Just a filter on an extracted u16 column.
4. **Redis → Arrow round-trip**: `DN.GET Ada:A:soul:identity` → `addr = dn_path_to_addr()` → `SELECT * FROM nodes WHERE addr = ?`. Same key, same path, same record.
5. **Flight DoGet**: Client sends `addr` as ticket. Server does `array[addr.prefix()][addr.slot()]`. Zero-copy RecordBatch out.
6. **GQL MATCH**: `MATCH (a)-[:CAUSES]->(b)` internally resolves `a.addr` and `b.addr`. The pattern match result is a pair of UInt16 columns.

**Tree operations as SQL (because why not):**

```sql
-- All children of a node (O(k) via sibling chain, or via extracted column):
SELECT * FROM nodes WHERE parent_addr = 0xB122;

-- Subtree scan (T1-T4 sort adjacency, O(log n + k)):
SELECT * FROM nodes
WHERE dn_anchor = X'A37B0000'  -- same T1-T4
  AND depth > 3;

-- Leaf nodes only:
SELECT addr FROM nodes WHERE first_child = 0xFFFF;

-- All roots:
SELECT addr FROM nodes WHERE parent_addr = 0xFFFF AND depth = 0;
```

Two SIMD columns (`structure` and `fingerprint`) plus extracted scalars. A query that only needs graph structure reads only the `structure` column (512 bytes/row). A query that only needs similarity reads only the `fingerprint` column (1,536 bytes/row). A combined query reads both.

**Alignment:**
- `FixedSizeBinary(512)`: row N at offset `N × 512`. 512 = 8 × 64. Every row 64-byte aligned.
- `FixedSizeBinary(1536)`: row N at offset `N × 1536`. 1536 = 24 × 64. Every row 64-byte aligned.

---

## 7. DataFusion Execution Pipeline

```
TopK(k=20)
  └── HdrCascadeExec (reads: structure C7 for scent, fingerprint C8-C31 for Hamming)
       │  L0: scent from structure bytes [448..495] (C7 words 56-61)
       │  L1: popcount from structure bytes [496..497] (C7 word 62)
       │  L2: first 64 bytes of fingerprint (C8)
       │  L3: full fingerprint (C8-C31)
       │  L4: Mexican hat on survivors
       │  Output: _distance column (u16)
       │
       └── GraphTraversalExec (reads: structure C1-C3 only)
            │  BFS on adjacency bitvectors
            │  Verb filter on C3
            │  Output: _reachable boolean column
            │
            └── NarsFilterExec (reads: structure C5 only)
                 │  WHERE nars_confidence > threshold
                 │  Pure integer comparison on C5 word 40
                 │
                 └── FilterExec (reads: extracted scalars)
                      │  WHERE context_id = 0 AND rung <= 5
                      │
                      └── ParquetExec
                           projection = [key, dn_anchor, context_id,
                                         rung, structure, fingerprint, label]
```

**What each operator touches:**

| Operator | Columns read | Compartments touched | Bytes/row |
|----------|-------------|---------------------|-----------|
| FilterExec | extracted scalars | — | ~10 |
| NarsFilterExec | structure | C5 only | 64 |
| GraphTraversalExec | structure | C1-C3 | 192 |
| HdrCascadeExec L0-L1 | structure | C7 | 64 |
| HdrCascadeExec L2 | fingerprint | C8 | 64 |
| HdrCascadeExec L3 | fingerprint | C8-C31 | 1,536 |

**90% of candidates eliminated by the time we read more than 64 bytes.**

---

## 8. The Rust Struct

```rust
/// 256 × u64 cognitive record.
///
/// 32 compartments of 8 words (64 bytes) each.
/// C0-C7: structure (key, graph, SPO, NARS, kernel, scent).
/// C8-C31: semantic fingerprint (192 words, 12,288 bits).
///
/// Every field is integer. No float in storage.
/// SIMD on any compartment: exactly 1 AVX-512 iteration per compartment.
#[repr(C, align(64))]
pub struct CogRecord {
    words: [u64; 256],
}

/// Compartment index (0-31). Each compartment is 8 u64 = 64 bytes.
#[repr(u8)]
pub enum Comp {
    KeyDn       = 0,   // C0: routing key + DN tree + context
    AdjOut      = 1,   // C1: outgoing adjacency bitvector (512 bits)
    AdjIn       = 2,   // C2: incoming adjacency bitvector (512 bits)
    VerbMask    = 3,   // C3: verb type presence (512 bits)
    Spo         = 4,   // C4: SPO sketch
    Nars        = 5,   // C5: NARS belief (fixed-point integer)
    Kernel      = 6,   // C6: semantic kernel memo
    ScentParity = 7,   // C7: expanded scent + popcount + ECC
    Fp0         = 8,   // C8:  fingerprint words 0-7
    // ...
    Fp23        = 31,  // C31: fingerprint words 184-191
}

impl CogRecord {
    /// Read one compartment as 8 u64 words. One cache line. One AVX-512 register.
    #[inline(always)]
    pub fn comp(&self, c: Comp) -> &[u64; 8] {
        let offset = (c as usize) * 8;
        unsafe { &*(self.words.as_ptr().add(offset) as *const [u64; 8]) }
    }

    /// Read full structure (C0-C7 = 64 words = 512 bytes)
    #[inline(always)]
    pub fn structure(&self) -> &[u64; 64] {
        unsafe { &*(self.words.as_ptr() as *const [u64; 64]) }
    }

    /// Read full fingerprint (C8-C31 = 192 words = 1,536 bytes)
    #[inline(always)]
    pub fn fingerprint(&self) -> &[u64; 192] {
        unsafe { &*(self.words.as_ptr().add(64) as *const [u64; 192]) }
    }

    // ═══════════════════════════════════════════════════════════════════
    // C0: PRIMARY KEY + ROUTING
    // ═══════════════════════════════════════════════════════════════════

    /// The primary key. This IS the record's identity everywhere.
    #[inline(always)] pub fn addr(&self) -> u16 {
        (self.words[0] & 0xFFFF) as u16  // prefix:slot packed
    }
    #[inline(always)] pub fn prefix(&self)     -> u8  { self.words[0] as u8 }
    #[inline(always)] pub fn slot(&self)       -> u8  { (self.words[0] >> 8) as u8 }
    #[inline(always)] pub fn group48(&self)    -> u64 { self.words[0] >> 16 }
    #[inline(always)] pub fn disambig(&self)   -> u64 { self.words[1] }

    /// DN tier slots (T1-T8), word 4
    #[inline(always)] pub fn dn_anchor(&self) -> [u8; 4] {
        let b = self.words[4].to_le_bytes();
        [b[0], b[1], b[2], b[3]]  // T1, T2, T3, T4 (frozen)
    }
    #[inline(always)] pub fn dn_leaf(&self) -> [u8; 4] {
        let b = self.words[4].to_le_bytes();
        [b[4], b[5], b[6], b[7]]  // T5, T6, T7, T8 (ephemeral)
    }

    // ═══════════════════════════════════════════════════════════════════
    // C0 word 5: O(1) DN TREE (LCRS pointers)
    //
    // Parent is derived: strip last segment from DN path, hash again.
    //   "Ada:A:soul:identity" → parent = dn_path_to_addr("Ada:A:soul")
    // Siblings are derived: same parent, different leaf.
    // But we ALSO store the pointers for O(1) chasing without string ops.
    // ═══════════════════════════════════════════════════════════════════

    const NONE_ADDR: u16 = 0xFFFF;

    #[inline(always)] pub fn parent_addr(&self)    -> u16 {
        (self.words[5] & 0xFFFF) as u16
    }
    #[inline(always)] pub fn first_child(&self)    -> u16 {
        ((self.words[5] >> 16) & 0xFFFF) as u16
    }
    #[inline(always)] pub fn next_sibling(&self)   -> u16 {
        ((self.words[5] >> 32) & 0xFFFF) as u16
    }
    #[inline(always)] pub fn prev_sibling(&self)   -> u16 {
        ((self.words[5] >> 48) & 0xFFFF) as u16
    }
    #[inline(always)] pub fn is_leaf(&self) -> bool {
        self.first_child() == Self::NONE_ADDR
    }
    #[inline(always)] pub fn is_root(&self) -> bool {
        self.parent_addr() == Self::NONE_ADDR
    }

    /// Pack LCRS pointers into word 5
    #[inline(always)]
    pub fn set_tree_pointers(&mut self, parent: u16, first_child: u16,
                              next_sib: u16, prev_sib: u16) {
        self.words[5] = (parent as u64)
            | ((first_child as u64) << 16)
            | ((next_sib as u64) << 32)
            | ((prev_sib as u64) << 48);
    }

    /// Insert self as first child of parent. O(1) pointer surgery.
    /// Returns the old first_child (so caller can update its prev_sib).
    pub fn insert_as_child(
        records: &mut [CogRecord],  // the full 64K BindSpace
        parent_idx: usize,
        child_idx: usize,
    ) {
        let old_first = records[parent_idx].first_child();

        // New child points to parent, old first child as next sibling
        let child_parent = records[parent_idx].addr();
        records[child_idx].set_tree_pointers(
            child_parent,
            Self::NONE_ADDR,  // new child has no children yet
            old_first,        // next_sibling = parent's old first_child
            Self::NONE_ADDR,  // prev_sibling = none (we're the new head)
        );

        // Old first child's prev_sibling = new child
        if old_first != Self::NONE_ADDR {
            let old_w5 = records[old_first as usize].words[5];
            records[old_first as usize].words[5] =
                (old_w5 & 0x0000_FFFF_FFFF_FFFF)
                | ((records[child_idx].addr() as u64) << 48);
        }

        // Parent's first_child = new child
        let p_w5 = records[parent_idx].words[5];
        records[parent_idx].words[5] =
            (p_w5 & 0xFFFF_FFFF_0000_FFFF)
            | ((records[child_idx].addr() as u64) << 16);

        // Increment child count in word 6
        let w6 = records[parent_idx].words[6];
        let old_cnt = ((w6 >> 40) & 0xFF) as u8;
        records[parent_idx].words[6] =
            (w6 & !(0xFF << 40)) | (((old_cnt.saturating_add(1)) as u64) << 40);
    }

    // ═══════════════════════════════════════════════════════════════════
    // C0 word 6: CONTEXT + HIERARCHY + COUNTS
    // ═══════════════════════════════════════════════════════════════════

    #[inline(always)] pub fn context_id(&self) -> u16 {
        (self.words[6] & 0xFFFF) as u16
    }
    #[inline(always)] pub fn rung(&self) -> u8 {
        ((self.words[6] >> 16) & 0xFF) as u8
    }
    #[inline(always)] pub fn sigma(&self) -> u8 {
        ((self.words[6] >> 24) & 0xFF) as u8
    }
    #[inline(always)] pub fn depth(&self) -> u8 {
        ((self.words[6] >> 32) & 0xFF) as u8
    }
    #[inline(always)] pub fn child_count(&self) -> u8 {
        ((self.words[6] >> 40) & 0xFF) as u8
    }

    // ═══════════════════════════════════════════════════════════════════
    // C5: NARS (integer only, Q16.16 fixed-point)
    // ═══════════════════════════════════════════════════════════════════

    #[inline(always)]
    pub fn nars_f_fp(&self) -> u32 { self.words[40] as u32 }
    #[inline(always)]
    pub fn nars_c_fp(&self) -> u32 { (self.words[40] >> 32) as u32 }
    #[inline(always)]
    pub fn evidence_pos(&self) -> u32 { self.words[41] as u32 }
    #[inline(always)]
    pub fn evidence_neg(&self) -> u32 { (self.words[41] >> 32) as u32 }

    // ═══════════════════════════════════════════════════════════════════
    // C1-C3: GRAPH ADJACENCY + VERB MASK (separate from DN tree)
    // ═══════════════════════════════════════════════════════════════════

    #[inline(always)]
    pub fn has_edge_to_bucket(&self, b: u16) -> bool {
        let w = self.words[8 + (b / 64) as usize];  // C1 starts at word 8
        w & (1u64 << (b % 64)) != 0
    }

    #[inline(always)]
    pub fn has_verb_out(&self, v: u16) -> bool {
        let w = self.words[24 + (v / 64) as usize]; // C3 starts at word 24
        w & (1u64 << (v % 64)) != 0
    }

    // ═══════════════════════════════════════════════════════════════════
    // C8-C31: FINGERPRINT (192 words, 12,288 bits)
    // ═══════════════════════════════════════════════════════════════════

    pub fn hamming_fp(&self, other: &CogRecord) -> u32 {
        let a = self.fingerprint();
        let b = other.fingerprint();
        let mut dist = 0u32;
        for i in 0..192 { dist += (a[i] ^ b[i]).count_ones(); }
        dist
    }

    pub fn bind_fp(&self, other: &CogRecord) -> [u64; 192] {
        let a = self.fingerprint();
        let b = other.fingerprint();
        let mut r = [0u64; 192];
        for i in 0..192 { r[i] = a[i] ^ b[i]; }
        r
    }

    // ═══════════════════════════════════════════════════════════════════
    // C7: SCENT + POPCOUNT + PARITY
    // ═══════════════════════════════════════════════════════════════════

    pub fn scent_distance(&self, other: &CogRecord) -> u32 {
        let a = self.comp(Comp::ScentParity);
        let b = other.comp(Comp::ScentParity);
        let mut dist = 0u32;
        for i in 0..6 { dist += (a[i] ^ b[i]).count_ones(); }
        dist
    }

    pub fn popcount(&self) -> u16 { (self.words[62] & 0xFFFF) as u16 }

    // ═══════════════════════════════════════════════════════════════════
    // DN TREE ITERATION (all O(1) per step, no external structures)
    // ═══════════════════════════════════════════════════════════════════

    /// Iterate children: first_child → next_sibling → next_sibling → ...
    pub fn children_iter<'a>(records: &'a [CogRecord], addr: u16)
        -> impl Iterator<Item = u16> + 'a
    {
        let mut cursor = records[addr as usize].first_child();
        std::iter::from_fn(move || {
            if cursor == Self::NONE_ADDR { return None; }
            let current = cursor;
            cursor = records[current as usize].next_sibling();
            Some(current)
        })
    }

    /// Iterate ancestors: parent → parent → parent → ... → root
    pub fn ancestors_iter<'a>(records: &'a [CogRecord], addr: u16)
        -> impl Iterator<Item = u16> + 'a
    {
        let mut cursor = records[addr as usize].parent_addr();
        std::iter::from_fn(move || {
            if cursor == Self::NONE_ADDR { return None; }
            let current = cursor;
            cursor = records[current as usize].parent_addr();
            Some(current)
        })
    }
}
```

---

## 9. Migration Path

| Step | What | Files touched |
|------|------|---------------|
| 1 | Define `CogRecord` struct + `Comp` enum | New: `src/storage/cog_record.rs` |
| 2 | Replace `BindNode` with `CogRecord` in BindSpace arrays | `bind_space.rs` |
| 3 | Add LCRS tree pointers to `write_dn_path()` | `bind_space.rs:1218` — call `insert_as_child()` |
| 4 | Remove `BitpackedCsr` for DN tree ops (keep for edge-verb lookup) | `bind_space.rs` |
| 5 | Add `fp_mul` and fixed-point NARS to `nars/truth.rs` | Existing: add `TruthValueFp` type |
| 6 | Change `FINGERPRINT_WORDS` to 192, `FINGERPRINT_BYTES` to 1536 | `bind_space.rs`, `lib.rs` |
| 7 | Update `FingerprintBuffer` to 1,536-byte fingerprints | `lance_zero_copy/mod.rs` |
| 8 | Populate C1-C3 (adjacency, verb mask) on edge insert | `bind_space.rs` add_edge() |
| 9 | Populate C4 (SPO sketch) on triple insert | `extensions/spo/spo.rs` |
| 10 | Populate C5 (NARS) on inference | `nars/truth.rs`, `cog_redis.rs` |
| 11 | Populate C6 (kernel memo) on node classification | `orchestration/semantic_kernel.rs` |
| 12 | Populate C7 (scent, popcount) on fingerprint write | `core/scent.rs` |
| 13 | Update `HdrCascadeExec` constants (WORDS=192) | `search/hdr_cascade.rs` |
| 14 | Update `FingerprintTableProvider` → `CogRecordProvider` | `query/fingerprint_table.rs` |
| 15 | Add `addr: UInt16` as primary column in Arrow schema | `query/fingerprint_table.rs` |
| 16 | Extract `parent_addr`, `first_child` as Arrow columns | `query/fingerprint_table.rs` |
| 17 | Add `GraphTraversalExec` reading C1-C3 | New: `query/graph_traverse_exec.rs` |
| 18 | Add `NarsFilterExec` reading C5 | New: `query/nars_filter_exec.rs` |

**What changes:** `BindNode` → `CogRecord`. LCRS pointers replace CSR for tree ops. `addr` becomes Arrow column zero.

**What doesn't change:** `Addr(u16)` semantics, `dn_path_to_addr()`, `CogRedis` command syntax, Flight protocol, DN path parsing. The path `"Ada:A:soul:identity"` still hashes to the same address. The address still indexes the same array. Everything converges to the same record — it just has more pre-computed state now.

### The Path-Derived Insight

The parent is free. The siblings are free. No traversal cost:

```rust
// Parent = strip last segment, hash again. O(1) string op.
fn parent_path(path: &str) -> Option<&str> {
    path.rfind(':').map(|i| &path[..i])
}

// "Ada:A:soul:identity" → "Ada:A:soul" → dn_path_to_addr() → parent Addr
// No pointer chase needed to FIND the parent. The path IS the hierarchy.
// But we STORE the pointer anyway (word 5) so that:
//   - Records without their path string can still navigate
//   - Arrow queries can do WHERE parent_addr = X without string parsing
//   - Iterator chains (children, ancestors) work on raw Addr arrays
//   - The tree structure survives serialization/deserialization
//
// Two paths to the same truth:
//   1. String path: strip segment, hash → O(1), needs the path string
//   2. Stored pointer: read word 5 → O(1), needs only the record
//
// Both return the same Addr. Always.
```
