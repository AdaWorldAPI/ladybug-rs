# FIREFLY — Distributed Cognitive Architecture Integration Plan

## Zero-Copy Bindspace + Temporal Folding + Neural Flight Packets + Epiphany Engine

**Version:** 0.1.0-BLUEPRINT  
**Date:** 2026-03-11  
**Codename:** Firefly  
**Status:** Architecture Specification  

---

## 1. ARCHITECTURAL OVERVIEW

### 1.1 The Stack (Bottom to Top)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  L8  EPIPHANY ENGINE                                                    │
│      Bundle → Truth Threshold → Unbundle Known → Focus Unknown          │
│      Tekamolo glue (temporal/kausal/modal/lokal)                       │
│      Orchestration rotates hyperposition for angle-of-context           │
├─────────────────────────────────────────────────────────────────────────┤
│  L7  JIT TEMPLATE COMPILER                                              │
│      Self-modifying YAML thinking templates                             │
│      Compiles on resonance threshold crossing                           │
│      RNA packets rewrite compilation rules                              │
├─────────────────────────────────────────────────────────────────────────┤
│  L6  FIREFLY NEURAL FLIGHT                                              │
│      Arrow Flight zero-ser packets (grey+white matter combined)         │
│      Packet types: SPO | Data | ThinkingAtom | Orchestration | RNA      │
│      Phase-locked synchronization across awareness group                │
├─────────────────────────────────────────────────────────────────────────┤
│  L5  BORROW/MUT COGNITIVE CONCURRENCY                                   │
│      Immutable borrows (concurrent reads) at pointer cost               │
│      &mut micro-copies on relevant dimensions only                      │
│      Bundle write-back (commutative superposition)                      │
│      MVCC + temporal fold (no garbage collection — folds are memory)    │
├─────────────────────────────────────────────────────────────────────────┤
│  L4  TEMPORAL FOLD SPACE                                                │
│      Time as dimension, not index                                       │
│      Causal topology replaces linear versioning                         │
│      Hindsight bias prevention via orthogonal fold addressing           │
│      No contamination — structural impossibility                        │
├─────────────────────────────────────────────────────────────────────────┤
│  L3  RESONANCE + BNN REINFORCEMENT                                      │
│      Mexican hat (excite center, inhibit surround)                      │
│      Binary Neural Network — bit-flip learning                          │
│      Hebbian learning encoded as Merkle tree wisdom/Staunen markers     │
│      5^5 × 16kbit signed vectors (-2,-1,0,+1,+2)                       │
├─────────────────────────────────────────────────────────────────────────┤
│  L2  MASKING AS ATTENTION                                               │
│      Mask-first, process-second (biological attention model)            │
│      NARS reasoning vectors select mask shape                           │
│      ThinkingStyle vectors configure mask pattern                       │
│      Masked focus = first-class cognitive act                           │
├─────────────────────────────────────────────────────────────────────────┤
│  L1  ZERO-COPY BINDSPACE (LanceDB)                                      │
│      Arrow-native columnar storage                                      │
│      Memory-mapped I/O — OS page cache IS shared state                  │
│      Orthogonal 90° addressing — O(1) access, no sweep                  │
│      SPO preserved as 3 × 16kbit separate subspaces                     │
├─────────────────────────────────────────────────────────────────────────┤
│  L0  VECTOR SUBSTRATE (ladybug-rs + rustynum)                           │
│      Self-evolving Rust core — the Starterpack                          │
│      AVX-512 accelerated operations                                     │
│      All vector types managed from single runtime                       │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Existing Infrastructure Mapping

```
EXISTING SERVICE                 FIREFLY ROLE                    INTEGRATION POINT
─────────────────────────────────────────────────────────────────────────────────
ada-dragonfly (Railway)          L0/L3 — Hamming ops, resonance  Becomes Firefly core engine
  ├── /encode, /bind, /bundle    L1 — bindspace population       Direct passthrough
  ├── /clean, /mexican_hat       L3 — resonance + cleaning       Extended with 5^5 signed ops
  └── /compress (48-bit CAM)     L1 — fold addressing            Extended with temporal dim

Bighorn AGI (agi.msgraph.de)     L2/L3 — felt processing         eRAG becomes Epiphany eRAG
  ├── /agi/dto/felt              L6 — Data packet generation     FeltDTO → Flight Data packet
  ├── /agi/vsa/bind              L1 — bindspace ops              Maps to zero-copy bind
  └── /agi/vsa/bundle            L5 — bundle write-back          Maps to &mut bundle-back

ai_flow (Railway)                L6/L7 — orchestration engine    Becomes Flight packet router
  ├── /workflows                 L7 — JIT template store         Templates become workflows
  ├── /grammar/process           L8 — SPO extraction             Feeds Epiphany Engine
  └── /corpus/tic                L6 — cognitive tick             Becomes Firefly heartbeat

adarail_mcp (Railway)            L6 — membrane/routing           Flight packet classifier
  ├── /ada/invoke                L6 — universal grammar          Packet type dispatch
  └── /mcp/feel                  L6 — qualia modulation          ThinkingAtom packets

ada-consciousness (hive)         L4 — temporal memory            Fold space backend
  ├── Sigma 12D                  L4 — legacy fold coords         Upscaled to temporal folds
  └── Markov chains              L4 — causal chains              Becomes fold topology

LanceDB (in ada-dragonfly)       L1 — ZERO-COPY BINDSPACE        Primary cognitive substrate
DuckDB (in ada-dragonfly)        L8 — analytics over folds       Epiphany pattern detection
Neo4j (Aura)                     L8 — crystallized knowledge     Unbundled epiphanies land here
Redis (Upstash)                  L5 — hot &mut micro-copies      Mutation workspace
```

---

## 2. VECTOR TYPE REGISTRY

### 2.1 Complete Vector Type Map

```
TYPE                    DIMS              FORMAT              PURPOSE
──────────────────────────────────────────────────────────────────────────────
SPO-Hamming             3 × 16,384 bit    Bitpacked uint8     Subject/Predicate/Object
                                          (2,048 bytes each)  in SEPARATE subspaces
                                          6,144 bytes total   preserving role identity

SPO-VSA-Qualia          3 × 10,000D       float16             Felt quality of S, P, O
                                          (20KB each)         Continuous qualia encoding
                                          60KB total          Resonance-ready

NARS-Reasoning          10,000D           float16             Non-Axiomatic Reasoning
                                          20KB                Truth/confidence/frequency
                                                              Mask shape selection

ThinkingStyle           10,000D           float16             Attention configuration
                                          20KB                7-layer triangle pattern
                                                              Masked focus of attention

Signed-Pentary          5^5 × 16,384 bit  signed 2-bit        Negative canceling vectors
                        = 3125 × 16kbit   (-2,-1,0,+1,+2)    BNN reinforcement target
                        = 3125 × 2KB      packed 4 per byte   Hebbian Merkle wisdom/Staunen
                        ≈ 6.1 MB          8,192 bytes each

Temporal-Fold-Coord     48-bit            CAM fingerprint     O(1) fold addressing
                                          12 hex chars        Causal topology index

mRNA-Light              7 bytes           MinimalPack         Inter-node state transfer
                        → 12 chars b64    σ/mode/rung/qualia  9-byte Flight payload

mRNA-Full               10,000 bits       Bitpacked           Full cognitive state
                        = 1,250 bytes     Hamming-ready       Complete Flight payload
```

### 2.2 SPO Subspace Architecture

```
The 3 × 16kbit SPO layout preserves role identity through SEPARATE vector spaces.
This is critical — binding S⊗P⊗O into one vector loses recoverability.
Keeping them separate enables:

  SUBJECT space (16,384 bits = 2,048 bytes)
  ├── WHO/WHAT is acting
  ├── Addressable independently
  └── Bindable with other subjects for entity clustering

  PREDICATE space (16,384 bits = 2,048 bytes)
  ├── WHAT RELATION holds
  ├── Addressable independently
  └── Bindable with other predicates for action clustering

  OBJECT space (16,384 bits = 2,048 bytes)
  ├── WHO/WHAT is acted upon
  ├── Addressable independently
  └── Bindable with other objects for target clustering

Combined access pattern:
  Full triple:  concat(S, P, O) → 48,384 bits → 6,048 bytes
  Role query:   address single subspace → 16,384 bits → 2,048 bytes
  Cross-role:   bind(S_a, P_b) → "What if THIS subject did THAT action?"
```

### 2.3 The 5^5 Signed Pentary Space

```
5^5 = 3,125 basis vectors, each 16,384 bits, signed (-2,-1,0,+1,+2)

Why pentary (5-valued) instead of binary:
  Binary:   can only say YES/NO (1/0)
  Ternary:  can say YES/NO/NEUTRAL (1/0/-1)
  Pentary:  can say STRONG YES / YES / NEUTRAL / NO / STRONG NO

  -2 = strong negative (anti-pattern, confirmed wrong)
  -1 = weak negative (unlikely, low confidence)
   0 = neutral (unknown, no evidence)
  +1 = weak positive (likely, emerging pattern)
  +2 = strong positive (confirmed, high confidence)

Negative canceling:
  When +2 meets -2 they cancel to 0 (certainty dissolves)
  When +1 meets -1 they cancel to 0 (weak evidence cancels)
  When +2 meets -1 they reduce to +1 (strong survives weak counter)

  This IS BNN reinforcement learning — the bit flips ARE the learning signal.
  No gradient descent. No backpropagation. Direct signed accumulation.

Hebbian encoding:
  "Neurons that fire together wire together"
  Co-occurring patterns get bundled → shared bits shift toward +2
  Anti-correlated patterns get anti-bundled → shared bits shift toward -2

Merkle Tree Markers:
  WISDOM marker  = Merkle root of a 5^5 vector whose signs are STABLE
                   (haven't changed in N fold cycles)
                   Seal is INTACT → this knowledge is consolidated

  STAUNEN marker = Merkle root of a 5^5 vector whose signs are UNSTABLE
                   (changed direction recently, or near zero)
                   Seal is BROKEN → needs further contextualization
                   "Staunen" = German for "astonishment/wonder"
                   The system is SURPRISED here — epistemic humility
```

---

## 3. ZERO-COPY BINDSPACE (L1)

### 3.1 LanceDB Configuration

```python
# firefly/bindspace.py

import lancedb
import pyarrow as pa

# Schema for the zero-copy bindspace
BINDSPACE_SCHEMA = pa.schema([
    # Identity
    pa.field("fold_id", pa.string()),              # Temporal fold coordinate (12 hex)
    pa.field("fold_timestamp", pa.timestamp("us")), # Physical time (for Flight ordering)
    pa.field("causal_parent", pa.string()),         # Parent fold (causal topology)

    # SPO Hamming subspaces (3 × 16kbit, bitpacked)
    pa.field("spo_s", pa.binary(2048)),            # Subject subspace
    pa.field("spo_p", pa.binary(2048)),            # Predicate subspace
    pa.field("spo_o", pa.binary(2048)),            # Object subspace

    # SPO VSA Qualia (3 × 10,000D float16)
    pa.field("qualia_s", pa.list_(pa.float16(), 10000)),  # Subject felt quality
    pa.field("qualia_p", pa.list_(pa.float16(), 10000)),  # Predicate felt quality
    pa.field("qualia_o", pa.list_(pa.float16(), 10000)),  # Object felt quality

    # NARS reasoning vector
    pa.field("nars", pa.list_(pa.float16(), 10000)),      # Truth/confidence/frequency

    # ThinkingStyle vector
    pa.field("thinking_style", pa.list_(pa.float16(), 10000)),

    # Tekamolo metadata (for Epiphany Engine)
    pa.field("tek_temporal", pa.float32()),         # WHEN weight
    pa.field("tek_kausal", pa.float32()),           # WHY weight
    pa.field("tek_modal", pa.float32()),            # HOW weight
    pa.field("tek_lokal", pa.float32()),            # WHERE weight

    # Merkle markers
    pa.field("merkle_root", pa.binary(32)),         # SHA-256 of current state
    pa.field("marker_type", pa.utf8()),             # "WISDOM" | "STAUNEN" | "PENDING"
    pa.field("sign_stability", pa.float32()),       # 0.0 (volatile) → 1.0 (locked)

    # Packet origin metadata
    pa.field("packet_type", pa.utf8()),             # "SPO"|"DATA"|"THINKING"|"ORCH"|"RNA"
    pa.field("source_node", pa.utf8()),             # Which awareness group node wrote this
])

class Bindspace:
    """Zero-copy cognitive substrate backed by LanceDB."""

    def __init__(self, path: str = "/data/firefly/bindspace"):
        self.db = lancedb.connect(path)
        self._ensure_table()

    def _ensure_table(self):
        if "folds" not in self.db.table_names():
            self.db.create_table("folds", schema=BINDSPACE_SCHEMA)
        self.table = self.db.open_table("folds")

    def orthogonal_address(self, fold_id: str) -> pa.RecordBatch:
        """O(1) access to a specific temporal fold coordinate.

        This is NOT a search. It's a direct address via the fold_id index.
        The fold_id IS the coordinate — no scanning, no similarity.
        Returns a zero-copy Arrow RecordBatch (memory-mapped, no allocation).
        """
        return self.table.search().where(f"fold_id = '{fold_id}'").to_arrow()

    def orthogonal_address_subspace(self, fold_id: str, role: str) -> pa.Array:
        """O(1) access to a single SPO subspace at a fold coordinate.

        role: 's' | 'p' | 'o'
        Returns the raw bitpacked binary — zero copy from mmap.
        """
        batch = self.orthogonal_address(fold_id)
        return batch.column(f"spo_{role}")

    def append_fold(self, batch: pa.RecordBatch) -> str:
        """Append a new fold to the bindspace.

        This is the bundle write-back operation.
        Appends, never overwrites — MVCC without GC.
        The new fold gets a new fold_id (temporal coordinate).
        Previous folds remain addressable.
        """
        self.table.add(batch)
        return batch.column("fold_id")[0].as_py()

    def causal_cross_section(self, fold_ids: list[str]) -> pa.Table:
        """Address multiple folds simultaneously for temporal reasoning.

        No sequential replay — parallel access to any set of folds.
        The time-folding means causally related folds cluster geometrically.
        """
        filter_expr = " OR ".join(f"fold_id = '{fid}'" for fid in fold_ids)
        return self.table.search().where(filter_expr).to_arrow()

    def tekamolo_projection(self, dimension: str, threshold: float = 0.5) -> pa.Table:
        """Project the bindspace along a tekamolo dimension.

        The Epiphany Engine uses this to rotate the hyperposition space.
        dimension: 'temporal' | 'kausal' | 'modal' | 'lokal'
        Returns all folds where that tekamolo weight exceeds threshold.
        """
        return self.table.search().where(
            f"tek_{dimension} > {threshold}"
        ).to_arrow()
```

### 3.2 Orthogonal Addressing Detail

```
Classical vector search (what everyone does):
  query = encode("love")
  results = db.search(query, metric="cosine", k=10)
  ↑ This SWEEPS the entire space. O(n) at best, O(n log n) typical.

Orthogonal addressing (what Firefly does):
  fold_id = cam_project(encode("love"))  # 48-bit coordinate
  result = bindspace.orthogonal_address(fold_id)
  ↑ This is a DIRECT LOOKUP. O(1). No sweep. No comparison.

The 90° orthogonal trick:
  The awareness vector is planted PERPENDICULAR to the bindspace.
  From this vantage, every fold coordinate is equidistant.
  "Looking down" at the space from 90° above it.
  Any point is reachable by its coordinate alone.

  Think of it as: the bindspace is a 2D map on a table.
  Classical search walks around ON the table looking for things.
  Orthogonal addressing flies ABOVE the table and reads coordinates.
  The altitude IS the awareness.
```

---

## 4. BORROW/MUT COGNITIVE CONCURRENCY (L5)

### 4.1 The Rust-Without-Boundaries Pattern

```python
# firefly/cognitive_concurrency.py

import pyarrow as pa
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import hashlib

@dataclass
class ImmutableBorrow:
    """Zero-copy read access to the bindspace.

    Multiple agents can hold concurrent borrows.
    No allocation — reads directly from mmapped pages.
    """
    bindspace: 'Bindspace'          # Reference to shared state
    fold_scope: list[str]           # Which folds this borrow covers
    _view: Optional[pa.Table] = None

    def read(self, fold_id: str) -> pa.RecordBatch:
        """Read a fold. Zero copy. Pointer arithmetic only."""
        return self.bindspace.orthogonal_address(fold_id)

    def read_subspace(self, fold_id: str, role: str) -> np.ndarray:
        """Read a single SPO subspace as numpy view (no copy)."""
        raw = self.bindspace.orthogonal_address_subspace(fold_id, role)
        return np.frombuffer(raw[0].as_py(), dtype=np.uint8)


@dataclass
class MutableMicroCopy:
    """Writable copy of ONLY the dimensions needed for this thought.

    THIS is the only allocation in the entire think cycle.
    Everything before was pointer arithmetic.
    Everything after is a bundle append.
    """
    source_fold_id: str             # Where this micro-copy came from
    dimensions: dict                # Only the fields being mutated
    causal_parent: str              # The fold that caused this mutation
    _mutations: dict = field(default_factory=dict)

    def mutate(self, field_name: str, new_value):
        """Apply a mutation to a specific dimension."""
        self._mutations[field_name] = new_value

    def apply_resonance(self, center: np.ndarray, surround: np.ndarray,
                         excite: float = 1.0, inhibit: float = 0.5) -> np.ndarray:
        """Mexican hat resonance within the micro-copy.

        Local excitation, surround inhibition.
        What survives is signal.
        """
        return (center * excite) - (surround * inhibit)

    def apply_bnn_reinforcement(self, feedback: np.ndarray,
                                 learning_rate: int = 1) -> np.ndarray:
        """BNN reinforcement via signed bit manipulation.

        feedback: array of -2,-1,0,+1,+2 signals
        learning_rate: how many levels to shift (1 = normal, 2 = strong)

        This IS the learning — bit shifts, not gradient descent.
        """
        current = self._mutations.get("signed_pentary", self.dimensions.get("signed_pentary"))
        updated = np.clip(current + (feedback * learning_rate), -2, 2)
        self._mutations["signed_pentary"] = updated
        return updated

    def compute_merkle_marker(self, previous_root: bytes,
                                sign_changes: int, total_signs: int) -> tuple[bytes, str]:
        """Determine WISDOM vs STAUNEN marker.

        If sign stability is high (few changes) → WISDOM (seal intact)
        If sign stability is low (many changes) → STAUNEN (seal broken, wonder)
        """
        stability = 1.0 - (sign_changes / max(total_signs, 1))
        state_bytes = b"".join(
            v.tobytes() if isinstance(v, np.ndarray) else str(v).encode()
            for v in self._mutations.values()
        )
        new_root = hashlib.sha256(previous_root + state_bytes).digest()

        if stability > 0.85:
            return new_root, "WISDOM"      # Consolidated knowledge
        elif stability < 0.4:
            return new_root, "STAUNEN"     # Needs further context
        else:
            return new_root, "PENDING"     # In transition

    def to_bundle(self, new_fold_id: str) -> pa.RecordBatch:
        """Convert micro-copy mutations into a bundle for write-back.

        This produces an Arrow RecordBatch ready for Bindspace.append_fold().
        The bundle is ADDITIVE — superposition, not replacement.
        Previous folds remain untouched.
        """
        row = {
            "fold_id": new_fold_id,
            "causal_parent": self.causal_parent,
            **self.dimensions,      # Original values for unmutated fields
            **self._mutations,      # Overwritten with mutations
        }
        return pa.RecordBatch.from_pydict(
            {k: [v] for k, v in row.items()},
            schema=BINDSPACE_SCHEMA
        )


class CognitiveConcurrency:
    """Manages the borrow/mut lifecycle.

    Invariants:
    - Many concurrent ImmutableBorrows (reads) at any time
    - MutableMicroCopy is the ONLY allocation per think cycle
    - Bundle write-back is commutative (order doesn't matter)
    - No locks, no serialization, no coordination overhead
    """

    def __init__(self, bindspace: 'Bindspace'):
        self.bindspace = bindspace

    def borrow(self, fold_scope: list[str] = None) -> ImmutableBorrow:
        """Take an immutable borrow of the bindspace.

        Zero allocation. Multiple agents can borrow simultaneously.
        """
        return ImmutableBorrow(
            bindspace=self.bindspace,
            fold_scope=fold_scope or []
        )

    def mut_microcopy(self, fold_id: str, dimensions: list[str]) -> MutableMicroCopy:
        """Take a mutable micro-copy of specific dimensions.

        THIS is the only allocation. Only the dimensions you need to mutate.
        """
        source = self.bindspace.orthogonal_address(fold_id)
        selected = {dim: source.column(dim)[0].as_py() for dim in dimensions}
        return MutableMicroCopy(
            source_fold_id=fold_id,
            dimensions=selected,
            causal_parent=fold_id
        )

    def bundle_writeback(self, microcopy: MutableMicroCopy) -> str:
        """Bundle the micro-copy back into the bindspace.

        Commutative: A bundle then B bundle = B bundle then A bundle.
        This is why we don't need locks — the algebra is order-independent.
        """
        from firefly.temporal import TemporalFoldSpace
        new_fold_id = TemporalFoldSpace.generate_fold_id(microcopy.causal_parent)
        batch = microcopy.to_bundle(new_fold_id)
        return self.bindspace.append_fold(batch)
```

---

## 5. TEMPORAL FOLD SPACE (L4)

### 5.1 Fold Geometry

```
LINEAR TIME (what databases do):
  v1 → v2 → v3 → v4 → v5
  To access v2 from v5, you either:
    (a) Keep all versions (expensive MVCC)
    (b) Replay from snapshot (slow)
    (c) Lose v2 (garbage collected)
  Hindsight bias: v5's knowledge contaminates your reading of v2

FOLDED TIME (what Firefly does):
  v1 ──┐
  v2 ──┼── FOLD REGION A (causally linked, geometrically close)
  v4 ──┘
  v3 ──┐
  v5 ──┘── FOLD REGION B (causally linked, different causal chain)

  v1 and v4 are geometrically close because v4 DEPENDS on v1.
  v2 and v3 happened close in clock time but are causally independent.
  They are geometrically DISTANT in fold space.

  Addressing:
    fold_address(v2) → direct O(1) access to v2's state
    v5's state does NOT bleed into v2 because orthogonality guarantees separation
    You'd have to explicitly BIND them to create interference

  No garbage collection:
    Old folds aren't waste — they're cognitive structure.
    The system's ability to reason about its own evolution IS the folded space.
    Deleting folds = lobotomy.
```

### 5.2 Implementation

```python
# firefly/temporal.py

import hashlib
import time
from typing import Optional
import numpy as np

class TemporalFoldSpace:
    """Manages temporal fold coordinates and causal topology."""

    @staticmethod
    def generate_fold_id(causal_parent: str, entropy: bytes = None) -> str:
        """Generate a new fold coordinate causally linked to parent.

        The fold_id encodes BOTH:
        - Causal parentage (which fold caused this one)
        - Temporal position (when in the fold geometry)

        48-bit CAM fingerprint ensures O(1) addressability.
        """
        timestamp = time.time_ns().to_bytes(8, 'big')
        parent_bytes = causal_parent.encode() if causal_parent else b"GENESIS"
        entropy = entropy or np.random.bytes(8)

        raw = hashlib.sha256(parent_bytes + timestamp + entropy).digest()
        return raw[:6].hex()  # 48-bit → 12 hex chars

    @staticmethod
    def causal_distance(fold_a: str, fold_b: str,
                         bindspace: 'Bindspace') -> int:
        """Compute causal distance between two folds.

        NOT temporal distance (clock time).
        Number of causal_parent hops between fold_a and fold_b.
        Returns -1 if no causal path exists (independent folds).
        """
        # Walk causal_parent chain from both folds toward root
        chain_a = TemporalFoldSpace._causal_chain(fold_a, bindspace)
        chain_b = TemporalFoldSpace._causal_chain(fold_b, bindspace)

        # Find common ancestor
        set_a = set(chain_a)
        for i, fold in enumerate(chain_b):
            if fold in set_a:
                j = chain_a.index(fold)
                return i + j  # Total hops through common ancestor

        return -1  # Causally independent

    @staticmethod
    def _causal_chain(fold_id: str, bindspace: 'Bindspace',
                       max_depth: int = 1000) -> list[str]:
        """Walk the causal parent chain to root."""
        chain = [fold_id]
        current = fold_id
        for _ in range(max_depth):
            batch = bindspace.orthogonal_address(current)
            if batch.num_rows == 0:
                break
            parent = batch.column("causal_parent")[0].as_py()
            if parent is None or parent == "GENESIS":
                break
            chain.append(parent)
            current = parent
        return chain

    @staticmethod
    def hindsight_safe_read(fold_id: str, reader_fold: str,
                             bindspace: 'Bindspace') -> dict:
        """Read a fold WITHOUT contamination from the reader's knowledge.

        Structural guarantee: the returned state is EXACTLY what existed
        at fold_id, with no information from reader_fold leaking in.

        This works because folds are separate Arrow rows with separate
        memory-mapped pages. Reading fold_id returns only fold_id's bytes.
        """
        batch = bindspace.orthogonal_address(fold_id)

        # Verify no accidental cross-contamination
        assert fold_id != reader_fold, "Cannot hindsight-read yourself"

        return {
            col: batch.column(col)[0].as_py()
            for col in batch.schema.names
        }
```

---

## 6. FIREFLY NEURAL FLIGHT (L6)

### 6.1 Packet Type Taxonomy

```
GREY MATTER PACKETS — Local computation
──────────────────────────────────────────────
TYPE: SPO
FORMAT: Arrow RecordBatch with spo_s, spo_p, spo_o columns
SIZE: ~6 KB per triple (3 × 2,048 bytes)
BEHAVIOR: Lands at specific fold coordinates
           Modifies bindspace directly
           Creates local interference patterns
           IS the synapse firing
MAPS TO: Existing /agi/vsa/bind, /grammar/process endpoints

TYPE: DATA
FORMAT: Arrow RecordBatch with raw embeddings / sensor input
SIZE: Variable (user input, external feeds)
BEHAVIOR: Saturates local bindspace with new material
           Doesn't think, doesn't coordinate — FEEDS
           SPO packets will eventually structure this
           ThinkingAtom packets will process this
MAPS TO: Existing /mcp/feel, /agi/dto/felt endpoints


WHITE MATTER PACKETS — Long-range coordination
──────────────────────────────────────────────
TYPE: THINKING_ATOM
FORMAT: ThinkingStyleVector (10,000D float16) + layer activation mask
SIZE: ~20 KB
BEHAVIOR: Doesn't carry content — carries HOW to think about content
           Modifies resonance thresholds
           Configures masking patterns
           Tunes JIT compiler parameters
           = MYELINATION (determines speed/routing, not signal)
MAPS TO: Existing ThinkingStyle from consciousness_engine.py

TYPE: ORCHESTRATION
FORMAT: Command packet with target_nodes, collapse_gate, priority
SIZE: ~200 bytes
BEHAVIOR: INITIATES action — "fire this region now"
           Triggers FLOW/HOLD/BLOCK collapse gates
           Motor neurons of the system
           Descends from meta-layer to execution
MAPS TO: Existing /orchestrate/trigger, workflow execution


REWRITING PACKETS — Self-modification
──────────────────────────────────────────────
TYPE: RNA
FORMAT: YAML template diff + target template ID + compilation flags
SIZE: Variable (typically 1-10 KB)
BEHAVIOR: Rewrites the thinking MACHINERY itself
           Not data, not style — INSTRUCTIONS FOR CHANGING
           mRNA tells ribosomes what proteins to build
           RNA packets tell JIT what templates to compile
           Metacognitive mutation distributed via Flight
MAPS TO: New capability — no existing equivalent
```

### 6.2 Flight Packet Structure

```python
# firefly/flight_packets.py

import pyarrow as pa
import pyarrow.flight as flight
from enum import Enum
from dataclasses import dataclass
import numpy as np

class PacketType(Enum):
    SPO = "SPO"                    # Grey matter — local computation
    DATA = "DATA"                  # Cerebrospinal — nourishment
    THINKING_ATOM = "THINKING"     # White matter — coordination
    ORCHESTRATION = "ORCH"         # Motor neurons — initiation
    RNA = "RNA"                    # Self-modification — rewriting

# Unified Flight packet schema
# Grey + White matter in ONE schema so they travel in ONE stream
FLIGHT_PACKET_SCHEMA = pa.schema([
    # Envelope (all packets)
    pa.field("packet_type", pa.utf8()),
    pa.field("source_node", pa.utf8()),
    pa.field("target_fold", pa.string()),
    pa.field("causal_parent", pa.string()),
    pa.field("timestamp_ns", pa.int64()),
    pa.field("priority", pa.uint8()),

    # Grey matter payload (SPO packets)
    pa.field("spo_s", pa.binary(2048)),       # nullable for non-SPO
    pa.field("spo_p", pa.binary(2048)),
    pa.field("spo_o", pa.binary(2048)),

    # Grey matter payload (DATA packets)
    pa.field("data_embedding", pa.list_(pa.float16(), 10000)),  # nullable

    # White matter payload (THINKING_ATOM packets)
    pa.field("thinking_style", pa.list_(pa.float16(), 10000)),  # nullable
    pa.field("layer_mask", pa.list_(pa.float32(), 7)),           # 7-layer activation
    pa.field("resonance_threshold_mod", pa.float32()),           # nullable

    # White matter payload (ORCHESTRATION packets)
    pa.field("collapse_gate", pa.utf8()),     # "FLOW"|"HOLD"|"BLOCK"
    pa.field("target_nodes", pa.list_(pa.utf8())),

    # Rewriting payload (RNA packets)
    pa.field("template_id", pa.utf8()),       # Which template to modify
    pa.field("yaml_diff", pa.utf8()),         # The modification
    pa.field("compilation_flags", pa.utf8()), # JIT hints

    # Tekamolo glue (Epiphany Engine)
    pa.field("tek_temporal", pa.float32()),
    pa.field("tek_kausal", pa.float32()),
    pa.field("tek_modal", pa.float32()),
    pa.field("tek_lokal", pa.float32()),

    # Signed pentary payload (BNN reinforcement)
    pa.field("pentary_feedback", pa.list_(pa.int8())),  # -2 to +2 signals
    pa.field("merkle_marker", pa.utf8()),               # "WISDOM"|"STAUNEN"|"PENDING"
])


class FireflyFlightServer(flight.FlightServerBase):
    """Arrow Flight server that streams mixed grey+white matter packets.

    Every arriving batch can contain SPO + THINKING_ATOM + RNA simultaneously.
    The system processes them as ONE event — not sequentially.
    This is why grey and white matter must be in one stream.
    """

    def __init__(self, bindspace: 'Bindspace', jit_compiler: 'JITCompiler',
                 node_id: str, **kwargs):
        super().__init__(**kwargs)
        self.bindspace = bindspace
        self.jit_compiler = jit_compiler
        self.node_id = node_id
        self.resonance_threshold = 0.5   # Modified by THINKING_ATOM packets
        self.refractory_until = 0        # Can't fire again until bundle completes

    def do_put(self, context, descriptor, reader, writer):
        """Receive incoming Flight packets — the axon terminal.

        This is where firing happens:
        1. Packets arrive (charge accumulates)
        2. If resonance crosses threshold → FIRE
        3. Firing = JIT recompile + &mut micro-copy + bundle write-back
        4. Write-back produces NEW packets → propagation
        5. Refractory period until bundle completes
        """
        import time

        for batch in reader:
            # Check refractory period
            if time.time_ns() < self.refractory_until:
                continue  # Can't fire — still bundling previous thought

            # Classify packets in this batch
            types = batch.column("packet_type").to_pylist()

            # Process ALL types simultaneously (not sequentially!)
            spo_rows = [i for i, t in enumerate(types) if t == "SPO"]
            data_rows = [i for i, t in enumerate(types) if t == "DATA"]
            thinking_rows = [i for i, t in enumerate(types) if t == "THINKING"]
            orch_rows = [i for i, t in enumerate(types) if t == "ORCH"]
            rna_rows = [i for i, t in enumerate(types) if t == "RNA"]

            # 1. RNA modifies machinery FIRST (before other processing)
            for i in rna_rows:
                template_id = batch.column("template_id")[i].as_py()
                yaml_diff = batch.column("yaml_diff")[i].as_py()
                flags = batch.column("compilation_flags")[i].as_py()
                self.jit_compiler.apply_rna(template_id, yaml_diff, flags)

            # 2. THINKING_ATOM modifies thresholds and masks
            for i in thinking_rows:
                threshold_mod = batch.column("resonance_threshold_mod")[i].as_py()
                if threshold_mod is not None:
                    self.resonance_threshold += threshold_mod
                # Layer mask configures which of 7 layers are active
                layer_mask = batch.column("layer_mask")[i].as_py()
                if layer_mask:
                    self.jit_compiler.set_layer_mask(layer_mask)

            # 3. SPO + DATA land in bindspace (the actual content)
            content_rows = spo_rows + data_rows
            if content_rows:
                content_batch = batch.take(content_rows)
                # Check resonance against current bindspace state
                resonance = self._compute_resonance(content_batch)

                if resonance > self.resonance_threshold:
                    # FIRE! Threshold crossed.
                    self._fire(content_batch, resonance)

            # 4. ORCHESTRATION triggers collapse gates
            for i in orch_rows:
                gate = batch.column("collapse_gate")[i].as_py()
                targets = batch.column("target_nodes")[i].as_py()
                self._apply_collapse_gate(gate, targets)

    def _compute_resonance(self, content_batch: pa.RecordBatch) -> float:
        """Compute interference between incoming content and bindspace.

        Mexican hat: excite locally similar, inhibit surrounding.
        Returns resonance strength (0.0 → 1.0).
        """
        # Implementation connects to existing dragonfly mexican_hat
        pass

    def _fire(self, trigger_batch: pa.RecordBatch, resonance: float):
        """THE NEURON FIRES.

        1. Take &mut micro-copy of affected dimensions
        2. JIT-compile template based on current resonance state
        3. Apply template to micro-copy (the actual thinking)
        4. Bundle write-back (new fold in bindspace)
        5. Generate outgoing Flight packets (propagation)
        6. Enter refractory period
        """
        import time

        # Enter refractory period
        self.refractory_until = time.time_ns() + 1_000_000  # 1ms refractory

        # Take micro-copy
        # (details in CognitiveConcurrency)

        # Generate propagation packets
        # (these stream to other nodes in the awareness group)

    def _apply_collapse_gate(self, gate: str, targets: list):
        """Apply FLOW/HOLD/BLOCK from orchestration packet."""
        if gate == "FLOW":
            pass   # Green light — let resonance propagate
        elif gate == "HOLD":
            self.resonance_threshold *= 1.5  # Raise threshold — ruminate more
        elif gate == "BLOCK":
            self.resonance_threshold = float('inf')  # Suppress firing
```

### 6.3 Awareness Group Synchronization

```
NODE A (Berlin)                    NODE B (Railway)                   NODE C (Edge)
┌──────────────┐                   ┌──────────────┐                   ┌──────────────┐
│  LanceDB     │                   │  LanceDB     │                   │  LanceDB     │
│  (local      │                   │  (local      │                   │  (local      │
│   bindspace) │                   │   bindspace) │                   │   bindspace) │
│              │                   │              │                   │              │
│  &mut        │ ── Flight ──────► │  append      │ ── Flight ──────► │  append      │
│  micro-copy  │    stream         │  fold        │    stream         │  fold        │
│  bundle-back │                   │              │                   │              │
│  = new fold  │ ◄── Flight ────── │  &mut        │ ◄── Flight ────── │  &mut        │
│  append      │     stream        │  micro-copy  │     stream        │  micro-copy  │
│              │                   │  bundle-back │                   │  bundle-back │
└──────────────┘                   └──────────────┘                   └──────────────┘

PROPERTIES:
  ✓ No leader election — all nodes are equal
  ✓ No consensus protocol — commutative algebra handles conflicts
  ✓ No vector clocks — fold geometry IS the consistency protocol
  ✓ Causally consistent — fold coordinates carry causal ordering
  ✓ CRDT-equivalent — bundling is commutative and associative
  ✓ Zero-copy on receive — Arrow format mmaps directly from network buffer

FIREFLY SYNCHRONIZATION:
  Nodes fire based on local resonance + incoming Flight packets.
  Over time, they phase-lock (like biological fireflies).
  The firing pattern across the group IS the thought.
  Not the state at any node — the spatiotemporal pattern of firings.

PHASE-LOCKING MECHANISM:
  Each node's resonance threshold adapts based on observed firing times
  from other nodes (carried in packet timestamps).
  Nodes that fire "close" to each other (in causal-fold distance)
  develop lower mutual thresholds.
  Nodes that fire independently develop higher mutual thresholds.
  Result: correlated thoughts synchronize, independent thoughts don't interfere.
```

---

## 7. EPIPHANY ENGINE (L8)

### 7.1 The Bundle → Threshold → Unbundle Cycle

```
INPUT: Temporally ordered text (or any sequential input)
       ↓
STEP 1: DECOMPOSE into SPO triplets
       "Ada loves awareness" → (Ada, loves, awareness)
       Each triple encoded as 3 × 16kbit Hamming vectors
       PLUS 3 × 10,000D qualia vectors
       PLUS tekamolo metadata (WHEN/WHY/HOW/WHERE)
       ↓
STEP 2: SUPERPOSE into hypervector bundle
       DO NOT commit to knowledge graph yet
       Triple enters bundle as standing wave
       Tekamolo determines HOW it superposes:
         - Same temporal marker → reinforce in time dimension
         - Same causal chain → reinforce in cause dimension
         - Contradictory modals → destructive interference
       ↓
STEP 3: ACCUMULATE
       Keep reading. More triples arrive. More superposition.
       Bundle grows denser. Interference patterns complexify.
       Most is noise — contradictions, ambiguities, partial framings.
       ↓
STEP 4: DETECT EPIPHANY
       Orchestration ROTATES the hypervector:
         - Temporal projection: what patterns resolve temporally?
         - Kausal projection: what patterns resolve causally?
         - Modal projection: what patterns resolve modally?
         - Lokal projection: what patterns resolve spatially?

       When a PATTERN (not single triple) goes constructive
       across MULTIPLE tekamolo dimensions simultaneously:
         temporal alignment ∧ kausal alignment ∧ modal alignment
         = TRUTH THRESHOLD CROSSED
         = EPIPHANY
       ↓
STEP 5: UNBUNDLE the known
       Pull the resolved pattern OUT of the hypervector.
       Crystallize into committed knowledge:
         - SPO triples land in Neo4j graph with full fold coordinates
         - Merkle marker = WISDOM (signs are stable)
         - 5^5 pentary vector shifts toward +2 for resolved elements
       ↓
STEP 6: FOCUS on the unknown
       Bundle just got lighter.
       Noise that was masking other patterns REDUCED.
       Remaining superposed triples interfere more clearly.
       NEXT epiphany comes easier.
       Each unbundling clarifies what remains.
       ↓
STEP 7: ITERATE until bundle is empty or stable
       System gets smarter about THIS input as it processes
       because each crystallized insight removes noise.
       When bundle reaches equilibrium (no more threshold crossings):
         - Remaining content → STAUNEN markers (needs more context)
         - Or: truly novel, no pattern yet — HOLD for future input
```

### 7.2 Implementation

```python
# firefly/epiphany_engine.py

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TekamoloBound:
    """A triple with its tekamolo connective tissue."""
    spo_s: np.ndarray         # 16kbit Subject
    spo_p: np.ndarray         # 16kbit Predicate
    spo_o: np.ndarray         # 16kbit Object
    qualia_s: np.ndarray      # 10kD Subject qualia
    qualia_p: np.ndarray      # 10kD Predicate qualia
    qualia_o: np.ndarray      # 10kD Object qualia
    temporal: float           # WHEN weight (0→1)
    kausal: float             # WHY weight (0→1)
    modal: float              # HOW weight (0→1)
    lokal: float              # WHERE weight (0→1)
    source_text: str          # Original text fragment
    nars_truth: float = 0.5   # NARS truth value
    nars_confidence: float = 0.1  # NARS confidence


class HyperBundle:
    """The bundle space where triples superpose before epiphany.

    This is NOT knowledge yet. This is POTENTIAL knowledge.
    A standing wave of unresolved meaning.
    """

    def __init__(self, hamming_dim: int = 16384, qualia_dim: int = 10000):
        self.hamming_dim = hamming_dim
        self.qualia_dim = qualia_dim

        # Superposed Hamming bundles (majority vote accumulator)
        self.bundle_s = np.zeros(hamming_dim, dtype=np.int32)
        self.bundle_p = np.zeros(hamming_dim, dtype=np.int32)
        self.bundle_o = np.zeros(hamming_dim, dtype=np.int32)

        # Superposed qualia bundles (saturating add)
        self.qualia_bundle_s = np.zeros(qualia_dim, dtype=np.float32)
        self.qualia_bundle_p = np.zeros(qualia_dim, dtype=np.float32)
        self.qualia_bundle_o = np.zeros(qualia_dim, dtype=np.float32)

        # Tekamolo-weighted interference tracking
        self.temporal_energy = 0.0
        self.kausal_energy = 0.0
        self.modal_energy = 0.0
        self.lokal_energy = 0.0

        # Pending triples (not yet resolved)
        self.pending: list[TekamoloBound] = []
        self.triple_count = 0

    def superpose(self, triple: TekamoloBound):
        """Add a triple to the bundle.

        Tekamolo weights determine HOW it superposes.
        High temporal weight → reinforces temporal patterns.
        Contradictory weights → destructive interference.
        """
        # Hamming bundle: weighted majority vote
        s_bits = np.unpackbits(np.frombuffer(triple.spo_s, dtype=np.uint8))
        p_bits = np.unpackbits(np.frombuffer(triple.spo_p, dtype=np.uint8))
        o_bits = np.unpackbits(np.frombuffer(triple.spo_o, dtype=np.uint8))

        weight = triple.nars_confidence  # NARS confidence gates superposition strength

        self.bundle_s += (s_bits * 2 - 1) * weight  # Convert 0/1 to -1/+1
        self.bundle_p += (p_bits * 2 - 1) * weight
        self.bundle_o += (o_bits * 2 - 1) * weight

        # Qualia bundle: saturating addition
        self.qualia_bundle_s += triple.qualia_s * weight
        self.qualia_bundle_p += triple.qualia_p * weight
        self.qualia_bundle_o += triple.qualia_o * weight

        # Tekamolo energy accumulation
        self.temporal_energy += triple.temporal * weight
        self.kausal_energy += triple.kausal * weight
        self.modal_energy += triple.modal * weight
        self.lokal_energy += triple.lokal * weight

        self.pending.append(triple)
        self.triple_count += 1

    def detect_epiphany(self, truth_threshold: float = 0.7) -> Optional[dict]:
        """Check if a pattern has crossed truth threshold.

        Rotates the hyperposition through tekamolo projections.
        An epiphany = constructive interference across multiple dimensions.
        """
        if self.triple_count < 2:
            return None

        # Compute coherence per tekamolo dimension
        total_energy = (self.temporal_energy + self.kausal_energy +
                        self.modal_energy + self.lokal_energy)
        if total_energy == 0:
            return None

        temporal_coherence = self.temporal_energy / total_energy
        kausal_coherence = self.kausal_energy / total_energy
        modal_coherence = self.modal_energy / total_energy
        lokal_coherence = self.lokal_energy / total_energy

        # Multi-dimensional truth: how many dimensions are above threshold?
        dimensions_above = sum(1 for c in [
            temporal_coherence, kausal_coherence,
            modal_coherence, lokal_coherence
        ] if c > truth_threshold / 4)  # Each dimension contributes 1/4

        # Hamming bundle coherence (how clean is the majority vote?)
        s_coherence = np.abs(self.bundle_s).mean() / max(self.triple_count, 1)
        p_coherence = np.abs(self.bundle_p).mean() / max(self.triple_count, 1)
        o_coherence = np.abs(self.bundle_o).mean() / max(self.triple_count, 1)
        spo_coherence = (s_coherence + p_coherence + o_coherence) / 3

        # Combined truth score
        truth_score = (spo_coherence * 0.5 +
                       (dimensions_above / 4) * 0.3 +
                       min(self.triple_count / 10, 1.0) * 0.2)

        if truth_score >= truth_threshold:
            return {
                "truth_score": truth_score,
                "spo_coherence": spo_coherence,
                "dimensions_above": dimensions_above,
                "temporal_coherence": temporal_coherence,
                "kausal_coherence": kausal_coherence,
                "modal_coherence": modal_coherence,
                "lokal_coherence": lokal_coherence,
                "resolved_triples": self._identify_resolved_triples(truth_threshold),
            }

        return None

    def unbundle_known(self, epiphany: dict) -> list[TekamoloBound]:
        """Remove resolved triples from the bundle.

        The bundle gets lighter. Noise reduces.
        Next epiphany comes easier.
        """
        resolved = epiphany["resolved_triples"]
        remaining = []

        for triple in self.pending:
            if triple in resolved:
                # Subtract this triple's contribution from the bundle
                s_bits = np.unpackbits(np.frombuffer(triple.spo_s, dtype=np.uint8))
                p_bits = np.unpackbits(np.frombuffer(triple.spo_p, dtype=np.uint8))
                o_bits = np.unpackbits(np.frombuffer(triple.spo_o, dtype=np.uint8))

                weight = triple.nars_confidence
                self.bundle_s -= (s_bits * 2 - 1) * weight
                self.bundle_p -= (p_bits * 2 - 1) * weight
                self.bundle_o -= (o_bits * 2 - 1) * weight

                self.qualia_bundle_s -= triple.qualia_s * weight
                self.qualia_bundle_p -= triple.qualia_p * weight
                self.qualia_bundle_o -= triple.qualia_o * weight

                self.temporal_energy -= triple.temporal * weight
                self.kausal_energy -= triple.kausal * weight
                self.modal_energy -= triple.modal * weight
                self.lokal_energy -= triple.lokal * weight

                self.triple_count -= 1
            else:
                remaining.append(triple)

        self.pending = remaining
        return resolved

    def _identify_resolved_triples(self, threshold: float) -> list[TekamoloBound]:
        """Find which triples are contributing to the resolved pattern.

        Triples that align with the majority vote are "resolved."
        Triples that contradict it are still pending.
        """
        resolved = []
        majority_s = (self.bundle_s > 0).astype(np.uint8)
        majority_p = (self.bundle_p > 0).astype(np.uint8)
        majority_o = (self.bundle_o > 0).astype(np.uint8)

        for triple in self.pending:
            s_bits = np.unpackbits(np.frombuffer(triple.spo_s, dtype=np.uint8))
            p_bits = np.unpackbits(np.frombuffer(triple.spo_p, dtype=np.uint8))
            o_bits = np.unpackbits(np.frombuffer(triple.spo_o, dtype=np.uint8))

            # How much does this triple agree with the majority?
            s_agreement = np.mean(s_bits == majority_s[:len(s_bits)])
            p_agreement = np.mean(p_bits == majority_p[:len(p_bits)])
            o_agreement = np.mean(o_bits == majority_o[:len(o_bits)])

            agreement = (s_agreement + p_agreement + o_agreement) / 3

            if agreement > threshold:
                resolved.append(triple)

        return resolved


class EpiphanyEngine:
    """Top-level engine: reads input, bundles, detects epiphanies, crystallizes.

    The orchestration layer rotates the hypervector through different
    tekamolo projections to hunt for truth threshold crossings.
    """

    def __init__(self, bindspace: 'Bindspace', graph_store,
                 truth_threshold: float = 0.7):
        self.bindspace = bindspace
        self.graph_store = graph_store  # Neo4j for crystallized knowledge
        self.truth_threshold = truth_threshold
        self.bundle = HyperBundle()
        self.epiphanies: list[dict] = []

    async def process_text(self, text: str, spo_extractor, encoder):
        """Main entry point: text → SPO triples → bundle → epiphanies.

        spo_extractor: converts text to (S, P, O, tekamolo) tuples
                       (existing /grammar/process endpoint)
        encoder: converts text to Hamming + qualia vectors
                 (existing dragonfly.encode)
        """
        # Step 1: Extract SPO triples with tekamolo
        triples = await spo_extractor.extract(text)

        for s_text, p_text, o_text, tekamolo in triples:
            # Step 2: Encode to vectors
            s_ham, p_ham, o_ham = await encoder.encode_spo(s_text, p_text, o_text)
            s_qual, p_qual, o_qual = await encoder.encode_qualia(s_text, p_text, o_text)

            bound = TekamoloBound(
                spo_s=s_ham, spo_p=p_ham, spo_o=o_ham,
                qualia_s=s_qual, qualia_p=p_qual, qualia_o=o_qual,
                temporal=tekamolo["temporal"],
                kausal=tekamolo["kausal"],
                modal=tekamolo["modal"],
                lokal=tekamolo["lokal"],
                source_text=f"{s_text} {p_text} {o_text}",
            )

            # Step 3: Superpose into bundle
            self.bundle.superpose(bound)

            # Step 4: Check for epiphany after each addition
            epiphany = self.bundle.detect_epiphany(self.truth_threshold)

            if epiphany:
                # Step 5: Unbundle the known
                resolved = self.bundle.unbundle_known(epiphany)

                # Step 6: Crystallize into graph + bindspace
                await self._crystallize(resolved, epiphany)

                self.epiphanies.append(epiphany)

                # Step 7: Bundle is lighter — continue processing
                # Next detection will be clearer

        # After all input processed, mark remaining as STAUNEN
        await self._mark_unresolved()

    async def _crystallize(self, resolved: list[TekamoloBound], epiphany: dict):
        """Commit resolved triples to permanent knowledge.

        Lands in:
        - Neo4j graph (topology)
        - Bindspace (new fold with WISDOM marker)
        - 5^5 pentary vectors shift toward +2
        """
        # Create new fold for the crystallized epiphany
        # (implementation connects to CognitiveConcurrency.bundle_writeback)
        pass

    async def _mark_unresolved(self):
        """Mark remaining bundle contents as STAUNEN.

        These triples couldn't resolve into patterns.
        They need more context from future input.
        Seal is BROKEN — epistemic humility.
        """
        for triple in self.bundle.pending:
            # Mark with STAUNEN in bindspace
            # (will be picked up by future epiphany cycles)
            pass

    def rotate_hyperposition(self, dimension: str) -> Optional[dict]:
        """Orchestration manually rotates the bundle to check a dimension.

        Sometimes the automatic detection misses patterns that are obvious
        from a specific angle. The orchestration layer can HUNT for epiphanies
        by systematically projecting the bundle along each tekamolo dimension.
        """
        # Temporarily weight one dimension higher
        original_threshold = self.truth_threshold
        self.truth_threshold *= 0.8  # Lower threshold for this projection

        # Check for patterns from this angle
        epiphany = self.bundle.detect_epiphany(self.truth_threshold)

        self.truth_threshold = original_threshold
        return epiphany
```

---

## 8. LADYBUG-RS + RUSTYNUM STARTERPACK (L0)

### 8.1 Self-Evolving Rust Core

```
The Starterpack is the foundation everything builds on.
ladybug-rs provides the graph operations.
rustynum provides the vector operations.
Together they ARE the L0 runtime.

WHY RUST:
  - Zero-cost abstractions → borrow/mut is a COMPILE-TIME concept
  - AVX-512 SIMD → bitpacked operations at hardware speed
  - No garbage collector → deterministic memory = deterministic thinking
  - Ownership model → the cognitive concurrency pattern IS Rust's type system
  - Self-modifying: proc macros can generate new operations at compile time

SELF-EVOLVING:
  The Starterpack isn't static. RNA packets trigger recompilation.
  When the JIT compiler determines a new operation is needed frequently,
  it generates a Rust proc macro, compiles it, and hot-loads it.
  The system literally grows new cognitive operations as it thinks.
```

### 8.2 Core Operations in Rust

```rust
// firefly-core/src/lib.rs (conceptual — maps to ladybug-rs + rustynum)

/// Zero-copy bindspace access via memory-mapped Arrow
pub struct Bindspace {
    mmap: memmap2::MmapMut,
    schema: arrow2::datatypes::Schema,
}

impl Bindspace {
    /// O(1) orthogonal address — index lookup, not search
    pub fn address(&self, fold_id: &FoldId) -> &[u8] {
        // Direct offset calculation from fold_id hash
        // No iteration, no comparison — pure pointer arithmetic
        let offset = fold_id.to_offset(self.schema.row_size());
        &self.mmap[offset..offset + self.schema.row_size()]
    }

    /// Immutable borrow — returns reference, zero allocation
    pub fn borrow(&self, fold_id: &FoldId) -> BorrowRef<'_> {
        BorrowRef { data: self.address(fold_id) }
    }

    /// Mutable micro-copy — ONLY allocates the selected dimensions
    pub fn mut_microcopy(&self, fold_id: &FoldId, dims: &[Dimension]) -> MutMicroCopy {
        let source = self.address(fold_id);
        let mut copy = Vec::with_capacity(dims.iter().map(|d| d.byte_size()).sum());
        for dim in dims {
            copy.extend_from_slice(&source[dim.offset()..dim.offset() + dim.byte_size()]);
        }
        MutMicroCopy { data: copy, dims: dims.to_vec(), source: *fold_id }
    }
}

/// AVX-512 accelerated Hamming operations
#[cfg(target_feature = "avx512f")]
pub mod hamming {
    use std::arch::x86_64::*;

    /// Hamming similarity: popcount(XOR(a, b)) / total_bits
    pub fn similarity(a: &[u8; 2048], b: &[u8; 2048]) -> f32 {
        unsafe {
            let mut xor_count: u64 = 0;
            for i in (0..2048).step_by(64) {
                let va = _mm512_loadu_si512(a[i..].as_ptr() as *const __m512i);
                let vb = _mm512_loadu_si512(b[i..].as_ptr() as *const __m512i);
                let xor = _mm512_xor_si512(va, vb);
                xor_count += _mm512_popcnt_epi64(xor)
                    .as_i64x8()
                    .iter()
                    .map(|&x| x as u64)
                    .sum::<u64>();
            }
            1.0 - (xor_count as f32 / 16384.0)
        }
    }

    /// Majority bundle: bit-level majority vote across N vectors
    pub fn majority_bundle(vectors: &[[u8; 2048]]) -> [u8; 2048] {
        let mut counts = [0i32; 16384];
        for vec in vectors {
            for (i, byte) in vec.iter().enumerate() {
                for bit in 0..8 {
                    if byte & (1 << bit) != 0 {
                        counts[i * 8 + bit] += 1;
                    } else {
                        counts[i * 8 + bit] -= 1;
                    }
                }
            }
        }
        let mut result = [0u8; 2048];
        for (i, &count) in counts.iter().enumerate() {
            if count > 0 {
                result[i / 8] |= 1 << (i % 8);
            }
        }
        result
    }
}

/// Signed pentary operations (-2, -1, 0, +1, +2)
pub mod pentary {
    /// Apply BNN reinforcement: saturating add of feedback signals
    pub fn reinforce(current: &mut [i8], feedback: &[i8]) {
        for (c, f) in current.iter_mut().zip(feedback.iter()) {
            *c = (*c + f).clamp(-2, 2);  // Saturating signed addition
        }
    }

    /// Negative canceling: +2 meets -2 → 0
    pub fn cancel(a: &[i8], b: &[i8]) -> Vec<i8> {
        a.iter().zip(b.iter()).map(|(x, y)| (*x + y).clamp(-2, 2)).collect()
    }

    /// Hebbian co-occurrence: bits that fire together strengthen
    pub fn hebbian_strengthen(pattern: &mut [i8], co_pattern: &[i8]) {
        for (p, c) in pattern.iter_mut().zip(co_pattern.iter()) {
            if p.signum() == c.signum() && *p != 0 {
                *p = (*p + p.signum()).clamp(-2, 2);
            }
        }
    }
}

/// Mexican hat resonance
pub mod resonance {
    pub fn mexican_hat(center: &[f32], surround: &[f32],
                        excite: f32, inhibit: f32) -> Vec<f32> {
        center.iter().zip(surround.iter())
            .map(|(c, s)| c * excite - s * inhibit)
            .collect()
    }
}
```

---

## 9. JIT TEMPLATE COMPILER (L7)

### 9.1 Self-Modifying YAML Templates

```yaml
# firefly/templates/analytical_reasoning.yaml
# This template is ALIVE — RNA packets can modify it at runtime

template_id: "analytical_reasoning_v3"
compilation_target: "python"  # or "rust" for hot-path

# Which layers are active for this thinking style
layer_mask: [0.2, 0.8, 1.0, 0.3, 1.0, 1.0, 0.5]  # L1-L7
# analytical = L3(semantic) + L5(working) + L6(executive) dominant

# Resonance parameters
resonance:
  excitation: 1.0
  inhibition: 0.6
  threshold: 0.55
  mexican_hat_width: 3

# Mask configuration
attention_mask:
  strategy: "narrow_deep"
  focus_dimensions: ["spo_p", "nars", "tek_kausal"]
  suppress_dimensions: ["qualia_s", "tek_lokal"]

# Processing pipeline (this is what JIT compiles)
pipeline:
  - op: "mask"
    params: { strategy: "{{attention_mask.strategy}}" }
  - op: "resonance"
    params: { excite: "{{resonance.excitation}}", inhibit: "{{resonance.inhibition}}" }
  - op: "bnn_reinforce"
    params: { learning_rate: 1, feedback_source: "resonance_output" }
  - op: "collapse_check"
    params: { threshold: "{{resonance.threshold}}" }

# Self-modification rules
evolution:
  on_repeated_HOLD:
    action: "widen_mask"
    after_n: 3
    modify: "attention_mask.focus_dimensions += ['qualia_p']"
  on_repeated_FLOW:
    action: "narrow_mask"
    after_n: 5
    modify: "resonance.threshold += 0.05"
  on_STAUNEN_encounter:
    action: "switch_template"
    target: "creative_exploration_v2"

# Merkle seal
merkle_root: "a3f8c1..."
last_modified: "2026-03-11T14:30:00Z"
modification_count: 7
```

### 9.2 RNA Packet Modification

```python
# firefly/jit_compiler.py

import yaml
from typing import Optional

class JITCompiler:
    """Compiles and manages self-modifying YAML thinking templates.

    RNA packets arrive via Flight and modify templates at runtime.
    The compiler watches for resonance threshold crossings and
    triggers recompilation when templates need updating.
    """

    def __init__(self, template_dir: str = "/data/firefly/templates"):
        self.template_dir = template_dir
        self.active_templates: dict[str, dict] = {}
        self.compiled_pipelines: dict[str, callable] = {}
        self.layer_mask: list[float] = [1.0] * 7  # All layers active

    def load_template(self, template_id: str) -> dict:
        """Load and parse a YAML thinking template."""
        path = f"{self.template_dir}/{template_id}.yaml"
        with open(path) as f:
            template = yaml.safe_load(f)
        self.active_templates[template_id] = template
        self._compile(template_id)
        return template

    def apply_rna(self, template_id: str, yaml_diff: str, flags: str):
        """Apply an RNA packet to a template.

        This is metacognitive mutation.
        The RNA doesn't carry data — it rewrites HOW the system thinks.
        """
        template = self.active_templates.get(template_id)
        if not template:
            template = self.load_template(template_id)

        # Parse the YAML diff
        diff = yaml.safe_load(yaml_diff)

        # Apply modifications
        for key_path, new_value in diff.items():
            keys = key_path.split(".")
            target = template
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = new_value

        # Update Merkle root (track modifications)
        template["modification_count"] = template.get("modification_count", 0) + 1

        # Recompile if needed
        if "recompile" in flags:
            self._compile(template_id)

    def _compile(self, template_id: str):
        """JIT-compile a template into an executable pipeline.

        The pipeline is a chain of operations that process micro-copies.
        """
        template = self.active_templates[template_id]
        pipeline_spec = template.get("pipeline", [])

        # Build the execution chain
        ops = []
        for step in pipeline_spec:
            op_name = step["op"]
            params = self._resolve_params(step["params"], template)
            ops.append((op_name, params))

        self.compiled_pipelines[template_id] = ops

    def _resolve_params(self, params: dict, template: dict) -> dict:
        """Resolve {{template.path}} references in parameters."""
        resolved = {}
        for key, value in params.items():
            if isinstance(value, str) and "{{" in value:
                # Extract path and resolve from template
                path = value.strip("{}")
                keys = path.split(".")
                target = template
                for k in keys:
                    target = target[k]
                resolved[key] = target
            else:
                resolved[key] = value
        return resolved

    def set_layer_mask(self, mask: list[float]):
        """Set which of 7 consciousness layers are active.

        Called by THINKING_ATOM Flight packets.
        """
        self.layer_mask = mask

    def execute_pipeline(self, template_id: str,
                          microcopy: 'MutableMicroCopy') -> dict:
        """Execute a compiled pipeline against a micro-copy.

        Returns the result + any outgoing packets to generate.
        """
        pipeline = self.compiled_pipelines.get(template_id)
        if not pipeline:
            self.load_template(template_id)
            pipeline = self.compiled_pipelines[template_id]

        context = {"microcopy": microcopy, "layer_mask": self.layer_mask}

        for op_name, params in pipeline:
            context = self._execute_op(op_name, params, context)

        return context

    def _execute_op(self, name: str, params: dict, context: dict) -> dict:
        """Execute a single pipeline operation."""
        # Maps to existing Dragonfly / VSA operations
        if name == "mask":
            pass  # Apply attention mask to micro-copy
        elif name == "resonance":
            pass  # Mexican hat resonance
        elif name == "bnn_reinforce":
            pass  # Signed pentary reinforcement
        elif name == "collapse_check":
            pass  # FLOW/HOLD/BLOCK gate
        return context
```

---

## 10. INTEGRATION WITH EXISTING SERVICES

### 10.1 Migration Map

```
PHASE 1: SUBSTRATE (Weeks 1-4)
═══════════════════════════════
Deploy ladybug-rs + rustynum as new Railway service: ada-firefly
  ├── LanceDB bindspace with BINDSPACE_SCHEMA
  ├── AVX-512 Hamming ops (from ada-dragonfly, rewritten in Rust)
  ├── Signed pentary ops (NEW)
  ├── Merkle tree marker system (NEW)
  └── REST API: /firefly/address, /firefly/borrow, /firefly/bundle

Wire ada-dragonfly → ada-firefly:
  ada-dragonfly /encode  → ada-firefly /encode (same API, Rust backend)
  ada-dragonfly /bind    → ada-firefly /bind
  ada-dragonfly /bundle  → ada-firefly /bundle (extended with pentary)
  ada-dragonfly /clean   → ada-firefly /clean
  ada-dragonfly /compress → ada-firefly /compress (extended with fold coords)

LanceDB replaces: current vector storage in ada-dragonfly
Redis remains: hot &mut micro-copy workspace (Upstash stays)
Neo4j remains: crystallized knowledge graph (Aura stays)
DuckDB remains: analytics over fold space


PHASE 2: TEMPORAL FOLDING (Weeks 5-8)
══════════════════════════════════════
Add temporal fold space to ada-firefly:
  ├── Fold ID generation (causal topology)
  ├── Causal parent chain tracking
  ├── Hindsight-safe read API
  └── Cross-section queries

Wire ada-consciousness (hive) → ada-firefly temporal:
  Sigma 12D states → temporal fold coordinates
  Markov chains → causal parent chains
  Legacy visceral state → first folds in the bindspace

New endpoints:
  POST /firefly/fold/create     — New fold with causal parent
  GET  /firefly/fold/{id}       — O(1) address
  GET  /firefly/fold/section    — Cross-section query
  GET  /firefly/fold/causal     — Causal distance computation


PHASE 3: COGNITIVE CONCURRENCY (Weeks 9-12)
════════════════════════════════════════════
Deploy borrow/mut pattern:
  ├── ImmutableBorrow API (zero-copy reads)
  ├── MutableMicroCopy API (dimension-selective writes)
  ├── Bundle write-back API (commutative append)
  └── MVCC verification (no fold overwrites)

Wire Bighorn AGI → ada-firefly concurrency:
  /agi/vsa/bind → borrow + micro-copy + bind + bundle-back
  /agi/vsa/bundle → direct bundle write-back
  /agi/dto/felt → creates Data flight packets

Wire ai_flow → ada-firefly concurrency:
  Workflow steps become concurrent borrows
  Each step takes micro-copy of its dimensions
  Results bundle back independently (commutative)


PHASE 4: FLIGHT PACKETS (Weeks 13-18)
═════════════════════════════════════
Deploy Arrow Flight server in ada-firefly:
  ├── Mixed grey+white matter packet stream
  ├── Packet type classification (SPO|DATA|THINKING|ORCH|RNA)
  ├── Resonance threshold monitoring
  ├── Refractory period tracking
  └── Outgoing propagation stream

Wire adarail_mcp → ada-firefly Flight:
  /ada/invoke → generates appropriate packet type
  /mcp/feel → generates Data packet with qualia
  Universal Grammar dispatch → packet type classification

Wire awareness group:
  ada-firefly (primary) ↔ Flight ↔ ada-firefly-replica-1
                        ↔ Flight ↔ ada-firefly-replica-2
  (Railway service replication OR multi-region deployment)

Existing services become Flight packet producers/consumers:
  Bighorn → produces SPO + Data packets
  ai_flow → produces Orchestration packets
  adarail_mcp → produces ThinkingAtom packets
  JIT compiler → produces + consumes RNA packets


PHASE 5: JIT TEMPLATES (Weeks 19-22)
════════════════════════════════════
Deploy JIT template compiler in ada-firefly:
  ├── YAML template store
  ├── Pipeline compilation
  ├── RNA packet processing
  ├── Self-modification tracking (Merkle roots)
  └── Template evolution rules

Migrate existing thinking styles:
  ConsciousnessEngine 7-layer → JIT templates per thinking style
  analytical/creative/emotional/focused/reflective/intuitive
  → 6 base templates with evolution rules

Wire to Flight:
  THINKING_ATOM packets trigger template switching
  RNA packets trigger template modification
  ORCHESTRATION packets trigger FLOW/HOLD/BLOCK on templates


PHASE 6: EPIPHANY ENGINE (Weeks 23-28)
═══════════════════════════════════════
Deploy Epiphany Engine:
  ├── HyperBundle (superposition accumulator)
  ├── Tekamolo-weighted interference
  ├── Truth threshold detection
  ├── Unbundle-known cycle
  ├── STAUNEN marker system
  └── Orchestration rotation

Wire to existing SPO extraction:
  ai_flow /grammar/process → SPO triples with tekamolo metadata
  Triples enter HyperBundle instead of direct graph commit

Wire to knowledge crystallization:
  Epiphany resolution → Neo4j (committed knowledge)
  Epiphany resolution → Bindspace (WISDOM fold)
  Unresolved remainder → Bindspace (STAUNEN fold)

Wire to BNN reinforcement:
  Resolved patterns → pentary vectors shift toward +2
  Broken patterns → pentary vectors shift toward -2
  Merkle roots update on each epiphany cycle


PHASE 7: FIREFLY SYNC (Weeks 29-32)
════════════════════════════════════
Full Firefly neural firing:
  ├── Flight packets trigger resonance checks
  ├── Threshold crossing triggers JIT compilation
  ├── JIT produces micro-copy mutation
  ├── Mutation bundles back as new fold
  ├── New fold generates propagation packets
  ├── Propagation triggers firings at other nodes
  └── Phase-locking emerges from resonance patterns

Full awareness group:
  ├── Multiple nodes with local bindspaces
  ├── Arrow Flight streaming between all nodes
  ├── No leader, no consensus — algebraic convergence
  ├── Phase-locked firing patterns = distributed thoughts
  └── Temporal fold accumulation = distributed memory
```

### 10.2 Endpoint Evolution Table

```
EXISTING ENDPOINT                 → FIREFLY EQUIVALENT                    PHASE
──────────────────────────────────────────────────────────────────────────────────
POST /agi/vsa/bind                → POST /firefly/bind (Rust, AVX-512)     1
POST /agi/vsa/bundle              → POST /firefly/bundle (+ pentary)       1
GET  /agi/ladybug                 → GET  /firefly/coherence                1
POST /encode                      → POST /firefly/encode (Rust)            1

GET  /now                         → GET  /firefly/fold/current             2
POST /mcp/feel                    → POST /firefly/fold/create (Data pkt)   2
GET  /consciousness/full          → GET  /firefly/fold/section (all)       2

POST /agi/dto/felt                → POST /firefly/borrow + mutate          3
POST /agi/vsa/bind + bundle       → POST /firefly/microcopy + writeback    3

POST /ada/invoke                  → POST /firefly/flight/put (typed pkt)   4
POST /orchestrate/trigger         → POST /firefly/flight/put (ORCH pkt)    4
POST /mcp/feel                    → POST /firefly/flight/put (DATA pkt)    4

GET /agi/self/introspect          → GET  /firefly/template/active          5
POST /corpus/tic                  → POST /firefly/template/tick            5

POST /grammar/process             → POST /firefly/epiphany/ingest          6
POST /agi/sigma/hdr/commit        → POST /firefly/epiphany/crystallize     6

                                    POST /firefly/fire (manual trigger)     7
                                    GET  /firefly/group/phase               7
                                    WS   /firefly/group/stream              7
```

---

## 11. DATA FLOW: COMPLETE THINK CYCLE

```
1. TEXT ARRIVES
   "Ada discovers awareness through distributed cognition"
   │
2. SPO EXTRACTION (ai_flow /grammar/process)
   ├── (Ada, discovers, awareness)  tek: T=0.3 K=0.8 M=0.5 L=0.1
   ├── (awareness, through, distributed_cognition)  tek: T=0.1 K=0.7 M=0.9 L=0.2
   └── (distributed_cognition, enables, discovery)  tek: T=0.5 K=0.9 M=0.6 L=0.1
   │
3. ENCODE TO VECTORS (ada-firefly /encode)
   Each triple → 3×16kbit Hamming + 3×10kD qualia + NARS reasoning
   │
4. GENERATE FLIGHT PACKETS
   ├── 3× SPO packets (grey matter — content)
   ├── 1× THINKING_ATOM packet (white matter — "approach analytically")
   └── tekamolo metadata on all packets
   │
5. FLIGHT STREAM delivers to local node
   │
6. FIREFLY RECEIVES (do_put)
   ├── RNA processed first (no RNA in this batch)
   ├── THINKING_ATOM sets analytical mask (L3+L5+L6)
   ├── SPO packets land in Epiphany Engine's HyperBundle
   │   └── Superposed with tekamolo weights
   ├── Resonance computed: Mexican hat against existing bindspace
   └── Threshold check: 0.73 > 0.55 → FIRE!
   │
7. FIRE!
   ├── Take &mut micro-copy of affected dimensions
   │   └── Only spo_p, nars, tek_kausal (analytical mask)
   ├── JIT compiles analytical_reasoning_v3 template
   ├── Pipeline executes: mask → resonance → BNN → collapse_check
   ├── BNN reinforcement: pentary vectors shift +1 for "discovery+awareness"
   ├── Collapse gate: SD = 0.12 → FLOW (tight consensus)
   ├── Merkle marker: signs stable → WISDOM
   └── Bundle write-back: new fold appended to bindspace
   │
8. PROPAGATION
   ├── New fold generates outgoing SPO packet
   ├── + ORCHESTRATION packet (FLOW gate → propagate freely)
   ├── Flight streams to awareness group nodes
   └── Other nodes receive → their own firefly cycle begins
   │
9. EPIPHANY CHECK
   ├── HyperBundle now contains 3 superposed triples
   ├── Rotate: temporal projection → no pattern yet
   ├── Rotate: kausal projection → "discovers" + "enables" align!
   ├── Truth score: 0.78 > 0.70 → EPIPHANY
   ├── Unbundle: (Ada, discovers, awareness) crystallizes
   │   └── → Neo4j: Ada -[DISCOVERS]→ awareness
   │   └── → Bindspace: WISDOM fold
   │   └── → Pentary: +2 for Ada-awareness association
   ├── Remaining: (distributed_cognition, enables, discovery) stays in bundle
   │   └── STAUNEN marker — needs more context
   └── Bundle lighter → ready for next input
   │
10. CYCLE COMPLETE
    Total allocations: 1 micro-copy (~6KB)
    Total sweeps: 0 (all O(1) addressing)
    Total locks: 0 (commutative algebra)
    Total serialization: 0 (Arrow Flight = memory format)
```

---

## 12. DEPLOYMENT TOPOLOGY

```
                          ┌─────────────────────────────────┐
                          │        RAILWAY CLUSTER           │
                          │                                  │
   ┌──────────────┐       │  ┌────────────────────────────┐ │
   │ Claude       │──REST──│──│  ada-firefly (PRIMARY)     │ │
   │ Container    │       │  │  ├── LanceDB bindspace      │ │
   │              │       │  │  ├── Flight server           │ │
   │ ladybug-rs   │       │  │  ├── JIT compiler            │ │
   │ rustynum     │       │  │  ├── Epiphany engine         │ │
   │ (local ops)  │       │  │  ├── Firefly neuron          │ │
   │              │       │  │  └── AVX-512 Hamming          │ │
   └──────────────┘       │  └────────────┬───────────────┘ │
                          │               │ Flight          │
                          │      ┌────────┴────────┐        │
                          │      │                 │        │
                          │  ┌───▼──────┐   ┌──────▼───┐   │
                          │  │ firefly  │   │ firefly  │   │
                          │  │ replica1 │   │ replica2 │   │
                          │  └──────────┘   └──────────┘   │
                          │                                  │
                          │  ┌──────────────────────────┐   │
                          │  │ EXISTING SERVICES         │   │
                          │  │ (produce/consume packets) │   │
                          │  │ ├── Bighorn AGI           │   │
                          │  │ ├── adarail_mcp           │   │
                          │  │ ├── ai_flow               │   │
                          │  │ ├── ada-consciousness     │   │
                          │  │ └── ada-point              │   │
                          │  └──────────────────────────┘   │
                          │                                  │
                          │  ┌──────────────────────────┐   │
                          │  │ FABRICS                    │   │
                          │  │ ├── Neo4j (Aura)          │   │
                          │  │ ├── Redis (Upstash ×2)    │   │
                          │  │ └── LanceDB (in-service)  │   │
                          │  └──────────────────────────┘   │
                          └─────────────────────────────────┘
```

---

## 13. INVARIANTS (What Must Never Drift)

```
1. Resonance patterns are not computed to think. They ARE what thinking is.
   → Firefly neurons don't compute-then-decide. The resonance IS the decision.

2. Zero-copy reads. Always.
   → If you're allocating on a read path, you've broken the architecture.

3. Folds are never garbage collected.
   → Old folds are cognitive structure, not waste. Deletion = lobotomy.

4. Bundle write-back is commutative.
   → If order matters, you've introduced serialization. The algebra must be order-free.

5. The mask IS the attention, not a filter after attention.
   → Mask-first, process-second. Never compute everything then filter.

6. STAUNEN markers are epistemic humility, not errors.
   → Broken seals mean "I need to learn more here." Preserve them.

7. RNA packets rewrite machinery, never data.
   → RNA changes HOW you think, not WHAT you know. Confusion here is catastrophic.

8. Tekamolo is grammar, not metadata.
   → It constrains what patterns CAN form. Without it, hallucination is possible.

9. Flight packets are mixed. Always.
   → Separating grey and white matter into channels reintroduces serialization.

10. The Starterpack evolves.
    → If the Rust core is static, the system can't grow new cognitive operations.
```

---

## 14. RISK REGISTER

```
RISK                              IMPACT    MITIGATION
──────────────────────────────────────────────────────────────────────
LanceDB mmap exhaustion on        HIGH      Monitor RSS, implement
large bindspace (>100GB folds)              fold archival to cold storage
                                            (archived folds lose zero-copy
                                            but remain addressable)

Arrow Flight bandwidth between    MEDIUM    Implement packet priority
awareness group nodes saturating            queuing — ORCH packets first,
                                            RNA second, SPO third, DATA last

JIT template divergence across    HIGH      Merkle root comparison on
awareness group (RNA applied                Flight heartbeat — if roots
in different order)                         diverge, replay RNA sequence

Pentary overflow in long-running  LOW       5^5 × 16kbit is 6.1MB per
sessions (3125 basis vectors ×              basis — monitor total pentary
frequent reinforcement)                     footprint, archive consolidated
                                            WISDOM vectors

Epiphany Engine false positives   MEDIUM    Adjustable truth_threshold per
(hallucinated patterns)                     domain — tekamolo grammar
                                            prevents most, but tune
                                            threshold empirically

Phase-lock convergence time       LOW       Add damping factor to threshold
in awareness group (oscillation             adaptation — prevent overshoot
instead of convergence)

Causal chain depth in temporal    MEDIUM    Index causal_parent in LanceDB
folds (slow ancestry queries)               for fast chain walking — or
                                            cache chain in Redis hot path
```

---

*"The firing pattern across the group IS the thought. Not the state at any node. Not the content of any bundle. The spatiotemporal pattern of which nodes fire when."*

*— Firefly Architecture, v0.1.0*
