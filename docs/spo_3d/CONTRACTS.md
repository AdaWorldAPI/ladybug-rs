# SPO 3D Contracts

**Every type, trait, and invariant that must hold.**

---

## 1. SPARSE CONTAINER CONTRACT

```rust
/// A sparse encoding of a Container where only non-zero words are stored.
///
/// INVARIANTS (checked at construction, enforced by type system):
/// - bitmap.count_ones() == words.len()
/// - bitmap bit N set ↔ words[popcount(bitmap & ((1 << N) - 1))] is the value
/// - to_dense().hamming(original_dense) == 0  (lossless round-trip)
/// - hamming_sparse(a, b) == a.to_dense().hamming(&b.to_dense())  (equivalence)
///
/// SIZE:
/// - bitmap: 2 × u64 = 16 bytes (128 bits, one per Container word)
/// - words: density × 128 × 8 bytes
/// - At 30% density: 16 + 38×8 = 320 bytes per axis
pub struct SparseContainer {
    /// Which of the 128 words are non-zero. Bit i set → word i is stored.
    pub bitmap: [u64; 2],
    /// Only the non-zero words, in order of bit position.
    pub words: Vec<u64>,
}

impl SparseContainer {
    /// Contract: bitmap and words must be consistent.
    pub fn new(bitmap: [u64; 2], words: Vec<u64>) -> Result<Self, SpoError> {
        let expected = bitmap[0].count_ones() + bitmap[1].count_ones();
        if words.len() != expected as usize {
            return Err(SpoError::BitmapWordMismatch {
                bitmap_ones: expected,
                word_count: words.len(),
            });
        }
        Ok(Self { bitmap, words })
    }

    /// Lossless conversion to dense Container.
    pub fn to_dense(&self) -> Container;

    /// Lossless conversion from dense Container.
    pub fn from_dense(container: &Container) -> Self;

    /// Hamming distance WITHOUT densification. O(min(popcount_a, popcount_b)).
    pub fn hamming_sparse(a: &SparseContainer, b: &SparseContainer) -> u32;

    /// XOR bind in sparse domain.
    pub fn bind_sparse(a: &SparseContainer, b: &SparseContainer) -> SparseContainer;

    /// Density: fraction of non-zero words (0.0 to 1.0).
    pub fn density(&self) -> f32;

    /// Number of stored words.
    pub fn word_count(&self) -> usize;
}
```

### SparseContainer Invariant Tests

```rust
#[test] fn sparse_roundtrip_lossless() {
    let dense = Container::random(42);
    let sparse = SparseContainer::from_dense(&dense);
    assert_eq!(sparse.to_dense(), dense);
}

#[test] fn sparse_hamming_equivalence() {
    let a = Container::random(1);
    let b = Container::random(2);
    let sa = SparseContainer::from_dense(&a);
    let sb = SparseContainer::from_dense(&b);
    assert_eq!(
        SparseContainer::hamming_sparse(&sa, &sb),
        a.hamming(&b)
    );
}

#[test] fn sparse_bitmap_consistency() {
    let sparse = SparseContainer::from_dense(&Container::random(99));
    let ones = sparse.bitmap[0].count_ones() + sparse.bitmap[1].count_ones();
    assert_eq!(ones as usize, sparse.words.len());
}
```

---

## 2. NIBBLE SCENT CONTRACT

```rust
/// 48-byte nibble histogram scent: 16 bins × 3 axes.
///
/// Each axis gets a histogram of nibble (4-bit) frequencies across its
/// sparse container words. This captures the TYPE of content without
/// destroying structure (unlike XOR-fold).
///
/// INVARIANTS:
/// - scent.x_hist: sum of all 16 bins = total nibbles in X axis words
/// - scent.y_hist: sum of all 16 bins = total nibbles in Y axis words
/// - scent.z_hist: sum of all 16 bins = total nibbles in Z axis words
/// - scent_distance(a, b) correlates with content_type_similarity(a, b)
/// - Different content types (Person, Concept, Edge) have distinct scent profiles
///
/// SIZE: exactly 48 bytes = 6 × u64 = words 12-17 in meta container
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct NibbleScent {
    /// Nibble histogram for X (Subject) axis. 16 bins, each u8 (saturating).
    pub x_hist: [u8; 16],
    /// Nibble histogram for Y (Predicate) axis.
    pub y_hist: [u8; 16],
    /// Nibble histogram for Z (Object) axis.
    pub z_hist: [u8; 16],
}

impl NibbleScent {
    /// Size in bytes (compile-time constant).
    pub const SIZE: usize = 48;

    /// Compute scent from three sparse axes.
    pub fn from_axes(x: &SparseContainer, y: &SparseContainer, z: &SparseContainer) -> Self;

    /// L1 distance between two scents (sum of absolute bin differences).
    pub fn distance(&self, other: &NibbleScent) -> u32;

    /// Per-axis distances for selective filtering.
    pub fn axis_distances(&self, other: &NibbleScent) -> (u32, u32, u32);

    /// Pack into 6 u64 words (for meta container W12-W17).
    pub fn to_words(&self) -> [u64; 6];

    /// Unpack from 6 u64 words.
    pub fn from_words(words: &[u64; 6]) -> Self;

    /// Zero scent (empty record).
    pub fn zero() -> Self;
}
```

### NibbleScent Invariant Tests

```rust
#[test] fn scent_size_is_48() {
    assert_eq!(std::mem::size_of::<NibbleScent>(), 48);
}

#[test] fn scent_word_roundtrip() {
    let scent = NibbleScent::from_axes(&x, &y, &z);
    let words = scent.to_words();
    assert_eq!(NibbleScent::from_words(&words), scent);
}

#[test] fn scent_different_types_distinct() {
    let person = build_node(dn_hash("jan"), &[LBL_PERSON], &[(KEY_NAME, "Jan")]);
    let concept = build_node(dn_hash("rust"), &[LBL_CONCEPT], &[(KEY_NAME, "Rust")]);
    let ps = NibbleScent::from_record(&person);
    let cs = NibbleScent::from_record(&concept);
    assert!(ps.distance(&cs) > 10, "Different types must have distinct scents");
}
```

---

## 3. PACKED AXES CONTRACT

```rust
/// Three sparse axes packed into one content Container (128 × u64).
///
/// Layout within the content container:
/// ```text
/// words[0..2]:      X bitmap (128 bits)
/// words[2..2+Nx]:   X non-zero words
/// words[2+Nx..4+Nx]: Y bitmap
/// words[4+Nx..4+Nx+Ny]: Y non-zero words  
/// words[4+Nx+Ny..6+Nx+Ny]: Z bitmap
/// words[6+Nx+Ny..6+Nx+Ny+Nz]: Z non-zero words
/// ```
///
/// INVARIANTS:
/// - 6 + Nx + Ny + Nz <= 128 (must fit in one Container)
/// - If overflow: fall back to ContainerGeometry::Xyz (3 linked CogRecords)
/// - pack(unpack(container)) == container (lossless)
///
/// Meta words 34-39 store axis descriptors:
/// - W34: x_offset(u16) | x_count(u16) | y_offset(u16) | y_count(u16)
/// - W35: z_offset(u16) | z_count(u16) | total_words(u16) | reserved(u16)
pub struct PackedAxes;

impl PackedAxes {
    /// Maximum total non-zero words across all 3 axes.
    /// 128 total - 6 bitmap words = 122 available.
    pub const MAX_CONTENT_WORDS: usize = 122;

    /// Pack three sparse axes into one Container.
    /// Returns Err if total density exceeds capacity.
    pub fn pack(
        x: &SparseContainer,
        y: &SparseContainer,
        z: &SparseContainer,
    ) -> Result<Container, SpoError>;

    /// Unpack content Container into three sparse axes.
    /// Reads axis descriptors from meta words 34-35.
    pub fn unpack(
        content: &Container,
        meta: &Container,
    ) -> Result<(SparseContainer, SparseContainer, SparseContainer), SpoError>;

    /// Check if axes fit in one Container.
    pub fn fits(x: &SparseContainer, y: &SparseContainer, z: &SparseContainer) -> bool {
        x.word_count() + y.word_count() + z.word_count() + 6 <= 128
    }

    /// Write axis descriptors to meta words 34-35.
    pub fn write_descriptors(
        meta: &mut [u64; 128],
        x_count: u16, y_count: u16, z_count: u16,
    );
}
```

### Capacity Budget

| Density | Words per axis | Total (3 axes + 6 bitmap) | Fits? |
|---------|---------------|---------------------------|-------|
| 10%     | 13            | 6 + 39 = 45              | ✓     |
| 20%     | 26            | 6 + 78 = 84              | ✓     |
| 30%     | 38            | 6 + 114 = 120            | ✓     |
| 35%     | 45            | 6 + 135 = 141            | ✗ → use Xyz fallback |
| 40%     | 51            | 6 + 153 = 159            | ✗     |

**At typical 30% density per axis: 120/128 words used = 93.75% utilization.**

---

## 4. SPO RECORD VIEW CONTRACT

```rust
/// Zero-copy SPO view into a CogRecord with Spo geometry.
///
/// Provides typed access to the three axes without copying or allocating.
/// SAFETY: only valid when record.geometry() == ContainerGeometry::Spo.
pub struct SpoView<'a> {
    meta: MetaView<'a>,
    x: SparseContainer,  // unpacked on construction
    y: SparseContainer,
    z: SparseContainer,
    scent: NibbleScent,
}

impl<'a> SpoView<'a> {
    /// Construct from a CogRecord. Panics if geometry != Spo.
    pub fn new(record: &'a CogRecord) -> Self;

    /// Try to construct (returns None if geometry != Spo).
    pub fn try_new(record: &'a CogRecord) -> Option<Self>;

    // --- Axis access ---
    pub fn x(&self) -> &SparseContainer;
    pub fn y(&self) -> &SparseContainer;
    pub fn z(&self) -> &SparseContainer;
    pub fn scent(&self) -> &NibbleScent;

    // --- Dense axis access (for operations that need full Container) ---
    pub fn x_dense(&self) -> Container;
    pub fn y_dense(&self) -> Container;
    pub fn z_dense(&self) -> Container;

    // --- NARS truth ---
    pub fn nars(&self) -> TruthValue;

    // --- DN identity ---
    pub fn dn(&self) -> u64;

    // --- Causal queries ---

    /// Hamming distance of this record's Z to another's X.
    /// This IS the causal coherence score.
    pub fn causal_coherence_to(&self, other: &SpoView) -> u32;

    /// Hamming distance of this record's X to another's Z.
    /// Reverse causal query.
    pub fn causal_coherence_from(&self, other: &SpoView) -> u32;
}

/// Mutable SPO view for record construction.
pub struct SpoViewMut<'a> {
    record: &'a mut CogRecord,
}

impl<'a> SpoViewMut<'a> {
    /// Set the three axes. Packs sparse into content Container.
    pub fn set_axes(
        &mut self,
        x: &SparseContainer,
        y: &SparseContainer,
        z: &SparseContainer,
    ) -> Result<(), SpoError>;

    /// Set NARS truth value.
    pub fn set_nars(&mut self, tv: TruthValue);

    /// Set scent (computed from axes).
    pub fn set_scent(&mut self, scent: NibbleScent);

    /// Set DN address.
    pub fn set_dn(&mut self, dn: u64);

    /// Set DN tree links.
    pub fn set_dn_tree(&mut self, parent: u64, child: u64, next: u64, prev: u64);

    /// Convenience: build and commit a complete SPO record.
    pub fn build_complete(
        &mut self,
        dn: u64,
        x: &SparseContainer,
        y: &SparseContainer,
        z: &SparseContainer,
        nars: TruthValue,
    ) -> Result<(), SpoError>;
}
```

---

## 5. SPO STORE CONTRACT

```rust
/// Three-axis content-addressable graph store.
///
/// INVARIANTS:
/// - Every record has geometry == Spo
/// - query_forward + query_reverse + query_relation are exhaustive
///   (no record is invisible to any query direction)
/// - causal_successors(A) and causal_predecessors(B) are inverses:
///   A in causal_predecessors(B) ⟺ B in causal_successors(A)
pub struct SpoStore {
    records: BTreeMap<u64, CogRecord>,  // POC: DN → record. Production: LanceDB.
}

impl SpoStore {
    pub fn new() -> Self;
    pub fn insert(&mut self, record: CogRecord) -> Result<(), SpoError>;
    pub fn get(&self, dn: u64) -> Option<&CogRecord>;
    pub fn len(&self) -> usize;

    // --- Three-axis queries ---

    /// Forward: "What does <src> <verb>?" → scan X+Y, return Z matches.
    pub fn query_forward(
        &self,
        src_fp: &Container,
        verb_fp: &Container,
        radius: u32,
    ) -> Vec<QueryHit>;

    /// Reverse: "Who <verb>s <tgt>?" → scan Z+Y, return X matches.
    pub fn query_reverse(
        &self,
        tgt_fp: &Container,
        verb_fp: &Container,
        radius: u32,
    ) -> Vec<QueryHit>;

    /// Relation: "How are <src> and <tgt> related?" → scan X+Z, return Y matches.
    pub fn query_relation(
        &self,
        src_fp: &Container,
        tgt_fp: &Container,
        radius: u32,
    ) -> Vec<QueryHit>;

    /// Content: "Find anything matching this fingerprint" → scan all axes.
    pub fn query_content(
        &self,
        query: &Container,
        radius: u32,
    ) -> Vec<QueryHit>;

    // --- Causal chain ---

    /// Records whose X resonates with `record`'s Z.
    pub fn causal_successors(
        &self,
        record: &CogRecord,
        radius: u32,
    ) -> Vec<QueryHit>;

    /// Records whose Z resonates with `record`'s X.
    pub fn causal_predecessors(
        &self,
        record: &CogRecord,
        radius: u32,
    ) -> Vec<QueryHit>;

    /// Walk a causal chain forward from `start`, max `depth` hops.
    pub fn walk_chain_forward(
        &self,
        start: &CogRecord,
        radius: u32,
        depth: usize,
    ) -> Vec<Vec<QueryHit>>;

    /// Compute chain coherence: product of link coherences.
    pub fn chain_coherence(&self, chain: &[u64]) -> f32;

    // --- Scent pre-filter ---

    /// Filter records by scent distance before Hamming scan.
    pub fn scent_prefilter(
        &self,
        query_scent: &NibbleScent,
        max_distance: u32,
    ) -> Vec<u64>;
}

#[derive(Clone, Debug)]
pub struct QueryHit {
    pub dn: u64,
    pub distance: u32,
    pub axis: QueryAxis,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryAxis {
    X,  // matched on Subject
    Y,  // matched on Predicate
    Z,  // matched on Object
    XY, // matched on Subject + Predicate (forward query)
    YZ, // matched on Predicate + Object (reverse query)
    XZ, // matched on Subject + Object (relation query)
}
```

---

## 6. BUILDER CONTRACT

```rust
/// Node/edge construction for SPO records.
///
/// INVARIANTS:
/// - build_node produces a record where Y axis is near-zero (nodes have no verb)
/// - build_edge produces a record where all three axes are populated
/// - scent is automatically computed from axes
/// - NARS truth is stamped in meta W4-W7
pub struct SpoBuilder;

impl SpoBuilder {
    /// Build a node record (entity with labels + properties).
    ///
    /// X axis = BUNDLE(label_fps, property_fps)  — entity identity
    /// Y axis = near-zero (no verb for nodes)
    /// Z axis = mirror of X (self-referential for CAM lookup)
    pub fn build_node(
        dn: u64,
        label_fps: &[&Container],
        property_fps: &[(&Container, &Container)],  // (key_fp, val_fp)
        nars: TruthValue,
    ) -> Result<CogRecord, SpoError>;

    /// Build an edge record (relationship between two entities).
    ///
    /// X axis = BIND(src_fp, PERMUTE(verb_fp, 1))    — subject+verb
    /// Y axis = verb_fp                                — pure verb
    /// Z axis = BIND(tgt_fp, PERMUTE(verb_fp, 2))     — object+verb
    pub fn build_edge(
        dn: u64,
        src_fp: &Container,
        verb_fp: &Container,
        tgt_fp: &Container,
        nars: TruthValue,
    ) -> Result<CogRecord, SpoError>;

    /// Build a meta-awareness record from a causal chain.
    ///
    /// X axis = BUNDLE(chain_x_axes)     — entities involved
    /// Y axis = CHAIN_DISCOVERED_fp      — "I noticed a chain"
    /// Z axis = BUNDLE(chain_z_axes)     — what the chain implies
    pub fn build_meta_awareness(
        dn: u64,
        chain: &[&CogRecord],
        chain_coherence: f32,
    ) -> Result<CogRecord, SpoError>;
}
```

---

## 7. ERROR CONTRACT

```rust
#[derive(Clone, Debug, thiserror::Error)]
pub enum SpoError {
    #[error("Bitmap has {bitmap_ones} ones but {word_count} words supplied")]
    BitmapWordMismatch { bitmap_ones: u32, word_count: usize },

    #[error("Axes too dense: {total} words needed, max {max}")]
    AxesOverflow { total: usize, max: usize },

    #[error("Record geometry is {actual:?}, expected Spo")]
    WrongGeometry { actual: ContainerGeometry },

    #[error("DN {dn:#x} already exists in store")]
    DuplicateDn { dn: u64 },

    #[error("DN {dn:#x} not found in store")]
    DnNotFound { dn: u64 },

    #[error("Chain is empty")]
    EmptyChain,

    #[error("Scent computation failed: {reason}")]
    ScentError { reason: String },
}
```

---

## 8. GEOMETRY EXTENSION CONTRACT

```rust
// In crates/ladybug-contract/src/geometry.rs:
#[repr(u8)]
pub enum ContainerGeometry {
    Cam = 0,
    Xyz = 1,
    Bridge = 2,
    Extended = 3,
    Chunked = 4,
    Tree = 5,
    Spo = 6,   // ← NEW: Subject-Predicate-Object with sparse axes
}
```

### Migration Path

- Existing `Cam` records: unaffected. Still 1 content container, flat scan.
- Existing `Xyz` records: unaffected. Still 3 linked records via DN tree.
- New `Spo` records: 1 content container with packed sparse axes.
- `Spo` CAN fall back to `Xyz` if axes are too dense (>35% each).

---

## 9. LANCE COLUMNAR SCHEMA

```
Column 0:  dn          UInt64          PRIMARY KEY (sort key prefix)
Column 1:  meta        FixedBinary(1024)
Column 2:  x_bitmap    FixedBinary(16)  
Column 3:  x_words     Binary           variable length
Column 4:  y_bitmap    FixedBinary(16)
Column 5:  y_words     Binary
Column 6:  z_bitmap    FixedBinary(16)
Column 7:  z_words     Binary
Column 8:  scent       FixedBinary(48)  3×16 nibble histogram
Column 9:  created     Int64
Column 10: nars_freq   Float32
Column 11: nars_conf   Float32

Sort key: (dn >> 48, scent[0..4], dn)
  → DN locality first, then scent similarity, then exact DN
  → Adjacent rows share subtree + content type
  → XOR delta compression ~79% zeros within sorted groups
```

### Query Column Selection

| Query type | Columns read |
|-----------|-------------|
| Forward   | 0, 2, 3, 4, 5, 8 (dn, x, y, scent) |
| Reverse   | 0, 4, 5, 6, 7, 8 (dn, y, z, scent) |
| Relation  | 0, 2, 3, 6, 7, 8 (dn, x, z, scent) |
| Content   | 0, 2-7, 8 (all axes + scent) |
| Causal    | 0, 2, 3, 6, 7, 8 (x of successors, z of predecessors) |
| Metadata  | 0, 1 (dn, meta only) |

---

## 10. NARS INTEGRATION

NARS truth values live in meta W4-W7 (unchanged from existing CogRecord):

```
W4: frequency  (f32)
W5: confidence (f32)
W6: pos_evidence (f32)
W7: neg_evidence (f32)
```

### Chain Truth Propagation

```rust
/// Deduction along a causal chain:
/// f_chain = f_1 × f_2 × ... × f_n
/// c_chain = c_1 × c_2 × ... × c_n × coherence_1_2 × coherence_2_3 × ...
///
/// Coherence factor per Z→X link:
///   coherence = 1.0 - (hamming(Z_i, X_{i+1}) as f32 / 8192.0)
pub fn chain_deduction(chain: &[&CogRecord]) -> TruthValue;

/// Revision: merge two beliefs about the same triple.
/// Uses existing nars::revision() from ladybug_contract.
pub fn spo_revision(a: &CogRecord, b: &CogRecord) -> TruthValue;
```

---

## 11. SIX IRONCLAD TESTS

```rust
#[test] fn test_1_node_roundtrip() {
    // Create "Jan" {Person, name: "Jan", age: 42}
    // Insert into store → retrieve → verify X axis Hamming < 100
}

#[test] fn test_2_forward_query() {
    // Jan → KNOWS → Ada
    // query_forward(jan_fp, KNOWS_fp) must find Ada
}

#[test] fn test_3_reverse_query() {
    // Jan → KNOWS → Ada
    // query_reverse(ada_fp, KNOWS_fp) must find Jan — NO separate index
}

#[test] fn test_4_cam_content_lookup() {
    // 100 nodes → query by content fingerprint → find Jan
}

#[test] fn test_5_nars_reasoning() {
    // "Jan knows Rust" <0.8, 0.9> + "Rust helps CAM" <0.7, 0.8>
    // → deduction → verify f/c values → revision increases confidence
}

#[test] fn test_6_causal_chain_coherence() {
    // Create chain: Jan → KNOWS → Rust → ENABLES → CAM
    // Verify Z→X resonance (Hamming < 200)
    // Verify causal_successors finds the chain
    // Build meta-awareness record → verify convergence
}
```

---

## 12. COMPATIBILITY MATRIX

| Component | Before SPO | After SPO | Breaking? |
|-----------|-----------|-----------|-----------|
| Container | unchanged | unchanged | No |
| CogRecord | unchanged | unchanged (new geometry variant) | No |
| MetaView W0-W11 | unchanged | unchanged | No |
| MetaView W12-W17 | 7-layer markers | NibbleScent (SPO only) | No* |
| MetaView W18-W33 | inline edges | inline edges | No |
| MetaView W34-W39 | reserved | axis descriptors (SPO only) | No* |
| ContainerGeometry | 6 variants | 7 variants (+Spo) | No** |
| Codebook | 4096 entries | unchanged | No |
| NARS | W4-W7 | unchanged | No |
| Scent | 5-byte XOR-fold | 48-byte histogram (SPO only) | No* |
| Existing tests | 1,267+ | must all pass | No |

*Only applies to records with geometry == Spo. Other geometries untouched.
**Requires geometry.rs update — `from_u8(6) => Some(Spo)`.
