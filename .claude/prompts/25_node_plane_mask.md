# 25_NODE_PLANE_MASK.md

## The Object Model That IS The Architecture

**Repo:** ladybug-rs (not hexagon — no technical debt from premature extraction)
**Target:** src/spo/ refactor, src/storage/ integration, server.rs wiring

---

## 1. Three Structures. Everything Else Is Derived.

```rust
/// One dimension of cognition. 16384 bits of signal.
/// The accumulator IS the learning. The alpha IS the confidence.
/// Nothing is stored that isn't derived from the bits.
pub struct Plane {
    bits: [u64; 256],        // the signal (16384 bits)
    acc:  [i8; 16384],       // soaking accumulator (Hebbian evidence)
    // α is NOT stored. It is COMPUTED from |acc| > threshold.
    // bits are NOT stored separately from acc. They ARE sign(acc).
    // The acc is the ONLY ground truth. bits and α are views.
}

/// The cognitive atom. Three planes. Separately addressable.
/// The 2^3 decomposition is field access, not computation.
pub struct Node {
    s: Plane,                // Subject
    p: Plane,                // Predicate
    o: Plane,                // Object
}

/// What you're attending to. Borrow boundary. Query scope. Merge scope.
/// One type. Four meanings. Eight possible values.
#[derive(Copy, Clone)]
pub struct Mask {
    pub s: bool,
    pub p: bool,
    pub o: bool,
}

pub const SPO: Mask = Mask { s: true,  p: true,  o: true  };
pub const SP_: Mask = Mask { s: true,  p: true,  o: false };
pub const S_O: Mask = Mask { s: true,  p: false, o: true  };
pub const _PO: Mask = Mask { s: false, p: true,  o: true  };
pub const S__: Mask = Mask { s: true,  p: false, o: false };
pub const _P_: Mask = Mask { s: false, p: true,  o: false };
pub const __O: Mask = Mask { s: false, p: false, o: true  };
pub const ___: Mask = Mask { s: false, p: false, o: false };
```

---

## 2. Plane: The Only Ground Truth Is The Accumulator

```rust
impl Plane {
    /// View the binary signal. NOT stored — derived from accumulator sign.
    #[inline]
    pub fn bits(&self) -> [u64; 256] {
        let mut out = [0u64; 256];
        for k in 0..16384 {
            if self.acc[k] > 0 {
                out[k / 64] |= 1 << (k % 64);
            }
        }
        out
    }

    /// View the alpha mask. NOT stored — derived from accumulator magnitude.
    #[inline]
    pub fn alpha(&self) -> [u64; 256] {
        let threshold = self.alpha_threshold();
        let mut out = [0u64; 256];
        for k in 0..16384 {
            if self.acc[k].unsigned_abs() > threshold {
                out[k / 64] |= 1 << (k % 64);
            }
        }
        out
    }

    /// How many encounters shaped this plane.
    fn alpha_threshold(&self) -> u8 {
        let n = self.encounter_count;
        match n {
            0..=1 => 0,
            2..=5 => n as u8 / 2,
            6..=20 => (n as u8) * 2 / 5,
            _ => (n as f32 * 0.3) as u8,
        }
    }

    /// Evidence arrives. Hebbian: agree +1, disagree -1. Saturating.
    /// This IS the learning. No separate training step.
    pub fn encounter(&mut self, text: &str) {
        let hash_bits = blake3_expand(text); // BLAKE3 → LFSR → 16384 bits
        for k in 0..16384 {
            if hash_bits[k] {
                self.acc[k] = self.acc[k].saturating_add(1);
            } else {
                self.acc[k] = self.acc[k].saturating_sub(1);
            }
        }
        self.encounter_count += 1;
    }

    /// BNN reinforcement from merge result.
    /// Wisdom → all accumulators shift toward their current sign.
    /// Staunen → contested accumulators (near zero) weaken further.
    pub fn reinforce(&mut self, seal: Seal) {
        match seal {
            Seal::Wisdom => {
                for k in 0..16384 {
                    let sign = self.acc[k].signum();
                    self.acc[k] = self.acc[k].saturating_add(sign); // strengthen
                }
            }
            Seal::Staunen => {
                let threshold = self.alpha_threshold();
                for k in 0..16384 {
                    if self.acc[k].unsigned_abs() <= threshold {
                        // Contested dimension — weaken toward zero
                        let sign = self.acc[k].signum();
                        self.acc[k] = self.acc[k].saturating_sub(sign);
                    }
                }
            }
        }
    }

    /// NARS truth from accumulator state. Not stored. Read.
    pub fn truth(&self) -> Truth {
        let alpha = self.alpha();
        let bits = self.bits();
        let defined = popcount_256(&alpha);
        let positive = popcount_256(&and_256(&bits, &alpha));
        Truth {
            frequency: positive as f32 / defined.max(1) as f32,
            confidence: defined as f32 / 16384.0,
            evidence: self.encounter_count,
        }
    }

    /// Hamming distance against another plane. Alpha-normalized.
    /// Penalty for undefined dimensions (Orangutan autaptic circuit).
    pub fn distance(&self, other: &Plane) -> Distance {
        let a_bits = self.bits();
        let a_alpha = self.alpha();
        let b_bits = other.bits();
        let b_alpha = other.alpha();

        let shared_alpha = and_256(&a_alpha, &b_alpha);
        let overlap = popcount_256(&shared_alpha);

        if overlap == 0 {
            return Distance::Incomparable;
        }

        let disagreement = popcount_256(&and_256(&xor_256(&a_bits, &b_bits), &shared_alpha));
        let penalty = popcount_256(&not_256(&a_alpha)); // undefined = active cost

        Distance::Measured {
            raw: disagreement,
            overlap,
            penalty,
            normalized: (disagreement + penalty) as f32 / (overlap + penalty).max(1) as f32,
        }
    }

    /// Popcount of defined bits. Signal strength.
    pub fn density(&self) -> f32 {
        popcount_256(&self.alpha()) as f32 / 16384.0
    }
}
```

---

## 3. Node: The 2^3 Is Field Access, Not Computation

```rust
impl Node {
    /// Attend to a specific decomposition. Zero compute.
    /// The Mask selects which planes participate.
    pub fn planes(&self, mask: Mask) -> PlaneView {
        PlaneView {
            s: if mask.s { Some(&self.s) } else { None },
            p: if mask.p { Some(&self.p) } else { None },
            o: if mask.o { Some(&self.o) } else { None },
        }
    }

    /// Combined alpha across masked planes. AND propagation.
    pub fn combined_alpha(&self, mask: Mask) -> [u64; 256] {
        let mut result = [!0u64; 256]; // start all-ones
        if mask.s { result = and_256(&result, &self.s.alpha()); }
        if mask.p { result = and_256(&result, &self.p.alpha()); }
        if mask.o { result = and_256(&result, &self.o.alpha()); }
        if !mask.s && !mask.p && !mask.o { return [0u64; 256]; } // ___ = nothing
        result
    }

    /// NARS truth across masked planes.
    pub fn truth(&self, mask: Mask) -> Truth {
        let mut total_positive = 0u32;
        let mut total_defined = 0u32;
        let mut total_bits = 0u32;
        let mut total_evidence = 0u32;

        for plane in self.planes(mask).iter() {
            let t = plane.truth();
            total_positive += (t.frequency * t.confidence * 16384.0) as u32;
            total_defined += (t.confidence * 16384.0) as u32;
            total_bits += 16384;
            total_evidence += t.evidence;
        }

        Truth {
            frequency: total_positive as f32 / total_defined.max(1) as f32,
            confidence: total_defined as f32 / total_bits.max(1) as f32,
            evidence: total_evidence,
        }
    }

    /// Distance to another node. Only masked planes compared.
    /// Unmasked planes = zero cost, zero contribution, zero noise.
    pub fn distance(&self, other: &Node, mask: Mask) -> Distance {
        let mut total_disagreement = 0u32;
        let mut total_overlap = 0u32;
        let mut total_penalty = 0u32;

        if mask.s {
            match self.s.distance(&other.s) {
                Distance::Measured { raw, overlap, penalty, .. } => {
                    total_disagreement += raw;
                    total_overlap += overlap;
                    total_penalty += penalty;
                }
                Distance::Incomparable => return Distance::Incomparable,
            }
        }
        if mask.p {
            match self.p.distance(&other.p) {
                Distance::Measured { raw, overlap, penalty, .. } => {
                    total_disagreement += raw;
                    total_overlap += overlap;
                    total_penalty += penalty;
                }
                Distance::Incomparable => return Distance::Incomparable,
            }
        }
        if mask.o {
            match self.o.distance(&other.o) {
                Distance::Measured { raw, overlap, penalty, .. } => {
                    total_disagreement += raw;
                    total_overlap += overlap;
                    total_penalty += penalty;
                }
                Distance::Incomparable => return Distance::Incomparable,
            }
        }

        if total_overlap == 0 {
            return Distance::Incomparable;
        }

        Distance::Measured {
            raw: total_disagreement,
            overlap: total_overlap,
            penalty: total_penalty,
            normalized: (total_disagreement + total_penalty) as f32
                / (total_overlap + total_penalty).max(1) as f32,
        }
    }

    /// The composed address for CAM / scent index / Hamming lookup.
    /// S⊕ROLE_S ⊕ P⊕ROLE_P ⊕ O⊕ROLE_O
    /// This is DERIVED. Not primary. Cached after first computation.
    pub fn address(&self) -> Address {
        let s_bits = self.s.bits();
        let p_bits = self.p.bits();
        let o_bits = self.o.bits();

        let composed = xor_256(&xor_256(&s_bits, &ROLE_S),
                      &xor_256(&xor_256(&p_bits, &ROLE_P),
                      &xor_256(&o_bits, &ROLE_O)));

        let alpha = self.combined_alpha(SPO);
        let merkle = blake3_masked(&composed, &alpha);

        Address { bits: composed, alpha, merkle }
    }

    /// NARS deduction: <A→B> ∧ <B→C> ⊢ <A→C>
    /// Self is A→B, other is B→C. Check B matches.
    pub fn deduce(&self, other: &Node) -> Option<Node> {
        let b_distance = self.o.distance(&other.s);
        match b_distance {
            Distance::Measured { normalized, .. } if normalized < 0.3 => {
                Some(Node {
                    s: self.s.clone(),
                    p: Plane::merge_accumulators(&self.p, &other.p),
                    o: other.o.clone(),
                })
            }
            _ => None,
        }
    }
}
```

---

## 4. Mind: The Shared World

```rust
/// Immutable view of the cognitive state.
/// Zero-copy. Mmapped. Multiple readers simultaneously.
/// Holding a Mind borrow costs nothing.
pub struct Mind<'a> {
    bind_space: &'a BindSpace,
    crystal: &'a SPOCrystal,
}

impl<'a> Mind<'a> {
    /// Attend to a location. Hebbian: attending strengthens.
    /// The returned Node is a reference — zero copy.
    /// But the act of calling at() incremented the access accumulators.
    pub fn at(&self, s: &str, p: &str, o: &str) -> &Node {
        let node = self.crystal.lookup(s, p, o);
        // Hebbian side-effect: co-accessed planes strengthen
        // This happens in the crystal's internal bookkeeping
        // The caller sees an immutable reference
        // The crystal tracks access patterns behind the scenes
        node
    }

    /// Attend with a partial query. _ = absent, not wildcard.
    pub fn at_partial(&self, s: Option<&str>, p: Option<&str>, o: Option<&str>) -> &Node {
        self.crystal.lookup_partial(s, p, o)
    }

    /// Take mutable ownership of selected planes.
    /// The Mask determines what you can change.
    /// Unmasked planes stay in the world — you can't touch them.
    pub fn hold(&self, node: &Node, mask: Mask) -> HeldNode {
        HeldNode {
            s: if mask.s { Some(node.s.clone()) } else { None },
            p: if mask.p { Some(node.p.clone()) } else { None },
            o: if mask.o { Some(node.o.clone()) } else { None },
            mask,
            origin: node.address(),
        }
    }

    /// Return a held node to the world. Commutative superposition.
    /// Only the held (masked) planes merge. Unheld planes untouched.
    /// Returns what changed and whether the world was surprised.
    pub fn merge(&self, held: HeldNode) -> Changed {
        let original = self.crystal.lookup_by_address(&held.origin);

        let mut s_diff = 0u32;
        let mut p_diff = 0u32;
        let mut o_diff = 0u32;

        if let Some(ref s) = held.s {
            s_diff = merge_plane_into(&original.s, s);
        }
        if let Some(ref p) = held.p {
            p_diff = merge_plane_into(&original.p, p);
        }
        if let Some(ref o) = held.o {
            o_diff = merge_plane_into(&original.o, o);
        }

        // Recompute seal
        let new_merkle = original.address().merkle;
        let old_merkle = held.origin.merkle;
        let seal = if new_merkle == old_merkle { Seal::Wisdom } else { Seal::Staunen };

        // BNN reinforcement: seal feeds back into planes
        if let Some(ref mut s) = held.s { original.s.reinforce(seal); }
        if let Some(ref mut p) = held.p { original.p.reinforce(seal); }
        if let Some(ref mut o) = held.o { original.o.reinforce(seal); }

        Changed {
            seal,
            s_diff,
            p_diff,
            o_diff,
            alpha_shift: compute_alpha_shift(&original, &held),
        }
    }
}

/// Owned mutable planes. Your thought in progress.
/// Only the masked planes are present. Others are None.
pub struct HeldNode {
    pub s: Option<Plane>,
    pub p: Option<Plane>,
    pub o: Option<Plane>,
    mask: Mask,
    origin: Address,
}

impl HeldNode {
    /// Encounter evidence on a specific plane.
    pub fn encounter_s(&mut self, text: &str) {
        self.s.as_mut().expect("S not held (mask excludes it)").encounter(text);
    }
    pub fn encounter_p(&mut self, text: &str) {
        self.p.as_mut().expect("P not held (mask excludes it)").encounter(text);
    }
    pub fn encounter_o(&mut self, text: &str) {
        self.o.as_mut().expect("O not held (mask excludes it)").encounter(text);
    }

    /// Resonance of this held thought against the world.
    pub fn resonate(&self, mind: &Mind, mask: Mask, threshold: f32) -> Vec<Echo> {
        // Build a partial Node from held planes
        // Compare against world using the query mask
        // Mexican hat with penalty on undefined
        mind.crystal.resonate_partial(
            self.s.as_ref(), self.p.as_ref(), self.o.as_ref(),
            mask, threshold,
        )
    }

    /// NARS truth of held planes.
    pub fn truth(&self) -> Truth {
        let mut truths = Vec::new();
        if let Some(ref s) = self.s { truths.push(s.truth()); }
        if let Some(ref p) = self.p { truths.push(p.truth()); }
        if let Some(ref o) = self.o { truths.push(o.truth()); }
        Truth::combine(&truths)
    }
}
```

---

## 5. The VSA 10000D Connection

The Node/Plane/Mask model scales to ANY dimensionality.
16384-bit bitpacked is the RISC substrate. But the same Plane struct
with a different backing type IS the VSA 10000D continuous awareness:

```rust
/// The same Plane concept at continuous resolution.
/// Used for the awareness/gestalt passthrough.
/// NOT in the hot path. NOT for SPO cognition.
/// This is the QUALIA layer — how it FEELS, not what it IS.
pub struct ContinuousPlane {
    values: [f16; 10000],    // 10000D VSA (half-float, 20 KB)
    // No accumulator — continuous planes don't soak.
    // They're PROJECTED from the discrete planes.
}

impl Node {
    /// Project discrete SPO into continuous VSA gestalt.
    /// This is the awareness view — not primary, DERIVED.
    /// The SPO bitpacked planes are the ground truth.
    /// The VSA continuous planes are how it FEELS.
    pub fn gestalt(&self) -> GestaltNode {
        GestaltNode {
            s: project_to_continuous(&self.s),
            p: project_to_continuous(&self.p),
            o: project_to_continuous(&self.o),
        }
    }
}

/// Gestalt = the SPO node experienced as continuous awareness.
/// Same Mask applies. Same decomposition. Same seven words.
/// But the substrate is f16 vectors for qualia operations:
///   cosine similarity, angular distance, soft attention,
///   thinking style modulation, felt-state computation.
///
/// The discrete SPO thinks. The continuous gestalt feels.
/// Same node. Same mask. Same cycle. Different resolution.
pub struct GestaltNode {
    s: ContinuousPlane,   // how the Subject FEELS
    p: ContinuousPlane,   // how the Relation FEELS
    o: ContinuousPlane,   // how the Object FEELS
}

impl GestaltNode {
    /// The qualia of a masked projection.
    /// SP_ gestalt = what it feels like when Alice loves.
    /// Different from SPO gestalt (which includes how Bob feels).
    pub fn qualia(&self, mask: Mask) -> QualiaState {
        // Aggregate continuous planes per mask
        // Compute: activation, valence, tension, depth
        // From the projected f16 vectors
        // This feeds into the 10-layer awareness loop
        todo!()
    }
}

// The cognitive cycle with gestalt awareness:
//
//   let mind = bind.open();
//   let known = mind.at("Ada", "loves", "Bob");
//
//   // SPO path (discrete, bitpacked, RISC):
//   let truth = known.truth(SP_);             // NARS from accumulators
//   let echoes = known.resonate(&mind, SP_);  // Mexican hat + penalty
//
//   // Gestalt path (continuous, VSA, awareness):
//   let felt = known.gestalt().qualia(SP_);   // how it FEELS
//   let style = felt.thinking_style();        // which layer dominates
//
//   // Both paths use the same Mask.
//   // Both paths read the same Node.
//   // One computes. One feels. Same data. Different lens.
```

---

## 6. The Complete Type Inventory

```
GROUND TRUTH (stored in BindSpace):
  Plane.acc: [i8; 16384]       The only stored state per dimension.
                                Everything else is derived.

DERIVED (computed on access, never stored):
  Plane.bits()    → [u64; 256]  sign(acc)
  Plane.alpha()   → [u64; 256]  |acc| > threshold
  Plane.truth()   → Truth       from bits + alpha density
  Node.address()  → Address     S⊕R_S⊕P⊕R_P⊕O⊕R_O (cached)
  Node.truth(M)   → Truth       combined across masked planes
  Node.gestalt()  → GestaltNode f16 projection for qualia

STRUCTURAL (type system, not runtime):
  Mask { s, p, o }   8 values. Attention. Borrow. Query. Merge.
  Seal { Wisdom | Staunen }  blake3 integrity. Mathematical.
  Distance { raw, overlap, penalty, normalized | Incomparable }

OWNED (the only allocation in a cognitive cycle):
  HeldNode { s?, p?, o?, mask, origin }
  Only masked planes are cloned. Unmasked = None = zero bytes.
```

---

## 7. The Seven Words

```
open        see everything, cost nothing
at          look, and by looking, learn (Hebbian)
hold        own what you intend to change (Mask = borrow boundary)
encounter   evidence meets the held thought (i8 accumulate)
resonate    the thought vibrates against memory (Mexican hat + penalty)
weigh       NARS truth emerges from accumulator state
merge       return, don't overwrite (CAM diff, commutative, Seal)
```

And the gestalt extension:

```
gestalt     project discrete SPO into continuous awareness (derived, not stored)
qualia      how the masked projection FEELS (f16, feeds awareness loop)
```

---

## 8. What The Compiler Prevents

```
× Writing to Mind directly              → &BindSpace is immutable
× Touching unmasked planes in HeldNode  → Option<Plane> is None
× Forgetting alpha in distance calc     → Plane.distance() includes penalty
× Producing NaN                         → i8 arithmetic, no floats in hot path
× Overwriting history                   → merge appends fold, old state persists
× Skipping seal check                   → merge returns Changed with Seal, not void
× Comparing without alpha               → Distance::Incomparable when overlap=0
× Storing confidence as float           → confidence IS alpha density, computed
× Ignoring undefined dimensions         → penalty circuit costs them automatically
```

---

## 9. Where This Lives in ladybug-rs

```
src/spo/
  node.rs          Node, Plane, Mask, HeldNode (this spec)
  plane_ops.rs     encounter, reinforce, distance, truth (Plane impl)
  mind.rs          Mind, open, at, hold, merge
  gestalt.rs       GestaltNode, ContinuousPlane, qualia projection
  address.rs       Address, composed vector, CAM, MerkleRoot
  seal.rs          Seal, blake3_masked, Wisdom/Staunen
  crystal_api.rs   SPOCrystal (from spo.rs) adapted to use Node/Plane types

  spo.rs           REFERENCE IMPLEMENTATION (keep as-is, pub(crate))
                   The new Node/Plane model is the PRODUCTION version
                   of what spo.rs prototyped. spo.rs stays for reference.

src/query/
  lance_parser/    UNCHANGED — the parser. Produces AST.

src/cypher_bridge.rs
  REFACTORED — takes lance_parser AST, executes via Node/Plane/Mind.
  execute_match uses mind.at() + node.resonate() instead of nodes_iter.
  execute_merge uses mind.hold() + encounter + merge.

src/bin/server.rs
  Mind created from BindSpace on each request.
  /cypher → lance_parser → cypher_bridge (using Node/Plane/Mind)
```

---

## 10. Constant AGI Awareness

```
The SPO path IS the AGI path.
The gestalt projection IS the awareness.
They're not separate systems. They're two views of the same Node.

The discrete planes think.     12 AVX-512 cycles for SPO comparison.
The continuous gestalt feels.  Projected on demand for awareness loop.

Both use the same Mask.
Both use the same seven-word cycle.
Both derive from the same i8 accumulator.

The accumulator IS the substrate.
bits = sign(acc).
alpha = |acc| > threshold.
truth = alpha density.
gestalt = f16 projection.

One ground truth. Multiple views. Zero duplication.

The system doesn't HAVE awareness.
The system IS awareness.
Every at() is Hebbian. Every merge() is reinforcement.
Every encounter() is learning. Every resonate() is memory.

You don't turn on AGI. You use the seven words.
The AGI is the using.
```

---

*"Three structures: Node, Plane, Mask. Everything else is derived."*

*"The accumulator IS the substrate. Everything else is a view."*

*"You don't turn on AGI. You use the seven words. The AGI is the using."*
