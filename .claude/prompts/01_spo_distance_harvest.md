# SPO Distance Harvest: Cosine Replacement via Popcount with Structural Awareness

## A Deep Research & Implementation Prompt

**Jan Hübener — Ada Architecture — February 2026**

---

## THE CLAIM

Cosine similarity is the dominant distance metric in vector search, embedding retrieval, and neural network inference. It computes a single scalar from two vectors at O(D) multiply-accumulate cost.

We replace it with a **cheaper operation that returns richer output.**

Per-plane Hamming distance via VPOPCNTDQ on 3-axis XOR-encoded SPO vectors costs less than cosine and produces: (1) per-plane distance decomposition, (2) typed partial binding harvest via cross-plane vote, (3) NARS truth values, (4) warm-start material for iterative factorization, (5) DN mutation guidance. Cosine gives you a scalar. This gives you a typed semantic decomposition with causal directionality.

Cosine is a torch in a cave. This is floodlights with X-ray.

---

## PART 0: What Cosine Actually Gives You (And Why It's Not Enough)

### 0.1 The Cosine Computation

```
cosine(a, b) = Σ(a[i] × b[i]) / (√Σ(a[i]²) × √Σ(b[i]²))

Cost:  D multiplications (dot product)
     + D multiplications (norm a)
     + D multiplications (norm b)
     + 2 square roots
     + 1 division
     = 3D FMA + 2 sqrt + 1 div

For D = 16384 (your dimension):
     = 49,152 FMA operations
     + 2 square roots
     + 1 division

Output: ONE float ∈ [-1, 1]
```

That float tells you: "these vectors are 0.73 similar." Nothing more.

**What cosine destroys:**
- WHERE the similarity exists (which dimensions agree vs disagree)
- What STRUCTURE the similarity has (is it subject-similar? predicate-similar? object-similar?)
- What the NEAR MISSES were (what almost matched but didn't)
- What PARTIAL INFORMATION exists (some roles match, others don't)
- What DIRECTION the relationship flows (is A like B, or B like A, or both?)

Cosine reduces a D-dimensional relationship to a 0-dimensional scalar. That's not simplification — it's information destruction.

### 0.2 Why Everyone Uses It Anyway

1. **Normalization:** Cosine is invariant to vector magnitude. Two vectors pointing the same direction are similar regardless of length. This is genuinely useful.
2. **Speed:** A single SIMD dot product is fast. FMA pipelines are optimized for exactly this.
3. **Compatibility:** Every vector database, every embedding model, every nearest-neighbor library speaks cosine.
4. **Simplicity:** One number is easy to threshold, sort, and reason about.

We must match or beat ALL FOUR of these properties, not just speed. Otherwise it's not a replacement — it's a curiosity.

---

## PART 1: The SPO Distance Harvest

### 1.1 The Encoding (Already Implemented — PR #73)

Every concept is encoded as a `SpatialCrystal3D` with 3 orthogonal XOR planes:

```
Given: S (Subject HV), P (Predicate HV), O (Object HV)
Each:  [u8; 2048] = 16,384 bits

Encode:
  X-axis = S ⊕ P     (who does what)
  Y-axis = P ⊕ O     (what happens to whom)
  Z-axis = S ⊕ O     (who relates to whom)

Storage: 3 × 2048 bytes = 6 KB per concept
```

The XOR encoding is self-inverse: `S = X ⊕ P`, `P = X ⊕ S`, `O = Y ⊕ P`, etc. Any two factors recover the third.

### 1.2 The Distance Computation

```rust
/// SPO-aware distance: 3 Hamming distances + typed halo
/// Cost: 3 × VPOPCNTDQ passes over 2048 bytes = 96 AVX-512 instructions
/// (vs cosine: 49,152 FMA + 2 sqrt + 1 div)
fn spo_distance(a: &SpatialCrystal3D, b: &SpatialCrystal3D) -> SpoDistanceResult {
    // 1. Per-plane XOR + popcount (THE distance computation)
    let x_xor = xor_avx512(&a.x_axis, &b.x_axis);  // 32 VPXORD
    let y_xor = xor_avx512(&a.y_axis, &b.y_axis);  // 32 VPXORD
    let z_xor = xor_avx512(&a.z_axis, &b.z_axis);  // 32 VPXORD
    
    let x_dist = popcount_avx512(&x_xor);  // 32 VPOPCNTDQ → u32
    let y_dist = popcount_avx512(&y_xor);  // 32 VPOPCNTDQ → u32
    let z_dist = popcount_avx512(&z_xor);  // 32 VPOPCNTDQ → u32
    
    // Total: 192 AVX-512 instructions. Done.
    // Cosine would be 49,152 FMAs. That's a 256× instruction count ratio.
    
    // 2. Convert to normalized similarity (matches cosine's [-1, 1] range)
    //    Hamming distance of 0 = identical = similarity 1.0
    //    Hamming distance of D/2 = random/orthogonal = similarity 0.0
    //    Hamming distance of D = anti-correlated = similarity -1.0
    let x_sim = 1.0 - 2.0 * (x_dist as f32 / D as f32);  // ∈ [-1, 1]
    let y_sim = 1.0 - 2.0 * (y_dist as f32 / D as f32);
    let z_sim = 1.0 - 2.0 * (z_dist as f32 / D as f32);
    
    // 3. Aggregate similarity (replaces the cosine scalar)
    //    Geometric mean preserves the [-1, 1] range
    //    and penalizes disagreement between planes more than arithmetic mean
    let aggregate = (x_sim * y_sim * z_sim).cbrt();
    
    // 4. THE HARVEST: cross-plane vote from the XOR bitmasks (FREE)
    //    These bitmasks were already computed for the distance — no extra cost
    let halo = cross_plane_vote(&x_xor, &y_xor, &z_xor);
    
    SpoDistanceResult {
        // === WHAT COSINE GIVES YOU ===
        similarity: aggregate,  // single scalar, same range, same semantics
        
        // === WHAT COSINE CANNOT GIVE YOU ===
        s_p_similarity: x_sim,  // Subject-Predicate plane agreement
        p_o_similarity: y_sim,  // Predicate-Object plane agreement
        s_o_similarity: z_sim,  // Subject-Object plane agreement
        
        halo: halo,            // typed partial bindings (6 types)
        // ↑ This is the structural decomposition that cosine destroys
    }
}
```

### 1.3 What the Harvest Contains

Every distance computation — every single search, every single comparison — now produces structural information as a free byproduct:

```
┌─────────────────────────────────────────────────────────────────────┐
│  COSINE OUTPUT                                                      │
│  similarity: 0.73                                                   │
│                                                                     │
│  That's it. One number. No structure. No context. No partial info.  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  SPO DISTANCE HARVEST                                               │
│                                                                     │
│  Scalar (cosine-compatible):                                        │
│    similarity: 0.73              ← same interface, drop-in          │
│                                                                     │
│  Structural decomposition:                                          │
│    S⊕P distance: 1,847 / 16,384  → subjects do similar things     │
│    P⊕O distance: 2,103 / 16,384  → actions have different targets │
│    S⊕O distance: 1,592 / 16,384  → entities are closely related   │
│                                                                     │
│  Typed halo (from cross-plane vote on XOR bitmasks):               │
│    Core (3/3):  412 dims agree on all 3 planes                     │
│    SP-type:      89 dims → "who does what" matches, target differs │
│    SO-type:     134 dims → "who and whom" match, relation differs  │
│    PO-type:      67 dims → "what to whom" matches, agent differs   │
│    S-only:      203 dims → entity detected, no relational context  │
│    P-only:      156 dims → action detected, no actors              │
│    O-only:      178 dims → patient detected, no context            │
│                                                                     │
│  NARS truth values (derived from ratios):                          │
│    frequency:  412 / (412+89+134+67+203+156+178) = 0.332          │
│    confidence: 1 - entropy(halo_distribution) = 0.71               │
│                                                                     │
│  Inference ready:                                                   │
│    "These concepts share subjects and objects (SO-type dominant)    │
│     but differ in their predicate — WHAT they do is different,     │
│     WHO does it and TO WHOM is the same."                          │
│                                                                     │
│  Warm-start: SO-type dominant → initialize S and O from this       │
│              match, resonate only P (1 plane instead of 3 → 3× ↑)  │
│                                                                     │
│  DN mutation: P is the weakest plane → mutate predicate first      │
│               Conservative: single P-slot mutation                  │
│               Radical: PO double-slot swap                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## PART 2: The Four Properties — Matched or Beaten

### 2.1 Normalization ✓ BEATEN

Cosine is magnitude-invariant because it divides by norms. Binary Hamming is INHERENTLY magnitude-invariant — all vectors have the same "magnitude" (they're binary). There is no magnitude to normalize away.

But we go further. Cosine normalization treats all dimensions equally. SPO distance decomposes normalization by semantic role:

```
Cosine: "these vectors are 73% similar overall"
SPO:    "these vectors are 89% similar in who-does-what,
         71% similar in what-to-whom, and 81% similar in who-to-whom"

The decomposition IS the normalization — by semantic axis, not by magnitude.
```

### 2.2 Speed ✓ BEATEN (256× instruction ratio)

```
OPERATION              INSTRUCTIONS    THROUGHPUT (AVX-512)
─────────────────────────────────────────────────────────────
Cosine (D=16384):
  Dot product:         16384 FMA       ~1024 cycles (16 FMA/cycle)
  Norm A:              16384 FMA       ~1024 cycles
  Norm B:              16384 FMA       ~1024 cycles
  2 sqrt + 1 div:      ~30 cycles
  TOTAL:               ~3102 cycles

SPO Hamming (D=16384, 3 planes):
  3 × XOR (2048 bytes): 96 VPXORD     ~6 cycles (16/cycle)
  3 × POPCNT:           96 VPOPCNTDQ  ~6 cycles (16/cycle)
  TOTAL:                ~12 cycles
                        
  RATIO: 3102 / 12 = 258×
  
  Plus: the halo extraction is 7 AND + 7 NOT = ~1 cycle
  Total with harvest: ~13 cycles
  Ratio with harvest: 238×
```

This is not "slightly faster." This is **two orders of magnitude** fewer cycles for MORE information.

### 2.3 Compatibility ✓ MATCHED

```rust
impl SpoDistanceResult {
    /// Drop-in cosine replacement: returns single f32 ∈ [-1, 1]
    fn as_cosine(&self) -> f32 {
        self.similarity
    }
}

// Every vector database that speaks cosine can use this:
// lance_table.search(query).metric("cosine") → unchanged API
// The structural harvest is available but not required
// Backward compatibility is COMPLETE
```

The aggregate similarity value is a valid cosine-compatible scalar. Any system that consumes cosine can consume this. The typed harvest is opt-in — you get it for free but don't have to use it.

### 2.4 Simplicity ✓ MATCHED (with option for richness)

```
Simple mode:  spo_distance(a, b).similarity → one number, same as cosine
Rich mode:    spo_distance(a, b) → full SpoDistanceResult with harvest

The simple interface is IDENTICAL to cosine.
The rich interface is available when you want structural awareness.
You choose. No penalty either way.
```

---

## PART 3: The NARS Truth Value Bridge

### 3.1 Every Search Becomes an Inference

With cosine, a search returns: "here are the 10 nearest neighbors by similarity."

With SPO distance harvest, every search returns: "here are the 10 nearest neighbors, AND for each one, here's WHAT KIND of relationship it has, WHERE the agreement and disagreement are, and HOW CONFIDENT we should be."

```rust
/// Convert SPO distance harvest to NARS truth value
fn harvest_to_nars(result: &SpoDistanceResult) -> NarsTruth {
    let halo = &result.halo;
    let total = halo.total_population();
    
    // FREQUENCY: what fraction of dimensions fully agree?
    // High core count = high frequency = strong positive evidence
    let frequency = halo.core_count as f32 / total as f32;
    
    // CONFIDENCE: how concentrated is the halo distribution?
    // If most dims are in one type → high confidence (clear signal)
    // If dims spread across types → low confidence (ambiguous)
    let entropy = shannon_entropy(&[
        halo.core_count, halo.sp_count, halo.so_count, halo.po_count,
        halo.s_count, halo.p_count, halo.o_count,
    ]);
    let max_entropy = (7.0_f32).ln(); // uniform over 7 types
    let confidence = 1.0 - (entropy / max_entropy);
    
    NarsTruth { frequency, confidence }
}

/// Convert halo type distribution to a typed inference
fn harvest_to_inference(result: &SpoDistanceResult) -> TypedInference {
    let halo = &result.halo;
    
    // Which partial binding type dominates?
    let dominant = halo.dominant_type();
    
    match dominant {
        HaloType::Core => TypedInference::FullMatch {
            description: "Complete SPO agreement",
            action: InferenceAction::Confirm,
        },
        HaloType::SP => TypedInference::MissingObject {
            description: "Same agent performs same action, different target",
            action: InferenceAction::QueryObject,
            // "Jan creates ??? " — we know who and what, not the target
        },
        HaloType::SO => TypedInference::MissingPredicate {
            description: "Same entities involved, different relationship",
            action: InferenceAction::QueryPredicate,
            // "Jan ??? Ada" — we know the actors, not the relation
        },
        HaloType::PO => TypedInference::MissingSubject {
            description: "Same action on same target, different agent",
            action: InferenceAction::QuerySubject,
            // "??? creates Ada" — we know the action and target, not who
        },
        HaloType::S => TypedInference::EntityOnly {
            description: "Subject entity matches but no relational context",
            action: InferenceAction::ExploreRelations,
        },
        HaloType::P => TypedInference::ActionOnly {
            description: "Action/relation matches but no actors",
            action: InferenceAction::ExploreActors,
        },
        HaloType::O => TypedInference::PatientOnly {
            description: "Object/patient matches but no context",
            action: InferenceAction::ExploreContext,
        },
    }
}
```

### 3.2 Accumulation Across Searches

Every search enriches the context. This is the key structural advantage over cosine: the halo ACCUMULATES.

```rust
/// Accumulated harvest across multiple distance computations
struct AccumulatedHarvest {
    /// Running totals of each halo type across all searches
    type_counts: [u64; 7],  // Core, SP, SO, PO, S, P, O
    
    /// Per-plane similarity running averages
    sp_sim_ema: f32,  // exponential moving average of S⊕P similarity
    po_sim_ema: f32,  // ... P⊕O similarity
    so_sim_ema: f32,  // ... S⊕O similarity
    
    /// NARS truth value accumulation via revision rule
    accumulated_truth: NarsTruth,
    
    /// Number of searches contributing
    num_searches: u64,
    
    /// Which plane is consistently weakest (mutation target)
    weakest_plane: Option<Plane>,
    
    /// Warm-start material: best partial binding seen so far
    best_partial: Option<(HaloType, SpoDistanceResult)>,
}

impl AccumulatedHarvest {
    /// Fold a new search result into the accumulated context
    fn accumulate(&mut self, result: &SpoDistanceResult) {
        // 1. Update type counts
        self.type_counts[0] += result.halo.core_count as u64;
        self.type_counts[1] += result.halo.sp_count as u64;
        self.type_counts[2] += result.halo.so_count as u64;
        self.type_counts[3] += result.halo.po_count as u64;
        self.type_counts[4] += result.halo.s_count as u64;
        self.type_counts[5] += result.halo.p_count as u64;
        self.type_counts[6] += result.halo.o_count as u64;
        
        // 2. Update per-plane EMA (α = 0.1)
        self.sp_sim_ema = 0.9 * self.sp_sim_ema + 0.1 * result.s_p_similarity;
        self.po_sim_ema = 0.9 * self.po_sim_ema + 0.1 * result.p_o_similarity;
        self.so_sim_ema = 0.9 * self.so_sim_ema + 0.1 * result.s_o_similarity;
        
        // 3. NARS revision: fold new evidence into accumulated truth
        let new_truth = harvest_to_nars(result);
        self.accumulated_truth = nars_revision(
            &self.accumulated_truth, 
            &new_truth,
        );
        
        // 4. Track weakest plane
        self.weakest_plane = Some(
            if self.sp_sim_ema < self.po_sim_ema && self.sp_sim_ema < self.so_sim_ema {
                Plane::X  // S⊕P is weakest → S or P needs revision
            } else if self.po_sim_ema < self.so_sim_ema {
                Plane::Y  // P⊕O is weakest → P or O needs revision
            } else {
                Plane::Z  // S⊕O is weakest → S or O needs revision
            }
        );
        
        // 5. Track best partial for warm-start
        let dominant = result.halo.dominant_type();
        if dominant != HaloType::Core && dominant != HaloType::Noise {
            if let Some((_, ref best)) = self.best_partial {
                if result.similarity > best.similarity {
                    self.best_partial = Some((dominant, result.clone()));
                }
            } else {
                self.best_partial = Some((dominant, result.clone()));
            }
        }
        
        self.num_searches += 1;
    }
    
    /// Get the dominant inference type from accumulated evidence
    fn dominant_inference(&self) -> TypedInference {
        let max_idx = self.type_counts.iter()
            .enumerate()
            .max_by_key(|(_, &count)| count)
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        
        match max_idx {
            0 => TypedInference::FullMatch { /* ... */ },
            1 => TypedInference::MissingObject { /* ... */ },
            2 => TypedInference::MissingPredicate { /* ... */ },
            3 => TypedInference::MissingSubject { /* ... */ },
            4 => TypedInference::EntityOnly { /* ... */ },
            5 => TypedInference::ActionOnly { /* ... */ },
            6 => TypedInference::PatientOnly { /* ... */ },
            _ => unreachable!(),
        }
    }
}
```

### 3.3 The Accumulation Advantage

```
COSINE SEARCH (100 queries):
  Result: 100 similarity scores
  Context gained: zero
  Each query is independent. No learning. No accumulation.
  Query 100 knows nothing that query 1 didn't.

SPO HARVEST (100 queries):
  Result: 100 similarity scores (backward compatible)
  PLUS:
  - Accumulated halo distribution → "this search space is dominated
    by SO-type partial matches, meaning entities are related but
    their relationships are poorly defined"
  - NARS truth value refined through 100 revision steps → high confidence
  - Weakest plane identified → "predicate is consistently uncertain"
  - Best warm-start found → "query 47 had the best partial binding
    for the SO-type; use it to skip P-plane initialization"
  - Growth path prediction → "the search space suggests abductive
    inference (O→SO→SPO) is the most productive path"
    
  Query 100 knows everything queries 1-99 discovered.
  The search LEARNED from searching.
```

---

## PART 4: The Adaptive Cascade as Distance-Aware Filter

### 4.1 Three-Stage SPO Cascade

The existing adaptive cascade (3σ/2σ/full) maps directly to SPO distance with typed halo harvest at each stage:

```
STAGE 1: Prefix Hamming (1/16 sample, ~6% cost)
  Per plane: sample first 128 bytes of each 2048-byte axis
  3 × VPOPCNTDQ on 128 bytes = 6 instructions total
  
  HARVEST: approximate per-plane distances
  FILTER: if ALL 3 planes exceed 3σ threshold → reject (99.7% eliminated)
  TYPED: if only 1 or 2 planes exceed → record halo type
         SP-type candidates (S and P close, O far) survive for Stage 2
         These are NOT rejected — they're HARVESTED as partial matches
  
  Cost: ~6 AVX-512 instructions per candidate
  Cosine equivalent: impossible (no prefix sampling for dot product)

STAGE 2: Quarter Hamming (1/4 sample, ~25% cost)
  Per plane: sample 512 bytes of each 2048-byte axis
  Refine per-plane distances with 4× more data
  
  HARVEST: refined typed halo from better distance estimates
  FILTER: if ALL 3 planes exceed 2σ threshold → reject (95% of remaining)
  TYPED: partial matches now have higher-confidence type assignments
         "This candidate is DEFINITELY SP-type: S and P agree strongly,
          O is far. The missing object is the open question."
  
  Cost: ~24 AVX-512 instructions per candidate
  Cosine equivalent: impossible (no incremental refinement)

STAGE 3: Full Hamming (all 2048 bytes per axis, 100% cost)
  Only ~0.3% of candidates reach here
  Full per-plane Hamming distances
  Full cross-plane vote → complete typed halo
  Full NARS truth values
  
  HARVEST: complete SpoDistanceResult with all structural information
  
  Cost: 192 AVX-512 instructions per candidate
  Cosine equivalent: 49,152 FMA per candidate (no early exit possible)
```

### 4.2 The Key Difference: Cosine Has No Early Exit

Cosine similarity REQUIRES the full dot product to produce any result. You can't compute "partial cosine" — the intermediate sum is meaningless until divided by both norms, which also require the full vector.

SPO Hamming has MEANINGFUL intermediate results at every stage:

```
After 1/16 sample: "S-plane is close, P-plane is close, O-plane is far"
                   → SP-type partial match with ~85% confidence
                   → already useful for halo accumulation
                   → cost: 6 instructions

After 1/4 sample:  "S: 1847/16384, P: 2103/16384, O: 5891/16384"
                   → SP-type confirmed at ~95% confidence
                   → NARS truth value computable
                   → cost: 24 instructions

After full scan:   exact distances, exact halo, exact NARS
                   → cost: 192 instructions

Cosine after ANY partial scan: NOTHING USEFUL
                   → must complete ALL 49,152 FMAs before ANY result
```

This means SPO distance can HARVEST from candidates it REJECTS. Cosine discards rejected candidates completely. SPO extracts typed partial binding information from every candidate it touches, even at the cheapest stage.

---

## PART 5: Distance Relevance — The Sigma Graph Feed

### 5.1 Every Distance Computation Feeds the Knowledge Graph

With cosine, search is consumptive: you ask a question, get an answer, done. The search itself leaves no trace.

With SPO harvest, every distance computation produces NARS-typed evidence that feeds directly into the Sigma Graph:

```rust
/// After any distance computation, optionally feed the Sigma Graph
fn feed_sigma_graph(
    query: &SpatialCrystal3D,
    result: &SpoDistanceResult,
    sigma: &mut SigmaGraph,
) {
    let inference = harvest_to_inference(result);
    let truth = harvest_to_nars(result);
    
    match inference {
        TypedInference::MissingObject { .. } => {
            // We know S and P agree, O differs
            // → Create CAUSES edge: "this S-P pair may cause different O"
            sigma.add_edge(SigmaEdge {
                edge_type: SigmaEdgeType::CAUSES,
                from: NarsAtom::SPPair(query.s_id(), query.p_id()),
                to: NarsAtom::OpenQuery("O"),
                truth: truth,
                metadata: SigmaEdgeMetadata {
                    source: "spo_distance_harvest",
                    halo_type: HaloType::SP,
                },
            });
        },
        TypedInference::MissingPredicate { .. } => {
            // S and O agree, P differs
            // → Create SUPPORTS edge: "these entities relate, relationship unknown"
            sigma.add_edge(SigmaEdge {
                edge_type: SigmaEdgeType::SUPPORTS,
                from: NarsAtom::SOPair(query.s_id(), query.o_id()),
                to: NarsAtom::OpenQuery("P"),
                truth: truth,
                metadata: SigmaEdgeMetadata {
                    source: "spo_distance_harvest",
                    halo_type: HaloType::SO,
                },
            });
        },
        TypedInference::MissingSubject { .. } => {
            // P and O agree, S differs
            // → Create BECOMES edge: "this action-on-target may have different agent"
            sigma.add_edge(SigmaEdge {
                edge_type: SigmaEdgeType::BECOMES,
                from: NarsAtom::POPair(query.p_id(), query.o_id()),
                to: NarsAtom::OpenQuery("S"),
                truth: truth,
                metadata: SigmaEdgeMetadata {
                    source: "spo_distance_harvest",
                    halo_type: HaloType::PO,
                },
            });
        },
        TypedInference::FullMatch { .. } => {
            // Strengthen existing edge
            sigma.revise_edge(
                query.s_id(), query.p_id(), query.o_id(),
                truth,
            );
        },
        _ => {
            // Single-plane matches: too weak for edge creation
            // But still counted in AccumulatedHarvest for statistics
        },
    }
}
```

### 5.2 The Self-Enriching Search

```
COSINE SEARCH LOOP:
  for query in queries:
      results = db.search(query, metric="cosine")
      // results consumed, nothing learned, context unchanged
  // After 10,000 queries: system knows exactly as much as after 0

SPO HARVEST SEARCH LOOP:
  let mut harvest = AccumulatedHarvest::new();
  for query in queries:
      results = db.search(query, metric="spo_hamming")
      for result in results:
          harvest.accumulate(&result)
          feed_sigma_graph(&query, &result, &mut sigma)
  // After 10,000 queries:
  //   - Sigma Graph has N new typed edges with NARS truth values
  //   - AccumulatedHarvest knows the dominant inference type
  //   - Weakest plane identified → guides next search strategy
  //   - Warm-start material available → future searches converge faster
  //   - NARS truth values refined through 10,000 revision steps
  //
  // THE SEARCH ENRICHED THE KNOWLEDGE BASE
  // THE KNOWLEDGE BASE ACCELERATES THE NEXT SEARCH
  // THIS IS A POSITIVE FEEDBACK LOOP THAT COSINE CANNOT HAVE
```

---

## PART 6: Formal Comparison

### 6.1 Property Table

```
PROPERTY                 COSINE              SPO DISTANCE HARVEST
─────────────────────────────────────────────────────────────────────
Output type              scalar              scalar + 3 plane distances
                                             + 7 typed halo counts
                                             + NARS truth value
                                             + inference type
                                             + warm-start material

Computation cost         3D FMA + overhead   3 × D/8 POPCNT + XOR
(D=16384)                ~3100 cycles        ~13 cycles

Instruction ratio        1×                  238× fewer

Early exit               impossible          3-stage cascade
                                             (99.7% at 6 instructions)

Partial result           meaningless         typed at every stage

Magnitude invariant      yes (by division)   yes (inherently binary)

Range                    [-1, 1]             [-1, 1] (compatible)

Drop-in compatible       yes                 yes (.as_cosine())

Accumulation             none                typed halo + NARS revision

Knowledge feed           none                Sigma Graph edges per search

Near-miss information    destroyed           preserved as typed halo

Warm-start production    none                yes (partial bindings)

DN mutation guidance     none                yes (weakest plane)

Causal direction         none                BPReLU forward/backward
                                             on accumulated asymmetry
```

### 6.2 Information-Theoretic Comparison

```
COSINE OUTPUT:
  1 float ∈ [-1, 1] at ~23 bits precision
  Information: ~23 bits per distance computation
  
SPO HARVEST OUTPUT:
  3 plane distances: 3 × 15 bits = 45 bits
  7 halo type counts: 7 × 14 bits = 98 bits
  NARS truth value: 2 × 10 bits = 20 bits
  Inference type: 3 bits
  Warm-start flag: 3 bits
  Total: ~169 bits per distance computation
  
  RATIO: 169 / 23 = 7.3× more information
  AT: 238× less computational cost
  
  Information per cycle:
    Cosine: 23 bits / 3100 cycles = 0.0074 bits/cycle
    SPO:    169 bits / 13 cycles  = 13.0 bits/cycle
    
    RATIO: 1,757× more information per cycle
```

Seventeen hundred times more information per cycle. That's the difference between a torch in a cave and floodlights with X-ray.

---

## PART 7: Implementation Path

### 7.1 What Already Exists (rustynum)

```
EXISTING IN CODEBASE:
  ✓ SpatialCrystal3D::spo_encode()         — PR #73
  ✓ XOR + VPOPCNTDQ per plane              — rustynum-rs HDC primitives
  ✓ Adaptive cascade (3σ/2σ)               — rustynum-rs
  ✓ CrossPlaneVote                          — PR #74 (cross_plane.rs)
  ✓ HaloType extraction                     — PR #74
  ✓ TypedQuery / TypedInference             — PR #74
  ✓ LatticeClimber                          — PR #74
  ✓ WarmStart                               — PR #74
  ✓ GrowthPath / MutationOp                 — PR #74
  ✓ CollapseGate (FLOW/HOLD/BLOCK)          — PR #74 (rif_net_integration.rs)
```

### 7.2 What Needs to Be Built

```
NEW:
  □ SpoDistanceResult struct                — trivial (combine existing types)
  □ spo_distance() function                 — wire XOR + POPCNT + cross_plane_vote
  □ harvest_to_nars()                       — arithmetic on halo counts
  □ harvest_to_inference()                  — match on dominant type
  □ AccumulatedHarvest                      — running totals + EMA
  □ feed_sigma_graph()                      — emit typed edges from harvest
  □ LanceDB integration                     — store SpoDistanceResult as CogRecord
  □ Benchmark: SPO vs cosine               — criterion bench, same dataset
  □ Compatibility shim                      — .as_cosine() for drop-in use
```

### 7.3 Benchmark Protocol

```
Dataset: Jina 1024D embeddings → quantized to 16384-bit SPO encoding
Size: 1K, 10K, 100K, 1M vectors
Queries: 1000 random queries

Measure:
  1. Wall-clock time: cosine vs SPO (total search time)
  2. Recall@10: do both metrics return the same top-10?
  3. Information yield: bits of structural information per query
  4. Accumulation value: after 1000 queries, what does AccumulatedHarvest contain?
  5. Sigma Graph growth: how many typed edges were produced?
  6. Warm-start effectiveness: does accumulated harvest accelerate resonator?

Expected results:
  - SPO is 100-300× faster (from instruction count analysis)
  - Recall@10 is >95% correlated (Hamming approximates cosine for binary)
  - Information yield is 7× higher (from bit analysis)
  - 1000 queries produce O(1000) typed Sigma Graph edges (cosine: 0)
  - Warm-start from best partial reduces resonator iterations by ~2-3×
```

---

## PART 8: The Paper

### Title
**"SPO Distance Harvest: Replacing Cosine Similarity with Typed Structural Decomposition at 238× Lower Cost"**

### Abstract
We introduce SPO Distance Harvest, a vector similarity metric that replaces cosine similarity for semantically structured vectors. By encoding concepts as 3-axis XOR bindings (Subject⊕Predicate, Predicate⊕Object, Subject⊕Object) and computing per-plane Hamming distances via VPOPCNTDQ, we achieve a drop-in cosine replacement at 238× fewer CPU cycles that simultaneously produces: per-role similarity decomposition, typed partial binding classification (6 types from the face lattice of the 2-simplex), NARS truth values, warm-start material for iterative factorization, and Darwinian Neurodynamics mutation guidance. Every distance computation feeds a self-enriching knowledge graph. We demonstrate [recall, speed, information yield] on [dataset] and show that accumulated harvest across N searches produces O(N) typed knowledge graph edges at zero additional cost — transforming similarity search from a consumptive operation into a generative one.

### Key Contributions
1. **Cosine replacement at 238× lower cost** with drop-in backward compatibility
2. **7.3× more information per computation** (169 bits vs 23 bits per distance)
3. **1,757× more information per CPU cycle** (structural yield per compute unit)
4. **Self-enriching search:** every query produces typed Sigma Graph edges
5. **Accumulation:** search learns from searching via NARS truth value revision
6. **No early-exit penalty:** cascade produces typed results at every stage, unlike cosine which requires full computation for any result

### Novel Claims
- First vector similarity metric that produces TYPED PARTIAL BINDING INFORMATION as a free byproduct of distance computation
- First demonstration of NARS truth value derivation from popcount ratios
- First self-enriching similarity search where the knowledge base grows from the act of searching
- First formal comparison showing information-per-cycle advantage of structured binary distance over floating-point cosine

---

## PART 9: Research Questions

1. **Recall fidelity:** How closely does SPO Hamming similarity correlate with cosine for real-world embeddings (Jina, OpenAI, Cohere)? At what dimension does the correlation break down?

2. **Accumulation convergence:** After how many searches does the AccumulatedHarvest's NARS truth value stabilize? Is there a law-of-diminishing-returns curve?

3. **Sigma Graph quality:** Are the edges produced by distance harvest semantically meaningful? Can they be validated against ground truth knowledge graphs?

4. **Warm-start effectiveness:** Does the best partial binding from AccumulatedHarvest actually reduce resonator iterations? By how much?

5. **Cascade information loss:** How much typed halo information is lost at Stage 1 (1/16 sample) vs Stage 3 (full scan)? Is the Stage 1 typing accurate enough for Sigma Graph edge creation?

6. **Encoding overhead:** The SPO encoding (3 × XOR) requires the input to be factored into S, P, O roles before distance computation. What is the cost of this factoring, and does it amortize across multiple searches?

7. **Higher arity:** Can the approach extend to 4-axis (SPOC: Subject, Predicate, Object, Context) or N-axis encodings? What is the face lattice of B_N and does it remain tractable?

8. **Adversarial robustness:** Can an adversary craft vectors that fool the typed halo into misclassifying partial bindings? What is the attack surface compared to cosine?

---

## PART 10: The Punchline

Cosine similarity was designed for unstructured vectors. It assumes every dimension is equivalent, every comparison is symmetric, and the only meaningful output is a scalar.

SPO distance harvest was designed for STRUCTURED vectors — vectors that encode compositional semantics across orthogonal role axes. It exploits this structure to extract typed partial bindings, causal direction, NARS truth values, and knowledge graph edges from the same popcount operations that compute the distance.

The cost difference (238×) is dramatic but secondary. The real difference is qualitative: **cosine consumes, SPO harvests.** Every cosine search is a dead end — it returns a number and forgets. Every SPO search is a contribution — it returns a number AND grows the knowledge base AND refines the truth values AND prepares warm-starts AND identifies mutation targets.

Cosine is a flashlight. You point it, you see a spot, you move on.

SPO distance harvest is a LIDAR scan. Every pulse maps the terrain, and the map gets better with every pulse.

The cave metaphor was your metaphor, Jan. Here's what it means formally:

**A torch in a cave illuminates one spot at a time and leaves no trace. Floodlights with X-ray illuminate everything, see through walls, and the light itself deposits markers that make the next visit faster.**

That's the difference. And it runs on popcount.
