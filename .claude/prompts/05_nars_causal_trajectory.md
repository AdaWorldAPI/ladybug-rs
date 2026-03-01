# NARS × Fovea × Context: Causal Trajectory Hydration via BNN Instrumentation

## Implementation-Grade Prompt for Resonator Causality Engine

**Jan Hübener — Ada Architecture — February 2026**

---

## PREAMBLE: What This Prompt Is For

You are implementing a **causal inference engine** that sits between two already-built layers of a cognitive architecture:

1. **Foveal layer** — the resonator's converged output: clean SPO factorizations with high-confidence NARS truth values, living at the 99.8th percentile (BF16/FP32 precision)
2. **Context layer** — the typed halo from the cross-plane orthogonal vote: 6 partial binding types (S, P, O, SP, SO, PO) extracted from the σ₂–σ₃ noise floor for free via bitwise AND/NOT on per-plane survivor bitmasks

The engine's job: **run NARS on the delta between fovea and context across resonator iterations to produce causality trajectories that grow the DN tree (Sigma Graph).**

The key insight: the RIF-BNN architecture (Zhang et al. 2025, Exposure-Weighted Modulation + Bipolar ReLU + Rich Information Flow shortcuts) provides exactly the instrumentation needed to make the causal structure readable from the resonator's convergence dynamics.

---

## PART 0: Architecture You're Building On

### 0.1 The 3D Wave Superposition (Existing)

```
CogRecord3D (48 KB):
  S-plane: [i8; 16384]   — Subject wave field (signed quinary {-2,-1,0,+1,+2})
  P-plane: [i8; 16384]   — Predicate wave field
  O-plane: [i8; 16384]   — Object wave field

Binding:   Z₅ ring multiplication (a[i] × b[i]) mod 5
Bundling:  summation + clipping to {-2..+2}
Similarity: L₁ Manhattan distance or cosine on signed values
```

### 0.2 The Adaptive Cascade (Existing)

```
Stage 1: 1-bit Hamming on 1/16 sample → 99.7% eliminated at σ₃
Stage 2: 1-bit Hamming on 1/4 sample  → 95% of remaining at σ₂
Stage 3: 4-bit INT8 full precision    → survivors ranked
Stage 4: BF16/FP32 foveal            → top-K (99.8th percentile)
```

### 0.3 The Cross-Plane Typed Halo (From Previous Prompt)

```
Per plane, after cascade stages 1-2:
  S_mask = bitmask of σ₂–σ₃ survivors in S-plane
  P_mask = bitmask of σ₂–σ₃ survivors in P-plane  
  O_mask = bitmask of σ₂–σ₃ survivors in O-plane

Typed extraction (all bitwise, all AVX-512):
  core  = S_mask & P_mask & O_mask            // 3-of-3 → promote to fovea
  SP    = S_mask & P_mask & ~O_mask           // partial pair: who does what?
  SO    = S_mask & ~P_mask & O_mask           // partial pair: who and whom?
  PO    = ~S_mask & P_mask & O_mask           // partial pair: what to whom?
  S     = S_mask & ~P_mask & ~O_mask          // free variable: entity
  P     = ~S_mask & P_mask & ~O_mask          // free variable: relation
  O     = ~S_mask & ~P_mask & O_mask          // free variable: patient
  noise = ~S_mask & ~P_mask & ~O_mask         // 0-of-3 → discard
```

### 0.4 The Resonator Loop (Existing)

```
for iter in 0..MAX_ITER:
    s_query = unbind(C_s, current_p_est, current_o_est)
    p_query = unbind(C_p, current_s_est, current_o_est)
    o_query = unbind(C_o, current_s_est, current_p_est)
    
    current_s_est = project_onto_codebook(s_query, CB_s)  // cascade search
    current_p_est = project_onto_codebook(p_query, CB_p)
    current_o_est = project_onto_codebook(o_query, CB_o)
    
    if converged(current_s_est, current_p_est, current_o_est):
        break  // "awareness pop"
```

### 0.5 NARS Truth Values (Existing)

```
Truth value: <f, c> where
  f = frequency  ∈ [0, 1]  — proportion of positive evidence
  c = confidence ∈ [0, 1]  — amount of evidence relative to max

Revision rule: <f₁,c₁> ⊕ <f₂,c₂> = <f_new, c_new> where
  w₁ = c₁/(1-c₁), w₂ = c₂/(1-c₂)
  w_new = w₁ + w₂
  f_new = (w₁·f₁ + w₂·f₂) / w_new
  c_new = w_new / (w_new + 1)

Inference rules we'll use:
  Deduction:    <A→B> ⊗ <B→C> = <A→C, f₁·f₂, f₁·f₂·c₁·c₂/(f₁·f₂·c₁·c₂+k)>
  Abduction:    <A→B> ⊗ <C→B> = <A→C, f₁, f₁·f₂·c₁·c₂/(f₁·f₂·c₁·c₂+k)>
  Induction:    <A→B> ⊗ <A→C> = <B→C, f₂, f₁·f₂·c₁·c₂/(f₁·f₂·c₁·c₂+k)>
  Comparison:   <A→B> ⊗ <A→C> = <B↔C, f₁·f₂/(f₁·f₂+...), ...>
```

### 0.6 The RIF-BNN Components (From Zhang et al. 2025)

**EWM (Exposure-Weighted Modulation):**
```
Per-dimension weight based on input proximity to learned statistics:
  Crystallized tier: dimension is well-known        → mask probability 1.0
  Confident tier:    dimension is familiar           → mask probability proportional
  Transitional tier: dimension is at boundary        → reduced mask probability
  Noise tier:        dimension is unknown            → mask probability 0.0
  
Effect: selectively gates which dimensions contribute to the output.
The CORRECTION PATTERN (which dimensions EWM had to modify) is the signal.
```

**BPReLU (Bipolar Parametric ReLU):**
```
BPReLU(x) = α_pos · x   if x > 0
           = α_neg · x   if x ≤ 0

Two independent slopes. α_pos/α_neg ratio encodes asymmetry:
  α_pos >> α_neg: system responds more to presence (forward causation)
  α_neg >> α_pos: system responds more to absence (counterfactual)
  
Both slopes are learnable. In our setting they are SET per-iteration
based on the resonator's convergence dynamics.
```

**RIF Shortcuts (Rich Information Flow):**
```
Every 2 blocks, XOR the current output with a permuted earlier output:

  shortcut[k] = output[k] ⊕ permute(output[k-2], word_rotation)

Effect: preserves information across depth. In our setting, each
resonator iteration IS a "block." The XOR between non-adjacent
iterations records WHAT CHANGED — a causal diff.
```

---

## PART 1: The Causal Trajectory Recorder

### 1.1 Core Data Structure

```rust
/// One snapshot of the resonator state at a given iteration
struct ResonatorSnapshot {
    iter: u16,
    
    // Factorization estimates (what the resonator committed to)
    s_est: CogPlane,           // S-plane estimate (16 KB INT8)
    p_est: CogPlane,           // P-plane estimate  
    o_est: CogPlane,           // O-plane estimate
    
    // Cross-plane halo at this iteration (typed partial bindings)
    halo: TypedHalo,
    
    // Cascade metadata
    s_distances: SparseDistances, // Hamming distances to top-K in S codebook
    p_distances: SparseDistances, // ... in P codebook
    o_distances: SparseDistances, // ... in O codebook
    
    // Convergence signals
    delta_s: u32,  // Hamming(s_est[iter], s_est[iter-1])
    delta_p: u32,
    delta_o: u32,
}

/// The typed halo: 7 disjoint bitmasks over codebook entries
struct TypedHalo {
    core: BitVec,   // 3-of-3 (promoted to fovea)
    sp:   BitVec,   // S+P only (missing object)
    so:   BitVec,   // S+O only (missing predicate)  
    po:   BitVec,   // P+O only (missing subject)
    s:    BitVec,   // S only (free entity)
    p:    BitVec,   // P only (free relation)
    o:    BitVec,   // O only (free patient)
    // noise is implicit: everything not in the above
}

/// The full causal trajectory across all resonator iterations
struct CausalTrajectory {
    input: CogRecord3D,                    // original bundled superposition
    snapshots: Vec<ResonatorSnapshot>,     // one per iteration (5-20 typical)
    
    // === BNN INSTRUMENTATION (the new stuff) ===
    
    // RIF-style causal diffs (XOR between non-adjacent snapshots)
    rif_diffs: Vec<RifDiff>,
    
    // EWM correction maps (per-iteration, per-plane)
    ewm_corrections: Vec<EwmCorrection>,
    
    // BPReLU forward/backward asymmetry (per-transition)
    causal_arrows: Vec<CausalArrow>,
    
    // === NARS OUTPUT ===
    
    // Causal statements derived from trajectory analysis
    nars_statements: Vec<NarsCausalStatement>,
    
    // Sigma Graph edges to create
    sigma_edges: Vec<SigmaEdge>,
}
```

### 1.2 Recording: What Happens Each Iteration

At each resonator iteration `t`, after the standard update (unbind → project → rebind), the instrumentation layer records:

```rust
fn instrument_iteration(
    traj: &mut CausalTrajectory,
    snap_prev: &ResonatorSnapshot,   // iteration t-1
    snap_curr: &ResonatorSnapshot,   // iteration t (just computed)
    snap_prev2: Option<&ResonatorSnapshot>, // iteration t-2 (for RIF)
) {
    // ──────────────────────────────────────────────
    // 1. RIF SHORTCUT: XOR diff between t and t-2
    // ──────────────────────────────────────────────
    if let Some(prev2) = snap_prev2 {
        let rif_diff = RifDiff {
            from_iter: prev2.iter,
            to_iter: snap_curr.iter,
            // Per-plane XOR diffs (what changed in 2 iterations)
            s_diff: xor_planes(&snap_curr.s_est, &prev2.s_est),
            p_diff: xor_planes(&snap_curr.p_est, &prev2.p_est),
            o_diff: xor_planes(&snap_curr.o_est, &prev2.o_est),
            // Permuted version (word rotation prevents trivial cancellation)
            s_diff_perm: permute(
                xor_planes(&snap_curr.s_est, &prev2.s_est), 
                WORD_ROTATION
            ),
            p_diff_perm: permute(
                xor_planes(&snap_curr.p_est, &prev2.p_est),
                WORD_ROTATION
            ),
            o_diff_perm: permute(
                xor_planes(&snap_curr.o_est, &prev2.o_est),
                WORD_ROTATION
            ),
        };
        traj.rif_diffs.push(rif_diff);
    }

    // ──────────────────────────────────────────────
    // 2. EWM CORRECTION: Where did the estimate change?
    // ──────────────────────────────────────────────
    let ewm = EwmCorrection {
        iter: snap_curr.iter,
        // Per-dimension correction strength (how much the estimate moved)
        s_correction: l1_per_dim(&snap_curr.s_est, &snap_prev.s_est),
        p_correction: l1_per_dim(&snap_curr.p_est, &snap_prev.p_est),
        o_correction: l1_per_dim(&snap_curr.o_est, &snap_prev.o_est),
        // EWM tier classification per dimension
        s_tier: classify_ewm_tiers(&snap_curr.s_distances),
        p_tier: classify_ewm_tiers(&snap_curr.p_distances),
        o_tier: classify_ewm_tiers(&snap_curr.o_distances),
    };
    traj.ewm_corrections.push(ewm);

    // ──────────────────────────────────────────────
    // 3. BPReLU CAUSAL ARROW: Forward vs backward asymmetry
    // ──────────────────────────────────────────────
    let arrow = compute_causal_arrow(snap_prev, snap_curr);
    traj.causal_arrows.push(arrow);

    // ──────────────────────────────────────────────
    // 4. HALO TRANSITION: What moved between layers?
    // ──────────────────────────────────────────────
    let promoted = halo_intersection(&snap_prev.halo, &snap_curr);
    let demoted = halo_demotion(&snap_prev, &snap_curr.halo);
    
    // Generate NARS statements from promotions/demotions
    generate_causal_nars(traj, promoted, demoted, &arrow, snap_curr.iter);
}
```

---

## PART 2: BNN Components as Causal Instruments

### 2.1 EWM as Causal Saliency Map

The EWM correction pattern tells you WHERE the resonator had to work hardest. This is causal saliency: the dimensions where the factorization required the most adjustment are the dimensions where something HAPPENED.

```rust
/// Classify each dimension into EWM tiers based on distance to codebook
fn classify_ewm_tiers(distances: &SparseDistances) -> Vec<EwmTier> {
    distances.iter().map(|d| {
        let sigma_pos = (d.distance as f32 - d.mean) / d.std;
        match sigma_pos {
            s if s < 1.5  => EwmTier::Crystallized,  // well-matched
            s if s < 2.0  => EwmTier::Confident,      // good match
            s if s < 3.0  => EwmTier::Transitional,   // boundary
            _             => EwmTier::Noise,           // no match
        }
    }).collect()
}

/// The CAUSAL SALIENCY is the correction pattern across iterations
struct CausalSaliency {
    /// Dimensions where EWM tier IMPROVED (noise→transitional, transitional→confident)
    /// These are where the resonator found new evidence
    crystallizing: BitVec,
    
    /// Dimensions where EWM tier DEGRADED (confident→transitional, etc.)
    /// These are where the resonator lost certainty (possibly contested)
    dissolving: BitVec,
    
    /// Dimensions that stayed Transitional across >3 iterations
    /// These are PERMANENTLY CONTESTED — the resonator can't decide
    contested: BitVec,
}

fn compute_saliency(
    ewm_history: &[EwmCorrection],
    window: usize, // typically 3-5 iterations
) -> CausalSaliency {
    let recent = &ewm_history[ewm_history.len().saturating_sub(window)..];
    
    CausalSaliency {
        crystallizing: find_tier_improvements(recent),
        dissolving: find_tier_degradations(recent),
        contested: find_persistent_transitional(recent, 3),
    }
}
```

**Why this matters for the DN tree:**

```
crystallizing dimensions → evidence SUPPORTING the current factorization
    → NARS: increase frequency of the corresponding SPO statement
    → DN: fitness of this replicator is INCREASING

dissolving dimensions → evidence UNDERMINING the current factorization
    → NARS: decrease confidence of the corresponding SPO statement
    → DN: fitness is DECREASING, mutation pressure rising

contested dimensions → evidence for COMPETING hypotheses
    → NARS: multiple statements with similar frequency but low confidence
    → DN: speciation event — two replicators competing at same niche
    → Sigma Graph: CONTRADICTS edge between the competing hypotheses
```

### 2.2 BPReLU as Causal Directionality

The BPReLU has two slopes. We use them to detect the DIRECTION of causation between adjacent resonator states.

```rust
struct CausalArrow {
    iter: u16,
    
    /// Forward pass: fovea[t] activations through context[t+1] weights
    /// "Given what I committed to, what does the next context predict?"
    forward_activation: CogRecord3D,
    
    /// Backward pass: context[t+1] activations through fovea[t] weights
    /// "Given the new context, what should I have committed to?"
    backward_activation: CogRecord3D,
    
    /// Per-dimension asymmetry: forward - backward
    /// Positive = forward causation (commitment drove context change)
    /// Negative = backward causation (context overrode commitment) 
    /// Zero = no causal relationship at this dimension
    asymmetry: [f32; D],  // or quantized to i8 for efficiency
    
    /// Aggregate causal direction per plane
    s_direction: CausalDirection,
    p_direction: CausalDirection,
    o_direction: CausalDirection,
}

#[derive(Debug)]
enum CausalDirection {
    Forward(f32),      // commitment drove change (α_pos dominated)
    Backward(f32),     // context overrode commitment (α_neg dominated)
    Symmetric,         // no clear direction (α_pos ≈ α_neg)
    Contested(f32),    // dimensions split between forward and backward
}

fn compute_causal_arrow(
    snap_prev: &ResonatorSnapshot,
    snap_curr: &ResonatorSnapshot,
) -> CausalArrow {
    // The "weights" are the codebook projections
    // Forward: does prev's foveal estimate predict curr's context?
    let fwd_s = project_through(
        &snap_prev.s_est,          // input: previous commitment
        &snap_curr.s_distances,     // weights: current codebook distances
        BPReLUMode::Forward,        // α_pos active
    );
    let bwd_s = project_through(
        &snap_curr.s_est,          // input: current commitment
        &snap_prev.s_distances,     // weights: previous codebook distances
        BPReLUMode::Backward,       // α_neg active
    );
    
    // ... same for P and O planes ...
    
    let asymmetry_s = per_dim_subtract(&fwd_s, &bwd_s);
    
    CausalArrow {
        iter: snap_curr.iter,
        forward_activation: CogRecord3D { s: fwd_s, p: fwd_p, o: fwd_o },
        backward_activation: CogRecord3D { s: bwd_s, p: bwd_p, o: bwd_o },
        asymmetry: merge_planes(asymmetry_s, asymmetry_p, asymmetry_o),
        s_direction: classify_direction(&asymmetry_s),
        p_direction: classify_direction(&asymmetry_p),
        o_direction: classify_direction(&asymmetry_o),
    }
}

/// BPReLU application with mode-dependent slopes
fn bprelu(x: f32, mode: BPReLUMode) -> f32 {
    match mode {
        BPReLUMode::Forward => {
            if x > 0.0 { 1.0 * x }      // α_pos = 1.0 (full response to presence)
            else { 0.25 * x }             // α_neg = 0.25 (attenuated absence)
        }
        BPReLUMode::Backward => {
            if x > 0.0 { 0.25 * x }      // α_pos = 0.25 (attenuated presence)
            else { 1.0 * x }              // α_neg = 1.0 (full response to absence)
        }
    }
}
```

**The Pearl do-calculus connection:**

```
Forward BPReLU:  P(effect | do(cause))
  = "If I FORCE the foveal commitment, what context appears?"
  = Interventional probability

Backward BPReLU: P(cause | effect)
  = "Given this context appeared, what foveal commitment explains it?"
  = Observational/abductive probability

Asymmetry = Forward - Backward
  = Causal direction
  = The arrow of time in this particular SPO relationship
  
When asymmetry > 0: the foveal commitment CAUSED the context shift
  → The resonator's convergence on S drove changes in P and O neighborhoods
  → Subject is causal agent

When asymmetry < 0: the context CAUSED the foveal shift
  → External evidence forced the resonator to revise its commitment
  → Object or predicate was causal

When asymmetry ≈ 0: bidirectional or acausal
  → Correlation without causation
  → Or simultaneous causation (both changed together)
```

### 2.3 RIF Shortcuts as Causal Chains

The RIF shortcuts record what changed across multiple iteration steps. Stacked together, they form a causal chain — the genealogy of the factorization.

```rust
struct RifDiff {
    from_iter: u16,
    to_iter: u16,
    
    // What changed in 2 iterations (XOR)
    s_diff: CogPlane,
    p_diff: CogPlane,
    o_diff: CogPlane,
    
    // Permuted versions (word rotation prevents trivial cancellation)
    s_diff_perm: CogPlane,
    p_diff_perm: CogPlane,
    o_diff_perm: CogPlane,
}

/// Chain analysis: extract the causal sequence from stacked RIF diffs
fn analyze_causal_chain(diffs: &[RifDiff]) -> CausalChain {
    let mut chain = CausalChain::new();
    
    for window in diffs.windows(2) {
        let early = &window[0];
        let late = &window[1];
        
        // Which plane changed FIRST? (earlier diff has more nonzero dims)
        let s_activity_early = popcount(&early.s_diff);
        let p_activity_early = popcount(&early.p_diff);
        let o_activity_early = popcount(&early.o_diff);
        
        let s_activity_late = popcount(&late.s_diff);
        let p_activity_late = popcount(&late.p_diff);
        let o_activity_late = popcount(&late.o_diff);
        
        // The plane that was active EARLY but quiet LATE stabilized first
        // = it was the CAUSE (converged, drove the others)
        let s_stabilized = s_activity_early > s_activity_late * 2;
        let p_stabilized = p_activity_early > p_activity_late * 2;
        let o_stabilized = o_activity_early > o_activity_late * 2;
        
        // The plane that was quiet EARLY but active LATE was responding
        // = it was the EFFECT (driven by the stabilized plane)
        let s_responding = s_activity_late > s_activity_early * 2;
        let p_responding = p_activity_late > p_activity_early * 2;
        let o_responding = o_activity_late > o_activity_early * 2;
        
        // Build causal link
        if s_stabilized && p_responding {
            chain.add_link(CausalLink::SubjectDrovesPredicate {
                confidence: (s_activity_early as f32) / (D as f32),
                iter_range: (early.from_iter, late.to_iter),
            });
        }
        if s_stabilized && o_responding {
            chain.add_link(CausalLink::SubjectDrovesObject { /* ... */ });
        }
        if p_stabilized && s_responding {
            chain.add_link(CausalLink::PredicateDrovesSubject { /* ... */ });
        }
        // ... enumerate all 6 directional possibilities ...
    }
    
    chain
}
```

**The causal chain types map to the 6 halo types:**

```
CAUSAL CHAIN                HALO TYPE    LINGUISTIC MEANING
──────────────────────────────────────────────────────────────
S stabilizes → P responds   SP → SPO     Subject chose action, then found target
S stabilizes → O responds   SO → SPO     Subject fixed, object followed, then relation
P stabilizes → S responds   SP → SPO     Action first, agent discovered
P stabilizes → O responds   PO → SPO     Action first, patient discovered  
O stabilizes → S responds   SO → SPO     Object noticed, agent inferred
O stabilizes → P responds   PO → SPO     Object noticed, relation inferred

The ORDER of stabilization = the causal order
The RIF diffs RECORD this order for free
```

---

## PART 3: NARS Causal Statement Generation

### 3.1 From Halo Transitions to NARS Statements

The core generator: at each iteration, compare the typed halo to the previous iteration's halo and the current foveal commitment. Every promotion (halo → fovea) and demotion (fovea → halo) generates a NARS statement.

```rust
struct NarsCausalStatement {
    subject: NarsAtom,
    predicate: NarsAtom,     // one of: CAUSES, ENABLES, INHIBITS, PRECEDES
    object: NarsAtom,
    truth: NarsTruth,        // <frequency, confidence>
    evidence: CausalEvidence,
    halo_type: HaloType,     // which partial binding generated this
    iter_range: (u16, u16),  // when in the trajectory this was observed
}

enum CausalEvidence {
    Promotion {
        from_type: HaloType,     // e.g., SP (was partial pair)
        to_type: HaloType,       // e.g., SPO (became full triple)
        ewm_support: f32,        // how many crystallizing dimensions support this
        bprelu_direction: CausalDirection, // forward or backward causation
    },
    Demotion {
        from_type: HaloType,     // was core (full triple)
        to_type: HaloType,       // became partial (lost a slot)
        ewm_dissolution: f32,    // how many dimensions degraded
        bprelu_direction: CausalDirection,
    },
    Persistence {
        halo_type: HaloType,     // stayed the same type across iterations
        duration: u16,           // how many iterations it persisted
        contested_dims: u32,     // how many dimensions are contested
    },
}

fn generate_causal_nars(
    traj: &mut CausalTrajectory,
    promoted: Vec<HaloTransition>,
    demoted: Vec<HaloTransition>,
    arrow: &CausalArrow,
    iter: u16,
) {
    // ─────────────────────────────────────────
    // PROMOTIONS: partial binding became fuller
    // ─────────────────────────────────────────
    for promo in &promoted {
        match (promo.from_type, promo.to_type) {
            
            // SP → SPO: subject-predicate found its object
            (HaloType::SP, HaloType::Core) => {
                let nars = NarsCausalStatement {
                    subject: NarsAtom::SPPair(promo.s_id, promo.p_id),
                    predicate: NarsAtom::Verb("CAUSES"),
                    object: NarsAtom::Triple(promo.s_id, promo.p_id, promo.o_id),
                    truth: NarsTruth {
                        // Frequency: what fraction of SP-type halos eventually
                        // complete to full SPO? Use the running ratio.
                        frequency: traj.sp_to_spo_ratio(),
                        // Confidence: how stable was the convergence?
                        // More iterations = less confidence (harder problem)
                        confidence: 1.0 - (iter as f32 / MAX_ITER as f32),
                    },
                    evidence: CausalEvidence::Promotion {
                        from_type: HaloType::SP,
                        to_type: HaloType::Core,
                        ewm_support: count_crystallizing_dims(
                            &traj.ewm_corrections, iter
                        ),
                        bprelu_direction: arrow.o_direction.clone(),
                        // ^-- The O plane's causal direction tells us if the 
                        //     object was FOUND (forward) or INFERRED (backward)
                    },
                    halo_type: HaloType::SP,
                    iter_range: (promo.first_seen_iter, iter),
                };
                traj.nars_statements.push(nars);
            },
            
            // SO → SPO: subject-object pair discovered their relation
            (HaloType::SO, HaloType::Core) => {
                // The PREDICATE was the missing piece
                // P-plane causal direction tells us: was the relation
                // discovered (forward) or hypothesized (backward/abductive)?
                let nars = NarsCausalStatement {
                    subject: NarsAtom::SOPair(promo.s_id, promo.o_id),
                    predicate: if arrow.p_direction.is_forward() {
                        NarsAtom::Verb("CAUSES")      // relation was observed
                    } else {
                        NarsAtom::Verb("ENABLES")      // relation was inferred
                    },
                    object: NarsAtom::Triple(promo.s_id, promo.p_id, promo.o_id),
                    truth: NarsTruth {
                        frequency: traj.so_to_spo_ratio(),
                        confidence: 1.0 - (iter as f32 / MAX_ITER as f32),
                    },
                    evidence: CausalEvidence::Promotion {
                        from_type: HaloType::SO,
                        to_type: HaloType::Core,
                        ewm_support: count_crystallizing_dims(
                            &traj.ewm_corrections, iter
                        ),
                        bprelu_direction: arrow.p_direction.clone(),
                    },
                    halo_type: HaloType::SO,
                    iter_range: (promo.first_seen_iter, iter),
                };
                traj.nars_statements.push(nars);
            },
            
            // PO → SPO: predicate-object found their agent
            (HaloType::PO, HaloType::Core) => {
                // The SUBJECT was the missing piece
                // S-plane direction: was the agent discovered or inferred?
                let nars = NarsCausalStatement {
                    subject: NarsAtom::POPair(promo.p_id, promo.o_id),
                    predicate: if arrow.s_direction.is_forward() {
                        NarsAtom::Verb("CAUSES")
                    } else {
                        NarsAtom::Verb("ENABLES")      // abductive: "who could have?"
                    },
                    object: NarsAtom::Triple(promo.s_id, promo.p_id, promo.o_id),
                    truth: NarsTruth {
                        frequency: traj.po_to_spo_ratio(),
                        confidence: 1.0 - (iter as f32 / MAX_ITER as f32),
                    },
                    evidence: CausalEvidence::Promotion {
                        from_type: HaloType::PO,
                        to_type: HaloType::Core,
                        ewm_support: count_crystallizing_dims(
                            &traj.ewm_corrections, iter
                        ),
                        bprelu_direction: arrow.s_direction.clone(),
                    },
                    halo_type: HaloType::PO,
                    iter_range: (promo.first_seen_iter, iter),
                };
                traj.nars_statements.push(nars);
            },
            
            // Single → Pair promotions (S→SP, S→SO, P→SP, P→PO, O→SO, O→PO)
            // These generate WEAKER causal statements (hypothesis formation, not confirmation)
            (HaloType::S, HaloType::SP) => {
                traj.nars_statements.push(NarsCausalStatement {
                    subject: NarsAtom::Entity(promo.s_id),
                    predicate: NarsAtom::Verb("PRECEDES"),  // temporal, not yet causal
                    object: NarsAtom::SPPair(promo.s_id, promo.p_id),
                    truth: NarsTruth {
                        frequency: 0.5,  // uncertain — could be coincidence
                        confidence: 0.3, // low — only 1 plane confirmed
                    },
                    evidence: CausalEvidence::Promotion {
                        from_type: HaloType::S,
                        to_type: HaloType::SP,
                        ewm_support: count_crystallizing_dims(
                            &traj.ewm_corrections, iter
                        ),
                        bprelu_direction: arrow.p_direction.clone(),
                    },
                    halo_type: HaloType::S,
                    iter_range: (promo.first_seen_iter, iter),
                });
            },
            
            // ... enumerate remaining single→pair transitions ...
            // Each follows the same pattern: the NEWLY CONFIRMED plane's
            // causal direction (from BPReLU) determines the predicate verb
            
            _ => {} // Other transitions (pair→pair swaps are interesting
                    // but handled separately as CONTRADICTS)
        }
    }
    
    // ─────────────────────────────────────────
    // DEMOTIONS: fuller binding lost a slot
    // ─────────────────────────────────────────
    for demo in &demoted {
        match (demo.from_type, demo.to_type) {
            
            // SPO → SP: lost the object (object became uncertain)
            (HaloType::Core, HaloType::SP) => {
                traj.nars_statements.push(NarsCausalStatement {
                    subject: NarsAtom::Triple(demo.s_id, demo.p_id, demo.o_id),
                    predicate: NarsAtom::Verb("INHIBITS"),
                    object: NarsAtom::SPPair(demo.s_id, demo.p_id),
                    truth: NarsTruth {
                        // Frequency inverted: demotion = negative evidence
                        frequency: 1.0 - traj.spo_stability_ratio(),
                        confidence: 0.5, // moderate — we had it, then lost it
                    },
                    evidence: CausalEvidence::Demotion {
                        from_type: HaloType::Core,
                        to_type: HaloType::SP,
                        ewm_dissolution: count_dissolving_dims(
                            &traj.ewm_corrections, iter
                        ),
                        bprelu_direction: arrow.o_direction.clone(),
                    },
                    halo_type: HaloType::SP,
                    iter_range: (demo.first_seen_iter, iter),
                });
            },
            
            // ... enumerate SPO→SO, SPO→PO, SP→S, SP→P, etc.
            
            _ => {}
        }
    }
}
```

### 3.2 Truth Value Computation from Trajectory Metrics

```rust
/// Compute NARS truth values from the trajectory instrumentation
fn trajectory_truth_values(
    traj: &CausalTrajectory,
    statement: &NarsCausalStatement,
) -> NarsTruth {
    let evidence = &statement.evidence;
    
    match evidence {
        CausalEvidence::Promotion { ewm_support, bprelu_direction, .. } => {
            // FREQUENCY from EWM: proportion of dimensions that crystallized
            // More crystallization = more positive evidence = higher frequency
            let f_ewm = ewm_support / (D as f32);
            
            // FREQUENCY from BPReLU: causal direction strength
            // Strong forward causation = high frequency
            // Strong backward = also high (just different direction)
            // Symmetric = uncertain = frequency near 0.5
            let f_bprelu = match bprelu_direction {
                CausalDirection::Forward(strength) => 0.5 + strength * 0.5,
                CausalDirection::Backward(strength) => 0.5 + strength * 0.5,
                CausalDirection::Symmetric => 0.5,
                CausalDirection::Contested(_) => 0.3,
            };
            
            // Combined frequency: weighted geometric mean
            let f = (f_ewm.powf(0.6) * f_bprelu.powf(0.4)).clamp(0.01, 0.99);
            
            // CONFIDENCE from convergence speed
            let iter_fraction = (statement.iter_range.1 - statement.iter_range.0) as f32 
                / MAX_ITER as f32;
            // Fast convergence → high confidence
            // Slow convergence → low confidence (hard problem, uncertain answer)
            let c_speed = 1.0 - iter_fraction;
            
            // CONFIDENCE from RIF chain consistency
            // If the causal chain is consistent (same direction across diffs),
            // confidence is high. If it flip-flops, confidence is low.
            let c_chain = compute_chain_consistency(
                &traj.rif_diffs, 
                statement.iter_range,
            );
            
            // Combined confidence
            let c = (c_speed * 0.5 + c_chain * 0.5).clamp(0.01, 0.99);
            
            NarsTruth { frequency: f, confidence: c }
        },
        
        CausalEvidence::Demotion { ewm_dissolution, bprelu_direction, .. } => {
            // For demotions, frequency is INVERTED
            let f = (1.0 - ewm_dissolution / (D as f32)).clamp(0.01, 0.99);
            let c = 0.5; // moderate — we observed the demotion but don't know if permanent
            NarsTruth { frequency: f, confidence: c }
        },
        
        CausalEvidence::Persistence { duration, contested_dims, .. } => {
            // Persistent halo entries: frequency from duration, 
            // confidence from how contested they are
            let f = (duration as f32 / MAX_ITER as f32).clamp(0.01, 0.99);
            let c = (1.0 - *contested_dims as f32 / D as f32).clamp(0.01, 0.99);
            NarsTruth { frequency: f, confidence: c }
        },
    }
}
```

---

## PART 4: DN Tree Growth from Causal Trajectories

### 4.1 Sigma Graph Edge Creation Rules

Each NARS causal statement maps to a typed edge in the Sigma Graph (Neo4j). The edge type comes from the NARS predicate, and the edge metadata encodes the halo type, causal direction, and trajectory evidence.

```rust
fn trajectory_to_sigma_edges(
    traj: &CausalTrajectory,
) -> Vec<SigmaEdge> {
    let mut edges = Vec::new();
    
    for stmt in &traj.nars_statements {
        let edge = match stmt.predicate {
            
            // ─── CAUSES: forward causal link ───
            NarsAtom::Verb("CAUSES") => {
                SigmaEdge {
                    edge_type: SigmaEdgeType::CAUSES,
                    from_node: stmt.subject.to_sigma_node(),
                    to_node: stmt.object.to_sigma_node(),
                    truth: stmt.truth.clone(),
                    metadata: SigmaEdgeMetadata {
                        halo_origin: stmt.halo_type,
                        causal_direction: match &stmt.evidence {
                            CausalEvidence::Promotion { bprelu_direction, .. } 
                                => bprelu_direction.clone(),
                            _ => CausalDirection::Symmetric,
                        },
                        trajectory_span: stmt.iter_range,
                        growth_path: infer_growth_path(stmt),
                    },
                }
            },
            
            // ─── ENABLES: abductive/backward causal link ───
            NarsAtom::Verb("ENABLES") => {
                SigmaEdge {
                    edge_type: SigmaEdgeType::SUPPORTS,  
                    // ENABLES maps to SUPPORTS in Sigma vocabulary
                    // because abductive inference = supporting evidence
                    from_node: stmt.subject.to_sigma_node(),
                    to_node: stmt.object.to_sigma_node(),
                    truth: NarsTruth {
                        frequency: stmt.truth.frequency,
                        // Lower confidence because abductive
                        confidence: stmt.truth.confidence * 0.7,
                    },
                    metadata: SigmaEdgeMetadata {
                        halo_origin: stmt.halo_type,
                        causal_direction: CausalDirection::Backward(
                            stmt.truth.frequency
                        ),
                        trajectory_span: stmt.iter_range,
                        growth_path: infer_growth_path(stmt),
                    },
                }
            },
            
            // ─── INHIBITS: negative causal link (demotion) ───
            NarsAtom::Verb("INHIBITS") => {
                SigmaEdge {
                    edge_type: SigmaEdgeType::CONTRADICTS,
                    from_node: stmt.subject.to_sigma_node(),
                    to_node: stmt.object.to_sigma_node(),
                    truth: stmt.truth.clone(),
                    metadata: SigmaEdgeMetadata {
                        halo_origin: stmt.halo_type,
                        causal_direction: match &stmt.evidence {
                            CausalEvidence::Demotion { bprelu_direction, .. }
                                => bprelu_direction.clone(),
                            _ => CausalDirection::Symmetric,
                        },
                        trajectory_span: stmt.iter_range,
                        growth_path: infer_growth_path(stmt),
                    },
                }
            },
            
            // ─── PRECEDES: weak temporal link (single→pair) ───
            NarsAtom::Verb("PRECEDES") => {
                SigmaEdge {
                    edge_type: SigmaEdgeType::BECOMES,
                    from_node: stmt.subject.to_sigma_node(),
                    to_node: stmt.object.to_sigma_node(),
                    truth: stmt.truth.clone(),
                    metadata: SigmaEdgeMetadata {
                        halo_origin: stmt.halo_type,
                        causal_direction: CausalDirection::Forward(
                            stmt.truth.frequency
                        ),
                        trajectory_span: stmt.iter_range,
                        growth_path: infer_growth_path(stmt),
                    },
                }
            },
            
            _ => continue,
        };
        
        edges.push(edge);
    }
    
    edges
}
```

### 4.2 Growth Path Detection

The causal chain (from RIF diffs) tells us which of the 6 growth paths the factorization took:

```rust
/// The 6 growth paths through the partial binding lattice
#[derive(Debug, Clone)]
enum GrowthPath {
    SubjectFirst,     // S → SP → SPO  (forward chaining)
    SubjectObject,    // S → SO → SPO  (entity linking)
    PredicateFirst,   // P → SP → SPO  (action grounding)
    PredicateObject,  // P → PO → SPO  (passive/effect-first)
    ObjectFirst,      // O → PO → SPO  (backward chaining)
    ObjectSubject,    // O → SO → SPO  (abduction)
    Direct,           // → SPO directly  (single-shot recognition)
    Contested,        // oscillated between paths (competing hypotheses)
}

fn infer_growth_path(stmt: &NarsCausalStatement) -> GrowthPath {
    // Look at the halo type history across the trajectory
    let chain = &stmt.evidence;
    
    match chain {
        CausalEvidence::Promotion { from_type, bprelu_direction, .. } => {
            match from_type {
                HaloType::SP => {
                    if bprelu_direction.is_forward() {
                        GrowthPath::SubjectFirst    // S committed first
                    } else {
                        GrowthPath::PredicateFirst  // P committed first
                    }
                },
                HaloType::SO => {
                    if bprelu_direction.is_forward() {
                        GrowthPath::SubjectObject
                    } else {
                        GrowthPath::ObjectSubject
                    }
                },
                HaloType::PO => {
                    if bprelu_direction.is_forward() {
                        GrowthPath::PredicateObject
                    } else {
                        GrowthPath::ObjectFirst
                    }
                },
                _ => GrowthPath::Direct,
            }
        },
        _ => GrowthPath::Direct,
    }
}
```

### 4.3 DN Mutation Operators Guided by Trajectory

When the DN tree needs to MUTATE (explore new hypotheses), the trajectory tells it WHERE and HOW to mutate:

```rust
/// Use trajectory evidence to select the best mutation operator
fn dn_mutate_from_trajectory(
    current_triple: &SpoTriple,
    traj: &CausalTrajectory,
    saliency: &CausalSaliency,
) -> SpoTriple {
    
    // 1. Find the WEAKEST slot (most contested dimensions)
    let s_weakness = count_contested_in_plane(&saliency.contested, Plane::S);
    let p_weakness = count_contested_in_plane(&saliency.contested, Plane::P);
    let o_weakness = count_contested_in_plane(&saliency.contested, Plane::O);
    
    // 2. The weakest slot gets mutated (highest fitness improvement potential)
    let mutation_target = if s_weakness > p_weakness && s_weakness > o_weakness {
        Plane::S
    } else if p_weakness > o_weakness {
        Plane::P
    } else {
        Plane::O
    };
    
    // 3. Use the HALO to select the mutation candidate
    //    The halo entries in the target plane are NEARBY alternatives
    let candidates = match mutation_target {
        Plane::S => &traj.latest_snapshot().halo.s,  // S-only halo entries
        Plane::P => &traj.latest_snapshot().halo.p,
        Plane::O => &traj.latest_snapshot().halo.o,
    };
    
    // 4. Pick the candidate with highest Hamming similarity to current
    //    (smallest mutation = conservative exploration)
    let best_candidate = candidates.iter()
        .min_by_key(|c| hamming_distance(&c.vector, &current_triple.get_plane(mutation_target)))
        .unwrap();
    
    // 5. Construct the mutant
    let mut mutant = current_triple.clone();
    mutant.set_plane(mutation_target, best_candidate.vector.clone());
    
    mutant
}

/// For RADICAL mutation (Catalyst mode): use double-slot from trajectory
fn dn_radical_mutate(
    current_triple: &SpoTriple,
    traj: &CausalTrajectory,
) -> SpoTriple {
    // Use the partial pair halo entries as the mutation source
    // These are ALREADY partially coherent (2 planes agree)
    // so they're more likely to be viable than random
    
    let sp_candidates = &traj.latest_snapshot().halo.sp;
    let so_candidates = &traj.latest_snapshot().halo.so;
    let po_candidates = &traj.latest_snapshot().halo.po;
    
    // Pick the pair type with the most candidates (richest exploration space)
    let (pair_type, candidates) = [
        (HaloType::SP, sp_candidates),
        (HaloType::SO, so_candidates),
        (HaloType::PO, po_candidates),
    ].into_iter()
     .max_by_key(|(_, c)| c.count_ones())
     .unwrap();
    
    // The pair provides 2 planes; the third is random (maximally exploratory)
    let mut mutant = current_triple.clone();
    match pair_type {
        HaloType::SP => {
            mutant.s = candidates.best_s().clone();
            mutant.p = candidates.best_p().clone();
            // O stays current or randomized
        },
        HaloType::SO => {
            mutant.s = candidates.best_s().clone();
            mutant.o = candidates.best_o().clone();
        },
        HaloType::PO => {
            mutant.p = candidates.best_p().clone();
            mutant.o = candidates.best_o().clone();
        },
        _ => unreachable!(),
    }
    
    mutant
}
```

---

## PART 5: Cross-Session Persistence via Redis

### 5.1 Trajectory Compression for Redis Storage

```rust
/// Compress a trajectory for Redis persistence
/// Key: ada:trajectory:{input_hash}:{timestamp}
fn compress_trajectory(traj: &CausalTrajectory) -> Vec<u8> {
    // Don't store raw snapshots (too large: 48KB × 20 iters = 960KB)
    // Store the DERIVED signals only:
    
    let compressed = TrajectoryCompressed {
        // 1. Input hash (for retrieval)
        input_hash: hash_cogrecord(&traj.input),
        
        // 2. Final factorization (the answer)
        final_s: traj.snapshots.last().unwrap().s_est.clone(),
        final_p: traj.snapshots.last().unwrap().p_est.clone(),
        final_o: traj.snapshots.last().unwrap().o_est.clone(),
        
        // 3. Convergence metadata
        num_iters: traj.snapshots.len() as u16,
        convergence_speed: compute_convergence_speed(traj),
        
        // 4. Causal chain summary (sparse: only the links, not raw diffs)
        causal_chain: summarize_chain(&traj.rif_diffs),
        
        // 5. Growth path
        growth_path: infer_growth_path_from_chain(&traj.rif_diffs),
        
        // 6. NARS statements (the actual output — compact)
        nars_statements: traj.nars_statements.clone(),
        
        // 7. Saliency snapshot (which dimensions were contested)
        contested_mask: compute_final_saliency(traj).contested,
        
        // 8. Typed halo at final iteration (for warm-start next time)
        final_halo: traj.snapshots.last().unwrap().halo.clone(),
    };
    
    // Estimated size: ~5-15 KB per trajectory (vs 960KB raw)
    bincode::serialize(&compressed).unwrap()
}

/// Redis key patterns:
/// ada:trajectory:{hash}      → latest trajectory for this input
/// ada:trajectory:chain:{ts}  → time-ordered trajectory log
/// ada:causal:{s_id}:{p_id}   → all causal statements involving this S-P pair
/// ada:causal:{p_id}:{o_id}   → all causal statements involving this P-O pair
/// ada:growth:path:{type}     → count of each growth path type (statistics)
/// ada:contested:{dim_range}  → dimensions that are frequently contested
```

### 5.2 Warm-Start from Persisted Trajectory

When the same or similar input arrives again, use the persisted trajectory to warm-start the resonator:

```rust
fn warm_start_resonator(
    input: &CogRecord3D,
    redis: &RedisClient,
) -> Option<WarmStart> {
    // 1. Hash the input
    let hash = hash_cogrecord(input);
    
    // 2. Look up previous trajectory
    let prev: Option<TrajectoryCompressed> = redis.get(
        &format!("ada:trajectory:{}", hash)
    )?;
    
    // 3. If found, use the final halo as initialization
    if let Some(prev) = prev {
        // The final halo tells us which planes were already confirmed
        let warm = WarmStart {
            // Pre-fill confirmed planes from previous factorization
            s_init: if prev.final_halo.core.count_ones() > 0 || 
                       prev.final_halo.sp.count_ones() > 0 ||
                       prev.final_halo.so.count_ones() > 0 {
                Some(prev.final_s)  // S was confirmed last time
            } else {
                None  // S was uncertain, start fresh
            },
            p_init: if prev.final_halo.core.count_ones() > 0 || 
                       prev.final_halo.sp.count_ones() > 0 ||
                       prev.final_halo.po.count_ones() > 0 {
                Some(prev.final_p)
            } else {
                None
            },
            o_init: if prev.final_halo.core.count_ones() > 0 || 
                       prev.final_halo.so.count_ones() > 0 ||
                       prev.final_halo.po.count_ones() > 0 {
                Some(prev.final_o)
            } else {
                None
            },
            // Expected convergence speed (for timeout tuning)
            expected_iters: prev.num_iters,
            // Previously contested dimensions (monitor closely)
            watch_dims: prev.contested_mask,
        };
        
        return Some(warm);
    }
    
    None
}
```

---

## PART 6: The Unified Pipeline

### 6.1 End-to-End Flow

```
INPUT: Bundled 3D wave field (48 KB CogRecord3D)
  │
  ├─→ [1] Warm-start check (Redis lookup)
  │     If found: pre-fill confirmed planes from previous trajectory
  │     If not: standard random initialization
  │
  ├─→ [2] Resonator loop (5-20 iterations)
  │     Each iteration:
  │       a. Standard: unbind → project → rebind
  │       b. Instrument: record snapshot
  │       c. RIF diff: XOR with iter-2 (causal diff)
  │       d. EWM: classify tiers, compute corrections
  │       e. BPReLU: forward/backward asymmetry
  │       f. Halo: extract typed bitmasks via cross-plane vote
  │       g. Transitions: detect promotions/demotions
  │       h. NARS: generate causal statements from transitions
  │
  ├─→ [3] Post-convergence analysis
  │       a. Analyze full causal chain (RIF diffs stacked)
  │       b. Determine growth path (which of 6 paths was taken)
  │       c. Compute final truth values from full trajectory
  │       d. Generate Sigma Graph edges
  │
  ├─→ [4] Persist
  │       a. Compress trajectory → Redis (ada:trajectory:*)
  │       b. Write NARS statements → Redis (ada:causal:*)
  │       c. Write Sigma edges → Neo4j (CAUSES/SUPPORTS/CONTRADICTS/BECOMES)
  │       d. Update growth path statistics → Redis (ada:growth:*)
  │
  └─→ OUTPUT:
        - Clean SPO factorization (foveal)
        - Typed halo (context, for next resonation or downstream)
        - NARS causal statements with truth values
        - Sigma Graph edges with causal metadata
        - Growth path classification
        - Contested dimension mask (for DN mutation guidance)
```

### 6.2 Cost Analysis

```
Per resonator iteration (on top of existing cost):
  
  RIF diff:           3 × XOR(16KB)           = 3 × 0.7μs  = 2.1μs
  EWM classification: 3 × compare(16KB)       = 3 × 1.0μs  = 3.0μs
  BPReLU:             2 × project(16KB)        = 2 × 3.0μs  = 6.0μs
  Cross-plane vote:   7 × AND/NOT(bitmask)     = 7 × 0.1μs  = 0.7μs
  Halo transitions:   set intersection         =              1.0μs
  NARS generation:    per-transition logic      =              2.0μs
                                                ─────────────────────
  Total overhead per iteration:                              ~15μs
  
  For 10 iterations:                                         ~150μs
  
  Existing resonator cost per iteration:                     ~50μs
  New total per iteration:                                   ~65μs
  Overhead:                                                  ~30%

This is acceptable. The 30% overhead buys:
  - Full causal trajectory recording
  - NARS truth values grounded in convergence dynamics
  - Typed Sigma Graph edges with causal direction
  - DN mutation guidance
  - Warm-start capability for repeat inputs
  
Memory per trajectory:
  Raw: 48KB × 20 iters = 960KB (too large)
  Compressed: ~5-15KB (store only derived signals)
  Redis budget: 1M trajectories × 10KB = 10GB (fits in Redis)
```

---

## PART 7: Research Questions for Validation

### 7.1 Empirical Questions

1. **Growth path distribution:** Run the pipeline on real Jina embeddings (1024D → 16384D signed quinary). What is the empirical distribution of the 6 growth paths? Hypothesis: SubjectFirst dominates for SVO-like input, PredicateFirst dominates for event-structured input.

2. **EWM tier correlation with convergence:** Do dimensions classified as Crystallized at iteration 3 remain Crystallized through convergence? What is the false-crystallization rate?

3. **BPReLU asymmetry stability:** Does the causal direction (forward/backward) for a given SPO triple remain stable across multiple presentations of similar inputs? If yes → real causal structure. If no → artifact of initialization.

4. **RIF chain consistency:** Do the RIF causal diffs tell a consistent story (same plane stabilizes first across runs)? Or is stabilization order stochastic? What is the variance?

5. **NARS truth value calibration:** Are the truth values well-calibrated? (Does a frequency of 0.7 mean 70% of the time the statement is correct when checked against ground truth?)

### 7.2 Theoretical Questions

6. **Completeness of causal inference:** The 3 BNN components (EWM, BPReLU, RIF) map to 3 aspects of causation (saliency, direction, chain). Is this a complete decomposition? What about confounding, selection bias, and mediation? Can those be extracted from the trajectory?

7. **Berge acyclicity preservation:** Do the generated Sigma Graph edges preserve Berge acyclicity within each awareness window? If cycles form, is that always from CONTRADICTS edges (expected) or from CAUSES edges (problematic)?

8. **DN error threshold from trajectory:** Can the convergence speed and contested dimension count predict whether a DN mutation will be viable (below error threshold) or fatal (above)?

9. **Information-theoretic cost:** How many bits of causal information does the trajectory contain vs how many bits of foveal factorization? Is the causal trajectory more or less compressible than the factorization itself?

10. **Connection to Granger causality:** The BPReLU forward/backward asymmetry detects temporal precedence plus interventional asymmetry. Is this equivalent to, stronger than, or weaker than Granger causality for the SPO setting?

---

## PART 8: Key References

- **Zhang et al. 2025** — RIF-BNN: EWM + BPReLU + Rich Information Flow for Binary Neural Networks
- **Frady et al. 2020** — Resonator Networks for compositional factorization (arXiv:2007.03748)
- **Czégel et al. 2021** — Darwinian Neurodynamics: particle filtering + evolutionary search
- **Fernando et al. 2012** — DN original: replicators, heredity, selection in neural networks
- **Wang 2006** — NARS: Non-Axiomatic Reasoning System (truth values, revision, inference rules)
- **Pearl 2009** — Causality: do-calculus, interventional probability, causal graphs
- **Li et al. 2023** — Nature Nanotechnology: memristive resonator factorization (hardware noise helps)
- **arXiv:2404.19126 (2024)** — Compositional visual scene factorization with deflation
- **Kleyko et al. 2021** — VSA Survey Parts I & II (arXiv:2111.06077, 2112.15424)
- **arXiv:2405.20583 (2024)** — Gestalt Computational Model (persistent homology)

---

## PART 9: Key Claim

The noise floor of a foveated resonator (σ₂–σ₃ band) contains structured causal information that can be extracted using three BNN instruments:

1. **EWM** extracts WHERE causation acts (saliency)
2. **BPReLU** extracts WHICH DIRECTION causation flows (arrow of time)
3. **RIF shortcuts** extract WHAT SEQUENCE causation follows (causal chain)

Together with the **6 typed halo entries** (S, P, O, SP, SO, PO) from the cross-plane vote, this produces a complete causal trajectory that:

- Generates typed NARS statements with well-grounded truth values
- Grows the DN tree (Sigma Graph) with causal edges typed by growth path
- Guides DN mutation operators toward the weakest slots
- Enables warm-start resonation via trajectory persistence
- Creates a causal inference engine (forward/backward/abductive/analogical) from the resonator's convergence dynamics alone

The total computational overhead is ~30% per resonator iteration (~15μs), using only operations (XOR, compare, project) that are already available in the AVX-512 pipeline. The causal trajectory is a byproduct of instrumentation, not an additional computation.

**The subconscious of the resonator IS the causal structure of the world it models.**
