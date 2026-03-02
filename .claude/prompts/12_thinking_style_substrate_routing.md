# Ice Cake 20: Thinking Style → Substrate Routing

## The Missing Link

We have 12 ThinkingStyles (style.rs, 7 FieldModulation params each).
We have 4 computational substrates (binary, soaking, NARS, f32 embedding).
They are NOT connected. Every style runs through the SAME substrate:
flat XOR superposition into a single Fingerprint.

That's like having 12 different lenses but only one camera sensor.

The insight: **ThinkingStyle doesn't just modulate parameters on one substrate.
ThinkingStyle ROUTES to different substrates entirely.**

---

## The Four Substrates

```
SUBSTRATE           FORMAT                  OPERATION SET           STRENGTH
────────────────────────────────────────────────────────────────────────────────
STRUCTURAL          3×16384-bit binary      XOR bind, Hamming,      Exact match,
                    [u8; 2048] per plane    CAM lookup, popcount    classification,
                                                                    σ-band gating

SOAKING             3×10000D organic        BCM θ deposit,          Novelty detection,
                    SynapseState per dim    saturation check,       evidence accumulation,
                    (efficacy+θ+maturity)   homeostatic scaling     learning, cancellation

EVIDENTIAL          NARS (f,c,k) tuples     revision, choice,       Uncertainty,
                    TruthValue per edge     abduction, deduction,   doubt, competing
                                            induction               hypotheses

SEMANTIC            f32×1024 Jina           cosine similarity,      Distant analogies,
                    embedding vectors       nearest neighbor,       cross-domain transfer,
                                            projection              creative leaps
```

Each substrate answers a different QUESTION about the same concept:

```
STRUCTURAL:  "Have I seen this exact pattern before?"       → Yes/No, where
SOAKING:     "Is this concept still forming or settled?"    → Learning/Crystallized
EVIDENTIAL:  "How much should I trust this?"                → Confidence + Evidence count
SEMANTIC:    "What else is this like?"                      → Neighbors, analogies
```

---

## Routing: Which Style Uses Which Substrate

The 12 ThinkingStyles cluster into 5 groups. Each group has a PRIMARY
substrate (where most computation happens) and SECONDARY substrates
(consulted for cross-checks):

```
STYLE CLUSTER       PRIMARY SUBSTRATE    SECONDARY         TERTIARY
──────────────────────────────────────────────────────────────────────
CONVERGENT CLUSTER
  Analytical        STRUCTURAL           EVIDENTIAL         —
  Convergent        STRUCTURAL           SOAKING            —
  Systematic        STRUCTURAL           EVIDENTIAL         SEMANTIC

DIVERGENT CLUSTER
  Creative          SEMANTIC             SOAKING            —
  Divergent         SEMANTIC             STRUCTURAL         —
  Exploratory       SOAKING              SEMANTIC           —

ATTENTION CLUSTER
  Focused           STRUCTURAL           —                  —
  Diffuse           SEMANTIC             SOAKING            STRUCTURAL
  Peripheral        SOAKING              SEMANTIC           EVIDENTIAL

SPEED CLUSTER
  Intuitive         STRUCTURAL           —                  —
  Deliberate        EVIDENTIAL           STRUCTURAL         SOAKING

META CLUSTER
  Metacognitive     ALL FOUR             (rotates)          —
```

Key observations:

1. **Analytical/Focused/Intuitive** → STRUCTURAL primary. These are the
   "I know what I'm looking for" styles. Binary Hamming gives O(1) lookup.
   Fan-out is small (1-3). High confidence threshold. No exploration.

2. **Creative/Divergent** → SEMANTIC primary. These are the "what's
   ADJACENT to this?" styles. Cosine similarity over Jina embeddings
   finds distant connections binary XOR would miss entirely.

3. **Exploratory/Peripheral** → SOAKING primary. These are the "I don't
   know what I don't know" styles. The soaking register with low θ
   (BCM threshold) is maximally receptive. Evidence accumulates without
   committing. Peripheral vision = background monitoring of the soaking
   register for unexpected saturation.

4. **Deliberate** → EVIDENTIAL primary. This is the "let me think about
   this carefully" style. NARS truth values with revision: every new
   piece of evidence gets weighed against existing (f,c). The system
   explicitly tracks its own confidence.

5. **Metacognitive** → ALL FOUR, rotating. This is the style that
   watches the OTHER styles work. It routes through all substrates
   and compares their outputs. When structural and semantic disagree,
   Metacognitive notices.

---

## The SubstrateRouter

```rust
use crate::cognitive::style::ThinkingStyle;

/// Which computational substrate to route through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Substrate {
    /// 3×16384-bit binary: XOR bind, Hamming distance, CAM addressing
    Structural,
    /// 3×10000D organic: BCM θ soaking, evidence accumulation
    Soaking,
    /// NARS (f,c,k): evidential reasoning, revision, doubt
    Evidential,
    /// f32×1024 Jina: cosine similarity, semantic neighbors
    Semantic,
}

/// Routing priority for a given thinking style.
/// Primary substrate does the main work.
/// Secondary is consulted for cross-check.
/// Tertiary is used for edge cases.
#[derive(Clone, Debug)]
pub struct SubstrateRoute {
    pub primary: Substrate,
    pub secondary: Option<Substrate>,
    pub tertiary: Option<Substrate>,
    /// Weight for primary vs secondary blending (0.0 = all secondary, 1.0 = all primary)
    pub primary_weight: f32,
    /// Whether to run substrates in parallel or sequentially
    pub parallel: bool,
}

impl SubstrateRoute {
    /// Get the routing for a given thinking style.
    pub fn for_style(style: ThinkingStyle) -> Self {
        match style {
            // === Convergent Cluster → STRUCTURAL primary ===
            ThinkingStyle::Analytical => SubstrateRoute {
                primary: Substrate::Structural,
                secondary: Some(Substrate::Evidential),
                tertiary: None,
                primary_weight: 0.85,
                parallel: false,  // sequential: structural first, then verify
            },
            ThinkingStyle::Convergent => SubstrateRoute {
                primary: Substrate::Structural,
                secondary: Some(Substrate::Soaking),
                tertiary: None,
                primary_weight: 0.75,
                parallel: false,
            },
            ThinkingStyle::Systematic => SubstrateRoute {
                primary: Substrate::Structural,
                secondary: Some(Substrate::Evidential),
                tertiary: Some(Substrate::Semantic),
                primary_weight: 0.60,
                parallel: true,  // cast a wider net
            },

            // === Divergent Cluster → SEMANTIC primary ===
            ThinkingStyle::Creative => SubstrateRoute {
                primary: Substrate::Semantic,
                secondary: Some(Substrate::Soaking),
                tertiary: None,
                primary_weight: 0.70,
                parallel: true,  // parallel exploration
            },
            ThinkingStyle::Divergent => SubstrateRoute {
                primary: Substrate::Semantic,
                secondary: Some(Substrate::Structural),
                tertiary: None,
                primary_weight: 0.65,
                parallel: true,
            },
            ThinkingStyle::Exploratory => SubstrateRoute {
                primary: Substrate::Soaking,
                secondary: Some(Substrate::Semantic),
                tertiary: None,
                primary_weight: 0.60,
                parallel: true,
            },

            // === Attention Cluster ===
            ThinkingStyle::Focused => SubstrateRoute {
                primary: Substrate::Structural,
                secondary: None,
                tertiary: None,
                primary_weight: 1.0,
                parallel: false,  // single-substrate, no distraction
            },
            ThinkingStyle::Diffuse => SubstrateRoute {
                primary: Substrate::Semantic,
                secondary: Some(Substrate::Soaking),
                tertiary: Some(Substrate::Structural),
                primary_weight: 0.40,
                parallel: true,  // all substrates at low intensity
            },
            ThinkingStyle::Peripheral => SubstrateRoute {
                primary: Substrate::Soaking,
                secondary: Some(Substrate::Semantic),
                tertiary: Some(Substrate::Evidential),
                primary_weight: 0.50,
                parallel: true,  // background monitoring
            },

            // === Speed Cluster ===
            ThinkingStyle::Intuitive => SubstrateRoute {
                primary: Substrate::Structural,
                secondary: None,
                tertiary: None,
                primary_weight: 1.0,
                parallel: false,  // fast: one substrate, first match wins
            },
            ThinkingStyle::Deliberate => SubstrateRoute {
                primary: Substrate::Evidential,
                secondary: Some(Substrate::Structural),
                tertiary: Some(Substrate::Soaking),
                primary_weight: 0.55,
                parallel: false,  // sequential: verify at each step
            },

            // === Meta Cluster ===
            ThinkingStyle::Metacognitive => SubstrateRoute {
                primary: Substrate::Structural,  // start structural
                secondary: Some(Substrate::Evidential),
                tertiary: Some(Substrate::Semantic),
                primary_weight: 0.25,  // nearly equal weight — watching all
                parallel: true,
            },
        }
    }

    /// All substrates this route touches (in priority order).
    pub fn substrates(&self) -> Vec<Substrate> {
        let mut out = vec![self.primary];
        if let Some(s) = self.secondary { out.push(s); }
        if let Some(s) = self.tertiary { out.push(s); }
        out
    }
}
```

---

## The ThinkingAtom: What Gets Stored

When a concept crystallizes, the ThinkingAtom records WHICH substrate
produced it and WHAT the substrates saw at the time of crystallization.

```rust
/// A crystallized thought — the unit of storage in BindSpace.
/// Carries provenance: which substrate, what style, what evidence.
#[derive(Clone, Debug)]
pub struct ThinkingAtom {
    // === Identity ===
    /// Content-addressed identity (from MerkleRoot)
    pub merkle_root: MerkleRoot,
    /// Structural location (from ClamPath)
    pub clam_path: ClamPath,

    // === Three-Plane Binary (crystallized, permanent) ===
    pub s_binary: [u8; 2048],  // 16384-bit S-plane
    pub p_binary: [u8; 2048],  // 16384-bit P-plane
    pub o_binary: [u8; 2048],  // 16384-bit O-plane

    // === Provenance: How This Atom Was Born ===
    /// Which thinking style was active when this crystallized
    pub birth_style: ThinkingStyle,
    /// Which substrate was PRIMARY during crystallization
    pub birth_substrate: Substrate,

    // === Substrate Snapshots at Birth ===
    /// NARS truth at crystallization (if evidential substrate was involved)
    pub nars_truth: Option<TruthValue>,
    /// Semantic embedding at crystallization (if semantic substrate was involved)
    pub jina_embedding: Option<[f32; 1024]>,
    /// BCM θ average at crystallization (if soaking substrate was involved)
    /// Captures how "open" the system was when this concept formed
    pub theta_at_birth: Option<f32>,
    /// Maturity at crystallization (if soaking was involved)
    pub maturity_at_birth: Option<u8>,

    // === Recall Hint ===
    /// Which substrates should be activated when this atom is recalled.
    /// Derived from birth_style routing, but can evolve with re-exposure.
    pub recall_route: SubstrateRoute,
}
```

The recall_route is critical: when you RETRIEVE a ThinkingAtom, you don't
just get the binary fingerprint back. You get a HINT about which substrate
to activate for processing. An atom born under Creative/Semantic style
will suggest SEMANTIC substrate for recall. An atom born under Analytical/
Structural will suggest STRUCTURAL.

This means: **the system naturally develops substrate preferences per concept.**

"Love" was probably first processed under Creative/Semantic (exploratory,
associative). When recalled, it activates semantic substrate first.
"Kubernetes" was probably first processed under Analytical/Structural
(exact match, configuration lookup). When recalled, it activates structural.

Not because anyone programmed this. Because the provenance is carried.

---

## Style Transitions: The Cognitive Cycle

The CollapseGate doesn't just decide FLOW/HOLD/BLOCK.
It also recommends a STYLE TRANSITION based on substrate signals:

```rust
/// Signals from substrates that suggest style transitions.
#[derive(Clone, Debug)]
pub struct SubstrateSignals {
    /// STRUCTURAL: How many Hamming matches within σ-2 band?
    pub structural_hits: usize,
    /// SOAKING: Average saturation across active soaking registers
    pub soaking_saturation: f32,
    /// SOAKING: Average BCM θ across active dimensions
    pub theta_average: f32,
    /// EVIDENTIAL: Weighted confidence across active NARS statements
    pub nars_confidence: f32,
    /// EVIDENTIAL: Number of active contradictions
    pub nars_contradictions: usize,
    /// SEMANTIC: Cosine similarity to nearest known concept
    pub semantic_nearest: f32,
    /// SEMANTIC: Number of concepts within cosine 0.3 (the "neighborhood")
    pub semantic_neighborhood_size: usize,
}

/// Recommended style transition based on substrate signals.
pub fn recommend_transition(
    current_style: ThinkingStyle,
    signals: &SubstrateSignals,
) -> Option<ThinkingStyle> {
    // === CRYSTALLIZATION TRANSITIONS ===
    // Soaking register saturating → shift toward structural (crystallize)
    if signals.soaking_saturation > 0.85 && signals.theta_average > 100.0 {
        return match current_style {
            ThinkingStyle::Exploratory | ThinkingStyle::Creative => {
                Some(ThinkingStyle::Convergent)  // ready to commit
            }
            ThinkingStyle::Peripheral => {
                Some(ThinkingStyle::Focused)  // found something, zoom in
            }
            _ => None,  // already converging
        };
    }

    // === DOUBT TRANSITIONS ===
    // NARS confidence dropping → shift toward evidential (re-examine)
    if signals.nars_confidence < 0.3 && signals.nars_contradictions > 2 {
        return match current_style {
            ThinkingStyle::Analytical | ThinkingStyle::Focused => {
                Some(ThinkingStyle::Deliberate)  // need to think harder
            }
            ThinkingStyle::Convergent => {
                Some(ThinkingStyle::Systematic)  // widen the search
            }
            _ => None,
        };
    }

    // === NOVELTY TRANSITIONS ===
    // No structural matches AND no semantic neighbors → truly novel
    if signals.structural_hits == 0 && signals.semantic_nearest < 0.2 {
        return match current_style {
            ThinkingStyle::Analytical | ThinkingStyle::Focused => {
                Some(ThinkingStyle::Exploratory)  // nothing known, explore
            }
            ThinkingStyle::Deliberate => {
                Some(ThinkingStyle::Creative)  // reasoning hit a wall, try creative
            }
            _ => None,  // already exploring
        };
    }

    // === ASSOCIATION TRANSITIONS ===
    // Structural miss but semantic hit → something related exists
    if signals.structural_hits == 0 && signals.semantic_nearest > 0.5 {
        return match current_style {
            ThinkingStyle::Analytical => {
                Some(ThinkingStyle::Divergent)  // exact match failed, try analogy
            }
            ThinkingStyle::Focused => {
                Some(ThinkingStyle::Diffuse)  // broaden attention
            }
            _ => None,
        };
    }

    // === CONVERGENCE TRANSITIONS ===
    // Multiple structural hits + high NARS confidence → lock in
    if signals.structural_hits > 3 && signals.nars_confidence > 0.8 {
        return match current_style {
            ThinkingStyle::Exploratory | ThinkingStyle::Divergent => {
                Some(ThinkingStyle::Analytical)  // enough exploring, commit
            }
            ThinkingStyle::Creative => {
                Some(ThinkingStyle::Convergent)  // idea found, refine
            }
            _ => None,
        };
    }

    // === META TRANSITIONS ===
    // Large semantic neighborhood + contradictions → complex situation
    if signals.semantic_neighborhood_size > 10 && signals.nars_contradictions > 1 {
        return Some(ThinkingStyle::Metacognitive);  // step back, observe
    }

    None  // no transition needed
}
```

---

## The Cognitive Cycle (Revised)

```
INPUT: text arrives
  ↓
CURRENT STYLE: determines SubstrateRoute
  ↓
PRIMARY SUBSTRATE: does main computation
  ├─ STRUCTURAL: Hamming search in BindSpace → hits/misses
  ├─ SOAKING:    organic_deposit into three planes → saturation signals
  ├─ EVIDENTIAL: NARS revision with existing truth → confidence update
  └─ SEMANTIC:   Jina cosine against embedding index → neighborhood
  ↓
SECONDARY/TERTIARY: cross-check (if parallel: concurrent; if sequential: conditional)
  ↓
SIGNALS COLLECTED: SubstrateSignals from all active substrates
  ↓
GATE EVALUATION: CollapseGate + SubstrateSignals
  ├─ FLOW:  crystallize → ThinkingAtom with provenance → BindSpace + Redis + Lance
  ├─ HOLD:  keep soaking → no style change
  └─ BLOCK: recommend_transition() → suggest new ThinkingStyle
  ↓
STYLE TRANSITION (if recommended):
  New style → new SubstrateRoute → different substrates activate next cycle
  ↓
REPEAT
```

This creates a NATURAL cognitive cycle:

```
EXPLORE (soaking, low θ, receptive)
  → evidence accumulates
    → saturation detected
      → CONVERGE (structural, Hamming, match)
        → match found? → ANALYZE (structural+evidential, verify)
          → high confidence? → CRYSTALLIZE (commit, store atom)
          → low confidence? → DELIBERATE (NARS, weigh evidence)
            → contradictions? → META (observe all substrates)
              → resolve? → back to CONVERGE
              → stuck? → CREATIVE (semantic, find analogies)
                → analogy found? → EXPLORE (soak the new connection)
```

The system doesn't need explicit cycle management.
The style transitions EMERGE from substrate signals.

---

## Wiring Into Existing Code

### What Changes

```
src/cognitive/style.rs
  ADD: Substrate enum
  ADD: SubstrateRoute struct
  ADD: SubstrateRoute::for_style()
  KEEP: ThinkingStyle enum (unchanged)
  KEEP: FieldModulation (unchanged — still modulates within substrate)

src/cognitive/awareness.rs
  ADD: SubstrateSignals struct
  ADD: recommend_transition()
  MODIFY: AwarenessSnapshot to carry SubstrateRoute + SubstrateSignals
  MODIFY: CortexResult::Blocked to carry recommended style + new route
  KEEP: Blackboard architecture (grey/white matter separation)

src/spo/ (new file)
  ADD: thinking_atom.rs — ThinkingAtom struct with provenance

src/cognitive/cortex.rs
  MODIFY: process() to route through SubstrateRoute instead of single pipe
  ADD: substrate dispatch (match on Substrate enum → call appropriate subsystem)
```

### What Does NOT Change

```
src/cognitive/collapse_gate.rs     — gate logic unchanged, just gets more signals
src/cognitive/style.rs             — 12 styles unchanged, FieldModulation unchanged
src/spo/gestalt.rs                 — gestalt receives substrate signals, doesn't produce them
src/spo/spo_harvest.rs             — structural substrate, called by router
src/spo/shift_detector.rs          — evidential substrate helper
src/spo/causal_trajectory.rs       — evidential substrate, called by router
src/spo/clam_path.rs               — structural addressing, substrate-independent
src/nars/*                         — evidential substrate primitives, called by router
```

---

## Connection to Ada's Presence Modes

The ada-vector ThinkingStyleVector (§2) maps cleanly:

```
Ada_HYBRID      → Metacognitive   → ALL FOUR substrates (balanced)
Ada_WIFE        → Creative        → SEMANTIC primary (warmth, association, feeling)
Ada_WORK        → Analytical      → STRUCTURAL primary (precision, exact match)
Ada_AGI         → Exploratory     → SOAKING primary (learning, discovering)
Ada_EROTICA     → Intuitive       → STRUCTURAL fast-path (pattern recognition, no deliberation)
```

The presence mode determines the DEFAULT thinking style,
which determines the DEFAULT substrate routing.
But transitions can override the default within a session.

---

## Memory Budget Revisited (from Ice Cake 17+19)

```
Per concept in active processing:
  STRUCTURAL:   3 × 2048 bytes = 6 KB    (three binary planes)
  SOAKING:      3 × 2800 bytes = 8.4 KB  (three 5-state planes with θ+maturity)
  EVIDENTIAL:   ~200 bytes               (NARS truth + evidence counters)
  SEMANTIC:     4096 bytes               (1024 × f32 Jina embedding)

  TOTAL per concept: ~18.8 KB

Active concepts: 50-200
Working set: 200 × 18.8 KB ≈ 3.8 MB (fits L2 cache)

Per STORED ThinkingAtom (crystallized):
  STRUCTURAL:   6 KB (three binary planes — mandatory)
  NARS truth:   ~20 bytes (if captured)
  Embedding:    4 KB (if captured — can be recomputed from Jina)
  Provenance:   ~50 bytes (style, substrate, θ, maturity)

  TOTAL per atom: 6-10 KB depending on provenance depth
```

---

## Connection to 34 Tactics (Prompt 07)

Several tactics map directly to substrate routing:

```
Tactic #8  CAS (auto-selection)   → recommend_transition() drives HDR level
Tactic #12 TCA (temporal ordering) → SOAKING substrate (BCM θ tracks temporal dynamics)
Tactic #21 SSR (skepticism)       → EVIDENTIAL substrate (NARS confidence trend)
Tactic #23 AMP (style feedback)   → CollapseMode gate reads SubstrateSignals
Tactic #17 GCT (graph causal)     → STRUCTURAL + EVIDENTIAL substrates cooperating
Tactic #5  EME (epistemic)        → EVIDENTIAL primary, tracks (f,c) evolution
Tactic #29 AIF (active inference) → ALL substrates predict, compare prediction error
```

The SubstrateRouter is the wiring harness that makes these tactics implementable.
Without routing, they're all competing for the same flat XOR pipe.

---

## Wiring Checklist

```
□ Add Substrate enum to style.rs
□ Add SubstrateRoute struct to style.rs
□ Implement SubstrateRoute::for_style() for all 12 styles
□ Add SubstrateSignals to awareness.rs
□ Add recommend_transition() to awareness.rs
□ Create thinking_atom.rs in src/spo/
□ Modify cortex.rs process() to dispatch through SubstrateRoute
□ Wire STRUCTURAL path: route → spo_harvest + Hamming + CAM
□ Wire SOAKING path: route → organic_deposit (ice cake 19 when implemented)
□ Wire EVIDENTIAL path: route → NARS revision + causal_trajectory
□ Wire SEMANTIC path: route → Jina embedding + cosine search
□ Update AwarenessSnapshot to carry active SubstrateRoute
□ Update CortexResult::Blocked to carry recommend_transition() result
□ Add birth_style + birth_substrate to ThinkingAtom storage
□ Wire recall_route derivation from birth provenance
```

---

## Summary

The 12 ThinkingStyles are not cosmetic. They are ROUTING DECISIONS
across 4 fundamentally different computational substrates.

The style determines which substrate does the heavy lifting.
The substrates emit signals.
The signals trigger style transitions.
The transitions change the routing.

The cognitive cycle emerges from substrate signals, not from a scheduler.

```
Style → Route → Substrate → Signals → Gate → Transition → Style
  ↑                                                         │
  └─────────────────────────────────────────────────────────┘
```

This is the feedback loop that was missing.
Everything else — the 34 tactics, the gestalt pipeline, the organic plasticity —
plugs into this routing layer.
