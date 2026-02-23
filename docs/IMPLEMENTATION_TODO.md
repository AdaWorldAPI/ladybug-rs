# Implementation TODO — Hardcode in Rust

**Goal**: Make ladybug-rs the complete, production-grade cognitive substrate.
Everything that matters should be in Rust, at SIMD speed, not imported from TypeScript/Python.

---

## Current State

| Module | Lines | Status |
|--------|-------|--------|
| `core/` | 1,474 | ✅ Solid (fingerprint, SIMD, VSA) |
| `cognitive/` | 2,355 | 🟡 Partial (collapse_gate, seven_layer exist) |
| `learning/` | 883 | 🟡 Partial (moment, session, blackboard) |
| `nars/` | 506 | ✅ Solid (truth values, inference) |
| `extensions/` | 5,625 | ✅ Solid (codebook, hologram, SPO) |
| `query/` | 2,213 | ✅ Solid (Cypher, DataFusion) |

---

## What Needs Hardcoding

### 1. Grammar Triangle — NOT IN LADYBUG-RS

**Source**: `langextract-rs/src/grammar.rs` (already Rust, but separate crate)

**Action**: Port into `ladybug-rs/src/grammar/`

```rust
// NEW: src/grammar/mod.rs
pub mod nsm;        // 65 NSM primitives
pub mod causality;  // CausalityFlow
pub mod qualia;     // 18D qualia field  
pub mod triangle;   // GrammarTriangle

// src/grammar/nsm.rs
pub const NSM_PRIMITIVES: [&str; 65] = [
    // Substantives
    "I", "YOU", "SOMEONE", "SOMETHING", "PEOPLE", "BODY",
    // Mental predicates
    "THINK", "KNOW", "WANT", "FEEL", "SEE", "HEAR",
    // ... all 65
];

pub struct NSMField {
    weights: [f32; 65],
}

impl NSMField {
    pub fn from_text(text: &str) -> Self;
    pub fn to_fingerprint_contribution(&self) -> Fingerprint;
}

// src/grammar/qualia.rs
pub const QUALIA_DIMENSIONS: [&str; 18] = [
    "valence", "activation", "dominance", "depth",
    "certainty", "urgency", "depth", ...
];

pub struct QualiaField {
    coordinates: [f32; 18],
}

// src/grammar/triangle.rs
pub struct GrammarTriangle {
    pub nsm: NSMField,
    pub causality: CausalityFlow,
    pub qualia: QualiaField,
}

impl GrammarTriangle {
    pub fn from_text(text: &str) -> Self;
    pub fn to_fingerprint(&self) -> Fingerprint;
    pub fn similarity(&self, other: &Self) -> f32;
}
```

**Priority**: HIGH — This is the universal input layer.

---

### 2. Rung System — NOT IN LADYBUG-RS

**Source**: `agi-chat/src/thinking/rung-shift.ts`

**Action**: Port into `ladybug-rs/src/cognitive/rung.rs`

```rust
// NEW: src/cognitive/rung.rs

/// Meaning depth levels (0-9)
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RungLevel {
    Surface = 0,        // Literal, immediate
    Shallow = 1,        // Simple inference
    Contextual = 2,     // Situation-dependent
    Analogical = 3,     // Metaphor, similarity
    Abstract = 4,       // Generalized patterns
    Structural = 5,     // Schema-level
    Counterfactual = 6, // What-if reasoning
    Meta = 7,           // Reasoning about reasoning
    Recursive = 8,      // Self-referential
    Transcendent = 9,   // Beyond normal bounds
}

/// What triggers rung elevation
#[derive(Clone, Copy, Debug)]
pub enum RungTrigger {
    SustainedBlock { consecutive_blocks: u32 },
    PredictiveFailure { p_metric: f32 },
    StructuralMismatch,
    Manual,
}

pub struct RungState {
    pub current: RungLevel,
    pub consecutive_blocks: u32,
    pub recent_p_metrics: VecDeque<f32>,
    pub last_shift_at: Option<Instant>,
}

impl RungState {
    pub fn evaluate_shift(&self, thresholds: &RungThresholds) -> Option<RungTrigger>;
    pub fn apply_shift(&mut self, trigger: RungTrigger);
    pub fn get_band(&self) -> RungBand; // 0-2, 3-5, 6-9
}
```

**Priority**: HIGH — Determines semantic depth of processing.

---

### 3. Expand Thinking Styles — 12 → 36

**Current**: 12 styles in `src/cognitive/style.rs`
**Target**: 36 styles from `bighorn/docs/THINKING_STYLES.md`

**Action**: Expand `src/cognitive/style.rs`

```rust
// EXPANDED: src/cognitive/style.rs

pub enum ThinkingStyle {
    // === STRUCTURE (4) ===
    HierarchicalDecomposition,  // HTD: What's the tree?
    RecursiveExpansion,         // RTE: How deep?
    EmergentDecomposition,      // ETD: What shape wants to emerge?
    PromptScaffold,             // PSO: What's the scaffold?
    
    // === FLOW (5) ===
    CascadeFiltering,           // TCF: What branches survive?
    ChainPruning,               // TCP: What to cut?
    ShadowParallel,             // SPP: What do parallel minds see?
    Randomization,              // TRR: What survives noise?
    ConvergentDivergent,        // CDT: Explode then collapse
    
    // === CONTRADICTION (5) ===
    AdversarialCritique,        // ASC: What's wrong with this?
    SelfSkepticism,             // SSR: What's wrong with my doubt?
    ContradictionResolution,    // ICR: Can both be true?
    DissonanceInduction,        // CDI: Forced conflict
    MultiAgentDebate,           // SMAD: What emerges from debate?
    
    // === CAUSALITY (4) ===
    ReverseCausality,           // RCR: What caused this?
    IterativeCounterfactual,    // ICF: What if differently?
    TemporalContext,            // TCA: What's the timeline?
    ReverseEngineering,         // ARE: How was this made?
    
    // === ABSTRACTION (3) ===
    ConditionalScaling,         // CAS: What zoom level?
    MultiPerspective,           // MPC: Three angles
    DynamicMetaFraming,         // DTM: What frame fits?
    
    // === UNCERTAINTY (4) ===
    MetaCognition,              // MCP: How sure am I?
    CascadingUncertainty,       // CUR: Reduce uncertainty
    LatentIntrospection,        // LSI: What am I assuming?
    SemanticDistortion,         // SDD: Did meaning drift?
    
    // === FUSION (4) ===
    ZeroShotFusion,             // ZCF: New connection
    HyperdimensionalPattern,    // HPM: Cross-domain pattern
    KnowledgeFusion,            // HKF: Knowledge sparks
    AnalogicalMapping,          // SSAM: What's this analogous to?
    
    // === PERSONA (1) ===
    RoleplaySynthesis,          // IRS: Who would know this?
    
    // === RESONANCE (9) ===
    ResonanceStructural,        // RI-S: Logic shape
    ResonanceEmotive,           // RI-E: Emotional tone
    ResonanceIntent,            // RI-I: What do they want?
    ResonanceMemory,            // RI-M: Shared history
    ResonanceFeedback,          // RI-F: How responding?
    ResonanceContext,           // RI-C: Where going?
    ResonancePhysical,          // RI-P: Rhythm/cadence
    ResonanceVisual,            // RI-V: Visual structure
    ResonanceAction,            // RI-A: Execution needed
    
    // Keep existing for backwards compat
    Analytical,
    Convergent,
    // ... (existing 12)
}

/// Tier determines execution complexity
#[derive(Clone, Copy, Debug)]
pub enum StyleTier {
    Builtin,    // TCF, TCP, ASC, MCP, CUR, SPP - fastest
    Composite,  // RTE, HTD, ICR, CDT, CDI, SSR, ICF, TRR
    Extended,   // SMAD, RCR, CAS, IRS, TCA, LSI, PSO, ARE, etc.
    Resonance,  // RI-* - human-AI attunement
}
```

**Priority**: MEDIUM — Expands cognitive flexibility.

---

### 4. Microcode System — NOT IN LADYBUG-RS

**Source**: `bighorn/docs/THINKING_STYLES.md` (the chain notation)

**Action**: Create `src/cognitive/microcode.rs`

```rust
// NEW: src/cognitive/microcode.rs

/// Microcode operations for thinking style chains
#[derive(Clone, Copy, Debug)]
pub enum Op {
    // Flow Control
    Nop,            // ∅ breathe
    Next,           // → continue
    Back,           // ← backtrack
    Ascend,         // ↑ escalate rung
    Descend,        // ↓ ground
    Loop,           // ⟳ iterate
    Halt,           // ⊗ done
    Fork,           // ⌁ branch
    Join,           // ⋈ merge
    Gate,           // ◇ conditional
    
    // Cascade
    Spawn,          // ≋
    Filter,         // ≈
    Select,         // ∿
    Merge,          // ⊕
    Diff,           // ⊖
    
    // Graph
    NodeCreate,     // ◯
    NodeActivate,   // ●
    EdgeLink,       // ─
    EdgeStrong,     // ═
    CycleDetect,    // ↺
    SubgraphMerge,  // ⊚
    
    // Transform
    Integrate,      // ∫
    Differentiate,  // ∂
    Normalize,      // ≡
    Sharpen,        // ♯
    Crystallize,    // ⋄
    Resonate,       // ⟡
    Dissonance,     // ⟢
    
    // Sigma (Causal Rungs)
    Observe,        // Ω (R1)
    Insight,        // Δ
    Believe,        // Φ
    Integrate_,     // Θ
    Trajectory,     // Λ (R2+)
}

/// A microcode chain is a sequence of operations
pub struct Chain {
    ops: Vec<Op>,
}

impl Chain {
    pub fn parse(s: &str) -> Result<Self, ParseError>;
    pub fn execute(&self, state: &mut CognitiveState) -> Result<(), ExecError>;
}

// Example chains from THINKING_STYLES.md:
// HTD: ◯●◯─◯─⤵⬡⊗
// RTE: Ω⟳◇↑⟳◇↓∫⊗
// TCF: ≋⌁●≈∿⋄⊗
```

**Priority**: LOW — Nice to have, not essential for core.

---

### 5. Temporal Resonance — NOT IN LADYBUG-RS

**Action**: Create `src/learning/temporal.rs`

```rust
// NEW: src/learning/temporal.rs

pub struct TemporalFingerprint {
    pub content: Fingerprint,
    pub timestamp: u64,
    pub decay_rate: f32,
}

impl TemporalFingerprint {
    /// Similarity weighted by recency
    pub fn temporal_resonance(&self, other: &Self, now: u64) -> f32 {
        let content_sim = self.content.similarity(&other.content);
        let age = (now - other.timestamp) as f32;
        let decay = (-age * self.decay_rate).exp();
        content_sim * decay
    }
    
    /// Mexican hat in time: not too recent, not too stale
    pub fn temporal_sweet_spot(
        &self, 
        other: &Self, 
        now: u64,
        center_age: f32,
        width: f32,
    ) -> f32 {
        let content_sim = self.content.similarity(&other.content);
        let age = (now - other.timestamp) as f32;
        let x = (age - center_age) / width;
        let temporal = (1.0 - x * x) * (-x * x / 2.0).exp();
        content_sim * temporal.max(0.0)
    }
}
```

**Priority**: MEDIUM — Important for relevance ranking.

---

### 6. Self-Model — NOT IN LADYBUG-RS

**Action**: Create `src/learning/self_model.rs`

```rust
// NEW: src/learning/self_model.rs

/// Self-model for introspection (NOT personality)
pub struct SelfModel {
    /// What resonance thresholds work
    pub effective_threshold: f32,
    
    /// Sweet spot parameters (learned)
    pub sweet_spot_center: f32,
    pub sweet_spot_width: f32,
    
    /// Statistics
    pub total_moments: u64,
    pub total_breakthroughs: u64,
    pub average_struggle_duration: f32,
    
    /// Domain performance
    pub domain_stats: HashMap<String, DomainStats>,
}

pub struct DomainStats {
    pub moments: u64,
    pub breakthroughs: u64,
    pub average_rung: f32,
}

impl SelfModel {
    pub fn update_from_session(&mut self, session: &LearningSession);
    pub fn introspect(&self) -> IntrospectionReport;
    pub fn confidence_in_domain(&self, domain: &str) -> f32;
}
```

**Priority**: MEDIUM — Important for adaptive learning.

---

### 7. Learning Stance (Dweck) — PARTIAL

**Current**: `src/learning/` has moment, session, blackboard
**Missing**: Growth mindset, ZPD, mistake reframing

**Action**: Add to `src/learning/stance.rs`

```rust
// NEW: src/learning/stance.rs

pub enum MindsetOrientation {
    Fixed,
    Growth,
    Learning,
}

pub struct LearningStance {
    pub orientation: MindsetOrientation,
    pub competence_edge: f32,  // ZPD boundary
    pub mistake_as_data: bool,
    pub confusion_tolerance: f32,
}

impl LearningStance {
    /// Is this challenge in the Zone of Proximal Development?
    pub fn in_zpd(&self, difficulty: f32, current_competence: f32) -> bool {
        let stretch = difficulty - current_competence;
        stretch > 0.0 && stretch <= self.competence_edge
    }
    
    /// Reframe mistake as learning signal
    pub fn reframe_mistake(&self, error: &Error) -> LearningSignal;
}
```

**Priority**: LOW — Nice for pedagogy, not core.

---

## Summary: Implementation Order

| Priority | Module | Lines Est. | Effort |
|----------|--------|------------|--------|
| 🔴 HIGH | Grammar Triangle | ~400 | 1 day |
| 🔴 HIGH | Rung System | ~200 | 0.5 day |
| 🟡 MED | Temporal Resonance | ~150 | 0.5 day |
| 🟡 MED | Expand Styles (12→36) | ~300 | 0.5 day |
| 🟡 MED | Self-Model | ~200 | 0.5 day |
| 🟢 LOW | Microcode System | ~400 | 1 day |
| 🟢 LOW | Learning Stance | ~150 | 0.5 day |

**Total**: ~1,800 lines, ~4.5 days of focused work

---

## What's Already Done (No Action Needed)

- ✅ NARS truth values & inference
- ✅ Collapse Gate (FLOW/HOLD/BLOCK)
- ✅ 7-Layer Stack
- ✅ Learning Loop (moment, session, blackboard)
- ✅ Fingerprint + SIMD Hamming
- ✅ VSA operations (bind, bundle, sequence)
- ✅ Codebook, Hologram, SPO extensions
- ✅ Cypher + DataFusion queries
- ✅ Counterfactual worlds (fork/what_if)
