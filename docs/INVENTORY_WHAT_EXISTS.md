# What Already Exists — Repository Inventory

**Date**: 2026-01-30
**Purpose**: Audit what's already built before adding to ladybug-rs UNIVERSAL_ARCHITECTURE.md

---

## Summary

Most of what was listed as "missing" in UNIVERSAL_ARCHITECTURE.md **already exists** — just scattered across repositories.

| Component | Status | Location |
|-----------|--------|----------|
| Grammar Triangle | ✅ EXISTS | `langextract-rs/src/grammar.rs`, `ada-unified/extensions/langextract/core/grammar_triangle.py`, `agi-chat/src/grammar/` |
| NSM Primitives | ✅ EXISTS | 65 primes in `grammar_triangle.py` |
| Qualia Field | ✅ EXISTS | 18D in `grammar_triangle.py`, `langextract-rs/src/grammar.rs` |
| Thinking Styles | ✅ EXISTS | 36 styles in `bighorn/docs/THINKING_STYLES.md`, `ada-unified/dto/thinking_style.py` |
| Rung System | ✅ EXISTS | 0-9 levels in `agi-chat/src/thinking/rung-shift.ts` |
| Collapse Gate | ✅ EXISTS | FLOW/HOLD/BLOCK in `agi-chat/src/thinking/collapse-gate.ts`, `ladybug-rs/src/cognitive/collapse_gate.rs` |
| 7-Layer Stack | ✅ EXISTS | `agi-chat/src/vsa/seven-layer.ts` |
| NARS Truth Values | ✅ EXISTS | `ladybug-rs/src/nars/truth.rs` |
| NARS Inference | ✅ EXISTS | `ladybug-rs/src/nars/inference.rs` |
| Resonance Parser | ✅ EXISTS | `agi-chat/src/grammar/resonance-parser.ts` |
| SPO Crystal | ✅ EXISTS | `ladybug-rs/src/extensions/spo/spo.rs` |
| Holographic Memory | ✅ EXISTS | `ladybug-rs/src/extensions/hologram/` |
| Codebook | ✅ EXISTS | `ladybug-rs/src/extensions/codebook/` |
| Learning Stance | ✅ EXISTS | `ada-consciousness/learning/learning_stance.py` |

---

## Detailed Inventory

### 1. Grammar Layer

**langextract-rs/src/grammar.rs** (Rust)
```rust
pub struct GrammarTriangle {
    pub nsm: NSMField,
    pub causality: CausalityFlow,
    pub qualia: QualiaField,
}

impl GrammarTriangle {
    pub fn from_text(text: &str) -> Self;
    pub fn to_fingerprint(&self) -> [u64; 157];
    pub fn similarity(&self, other: &Self) -> f32;
}
```

**ada-unified/extensions/langextract/core/grammar_triangle.py** (Python)
```python
class GrammarTriangleField:
    """
    Three-way field: NSM + Causality + ICC (Qualia)
    Creates continuous field where meaning flows without collapsing.
    """
```

**agi-chat/src/grammar/** (TypeScript)
- `triangle-types.ts` — GrammarRole, FrameType, FeatureBundle
- `resonance-parser.ts` — Full parser implementation
- `grammar-awareness.ts` — Awareness engine

### 2. NSM (Natural Semantic Metalanguage)

**65 Primitives** in `grammar_triangle.py`:
```python
NSM_PRIMITIVES = [
    # Substantives
    "I", "YOU", "SOMEONE", "SOMETHING", "PEOPLE", "BODY",
    # Mental predicates
    "THINK", "KNOW", "WANT", "FEEL", "SEE", "HEAR",
    # Speech
    "SAY", "WORDS", "TRUE",
    # Actions/Events
    "DO", "HAPPEN", "MOVE", "TOUCH",
    # Time
    "WHEN", "NOW", "BEFORE", "AFTER",
    # ... 65 total
]
```

### 3. Qualia Field

**18 Dimensions** in `grammar_triangle.py`:
```python
QUALIA_DIMENSIONS = [
    "valence",        # Positive/negative feeling
    "arousal",        # Activation level
    "dominance",      # Control/agency
    "intimacy",       # Closeness
    "certainty",      # Epistemic confidence
    "urgency",        # Temporal pressure
    "depth",          # Abstraction level
    # ... 18 total
]
```

### 4. Thinking Styles

**36 Styles** in `bighorn/docs/THINKING_STYLES.md`:

| Category | Styles |
|----------|--------|
| STRUCTURE | HTD, RTE, ETD, PSO |
| FLOW | TCF, TCP, SPP, TRR, CDT |
| CONTRADICTION | ASC, SSR, ICR, CDI, SMAD |
| CAUSALITY | RCR, ICF, TCA, ARE |
| ABSTRACTION | CAS, MPC, DTM |
| UNCERTAINTY | MCP, CUR, LSI, SDD |
| FUSION | ZCF, HPM, HKF, SSAM |
| PERSONA | IRS |
| RESONANCE | RI-S, RI-E, RI-I, RI-M, RI-F, RI-C, RI-P, RI-V, RI-A |

**ThinkingStyleDTO** in `ada-unified/dto/thinking_style.py`:
```python
@dataclass
class ThinkingStyleDTO:
    """33D Cognitive Fingerprint mapped to 10kD[256:320]."""
    
    @property
    def pearl(self) -> Tuple[float, float, float]:  # SEE, DO, IMAGINE
    
    @property
    def rung(self) -> Tuple[float, ...]:  # R1-R9 profile
    
    @property
    def sigma(self) -> Tuple[float, ...]:  # Ω, Δ, Φ, Θ, Λ
```

### 5. Rung System

**agi-chat/src/thinking/rung-shift.ts**:
```typescript
type RungLevel = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9;

const RUNG_SEMANTICS: Record<RungLevel, string> = {
  0: 'Surface — literal, immediate meaning',
  1: 'Shallow — simple inference',
  2: 'Contextual — situation-dependent',
  3: 'Analogical — metaphor, similarity',
  4: 'Abstract — generalized patterns',
  5: 'Structural — schema-level',
  6: 'Counterfactual — what-if reasoning',
  7: 'Meta — reasoning about reasoning',
  8: 'Recursive — self-referential',
  9: 'Transcendent — beyond normal bounds',
};

// Triggers for elevation:
// - sustained_block (BLOCK persists N turns)
// - predictive_failure (P metric drops)
// - structural_mismatch (no legal parse)
```

### 6. Collapse Gate

**Both TypeScript and Rust implementations**:

```typescript
// agi-chat/src/thinking/collapse-gate.ts
type GateState = 'FLOW' | 'HOLD' | 'BLOCK';

const SD_FLOW_THRESHOLD = 0.15;   // Low dispersion → collapse
const SD_BLOCK_THRESHOLD = 0.35; // High dispersion → block
```

```rust
// ladybug-rs/src/cognitive/collapse_gate.rs
pub enum GateState { Flow, Hold, Block }

pub const SD_FLOW_THRESHOLD: f32 = 0.30 * SD_MAX;  // ~0.15
pub const SD_BLOCK_THRESHOLD: f32 = 0.70 * SD_MAX; // ~0.35
```

### 7. 7-Layer Consciousness Stack

**agi-chat/src/vsa/seven-layer.ts**:
```typescript
type LayerId = 'L1' | 'L2' | 'L3' | 'L4' | 'L5' | 'L6' | 'L7';

const LAYER_NAMES = {
  L1: 'sensory',    // Raw input processing
  L2: 'pattern',    // Recognition and matching
  L3: 'semantic',   // Meaning and concepts
  L4: 'episodic',   // Memory and temporal context
  L5: 'working',    // Active manipulation
  L6: 'executive',  // Planning and decisions
  L7: 'meta'        // Self-awareness and monitoring
};
```

### 8. NARS (Non-Axiomatic Reasoning)

**ladybug-rs/src/nars/truth.rs**:
```rust
pub struct TruthValue {
    pub frequency: f32,   // Proportion of positive evidence
    pub confidence: f32,  // Reliability of frequency
}

impl TruthValue {
    pub fn revision(&self, other: &TruthValue) -> TruthValue;
    pub fn deduction(&self, other: &TruthValue) -> TruthValue;  // A→B, B→C ⊢ A→C
    pub fn induction(&self, other: &TruthValue) -> TruthValue;  // A→B, A→C ⊢ B→C
    pub fn abduction(&self, other: &TruthValue) -> TruthValue;  // A→B, C→B ⊢ A→C
    pub fn analogy(&self, other: &TruthValue) -> TruthValue;
}
```

### 9. Learning Stance (Dweck's Mindset)

**ada-consciousness/learning/learning_stance.py**:
```python
class MindsetOrientation(Enum):
    FIXED = "fixed"
    GROWTH = "growth"
    LEARNING = "learning"

class LearningStance:
    """
    Operationalizes Dweck's mindset theory.
    
    - CompetenceEdge: Vygotsky's Zone of Proximal Development
    - MistakeReframe: transforms mistakes from shame to data
    - ConfusionState: "confusion is the feeling of learning"
    - LearningJournal: tracks growth over time
    """
```

---

## What's Actually Missing

After this audit, here's what's **truly** missing for universal:

### 1. **Modality Adapters** (still missing)
No local CLIP, Whisper, tree-sitter integration exists. All repos rely on:
- Jina for text embeddings
- External APIs for other modalities

### 2. **Universal Projection Operators** (still missing)
No LSH projector that takes arbitrary embeddings → 10K Hamming.
Current flow: Jina 1024D → ad-hoc projection

### 3. **Temporal Resonance** (partial)
Timestamps exist, but no:
- Decay-weighted similarity
- Mexican hat in time domain
- Temporal sweet spot search

### 4. **Federation** (not started)
No CRDT, no multi-instance sync, no cross-agent resonance.

### 5. **Self-Model** (not started)
No introspection layer that models:
- What resonance thresholds work
- What domains cause struggle
- Statistics about own learning

### 6. **Unified Entry Point** (the real gap)
The components exist but aren't wired together:
- `langextract-rs` has Grammar Triangle
- `ladybug-rs` has NARS, Collapse Gate, Learning Loop
- `agi-chat` has Rung System, 7-Layer Stack
- `ada-unified` has Thinking Styles, ThinkingStyleDTO
- `bighorn` has the docs and Cypher

**No single crate ties them together.**

---

## Recommended Next Steps

1. **Create `ladybug-unified` crate** that imports from all others
2. **Build modality adapters** with local inference (candle CLIP, whisper-rs)
3. **Implement temporal resonance** with decay and Mexican hat
4. **Add self-model layer** for introspection
5. **Defer federation** until single-instance is complete

---

## Repository Map

```
AdaWorldAPI/
├── ladybug-rs          # Core: Hamming, NARS, Learning Loop, Extensions
├── langextract-rs      # Grammar Triangle (Rust)
├── ada-consciousness   # Learning Stance, RL, Theta
├── ada-unified         # Thinking Styles, Qualia, LanceGraph
├── agi-chat            # 7-Layer, Rung, Collapse Gate, Resonance Parser
├── bighorn             # Cypher, AGI Stack Extensions, Docs
└── (ada-docs)          # Documentation
```

**Total**: ~95% of cognitive primitives exist. Missing: modality adapters, temporal resonance, self-model, federation, and the glue that unifies them.
