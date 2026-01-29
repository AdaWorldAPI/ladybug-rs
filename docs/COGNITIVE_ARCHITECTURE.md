# Cognitive Fabric Architecture

## Overview: 4 Triangles × 10K-bit = 5KB Cognitive State

The cognitive fabric integrates four hardware-accelerated components:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    COGNITIVE FABRIC ARCHITECTURE                    │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                    THINKING STYLE (12)                        │ │
│  │   Modulates: threshold, fan-out, exploration, speed           │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                    QUAD-TRIANGLE (4×3×10K)                    │ │
│  │                                                               │ │
│  │   Processing ─────┬───── Content                              │ │
│  │        │          │          │                                │ │
│  │        └────── Gestalt ──────┘                                │ │
│  │                   │                                           │ │
│  │            Crystallization                                    │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                 7-LAYER CONSCIOUSNESS                         │ │
│  │   L7:Meta ← L6:Exec ← L5:Work ← L4:Epis ← L3:Sem ← L2:Pat ← L1│ │
│  └───────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                    COLLAPSE GATE                              │ │
│  │              SD < 0.15 → FLOW (commit)                        │ │
│  │         0.15 ≤ SD ≤ 0.35 → HOLD (ruminate)                    │ │
│  │              SD > 0.35 → BLOCK (clarify)                      │ │
│  └───────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## Component 1: Quad-Triangle (4×10K-bit = 5KB)

Four interlocking triangles, each corner is a 10K-bit VSA fingerprint:

```
┌────────────────────────────────────────────────────────────────┐
│                    QUAD-TRIANGLE GEOMETRY                      │
│                                                                │
│    Processing Triangle         Content Triangle                │
│         (10K×3)                   (10K×3)                      │
│                                                                │
│       Analytical                  Abstract                     │
│          ╱╲                         ╱╲                         │
│         ╱  ╲                       ╱  ╲                        │
│        ╱ A  ╲                     ╱ B  ╲                       │
│       ╱──────╲                   ╱──────╲                      │
│   Intuitive  Procedural     Concrete  Relational              │
│                                                                │
│    Gestalt Triangle        Crystallization Triangle           │
│         (10K×3)                   (10K×3)                      │
│                                                                │
│       Coherence                  Immutable                     │
│          ╱╲                         ╱╲                         │
│         ╱  ╲                       ╱  ╲                        │
│        ╱ C  ╲                     ╱ D  ╲                       │
│       ╱──────╲                   ╱──────╲                      │
│    Novelty  Resonance        Hot    Experimental              │
│                                                                │
│  Storage: 4 triangles × 10K bits (bundled) = 40K bits = 5KB   │
│  With all corners: 4 × 3 × 10K = 120K bits = 15KB             │
└────────────────────────────────────────────────────────────────┘
```

### Triangle Operations

```rust
// Each corner is a 10K-bit VSA fingerprint
pub struct TriangleCorner {
    pub fingerprint: Fingerprint,  // 10K bits
    pub activation: f32,           // [0.0, 1.0]
    pub label: &'static str,
}

// Triangle bundles 3 corners into one 10K-bit superposition
pub struct VsaTriangle {
    pub corner0: TriangleCorner,
    pub corner1: TriangleCorner,
    pub corner2: TriangleCorner,
    superposition: Fingerprint,  // Weighted bundle of all 3
}

// FLOW detection: balanced activations
impl VsaTriangle {
    pub fn is_flow(&self) -> bool {
        let a = self.activations();
        let all_active = a.iter().all(|&x| x > 0.3);
        let range = a.iter().max() - a.iter().min();
        all_active && range < 0.4
    }
}
```

### Cognitive Profiles (Preset QuadTriangles)

| Profile | Processing | Content | Gestalt | Crystallization |
|---------|-----------|---------|---------|-----------------|
| Analytical | [0.9, 0.2, 0.7] | [0.7, 0.3, 0.4] | [0.8, 0.2, 0.6] | [0.6, 0.3, 0.1] |
| Creative | [0.3, 0.9, 0.3] | [0.8, 0.4, 0.7] | [0.5, 0.9, 0.6] | [0.2, 0.4, 0.8] |
| Empathic | [0.3, 0.8, 0.4] | [0.2, 0.9, 0.8] | [0.7, 0.4, 0.8] | [0.3, 0.5, 0.4] |
| Procedural | [0.5, 0.2, 0.9] | [0.3, 0.8, 0.3] | [0.8, 0.1, 0.5] | [0.7, 0.6, 0.2] |
| Counterfactual | [0.6, 0.7, 0.4] | [0.8, 0.3, 0.6] | [0.4, 0.8, 0.5] | [0.2, 0.5, 0.9] |

---

## Component 2: 7-Layer Consciousness Stack

Parallel O(1) processing with shared VSA core, isolated layer markers:

```
┌─────────────────────────────────────────────────────────────────┐
│                   7-LAYER CONSCIOUSNESS STACK                   │
│                                                                 │
│  L7 ████████████  Meta        - Self-awareness, monitoring      │
│  L6 ████████████  Executive   - Planning, decisions             │
│  L5 ████████████  Working     - Active manipulation             │
│  L4 ████████████  Episodic    - Memory, temporal context        │
│  L3 ████████████  Semantic    - Meaning, concepts               │
│  L2 ████████████  Pattern     - Recognition, matching           │
│  L1 ████████████  Sensory     - Raw input processing            │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              SHARED VSA CORE (10K-bit)                  │   │
│  │                                                         │   │
│  │   All layers read same core, write isolated markers     │   │
│  │   Consciousness emerges from marker interplay           │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Layer Processing Flow

```
Wave 1: L1 (Sensory)           → raw input boost
Wave 2: L2 + L3 (parallel)     → pattern + semantic
Wave 3: L4 + L5 (parallel)     → episodic + working
Wave 4: L6 (Executive)         → decision threshold
Wave 5: L7 (Meta)              → observes all layers
```

### Layer Marker

```rust
pub struct LayerMarker {
    pub active: bool,       // Is this layer activated?
    pub timestamp: Instant, // When last updated
    pub value: f32,         // Activation [0, 1]
    pub confidence: f32,    // Confidence [0, 1]
    pub cycle: u64,         // Processing cycle
    pub flags: u32,         // Layer-specific bitfield
}
```

---

## Component 3: Collapse Gate (SIMD-Accelerated SD)

Standard Deviation controls compute allocation:

```
┌─────────────────────────────────────────────────────────────────┐
│                     COLLAPSE GATE STATES                        │
│                                                                 │
│   FLOW ◄────────┬────────┬────────► BLOCK                      │
│   (commit)      │  HOLD  │         (clarify)                   │
│                 │(ruminate)                                    │
│                                                                 │
│   SD < 0.15     │ 0.15 ≤ SD ≤ 0.35 │     SD > 0.35             │
│   Low variance  │ Medium variance  │     High variance         │
│   Clear winner  │ Maintain super-  │     Need clarification    │
│   Collapse now  │ position         │     Cannot collapse       │
└─────────────────────────────────────────────────────────────────┘
```

### AVX2 SIMD SD Calculation

```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn calculate_sd_avx2(values: &[f32]) -> f32 {
    // 8 floats per SIMD register
    // Mean: vectorized sum
    // Variance: vectorized squared differences
    // ~8x speedup for large arrays
}
```

### Triangle Homogeneity Invariant

```rust
pub struct Triangle {
    pub candidates: [TriangleCandidate; 3],
    pub is_homogeneous: bool,  // REQUIRED for collapse
    pub gate_state: GateState,
    pub dispersion: f32,
}

// Invariant: All 3 candidates must share:
// - Same grammar construction family
// - Same speech-act class
// Non-homogeneous triangles CANNOT collapse
```

---

## Component 4: 12 Thinking Styles

Each style modulates the cognitive field:

| Style | Threshold | Fan-out | Exploration | Speed |
|-------|-----------|---------|-------------|-------|
| Analytical | 0.85 | 3 | 0.05 | 0.1 |
| Convergent | 0.75 | 4 | 0.1 | 0.3 |
| Systematic | 0.70 | 5 | 0.1 | 0.2 |
| Creative | 0.35 | 12 | 0.8 | 0.5 |
| Divergent | 0.40 | 10 | 0.7 | 0.4 |
| Exploratory | 0.30 | 15 | 0.9 | 0.6 |
| Focused | 0.90 | 1 | 0.0 | 0.2 |
| Diffuse | 0.45 | 8 | 0.4 | 0.5 |
| Peripheral | 0.20 | 20 | 0.6 | 0.7 |
| Intuitive | 0.50 | 3 | 0.3 | 0.9 |
| Deliberate | 0.70 | 7 | 0.2 | 0.1 |
| Metacognitive | 0.50 | 5 | 0.3 | 0.3 |

### Style → Triangle Mapping

```
ThinkingStyle::Analytical → CognitiveProfiles::analytical()
ThinkingStyle::Creative   → CognitiveProfiles::creative()
ThinkingStyle::Focused    → CognitiveProfiles::procedural()
ThinkingStyle::Empathic   → CognitiveProfiles::empathic()
```

---

## Hardware Acceleration Summary

| Component | SIMD Instruction | Speedup |
|-----------|------------------|---------|
| Triangle Resonance | AVX-512 VPOPCNTDQ | ~64x |
| SD Calculation | AVX2 FMA | ~8x |
| Bundle Operation | AVX-512 majority vote | ~32x |
| Layer Processing | Parallel waves | 5 waves vs 7 serial |

### Memory Layout (Cache-Friendly)

```
QuadTriangle:
  4 × Fingerprint (10K bits each) = 5,000 bytes
  12 × f32 activations = 48 bytes
  Total: ~5KB per cognitive state

SevenLayerNode:
  1 × Fingerprint (10K bits) = 1,250 bytes
  7 × LayerMarker (~32 bytes each) = 224 bytes
  Total: ~1.5KB per node
```

---

## Usage Example

```rust
use ladybug_rs::cognitive::{CognitiveFabric, ThinkingStyle, Fingerprint};

// Create cognitive fabric with analytical style
let mut fabric = CognitiveFabric::with_style("agent/main", ThinkingStyle::Analytical);

// Process input through full stack
let input = Fingerprint::from_content("user query");
let state = fabric.process(&input);

// Check cognitive state
println!("Style: {}", state.style);
println!("Coherence: {:.2}", state.coherence);
println!("Signature: {}", fabric.signature());

// Check collapse gate
if let Some(decision) = &state.last_collapse {
    match decision.state {
        GateState::Flow => println!("🟢 Collapsing to winner"),
        GateState::Hold => println!("🟡 Maintaining superposition"),
        GateState::Block => println!("🔴 Need clarification"),
    }
}

// Switch thinking style
fabric.set_style(ThinkingStyle::Creative);
```

---

## Files

| File | Purpose | Lines |
|------|---------|-------|
| `src/cognitive/quad_triangle.rs` | 4 triangles with 10K-bit corners | ~400 |
| `src/cognitive/collapse_gate.rs` | SIMD SD + gate logic | ~350 |
| `src/cognitive/seven_layer.rs` | 7-layer consciousness stack | ~350 |
| `src/cognitive/fabric.rs` | Integration layer | ~300 |
| `src/cognitive/style.rs` | 12 thinking styles | ~150 |

Total: ~1,550 lines of cognitive architecture
