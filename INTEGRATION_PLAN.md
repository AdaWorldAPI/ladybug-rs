# AdaWorld Unified Cognitive Integration Plan

## Status: IN PROGRESS (Phase 1 near-complete, Phase 2 started)

This document is the canonical integration plan for connecting ladybug-rs,
crewai-rust, n8n-rs, cubus, and rustynum into a unified cognitive architecture.
It survives context resets and serves as the task skeleton for agents.

---

## Architecture Overview

```
                        n8n-rs (meta-orchestrator)
                        ┌─────────────────────────────┐
                        │ ImpactGate (RBAC, 8 roles)  │
                        │ FreeWillPipeline (self-mod)  │
                        │ UnifiedExecution pipeline    │
                        │ A2A blackboard discovery     │
                        └────────────┬────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
           crewai-rust          ladybug-rs          openclaw-rs
           (orchestration)      (cognitive DB)      (multi-channel)
           ┌──────────────┐     ┌──────────────┐   ┌────────────┐
           │ 36 styles    │     │ 12 styles    │   │ channels   │
           │ personas     │◄───►│ Grammar △    │   │ assistant  │
           │ agents/crews │     │ QuadTriangle │   └────────────┘
           │ SemanticKrnl │     │ 10-Layer     │
           │ Blackboard   │     │ CollapseGate │
           └──────┬───────┘     │ CogKernel    │
                  │             │ BindSpace ◄──────── THE HUB
                  │             └──────┬───────┘
                  │                    │
                  └──────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │ ladybug-contract │
                    │ Container 16384b │
                    │ CogRecord 16K    │
                    │ CogPacket wire   │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │ cubus (BF16)     │
                    │ awareness.rs     │
                    │ ThinkingStyleMix │
                    │ EnrichedGate     │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │ rustynum-core    │
                    │ DeltaLayer       │
                    │ LayerStack       │
                    │ CollapseGate     │
                    │ BF16 Hamming     │
                    │ Superposition    │
                    │ AwarenessState   │
                    └──────────────────┘
```

## Three Operating Modes

### Mode 1: crewai-rust Standalone (Passive RAG)

ladybug-rs serves as a cognitive database. crewai-rust drives all orchestration.

```
crewai-rust                 ladybug-rs
═══════════                 ══════════
Agents + Crews ──lb.query──► CognitiveService(PassiveRag)
                              │
                              ├─ query_text() → resonance score
                              ├─ query_resonance() → similarity
                              └─ evaluate_gate() → FLOW/HOLD/BLOCK
                              (no state mutation)
```

**What exists today:**
- [x] CognitiveService with PassiveRag mode (service.rs)
- [x] ThinkingStyleBridge (36→12 mapping)
- [x] Grammar Triangle → fingerprint pipeline
- [x] LbStepHandler routing lb.query/resonate/process/gate/style/snapshot/mode/reset
- [x] LadybugSubsystem with lifecycle bridge + agent descriptors
- [ ] Registration in crewai-rust SubsystemRegistry
- [ ] Integration test with crewai-rust's Pipeline

### Mode 2: ladybug-rs as Brain

ladybug-rs drives the cognitive loop, imports crewai-rust personas as style layers.

```
ladybug-rs                  crewai-rust
══════════                  ═══════════
CognitiveService(Brain)     Persona registry
  │                            │
  ├─ process_text()            │
  │  Grammar △ → fingerprint   │
  │  QuadTriangle update       │
  │  10-layer processing       │
  │  CollapseGate              │
  │                            │
  ├─ style = Bridge.resolve(persona.style)
  │                            │
  └─ emit crew.* steps ──────►│ Agent execution
     for each persona layer    │ Task delegation
                               │ LLM calls
```

**What exists today:**
- [x] CognitiveService with Brain mode (service.rs)
- [x] Full process_text() pipeline
- [x] process_with_style() for per-step overrides
- [ ] Step emission to crewai-rust pipeline
- [ ] Persona → ThinkingStyle bidirectional sync
- [ ] CognitiveKernel ↔ CognitiveService unification

### Mode 3: n8n-rs Orchestrated

n8n-rs manages the execution pipeline with policies, RBAC, and A2A discovery.

```
n8n-rs
══════
UnifiedExecution
  │
  ├── crew.agent steps → crewai-rust
  ├── lb.resonate steps → ladybug-rs
  ├── lb.process steps → ladybug-rs (Brain)
  ├── oc.channel steps → openclaw-rs
  │
  ├── ImpactGate → RBAC check per step
  ├── FreeWillPipeline → self-modification boundary
  │
  └── Blackboard
       ├── CognitiveSnapshot entries (style, coherence, gate)
       ├── A2A registry (agent discovery by capability)
       └── Typed slots (zero-serde in one-binary mode)
```

**What exists today:**
- [x] CognitiveSnapshot for blackboard exchange (service.rs)
- [x] apply_snapshot() for external state injection
- [x] LadybugSubsystem with mode-aware lifecycle (subsystem_impl.rs)
- [x] AgentDescriptor for lb:analyst and lb:advisor (A2A registry)
- [ ] N8nSubsystem implementing Subsystem trait
- [ ] ImpactGate integration with CognitiveService
- [ ] FreeWill → style self-modification wiring

---

## Task Breakdown by Repository

### 1. ladybug-rs (Cognitive Substrate)

#### 1.1 CognitiveService [DONE]
- [x] `service.rs` — Mode-aware facade (PassiveRag/Brain/Orchestrated)
- [x] `ThinkingStyleBridge` — 36→12 style mapping
- [x] `CognitiveSnapshot` — Cross-crate state exchange
- [x] Grammar Triangle → CognitiveFabric → CollapseGate pipeline
- [x] Serialization/deserialization
- [x] Tests

#### 1.2 StepHandler Implementation [DONE]
- [x] Create `src/cognitive/step_handler.rs`
- [x] `LbStepHandler` wraps CognitiveService with UnifiedStep dispatch
- [x] Route `lb.query` → query_text() (all modes)
- [x] Route `lb.resonate` → query_resonance() (all modes, base64 or text)
- [x] Route `lb.process` → process_text() (Brain/Orchestrated, optional style)
- [x] Route `lb.gate` → evaluate_gate() (arbitrary scores)
- [x] Route `lb.style` → set_style_external() + modulation JSON output
- [x] Route `lb.snapshot` → snapshot() or basic state info
- [x] Route `lb.mode` → set_mode() with previous/new tracking
- [x] Route `lb.reset` → reset() cognitive state
- [x] 16 tests passing

#### 1.3 Subsystem Trait [DONE]
- [x] Create `src/cognitive/subsystem_impl.rs`
- [x] `LadybugSubsystem` with mode-aware lifecycle
  - `step_handler()` → returns LbStepHandler with configured mode + style
  - `init_state()` → JSON with cognitive snapshot, modulation, capabilities
  - `agent_descriptors()` → lb:analyst (query/resonate/gate) + lb:advisor (style/process)
  - `shutdown()` → lifecycle hook (no-op, future: flush BindSpace)
- [x] `SubsystemBuilder` for fluent configuration
- [x] `AgentDescriptor` with JSON export for A2A registry
- [x] 9 tests passing (including full lifecycle test)

#### 1.4 CognitiveKernel ↔ Service Unification [TODO]
- [ ] Bridge CognitiveKernel (BindSpace operations) into CognitiveService
- [ ] CognitiveService.process_packet() for CogPacket protocol
- [ ] Container ↔ Fingerprint bridging through service layer
- [ ] Satisfaction gate synchronization between Kernel and Fabric

#### 1.5 Fabric Consolidation [TODO]
- [ ] Reconcile `cognitive/fabric.rs` (10-layer, 2-stroke) with
      `cognitive/substrate.rs` (7-layer, mRNA, butterfly)
- [ ] Choose primary: fabric.rs (richer) as canonical
- [ ] Port mRNA + butterfly detection into fabric.rs
- [ ] Deprecate substrate.rs with re-exports

#### 1.6 Grammar → Cognitive Pipeline [TODO]
- [ ] Wire `GrammarTriangle.to_fingerprint()` as the canonical text entry point
- [ ] Feed Grammar qualia dimensions into QuadTriangle Gestalt corner
- [ ] Feed Grammar causality into QuadTriangle Content corner
- [ ] Feed Grammar NSM activations into QuadTriangle Processing corner
- [ ] Bidirectional: Cognitive state → Grammar summary for blackboard

### 2. crewai-rust (Orchestration Layer)

#### 2.1 LadybugSubsystem Registration [TODO]
- [ ] Add `ladybug` feature flag to Cargo.toml
- [ ] Create `src/subsystems/ladybug.rs`
- [ ] Implement `Subsystem` for ladybug integration
- [ ] Register in `SubsystemRegistry` at startup
- [ ] Feature-gated: works without ladybug dependency

#### 2.2 ThinkingStyle Bidirectional Sync [TODO]
- [ ] Map persona ThinkingStyle (36 styles, 23D) → BlackboardEntry.active_style
- [ ] On lb.* step completion, read CognitiveSnapshot from blackboard
- [ ] Update persona emphasis/weight based on cognitive coherence
- [ ] Style switch suggestions from ladybug → persona adjustment

#### 2.3 SemanticKernel → In-Process Bridge [TODO]
- [ ] SemanticKernel currently uses HTTP to reach ladybug BindSpace
- [ ] Add in-process path when compiled together (one-binary mode)
- [ ] Feature gate: `#[cfg(feature = "ladybug")]` → direct, else HTTP
- [ ] BindSpace addresses remain the same (0x0C-0x0F)

#### 2.4 Blackboard CognitiveSnapshot Integration [TODO]
- [ ] Define `BlackboardEntry` compatible with ladybug's `CognitiveSnapshot`
- [ ] Write cognitive state to blackboard after each step
- [ ] Read cognitive state from blackboard in chat handler
- [ ] Use coherence/flow_state to modulate LLM parameters

### 3. n8n-rs (Meta-Orchestrator)

#### 3.1 CognitiveSnapshot in DataEnvelope [TODO]
- [ ] Extend `EnvelopeMetadata` with cognitive awareness fields
- [ ] Map CognitiveSnapshot fields to metadata
- [ ] Propagate through step delegation pipeline

#### 3.2 LadybugRouter Enhancement [TODO]
- [ ] Add `lb.process` step type to LadybugRouter
- [ ] Add `lb.style` step type for style commands
- [ ] Add `lb.gate` step type for gate evaluation
- [ ] Route through CogPacket when in binary mode

#### 3.3 ImpactGate + Cognitive Integration [TODO]
- [ ] Gate cognitive self-modification by RBAC role
- [ ] Only `kernel` and `autonomous_kernel` can modify styles
- [ ] Budget tracking for cognitive cycles (they're not free)
- [ ] Evidence threshold from NARS for style changes

#### 3.4 FreeWill + Style Evolution [TODO]
- [ ] Map style changes to ModificationLimits
- [ ] Self-modification boundary: can the system change its own thinking style?
- [ ] Evidence accumulation before style crystallization
- [ ] A2A: agents negotiate style switches via blackboard

#### 3.5 A2A Blackboard Protocol [TODO]
- [ ] CognitiveSnapshot as A2A message payload
- [ ] Agents discover each other by capability + thinking style
- [ ] Style-based routing: "find me an analytical agent"
- [ ] Cross-agent coherence tracking

### 4. cubus (BF16 Numeric Layer)

#### 4.1 Awareness ↔ CognitiveSnapshot Bridge [TODO]
- [ ] Map cubus `ThinkingStyleMix` to ladybug `ThinkingStyle`
- [ ] Map cubus `EnrichedGate` to ladybug `CollapseGate`
- [ ] Map cubus `AwarenessLoop` to ladybug `CognitiveFabric.process()`
- [ ] Optional dependency: cubus can work standalone or with ladybug types

#### 4.2 BF16 Container Operations [TODO]
- [ ] cubus BF16 decomposition → ladybug WideContainer
- [ ] Shared CollapseGate math (via rustynum-core)
- [ ] BF16 moment metadata → CognitiveSnapshot fields

### 5. rustynum-core (Math Foundation)

#### 5.1 Shared Primitives [DONE]
- [x] DeltaLayer<N> (ephemeral XOR diffs, ground truth `&self` forever)
- [x] LayerStack<N> (multi-writer concurrent stacks with CollapseGate)
- [x] CollapseGate (conflict-threshold gating: Flow/Hold/Block)
- [x] Fingerprint<N> (XOR group with BitXor/BitAnd impls, Hamming distance)
- [x] BF16 Hamming with AVX-512 dispatch + scalar fallback
- [x] BF16 structural diff (sign_flips, exponent_bits, mantissa_bits per dim)
- [x] BF16 superposition decompose → 4-state awareness (Crystallized/Tensioned/Uncertain/Noise)
- [x] Awareness packing (2 bits per dim) and unpacking
- [x] fp32_to_bf16_bytes / bf16_bytes_to_fp32 conversion
- [x] ComputeCaps + ComputeTier hardware detection (AVX-512, AMX, GPU)
- [x] Blackboard (64-byte aligned memory arena)

#### 5.2 Future Primitives [TODO]
- [ ] Majority vote bundle (used by CognitiveKernel.bundle_recent)
- [ ] Hamming similarity (portable version of AVX-512 VPOPCNTDQ)
- [ ] Fingerprint XOR-fold (Container = Fingerprint = 16384 bits)
- [ ] Focus mask: dimension selection based on codebook crystallization history

### 6. ladybug-contract (Pure Types)

#### 6.1 Current State [DONE]
- [x] Container (16384-bit, 256 × u64) *(updated Feb 2026 from 8192-bit)*
- [x] CogRecord (each container = 16K = 2 KB)
- [x] CogPacket wire protocol
- [x] EmbeddingFormat enum

#### 6.2 CognitiveSnapshot Wire Format [TODO]
- [ ] Binary encoding of CognitiveSnapshot for CogPacket
- [ ] Pack style (4 bits) + coherence (8 bits) + gate (2 bits) into header
- [ ] Pack triangle activations (12 × f16 = 24 bytes) into payload
- [ ] Wire-compatible with existing CogPacket layout

### 7. n8n-contract (Execution Types)

#### 7.1 Current State [DONE]
- [x] UnifiedStep / UnifiedExecution / DataEnvelope
- [x] StepDelegation Request/Response
- [x] CrewRouter / LadybugRouter
- [x] ImpactGate / FreeWillPipeline

#### 7.2 Cognitive Awareness [TODO]
- [ ] Add `CognitiveAwareness` fields to `EnvelopeMetadata`
- [ ] Step-level style override in `UnifiedStep.metadata`
- [ ] Gate state propagation through delegation chain

---

## Priority Order

### Phase 1: Vertical Slice (Mode 1 — Passive RAG) [NEAR-COMPLETE]
1. [x] CognitiveService + ThinkingStyleBridge (ladybug-rs) **DONE**
2. [x] LbStepHandler for `lb.*` domain — 8 step types (ladybug-rs) **DONE**
3. [x] LadybugSubsystem with lifecycle + agent descriptors (ladybug-rs) **DONE**
4. [ ] Registration in crewai-rust SubsystemRegistry
5. [ ] End-to-end test: crewai-rust crew → lb.query → response

### Phase 2: Brain Mode (Mode 2) [IN PROGRESS]
6. [ ] Grammar → QuadTriangle bidirectional wiring
7. [ ] CognitiveKernel ↔ Service unification
8. [ ] Persona → ThinkingStyle bidirectional sync
9. [ ] Style-driven cognitive cycle with blackboard output

### Phase 3: Orchestrated (Mode 3)
10. [ ] N8nSubsystem + enhanced LadybugRouter
11. [ ] ImpactGate + cognitive budget
12. [ ] FreeWill + style evolution
13. [ ] A2A blackboard protocol

### Phase 4: Numeric Foundation + BF16 Superposition
14. [ ] BindSpace ↔ rustynum DeltaLayer/LayerStack integration
15. [ ] BF16 superposition sensing in BindSpace (hot path)
16. [ ] Crystal codebook as cold accumulation (feedback loop)
17. [ ] cubus awareness ↔ ladybug CognitiveSnapshot bridge
18. [ ] CognitiveSnapshot wire format in ladybug-contract

---

## BF16 Superposition Architecture (Hot/Cold/Feedback)

The core architectural insight connecting rustynum-core to ladybug-rs BindSpace
is a three-layer loop: **hot sensing**, **cold accumulation**, **feedback bias**.

### The Problem

Scanning all 1024 Jina dimensions (or all 16,384 fingerprint bits) for every
operation is wasteful. Most dimensions are noise for any given context. The
system needs to know WHERE to focus — not through configuration, but through
lived experience.

### The Solution: Three Loops

```
                    ┌──────────────────────────────────────────────────┐
                    │          HOT PATH (microseconds)                 │
                    │                                                  │
                    │  On every insert / query / agent flux:           │
                    │  1. Pick 2-3 relevant vectors from BindSpace     │
                    │  2. superposition_decompose(vectors, thresholds) │
                    │  3. → 4-state per dimension:                     │
                    │       Crystallized = sign+exp agree (real signal)│
                    │       Tensioned = sign disagree (contradiction)  │
                    │       Uncertain = sign agree, exp spread (maybe) │
                    │       Noise = only mantissa differs (ignore)     │
                    │  4. Pack to 2 bits/dim → write to BindSpace meta │
                    │                                                  │
                    │  Cost: ~2µs per decomposition                    │
                    │  Runs: continuously, ambient, every operation    │
                    └──────────────────┬───────────────────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────────────────┐
                    │          COLD PATH (Lance / Crystal)             │
                    │                                                  │
                    │  CrystalCodebook accumulates which superpositions│
                    │  crystallized over time:                         │
                    │  - Which dimension triples showed sign consensus │
                    │  - Which exponent ranges stabilized              │
                    │  - 125 cells (5×5×5 grid) as learned centroids  │
                    │  - NOT a trained model — a lived history         │
                    │                                                  │
                    │  Storage: SPO Crystal (3D content-addressable)   │
                    │  Update: Lloyd iteration on codebook centroids   │
                    │  Compression: centroid + sparse residual (50KB)  │
                    └──────────────────┬───────────────────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────────────────┐
                    │          FEEDBACK LOOP                           │
                    │                                                  │
                    │  Codebook biases WHICH dimensions get weighted   │
                    │  in the next superposition:                      │
                    │                                                  │
                    │  NOT: "scan all 1024 Jina dims"                  │
                    │  BUT: "these 47 dims crystallized before,        │
                    │         weight them in next decomposition"       │
                    │                                                  │
                    │  Focus mask = top dimension triples from history │
                    │  Updated lazily (not every cycle)                │
                    │  Makes superposition smarter over time           │
                    └──────────────────────────────────────────────────┘
```

### CAM = Content-Addressable Memory (Address IS Content)

When 3 dimensions consistently show sign agreement across different contexts,
those 3 dimensions ARE the address of that concept. They emerge from the data,
not from design. The top triples from counting = CAM codebook entries.

```
Dimension triples that crystallize:
  (valence, agency, temporality)  → emotional intent fingerprint
  (depth, complexity, coherence)  → analytical difficulty address
  (novelty, salience, aesthetic)  → creative inspiration address

These aren't designed — they're discovered by counting sign consensus.
```

### Where BF16 Benefits Most

| Component | Current | With BF16 | Benefit |
|-----------|---------|-----------|---------|
| **SemanticKernel** | HTTP + JSON serialization | In-process BF16 Hamming | 1000x faster, zero-copy |
| **Crystal4K / SPO** | f64 similarity | BF16 structured diff | Sign=polarity, Exp=magnitude, Mantissa=noise separation |
| **Sentence Crystal** | Jina 1024D → random projection | BF16 superposition → 4-state awareness | Knows WHERE focus is, not just popcount |
| **Crystal Compress** | Codebook + residual | BF16 codebook centroids | 2B per dim vs 4B, same semantic fidelity |
| **BindSpace** | Fingerprint [u64;256] | DeltaLayer<256> overlays | Multi-writer concurrent, XOR algebra, CollapseGate |

### BindSpace as the Hub

All services should be nudged toward BindSpace:

```
              BindSpace (65,536 addresses)
              ┌─────────────────────────────────────┐
              │ Surface 0x00-0x0B: cognitive state   │
              │ Orchestration 0x0C-0x0F: agents/A2A  │
              │ Fluid 0x10-0x7F: working memory      │
              │ Nodes 0x80-0xFF: crystallized truth   │
              │                                       │
              │ Each address:                         │
              │   [u64; 256] = Fingerprint2K          │
              │   ≡ rustynum Fingerprint<256>          │
              │   Overlaid with DeltaLayer<256>        │
              │   Multi-writer via LayerStack<256>     │
              │   Gated by CollapseGate (Flow/Hold/   │
              │   Block)                               │
              └───────────────┬─────────────────────┘
                              │
          ┌───────────────────┼─────────────────────┐
          ▼                   ▼                     ▼
   Grammar Triangle      CognitiveKernel      SemanticKernel
   text → fp → write     L1-L10 → BindOps    crewai ↔ BindSpace
                              │
                              ▼
                    BF16 superposition sensing
                    (on every read/write pair)
                              │
                              ▼
                    CrystalCodebook accumulation
                    (background, lazy update)
                              │
                              ▼
                    Focus mask → next sensing weighted
```

### rustynum-core ↔ ladybug-rs Type Mapping

| rustynum-core | ladybug-rs | Notes |
|---------------|------------|-------|
| `Fingerprint<256>` | `Fingerprint` ([u64;256]) | Same layout, zero-copy cast |
| `DeltaLayer<256>` | — (new) | XOR overlay on BindSpace nodes |
| `LayerStack<256>` | — (new) | Multi-writer per BindSpace address |
| `CollapseGate` | `GateState` (Flow/Hold/Block) | Same semantics, unify enum |
| `BF16Weights` | — (new) | Per-search weight tuning |
| `SuperpositionState` | — (new) | 4-state awareness per dimension |
| `AwarenessThresholds` | — (new) | Configurable per thinking style |
| `BF16StructuralDiff` | — (new) | Learning signal per update |
| `Blackboard` | `BindSpace` | Converge to BindSpace as canonical |

### 256Kbit Container Architecture (Future)

The next-generation CogRecord format expands to 4 containers:

```
┌──────────────────────────────────────────────────────┐
│  Container 0 (upper): METADATA                       │
│  - neo4j-rs DN tree pointers (LCRS encoding)         │
│  - Hash search index (O(1) prefix lookup)            │
│  - Format field: describes how to interpret C1-C3    │
│  - Quick node/edge metadata for graph traversal      │
│                                                      │
│  Container 1-3 (lower): PAYLOAD                      │
│  Option A: 3D spatial leaf (one container per axis)   │
│  Option B: Hybrid bitpacked                          │
│    - Jina 1024D embedding (BF16: 2KB)                │
│    - Semantic fingerprint (remaining bits)            │
│    - Format field in C0 selects interpretation        │
│                                                      │
│  The format field enables:                           │
│  - DN tree lookup (graph traversal)                  │
│  - Hash search (content addressing)                  │
│  - BF16 awareness (superposition sensing)            │
│  - All without touching containers you don't need    │
└──────────────────────────────────────────────────────┘
```

---

## Key Files Reference

| File | Purpose |
|------|---------|
| **ladybug-rs cognitive pipeline** | |
| `src/cognitive/service.rs` | Mode-aware CognitiveService (3 modes) |
| `src/cognitive/step_handler.rs` | LbStepHandler — lb.* step dispatch (8 types) |
| `src/cognitive/subsystem_impl.rs` | LadybugSubsystem lifecycle + agent descriptors |
| `src/cognitive/fabric.rs` | CognitiveFabric (10-layer, 2-stroke, satisfaction) |
| `src/cognitive/substrate.rs` | CognitiveSubstrate (7-layer, mRNA, butterfly) |
| `src/cognitive/cognitive_kernel.rs` | CognitiveKernel (BindSpace bridge, L1-L10 ops) |
| `src/cognitive/style.rs` | 12 ThinkingStyles + FieldModulation |
| `src/cognitive/collapse_gate.rs` | SD-based FLOW/HOLD/BLOCK |
| `src/cognitive/quad_triangle.rs` | 4×3 QuadTriangle (120K bits) |
| **ladybug-rs grammar pipeline** | |
| `src/grammar/triangle.rs` | Grammar Triangle (NSM+Causality+Qualia→10Kbit) |
| `src/grammar/nsm.rs` | 65 NSM semantic primitives |
| `src/grammar/qualia.rs` | 18D qualia field |
| `src/grammar/causality.rs` | Causality flow (WHO→DID→WHAT→WHY) |
| **ladybug-rs extensions (BF16 target)** | |
| `src/extensions/spo/spo.rs` | SPO Crystal — 5×5×5 content-addressable graph |
| `src/extensions/sentence_crystal.rs` | Sentence Crystal — Jina 1024D → 5^5 grid + NSM |
| `src/extensions/compress/compress.rs` | Crystal Compress — learned codebook quantization |
| **ladybug-rs storage** | |
| `src/storage/bind_space.rs` | BindSpace (65K addresses, [u64;256] per node) |
| `src/contract/` | UnifiedStep/DataEnvelope/EnrichmentEngine/Spectator |
| **crewai-rust** | |
| `src/contract/pipeline.rs` | Pipeline + StepRouter + Phase |
| `src/contract/router.rs` | StepDomain enum + StepHandler trait |
| `src/contract/subsystem.rs` | Subsystem trait + SubsystemRegistry |
| `src/persona/thinking_style.rs` | 36 styles, 23D, 6 clusters |
| `src/chat/semantic_kernel.rs` | HTTP bridge to ladybug BindSpace |
| **rustynum-core (BF16 foundation)** | |
| `src/bf16_hamming.rs` | BF16 Hamming, structural diff, superposition decompose |
| `src/delta.rs` | DeltaLayer<N> — ephemeral XOR on immutable ground truth |
| `src/layer_stack.rs` | LayerStack<N> + CollapseGate (Flow/Hold/Block) |
| `src/fingerprint.rs` | Fingerprint<N> — XOR group, Hamming, type aliases |
| `src/compute.rs` | ComputeCaps + ComputeTier hardware detection |
| **other repos** | |
| `n8n-rs/n8n-rust/crates/n8n-contract/src/` | Unified execution types |
| `cubus/cubus/src/awareness.rs` | BF16 awareness substrate |
| `ladybug-contract/src/` | Container, WideContainer, CogRecord8K, CogPacket |

---

## ThinkingStyle Mapping Reference

### crewai-rust 36 → ladybug 12

| crewai-rust Style | Cluster | ladybug Style | ladybug Cluster |
|-------------------|---------|---------------|-----------------|
| analytical, logical, deductive | Analytical | Analytical | Convergent |
| critical, structured, methodical | Analytical | Systematic | Convergent |
| convergent, precise, rigorous | Analytical | Convergent | Convergent |
| creative, imaginative, artistic | Creative | Creative | Divergent |
| innovative, lateral, brainstorming | Creative | Divergent | Divergent |
| visionary, generative, inventive | Creative | Exploratory | Divergent |
| empathic, compassionate, understanding | Empathic | Diffuse | Attention |
| supportive, emotional, nurturing | Empathic | Peripheral | Attention |
| intuitive, instinctive, gut-feeling | Empathic | Intuitive | Speed |
| direct, decisive, action-oriented | Direct | Focused | Attention |
| pragmatic, efficient, results-driven | Direct | Systematic | Convergent |
| exploratory, curious, investigative | Exploratory | Exploratory | Divergent |
| research-oriented, questioning | Exploratory | Deliberate | Speed |
| metacognitive, reflective | Meta | Metacognitive | Meta |
| philosophical, abstract, systemic | Meta | Metacognitive | Meta |
| integrative, holistic, synthesizing | Meta | Convergent | Convergent |

### ladybug 12 → FieldModulation

| Style | Threshold | Fan-out | Depth | Breadth | Noise | Speed | Explore |
|-------|-----------|---------|-------|---------|-------|-------|---------|
| Analytical | 0.85 | 3 | 1.0 | 0.1 | 0.05 | 0.1 | 0.05 |
| Convergent | 0.75 | 4 | 0.8 | 0.2 | 0.10 | 0.3 | 0.10 |
| Systematic | 0.70 | 5 | 0.7 | 0.3 | 0.15 | 0.2 | 0.10 |
| Creative | 0.35 | 12 | 0.2 | 1.0 | 0.40 | 0.5 | 0.80 |
| Divergent | 0.40 | 10 | 0.3 | 0.9 | 0.35 | 0.4 | 0.70 |
| Exploratory | 0.30 | 15 | 0.4 | 0.8 | 0.50 | 0.6 | 0.90 |
| Focused | 0.90 | 1 | 1.0 | 0.0 | 0.02 | 0.2 | 0.00 |
| Diffuse | 0.45 | 8 | 0.3 | 0.7 | 0.30 | 0.5 | 0.40 |
| Peripheral | 0.20 | 20 | 0.1 | 0.5 | 0.60 | 0.7 | 0.60 |
| Intuitive | 0.50 | 3 | 0.3 | 0.4 | 0.25 | 0.9 | 0.30 |
| Deliberate | 0.70 | 7 | 0.6 | 0.5 | 0.10 | 0.1 | 0.20 |
| Metacognitive | 0.50 | 5 | 0.5 | 0.5 | 0.20 | 0.3 | 0.30 |

---

## Commit History (this session)

| Commit | Description |
|--------|-------------|
| `1800ee3` | feat: CognitiveService + ThinkingStyleBridge + INTEGRATION_PLAN |
| `8fe497e` | feat: LbStepHandler + LadybugSubsystem for lb.* domain routing |

## Backups

- **crewai-rust**: DO NOT force push. Branch `claude/vsaclip-hamming-recognition-y0b94` has merge of main.
- **n8n-rs**: Untouched backup. Do not modify.
- **cubus**: Branch pushed to `adaworld` remote. Safe.
- **rustynum**: Branch pushed. Safe.
- **ladybug-rs**: Active development branch. This integration plan committed here.

---

*Last updated: 2026-02-23 (session 2 — BF16 architecture + step handler + subsystem)*
*Session: claude/vsaclip-hamming-recognition-y0b94*
*Total cognitive tests: 44 (19 service + 16 step_handler + 9 subsystem)*
