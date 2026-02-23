# AdaWorld Unified Cognitive Integration Plan

## Status: IN PROGRESS

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
                  │             │ BindSpace    │
                  │             └──────┬───────┘
                  │                    │
                  └──────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │ ladybug-contract │
                    │ Container 8192b  │
                    │ WideContainer    │
                    │ CogRecord8K     │
                    │ CogPacket wire  │
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
                    │ NumElement trait │
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
- [ ] StepHandler for `lb.query` step type
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
- [ ] N8nSubsystem implementing Subsystem trait
- [ ] LadybugSubsystem implementing Subsystem trait
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

#### 1.2 StepHandler Implementation [TODO]
- [ ] Create `src/cognitive/step_handler.rs`
- [ ] Implement `StepHandler` trait for ladybug domain (`lb.*`)
- [ ] Route `lb.query` → CognitiveService.query_text()
- [ ] Route `lb.resonate` → CognitiveService.query_resonance()
- [ ] Route `lb.process` → CognitiveService.process_text()
- [ ] Route `lb.gate` → CognitiveService.evaluate_gate()
- [ ] Route `lb.style` → CognitiveService.set_style_external()
- [ ] Write step output to Blackboard as CognitiveSnapshot

#### 1.3 Subsystem Trait [TODO]
- [ ] Create `src/cognitive/subsystem_impl.rs`
- [ ] Implement `Subsystem` for `LadybugSubsystem`
  - `step_handler()` → returns StepHandler from 1.2
  - `init_blackboard()` → pre-populate with default CognitiveSnapshot
  - `register_agents()` → register cognitive agents in A2A
  - `install_hooks()` → lifecycle hooks for style sync
- [ ] Mode-aware: PassiveRag vs Brain behavior

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
- [x] DeltaLayer (address-sparse XOR)
- [x] LayerStack (sequential processing with merge)
- [x] CollapseGate (SD-based gating: FLOW/HOLD/BLOCK)
- [x] NumElement trait alias

#### 5.2 Future Primitives [TODO]
- [ ] Majority vote bundle (used by CognitiveKernel.bundle_recent)
- [ ] Hamming similarity (portable version of AVX-512 VPOPCNTDQ)
- [ ] Fingerprint XOR-fold (Container 8192 ↔ Fingerprint 16384)

### 6. ladybug-contract (Pure Types)

#### 6.1 Current State [DONE]
- [x] Container (8192-bit, 128 × u64)
- [x] WideContainer (16384-bit, 256 × u64)
- [x] CogRecord / CogRecord8K
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

### Phase 1: Vertical Slice (Mode 1 — Passive RAG)
1. [x] CognitiveService + ThinkingStyleBridge (ladybug-rs) **DONE**
2. [ ] StepHandler for `lb.*` domain (ladybug-rs)
3. [ ] LadybugSubsystem trait impl (ladybug-rs)
4. [ ] Registration in crewai-rust SubsystemRegistry
5. [ ] End-to-end test: crewai-rust crew → lb.query → response

### Phase 2: Brain Mode (Mode 2)
6. [ ] Grammar → QuadTriangle bidirectional wiring
7. [ ] CognitiveKernel ↔ Service unification
8. [ ] Persona → ThinkingStyle bidirectional sync
9. [ ] Style-driven cognitive cycle with blackboard output

### Phase 3: Orchestrated (Mode 3)
10. [ ] N8nSubsystem + enhanced LadybugRouter
11. [ ] ImpactGate + cognitive budget
12. [ ] FreeWill + style evolution
13. [ ] A2A blackboard protocol

### Phase 4: Numeric Foundation
14. [ ] cubus awareness ↔ ladybug CognitiveSnapshot bridge
15. [ ] Shared rustynum-core primitives (majority vote, Hamming)
16. [ ] CognitiveSnapshot wire format in ladybug-contract

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `ladybug-rs/src/cognitive/service.rs` | **NEW** Mode-aware CognitiveService |
| `ladybug-rs/src/cognitive/fabric.rs` | CognitiveFabric (10-layer, 2-stroke) |
| `ladybug-rs/src/cognitive/substrate.rs` | CognitiveSubstrate (7-layer, mRNA) |
| `ladybug-rs/src/cognitive/cognitive_kernel.rs` | CognitiveKernel (BindSpace bridge) |
| `ladybug-rs/src/cognitive/style.rs` | 12 ThinkingStyles + FieldModulation |
| `ladybug-rs/src/cognitive/collapse_gate.rs` | SD-based FLOW/HOLD/BLOCK |
| `ladybug-rs/src/cognitive/quad_triangle.rs` | 4×3 QuadTriangle |
| `ladybug-rs/src/grammar/triangle.rs` | Grammar Triangle (NSM+Causality+Qualia) |
| `ladybug-rs/src/grammar/nsm.rs` | 65 NSM semantic primitives |
| `ladybug-rs/src/grammar/qualia.rs` | 18D qualia field |
| `ladybug-rs/src/grammar/causality.rs` | Causality flow (WHO→DID→WHAT→WHY) |
| `ladybug-rs/src/extensions/` | NSM substrate, SPO crystal, hologram, CAM |
| `crewai-rust/src/contract/pipeline.rs` | Pipeline + StepRouter + Phase |
| `crewai-rust/src/contract/router.rs` | StepDomain enum + StepHandler trait |
| `crewai-rust/src/contract/subsystem.rs` | Subsystem trait + SubsystemRegistry |
| `crewai-rust/src/persona/thinking_style.rs` | 36 styles, 23D, 6 clusters |
| `crewai-rust/src/chat/semantic_kernel.rs` | HTTP bridge to ladybug BindSpace |
| `n8n-rs/n8n-rust/crates/n8n-contract/src/` | Unified execution types |
| `cubus/cubus/src/awareness.rs` | BF16 awareness substrate |
| `rustynum/rustynum-core/src/ops/` | DeltaLayer, LayerStack, CollapseGate |

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

## Backups

- **crewai-rust**: DO NOT force push. Branch `claude/vsaclip-hamming-recognition-y0b94` has merge of main.
- **n8n-rs**: Untouched backup. Do not modify.
- **cubus**: Branch pushed to `adaworld` remote. Safe.
- **rustynum**: Branch pushed. Safe.
- **ladybug-rs**: Active development branch. This integration plan committed here.

---

*Last updated: 2026-02-23*
*Session: claude/vsaclip-hamming-recognition-y0b94*
