# Strategic Node Integration Plan — Unified Reasoning ↔ Agency ↔ Self-Actualization

> **Created**: 2026-02-15
> **Branch**: `claude/chess-ai-multiagent-zDaCV`
> **Repos**: ladybug-rs, crewai-rust, n8n-rs, neo4j-rs, aiwar-neo4j-harvest, stonksfish
> **Status**: Architecture approved — implementation subtasks defined below
> **Principle**: The 10-layer cognitive stack + 2-stroke engine IS the strategy.
> Strategy emerges from thinking_style modulation, not from a separate reasoning module.

---

## 1. Core Insight

Four repos currently have independent node representations:

| Repo | Node Type | What It Encodes |
|------|-----------|-----------------|
| **ladybug-rs** | `CogRecord` (8192-bit container, 128×u64 metadata) | Fingerprint + NARS truth + edges + Q-values + qualia + graph metrics |
| **crewai-rust** | `AgentBlueprint` + `ModuleInner` (YAML) | role, goal, thinking_style[10], persona, skills |
| **n8n-rs** | Workflow nodes | Orchestration steps, triggers, connections, execution state |
| **neo4j-rs** | `Node { id, labels, properties }` | Property graph model, Cypher queryable |

**The unification**: The ladybug-rs container metadata already encodes ALL of these
as word-ranges in a single 1KB binary structure. Rather than transcoding between
formats, we make the container the single source of truth and provide typed views
for each consumer.

---

## 2. The Feedback Loop (Full Cycle)

```
    ┌──────────────────────────────────────────────────────────────────────┐
    │                    THE SELF-ACTUALIZATION LOOP                       │
    │                                                                      │
    │  ┌─────────────┐     ┌──────────────┐     ┌─────────────────────┐  │
    │  │  REASONING   │────►│   AGENCY      │────►│  SELF-ACTUALIZATION │  │
    │  │  (ladybug)   │     │  (crewai/n8n) │     │  (crystallization)  │  │
    │  └──────▲───────┘     └──────┬────────┘     └─────────┬───────────┘  │
    │         │                    │                         │              │
    │         │         ┌──────────▼────────┐               │              │
    │         │         │ Agent executes     │               │              │
    │         │         │ with thinking_style│               │              │
    │         │         │ modulating 10-layer│               │              │
    │         │         │ 2-stroke engine    │               │              │
    │         │         └──────────┬────────┘               │              │
    │         │                    │                         │              │
    │         │         ┌──────────▼────────┐               │              │
    │         │         │ L8 Integration:    │               │              │
    │         │         │ merge multi-agent  │               │              │
    │         │         │ evidence via       │               │              │
    │         │         │ majority-vote      │               │              │
    │         │         │ bundling           │               │              │
    │         │         └──────────┬────────┘               │              │
    │         │                    │                         │              │
    │         │         ┌──────────▼────────┐               │              │
    │         │         │ L9 Validation:     │               │              │
    │         │         │ NARS truth harden  │◄──────────────┘              │
    │         │         │ + Brier calibrate  │                              │
    │         │         │ + Dunning-Kruger   │                              │
    │         │         └──────────┬────────┘                              │
    │         │                    │                                        │
    │         │         ┌──────────▼────────┐                              │
    │         │         │ L10 Crystallize:   │                              │
    │         │         │ bind result with   │                              │
    │         │         │ modulation FP      │                              │
    │         │         │ → write to 0x80    │                              │
    │         │         │   (Node zone)      │                              │
    │         │         └──────────┬────────┘                              │
    │         │                    │                                        │
    │         │         ┌──────────▼─────────────────────┐                 │
    │         │         │ SELF-MODIFICATION:              │                 │
    │         │         │ Recover which thinking_style    │                 │
    │         │         │ produced this result:           │                 │
    │         │         │   modulation = crystal ⊕ content│                 │
    │         │         │                                 │                 │
    │         │         │ If self_modify == Constrained:  │                 │
    │         │         │   tune thinking_style ±0.1/step │                 │
    │         │         │   (PersonaProfile constraint)   │                 │
    │         │         │                                 │                 │
    │         │         │ Updated style → next YAML cycle │                 │
    │         └─────────│ → feeds L2 Resonance on reentry │                 │
    │                   └────────────────────────────────┘                 │
    │                                                                      │
    └──────────────────────────────────────────────────────────────────────┘
```

### 2.1 Step-by-step cycle

1. **Input arrives** (chess position, workflow trigger, user query)
2. **L1 Recognition** (threshold from `thinking_style[0]`): Pattern match against
   codebook fingerprints in BindSpace. Chess: identify position type, piece patterns.
3. **L2 Resonance** (threshold from `thinking_style[1]`): Similarity search across
   Fluid + Node zones. Finds similar positions, prior crystallized knowledge.
4. **L3 Appraisal** (threshold from `thinking_style[2]`): Form hypothesis. Maps to
   Pearl Rung 1 (SEE — correlation). NARS truth values updated.
5. **L4 Routing** (threshold from `thinking_style[3]`): Select thinking style for
   this problem via `select_style_by_resonance()`. Dispatch to best-matching style.
6. **L5 Execution** (threshold from `thinking_style[4]`): Produce output. Maps to
   Pearl Rung 2 (DO — intervention). Write result to Fluid zone (0x10).

   --- SINGLE AGENT BOUNDARY ---

7. **L6 Delegation** (threshold from `thinking_style[5]`): Fan out to other agents.
   **This is where crewai-rust and n8n-rs enter.** The Strategist's 0.95 delegation
   fires hard, dispatching to Tactician, Critic, Psychologist, etc.
8. **L7 Contingency** (threshold from `thinking_style[6]`): "Could be otherwise."
   XOR-counterfactual search (Pearl Rung 3: IMAGINE). The Critic's 0.95 contingency
   searches for refutations of every proposed move.
9. **L8 Integration** (threshold from `thinking_style[7]`): Merge all specialist
   outputs via majority-vote bundling. Evidence from 6 chess agents converges.
10. **L9 Validation** (threshold from `thinking_style[8]`): Three orthogonal checks:
    - NARS: `confidence > 0.5 && frequency > 0.3`
    - Brier calibration: `!should_admit_ignorance && calibration_error < 0.2`
    - Dunning-Kruger: `confidence > 0.8 && uncertainty > 0.4` → REJECT
11. **L10 Crystallization** (threshold from `thinking_style[9]`): Bind validated
    result with the modulation fingerprint that produced it. Write to Node zone.
    **This IS learning.** No rule table. The fingerprint IS the knowledge.

12. **Self-Modification** (if `PersonaProfile.self_modify == Constrained`):
    - Recover modulation: `modulation_fp = crystal_fp ⊕ content_fp`
    - Compare recovered modulation to current thinking_style
    - If result was good (L9 Pass): nudge style toward what worked (±0.1 max)
    - Updated thinking_style feeds back into next cycle's thresholds
    - **This is self-actualization** — the agent learns HOW to think, not just WHAT

---

## 3. Container Metadata as Universal Node

### 3.1 Mapping: thinking_style[10] → Metadata Words

```
thinking_style[0]  recognition     → W12-15 Layer 1 markers (strength, frequency, recency)
thinking_style[1]  resonance       → W12-15 Layer 2 markers
thinking_style[2]  appraisal       → W12-15 Layer 3 markers
thinking_style[3]  routing         → W12-15 Layer 4 markers
thinking_style[4]  execution       → W12-15 Layer 5 markers
thinking_style[5]  delegation      → W12-15 Layer 6 markers
thinking_style[6]  contingency     → W12-15 Layer 7 markers
thinking_style[7]  integration     → Expand to W12-15 Layer 8 (uses reserved W14 bits)
thinking_style[8]  validation      → Expand to W12-15 Layer 9 (uses reserved W15 bits)
thinking_style[9]  crystallization → Expand to W12-15 Layer 10 (uses reserved W15 bits)
```

> **NOTE**: W12-15 currently encodes 7 layers × 5 bytes = 35 bytes (280 bits).
> The 256-bit budget (W12-15) accommodates 10 layers × 3 bytes = 30 bytes (240 bits)
> by reducing per-layer from 5 bytes to 3 bytes (strength:u8, frequency:u8, flags:u8),
> or using 10 layers × 25.6 bits = 256 bits exactly with custom packing.
> **This is a backward-compatible layout change** within the same word range.

### 3.2 Mapping: crewai-rust Agent → Container Words

```
AgentBlueprint.role          → W0    DN address (identity)
AgentBlueprint.domain        → W1    node_kind byte (SavantDomain → u8)
AgentBlueprint.skills        → W16-31 inline edges (skill_id → verb:target packed)
ModuleInner.thinking_style   → W12-15 layer markers (see §3.1)
PersonaProfile.volition_axes → W56-63 qualia channels [0-4] (curiosity, autonomy, persistence, caution, empathy)
PersonaProfile.affect_baseline → W56-63 qualia channels [5-12] (Plutchik's 8 emotions)
ModuleInner.collapse_gate    → W8    gate_state byte (FLOW=0, HOLD=1, BLOCK=2)
AgentBlueprint.allow_delegation → W1 flags byte, bit 4
SkillDescriptor.proficiency  → W4-7  NARS confidence (skill confidence ≈ belief confidence)
```

### 3.3 Mapping: n8n-rs Workflow Node → Container Words

```
Workflow node ID             → W0    DN address
Workflow node type           → W1    node_kind (trigger, action, condition, etc.)
Workflow execution state     → W8    gate_state (pending=HOLD, running=FLOW, error=BLOCK)
Workflow connections (edges) → W16-31 inline edges (verb=TRIGGERS/FEEDS/GUARDS)
Workflow node parameters     → Content container (geometry=Chunked for large payloads)
Orchestration decisions      → W32-39 Q-values (which downstream node to route to)
Workflow node health metrics → W48-55 graph metrics (throughput=pagerank proxy)
```

### 3.4 Mapping: neo4j-rs Node → Container Words

```
Node.id                      → W0    DN address
Node.labels                  → W40-47 Bloom filter (label membership test)
Node.properties              → Content container (geometry=CAM or Chunked)
Relationship edges           → W16-31 inline edges (rel_type → verb_id)
Graph metrics (pagerank, etc.)→ W48-55 graph metrics (stored natively)
```

---

## 4. strategy.rs — Domain Binding Layer

### 4.1 Purpose

`strategy.rs` lives in ladybug-rs and provides:
1. **StrategicNode** — a typed view over `CogRecord` that unifies all 4 representations
2. **AI War tactical concept codebook** — fingerprints for the 10 tactical bridge concepts
3. **SPO crystal bindings** — strategy triples (Agent, Action, Outcome)
4. **thinking_style ↔ FieldModulation bridge** — converts YAML vectors to 2-stroke parameters
5. **Self-modification protocol** — constrained style tuning via crystallization recovery

### 4.2 StrategicNode API

```rust
/// Zero-copy view over a CogRecord that provides unified access
/// to the cognitive, agentic, and orchestration layers.
pub struct StrategicNode<'a> {
    record: &'a CogRecord,
}

impl<'a> StrategicNode<'a> {
    // === Identity (W0-W3) ===
    fn dn_address(&self) -> PackedDn;
    fn node_kind(&self) -> NodeKind;
    fn domain(&self) -> SavantDomain;

    // === Cognitive State (W4-W15) ===
    fn nars_truth(&self) -> NarsTruth;           // W4-7: frequency, confidence, evidence
    fn gate_state(&self) -> GateState;           // W8: FLOW/HOLD/BLOCK
    fn thinking_style(&self) -> [f32; 10];       // W12-15: 10-layer activation
    fn to_field_modulation(&self) -> FieldModulation;  // Convert style → 2-stroke params

    // === Graph Topology (W16-W55) ===
    fn inline_edges(&self) -> impl Iterator<Item = (VerbId, TargetAddr)>;  // W16-31
    fn q_values(&self) -> [f32; 16];             // W32-39: RL action values
    fn bloom_contains(&self, id: u64) -> bool;   // W40-47: neighbor membership
    fn graph_metrics(&self) -> GraphMetrics;     // W48-55: pagerank, degree, clustering

    // === Awareness (W56-W63) ===
    fn qualia(&self) -> QualiaChannels;          // 18 × f16 affect channels
    fn volition_axes(&self) -> [f32; 5];         // curiosity, autonomy, persistence, caution, empathy
    fn affect_baseline(&self) -> [f32; 8];       // Plutchik's wheel

    // === Strategic Operations ===
    fn tactical_similarity(&self, other: &StrategicNode) -> f32;  // Hamming-based
    fn bind_role(&self, role: &Fingerprint) -> Fingerprint;        // XOR role binding
    fn unbind_role(&self, bound: &Fingerprint) -> Fingerprint;     // XOR recovery
    fn store_spo_triple(&mut self, s: &Fingerprint, p: &Fingerprint, o: &Fingerprint);
    fn query_spo(&self, s: Option<&Fingerprint>, p: Option<&Fingerprint>) -> Vec<SpoMatch>;

    // === Self-Modification ===
    fn recover_modulation(&self, content_fp: &Fingerprint) -> FieldModulation;
    fn propose_style_update(&self, outcome: ValidationResult) -> Option<[f32; 10]>;
}
```

### 4.3 AI War Tactical Concept Codebook

The 10 chess↔AI War bridge concepts become fingerprints in the CAM codebook,
not property graph nodes in neo4j-rs:

```rust
/// CAM codebook entries for AI War tactical concepts.
/// Category 0x600 (Crystal/temporal) + 0x700 (NSM semantic).
pub struct TacticalCodebook {
    // Chess domain fingerprints (bound with ROLE_CHESS)
    material:           Fingerprint,  // 0x600:00
    pawn_structure:     Fingerprint,  // 0x600:01
    king_safety:        Fingerprint,  // 0x600:02
    piece_activity:     Fingerprint,  // 0x600:03
    tactical_threats:   Fingerprint,  // 0x600:04
    strategic_plan:     Fingerprint,  // 0x600:05
    game_phase:         Fingerprint,  // 0x600:06
    opening_theory:     Fingerprint,  // 0x600:07
    endgame_technique:  Fingerprint,  // 0x600:08
    time_pressure:      Fingerprint,  // 0x600:09

    // AI War domain fingerprints (bound with ROLE_AIWAR)
    capabilities:         Fingerprint,  // 0x700:00
    infrastructure:       Fingerprint,  // 0x700:01
    vulnerability_surface: Fingerprint, // 0x700:02
    operational_tempo:    Fingerprint,  // 0x700:03
    attack_vectors:       Fingerprint,  // 0x700:04
    deployment_strategy:  Fingerprint,  // 0x700:05
    system_maturity:      Fingerprint,  // 0x700:06
    known_patterns:       Fingerprint,  // 0x700:07
    capability_conversion: Fingerprint, // 0x700:08
    decision_latency:     Fingerprint,  // 0x700:09
}

impl TacticalCodebook {
    /// Cross-domain binding: chess concept ⊕ ROLE_CHESS = chess-contextualized FP
    /// Same concept ⊕ ROLE_AIWAR = aiwar-contextualized FP
    /// Similarity between domains = hamming(chess_bound, aiwar_bound)
    pub fn cross_domain_similarity(&self, chess_idx: usize, aiwar_idx: usize) -> f32;

    /// Bind a chess position fingerprint with a tactical concept
    /// to find "how much Material advantage does this position have?"
    pub fn measure_concept(&self, position_fp: &Fingerprint, concept: TacticalConcept) -> f32;
}
```

### 4.4 SPO Crystal for Strategy Triples

```rust
/// Store strategic plans as SPO triples in the 5×5×5 crystal:
///   SPO(Strategist, e4, KingSideAttack) → crystal[h(S)%5, h(P)%5, h(O)%5]
///
/// Query by any axis:
///   "What plans has Strategist proposed?" → query(S=Strategist, P=*, O=*)
///   "What actions lead to KingSideAttack?" → query(S=*, P=*, O=KingSideAttack)
///   "Who has used e4?" → query(S=*, P=e4, O=*)
pub struct StrategyCrystal {
    crystal: SpoCrystal,  // 5×5×5 × 16384-bit cells
}
```

---

## 5. Agency Integration: n8n-rs + crewai-rust

### 5.1 How Reasoning Feeds Into Agency

The 2-stroke engine's L6 (Delegation) layer is the bridge from reasoning to agency:

```
ladybug-rs L6 fires (thinking_style[5] > threshold)
    │
    ▼
crewai-rust receives delegation via Arrow Flight DoAction("crew.delegate_task")
    │
    ▼
n8n-rs orchestrates the multi-agent workflow:
    ├── Route to Tactician (if tactical verification needed)
    ├── Route to Psychologist (if opponent modeling needed)
    ├── Route to Critic (if refutation search needed)
    ├── Route to Advocatus Diaboli (if stress-testing needed)
    └── Route to Endgame Specialist (if piece_count < 10)
    │
    ▼
Each agent runs its OWN 2-stroke cycle with its OWN thinking_style:
    Tactician:  [0.95, 0.4, 0.7, 0.5, 0.95, 0.1, 0.8, 0.4, 0.95, 0.6]
    Critic:     [0.9, 0.5, 0.95, 0.7, 0.7, 0.1, 0.95, 0.5, 0.95, 0.6]
    etc.
    │
    ▼
Results flow back to Strategist's BindSpace (Fluid zone 0x10)
    │
    ▼
Strategist's L8 (Integration=0.95) merges all specialist outputs
Strategist's L9 (Validation=0.8) truth-hardens the merged result
Strategist's L10 (Crystallization=0.9) persists what survives
```

### 5.2 How Agency Feeds Back Into Thinking Styles

This is the self-actualization loop:

```rust
// After L10 crystallization:
let crystal_fp = crystallize(&validated_result, &current_modulation);

// Store in Node zone — this IS the learned knowledge
bind_space.write_at(node_prefix, slot, crystal_fp);

// === SELF-MODIFICATION (if PersonaProfile allows) ===

// 1. Recover which modulation produced this good result
let recovered_mod = recover_modulation(&crystal_fp, &validated_result);

// 2. Compare to current thinking_style
let current_style = agent.thinking_style();
let recovered_style = modulation_to_style(&recovered_mod);

// 3. Compute delta (constrained to ±0.1 per step)
let delta: [f32; 10] = array::from_fn(|i| {
    (recovered_style[i] - current_style[i]).clamp(-0.1, 0.1)
});

// 4. Apply update (only if self_modify == Constrained or Open)
if agent.persona.self_modify != SelfModifyBounds::None {
    agent.update_thinking_style(|style| {
        for i in 0..10 {
            style[i] = (style[i] + delta[i]).clamp(0.0, 1.0);
        }
    });

    // 5. Persist updated style to YAML / BindSpace
    //    → Next cycle uses the updated thresholds
    //    → Agent has LEARNED how to think about this type of problem
}
```

### 5.3 The Self-Orchestration Dimension

n8n-rs workflow nodes can themselves be strategic nodes:

```
n8n-rs workflow:
  Node A (trigger) → Node B (chess_evaluate) → Node C (delegation_gate)
                                                  │
                                           ┌──────┴──────┐
                                           ▼              ▼
                                    Node D (tactician) Node E (critic)
                                           │              │
                                           └──────┬──────┘
                                                  ▼
                                           Node F (integration)
                                                  │
                                                  ▼
                                           Node G (crystallize)
```

Each workflow node IS a strategic node with:
- Its own Q-values (W32-39) for routing decisions
- Its own gate state (FLOW/HOLD/BLOCK)
- Its own thinking_style modulation

**Self-orchestration** = the workflow itself learns which routing works:
- Node C's Q-values encode: "for sharp tactical positions, always route to Tactician first"
- This is learned through L10 crystallization of routing outcomes
- n8n-rs doesn't need explicit routing rules — it resonates with crystallized patterns

---

## 6. Neo4j-rs Role (Reduced Scope)

Neo4j-rs becomes a **query language adapter**, not a separate data store:

### 6.1 What neo4j-rs KEEPS
- Cypher parser + AST (valuable for human-written queries)
- Property graph model (useful as a VIEW over containers)
- StorageBackend trait (clean abstraction)
- aiwar.rs bridge concept definitions (reused as codebook seed data)

### 6.2 What neo4j-rs GAINS
- `LadybugBackend` implementation of `StorageBackend`
- Translates Cypher MATCH/WHERE/RETURN to BindSpace resonance queries
- Node properties → container content + metadata word reads
- Relationship traversal → inline edge walks (W16-31) + Bloom filter (W40-47)

### 6.3 What neo4j-rs DOES NOT need
- Its own execution engine (ladybug 2-stroke engine handles this)
- Separate transcoding layer (no transcode needed — same containers)
- NARS/ACT-R integration (handled by ladybug-rs L9 Validation)
- Bolt protocol (direct container access, no network protocol needed)

---

## 7. Implementation Subtasks

### PHASE 1: strategy.rs in ladybug-rs (Foundation)

| # | Task | File | Depends On | Estimated Size |
|---|------|------|------------|----------------|
| 1.1 | Expand W12-15 layer markers from 7 to 10 layers | `crates/ladybug-contract/src/meta.rs` | — | ~100 LOC |
| 1.2 | Add `StrategicNode` zero-copy view struct | `src/cognitive/strategy.rs` (new) | 1.1 | ~300 LOC |
| 1.3 | Implement thinking_style[10] ↔ FieldModulation bridge | `src/cognitive/strategy.rs` | 1.2 | ~150 LOC |
| 1.4 | Add TacticalCodebook with 10 chess + 10 AI War fingerprints | `src/cognitive/strategy.rs` | 1.2 | ~200 LOC |
| 1.5 | Wire SPO crystal for strategy triples | `src/cognitive/strategy.rs` | 1.2 | ~100 LOC |
| 1.6 | Implement self-modification protocol (recover_modulation → propose_style_update) | `src/cognitive/strategy.rs` | 1.3 | ~200 LOC |
| 1.7 | Tests for strategic node view, codebook, self-modification | `tests/strategy_tests.rs` | 1.2-1.6 | ~400 LOC |

### PHASE 2: crewai-rust Agency Bridge

| # | Task | File | Depends On | Estimated Size |
|---|------|------|------------|----------------|
| 2.1 | Add thinking_style field to AgentBlueprint struct | `src/meta_agents/types.rs` | — | ~30 LOC |
| 2.2 | Load thinking_style from module YAML into AgentBlueprint | `src/modules/loader.rs` | 2.1 | ~50 LOC |
| 2.3 | Wire chess savant blueprints to use YAML thinking_styles | `src/meta_agents/savants.rs` | 2.1, 2.2 | ~100 LOC |
| 2.4 | Add style_update callback to InnerThoughtHook | `src/persona/inner_loop.rs` | 2.1 | ~100 LOC |
| 2.5 | Connect PersonaProfile.self_modify to L10 crystallization feedback | `src/persona/inner_loop.rs` | 2.4, P1.6 | ~150 LOC |
| 2.6 | Expose `strategic_node_view()` from crewai-rust's ladybug bridge tools | `src/tools/chess/tools.rs` | P1.2, 2.1 | ~100 LOC |
| 2.7 | Tests: style update round-trip, inner loop with crystallization | `tests/` | 2.4-2.6 | ~300 LOC |

### PHASE 3: n8n-rs Self-Orchestration

| # | Task | File | Depends On | Estimated Size |
|---|------|------|------------|----------------|
| 3.1 | Map n8n workflow nodes to strategic node containers | TBD (n8n-rs workflow engine) | P1.2 | ~200 LOC |
| 3.2 | Wire Q-values (W32-39) for workflow routing decisions | TBD | 3.1 | ~150 LOC |
| 3.3 | Implement gate_state mapping (FLOW/HOLD/BLOCK → workflow state) | TBD | 3.1 | ~100 LOC |
| 3.4 | Add crystallization-based learning for workflow routing | TBD | 3.2, P1.6 | ~200 LOC |
| 3.5 | Tests: workflow self-optimization via crystallization | TBD | 3.1-3.4 | ~300 LOC |

### PHASE 4: neo4j-rs LadybugBackend

| # | Task | File | Depends On | Estimated Size |
|---|------|------|------------|----------------|
| 4.1 | Implement LadybugBackend for StorageBackend trait | `src/storage/ladybug.rs` (new) | P1.2 | ~400 LOC |
| 4.2 | Map Cypher MATCH → BindSpace resonance queries | `src/storage/ladybug.rs` | 4.1 | ~300 LOC |
| 4.3 | Map Cypher RETURN → container metadata word reads | `src/storage/ladybug.rs` | 4.1 | ~200 LOC |
| 4.4 | Map Cypher relationship traversal → inline edge walks | `src/storage/ladybug.rs` | 4.1 | ~200 LOC |
| 4.5 | Migrate aiwar.rs bridge concepts to TacticalCodebook seeds | `src/aiwar.rs` | P1.4 | ~100 LOC |
| 4.6 | Tests: Cypher queries over BindSpace containers | `tests/` | 4.1-4.4 | ~400 LOC |

---

## 8. Thinking Style Vectors for Reference

These are the 6 chess module YAMLs already committed to crewai-rust:

```
                 recog reson appr  rout  exec  deleg conti integ valid cryst
Strategist:     [0.85, 0.90, 0.85, 0.80, 0.60, 0.95, 0.70, 0.95, 0.80, 0.90]
Tactician:      [0.95, 0.40, 0.70, 0.50, 0.95, 0.10, 0.80, 0.40, 0.95, 0.60]
Endgame:        [0.90, 0.85, 0.90, 0.60, 0.80, 0.10, 0.50, 0.60, 0.85, 0.95]
Psychologist:   [0.60, 0.95, 0.90, 0.85, 0.40, 0.30, 0.80, 0.85, 0.60, 0.70]
Inner Critic:   [0.90, 0.50, 0.95, 0.70, 0.70, 0.10, 0.95, 0.50, 0.95, 0.60]
Advocatus D.:   [0.80, 0.80, 0.85, 0.90, 0.85, 0.20, 0.90, 0.80, 0.80, 0.75]
```

Each vector IS the agent's cognitive fingerprint. Strategic behavior EMERGES
from the 2-stroke engine firing layers at these thresholds.

### 8.1 Existing Module YAMLs (non-chess, for reference)

```
                 recog reson appr  rout  exec  deleg conti integ valid cryst
Coding Agent:   [0.70, 0.80, 0.60, 0.50, 0.70, 0.75, 0.90, 0.80, 0.70, 0.85]
SOC Analyst:    [0.90, 0.30, 0.80, 0.50, 0.70, 0.95, 0.60, 0.85, 0.90, 0.75]
O365 Admin:     [0.70, 0.20, 0.95, 0.30, 0.60, 0.80, 0.40, 0.80, 0.85, 0.65]
Exchange Migr:  [0.70, 0.20, 0.95, 0.30, 0.50, 0.90, 0.40, 0.85, 0.90, 0.70]
Minecraft Orch: [0.40, 0.95, 0.50, 0.70, 0.60, 0.50, 0.90, 0.60, 0.50, 0.70]
```

---

## 9. Satisfaction Gate as Emergent Focus

The satisfaction gate acts as Maslow's hierarchy for cognition:

```
Layer minimums: [0.3, 0.3, 0.4, 0.4, 0.5, 0.5, 0.5, 0.6, 0.7, 0.8]
```

If L1 Recognition is unsatisfied (didn't find pattern), L9 Validation gets a
HIGHER effective threshold (harder to validate something you didn't recognize).

Combined with per-agent thinking_style, this creates implicit attention:

- **Analytical agent** (high thresholds across the board) → narrow focus, few layers fire
- **Creative agent** (low thresholds) → wide awareness, many layers fire simultaneously
- **Critic agent** (high contingency + validation, low delegation) → obsessive refutation

The satisfaction gate + thinking_style = emergent strategic personality.
No if/else dispatch needed. No strategy rules. Behavior emerges from the
interaction of threshold modulation with the 2-stroke resonance cycle.

---

## 10. Key Files Across Repos

### ladybug-rs (substrate)
- `src/cognitive/layer_stack.rs` — 10-layer stack definition (~750 LOC)
- `src/cognitive/two_stroke.rs` — 2-stroke engine (RESONATE → ACT) (~500 LOC)
- `src/cognitive/satisfaction_gate.rs` — Maslow for layers (~250 LOC)
- `src/cognitive/style.rs` — 12 thinking styles + FieldModulation (~250 LOC)
- `src/cognitive/cognitive_kernel.rs` — Layer ↔ BindSpace bridge (~500 LOC)
- `src/storage/bind_space.rs` — 8+8 addressing (surface/fluid/nodes) (~600 LOC)
- `crates/ladybug-contract/src/meta.rs` — Container metadata schema (128 words)
- `crates/ladybug-contract/src/codebook.rs` — 4096-op CAM codebook
- `crates/ladybug-contract/src/layers.rs` — Layer type IDs + backward compat
- `src/extensions/spo/spo.rs` — 5×5×5 SPO crystal

### crewai-rust (agency)
- `src/meta_agents/savants.rs` — 6 chess specialists (programmatic)
- `src/meta_agents/types.rs` — AgentBlueprint, SkillDescriptor, SavantDomain
- `src/modules/module_def.rs` — Module YAML schema (thinking_style, persona, gates)
- `src/modules/loader.rs` — YAML → ModuleInstance loader
- `src/persona/inner_loop.rs` — InnerThoughtHook + AgentState
- `src/persona/thinking_style.rs` — 36 ThinkingStyle enum (persona-level, complements 10-axis)
- `src/persona/composite.rs` — CompositeStyle blending
- `src/tools/chess/tools.rs` — Chess tool implementations (stonksfish, ladybug, neo4j)
- `modules/chess_*.yaml` — 6 chess module YAMLs with thinking_style vectors

### neo4j-rs (query layer)
- `src/aiwar.rs` — AI War plugin (chess-AIWar bridge concepts)
- `src/storage/mod.rs` — StorageBackend trait (needs LadybugBackend)
- `src/model/` — Node, Relationship, Value, PropertyMap DTOs

### n8n-rs (orchestration)
- Workflow engine (maps to strategic nodes via Q-values + gate state)

### stonksfish (evaluation)
- UCI engine providing eval_cp, game phase, legal moves

---

## 11. Design Principles

1. **No transcode** — The container IS the node. Views, not copies.
2. **No strategy rules** — Strategy emerges from thinking_style × 2-stroke × satisfaction.
3. **No separate reasoning module** — L1-L10 ARE the reasoning.
4. **No separate agency module** — L6 Delegation IS the agency bridge.
5. **No separate learning module** — L10 Crystallization IS the learning.
6. **No separate self-modification** — Modulation recovery from crystallized FP IS self-actualization.
7. **Hamming distance IS similarity** — No embedding model, no cosine distance, no transcoding.
8. **XOR binding IS association** — Role bindings, sequence encoding, cross-domain mapping.
9. **Majority-vote bundling IS evidence merge** — L8 Integration uses native VSA operation.
10. **The fingerprint IS the knowledge** — No rule table, no weight matrix, no gradient.

---

## 12. Open Questions

1. **W12-15 packing**: Should we reduce per-layer bytes from 5→3 (fitting 10 layers in 256 bits)
   or expand to use W64-79 reserved space (keeping 5 bytes per layer)?
2. **Style convergence**: Should agents converge toward similar thinking_styles over time
   (via crystallization feedback), or should diversity be enforced?
3. **n8n-rs node representation**: Does n8n-rs currently have a node struct we can extend,
   or do we need to define the strategic node mapping from scratch?
4. **Arrow Flight interface**: Should crewai-rust talk to ladybug-rs via Arrow Flight
   (as documented in CREWAI_INTEGRATION.md), or direct Rust function calls?
5. **Neo4j-rs LadybugBackend priority**: Should this come before or after strategy.rs?
   (Recommendation: after — strategy.rs defines the contract that LadybugBackend implements)

---

*This document is the authoritative integration plan. All subtasks reference sections
by number (e.g., "Implement §4.2 StrategicNode API"). Session-safe: contains all
context needed to resume work from any point.*
