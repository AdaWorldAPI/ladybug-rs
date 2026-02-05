# crewAI Integration — Orchestration Layer for ladybug-rs

> **Last Updated**: 2026-02-05
> **Feature Flag**: `crewai` (depends on `flight`)
> **Branch**: claude/code-review-X0tu2
> **Status**: Implemented, 18+ tests passing

---

## Overview

The crewAI integration turns ladybug-rs into a cognitive substrate for [crewAI](https://github.com/AdaWorldAPI/crewAI) multi-agent orchestration. crewAI (Python) handles crew coordination; ladybug-rs (Rust) provides:

- **Ice-caked blackboard awareness** per agent
- **10K-bit fingerprint encoding** of agent identity, knowledge, and persona
- **Zero-copy Arrow Flight transport** between Python and Rust
- **A2A (agent-to-agent) messaging** via XOR-composed BindSpace channels
- **Thinking style modulation** from YAML templates
- **Persona-driven task selection** via volition alignment

**Design principle**: ladybug-rs stays standalone. If crewAI is not connected, these endpoints simply return empty results. All orchestration code is behind `#[cfg(feature = "crewai")]`.

---

## Architecture

```
crewAI (Python)                          Arrow Flight (gRPC)          ladybug-rs (Rust)
─────────────────                        ──────────────────           ─────────────────

agents.yaml ──────► DoAction("crew.register_agent") ──────► AgentRegistry (0x0C)
tasks.yaml  ──────► DoAction("crew.submit_task")    ──────► TaskQueue → Blackboard
styles.yaml ──────► DoAction("crew.register_style") ──────► ThinkingTemplateRegistry (0x0D)
persona.yaml ─────► DoAction("persona.attach")      ──────► PersonaRegistry → 0x0C:80+N

Agent A ──────────► DoAction("a2a.send")             ──────► A2AChannel (0x0F)
Agent B ──────────► DoAction("a2a.receive")           ◄───── Pending messages

DoGet("agents")    ◄──────── zero-copy agent list
DoGet("styles")    ◄──────── thinking template fingerprints
DoGet("blackboards") ◄────── agent state snapshots
DoGet("orchestration") ◄──── all 0x0C-0x0F prefixes
```

---

## Prefix Allocation (0x0C-0x0F)

| Prefix | Name | Slot Layout | Capacity |
|--------|------|-------------|----------|
| **0x0C** | Agent Registry | 0x00-0x7F: agent cards; 0x80-0xFF: persona fingerprints | 128 agents |
| **0x0D** | Thinking Styles | 0x00-0x0B: 12 base styles; 0x0C-0xFF: custom variants | 256 templates |
| **0x0E** | Blackboard | 0x00-0xFF: per-agent state (mirrors agent slot) | 256 boards |
| **0x0F** | A2A Routing | 0x00-0xFF: message channels (sender:receiver hash) | 256 channels |

These prefixes were previously reserved (0x0C-0x0F) in the surface zone. Constants defined in `bind_space.rs`:

```rust
pub const PREFIX_AGENTS: u8 = 0x0C;
pub const PREFIX_THINKING: u8 = 0x0D;
pub const PREFIX_BLACKBOARD: u8 = 0x0E;
pub const PREFIX_A2A: u8 = 0x0F;
```

---

## Modules

### `src/orchestration/agent_card.rs`

**Structs**: `AgentCard`, `AgentRole`, `AgentGoal`, `AgentCapability`, `AgentRegistry`

Agent cards are crewAI `agents.yaml` compatible. Each card stores:
- Identity (id, name, role, goal, backstory)
- Thinking style reference (resolved to one of 12 `ThinkingStyle` variants)
- Capabilities (tools with optional CAM opcodes and sci/v1 validation flags)
- Persona (optional deep personality profile)

**Fingerprint**: `identity_fingerprint()` hashes `role:description:objective:backstory` via SHA256, expanded to 10K bits across 156 u64 words.

**Registry**: Manages up to 128 agents. Auto-assigns slots 0x00-0x7F. Capability/persona fingerprints go to 0x80-0xFF (mirrored by `slot | 0x80`).

### `src/orchestration/thinking_template.rs`

**Structs**: `ThinkingTemplate`, `StyleOverride`, `ThinkingTemplateRegistry`

Bridges crewAI YAML templates to ladybug-rs's 12 thinking styles and their `FieldModulation` parameters (resonance_threshold, fan_out, depth_bias, breadth_bias, noise_tolerance, speed_bias, exploration).

**12 Base Styles** (auto-seeded at slots 0x00-0x0B):
Analytical, Convergent, Systematic, Creative, Divergent, Exploratory, Focused, Diffuse, Peripheral, Intuitive, Deliberate, Metacognitive

**Custom Templates**: Slots 0x0C-0xFF. Define a `base_style` + optional parameter overrides.

**Fingerprint**: `modulation_fingerprint()` uses thermometer coding to encode 7 modulation parameters, enabling Hamming-based similarity search between thinking styles.

### `src/orchestration/blackboard_agent.rs`

**Structs**: `AgentBlackboard`, `AgentAwareness`, `TaskRecord`, `BlackboardRegistry`

Per-agent persistent state at prefix 0x0E, mirroring the agent's slot in 0x0C:

- `AgentAwareness`: active_style, coherence (0.0-1.0), progress (0.0-1.0), ice_caked decisions, active_goals, available_tools, resonance_hits, pending_messages
- `TaskRecord`: completed task history (last 50)
- `knowledge_addrs`: addresses the agent has learned
- `cycle`: session cycle counter

**Ice-caking**: `ice_cake(decision)` freezes a commitment. Ice-caked decisions persist across task executions and inform future reasoning.

**Fingerprint**: `state_fingerprint()` combines agent identity, style, coherence, cycle, and knowledge addresses via SHA256 + Fibonacci hashing.

### `src/orchestration/a2a.rs`

**Structs**: `A2AMessage`, `A2AChannel`, `A2AProtocol`
**Enums**: `MessageKind`, `DeliveryStatus`

**MessageKind variants**: Delegate, Result, Status, Knowledge, Sync, Query, Response, PersonaExchange

**Channel addressing**: `compute_channel(sender, receiver)` uses XOR + rotation to map sender:receiver pairs to 0x0F:XX slots. Channels are asymmetric (A->B != B->A).

**XOR composition**: Messages are encoded as fingerprints and XOR-composed into the channel's BindSpace slot, allowing multiple messages to be stacked.

### `src/orchestration/crew_bridge.rs`

**Structs**: `CrewBridge`, `CrewTask`, `CrewDispatch`, `DispatchResult`, `BridgeStatus`
**Enum**: `TaskStatus` (Queued, Assigned, InProgress, Completed, Failed, Delegated)

The main coordinator holding all registries:
- `AgentRegistry` (0x0C)
- `ThinkingTemplateRegistry` (0x0D)
- `BlackboardRegistry` (0x0E)
- `PersonaRegistry` (0x0C:80+N)
- `A2AProtocol` (0x0F)

**Task lifecycle**: submit -> auto-assign (if agent_id specified) -> update blackboard -> complete -> record in history

### `src/orchestration/persona.rs`

**Structs**: `Persona`, `VolitionDTO`, `PersonalityTrait`, `CommunicationStyle`, `FeatureAd`, `PersonaExchange`, `VolitionSummary`, `PersonaRegistry`

Deep personality encoding for feature-aware A2A customization. See `docs/PERSONA_EXTENSION.md` for full documentation.

---

## Flight Actions (DoAction Handlers)

All actions are in `src/flight/crew_actions.rs`.

### Agent Management

| Action | Input | Output | Description |
|--------|-------|--------|-------------|
| `crew.register_agent` | YAML (agents.yaml) | OK/error | Register 1+ agents |
| `agent.list` | `{}` | Arrow IPC RecordBatch | List all agents |
| `agent.blackboard` | `{agent_slot: u8}` | Arrow IPC RecordBatch | Agent awareness state |
| `agent.blackboard.yaml` | `{agent_slot: u8}` | YAML string | Blackboard for handover |

### Style Management

| Action | Input | Output | Description |
|--------|-------|--------|-------------|
| `crew.register_style` | YAML (templates) | OK/error | Register custom templates |
| `style.resolve` | template name string | Arrow IPC RecordBatch | Resolve modulation params |
| `style.list` | `{}` | Arrow IPC RecordBatch | All templates + params |

### Task Dispatch

| Action | Input | Output | Description |
|--------|-------|--------|-------------|
| `crew.submit_task` | JSON CrewTask | JSON DispatchResult | Submit single task |
| `crew.dispatch` | JSON CrewDispatch | Arrow IPC RecordBatch | Batch crew dispatch |
| `crew.complete_task` | `{task_id, outcome}` | JSON DispatchResult | Complete a task |

### Orchestration Control

| Action | Input | Output | Description |
|--------|-------|--------|-------------|
| `crew.status` | `{}` | Arrow IPC RecordBatch | Bridge status summary |
| `crew.bind` | `{}` | OK/error | Commit all state to BindSpace |

### A2A Messaging

| Action | Input | Output | Description |
|--------|-------|--------|-------------|
| `a2a.send` | JSON A2AMessage | OK/error | Send via XOR channel |
| `a2a.receive` | `{agent_slot: u8}` | JSON [A2AMessage] | Drain pending messages |

### Persona

| Action | Input | Output | Description |
|--------|-------|--------|-------------|
| `persona.attach` | `{agent_slot, persona}` | OK/error | Attach persona to agent |
| `persona.attach_yaml` | YAML with agent_slot + persona | OK/error | Attach from YAML |
| `persona.get` | `{agent_slot: u8}` | JSON Persona | Get persona |
| `persona.get_yaml` | `{agent_slot: u8}` | YAML Persona | Get persona as YAML |
| `persona.compatible` | `{agent_slot, threshold}` | JSON [(slot, score)] | Find compatible agents |
| `persona.best_for_task` | task description string | JSON {slot, score} | Volition-based task matching |

---

## DoGet Tickets

| Ticket | Prefix Range | Description |
|--------|--------------|-------------|
| `agents` | 0x0C | All agent fingerprints |
| `styles` | 0x0D | All thinking template fingerprints |
| `blackboards` | 0x0E | All blackboard state fingerprints |
| `a2a` | 0x0F | All A2A channel fingerprints |
| `orchestration` | 0x0C-0x0F | Complete orchestration state |

---

## YAML Formats

### agents.yaml (crewAI compatible)

```yaml
- id: "researcher"
  name: "Research Agent"
  role:
    name: "researcher"
    description: "Performs deep research on scientific topics"
  goal:
    objective: "Find and validate scientific claims"
    success_criteria:
      - "p < 0.05"
    constraints:
      - "Use sci/v1 validation"
  backstory: "Expert in statistical analysis"
  thinking_style: "analytical"
  capabilities:
    - name: "sci_query"
      description: "Query sci/v1 endpoints"
      cam_opcode: 96
      requires_validation: true
  allow_delegation: true
  memory: true
  verbose: false
  max_iter: 25
  persona:
    volition:
      drive: "Discover hidden patterns"
      curiosity: 0.9
      autonomy: 0.7
      persistence: 0.85
      risk_tolerance: 0.6
      collaboration: 0.4
      affinities: ["statistics", "patterns"]
      aversions: ["formatting"]
    traits:
      - name: "openness"
        value: 0.9
      - name: "precision"
        value: 0.85
        frozen: true
    communication:
      formality: 0.6
      verbosity: 0.4
      directness: 0.8
      technical_depth: 0.9
      emotional_tone: 0.2
    preferred_styles: ["analytical", "exploratory"]
    features:
      - name: "sci_query"
        proficiency: 0.95
        preference: 0.9
        cam_opcode: 96
```

### styles.yaml

```yaml
templates:
  - name: "deep_research"
    base_style: "analytical"
    description: "Deep statistical research"
    overrides:
      resonance_threshold: 0.95
      depth_bias: 1.0
      noise_tolerance: 0.02
  - name: "brainstorm"
    base_style: "creative"
    overrides:
      fan_out: 20
      exploration: 0.95
```

---

## Data Flow

### Registration Phase

```
1. Python: Read agents.yaml
2. Python: DoAction("crew.register_agent", yaml_bytes)
3. Rust:   Parse YAML -> Vec<AgentCard>
4. Rust:   For each card:
           a. Assign slot 0x0C:N (N = 0,1,2...)
           b. Create blackboard at 0x0E:N
           c. If persona present: attach to PersonaRegistry
5. Rust:   Return assigned addresses

6. Python: Read styles.yaml
7. Python: DoAction("crew.register_style", yaml_bytes)
8. Rust:   Assign slots 0x0D:0C+ (base styles at 0x00-0x0B)

9. Python: DoAction("crew.bind", "")
10. Rust:  Write all fingerprints to BindSpace (0x0C-0x0F)
```

### Execution Phase

```
1. Python: Crew.kickoff()
2. Python: DoAction("crew.dispatch", dispatch_json)
3. Rust:   For each task:
           a. If agent_id specified: auto-assign to slot
           b. Update blackboard: awareness.active_goals += task.description
           c. If thinking_style override: bb.set_thinking_style(template)
4. Rust:   Return DispatchResult[]

5. Python: Agent executes task
6. Python: DoAction("crew.complete_task", {task_id, outcome})
7. Rust:   Record in blackboard.task_history, increment cycle
```

### A2A Communication

```
1. Agent A: Construct A2AMessage (Delegate kind)
2. Python:  DoAction("a2a.send", message_json)
3. Rust:    Compute channel_slot = hash(sender, receiver)
4. Rust:    Encode message -> fingerprint
5. Rust:    XOR-compose into 0x0F:channel_slot
6. Rust:    Mark message as Delivered

7. Agent B: DoAction("a2a.receive", {agent_slot: B})
8. Rust:    Drain pending messages for B
```

---

## Feature Flags

```toml
# Cargo.toml
[features]
crewai = ["flight"]  # crewAI orchestration (A2A, agent cards, thinking templates, personas)
```

The `crewai` feature depends on `flight` (Arrow Flight gRPC). When disabled, all orchestration code is excluded from compilation. The standalone ladybug-rs is unaffected.

---

## Testing

```bash
# Run orchestration tests
cargo test --features "crewai" orchestration

# Run all tests with crewai
cargo test --features "simd,parallel,crewai"

# Check compilation only
cargo check --features "simd,parallel,crewai"
```

### Test Coverage

| Module | Tests | Status |
|--------|-------|--------|
| agent_card | 3 | Passing |
| thinking_template | 4 | Passing |
| a2a | 3 | Passing |
| blackboard_agent | 5 | Passing |
| crew_bridge | 3 | Passing |
| persona | 10 | Passing |

---

## Source Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/orchestration/mod.rs` | ~87 | Module exports + architecture diagram |
| `src/orchestration/agent_card.rs` | ~300 | AgentCard + AgentRegistry |
| `src/orchestration/thinking_template.rs` | ~307 | ThinkingTemplate + registry |
| `src/orchestration/a2a.rs` | ~263 | A2A protocol + XOR channels |
| `src/orchestration/blackboard_agent.rs` | ~287 | AgentBlackboard + awareness |
| `src/orchestration/crew_bridge.rs` | ~410 | CrewBridge coordinator |
| `src/orchestration/persona.rs` | ~430 | Persona + VolitionDTO + registry |
| `src/flight/crew_actions.rs` | ~490 | 20 DoAction handlers |
