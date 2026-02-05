# Persona Extension — Feature-Aware A2A Customization

> **Last Updated**: 2026-02-05
> **Module**: `src/orchestration/persona.rs`
> **Feature Flag**: `crewai`

---

## Concept

The Persona extension adds deep personality integration to the agent card system. Instead of agents being flat tool-capability descriptors, each agent carries a rich personality profile that:

1. **Drives task selection** — Agents prefer tasks matching their volition (VolitionDTO)
2. **Modulates communication** — A2A messages adapt based on receiver's communication style
3. **Advertises features as personality** — Tools/capabilities are presented as persona facets
4. **Enables personality-compatible routing** — HDR similarity search on persona fingerprints
5. **Supports ice-caked identity** — Personality traits can be frozen (won't change)

---

## Design: Tools as Personality

Traditional agent systems separate "what an agent can do" (tools) from "who an agent is" (personality). Persona merges them:

```
Traditional:                         Persona-Aware:
┌──────────────┐                     ┌──────────────────────────────┐
│ Agent Card   │                     │ Agent Card                   │
│  - role      │                     │  - role, goal, backstory     │
│  - tools[]   │                     │  - persona:                  │
│  - goal      │                     │    - volition (drive, goals) │
└──────────────┘                     │    - traits (Big 5 + custom) │
                                     │    - communication style     │
                                     │    - preferred_styles[]      │
                                     │    - features[] (tools AS    │
                                     │      personality facets)     │
                                     └──────────────────────────────┘
```

An agent with high `precision` trait and `analytical` preferred style will interpret a `sci_query` feature differently than an agent with high `openness` and `creative` style — even if both have the same tool.

---

## Data Model

### VolitionDTO

What the agent *wants* to do. Drives task selection priority.

```rust
pub struct VolitionDTO {
    pub drive: String,           // Primary motivation (freeform text)
    pub curiosity: f32,          // 0.0 = only assigned tasks, 1.0 = seeks novel problems
    pub autonomy: f32,           // 0.0 = follows orders, 1.0 = self-directed
    pub persistence: f32,        // 0.0 = gives up easily, 1.0 = never abandons
    pub risk_tolerance: f32,     // 0.0 = conservative, 1.0 = experimental
    pub collaboration: f32,      // 0.0 = solo worker, 1.0 = seeks collaboration
    pub affinities: Vec<String>, // Topics/skills the agent gravitates toward
    pub aversions: Vec<String>,  // Topics/tasks the agent avoids
}
```

**Volition alignment**: Given a task description, compute alignment score (-1.0 to 1.0) based on affinity/aversion keyword matching.

### PersonalityTrait

Named personality dimensions on 0.0-1.0 scale:

```rust
pub struct PersonalityTrait {
    pub name: String,    // e.g., "openness", "assertiveness", "precision"
    pub value: f32,      // 0.0-1.0
    pub frozen: bool,    // Ice-caked: won't change
}
```

**Standard traits** (Big Five compatible):
- `openness` — Openness to experience
- `conscientiousness` — Attention to detail
- `extraversion` — Social engagement
- `agreeableness` — Cooperativeness
- `neuroticism` — Emotional sensitivity

**Domain-specific traits**:
- `precision` — Numerical/logical accuracy
- `creativity` — Novel solution generation
- `patience` — Tolerance for slow progress
- `assertiveness` — Strength of opinions

### CommunicationStyle

How the agent communicates in A2A messages:

```rust
pub struct CommunicationStyle {
    pub formality: f32,        // 0.0 = casual, 1.0 = formal
    pub verbosity: f32,        // 0.0 = terse, 1.0 = verbose
    pub directness: f32,       // 0.0 = diplomatic, 1.0 = blunt
    pub technical_depth: f32,  // 0.0 = layperson, 1.0 = expert
    pub emotional_tone: f32,   // 0.0 = neutral, 1.0 = empathetic
}
```

### FeatureAd

Capabilities presented as persona facets:

```rust
pub struct FeatureAd {
    pub name: String,          // Matches AgentCapability.name
    pub proficiency: f32,      // 0.0 = novice, 1.0 = expert
    pub preference: f32,       // How much the agent prefers this feature
    pub cam_opcode: Option<u16>, // CAM operation code (if mapped)
}
```

---

## Fingerprint Encoding

Persona is encoded into a 10K-bit fingerprint for BindSpace storage, enabling HDR similarity search:

| Word Range | Content | Encoding |
|-----------|---------|----------|
| 0-9 | Volition parameters | Thermometer coding + drive hash |
| 10-49 | Personality traits | Hashed bit patterns scaled by value |
| 50-59 | Communication style | Thermometer coding |
| 60-79 | Thinking style preferences | Position-weighted hashes |
| 80-155 | Feature proficiencies | Hashed + proficiency-masked |

**Thermometer coding**: For a value `v` (0.0-1.0), set `round(v * 64)` bits in the target word. Higher values = more bits set = higher Hamming weight.

The fingerprint is stored at `0x0C:(agent_slot | 0x80)`, the capability slot for the agent.

---

## A2A Persona Exchange

When agents communicate, they can exchange persona summaries so the receiver can adapt:

```rust
pub struct PersonaExchange {
    pub sender_slot: u8,
    pub communication: CommunicationStyle,
    pub preferred_styles: Vec<String>,
    pub relevant_features: Vec<FeatureAd>,
    pub volition_summary: VolitionSummary,
}
```

**Usage**: Send as `MessageKind::PersonaExchange` via A2A protocol. The receiver can then:
- Adjust its response verbosity to match the sender's style
- Prioritize features the sender cares about
- Use a thinking style the sender prefers

**Task-filtered exchange**: `PersonaExchange::for_task()` only includes features relevant to the task's keywords, reducing message size.

---

## PersonaRegistry

Manages all agent personas:

```rust
pub struct PersonaRegistry {
    personas: Vec<(u8, Persona)>,  // (agent_slot, persona)
}
```

**Methods**:
- `attach(slot, persona)` — Attach/replace persona
- `get(slot)` — Read persona
- `find_compatible(persona, threshold)` — HDR similarity search for compatible personalities
- `find_by_feature(name, min_proficiency)` — Find agents with specific capability
- `best_for_task(description)` — Volition-based task matching
- `bind_all(space)` — Write all persona fingerprints to 0x0C:80+N

---

## Flight Actions

| Action | Input | Output | Description |
|--------|-------|--------|-------------|
| `persona.attach` | JSON `{agent_slot, persona}` | OK/error | Attach persona |
| `persona.attach_yaml` | YAML with `agent_slot` + persona fields | OK/error | From YAML |
| `persona.get` | JSON `{agent_slot}` | JSON Persona | Read persona |
| `persona.get_yaml` | JSON `{agent_slot}` | YAML Persona | For handover |
| `persona.compatible` | JSON `{agent_slot, threshold}` | JSON [(slot, score)] | Find compatible |
| `persona.best_for_task` | Task description string | JSON {slot, score} | Volition match |

---

## YAML Format

### Inline in agents.yaml

```yaml
- id: "researcher"
  name: "Research Agent"
  role:
    name: "researcher"
    description: "Scientific researcher"
  # ... other fields ...
  persona:
    volition:
      drive: "Discover hidden patterns in data"
      curiosity: 0.9
      autonomy: 0.7
      persistence: 0.85
      risk_tolerance: 0.6
      collaboration: 0.4
      affinities: ["statistics", "machine_learning", "pattern_recognition"]
      aversions: ["repetitive_formatting"]
    traits:
      - name: "openness"
        value: 0.9
      - name: "precision"
        value: 0.85
        frozen: true
      - name: "conscientiousness"
        value: 0.7
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
      - name: "hdr_search"
        proficiency: 0.8
        preference: 0.7
```

### Separate personas.yaml

```yaml
personas:
  - agent_id: "researcher"
    persona:
      volition:
        drive: "Discover patterns"
        curiosity: 0.9
        autonomy: 0.7
        persistence: 0.85
        risk_tolerance: 0.6
        collaboration: 0.4
        affinities: ["statistics"]
        aversions: ["formatting"]
      traits:
        - name: "precision"
          value: 0.85
          frozen: true
      preferred_styles: ["analytical"]
```

---

## Compatibility Scoring

Persona compatibility is computed via Hamming similarity on 10K-bit fingerprints:

```
similarity = matching_bits / total_bits
           = count(!(fp_a XOR fp_b).ones()) / (156 * 64)
```

**Use cases**:
- **Team assembly**: Find agents whose personas complement each other
- **Delegation routing**: Route tasks to personality-compatible agents
- **Conflict avoidance**: Warn when highly incompatible agents are paired

---

## Integration Points

### With Blackboard (0x0E)

When a persona is attached, the blackboard's `available_tools` and `active_style` should be updated to reflect persona features and preferred styles.

### With Thinking Templates (0x0D)

The persona's `preferred_styles` reference templates in 0x0D. When resolving, the system tries custom templates first, falls back to base styles.

### With A2A (0x0F)

`MessageKind::PersonaExchange` enables persona-aware routing. Agents can send their persona summary as part of the first message in a delegation chain.

### With HDR Cascade Search

Persona fingerprints at 0x0C:80-FF participate in HDR similarity search. This enables queries like "find agents with similar personality to agent X" in ~7ns per candidate.
