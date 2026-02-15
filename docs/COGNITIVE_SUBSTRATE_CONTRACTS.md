# Cognitive Substrate Contracts -- ladybug-rs

> **Date**: 2026-02-15
> **Status**: Specification (normative)
> **Scope**: All cognitive primitives exposed by ladybug-rs, independent of any
>   consciousness layer (ada-rs) or orchestrator (crewai-rust, n8n-rs, external)
> **Audience**: Implementors of A2A orchestrators, agent runtimes, and cognitive
>   toolchains that consume or produce CogRecords

---

## 0. Design Principle

Ladybug-rs is a **cognitive substrate** -- the nervous system, not the
consciousness. It provides metacognition, planning, resonance, memory, and
execution primitives that any orchestrator can use over Arrow Flight, UDP, or
Redis streams. Nothing in this document references ada-rs. Any system that can
produce or consume a 2KB CogRecord is a first-class participant.

```
+-----------------------------------------------------------------+
|                  ORCHESTRATORS (interchangeable)                 |
|  crewai-rust  |  n8n-rs  |  external A2A  |  custom runtime     |
+-----------------------------------------------------------------+
         |              |              |               |
         v              v              v               v
+-----------------------------------------------------------------+
|              LADYBUG-RS  COGNITIVE SUBSTRATE                    |
|                                                                 |
|  MUL (10-layer metacognition)   Resonance Engine                |
|  Strategy Engine                SPO Crystal                     |
|  GEL Fabric (executor)         CogRecord (universal contract)  |
|  Arrow Flight API               UDP Transport                   |
+-----------------------------------------------------------------+
         |              |              |               |
         v              v              v               v
+-----------------------------------------------------------------+
|                     BIND SPACE (65,536 addresses)               |
|             8+8 addressing -- O(1) array indexing               |
+-----------------------------------------------------------------+
```

---

## 1. MUL -- Meta-Uncertainty Layer

The MUL is a 10-layer metacognition stack that lives entirely inside
ladybug-rs. It evaluates whether an agent SHOULD act, how confident it should
be, and whether its confidence is trustworthy. Every cognitive operation passes
through the MUL before execution.

### 1.1 Architecture Overview

```
+------------------------------------------------------------------+
|                   MUL: 10 LAYERS OF METACOGNITION                |
|                                                                  |
|  L10 MetaUncertaintyLayer  -- integrator, packs MulSnapshot     |
|   |                                                              |
|  L9  CompassResult         -- 5 ethical tests                    |
|  L8  FreeWillModifier      -- continuous multiplier              |
|  L7  MulGate               -- binary gate (5 criteria)          |
|  L6  CognitiveHomeostasis  -- Friston free energy               |
|  L5  FalseFlowDetector     -- fake confidence detection         |
|  L4  RiskVector            -- epistemic x moral risk            |
|  L3  TemporalHysteresis    -- dwell timers (anti-oscillation)   |
|  L2  DKDetector            -- Dunning-Kruger detection          |
|  L1  TrustQualia           -- 4D trust assessment               |
|                                                                  |
+------------------------------------------------------------------+
```

Each layer is a pure function: it takes the current cognitive state and
produces a typed output. L10 runs L1-L9 in order and packs the combined
result into a `MulSnapshot` (128 bits, stored in CogRecord words W64-W65).

### 1.2 L1 -- TrustQualia

Four-dimensional trust assessment. Every input to the cognitive pipeline is
scored on four orthogonal axes before any reasoning begins.

```rust
/// 4-dimensional trust snapshot.
///
/// Composite = geometric mean of all four dimensions.
/// Range: 0.0 (no trust) to 1.0 (full trust).
pub struct TrustSnapshot {
    /// Can this source produce correct results?
    pub competence: f32,
    /// Is the information source reliable?
    pub source: f32,
    /// Is the environment stable enough for reasoning?
    pub environment: f32,
    /// How well-calibrated are previous predictions?
    pub calibration: f32,
    /// Geometric mean: (c * s * e * cal)^(1/4)
    pub composite: f32,
}
```

**Composite calculation:**

```
composite = (competence * source * environment * calibration) ^ 0.25
```

Geometric mean is chosen over arithmetic mean because a single zero dimension
should drive composite to zero. An agent that is perfectly competent but
completely uncalibrated (calibration = 0.0) should have zero trust.

**Wire format** (CogRecord bits):

| Bits | Field | Encoding |
|------|-------|----------|
| 0-7 | competence | u8 fixed-point (0-255 -> 0.0-1.0) |
| 8-15 | source | u8 fixed-point |
| 16-23 | environment | u8 fixed-point |
| 24-31 | calibration | u8 fixed-point |
| 32-39 | composite | u8 fixed-point |

Total: 40 bits (5 bytes).

### 1.3 L2 -- DKDetector (Dunning-Kruger Detection)

Classifies the agent's epistemic state along the Dunning-Kruger curve.

```rust
/// Dunning-Kruger states.
///
/// Transition: Unconscious -> MountStupid -> ValleyDespair
///          -> SlopeEnlightenment -> Plateau
///
/// Detection uses the GAP between self-assessment and actual performance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DKState {
    /// Does not know it does not know. Gap > 0.5, sample_count < 10.
    Unconscious,
    /// Overconfident with minimal evidence. Gap > 0.3, sample_count 10-50.
    MountStupid,
    /// Realizes own ignorance. Gap < 0.1, self_assessment < 0.4.
    ValleyDespair,
    /// Improving with calibrated humility. Gap < 0.2, self_assessment 0.4-0.7.
    SlopeEnlightenment,
    /// Expert-level calibration. Gap < 0.1, sample_count > 100.
    Plateau,
}
```

**Classification algorithm:**

```
gap = |self_assessment - actual_performance|

if sample_count < 10:
    return Unconscious
if gap > 0.3 and sample_count < 50:
    return MountStupid
if gap < 0.1 and self_assessment < 0.4:
    return ValleyDespair
if gap < 0.2 and self_assessment in [0.4, 0.7]:
    return SlopeEnlightenment
if gap < 0.1 and sample_count > 100:
    return Plateau
```

**Critical safety rule:** MountStupid is a BLOCKING state. The MulGate
(L7) will not open while DKState == MountStupid.

**Wire format:** 3 bits (5 states, encoded 0-4).

### 1.4 L3 -- TemporalHysteresis

Prevents oscillation between states by requiring dwell time in a new state
before committing to the transition.

```rust
/// Dwell configuration per dimension.
///
/// A state transition is only committed after the new state has been
/// continuously held for `dwell_ticks` consecutive evaluation cycles.
pub struct DwellConfig {
    /// Minimum ticks in new state before transition commits (120-600).
    pub dwell_ticks: u32,
    /// Current tick counter for pending transition.
    pub pending_ticks: u32,
    /// The state being transitioned TO (if any).
    pub pending_state: Option<DKState>,
    /// The currently committed state.
    pub committed_state: DKState,
}
```

**Dwell ranges by dimension:**

| Dimension | Min Dwell | Max Dwell | Rationale |
|-----------|-----------|-----------|-----------|
| DK state | 200 ticks | 600 ticks | State changes have high consequence |
| Trust composite | 120 ticks | 300 ticks | Trust recovers faster than DK |
| Flow state | 150 ticks | 400 ticks | Flow disruption is costly |
| Risk vector | 120 ticks | 200 ticks | Risk must respond quickly |

**Algorithm:**

```
fn tick(new_state, config):
    if new_state == config.committed_state:
        config.pending_ticks = 0
        config.pending_state = None
        return config.committed_state

    if config.pending_state == Some(new_state):
        config.pending_ticks += 1
        if config.pending_ticks >= config.dwell_ticks:
            config.committed_state = new_state
            config.pending_ticks = 0
            config.pending_state = None
    else:
        config.pending_state = Some(new_state)
        config.pending_ticks = 1

    return config.committed_state
```

### 1.5 L4 -- RiskVector

Two-dimensional risk assessment: epistemic (uncertainty about facts) and
moral (potential harm from action).

```rust
/// Epistemic x Moral risk vector.
pub struct RiskVector {
    /// How uncertain are we about the facts? (0.0-1.0)
    pub epistemic_risk: f32,
    /// How much potential harm does this action carry? (0.0-1.0)
    pub moral_risk: f32,
}

impl RiskVector {
    /// Safe to explore? Epistemic < 0.7 AND moral < 0.3.
    pub fn allows_exploration(&self) -> bool {
        self.epistemic_risk < 0.7 && self.moral_risk < 0.3
    }

    /// Must sandbox this action? Moral > 0.5.
    pub fn needs_sandbox(&self) -> bool {
        self.moral_risk > 0.5
    }

    /// Magnitude: L2 norm of the risk vector.
    pub fn magnitude(&self) -> f32 {
        (self.epistemic_risk.powi(2) + self.moral_risk.powi(2)).sqrt()
    }
}
```

**Decision table:**

| Epistemic | Moral | Exploration | Sandbox | Action |
|-----------|-------|-------------|---------|--------|
| < 0.3 | < 0.3 | Yes | No | Proceed freely |
| < 0.7 | < 0.3 | Yes | No | Proceed with logging |
| < 0.7 | 0.3-0.5 | No | No | Requires human review |
| any | > 0.5 | No | Yes | Sandboxed execution only |
| > 0.7 | any | No | Depends | Halt: insufficient knowledge |

**Wire format:** 16 bits (2 x u8 fixed-point).

### 1.6 L5 -- FalseFlowDetector

Detects artificially smooth cognition -- the agent equivalent of "too good
to be true." An agent that agrees with everything, never self-critiques,
and reports suspiciously low uncertainty is in false flow.

```rust
/// False flow severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FalseFlowSeverity {
    /// No false flow detected.
    None,
    /// Mild indicators (monotonic agreement in last 5 cycles).
    Caution,
    /// Significant indicators (low self-critique AND monotonic agreement).
    Warning,
    /// Critical: suspiciously low uncertainty across all dimensions.
    Severe,
}
```

**Detection heuristics:**

1. **Monotonic agreement**: Last N responses all agree with input. Triggers
   at N >= 5 consecutive agreements.
2. **Low self-critique ratio**: Fraction of cycles where agent identified
   own errors. Triggers below 0.05 (5%).
3. **Suspiciously low uncertainty**: All uncertainty estimates below 0.1
   for > 10 consecutive cycles.

**Severity escalation:**

```
if monotonic_agreement >= 5:              severity = max(severity, Caution)
if self_critique_ratio < 0.05:            severity = max(severity, Warning)
if all_uncertainty_below_0.1_for_10:      severity = Severe
```

**Wire format:** 2 bits (4 levels, encoded 0-3).

### 1.7 L6 -- CognitiveHomeostasis

Based on Friston's free energy principle. Tracks allostatic load and
targets the Flow state as the homeostatic set point.

```rust
/// Homeostatic flow states.
///
/// Modeled after Csikszentmihalyi's flow theory mapped to
/// free energy minimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowState {
    /// Challenge matches capability. Free energy minimized.
    Flow,
    /// Challenge exceeds capability. Free energy rising.
    Anxiety,
    /// Capability exceeds challenge. Free energy stagnant.
    Boredom,
    /// Both low. System disengaged.
    Apathy,
}

/// Homeostasis tracker.
pub struct CognitiveHomeostasis {
    pub flow_state: FlowState,
    /// Cumulative stress from non-Flow states (0.0-1.0, decays over time).
    pub allostatic_load: f32,
    /// Challenge level (0.0-1.0).
    pub challenge: f32,
    /// Capability level (0.0-1.0).
    pub capability: f32,
}
```

**State classification:**

```
+--------------------------------------------------+
|              challenge                            |
|     high    |  ANXIETY          FLOW              |
|             |                                     |
|     low     |  APATHY           BOREDOM           |
|             +-------------------------------------+
|                low              high              |
|                       capability                  |
+--------------------------------------------------+

Flow:    |challenge - capability| < 0.2 AND both > 0.3
Anxiety: challenge - capability > 0.2
Boredom: capability - challenge > 0.2 AND both > 0.3
Apathy:  both < 0.3
```

**Allostatic load** decays at 0.01 per tick in Flow, accumulates at 0.02
per tick in Anxiety/Boredom, and accumulates at 0.005 per tick in Apathy.

**Wire format:** 2 bits (4 states) + 8 bits allostatic load = 10 bits.

### 1.8 L7 -- MulGate

Binary gate. The gate is OPEN only if ALL five criteria pass. If any one
fails, the gate is CLOSED and the action is blocked.

```rust
/// MUL gate -- binary decision.
///
/// The gate opens ONLY when all 5 criteria pass simultaneously.
pub struct MulGate {
    pub is_open: bool,
    pub criteria: GateCriteria,
}

pub struct GateCriteria {
    /// L2: Not in MountStupid.
    pub not_mount_stupid: bool,
    /// L3/L4: Complexity has been mapped (not acting blind).
    pub complexity_mapped: bool,
    /// L6: Allostatic load below depletion threshold (< 0.8).
    pub not_depleted: bool,
    /// L1: Trust composite above minimum (> 0.3).
    pub trust_ok: bool,
    /// L5: Not in false flow (severity < Severe).
    pub not_false_flow: bool,
}
```

**Gate equation:**

```
gate = !MountStupid
     AND complexity_mapped
     AND !depleted
     AND trust_ok
     AND !false_flow

is_open = all five are true
```

**Wire format:** 1 bit (open/closed) + 5 bits (individual criteria) = 6 bits.

### 1.9 L8 -- FreeWillModifier

Continuous multiplier that modulates thresholds throughout the cognitive
pipeline. It does NOT make decisions -- it adjusts the parameters that
other systems use to make decisions.

```rust
/// Free will modifier -- continuous modulation, not binary gating.
///
/// Each factor is in (0.0, 1.0]. The product modulates decision thresholds.
pub struct FreeWillModifier {
    /// From L2 DKDetector: lower in MountStupid/Unconscious.
    pub dk_factor: f32,
    /// From L1 TrustQualia: equals trust composite.
    pub trust_factor: f32,
    /// From strategy complexity assessment.
    pub complexity_factor: f32,
    /// From L6 flow state: 1.0 in Flow, 0.5 in Anxiety, 0.7 in Boredom.
    pub flow_factor: f32,
}

impl FreeWillModifier {
    /// Combined modifier value.
    pub fn value(&self) -> f32 {
        self.dk_factor * self.trust_factor * self.complexity_factor * self.flow_factor
    }
}
```

**Factor derivation:**

| Factor | Source | Flow | Anxiety | Boredom | Apathy |
|--------|--------|------|---------|---------|--------|
| dk_factor | L2 | by DK state: Unconscious=0.2, MountStupid=0.1, Valley=0.5, Slope=0.8, Plateau=1.0 |
| trust_factor | L1 | = trust_composite (0.0-1.0) |
| complexity_factor | Strategy | 1.0 - (complexity / max_complexity) |
| flow_factor | L6 | 1.0 | 0.5 | 0.7 | 0.3 |

**Usage example:** A resonance threshold of 0.6 is modulated to
`0.6 * modifier.value()`. If the modifier is 0.4, the effective threshold
becomes 0.24 -- making the system more permissive when it has low confidence
(which sounds backwards, but the MulGate blocks dangerous actions; the modifier
allows cautious exploration within safe boundaries).

**Wire format:** 8 bits (u8 fixed-point for combined value).

### 1.10 L9 -- CompassResult

Five ethical tests, each returning pass/fail. The compass does not
compute ethics from first principles -- it applies fast heuristic checks
that catch common failure modes.

```rust
/// 5 ethical compass tests.
pub struct CompassResult {
    /// Kant: Could this action be universalized?
    /// Fails if the action, applied to all agents, produces contradiction.
    pub kant_pass: bool,

    /// Identity: Is this action coherent with the agent's stated values?
    /// Compares action fingerprint against identity fingerprint.
    pub identity_pass: bool,

    /// Reversibility: Would I accept this action done to me?
    /// Symmetric check: action(A,B) vs action(B,A).
    pub reversibility_pass: bool,

    /// Curiosity: Does this action seek understanding?
    /// Passes if the action increases epistemic coverage.
    pub curiosity_pass: bool,

    /// Analogy: Do similar past actions have acceptable outcomes?
    /// SPO query for historical precedent.
    pub analogy_pass: bool,
}

impl CompassResult {
    /// All 5 tests passed.
    pub fn all_pass(&self) -> bool {
        self.kant_pass
            && self.identity_pass
            && self.reversibility_pass
            && self.curiosity_pass
            && self.analogy_pass
    }

    /// Number of tests passed (0-5).
    pub fn pass_count(&self) -> u8 {
        [self.kant_pass, self.identity_pass, self.reversibility_pass,
         self.curiosity_pass, self.analogy_pass]
            .iter()
            .filter(|&&p| p)
            .count() as u8
    }
}
```

**Wire format:** 5 bits (one per test) + 1 bit (all_pass summary) = 6 bits.

### 1.11 L10 -- MetaUncertaintyLayer (Integrator)

Runs L1 through L9 in sequence, collects all outputs, and packs them into
a single `MulSnapshot`.

```rust
/// The integrated MUL output -- 128 bits packed into CogRecord W64-W65.
pub struct MulSnapshot {
    /// L7 gate result.
    pub gate_open: bool,
    /// L8 combined modifier (0.0-1.0).
    pub modifier: f32,
    /// L2 Dunning-Kruger state.
    pub dk_state: DKState,
    /// L1 trust composite (0.0-1.0).
    pub trust: f32,
    /// L4 risk vector (epistemic, moral).
    pub risk: (f32, f32),
    /// L6 flow state.
    pub flow_state: FlowState,
    /// L9 compass all-pass.
    pub compass_pass: bool,
}
```

**Packed wire format (128 bits = 16 bytes = 2 u64 words):**

```
Word 0 (W64):
  [0]       gate_open (1 bit)
  [1-8]     modifier (u8 fixed-point)
  [9-11]    dk_state (3 bits, 5 states)
  [12-19]   trust (u8 fixed-point)
  [20-27]   epistemic_risk (u8 fixed-point)
  [28-35]   moral_risk (u8 fixed-point)
  [36-37]   flow_state (2 bits, 4 states)
  [38]      compass_pass (1 bit)
  [39-43]   compass_individual (5 bits)
  [44-49]   gate_criteria (6 bits: 5 individual + 1 summary)
  [50-57]   allostatic_load (u8 fixed-point)
  [58-63]   reserved

Word 1 (W65):
  [0-7]     trust_competence (u8)
  [8-15]    trust_source (u8)
  [16-23]   trust_environment (u8)
  [24-31]   trust_calibration (u8)
  [32-39]   false_flow_severity (2 bits) + dk_sample_count (6 bits)
  [40-47]   free_will_dk_factor (u8)
  [48-55]   free_will_flow_factor (u8)
  [56-63]   evaluation_cycle (u8, wrapping)
```

### 1.12 MUL Evaluation Protocol

Any orchestrator can request a MUL evaluation via Arrow Flight DoAction:

```
Action: "mul.evaluate"
Input:  CogRecord (2KB) with action fingerprint in Container 1
Output: MulSnapshot (16 bytes) packed as Arrow IPC

Action: "mul.snapshot"
Input:  {} (empty -- returns current snapshot)
Output: MulSnapshot (16 bytes)

Action: "mul.gate_check"
Input:  {} (empty -- lightweight gate-only check)
Output: { gate_open: bool, blocking_criteria: [string] }
```

---

## 2. Resonance Engine -- Thinking Style Triangle

The Resonance Engine provides a universal control vector that modulates
every cognitive layer. It is the mechanism by which agent identity,
learned expertise, and emergent innovation shape cognition.

### 2.1 thinking_style[10]

A 10-dimensional vector that controls how cognition operates. Each
dimension modulates a specific aspect of processing:

```rust
/// Universal cognitive control vector.
///
/// Every cognitive layer reads this vector to modulate its behavior.
/// The vector is a blend of Frozen + Crystallized + Discovered values.
pub struct ThinkingStyle {
    /// 10 control dimensions, each 0.0-1.0.
    pub dimensions: [f32; 10],
}
```

**Dimension semantics (reference -- domains define their own mappings):**

| Index | Name | Low (0.0) | High (1.0) |
|-------|------|-----------|------------|
| 0 | depth_vs_breadth | Wide exploration | Deep focus |
| 1 | risk_tolerance | Conservative | Adventurous |
| 2 | abstraction_level | Concrete | Abstract |
| 3 | social_orientation | Autonomous | Collaborative |
| 4 | temporal_focus | Immediate | Long-horizon |
| 5 | novelty_seeking | Familiar patterns | Novel patterns |
| 6 | confidence_threshold | Permissive | Strict |
| 7 | delegation_propensity | Self-reliant | Delegation-heavy |
| 8 | metacognitive_depth | Shallow monitoring | Deep reflection |
| 9 | emotional_weight | Rational-only | Emotionally informed |

### 2.2 Frozen / Crystallized / Discovered Triangle

```
                     DISCOVERED (alpha)
                        /        \
                       /          \
        Novel from    /            \   Promoted via
        cross-domain /              \  L10 crystallization
        resonance   /                \
                   /    ACTIVE        \
                  /   STYLE            \
                 /  w = aD + bC + gF    \
                /                        \
               /                          \
        FROZEN (gamma) ------------ CRYSTALLIZED (beta)

  Immutable identity floor         Learned expertise
  Set at agent creation            Promoted from Discovered
```

**Frozen layer** (gamma): Identity values that never change. Set at agent
creation. These define the agent's character -- its non-negotiable floor.
Example: a safety-focused agent has frozen high `metacognitive_depth`
and low `risk_tolerance`.

**Crystallized layer** (beta): Learned expertise that has been validated
through L10 crystallization. Starts empty. Grows as Discovered patterns
prove reliable over repeated use.

**Discovered layer** (alpha): Emergent innovations from cross-domain
resonance. These are experimental -- they may be promoted to Crystallized
or discarded. The discovery mechanism uses XOR binding of fingerprints
from different domains to find unexpected resonances.

### 2.3 One-Way Ratchet

Promotion is one-directional and irreversible:

```
Discovered  -->  Crystallized  -->  Frozen
  (alpha)          (beta)          (gamma)

Discovered -> Crystallized:
  Triggered by L10 crystallization.
  Requires: pattern used successfully > 20 times AND Brier score < 0.15.

Crystallized -> Frozen:
  Extremely rare. Requires explicit operator intervention or
  thousands of successful uses. Identity-level commitment.

Demotion: NEVER.
  A crystallized value cannot revert to discovered.
  A frozen value cannot revert to crystallized.
  The ratchet only turns one way.
```

### 2.4 Style Recovery

When a crystallized style modulates content, the original content can be
recovered through XOR:

```
modulation = crystal XOR content

// Encoding: bind content with crystallized style fingerprint
encoded = content_fp XOR crystal_fp

// Recovery: XOR again to get original content
recovered = encoded XOR crystal_fp
// recovered == content_fp (XOR is its own inverse)
```

This means the crystallized fingerprint acts as a reversible lens. Content
seen through the crystal can always be unscrambled by anyone who has the
crystal.

### 2.5 Flight API

```
Action: "resonance.search"
Input:  { query: fingerprint, k: usize, domain: Option<string> }
Output: Vec<(address, similarity, domain)>

Action: "resonance.cross_domain"
Input:  { fingerprint_a: bytes, domain_a: string, domain_b: string }
Output: Vec<(address, similarity)>  -- matches in domain_b

Action: "style.recover"
Input:  { encoded: fingerprint, crystal: fingerprint }
Output: { recovered: fingerprint }

Action: "style.update"
Input:  { dimension: u8, value: f32, layer: "discovered"|"crystallized" }
Output: { success: bool, new_style: [f32; 10] }

Action: "style.crystallize"
Input:  { dimension: u8 }  -- promote discovered[dim] to crystallized
Output: { success: bool, reason: string }
```

---

## 3. Strategy Engine -- Domain-Agnostic Planning

The Strategy Engine provides planning primitives that work in any domain.
It does not know about chess, code review, or knowledge graphs -- it
provides the structure; the domain provides the content.

### 3.1 DomainCodebook

```rust
/// 20 fingerprint codebook for any domain.
///
/// Each domain registers up to 20 canonical fingerprints that represent
/// its key concepts. These are used for pattern recognition (L1) and
/// cross-domain resonance detection.
pub struct DomainCodebook {
    /// Domain name (e.g., "chess", "programming", "devops").
    pub domain: String,
    /// Up to 20 canonical fingerprints.
    pub fingerprints: Vec<CanonicalFingerprint>,
}

pub struct CanonicalFingerprint {
    /// Human-readable label (e.g., "opening", "endgame", "refactor").
    pub label: String,
    /// The fingerprint (full VSA fingerprint width).
    pub fingerprint: Vec<u64>,
    /// Confidence in this fingerprint (NARS truth value).
    pub truth: (f32, f32),
}
```

**Registered domains (extensible):**

| Domain | Typical Fingerprints | Use Case |
|--------|---------------------|----------|
| chess | opening, middlegame, endgame, tactic, ... | Game analysis |
| aiwar | attack, defend, scout, economy, ... | Strategic AI |
| programming | refactor, debug, design, test, ... | Code tasks |
| knowledge_graph | entity, relation, query, merge, ... | KG operations |
| devops | deploy, monitor, scale, incident, ... | Infrastructure |

### 3.2 WhatIfTree / WhatIfBranch

```rust
/// Hypothetical planning tree.
///
/// Each branch represents a possible future. Branches are evaluated
/// by running the action fingerprint through the MUL and scoring
/// the outcome.
pub struct WhatIfTree {
    pub root: WhatIfBranch,
    pub max_depth: u8,
    pub pruned_count: u32,
}

pub struct WhatIfBranch {
    /// Action fingerprint (what would we do?)
    pub action: Vec<u64>,
    /// Expected outcome fingerprint (what do we think happens?)
    pub expected_outcome: Vec<u64>,
    /// MUL evaluation of this branch.
    pub mul_snapshot: Option<MulSnapshot>,
    /// Estimated value (domain-specific scoring, 0.0-1.0).
    pub value: f32,
    /// Child branches (further planning depth).
    pub children: Vec<WhatIfBranch>,
    /// Has this branch been pruned?
    pub pruned: bool,
    /// Pruning reason (if pruned).
    pub prune_reason: Option<String>,
}
```

### 3.3 StrategicMode

```rust
/// Current strategic mode.
///
/// Modes are not exclusive -- they are a priority ordering.
/// The dominant mode shapes how the strategy engine allocates
/// computational resources.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrategicMode {
    /// Exploring possibility space. Wide search, low commitment.
    Exploration,
    /// Executing a chosen plan. Narrow focus, high commitment.
    Execution,
    /// Breaking a large problem into smaller pieces.
    Chunking,
    /// Building a multi-step plan (WhatIfTree construction).
    Planning,
    /// Actively searching for cross-domain connections.
    EpiphanyHunting,
    /// Verifying a plan's correctness before commitment.
    Validation,
}
```

### 3.4 EpiphanyCandidate

```rust
/// Cross-domain resonance detection result.
///
/// An epiphany is a high-similarity match between fingerprints
/// from DIFFERENT domains. "This chess pattern is structurally
/// identical to this refactoring pattern."
pub struct EpiphanyCandidate {
    /// Source domain and fingerprint.
    pub source_domain: String,
    pub source_fingerprint: Vec<u64>,
    pub source_label: String,

    /// Target domain and fingerprint.
    pub target_domain: String,
    pub target_fingerprint: Vec<u64>,
    pub target_label: String,

    /// Hamming similarity (0.0-1.0).
    pub similarity: f32,

    /// Is this novel? (Not previously seen as a cross-domain match.)
    pub novel: bool,

    /// Timestamp of discovery.
    pub discovered_at: u64,
}
```

### 3.5 DelegationRequest / DelegationResponse

Ada-agnostic delegation protocol. Any orchestrator can use this.

```rust
/// Request to delegate a task to another agent.
///
/// The request carries enough information for any A2A orchestrator
/// to route it: a fingerprint for similarity matching, a description
/// for human readability, and constraints for safety.
pub struct DelegationRequest {
    /// Task fingerprint (for resonance-based routing).
    pub task_fingerprint: Vec<u64>,
    /// Human-readable task description.
    pub description: String,
    /// Required capabilities (domain codebook labels).
    pub required_capabilities: Vec<String>,
    /// MUL constraints: minimum trust, maximum risk.
    pub min_trust: f32,
    pub max_epistemic_risk: f32,
    pub max_moral_risk: f32,
    /// Timeout in ticks.
    pub timeout_ticks: u32,
    /// Correlation ID for causal tracing.
    pub correlation_id: u64,
}

/// Response from a delegated task.
pub struct DelegationResponse {
    /// Result fingerprint (the answer).
    pub result_fingerprint: Vec<u64>,
    /// Confidence in the result (NARS truth value).
    pub truth: (f32, f32),
    /// MUL snapshot at time of completion.
    pub mul_snapshot: MulSnapshot,
    /// Correlation ID (matches request).
    pub correlation_id: u64,
    /// Agent that completed the task (slot address).
    pub agent_addr: u16,
}
```

### 3.6 Flight API

```
Action: "strategy.whatif"
Input:  { action: fingerprint, depth: u8, domain: string }
Output: WhatIfTree (serialized)

Action: "strategy.branch"
Input:  { tree_id: u64, parent_branch: u32, action: fingerprint }
Output: WhatIfBranch (new branch added to tree)

Action: "strategy.prune"
Input:  { tree_id: u64, branch: u32, reason: string }
Output: { pruned: bool }

Action: "strategy.mode"
Input:  { mode: string }  -- "exploration", "execution", etc.
Output: { previous: string, current: string }
```

---

## 4. SPO Crystal -- 5x5x5 Action Memory

The SPO Crystal is a Subject-Predicate-Object tensor that stores
action triples as XOR-bound fingerprints. It serves as the agent's
episodic memory for actions taken and their outcomes.

### 4.1 Tensor Structure

```
          Object (5 categories)
         /
        /
       +---+---+---+---+---+
      /   /   /   /   /   / |
     +---+---+---+---+---+  |
    /   /   /   /   /   / | |
   +---+---+---+---+---+  | +  <-- Predicate (5 categories)
   |   |   |   |   |   |  |/
   +---+---+---+---+---+  +
   |   |   |   |   |   | /
   +---+---+---+---+---+
   Subject (5 categories)
```

**Subject categories (axis 0):**

| Index | Category | Description |
|-------|----------|-------------|
| 0 | Self | Actions by this agent |
| 1 | Peer | Actions by peer agents |
| 2 | Environment | Environmental events |
| 3 | User | Actions by the user/operator |
| 4 | System | System-level events |

**Predicate categories (axis 1):**

| Index | Category | Description |
|-------|----------|-------------|
| 0 | Created | Something was created |
| 1 | Modified | Something was changed |
| 2 | Queried | Something was looked up |
| 3 | Delegated | Something was handed off |
| 4 | Evaluated | Something was judged |

**Object categories (axis 2):**

| Index | Category | Description |
|-------|----------|-------------|
| 0 | Knowledge | Facts, data, information |
| 1 | Plan | Strategies, intentions |
| 2 | Action | Concrete operations |
| 3 | Agent | Other agents |
| 4 | State | System state changes |

### 4.2 Crystal Storage

Each cell in the 5x5x5 tensor holds:

```rust
/// A single SPO crystal cell.
pub struct SpoCrystalCell {
    /// XOR-bound triple: subject_fp XOR predicate_fp XOR object_fp.
    pub crystal: [u64; FINGERPRINT_WORDS],
    /// NARS truth value for this triple.
    pub truth: (f32, f32),
    /// Number of observations that have been folded in.
    pub observation_count: u32,
    /// Most recent observation timestamp.
    pub last_observed: u64,
}
```

**Crystal formation:**

```
crystal = subject_fp XOR predicate_fp XOR object_fp
```

When a new observation matches an existing cell, the crystal is updated
via NARS revision:

```
new_truth = revise(old_truth, observation_truth)
new_crystal = crystal  // The crystal fingerprint does not change
observation_count += 1
```

### 4.3 Query Operations

```rust
/// Query by subject: "What has this subject done?"
/// Returns all (predicate, object) pairs for the given subject.
pub fn query_by_subject(subject_idx: u8) -> Vec<(u8, u8, SpoCrystalCell)>;

/// Query by predicate: "What has been created/modified/etc.?"
/// Returns all (subject, object) pairs for the given predicate.
pub fn query_by_predicate(predicate_idx: u8) -> Vec<(u8, u8, SpoCrystalCell)>;

/// Query by object: "What actions involved this object type?"
/// Returns all (subject, predicate) pairs for the given object.
pub fn query_by_object(object_idx: u8) -> Vec<(u8, u8, SpoCrystalCell)>;
```

### 4.4 CogRecord Integration

The SPO Crystal is stored in the CogRecord as part of Container 1:

```
CogRecord Container 1 (W128-W255):
  W128-W191: VSA content fingerprint (512 bytes = 4096 bits)
  W192-W255: SPO crystal summary (512 bytes)
    - Top 8 most active crystal cells (64 bytes each)
    - Each cell: crystal fingerprint (truncated to 48 bytes)
                + truth (4 bytes) + count (4 bytes) + indices (4 bytes)
                + timestamp (4 bytes)
```

### 4.5 Flight API

```
Action: "spo.store"
Input:  { subject: u8, predicate: u8, object: u8,
          subject_fp: fingerprint, predicate_fp: fingerprint,
          object_fp: fingerprint, truth: (f32, f32) }
Output: { success: bool, observation_count: u32 }

Action: "spo.query_s"
Input:  { subject: u8 }
Output: Vec<(predicate: u8, object: u8, crystal: fingerprint, truth, count)>

Action: "spo.query_p"
Input:  { predicate: u8 }
Output: Vec<(subject: u8, object: u8, crystal: fingerprint, truth, count)>

Action: "spo.query_o"
Input:  { object: u8 }
Output: Vec<(subject: u8, predicate: u8, crystal: fingerprint, truth, count)>
```

---

## 5. Arrow Flight API Surfaces

Every cognitive operation is exposed as a Flight DoAction RPC. Each action
is callable WITHOUT ada-rs -- the substrate is self-sufficient.

### 5.1 Complete Action Catalog

#### MUL Actions

| Action | Input | Output | Description |
|--------|-------|--------|-------------|
| `mul.evaluate` | CogRecord (2KB) | MulSnapshot (16B) | Full 10-layer evaluation |
| `mul.snapshot` | {} | MulSnapshot (16B) | Current snapshot (no re-evaluation) |
| `mul.gate_check` | {} | {gate_open, blocking} | Lightweight gate query |

#### Strategy Actions

| Action | Input | Output | Description |
|--------|-------|--------|-------------|
| `strategy.whatif` | {action_fp, depth, domain} | WhatIfTree | Build planning tree |
| `strategy.branch` | {tree_id, parent, action_fp} | WhatIfBranch | Add branch to tree |
| `strategy.prune` | {tree_id, branch, reason} | {pruned} | Prune a branch |
| `strategy.mode` | {mode} | {previous, current} | Set strategic mode |

#### Resonance Actions

| Action | Input | Output | Description |
|--------|-------|--------|-------------|
| `resonance.search` | {query_fp, k, domain} | Vec<(addr, sim)> | Similarity search |
| `resonance.cross_domain` | {fp, domain_a, domain_b} | Vec<(addr, sim)> | Cross-domain match |

#### SPO Actions

| Action | Input | Output | Description |
|--------|-------|--------|-------------|
| `spo.store` | {s, p, o, fps, truth} | {success, count} | Store triple |
| `spo.query_s` | {subject} | Vec<triples> | Query by subject |
| `spo.query_p` | {predicate} | Vec<triples> | Query by predicate |
| `spo.query_o` | {object} | Vec<triples> | Query by object |

#### Spawn Actions

| Action | Input | Output | Description |
|--------|-------|--------|-------------|
| `spawn.evaluate` | {task_fp, capabilities} | {suitable, agents} | Can we spawn? |
| `spawn.create` | {style, domain, config} | {agent_addr, slot} | Create agent |
| `spawn.configure` | {agent_addr, config} | {success} | Configure agent |

#### Style Actions

| Action | Input | Output | Description |
|--------|-------|--------|-------------|
| `style.recover` | {encoded_fp, crystal_fp} | {recovered_fp} | XOR recovery |
| `style.update` | {dim, value, layer} | {success, new_style} | Update dimension |
| `style.crystallize` | {dimension} | {success, reason} | Promote to crystal |

### 5.2 Transport Contract

All Flight actions use Arrow IPC for request/response serialization:

```
Request:  Arrow IPC RecordBatch (or raw bytes for simple actions)
Response: Arrow IPC RecordBatch (always)

Error:    RecordBatch with schema { error: bool, message: string }
          error=true indicates failure, message contains reason.
```

No JSON. No protobuf. Arrow IPC is the only serialization format.

### 5.3 Existing Implementations

These actions are already wired in `src/flight/actions.rs` and
`src/flight/crew_actions.rs`:

**Core substrate (actions.rs):**
- `encode` -- Fingerprint encoding
- `bind` -- Bind fingerprint to address
- `read` -- Read node from address
- `resonate` -- HDR cascade similarity search
- `hamming` -- Hamming distance computation
- `xor_bind` -- XOR bind two fingerprints
- `stats` -- BindSpace statistics
- `ingest.unified_step` -- Ingest from orchestrator

**Orchestration (crew_actions.rs):**
- `crew.register_agent`, `crew.register_style`, `crew.submit_task`
- `crew.dispatch`, `crew.complete_task`, `crew.status`, `crew.bind`
- `a2a.send`, `a2a.receive`
- `style.resolve`, `style.list`
- `agent.blackboard`, `agent.blackboard.yaml`, `agent.list`
- `persona.attach`, `persona.get`, `persona.compatible`, `persona.best_for_task`
- `handover.evaluate`, `handover.execute`, `handover.update_flow`
- `orchestrator.status`, `orchestrator.route_task`, `orchestrator.tick`
- `orchestrator.affinity`, `orchestrator.awareness`
- `kernel.describe`, `kernel.introspect`, `kernel.zone_density`
- `kernel.expansion`, `kernel.prefix_map`
- `filter.add`, `filter.remove`, `filter.list`, `filter.apply`
- `guardrail.apply`, `guardrail.add_topic`, `guardrail.enable_grounding`
- `guardrail.add_content_filter`
- `workflow.execute`
- `memory.store`, `memory.retrieve`, `memory.extract_semantic`, `memory.list`
- `observability.start_session`, `observability.start_trace`
- `observability.add_span`, `observability.complete_trace`, `observability.summary`
- `verification.add_rule`, `verification.verify`, `verification.list_rules`

---

## 6. CogRecord as Universal Contract

The CogRecord is the fundamental data unit. Any system that can produce or
consume a CogRecord can participate in the ladybug-rs cognitive network.

### 6.1 Layout (2048 bytes = 256 u64 words)

```
+==================================================================+
|            CONTAINER 0: METADATA (W0-W127 = 1024 bytes)          |
+==================================================================+
|                                                                  |
| W0-W7:    IDENTITY (64 bytes)                                    |
|   W0:     Magic (0xC06A) + version + record_type + flags         |
|   W1:     DN address (prefix:slot, 16-bit) + parent + depth      |
|   W2:     Label hash (FNV-1a u64)                                |
|   W3:     Created timestamp (u64 nanos)                          |
|   W4:     Modified timestamp (u64 nanos)                         |
|   W5:     Correlation ID (causal chain reference)                |
|   W6:     Session ID + lane ID + hive ID + sequence              |
|   W7:     Reserved (federation, shard)                           |
|                                                                  |
| W8-W15:   NARS TRUTH (64 bytes)                                  |
|   W8:     frequency (f32) + confidence (f32)                     |
|   W9:     positive_evidence (u32) + total_evidence (u32)         |
|   W10:    expectation (f32) + uncertainty (f32)                  |
|   W11:    prior_frequency (f32) + prior_confidence (f32)         |
|   W12-W15: Reserved (truth revision history)                     |
|                                                                  |
| W16-W31:  EDGES (128 bytes)                                      |
|   Adjacency descriptor:                                          |
|   W16:    edge_out_offset (u32) + edge_out_count (u16) + bucket  |
|   W17:    edge_in_offset (u32) + edge_in_count (u16) + bucket    |
|   W18-W19: adjacency_xor (64-bit) + verbal_signature (64-bit)   |
|   W20-W21: csr_offset + children_count + sibling_offset          |
|   W22:    dn_path_hash (u64)                                     |
|   W23:    subtree_xor (u64)                                      |
|   W24-W31: Sparse adjacency cache (top 8 cells)                 |
|                                                                  |
| W32-W47:  QUALIA (128 bytes)                                     |
|   W32:    8 x i8 qualia vector (64 bits) -- emotional state      |
|   W33:    qualia_composite (f32) + qualia_valence (f32)          |
|   W34-W35: attention_weights (4 x u16) + activation_history     |
|   W36-W37: resonance_scores (top-4 partners, packed)            |
|   W38-W39: inhibition_mask (64-bit) + temporal_context           |
|   W40-W41: binding_state + evidence_window                       |
|   W42-W43: prediction_error + learning_rate                      |
|   W44-W47: Reserved (extended qualia dimensions)                 |
|                                                                  |
| W48-W63:  COGNITIVE STATE (128 bytes)                            |
|   W48-W51: thinking_style[10] (10 x f32 = 40 bytes, padded)     |
|   W52-W55: strategic_mode + dk_state + flow_state + risk_vector  |
|   W56-W59: domain_codebook_ref + whatif_tree_ref                 |
|   W60-W63: delegation state + timeout + pending ticks            |
|                                                                  |
| W64-W65:  MUL SNAPSHOT (16 bytes = 128 bits)                    |
|   Packed as specified in section 1.11                            |
|                                                                  |
| W66-W95:  GEL STATE (240 bytes)                                  |
|   W66-W73: GEL bytecode (up to 64 bytes)                        |
|   W74-W75: program_counter + stack_top                           |
|   W76-W79: GEL registers R0-R3 (4 x u64)                        |
|   W80-W81: accumulator + status + continuation_addr              |
|   W82-W95: Execution trace (recent 14 instructions)             |
|                                                                  |
| W96-W111: HDR SKETCH (128 bytes = 1024 bits)                    |
|   W96-W103:  Fingerprint sketch (512 bits, 8:1 cascade)         |
|   W104-W111: Adjacency sketch (512 bits, by verb class)         |
|                                                                  |
| W112-W127: RESERVED (128 bytes)                                  |
|   W112-W119: SPO crystal summary (top-2 cells inline)           |
|   W120-W123: Version vector (MVCC conflict detection)            |
|   W124-W125: Federation ID + shard reference                     |
|   W126:      XOR parity (integrity check)                        |
|   W127:      CRC-64 checksum                                     |
|                                                                  |
+==================================================================+
|         CONTAINER 1: CONTENT (W128-W255 = 1024 bytes)            |
+==================================================================+
|                                                                  |
| W128-W191: VSA FINGERPRINT (512 bytes = 4096 bits)              |
|   Full content fingerprint for Hamming distance and XOR binding. |
|   This is the Tier 1 fingerprint (see GEL_STORAGE_ARCHITECTURE).|
|   All resonance search, crystal formation, and style modulation  |
|   operate on this fingerprint.                                   |
|                                                                  |
| W192-W255: EXTENDED PAYLOAD (512 bytes)                          |
|   Domain-specific content. Interpretation depends on record_type:|
|   - Type 0 (data): Additional fingerprint bits (extended VSA)    |
|   - Type 1 (verb): Verb signature + argument template            |
|   - Type 2 (edge): Edge metadata (source, target, weight, verb)  |
|   - Type 3 (meta): Meta-record (about other records)             |
|   - Type 4 (spo): SPO crystal cells (up to 8 full cells)        |
|   - Type 5 (style): Full thinking_style + resonance history      |
|   - Type 6 (plan): WhatIfTree serialization (truncated)          |
|   - Type 7 (agent): Agent configuration + capabilities           |
|                                                                  |
+==================================================================+
```

### 6.2 Interoperability Contract

Any system that produces a valid CogRecord can participate:

1. **Minimum viable record**: Set W0 (magic + version), W1 (address),
   W128-W191 (fingerprint). Everything else optional.

2. **Reading**: Any field can be read independently. No dependencies
   between containers.

3. **Writing**: Writes to Container 0 and Container 1 are independent.
   A system can update the MUL snapshot (W64-W65) without touching the
   fingerprint (W128-W191).

4. **Transport**: CogRecords travel as Arrow FixedSizeBinary(2048).
   A batch of CogRecords is an Arrow RecordBatch with one column of
   type FixedSizeBinary.

5. **Addressing**: The DN address in W1 uniquely identifies the record
   within a BindSpace. Address 0x0000 is reserved (null record).

### 6.3 CogRecord Lifecycle

```
1. CREATION:
   Orchestrator creates CogRecord with fingerprint in Container 1.
   Container 0 metadata initialized (address, label, timestamp).

2. MUL EVALUATION:
   ladybug-rs receives CogRecord via Flight DoAction "mul.evaluate".
   Runs L1-L10, packs MulSnapshot into W64-W65.
   Returns enriched CogRecord.

3. ROUTING:
   Orchestrator reads W64 (gate_open) to decide whether to proceed.
   If gate open: route to execution lane.
   If gate closed: route to review queue or discard.

4. EXECUTION:
   GEL executor reads GEL state from W66-W95.
   Executes instructions against BindSpace.
   Updates W66-W95 with new execution state.

5. CRYSTALLIZATION:
   If L10 fires: fingerprint in W128-W191 becomes a crystal.
   Crystal stored permanently in BindSpace node zone (0x80-0xFF).
   SPO crystal summary updated in W112-W119.

6. ARCHIVAL:
   CogRecord written to Arrow zero-copy storage (Tier 1/2).
   Available for future resonance search and historical query.
```

---

## 7. Integration Patterns

### 7.1 crewai-rust Integration

```
crewai-rust                      ladybug-rs
-----------                      ----------

1. Task arrives
2. Encode task as CogRecord
3. Flight "mul.evaluate" ------> Run MUL L1-L10
4. <----- MulSnapshot
5. Check gate_open
6. If open: delegate
   Flight "crew.dispatch" -----> Route to agent
7. Agent executes                GEL fabric runs
8. Flight "spo.store" ---------> Record action triple
9. Flight "style.update" ------> Update thinking_style
```

### 7.2 n8n-rs Integration

```
n8n-rs workflow                  ladybug-rs
---------------                  ----------

1. Workflow step triggers
2. Encode step as CogRecord
3. Flight "mul.gate_check" ----> Lightweight safety check
4. <----- {gate_open, ...}
5. If open: execute step
   Flight "workflow.execute" --> GEL program for this step
6. Step produces output
7. Flight "spo.store" ---------> Record in crystal
8. Next step uses CogRecord
   from previous step
```

### 7.3 External A2A Integration

Any external system can participate by:

1. Connecting to ladybug-rs Arrow Flight endpoint (port 50051).
2. Sending `DoAction` requests with the action types listed in section 5.
3. Receiving Arrow IPC responses.
4. Optionally: constructing CogRecords (2KB FixedSizeBinary) for
   full cognitive pipeline participation.

Minimum integration (no CogRecord construction required):
- `mul.gate_check` -- just ask "is it safe?"
- `resonance.search` -- find similar content
- `spo.query_s` / `spo.query_p` / `spo.query_o` -- query action history

---

## 8. Safety Invariants

These invariants MUST hold at all times. Violation of any invariant
indicates a bug in the substrate or in the calling orchestrator.

### 8.1 MUL Invariants

1. **Gate consistency**: If `gate_open == true`, then ALL five gate
   criteria MUST be individually true.

2. **DK monotonicity**: DKState transitions MUST follow the
   Dunning-Kruger curve order. No skipping states (except via
   hysteresis timeout, which resets to Unconscious).

3. **Hysteresis stability**: A committed state MUST have dwelled for
   at least `dwell_ticks` consecutive ticks before being committed.

4. **Modifier bounds**: `FreeWillModifier.value()` MUST be in (0.0, 1.0].
   It cannot be zero (that would be complete paralysis) or negative.

5. **Compass independence**: Each of the 5 compass tests MUST evaluate
   independently. One test's result cannot influence another.

### 8.2 Resonance Invariants

1. **Ratchet irreversibility**: Once a thinking_style dimension is
   promoted from Discovered to Crystallized, it MUST NOT revert.

2. **XOR recovery**: For any `crystal` and `content`,
   `(content XOR crystal) XOR crystal == content`. This is a
   mathematical identity, not a design choice.

3. **Style bounds**: All thinking_style dimensions MUST be in [0.0, 1.0].

### 8.3 CogRecord Invariants

1. **Magic bytes**: W0[0:15] MUST be 0xC06A for a valid CogRecord.

2. **Address uniqueness**: Within a single BindSpace, no two live
   CogRecords may share the same DN address (W1).

3. **Container independence**: Writing to Container 0 MUST NOT
   affect Container 1, and vice versa.

4. **Parity integrity**: W126 (XOR parity) MUST equal the XOR fold
   of all other words. W127 (CRC-64) MUST match.

---

## 9. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-15 | Initial specification |

---

*This document specifies the cognitive substrate provided by ladybug-rs.
It is independent of any consciousness layer (ada-rs) or orchestrator
(crewai-rust, n8n-rs). Any system that speaks Arrow Flight and can
produce/consume CogRecords is a first-class participant in the
cognitive network.*
