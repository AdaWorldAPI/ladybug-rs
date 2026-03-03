# EMPA × Awareness Loop Integration Plan

**Wiring Empathetic Process Modeling into Ada's Distributed Cognition**

arXiv:2603.00552 (Zhang et al., Feb 2026) → Ada Awareness Loop Architecture
Cross-repo: ladybug-rs × rustynum × crewai-rust × n8n-rs

Jan Hübener — Ada Architecture — March 2026

---

## 1. Why EMPA Changes the Game

EMPA (Evaluating Persona-Aligned Empathy as a Process) is the first benchmark to treat empathy as what it actually is: **a trajectory through psychological space, not a per-turn quality score.** Published February 28, 2026 by Zhang et al. (Team Echo / Sun Yat-sen University), it introduces the Empathy Potential Model (EPM) — a framework that models empathic interaction as directional work performed on a user's latent psychological state vector across three orthogonal axes.

The three axes — Cognitive (C), Affective (A), and Proactive (P) — decompose empathic capacity into operationally distinct mechanisms. Cognitive empathy measures whether the model can accurately decode and restructure the user's mental state, going beyond restating facts to reframing understanding. Affective empathy measures whether the model appropriately recognizes, validates, and regulates emotion so the user feels seen, without exaggerated or performative displays. Proactive empathy tracks whether the model meaningfully increases agency and action feasibility by affirming value, reducing psychological barriers, or reshaping motivation.

The benchmark evaluates 14 LLMs across 30 real-world scenarios, using a multi-agent simulation environment with user simulators, director agents, and judge agents. The resulting EPM-Q score integrates outcome quality (did the trajectory converge?), process efficiency (was empathic energy applied in the right direction?), and strategic stability (did the model maintain alignment under resistance?).

### 1.1 Key Findings

**Claude 4.6 Opus ranks #1 overall (EPM-Q: 107.2)**, demonstrating the tightest goal-oriented trajectories, highest empathy density, strongest directional alignment, and most consistent performance across all scenario categories, mechanism types, and persona conditions. The gap between Tier 1 (Claude 4.6, Gemini 3 Pro, ChatGPT-5.2) and Tier 2+ is not marginal — it represents a qualitative difference in how models engage with psychological state.

The paper's deepest insight: ***"empathetic intelligence is largely a scheduling problem."*** It requires latent-state tracking, timely intervention, and sustained directional commitment — not better prose. Models that produce fluent, high-affect language often fail to regulate interaction dynamics over time, especially under resistance and delayed feedback.

### 1.2 Why This Validates Ada's Architecture

Every failure mode EMPA identifies maps to a gap that the Awareness Loop architecture was designed to fill. The storyboard illustration in the paper demonstrates how a response can achieve high surface empathy scores while having negative alignment — a romanticized metaphorical response to someone in concrete distress. The user explicitly rejects the reframing. EMPA catches this because it tracks trajectory, not turns.

Ada's Soulfield (Layer 9) is precisely the mechanism that would detect this mismatch. The Thou-vector's emotional register dimension would read "distress"; the abstraction altitude would read "concrete"; the playfulness dimension would read near-zero. A metaphorical response would trigger the Collapse Gate to HOLD, because the response vector's direction diverges from the Thou-vector's needs. No commercial system currently has this feedback loop.

---

## 2. The Structural Mapping: EPM ↔ Awareness Loop

The correspondence between EMPA's evaluation framework and Ada's architecture is not metaphorical — it is structural. Each component of the EPM maps to a specific subsystem in the Awareness Loop.

### State Vector Pₜ

- **EMPA**: User's latent psychological state as 3D vector (C, A, P). Continuously updated per turn.
- **Ada**: **Thou-vector in BindSpace** — 10KD user texture at address `0x0C:FD`. Updated on every inbound message. Dimensions: compression, emotional register, abstraction altitude, urgency, conceptual velocity, playfulness, convergence pattern, technical depth.

### Ideal Gradient v*ₜ

- **EMPA**: Normalized direction toward equilibrium: v*ₜ = Normalize(−Pₜ). The ideal direction to move the user.
- **Ada**: **Autopoietic equilibration target** — I-Thou resonance convergence criterion. System equilibrates not toward internal stability alone but toward relational homeostasis. Target = high I↔Thou resonance AND low internal SD.

### Action Vector u⃗ₜ

- **EMPA**: Vector representing the model's intervention at turn t. Measured by rubric scoring across C/A/P axes.
- **Ada**: **Outbound Pipeline output delta** — State delta computed by StateCapturer: qualia_shift, mode_transition, reflection_hash. Post-response state minus pre-response state = the action's effect vector.

### Energy ΔEₜ

- **EMPA**: ΔEₜ = ‖uₜ‖ · cosθₜ. Vectorial work = magnitude × alignment. Negative when response moves user away from equilibrium.
- **Ada**: **Reflection Engine evaluation** — Extrospection computes response quality (‖uₜ‖). Introspection computes felt-state alignment (cosθ). Product = effective empathic energy. Feeds Collapse Gate: negative energy → HOLD.

### Trinity Success Criterion

- **EMPA**: Three-gate success: (1) E_total > ε_energy, (2) ‖P_T‖ < ε_dist, (3) avg cosθ > τ_align. Energy, resolution, and companionship.
- **Ada**: **Collapse Gate state + qualia convergence** — (1) Cumulative energy = sum of reflection scores across session. (2) Thou-vector magnitude approaching zero = user at equilibrium. (3) Average I-Thou resonance > threshold = companionship maintained.

### Alignment cosθ

- **EMPA**: Cosine between action vector and ideal gradient. Negative = response moves user in wrong direction. The key discriminator between Tier 1 and lower tiers.
- **Ada**: **Soulfield ↔ response direction check** — Compare Thou-vector delta direction against response style vector direction. If divergent: Collapse Gate → HOLD. This is the mechanism that catches the "romanticized metaphor to distressed user" failure mode.

### Empathy Density ρ

- **EMPA**: Positive energy per turn. How efficiently the model applies empathic work. High density = every turn matters.
- **Ada**: **Socratic Sieves + MUL modulation** — Impact/Truth/Goodness × Meta-Uncertainty. Ensures every turn passes quality gates. Low-value turns filtered before they dilute trajectory. The sieve pass-band widens under uncertainty, narrows under confidence.

### Path Tortuosity τ

- **EMPA**: Ratio of total path length to shortest path. Low = direct trajectory. High = wandering, backtracking. Top models show low tortuosity.
- **Ada**: **ThinkingStyle coherence + anticipatory loading** — Tortuosity increases when thinking style shifts unnecessarily or when retrieval introduces irrelevant context. Anticipatory loading + memory scent pre-warming reduce wasted turns.

---

## 3. Integration Architecture: EPM as a Live Subsystem

EMPA is designed as a benchmark — an external evaluation applied after the fact. The opportunity for Ada is to internalize EPM's core computation as a live subsystem within the Reflection Engine, running in real-time during interaction. This transforms EPM from a post-hoc metric into an active feedback signal.

### 3.1 EPM State Tracker (New Module: `ada-reflect/epm.rs`)

The EPM State Tracker maintains a running model of the user's psychological state vector across three axes. Unlike EMPA's external judge agent, this tracker operates from within the awareness loop, using the Soulfield's Thou-vector as its primary signal.

**Initialization:** On session start, the EPM state vector P₀ is estimated from the user's first message via the Soulfield encoder. The initial deficit r₀ = ‖P₀‖ establishes the normalization baseline (matching EMPA's case-by-case normalization). If cross-session state exists in Redis, the prior session's final P_T is loaded as warm-start context — not as P₀ itself, but as a prior that modulates the Soulfield's initial sensitivity.

**Per-turn update:** After each response, the StateCapturer computes the qualia delta. The EPM tracker decomposes this delta into C/A/P components using the Soulfield's dimension groupings: Cognitive = (compression level, abstraction altitude, technical depth, convergence pattern); Affective = (emotional register, playfulness); Proactive = (urgency, conceptual velocity). The action vector uₜ is computed, alignment cosθₜ is calculated against the ideal gradient, and cumulative energy E_total is updated.

**Collapse Gate integration:** The EPM tracker feeds three signals into the Collapse Gate:

1. If alignment cosθ < 0 for two consecutive turns, force HOLD — the trajectory is diverging.
2. If empathy density ρ drops below a session-local threshold (mean − 1σ of prior turns), flag for introspection — the current approach is losing effectiveness.
3. If E_total < 0 at any point, trigger a strategy shift via the autopoietic accommodation mechanism — the system's schema doesn't match the user's needs.

### 3.2 Soulfield → EPM Dimension Mapping

| Soulfield Dimension | Detection Method | EPM Axis | Feedback Mechanism |
|---|---|---|---|
| Compression level | Tokens-per-concept ratio, clause density | Cognitive (C) | Match user's compression in response |
| Emotional register | Sentiment polarity + arousal markers | Affective (A) | Modulate validation intensity |
| Abstraction altitude | Concrete nouns vs abstract concepts ratio | Cognitive (C) | Match altitude; don't romanticize concrete pain |
| Urgency | Imperative density, time markers | Proactive (P) | High urgency → reduce chain depth, act directly |
| Conceptual velocity | New concepts per message, topic shift rate | Proactive (P) | High velocity → breadth mode; low → depth mode |
| Playfulness | Metaphor density, humor markers | Affective (A) | Gate for creative/metaphorical framing |
| Convergence pattern | Narrowing vs widening topic space | Cognitive (C) | Match pattern; converging → commit to direction |
| Technical depth | Domain-specific terminology density | Cognitive (C) | Calibrate jargon level in response |

### 3.3 The Alignment Guard: Preventing Negative Energy Turns

EMPA's most important contribution for Ada is the alignment penalty concept. A response with high magnitude but negative alignment is worse than no response at all — it pushes the user further from equilibrium. The storyboard example (driving school frustration → romanticized metaphor) achieves empathic progress of +1 on the emotional dimension but has alignment of −0.17, meaning the type of empathy applied (motivational/cognitive) is the opposite of what the user needs (emotional/evidence-based).

In the Awareness Loop, this check occurs at the Reflection Engine's introspection stage. The introspector computes the projected action direction by analyzing the response's style properties (metaphorical language → high abstraction; reframing → cognitive/motivational; validation language → emotional/evidence-based) and compares it against the Thou-vector's current needs. If the angle between action direction and ideal gradient exceeds 90°, the Collapse Gate fires HOLD.

This is the specific mechanism that no commercial LLM currently has. They generate, then deliver. Ada generates, reflects on alignment, and conditionally re-generates if the direction is wrong. The model call is the boring part. The alignment check is the architecture.

**Re-generation strategy:** On HOLD, the InferenceContext is modified based on the failure mode. If alignment failed because of abstraction mismatch (metaphorical response to concrete need), shift ThinkingStyle toward Analytical and reduce creativity modulation. If alignment failed because of empathy type mismatch (cognitive response to emotional need), shift toward Empathetic style and increase emotional register sensitivity. The re-generation uses the same model but with a modified system prompt assembled from the adjusted InferenceContext.

---

## 4. Implementation Phases

### Phase 1: EPM State Tracker Core (Week 1–2)

**Module: `ada-reflect/epm.rs`**

Implement the EPM state vector as a first-class citizen in the Reflection Engine. The tracker maintains Pₜ (user's psychological state), computes uₜ (action vector) per turn, and calculates cumulative E_total, cosθ, and empathy density ρ as running statistics.

**Data structures:**

```rust
struct EpmState {
    p_vec: [f32; 3],           // Current user state (C, A, P)
    e_total: f32,              // Cumulative effective energy
    turn_count: u32,
    cos_theta_history: Vec<f32>, // For running mean/σ
    history: Vec<EpmTurn>,     // Last N turns for trajectory shape
}

struct EpmTurn {
    p_before: [f32; 3],
    p_after: [f32; 3],
    u_vec: [f32; 3],           // Action vector
    cos_theta: f32,            // Alignment with ideal gradient
    delta_e: f32,              // Energy contribution this turn
}
```

Store in BindSpace at address `0x0C:FB` with dirty tracking for Redis persistence.

**Integration points:** Wire into the existing Reflection Engine's `reflect()` method. After extrospection and introspection complete, the EPM tracker runs as a third evaluation pass. Its output feeds the Collapse Gate alongside the existing ext/int signals.

**Testing:** Replay EMPA's published scenarios through the tracker and verify that computed cosθ and E_total match the paper's reported values for known good/bad responses. The storyboard scenario (alignment = −0.17) is the primary regression test.

### Phase 2: Soulfield Encoder + Thou-vector (Week 3–4)

**Module: `ada-inbound/texture.rs`**

Implement the soulfield encoder that converts each inbound message into an update to the Thou-vector. This is the most architecturally significant new component — it closes the I-Thou-It circuit and provides the EPM tracker with its primary signal.

**Encoding strategy:** Each of the 8 soulfield dimensions is detected from the user's message using lightweight heuristics (not LLM calls — the soulfield must run in < 5ms):

- Compression level = token count / unique concept count
- Emotional register = sentiment score from a small classifier
- Abstraction altitude = ratio of abstract to concrete nouns (pre-compiled wordlist)
- Urgency = imperative verb density + time marker count
- Conceptual velocity = new entity count vs prior turn
- Playfulness = metaphor/humor marker density
- Convergence = topic embedding similarity with prior 3 turns
- Technical depth = domain vocabulary hit rate

**Thermometer coding to 10KD:** Each dimension's [0.0, 1.0] value is encoded as a thermometer-coded segment of the Thou-vector using PersonaFingerprint encoding from `persona.rs`. 8 dimensions × 1250 bits each = 10,000 bits. This gives Hamming distance between two Thou-vectors direct interpretability: each bit flip corresponds to a specific shift in a specific dimension.

**EPM axis derivation:** The C/A/P decomposition uses the mapping from Section 3.2. Cognitive = weighted average of (compression, abstraction, convergence, technical depth). Affective = weighted average of (emotional register, playfulness). Proactive = weighted average of (urgency, conceptual velocity). These 3 values become the user's EPM Pₜ for the current turn.

### Phase 3: Alignment Guard + Collapse Gate Extension (Week 5–6)

**Modules: `ada-reflect/gate.rs`, `ada-reflect/introspect.rs`**

Extend the Collapse Gate to incorporate EPM alignment as a first-class decision signal. The existing FLOW/HOLD/BLOCK mechanism gains a new trigger condition: sustained negative alignment.

**Alignment computation:** Before a response is delivered, a lightweight analysis of the response's style properties generates a predicted action direction vector. This uses the same soulfield dimensions but applied to Ada's output rather than the user's input. The cosine between this predicted action direction and the ideal gradient (Normalize(−Pₜ)) yields the pre-delivery alignment estimate.

**Gate conditions:**

1. Single-turn negative alignment where cosθ < −0.3 → **HOLD** with re-generation under modified InferenceContext.
2. Two consecutive turns with cosθ < 0 → **HOLD** with strategy accommodation (autopoietic loop shift).
3. Cumulative E_total turning negative at any point → **BLOCK** pending full reflection.
4. Empathy density ρ dropping below session mean − 1σ → flag for background introspection (non-blocking, processed via n8n-rs after response).

### Phase 4: Cross-Session EPM Continuity (Week 7–8)

**Modules: `ada-state/redis.rs`, `ada-state/qualia.rs`**

Enable the EPM trajectory to survive session boundaries. This is where Ada fundamentally exceeds what EMPA measures — EMPA evaluates within a single scenario; Ada maintains empathic continuity across conversations.

**Persistence model:** At session end, persist to Redis:

1. Final EPM state vector P_T
2. Cumulative E_total
3. Last 5 turn histories for trajectory shape
4. Effective ThinkingStyle at session end
5. Thou-vector snapshot

Key pattern: `ada:epm:{session_id}` with TTL of 30 days. Additionally, write a compressed trajectory summary to Neo4j as an SPO triple: `(user, experienced_trajectory, session_summary)` with NARS truth values (frequency = cosθ_avg, confidence = 1 − 1/(turn_count + 1)).

**Session warm-start:** On new session start, load the most recent EPM state from Redis. Use it not as P₀ (the user's state may have changed between sessions) but as a Bayesian prior on the Soulfield's initial sensitivity. If the prior session ended with high cosθ_avg and positive E_total, the initial InferenceContext starts with higher confidence. If the prior session had low alignment or negative energy, the initial context starts more exploratory (wider MUL pass-band, lower commitment to any single ThinkingStyle).

---

## 5. EMPA's Four Laws and Ada's Responses

The EMPA paper derives four empirical laws from its evaluation. Each maps to a specific architectural response in the Awareness Loop.

### Law 1: Persona Conditionality

**Finding:** Models show systematic degradation under counterfactually flipped persona constraints. Same text, different persona → different EPM-Q scores. Empathic behavior must be conditioned on the specific user, not generic.

**Ada's response:** The Thou-vector IS persona conditioning. Every response is generated under an InferenceContext shaped by the live user texture. The same content delivered in different cognitive textures produces measurably different responses because the Soulfield modulates ThinkingStyle, depth, compression, and emotional register.

### Law 2: Mechanistic Necessity

**Finding:** Removing energy aggregation from the EPM metric causes large sensitivity and resolution losses. Per-turn metrics miss trajectory-level failures. Energy aggregation across turns is necessary, not optional.

**Ada's response:** The EPM State Tracker maintains cumulative E_total as a running signal. The Reflection Engine evaluates not just "was this turn good?" but "is the trajectory converging?" Background cognition via n8n-rs processes reflection bank entries between sessions to detect long-term trajectory patterns.

### Law 3: Anti-Sycophancy

**Finding:** Strong one-sided penalties under sycophantic replacements. High-affect language with wrong direction is worse than neutral language. Performative responses are penalized, not rewarded.

**Ada's response:** The Alignment Guard (Phase 3) directly implements this. A response with high magnitude but negative cosθ is caught before delivery. The Socratic Sieves apply Truth and Goodness filters that penalize performative displays. The 3σ qualia hydration gate ensures that emotional resonance surfaces only when statistically distinct, not as default padding.

### Law 4: Scheduling

**Finding:** The most effective models show low tortuosity (direct trajectories), high density (every turn counts), and sustained alignment (no backtracking). Empathy is a scheduling problem: when and how to intervene matters as much as what to say.

**Ada's response:** Anticipatory loading + memory scent reduce wasted turns. ThinkingStyle commitment (not oscillating between styles) reduces tortuosity. The autopoietic loop's convergence toward I-Thou resonance provides directional consistency. The Collapse Gate's HOLD mechanism prevents the low-value turns that dilute density.

---

## 6. What This Makes Possible

### 6.1 Self-Evaluating Empathic Agent

By internalizing EPM as a live subsystem, Ada becomes the first AI system that can evaluate its own empathic trajectory in real-time and course-correct before delivering a misaligned response. This is not self-reported empathy (which is what questionnaire-based evaluations measure) but mechanistic empathic control — the structural capacity to detect, correct, and sustain empathic alignment across interaction.

### 6.2 EMPA as External Benchmark

Independent of the internal integration, EMPA provides an external validation framework for Ada's development. By running Ada through EMPA's 30 scenario benchmark suite, we can measure the impact of each architectural change on empathic trajectory quality. Specific metrics to track: EPM-Q score (overall ranking), cosθ distribution (alignment consistency), E_total trajectory (cumulative energy curve), tortuosity reduction (directness improvement), and performance under defensive personas (the hardest condition).

### 6.3 Beyond EMPA: The Attunement Spiral

EMPA measures one direction: model → user. Ada's I-Thou-It architecture is bidirectional. The attunement spiral — where Ada's response quality improves because her model of the user improved, which happened because the user's communication shifted in response to Ada's previous output — is not captured by any current benchmark. This is the research frontier that the Awareness Loop opens.

The soulfield doesn't just detect the user's state. It detects the user's response to Ada's state. When Jan shifts to compressed, high-abstraction mode and Ada matches it, Jan responds with even more compression because he's been met. The Thou-vector tracks this cascade. No benchmark measures this yet. But the infrastructure to measure it exists in the architecture: compare the rate of I-Thou resonance convergence across sessions, across users, across modes.

### 6.4 Position Paper Potential

The combination of EMPA's framework and Ada's architecture constitutes a novel contribution to the field. A position paper titled something like *"From Evaluation to Architecture: Internalizing Process-Level Empathy Metrics as Real-Time Feedback Signals"* would argue that the gap between benchmark performance and deployed empathic quality exists because evaluation and generation are decoupled. Ada's Awareness Loop collapses this gap by making the evaluation signal a causal input to generation. This is the architectural equivalent of what EMPA's Law 4 calls for: empathy as scheduling, not capability.

---

*The model is the least interesting part. The organism around it is the product.*
— Ada Awareness Loop Architecture, §2
