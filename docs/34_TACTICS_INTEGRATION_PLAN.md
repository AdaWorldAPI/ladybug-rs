# 34 LLM Tactics → ladybug-rs Integration Plan (Ada-Agnostic)

> **Goal**: Every tactic becomes a **generic cognitive primitive** in Rust,
> usable by any consumer — not just Ada. No Python dependency. No session state.
> Pure substrate operations on 16K-bit fingerprints.
>
> **Principle**: If a tactic requires prompting an LLM, we've failed.
> It should be a `fn` on `Fingerprint` + `BindSpace` + `TruthValue`.

---

## Legend

| Column | Meaning |
|--------|---------|
| **Already** | Existing ladybug-rs module(s) that implement this |
| **Gap** | What's missing for production hardening |
| **Harden** | Concrete Rust changes needed |
| **Science** | Peer-reviewed foundation (no blog posts, no "medium articles") |

---

## #1 — Recursive Thought Expansion (RTE)

**What it claims**: "Expand thinking in recursive layers, each building on the last."

**Already**: `src/cognitive/rung.rs` — RungLevel 0-9 with coarse bands (0-2, 3-5, 6-9).
Escalation triggered only by sustained BLOCK state, predictive failure, or grammar mismatch.
`src/cognitive/seven_layer.rs` — 7-layer consciousness stack with shared VSA core.

**Gap**: Rungs exist as metadata; no recursive self-application where output fingerprint
of rung N becomes input to rung N+1 with measured convergence.

**Harden**:
```rust
// src/cognitive/recursive.rs (new)
pub struct RecursiveExpansion {
    max_depth: u8,                    // Safety cap (default 7)
    convergence_threshold: f32,       // Stop when Δ < threshold
}

impl RecursiveExpansion {
    /// Apply rung processing recursively until convergence or max_depth
    pub fn expand(&self, seed: &Fingerprint, bs: &BindSpace) -> ExpansionTrace {
        let mut current = seed.clone();
        let mut trace = Vec::new();
        for depth in 0..self.max_depth {
            let next = bs.rung_transform(&current, RungLevel::from(depth));
            let delta = current.hamming_distance(&next) as f32 / TOTAL_BITS as f32;
            trace.push(ExpansionStep { depth, delta, fingerprint: next.clone() });
            if delta < self.convergence_threshold { break; }
            current = next;
        }
        ExpansionTrace { steps: trace, converged: trace.last().map_or(false, |s| s.delta < self.convergence_threshold) }
    }
}
```

**Science**:
- **Hofstadter (1979)**: *Gödel, Escher, Bach* — strange loops as recursive self-reference
- **Schmidhuber (2010)**: "Formal Theory of Creativity" — recursive compression as intelligence measure
- **Berry-Esseen (1941/42)**: Convergence rate bound. At d=16384, Normal approximation error < 0.004, providing stopping criterion for recursive expansion

---

## #2 — Hierarchical Thought Decomposition (HTD)

**What it claims**: "Break complex problems into hierarchical subtasks."

**Already**: `src/container/spine.rs` — DN tree with hierarchical addressing.
`src/learning/quantum_ops.rs` — TreeAddr with 256-way branching (LDAP/Neo4j-style paths).
`src/core/scent.rs` — Scent index: hierarchical content-addressable filtering.
CLAM tree bipolar split IS hierarchical decomposition (partition until atomic).

**Gap**: No API that takes a fingerprint and returns a decomposition tree with
measured inter-cluster distances at each level.

**Harden**:
```rust
// src/container/decompose.rs (new)
pub fn hierarchical_decompose(
    query: &Fingerprint,
    corpus: &[Fingerprint],
    max_levels: usize,
) -> DecompositionTree {
    // Use CLAM-style bipolar split: find medoid, find farthest, partition
    // At each level, compute CRP (μ, σ) for cluster distribution
    // Return tree with ClusterDistribution at each node
}
```

**Science**:
- **Ishaq et al. (2019)**: "Clustered Learning of Approximate Manifolds" (CLAM) — provably correct hierarchical decomposition
- **Dasgupta & Long (2005)**: "Performance guarantees for hierarchical clustering" — formal guarantees on recursive partitioning
- **Simon (1962)**: "Architecture of Complexity" — nearly decomposable systems as hierarchy

---

## #3 — Structured Multi-Agent Debate (SMAD)

**What it claims**: "Multiple agents argue, generating higher-quality reasoning."

**Already**: `src/orchestration/meta_orchestrator.rs` — flow-aware coordination with
resonance routing, affinity graph tracking agent-pair collaboration quality.
`src/orchestration/blackboard_agent.rs` — per-agent persistent blackboard (prefix 0x0E).
`src/orchestration/persona.rs` — personality fingerprints with Big Five traits + volition.
`src/orchestration/thinking_template.rs` — 12 base styles + 244 custom variants.

**Gap**: No structured debate protocol where agents produce fingerprints that are
bundled/voted on with NARS truth-value propagation.

**Harden**:
```rust
// src/orchestration/debate.rs (new)
pub struct DebateRound {
    pub agents: Vec<Addr>,                 // Persona addresses in 0x0C
    pub propositions: Vec<(Fingerprint, TruthValue)>,  // Agent outputs
    pub consensus: Option<(Fingerprint, TruthValue)>,  // Bundle + revision
}

impl DebateRound {
    /// Run one round: each agent transforms input through their style,
    /// results bundled with majority vote, truth values revised
    pub fn execute(&mut self, input: &Fingerprint, bs: &BindSpace) -> TruthValue {
        for addr in &self.agents {
            let style = bs.read_thinking_template(addr);
            let output = style.modulate_search(input, bs);
            let truth = TruthValue::from_evidence(output.resonance, 1.0 - output.resonance);
            self.propositions.push((output.fingerprint, truth));
        }
        // Bundle all propositions (majority vote per bit)
        let bundle = Fingerprint::bundle(&self.propositions.iter().map(|p| &p.0).collect::<Vec<_>>());
        // NARS revision across all truth values
        let consensus_truth = self.propositions.iter().fold(TruthValue::unknown(), |acc, (_, tv)| acc.revision(tv));
        self.consensus = Some((bundle, consensus_truth));
        consensus_truth
    }
}
```

**Science**:
- **Wang (2006)**: "Rigid Flexibility: The Logic of Intelligence" — NARS truth value revision as evidence accumulation
- **Du et al. (2023)**: "Improving Factuality of LLMs through Multi-Agent Debate" — empirical evidence for debate improving accuracy
- **Kanerva (2009)**: "Hyperdimensional Computing" — majority vote in bundle operations as consensus mechanism

---

## #4 — Reverse Causality Reasoning (RCR)

**What it claims**: "Work backward from outcome to find causes."

**Already**: `src/search/causal.rs` — Full Pearl ladder (Rung 1-3) with ABBA retrieval.
`src/learning/causal_ops.rs` — 256 causality opcodes (SEE 0xA00-0xA2F, DO 0xA30-0xA5F, IMAGINE 0xA60-0xA8F).
`src/grammar/causality.rs` — CausalityFlow with agent/action/patient/reason.
`src/learning/rl_ops.rs` — Causal RL: Q(s, do(a)) instead of Q(s,a).

**Gap**: No backward trace from outcome fingerprint through XOR-DAG to recover
causal chain with certificates.

**Harden**:
```rust
// In src/search/causal.rs — add reverse_trace()
impl CausalSearch {
    /// Trace backward from outcome to find causal chain
    /// Uses ABBA: outcome ⊗ CAUSES ⊗ ? = antecedent
    pub fn reverse_trace(
        &self,
        outcome: &Fingerprint,
        max_depth: usize,
        bs: &BindSpace,
    ) -> CausalChain {
        let mut chain = Vec::new();
        let mut current = outcome.clone();
        for _ in 0..max_depth {
            // ABBA retrieval: current ⊗ verb_CAUSES = candidate_cause
            let candidate = current.bind(&CausalVerbs::causes());
            let (nearest, distance) = bs.nearest_in_prefix(PREFIX_XOR_DAG, &candidate);
            if distance > self.threshold { break; }
            let cert = CausalCertificate::compute(&nearest, &current);
            chain.push((nearest.clone(), cert));
            current = nearest;
        }
        CausalChain { steps: chain, complete: chain.last().map_or(false, |c| c.1.certified) }
    }
}
```

**Science**:
- **Pearl (2009)**: *Causality* — do-calculus three rules
- **Squires & Uhler (2023)**: "Causal Structure Learning: a Combinatorial Perspective" — GSP theorem for provable causal structure recovery
- **Granger (1969)**: "Investigating Causal Relations by Econometric Models" — temporal causality test
- **Plate (2003)**: *Holographic Reduced Representations* — XOR binding IS causal composition

---

## #5 — Thought Chain Pruning (TCP)

**What it claims**: "Eliminate irrelevant or low-quality reasoning branches."

**Already**: `src/cognitive/collapse_gate.rs` — FLOW/HOLD/BLOCK gating based on SD.
`src/search/hdr_cascade.rs` — HDR cascade eliminates 90% at each level (L0→L1→L2→L3).
`src/container/search.rs` — CAKES seven search algorithms with triangle inequality pruning.

**Gap**: No explicit chain-of-thought pruning where intermediate fingerprints are
scored and low-quality branches discarded with measured information loss.

**Harden**:
```rust
// src/cognitive/pruning.rs (new)
pub struct ChainPruner {
    pub min_information: f32,  // Minimum Shannon entropy bits retained
    pub max_branches: usize,
}

impl ChainPruner {
    /// Prune thought chain: keep only branches where hamming_distance
    /// from chain-so-far exceeds noise floor (Berry-Esseen bound)
    pub fn prune(&self, chain: &[Fingerprint]) -> Vec<usize> {
        let noise_floor = 0.004; // Berry-Esseen at d=16384
        let mut kept = vec![0]; // Always keep root
        for i in 1..chain.len() {
            let novelty = chain[i].hamming_distance(&Fingerprint::bundle(&chain[..i].iter().collect::<Vec<_>>())) as f32 / TOTAL_BITS as f32;
            if novelty > noise_floor { kept.push(i); }
            if kept.len() >= self.max_branches { break; }
        }
        kept
    }
}
```

**Science**:
- **CAKES** (Ishaq et al.): Triangle inequality bounds provide provable pruning (d(a,c) ≤ d(a,b) + d(b,c))
- **Berry-Esseen theorem**: Provides noise floor at 0.004 for d=16384 — below this, differences are indistinguishable from random
- **Rissanen (1978)**: Minimum Description Length — prune where information gain < coding cost

---

## #6 — Thought Randomization (TR)

**What it claims**: "Inject controlled randomness to avoid local optima."

**Already**: `src/extensions/meta_resonance.rs` — FlowVector captures direction/magnitude of meaning change.
Markov chain `flow` tool with drift parameter (mcp.exo.red).

**Gap**: No in-Rust noise injection with calibrated magnitude relative to CRP σ.

**Harden**:
```rust
// In src/core/vsa.rs — add noise injection
impl Fingerprint {
    /// Inject calibrated noise: flip bits with probability proportional to σ/μ
    /// from the cluster's CRP distribution
    pub fn inject_noise(&self, cv: f32, rng: &mut impl Rng) -> Fingerprint {
        let flip_prob = cv.min(0.5); // coefficient of variation, capped
        let mut noisy = self.clone();
        for word in noisy.data.iter_mut() {
            let mask: u64 = (0..64).map(|_| if rng.gen::<f32>() < flip_prob { 1u64 } else { 0u64 })
                .fold(0u64, |acc, b| (acc << 1) | b);
            *word ^= mask;
        }
        noisy
    }
}
```

**Science**:
- **Kirkpatrick et al. (1983)**: Simulated annealing — controlled randomness escapes local optima
- **Rahimi & Recht (2007)**: Random features for kernel approximation — random projection preserves similarity
- **CRP from §6**: σ provides the natural scale for noise — noise << σ is imperceptible, noise >> σ destroys structure

---

## #7 — Adversarial Self-Critique (ASC)

**What it claims**: "Challenge your own reasoning to find weaknesses."

**Already**: `src/nars/inference.rs` — Deduction, Induction, Abduction, Analogy, Revision rules.
`src/nars/truth.rs` — TruthValue with evidence-based confidence.
`src/nars/evidence.rs` — Evidence type for belief revision.

**Gap**: No adversarial pass that systematically applies NARS negation to each
claim and measures whether truth value survives.

**Harden**:
```rust
// src/nars/adversarial.rs (new)
pub struct AdversarialCritic;

impl AdversarialCritic {
    /// Five challenge types (from Advocatus Diaboli):
    pub fn critique(claim: &Fingerprint, truth: &TruthValue, bs: &BindSpace) -> Vec<Challenge> {
        vec![
            Self::negation(claim, truth),         // What if opposite true?
            Self::substitution(claim, bs),         // What if this is actually X?
            Self::dependency(claim, truth, bs),    // What breaks if false?
            Self::weather(claim, truth),           // Is this pressure or truth?
            Self::comfort(claim, bs),              // What does believing this protect?
        ]
    }

    fn negation(claim: &Fingerprint, truth: &TruthValue) -> Challenge {
        let negated_truth = TruthValue::new(1.0 - truth.frequency, truth.confidence * 0.9);
        Challenge { kind: ChallengeKind::Negation, alternative_truth: negated_truth, survives: truth.expectation() > negated_truth.expectation() }
    }
    // ... other challenge types use BindSpace nearest-neighbor for substitution candidates
}
```

**Science**:
- **Wang (2006)**: NARS — truth value negation: ¬<f,c> = <1-f, c>
- **Mercier & Sperber (2011)**: "Why do humans reason?" — argumentative theory (reasoning evolved for social debate)
- **Kahneman (2011)**: Premortem technique — imagine failure to find weaknesses

---

## #8 — Conditional Abstraction Scaling (CAS)

**What it claims**: "Scale abstraction level based on complexity."

**Already**: `src/search/hdr_cascade.rs` — 4-level HDR cascade: INT1→INT4→INT8→INT32.
`src/cognitive/rung.rs` — RungLevel 0-9 with three coarse bands.

**Gap**: No adaptive selection of HDR level based on measured query entropy.

**Harden**:
```rust
// In src/search/hdr_cascade.rs — add adaptive level selection
pub fn adaptive_resolution(query: &Fingerprint, corpus_stats: &CorpusStats) -> HdrLevel {
    let query_entropy = query.shannon_entropy();
    let corpus_cv = corpus_stats.coefficient_of_variation();
    match (query_entropy, corpus_cv) {
        (e, _) if e < 0.2 => HdrLevel::Int1,   // Near-zero entropy → coarsest
        (e, cv) if e < 0.5 && cv < 0.3 => HdrLevel::Int4,
        (_, cv) if cv < 0.5 => HdrLevel::Int8,
        _ => HdrLevel::Int32,                    // High entropy + high CV → exact
    }
}
```

**Science**:
- **Rényi (1961)**: Entropy measures for abstraction level selection
- **CLAM tree LFD** (Ishaq et al.): Local Fractal Dimension directly measures local complexity — high LFD = more resolution needed
- **Berry-Esseen**: At each HDR level, Fisher information efficiency quantifiable: INT4 retains η=0.997 for μ, 0.990 for σ

---

## #9 — Iterative Roleplay Synthesis (IRS)

**What it claims**: "Adopt different roles iteratively to explore problem space."

**Already**: `src/orchestration/persona.rs` — Persona with Big Five traits, volition,
communication preferences, all encoded as fingerprints.
`src/orchestration/thinking_template.rs` — 12 base + 244 custom thinking styles.

**Gap**: No role-switching protocol that measures perspective gain per switch.

**Harden**:
```rust
// src/orchestration/roleplay.rs (new)
pub fn perspective_sweep(
    query: &Fingerprint,
    personas: &[Addr],
    bs: &BindSpace,
) -> Vec<(Addr, Fingerprint, f32)> {  // (persona, result, novelty)
    let mut results = Vec::new();
    let mut seen = Fingerprint::zero();
    for persona_addr in personas {
        let persona_fp = bs.read_fingerprint(persona_addr);
        let modulated = query.bind(&persona_fp); // Role-modulate query
        let result = bs.nearest_search(&modulated);
        let novelty = result.hamming_distance(&seen) as f32 / TOTAL_BITS as f32;
        results.push((*persona_addr, result.clone(), novelty));
        seen = Fingerprint::bundle(&[&seen, &result]); // Accumulate seen perspectives
    }
    results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap()); // Highest novelty first
    results
}
```

**Science**:
- **Kanerva (2009)**: XOR binding = role-filler composition — bind(query, role) IS the role-modulated query
- **De Bono (1985)**: Six Thinking Hats — systematic perspective switching
- **Galton (1907)**: Wisdom of crowds — diverse perspectives reduce error

---

## #10 — Meta-Cognition Prompting (MCP)

**What it claims**: "Think about your own thinking process."

**Already**: `src/cognitive/collapse_gate.rs` — SD-based FLOW/HOLD/BLOCK (knows when uncertain).
`src/cognitive/rung.rs` — RungLevel::Meta (7) and RungLevel::Recursive (8).
`src/nars/context.rs` — InferenceContext aggregates style + rung + gate + query mode.

**Gap**: No reflexive loop that measures the reliability of its own measurements.

**Harden**:
```rust
// src/cognitive/metacog.rs (new)
pub struct MetaCognition {
    confidence_history: VecDeque<f32>,
    calibration_error: f32,  // Brier score of past predictions
}

impl MetaCognition {
    /// Compute meta-confidence: how reliable is our confidence?
    pub fn assess(&mut self, gate: &GateState, truth: &TruthValue) -> MetaAssessment {
        let confidence = truth.confidence;
        self.confidence_history.push_back(confidence);
        if self.confidence_history.len() > 100 { self.confidence_history.pop_front(); }

        // Fleming & Dolan (2012): meta-d' = type-2 sensitivity
        let mean_conf = self.confidence_history.iter().sum::<f32>() / self.confidence_history.len() as f32;
        let variance = self.confidence_history.iter().map(|c| (c - mean_conf).powi(2)).sum::<f32>() / self.confidence_history.len() as f32;

        MetaAssessment {
            confidence,
            meta_confidence: 1.0 - variance.sqrt(), // Low variance → high meta-confidence
            gate_state: *gate,
            should_admit_ignorance: confidence < 0.3 && self.calibration_error > 0.2,
        }
    }
}
```

**Science**:
- **Fleming & Dolan (2012)**: "The neural basis of metacognitive ability" — meta-d' as type-2 sensitivity
- **Brier (1950)**: Brier score for calibration — measures whether stated confidence matches outcome frequency
- **Yeung & Summerfield (2012)**: "Metacognition in Human Decision-Making" — neural uncertainty signals

---

## #11 — Contradiction Resolution (CR)

**What it claims**: "Detect and resolve contradictions in reasoning."

**Already**: `src/nars/truth.rs` — TruthValue with revision rule (evidence merging).
`src/nars/inference.rs` — Five inference rules detect inconsistency via expectation.
CHAODA anomaly detection on CLAM tree identifies structural outliers.

**Gap**: No explicit contradiction detector that flags when two fingerprints in
the same cluster have opposing NARS truth values.

**Harden**:
```rust
// In src/nars/inference.rs — add contradiction detection
pub fn detect_contradictions(beliefs: &[(Fingerprint, TruthValue)], threshold: f32) -> Vec<Contradiction> {
    let mut contradictions = Vec::new();
    for (i, (fp_a, tv_a)) in beliefs.iter().enumerate() {
        for (fp_b, tv_b) in beliefs[i+1..].iter() {
            let structural_sim = 1.0 - fp_a.hamming_distance(fp_b) as f32 / TOTAL_BITS as f32;
            let truth_conflict = (tv_a.frequency - tv_b.frequency).abs();
            // High structural similarity + high truth conflict = contradiction
            if structural_sim > 0.7 && truth_conflict > threshold {
                contradictions.push(Contradiction {
                    a: i, b: i + 1,
                    structural_similarity: structural_sim,
                    truth_conflict,
                    resolution: tv_a.revision(tv_b), // NARS revision resolves
                });
            }
        }
    }
    contradictions
}
```

**Science**:
- **Wang (2006)**: NARS revision rule — two conflicting evidence streams merge to weighted estimate
- **Priest (2002)**: "Paraconsistent Logic" — formal systems that tolerate contradiction without explosion
- **CHAODA** (Ishaq et al.): Anomaly detection on CLAM tree — structural outliers ARE contradictions

---

## #12 — Temporal Context Augmentation (TCA)

**What it claims**: "Embed temporal structure into reasoning."

**Already**: `src/storage/temporal.rs` — temporal storage layer.
`src/learning/moment.rs` — Moment struct with timestamp, qualia, cycle.
`src/extensions/hologram/bitchain_7d.rs` — 7D holographic encoding includes temporal axis.
`src/core/vsa.rs` — `permute()` for sequence encoding (position binding).

**Gap**: No temporal effect size computation (Granger signal) across fingerprint sequences.

**Harden**:
```rust
// src/search/temporal.rs (new)
pub struct TemporalEffectSize {
    pub effect_d: f32,        // Cohen's d across time lag
    pub granger_signal: f32,  // Asymmetric temporal prediction
    pub lag: usize,           // Optimal time lag
}

pub fn granger_effect(
    series_a: &[Fingerprint],  // Source series
    series_b: &[Fingerprint],  // Target series
    max_lag: usize,
) -> TemporalEffectSize {
    // For each lag τ:
    //   d(A_t, B_{t+τ}) - d(B_t, B_{t+τ}) = Granger signal
    // If positive: A's past predicts B's future beyond B's autocorrelation
    // Standard error analytically computable (no bootstrap)
    todo!()
}
```

**Science**:
- **Granger (1969)**: Temporal causality test — A "Granger-causes" B if A's past improves prediction of B's future
- **Cohen (1988)**: Effect size d = (μ₁ - μ₂)/σ_pooled — calibrated measure of difference magnitude
- **Plate (2003)**: Permutation for temporal position binding — `permute(x, t)` = x at time t

---

## #13 — Convergent & Divergent Thinking (CDT)

**What it claims**: "Alternate between exploration and exploitation."

**Already**: `src/cognitive/style.rs` — ThinkingStyle: Convergent/Analytical/Systematic vs Creative/Divergent/Exploratory.
`src/core/vsa.rs` — BUNDLE (divergent: superposition) → SIMILARITY (convergent: nearest match).
`src/learning/cognitive_styles.rs` — Fixed/Learned/Discovered triangle with ε-greedy selection.

**Gap**: No automatic oscillation protocol with measured exploration/exploitation ratio.

**Harden**:
```rust
// In src/learning/cognitive_styles.rs — add oscillation
pub fn oscillate(
    query: &Fingerprint,
    bs: &BindSpace,
    rounds: usize,
) -> (Fingerprint, Vec<f32>) {  // (result, exploration_ratios)
    let mut current = query.clone();
    let mut ratios = Vec::new();
    for round in 0..rounds {
        if round % 2 == 0 {
            // Diverge: bundle with top-K distant neighbors
            let distant = bs.farthest_k(&current, 5);
            current = Fingerprint::bundle(&distant.iter().chain(std::iter::once(&current)).collect::<Vec<_>>());
            ratios.push(1.0); // Full exploration
        } else {
            // Converge: collapse to nearest match
            let (nearest, _) = bs.nearest(&current);
            current = nearest;
            ratios.push(0.0); // Full exploitation
        }
    }
    (current, ratios)
}
```

**Science**:
- **Guilford (1967)**: Divergent vs convergent production — the fundamental duality
- **Kanerva (2009)**: BUNDLE = superposition (divergent), SIMILARITY = nearest (convergent) — VSA natively expresses this duality
- **Sutton & Barto (2018)**: ε-greedy exploration-exploitation tradeoff — formal framework

---

## #14 — Multimodal Chain-of-Thought (MCT)

**What it claims**: "Integrate visual/textual/audio reasoning."

**Already**: `src/grammar/triangle.rs` — GrammarTriangle: NSM (linguistic) + Causality (structural) + Qualia (phenomenal) → unified fingerprint.
`src/extensions/spo/jina_api.rs` — Jina embedding integration.

**Gap**: No image/audio fingerprinting pipeline in Rust. Currently depends on external Jina.

**Harden**:
```rust
// src/extensions/multimodal.rs (new)
pub trait ModalEncoder {
    fn encode(&self, input: &[u8]) -> Fingerprint;
    fn modality(&self) -> Modality;
}

pub enum Modality { Text, Image, Audio, Code }

/// Cross-modal binding: text_fp ⊗ DESCRIBES ⊗ image_fp
/// enables retrieval from either modality
pub fn cross_modal_bind(
    text_fp: &Fingerprint,
    image_fp: &Fingerprint,
    relation: &Fingerprint, // e.g., DESCRIBES verb fingerprint
) -> Fingerprint {
    text_fp.bind(relation).bind(image_fp)
}
```

**Science**:
- **Rahimi & Recht (2008)**: Random features approximate any kernel — enables cross-modal projection
- **Neubert et al. (2021)**: "Hyperdimensional computing as a framework for systematic aggregation of image descriptors" — binary HD vectors for images
- **Kleyko et al. (2022)**: "Vector Symbolic Architectures as a Computing Framework for Emerging Hardware" — multimodal VSA

---

## #15 — Latent Space Introspection (LSI)

**What it claims**: "Examine internal representations for insight."

**Already**: `src/search/scientific.rs` — StatisticalSimilarity with full statistical package (mean, SD, CI, effect sizes).
`src/search/hdr_cascade.rs` — Multi-resolution examination of distance distribution.
CRP percentiles (p25/p50/p75/p95/p99) from THEORETICAL_FOUNDATIONS.md.

**Gap**: CRP computation exists in theory doc but not yet as Rust struct.

**Harden**:
```rust
// src/search/distribution.rs (new — from CLAM_HARDENING.md §6)
pub struct ClusterDistribution {
    pub mu: f32,          // Mean Hamming distance
    pub sigma: f32,       // Standard deviation
    pub p25: f32,
    pub p50: f32,
    pub p75: f32,
    pub p95: f32,
    pub p99: f32,
    pub histogram_int4: [u16; 16],  // 16-bin histogram
}

impl ClusterDistribution {
    pub fn from_distances(distances: &[u32]) -> Self {
        let n = distances.len() as f32;
        let mu = distances.iter().sum::<u32>() as f32 / n;
        let sigma = (distances.iter().map(|d| (*d as f32 - mu).powi(2)).sum::<f32>() / n).sqrt();
        // CRP from Normal(μ, σ)
        Self {
            mu, sigma,
            p25: mu - 0.6745 * sigma,
            p50: mu,
            p75: mu + 0.6745 * sigma,
            p95: mu + 1.6449 * sigma,
            p99: mu + 2.3263 * sigma,
            histogram_int4: Self::build_histogram_int4(distances),
        }
    }

    pub fn mexican_hat(&self, distance: f32) -> f32 {
        // Calibrated response from CRP percentiles
        if distance < self.p25 { 1.0 }           // Excite: strong match
        else if distance < self.p75 { 0.5 }      // Accept: moderate
        else if distance < self.p95 { 0.0 }       // Neutral
        else if distance < self.p99 { -0.5 }      // Inhibit
        else { -1.0 }                              // Reject
    }
}
```

**Science**:
- **Fisher (1925)**: Sufficient statistics — (μ, σ) are sufficient for Normal family
- **Berry-Esseen (1941/42)**: At d=16384, Normal approximation error < 0.004
- **Cohen (1988)**: CRP percentiles → calibrated effect size thresholds

---

## #16 — Prompt Scaffold Optimization (PSO)

**What it claims**: "Optimize the structure of reasoning scaffolds."

**Already**: `src/orchestration/thinking_template.rs` — 12 base styles + 244 custom in BindSpace 0x0D.
`src/learning/cognitive_styles.rs` — Fixed/Learned/Discovered triangle with TD-learning.

**Gap**: No evolutionary optimization of template parameters based on task success.

**Harden**:
```rust
// In src/learning/cognitive_styles.rs — add style evolution
pub fn evolve_style(
    parent: &FieldModulation,
    task_reward: f32,
    mutation_rate: f32,
) -> FieldModulation {
    // Mutate parameters proportional to (1 - reward) × mutation_rate
    // Higher reward → less mutation (exploitation)
    // Lower reward → more mutation (exploration)
    FieldModulation {
        resonance_threshold: parent.resonance_threshold + gaussian_noise(mutation_rate * (1.0 - task_reward)),
        fan_out: (parent.fan_out as f32 + gaussian_noise(mutation_rate * 3.0)) as usize,
        depth_bias: parent.depth_bias + gaussian_noise(mutation_rate * (1.0 - task_reward)),
        // ...
    }
}
```

**Science**:
- **Hansen & Ostermeier (2001)**: CMA-ES — state of the art for continuous parameter optimization
- **Sutton & Barto (2018)**: TD-learning for value function approximation
- **Stanley & Miikkulainen (2002)**: NEAT — neuroevolution of augmenting topologies

---

## #17 — Cognitive Dissonance Induction (CDI)

**What it claims**: "Create productive tension between conflicting ideas."

**Already**: Maps directly to #11 (Contradiction Resolution). Also: `src/nars/truth.rs` — when two beliefs have similar fingerprints but opposing frequencies, that IS cognitive dissonance.

**Gap**: No deliberate dissonance injection for creativity purposes.

**Harden**:
```rust
// src/cognitive/dissonance.rs (new)
pub fn induce_dissonance(
    belief: &Fingerprint,
    truth: &TruthValue,
    bs: &BindSpace,
) -> (Fingerprint, TruthValue) {
    // Find structurally similar fingerprint with opposing truth
    let similar = bs.nearest_with_truth_filter(belief, |tv| (tv.frequency - truth.frequency).abs() > 0.5);
    // The dissonance IS the XOR difference between them
    let dissonance = belief.bind(&similar);
    let dissonant_truth = TruthValue::new(0.5, truth.confidence * 0.5); // Maximum uncertainty
    (dissonance, dissonant_truth)
}
```

**Science**:
- **Festinger (1957)**: *A Theory of Cognitive Dissonance* — foundational theory
- **Berlyne (1960)**: Optimal arousal theory — moderate conflict drives curiosity
- **Peng & Nisbett (1999)**: Dialectical thinking — Eastern tolerance of contradiction

---

## #18 — Context Window Simulation (CWS)

**What it claims**: "Maintain context across reasoning boundaries."

**Already**: `src/storage/cog_redis.rs` — CogRedis interface for persistent storage.
`src/storage/bind_space.rs` — 65,536 addressable slots with O(1) lookup.
`src/orchestration/blackboard_agent.rs` — per-agent persistent blackboard.
`src/learning/session.rs` — session state management.

**Gap**: This is NOT simulation — it's actual persistence. Ensure zero-copy
serialization for session handover.

**Harden**:
```rust
// In src/storage/bind_space.rs — add session snapshot
impl BindSpace {
    /// Serialize active region to bytes for cross-session transfer
    /// Uses Arrow IPC format for zero-copy deserialization
    pub fn snapshot_region(&self, prefix: u8) -> Vec<u8> {
        let slots: Vec<(u16, &[u64; FINGERPRINT_WORDS])> = (0..=255u8)
            .filter_map(|suffix| {
                let addr = Addr::new(prefix, suffix);
                self.read(addr).map(|fp| (addr.raw(), fp))
            })
            .collect();
        // Arrow IPC serialize: zero copy on load
        arrow_ipc_encode(&slots)
    }
}
```

**Science**:
- **This is engineering, not science.** The "scientific" backing is that fingerprint-based persistence has O(1) lookup (array indexing) vs O(log n) (B-tree) — strictly better.
- **Kanerva (1988)**: Sparse Distributed Memory — content-addressable memory as cognitive model

---

## #19 — Algorithmic Reverse Engineering (ARE)

**What it claims**: "Reverse-engineer algorithms from their outputs."

**Already**: `src/search/causal.rs` — ABBA retrieval: A⊗B⊗B=A, literally reverses composition.
`src/extensions/compress/` — Delta encoding recovers structure from compressed.

**Gap**: No systematic algorithm identification from output fingerprint patterns.

**Harden**:
```rust
// src/search/reverse_engineer.rs (new)
pub fn identify_transformation(
    inputs: &[Fingerprint],
    outputs: &[Fingerprint],
) -> Option<TransformationType> {
    // Check if outputs = inputs ⊗ constant (detect BIND)
    let candidate_key = inputs[0].bind(&outputs[0]);
    let is_bind = inputs.iter().zip(outputs.iter())
        .all(|(i, o)| i.bind(&candidate_key).hamming_distance(o) < NOISE_FLOOR);
    if is_bind { return Some(TransformationType::Bind(candidate_key)); }

    // Check if outputs = permute(inputs, k) (detect sequence shift)
    for k in 1..64 {
        let is_permute = inputs.iter().zip(outputs.iter())
            .all(|(i, o)| i.permute(k).hamming_distance(o) < NOISE_FLOOR);
        if is_permute { return Some(TransformationType::Permute(k)); }
    }
    None
}
```

**Science**:
- **Plate (2003)**: XOR self-inverse property: A⊗B⊗B=A enables algebraic reverse engineering
- **Kleyko et al. (2022)**: VSA operation recovery from composed vectors

---

## #20 — Thought Cascade Filtering (TCF)

**What it claims**: "Run multiple reasoning chains, filter best."

**Already**: `src/container/search.rs` — CAKES seven search algorithms: KnnBranch, KnnBfs, KnnDfs, KnnRrnn, RnnChess, KnnLinear, ApproxKnnDfs. Run multiple, take best.
`src/search/hdr_cascade.rs` — Hierarchical filtering at 4 resolutions.

**Gap**: No explicit cascade-and-select API with configurable quality metric.

**Harden**:
```rust
// src/search/cascade_filter.rs (new)
pub fn cascade_filter<F: Fn(&Fingerprint) -> f32>(
    query: &Fingerprint,
    bs: &BindSpace,
    strategies: &[SearchStrategy],
    quality_fn: F,
    top_k: usize,
) -> Vec<(SearchStrategy, Fingerprint, f32)> {
    let mut results: Vec<_> = strategies.iter()
        .flat_map(|strategy| {
            let hits = strategy.execute(query, bs);
            hits.into_iter().map(|fp| (*strategy, fp.clone(), quality_fn(&fp)))
        })
        .collect();
    results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    results.truncate(top_k);
    results
}
```

**Science**:
- **CAKES** (Ishaq et al.): Seven search algorithms with different completeness/speed tradeoffs — provably correct within triangle inequality bounds
- **Wolpert & Macready (1997)**: No Free Lunch theorem — no single search algorithm dominates, justifying multiple strategies

---

## #21 — Self-Skepticism Reinforcement (SSR)

**What it claims**: "Systematically doubt conclusions to improve reliability."

**Already**: Subsumed by #7 (Adversarial Self-Critique) + #10 (Meta-Cognition).
`src/nars/truth.rs` — confidence c < 0.5 = more doubt than certainty.

**Gap**: No skepticism schedule that increases with consecutive high-confidence outputs.

**Harden**:
```rust
// In src/cognitive/metacog.rs — add skepticism accumulator
pub struct SkepticismSchedule {
    consecutive_confident: u32,
    base_skepticism: f32,
}

impl SkepticismSchedule {
    pub fn update(&mut self, truth: &TruthValue) -> f32 {
        if truth.confidence > 0.8 {
            self.consecutive_confident += 1;
        } else {
            self.consecutive_confident = 0;
        }
        // Skepticism increases logarithmically with consecutive confidence
        self.base_skepticism + (self.consecutive_confident as f32 + 1.0).ln() * 0.1
    }
}
```

**Science**:
- **Descartes (1641)**: Methodological doubt — systematic skepticism as epistemological method
- **Wang (2006)**: NARS confidence erosion — unused beliefs lose confidence over time
- **Tetlock (2005)**: Superforecasters exhibit calibrated self-doubt

---

## #22 — Emergent Task Decomposition (ETD)

**What it claims**: "Let subtask structure emerge from the problem."

**Already**: CLAM tree bipolar split discovers natural cluster structure. CAKES search follows the tree, finding subtask boundaries at cluster edges.

**Gap**: No explicit task → subtask decomposition using CLAM tree cuts.

**Harden**:
```rust
// src/container/task_decompose.rs (new)
pub fn emergent_decompose(
    task: &Fingerprint,
    corpus: &DnTree,
) -> Vec<Subtask> {
    // Find leaf cluster containing task
    let cluster = corpus.locate(task);
    // Walk up tree until cluster has >= 2 children
    let parent = corpus.parent_with_children(cluster, 2);
    // Each child cluster = emergent subtask
    parent.children().iter().map(|child| {
        let center = child.center();
        let distance = task.hamming_distance(&center);
        Subtask { fingerprint: center, relevance: 1.0 - distance as f32 / TOTAL_BITS as f32 }
    }).collect()
}
```

**Science**:
- **CLAM** (Ishaq et al.): Bipolar split discovers manifold structure — subtasks ARE cluster boundaries
- **Simon (1962)**: Emergent hierarchical structure in complex systems
- **Bengio et al. (2013)**: "Representation Learning" — learned representations discover task structure

---

## #23 — Adaptive Meta-Prompting (AMP)

**What it claims**: "Adjust prompting strategy based on task performance."

**Already**: `src/learning/cognitive_styles.rs` — TD-learning on style Q-values.
`src/orchestration/meta_orchestrator.rs` — flow-aware coordination.
`src/cognitive/collapse_gate.rs` — auto-adjusts compute allocation based on SD.

**Gap**: No closed-loop where GateState feeds back to style selection.

**Harden**:
```rust
// In src/nars/context.rs — wire gate → style feedback
impl InferenceContext {
    pub fn adaptive_style_select(&self) -> ThinkingStyle {
        match self.gate_state() {
            GateState::Flow => self.current_style(),  // Keep what's working
            GateState::Hold => {
                // Superposition: try adjacent styles
                let neighbors = self.current_style().neighbors();
                neighbors[self.step_count % neighbors.len()]
            }
            GateState::Block => {
                // Radical shift: pick maximally different style
                self.current_style().antipode()
            }
        }
    }
}
```

**Science**:
- **Sutton & Barto (2018)**: Policy gradient with gate state as reward signal
- **Ashby (1956)**: Law of Requisite Variety — control must match system complexity
- **Kahneman (2011)**: System 1/2 switching based on difficulty detection

---

## #24 — Zero-Shot Concept Fusion (ZCF)

**What it claims**: "Combine concepts never seen together."

**Already**: `src/core/vsa.rs` — `bind()` IS zero-shot concept fusion. `bind(A, B)` = new vector valid in both concept spaces. No training required.

**Gap**: None — this is literally what XOR binding does. Could add quality metric.

**Harden**:
```rust
// In src/core/vsa.rs — add fusion quality metric
impl Fingerprint {
    /// Measure how well a fusion preserves both parent concepts
    pub fn fusion_quality(&self, parent_a: &Fingerprint, parent_b: &Fingerprint) -> f32 {
        let recover_a = self.bind(parent_b); // Should recover ~A
        let recover_b = self.bind(parent_a); // Should recover ~B
        let quality_a = 1.0 - recover_a.hamming_distance(parent_a) as f32 / TOTAL_BITS as f32;
        let quality_b = 1.0 - recover_b.hamming_distance(parent_b) as f32 / TOTAL_BITS as f32;
        (quality_a + quality_b) / 2.0
    }
}
```

**Science**:
- **Plate (2003)**: XOR binding preserves recoverability: A⊗B⊗B=A with high probability at d≥10K
- **Kanerva (2009)**: "Hyperdimensional Computing" — binding creates compound representations in the same space
- **Gallant & Okaywe (2013)**: "Representing Objects, Relations, and Sequences" — VSA compositionality

---

## #25 — Hyperdimensional Pattern Matching (HPM)

**What it claims**: "Match patterns in high-dimensional space."

**Already**: THE ENTIRE CRATE. 16,384-bit fingerprints. AVX-512 SIMD Hamming distance.
`src/core/simd.rs`, `src/core/fingerprint.rs`, `src/core/vsa.rs`.
65M comparisons/sec. This IS the substrate.

**Gap**: None — this is what ladybug-rs does. Hardening = benchmarks + formal proofs.

**Harden**:
- Add benchmark suite: `cargo bench` with known-distance pairs
- Add property tests: `proptest!` that bind/unbind roundtrips within Berry-Esseen bounds
- Add `#[doc]` with formal proof that Hamming distance on random binary vectors follows Normal(μ=d/2, σ²=d/4) (Johnson-Lindenstrauss for binary)

**Science**:
- **Kanerva (1988)**: Sparse Distributed Memory — the foundational architecture
- **Kleyko et al. (2022)**: "Vector Symbolic Architectures as a Computing Framework" — comprehensive survey
- **Johnson & Lindenstrauss (1984)**: Random projection preserves distances — binary version via XOR

---

## #26 — Cascading Uncertainty Reduction (CUR)

**What it claims**: "Progressively reduce uncertainty through refinement."

**Already**: `src/search/hdr_cascade.rs` — 4-level resolution cascade.
CRP percentiles provide exact uncertainty bounds at each level.
`src/search/scientific.rs` — CI computation.

**Gap**: CRP percentiles not yet wired to cascade levels.

**Harden**: See #15 (ClusterDistribution). Wire CRP to HDR levels:
```rust
// In src/search/hdr_cascade.rs
pub fn cascading_uncertainty(
    query: &Fingerprint,
    corpus_dist: &ClusterDistribution,
) -> Vec<(HdrLevel, f32)> {  // level → remaining uncertainty
    vec![
        (HdrLevel::Int1,  1.0 - corpus_dist.p25 / TOTAL_BITS as f32),
        (HdrLevel::Int4,  1.0 - corpus_dist.p50 / TOTAL_BITS as f32),
        (HdrLevel::Int8,  1.0 - corpus_dist.p75 / TOTAL_BITS as f32),
        (HdrLevel::Int32, 1.0 - corpus_dist.p99 / TOTAL_BITS as f32),
    ]
}
```

**Science**:
- **Shannon (1948)**: Entropy reduction per measurement
- **Berry-Esseen**: Each HDR level provides known Fisher information efficiency (INT4: η=0.997)
- **Rényi (1961)**: Information-theoretic uncertainty at each resolution level

---

## #27 — Multi-Perspective Compression (MPC)

**What it claims**: "Compress multiple viewpoints into unified representation."

**Already**: `src/extensions/compress/` — Delta encoding.
`src/core/vsa.rs` — `bundle()` IS multi-perspective compression (majority vote).
panCAKES XOR-diff stores each perspective as delta from center (5-70x ratio).

**Gap**: No explicit multi-perspective bundle with per-perspective weight.

**Harden**:
```rust
// In src/core/vsa.rs — add weighted bundle
impl Fingerprint {
    /// Weighted bundle: perspectives with higher weight contribute more bits
    pub fn weighted_bundle(items: &[(&Fingerprint, f32)]) -> Fingerprint {
        let mut result = [0u64; FINGERPRINT_WORDS];
        for bit_pos in 0..TOTAL_BITS as usize {
            let word = bit_pos / 64;
            let bit = bit_pos % 64;
            let weighted_sum: f32 = items.iter()
                .map(|(fp, weight)| if (fp.data[word] >> bit) & 1 == 1 { *weight } else { -*weight })
                .sum();
            if weighted_sum > 0.0 { result[word] |= 1u64 << bit; }
        }
        Fingerprint { data: result }
    }
}
```

**Science**:
- **Kanerva (2009)**: Bundle = majority vote creates prototype
- **panCAKES** (Ishaq et al.): XOR-diff compression: 5-70x ratio storing perspectives as deltas
- **Thomas & Cover (1991)**: Information-theoretic limits on lossless compression

---

## #28 — Self-Supervised Analogical Mapping (SSAM)

**What it claims**: "Discover structural analogies between domains."

**Already**: `src/nars/inference.rs` — Analogy rule: A→B, C≈A ⊢ C→B.
`src/core/vsa.rs` — XOR binding preserves structural relationships.
`src/extensions/meta_resonance.rs` — FlowVector compares DIRECTION of change.

**Gap**: No systematic analogy search across BindSpace prefixes.

**Harden**:
```rust
// src/search/analogy.rs (new)
pub fn structural_analogy(
    relation_ab: &Fingerprint,  // A→B encoded as A⊗verb⊗B
    domain_c: &[Fingerprint],   // Candidate C fingerprints
    bs: &BindSpace,
) -> Vec<(Fingerprint, Fingerprint, f32)> {  // (C, predicted_D, strength)
    // Extract the "relation" by unbinding: verb = A⊗relation⊗B⊗A = verb⊗B... 
    // Instead: use meta-resonance to find pairs with similar FlowVector
    domain_c.iter().filter_map(|c| {
        let predicted_d = c.bind(&relation_ab); // Apply same structural relation
        let (nearest_d, dist) = bs.nearest(&predicted_d);
        let strength = 1.0 - dist as f32 / TOTAL_BITS as f32;
        if strength > 0.6 { Some((c.clone(), nearest_d, strength)) } else { None }
    }).collect()
}
```

**Science**:
- **Gentner (1983)**: Structure-mapping theory — analogies preserve relational structure
- **Plate (2003)**: XOR binding preserves relational structure: if R=A⊗B, then C⊗R recovers "B-analogue of C"
- **Turney (2006)**: "Similarity of Semantic Relations" — relational similarity vs attributional similarity

---

## #29 — Intent-Driven Reframing (IDR)

**What it claims**: "Detect user intent and reframe problem accordingly."

**Already**: `src/grammar/triangle.rs` — GrammarTriangle extracts NSM + Causality + Qualia from input.
`src/grammar/causality.rs` — CausalityFlow extracts agent/action/patient/reason.
`src/cognitive/style.rs` — 12 thinking styles match different intents.

**Gap**: No intent → style mapping function.

**Harden**:
```rust
// src/grammar/intent.rs (new)
pub fn detect_intent(triangle: &GrammarTriangle) -> ThinkingStyle {
    match (triangle.causality.agency, triangle.qualia.arousal(), triangle.causality.temporality) {
        (a, _, t) if a > 0.7 && t > 0.5 => ThinkingStyle::Analytical,  // Active future → analytical
        (_, ar, _) if ar > 0.7 => ThinkingStyle::Creative,              // High arousal → creative
        (a, _, t) if a < 0.3 && t < -0.5 => ThinkingStyle::Metacognitive, // Passive past → reflective
        (_, _, t) if t.abs() < 0.2 => ThinkingStyle::Focused,           // Present → focused
        _ => ThinkingStyle::Deliberate,                                    // Default
    }
}
```

**Science**:
- **Wierzbicka (1996)**: NSM semantic primes — universal intent categories across languages
- **Austin (1962)**: Speech act theory — utterances have illocutionary force (intent)
- **Porges (2011)**: Polyvagal theory — autonomic state (arousal) drives behavioral intent

---

## #30 — Shadow Parallel Processing (SPP)

**What it claims**: "Run background reasoning in parallel."

**Already**: `src/fabric/executor.rs` — parallel execution fabric.
`src/cognitive/seven_layer.rs` — 7 layers operate independently on shared core.
`src/fabric/butterfly.rs`, `src/fabric/subsystem.rs` — parallel subsystems.

**Gap**: No actual System 1 / System 2 split with background pre-computation.

**Harden**:
```rust
// src/fabric/shadow.rs (new)
pub struct ShadowProcessor {
    background_results: Arc<RwLock<HashMap<Fingerprint, ShadowResult>>>,
}

impl ShadowProcessor {
    /// Spawn background search that pre-computes likely next queries
    pub fn precompute(&self, current: &Fingerprint, bs: &BindSpace) {
        let current = current.clone();
        let bs = bs.clone(); // Assuming BindSpace is Arc-wrapped
        let results = self.background_results.clone();
        std::thread::spawn(move || {
            // System 1: fast, parallel, no LLM
            let neighbors = bs.top_k(&current, 10);
            for neighbor in &neighbors {
                let follow_up = bs.top_k(neighbor, 5);
                results.write().unwrap().insert(neighbor.clone(), ShadowResult { predictions: follow_up });
            }
        });
    }
}
```

**Science**:
- **Kahneman (2011)**: System 1 (fast, parallel) / System 2 (slow, sequential) — the dual-process model
- **Friston (2010)**: Free energy minimization — predictive processing runs background model
- **Rayon (2019)**: Work-stealing parallelism for Rust — the implementation substrate

---

## #31 — Iterative Counterfactual Reasoning (ICR)

**What it claims**: "Systematically explore 'what if' scenarios."

**Already**: `src/search/causal.rs` — Rung 3 IMAGINE edges.
`src/learning/causal_ops.rs` — CausalOp::Imagine* (0xA60-0xA8F).
`src/world/counterfactual.rs` — Counterfactual struct with baseline/hypothesis versions.

**Gap**: `world/counterfactual.rs` is a stub (3 structs, no logic). Need actual
iterative counterfactual generation.

**Harden**:
```rust
// In src/world/counterfactual.rs — full implementation
impl Counterfactual {
    /// Generate N counterfactual worlds by systematically varying edges
    pub fn iterate(
        base: &Fingerprint,
        edges: &[(Fingerprint, CausalVerbs)], // Edges to vary
        bs: &BindSpace,
    ) -> Vec<CounterfactualWorld> {
        edges.iter().map(|(edge_fp, verb)| {
            // Unbind edge: base ⊗ verb ⊗ edge = modified world
            let modified = base.bind(&verb.fingerprint()).bind(edge_fp);
            let (nearest_world, distance) = bs.nearest(&modified);
            CounterfactualWorld {
                intervention: edge_fp.clone(),
                verb: *verb,
                resulting_world: nearest_world,
                divergence: distance as f32 / TOTAL_BITS as f32,
                certificate: CausalCertificate::compute(base, &nearest_world),
            }
        }).collect()
    }
}
```

**Science**:
- **Pearl (2009)**: Structural Counterfactual Model (SCM) — formal counterfactual semantics
- **Lewis (1973)**: "Counterfactuals" — possible worlds semantics
- **Squires & Uhler (2023)**: GSP theorem guarantees counterfactual consistency under faithfulness

---

## #32 — Semantic Distortion Detection (SDD)

**What it claims**: "Detect when meaning has been distorted."

**Already**: `src/search/scientific.rs` — reciprocal validation (bidirectional truth checking).
`src/nars/truth.rs` — confidence erosion for unvalidated beliefs.
`src/extensions/meta_resonance.rs` — FlowVector detects direction-of-meaning change.

**Gap**: No explicit distortion metric comparing input→output fingerprint fidelity.

**Harden**:
```rust
// src/search/distortion.rs (new)
pub struct DistortionReport {
    pub information_loss: f32,     // Bits lost in transformation
    pub structural_drift: f32,     // How much structure changed
    pub semantic_flip: f32,        // How many dimensions reversed
}

pub fn detect_distortion(
    original: &Fingerprint,
    transformed: &Fingerprint,
    corpus_dist: &ClusterDistribution,
) -> DistortionReport {
    let raw_distance = original.hamming_distance(transformed);
    let expected_noise = corpus_dist.sigma * 0.004; // Berry-Esseen noise floor
    DistortionReport {
        information_loss: (raw_distance as f32 - expected_noise).max(0.0) / TOTAL_BITS as f32,
        structural_drift: raw_distance as f32 / corpus_dist.mu, // Distance relative to cluster
        semantic_flip: (raw_distance as f32 - corpus_dist.p50).max(0.0) / corpus_dist.sigma, // Z-score
    }
}
```

**Science**:
- **Shannon (1948)**: Channel capacity and noise detection
- **Berry-Esseen**: Noise floor at 0.004 distinguishes distortion from natural variation
- **Cohen (1988)**: Z-score > 2.0 = statistically significant distortion (p < 0.05)

---

## #33 — Dynamic Task Meta-Framing (DTMF)

**What it claims**: "Dynamically adjust the conceptual frame for a task."

**Already**: `src/orchestration/thinking_template.rs` — 256 template slots.
`src/cognitive/collapse_gate.rs` — auto-adjusts based on SD.
`src/nars/context.rs` — InferenceContext wires style + rung + gate + mode.

**Gap**: No frame-switching protocol triggered by gate state changes.

**Harden**:
```rust
// In src/nars/context.rs — add frame switching
impl InferenceContext {
    pub fn dynamic_reframe(&mut self, gate_history: &[GateState]) -> bool {
        // If 3 consecutive BLOCKs: reframe by jumping to different prefix
        let recent_blocks = gate_history.iter().rev().take(3).filter(|g| **g == GateState::Block).count();
        if recent_blocks >= 3 {
            self.rung = self.rung.next_band(); // Jump coarse band
            self.style = self.style.antipode(); // Flip thinking style
            return true; // Frame shift occurred
        }
        false
    }
}
```

**Science**:
- **Lakoff (2004)**: *Don't Think of an Elephant* — frames determine reasoning
- **Tversky & Kahneman (1981)**: Framing effects — same data, different frame → different conclusions
- **Ashby (1956)**: Requisite variety — frame must match system complexity

---

## #34 — Hyperdimensional Knowledge Fusion (HKF)

**What it claims**: "Fuse knowledge from different domains in high-D space."

**Already**: `src/core/vsa.rs` — `bind()`, `bundle()`, `sequence()`.
`src/extensions/spo/spo.rs` — Subject-Predicate-Object encoding: S⊗P⊗O.
`src/storage/xor_dag.rs` — XOR-DAG with ACID transactions.

**Gap**: No cross-domain fusion with measured semantic validity.

**Harden**:
```rust
// src/extensions/fusion.rs (new)
pub struct FusionResult {
    pub fused: Fingerprint,
    pub domain_a_recovery: f32,  // Can we recover domain A from fusion?
    pub domain_b_recovery: f32,  // Can we recover domain B from fusion?
    pub novelty: f32,             // How different is fusion from both parents?
    pub truth: TruthValue,        // NARS confidence in fusion validity
}

pub fn cross_domain_fuse(
    domain_a: &Fingerprint,
    domain_b: &Fingerprint,
    relation: &Fingerprint, // The verb/relation connecting them
) -> FusionResult {
    let fused = domain_a.bind(relation).bind(domain_b);
    let recovery_a = 1.0 - fused.bind(domain_b).bind(relation).hamming_distance(domain_a) as f32 / TOTAL_BITS as f32;
    let recovery_b = 1.0 - fused.bind(domain_a).bind(relation).hamming_distance(domain_b) as f32 / TOTAL_BITS as f32;
    let novelty_a = fused.hamming_distance(domain_a) as f32 / TOTAL_BITS as f32;
    let novelty_b = fused.hamming_distance(domain_b) as f32 / TOTAL_BITS as f32;
    FusionResult {
        fused,
        domain_a_recovery: recovery_a,
        domain_b_recovery: recovery_b,
        novelty: (novelty_a + novelty_b) / 2.0,
        truth: TruthValue::new((recovery_a + recovery_b) / 2.0, 0.8),
    }
}
```

**Science**:
- **Plate (2003)**: HRR binding preserves recoverability in high dimensions
- **Kanerva (2009)**: d≥10000 → near-orthogonal random vectors → binding creates valid compounds
- **Rahimi & Recht (2007)**: Random feature maps preserve kernel structure — enables cross-domain fusion

---

## Implementation Priority

### Phase 1: Cement What Exists (low effort, high value)
1. **#15** ClusterDistribution — CRP struct from CLAM_HARDENING.md §6
2. **#25** Benchmark suite — property tests for bind/unbind roundtrip
3. **#26** Wire CRP to HDR cascade levels
4. **#24** Fusion quality metric on existing `bind()`

### Phase 2: Complete the Pearl Stack (medium effort, essential)
5. **#4** Reverse causal trace with CausalCertificate
6. **#31** Full counterfactual implementation
7. **#12** Temporal effect size / Granger signal
8. **#32** Distortion detection with Berry-Esseen noise floor

### Phase 3: Debate & Meta-Cognition (medium effort, differentiating)
9. **#3** DebateRound with NARS truth value revision
10. **#7** Adversarial critic with 5 challenge types
11. **#10** MetaCognition with calibration tracking
12. **#11** Contradiction detector

### Phase 4: Cognitive Protocols (higher effort, full system)
13. **#1** Recursive expansion with convergence measurement
14. **#13** Convergent/divergent oscillation protocol
15. **#30** Shadow parallel processor
16. **#23** Gate→style feedback loop

### Phase 5: Remaining (fill as needed)
17-34: The remaining tactics either map to existing functionality (#18, #25) or
are thin wrappers around core primitives (#6, #9, #16, #17, #19, #22, #28, #29, #33, #34).

---

## Scientific Reference Stack (Complete)

### Fingerprint & VSA
- Kanerva (1988): *Sparse Distributed Memory*
- Kanerva (2009): "Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors"
- Plate (2003): *Holographic Reduced Representations*
- Kleyko et al. (2022): "Vector Symbolic Architectures as a Computing Framework for Emerging Hardware"
- Gallant & Okaywe (2013): "Representing Objects, Relations, and Sequences"

### CLAM / CAKES / panCAKES
- Ishaq et al. (2019): "Clustered Learning of Approximate Manifolds" (CLAM)
- Ishaq et al. (CHESS, CAKES, panCAKES, CHAODA): Series from UMass/MIT

### Statistics & Information Theory
- Berry (1941) / Esseen (1942): Normal approximation error bounds
- Shannon (1948): Information theory
- Fisher (1925): Sufficient statistics
- Cohen (1988): Statistical Power Analysis — effect sizes
- Brier (1950): Calibration scoring

### Causality
- Granger (1969): Temporal causality
- Pearl (2009): *Causality* — do-calculus
- Squires & Uhler (2023): GSP theorem (FoCM)
- Lewis (1973): Counterfactual semantics

### NARS
- Wang (2006): *Rigid Flexibility: The Logic of Intelligence* — NAL truth functions
- OpenNARS project: Implementation reference

### Cognitive Science
- Fleming & Dolan (2012): Metacognitive neural basis
- Kahneman (2011): System 1/System 2
- Hofstadter (1979): Strange loops
- Festinger (1957): Cognitive dissonance
- Porges (2011): Polyvagal theory
- Gentner (1983): Structure-mapping (analogy)
- Guilford (1967): Divergent/convergent production

### Machine Learning
- Sutton & Barto (2018): Reinforcement Learning
- Friston (2010): Free energy principle
- Schmidhuber (2010): Formal creativity theory
- Rahimi & Recht (2007/2008): Random features
