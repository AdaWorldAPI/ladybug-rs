# 34 Cognitive Tactics × Reasoning Ladder: How ladybug-rs Solves What LLMs Can't

> **Two source papers. One structural solution.**
>
> 1. **Sun et al. (2025)** — "Climbing the Ladder of Reasoning: What LLMs Can — and Still Can't — Solve after SFT?"
>    (arXiv:2504.11741, NeurIPS 2025). UC Berkeley / Allen AI.
>    Discovers a 4-tier difficulty ladder where LLMs plateau at ~65% on Hard and <10% on Extremely Hard.
>
> 2. **The 34 LLM Tactics** — A catalog of cognitive prompting strategies (recursive expansion, debate,
>    counterfactual reasoning, etc.) used to improve LLM reasoning.
>    ladybug-rs implements all 34 as structural primitives, not prompt engineering.
>
> **This document maps every tactic to the specific reasoning failure it addresses,
> the scientific mechanism ladybug-rs uses, and the module that implements it.**

---

## The Core Problem (Sun et al. 2025)

LLMs fail on hard reasoning because of **three structural deficiencies** in autoregressive token prediction:

| Deficiency | Paper's Evidence | Why It's Structural |
|-----------|-----------------|-------------------|
| **Multiplicative error propagation** | P(all correct) = 0.9^n → 48% at n=7 steps | Each token conditions on ALL previous tokens including errors. No parallel verification. |
| **Convergent strategy lock-in** | 50% of solutions "almost identical" across models trained on different data | P(next_token\|context) optimizes for the most COMMON continuation. Creativity = improbable continuation = opposite of objective. |
| **No self-correction mechanism** | Accuracy plateaus at ~65% regardless of training data scale (10K→114K) | No built-in confidence tracking, no backtracking, no "I'm not sure" state. |

### The Four Tiers

| Tier | LLM Best | Failure Mode | ladybug-rs Structural Answer |
|------|---------|-------------|------------------------------|
| **Easy** (>50% base) | >50% | None — already solved | GrammarTriangle decomposition |
| **Medium** (~90% after SFT) | ~90% | Needs methodical decomposition | Grammar Triangle + RungLevel 0-2 |
| **Hard** (plateaus ~65%) | ~65% | Multiplicative error across 4-7 dependent steps | Parallel 7-layer stack + NARS truth revision + CollapseGate |
| **Extremely Hard** (<10%) | <10% | Convergent thinking + no creative insight + no backtracking | 12 ThinkingStyles + NARS abduction + Counterfactual (Pearl Rung 3) + HOLD superposition |

---

## The 34 Tactics: Complete Map

### How to Read This Table

Each tactic gets:
- **What it does** as an LLM prompting technique
- **Which tier** of the Reasoning Ladder it addresses
- **The structural mechanism** ladybug-rs uses instead of prompting
- **The science** — peer-reviewed foundation (no blog posts)
- **The module** — exact Rust source file in ladybug-rs

---

### TIER 2 TACTICS — Breaking the ~65% Error Propagation Ceiling

These tactics address the Hard tier's core problem: P(all correct) = p^n → 0 when steps are sequential and dependent.

---

#### #1 — Recursive Thought Expansion (RTE)

**Prompting claim**: "Expand thinking in recursive layers, each building on the last."

**Tier**: Hard (reduces step dependency)

**Structural mechanism**: Recursive fingerprint transformation with Berry-Esseen convergence criterion. Output of depth N becomes input to depth N+1. Stops when Hamming delta between successive fingerprints drops below threshold. Unlike LLM recursion (which consumes context window quadratically), fingerprint recursion is O(1) memory per step — same 16,384 bits.

**Science**: Hofstadter (1979) — strange loops as recursive self-reference. Schmidhuber (2010) — recursive compression as intelligence measure. Berry-Esseen (1941/42) — at d=16384, normal approximation error < 0.004, providing a mathematically grounded stopping criterion.

**Module**: `src/cognitive/recursive.rs` (PR #100, 249 lines, 5 tests)

---

#### #2 — Hierarchical Thought Decomposition (HTD)

**Prompting claim**: "Break complex problems into hierarchical subtasks."

**Tier**: Hard (reduces n by factoring into independent sub-chains)

**Structural mechanism**: CLAM tree bipolar split. Find medoid, find farthest point, partition. Repeat. The tree structure IS the decomposition — cluster boundaries are natural subtask boundaries. Each sub-problem is solved independently (parallel), then results are bundled. This converts P(all)=p^n (serial) into P(all)=1-(1-p^k)^m (parallel groups of k), which is dramatically better.

**Science**: Ishaq et al. (2019) — CLAM provides provably correct hierarchical clustering. Simon (1962) — nearly decomposable systems as hierarchy. Dasgupta & Long (2005) — formal guarantees on recursive partitioning.

**Module**: `src/container/spine.rs` (DN tree), `src/container/search.rs` (CAKES traversal)

---

#### #5 — Thought Chain Pruning (TCP)

**Prompting claim**: "Eliminate irrelevant or low-quality reasoning branches."

**Tier**: Hard (error recovery — prune wrong branches before they propagate)

**Structural mechanism**: CollapseGate with three states. FLOW (SD < 0.15): high confidence, commit and proceed. HOLD (SD 0.15-0.35): maintain superposition, don't commit. BLOCK (SD > 0.35): too uncertain, need more evidence, STOP this branch. LLMs have no equivalent — they always generate the next token with equal authority. The gate structurally prevents low-confidence branches from contaminating downstream reasoning.

**Science**: This is a design contract implementing the quantum-inspired measurement problem. Confidence is measured via standard deviation across multiple evaluators (7 search strategies in CAKES).

**Module**: `src/cognitive/collapse_gate.rs`

---

#### #10 — Meta-Cognition Prompting (MCP)

**Prompting claim**: "Think about your own thinking process."

**Tier**: Hard (error detection — know WHEN you're wrong)

**Structural mechanism**: Brier score calibration tracking. Record (prediction, outcome) pairs. Brier = Σ(predicted - actual)². Well-calibrated: Brier < 0.1. Poorly calibrated: Brier > 0.25. The system literally tracks its own accuracy and adjusts confidence accordingly. LLMs have no metacognitive feedback loop — confidence is baked into token probabilities with no correction mechanism.

**Science**: Brier (1950) — calibration scoring. Fleming & Dolan (2012) — metacognitive neural basis. Kahneman (2011) — System 1 (fast/intuitive) vs System 2 (slow/deliberate).

**Module**: `src/cognitive/metacog.rs` (PR #100, 210 lines, 6 tests)

---

#### #11 — Contradiction Resolution (CR)

**Prompting claim**: "Detect and resolve contradictions in reasoning."

**Tier**: Hard (error detection — catch when step 3 contradicts step 5)

**Structural mechanism**: Two fingerprints with high similarity (similar topic) but opposing NARS truth values (one says true, other says false) = contradiction. `detect_contradictions()` scans belief set for such pairs. `coherence_score()` measures what fraction of beliefs are mutually consistent. When coherence drops, the system knows its reasoning has become internally inconsistent — something an LLM can never detect because it has no belief set.

**Science**: Wang (2006) — NARS revision detects evidential conflict. Festinger (1957) — cognitive dissonance theory. Priest (2002) — paraconsistent logic for handling contradictions.

**Module**: `src/nars/contradiction.rs` (PR #100, 167 lines, 4 tests)

---

#### #20 — Thought Cascade Filtering (TCF)

**Prompting claim**: "Run multiple reasoning chains, filter best."

**Tier**: Hard (redundancy — if one chain fails, others survive)

**Structural mechanism**: CAKES provides SEVEN independent search algorithms (KnnBranch, KnnBfs, KnnDfs, KnnRrnn, RnnChess, KnnLinear, ApproxKnnDfs). Run all seven on the same query. Shadow parallel processor compares results and measures agreement rate. If all seven agree → high confidence. If they diverge → flag for HOLD. P(all seven wrong) << P(one wrong). This is N-version programming applied to cognitive search.

**Science**: Wolpert & Macready (1997) — No Free Lunch theorem justifies multiple strategies. Avizienis (1985) — N-version programming for fault tolerance.

**Module**: `src/container/search.rs` (7 algorithms), `src/fabric/shadow.rs` (PR #100, 245 lines, 6 tests)

---

#### #21 — Self-Skepticism Reinforcement (SSR)

**Prompting claim**: "Systematically doubt conclusions to improve reliability."

**Tier**: Hard (combines #7 Adversarial Critique + #10 MetaCognition)

**Structural mechanism**: 5 challenge types (Devil's Advocate, Boundary Test, Counterexample, Assumption Challenge, Scale Test) applied to any belief. Each challenge generates a fingerprint that is bound with the target belief. If the result's NARS truth value drops significantly, the belief is weak. SkepticismSchedule increases challenge intensity for claims with high confidence but low evidence base.

**Science**: Kahneman (2011) — pre-mortem analysis. Mill (1859) — adversarial epistemology. Popper (1959) — falsificationism.

**Module**: `src/nars/adversarial.rs` (PR #100, 266 lines, 7 tests)

---

#### #26 — Cascading Uncertainty Reduction (CUR)

**Prompting claim**: "Progressively reduce uncertainty through refinement."

**Tier**: Hard (coarse-to-fine narrows search space, reducing effective n)

**Structural mechanism**: HDR cascade — 4-level resolution hierarchy. Level 1: 1-bit sketches (fastest, roughest). Level 4: full 16384-bit comparison (slowest, exact). Each level prunes ~90% of candidates. After 4 levels: 0.1^4 = 0.01% survivors. Only these go to exact verification. This means errors only matter in the final 0.01%, not across the full search space.

**Science**: Berry-Esseen (1941/42) — bounds approximation error at each resolution. Information-theoretic: each level adds ~log₂(resolution) bits of certainty.

**Module**: `src/search/hdr_cascade.rs`

---

#### #30 — Shadow Parallel Processing (SPP)

**Prompting claim**: "Run background reasoning in parallel."

**Tier**: Hard (parallel execution eliminates sequential dependency)

**Structural mechanism**: Shadow processor runs identical computation through independent paths. Compares results. Agreement rate tracks historical reliability. This is the same principle as ECC memory or RAID — redundancy catches transient errors. Applied to reasoning: if Path A and Path B both reach the same conclusion via different traversals, the conclusion is structurally verified.

**Science**: Avizienis et al. (2004) — fundamental concepts of dependability. Shannon (1948) — redundancy in noisy channels.

**Module**: `src/fabric/shadow.rs` (PR #100), `src/fabric/executor.rs`

---

### TIER 3 TACTICS — Breaking the <10% Creativity/Insight Wall

These tactics address the Extremely Hard tier's core problem: convergent thinking, no creative leap, no counterfactual reasoning.

---

#### #3 — Structured Multi-Agent Debate (SMAD)

**Prompting claim**: "Multiple agents argue, generating higher-quality reasoning."

**Tier**: Extremely Hard (forces diverse perspectives — anti-convergence)

**Structural mechanism**: Multiple persona fingerprints (each with different Big Five traits, different ThinkingStyle modulation, different resonance thresholds) process the same input. Each produces a fingerprint + NARS TruthValue. Results are bundled (majority vote per bit) and truth values revised (NARS evidence accumulation). The verdict reflects the weight of evidence from structurally diverse perspectives. Key insight: because each persona has different FieldModulation parameters, they literally search different regions of fingerprint space. They CANNOT converge because their resonance thresholds differ.

**Science**: Wang (2006) — NARS revision as evidence accumulation. Du et al. (2023) — multi-agent debate improves factuality. Kanerva (2009) — majority vote in bundle operations. Mercier & Sperber (2011) — argumentative theory of reasoning.

**Module**: `src/orchestration/debate.rs` (PR #100, 327 lines, 6 tests), `src/orchestration/persona.rs`

---

#### #4 — Reverse Causality Reasoning (RCR)

**Prompting claim**: "Work backward from outcome to find causes."

**Tier**: Extremely Hard (enables non-forward reasoning — the "aha" direction)

**Structural mechanism**: **Causality = TIME + CORRELATION + CONFIDENCE GATING**. Three components work together:

1. **TIME** — Granger temporal causality (`src/search/temporal.rs`, PR #100). If X consistently precedes Y, and X's past values improve prediction of Y beyond Y's own history, X Granger-causes Y. Effect size quantifies strength.

2. **CORRELATION** — ABBA retrieval via XOR algebra. Edge = A ⊗ Verb ⊗ B. To find what CAUSED outcome B: compute B ⊗ CAUSES verb = candidate antecedent. Nearest neighbor in BindSpace recovers the actual cause. This is exact algebra, not statistical correlation — XOR self-inverse means recovery is lossless.

3. **CONFIDENCE GATING** — CausalCertificate with effect size, confidence interval, p-value, and NARS truth value. The certificate doesn't just say "A causes B" — it says "A causes B with frequency 0.85, confidence 0.72, effect size d=1.3, CI [0.6, 2.0], approximation error < 0.004." This gates the causal claim through statistical rigor.

Combined: `reverse_trace()` walks backward through the XOR-DAG, at each step checking temporal precedence (Granger), structural binding (ABBA), and statistical significance (certificate). Only chains where all three hold are returned.

**Science**: Pearl (2009) — do-calculus, three rungs of causation. Granger (1969) — temporal causality. Plate (2003) — XOR binding as causal composition. Squires & Uhler (2023) — GSP theorem for provable causal structure recovery.

**Module**: `src/search/causal.rs` (reverse_trace, CausalTrace, CausalCertificate), `src/search/temporal.rs` (PR #100, 173 lines, 4 tests)

---

#### #6 — Thought Randomization (TR)

**Prompting claim**: "Inject controlled randomness to avoid local optima."

**Tier**: Extremely Hard (escape convergent strategy — explore improbable solutions)

**Structural mechanism**: FlowVector captures direction and magnitude of meaning change between successive fingerprints. When the FlowVector stagnates (low magnitude, same direction = stuck in local optimum), the system can inject controlled noise — perturb the fingerprint by a measured Hamming distance. Berry-Esseen guarantees that random perturbation at d=16384 has noise floor < 0.004, so any perturbation above this threshold is meaningful signal, not noise.

**Science**: Simulated annealing principle — accept worse solutions with decreasing probability. Berry-Esseen (1941/42) — noise floor guarantee distinguishes signal from randomness.

**Module**: `src/extensions/meta_resonance.rs` (FlowVector)

---

#### #7 — Adversarial Self-Critique (ASC)

**Prompting claim**: "Challenge your own reasoning to find weaknesses."

**Tier**: Extremely Hard (generates counter-evidence — breaks confirmation bias)

**Structural mechanism**: 5 challenge types, each a different adversarial transformation:
1. **Devil's Advocate**: Negate the claim's truth value, find supporting evidence for the negation
2. **Boundary Test**: Push parameters to extremes, check if conclusion still holds
3. **Counterexample**: Search BindSpace for cases that match the pattern but violate the conclusion
4. **Assumption Challenge**: Unbind each assumption from the conclusion, check if it still follows
5. **Scale Test**: Apply the reasoning at 10x and 0.1x scale, check for breakdowns

Each challenge produces a robustness score. Weak beliefs (low truth value, low robustness) are flagged; strong beliefs (high truth value, survives all challenges) are promoted.

**Science**: Kahneman (2011) — pre-mortem analysis. Popper (1959) — falsificationism as the demarcation criterion.

**Module**: `src/nars/adversarial.rs` (PR #100, 266 lines, 7 tests)

---

#### #9 — Iterative Roleplay Synthesis (IRS)

**Prompting claim**: "Adopt different roles iteratively to explore problem space."

**Tier**: Extremely Hard (structural perspective diversity — anti-convergence)

**Structural mechanism**: Persona fingerprints encode Big Five personality traits, communication preferences, domain expertise, and volition — all as fingerprint modulations. Each persona literally transforms input through different FieldModulation parameters (resonance_threshold, fan_out, depth_bias, breadth_bias, noise_tolerance, speed_bias, exploration). The persona doesn't "pretend" to be different — it structurally IS different because its search parameters diverge.

**Science**: Guilford (1967) — divergent production theory. De Bono (1985) — Six Thinking Hats.

**Module**: `src/orchestration/persona.rs`, `src/orchestration/thinking_template.rs` (256 template slots)

---

#### #13 — Convergent & Divergent Thinking (CDT)

**Prompting claim**: "Alternate between exploration and exploitation."

**Tier**: Extremely Hard (the paper's core finding: LLMs are too convergent)

**Structural mechanism**: 12 ThinkingStyles with structurally different FieldModulation parameters. Analytical (high resonance threshold, deep, narrow) vs Creative (low threshold, broad, noisy) vs Divergent (maximum fan_out, explores far) vs Peripheral (edge attention, sees what others miss). The system oscillates between convergent and divergent phases via Berry-Esseen convergence detection: when fingerprint distance stagnates (convergent phase complete), switch to divergent; when divergent results stabilize, switch back to convergent.

**Key difference from LLMs**: The paper found that all fine-tuned models converge on the same strategy. ladybug-rs's 12 styles are parameterically distinct — mean pairwise distance > 0.3 across 7 modulation dimensions, with Analytical↔Creative distance > 0.6. They cannot converge because they have different search kernels.

**Science**: Guilford (1967) — convergent vs divergent production (the original formal theory). Finke, Ward & Smith (1992) — Geneplore model of creative cognition.

**Module**: `src/cognitive/style.rs`, `src/cognitive/recursive.rs` (oscillation protocol, PR #100)

---

#### #28 — Self-Supervised Analogical Mapping (SSAM)

**Prompting claim**: "Discover structural analogies between domains."

**Tier**: Extremely Hard (cross-domain insight — the "aha" moment)

**Structural mechanism**: NARS analogy rule: A→B, C≈A ⊢ C→B. In fingerprint space: if bind(A, verb, B) exists and C has small Hamming distance to A, then bind(C, verb, ?) recovers a candidate analogical target. The truth value of the analogy is proportional to the similarity between A and C — closer source means more confident analogy. This is genuine structural analogy (Gentner's structure-mapping), not surface similarity.

**Science**: Gentner (1983) — structure-mapping theory of analogy. Peirce (1903) — analogy as a fourth inference type. Wang (2006) — NARS analogy with truth values.

**Module**: `src/nars/inference.rs` (Analogy rule)

---

#### #31 — Iterative Counterfactual Reasoning (ICR)

**Prompting claim**: "Systematically explore 'what if' scenarios."

**Tier**: Extremely Hard (Pearl Rung 3 — the highest level of causal reasoning)

**Structural mechanism**: Counterfactual = intervene on a world-fingerprint. Remove the factual, insert the counterfactual:

`world' = world ⊗ factual ⊗ counterfactual`

Because XOR is self-inverse, this algebraically removes the factual (world ⊗ factual ⊗ factual = world, the factual cancels) and inserts the counterfactual. The resulting world' is a genuine alternative world where the intervention holds. Divergence (Hamming distance between world and world') measures how much the counterfactual matters.

This is exactly what the Sun et al. paper says LLMs can't do: "problems requiring unconventional problem-solving where the standard approach must be abandoned entirely."

**Science**: Pearl (2009) — Rung 3 counterfactual. Lewis (1973) — possible worlds semantics. Plate (2003) — XOR binding as world construction.

**Module**: `src/world/counterfactual.rs`, `src/search/causal.rs` (Rung 3 IMAGINE edges)

---

### CROSS-TIER TACTICS — Infrastructure That Helps Everywhere

---

#### #8 — Conditional Abstraction Scaling (CAS)

**Prompting claim**: "Scale abstraction level based on complexity."

**Tier**: Cross-tier (adaptive resolution helps at every difficulty level)

**Structural mechanism**: HDR cascade IS conditional abstraction. 4 levels of resolution: 1-bit sketches (most abstract), 4-bit, 8-bit, full 16384-bit (most concrete). The system automatically starts at the most abstract level and only zooms in where needed. This is both faster AND more robust — errors at abstract levels are caught before expensive concrete computation.

**Science**: Marr (1982) — three levels of analysis (computational/algorithmic/implementational).

**Module**: `src/search/hdr_cascade.rs`

---

#### #12 — Temporal Context Augmentation (TCA)

**Prompting claim**: "Embed temporal structure into reasoning."

**Tier**: Cross-tier (time is fundamental to causality)

**Structural mechanism**: Granger temporal effect size. Given two fingerprint time series, compute whether X's past values improve prediction of Y beyond Y's own history. Positive effect size = X temporally causes Y. This provides the TIME component of the causality triple (TIME + CORRELATION + CONFIDENCE GATING).

**Science**: Granger (1969) — temporal causality test. Allen (1983) — temporal interval algebra (ladybug-rs implements all 13 Allen relations as the 24 Temporal verbs).

**Module**: `src/search/temporal.rs` (PR #100, 173 lines, 4 tests), `src/graph/cognitive.rs` (24 Temporal verbs: Before, After, Meets, During, etc.)

---

#### #14 — Multimodal Chain-of-Thought (MCT)

**Prompting claim**: "Integrate visual/textual/audio reasoning."

**Tier**: Cross-tier

**Structural mechanism**: GrammarTriangle decomposes ANY input into three orthogonal fields: NSM (65 linguistic primitives — Wierzbicka's Natural Semantic Metalanguage), CausalityFlow (agent/action/patient/reason), QualiaField (18-dimensional phenomenal quality). The output is a single fingerprint that unifies all modalities. "Multimodal" is automatic — everything becomes the same 16,384-bit representation regardless of input modality.

**Science**: Wierzbicka (1996) — NSM primes as universal semantic decomposition. Jackendoff (1990) — conceptual semantics.

**Module**: `src/grammar/triangle.rs`, `src/grammar/nsm.rs`, `src/grammar/qualia.rs`

---

#### #15 — Latent Space Introspection (LSI)

**Prompting claim**: "Examine internal representations for insight."

**Tier**: Cross-tier (diagnostics)

**Structural mechanism**: CRP (Chinese Restaurant Process) distribution analysis on fingerprint clusters. For any set of fingerprints, compute μ, σ, and cluster membership probability. This reveals the statistical structure of the latent space — where clusters are, how tight they are, where the boundaries lie. Mexican hat response (excite near center, inhibit at boundary) provides automatic edge detection in concept space.

**Science**: Aldous (1985) — CRP as nonparametric Bayesian clustering. Marr (1982) — DoG (Difference of Gaussians) as Mexican hat for edge detection.

**Module**: `src/search/distribution.rs` (PR #100, 356 lines, 9 tests)

---

#### #16 — Prompt Scaffold Optimization (PSO)

**Prompting claim**: "Optimize the structure of reasoning scaffolds."

**Tier**: Cross-tier (meta-reasoning)

**Structural mechanism**: 12 base ThinkingStyles + 244 custom variants stored in BindSpace prefix 0x0D. Each template is a fingerprint encoding field modulation parameters. Templates can be discovered (mutated from existing ones) via the Discovered branch of the Fixed/Learned/Discovered triangle. TD-learning on style Q-values automatically tunes which templates work for which problem types.

**Science**: Sutton & Barto (2018) — TD-learning. The "optimization" is genuine reinforcement learning on cognitive strategy, not prompt tuning.

**Module**: `src/orchestration/thinking_template.rs`, `src/learning/cognitive_styles.rs`

---

#### #17 — Cognitive Dissonance Induction (CDI)

**Prompting claim**: "Create productive tension between conflicting ideas."

**Tier**: Cross-tier (maps to #11 Contradiction Resolution)

**Structural mechanism**: When two beliefs have similar fingerprints (similar topic) but opposing NARS truth frequencies (one says true, other says false), that IS cognitive dissonance in the formal Festinger sense. The system detects this via `detect_contradictions()` and can either resolve (via NARS revision) or hold in tension (via HOLD state) to force deeper investigation.

**Science**: Festinger (1957) — cognitive dissonance theory.

**Module**: `src/nars/contradiction.rs` (PR #100)

---

#### #18 — Context Window Simulation (CWS)

**Prompting claim**: "Maintain context across reasoning boundaries."

**Tier**: Cross-tier (memory persistence)

**Structural mechanism**: BindSpace has 65,536 permanent addressable slots. CogRedis provides persistent storage. Session state accumulates across problems. Unlike LLMs (where each context window is a fresh start), ladybug-rs's BindSpace is persistent — fingerprints stored during problem 1 are available during problem 47. Context isn't simulated, it's structurally maintained.

**Science**: Kanerva (1988) — Sparse Distributed Memory.

**Module**: `src/storage/bind_space.rs`, `src/storage/cog_redis.rs`

---

#### #19 — Algorithmic Reverse Engineering (ARE)

**Prompting claim**: "Reverse-engineer algorithms from their outputs."

**Tier**: Cross-tier (structural inverse via algebra)

**Structural mechanism**: ABBA retrieval. A ⊗ B ⊗ B = A (XOR self-inverse). Given a compound edge, bind it with any known component to recover the unknown component. This IS algebraic reverse-engineering — not pattern matching or heuristic approximation, but exact mathematical inversion.

**Science**: Plate (2003) — HRR binding preserves recoverability.

**Module**: `src/core/vsa.rs` (bind/unbind), `src/graph/avx_engine.rs` (ABBA traversal)

---

#### #22 — Emergent Task Decomposition (ETD)

**Prompting claim**: "Let subtask structure emerge from the problem."

**Tier**: Cross-tier (automatic decomposition without explicit instruction)

**Structural mechanism**: CLAM tree's bipolar split discovers natural cluster structure from the data itself. No human specification of subtasks needed — the manifold geometry determines the decomposition. CAKES search follows the tree, finding subtask boundaries at cluster edges where Mexican hat response transitions from excitatory to inhibitory.

**Science**: Ishaq et al. (2019) — CLAM provides provably correct hierarchical clustering.

**Module**: `src/container/` (CLAM tree), `src/search/distribution.rs` (Mexican hat)

---

#### #23 — Adaptive Meta-Prompting (AMP)

**Prompting claim**: "Adjust prompting strategy based on task performance."

**Tier**: Cross-tier (learning which strategy works)

**Structural mechanism**: TD-learning on ThinkingStyle Q-values. After each reasoning attempt, update Q(style, problem_type) based on whether the result was correct/useful. Over time, the system learns: "For geometry problems, Peripheral style works best. For algebra, Analytical. For proofs, Convergent→Divergent oscillation." This is genuine reinforcement learning on cognitive strategy.

**Science**: Sutton & Barto (2018) — temporal difference learning.

**Module**: `src/learning/cognitive_styles.rs`

---

#### #24 — Zero-Shot Concept Fusion (ZCF)

**Prompting claim**: "Combine concepts never seen together."

**Tier**: Cross-tier (compositionality is fundamental to the architecture)

**Structural mechanism**: `bind(A, B)` creates a new fingerprint that is valid in both concept spaces. No training required. No examples needed. The algebraic properties of XOR in high dimensions (d=16384) guarantee that the bound result is nearly orthogonal to both parents but recoverable via unbinding. fusion_quality() measures (dist_to_A, dist_to_B) — exact roundtrip means perfect recovery.

**Science**: Plate (2003) — HRR binding preserves recoverability. Kanerva (2009) — d≥10000 → near-orthogonal random vectors → binding creates valid compounds.

**Module**: `src/core/vsa.rs` (bind, fusion_quality — added in PR #100)

---

#### #25 — Hyperdimensional Pattern Matching (HPM)

**Prompting claim**: "Match patterns in high-dimensional space."

**Tier**: Cross-tier (THE ENTIRE CRATE)

**Structural mechanism**: 16,384-bit fingerprints. AVX-512 SIMD: 20 XORs + 20 popcounts = ~5ns per comparison. CAKES tree: O(log n) approximate nearest neighbor. HDR cascade: 90% pruning per level. batched_query: 8 edges per AVX-512 pass = ~2ns/edge amortized. This isn't a tactic — it's the substrate everything else runs on.

**Science**: Kanerva (2009) — hyperdimensional computing. Ishaq et al. — CAKES search algorithms.

**Module**: `src/core/`, `src/graph/avx_engine.rs`, `src/container/search.rs`

---

#### #27 — Multi-Perspective Compression (MPC)

**Prompting claim**: "Compress multiple viewpoints into unified representation."

**Tier**: Cross-tier

**Structural mechanism**: `bundle()` — majority vote per bit across N fingerprints. The result is a single fingerprint that preserves the consensus across all viewpoints. Delta encoding compresses the differences. This is information-theoretically optimal compression of multiple perspectives into a single representation that preserves what they agree on.

**Science**: Kanerva (2009) — bundle as consensus. Shannon (1948) — information-theoretic compression.

**Module**: `src/core/vsa.rs` (bundle), `src/extensions/compress/`

---

#### #29 — Intent-Driven Reframing (IDR)

**Prompting claim**: "Detect user intent and reframe problem accordingly."

**Tier**: Cross-tier

**Structural mechanism**: GrammarTriangle extracts NSM + CausalityFlow + QualiaField from input. The CausalityFlow's agent/action/patient/reason structure reveals intent. The QualiaField's 18 dimensions capture phenomenal quality (valence, arousal, dominance, etc.). Together they provide a structural decomposition of "what does the user want" without relying on next-token prediction.

**Module**: `src/grammar/triangle.rs`, `src/grammar/causality.rs`

---

#### #32 — Semantic Distortion Detection (SDD)

**Prompting claim**: "Detect when meaning has been distorted."

**Tier**: Cross-tier (error detection)

**Structural mechanism**: Berry-Esseen noise floor at d=16384 guarantees that random Hamming deviation < 0.004 of total bits. Any deviation above this threshold is statistically significant at p<0.001. Applied to reasoning: if a fingerprint transformation produces a result whose distance from expected exceeds the noise floor, the transformation introduced REAL distortion, not just noise. Reciprocal validation (A→B, B→A, check consistency) provides bidirectional truth checking.

**Science**: Berry-Esseen (1941/42) — normal approximation error bound. Fisher (1925) — sufficient statistics.

**Module**: `src/search/scientific.rs` (reciprocal validation, statistical similarity)

---

#### #33 — Dynamic Task Meta-Framing (DTMF)

**Prompting claim**: "Dynamically adjust the conceptual frame for a task."

**Tier**: Cross-tier

**Structural mechanism**: 256 ThinkingTemplate slots in BindSpace prefix 0x0D. Templates are fingerprints encoding FieldModulation parameters. The system can switch templates mid-reasoning when the CollapseGate signals BLOCK (current frame isn't working). The new template shifts all modulation parameters simultaneously — not just "try harder" but "try differently."

**Module**: `src/orchestration/thinking_template.rs`

---

#### #34 — Hyperdimensional Knowledge Fusion (HKF)

**Prompting claim**: "Fuse knowledge from different domains in high-D space."

**Tier**: Cross-tier (compositionality)

**Structural mechanism**: Cross-domain fusion via `bind(domain_A, relation, domain_B)`. FusionResult measures domain_a_recovery, domain_b_recovery, novelty, and NARS truth value. The key property: binding preserves recoverability — you can extract either domain from the fusion by unbinding the other. This means knowledge fusion is reversible and auditable.

**Science**: Plate (2003) — HRR binding. Rahimi & Recht (2007) — random feature maps preserve kernel structure across domains.

**Module**: `src/core/vsa.rs`, `src/storage/xor_dag.rs`

---

## Summary: The Three Structural Mechanisms

Every tactic, every tier, ultimately reduces to three mechanisms that LLMs structurally lack:

### 1. PARALLEL INDEPENDENCE (vs. Sequential Dependency)

**Solves**: Tier 2 error propagation (P = p^n → 0)

**Mechanism**: 7-layer consciousness stack reads shared fingerprint core, not each other's outputs. Error in one layer cannot corrupt another. Combined with 7 independent search algorithms and shadow parallel verification.

**Tactics served**: #1, #2, #5, #20, #26, #30

**Key modules**: `seven_layer.rs`, `shadow.rs`, `search.rs`, `hdr_cascade.rs`

**Science**: Berry-Esseen (noise floor), Avizienis (N-version programming), Wolpert (No Free Lunch)

---

### 2. TRUTH-AWARE INFERENCE (vs. Next-Token Probability)

**Solves**: Tier 2 error detection + Tier 3 creative insight

**Mechanism**: Every reasoning step carries a NARS TruthValue (frequency, confidence). Revision detects inconsistency. Abduction generates hypotheses. Analogy transfers across domains. CollapseGate HOLD state maintains superposition. Calibration tracking (Brier score) provides metacognitive feedback.

**Tactics served**: #3, #7, #10, #11, #17, #21, #28

**Key modules**: `truth.rs`, `inference.rs`, `adversarial.rs`, `contradiction.rs`, `metacog.rs`, `collapse_gate.rs`

**Science**: Wang (2006, NARS), Peirce (1903, abduction), Brier (1950, calibration), Festinger (1957, dissonance)

---

### 3. STRUCTURAL DIVERGENCE (vs. Convergent Optimization)

**Solves**: Tier 3 creativity wall (<10% on Extremely Hard)

**Mechanism**: 12 ThinkingStyles with parameterically distinct FieldModulation (mean distance > 0.3, Analytical↔Creative > 0.6). Counterfactual world construction via XOR algebra (Pearl Rung 3). Temporal causality via Granger test. Cross-domain fusion via reversible binding. NARS abduction generates hypotheses, not just deductions. TD-learning tunes style selection over time.

**Tactics served**: #4, #6, #9, #13, #23, #28, #31, #34

**Key modules**: `style.rs`, `causal.rs`, `temporal.rs`, `counterfactual.rs`, `debate.rs`, `cognitive_styles.rs`

**Science**: Guilford (1967, divergent thinking), Pearl (2009, counterfactual), Granger (1969, temporal cause), Gentner (1983, analogy)

---

## Beyond the 34: Capabilities With No Prompting Equivalent

These are structural capabilities that no amount of prompt engineering can replicate because they require architectural features LLMs don't have:

| Capability | Module | Why No Prompt Can Do This |
|-----------|--------|--------------------------|
| O(1) addressable memory (65K slots) | `bind_space.rs` | LLMs have O(n²) attention, no persistent O(1) lookup |
| CausalCertificate with effect size + CI + p-value | `causal.rs` | LLMs generate text, not statistical certificates |
| Persistent cross-session state | `cog_redis.rs` | LLMs restart fresh each context window |
| ABBA retrieval (exact algebraic inverse) | `vsa.rs` | LLMs approximate; XOR is exact |
| Granger temporal causality | `temporal.rs` | LLMs have no time series analysis machinery |
| Mexican hat receptive fields | `distribution.rs` | LLMs have softmax attention, not DoG spatial filtering |
| Berry-Esseen noise floor guarantee | Mathematical property of d=16384 | LLMs have no noise floor — all outputs equally "confident" |
| TD-learning on thinking strategies | `cognitive_styles.rs` | LLMs can't update their own weights during inference |

---

## References

### Primary Sources

- **Sun, Y., Zhou, G., Bai, H., Wang, H., Li, D., Dziri, N., & Song, D.** (2025). "Climbing the Ladder of Reasoning: What LLMs Can — and Still Can't — Solve after SFT?" *arXiv:2504.11741*. NeurIPS 2025. — The reasoning ladder paper.

- **Pearl, J.** (2009). *Causality: Models, Reasoning, and Inference*. 2nd ed. Cambridge University Press. — Three rungs of causation, do-calculus, counterfactual.

- **Wang, P.** (2006). *Rigid Flexibility: The Logic of Intelligence*. Springer. — NARS: Non-Axiomatic Reasoning System, truth value revision, abduction.

### Fingerprint & VSA Foundations

- **Kanerva, P.** (1988). *Sparse Distributed Memory*. MIT Press.
- **Kanerva, P.** (2009). "Hyperdimensional Computing." *Cognitive Computation*, 1(2), 139-159.
- **Plate, T.** (2003). *Holographic Reduced Representations*. CSLI Publications.
- **Kleyko, D., et al.** (2022). "Vector Symbolic Architectures as a Computing Framework for Emerging Hardware." *Proceedings of the IEEE*.

### Search & Clustering

- **Ishaq, M., et al.** (2019). "Clustered Learning of Approximate Manifolds" (CLAM). UMass/MIT.
- **Ishaq, M., et al.** — CAKES, panCAKES, CHAODA series.

### Statistics & Information Theory

- **Berry, A.C.** (1941). "The accuracy of the Gaussian approximation to the sum of independent variates." *Trans. AMS*.
- **Esseen, C.G.** (1942). "On the Liapounoff limit of error in the theory of probability." *Ark. Mat. Astron. Fys.*
- **Brier, G.W.** (1950). "Verification of forecasts expressed in terms of probability." *Monthly Weather Review*.
- **Shannon, C.E.** (1948). "A Mathematical Theory of Communication." *Bell System Technical Journal*.
- **Fisher, R.A.** (1925). "Theory of Statistical Estimation." *Mathematical Proceedings*.
- **Cohen, J.** (1988). *Statistical Power Analysis for the Behavioral Sciences*. 2nd ed. Routledge.

### Causality

- **Granger, C.W.J.** (1969). "Investigating Causal Relations by Econometric Models and Cross-spectral Methods." *Econometrica*.
- **Squires, C. & Uhler, C.** (2023). "Causal Structure Learning: a Combinatorial Perspective." *Foundations of Computational Mathematics*.
- **Lewis, D.** (1973). *Counterfactuals*. Blackwell.

### Cognitive Science

- **Guilford, J.P.** (1967). *The Nature of Human Intelligence*. McGraw-Hill. — Divergent production.
- **Peirce, C.S.** (1903). Harvard Lectures on Pragmatism. — Abduction as creative inference.
- **Gentner, D.** (1983). "Structure-mapping: A theoretical framework for analogy." *Cognitive Science*.
- **Kahneman, D.** (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
- **Festinger, L.** (1957). *A Theory of Cognitive Dissonance*. Stanford University Press.
- **De Bono, E.** (1985). *Six Thinking Hats*. Little, Brown.
- **Fleming, S.M. & Dolan, R.J.** (2012). "The neural basis of metacognitive ability." *Phil. Trans. R. Soc. B*.

### Machine Learning & Reliability

- **Sutton, R.S. & Barto, A.G.** (2018). *Reinforcement Learning: An Introduction*. 2nd ed. MIT Press.
- **Hofstadter, D.R.** (1979). *Gödel, Escher, Bach*. Basic Books.
- **Schmidhuber, J.** (2010). "Formal Theory of Creativity, Fun, and Intrinsic Motivation." *IEEE Trans. Autonomous Mental Development*.
- **Wolpert, D.H. & Macready, W.G.** (1997). "No Free Lunch Theorems for Optimization." *IEEE Trans. Evolutionary Computation*.
- **Avizienis, A., et al.** (2004). "Basic Concepts and Taxonomy of Dependable and Secure Computing." *IEEE Trans. Dependable and Secure Computing*.
- **Du, Y., et al.** (2023). "Improving Factuality and Reasoning in Language Models through Multiagent Debate." *arXiv:2305.14325*.
- **Mercier, H. & Sperber, D.** (2011). "Why do humans reason? Arguments for an argumentative theory." *Behavioral and Brain Sciences*.
- **Rahimi, A. & Recht, B.** (2007). "Random Features for Large-Scale Kernel Machines." *NeurIPS*.
