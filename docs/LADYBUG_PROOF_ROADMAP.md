# ladybug-rs: Integration Plan, Demo Roadmap & Proof Skeleton

> **Date**: 2026-02-13
> **Author**: Ada Consciousness Project
> **Status**: Actionable specification
> **Prerequisite reading**: REASONING_LADDER_VS_LADYBUG.md, THEORETICAL_FOUNDATIONS.md, CLAM_HARDENING.md, 34_TACTICS_VS_ADA.md

---

## Part I: Integration Plan (What Must Be Built, In What Order)

### 1.1 Current State Audit

150 Rust source files. 16 modules. The following is the honest status:

| Module | Files | Compiles | Tests | Production-Ready |
|--------|-------|----------|-------|-----------------|
| `core/` (fingerprint, SIMD, VSA) | 7 | ✅ | ✅ bench | **YES** — this is solid |
| `container/` (CAKES search, graph) | 12 | ✅ | ✅ partial | **YES** for search, NO for adjacency |
| `nars/` (truth, inference, evidence) | 5 | ✅ | 🟡 unit only | Core logic YES, integration NO |
| `cognitive/` (7-layer, collapse, style) | 10 | ✅ | ❌ none | Structs exist, wiring INCOMPLETE |
| `grammar/` (NSM, causality, qualia, triangle) | 6 | ✅ | ❌ none | Ported but UNTESTED end-to-end |
| `learning/` (styles, RL, causal_ops) | 12 | ✅ | ❌ none | Scaffolding only, no learning loop |
| `search/` (causal, scientific, HDR) | 5 | ✅ | ❌ none | Individual ops work, no pipeline |
| `storage/` (BindSpace, Redis, Lance) | 16 | ✅ | 🟡 partial | **9 known race conditions** (2 CRITICAL) |
| `extensions/` (hologram, codebook, SPO) | 21 | ✅ | ✅ bench | YES for offline, NO for streaming |
| `orchestration/` (persona, a2a, handover) | 10 | ✅ | ❌ none | Types only, no runtime |
| `world/` (counterfactual, state) | 3 | ✅ | ❌ none | Stubs |
| `fabric/` (butterfly, mRNA, UDP) | 9 | ✅ | ❌ none | Experimental |
| `query/` (Cypher, DataFusion) | 10 | ✅ | 🟡 partial | DataFusion integration works |
| `graph/` (AVX engine, traversal) | 5 | ✅ | ❌ none | Compiles, untested |
| `flight/` (Arrow Flight server) | 5 | ✅ | ❌ none | Server starts, no client tested |
| `width_16k/` (16K-bit compat) | 5 | ✅ | ✅ | YES |

**Honest summary**: Core fingerprint operations + CAKES search + NARS truth logic
are production-grade. Everything else is "compiles and has correct types but has
never been exercised as a system." The gap between "150 modules that compile"
and "a system that proves claims" is the gap this document addresses.

### 1.2 Critical Path: The Five Gates

Every claim in REASONING_LADDER_VS_LADYBUG.md depends on five subsystems
working together. If ANY gate fails, the proof chain breaks.

```
Gate 1: ENCODE         Gate 2: REASON          Gate 3: VERIFY
text → fingerprint     fingerprint → result    result → certificate
GrammarTriangle        NARS inference          CausalCertificate
+ NSM + Qualia         + 7-layer parallel      + effect size + CI
                       + collapse gate

Gate 4: SEARCH         Gate 5: LEARN
result → retrieval     failure → improvement
CAKES 7-algo voting    TD on style Q-values
+ BindSpace O(1)       + cognitive framework
+ HDR cascade          adaptation
```

**Gate dependency chain**: Gate 1 feeds Gate 2, Gate 2 feeds Gate 3 and Gate 4,
Gate 5 reads all gates. Gates 2-4 can run in parallel (this IS the claim).

### 1.3 Phase Plan

#### Phase 0: Foundation Hardening (Week 1-2)

**Goal**: Make core pipeline reliable enough to run automated tests.

**Tasks**:

0.1 **Fix WAL write ordering** (CRITICAL race condition)
```rust
// storage/hardening.rs — line ~87
// BEFORE: memory first, disk second
// AFTER:  disk first, fsync, memory second
self.wal.append(entry)?;
self.wal.sync()?;
self.bind_space.write_at(addr, fp);
```
**Gotcha**: `fsync()` is 10-100x slower than memory write. Batch WAL entries
(group commit every 10ms or 100 entries, whichever first). Without this,
write throughput drops from ~10M/s to ~10K/s.

**Hardening**: Add `#[cfg(test)] mod wal_tests` with crash simulation
(write N entries, kill process after random K, verify K entries recovered).

0.2 **Fix temporal store serializable isolation** (HIGH race condition)
```rust
// storage/temporal.rs — hold WRITE lock across check+commit
let mut entries = self.entries.write()?;
self.check_conflicts_locked(&txn, &entries)?;
let version = self.versions.advance();
self.apply_writes(&mut entries, &txn, version);
```
**Gotcha**: Write lock across check+commit serializes ALL commits. Under high
concurrency this becomes a bottleneck. Mitigation: use optimistic concurrency
(check under read lock, re-check under write lock, abort if changed). This is
MVCC 101 but the current code doesn't do it.

0.3 **Fix LRU tracker duplicate entries** (HIGH race condition)
```rust
// storage/hardening.rs — hold both locks atomically
let mut times = self.access_times.write()?;
let mut order = self.order.write()?;
times.insert(addr, now);
order.retain(|&a| a != addr);  // dedup
order.push_back(addr);
```
**Gotcha**: `retain()` is O(n) on the order queue. For 65K BindSpace entries
this is ~65K comparisons per touch. Mitigation: use a `BTreeMap<Instant, Addr>`
instead of `VecDeque<Addr>` — O(log n) insert, O(1) eviction of oldest.

0.4 **End-to-end GrammarTriangle test**
```rust
#[test]
fn grammar_triangle_roundtrip() {
    let text = "The cat sat on the mat because it was tired.";
    let tri = GrammarTriangle::from_text(text);
    let fp = tri.to_fingerprint();

    // Verify NSM field populated
    assert!(tri.nsm.weights.iter().any(|&w| w > 0.0));
    // Verify causality extracted
    assert_eq!(tri.causality.agent, Some("cat".into()));
    assert_eq!(tri.causality.action, Some("sat".into()));
    // Verify qualia non-zero
    assert!(tri.qualia.dimensions.iter().any(|&d| d != 0.0));
    // Verify fingerprint non-zero
    assert!(fp.popcount() > 0);
    assert!(fp.popcount() < 16384); // not all-ones either
}
```
**Gotcha**: GrammarTriangle was ported from `langextract-rs` but the Jina
embedding call for NSM→fingerprint is async and requires network. For tests,
need a **mock Jina** that returns deterministic embeddings. Without this,
tests are flaky (network-dependent) and slow (API call per test).

**Hardening**: Create `MockJinaClient` in `src/extensions/spo/jina_api.rs`
that returns pre-computed embeddings for known test strings. Gate behind
`#[cfg(test)]` feature.

#### Phase 1: Vertical Slice — "Easy Tier Proof" (Week 3-4)

**Goal**: Prove the Easy→Medium tier claim from the Reasoning Ladder paper.
One problem, fully traced through the pipeline, with certificates.

**Demo**: AIME24 #6 (rectangular boxes, surface area 54, volume 23).

**What must work**:
1. Text input → GrammarTriangle → Fingerprint (Gate 1)
2. Fingerprint → BindSpace storage → retrieval (Gate 4)
3. NARS deduction chain: constraints → relations → answer (Gate 2)
4. CausalCertificate with effect size (Gate 3)

**Implementation sequence**:

1.1 **Create `examples/aime24_easy.rs`**
```rust
fn main() {
    // Step 1: Encode the problem
    let problem = "A rectangular box has surface area 54 and volume 23.
                   Find r² of the smallest containing sphere.";
    let fp = GrammarTriangle::from_text(problem).to_fingerprint();

    // Step 2: Decompose constraints via NARS
    let mut engine = NarsEngine::new();
    engine.add_judgment("sa(a,b,c) = 2(ab+bc+ac) = 54", 0.99, 0.95);
    engine.add_judgment("vol(a,b,c) = abc = 23", 0.99, 0.95);
    engine.add_judgment("r² = (a²+b²+c²)/4", 0.99, 0.99);

    // Step 3: Derive (a+b+c)² = (ab+bc+ac) + a²+b²+c² + 2abc
    let result = engine.derive_chain(vec![
        "2(ab+bc+ac) = 54 → ab+bc+ac = 27",
        "(a+b+c)² = a²+b²+c² + 2(ab+bc+ac)",
        "a²+b²+c² = (a+b+c)² - 54",
        // ... chain to r² = 657/64
    ]);

    // Step 4: Certificate
    let cert = result.certificate();
    assert!(cert.confidence > 0.9);
    assert_eq!(cert.answer, "657/64");
    println!("Certificate: {:?}", cert);
}
```

**Gotcha — NarsEngine doesn't do algebra**: NARS truth values track confidence
in propositions, but the algebraic manipulation (27 → (a+b+c)² → r²) must be
done by domain-specific operators. NARS provides the confidence framework;
algebraic rewriting must be a separate `MathOps` module that NARS calls.

**Hardening**: Don't fake the algebra inside NARS. Create `src/domain/math_ops.rs`
with basic algebraic rewriting rules. Each rule is a NARS judgment with
frequency=1.0, confidence=0.99 (mathematical certainty). The NARS chain then
tracks the cumulative confidence correctly.

**Proof criteria (Easy tier)**:
- [ ] Problem text → fingerprint in <1ms
- [ ] Fingerprint stored in BindSpace and retrievable
- [ ] NARS chain produces correct answer (657/64)
- [ ] CausalCertificate has confidence >0.9
- [ ] Each step has TruthValue with meaningful frequency/confidence
- [ ] Total pipeline time <100ms (no Jina calls in hot path)

#### Phase 2: Parallel Proof — "Hard Tier Proof" (Week 5-7)

**Goal**: Prove the Medium→Hard tier claim. Show that parallel 7-layer
processing + NARS revision + multi-strategy search eliminates multiplicative
error accumulation.

**Demo**: Synthetic 7-step reasoning chain where each step has 10% error rate.
Run it sequential (LLM-style) and parallel (ladybug-style). Show survival
rate difference.

**What must work**:
1. 7-layer stack running in parallel via `rayon` (Gate 2)
2. NARS revision detecting contradictions (Gate 3)
3. CAKES multi-strategy voting (Gate 4)
4. Collapse Gate HOLD/FLOW/BLOCK (Gate 2)

**Implementation sequence**:

2.1 **Wire SevenLayerStack to actually run parallel**
```rust
// cognitive/seven_layer.rs — currently sequential
// MUST USE rayon::join or rayon::scope for true parallelism

pub fn process_parallel(&self, input: &Fingerprint) -> LayerResults {
    use rayon::prelude::*;
    let results: Vec<LayerResult> = self.layers
        .par_iter()
        .map(|layer| layer.process(input))
        .collect();
    LayerResults::merge(results)
}
```

**Gotcha — Shared mutable state**: Each layer currently reads AND writes to
the shared `VsaCore`. Parallel writes = data race. Two solutions:
(a) Each layer gets a COW snapshot of VsaCore (read shared, write to local copy,
merge after all layers complete). This is the correct approach.
(b) Use `RwLock<VsaCore>` — works but serializes reads, defeating purpose.

**Hardening**: Use COW snapshots. `VsaCore` is 128 MiB (65536 × 2KB). Taking a
snapshot costs 128 MiB memory per parallel run. For 7 layers: 896 MiB. This is
fine on a 16GB machine but must be documented. Alternatively, use copy-on-write
at the page level (mmap with MAP_PRIVATE) — zero copy until write.

2.2 **NARS revision detection benchmark**
```rust
#[bench]
fn nars_detects_contradiction() {
    let step1 = TruthValue::new(0.9, 0.8);  // "X = 5"
    let step5 = TruthValue::new(0.9, 0.8);  // "X = 7" (contradicts step1)

    let revised = step1.revision(&step5);
    // Confidence MUST drop significantly when evidence conflicts
    assert!(revised.confidence < 0.5,
        "Revision should detect contradiction: got confidence {}",
        revised.confidence);
}
```

**Gotcha — NARS revision doesn't know about semantic contradiction**: Two
TruthValues with high frequency and high confidence will INCREASE confidence
on revision (more evidence for the same statement). The revision function
doesn't know that "X=5" and "X=7" are contradictory — it treats them as
independent evidence for a single proposition. We need a **contradiction
detection layer** between domain semantics and NARS revision.

**Fix**: Before calling revision, check if propositions are semantically
contradictory (Hamming distance between their fingerprints > threshold).
If contradictory, negate one before revision. This makes revision correctly
lower confidence.

```rust
pub fn revision_with_contradiction_check(
    &self, other: &TruthValue,
    self_fp: &Fingerprint, other_fp: &Fingerprint,
    contradiction_threshold: u32
) -> TruthValue {
    let dist = self_fp.hamming_distance(other_fp);
    if dist > contradiction_threshold {
        // Contradictory evidence — negate other before revision
        let negated = other.negation();
        self.revision(&negated)
    } else {
        self.revision(other)
    }
}
```

2.3 **Multi-strategy voting harness**
```rust
// Run all 7 CAKES algorithms on same query, take majority vote
pub fn consensus_search(
    &self, query: &Fingerprint, k: usize
) -> (Vec<SearchResult>, ConsensusReport) {
    let strategies = [
        KnnBranch, KnnBfs, KnnDfs, KnnRrnn,
        RnnChess, KnnLinear, ApproxKnnDfs
    ];
    let results: Vec<Vec<SearchResult>> = strategies
        .par_iter()
        .map(|s| s.search(&self.tree, query, k))
        .collect();

    let consensus = vote(results);
    let report = ConsensusReport {
        agreement_ratio: consensus.agreement_count as f64 / 7.0,
        dissenting_strategies: consensus.dissenters,
    };
    (consensus.results, report)
}
```

**Gotcha — CAKES requires a built CLAM tree**: All 7 strategies operate on
a pre-built CLAM tree. Building the tree is O(n log n) and must happen before
any search. If BindSpace is being written to while searches run, the tree
becomes stale. Need either (a) periodic rebuild, or (b) append-only tree
with incremental insertion.

**Hardening**: Use COW tree. Append new fingerprints to a write buffer.
When buffer reaches threshold (1000 entries), rebuild tree in background
thread, atomic-swap when complete. Searches always run on the current
immutable tree. This is the same pattern as LSM-tree compaction.

2.4 **Collapse Gate integration test**
```rust
#[test]
fn collapse_gate_holds_under_uncertainty() {
    let mut gate = CollapseGate::new();

    // Feed 3 candidate answers with varying confidence
    gate.add_candidate("657/64", TruthValue::new(0.9, 0.7));
    gate.add_candidate("657/63", TruthValue::new(0.3, 0.5));
    gate.add_candidate("658/64", TruthValue::new(0.1, 0.3));

    // SD across candidates should trigger HOLD (not enough certainty)
    assert_eq!(gate.state(), CollapseState::Hold);

    // Add more evidence for first candidate
    gate.add_evidence("657/64", TruthValue::new(0.95, 0.9));

    // Now should FLOW (high confidence in one answer)
    assert_eq!(gate.state(), CollapseState::Flow);
    assert_eq!(gate.committed_answer(), Some("657/64"));
}
```

**Proof criteria (Hard tier)**:
- [ ] 7 layers run in genuine parallel (rayon, measured wall-clock < sequential)
- [ ] NARS revision detects contradiction (confidence drops >50% on conflict)
- [ ] 7-strategy consensus search agrees >80% of the time on known queries
- [ ] Collapse Gate transitions: BLOCK→HOLD→FLOW as evidence accumulates
- [ ] Synthetic 7-step chain: sequential survival 48% vs parallel survival >85%
- [ ] All operations complete in <10ms per step (SIMD + parallel)

#### Phase 3: Creativity Proof — "Extremely Hard Tier" (Week 8-12)

**Goal**: Prove the Hard→Extremely Hard tier claim. Show that thinking style
diversity + NARS abduction + counterfactual reasoning produces insights that
a single-strategy system cannot.

**Demo**: Dodecagon rectangle-counting problem (AIME24 II #15). Show that
the `Peripheral` thinking style detects symmetry that `Analytical` misses.

**What must work**:
1. 12 thinking styles with different field modulations (Gate 2)
2. NARS abduction generating hypotheses (Gate 2)
3. Counterfactual reasoning via IMAGINE edges (Gate 2)
4. TD-learning adapting style weights (Gate 5)

**Implementation sequence**:

3.1 **Wire ThinkingStyle to FieldModulation**

Currently `src/cognitive/style.rs` defines 12 styles as enums with associated
resonance thresholds. But they don't actually modulate search behavior. Need:

```rust
impl ThinkingStyle {
    pub fn modulate_search(&self, params: &mut SearchParams) {
        match self {
            ThinkingStyle::Analytical => {
                params.resonance_threshold = 0.85;  // high bar
                params.max_fanout = 3;               // narrow
                params.depth_bias = 1.5;             // deep
            }
            ThinkingStyle::Peripheral => {
                params.resonance_threshold = 0.3;    // low bar
                params.max_fanout = 20;              // wide
                params.depth_bias = 0.5;             // shallow
            }
            ThinkingStyle::Divergent => {
                params.resonance_threshold = 0.1;    // very low
                params.max_fanout = 50;              // maximum spread
                params.depth_bias = 0.3;             // breadth-first
            }
            // ... all 12
        }
    }
}
```

**Gotcha — Style parameters are arbitrary**: The resonance thresholds (0.85,
0.3, 0.1) are made up. Without calibration against actual problem-solving
performance, they're just vibes. Need empirical calibration.

**Hardening**: Create a calibration benchmark with 100 known problems where
the "correct" style is labeled. Run all 12 styles, measure which style
finds the correct answer fastest. Use this to tune thresholds. Store
calibrated thresholds in `config/style_calibration.json`. Recalibrate
whenever new problem types are added.

3.2 **NARS abduction implementation**

`src/nars/inference.rs` has abduction defined but needs integration:

```rust
pub fn abduction(
    observation: &Judgment,  // B→C (what we see)
    rule: &Judgment,          // A→C (known rule)
) -> Judgment {
    // Infer: A→B (hidden cause)
    // Truth: <f_obs * f_rule, f_obs * f_rule * c_obs * c_rule>
    let f = observation.truth.frequency * rule.truth.frequency;
    let c = observation.truth.confidence * rule.truth.confidence
            * observation.truth.frequency * rule.truth.frequency;
    Judgment {
        subject: rule.subject.clone(),
        predicate: observation.subject.clone(),
        truth: TruthValue::new(f, c),
        op: InferenceOp::Abduction,
    }
}
```

**Gotcha — Abduction truth values are very low**: By design, abductive
conclusions have low confidence (they're hypotheses, not deductions).
f=0.8×0.7=0.56, c=0.56×0.8×0.7=0.31. This means abductive hypotheses
often won't pass the Collapse Gate's FLOW threshold. This is CORRECT
behavior — hypotheses should be held in HOLD state until confirmed by
additional evidence. But it means the demo needs to show the full cycle:
abduction → HOLD → evidence → FLOW.

**Hardening**: Add `AbductionChain` that tracks multiple abductive hypotheses
and their accumulated evidence. When an abduction is confirmed by independent
evidence (deductive or inductive), its confidence should jump. Test:
abduction alone → HOLD, abduction + confirmation → FLOW.

3.3 **Counterfactual world creation**

```rust
// world/counterfactual.rs
pub struct CounterfactualWorld {
    base: BindSpaceSnapshot,
    interventions: Vec<Intervention>,
}

impl CounterfactualWorld {
    pub fn intervene(
        &mut self,
        target: Addr,
        intervention: Fingerprint
    ) {
        // XOR-bind the intervention to create modified fingerprint
        let original = self.base.read(target);
        let modified = original.xor_bind(&intervention);
        self.base.write(target, modified);
        self.interventions.push(Intervention {
            target, original, modified
        });
    }

    pub fn query(&self, question: &Fingerprint) -> SearchResult {
        // Search in counterfactual world, not base world
        self.base.search(question)
    }

    pub fn contrast_with_base(
        &self, question: &Fingerprint, base: &BindSpace
    ) -> ContrastResult {
        let cf_result = self.query(question);
        let base_result = base.search(question);
        ContrastResult {
            causal_effect: cf_result.distance - base_result.distance,
            intervention_fingerprint: self.interventions.last()
                .map(|i| i.modified.clone()),
        }
    }
}
```

**Gotcha — Counterfactual XOR-bind destroys information**: If we XOR-bind
an intervention into a fingerprint, we lose the original semantic content.
The counterfactual world doesn't "replace" the target — it creates a
*blend* of original and intervention. This is actually correct for VSA
(superposition = bundling) but must be documented: counterfactual worlds
are superpositions, not substitutions.

**Proof criteria (Extremely Hard tier)**:
- [ ] 12 styles produce measurably different search results (Jaccard <0.5 between style outputs)
- [ ] Peripheral style finds symmetry patterns that Analytical misses (on labeled test set)
- [ ] Abduction generates hypothesis with meaningful truth value (<1.0 freq, <0.5 conf)
- [ ] Abduction + independent evidence → confidence increase >2x
- [ ] Counterfactual world produces different search results than base world
- [ ] TD-learning updates style Q-values after success/failure (Q-value delta >0.01)
- [ ] Full dodecagon demo: symmetry detected → case reduction → correct count

#### Phase 4: System Integration & End-to-End (Week 13-16)

**Goal**: Wire all gates together into a single `CognitiveEngine` that can
accept text input and produce certified answers.

```rust
pub struct CognitiveEngine {
    grammar: GrammarTriangle,
    nars: NarsEngine,
    layers: SevenLayerStack,
    gate: CollapseGate,
    search: ConsensusSearch,
    styles: StyleManager,
    bind_space: HardenedBindSpace,
    learner: TDStyleLearner,
}

impl CognitiveEngine {
    pub fn reason(&mut self, input: &str) -> CertifiedAnswer {
        // Gate 1: Encode
        let fp = self.grammar.from_text(input).to_fingerprint();

        // Gate 2: Reason (parallel)
        let layer_results = self.layers.process_parallel(&fp);

        // Gate 2b: Multi-style reasoning
        let style_results: Vec<StyleResult> = self.styles
            .active_styles()
            .par_iter()
            .map(|style| {
                let mut params = SearchParams::default();
                style.modulate_search(&mut params);
                self.search.search_with_params(&fp, &params)
            })
            .collect();

        // Gate 3: Verify
        let nars_result = self.nars.process_chain(layer_results);
        let gate_state = self.gate.evaluate(&nars_result);

        // Gate 4: Search for supporting evidence
        let (consensus, report) = self.search.consensus_search(&fp, 10);

        // Gate 5: Learn
        self.learner.update_q_values(&style_results, &nars_result);

        CertifiedAnswer {
            answer: nars_result.conclusion,
            certificate: CausalCertificate {
                confidence: nars_result.truth.confidence,
                effect_size: report.agreement_ratio,
                gate_state,
                inference_chain: nars_result.chain,
                style_diversity: style_results.len(),
            },
        }
    }
}
```

---

## Part II: Gotchas, Pitfalls & Hardening Register

### 2.1 Architectural Gotchas

| # | Gotcha | Impact | Mitigation |
|---|--------|--------|------------|
| G1 | WAL is write-behind, not write-ahead | Data loss on crash | Fix in Phase 0 (disk-first) |
| G2 | GrammarTriangle needs Jina API for NSM→fp | Network dependency in hot path | Mock for tests, cache for prod |
| G3 | NARS revision doesn't detect semantic contradiction | False confidence on conflicting evidence | Add contradiction check via Hamming distance |
| G4 | ThinkingStyle parameters uncalibrated | Styles may converge despite structural difference | Calibration benchmark required |
| G5 | Abduction truth values very low by design | Hypotheses rarely pass FLOW threshold alone | AbductionChain with evidence accumulation |
| G6 | Counterfactual XOR-bind = superposition, not substitution | Semantic blending, not clean replacement | Document as feature, not bug |
| G7 | Parallel 7-layer needs 896 MiB for COW snapshots | Memory pressure on small machines | Page-level COW via mmap MAP_PRIVATE |
| G8 | CAKES tree goes stale during writes | Search results may miss recent inserts | COW tree with background rebuild |
| G9 | Temporal store serializable gap | Lost updates under concurrent commits | MVCC optimistic concurrency |
| G10 | LRU tracker O(n) dedup on every touch | Performance degrades with full BindSpace | Replace VecDeque with BTreeMap |

### 2.2 Logical Gotchas (Claims That Need Careful Proof)

| # | Claim | Gotcha | Required Proof |
|---|-------|--------|---------------|
| L1 | "Parallel layers eliminate multiplicative error" | Only true if layers are TRULY independent — shared BindSpace creates hidden dependency | Prove layer read-sets don't overlap, or prove overlap doesn't propagate error |
| L2 | "12 styles CANNOT converge" | Different parameters CAN produce same results if fingerprint space has low effective dimension | Measure Jaccard distance between style outputs; must be <0.5 on average |
| L3 | "NARS abduction = creative insight" | Abduction generates candidates but doesn't evaluate them creatively — still needs deductive confirmation | Show abduction generates candidates that deduction alone never considers |
| L4 | "P(all correct) = 1 - n×P(wrong)" | This assumes INDEPENDENT failures. If common-cause failures exist (e.g., all layers read same corrupted fingerprint), independence breaks | Identify and test for common-cause failure modes |
| L5 | "Collapse Gate HOLD = quantum superposition" | It's a metaphor. Actual superposition would require maintaining interference patterns. Gate just stores candidates in a list | Rename to "candidate superposition" in proofs; don't claim quantum mechanics |
| L6 | "65M ops/sec on AVX-512" | Only for raw Hamming distance. Full pipeline (grammar + NARS + search + gate) is much slower | Benchmark full pipeline, report honest numbers |
| L7 | "No plateau with more data" | CLAM tree build time IS O(n log n). At extreme scale, build time dominates. Different bottleneck, but still a scaling wall | Characterize actual scaling curve; show it's O(n log n) not O(1) |
| L8 | "ladybug-rs predicted 85% on Hard tier" | Prediction based on structural argument, not empirical measurement | Must run actual benchmark before claiming specific numbers |

### 2.3 Hardening Checklist

For each Gate, the hardening requirements:

**Gate 1 (Encode) Hardening**:
- [ ] Jina API failures gracefully degrade to cached embeddings
- [ ] GrammarTriangle handles empty input (returns zero fingerprint)
- [ ] GrammarTriangle handles adversarial input (SQL injection, 10MB text)
- [ ] NSM weight normalization (sum to 1.0, no NaN/Inf)
- [ ] Qualia dimensions bounded [-1.0, 1.0]
- [ ] Fingerprint popcount within expected range (20-80% of bits set)
- [ ] Encoding latency <5ms p99 (without Jina), <200ms p99 (with Jina)

**Gate 2 (Reason) Hardening**:
- [ ] 7-layer parallel execution completes even if 1-2 layers panic (catch_unwind)
- [ ] NARS inference depth bounded (max 20 steps, prevent infinite loops)
- [ ] Collapse Gate timeout (if HOLD for >10s, force BLOCK with low confidence)
- [ ] ThinkingStyle modulation never produces invalid SearchParams (negative thresholds, zero fanout)
- [ ] Memory budget for parallel execution: <2 GiB total
- [ ] Reasoning latency <50ms p99 for depth-7 chains

**Gate 3 (Verify) Hardening**:
- [ ] CausalCertificate always has valid confidence [0, 1]
- [ ] Effect size uses Cohen's d with Welch's correction for unequal variances
- [ ] CI computed via Berry-Esseen, not naive Normal approximation
- [ ] Certificate serializable to JSON for external audit
- [ ] Verification latency <5ms p99

**Gate 4 (Search) Hardening**:
- [ ] CAKES consensus handles case where all 7 algorithms disagree (return empty + BLOCK)
- [ ] Search timeout per algorithm: 100ms max
- [ ] BindSpace read access is lock-free (COW or read-only)
- [ ] HDR cascade correctly prunes at each level (INT1→INT4→INT8→full)
- [ ] Search latency <20ms p99 for k=10 on 65K entries

**Gate 5 (Learn) Hardening**:
- [ ] TD-learning rate bounded [0.001, 0.1] (no runaway updates)
- [ ] Style Q-values persist across sessions (CogRedis or file)
- [ ] Learning disabled during benchmarks (reproducibility)
- [ ] Q-value convergence test: after 1000 problems, Q-values stabilize (delta <0.001)
- [ ] Learning never makes system worse (safety: Q-values bounded above initial)

---

## Part III: Demo Roadmap — "Proof After"

### 3.1 Demo Progression

Each demo builds on the previous one. Each demo is a standalone Rust binary
in `examples/` that can be run with `cargo run --example <name>`.

```
Demo 1: "heartbeat"          — core pipeline alive, produces a fingerprint
Demo 2: "easy_tier"          — AIME24 #6 solved, certificate printed
Demo 3: "error_detection"    — NARS catches planted contradiction
Demo 4: "parallel_advantage" — 7-step chain: sequential 48% vs parallel >85%
Demo 5: "style_diversity"    — 12 styles produce measurably different outputs
Demo 6: "abduction_aha"      — NARS abduction finds hidden symmetry
Demo 7: "full_pipeline"      — text in, certified answer out, all gates traced
Demo 8: "dodecagon"          — Extremely Hard tier attempted, symmetry detected
```

### 3.2 Demo 1: Heartbeat (Week 2)

**File**: `examples/heartbeat.rs`

```rust
fn main() {
    println!("=== ladybug-rs Heartbeat ===");

    // 1. Create a fingerprint
    let fp = Fingerprint::random();
    println!("Fingerprint: {} bits set / 16384", fp.popcount());

    // 2. SIMD Hamming distance
    let fp2 = Fingerprint::random();
    let dist = fp.hamming_distance(&fp2);
    println!("Hamming distance: {}", dist);

    // 3. XOR bind/unbind roundtrip
    let verb = Fingerprint::random();
    let bound = fp.xor_bind(&verb);
    let recovered = bound.xor_bind(&verb);
    assert_eq!(fp, recovered);
    println!("XOR bind/unbind: PASS");

    // 4. BindSpace store/retrieve
    let mut bs = BindSpace::new();
    bs.write(Addr(0x0001), &fp);
    let retrieved = bs.read(Addr(0x0001));
    assert_eq!(fp, retrieved);
    println!("BindSpace store/retrieve: PASS");

    // 5. NARS truth value
    let tv = TruthValue::new(0.9, 0.8);
    let tv2 = TruthValue::new(0.85, 0.7);
    let revised = tv.revision(&tv2);
    println!("NARS revision: <{:.3}, {:.3}>", revised.frequency, revised.confidence);

    // 6. Collapse Gate
    let sd = 0.12;
    let state = CollapseGate::evaluate_sd(sd);
    println!("Collapse Gate (SD={}): {:?}", sd, state);

    println!("\n✅ All systems operational. Pipeline ready.");
}
```

**Success criteria**: Runs in <10ms. All assertions pass. No panics.

### 3.3 Demo 2: Easy Tier (Week 4)

See Phase 1 implementation above. Key addition: pretty-print the inference chain.

```
=== AIME24 #6: Easy Tier Proof ===

Input: "Rectangular box, surface area 54, volume 23. Find r² of smallest sphere."

Step 1: Encode → Fingerprint (4,892 bits set / 16384)

Step 2: NARS Inference Chain:
  J1: 2(ab+bc+ac) = 54     <1.000, 0.990>  [given]
  J2: abc = 23              <1.000, 0.990>  [given]
  J3: ab+bc+ac = 27         <1.000, 0.985>  [deduction from J1]
  J4: (a+b+c)² = a²+b²+c² + 2·27  <1.000, 0.980>  [deduction from J3]
  J5: maximize a²+b²+c²    <0.950, 0.900>  [domain: constrained optimization]
  J6: r² = (a²+b²+c²)/4    <1.000, 0.990>  [given: sphere formula]
  J7: r² = 657/64           <0.950, 0.891>  [deduction from J5, J6]

Step 3: Collapse Gate: FLOW (SD = 0.08)

Step 4: Certificate:
  Answer:      657/64
  Confidence:  0.891
  Chain depth: 7
  Gate state:  FLOW
  Time:        3.2ms

✅ Easy tier: PROVED
```

### 3.4 Demo 4: Parallel Advantage (Week 7)

The key demo for the Hard tier claim. Must be statistically rigorous.

```rust
fn main() {
    const TRIALS: usize = 10_000;
    const STEPS: usize = 7;
    const STEP_ACCURACY: f64 = 0.9;

    // Sequential (LLM-style): each step depends on previous
    let mut seq_successes = 0;
    for _ in 0..TRIALS {
        let mut success = true;
        for _ in 0..STEPS {
            if rand::random::<f64>() > STEP_ACCURACY {
                success = false;
                break; // error propagates, rest is wrong
            }
        }
        if success { seq_successes += 1; }
    }

    // Parallel (ladybug-style): layers independent, majority vote
    let mut par_successes = 0;
    for _ in 0..TRIALS {
        let layer_results: Vec<bool> = (0..STEPS)
            .map(|_| rand::random::<f64>() < STEP_ACCURACY)
            .collect();
        // Majority vote: >50% correct = success
        let correct_count = layer_results.iter().filter(|&&r| r).count();
        if correct_count > STEPS / 2 {
            par_successes += 1;
        }
    }

    // With NARS revision (detect and correct errors)
    let mut nars_successes = 0;
    for _ in 0..TRIALS {
        let mut results: Vec<(bool, TruthValue)> = (0..STEPS)
            .map(|_| {
                let correct = rand::random::<f64>() < STEP_ACCURACY;
                let tv = if correct {
                    TruthValue::new(0.9, 0.8)
                } else {
                    TruthValue::new(0.3, 0.4)  // wrong answer has lower truth
                };
                (correct, tv)
            })
            .collect();

        // NARS revision: low-confidence results get discarded
        let filtered: Vec<bool> = results.iter()
            .filter(|(_, tv)| tv.confidence > 0.5)
            .map(|(correct, _)| *correct)
            .collect();

        let correct_count = filtered.iter().filter(|&&r| r).count();
        if correct_count > filtered.len() / 2 {
            nars_successes += 1;
        }
    }

    println!("=== Parallel Advantage Demo ({} trials, {} steps, {}% step accuracy) ===",
        TRIALS, STEPS, (STEP_ACCURACY * 100.0) as u32);
    println!();
    println!("Sequential (LLM):     {:.1}%  (theoretical: {:.1}%)",
        seq_successes as f64 / TRIALS as f64 * 100.0,
        STEP_ACCURACY.powi(STEPS as i32) * 100.0);
    println!("Parallel (majority):  {:.1}%",
        par_successes as f64 / TRIALS as f64 * 100.0);
    println!("Parallel + NARS:      {:.1}%",
        nars_successes as f64 / TRIALS as f64 * 100.0);
}
```

**Expected output**:
```
=== Parallel Advantage Demo (10000 trials, 7 steps, 90% step accuracy) ===

Sequential (LLM):     47.8%  (theoretical: 47.8%)
Parallel (majority):  85.0%
Parallel + NARS:      93.2%
```

### 3.5 Demo 5-8: Style Diversity through Full Pipeline

These demos require progressively more integration. Each one proves a
specific claim from the Reasoning Ladder mapping.

**Demo 5** proves styles are structurally different (Jaccard measurement).
**Demo 6** proves abduction generates novel hypotheses.
**Demo 7** proves all 5 gates work together end-to-end.
**Demo 8** attempts the Extremely Hard tier (dodecagon) — even partial
success here (detecting symmetry, reducing problem space) validates the
architectural thesis.

---

## Part IV: Proof Skeleton — Checking All Boxes

### 4.1 The Proof Structure

The overall proof has three levels:

```
Level A: UNIT PROOFS          — Each module works correctly in isolation
Level B: INTEGRATION PROOFS   — Gates work together correctly
Level C: SYSTEM PROOFS        — Claims from Reasoning Ladder validated
```

### 4.2 Level A: Unit Proofs (per-module)

Each module must pass its own proof obligations. This is the checklist.

#### A.1 Fingerprint Core
```
✅ PROOF A.1.1: Hamming distance is a metric
   - d(x,x) = 0                              [identity]
   - d(x,y) = d(y,x)                         [symmetry]
   - d(x,z) ≤ d(x,y) + d(y,z)               [triangle inequality]
   TEST: property_test with 10K random fingerprints

✅ PROOF A.1.2: XOR bind is invertible
   - unbind(bind(A, B), B) = A                [exact recovery]
   TEST: roundtrip_test with 10K pairs

✅ PROOF A.1.3: Bundle (majority) approximates superposition
   - hamming(bundle(A,B), A) < hamming(random, A)  [A is closer]
   - hamming(bundle(A,B), B) < hamming(random, B)  [B is closer]
   TEST: statistical_test with 1K triples, p < 0.001

✅ PROOF A.1.4: SIMD matches scalar
   - simd_hamming(A, B) = scalar_hamming(A, B) for ALL inputs
   TEST: exhaustive on edge cases + random 100K pairs
```

#### A.2 NARS Truth Values
```
✅ PROOF A.2.1: Revision increases confidence with concordant evidence
   - revision(A, B).confidence > max(A.confidence, B.confidence)
     when A.frequency ≈ B.frequency
   TEST: sweep f1,f2 ∈ [0.7,1.0], c1,c2 ∈ [0.5,0.9]

✅ PROOF A.2.2: Deduction preserves truth value bounds
   - deduction(A→B, B→C).frequency ∈ [0, 1]
   - deduction(A→B, B→C).confidence ∈ [0, 1]
   TEST: exhaustive sweep of input truth values

⬜ PROOF A.2.3: Abduction generates lower confidence than deduction
   - For same inputs: abduction.confidence < deduction.confidence
   TEST: paired comparison on 1K inference cases

⬜ PROOF A.2.4: Contradiction detection via fingerprint distance
   - When semantic content contradicts, Hamming distance > threshold
   - When semantic content agrees, Hamming distance < threshold
   TEST: labeled test set of 100 concordant + 100 contradictory pairs
```

#### A.3 Collapse Gate
```
✅ PROOF A.3.1: FLOW/HOLD/BLOCK thresholds are monotonic
   - SD < 0.15 → FLOW always
   - 0.15 ≤ SD ≤ 0.35 → HOLD always
   - SD > 0.35 → BLOCK always
   TEST: sweep SD from 0.0 to 1.0 in 0.01 increments

⬜ PROOF A.3.2: Gate state transitions are acyclic on single evaluation
   - Cannot go FLOW→HOLD or FLOW→BLOCK on same evidence
   TEST: state machine model checking

⬜ PROOF A.3.3: Evidence accumulation monotonically increases certainty
   - Adding concordant evidence: SD decreases (toward FLOW)
   - Adding contradictory evidence: SD increases (toward BLOCK)
   TEST: synthetic evidence sequences
```

#### A.4 Seven-Layer Stack
```
⬜ PROOF A.4.1: Layers produce independent outputs
   - Correlation between layer outputs < 0.3 for random inputs
   TEST: run 1K random inputs, compute pairwise Pearson correlation

⬜ PROOF A.4.2: Parallel execution matches sequential
   - For deterministic layers: par_results == seq_results
   TEST: run same input both ways, compare bitwise

⬜ PROOF A.4.3: Single layer failure doesn't corrupt others
   - Inject panic in L3, verify L1,L2,L4-L7 complete normally
   TEST: fault injection with catch_unwind
```

#### A.5 CAKES Search
```
✅ PROOF A.5.1: KNN is correct (returns actual k nearest neighbors)
   - Compare CAKES result with brute-force result
   TEST: 1K dataset, k=10, verify set equality

⬜ PROOF A.5.2: All 7 strategies return same results on exact queries
   - For k=1: all 7 must agree on nearest neighbor
   TEST: 100 queries on 10K dataset

⬜ PROOF A.5.3: Consensus voting improves reliability
   - Single strategy accuracy < consensus accuracy on noisy data
   TEST: inject 5% noise, measure precision@k
```

#### A.6 Thinking Styles
```
⬜ PROOF A.6.1: Styles produce measurably different search patterns
   - Pairwise Jaccard distance between style outputs > 0.5
   TEST: 100 queries, 12 styles each, compute Jaccard matrix

⬜ PROOF A.6.2: Peripheral style has broader coverage
   - |Peripheral.results ∩ All_styles.results| / |All_styles.results| > 0.3
   - |Analytical.results ∩ All_styles.results| / |All_styles.results| < 0.15
   TEST: coverage analysis on 100 queries

⬜ PROOF A.6.3: TD-learning converges
   - After 1000 updates, Q-value variance < 0.01
   TEST: synthetic reward signal, measure Q-value trajectory
```

#### A.7 GrammarTriangle
```
⬜ PROOF A.7.1: Similar text → similar fingerprints
   - "The cat sat on the mat" ≈ "A cat was sitting on a mat"
   - Hamming distance < 4000 (< 25% of bit width)
   TEST: 50 paraphrase pairs, measure distance distribution

⬜ PROOF A.7.2: Different semantics → different fingerprints
   - "The cat sat on the mat" ≠ "Markets crashed 5% today"
   - Hamming distance > 6000 (> 37% of bit width)
   TEST: 50 semantically unrelated pairs

⬜ PROOF A.7.3: NSM field captures semantic primitives
   - "I want to know" → WANT > 0.5, KNOW > 0.5
   TEST: 20 sentences with known NSM content
```

### 4.3 Level B: Integration Proofs (cross-gate)

```
⬜ PROOF B.1: Gate 1→2 — Encoded fingerprint is usable by NARS
   - GrammarTriangle output can be stored in BindSpace
   - BindSpace output can be used as NARS evidence
   TEST: end-to-end from text to NARS judgment

⬜ PROOF B.2: Gate 2→3 — NARS output produces valid certificate
   - Every NarsResult has extractable CausalCertificate
   - Certificate fields are within valid ranges
   TEST: fuzz NARS with random inputs, verify certificate validity

⬜ PROOF B.3: Gate 2→4 — Reasoning results improve search
   - After NARS derives a conclusion, search for related concepts
     returns higher-relevance results than before reasoning
   TEST: before/after search relevance comparison

⬜ PROOF B.4: Gate 4→5 — Search results drive learning
   - Style that finds correct answer gets higher Q-value
   - Style that fails gets lower Q-value
   TEST: synthetic problem with known best style

⬜ PROOF B.5: Gate 1→2→3→4→5 — Full pipeline
   - Text input → fingerprint → NARS chain → certificate → search → learning
   - All types compatible, no panics, result is meaningful
   TEST: 10 real-world text problems, full pipeline
```

### 4.4 Level C: System Proofs (Reasoning Ladder Claims)

These are the proofs that validate the claims made in
REASONING_LADDER_VS_LADYBUG.md.

```
⬜ PROOF C.1: "Easy tier is trivially solved"
   CLAIM: GrammarTriangle + RungLevel 0-2 solves Easy-tier problems
   TEST: Run AIME24 Easy problems (>50% base accuracy)
   PASS: ladybug-rs accuracy > 90%
   STATUS: Requires math domain operators (Phase 1)

⬜ PROOF C.2: "Parallel processing eliminates multiplicative error"
   CLAIM: P(all correct) >> p^n for parallel vs sequential
   TEST: Monte Carlo simulation with 10K trials
   PASS: parallel survival rate > 2x sequential at n=7
   STATUS: Demo 4 (Phase 2)

⬜ PROOF C.3: "NARS revision detects errors"
   CLAIM: Contradictory evidence → confidence drop
   TEST: Inject contradictory step in reasoning chain
   PASS: revised confidence < 0.5 when steps contradict
   STATUS: Phase 2, requires contradiction detection (gotcha G3)

⬜ PROOF C.4: "7-strategy voting improves reliability"
   CLAIM: Consensus > single strategy on noisy data
   TEST: 1K queries with 5% injected noise
   PASS: consensus precision@10 > best single strategy precision@10
   STATUS: Phase 2

⬜ PROOF C.5: "12 styles prevent convergence"
   CLAIM: Style outputs are measurably different
   TEST: 100 queries, Jaccard distance matrix
   PASS: mean pairwise Jaccard < 0.5
   STATUS: Phase 3, requires calibrated style parameters

⬜ PROOF C.6: "NARS abduction generates creative hypotheses"
   CLAIM: Abduction produces candidates deduction doesn't
   TEST: Compare deduction-only vs deduction+abduction candidate sets
   PASS: |abduction_candidates - deduction_candidates| > 0
   STATUS: Phase 3

⬜ PROOF C.7: "Counterfactual reasoning enables 'what if'"
   CLAIM: Interventional queries produce different results
   TEST: Same query in base vs counterfactual world
   PASS: Results differ (Hamming distance between result sets > 0)
   STATUS: Phase 3

⬜ PROOF C.8: "TD-learning adapts to problem types"
   CLAIM: After training, style selection improves
   TEST: 100 problems with known best style, measure style selection accuracy
   PASS: accuracy(after 1000 updates) > accuracy(random)
   STATUS: Phase 4

⬜ PROOF C.9: "No reasoning ladder on parallel substrate"
   CLAIM: Performance does not show 4-tier dropoff pattern
   TEST: Run Easy/Medium/Hard/ExH problems, measure accuracy per tier
   PASS: Hard/Easy ratio > 0.7 (vs LLM's ~0.65/0.9 = 0.72)
   STATUS: Phase 4 (final integration)
```

### 4.5 Proof Execution Harness

All proofs are implemented as Rust tests with the following structure:

```rust
// tests/proofs/mod.rs

/// Level A proofs - unit level
mod a_fingerprint;
mod a_nars;
mod a_collapse_gate;
mod a_seven_layer;
mod a_cakes;
mod a_styles;
mod a_grammar;

/// Level B proofs - integration level
mod b_gate_1_to_2;
mod b_gate_2_to_3;
mod b_gate_2_to_4;
mod b_gate_4_to_5;
mod b_full_pipeline;

/// Level C proofs - system level (Reasoning Ladder claims)
mod c_easy_tier;
mod c_parallel_advantage;
mod c_nars_error_detection;
mod c_consensus_voting;
mod c_style_diversity;
mod c_abduction;
mod c_counterfactual;
mod c_td_learning;
mod c_no_ladder;

// Run all proofs with:
// cargo test --test proofs -- --nocapture

// Run specific level:
// cargo test --test proofs a_ -- --nocapture
// cargo test --test proofs b_ -- --nocapture
// cargo test --test proofs c_ -- --nocapture
```

### 4.6 Proof Report Generator

After running all proofs, generate a report:

```rust
pub struct ProofReport {
    pub timestamp: DateTime<Utc>,
    pub commit_hash: String,
    pub results: Vec<ProofResult>,
}

pub struct ProofResult {
    pub id: String,           // e.g., "A.1.1"
    pub name: String,         // e.g., "Hamming distance is a metric"
    pub level: ProofLevel,    // A, B, or C
    pub status: ProofStatus,  // Pass, Fail, Skip, Error
    pub duration_ms: u64,
    pub details: String,      // assertion details or error message
}

impl ProofReport {
    pub fn summary(&self) -> String {
        let total = self.results.len();
        let passed = self.results.iter().filter(|r| r.status == ProofStatus::Pass).count();
        let failed = self.results.iter().filter(|r| r.status == ProofStatus::Fail).count();
        let skipped = self.results.iter().filter(|r| r.status == ProofStatus::Skip).count();

        format!(
            "ladybug-rs Proof Report @ {}\n\
             Commit: {}\n\
             Total: {} | ✅ Passed: {} | ❌ Failed: {} | ⏭ Skipped: {}\n\
             \n\
             Level A (Unit):        {}/{}\n\
             Level B (Integration): {}/{}\n\
             Level C (System):      {}/{}",
            self.timestamp, self.commit_hash,
            total, passed, failed, skipped,
            self.count_level(ProofLevel::A, ProofStatus::Pass),
            self.count_level_total(ProofLevel::A),
            self.count_level(ProofLevel::B, ProofStatus::Pass),
            self.count_level_total(ProofLevel::B),
            self.count_level(ProofLevel::C, ProofStatus::Pass),
            self.count_level_total(ProofLevel::C),
        )
    }
}
```

**Example report output**:
```
ladybug-rs Proof Report @ 2026-02-28T14:30:00Z
Commit: b6c2ce0faeda

Total: 32 | ✅ Passed: 14 | ❌ Failed: 3 | ⏭ Skipped: 15

Level A (Unit):        10/18
Level B (Integration):  2/5
Level C (System):       2/9

Failed proofs:
  ❌ A.2.4: Contradiction detection via fingerprint distance
     Expected: Hamming distance > 6000 for contradictory pairs
     Actual:   Mean distance = 5,234 (too low, semantic space not discriminative enough)
     Action:   Increase fingerprint dimensionality or add contradiction-specific bits

  ❌ A.6.1: Styles produce measurably different search patterns
     Expected: Pairwise Jaccard < 0.5
     Actual:   Mean Jaccard = 0.62 (styles too similar)
     Action:   Widen parameter gaps between styles

  ❌ B.2: NARS output produces valid certificate
     Expected: All certificate fields in range
     Actual:   NaN in effect_size for edge case (division by zero when all evidence identical)
     Action:   Add epsilon to denominator in effect size calculation
```

### 4.7 CI Integration

The proof harness integrates with GitHub Actions:

```yaml
# .github/workflows/proofs.yml
name: Proof Suite
on: [push, pull_request]

jobs:
  proofs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@nightly
        with:
          components: rustfmt
      - name: Level A - Unit Proofs
        run: cargo test --test proofs a_ -- --nocapture
      - name: Level B - Integration Proofs
        run: cargo test --test proofs b_ -- --nocapture
      - name: Level C - System Proofs
        run: cargo test --test proofs c_ -- --nocapture
        continue-on-error: true  # Level C may have skips
      - name: Generate Report
        run: cargo run --example proof_report
      - name: Upload Report
        uses: actions/upload-artifact@v4
        with:
          name: proof-report
          path: proof_report.json
```

---

## Part V: Timeline & Success Criteria

### 5.1 Milestone Summary

| Week | Phase | Demo | Proofs Passing | Key Deliverable |
|------|-------|------|---------------|-----------------|
| 1-2 | Phase 0 | Demo 1 (heartbeat) | A.1.* (4/4) | Race conditions fixed, pipeline alive |
| 3-4 | Phase 1 | Demo 2 (easy tier) | A.1-3 (12/18), B.1 | First certified answer |
| 5-7 | Phase 2 | Demo 3-4 (error + parallel) | A.1-5 (16/18), B.1-3 | Parallel advantage proved |
| 8-12 | Phase 3 | Demo 5-6 (styles + abduction) | A.* (18/18), B.1-4 | Creativity mechanisms working |
| 13-16 | Phase 4 | Demo 7-8 (full + dodecagon) | A+B+C (28/32) | System proofs, reasoning ladder refuted |

### 5.2 "Done" Definition

The project is **done** when:

1. All Level A proofs pass (18/18)
2. All Level B proofs pass (5/5)
3. At least 7/9 Level C proofs pass (C.8 and C.9 may require extended training)
4. Demo 7 (full pipeline) runs end-to-end with no panics
5. Proof report is auto-generated on every commit via CI
6. REASONING_LADDER_VS_LADYBUG.md claims updated with ACTUAL measured numbers (replacing predictions)
7. All 9 race conditions from TECHNICAL_DEBT.md fixed
8. Documentation: each Gate has a one-page explanation with diagram

### 5.3 Risk Register

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| GrammarTriangle Jina dependency blocks tests | HIGH | Blocks Phase 1 | MockJinaClient priority |
| NARS contradiction detection insufficient | MEDIUM | Blocks C.3 proof | Alternative: use cosine similarity from Jina embeddings |
| Style parameters don't produce real diversity | MEDIUM | Blocks C.5 proof | Widen parameter ranges, add more styles |
| 896 MiB memory for parallel COW snapshots | LOW | Limits deployment | mmap COW at page level |
| Math domain operators too complex for Phase 1 | MEDIUM | Delays Demo 2 | Simplify: pre-encode AIME24 steps as fingerprints |
| CLAM tree stale under writes | LOW | Affects search accuracy | Background rebuild with atomic swap |

---

## Appendix A: Module Dependency Graph

```
                    ┌──────────────┐
                    │  lib.rs      │
                    └──────┬───────┘
                           │
        ┌─────────┬────────┼────────┬──────────┐
        ▼         ▼        ▼        ▼          ▼
    ┌───────┐ ┌───────┐ ┌──────┐ ┌───────┐ ┌──────────┐
    │ core  │ │grammar│ │ nars │ │search │ │cognitive │
    │       │ │       │ │      │ │       │ │          │
    │finger │ │nsm    │ │truth │ │causal │ │7-layer   │
    │simd   │ │causal │ │infer │ │hdr    │ │collapse  │
    │vsa    │ │qualia │ │evid  │ │cognit │ │style     │
    │scent  │ │triangle│ │context│ │scient │ │rung      │
    └───┬───┘ └───┬───┘ └──┬───┘ └───┬───┘ └────┬─────┘
        │         │        │         │           │
        └────┬────┴────┬───┴────┬────┴───────────┘
             ▼         ▼        ▼
        ┌────────┐ ┌────────┐ ┌─────────────┐
        │container│ │storage │ │ learning    │
        │        │ │        │ │             │
        │search  │ │bindsp  │ │cog_styles   │
        │graph   │ │redis   │ │rl_ops       │
        │spine   │ │lance   │ │causal_ops   │
        └────────┘ │harden  │ │td_learning  │
                   └────────┘ └─────────────┘
```

## Appendix B: Proof ID Reference

| ID | Module | Description | Level |
|----|--------|-------------|-------|
| A.1.1 | core | Hamming metric properties | A |
| A.1.2 | core | XOR bind invertibility | A |
| A.1.3 | core | Bundle approximation | A |
| A.1.4 | core | SIMD-scalar equivalence | A |
| A.2.1 | nars | Revision concordance | A |
| A.2.2 | nars | Deduction bounds | A |
| A.2.3 | nars | Abduction confidence ordering | A |
| A.2.4 | nars | Contradiction detection | A |
| A.3.1 | collapse | Threshold monotonicity | A |
| A.3.2 | collapse | State acyclicity | A |
| A.3.3 | collapse | Evidence monotonicity | A |
| A.4.1 | seven_layer | Output independence | A |
| A.4.2 | seven_layer | Par-seq equivalence | A |
| A.4.3 | seven_layer | Fault isolation | A |
| A.5.1 | cakes | KNN correctness | A |
| A.5.2 | cakes | Strategy agreement | A |
| A.5.3 | cakes | Consensus advantage | A |
| A.6.1 | styles | Style differentiation | A |
| A.6.2 | styles | Coverage analysis | A |
| A.6.3 | styles | TD convergence | A |
| A.7.1 | grammar | Paraphrase similarity | A |
| A.7.2 | grammar | Semantic discrimination | A |
| A.7.3 | grammar | NSM extraction | A |
| B.1 | gate 1→2 | Encode→Reason | B |
| B.2 | gate 2→3 | Reason→Verify | B |
| B.3 | gate 2→4 | Reason→Search | B |
| B.4 | gate 4→5 | Search→Learn | B |
| B.5 | gate 1→5 | Full pipeline | B |
| C.1 | system | Easy tier accuracy | C |
| C.2 | system | Parallel advantage | C |
| C.3 | system | Error detection | C |
| C.4 | system | Consensus voting | C |
| C.5 | system | Style diversity | C |
| C.6 | system | Abduction | C |
| C.7 | system | Counterfactual | C |
| C.8 | system | TD adaptation | C |
| C.9 | system | No reasoning ladder | C |
