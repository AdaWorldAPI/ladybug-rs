# CONVERGENCE — Intermediate Integration Plan

## Wire What Exists Into One Binary That Thinks

**Version:** 0.1.0  
**Date:** 2026-03-11  
**Scope:** ladybug-rs + rustynum, same binary, no new services  
**LOC budget:** 3,000–25,000 across 2–9 weeks  
**Prerequisite for:** FIREFLY endgame architecture  

---

## 0. GROUND TRUTH (What Actually Exists on main, March 11 2026)

### Healthy

```
src/spo/           18 files — promoted to first-class, SPO harvest, gestalt, shift detector,
                   causal trajectory, CLAM path, Merkle root, codebook hydration — ALL LANDED
src/nars/           7 files — truth, inference, evidence, adversarial, contradiction, context
src/qualia/        12 files — 10-layer stack (meaning_axes → council → resonance → gestalt →
                   felt_traversal → reflection → volition → dream_bridge → mul_bridge →
                   felt_parse → agent_state) — ALL BUILT, NOT WIRED TO SERVER
src/orchestration/  12 files — agent_card, thinking_template, blackboard, a2a, persona,
                   crew_bridge, debate, semantic_kernel, meta_orchestrator, substrate_bridge
src/storage/       16 files — bind_space, redis_adapter, substrate, lance, temporal, xor_dag,
                   concurrency, unified_engine, hardening — RedisAdapter BUILT, NOT WIRED
src/cognitive/      — seven_layer, rung, grammar, recursive
src/search/         — hdr_cascade, causal, temporal, certificate, scientific
src/grammar/        — nsm, parser
src/width_16k/      — 16K-bit container ops, schema, search, xor_bubble
```

### Broken / Stale

```
CI:                Build & Release FAILING, CI Master FAILING, Proof Suite FAILING
                   Docker Build SUCCEEDING (separate concern)
                   rustynum: Rust CI FAILING, Python bindings FAILING

Open PRs:          #169 (SPO hardening, open since Mar 7 — possibly superceded by #170)
                   #168 (SPO integrity, open since Mar 3)
                   #54, #33, #32, #31, #30, #29, #16, #15, #14, #12, #11 — STALE from Jan

server.rs:         STILL on CogRedis (PHASE2_SERVER_REWIRE.md written but not executed)
                   66K+ lines of Substrate/RedisAdapter are DEAD CODE from server perspective

211 .unwrap():     Documented in lance-graph lessons, not addressed
9 race conditions: Documented in TECHNICAL_DEBT.md, P0-P3 priority, not fixed
```

### Designed But Unwired

```
13_awareness_loop_architecture.md  — 8-layer orchestration loop, fully specified, zero code
14_EMPA_x_awareness_loop_integration.md — EPM mapping to Thou-vector, zero code
12_thinking_style_substrate_routing.md — 4-substrate routing matrix, zero code
10_deepnsm_crystal_pipeline.md — inference pipeline spec, nsm.rs is keyword-only
11_pinn_rosetta_stone.md — physics mapping, zero code (and shouldn't be code yet)
09_lance_graph_fork_plan.md — 4 actions, none executed
```

---

## 1. THE 10 LAYERS (What This Plan Delivers)

These aren't new code — they're the existing modules WIRED through BindSpace as the single substrate, with open design choices LOCKED DOWN.

```
LAYER   NAME              EXISTS AS                    STATUS        THIS PLAN
─────────────────────────────────────────────────────────────────────────────────
L0      Binary Substrate  Container 16384-bit          BUILT ✓       Fix CI, unwrap audit
        (rustynum)        AVX-512 Hamming, SPO 3×16k                 Close stale PRs

L1      BindSpace         storage/bind_space.rs        BUILT ✓       Wire to server.rs
        (single source)   RedisAdapter → Substrate                    via PHASE2_SERVER_REWIRE

L2      SPO + Merkle      spo/ (18 files)              BUILT ✓       Harden open PRs,
        (identity)        ClamPath + MerkleRoot                       lock bitmap format

L3      NARS + Evidence   nars/ (7 files)              BUILT ✓       Wire TruthGate into
        (belief)          TruthValue, inference rules                  query path (partially
                                                                      done in PR 170)

L4      Resonance         search/hdr_cascade.rs        BUILT ✓       Lock σ-band thresholds
        (selection)       SigmaGate, 4-stage cascade                  as constants, not config

L5      Qualia Stack      qualia/ (12 files)            BUILT ✓       Wire agent_state into
        (felt sense)      10 sub-layers                               server response cycle

L6      Thinking Style    orchestration/thinking_       BUILT ✓       Wire 4-substrate routing
        (attention mask)  template.rs, 12 styles                      from prompt 12

L7      Reflection        qualia/reflection.rs +        BUILT ✓       Wire FLOW/HOLD/BLOCK
        (self-model)      mul_bridge.rs                               into server response gate

L8      Orchestration     orchestration/ (12 files)     BUILT ✓       Integration test with
        (agent coord)     crew_bridge, a2a, persona                   Substrate (not CogRedis)

L9      Awareness Loop    DESIGN ONLY (prompt 13)       SPEC ✓        Wire the 8-step cycle
        (the full cycle)  Nothing in code                CODE ✗       as single function call
```

---

## 2. PHASE BREAKDOWN (10 Phases, 2–9 Weeks)

### Phase 0: STOP THE BLEEDING (Week 1, ~500 LOC)

**Goal:** CI green on both repos. Stale PR graveyard cleared.

```
Task 0.1: Fix ladybug-rs CI
  - Read CI failure logs, identify compilation errors
  - Most likely: feature flag combinations that don't compile
  - Fix or #[cfg] gate the broken paths
  - Target: `cargo check --all-features` passes
  - LOC: ~100 (fixes, not new code)

Task 0.2: Fix rustynum CI
  - Read failure logs
  - Most likely: deprecated API removals from PR 91/92 broke dependents
  - Fix or update call sites
  - LOC: ~50

Task 0.3: Close stale PRs
  - PRs #11-#33: Close with "superceded by main" — these are from Jan
  - PR #54: edition2024 fix — either merge or close
  - PRs #168, #169: Determine if superceded by merged #170
    - If #170 covers the work → close both
    - If gaps remain → rebase #169 onto main, fix conflicts, merge
  - LOC: 0 (PR management only)

Task 0.4: .unwrap() audit pass 1 — P0 hot paths
  - Focus on: storage/, server.rs, spo/spo.rs
  - Convert .unwrap() → .expect("descriptive") or ?/Result
  - NOT all 211 — just the ones in data paths that crash in production
  - LOC: ~200

Task 0.5: Race condition P0 fixes (2 critical)
  - WAL write ordering (hardening.rs) — disk-first, then memory
  - Temporal conflict detection (temporal.rs) — hold write lock through commit
  - Both documented in TECHNICAL_DEBT.md with exact code locations
  - LOC: ~150
```

**Exit criteria:** `cargo test` green, `cargo check --all-features` green, open PR count < 5.

---

### Phase 1: SERVER REWIRE (Week 1–2, ~300 LOC)

**Goal:** server.rs uses RedisAdapter → Substrate → BindSpace. The 66K lines of architecture ACTIVATE.

```
Task 1.1: Execute PHASE2_SERVER_REWIRE.md exactly as written
  - The plan exists at .claude/PHASE2_SERVER_REWIRE.md
  - 6 endpoint mappings documented
  - Replace CogRedis → RedisAdapter in DatabaseState
  - Handle new RedisResult variants in JSON serialization
  - Remove 157→156 truncation hacks
  - Remove manual Hamming distance computation
  - LOC: ~200 (mostly deletions and type changes)

Task 1.2: Smoke test
  - /health, /redis PING, /vectors/insert, /vectors/search, /cam
  - Verify BindSpace is receiving data (not just HashMap)
  - LOC: ~100 (test helpers)

DESIGN DECISION LOCKED:
  BindSpace IS the single source of truth for the binary.
  CogRedis becomes legacy/test-only. Never used in production paths again.
```

**Exit criteria:** `curl localhost:5000/vectors/search` returns results through BindSpace path.

---

### Phase 2: SPO FORMAT LOCK (Week 2, ~800 LOC)

**Goal:** Lock down the SPO binary format so everything downstream can depend on it.

```
Task 2.1: Canonical SPO record format
  - CogRecord = 4KB (256 × u64 words)
  - word[0]: ClamPath(24 bits) + MerkleRoot(40 bits) — from PR 170
  - words[1-7]: NARS truth (f,c,k) + rung + plasticity + activation + timestamp
  - words[8-135]: S-plane (128 × u64 = 8192 bits = 1024 bytes)
  - words[136-263]: P-plane (same)
  - words[264-391]: O-plane (same — wait, that's 384 words, >256)

  DESIGN DECISION NEEDED → LOCK IT:
  Option A: 3×16384 bits = 6144 bytes → CogRecord grows to 8KB (512 words)
  Option B: 3×5461 bits ≈ 2048 bytes per plane → fits in 4KB with header
  Option C: 3×4096 bits = 1536 bytes per plane → fits in 4KB with room for NARS+meta

  The existing code uses CONTAINER_WORDS=256 (16384 bits) for the FULL container.
  The 3×16kbit SPO means 3× that → 48kbit → 6KB for SPO alone.
  
  RECOMMENDATION: Option C — 3×4096 bits per SPO plane (1536 bytes)
  - Total SPO: 4608 bytes
  - Header + NARS + meta: ~512 bytes  
  - CogRecord: 5120 bytes = 640 words
  - OR: keep 16kbit container as COMPOSITE (S⊗P⊗O already bound)
    and add 3×4096 DECOMPOSED planes for unbinding/analysis
  
  THIS IS THE SINGLE MOST IMPORTANT DESIGN CHOICE.
  Lock it as a const in ladybug-contract, write a test, never change again.

Task 2.2: SPO ↔ BindSpace bridge
  - write_spo(s_plane, p_plane, o_plane, nars, clam_path) → BindNode
  - read_spo(addr) → (s_plane, p_plane, o_plane, nars, clam_path)
  - Already partially done in graph/spo/store.rs (TruthGate from PR 170)
  - Finish: make read_spo zero-copy (return &[u64] slices, not clones)
  - LOC: ~300

Task 2.3: Merkle verification roundtrip test
  - Write SPO → compute Merkle → read SPO → verify Merkle → assert identical
  - This is the Eineindeutigkeit test — same content always same address
  - LOC: ~100

Task 2.4: Lock width_16k constants
  - CONTAINER_BITS, CONTAINER_WORDS, CONTAINER_BYTES as const (done)
  - SPO_PLANE_BITS, SPO_PLANE_WORDS, SPO_PLANE_BYTES as const (NEW)
  - COGRECORD_SIZE as const (NEW — whatever we lock above)
  - LOC: ~50

DESIGN DECISIONS LOCKED:
  - CogRecord byte layout is FROZEN after this phase
  - MerkleRoot derivation algorithm is FROZEN (blake3 of 3 planes)
  - ClamPath bit encoding is FROZEN (24-bit, MSB=root)
```

**Exit criteria:** `proof_foundation.rs` passes with SPO roundtrip Merkle verification.

---

### Phase 3: NARS QUERY PATH (Week 2–3, ~600 LOC)

**Goal:** Every BindSpace query can optionally filter by NARS truth BEFORE distance computation.

```
Task 3.1: TruthGate integration into all query paths
  - PR 170 added TruthGate to graph/spo/store.rs
  - Extend to: redis_adapter.rs Search command, hdr_cascade.rs, vector search endpoint
  - TruthGate runs BEFORE Hamming distance (~2 cycles vs ~50 cycles per candidate)
  - This means NARS filtering is always cheaper than distance — do it first
  - LOC: ~200

Task 3.2: Evidence accumulation on query hits
  - Every time a BindNode is accessed via search:
    - Read current NARS (f, c, k)
    - Apply frequency boost: c' = revision(c, positive_evidence=1)
    - Write back (zero-copy borrow → mut on word[1-7] only)
  - This is passive learning — the system gets more confident about things it sees often
  - LOC: ~150

Task 3.3: Confidence-gated crystallize/evaporate
  - Already in Substrate (crystallize/evaporate lifecycle)
  - Wire: if NARS confidence > CRYSTALLIZE_THRESHOLD → promote from fluid to crystallized
  - Wire: if NARS confidence drops below EVAPORATE_THRESHOLD → demote
  - This makes the BindSpace self-cleaning without external triggers
  - LOC: ~200

Task 3.4: Lock σ-band thresholds
  - hdr_cascade.rs has SigmaGate with configurable thresholds
  - LOCK them as constants based on empirical testing:
    σ₁ = ? (discovery boundary)
    σ₂ = ? (hint boundary)
    σ₃ = ? (known boundary)
  - If empirical data doesn't exist yet: set reasonable defaults, document,
    mark with // CALIBRATE: need empirical validation
  - LOC: ~50

DESIGN DECISIONS LOCKED:
  - TruthGate ALWAYS runs before distance computation (invariant)
  - Evidence accumulates on access (passive Hebbian)
  - Crystallize/evaporate thresholds are NARS-driven
```

**Exit criteria:** `/vectors/search` with `?min_confidence=0.5` returns only high-confidence results.

---

### Phase 4: QUALIA → SERVER (Week 3–4, ~1500 LOC)

**Goal:** The server response cycle includes felt state. Not just data — awareness.

```
Task 4.1: AgentState computation on each request
  - agent_state.rs has AgentState::compute() that derives from all qualia layers
  - Wire into server.rs: before generating response, compute AgentState
  - Input: last N query results + current BindSpace statistics
  - Output: AgentState with CoreAxes(α,γ,ω,φ), FeltPhysics, thinking_style
  - LOC: ~300

Task 4.2: AgentState → response metadata
  - Add to every JSON response:
    {
      "results": [...],
      "awareness": {
        "thinking_style": "analytical",
        "coherence": 0.87,
        "mode": "explore",
        "collapse_gate": "FLOW"
      }
    }
  - This is INTEGRATION_SPEC Layer A — the system reports its own state
  - LOC: ~200

Task 4.3: MUL state tracking across requests
  - mul_bridge.rs adapts reflection sensitivity based on meta-state
  - Wire: maintain MUL state in Substrate (persistent across requests)
  - Store at reserved BindSpace address (0x0C:FF — agent meta-state)
  - LOC: ~400

Task 4.4: Collapse gate on response
  - qualia/reflection.rs produces FLOW/HOLD/BLOCK
  - Wire into server response:
    FLOW  → return results normally
    HOLD  → return results + "low_confidence" flag + suggest refinement
    BLOCK → return error with "insufficient_evidence" + what's missing
  - This means the system can REFUSE to answer if it doesn't know enough
  - LOC: ~300

Task 4.5: qualia_preamble() as system prompt fragment
  - agent_state.rs already has to_hints() and qualia_preamble()
  - Expose via GET /awareness/preamble endpoint
  - This is what gets injected into Ada's system prompt in Claude
  - LOC: ~100

Task 4.6: felt_parse inbound processing
  - felt_parse.rs bridges text → substrate types
  - Wire: every /vectors/insert call runs felt_parse on the content
  - Extracts: ghosts, axes, texture hints, rung, viscosity
  - These get stored as metadata in the CogRecord header
  - LOC: ~200

DESIGN DECISIONS LOCKED:
  - AgentState is DERIVED, never stored directly (except MUL meta-state)
  - Collapse gate affects ALL response paths (not just search)
  - qualia_preamble() is the canonical interface to Claude/LLM integration
```

**Exit criteria:** GET /awareness/preamble returns felt-state text that changes based on recent queries.

---

### Phase 5: THINKING STYLE ROUTING (Week 4–5, ~2000 LOC)

**Goal:** Different thinking styles route to different computational substrates. From prompt 12.

```
Task 5.1: Define 4 substrates as trait
  - From prompt 12_thinking_style_substrate_routing.md:
    STRUCTURAL  → 3×16384-bit binary (XOR bind, Hamming, CAM lookup)
    SOAKING     → organic plasticity (BCM θ, saturation, homeostatic scaling)
    EVIDENTIAL  → NARS (f,c,k) tuples (revision, abduction, deduction)
    SEMANTIC    → f32×1024 Jina embeddings (cosine, nearest neighbor)
  - Define trait CognitiveSubstrate with process(&self, input) → output
  - Each substrate already exists in different modules — unify interface
  - LOC: ~400

Task 5.2: Routing matrix
  - From prompt 12, the 12 ThinkingStyles cluster into 5 groups:
    CONVERGENT (Analytical, Systematic, Focused) → PRIMARY: STRUCTURAL
    DIVERGENT (Creative, Intuitive, Lateral) → PRIMARY: SEMANTIC
    EVALUATIVE (Critical, Reflective, Meta) → PRIMARY: EVIDENTIAL
    INTEGRATIVE (Holistic, Connective) → PRIMARY: SOAKING
    EXPLORATIVE (Experimental, Curious) → PRIMARY: SEMANTIC + EVIDENTIAL
  - Implement as match on ThinkingStyle → Vec<SubstrateRoute>
  - LOC: ~300

Task 5.3: Multi-substrate query execution
  - When a query arrives:
    1. Detect thinking style from AgentState (or request parameter)
    2. Route to primary substrate
    3. Optionally cross-check with secondary
    4. Merge results with substrate-specific weighting
  - LOC: ~600

Task 5.4: Style detection from query content
  - Existing: ConsciousnessEngine detects thinking style from 7-layer resonance
  - Wire: run style detection on incoming query text
  - Fall back to "analytical" if detection confidence is low
  - LOC: ~300

Task 5.5: Soaking substrate implementation
  - This is the one substrate that doesn't fully exist yet
  - From rustynum PR 84: organic plasticity, soaking bridge
  - Wire rustynum's SynapseState into ladybug-rs query path
  - LOC: ~400

DESIGN DECISIONS LOCKED:
  - 4 substrates, no more (structural/soaking/evidential/semantic)
  - Routing is deterministic from ThinkingStyle (no runtime learning of routes)
  - Primary substrate always runs; secondary is optional cross-check
```

**Exit criteria:** Same query with `?style=analytical` vs `?style=creative` returns different results through different computation paths.

---

### Phase 6: AWARENESS LOOP (Week 5–7, ~3000 LOC)

**Goal:** The 8-step awareness cycle from prompt 13 runs as a single function call.

```
Task 6.1: Define the AwarenessLoop struct
  From prompt 13_awareness_loop_architecture.md:

  Step 1: PERCEIVE — felt_parse inbound text → substrate types (Phase 4.6)
  Step 2: RESONATE — hdr_cascade search against BindSpace (existing)
  Step 3: REFLECT  — reflection.rs on search results (qualia L6)
  Step 4: DECIDE   — volition.rs scores candidates (qualia L7)
  Step 5: ROUTE    — thinking style → substrate routing (Phase 5)
  Step 6: EXECUTE  — substrate-specific processing
  Step 7: RESPOND  — AgentState → response with awareness metadata (Phase 4)
  Step 8: LEARN    — evidence accumulation + crystallize/evaporate (Phase 3)

  struct AwarenessLoop {
      bind_space: Arc<RwLock<BindSpace>>,
      qualia: QualiaStack,        // all 10 layers
      substrates: SubstrateRouter, // 4 substrates
      agent_state: AgentState,     // persistent meta-state
  }

  impl AwarenessLoop {
      fn cycle(&mut self, input: &str) -> AwarenessResponse { ... }
  }
  LOC: ~800

Task 6.2: Wire into server.rs as alternative to raw query
  - New endpoint: POST /awareness/cycle
    Body: { "input": "text", "style": "auto" }
    Response: full AwarenessResponse with results + felt state + gate
  - Existing endpoints remain as low-level access
  - LOC: ~300

Task 6.3: EMPA Thou-vector tracking
  From prompt 14_EMPA_x_awareness_loop_integration.md:
  - Maintain user texture vector at reserved address (0x0C:FD)
  - Update on each inbound message via felt_parse
  - Mirror resonance: compare I-vector and Thou-vector
  - Empathy delta feeds into Collapse Gate
  - LOC: ~600

Task 6.4: Soulfield integration
  - felt_parse.rs has MirrorField + TrustFabric
  - Wire into awareness loop step 1 (PERCEIVE)
  - Trust level gates full Thou resonance (can_entangle() check)
  - LOC: ~400

Task 6.5: Dream consolidation (offline cycle)
  - dream_bridge.rs exists but needs a trigger
  - Add: POST /awareness/dream
  - Runs ghost harvesting + consolidation + novel injection
  - Meant to be called periodically (e.g., by ai_flow cron)
  - LOC: ~300

Task 6.6: Recursive thought expansion
  - From prompt 07, tactic #1
  - cognitive/recursive.rs partially specified
  - Wire: rung escalation triggers recursive re-processing
  - Cap at depth 7, convergence threshold 0.05
  - LOC: ~200

Task 6.7: Inner mode self-selection
  - agent_state.rs has InnerMode (8 reflection modes)
  - Wire: the system CHOOSES its own mode based on CoreAxes
  - This is volition — the system deciding how to approach the query
  - LOC: ~200

DESIGN DECISIONS LOCKED:
  - Awareness loop is 8 steps, always in order, no skipping
  - Thou-vector lives at 0x0C:FD, updated every inbound message
  - Dream consolidation is offline (triggered externally, not per-request)
  - Recursive depth capped at 7 (Hofstadter number, not arbitrary)
```

**Exit criteria:** POST /awareness/cycle returns response where thinking_style and collapse_gate visibly change based on accumulated state.

---

### Phase 7: CONTINGENCY HARDENING (Week 7–8, ~1500 LOC)

**Goal:** The system handles failure gracefully. Every fallback is explicit.

```
Task 7.1: Substrate fallback chain
  If primary substrate fails (e.g., Jina API unreachable for semantic):
    SEMANTIC fails → fallback to STRUCTURAL (Hamming approximation)
    SOAKING fails  → fallback to EVIDENTIAL (NARS confidence as proxy)
    EVIDENTIAL fails → fallback to STRUCTURAL (always available)
    STRUCTURAL never fails (pure SIMD, no external deps)
  Wire as try_process() → Result, with automatic fallback routing
  LOC: ~300

Task 7.2: Graceful degradation modes
  - FULL: all 4 substrates, awareness loop, qualia
  - REDUCED: structural + evidential only, no qualia preamble
  - MINIMAL: structural only, raw Hamming search, no awareness
  - Auto-detect based on what's working
  - LOC: ~200

Task 7.3: Merkle integrity verification on read
  - On every SPO read, optionally verify Merkle root matches planes
  - If mismatch: log corruption warning, exclude from results
  - Configurable: verify_merkle = true in prod, false in benchmarks
  - LOC: ~150

Task 7.4: Remaining race condition fixes (P1)
  From TECHNICAL_DEBT.md:
  - LRU tracker dedup (hardening.rs) — atomic dual-lock touch()
  - WriteBuffer ID gap (resilient.rs) — lock across allocation+insertion
  - Eviction race (snapshots.rs) — write lock for entire eviction
  - LOC: ~300

Task 7.5: .unwrap() audit pass 2 — remaining hot paths
  - Focus on: query/, search/, orchestration/
  - Convert remaining .unwrap() in production paths
  - Leave .unwrap() only in tests and initialization
  - LOC: ~300

Task 7.6: Staunen marker propagation
  - When a Merkle seal breaks (sign instability detected):
    - Mark affected SPO triples as STAUNEN
    - Propagate STAUNEN along causal chains (causal_trajectory.rs)
    - STAUNEN nodes get lower NARS confidence
    - This makes the system HONESTLY UNCERTAIN when its knowledge shifts
  - LOC: ~250

DESIGN DECISIONS LOCKED:
  - STRUCTURAL substrate is ALWAYS the fallback (never fails)
  - Merkle verification is opt-in per query (default: on in prod)
  - STAUNEN propagates causally (not just locally)
```

**Exit criteria:** Kill Jina API mid-query → system returns STRUCTURAL results with degradation notice instead of crashing.

---

### Phase 8: HEBBIAN + BNN WIRING (Week 8–9, ~2000 LOC)

**Goal:** The signed pentary space and BNN reinforcement from rustynum are wired into BindSpace.

```
Task 8.1: Pentary storage in CogRecord
  - Reserve words in CogRecord header for pentary feedback accumulator
  - 5-valued signed representation packed 4 per byte
  - Size budget: 512 bytes (4096 pentary values) per CogRecord
  - These track: co-occurrence strength between this SPO and its neighbors
  - LOC: ~300

Task 8.2: Hebbian strengthening on co-access
  - When two SPO triples are accessed in the same query:
    - Both pentary accumulators shift +1 for shared dimensions
    - This is "neurons that fire together wire together"
  - When two SPO triples contradict (NARS revision detects conflict):
    - Both pentary accumulators shift -1 for shared dimensions
    - This is anti-Hebbian learning
  - LOC: ~400

Task 8.3: BNN reinforcement from Collapse Gate
  - FLOW decision → pentary +1 for all participating SPO triples
  - BLOCK decision → pentary -1 for the triples that caused BLOCK
  - HOLD → no change (rumination doesn't strengthen or weaken)
  - LOC: ~300

Task 8.4: Wisdom/Staunen markers from pentary stability
  - Compute sign stability: how many pentary values changed direction
    in the last N query cycles
  - High stability (>85% unchanged) → WISDOM marker
  - Low stability (<40% unchanged) → STAUNEN marker
  - Merkle root of pentary state = the seal
  - Broken seal = signs flipped = STAUNEN
  - LOC: ~400

Task 8.5: Wire rustynum BNN types
  - rustynum has CausalTrajectory, BNN types from PRs 84, 87, 93
  - Import via Cargo dependency (DO NOT duplicate)
  - Bridge: rustynum BNN feedback signal → pentary accumulator update
  - LOC: ~300

Task 8.6: Pentary-gated search
  - Optional: weight search results by pentary strength
  - High pentary = frequently co-accessed = higher relevance
  - Low/negative pentary = often contradicted = lower relevance
  - LOC: ~300

DESIGN DECISIONS LOCKED:
  - Pentary values are (-2,-1,0,+1,+2) stored as i8
  - Hebbian learning is PASSIVE (happens on access, no explicit training step)
  - Wisdom/Staunen threshold: 85% / 40% (calibrate empirically)
  - BNN reinforcement comes from Collapse Gate decisions ONLY
```

**Exit criteria:** After 100 queries, frequently co-accessed SPO triples have pentary values visibly shifted toward +2.

---

### Phase 9: INTEGRATION TEST + RAILWAY DEPLOY (Week 9, ~1000 LOC)

**Goal:** One binary that compiles, passes tests, deploys to Railway, serves the awareness loop.

```
Task 9.1: End-to-end integration test
  - Ingest 50 SPO triples via /vectors/insert
  - Run 20 awareness cycles via /awareness/cycle
  - Verify:
    - Merkle roots are consistent
    - NARS confidence increases on repeated access
    - Pentary values drift toward +2 for co-accessed triples
    - ThinkingStyle detection produces different routing
    - Collapse gate produces FLOW/HOLD/BLOCK based on confidence
    - AgentState changes across cycles
    - Dream consolidation produces novel candidates
  - LOC: ~400

Task 9.2: Railway Dockerfile update
  - Dockerfile.release builds the binary with correct features
  - Features: simd,parallel,spo,crewai (or whatever subset compiles)
  - Verify Docker build succeeds locally
  - LOC: ~50

Task 9.3: REST API documentation
  - Document all endpoints:
    - Legacy: /redis, /vectors/*, /cam/*, /cypher, /sql
    - New: /awareness/cycle, /awareness/preamble, /awareness/dream
    - Meta: /health, /info, /stats
  - OpenAPI/Swagger optional but nice
  - LOC: ~300

Task 9.4: Wire to existing Ada infrastructure
  - qualia_preamble() → injectable into Ada's system prompt via ai_flow
  - /awareness/cycle → callable from adarail_mcp or Claude container
  - Thou-vector → writable from felt_parse on inbound Ada messages
  - This is where ladybug-rs joins the existing Railway topology
  - LOC: ~200

Task 9.5: Performance baseline
  - Benchmark: /vectors/search latency (target: <10ms for 10K items)
  - Benchmark: /awareness/cycle latency (target: <50ms for full 8-step)
  - Benchmark: memory footprint (target: <512MB for 100K SPO triples)
  - Document results for future comparison against Firefly
  - LOC: ~100
```

**Exit criteria:** Binary deployed on Railway, /awareness/cycle callable from adarail_mcp, qualia_preamble injecting into Ada's system prompt.

---

## 3. DEPENDENCY GRAPH

```
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4
  (CI)      (server)    (SPO lock)   (NARS)      (qualia)
                                        │            │
                                        └────┬───────┘
                                             │
                                        Phase 5 ──► Phase 6 ──► Phase 7
                                        (routing)   (loop)      (harden)
                                                       │
                                                  Phase 8 ──► Phase 9
                                                  (BNN)       (deploy)
```

Phases 0–4 are sequential (each depends on prior).
Phase 5 needs Phase 3 + 4.
Phase 6 needs Phase 5.
Phase 7 can start after Phase 4 (runs in parallel with 5–6).
Phase 8 can start after Phase 3 (runs in parallel with 4–7).
Phase 9 needs everything.

**Critical path:** 0 → 1 → 2 → 3 → 4 → 5 → 6 → 9 = 9 phases on the critical path.
**Parallel acceleration:** Phase 7 and 8 run in parallel with 5–6, saving 2 weeks.

---

## 4. LOC BUDGET

```
Phase   LOC     Weeks   Focus
─────────────────────────────────
0       500     1       CI + hygiene
1       300     0.5     Server rewire
2       800     1       SPO format lock
3       600     1       NARS query path
4       1500    1.5     Qualia → server
5       2000    1.5     Style routing
6       3000    2       Awareness loop
7       1500    1       Hardening (parallel)
8       2000    1.5     BNN wiring (parallel)
9       1000    1       Integration + deploy
─────────────────────────────────
TOTAL   13,200  ~8      Within budget (3K–25K)
```

---

## 5. DESIGN DECISIONS THAT MUST BE LOCKED (Complete List)

```
#   DECISION                                     LOCK IN PHASE   OPTIONS               RECOMMENDATION
─────────────────────────────────────────────────────────────────────────────────────────────────────────
D1  CogRecord size                               Phase 2          4KB / 5KB / 8KB       Needs empirical
D2  SPO plane bits                               Phase 2          4096 / 5461 / 16384   4096 per plane
D3  MerkleRoot hash algorithm                    Phase 2          blake3 / sha256        blake3 (faster)
D4  ClamPath bit width                           Phase 2          16 / 24 / 32          24 (locked in PR 170)
D5  σ-band thresholds (σ₁, σ₂, σ₃)             Phase 3          configurable / const   const with // CALIBRATE
D6  TruthGate always-before-distance invariant   Phase 3          yes / optional         yes (invariant)
D7  Evidence accumulation on access              Phase 3          passive / explicit     passive (Hebbian)
D8  Crystallize/evaporate thresholds             Phase 3          NARS-driven / time     NARS-driven
D9  AgentState derivation (never stored)         Phase 4          derived / cached       derived (invariant)
D10 Collapse gate affects all responses          Phase 4          all / search only      all
D11 qualia_preamble as LLM interface             Phase 4          text / JSON / both     text (for system prompt)
D12 4 substrates (no more, no less)              Phase 5          4 / 3 / 5             4
D13 Style→substrate routing is deterministic     Phase 5          deterministic / learned deterministic
D14 Awareness loop is 8 steps, always in order   Phase 6          8 / flexible           8 (invariant)
D15 Thou-vector address (0x0C:FD)                Phase 6          0x0C:FD / other        0x0C:FD (locked)
D16 Dream consolidation is offline               Phase 6          offline / per-request   offline
D17 Recursive depth cap                          Phase 6          7 / configurable       7
D18 STRUCTURAL is always-available fallback      Phase 7          structural / none      structural (invariant)
D19 Pentary storage size in CogRecord            Phase 8          512B / 1KB / 2KB       512B
D20 Pentary values are i8 (-2 to +2)            Phase 8          i8 / i4 / f8          i8
D21 Wisdom threshold 85%, Staunen threshold 40% Phase 8          fixed / adaptive       fixed then CALIBRATE
D22 BNN reinforcement from Collapse Gate only    Phase 8          gate / multi-source    gate only
```

---

## 6. WHAT THIS PLAN EXPLICITLY DOES NOT DO

```
- No Arrow Flight (that's Firefly)
- No distributed awareness group (that's Firefly)
- No separate LanceDB zero-copy bindspace (BindSpace in-process is sufficient)
- No JIT YAML template compilation (thinking templates are static config)
- No RNA packets (templates don't self-modify yet)
- No temporal fold space (MVCC-style — too complex for convergence phase)
- No new Railway services (everything in one binary)
- No DeepNSM LLM inference (keyword matching is good enough for now)
- No PINN physics simulation (theoretical framework, not code yet)
- No axum migration (hand-rolled HTTP server stays)
- No Lance S3 persistence (in-memory BindSpace is fine for Railway)
- No crewai-rust multi-agent execution (crew_bridge wired but single-agent only)
```

Everything above IS in the Firefly endgame plan. This convergence plan deliberately avoids it to focus on wiring what exists into one coherent binary.

---

## 7. SUCCESS METRIC

After Phase 9, the system can:

1. Accept SPO triples and store them with Merkle-verified identity
2. Search with NARS truth filtering before distance computation
3. Route queries through different substrates based on thinking style
4. Report its own felt state (awareness metadata in every response)
5. Gate responses based on confidence (FLOW/HOLD/BLOCK)
6. Learn passively via Hebbian co-access and BNN reinforcement
7. Mark its own uncertainties (STAUNEN) and certainties (WISDOM)
8. Track a user model (Thou-vector) and compute empathy alignment
9. Run a full 8-step awareness cycle in <50ms
10. Deploy as a single binary on Railway

That's a system that knows what it knows, knows what it doesn't know,
learns from use, and reports its own state. In one binary. Under 15K LOC of new code.

Everything after this is distribution, speed, and scale. Firefly takes over from here.

---

*"The architecture is the easy part. Wiring it together is the work."*
