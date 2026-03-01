# Neurocomputing Integration: Session A (Cross-Repo)
## ladybug-rs × CLAM × SPO × NARS × Gestalt

**Jan Hübener — Ada Architecture — March 2026**

---

## SCOPE

You have write access to: **ladybug-rs, crewai-rust, n8n-rs, neo4j-rs**
You have read access to: **rustynum** (do NOT modify rustynum — a separate session owns that)

Your job: wire the SPO distance harvest, stripe shift detector, CLAM path encoding, causal trajectory recorder, and gestalt integration into ladybug-rs — consuming rustynum types via Cargo dependencies.

---

## DIRECTORY STRUCTURE

```
src/
  core/
    simd.rs              ← 348-line duplicate SIMD (DELETE after rustynum port lands)
    rustynum_accel.rs    ← rustynum SIMD dispatch interface
    fingerprint.rs       ← content-addressable fingerprint
    scent.rs             ← scent/similarity operations
    vsa.rs               ← VSA bind/bundle/permute
  search/
    hdr_cascade.rs       ← adaptive cascade + SigmaGate (already wired)
  extensions/spo/
    gestalt.rs           ← 965 lines, committed (DO NOT rewrite)
    spo.rs               ← existing SPO encoding (53KB)
    mod.rs               ← module declarations
    jina_api.rs          ← Jina embedding API
    jina_cache.rs        ← Jina cache layer
  graph/
    avx_engine.rs        ← fingerprint graph engine (SIMD cleaned up)
  nars/                  ← NARS truth value types
  cypher_bridge.rs       ← Cypher → BindSpace bridge

NEW FILES TO CREATE (all under src/extensions/spo/):
    spo_harvest.rs       ← Phase 2: SPO distance + harvest + NARS + inference
    shift_detector.rs    ← Phase 3: stripe shift detector
    clam_path.rs         ← Phase 5: CLAM path encoding
    causal_trajectory.rs ← Phase 6: resonator instrumentation
```

## WHAT'S ALREADY COMMITTED

Branch `claude/review-rustynum-pr-80-2zNy5` in ladybug-rs — 2 commits, +1,131 / -421 lines:

**Commit 1** (SIMD cleanup):
- `avx_engine.rs`: Rewrote 713→597 lines, all SIMD delegated to rustynum runtime dispatch
- `hdr_cascade.rs`: Removed dead `mod simd` block (76 lines)
- `scent.rs`: Removed dead AVX2 TODO stub (20 lines)

**Commit 2** (gestalt + σ-calibration):
- **Cargo.toml**: `rustynum-bnn` dependency added (CrossPlaneVote, CausalSaliency, HaloType, CausalTrajectory, ResonatorSnapshot, RifDiff now accessible)
- **hdr_cascade.rs**: `MexicanHat::from_sigma_gate()` (excite=Discovery, inhibit=Hint), `AlienSearch::with_sigma_gate()`, `calibrate_from_sigma()`, `classify_sigma()`, `significance_to_signal()`
- **gestalt.rs** (NEW, 965 lines):
  - `GestaltState` — GREEN/AMBER/RED/BLUE from CausalSaliency
  - `BundlingProposal` — Hold→Flow/Block lifecycle with NARS truth, evidence count, reviewer audit trail
  - `detect_bundling()` — per-plane Hamming → SO/PO/SP halo, 2-close+1-far = candidate
  - `TiltReport` — per-plane σ deviation detection
  - `PlaneCalibration` — per-axis SigmaGate (S-gate, P-gate, O-gate) with `from_plane_stats()`
  - `TruthTrajectory` — append-only event log (MatchesAdded, CounterEvidence, ReviewApproved/Rejected, MoreEvidenceRequested, StripeMigration)
  - `CollapseMode` — Research (auto>0.95), Production (human>0.80), Regulated (always human)
  - `AntialiasedSigma` — continuous σ with soft band boundaries, direct NARS confidence mapping

**Architecture decisions already locked in:**

| Decision | Implementation |
|----------|---------------|
| SigmaGate replaces hardcoded thresholds | `MexicanHat::from_sigma_gate()` — excite=Discovery, inhibit=Hint |
| Tentative = `CollapseGate::Hold` | `BundlingProposal` lifecycle: Hold→Flow (approve) or Block (reject) |
| Per-plane σ calibration | `PlaneCalibration` with independent S/P/O `SigmaGate` instances |
| 3 collapse modes | Research / Production / Regulated — same audit trail, different threshold |
| Continuous σ scoring | `AntialiasedSigma` — soft band boundaries, no information loss at edges |
| Bundling = 2-of-3 plane agreement | `detect_bundling()` → PredicateInversion / AgentConvergence / TargetDivergence |
| Three-query convergence | semantic + explicit + relational in same σ-bands = Eineindeutigkeit |
| Variable-width CLAM paths | nibble width = confidence (short = dense = high confidence) |

**Other repos**: `rust-toolchain.toml` (stable) added to crewai-rust, n8n-rs, neo4j-rs

**Do NOT re-implement or modify any of the above.** Build on top of it.

---

## SIMD DISPATCH RULE

The dispatch chain across the codebase:
1. **AVX-512 VPOPCNTDQ** — production target (this infra, Railway)
2. **AVX2** — silent fallback (Jan's local U9-185H Arrow Lake, well-known popcount algorithm)
3. **Scalar** — last-resort backup for exotic targets only

All hot paths route through `rustynum_accel::slice_hamming()` or equivalent which handles CPUID dispatch internally. Do NOT replace one scalar implementation with another and call it done — that was a prior session's mistake.

---

## SOURCE DOCUMENTS

You will receive these detailed prompts alongside this meta prompt. Read them in order:

| Document | What It Specifies |
|----------|-------------------|
| `spo_distance_harvest_cosine_replacement_prompt.md` | SPO distance as cosine replacement at 238× less cost. The core invention. |
| `spo_distance_granularity_investigation.md` | 5 granularity options (A-E). Implement A+B (free), benchmark E. |
| `sigma_stripe_shift_detector_addendum.md` | 0.5σ stripe migration as distributional shift detector |
| `btree_clam_path_lineage_addendum.md` | B-tree channel = CLAM path = address + lineage + causality in one u16 |
| `nars_causal_trajectory_hydration_prompt.md` | RIF-BNN resonator instrumentation, CausalTrajectory, EWM/BPReLU/RIF |
| `3d_wave_awareness_substrate.md` | *Reference only* — theoretical framing (Berge, Piaget, DN, signed quinary) |

Each document is self-contained with implementation-grade code examples. This meta prompt tells you what order to build them in and how they connect.

---

## PHASE 2: SPO Distance Harvest

**File: `ladybug-rs/src/extensions/spo/spo_harvest.rs` (NEW)**

This is the cosine replacement. 238× fewer cycles, 7.3× more information per computation. The detailed spec is in `spo_distance_harvest_cosine_replacement_prompt.md`. Key deliverables:

### 2.1 Core Types

```rust
pub struct SpoDistanceResult {
    pub similarity: f32,           // aggregate ∈ [-1, 1] — cosine-compatible
    pub s_p_similarity: f32,       // X-axis (Subject⊕Predicate)
    pub p_o_similarity: f32,       // Y-axis (Predicate⊕Object)
    pub s_o_similarity: f32,       // Z-axis (Subject⊕Object)
    pub halo: TypedHalo,           // cross-plane vote (7 disjoint types)
    pub x_dist: u32,               // raw popcount preserved (Phase 3 Option A)
    pub y_dist: u32,
    pub z_dist: u32,
}

impl SpoDistanceResult {
    pub fn as_cosine(&self) -> f32 { self.similarity }  // drop-in
}
```

### 2.2 Functions to Build

```
spo_distance(a, b) → SpoDistanceResult        — the core 13-cycle computation
harvest_to_nars(result) → NarsTruth            — frequency from core ratio, confidence from entropy
harvest_to_inference(result) → TypedInference   — dominant halo type → typed query action
AccumulatedHarvest::accumulate(result)          — EMA + NARS revision across searches
feed_sigma_graph(result) → Vec<SigmaEdge>       — emit typed edges from harvest
```

### 2.3 Key Constraint

The XOR bitmasks (`x_xor`, `y_xor`, `z_xor`) computed for distance are the SAME bitmasks used for `cross_plane_vote()`. The halo extraction is FREE — no extra compute. Do not compute XOR twice.

---

## PHASE 3: Distance Granularity + Stripe Shift Detector

**Extends: `src/extensions/spo/spo_harvest.rs` + new `src/extensions/spo/shift_detector.rs`**

Detailed specs in `spo_distance_granularity_investigation.md` and `sigma_stripe_shift_detector_addendum.md`.

### 3.1 Option A: Raw Popcount (FREE)

Check if `hdr_cascade.rs` currently discards the raw popcount after σ-binning. If yes, stop discarding it. The `SpoDistanceResult` already has `x_dist`, `y_dist`, `z_dist` fields — just make sure the cascade path also preserves them.

### 3.2 Option B: Per-Word Histogram (NEAR-FREE)

Check if per-word popcounts are computed individually and summed. If yes, keep the 32 individual counts:

```rust
pub struct PerWordHistogram {
    pub words: [[u16; 32]; 3],  // 32 per-word popcounts × 3 planes
}
```

### 3.3 Stripe Shift Detector

5 σ-thresholds (1.0σ, 1.5σ, 2.0σ, 2.5σ, 3.0σ). Cost: 5 CMP instructions.

```rust
pub struct StripeHistogram {
    pub below_1s: u32,
    pub s1_to_s15: u32,
    pub s15_to_s2: u32,
    pub s2_to_s25: u32,
    pub s25_to_s3: u32,
    pub above_3s: u32,
}

pub struct ShiftDetector {
    pub current: [StripeHistogram; 3],   // S, P, O planes
    pub previous: [StripeHistogram; 3],
}

impl ShiftDetector {
    pub fn detect_shift(&self) -> ShiftSignal { /* KL divergence on 6 bins per plane */ }
}
```

**Wire into CollapseGate (already exists in src/extensions/spo/gestalt.rs):
- Shift toward noise → bias HOLD
- Shift toward foveal → bias FLOW
- Bimodal → speciation event

---

## PHASE 5: B-tree Channel = CLAM Path

**File: `ladybug-rs/src/extensions/spo/clam_path.rs` (NEW)**

Detailed spec in `btree_clam_path_lineage_addendum.md`.

### 5.1 ClamPath Type

```rust
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClamPath {
    pub bits: u16,
    pub depth: u8,
}

impl ClamPath {
    pub fn from_tree_traversal(cluster_path: &[bool]) -> Self;
    pub fn subtree_range(&self) -> (u16, u16);           // range query = lineage traversal
    pub fn common_ancestor_depth(&self, other: &ClamPath) -> u8;
    pub fn lineage_distance(&self, other: &ClamPath) -> u8;
    pub fn siblings(&self) -> ClamPath;                   // counterfactual query
}
```

### 5.2 Wire Into CogRecord

```rust
impl CogRecord {
    pub fn set_clam_path(&mut self, path: ClamPath) {
        self.btree_channel[0..2].copy_from_slice(&path.bits.to_be_bytes());
        self.btree_channel[2] = path.depth;
    }
    pub fn clam_path(&self) -> ClamPath { /* inverse */ }
}
```

### 5.3 Triple Semantics

One u16. Three query types. O(log n + k):
- **Address**: Domain.Node.branch.twig.leaf
- **Lineage**: ancestor → parent → self → child (range scan = phylogeny)
- **Causality**: cause → mediator → effect (range scan = causal chain)

---

## PHASE 6: Causal Trajectory Recorder

**File: `ladybug-rs/src/extensions/spo/causal_trajectory.rs` (NEW)**

Detailed spec in `nars_causal_trajectory_hydration_prompt.md`. This is the biggest new module.

### 6.1 Core Structures

```rust
pub struct CausalTrajectory {
    pub input: CogRecord3D,
    pub snapshots: Vec<ResonatorSnapshot>,
    pub rif_diffs: Vec<RifDiff>,
    pub ewm_corrections: Vec<EwmCorrection>,
    pub causal_arrows: Vec<CausalArrow>,
    pub nars_statements: Vec<NarsCausalStatement>,
    pub sigma_edges: Vec<SigmaEdge>,
}

pub struct CausalSaliency {
    pub crystallizing: BitVec,  // dimensions gaining certainty
    pub dissolving: BitVec,     // dimensions losing certainty
    pub contested: BitVec,      // stuck transitional >3 iters
}
```

### 6.2 Per-Iteration Instrumentation

After each resonator update (unbind → project → rebind):

1. **RIF diff** — XOR(iter_t, iter_{t-2}) → what changed in 2 steps (causal chain)
2. **EWM correction** — per-dimension tier classification → WHERE causation acts (saliency)
3. **BPReLU arrow** — forward/backward asymmetry → WHICH DIRECTION it flows
4. **Halo transition** — promotions/demotions → NARS statements

### 6.3 Growth Paths (6 types)

```
SubjectFirst     — S crystallizes first → strong entity
PredicateFirst   — P crystallizes first → strong relation
ObjectFirst      — O crystallizes first → strong target
SPFirst          — S+P simultaneous → "who does what" clear
SOFirst          — S+O simultaneous → "who and whom" clear
FullSimultaneous — all three at once → Gestalt snap (rare)
```

### 6.4 Cost Budget

~15μs overhead per resonator iteration (~30% on existing ~50μs). Acceptable for: full causal trajectory, NARS truth grounded in convergence dynamics, Sigma Graph edges, DN mutation guidance, warm-start capability.

---

## PHASE 7: Gestalt Integration

**Extends existing: `ladybug-rs/src/extensions/spo/gestalt.rs` (DO NOT rewrite — add to it)**

### 7.1 Wire detect_bundling() Into CLAM Harvest Loop

When `AccumulatedHarvest` SO/SP/PO evidence crosses `CollapseMode` threshold → auto-create `BundlingProposal`.

### 7.2 Expose TruthTrajectory via Semantic Redis Protocol

```
GET ada:clam:bundled_with:*    → tentative proposals, "status": "tentative"
GET ada:clam:truth:*           → current NARS truth + trajectory summary
SCAN ada:clam:shift:*          → stripe migration history
```

### 7.3 Connect to neo4j-rs

Edge types: `PROPOSED_BUNDLE` (tentative), `BUNDLED` (committed)
Properties: NARS truth values, σ-significance, reviewer audit trail, growth path

### 7.4 Cross-Validate CHAODA with Stripe Shift

```rust
fn chaoda_confirms_shift(anomaly: &AnomalyResult, shift: &ShiftSignal) -> ConfirmedShift {
    match (anomaly.calibration_type, shift.direction) {
        (Schaltminute, TowardNoise)   => ConfirmedShift::GlobalDrift { confidence: 0.95 },
        (Schaltsekunde, TowardFoveal) => ConfirmedShift::LocalRefinement { confidence: 0.85 },
        (Schaltminute, Bimodal)       => ConfirmedShift::Speciation { confidence: 0.90 },
        _                             => ConfirmedShift::Monitoring { confidence: 0.5 },
    }
}
```

---

## EINEINDEUTIG MAPPING TABLE

Every concept has ONE canonical address across ALL systems. Bijective.

| Concept | Redis Key | Neo4j Property | CLAM Path | Lance Column | Semantic Redis |
|---------|-----------|---------------|-----------|--------------|----------------|
| SPO triple | `ada:spo:{hash}` | `n.spo_hash` | `ClamPath.bits` | `btree_key` | `GET spo:{hash}` |
| NARS truth | `ada:truth:{hash}` | `e.frequency, e.confidence` | — | `meta` field | `GET truth:{hash}` |
| Halo type | `ada:halo:{hash}` | `e.halo_type` | — | derived | `GET halo:{hash}` |
| Trajectory | `ada:trajectory:{hash}` | — | — | `meta` field | `GET trajectory:{hash}` |
| Bundle proposal | `ada:bundle:{id}` | `PROPOSED_BUNDLE` edge | shared prefix | `meta` field | `GET bundle:{id}` |
| Shift signal | `ada:shift:{window}` | — | — | — | `GET shift:{window}` |
| Anomaly score | `ada:anomaly:{hash}` | `n.chaoda_score` | leaf depth | — | `ANOMALY {hash}` |
| σ-significance | `ada:sigma:{hash}` | `n.sigma_level` | — | `meta` field | `NARS {hash}` |

### CogRecord 4-Channel Layout

```
META  (2KB): timestamps, σ-significance, convergence metadata, trajectory summary
CAM   (2KB): content-addressable fingerprint — Hamming search via VPOPCNTDQ
B-tree(2KB): CLAM path — structural/lineage/causal range queries
Embed (2KB): SPO 3-axis XOR encoding — cosine-compatible + typed halo harvest
```

---

## EXECUTION ORDER

```
Phase 2  → Core value (SPO distance harvest — the cosine replacement)
Phase 3  → Near-free upgrades on Phase 2 (raw popcount, histogram, stripe shift)
Phase 5  → ClamPath encoding (type definitions, CogRecord wire)
Phase 6  → Causal trajectory recorder (biggest module, uses Phase 2 types)
Phase 7  → Wires Phase 2-6 into existing src/extensions/spo/gestalt.rs + neo4j-rs + Redis
```

### Minimum Viable

If time is limited: **Phase 2 + Phase 3**. That gives:
- `spo_distance()` as cosine replacement at 238× less cost
- `harvest_to_nars()` + `harvest_to_inference()` from every search
- `AccumulatedHarvest` (search learns from searching)
- Raw popcount + stripe shift detector at zero overhead
- Wire into existing `CollapseGate` via `ShiftSignal`

Everything else builds on that foundation.

---

## WHAT NOT TO CHANGE

```
DO NOT modify rustynum (separate session owns it)
DO NOT rewrite src/extensions/spo/gestalt.rs (add to it, don't replace it)
DO NOT touch the SIMD dispatch in avx_engine.rs (already cleaned up)

KEEP: AVX-512 VPOPCNTDQ path (more specialized than CLAM's distances crate)
KEEP: BindSpace O(1) content-addressable lookup
KEEP: XOR retrieval (A⊗verb⊗B=A — VSA-specific, unique contribution)
KEEP: Arrow/Lance integration (columnar, not Vec<(Id, I)>)
KEEP: COW immutability (freeze after build)
KEEP: HDR cascade + σ-significance (CLAM has no multi-resolution cascade)
KEEP: SPO typed halo harvest (novel contribution, no equivalent anywhere)
```

---

## TEST EXPECTATIONS

```bash
cargo test --package ladybug-rs -- spo_distance      # harvest works
cargo test --package ladybug-rs -- shift_detector     # stripe migration works
cargo test --package ladybug-rs -- clam_path          # encoding round-trips
cargo test --package ladybug-rs -- causal_trajectory  # instrumentation records correctly
cargo test --package ladybug-rs -- gestalt            # bundling detection works
cargo bench -- spo_distance                           # <15 cycles per comparison
```

---

## THE PUNCHLINE

Cosine consumes. SPO harvests. Every search grows the knowledge graph, refines truth values, detects distributional shifts, identifies causal structure, and prepares warm-starts. The resonator's subconscious IS the causal structure of the world it models. And it runs on popcount at 238× less cost than the operation it replaces.

The architecture is confirmed. The types exist. Make it compile.
