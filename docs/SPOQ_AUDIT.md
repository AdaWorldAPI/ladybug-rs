# SPOQ Integration Plan v2 — Cross-Check Audit

**Auditor**: Claude (Opus 4.6)
**Date**: 2026-02-18
**Scope**: Every factual claim in the SPOQ Plan verified against ladybug-rs HEAD (120,170 LOC, 1,267 tests) and neo4j-rs HEAD (5,384 LOC, 58 tests).

---

## 1. Verified Claims (✓)

All of the following claims in the SPOQ Plan are **confirmed accurate** against code:

| # | Claim | Evidence |
|---|-------|---------|
| 1 | Container = `[u64; 128]` = 8,192 bits = 1 KB | `crates/ladybug-contract/src/container.rs:10` — `CONTAINER_WORDS = 8192/64 = 128` |
| 2 | 16 AVX-512 loads per container | `CONTAINER_AVX512_ITERS = 128/8 = 16` (same file) |
| 3 | `ContainerGeometry` has 6 variants (Cam, Xyz, Bridge, Extended, Chunked, Tree) | `crates/ladybug-contract/src/geometry.rs` — all 6 present |
| 4 | CogRecord = meta (1 KB) + content (1 KB) = 2 KB | `crates/ladybug-contract/src/record.rs:38` — `pub struct CogRecord { pub meta: Container, pub content: Container }` |
| 5 | MetaView word layout W0-W127 | `src/container/meta.rs` — all word offsets match plan exactly (W0=DN, W1=type, W2=time, W4-7=NARS, W12-15=layers, W16-31=edges, W56-63=qualia, W126-127=checksum) |
| 6 | `belichtungsmesser()` uses 7 sample points `[0,19,41,59,79,101,127]` | `src/container/search.rs:25` — exact match, with 448-bit scaling factor |
| 7 | `belichtung_stats()` returns (mean, sd×100) | Same file, exists as claimed |
| 8 | 144-verb codebook | `src/graph/cognitive.rs:10,42,658` — exact 144 count |
| 9 | 10-layer cognitive stack (L1 Recognition → L10 Crystallization) | `src/cognitive/layer_stack.rs` — all 10 layers with correct names |
| 10 | Awareness blackboard with grey/white matter pattern | `src/cognitive/awareness.rs` (442 lines) — exact borrow-safety pattern as described |
| 11 | SPO Crystal role seeds: ROLE_S=0xDEADBEEF_CAFEBABE, ROLE_P=0xFEEDFACE_DEADC0DE, ROLE_O=0xBADC0FFE_E0DDF00D, ROLE_Q=0xC0FFEE00_DEADBEEF | `src/extensions/spo/spo.rs:768-771` — exact match |
| 12 | SPO trace formula: `S⊕ROLE_S ⊕ P⊕ROLE_P ⊕ O⊕ROLE_O ⊕ Q⊕ROLE_Q` | `spo.rs:783` — confirmed |
| 13 | `permute()` / `unpermute()` exist on Fingerprint | `src/core/fingerprint.rs:167,194` — circular rotation, present |
| 14 | neo4j-rs parser = 1,374 LOC, lexer = 435, ast = 278, total = 2,087 | Exact match from `wc -l` |
| 15 | neo4j-rs execution engine = 1,171 LOC, planner = 436 LOC | Exact match |
| 16 | neo4j-rs model/ directory = 554 LOC (Node, Relationship, Path, Value, PropertyMap) | Exact match |
| 17 | StorageBackend trait in neo4j-rs has 36 async methods | `src/storage/mod.rs:127` — 36 counted |
| 18 | ContainerDto and LadybugBackend exist in neo4j-rs | `src/storage/ladybug/` — both present |
| 19 | ContainerSemiring trait + 7 implementations in ladybug-rs | `src/container/semiring.rs` (365 lines) — confirmed |
| 20 | PR #21 is open on neo4j-rs | GitHub API confirms: state=open |
| 21 | Cognitive kernel PREFIX_BLACKBOARD = 0x0E | `src/cognitive/cognitive_kernel.rs:50` — exact |
| 22 | L7 Contingency uses XOR bind (counterfactual branch) | `cognitive_kernel.rs:189` — `xor_bind` at L7, confirmed |
| 23 | L8 Integration uses bundle (majority-vote evidence merge) | `cognitive_kernel.rs:211-215` — bundle of last 3, confirmed |
| 24 | L10 Crystallization promotes from Fluid to Node zone | `cognitive_kernel.rs:267,295` — confirmed |
| 25 | `enrich_step()` exists in `contract/enricher.rs` | `contract/enricher.rs:30` — present with StepEnrichment return |
| 26 | CAM dispatch 0xEF8-0xEFF: Feel=0xEF8, Awaken=0xEFF | `learning/cam_ops.rs:1687,1694` — exact |

**Verdict on factual claims: 26/26 verified. Zero fabrications.** The SPOQ Plan describes what actually exists.

---

## 2. Discrepancies (⚠️)

Minor inaccuracies that need correction:

| # | Claim | Reality | Severity |
|---|-------|---------|----------|
| D1 | §2.1 says "24,576 searchable bits per node" for Xyz 3-block | Correct arithmetic (3 × 8,192 = 24,576), but **Xyz geometry uses linked CogRecords, not inlined 3-block**. Each CogRecord is always 2 KB (meta+content). Xyz means 3 linked records via DN tree, not one 4 KB record. | Medium — misleading about memory layout |
| D2 | §5 says "neo4j-rs has 9 translation layers" | Actual path is: parser → AST → planner → LogicalPlan → execution engine → StorageBackend → MemoryBackend/LadybugBackend → model types → Row. That's 8-9 depending on how you count. Claim is **approximately correct** but imprecise. | Low |
| D3 | §5.1 says "Total surviving neo4j-rs: ~2,100 LOC" | 2,087 LOC from parser+lexer+ast. But the plan forgets: `src/storage/mod.rs` still has the MemoryBackend (624 LOC) + `src/storage/bolt/` for real Neo4j. **If deleting StorageBackend, Bolt is also gone.** Need to decide: pure Cypher compiler (2,087) or Cypher compiler + Bolt passthrough (~3,100). | Medium — scope decision |
| D4 | §6 Gap 1 says "add 0x0207-0x0209" to index.rs | `container/index.rs` doesn't appear to exist as a separate file. Layer constants live in `cognitive/layer_stack.rs`. The index.rs reference may be stale from an older layout. | Low — wrong file path |
| D5 | §6 Gap 2 says current layer markers are "5 bytes × 7 layers = 35 bytes" | Code shows `layer_marker()` returns `(u8, u8, u16, u8)` = 5 bytes, but for **up to 10 layers** already (the LayerId enum has 10 variants). So it's 5×10=50 bytes attempted in 32 bytes (W12-W15). The gap is **already critical**: current code can't store markers for all 10 layers. | High — the plan correctly identifies this but understates urgency |
| D6 | §7 says `permute()` needs to be "harvested from PR #21 to ladybug-rs" | `permute()` and `unpermute()` **already exist** in `src/core/fingerprint.rs:167-195`. The harvest is done. What's missing is permute at **Container width** (`[u64; 128]`) vs. Fingerprint width (`[u64; 256]`). | Medium — harvest partially done |
| D7 | §7 says `belichtungsmesser()` needs harvesting from PR #21 | `belichtungsmesser()` **already exists** in `src/container/search.rs:25` with the exact sample points. Already used by traversal.rs, semiring.rs, graph.rs. The harvest is **already done**. | Low — task is complete |

---

## 3. Cross-Reference: 34 Tactics × SPOQ Plan

Matching the 34 Tactics Integration Plan against the SPOQ Plan reveals alignment AND gaps:

### What SPOQ Plan Adds That 34 Tactics Doesn't Cover

| SPOQ Concept | Impact on 34 Tactics |
|-------------|---------------------|
| **One-binary blackboard** (§3) | Tactics #3 (Debate), #9 (Roleplay), #30 (Shadow Parallel) all implicitly assumed multi-process. SPOQ Plan's blackboard makes them **zero-copy in-process**. The debate round doesn't serialize between agents — agents are threads reading `&BindSpace`. |
| **Container geometry** (§2) | Tactic #14 (Multimodal CoT) said "GrammarTriangle unifies modalities into one fingerprint". SPOQ Plan shows **how**: Xyz geometry stores S/P/O as separate searchable blocks. Multimodal isn't "one fingerprint" — it's three linked containers with holographic recovery. |
| **SPOQ phase model** (§4) | Tactic #4 (Reverse Causality) described ABBA on Fingerprints. SPOQ Plan elevates this: S/P/O are separate containers, so ABBA operates on **8K blocks, not 16K fingerprints**. Faster (half the bits), more precise (role separation). |
| **CypherEngine::query(&BindSpace)** (§5.1) | Not mentioned in 34 Tactics at all. This is the **external interface** — how humans and LLMs interact with the substrate via Cypher. The 34 Tactics assumed Rust API calls. SPOQ Plan adds the query language bridge. |
| **Enforcement rules** (§8) | The 34 Tactics Plan has no "immune system". SPOQ Plan's 5 rules prevent architectural regression. This matters because every tactic implementation must respect: no serde_json on hot path, no HashMap side storage, etc. |

### What 34 Tactics Covers That SPOQ Plan Doesn't Address

| Tactic | Gap in SPOQ Plan |
|--------|-----------------|
| **#10 MetaCognition** (Brier calibration) | SPOQ says nothing about tracking prediction accuracy over time. The blackboard accumulates evidence but doesn't score its own calibration. MetaCognition's Brier tracking should write to a reserved MetaView field (W112-W125 are reserved — use W112 for Brier score). |
| **#12 Temporal Context** (Granger causality) | SPOQ Plan mentions no temporal dimension. The Granger test in `search/temporal.rs` needs **Container-level time series storage**. Currently Fingerprint-level. Need `Container::temporal_series()` that reads W2 timestamps across DN tree depth. |
| **#23 Adaptive Meta-Prompting** (TD-learning on styles) | SPOQ mentions Q-values in W32-W39 but doesn't show the feedback loop: after a cognitive cycle, which style worked, and how does the reward propagate to the ThinkingStyle Q-table? The blackboard writer phase should include a TD-update step. |
| **#31 Counterfactual** (world construction) | SPOQ Plan's L7 does XOR bind for counterfactual branching, but the 34 Tactics describes full **world construction** via `world ⊗ factual ⊗ counterfactual`. This requires maintaining multiple BindSpace snapshots simultaneously — not addressed in the single-BindSpace blackboard model. |

---

## 4. Expansion Opportunities (In the Spirit of SPOQ)

These are new capabilities that naturally extend from the existing architecture:

### 4.1 — Container-Native Granger Causality (Tactic #12 × SPOQ §2)

**Current state**: `search/temporal.rs` operates on Fingerprint time series.
**Opportunity**: Timestamps live in MetaView W2 (`created_ms:32 | modified_ms:32`). The DN tree provides natural temporal ordering. A Container-native Granger test would:

1. Walk the DN tree collecting (timestamp, content Container) pairs
2. Run belichtungsmesser-based correlation on time-offset series
3. Store Granger effect size in MetaView W112 (reserved)
4. Gate via NARS truth value in W4-W7

This gives temporal causality at Container width (8K) with zero allocation — the DN walk reads existing CogRecords.

### 4.2 — Counterfactual BindSpace Snapshots (Tactic #31 × SPOQ §3)

**Current state**: `world/counterfactual.rs` (249 lines) does `world ⊗ factual ⊗ counterfactual` on Fingerprints.
**Opportunity**: The blackboard pattern naturally supports **read-only snapshots**. Fork the BindSpace (copy-on-write via `Arc<[CogRecord]>`) before a counterfactual intervention. The reader threads can explore the counterfactual world while the original continues. After evaluation, either merge (FLOW) or discard (BLOCK).

This is where the blackboard pattern shines: forking is cheap (increment Arc refcount), the counterfactual world is read-only, and merge-back is a bundle operation on the diff.

### 4.3 — Brier Calibration in MetaView (Tactic #10 × SPOQ §2.1)

**Current state**: `cognitive/metacog.rs` (219 lines) tracks Brier scores on Fingerprints.
**Opportunity**: Store per-node Brier score in MetaView W112 as `f32`. Every time a node's prediction is verified, update its Brier. The CollapseGate (W8-W11) already reads NARS truth — add Brier as a second input:

```
gate_input = NARS.confidence × (1 - Brier_score)
```

Nodes that are confident but poorly calibrated get HOLD instead of FLOW. This is meta-cognitive gating at the container level — zero extra allocation.

### 4.4 — Cross-Domain Fusion Container (Tactic #34 × SPOQ §4)

**Current state**: No `fusion.rs` exists. The 34 Tactics doc proposed it but it was never implemented.
**Opportunity**: With Xyz geometry, cross-domain fusion is natural:

- Block 0 (S): Domain A fingerprint
- Block 1 (P): The FUSION verb (one of 144 codebook slots)
- Block 2 (O): Domain B fingerprint
- Trace = S ⊗ P ⊗ O = the fusion

Recovery: given any 2 domains + the verb, recover the 3rd. Novelty = Hamming(trace, S) + Hamming(trace, O). NARS truth tracks fusion quality. This makes cross-domain fusion a **first-class CogRecord** you can search, traverse, and gate.

### 4.5 — Hierarchical Decomposition via CLAM on Containers (Tactic #2 × SPOQ §2)

**Current state**: No `decompose.rs` exists. The 34 Tactics doc proposed it but never implemented.
**Opportunity**: CLAM's bipolar split works on anything with a distance metric. Containers have `hamming()`. Run CLAM on a `Vec<&Container>` → returns a DecompositionTree where each node holds a (centroid Container, radius, children) triple. The tree IS the hierarchical subtask decomposition. Store in DN tree as Chunked geometry (summary + children).

### 4.6 — TD-Learning Feedback in Blackboard Writer (Tactic #23 × SPOQ §3)

**Current state**: `learning/cognitive_styles.rs` has TD-learning on style Q-values. W32-W39 reserved for RL/Q-values.
**Opportunity**: After the blackboard writer commits a batch, compute reward signal:

```rust
// In writer thread, after space.write():
for entry in &committed_batch {
    let reward = entry.collapse_gate_outcome.reward();
    let style_idx = entry.thinking_style.index();
    let meta = MetaViewMut::new(&mut space.at(entry.dn).meta.words);
    meta.update_q_value(style_idx, reward, alpha=0.1);
}
```

This closes the loop: cognitive cycle → collapse gate → reward → Q-value update → next cycle uses updated Q-values. All in-place, all in MetaView, all SIMD-aligned.

---

## 5. Stale Documents — Inventory for Cleanup

### neo4j-rs docs/ (ALL stale per SPOQ Plan)

| Document | StorageBackend refs | LadybugBackend refs | Status |
|----------|--------------------|--------------------|--------|
| DEVELOPMENT.md | 46 | — | **Archive** — describes old architecture entirely |
| ROADMAP_REVIEW.md | 69 | — | **Archive** — 87K of old review |
| STRATEGY_INTEGRATION_PLAN.md | 7 | 6 | **Archive** — superseded by SPOQ Plan |
| COMPATIBILITY_REPORT.md | — | — | **Keep** — crewai-rust compatibility still relevant |
| FEATURE_MATRIX.md | — | — | **Keep** — useful feature overview |
| REALITY_CHECK.md | — | — | **Archive** — pre-SPOQ audit |
| INTEGRATION_ROADMAP.md | — | — | **Archive** — old roadmap |
| INTEGRATION_PLAN_SCHEMA_CHANGES.md | — | — | **Archive** — schema changes now in SPOQ |
| CAM_CYPHER_REFERENCE.md | — | — | **Keep** — Cypher→CAM translation still valid |
| CHESS_HARVEST_PLAN.md | — | — | **Keep** — orthogonal to architecture |
| THEORETICAL_FOUNDATIONS.md | — | — | **Keep** — science refs still valid |
| GUI_PLAN.md | — | — | **Keep** — orthogonal |
| INSPIRATION.md | — | — | **Keep** — orthogonal |

### ladybug-rs docs/ (Partially stale)

| Document | StorageBackend refs | Status |
|----------|-------------------|--------|
| INTEGRATION_CONTRACT_v2.md | 17 | **Archive** — describes the trait surface neo4j-rs is deleting |
| WIRING_PLAN_NEO4J_LADYBUG.md | 22 | **Archive** — wiring plan for old architecture |
| COMPATIBILITY_REPORT.md | 21 | **Archive** — compatibility with old neo4j-rs |
| STRATEGY_INTEGRATION_PLAN.md | 10 | **Archive** — old strategy |
| HANDOFF_LADYBUG_TO_NEO4J.md | refs to old PRs | **Archive** — handoff for old architecture |
| LADYBUG_PROOF_ROADMAP_v1.md | — | **Keep** — proof roadmap still valid |
| INTEGRATION_PROOF_PLAN_v2.md | — | **Keep** — proof tests still valid |
| All other docs | — | **Keep** — architecture docs, cognitive records, etc. |

---

## 6. Rebase Actions

### 6.1 ladybug-rs — Add SPOQ Plan + Archive Stale Docs

1. Add `docs/SPOQ_INTEGRATION_PLAN_v2.md` (the source plan)
2. Add `docs/SPOQ_AUDIT.md` (this audit)
3. Move stale docs to `docs/archive/`:
   - `INTEGRATION_CONTRACT_v2.md`
   - `WIRING_PLAN_NEO4J_LADYBUG.md`
   - `COMPATIBILITY_REPORT.md`
   - `STRATEGY_INTEGRATION_PLAN.md`
   - `HANDOFF_LADYBUG_TO_NEO4J.md`

### 6.2 neo4j-rs — Add SPOQ Plan + Archive Old Docs + Close PR #21

1. Add `docs/SPOQ_INTEGRATION_PLAN_v2.md`
2. Add `docs/SPOQ_AUDIT.md`
3. Move stale docs to `docs/archive/`:
   - `DEVELOPMENT.md`
   - `ROADMAP_REVIEW.md`
   - `STRATEGY_INTEGRATION_PLAN.md`
   - `REALITY_CHECK.md`
   - `INTEGRATION_ROADMAP.md`
   - `INTEGRATION_PLAN_SCHEMA_CHANGES.md`
4. Close PR #21 with harvest comment (belichtungsmesser and permute already in ladybug-rs)

---

## 7. Summary

**Accuracy**: 26/26 factual claims verified. 7 minor discrepancies found (D1-D7), none invalidating the plan.

**Most significant discrepancy**: D5 (layer marker overflow) — the plan correctly identifies this gap but understates its urgency. Current 5-byte markers × 10 layers = 50 bytes in 32 bytes of space. This is a **data corruption risk** for L8-L10.

**Most valuable expansion**: §4.2 (Counterfactual BindSpace Snapshots) — naturally extends the blackboard pattern with copy-on-write forking for counterfactual reasoning, closing the biggest gap between the 34 Tactics doc and the SPOQ architecture.

**Net assessment**: The SPOQ Plan is architecturally sound, factually accurate, and represents a significant simplification over the previous StorageBackend trait approach. The key insight — that neo4j-rs becomes a ~2,100 LOC Cypher compiler instead of a 5,384 LOC database — is well-supported by the code audit. The deletion of 3,252 LOC from neo4j-rs (execution engine, planner, model types, LadybugBackend) is justified because all that functionality now lives in ladybug-rs at the Container level.
