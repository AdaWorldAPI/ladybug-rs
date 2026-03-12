# CLAUDE.md — Ladybug-RS

> **Updated**: 2026-03-12
> **Branch**: main
> **CI Status**: Docker PASS, Build FAIL (path dep resolution), Proof FAIL

---

## ⚠ READ BEFORE WRITING ANY CODE

### 1. PATH DEPENDENCIES REQUIRE SIBLING REPOS

ladybug-rs CANNOT compile alone. It depends on sibling repos via relative paths:

```
REQUIRED (clone alongside ladybug-rs):
  ../rustynum/          rustynum-rs, rustynum-core, rustynum-bnn, rustynum-arrow,
                        rustynum-holo, rustynum-clam
  ../crewai-rust/       crewai-vendor (feature-gated behind "crewai")
  ../n8n-rs/            n8n-core, n8n-workflow, n8n-arrow, n8n-grpc, n8n-hamming
```

**To compile locally:**
```bash
mkdir adaworld && cd adaworld
git clone https://github.com/AdaWorldAPI/ladybug-rs
git clone https://github.com/AdaWorldAPI/rustynum
git clone https://github.com/AdaWorldAPI/crewai-rust
git clone https://github.com/AdaWorldAPI/n8n-rs
cd ladybug-rs
cargo check  # NOW it can resolve path deps
```

**To compile without sibling repos (minimal):**
```bash
cargo check --no-default-features --features "simd"
# This skips: lancedb, crewai, spo_jina, parallel
# Most modules still compile — only orchestration/ and flight/ break
```

**CI fails because** GitHub Actions checks out ladybug-rs alone. The Docker build
clones siblings into the build context (Dockerfile.release). CI Master does not.

### 2. spo.rs IS PRIVATE

`src/spo/spo.rs` (1568 lines) contains the REFERENCE architecture.
**It is `mod spo` NOT `pub mod spo`.** All types are private.
Nothing outside `src/spo/` can call SPOCrystal, OrthogonalCodebook, etc.
See `.claude/prompts/17a_spo_rosetta_stone_addendum.md` for full analysis.

**DO NOT** create public types that duplicate spo.rs concepts.
**DO** read spo.rs before implementing anything SPO-related. It is the source of truth.

### 3. FIVE CYPHER PATHS EXIST (Should Be One)

```
P1  src/query/cypher.rs        1560 lines  TRANSPILE ONLY, never executes → TO BE DELETED
P2  src/cypher_bridge.rs        897 lines  ONLY BindSpace writer → KEEP
P3  src/query/lance_parser/    5532 lines  Copy of lance-graph, orphaned → TO BE DELETED
P4  src/learning/cam_ops.rs     ~400 lines  Opcode enum, no executor → KEEP, MOVE
P5  lance-graph (external)     ~15K lines  Real parser+planner → TO BE IMPORTED
```

**DO NOT** add a sixth Cypher path. See `.claude/prompts/17_five_path_teardown.md`.
**DO NOT** call `cypher_to_sql()` expecting results — it returns a SQL string, doesn't execute.
**DO** use `cypher_bridge::execute_cypher()` for writes to BindSpace.

### 4. server.rs STILL USES CogRedis

`src/bin/server.rs` (3681 lines) uses `CogRedis` directly (line 480).
The `RedisAdapter` → `Substrate` → `BindSpace` stack (66K+ lines) is NOT activated.
`PHASE2_SERVER_REWIRE.md` documents the fix but it hasn't been executed.

**DO NOT** assume the server uses RedisAdapter. It doesn't.
**DO NOT** assume `/cypher` executes queries. It transpiles and returns SQL text.

### 5. THREE TruthValue TYPES

```
src/spo/spo.rs:382        private, f64, no evidence count, no inference rules
src/nars/truth.rs          PUBLIC, f32, full NARS inference (deduction/abduction/induction)
lance-graph truth.rs       external, f32, clean TruthGate (3 variants)
```

**USE** `nars::TruthValue` for all new code. The others are prototypes or external.

### 6. FIVE text→fingerprint ENCODERS

```
core/fingerprint.rs         from_content()   LFSR hash, no semantics
spo/nsm_substrate.rs        encode()         NSM primes, keyword-based
spo/codebook_training.rs    encode()         Trained weights
spo/deepnsm_integration.rs  encode()         LLM-based (needs GPU for training)
spo/crystal_lm.rs           encode_clean()   Iterative cleanup
```

None of these are called from any Cypher path. `from_content()` is used by
the server for basic fingerprinting. The NSM/codebook paths need explicit wiring.

### 7. HOT/COLD INVARIANT

**NEVER** let cold path metadata modify hot path state.
See `.claude/prompts/19_hot_cold_separation_constraint.md`.
There is no `fn cold_to_hot()`. Their absence IS the architecture.

---

## Architecture Quick Reference

```
ladybug-rs role: "The Brain" in the four-repo architecture.
  rustynum     = The Muscle (SIMD substrate)
  ladybug-rs   = The Brain (BindSpace, SPO, server)  ← THIS REPO
  staunen      = The Bet (6 instructions, no GPU)
  lance-graph  = The Face (Cypher/SQL query surface)
```

### Key Module Map

```
src/spo/              SPO encoding (18 files, ~14K lines) — the core
src/graph/spo/        SPO on BindSpace (9 files, ~4K lines) — the bridge
src/storage/          BindSpace + Substrate + Redis + Lance (~25K lines)
src/nars/             NARS inference (7 files) — truth values
src/qualia/           10-layer qualia stack (~10K lines) — felt sense
src/orchestration/    crewAI integration (12 files) — agent orchestration
src/query/            DataFusion + parsers (~10K lines) — MESSY, mid-surgery
src/learning/         cam_ops, BNN, causal (~14K lines)
src/cognitive/        7-layer consciousness, rung system
src/search/           HDR cascade, causal search
src/bin/server.rs     HTTP server (3681 lines) — uses CogRedis, not Substrate
src/cypher_bridge.rs  Cypher → BindSpace executor (897 lines) — disconnected
```

### Build Commands

```bash
# Minimal check (no sibling repos needed):
cargo check --no-default-features --features "simd"

# Full check (requires sibling repos):
cargo check

# Default features: simd, parallel, lancedb, crewai, spo_jina

# Run tests:
cargo test --features "spo_jina"

# Build server:
cargo build --bin ladybug-server
```

---

## Session Documents (read in order for full context)

```
.claude/prompts/15_RISC_brain_convergence_vision.md    Vision: 6 instructions, plasticity
.claude/prompts/16_open_brain_surgery_handover.md      5 Cypher paths, disconnection map
.claude/prompts/17_five_path_teardown.md               File-by-file verdicts
.claude/prompts/17a_spo_rosetta_stone_addendum.md      spo.rs analysis, private types
.claude/prompts/18_brain_surgery_orchestration.md      Multi-agent execution plan
.claude/prompts/19_hot_cold_separation_constraint.md   One-way mirror invariant
.claude/prompts/20_four_invariants.md                  Four-repo architecture
.claude/prompts/21_boring_version_plan.md              lance-graph crate separation
.claude/SURGERY_BLACKBOARD.md                          Shared agent state (all PENDING)
.claude/PHASE2_SERVER_REWIRE.md                        server.rs rewire spec
```

---

## Known Issues (Do Not Rediscover)

```
ISSUE                                          STATUS          DOCUMENTED IN
──────────────────────────────────────────────────────────────────────────────
CI Master fails (path dep resolution)          KNOWN           This file §1
211 .unwrap() calls in production paths        KNOWN           TECHNICAL_DEBT.md
9 race conditions (2 critical)                 KNOWN           TECHNICAL_DEBT.md
/cypher endpoint is a stub                     KNOWN           §4 above
server.rs bypasses Substrate                   KNOWN           §4 above
spo.rs types are private                       KNOWN           §2 above
P1+P3 (7092 lines) are dead code              KNOWN           prompt 17
query/lance_parser/ duplicates lance-graph     KNOWN           prompt 17
FINGERPRINT_WORDS vs FINGERPRINT_U64 (156/157) FIXED           PR 163
16K-bit container upgrade                      DONE            PR 150
SPO promoted from extensions/ to src/spo/      DONE            PR 159
SPO Merkle hardening                           DONE            PR 170
```

## What NOT To Do

```
× Don't create .docx files (Jan's rule: "doc" = documentation, not Word)
× Don't add a sixth Cypher parser
× Don't duplicate spo.rs types as public — unlock them or create a facade
× Don't let cold metadata write to hot path
× Don't assume server.rs uses RedisAdapter
× Don't assume /cypher executes queries
× Don't add floats to the SPO vector path (zero floats in hot path)
× Don't import lance-graph as Cargo dep (rewrite natively — prompt 19)
× Don't use `$var:` in PowerShell strings — use `${var}:`
```
