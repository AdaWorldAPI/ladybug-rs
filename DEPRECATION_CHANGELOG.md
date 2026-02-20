# Deprecation Changelog — 2026-02-17

## Architectural Decision: RISC not CISC

**One binary. Zero JSON on hot path. `&BindSpace` borrows, not HTTP.**

Internal operations between ladybug-rs, crewai-rust, n8n-rs, and neo4j-rs
must never serialize. They share one process, one memory space. The blackboard
borrow pattern (`&self` for reads, `&mut self` for writes) replaces all
inter-crate HTTP/JSON/REST/Arrow Flight communication.

JSON/REST endpoints exist ONLY for external consumers (dashboards, third-party
integrations). They are exhaust, not engine.

neo4j-rs is being redesigned as a **Cypher parser that emits BindSpace
operations directly** — not a database, not a service, a query language
compiler. Like the CISC→RISC transition: stop translating, start executing.

---

## PR #129 — ladybug-rs

**Branch**: `deprecate/pr127-128-json-hydrate`
**Merged from**: `main` @ `e2dfd54`
**URL**: https://github.com/AdaWorldAPI/ladybug-rs/pull/129

### Files moved to `.deprecated/pr127_128_json_hydrate/`

| File | Lines | Description |
|------|------:|-------------|
| `server_hydrate_block.rs` | 406 | Extracted handler code from `src/bin/server.rs` |
| `README.md` | — | Rationale, salvage notes |

### Changes to `src/bin/server.rs` (−418 lines)

**Removed handlers:**
- `fn handle_qualia_hydrate()` — POST `/api/v1/qualia/hydrate` (~240 lines)
- `fn handle_qualia_writeback()` — POST `/api/v1/qualia/write-back` (~100 lines)
- `fn text_to_container()` — hash-based Container from message text
- `fn text_to_dn()` — hash-based PackedDn from message text (3-level DN)
- `fn serde_json_escape()` — JSON string escaper (only used by hydrate response)

**Removed from `DbState` struct:**
- `qualia_graph: ContainerGraph` — DN-keyed graph for qualia ops
- `self_dims: SelfDimensions` — mutable self-model persistence
- Corresponding initializers in `DbState::new()`

**Removed route entries:**
- `("POST", "/api/v1/qualia/hydrate") => handle_qualia_hydrate(...)`
- `("POST", "/api/v1/qualia/write-back") => handle_qualia_writeback(...)`
- Comment line: `// Qualia substrate endpoints (holy grail pipeline)`

**Removed imports (hydrate-only):**
- `use ladybug::container::{Container, CONTAINER_BITS};` → removed entirely (Container unused elsewhere)
- `use ladybug::container::adjacency::PackedDn;`
- `use ladybug::container::graph::ContainerGraph;`
- `use ladybug::container::record::CogRecord;`
- `use ladybug::container::geometry::ContainerGeometry;`
- `use ladybug::qualia::texture::GraphMetrics;`
- `use ladybug::qualia::agent_state::{AgentState, PresenceMode, SelfDimensions};`
- `use ladybug::qualia::felt_parse::{GhostEcho, GhostType};`
- `use ladybug::qualia::volition::CouncilWeights;`
- `use ladybug::qualia::{felt_walk, volitional_cycle, harvest_ghosts};`
- `use ladybug_contract::nars::TruthValue as ContractTruthValue;`
- `use ladybug::cognitive::RungLevel;`

**NOT touched:**
- PR #126 (`felt_parse.rs`, `agent_state.rs`) — real substrate work, stays
- All `/api/v1/graph/*` endpoints — unchanged
- UDP bitpacked Hamming handler — unchanged
- All existing tests — unchanged

### Why deprecated

1. **JSON forbidden on internal hot path.** `serde_json::to_string` between
   crates in the same binary = bug. `reqwest::post()` between crates in the
   same binary = bug.
2. **`text_to_dn()` is hash soup.** `DefaultHasher` on message text produces
   meaningless DN positions. SPOQ requires DN paths to encode perspective
   (semantic viewpoint in the tree), not hash collisions.
3. **Hollow pipeline.** PR #127 constructed `FeltPath { choices: vec![] }` and
   `VolitionalAgenda { acts: vec![] }` — empty structs pretending to be
   computed state. PR #128 improved by calling `felt_walk()` but assigned ghost
   types via `match i % 8` (cycling through types by index = random noise).
4. **Wrong paradigm.** Assumed crewai-rust calls ladybug-rs via HTTP POST.
   Correct: one binary, `&BindSpace` borrow.

### What to salvage for the rewrite

- `AgentState::compute()` integration pattern (PR #126 has it natively)
- The route: message → qualia state → preamble for LLM prompt. But as
  `&Container → &Container`, not `JSON → JSON`.
- INTEGRATION_SPEC Layer A concept (preamble for system prompt injection)

---

## PR #20 — neo4j-rs

**Branch**: `deprecate/pr19-container-dto`
**Merged from**: `main` @ `83b80e1`
**URL**: https://github.com/AdaWorldAPI/neo4j-rs/pull/20

### Files moved to `.deprecated/pr19_container_dto/`

| File | Lines | Description |
|------|------:|-------------|
| `fingerprint.rs` | 333 | `ContainerDto` — reimplements `ladybug_contract::Container` |
| `ladybug_module/mod.rs` | 450 | `LadybugBackend` struct + `StorageBackend` impl |
| `ladybug_module/procedures.rs` | 308 | CALL `ladybug.*` procedure dispatch |
| `README.md` | — | Rationale, salvage notes |

### Changes to `src/storage/mod.rs`

```diff
-#[cfg(feature = "ladybug")]
-pub mod ladybug;
+// DEPRECATED: moved to .deprecated/pr19_container_dto/
+// #[cfg(feature = "ladybug")]
+// pub mod ladybug; // → .deprecated/pr19_container_dto/ladybug_module/
```

Comment added to `StorageConfig::Ladybug` variant:
```diff
-    /// ladybug-rs local storage
+    /// ladybug-rs local storage (DEPRECATED: module moved to .deprecated/)
```

The `Ladybug` variant stays in the enum behind `#[cfg(feature = "ladybug")]` —
it won't compile unless the feature is explicitly enabled, which nobody does.

`Cargo.toml` unchanged — `ladybug` feature flag preserved for the RISC rewrite.

**NOT touched:**
- `src/cypher/` (parser, lexer, AST) — the parser is the keeper
- `src/execution/` — will be rewritten but stays for now
- `src/storage/memory.rs` — MemoryBackend stays as test oracle
- `src/model/` — Neo4j value types stay
- `src/chess.rs` — feature-gated, has its own `cfg(feature = "ladybug")` block
- All tests

### Why deprecated

1. **`ContainerDto` duplicates `ladybug_contract::Container`.** 333 lines
   reimplementing `xor()`, `hamming()`, `similarity()`, `random()`, `popcount()`.
   In one-binary model, `use ladybug::container::Container` directly. Zero copy.
2. **9-layer CISC translation.** Cypher → parser → planner → executor →
   StorageBackend dispatch → LadybugBackend → `id_to_addr` HashMap →
   BindSpace → reconstruct Neo4j `Row`. RISC: parser → BindSpace call → done.
3. **`PropertyMap` side-HashMap.** `node_props: HashMap<NodeId, PropertyMap>` stores
   original strings alongside fingerprints. In SPOQ model, properties live in DN
   tree as Container values at path positions.
4. **`NodeId ↔ Addr` BiMap.** Neo4j uses sequential u64 IDs, BindSpace uses
   prefix:slot addressing. The bridge adds a HashMap lookup per operation.
   In RISC model, Cypher variables bind directly to `PackedDn` addresses.

### What to salvage for the RISC rewrite

- **CALL procedures surface** (`ladybug.search`, `ladybug.bind`, `ladybug.similarity`,
  `ladybug.truth`, `ladybug.revise`, `ladybug.spine`, `ladybug.dn.navigate`).
  These become native query semantics, not extension procedures.
- **Verb resolution via Surface 0x07** — correct pattern, keep it.
- **NARS truth revision on relationship creation** — the idea that `create_relationship`
  is evidence accumulation is right.

### The RISC target architecture

```rust
pub struct CypherEngine {
    parser: CypherParser,  // keep — good parser
}

impl CypherEngine {
    pub fn query<'a>(&self, space: &'a BindSpace, cypher: &str) -> QueryResult<'a> {
        let ast = self.parser.parse(cypher);
        execute_ast(space, &ast)  // MATCH → traverse(), WHERE → hamming filter
    }

    pub fn mutate(&self, space: &mut BindSpace, cypher: &str) -> MutationResult {
        let ast = self.parser.parse(cypher);
        execute_mutations(space, &ast)  // CREATE → write(), SET → revise()
    }
}
```

---

## PR #22 — n8n-rs

**Branch**: `deprecate/pr20-21-json-service`
**Merged from**: `master` @ `60428ff`
**URL**: https://github.com/AdaWorldAPI/n8n-rs/pull/22

### Files moved to `.deprecated/`

| Destination | Source | Lines | Description |
|-------------|--------|------:|-------------|
| `.deprecated/pr20_json_workflow/autopoiesis.json` | `n8n-rust/workflows/autopoiesis.json` | ~200 | JSON workflow template with service discovery env vars |
| `.deprecated/pr20_json_workflow/README.md` | — | — | Rationale |
| `.deprecated/pr21_service_contracts/COGNITIVE_WORKFLOW_CONTRACTS.md` | `docs/COGNITIVE_WORKFLOW_CONTRACTS.md` | 1,147 | Arrow Flight RPC contracts between services |
| `.deprecated/pr21_service_contracts/README.md` | — | — | Rationale |

### Files deleted

- `n8n-rust/workflows/autopoiesis.json`
- `docs/COGNITIVE_WORKFLOW_CONTRACTS.md`

### Files kept (NOT deprecated)

- `docs/AUTOPOIESIS_SPEC.md` — the Maturana & Varela model is sound. Q-value
  routing, MUL as immune system, organizational closure. Theory stays.
- `docs/INTEGRATION_EXECUTION_PLAN.md` — 5-phase execution plan stays.
- `docs/COMPATIBILITY_REPORT.md` — stays.
- All Rust source — unchanged.

### Why deprecated

1. **JSON workflow uses service URLs.** `ADA_URL`, `CREWAI_URL`, `LADYBUG_URL`
   environment variables for HTTP calls between components. One binary = direct
   function calls.
2. **Arrow Flight RPC contracts.** Defines gRPC surfaces between n8n-rs and
   ladybug-rs. In one-binary model, these become `&self` method calls on shared
   substrate.

### What to salvage

- The autopoiesis workflow sequence: sovereignty check → felt assessment → body
  scan → visceral composite → qualia modulation → hook evaluation → state
  persistence → dream check. Rewrite as a Rust function chain, not JSON nodes.
- `FreeWillPipeline` 7-step evaluation (type/scope/reversibility/evidence/
  satisfaction/rate-limit/RBAC)
- `TopologyChange` enum: PruneEdge, GrowEdge, DeactivateNode, Replicate
- Q-value routing with MUL-modulated epsilon-greedy

---

## PR #29 — crewai-rust

**Branch**: `deprecate/pr27-review-doc`
**Merged from**: `main` @ `28faa74`
**URL**: https://github.com/AdaWorldAPI/crewai-rust/pull/29

### Files moved to `.deprecated/pr27_review_doc/`

| File | Lines | Description |
|------|------:|-------------|
| `ROADMAP_REVIEW.md` | 1,632 | Cross-ecosystem roadmap review with 22 recommendations |
| `README.md` | — | Rationale |

### Files deleted

- `docs/ROADMAP_REVIEW.md`

### Files kept (NOT deprecated)

- `docs/INTEGRATION_PLAN_SCHEMA_CHANGES.md` — schema plan is sound ✅
- `docs/STRATEGY_INTEGRATION_PLAN.md` — strategy stays ✅
- `docs/AGENT_MUL_CONTRACTS.md` — stays
- `docs/AGENT_ORCHESTRATION_SPEC.md` — stays
- `docs/INTEGRATION_EXECUTION_PLAN.md` — stays
- All Rust source — unchanged.

### Why deprecated

1. Analysis document, not code — belongs in docs repo.
2. References neo4j-rs extensively — wrong repo for this content.
3. Many recommendations superseded by SPOQ/RISC architectural shift.

### What to salvage

- StorageBackend trait analysis (well-designed integration seam assessment)
- Effort estimate validation table (realistic LOC/time estimates)
- Risk #9: BindNode AoS→SoA refactor blocks Phase 4
- Risk #10: fingerprint 156→157 word split (silent corruption)
- Risk #11: CogRecord 256-word redesign changes integration surface

---

## Summary for Claude Code sessions

### The policy (enforce in all future PRs)

1. **No `serde_json` on internal hot path.** If you see `serde_json::to_string`
   or `serde_json::from_str` between crates in the same binary, reject the PR.
2. **No `reqwest::post()` between crates in the same binary.** If two crates
   share a `Cargo.toml` workspace, they communicate via borrows.
3. **No DTO types that duplicate `ladybug-contract`.** `Container`, `CogRecord`,
   `PackedDn`, `TruthValue` — use directly, never reimplement.
4. **No `HashMap<NodeId, PropertyMap>` side storage.** Properties live in the DN
   tree as Containers at path positions.
5. **neo4j-rs = Cypher parser + BindSpace caller.** No executor, no StorageBackend
   trait, no intermediate Row representation.

### What's still in each repo after deprecation

| Repo | Rust LOC (approx) | Key surviving components |
|------|-------------------:|--------------------------|
| ladybug-rs | ~25,000 | BindSpace, Container, SPO Crystal, AVX engine, qualia stack (felt_parse, agent_state, texture, reflection, volition, mul_bridge, dream_bridge), server.rs (graph endpoints + UDP) |
| neo4j-rs | ~5,000 | Cypher parser/lexer/AST, MemoryBackend (test oracle), model types, execution engine (to be rewritten) |
| n8n-rs | ~4,800 | Unified executor, workflow engine, cognitive layer stack, LanceDB/Arrow Flight (feature-gated) |
| crewai-rust | ~60,000 | MetaOrchestrator, Triune persona, flow engine, memory layers, RAG, interface gateway, contract bridge |
