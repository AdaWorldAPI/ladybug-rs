# neo4j-rs Compatibility Report

Cross-project evaluation of **neo4j-rs** against **ladybug-rs**, **n8n-rs**, **crewai-rust**, and **aiwar-neo4j-harvest**.

> **Key insight**: n8n-rs, ladybug-rs, and crewai-rust are already optimized for each other via shared `ladybug-contract` types and unified execution contracts. **neo4j-rs is the newcomer** — this report evaluates how cleanly it slots into the existing ecosystem.

---

## 1. Dependency Alignment Matrix

| Dimension | neo4j-rs | ladybug-rs | n8n-rs | crewai-rust | aiwar-neo4j-harvest |
|-----------|----------|------------|--------|-------------|---------------------|
| **Edition** | 2024 | 2024 | 2021 | 2021 | 2021 |
| **MSRV** | 1.88 | 1.88 | 1.93 | (unset) | (unset) |
| **Async runtime** | Tokio 1.49 | Tokio 1.49 | Tokio 1.35 | Tokio 1.x | Tokio 1.x |
| **serde** | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| **chrono** | 0.4 | 0.4 | 0.4.38 | 0.4 | -- |
| **parking_lot** | 0.12 | 0.12 | 0.12 | 0.12 | -- |
| **async-trait** | 0.1 | 0.1 | 0.1 | 0.1 | -- |
| **thiserror** | 2.0 | 2.0 | 1.0 | 2.0 | -- |
| **Arrow** | 57 (opt) | 57 | 57 | -- | -- |
| **DataFusion** | -- | 51 | 51 | -- | -- |
| **tracing** | 0.1 | 0.1 | 0.1 | -- | 0.1 |
| **hashbrown** | 0.15 | 0.15 | -- | -- | -- |
| **smallvec** | 1.15 | 1.15 | -- | -- | -- |
| **Neo4j driver** | (self) | neo4rs 0.8 | -- | -- | neo4rs 0.8 |

**Verdict**: No version conflicts. All Rust projects share the Tokio ecosystem and serde stack. The edition differences (2024 vs 2021) are transparent to downstream dependents since edition only affects the crate's own source, not its public API.

---

## 2. neo4j-rs <> ladybug-rs

### Status: **Designed to integrate** (explicit feature flag exists)

neo4j-rs already declares ladybug-rs as an optional dependency:

```toml
# neo4j-rs/Cargo.toml
ladybug = { path = "../ladybug-rs", optional = true }

[features]
ladybug = ["dep:ladybug", "dep:tokio"]
```

The `StorageBackend` trait in `neo4j-rs/src/storage/mod.rs` defines the integration contract with ~25 required methods + ~20 default methods. A `LadybugBackend` module is declared but not yet implemented:

```rust
#[cfg(feature = "ladybug")]
pub mod ladybug;  // gated, no file yet
```

### Compatibility Assessment

| Aspect | Compatible? | Notes |
|--------|-------------|-------|
| Async runtime | **Yes** | Both Tokio 1.49 |
| Data model | **Mappable** | neo4j-rs `Node/Relationship/PropertyMap` maps to ladybug's 8+8 BindSpace addresses |
| Cypher | **Complementary** | neo4j-rs has full Cypher parser+executor; ladybug has Cypher-to-SQL transpiler |
| Storage | **Interface ready** | `StorageBackend` trait is the contract; ladybug needs to implement it |
| Arrow | **Aligned** | Both use Arrow 57; neo4j-rs `arrow-results` feature streams `RecordBatch` |
| Vector search | **Natural fit** | `StorageBackend::vector_query()` maps to ladybug's HDR cascade + SIMD Hamming |
| Procedures | **Extension point** | `call_procedure()` can expose ladybug's 4096 CAM operations as Neo4j procedures |

### Integration Architecture

```
                          neo4j-rs
                     ┌─────────────────┐
  Cypher query ───>  │ Parser → Planner │
                     │    → Executor    │
                     └────────┬────────┘
                              │ StorageBackend trait
                              ▼
                     ┌─────────────────┐
                     │  LadybugBackend │
                     │                 │
                     │  create_node()  │ → BindSpace.write(addr, fp)
                     │  expand()       │ → BitpackedCSR traversal
                     │  vector_query() │ → HDR cascade + SIMD Hamming
                     │  call_procedure │ → CAM operations (4096 ops)
                     └─────────────────┘
```

### Key Mapping

| neo4j-rs concept | ladybug-rs concept |
|------------------|--------------------|
| `NodeId(u64)` | `Addr(prefix:u8, slot:u8)` — pack into u16, embed in u64 |
| `labels: Vec<String>` | Fingerprint dimensions (encode labels as bit patterns) |
| `PropertyMap` | Fingerprint + metadata in BindSpace cell |
| `Relationship` | 144 cognitive verbs (IS_A, CAUSES, BEFORE, ...) |
| `Path` | Traversal result from BitpackedCSR |
| `vector_query()` | HDR cascade search with Hamming distance |
| `call_procedure()` | CAM operation dispatch (RESONATE, SUPERPOSE, INHIBIT, ...) |

### What Needs Building

1. **`src/storage/ladybug.rs`** — Implement `StorageBackend` for ladybug's BindSpace
2. **Address translation** — Map `NodeId(u64)` to/from 8+8 BindSpace addresses
3. **Label encoding** — Encode Neo4j labels as fingerprint dimensions
4. **Relationship mapping** — Map arbitrary rel_types to ladybug's 144 verbs (or extend)
5. **Capability reporting** — Return `similarity_accelerated: true`, `supports_vector_index: true`

### Bilateral Value

- **neo4j-rs gains**: SIMD-accelerated similarity search, 16K-bit fingerprints, HDR cascade filtering, cognitive operations
- **ladybug-rs gains**: Full Cypher language (currently only transpiler), proper graph traversal semantics, ACID transaction framework

---

## 3. neo4j-rs <> n8n-rs

### Status: **No direct dependency yet; high synergy potential**

n8n-rs is a workflow automation engine that already depends on ladybug-rs. There is no direct dependency on neo4j-rs, but the integration path is clear through ladybug-rs as middleware.

### Compatibility Assessment

| Aspect | Compatible? | Notes |
|--------|-------------|-------|
| Async runtime | **Yes** | Both Tokio (1.49 vs 1.35 — compatible within semver) |
| Arrow | **Aligned** | Both Arrow 57 |
| DataFusion | **Shared** | ladybug 51, n8n 51 |
| Data model | **Complementary** | n8n workflows are DAGs; neo4j-rs stores property graphs |
| Transport | **Flight-compatible** | n8n-rs has Arrow Flight; neo4j-rs has `arrow-results` feature |

### Integration Architecture

n8n-rs can use neo4j-rs in two modes:

**Mode A: Direct dependency** — Add neo4j-rs as a node executor backend

```
n8n Workflow
    │
    ▼
┌───────────────────┐
│ Neo4jNodeExecutor  │  ← New node type: "neo4j-rs.cypherQuery"
│                    │
│  Accepts: Cypher   │
│  Returns: Arrow    │  ← via neo4j-rs arrow-results feature
└────────┬──────────┘
         │
         ▼
    neo4j-rs::Graph<MemoryBackend>  (embedded)
    neo4j-rs::Graph<LadybugBackend> (via ladybug-rs)
```

**Mode B: Through ladybug-rs** — Leverage existing ladybug dependency

```
n8n-rs → ladybug-rs → (internal LadybugBackend) → neo4j-rs
```

n8n-rs already depends on ladybug-rs via workspace path. Once ladybug implements `StorageBackend`, n8n workflows automatically get Cypher query support.

### Value Proposition

| For n8n-rs | Benefit |
|------------|---------|
| **Workflow graph persistence** | Store workflow DAGs as Neo4j property graphs instead of JSON blobs in PostgreSQL |
| **Execution lineage** | Track `(Execution)-[:STARTED]->(Node)-[:OUTPUTS_TO]->(Node)` as native graph traversals |
| **Credential dependency graphs** | Query `(Credential)-[:USED_BY]->(Node)-[:IN]->(Workflow)` for rotation impact analysis |
| **Hamming-based workflow similarity** | n8n-hamming vectors + neo4j-rs `vector_query()` = find similar workflows |

### What Needs Building

1. **Neo4jNodeExecutor** — Implement `NodeExecutor` trait for Cypher queries
2. **Arrow bridge** — Stream `QueryResult` as Arrow `RecordBatch` (neo4j-rs `arrow-results` feature already exists)
3. **Workflow-to-graph** — Serializer from n8n `Workflow` struct to neo4j-rs `CREATE` statements

---

## 4. neo4j-rs <> crewai-rust

### Status: **Fully compatible, native Rust integration**

[crewai-rust](https://github.com/AdaWorldAPI/crewai-rust) is a standalone **58K-line Rust port** of the crewAI framework (v1.9.3) with 443 passing tests, 0 compile errors. It already integrates with ladybug-rs via `ladybug-contract` shared types.

### Compatibility Assessment

| Aspect | neo4j-rs | crewai-rust | Compatible? |
|--------|----------|-------------|-------------|
| Edition | 2024 | 2021 | **Yes** (transparent to Cargo) |
| Async runtime | Tokio 1.49 | Tokio 1.x (full) | **Yes** |
| serde | 1.0 | 1.0 | **Yes** |
| chrono | 0.4 | 0.4 | **Yes** |
| parking_lot | 0.12 | 0.12 | **Yes** |
| async-trait | 0.1 | 0.1 | **Yes** |
| thiserror | 2.0 | 2.0 | **Yes** |
| dashmap | -- | 6.0 | N/A |
| Storage | `StorageBackend` trait | rusqlite (LTM), in-memory RAG | **Complementary** |
| ladybug-rs | path dep (optional) | ladybug-contract path dep | **Shared substrate** |

**Zero dependency conflicts.** Both crates can coexist in the same workspace.

### crewai-rust Architecture (Key Components for Integration)

```
crewai-rust (58K LOC, 258 files)
├── Agent           → role/goal/backstory, LLM provider, tools, memory
├── Crew            → orchestrates agents: Sequential / Hierarchical / Consensual
├── Task            → description, expected output, agent assignment, guardrails
├── MetaOrchestrator→ auto-attended controller (spawns agents dynamically)
├── SkillEngine     → EMA proficiency tracking, cross-agent skill transfer
├── Delegation      → Request → Dispatch → Response → Result protocol
├── Contract        → Unified execution steps (crew.* / lb.* / n8n.* routing)
├── Memory          → SQLite LTM, in-memory RAG, short/long/entity/contextual
├── Flow            → Event-driven workflows (@start, @listen, @router)
├── Events          → Full lifecycle event bus (agent/crew/task/memory/LLM)
├── A2A             → Agent-to-Agent HTTP protocol (agent cards, messages)
└── LLM Providers   → OpenAI, Anthropic, xAI (Azure/Bedrock/Gemini stubbed)
```

### Unified Execution Contract (Already Built)

crewai-rust has a `contract/` module that routes execution steps by prefix:

```rust
// contract/types.rs — already implemented
impl UnifiedStep {
    pub fn is_crew(&self)    -> bool  // step_type starts with "crew."
    pub fn is_ladybug(&self) -> bool  // step_type starts with "lb."
    pub fn is_n8n(&self)     -> bool  // step_type starts with "n8n."
}
```

This means **adding `neo4j.*` step routing is a natural extension** — crewai-rust already understands multi-backend dispatch.

### Integration Architecture

**Path 1: Direct neo4j-rs dependency** (recommended)

```rust
// crewai-rust/Cargo.toml
neo4j-rs = { path = "../neo4j-rs" }

// Neo4j-backed knowledge base for agents
struct Neo4jKnowledgeStore {
    graph: neo4j_rs::Graph<MemoryBackend>,
}

#[async_trait]
impl Storage for Neo4jKnowledgeStore {
    async fn save(&self, key: &str, data: &str, metadata: &HashMap<String, String>) {
        self.graph.mutate(
            "CREATE (k:Knowledge {key: $key, content: $data})",
            [("key", key.into()), ("data", data.into())]
        ).await.ok();
    }

    async fn search(&self, query: &str, limit: usize) -> Vec<StorageResult> {
        let result = self.graph.execute(
            "MATCH (k:Knowledge) WHERE k.content CONTAINS $q RETURN k LIMIT $limit",
            [("q", query.into()), ("limit", (limit as i64).into())]
        ).await.unwrap();
        // map to StorageResult
    }
}
```

**Path 2: Via ladybug-contract** (decoupled)

crewai-rust already depends on `ladybug-contract` for shared types. Once ladybug-rs implements neo4j-rs's `StorageBackend`, crewai-rust gets Cypher support transitively through the unified contract:

```
crewai-rust → UnifiedStep { step_type: "lb.cypher_query" }
           → ladybug-rs (LadybugBackend)
           → neo4j-rs (Cypher parser + executor)
```

**Path 3: SkillEngine as Neo4j graph** (highest-value target)

The MetaOrchestrator + SkillEngine + Delegation protocol form a natural property graph:

```cypher
// Agent skill proficiency
(Agent {role: "Strategist"})-[:KNOWS_SKILL {proficiency: 0.85, ema: 0.92}]->(Skill {name: "opening_theory"})

// Delegation traces
(Agent1)-[:DELEGATED {task_id: "t1", timestamp: datetime()}]->(Agent2)

// Task dependency graph
(Task {name: "Analyze"})-[:DEPENDS_ON]->(Task {name: "Research"})

// Crew execution history
(Crew)-[:EXECUTED]->(Task)-[:ASSIGNED_TO]->(Agent)-[:USED_TOOL]->(Tool)
```

### Value Proposition

| For crewai-rust | Benefit |
|-----------------|---------|
| **SkillEngine persistence** | Replace in-memory DashMap with Neo4j graph; survive restarts; query cross-agent skill networks |
| **Delegation trace graph** | `(Agent)-[:DELEGATED]->(Agent)` call graphs for debugging and optimization |
| **Agent knowledge graph** | Multi-hop retrieval: "Find knowledge related to X via agents who solved similar tasks" |
| **MetaOrchestrator memory** | Store orchestration decisions as graph; learn optimal agent composition patterns |
| **Execution trace analysis** | Full DAG of `Crew → Task → Agent → Tool` for post-mortem analysis |
| **A2A network topology** | Map agent-to-agent communication as a graph; identify bottlenecks |

### What Needs Building

1. **`Neo4jKnowledgeStore`** — Implement crewai-rust's `Storage` trait backed by neo4j-rs
2. **`Neo4jSkillBackend`** — Persist SkillEngine proficiency graph to Neo4j
3. **Contract step router** — Add `neo4j.*` prefix handling alongside existing `crew.*`/`lb.*`/`n8n.*`
4. **Feature flag** — `#[cfg(feature = "neo4j")]` gating in crewai-rust's Cargo.toml

---

## 5. aiwar-neo4j-harvest <> The Full Stack

### Current State

aiwar-neo4j-harvest currently uses `neo4rs 0.8` (the third-party Neo4j Bolt driver) to ingest 221 nodes + 356 edges from the AI War Cloud dataset into an external Neo4j server.

**Key distinction**: `neo4rs` (third-party Bolt client) vs `neo4j-rs` (this project, a full Rust reimplementation).

### Can aiwar-neo4j-harvest use neo4j-rs instead?

**Yes, and it gains significant capabilities:**

| Capability | With neo4rs 0.8 | With neo4j-rs |
|------------|-----------------|---------------|
| External Neo4j | Yes (Bolt) | Yes (Bolt feature, planned) |
| Embedded graph | No | **Yes** (MemoryBackend) |
| Full Cypher | External server | **Built-in** parser + executor |
| Offline analysis | No (needs server) | **Yes** (in-process) |
| Arrow streaming | No | **Yes** (arrow-results feature) |
| Ladybug similarity | No | **Yes** (ladybug feature) |
| Batch ingest | Manual | **Yes** (create_nodes_batch, create_relationships_batch) |

### Integration Architecture: aiwar-neo4j-harvest on the Full Stack

```
┌──────────────────────────────────────────────────────────────────┐
│                    aiwar-neo4j-harvest                           │
│              (CLI: cypher | neo4j | analyze)                     │
└───────┬──────────────────────┬──────────────────┬────────────────┘
        │                      │                  │
        ▼                      ▼                  ▼
   neo4j-rs              ladybug-rs           crewAI agents
   (embedded)            (similarity)         (analysis)
   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
   │ Graph<Memory>│      │ HDR cascade │      │ Risk Analyst│
   │ 221 nodes    │      │ Fingerprint │      │ Stakeholder │
   │ 356 edges    │      │ SIMD Hamming│      │   Mapper    │
   │ Full Cypher  │      │ CAM ops     │      │ Graph Query │
   └──────┬──────┘      └──────┬──────┘      └──────┬──────┘
          │                     │                     │
          └──────────┬──────────┘                     │
                     ▼                                │
              neo4j-rs::Graph<LadybugBackend>         │
              (Cypher + similarity in one engine)     │
                     │                                │
                     ├────────────────────────────────┘
                     ▼
              n8n-rs workflows
              ┌──────────────────────────────┐
              │ Trigger: new AI system report │
              │ → Ingest into neo4j-rs graph │
              │ → Compute similarity scores  │
              │ → Alert if high-risk pattern │
              │ → crewAI deep analysis       │
              └──────────────────────────────┘
```

### The 7 Novel Patterns Meet the Stack

The 7 graph patterns harvested by aiwar-neo4j-harvest map naturally to the stack:

| Pattern | neo4j-rs | ladybug-rs | n8n-rs | crewAI |
|---------|----------|------------|--------|--------|
| **1. Faceted multi-label** | Native (labels on Node) | Fingerprint dimensions | Workflow tags | Agent role labels |
| **2. Schema-as-data** | Queryable via Cypher | BindSpace meta-addresses | Workflow schema discovery | Agent capability ontology |
| **3. Dual-role bipartite** | Full relationship model | 144 verb graph | Node connection types | Agent delegation patterns |
| **4. Icon-addressed** | Property on nodes | noun_key as memory address | Visual workflow builder | Agent avatar/identity |
| **5. Hierarchical meta-edges** | Self-referential Cypher | BindSpace prefix hierarchy | Nested workflow execution | Crew hierarchy |
| **6. Temporal status flow** | Property + label changes | Temporal CogRedis TTL | Execution history | Task lifecycle |
| **7. AIRO ontology** | Properties on nodes | Fingerprint metadata | Risk alert conditions | Agent risk assessment |

### Concrete Migration Path

Replace `neo4rs 0.8` with `neo4j-rs` in `Cargo.toml`:

```toml
# Before
neo4rs = "0.8"

# After
neo4j-rs = { path = "../neo4j-rs" }
# Or, for external Neo4j fallback:
neo4j-rs = { path = "../neo4j-rs", features = ["bolt"] }
```

Then rewrite `ingest.rs` from string-based Cypher generation to typed API:

```rust
// Before: generate Cypher strings, send via neo4rs Bolt
let cypher = format!("CREATE (n:System {{id: '{}'}})", system.id);
graph.run(query(&cypher)).await?;

// After: use neo4j-rs typed API (embedded, no server needed)
let graph = Graph::<MemoryBackend>::open_memory()?;
let mut tx = graph.begin(TxMode::ReadWrite).await?;
graph.mutate(
    "CREATE (n:System {id: $id, name: $name})",
    [("id", system.id.into()), ("name", system.name.into())]
).await?;
```

---

## 6. Compatibility Risk Summary

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Edition 2024 vs 2021** | Low | Edition only affects the crate's own source code, not its API. Mixed-edition dependency graphs are fully supported by Cargo. |
| **thiserror 2.0 vs 1.0** | Low | n8n-rs uses thiserror 1.0; neo4j-rs uses 2.0. These can coexist in the same dependency graph (different major versions = different crates). |
| **Tokio version spread** (1.35-1.49) | None | Cargo unifies to the highest compatible version within semver. Tokio 1.x is fully backward-compatible. |
| **neo4j-rs Bolt backend not implemented** | Medium | aiwar-neo4j-harvest currently needs Bolt to talk to external Neo4j. Can use MemoryBackend for offline analysis now, Bolt when implemented. |
| **ladybug StorageBackend not implemented** | Medium | The trait contract exists; implementation is the next step. No design risk. |
| **crewai-rust agent wiring gap** | Low | Agent.execute_without_timeout() needs ~50 lines to call CrewAgentExecutor. Infrastructure is complete. |

---

## 7. Recommendation: Integration Priority

### Phase 1 — Foundation
1. **Implement `LadybugBackend`** in `neo4j-rs/src/storage/ladybug.rs`
   - Address translation (NodeId ↔ 8+8 BindSpace)
   - Node/relationship CRUD mapped to BindSpace operations
   - `vector_query()` → HDR cascade + SIMD Hamming
   - `call_procedure()` → CAM operation dispatch

### Phase 2 — Harvest Migration
2. **Migrate aiwar-neo4j-harvest** from `neo4rs 0.8` to `neo4j-rs`
   - Use `MemoryBackend` for offline analysis (no server dependency)
   - Preserve Cypher generation for external Neo4j compatibility
   - Add `LadybugBackend` mode for similarity-augmented harvesting

### Phase 3 — Workflow Integration
3. **Add Neo4jNodeExecutor to n8n-rs**
   - New node type for Cypher queries in workflows
   - Arrow streaming of results via `arrow-results` feature
   - Workflow-to-graph serialization for execution lineage

### Phase 4 — Agent Intelligence
4. **crewai-rust Neo4j knowledge base**
   - Agents query the graph for knowledge retrieval
   - Execution traces stored as graph paths
   - Agent capability matching via Cypher patterns

---

## 8. Conclusion

neo4j-rs is architecturally aligned with the entire Ada ecosystem. The `StorageBackend` trait is the linchpin: once ladybug-rs implements it, all downstream consumers (n8n-rs, crewAI, aiwar-neo4j-harvest) gain Cypher query support and graph semantics automatically.

The dependency versions are compatible across all projects with zero conflicts. The main work ahead is implementation of the `LadybugBackend` — the interfaces are already defined and waiting.

aiwar-neo4j-harvest specifically benefits from the full stack by:
- Eliminating the external Neo4j server dependency (embedded `MemoryBackend`)
- Adding similarity search over AI systems (ladybug's Hamming fingerprints)
- Automating harvest pipelines (n8n-rs workflows)
- Enabling multi-agent risk analysis (crewai-rust crews querying the graph)

---

## 9. Vision: Chess AI on the Full Stack

Building on aiwar-neo4j-harvest's graph pattern insights, the stack can power a
chess engine that uses **spawned agents, inner thought loops, and higher reasoning**
rather than brute-force search.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         n8n-rs  (Game Loop)                         │
│                                                                     │
│  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌───────────────┐  │
│  │ Trigger  │───>│ Position │───>│ Agent    │───>│ Move Execute  │  │
│  │ (clock)  │    │ Ingest   │    │ Dispatch │    │ + Board Update│  │
│  └─────────┘    └──────────┘    └──────────┘    └───────────────┘  │
│        ▲                              │                    │        │
│        └──────────────────────────────┴────────────────────┘        │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ UnifiedStep { step_type: "crew.chess_move" }
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    crewai-rust  (Agent Crews)                       │
│                                                                     │
│  MetaOrchestrator                                                   │
│  ├── Spawns agents dynamically based on game phase                  │
│  ├── SkillEngine tracks proficiency per opening/endgame             │
│  └── Delegation protocol routes sub-problems                        │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ Crew: "ChessThinkTank"                                       │   │
│  │ Process: Hierarchical (manager = Strategist)                 │   │
│  │                                                              │   │
│  │  Agent: Strategist (manager)                                 │   │
│  │  ├─ role: "Grand Strategist"                                 │   │
│  │  ├─ goal: "Choose the best plan for this position"           │   │
│  │  ├─ tools: [neo4j_query, ladybug_similarity, board_eval]     │   │
│  │  └─ Inner thought loop:                                      │   │
│  │       1. Query opening book graph (neo4j-rs)                 │   │
│  │       2. Find similar historical positions (ladybug)         │   │
│  │       3. Evaluate candidate plans                            │   │
│  │       4. Delegate to specialists                             │   │
│  │                                                              │   │
│  │  Agent: Tactician                                            │   │
│  │  ├─ role: "Tactical Calculator"                              │   │
│  │  ├─ goal: "Find forcing sequences and combinations"          │   │
│  │  ├─ tools: [move_generator, position_eval, threat_detector]  │   │
│  │  └─ Inner thought loop:                                      │   │
│  │       1. Generate candidate moves                            │   │
│  │       2. For each: simulate N plies                          │   │
│  │       3. Evaluate tactical consequences                      │   │
│  │       4. Report best tactical line to Strategist             │   │
│  │                                                              │   │
│  │  Agent: Endgame Specialist                                   │   │
│  │  ├─ role: "Endgame Expert"                                   │   │
│  │  ├─ goal: "Convert advantages in simplified positions"       │   │
│  │  ├─ tools: [tablebase_lookup, pawn_structure_eval]           │   │
│  │  └─ Spawned only when piece_count < 10                       │   │
│  │                                                              │   │
│  │  Agent: Psychologist  (self-play / opponent modeling)        │   │
│  │  ├─ role: "Opponent Modeler"                                 │   │
│  │  ├─ goal: "Predict opponent's plan and exploit tendencies"   │   │
│  │  ├─ tools: [game_history_query, style_fingerprint]           │   │
│  │  └─ Queries neo4j for opponent's past games pattern          │   │
│  │                                                              │   │
│  │  Agent: Inner Critic  (higher reasoning)                     │   │
│  │  ├─ role: "Devil's Advocate"                                 │   │
│  │  ├─ goal: "Challenge every proposed move with refutations"   │   │
│  │  ├─ tools: [counter_argument, blunder_check]                 │   │
│  │  └─ Receives Strategist's plan, tries to refute it          │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    neo4j-rs  (Game Knowledge Graph)                  │
│                                                                     │
│  Graph<MemoryBackend> or Graph<LadybugBackend>                      │
│                                                                     │
│  Node Types:                                                        │
│  ├── (:Position {fen, eval, phase, piece_count})                    │
│  ├── (:Move {san, uci, piece, capture, check})                     │
│  ├── (:Opening {eco, name, main_line})                              │
│  ├── (:Game {id, result, white, black, date})                       │
│  ├── (:Plan {description, theme, success_rate})                     │
│  ├── (:Pattern {name, type, fingerprint})  ← aiwar-style facets    │
│  └── (:AgentDecision {agent, reasoning, confidence, timestamp})     │
│                                                                     │
│  Relationships:                                                     │
│  ├── (Position)-[:MOVE {san, eval_delta}]->(Position)               │
│  ├── (Position)-[:BELONGS_TO]->(Opening)                            │
│  ├── (Game)-[:PLAYED]->(Position)                                   │
│  ├── (Position)-[:SIMILAR_TO {hamming_dist}]->(Position)  ← ladybug│
│  ├── (AgentDecision)-[:CHOSE]->(Move)                               │
│  ├── (AgentDecision)-[:EVALUATED]->(Position)                       │
│  ├── (AgentDecision)-[:DELEGATED_BY]->(AgentDecision)               │
│  └── (Plan)-[:APPLIES_TO]->(Pattern)                                │
│                                                                     │
│  Cypher Queries (inner thought loop tools):                         │
│                                                                     │
│  // Opening book lookup                                             │
│  MATCH (p:Position {fen: $fen})-[m:MOVE]->(next)                    │
│  RETURN m.san, next.eval, count{ (next)-[:PLAYED]->(:Game) } as freq│
│  ORDER BY freq DESC LIMIT 5                                         │
│                                                                     │
│  // Find similar positions agent has seen before                    │
│  MATCH (p:Position)-[:SIMILAR_TO]-(known:Position)                  │
│  WHERE known.eval IS NOT NULL                                       │
│  RETURN known.fen, known.eval, known.phase                          │
│                                                                     │
│  // Replay agent reasoning chain (higher reasoning)                 │
│  MATCH path = (d1:AgentDecision)-[:DELEGATED_BY*]->(d0)             │
│  WHERE d0.position_fen = $fen                                       │
│  RETURN [n IN nodes(path) | n.reasoning] AS thought_chain           │
│                                                                     │
│  // Learn from self-play: what plans worked?                        │
│  MATCH (plan:Plan)-[:APPLIES_TO]->(pat:Pattern)                     │
│  WHERE plan.success_rate > 0.7                                      │
│  RETURN plan.description, plan.theme, pat.name                      │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ladybug-rs  (Position Fingerprints)               │
│                                                                     │
│  Every chess position → 16,384-bit fingerprint encoding:            │
│  ├── Piece placement (64 squares x 12 piece types)                  │
│  ├── Pawn structure (doubled, isolated, passed, chains)             │
│  ├── King safety (pawn shield, open files, attack vectors)          │
│  ├── Piece activity (mobility, centralization, coordination)        │
│  ├── Material balance                                               │
│  └── Tactical motifs (pins, forks, skewers, discovered attacks)     │
│                                                                     │
│  Operations:                                                        │
│  ├── RESONATE(position_fp, k=10) → 10 most similar known positions  │
│  ├── SUPERPOSE(plan_fps...) → combined strategic concept            │
│  ├── INHIBIT(position_fp, blunder_fp) → avoid blunder patterns      │
│  └── XOR(white_plan, black_plan) → strategic tension measure        │
│                                                                     │
│  HDR Cascade:                                                       │
│  ├── Level 0: "Is this an open or closed position?" (1-bit)         │
│  ├── Level 1: "Which piece structure family?" (4-bit)               │
│  ├── Level 2: "Which tactical theme cluster?" (8-bit)               │
│  └── Level 3: Full Hamming distance (exact similarity)              │
│                                                                     │
│  Stored in neo4j-rs via LadybugBackend:                             │
│  ├── vector_query("positions", 10, position_fp) → similar positions │
│  └── call_procedure("resonate", [position_fp, 10]) → CAM search    │
└─────────────────────────────────────────────────────────────────────┘
```

### aiwar-neo4j-harvest Patterns Applied to Chess

The 7 novel graph patterns from aiwar-neo4j-harvest map directly:

| aiwar Pattern | Chess Application |
|---------------|-------------------|
| **1. Faceted multi-label** | `(:Position:Middlegame:OpenPosition:KingsidePressure)` — positions carry multiple strategic labels simultaneously |
| **2. Schema-as-data** | `(:SchemaAxis {name: "phase"})-[:VALID_FOR]<-(:SchemaValue {value: "Middlegame"})` — chess ontology as queryable graph |
| **3. Dual-role bipartite** | Same position appears in opening book AND endgame tablebase; agents are both decision-makers AND subjects of analysis |
| **4. Icon-addressed** | `noun_key` for positions → visual board thumbnails as memory addresses |
| **5. Hierarchical meta-edges** | Strategist's plan → Tactician's calculation → Move selection (nested decision DAG) |
| **6. Temporal status flow** | Opening → Middlegame → Endgame lifecycle; agent confidence changes over game |
| **7. AIRO ontology** | Risk assessment on moves: `{risk: "speculative", reversibility: "low", refutation_depth: 5}` |

### Inner Thought Loops

Each agent runs a **ReAct loop** (already implemented in crewai-rust's `CrewAgentExecutor`):

```
Strategist receives: board position + game history + opponent model
  │
  ├─ Thought: "Position is slightly better, I should press on the kingside"
  │   Action: neo4j_query("MATCH (p:Position {fen: $fen})-[:MOVE]->(n) RETURN ...")
  │   Observation: "3 candidate moves, Nf5 played in 67% of GM games"
  │
  ├─ Thought: "Nf5 looks promising but I need tactical verification"
  │   Action: delegate_to("Tactician", "Verify Nf5 to depth 8")
  │   Observation: "Tactician: Nf5 is sound, eval +0.8 after Bxf5 exf5"
  │
  ├─ Thought: "What if opponent plays h6 instead of taking?"
  │   Action: delegate_to("Inner Critic", "Refute Nf5 h6 line")
  │   Observation: "Critic: After h6 Nh4, position is still +0.5, no refutation"
  │
  ├─ Thought: "Check similar positions for strategic themes"
  │   Action: ladybug_similarity(current_position_fingerprint, k=5)
  │   Observation: "4/5 similar positions won by side with kingside initiative"
  │
  └─ Final Answer: "Play Nf5 — supported by opening book frequency, tactical
     soundness, and historical similarity. Strategic plan: kingside attack."
```

### Self-Play Learning Loop

```
n8n-rs workflow: "SelfPlayTrainer"
  │
  ├─ Node: InitGame
  │   └─ Create two ChessThinkTank crews (White, Black)
  │
  ├─ Node: PlayLoop (stack-based, resumable)
  │   └─ Alternate White/Black crew.kickoff() per move
  │       └─ Each move: store AgentDecision in neo4j-rs graph
  │
  ├─ Node: GameOver
  │   └─ Annotate all positions with game result
  │   └─ Update SkillEngine proficiencies (crewai-rust)
  │   └─ Store game in neo4j-rs: (Game)-[:PLAYED]->(Position) chain
  │
  ├─ Node: LearnFromGame
  │   └─ Cypher: find positions where winning side's eval was wrong
  │   └─ ladybug: update position fingerprints with outcome data
  │   └─ crewai-rust: adjust agent skill proficiencies via EMA
  │
  └─ Node: TriggerNextGame (loop back to InitGame)
```

### Why This Stack (Not Traditional Engines)

| Traditional Engine | This Stack |
|--------------------|------------|
| Alpha-beta search | Agent deliberation + delegation |
| Evaluation function | LLM reasoning + position fingerprint similarity |
| Opening book (flat file) | Neo4j graph with frequency, eval, game history |
| Endgame tablebase | Specialized agent spawned on demand |
| No self-awareness | Inner Critic agent challenges every decision |
| Fixed depth | Dynamic: deeper on critical positions, shallow on forced |
| No learning | SkillEngine EMA + position graph grows every game |
| Single perspective | 5 agents with different strategic lenses |
| No explanation | Full reasoning chain stored as graph path |

The chess AI won't beat Stockfish on raw calculation — but it **thinks like a human grandmaster**: strategic plans, pattern recognition, opponent modeling, and self-critique. Every decision is explainable as a graph traversal through agent reasoning chains.
