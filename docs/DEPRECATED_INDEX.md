# Deprecated Documents Index

> These documents describe the **pre-SPOQ architecture** where neo4j-rs had its own
> StorageBackend trait, LadybugBackend, execution engine, and planner.
>
> The SPOQ Integration Plan v2 supersedes this architecture:
> neo4j-rs becomes a ~2,100 LOC Cypher compiler. All storage, traversal,
> and cognitive operations live in ladybug-rs at the Container level.
>
> **Current architecture**: See `SPOQ_INTEGRATION_PLAN_v2.md` and `SPOQ_AUDIT.md`

## Superseded Documents

| Document | Reason Superseded |
|----------|-------------------|
| `INTEGRATION_CONTRACT_v2.md` | StorageBackend trait (43 methods) deleted — neo4j-rs calls BindSpace directly |
| `WIRING_PLAN_NEO4J_LADYBUG.md` | LadybugBackend struct deleted — no wiring needed when crates share memory |
| `COMPATIBILITY_REPORT.md` | Evaluated old integration path via StorageBackend |
| `STRATEGY_INTEGRATION_PLAN.md` | Old strategy: neo4j-rs as database. New: neo4j-rs as Cypher compiler |
| `HANDOFF_LADYBUG_TO_NEO4J.md` | Handoff for old architecture's trait implementation |
| `HANDOVER.md` | Generic handover — subsumed by SPOQ Plan §9 execution sequence |

## Still Valid Documents

All documents not listed above remain current, especially:
- `34_TACTICS_INTEGRATION_PLAN.md` — cognitive primitive catalog
- `34_TACTICS_x_REASONING_LADDER.md` — tactics × reasoning ladder cross-reference
- `COGNITIVE_ARCHITECTURE.md`, `COGNITIVE_FABRIC.md` — cognitive stack docs
- `CLAM_HARDENING.md` — CAKES/CLAM search hardening
- All `COGNITIVE_RECORD_*.md` — container record formats
- `INTEGRATION_PROOF_PLAN_v2.md`, `LADYBUG_PROOF_ROADMAP_v1.md` — test plans (still valid)
