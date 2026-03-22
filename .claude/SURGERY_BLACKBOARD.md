# Brain Surgery Blackboard — Session State

## Audit Status (2026-03-22)

- All 25 tasks across 5 roles (Surgeon, Locksmith, Bridge, Bouncer, Seal) remain PENDING — 0/25 completed
- server.rs (3717 lines) still uses CogRedis directly, not RedisAdapter
- P1 (src/query/cypher.rs) and P3 (src/query/lance_parser/) still exist — deletion tasks not started
- SPO module promoted to src/spo/ (18 files) — DONE
- SPO Merkle hardening — DONE
- 16K-bit container — DONE
- Graph SPO on BindSpace (src/graph/spo/) — PARTIAL (9 files, disconnected from server)
- Awareness loop — NOT STARTED (no awareness_loop files found)
- Path dependencies: 11 path deps, compilation requires all siblings present

session_id: "brain-surgery-2026-03"
started: "2026-03-12"
orchestration_prompt: ".claude/prompts/18_brain_surgery_orchestration.md"

surgeon:
  S1_delete_P1: PENDING
  S2_delete_P3: PENDING
  S3_stale_prs: PENDING
  S4_ci_green: PENDING
  S5_rename_p4: PENDING

locksmith:
  L1_project_out: PENDING
  L2_crystal_api: PENDING
  L3_codebook: PENDING
  L4_cleanup: PENDING
  L5_truthvalue: PENDING

bridge:
  B1_match_spo: PENDING
  B2_merge_spo: PENDING
  B3_edge_spo: PENDING
  B4_server_cypher: PENDING
  B5_crystal_state: PENDING

bouncer:
  N1_cargo_dep: PENDING
  N2_bouncer: PENDING
  N3_server_wire: PENDING
  N4_dedup_spo3: PENDING
  N5_logical_plan: PENDING

seal:
  K1_udf: PENDING
  K2_query_seal: PENDING
  K3_propagate: PENDING
  K4_neo4j: PENDING
  K5_register: PENDING

blocking_issues: []
decisions_made: []
notes: |
  Read .claude/prompts/18_brain_surgery_orchestration.md for full context.
  Read prompts 15, 16, 17, 17a BEFORE starting any work.

## Integration Plan Reference (2026-03-22)

The brain surgery tasks are now sequenced within the master integration plan.
They fall under **Plateau 2, Phase 2C** — only executed AFTER rustynum→ndarray
migration (2A) and lance-graph wiring (2B) are stable.

**Pre-conditions for surgery:**
- ndarray builds and tests pass (Plateau 0)
- ladybug-rs compiles with ndarray (Phase 2A)
- P1/P3 dead code deleted (Phase 2B.1-2B.2)
- lance-graph Cypher works end-to-end through server.rs (Phase 2B.4)

**Surgery task dependencies on integration phases:**
- S1/S2 (delete P1/P3): Unblocked → moved to Phase 2B.1/2B.2
- L1-L5 (crystal API, codebook, truth): Blocked on Phase 2A.4 (ndarray types)
- B1-B5 (wire SPO to server): Blocked on Phase 2B.5 (lance-graph SPO backend)
- N1-N5 (cargo deps, dedup): Blocked on Phase 2B.4 (lance-graph Cypher)
- K1-K5 (UDF, query seal): Blocked on Phase 2B.6 (server.rs rewire)

See: /home/user/INTEGRATION_PLAN.md
