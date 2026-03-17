# Brain Surgery Blackboard — Session State

session_id: "brain-surgery-2026-03"
started: "2026-03-12"
orchestration_prompt: ".claude/prompts/18_brain_surgery_orchestration.md"

surgeon:
  S1_delete_P1: DONE       # Deleted src/query/cypher.rs (1560 lines)
  S2_delete_P3: SKIPPED    # P3 (lance_parser) is the PRODUCTION parser — kept
  S3_stale_prs: PENDING
  S4_ci_green: PENDING
  S5_rename_p4: DONE       # CypherOp → CypherInstruction in cam_ops.rs

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
  B4_server_cypher: DONE    # /cypher now: parse_cypher_query → execute_cypher → BindSpace
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
decisions_made:
  - "P3 (lance_parser) is the production parser. S2_delete_P3 skipped — it was misnamed."
  - "cypher_bridge.rs rewritten: CypherOp/NodeRef/WhereClause/CypherValue removed. Takes P3 AST directly."
  - "CypherResult.rows changed from HashMap<String, CypherValue> to HashMap<String, serde_json::Value> (no more CypherValue)."
  - "server.rs /cypher now EXECUTES against BindSpace (was: transpile-only stub)."
notes: |
  Read .claude/prompts/18_brain_surgery_orchestration.md for full context.
  Read prompts 15, 16, 17, 17a BEFORE starting any work.
