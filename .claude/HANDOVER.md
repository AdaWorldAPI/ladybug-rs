# Session Handover — 2026-03-12

## Branch: `main`

## Active Session: Brain Surgery

**Orchestration prompt:** `.claude/prompts/18_brain_surgery_orchestration.md`
**Blackboard:** `.claude/SURGERY_BLACKBOARD.md`

## What Was Mapped (This Session — claude.ai, not Claude Code)

Complete archaeological mapping of ladybug-rs (164K LOC), lance-graph (19K LOC),
holograph (34K LOC). Found 5 Cypher paths where there should be 1.
Found spo.rs is the Rosetta Stone locked behind `mod spo` (private).
Found 3 TruthValue implementations, 5 text→fingerprint encoders, zero wired to Cypher.

### Documents Created (read in order)

| Prompt | File | What |
|--------|------|------|
| 15 | `15_RISC_brain_convergence_vision.md` | Glass-clear vision: 6 RISC instructions, live plasticity, Neo4j PET scan |
| 16 | `16_open_brain_surgery_handover.md` | 5 Cypher paths mapped, disconnection map, reconnection plan |
| 17 | `17_five_path_teardown.md` | File-by-file verdict: delete 7894 lines, write 1650, net -6244 |
| 17a | `17a_spo_rosetta_stone_addendum.md` | spo.rs is private, P4 opcodes map 1:1, TruthValue triplication |
| 18 | `18_brain_surgery_orchestration.md` | **A2A execution prompt — 5 agents, dependency graph, blackboard** |

### Also in ada-docs

- `FIREFLY-Integration-Plan.md` — Endgame architecture (Arrow Flight, temporal folding)
- `CONVERGENCE-Plan.md` — Intermediate plan (needs revision per new findings)
- `RISC-Brain-Convergence-Vision.md` — Copy of prompt 15
- `Open-Brain-Surgery-Handover.md` — Copy of prompt 16
- `Five-Path-Teardown.md` — Copy of prompt 17
- `SPO-Rosetta-Stone-Addendum.md` — Copy of prompt 17a
- `Brain-Surgery-Orchestration.md` — Copy of prompt 18

## How To Start the Surgery

```bash
# In Claude Code, read the orchestration prompt:
cat .claude/prompts/18_brain_surgery_orchestration.md

# It references these mandatory reads:
cat .claude/prompts/15_RISC_brain_convergence_vision.md
cat .claude/prompts/16_open_brain_surgery_handover.md
cat .claude/prompts/17_five_path_teardown.md
cat .claude/prompts/17a_spo_rosetta_stone_addendum.md

# Check blackboard state:
cat .claude/SURGERY_BLACKBOARD.md
```

## The Five Cypher Paths (Quick Reference)

```
P1  query/cypher.rs       1560 lines  TRANSPILE ONLY → DELETE
P2  cypher_bridge.rs       897 lines  EXECUTES against BindSpace → KEEP (only writer)
P3  query/lance_parser/   5532 lines  COPY of P5, orphaned → DELETE
P4  learning/cam_ops.rs    ~400 lines  OPCODE ENUM (80 opcodes) → KEEP + MOVE
P5  lance-graph (ext repo) ~15K lines  PARSER + PLANNER + ENGINE → IMPORT as Cargo dep
```

## spo.rs Status

- `src/spo/spo.rs` — 1568 lines, ALL PRIVATE (mod spo not pub mod spo)
- Contains: SPOCrystal, OrthogonalCodebook, QuorumField, CubicDistance,
  FieldCloseness, Triple, Qualia, TruthValue, Fingerprint
- Zero external references. The Rosetta Stone behind a locked door.
- `test_cypher_comparison` (line 1472) maps Cypher→SPO Crystal methods

## Key Decisions Pending

- [ ] SPO plane bit width: 3×4096? 3×16384? (impacts CogRecord size)
- [ ] lance-graph Cargo dep: git dep or workspace member?
- [ ] spo.rs unlock strategy: make pub or create crystal_api.rs facade?
- [ ] core::Fingerprint vs spo.rs Fingerprint: port project_out or bridge?

## CI Status

- ladybug-rs: Build & Release FAILING, CI Master FAILING, Docker SUCCEEDING
- rustynum: Rust CI FAILING, Python bindings FAILING
- Must fix before surgery begins (SURGEON task S4)

## Stale PRs

13 open PRs from Jan-Feb (#11-#33, #54, #168, #169). Close or merge as part of S3.
