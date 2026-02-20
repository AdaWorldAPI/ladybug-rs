# Deprecated: PR #127 + #128 — JSON Hydrate Endpoints

**Date**: 2026-02-17
**PRs**: #127 (POST /api/v1/hydrate), #128 (POST /api/v1/qualia/hydrate + write-back)

## Why deprecated

1. **JSON is forbidden on the internal hot path.** Internal operations use
   `&self` / `&mut self` borrows on shared substrate. Zero serialization.
   
2. **text_to_dn() is hash soup.** DN addressing must carry semantic meaning.
   The SPOQ model requires positions in the DN tree to represent viewpoints,
   not random hashes of message strings.

3. **Hollow pipeline.** PR #127 manually constructed FeltPath, ReflectionResult,
   and VolitionalAgenda with empty data and hardcoded defaults. PR #128 tried
   to fix this but used `match i % 8` for ghost type assignment (random noise).

4. **Wrong paradigm.** The hydrate endpoint assumed crewai-rust calls ladybug-rs
   via HTTP. The correct architecture: one binary, shared `&BindSpace`, blackboard
   borrow pattern (grey matter reads / white matter writes).

## What to salvage

- The `AgentState::compute()` integration pattern is correct (PR #126 has it)
- The route from message → qualia is right, but should be `&Container` → `&Container`, not JSON
- The INTEGRATION_SPEC Layer A concept (preamble for system prompt) is valid

## Files

- `server_hydrate_block.rs` — extracted 400-line code block from src/bin/server.rs
