# ladybug-rs — Progress Tracker

> Tracks progress against the master integration plan plateaus.
> See /home/user/INTEGRATION_PLAN.md for full context.

## Plateau 0: Everything Compiles

- [ ] ladybug-rs is EXCLUDED from Plateau 0 (needs sibling repos)
- [x] Minimal build works: `cargo check --no-default-features --features "simd"`
- [ ] Full build requires all siblings present

## Plateau 2: rustynum → ndarray + lance-graph Wiring

### Phase 2A: ndarray Replaces rustynum
- [ ] 2A.1: Add ndarray as path dep alongside rustynum
- [ ] 2A.2: Create src/compat/ndarray_bridge.rs
- [ ] 2A.3: Migrate src/spo/ to ndarray Fingerprint — CHECKPOINT
- [ ] 2A.4: Migrate src/nars/ to ndarray BF16Truth
- [ ] 2A.5: Migrate src/storage/ BindSpace — HIGH RISK CHECKPOINT
- [ ] 2A.6: Full test suite passes
- [ ] 2A.7: Remove rustynum path deps

### Phase 2B: lance-graph Wiring
- [ ] 2B.1: Delete P1 (src/query/cypher.rs, 1560 lines)
- [ ] 2B.2: Delete P3 (src/query/lance_parser/, 5532 lines)
- [ ] 2B.3: Wire lance-graph semirings into src/graph/spo/
- [ ] 2B.4: Delegate Cypher execution to lance-graph — CHECKPOINT
- [ ] 2B.5: Wire lance-graph SPO store as backend
- [ ] 2B.6: Connect server.rs to lance-graph query surface

### Phase 2C: Brain Surgery (25 tasks, 0/25 done)
- [ ] Surgeon S1-S5: Delete dead code, clean PRs, rename P4
- [ ] Locksmith L1-L5: Crystal API, codebook, TruthValue unification
- [ ] Bridge B1-B5: Wire SPO to server, merge SPO variants
- [ ] Bouncer N1-N5: Cargo deps, dedup, logical plan
- [ ] Seal K1-K5: UDF registration, query sealing, Neo4j

## Plateau 3: Full Stack

- [ ] 3C.1: Implement SubstrateView in ladybug-rs (crewai-rust connection)
- [ ] 3C.2: Wire JitProfile into n8n-rs ModuleRuntime

---
*Last updated: 2026-03-22*
