# Ladybug-rs Integration Execution Plan

> **Date**: 2026-02-15
> **Branch**: `claude/ada-rs-consolidation-6nvNm`
> **Scope**: What ladybug-rs needs to build to complete the cognitive substrate

---

## Role: Ladybug-rs IS the Substrate

Ladybug-rs provides the CPU + memory for the cognitive stack. Everything runs on it.
crewai-rust (agency) and n8n-rs (orchestration) are consumers. ada-rs (consciousness)
is an optional layer on top.

```
Ladybug-rs provides:
├── BindSpace: Container storage + resonance search
├── CogRecord: Universal 2KB node format
├── GEL: Graph Execution Language (parser, compiler, executor) ← EXISTS
├── Fabric: FireflyFrame, UDP transport, LaneRouter           ← EXISTS
├── NARS: Non-Axiomatic Reasoning System                       ← EXISTS
├── MUL: Meta-Uncertainty Layer (10 layers)                    ← TO BUILD
├── Redis Lanes: Hot-path execution fabric                     ← TO BUILD
└── Arrow Flight: DoAction RPC transport                       ← PARTIALLY EXISTS
```

---

## Phase 1: MUL Module (Priority: HIGHEST)

**Every cognitive decision depends on MUL**. It must exist before crewai-rust or n8n-rs
can integrate smart gating.

### New Module: `src/mul/`

```
src/mul/
├── mod.rs              (~100 LOC) Module exports + MulSnapshot struct
├── trust_qualia.rs     (~150 LOC) L1: 4D trust assessment
├── dk_detector.rs      (~180 LOC) L2: Dunning-Kruger state machine
├── hysteresis.rs       (~120 LOC) L3: Temporal dwell timers
├── risk_vector.rs      (~100 LOC) L4: epistemic × moral risk
├── false_flow.rs       (~140 LOC) L5: False confidence detection
├── homeostasis.rs      (~160 LOC) L6: Friston free energy states
├── gate.rs             (~80 LOC)  L7: Binary gate (5 criteria)
├── free_will_mod.rs    (~100 LOC) L8: Multiplicative modifier
├── compass.rs          (~150 LOC) L9: 5 ethical tests
└── integrator.rs       (~200 LOC) L10: Runs all, produces MulSnapshot
```

**Total**: ~1,480 LOC, pure Rust, no new dependencies.

### Key Output Type

```rust
pub struct MulSnapshot {
    pub gate_open: bool,
    pub modifier: f32,           // 0.0-2.0 multiplicative
    pub dk_state: DKState,
    pub trust_composite: f32,    // geometric mean
    pub risk: (f32, f32),        // (epistemic, moral)
    pub flow_state: FlowState,
    pub compass_pass: bool,
    pub allostatic_load: f32,
    pub tick: u64,
}

impl MulSnapshot {
    pub fn pack(&self) -> [u64; 2];            // → CogRecord W64-W65
    pub fn unpack(words: [u64; 2]) -> Self;    // ← CogRecord W64-W65
}
```

### Wiring

1. Add `pub mod mul;` to `src/lib.rs`
2. Re-export `MulSnapshot` from lib root
3. Add `mul.evaluate` / `mul.snapshot` Flight DoAction handlers
4. Add Trap:0x20-0x2F in `fabric/executor.rs` for GEL MUL operations

### Tests

- Each layer has unit tests (dk_detector transitions, trust computation, etc.)
- Integration test: full L1-L10 pipeline → MulSnapshot → pack → unpack → verify
- Edge cases: MountStupid → gate closed, FalseFlow severe → modifier near 0

---

## Phase 2: Redis Lane Transport (Priority: HIGH)

### New File: `src/fabric/redis_lanes.rs` (~400 LOC)

Uses the existing `redis` feature flag and `FramePacket` encoding.

```rust
pub struct RedisLaneTransport {
    client: redis::Client,
    lanes: Vec<LaneConfig>,
}

impl RedisLaneTransport {
    pub fn send_to_lane(&self, frame: &FireflyFrame, lane_id: u8) -> Result<()>;
    pub fn run_lane_worker(&self, lane_id: u8, worker: &str, exec: &mut Executor) -> Result<()>;
    pub fn create_pipeline(&self, num_lanes: u8) -> Result<()>;
}
```

### Wiring

1. Add `pub mod redis_lanes;` to `src/fabric/mod.rs` (cfg-gated on `redis` feature)
2. Add re-exports: `RedisLaneTransport`, `LaneConfig`
3. No new Cargo.toml changes — `redis` feature already exists

### Tests

- Mock Redis with in-memory channel (test without actual Redis)
- Verify FramePacket → XADD → XREADGROUP → decode → execute → result
- 10-lane pipeline test: L1→L10 all execute in order

---

## Phase 3: GEL Executor Extensions (Priority: MEDIUM)

### FORK/JOIN (Control:0x10, Control:0x11)

Extend `src/fabric/executor.rs`:

```rust
// FORK: Create N frames in N lanes
0x10 => {
    let target_lanes = self.parse_fork_targets(inst);
    for lane_id in target_lanes {
        let mut fork_frame = current_frame.clone();
        fork_frame.header.lane_id = lane_id;
        self.outbox.push(fork_frame);
    }
    ExecResult::Ok(None)
}

// JOIN: Wait for N results
0x11 => {
    // Block until all expected results arrive
    // (in Redis: XREADGROUP on result streams)
    ExecResult::Ok(None)
}
```

### MUL Traps (Trap:0x20-0x2F)

```rust
0x20 => {  // MUL_EVALUATE
    let context = self.read_context_from_register(inst.src1);
    let snapshot = self.mul.evaluate(&context);
    self.write_snapshot_to_register(inst.dest, &snapshot);
    ExecResult::Ok(None)
}

0x21 => {  // MUL_GATE_CHECK
    let snapshot = self.read_snapshot_from_register(inst.src1);
    self.flags.zero = snapshot.gate_open;
    ExecResult::Ok(None)
}
```

### GEL Compiler Extensions

Add new mnemonics to `src/fabric/gel.rs`:
- `fork <lane1>, <lane2>, <lane3>` → Control:0x10
- `join <count>` → Control:0x11
- `mul_eval <dest>, <context>` → Trap:0x20
- `mul_gate <snapshot>` → Trap:0x21

---

## Phase 4: Arrow Flight API Surfaces (Priority: MEDIUM)

### New DoAction Handlers

Add to Flight server (under `flight` feature):

```
mul.evaluate        → Run full MUL L1-L10, return MulSnapshot as RecordBatch
mul.snapshot        → Return current MulSnapshot for given context
mul.gate_check      → Quick boolean: is the gate open?

gel.compile         → Compile GEL source → frames (returns serialized frames)
gel.execute         → Execute frames, return results
gel.pipeline        → Set up a 10-lane Redis pipeline

resonance.search    → Hamming similarity search (already exists, formalize)
resonance.cross     → Cross-domain resonance scan

spo.store           → Store SPO triple as CogRecord
spo.query_s         → Query by subject
spo.query_p         → Query by predicate
spo.query_o         → Query by object

style.recover       → XOR recovery: crystal ⊕ content = modulation
style.crystallize   → L10: content ⊕ modulation = crystal → store
```

---

## Execution Timeline

```
Week 1: Phase 1 (MUL) — all 10 layers + tests
Week 2: Phase 2 (Redis lanes) + Phase 3 (GEL extensions)
Week 3: Phase 4 (Flight APIs) + integration with n8n-rs/crewai-rust
```

---

## Verification Checklist

- [ ] `cargo check` — clean (no new warnings in MUL module)
- [ ] `cargo test mul::` — all MUL tests pass
- [ ] `cargo check --features redis` — Redis lanes compile
- [ ] `cargo test fabric::redis_lanes::` — lane routing tests pass
- [ ] `cargo check --features flight` — Flight APIs compile
- [ ] `grep -rn "ada.rs\|ada_rs" src/mul/` → zero hits (MUL is ada-agnostic)
- [ ] MulSnapshot packs to 128 bits (CogRecord W64-W65) and round-trips
- [ ] GEL FORK/JOIN produce correct fan-out/fan-in behavior
- [ ] MUL trap opcodes execute correctly in GEL programs
