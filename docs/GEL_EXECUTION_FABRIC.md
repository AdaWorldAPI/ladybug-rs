# GEL Execution Fabric -- The Cognitive CPU in ladybug-rs

> **Date**: 2026-02-15
> **Status**: Discovery + Architecture Mapping
> **Source**: `ladybug-rs/src/fabric/` (4,485 lines, production-ready)
> **Core Insight**: The fabric module is a **complete Graph Execution Language**
>   with parser, compiler, executor, UDP transport, and lane routing. This
>   document maps the existing implementation to the Redis-as-CPU vision and
>   shows how it enables queuing ANY task -- A2A orchestration, workflow
>   execution, knowledge graph operations -- through a unified execution fabric.
>   Everything described here lives in ladybug-rs. No external consciousness
>   layer is required.

---

## 1. What Already Exists (Inventory)

### 1.1 Source File Inventory

```
ladybug-rs/src/fabric/
  gel.rs              880 lines   GEL parser + compiler + disassembler
  firefly_frame.rs    701 lines   16,384-bit frame format + builder
  executor.rs         761 lines   ALU: fetch-decode-dispatch-write-commit
  udp_transport.rs    580 lines   Zero-copy inter-node distribution
  zero_copy.rs        571 lines   Zero-copy frame operations
  mrna.rs             379 lines   mRNA transcription (frame transform)
  butterfly.rs        255 lines   Butterfly execution patterns
  shadow.rs           245 lines   Shadow execution (speculative)
  mod.rs               54 lines   Module exports
  subsystem.rs         59 lines   Subsystem trait definitions
  ──────────────────────────────
  TOTAL             4,485 lines
```

### 1.2 GEL Parser + Compiler (`fabric/gel.rs` -- 880 lines)

A human-readable assembly language for the cognitive CPU:

```gel
; Example: Bind two fingerprints and find resonance
.session 100
.lane 0
.hive 0

main:
    load r0, [0x8000]        ; Load fingerprint from node zone
    load r1, [0x8001]        ; Load another
    bind r2, r0, r1          ; XOR binding (A ^ B)
    resonate r3, r2, 10      ; Find 10 nearest to bound result
    halt
```

**Parser** (GelParser): Full tokenizer with source location tracking, directives
(`.session`, `.lane`, `.hive`, `.origin`), labels, register syntax (`r0-r255`),
address syntax (`[0x8000]`), immediate values (`#42`, `#0xFF`).

**Compiler** (GelCompiler): Converts GEL source to `Vec<FireflyFrame>`. Session,
lane, and hive tracking, label resolution, operand type handling.

**Disassembler**: `disassemble(frames)` produces GEL source (round-trip verified).

### 1.3 Nine Language Families (4-bit prefix x 8-bit opcode = 2,048 opcodes)

| Prefix | Family | Opcodes | Purpose |
|--------|--------|---------|---------|
| `0x0` | **Lance** | resonate, insert, hamming | Vector similarity, Hamming distance |
| `0x1` | **SQL** | select, filter, join | Relational queries (stub, DataFusion path) |
| `0x2` | **Cypher** | match, traverse, path | Graph pattern matching |
| `0x3` | **NARS** | deduce, induce, abduce, revise, negate | Inference engine |
| `0x4` | **Causal** | see, do, imagine | Pearl's 3 rungs of causation |
| `0x5` | **Quantum** | superpose, collapse, interfere | Superposition and measurement |
| `0x6` | **Memory** | bind, unbind, bundle, permute, load, store | BindSpace VSA ops |
| `0x7` | **Control** | nop, jump, branch, beq, bne, call, ret, cmp, halt | Flow control |
| `0xF` | **Trap** | syscall, debug, panic | System calls |

The 4-bit prefix + 8-bit opcode = **12-bit instruction encoding**. Nine
defined families with 256 opcodes each give 2,304 addressable opcodes.
Prefixes 0x8 through 0xE are reserved for domain-specific extensions
(6 families x 256 = 1,536 additional slots).

### 1.4 FireflyFrame Format (`fabric/firefly_frame.rs` -- 701 lines)

16,384-bit (256 u64 words / 2048 bytes) microinstruction frame:

```
Word 0:     Header (magic=0xADA1, version, session_id, lane_id, hive_id, sequence)
Word 1:     Instruction (prefix:4, opcode:8, flags:4, dest:16, src1:16, src2:16)
Words 2-3:  Operand (128-bit extended payload)
Words 4-9:  Data (384-bit fingerprint fragment / embedded payload)
Words 10-15: Context (qualia[8xi8], truth<f,c>, version, correlation_id)
Words 16-19: ECC (XOR-based parity, BCH slot reserved)
```

- **FrameBuilder**: Convenience API -- `.resonate()`, `.bind()`, `.nars_deduce()`,
  `.cypher_match()`, `.branch()`, `.trap()`, `.halt()`
- **ECC**: Simplified XOR parity (production path: BCH(1247,1024))
- **Encode/Decode**: Full round-trip with error detection

### 1.5 Executor (`fabric/executor.rs` -- 761 lines)

The ALU of the cognitive CPU:

```
Pipeline: FETCH -> DECODE -> DISPATCH -> WRITE -> COMMIT
```

**Register File = BindSpace** (the critical design choice):
- 65,536 addressable locations via 8+8 addressing
- Surface (0x00-0x0F): System registers
- Fluid (0x10-0x7F): Working memory
- Nodes (0x80-0xFF): Persistent storage
- Reading a register = reading from BindSpace
- Writing a register = writing to BindSpace
- Resonance search = native to the register file

**Implemented dispatchers**:
- LANCE: `resonate()` (via Substrate), `insert()`, `hamming()` (full popcount)
- NARS: `deduce()`, `induce()`, `abduce()`, `revise()`, `negate()` (real inference)
- QUANTUM: `superpose()` (OR), `collapse()` (identity), `interfere()` (XOR)
- MEMORY: `bind()` (XOR), `unbind()`, `permute()` (rotate), `load()`, `store()`
- CONTROL: `nop`, `jump`, `branch` (conditional), `call`, `ret`, `cmp` (Hamming compare)
- TRAP: `halt`, `panic`, `debug`

**Stub dispatchers** (architecture ready, logic not yet):
- SQL: Would integrate with DataFusion
- CYPHER: Would integrate with graph traversal
- CAUSAL: SEE/DO/IMAGINE (framework only)

**Statistics tracking**: Per-language instruction counts, cycle count, total time.

### 1.6 UDP Transport (`fabric/udp_transport.rs` -- 580 lines)

Zero-copy inter-node distribution:

- **FramePacket**: Transport header (magic + version + ack + sequence) + FireflyFrame
- **UdpSender**: Frame encoding, batch send, broadcast to multiple destinations
- **UdpReceiver**: Packet decode, executor loop, callback for results, timeout handling
- **LaneRouter**: Route frames to different executors based on `lane_id`
- **Async transport** (under `flight` feature): Tokio-based `AsyncSender`/`AsyncReceiver`

### 1.7 Additional Fabric Modules

**zero_copy.rs** (571 lines): Zero-copy frame operations for performance-critical
paths. Avoids allocation on the hot path by operating directly on byte slices.

**mrna.rs** (379 lines): mRNA-inspired frame transformation. Transcribes frames
between formats, enabling protocol evolution without breaking existing consumers.

**butterfly.rs** (255 lines): Butterfly execution patterns for parallel
fan-out/fan-in operations across multiple lanes.

**shadow.rs** (245 lines): Shadow (speculative) execution. Runs frames
speculatively on predicted paths; discards results if the prediction was wrong.

**subsystem.rs** (59 lines): Trait definitions for fabric subsystems, enabling
pluggable execution backends.

---

## 2. 512-Byte Node Record Context

The GEL executor operates on BindSpace nodes. The planned 512-byte node
record design (from `docs/GEL_STORAGE_ARCHITECTURE.md`) provides the
storage substrate that GEL instructions execute against.

### 2.1 Record Layout Summary

```
Offset  Size   Segment          Purpose
------  -----  ---------------  ---------------------------------
0x000   8 B    IDENTITY         Address + type + flags
0x008   8 B    NARS+ACCESS      Truth values (u8 fixed-point) + access
0x010   48 B   DN TREE+META     Tree structure + label + timestamps
0x040   64 B   ADJACENCY        Sparse pointers into Arrow edge columns
0x080   128 B  HDR SKETCH       1024-bit sketch for cascade filtering
0x100   128 B  META-THINKING    Reasoning state + GEL execution state
0x180   128 B  RESERVED         Sparse adjacency cache + checksums
------  -----
0x200   512 B  TOTAL
```

**GEL State within Segment 5 (META-THINKING, second half -- 64 bytes):**

```
Offset  Size  Field              Description
------  ----  -----------------  ----------------------------------
0x40    16 B  gel_bytecode       Up to 16 bytes of GEL instructions
0x50    8 B   program_counter    Current execution position
0x58    8 B   stack_top          Top of execution stack
0x60    16 B  registers          2 x u64 GEL registers (R0-R1)
0x70    8 B   accumulator        Running result (XOR, sum, product)
0x78    8 B   status+cont        Execution status + continuation addr
```

**16-opcode inline GEL** (fits in the 16-byte bytecode segment):

```
0x00 NOP, 0x01 LOAD, 0x02 WALK, 0x03 FILTER, 0x04 XOR, 0x05 POPCNT,
0x06 SKETCH, 0x07 HAMMING, 0x08 EMIT, 0x09 BRANCH, 0x0A GATHER,
0x0B REDUCE, 0x0C YIELD, 0x0D BIND, 0x0E STORE, 0x0F HALT
```

### 2.2 Tier Separation

```
+----------------------------------------------------------------------+
|                TIER 0: NODE RECORD (512 bytes)                        |
|    Always in memory. 8 cache lines. GEL executes directly here.      |
+----------------------------------------------------------------------+
|                TIER 1: HAMMING 4096 (512 bytes)                       |
|    Zero-copy Arrow buffer. mmap'd. For Hamming/XOR ops.              |
+----------------------------------------------------------------------+
|                TIER 2: EXTENDED (10K or 64K bits)                     |
|    Zero-copy Arrow buffer. On demand. Deep similarity only.          |
+----------------------------------------------------------------------+
```

GEL instructions that operate on the node record (LOAD, WALK, FILTER,
BRANCH, GATHER, REDUCE, STORE, HALT) never touch Tier 1 or Tier 2.
Only HAMMING and similarity search (SKETCH) may trigger Tier 1 loads.

---

## 3. Mapping: GEL Implementation to Redis-as-CPU-Lanes Vision

### 3.1 The Correspondence

```
COGNITIVE CPU VISION:                     LADYBUG-RS IMPLEMENTATION:
=========================                 ==========================

Redis stream = execution lane          -> LaneRouter (fabric/udp_transport.rs)
                                           lanes: Vec<Option<Executor>>
                                           route(packet) dispatches by lane_id

1.25KB UDP packet                      -> FireflyFrame (fabric/firefly_frame.rs)
                                           20 u64 words = 160 bytes core
                                           (compact -- fits in single UDP packet)

8+8 addressing                         -> Instruction.dest/src1/src2 (16-bit each)
                                           RegisterFile addresses: 0x00-0xFF prefix
                                           -> Surface/Fluid/Node zones

4096-opcode execution syntax           -> 9 LanguagePrefix x 256 opcodes = 2,304
                                           (4-bit prefix + 8-bit opcode = 12-bit)

Chaining pointer (next lane)           -> FrameHeader.lane_id + .sequence
                                           (explicit lane routing in header)

Worker picks up, computes, writes      -> Executor.execute(frame) -> ExecResult
                                           (pure instruction execution)

Fan-out to specialist lanes            -> FrameHeader.lane_id selects target
                                           LaneRouter.route() dispatches

Pipeline stages L1->L10                -> 10 lanes, each with its own Executor
                                           (one Executor per cognitive layer)
```

### 3.2 What Redis Adds That UDP Alone Does Not

The existing UDP transport gives point-to-point frame delivery. Redis streams
add persistence, ordering, fan-out, and observability:

```rust
/// Redis execution lane -- wraps a Redis stream as a CPU pipeline stage.
///
/// Each lane is a Redis stream. Messages are FireflyFrame-sized packets.
/// Consumer groups enable parallel workers per lane (SIMD for cognitive ops).
pub struct RedisLane {
    /// Stream name: "gel:lane:{lane_id}" e.g. "gel:lane:0" for L1 Recognition
    stream: String,
    /// Consumer group for parallel workers
    group: String,
    /// Lane ID (0-255, maps to LaneRouter index)
    lane_id: u8,
}

/// What Redis gives over raw UDP:
///
/// 1. ORDERING:    Redis streams are ordered -> pipeline stages execute in order
/// 2. PERSISTENCE: Optional XADD with MAXLEN -> execution traces for debugging
/// 3. CONSUMER GROUPS: Multiple workers per lane -> SIMD-style parallel exec
/// 4. BACKPRESSURE: XREADGROUP BLOCK -> natural flow control
/// 5. MONITORING:  XINFO/MONITOR -> see every packet flowing through every lane
/// 6. REPLAY:      XRANGE -> re-execute any portion of the pipeline
```

### 3.3 The 10-Lane Pipeline (Concrete)

```
Lane 0  (Input):         XADD gel:lane:0 * frame <encoded_firefly_frame>
  | Consumer group "l1-workers" (N workers)
Lane 1  (L1 Recognize):  LANCE.resonate -> codebook match
  | Result frame -> XADD gel:lane:2
Lane 2  (L2 Resonate):   LANCE.hamming -> similarity search
  |
Lane 3  (L3 Appraise):   CAUSAL.see -> entropy assessment
  |
Lane 4  (L4 Route):      CONTROL.branch -> select execution path
  | May skip lanes (branch target = gel:lane:5 or gel:lane:7)
Lane 5  (L5 Execute):    <domain-specific> -> produce output
  |
Lane 6  (L6 Delegate):   CONTROL.fork -> fan-out to specialist lanes
  | XADD to multiple lanes (gel:lane:6a, gel:lane:6b, gel:lane:6c)
Lane 7  (L7 Contingency): CAUSAL.imagine -> "what if otherwise?"
  | Fan-in: CONTROL.join -> wait for all specialists
Lane 8  (L8 Integrate):  MEMORY.bundle -> majority vote merge
  |
Lane 9  (L9 Validate):   NARS.revise -> evidence combination + DK check
  |
Lane 10 (L10 Crystal):   MEMORY.bind -> crystal_fp = content ^ modulation -> store
```

Each lane has its own `Executor` instance (via `LaneRouter`). Workers in a
consumer group share the lane's executor state via the underlying `BindSpace`
(which is the register file).

---

## 4. Queuing A2A Tasks: Any Orchestrator on GEL Lanes

### 4.1 The Key Insight

GEL frames are **agnostic** to what they carry. A frame is:

```
Header:      WHO is sending, WHICH lane, WHAT sequence
Instruction: WHICH language family, WHICH operation, WHERE to read/write
Context:     Truth value, qualia, correlation ID
Data:        384-bit payload fragment
```

This means an A2A task from any orchestrator is JUST a frame:

```gel
; Orchestrator: Delegate research task to specialist agent
.session 200          ; Agent session
.lane 6              ; L6 Delegation lane
.hive 1              ; Agent hive 1

; Package the task as a frame
load r0, [0x8100]    ; Load task description fingerprint
load r1, [0x8101]    ; Load agent profile fingerprint
resonate r2, r1, 5   ; Find 5 best-matching agents
store [0x0E03], r2   ; Write to blackboard slot 0x0E:03

; Route to specialist
; (fan-out: this creates frames in the specialist's input lane)
```

### 4.2 Frame Types for A2A Orchestration

```
TASK DELEGATION (crewai-rust, any orchestrator):
  Prefix: 0x7 (Control) + opcode 0x10 (FORK)
  dest:   Target agent's prefix:slot (e.g. 0x8103 = agent #3 in node zone)
  src1:   Task fingerprint address
  src2:   Urgency/priority (immediate value)
  context.correlation_id: Links to parent task chain

WORKFLOW STEP (n8n-rs, any workflow engine):
  Prefix: 0x7 (Control) + opcode 0x20 (YIELD)
  dest:   Workflow node's prefix:slot
  src1:   Input data address
  src2:   Step index
  context.truth: NARS evidence threshold for this step

MUL EVALUATION:
  Prefix: 0xF (Trap) + opcode 0x20 (AUTH -- repurposed as MUL gate)
  dest:   Decision output address
  src1:   Action fingerprint (what we want to do)
  src2:   Context fingerprint (current state)
  context.truth: Current trust level

AGENT SPAWN:
  Prefix: 0x7 (Control) + opcode 0x10 (FORK)
  data[0-5]: New agent's initial thinking_style values (packed)
  context.qualia: Emotional initialization
  context.correlation_id: Parent agent's ID
```

### 4.3 Concrete: Workflow as GEL Program

```gel
; Workflow: "Evaluate and route incoming request"
; This is what a workflow engine compiles each step into

.session 300
.lane 0              ; Input lane

evaluate:
    load r0, [0x8200]        ; Load incoming request fingerprint
    resonate r1, r0, 3       ; Find 3 closest known request patterns

    ; MUL gate check
    hamming r2, r0, r1       ; How confident are we in the match?
    cmp r2, #2048            ; Threshold: Hamming < 2048 = confident
    bne mul_uncertain        ; If not confident -> MUL evaluation

    ; Confident path -> route directly
    load r3, [0x8201]        ; Load route table
    bind r4, r0, r3          ; XOR bind request with route
    store [0x0E00], r4       ; Write to blackboard -> triggers next step
    halt

mul_uncertain:
    ; Not confident -> delegate to MUL for metacognitive evaluation
    ; (routes to MUL lane for DK detection, trust check, etc.)
    store [0x0300], r0       ; Write to MUL input slot
    ; (MUL consumer group picks this up on lane 3)
    halt
```

### 4.4 Concrete: Inner Council as GEL Fan-Out

```gel
; Inner council: Balanced, Catalyst, Guardian evaluate in parallel

.session 400
.lane 6              ; Delegation lane

council:
    load r0, [0x8300]        ; Load proposal fingerprint

    ; Fan out to three specialist lanes
    ; Each specialist has its own lane and executor
    store [0x10A0], r0       ; -> Balanced analyst (lane 0xA0)
    store [0x10A1], r0       ; -> Catalyst optimist (lane 0xA1)
    store [0x10A2], r0       ; -> Guardian protector (lane 0xA2)

    ; Wait for all three (fan-in on lane 8)
    ; Each specialist writes its vote to the integration lane
    halt

; --- On lane 0xA2 (Guardian) ---
; .lane 0xA2
; guardian:
;     load r0, [0x10A2]      ; Load proposal
;     load r1, [0x8400]      ; Load moral compass reference
;     hamming r2, r0, r1     ; Moral distance
;     cmp r2, #5734          ; 70% of 8192 content bits = moral risk threshold
;     bne guardian_veto       ; If distance > threshold -> VETO
;     ...
```

---

## 5. The Opcode Mapping: Vision to Implementation

### 5.1 Cognitive Vision Mapped to GEL

```
VISION OPCODES:                    GEL IMPLEMENTATION:
===============                    ===================

0x000-0x0FF: BindSpace ops     ->  0x6:00-0x6:FF (Memory family)
  read, write                      load (0x6:10), store (0x6:11)
  resonance                        resonate (0x0:00) [Lance family]
  bind, unbind                     bind (0x6:00), unbind (0x6:01)
  bundle                           bundle (0x6:02)
  permute                          permute (0x6:03)

0x100-0x1FF: Layer operations  ->  Encoded as LANE ROUTING
  L1-L10 triggers                  .lane directive selects which L
  gate                             CONTROL.branch + condition flags
  collapse                         QUANTUM.collapse (0x5:01)

0x200-0x2FF: Transport         ->  0x7:00-0x7:FF (Control family)
  route                            jump (0x7:01), branch (0x7:02)
  fan-out                          fork (0x7:10) [planned]
  fan-in                           join (0x7:11) [planned]
  delegate                         call (0x7:03)

0x300-0x3FF: MUL operations    ->  0xF:20 (Trap.AUTH repurposed)
  evaluate                         + custom trap service IDs
  tick, learn, compass             (to be defined as trap 0x30-0x3F)

0x400-0x4FF: SPO operations    ->  0x2:XX (Cypher family)
  store, query-S/P/O               match (0x2:00), traverse (0x2:01)
                                   + extended opcodes for SPO-specific

0x500-0x5FF: Strategy          ->  0x4:XX (Causal family)
  whatif                           imagine (0x4:02)
  branch                          see (0x4:00), do (0x4:01)
  prune, crystallize               + extended opcodes

0x600-0x6FF: Agent operations  ->  0x7:10-0x7:1F (Control, extended)
  spawn                            fork (0x7:10) + frame data payload
  delegate                         call (0x7:03)
  integrate                        ret (0x7:04) + frame data
  modify                           custom trap (0xF:60-0x6F)

0x700-0x7FF: Workflow ops      ->  0x7:20 (Control.YIELD)
  execute-step                     + lane routing metadata
  route, topology-change           Custom traps or extended control

0x800-0xFFF: Domain-specific   ->  Reserved language prefixes 0x8-0xE
  User-defined extensions          6 unused prefixes x 256 opcodes = 1,536 slots
```

### 5.2 Coverage Assessment

| Category | Vision Opcodes | Implemented | Gap |
|----------|---------------|-------------|-----|
| BindSpace ops | 256 | ~10 (core VSA) | Low priority -- 10 covers 99% |
| Layer operations | 256 | Via lane routing | **Design pattern, not opcodes** |
| Transport | 256 | ~8 (flow control) | FORK/JOIN need wiring |
| MUL operations | 256 | 0 | **High priority** -- needs trap extensions |
| SPO operations | 256 | ~3 (Cypher stubs) | Medium -- needs SPO semantics |
| Strategy | 256 | ~3 (Causal stubs) | Medium -- needs what-if tree |
| Agent operations | 256 | 0 | **High priority** -- needs spawn/delegate |
| Workflow ops | 256 | ~1 (YIELD) | Medium -- needs step/route |
| Domain-specific | 2048 | 0 | By design -- user extends |
| **Total** | **4096** | **~25** | Most gaps are extensions, not fundamentals |

**The fundamentals are solid**: Memory (VSA), Lance (vector), NARS (inference),
Control (flow), and the transport layer all work. The gaps are in higher-level
cognitive operations (MUL, SPO, Strategy, Agents) that can be built as:
1. New trap service IDs (simplest)
2. Extended opcodes in existing families (moderate)
3. New language prefixes using 0x8-0xE (most flexible)

---

## 6. Redis Lane Implementation Strategy

### 6.1 The Bridge: UdpSender/Receiver to Redis Streams

The existing `UdpSender` and `UdpReceiver` operate on `FramePacket` objects.
Redis lanes use the exact same encoding -- just a different transport:

```rust
/// Bridge: FramePacket -> Redis stream message
///
/// The FramePacket.encode() already produces bytes suitable for Redis XADD.
/// We just change the transport from UDP socket to Redis stream.
///
/// Key: "gel:lane:{lane_id}:{hive_id}"
/// Field: "frame" -> FramePacket.encode() bytes
/// Field: "seq" -> packet.sequence
///
/// Consumer group: "gel:workers:{lane_id}"
/// Consumer name: "worker-{worker_id}"

pub struct RedisLaneTransport {
    /// Redis connection
    client: redis::Client,
    /// Lane configuration
    lanes: Vec<LaneConfig>,
}

pub struct LaneConfig {
    lane_id: u8,
    hive_id: u8,
    stream_key: String,       // "gel:lane:0:0"
    consumer_group: String,   // "gel:workers:0"
    max_len: Option<u64>,     // MAXLEN for stream trimming
    block_ms: u64,            // XREADGROUP BLOCK timeout
}

impl RedisLaneTransport {
    /// Send frame to a specific lane (replaces UdpSender.send_frame)
    pub fn send_to_lane(&self, frame: &FireflyFrame, lane_id: u8) -> Result<()> {
        let packet = FramePacket::from_frame(frame);
        let bytes = packet.encode();
        let key = format!("gel:lane:{}:{}", lane_id, frame.header.hive_id);

        // XADD with optional trimming
        redis::cmd("XADD")
            .arg(&key)
            .arg("MAXLEN").arg("~").arg(10000)
            .arg("*")
            .arg("frame").arg(&bytes)
            .arg("seq").arg(frame.header.sequence)
            .query(&self.client)?;

        Ok(())
    }

    /// Receive and execute frames from a lane
    pub fn run_lane_worker(
        &self,
        lane_id: u8,
        worker_id: &str,
        executor: &mut Executor,
    ) -> Result<()> {
        let key = format!("gel:lane:{}:0", lane_id);
        let group = format!("gel:workers:{}", lane_id);

        loop {
            // XREADGROUP: blocking read from consumer group
            let results = redis::cmd("XREADGROUP")
                .arg("GROUP").arg(&group).arg(worker_id)
                .arg("BLOCK").arg(1000)  // 1s timeout
                .arg("COUNT").arg(10)    // batch up to 10
                .arg("STREAMS").arg(&key).arg(">")
                .query(&self.client)?;

            for entry in results {
                let bytes = entry.get("frame");
                let packet = FramePacket::decode(&bytes)?;
                let result = executor.execute(&packet.frame);

                // Chain to next lane if not halted
                if let Some(next_lane) = self.next_lane(lane_id, &result) {
                    let mut next_frame = packet.frame.clone();
                    next_frame.header.lane_id = next_lane;
                    self.send_to_lane(&next_frame, next_lane)?;
                }

                // ACK the message
                redis::cmd("XACK")
                    .arg(&key).arg(&group).arg(entry.id)
                    .query(&self.client)?;
            }
        }
    }
}
```

### 6.2 The Three-Transport Stack

```
+----------------------------------------------------------------------+
|                        TRANSPORT SELECTION                            |
+----------------------------------------------------------------------+
|                                                                      |
|  Redis Streams (hot path, ~1us per hop):                             |
|  +-- Intra-machine pipeline stages (L1->L2->...->L10)               |
|  +-- Fan-out/fan-in for inner council                                |
|  +-- A2A task queuing (any orchestrator)                             |
|  +-- MUL evaluation pipeline                                        |
|  +-- Observable: XINFO STREAM / MONITOR                              |
|                                                                      |
|  UDP (medium path, ~1ms):                                            |
|  +-- Inter-machine frame delivery (Pi fleet, cloud)                  |
|  +-- Cross-hive gossip (federated BindSpace updates)                 |
|  +-- Low-latency sensor input (IoT -> cognitive pipeline)            |
|  +-- Already implemented: UdpSender/UdpReceiver/LaneRouter           |
|                                                                      |
|  Arrow Flight / Ballista (cold path, ~10ms):                         |
|  +-- Distributed resonance search over millions of containers        |
|  +-- Bulk fingerprint comparison (batch similarity)                  |
|  +-- Cross-partition analytics (SQL over CogRecord batches)          |
|  +-- Persistent storage replication                                  |
|                                                                      |
|  All three share the SAME frame encoding:                            |
|    FireflyFrame -> FramePacket -> bytes -> Redis/UDP/Flight          |
|                                                                      |
+----------------------------------------------------------------------+
```

---

## 7. A2A Orchestration on GEL: Complete Example

### Scenario: An Orchestrator Receives a Code Review Task

```
Step 1:  Orchestrator receives task "Review pull request #42"
         -> Compiles task to FireflyFrame (Lance.resonate for PR fingerprint)
         -> XADD gel:lane:0 * frame <frame_bytes>

Step 2:  L1 Recognition worker picks up frame
         -> Executor.exec_lance(RESONATE) against codebook
         -> Matches: code_review pattern (sim=0.87)
         -> Result frame -> XADD gel:lane:1

Step 3:  L2 Resonance worker
         -> Executor.exec_lance(HAMMING) against previous reviews
         -> Finds 5 similar past reviews
         -> XADD gel:lane:3

Step 4:  L4 Route worker
         -> Executor.exec_control(BRANCH) based on complexity score
         -> Simple PR -> direct review (lane 5)
         -> Complex PR -> delegate to specialists (lane 6)
         (Let's say complex)
         -> XADD gel:lane:6

Step 5:  L6 Delegation worker (inner council)
         -> Fan-out to 3 specialist lanes:
         -> XADD gel:lane:A0 * frame <security_reviewer_frame>
         -> XADD gel:lane:A1 * frame <architecture_reviewer_frame>
         -> XADD gel:lane:A2 * frame <style_reviewer_frame>

Step 6:  Each specialist runs independently
         Security:     NARS.deduce(CVE patterns, PR code) -> truth(0.3, 0.8) = no CVE
         Architecture: CAUSAL.see(coupling, PR changes) -> moderate coupling increase
         Style:        LANCE.hamming(style guide, PR code) -> 3 style violations

Step 7:  L8 Integration
         -> Fan-in: all 3 results arrive at gel:lane:8
         -> MEMORY.bundle(security, architecture, style) -> merged review
         -> NARS.revise(merged, prior confidence) -> final truth

Step 8:  L9 Validation
         -> MUL gate: DK check passes (we have done code reviews before)
         -> Trust check: all sources credible
         -> Gate: OPEN

Step 9:  L10 Crystallization
         -> MEMORY.bind(review_content, PR_fingerprint) -> crystal
         -> Store crystal at [0x8500] in node zone
         -> This review becomes searchable knowledge for future reviews

Step 10: Output
         -> Orchestrator reads result from blackboard
         -> Formats as PR comment
         -> Posts via GitHub API
```

---

## 8. Integration with Cognitive Substrate Contracts

The GEL execution fabric is the runtime that executes the contracts defined
in `COGNITIVE_SUBSTRATE_CONTRACTS.md`. The relationship is:

| Contract | GEL Implementation |
|----------|-------------------|
| MUL L1-L10 | Trap extensions (0xF:20-0x2F), one lane per MUL layer |
| Resonance Engine | LANCE.resonate + MEMORY.bind for style recovery |
| Strategy Engine | CAUSAL family (see/do/imagine) + WhatIfTree via CONTROL |
| SPO Crystal | CYPHER family extensions (0x2:10-0x1F) for triple ops |
| CogRecord | FireflyFrame is the transport; CogRecord is the storage |

The distinction: **CogRecord is the data at rest. FireflyFrame is the data
in motion.** GEL instructions transform CogRecords by reading from and
writing to BindSpace addresses. The frame carries the instruction; the
BindSpace holds the state.

---

## 9. What Needs Building (Priority Order)

### 9.1 High Priority: Redis Lane Transport

**Files**: New `fabric/redis_lanes.rs` in ladybug-rs
**Depends on**: `redis` crate (lightweight)
**LOC estimate**: ~400 lines
**What**: `RedisLaneTransport` wrapping existing `FramePacket` encoding + `Executor`

### 9.2 High Priority: FORK/JOIN Opcodes

**Files**: Extend `fabric/executor.rs`
**Adds**: Control:0x10 (FORK) and Control:0x11 (JOIN) with multi-lane targeting
**LOC estimate**: ~150 lines
**What**: FORK creates N frames in N lanes, JOIN waits for N results

### 9.3 Medium Priority: MUL Trap Extensions

**Files**: Extend `fabric/executor.rs` Trap handler
**Adds**: Trap:0x20-0x2F for MUL operations (evaluate, dk_check, trust_score, etc.)
**LOC estimate**: ~300 lines (mostly calling into MUL module)
**What**: MUL evaluation as a trap service within the execution pipeline

### 9.4 Medium Priority: SPO Crystal via Cypher Family

**Files**: Extend `fabric/executor.rs` Cypher handler
**Adds**: Cypher:0x10-0x1F for SPO store/query (subject, predicate, object)
**LOC estimate**: ~200 lines
**What**: SPO crystal operations exposed as GEL instructions

### 9.5 Lower Priority: GEL Compiler Extensions

**Files**: Extend `fabric/gel.rs`
**Adds**: New mnemonics for FORK, JOIN, MUL traps, SPO ops
**LOC estimate**: ~100 lines per new family
**What**: Human-readable GEL syntax for the new opcodes

### 9.6 Lower Priority: Workflow-to-GEL Compiler

**Files**: New module (usable by any workflow engine)
**What**: Compile workflow definitions into GEL programs
**LOC estimate**: ~500 lines
**Enables**: Every workflow step becomes a GEL frame in a Redis lane

---

## 10. Architecture Summary

```
+----------------------------------------------------------------------+
|                     THE COMPLETE EXECUTION FABRIC                     |
+----------------------------------------------------------------------+
|                                                                      |
|  SOURCE LANGUAGES:                                                   |
|  +-- GEL assembly (.gel files)   -> GelParser -> GelCompiler -> frames
|  +-- Orchestrator tasks          -> TaskCompiler (planned) -> frames |
|  +-- Workflow steps              -> WorkflowCompiler (planned) -> f. |
|  +-- Domain-specific programs    -> DomainCompiler (extensible) -> f.|
|                                                                      |
|  TRANSPORT (all share FramePacket encoding):                         |
|  +-- Redis streams    -> intra-machine pipeline (~1us)               |
|  +-- UDP packets      -> inter-machine delivery (~1ms)               |
|  +-- Arrow Flight     -> distributed queries (~10ms)                 |
|                                                                      |
|  EXECUTION:                                                          |
|  +-- Executor (per lane) -> dispatches by language prefix            |
|  +-- RegisterFile = BindSpace -> 65,536 addressable containers       |
|  +-- LaneRouter -> routes frames to lane-specific executors          |
|  +-- Pipeline: FETCH -> DECODE -> DISPATCH -> WRITE -> COMMIT        |
|                                                                      |
|  LANGUAGE RUNTIMES (in the Executor):                                |
|  +-- Lance: Vector similarity (resonate, hamming, insert)    [done]  |
|  +-- NARS: Inference (deduce, induce, abduce, revise, negate)[done]  |
|  +-- Quantum: Superposition (superpose, collapse, interfere) [done]  |
|  +-- Memory: VSA (bind, unbind, bundle, permute, load, store)[done]  |
|  +-- Control: Flow (jump, branch, call, ret, cmp)            [done]  |
|  +-- Trap: System (halt, panic, debug)                       [done]  |
|  +-- SQL: Relational (select, filter, join)                  [stub]  |
|  +-- Cypher: Graph (match, traverse, path)                   [stub]  |
|  +-- Causal: Pearl's rungs (see, do, imagine)                [stub]  |
|                                                                      |
|  CONSUMERS (all interchangeable):                                    |
|  +-- crewai-rust: A2A task delegation, inner council fan-out/in      |
|  +-- n8n-rs: Workflow step execution, topology change                |
|  +-- External A2A: via UDP or Arrow Flight -> frames -> execution    |
|  +-- Custom runtimes: any system that produces FireflyFrames         |
|                                                                      |
|  The GEL execution fabric is INDEPENDENT of any consciousness layer. |
|  It lives in ladybug-rs. Any A2A orchestrator can submit frames.     |
|  The cognitive stack IS a real processor, not just a library.         |
|                                                                      |
+----------------------------------------------------------------------+
```

---

## 11. Relationship to Other ladybug-rs Documentation

| Document | Relationship |
|----------|-------------|
| `COGNITIVE_SUBSTRATE_CONTRACTS.md` | Defines the contracts that GEL executes |
| `GEL_STORAGE_ARCHITECTURE.md` | 512-byte node record that GEL operates on |
| `COGNITIVE_ARCHITECTURE.md` | 10-layer cognitive stack mapped to GEL lanes |
| `COGNITIVE_FABRIC.md` | High-level fabric overview |
| `STORAGE_CONTRACTS.md` | Race conditions and safety invariants |
| `api/FLIGHT_ENDPOINTS.md` | Flight endpoints that wrap GEL actions |
| `architecture/ADDRESS_MODEL.md` | 8+8 addressing used by GEL registers |
| `architecture/MEMORY_ZONES.md` | Memory zones that GEL addresses map to |

---

*The fabric module in ladybug-rs is a fully operational cognitive CPU. The GEL
parser, compiler, executor, and UDP transport are production-ready at 4,485
lines. The remaining work is: (1) Redis lane transport for intra-machine
pipelining, (2) FORK/JOIN for fan-out/fan-in, (3) MUL/SPO/Strategy as
extended opcodes, (4) workflow-to-GEL compiler for workflow engines. The
existing code provides the complete foundation. The 9 language families with
12-bit (4+8) instruction encoding execute on BindSpace-as-register-file. Any
orchestrator -- crewai-rust, n8n-rs, or external -- can submit FireflyFrames
and participate in the cognitive network without any dependency on a
consciousness layer.*
