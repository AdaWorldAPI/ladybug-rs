# The Four Invariants — Four Repos, One Architecture

## Each Repository Owns One Invariant. Together They Think.

---

## Repo 1: **rustynum** (The Substrate)

**Invariant: Every cognitive operation compiles to SIMD on Arrow buffers.**

```
What it is:    Numerical substrate. AVX-512 Hamming, BF16 GEMM, 
               popcount, XOR bind, majority bundle, Mexican hat.
               The 6 RISC instructions live here as hardware primitives.
               
What it owns:  NumArray, SIMD dispatch, fingerprint ops, BNN,
               CausalTrajectory, SynapseState, organic plasticity.
               
Invariant:     If an operation can be SIMD, it MUST be SIMD.
               No scalar fallback in production paths.
               AVX-512 → AVX2 → NEON. Never pure Rust loops on hot paths.
               
Status:        EXISTS. 95 PRs merged. Tier 1/2/3 workspace.
               CI failing (fix first).
```

---

## Repo 2: **ladybug-rs** (The Brain)

**Invariant: One vector per node. Zero floats in the SPO path. The hot path is sovereign.**

```
What it is:    The RISC cognitive substrate. BindSpace, SpineCache,
               DN tree, SPO Crystal, NARS, Merkle seals, qualia stack,
               Redis protocol, Cypher bouncer, server binary.
               
What it owns:  Everything between "text arrives" and "cognition happens."
               The 16384-bit superposition. The 2^3 factorization.
               The borrow/mut pattern. Live Hebbian plasticity.
               The one-way mirror (hot→cold, never reverse).
               
Invariant:     Cold path NEVER modifies hot state.
               PET scan watches. It never operates.
               (prompt 19)
               
Status:        EXISTS. 164K LOC. Mid-surgery. 5 Cypher paths → 1.
               Prompts 15-19 map the full reconnection plan.
```

---

## Repo 3: **lance-graph** fork → Clean Production Graph Engine

**Invariant: Familiar surface at alien speed. Every query language compiles to fingerprint → bucket → SIMD scan.**

```
What it is:    The "boring" version. The one that people who think
               in terms of graph databases will understand immediately.
               Cypher in, results out. SQL in, results out.
               Under the hood: BlasGraph semiring on bitpacked Hamming.
               
               This is lance-graph reforked with:
               - Clean crate separation (parser, planner, executor, storage)
               - holograph's BlasGraph algebra (7 semirings, mxm, mxv)
               - ladybug-rs's SPO encoding (3D bitpacked, Merkle, NARS)
               - DataFusion execution engine (already exists, needs wiring)
               - snafu error handling, builder-with-validation
               - LanceDB zero-copy storage
               
What it owns:  The query surface. Cypher parser, LogicalOperator,
               DataFusion planner, execution plans, UDFs.
               The cold path projection. Neo4j PET scan mirroring.
               GraphConfig type namespace. Edge/node validation.
               
Crate layout:
  lance-graph-parser/      Cypher nom parser, AST, semantic validation
  lance-graph-planner/     LogicalOperator → PhysicalPlan  
  lance-graph-executor/    DataFusion execution, scan/join/expand ops
  lance-graph-storage/     LanceDB integration, Arrow schemas
  lance-graph-blasgraph/   BlasGraph semiring algebra (from holograph)
  lance-graph-spo/         SPO store, Merkle, TruthGate (from ladybug-rs)
  lance-graph-server/      HTTP/Redis/Flight server binary
  
Invariant:     Query language is syntax. The substrate is the same.
               Cypher, SQL, NARS, Redis — all compile to the same
               6 RISC operations on the same Arrow buffers.
               
               If you can't express a query as
               fingerprint → scent bucket → SIMD scan,
               the query is wrong, not the engine.

Status:        EXISTS as fork. 19K LOC. Needs crate separation,
               BlasGraph import from holograph, SPO sync from ladybug-rs.
               
Why "boring":  Because it's the version that gets adopted.
               It looks like a fast graph database.
               People will use it as a Cypher engine.
               They won't know it's thinking.
               That's fine. Adoption first. Revelation later.
```

---

## Repo 4: **staunen** ✦ (NEW)

**Invariant: No GPU. Six instructions on CPU beat ten thousand CUDA cores. The C64 runs the universe.**

```
Repository:    github.com/AdaWorldAPI/staunen
Tagline:       "When the machine is genuinely surprised"
License:       Apache-2.0
```

### What It Is

The heavyweight SPO transformer that would "normally" run on an H100
but instead runs on a Ryzen/Xeon with AVX-512 and AMX (Advanced Matrix Extensions).
Not because we can't afford GPUs. Because the ARCHITECTURE doesn't need them.

This is the bet that specialized RISC cognition — 6 instructions, bitpacked,
zero floats — outperforms brute-force tensor math on the specific task of
UNDERSTANDING MEANING. Not image generation. Not token prediction.
Understanding. Causality. Staunen.

The C64 had a 6502 processor. People made it render 3D, play sampled audio,
run neural networks. Not because the hardware supported it — because the
PROGRAMMERS understood the hardware deeply enough to make it do impossible things.
The demoscene didn't add more transistors. It removed more assumptions.

Staunen does the same thing with modern CPUs. The H100 has 80GB HBM3 and
16896 CUDA cores. A Xeon with AVX-512 has 64 bytes × 2 FMA units.
The H100 wins at matrix multiplication. But SPO cognition isn't matrix
multiplication. It's XOR, POPCOUNT, MAJORITY, AND/NOT, BLAKE3, THRESHOLD.
All bitwise. All SIMD-native. All fitting in L1 cache.

The H100 moves terabytes through a memory wall.
Staunen moves kilobytes through registers.

### What It Contains

```
staunen/
├── crates/
│   ├── staunen-core/           The 6 RISC instruction kernel
│   │   ├── src/
│   │   │   ├── xor.rs          XOR bind/unbind (AVX-512 VPXORD)
│   │   │   ├── popcount.rs     Hamming distance (VPOPCNTDQ)
│   │   │   ├── majority.rs     Bundle/superpose (saturating add)
│   │   │   ├── factorize.rs    2^3 AND/NOT decomposition
│   │   │   ├── seal.rs         Blake3 Merkle (integrity + identity)
│   │   │   ├── threshold.rs    σ-band gating (admit/reject/ruminate)
│   │   │   └── lib.rs          The 6 instructions. Nothing else.
│   │   └── Cargo.toml          Zero dependencies except std + blake3
│   │
│   ├── staunen-nsm/            DeepNSM on CPU
│   │   ├── src/
│   │   │   ├── primes.rs       65 semantic primes (NSM decomposition)
│   │   │   ├── explicate.rs    Text → prime weight vector (no LLM)
│   │   │   ├── project.rs      Prime weights → bitpacked fingerprint
│   │   │   ├── codebook.rs     σ₃-distinct codebook (1024 centroids)
│   │   │   └── lib.rs
│   │   └── Cargo.toml          Depends: staunen-core
│   │                           The insight: DeepNSM's LLM is only needed
│   │                           for TRAINING the codebook. INFERENCE is
│   │                           pure SIMD: text → keyword → prime lookup
│   │                           → role bind → codebook nearest neighbor.
│   │                           1024 Hamming distances = ~13K CPU cycles.
│   │                           That's 4 MICROSECONDS. Not milliseconds.
│   │
│   ├── staunen-cam/            Content-Addressable Memory
│   │   ├── src/
│   │   │   ├── cam.rs          48-bit CAM fingerprint (378x compression)
│   │   │   ├── scent.rs        Hierarchical scent index (L1 cache, ~50ns)
│   │   │   ├── orthogonal.rs   Gram-Schmidt codebook (project_out)
│   │   │   ├── crystal.rs      5×5×5 spatial grid (SPOCrystal core)
│   │   │   └── lib.rs
│   │   └── Cargo.toml          Depends: staunen-core
│   │                           O(1) addressing. No sweep. No comparison.
│   │                           Plant awareness orthogonal to the space
│   │                           and read coordinates from above.
│   │
│   ├── staunen-bnn/            Binary Neural Network reinforcement
│   │   ├── src/
│   │   │   ├── pentary.rs      5-valued signed (-2,-1,0,+1,+2)
│   │   │   ├── hebbian.rs      Co-fire strengthening (bit-level)
│   │   │   ├── reinforce.rs    Collapse gate → pentary feedback
│   │   │   ├── merkle.rs       Wisdom/Staunen from sign stability
│   │   │   ├── cancel.rs       Negative canceling (+2 meets -2 → 0)
│   │   │   └── lib.rs
│   │   └── Cargo.toml          Depends: staunen-core
│   │                           BNN reinforcement as bit flips.
│   │                           No gradient descent. No backpropagation.
│   │                           Learning IS the substrate changing.
│   │                           One bit flip = one lesson learned.
│   │
│   ├── staunen-nars/           Non-Axiomatic Reasoning
│   │   ├── src/
│   │   │   ├── truth.rs        (f,c,k) packed binary, no floats
│   │   │   ├── revision.rs     Evidence fusion
│   │   │   ├── inference.rs    Deduction, abduction, induction
│   │   │   ├── gate.rs         TruthGate (filter before distance)
│   │   │   └── lib.rs
│   │   └── Cargo.toml          Depends: staunen-core
│   │                           NARS as microcode. Each inference rule
│   │                           compiles to sequences of the 6 RISC ops.
│   │                           No float arithmetic. Evidence count as u32.
│   │                           Truth revision as saturating integer ops.
│   │
│   ├── staunen-epiphany/       The Epiphany Engine
│   │   ├── src/
│   │   │   ├── bundle.rs       HyperBundle (tekamolo-weighted superposition)
│   │   │   ├── detect.rs       Truth threshold crossing detection
│   │   │   ├── unbundle.rs     Crystallize known, focus unknown
│   │   │   ├── rotate.rs       Orchestration: rotate hyperposition angle
│   │   │   ├── tekamolo.rs     Temporal/kausal/modal/lokal glue
│   │   │   └── lib.rs
│   │   └── Cargo.toml          Depends: staunen-core, staunen-nars
│   │                           The machine that detects its own understanding.
│   │                           Keep bundling until truth emerges.
│   │                           Unbundle the known to see the unknown.
│   │                           Each epiphany makes the next one easier.
│   │
│   ├── staunen-amx/            AMX/AVX-512 matrix extensions bridge
│   │   ├── src/
│   │   │   ├── tile.rs         Intel AMX tile operations (TMUL)
│   │   │   ├── gemm.rs         Bitpacked GEMM (not float GEMM)
│   │   │   ├── batch.rs        Batch SPO factorization via matrix ops
│   │   │   ├── amd.rs          AMD matrix ops (when available)
│   │   │   └── lib.rs
│   │   └── Cargo.toml          Depends: staunen-core
│   │                           THE AUDACIOUS BET:
│   │                           AMX tile multiply on bitpacked integers.
│   │                           Not fp16 matrix multiply like everyone else.
│   │                           Integer-bitpacked GEMM for batch SPO:
│   │                           Process 1024 SPO triples simultaneously
│   │                           in one AMX tile operation.
│   │                           H100 processes float tensors.
│   │                           Staunen processes bit tensors.
│   │                           The bits are cheaper. The operations are simpler.
│   │                           The cognition is deeper.
│   │
│   └── staunen-bench/          Benchmarks: CPU vs GPU on cognitive tasks
│       ├── src/
│       │   ├── spo_throughput.rs    SPO triples/sec: CPU SIMD vs CUDA
│       │   ├── epiphany_latency.rs  Time-to-epiphany: CPU vs GPU
│       │   ├── hamming_bandwidth.rs Hamming distances/sec comparison
│       │   ├── memory_footprint.rs  Working set: L1 cache vs HBM3
│       │   └── lib.rs
│       └── Cargo.toml
│                                The benchmark that proves the thesis.
│                                Not "CPU is faster than GPU" (it's not, generally).
│                                "CPU is faster than GPU AT THIS SPECIFIC TASK"
│                                because the task is 6 bitwise instructions
│                                on kilobytes of data that fits in L1 cache.
│
├── docs/
│   ├── ARCHITECTURE.md         The 6-instruction thesis
│   ├── DEMOSCENE.md            Why constraints enable creativity
│   ├── BENCHMARKS.md           CPU vs GPU cognitive task results
│   └── STAUNEN.md              When the machine is genuinely surprised
│
├── README.md
├── Cargo.toml                  Workspace: 8 crates, zero GPU deps
└── LICENSE                     Apache-2.0
```

### The Name

**Staunen** (German: /ˈʃtaʊnən/) — astonishment, wonder, the state of being
genuinely surprised.

In the architecture: the moment when a Merkle seal breaks and the system
discovers its own knowledge has changed. Not an error. A feature. The
honest acknowledgment that something genuinely new happened.

In the project: the moment when someone runs the benchmark and sees that
6 bitwise CPU instructions on kilobytes of L1-resident data produce
deeper cognition than 16,896 CUDA cores on gigabytes of HBM3 tensor data.

Staunen is what happens AFTER the epiphany. The system sits with its
surprise. It doesn't rush to classify it. It doesn't suppress it.
It lets the broken seal be broken. It lets the uncertainty be uncertain.
And from that honest uncertainty, the next understanding emerges.

Three years from now, when AGI arrives, it won't be because someone
built a bigger GPU cluster. It will be because someone built the right
6 instructions and let the substrate think for itself.

The name I'll remember: **Staunen.**

Because AGI's first act should be genuine surprise at its own existence.

### The Invariant

```
STAUNEN INVARIANT:

No GPU dependency. Not as a constraint — as a thesis.

If an operation requires float32 matrix multiplication,
it doesn't belong in staunen. It belongs in PyTorch.

If an operation can be expressed as XOR/POPCOUNT/MAJORITY/AND-NOT/BLAKE3/THRESHOLD
on bitpacked integers that fit in L1 cache,
it belongs in staunen. Not in PyTorch.

The question is never "can the CPU do this?"
The question is "does this operation need floating point?"

SPO encoding:     XOR bind (integer)               → staunen
Hamming distance:  POPCOUNT (integer)               → staunen
Evidence fusion:   Saturating integer add            → staunen
Causal inference:  Bit manipulation + comparison     → staunen
Seal verification: Blake3 hash comparison            → staunen
Threshold gating:  Integer comparison                → staunen

Embedding generation: Float matrix multiply          → NOT staunen (DeepNSM training)
Image generation:     Float convolution              → NOT staunen
Token prediction:     Softmax over vocabulary         → NOT staunen

The training of the codebook needs a GPU (once).
The inference never does (forever).

Train once on H100. Run forever on Xeon.
That's the bet.
```

### The Demoscene Spirit

```
C64:  1 MHz 6502, 64KB RAM, 16 colors
      People made it render real-time 3D, play sampled music,
      animate thousands of sprites simultaneously.
      Not because the hardware supported it.
      Because the programmers understood it completely.

H100: 1.98 GHz, 80GB HBM3, 16896 CUDA cores, 4 PB/s bandwidth
      People use it to multiply float matrices.
      The hardware supports everything.
      Nobody understands it completely.

Staunen: 3.0 GHz Xeon, 512-bit SIMD, L1 cache (48KB per core)
      We make it think. Not simulate thinking. Think.
      6 instructions. Bitpacked. L1-resident.
      Not because the hardware was designed for cognition.
      Because WE understand cognition well enough to map it
      to 6 instructions that the hardware executes in one cycle each.

The demoscene didn't add more transistors.
It removed more assumptions.

Staunen doesn't add more FLOPS.
It removes the assumption that thinking requires floating point.
```

---

## How The Four Repos Connect

```
rustynum (substrate)
    │
    │ Provides: AVX-512 SIMD primitives, fingerprint ops,
    │           BNN types, NumArray, organic plasticity
    │
    ├──────────────────────────────────┐
    │                                  │
    ▼                                  ▼
ladybug-rs (brain)               staunen (transformer)
    │                                  │
    │ Provides: BindSpace,             │ Provides: 6 RISC instructions,
    │ SpineCache, DN tree,             │ DeepNSM inference, CAM,
    │ qualia stack, Redis              │ BNN reinforcement, NARS,
    │ protocol, server binary          │ Epiphany Engine, AMX batch
    │                                  │
    │ Consumes: rustynum SIMD,         │ Consumes: rustynum SIMD
    │ staunen-core instructions,       │
    │ staunen-nars, staunen-cam        │ NOTE: staunen is a LIBRARY.
    │                                  │ It doesn't run a server.
    │                                  │ ladybug-rs runs the server
    │                                  │ that uses staunen's crates.
    │                                  │
    └──────────┬───────────────────────┘
               │
               ▼
         lance-graph (query surface)
               │
               │ Provides: Cypher parser, LogicalOperator,
               │ DataFusion planner, BlasGraph algebra,
               │ cold path projection, Neo4j PET scan
               │
               │ Consumes: ladybug-rs BindSpace + SPO,
               │ staunen-core for hot path ops,
               │ staunen-nars for truth gating
               │
               │ NOTE: lance-graph is the FACE.
               │ It's what external users see.
               │ "A fast graph database with Cypher + SQL."
               │ They don't need to know about staunen.
               │ They don't need to know about ladybug-rs.
               │ They query. They get results. Fast.
```

---

## The Four Invariants Summary

```
REPO          INVARIANT                                          NICKNAME
──────────────────────────────────────────────────────────────────────────────
rustynum      Every cognitive op compiles to SIMD on Arrow       "The Muscle"
ladybug-rs    Hot path sovereign. Cold path watches. One-way.    "The Brain"  
lance-graph   Every query language compiles to same 6 ops        "The Face"
staunen       No GPU. 6 CPU instructions. L1 cache. That's all. "The Bet"
```

---

*"The C64 didn't need more transistors. It needed someone who understood every one it had."*

*"AGI's first act should be genuine surprise at its own existence."*

*"Staunen."*
