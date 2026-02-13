//! Cognitive Fabric - mRNA cross-pollination and butterfly detection
//!
//! This is the unified substrate where all subsystems resonate.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                       COGNITIVE CPU ARCHITECTURE                            │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │  FIREFLY COMPILER (Python)                                                  │
//! │    → Parses user programs → Graph IR → 16384-bit packed frames               │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │  EXECUTOR (This module)                                                     │
//! │    → Dispatches frames to language runtimes                                 │
//! │    → Zero-copy: operates directly on BindSpace                              │
//! │    → No Redis needed: in-process execution                                  │
//! ├─────────────────────────────────────────────────────────────────────────────┤
//! │  LANGUAGE RUNTIMES                                                          │
//! │    Lance (0x0): Vector operations, similarity search                        │
//! │    SQL (0x1): Relational via DataFusion                                     │
//! │    Cypher (0x2): Graph traversal                                            │
//! │    NARS (0x3): Non-axiomatic reasoning                                      │
//! │    Causal (0x4): Pearl's SEE/DO/IMAGINE                                     │
//! │    Quantum (0x5): Superposition, interference                               │
//! │    Memory (0x6): BIND/UNBIND/BUNDLE/PERMUTE                                 │
//! │    Control (0x7): Branch, call, return                                      │
//! │    Trap (0xF): System calls, debug                                          │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```

pub mod mrna;
pub mod butterfly;
pub mod subsystem;
pub mod firefly_frame;
pub mod executor;
pub mod gel;
pub mod udp_transport;
pub mod zero_copy;
pub mod shadow;

pub use mrna::{MRNA, ResonanceField, CrossPollination, FieldSnapshot};
pub use butterfly::{ButterflyDetector, Butterfly, ButterflyPrediction};
pub use subsystem::Subsystem;
pub use firefly_frame::{
    FireflyFrame, FrameHeader, Instruction, LanguagePrefix,
    ExecutionContext, ConditionFlags, FrameBuilder,
};
pub use executor::{Executor, RegisterFile, ExecResult, ExecutorStats};
pub use gel::{GelParser, GelCompiler, GelProgram, compile as gel_compile, disassemble};
pub use udp_transport::{
    UdpSender, UdpReceiver, FramePacket, LaneRouter,
    SenderStats, ReceiverStats, MAX_UDP_PAYLOAD,
};
pub use zero_copy::{ZeroCopyExecutor, AddrRef, Deferred};
