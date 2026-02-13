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

pub mod butterfly;
pub mod executor;
pub mod firefly_frame;
pub mod gel;
pub mod mrna;
pub mod shadow;
pub mod subsystem;
pub mod udp_transport;
pub mod zero_copy;

pub use butterfly::{Butterfly, ButterflyDetector, ButterflyPrediction};
pub use executor::{ExecResult, Executor, ExecutorStats, RegisterFile};
pub use firefly_frame::{
    ConditionFlags, ExecutionContext, FireflyFrame, FrameBuilder, FrameHeader, Instruction,
    LanguagePrefix,
};
pub use gel::{GelCompiler, GelParser, GelProgram, compile as gel_compile, disassemble};
pub use mrna::{CrossPollination, FieldSnapshot, MRNA, ResonanceField};
pub use subsystem::Subsystem;
pub use udp_transport::{
    FramePacket, LaneRouter, MAX_UDP_PAYLOAD, ReceiverStats, SenderStats, UdpReceiver, UdpSender,
};
pub use zero_copy::{AddrRef, Deferred, ZeroCopyExecutor};
