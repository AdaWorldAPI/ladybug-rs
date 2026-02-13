//! # LadybugDB
//!
//! Unified cognitive database: SQL + Cypher + Vector + Hamming + NARS + Counterfactuals.
//! Built on Lance columnar storage with AGI operations as first-class primitives.
//!
//! ## Quick Start
//! ```rust,ignore
//! use ladybug::{Database, Thought, NodeRecord, cypher_to_sql};
//!
//! // Open database
//! let db = Database::open("./mydb").await?;
//!
//! // SQL queries (via DataFusion)
//! let results = db.sql("SELECT * FROM nodes WHERE label = 'Thought'").await?;
//!
//! // Cypher queries (auto-transpiled to recursive CTEs)
//! let paths = db.cypher("MATCH (a)-[:CAUSES*1..5]->(b) RETURN b").await?;
//!
//! // Vector search (via LanceDB ANN)
//! let similar = db.vector_search(&embedding, 10).await?;
//!
//! // Resonance search (Hamming similarity on 10K-bit fingerprints)
//! let resonant = db.resonate(&fingerprint, 0.7, 10);
//!
//! // Grammar Triangle (universal input layer)
//! use ladybug::grammar::GrammarTriangle;
//! let triangle = GrammarTriangle::from_text("I want to understand this");
//! let fingerprint = triangle.to_fingerprint();
//!
//! // Butterfly detection (causal amplification chains)
//! let butterflies = db.detect_butterflies("change_id", 5.0, 10).await?;
//!
//! // Counterfactual reasoning
//! let forked = db.fork();
//! ```
//!
//! ## Architecture
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                        LADYBUGDB                                 │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │   Grammar  → NSM + Causality + Qualia → 10K Fingerprint         │
//! │   SQL      → DataFusion + Custom UDFs (hamming, similarity)     │
//! │   Cypher   → Parser + Transpiler → Recursive CTEs               │
//! │   Vector   → LanceDB native ANN indices                         │
//! │   Hamming  → AVX-512 SIMD (65M comparisons/sec)                 │
//! │   NARS     → Non-Axiomatic Reasoning System                     │
//! │   Storage: Lance columnar format, zero-copy Arrow               │
//! │   Indices: IVF-PQ (vector), scalar (labels), Hamming (custom)   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

// portable_simd requires nightly - use fallback popcount instead
// #![cfg_attr(feature = "simd", feature(portable_simd))]
#![allow(dead_code)] // During development
// Clippy: allow stylistic lints across the codebase
#![allow(
    clippy::collapsible_if,
    clippy::needless_range_loop,
    clippy::manual_range_contains,
    clippy::derivable_impls,
    clippy::manual_div_ceil,
    clippy::redundant_closure,
    clippy::manual_is_multiple_of,
    clippy::needless_borrow,
    clippy::needless_borrows_for_generic_args,
    clippy::large_enum_variant,
    clippy::unnecessary_map_or,
    clippy::manual_clamp,
    clippy::if_same_then_else,
    clippy::let_and_return,
    clippy::redundant_pattern_matching,
    clippy::manual_memcpy,
    clippy::unnecessary_cast,
    clippy::new_without_default,
    clippy::len_without_is_empty,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::single_match,
    clippy::needless_return,
    clippy::single_char_add_str,
    clippy::unnecessary_unwrap,
    clippy::manual_find,
    clippy::manual_strip,
    clippy::get_first,
    clippy::len_zero,
    clippy::unnecessary_sort_by,
    clippy::implicit_saturating_sub,
    clippy::cast_abs_to_unsigned,
    clippy::manual_abs_diff,
    clippy::cloned_ref_to_slice_refs,
    clippy::unwrap_or_default,
    clippy::doc_lazy_continuation
)]

// === Core modules ===
pub mod cognitive;
pub mod container;
pub mod core;
pub mod fabric;
pub mod grammar; // NEW: Grammar Triangle
pub mod graph;
pub mod learning;
pub mod nars;
pub mod query;
pub mod search;
pub mod storage;
pub mod width_16k;
pub mod world;

// === Optional extensions ===
#[cfg(any(
    feature = "codebook",
    feature = "hologram",
    feature = "spo",
    feature = "compress"
))]
pub mod extensions;

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "bench")]
pub mod bench;

#[cfg(feature = "flight")]
pub mod flight;

#[cfg(feature = "crewai")]
pub mod orchestration;

// === Re-exports for convenience ===

// Core types
pub use crate::core::{DIM, DIM_U64, Embedding, Fingerprint, VsaOps};

// Cognitive types
pub use crate::cognitive::{Belief, Concept, ThinkingStyle, Thought};

// NARS (Non-Axiomatic Reasoning)
pub use crate::nars::{Abduction, Deduction, Evidence, Induction, TruthValue};

// Grammar Triangle (universal input layer)
pub use crate::grammar::{CausalityFlow, GrammarTriangle, NSMField, QualiaField};

// Graph traversal
pub use crate::graph::{Edge, EdgeType, Traversal};

// Counterfactual worlds
pub use crate::world::{Change, Counterfactual, World};

// Query engine
pub use crate::query::{Query, QueryBuilder, QueryResult, SqlEngine, cypher_to_sql};

// Storage
#[cfg(feature = "lancedb")]
pub use crate::storage::{Database, EdgeRecord, LanceStore, NodeRecord};

// Orchestration (crewAI integration)
#[cfg(feature = "crewai")]
pub use crate::orchestration::{
    A2AChannel,
    A2AMessage,
    A2AProtocol,
    AgentAwareness,
    AgentBlackboard,
    AgentCapability,
    AgentCard,
    AgentGoal,
    AgentRegistry,
    AgentRole,
    BlackboardRegistry,
    CommunicationStyle,
    CrewBridge,
    CrewDispatch,
    CrewTask,
    DispatchResult,
    FeatureAd,
    FilterPhase,
    // Kernel extensions (cross-platform best practices)
    FilterPipeline,
    GuardrailAction,
    GuardrailResult,
    KernelFilter,
    KernelGuardrail,
    KernelMemory,
    KernelSession,
    KernelSpan,
    KernelTrace,
    MemoryBank,
    MemoryKind,
    MessageKind,
    ObservabilityManager,
    Persona,
    PersonaExchange,
    PersonaRegistry,
    PersonalityTrait,
    StyleOverride,
    TaskStatus,
    ThinkingTemplate,
    ThinkingTemplateRegistry,
    VerificationEngine,
    VerificationKind,
    VerificationRule,
    VolitionDTO,
    VolitionSummary,
    WorkflowNode,
    WorkflowOp,
    WorkflowStep,
    execute_workflow,
};

// === Error types ===

/// Crate-level error type
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Query error: {0}")]
    Query(String),

    #[error("Invalid fingerprint: expected {expected} bytes, got {got}")]
    InvalidFingerprint { expected: usize, got: usize },

    #[error("Node not found: {0}")]
    NodeNotFound(String),

    #[error("Invalid inference: {0}")]
    InvalidInference(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("DataFusion error: {0}")]
    DataFusion(#[from] datafusion::error::DataFusionError),
}

// StorageError conversion removed - use Error::Storage directly

impl From<query::QueryError> for Error {
    fn from(e: query::QueryError) -> Self {
        Error::Query(e.to_string())
    }
}

#[cfg(feature = "lancedb")]
impl From<lance::Error> for Error {
    fn from(e: lance::Error) -> Self {
        Error::Storage(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

// === Constants ===

/// Version info
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Fingerprint dimensions (16K = 2^14, exact u64 alignment, no partial word)
pub const FINGERPRINT_BITS: usize = 16_384;
pub const FINGERPRINT_U64: usize = 256; // 16384/64, exact
pub const FINGERPRINT_BYTES: usize = 2048; // 256×8
