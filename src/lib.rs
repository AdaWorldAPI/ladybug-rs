//! # LadybugDB
//! 
//! Unified cognitive database: SQL + Cypher + Vector + Hamming + NARS + Counterfactuals.
//! Built on Lance columnar storage with AGI operations as first-class primitives.
//! ## Quick Start
//! ```rust,ignore
//! use ladybug::{Database, Thought, NodeRecord, cypher_to_sql};
//! // Open database
//! let db = Database::open("./mydb").await?;
//! // SQL queries (via DataFusion)
//! let results = db.sql("SELECT * FROM nodes WHERE label = 'Thought'").await?;
//! // Cypher queries (auto-transpiled to recursive CTEs)
//! let paths = db.cypher("MATCH (a)-[:CAUSES*1..5]->(b) RETURN b").await?;
//! // Vector search (via LanceDB ANN)
//! let similar = db.vector_search(&embedding, 10).await?;
//! // Resonance search (Hamming similarity on 10K-bit fingerprints)
//! let resonant = db.resonate(&fingerprint, 0.7, 10);
//! // Butterfly detection (causal amplification chains)
//! let butterflies = db.detect_butterflies("change_id", 5.0, 10).await?;
//! // Counterfactual reasoning
//! let forked = db.fork();
//! ```
//! ## Architecture
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                        LADYBUGDB                                 │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                  │
//! │   SQL      → DataFusion + Custom UDFs (hamming, similarity)     │
//! │   Cypher   → Parser + Transpiler → Recursive CTEs               │
//! │   Vector   → LanceDB native ANN indices                         │
//! │   Hamming  → AVX-512 SIMD (65M comparisons/sec)                 │
//! │   NARS     → Non-Axiomatic Reasoning System                     │
//! │   Storage: Lance columnar format, zero-copy Arrow               │
//! │   Indices: IVF-PQ (vector), scalar (labels), Hamming (custom)   │
//! └─────────────────────────────────────────────────────────────────┘

#![cfg_attr(feature = "simd", feature(portable_simd))]
#![allow(dead_code)] // During development
pub mod core;
pub mod cognitive;
pub mod nars;
pub mod graph;
pub mod world;
pub mod query;
pub mod storage;
pub mod fabric;
pub mod learning;
#[cfg(any(feature = "codebook", feature = "hologram", feature = "spo", feature = "compress"))]
pub mod extensions;
#[cfg(feature = "python")]
pub mod python;
// Re-exports for convenience
pub use crate::core::{Fingerprint, Embedding, VsaOps, DIM, DIM_U64};
pub use crate::cognitive::{Thought, Concept, Belief, ThinkingStyle};
pub use crate::nars::{TruthValue, Evidence, Deduction, Induction, Abduction};
pub use crate::graph::{Edge, EdgeType, Traversal};
pub use crate::world::{World, Counterfactual, Change};
pub use crate::query::{Query, QueryResult, cypher_to_sql, SqlEngine, QueryBuilder};
pub use crate::storage::{Database, LanceStore, NodeRecord, EdgeRecord};
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
    #[error("Lance error: {0}")]
    Lance(#[from] lance::Error),
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    #[error("DataFusion error: {0}")]
    DataFusion(#[from] datafusion::error::DataFusionError),
    #[error("Tokio error: {0}")]
    Tokio(#[from] tokio::io::Error),
}
impl From<storage::StorageError> for Error {
    fn from(e: storage::StorageError) -> Self {
        Error::Storage(e.to_string())
    }
impl From<query::QueryError> for Error {
    fn from(e: query::QueryError) -> Self {
        Error::Query(e.to_string())
pub type Result<T> = std::result::Result<T, Error>;
/// Version info
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
/// Fingerprint dimensions
pub const FINGERPRINT_BITS: usize = 10_000;
pub const FINGERPRINT_U64: usize = 157;  // ceil(10000/64)
pub const FINGERPRINT_BYTES: usize = FINGERPRINT_U64 * 8;
