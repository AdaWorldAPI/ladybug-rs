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

// === Core modules ===
pub mod core;
pub mod cognitive;
pub mod nars;
pub mod grammar;  // NEW: Grammar Triangle
pub mod graph;
pub mod world;
pub mod search;
pub mod query;
pub mod storage;
pub mod fabric;
pub mod learning;

// === Optional extensions ===
#[cfg(any(feature = "codebook", feature = "hologram", feature = "spo", feature = "compress"))]
pub mod extensions;

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "bench")]
pub mod bench;

// === Re-exports for convenience ===

// Core types
pub use crate::core::{Fingerprint, Embedding, VsaOps, DIM, DIM_U64};

// Cognitive types
pub use crate::cognitive::{Thought, Concept, Belief, ThinkingStyle};

// NARS (Non-Axiomatic Reasoning)
pub use crate::nars::{TruthValue, Evidence, Deduction, Induction, Abduction};

// Grammar Triangle (universal input layer)
pub use crate::grammar::{GrammarTriangle, NSMField, CausalityFlow, QualiaField};

// Graph traversal
pub use crate::graph::{Edge, EdgeType, Traversal};

// Counterfactual worlds
pub use crate::world::{World, Counterfactual, Change};

// Query engine
pub use crate::query::{Query, QueryResult, cypher_to_sql, SqlEngine, QueryBuilder};

// Storage
#[cfg(feature = "lancedb")]
pub use crate::storage::{Database, LanceStore, NodeRecord, EdgeRecord};

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

pub type Result<T> = std::result::Result<T, Error>;

// === Constants ===

/// Version info
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Fingerprint dimensions
pub const FINGERPRINT_BITS: usize = 10_000;
pub const FINGERPRINT_U64: usize = 157;  // ceil(10000/64)
pub const FINGERPRINT_BYTES: usize = FINGERPRINT_U64 * 8;
