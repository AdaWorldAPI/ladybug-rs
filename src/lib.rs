//! # LadybugDB
//! 
//! Unified cognitive database: SQL + Cypher + Vector + Hamming + NARS + Counterfactuals.
//! 
//! Built on Lance columnar storage with AGI operations as first-class primitives.
//! 
//! ## Quick Start
//! 
//! ```rust
//! use ladybug::{Database, Thought};
//! 
//! // Open database
//! let db = Database::open("./mydb")?;
//! 
//! // Conventional: SQL
//! let results = db.sql("SELECT * FROM thoughts WHERE confidence > 0.7")?;
//! 
//! // Conventional: Vector search
//! let similar = db.vector_search(&embedding, 10)?;
//! 
//! // AGI: Resonance (Hamming similarity on 10K-bit fingerprints)
//! let resonant = db.resonate(&fingerprint, 0.7)?;
//! 
//! // AGI: Graph traversal with amplification
//! let chains = db.query()
//!     .from("config_change")
//!     .causes()
//!     .amplifies(2.0)
//!     .depth(1..=5)
//!     .execute()?;
//! 
//! // AGI: Counterfactual reasoning
//! let what_if = db.fork()
//!     .apply(|w| w.remove("feature_flag"))
//!     .propagate()
//!     .diff()?;
//! 
//! // AGI: NARS inference
//! let conclusion = premise1.infer::<Deduction>(&premise2)?;
//! ```

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
pub use crate::core::{Fingerprint, Embedding, VsaOps};
pub use crate::cognitive::{Thought, Concept, Belief, ThinkingStyle};
pub use crate::nars::{TruthValue, Evidence, Deduction, Induction, Abduction};
pub use crate::graph::{Edge, EdgeType, Traversal};
pub use crate::world::{World, Counterfactual, Change};
pub use crate::query::{Query, QueryResult};
pub use crate::storage::Database;

/// Crate-level error type
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Storage error: {0}")]
    Storage(#[from] storage::StorageError),
    
    #[error("Query error: {0}")]
    Query(#[from] query::QueryError),
    
    #[error("Invalid fingerprint: expected {expected} bytes, got {got}")]
    InvalidFingerprint { expected: usize, got: usize },
    
    #[error("Node not found: {0}")]
    NodeNotFound(String),
    
    #[error("Invalid inference: {0}")]
    InvalidInference(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Version info
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Fingerprint dimensions
pub const FINGERPRINT_BITS: usize = 10_000;
pub const FINGERPRINT_U64: usize = 157;  // ceil(10000/64)
pub const FINGERPRINT_BYTES: usize = FINGERPRINT_U64 * 8;

/// Default embedding dimension (for dense vectors)
pub const EMBEDDING_DIM: usize = 1024;

// Learning module re-exports
pub use crate::learning::{
    Blackboard, LearningSession, LearningMoment, 
    ConceptExtractor, ResonanceTracker,
};
