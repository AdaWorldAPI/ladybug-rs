//! Storage layer - LanceDB integration
//!
//! Provides persistent storage via LanceDB with:
//! - Columnar Arrow format
//! - Native vector ANN indices
//! - Zero-copy operations
//! - Versioned datasets

mod database;
mod lance;

pub use database::Database;
pub use lance::{
    LanceStore, 
    NodeRecord, 
    EdgeRecord,
    nodes_schema,
    edges_schema,
    sessions_schema,
    FINGERPRINT_BYTES,
    EMBEDDING_DIM,
    THINKING_STYLE_DIM,
};

#[derive(thiserror::Error, Debug)]
pub enum StorageError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Not found: {0}")]
    NotFound(String),
    #[error("Lance error: {0}")]
    Lance(String),
    #[error("Arrow error: {0}")]
    Arrow(String),
}
