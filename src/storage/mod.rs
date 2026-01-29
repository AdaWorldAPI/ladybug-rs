//! Storage Layer
//!
//! Provides persistent storage via LanceDB with:
//! - Columnar storage for thoughts, edges, fingerprints
//! - Vector ANN index for embeddings
//! - Batch scan for Hamming similarity

mod database;
mod lance;

pub use database::Database;
pub use lance::{LanceStore, ThoughtRow, EdgeRow, FINGERPRINT_BYTES};
