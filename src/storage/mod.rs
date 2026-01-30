//! Storage module - Persistence layers
//!
//! Three-layer architecture:
//! - WHAT: LanceDB (content, fingerprints, vectors)
//! - WHERE: Kuzu (graph structure, relationships)  
//! - WHEN: Redis/Dragonfly (temporal, execution queue)

pub mod lance;
pub mod database;
pub mod kuzu;

pub use lance::LanceStore;
pub use database::Database;
pub use kuzu::{KuzuStore, NodeRecord, EdgeRecord, PathRecord};
