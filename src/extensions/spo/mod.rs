//! SPO Extension - 3D Content-Addressable Knowledge Graph
//! O(1) triple queries via VSA resonance (vs Cypher O(log N))
//!
//! ## Modules
//!
//! - `spo` — SPO Crystal: 3D spatial indexing, orthogonal codebook, qualia coloring
//! - `gestalt` — Bundling detection, tilt correction, truth trajectories (PRs #74-81)
//! - `jina_api` / `jina_cache` — Jina embedding API + cache

mod jina_api;
mod jina_cache;
pub mod gestalt;
mod spo;

pub use jina_api::{JinaClient, jina_embed_curl};
pub use jina_cache::JinaCache;
pub use spo::*;
