//! SPO Extension - 3D Content-Addressable Knowledge Graph
//! O(1) triple queries via VSA resonance (vs Cypher O(log N))

mod spo;
mod jina_api;
mod jina_cache;

pub use spo::*;
pub use jina_api::{JinaClient, jina_embed_curl};
pub use jina_cache::JinaCache;
