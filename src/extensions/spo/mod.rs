//! SPO Extension - 3D Content-Addressable Knowledge Graph
//! O(1) triple queries via VSA resonance (vs Cypher O(log N))

mod jina_api;
mod jina_cache;
mod spo;

pub use jina_api::{JinaClient, jina_embed_curl};
pub use jina_cache::JinaCache;
pub use spo::*;
