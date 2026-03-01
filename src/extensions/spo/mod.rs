//! SPO Extension - 3D Content-Addressable Knowledge Graph
//! O(1) triple queries via VSA resonance (vs Cypher O(log N))
//!
//! ## Modules
//!
//! - `spo` — SPO Crystal: 3D spatial indexing, orthogonal codebook, qualia coloring
//! - `gestalt` — Bundling detection, tilt correction, truth trajectories (PRs #74-81)
//! - `spo_harvest` — SPO distance harvest: cosine replacement at 238× less cost
//! - `shift_detector` — 0.5σ stripe migration as distributional shift detector
//! - `causal_trajectory` — BNN causal trajectory hydration → NARS truth + gestalt + DN growth
//! - `jina_api` / `jina_cache` — Jina embedding API + cache

mod jina_api;
mod jina_cache;
pub mod gestalt;
pub mod spo_harvest;
pub mod shift_detector;
pub mod causal_trajectory;
mod spo;

pub use jina_api::{JinaClient, jina_embed_curl};
pub use jina_cache::JinaCache;
pub use spo::*;
