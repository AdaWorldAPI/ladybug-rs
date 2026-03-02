//! SPO — Subject-Predicate-Object Cognitive Substrate
//!
//! Core module for 3D content-addressable knowledge graph operations.
//! O(1) triple queries via VSA resonance (vs Cypher O(log N)).
//!
//! This is the cognitive substrate — always-on, no feature gates.
//! BNN and SPO are unconditional dependencies after the rustynum integration.
//!
//! ## Modules
//!
//! - `spo` — SPO Crystal: 3D spatial indexing, orthogonal codebook, qualia coloring
//! - `gestalt` — Bundling detection, tilt correction, truth trajectories
//! - `spo_harvest` — SPO distance harvest: cosine replacement at 238× less cost
//! - `shift_detector` — 0.5σ stripe migration as distributional shift detector
//! - `causal_trajectory` — BNN causal trajectory hydration → NARS truth + gestalt + DN growth
//! - `context_crystal` — Context crystal qualia vectors
//! - `meta_resonance` — Meta-resonance patterns
//! - `nsm_substrate` — Neural substrate modeling
//! - `codebook_training` — Codebook training routines
//! - `deepnsm_integration` — Deep NSM integration layer
//! - `cognitive_codebook` — Cognitive codebook management
//! - `crystal_lm` — Crystal language model
//! - `sentence_crystal` — Sentence crystal encoding
//! - `jina_api` / `jina_cache` — Jina embedding API + cache

mod jina_api;
pub(crate) mod jina_cache;
pub mod gestalt;
pub mod spo_harvest;
pub mod shift_detector;
pub mod causal_trajectory;
pub mod clam_path;
pub mod context_crystal;
pub mod meta_resonance;
pub mod nsm_substrate;
pub mod codebook_training;
pub mod deepnsm_integration;
pub mod cognitive_codebook;
pub mod codebook_hydration;
pub mod crystal_lm;
pub mod sentence_crystal;
mod spo;

pub use jina_api::{JinaClient, jina_embed_curl};
pub use jina_cache::JinaCache;
pub use spo::*;
