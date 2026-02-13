//! Graph module - Cognitive Graph Substrate
//!
//! Nodes = Fingerprints (concepts, entities, states)
//! Edges = Fingerprint ⊗ Verb ⊗ Fingerprint (relationships)
//! Verbs = 144 core relations on Go board topology

pub mod cognitive;
mod edge;
mod traversal;

pub use cognitive::{CogEdge, CogGraph, CogNode, NodeType, Verb, VerbCategory};
pub use edge::{Edge, EdgeType};
pub use traversal::Traversal;

pub mod avx_engine;
pub use avx_engine::{
    FingerprintGraph, QueryMatch, avx512_available, batched_query, hamming_distance, simd_info,
};
