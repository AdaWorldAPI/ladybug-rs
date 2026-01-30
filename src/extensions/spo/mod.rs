//! SPO Extension - 3D Content-Addressable Knowledge Graph
//!
//! Replaces Cypher queries with O(1) VSA resonance lookup.
//!
//! # Architecture
//!
//! ```text
//! Traditional:  MATCH (s)-[p]->(o) WHERE s.name = "Ada"
//!               → O(log N) index lookup + graph traversal
//!
//! Crystal:      query(S="Ada", P="feels", O=?)
//!               → 3D address + orthogonal cleanup + qualia overlay
//!               → O(1) resonance
//!
//! Encoding:     S ⊕ ROLE_S ⊕ P ⊕ ROLE_P ⊕ O ⊕ ROLE_O ⊕ Q ⊕ ROLE_Q
//!                   ↓           ↓           ↓           ↓
//!                x-axis      y-axis      z-axis      qualia
//!               address     address     address     coloring
//! ```
//!
//! # Features
//!
//! - **3D Spatial Hashing**: S→x, P→y, O→z coordinates in 5×5×5 grid
//! - **VSA Encoding**: 10,000-bit fingerprints with role-filler binding
//! - **Orthogonal Codebook**: Gram-Schmidt-like cleaning for high SNR

mod spo;
mod jina_api;
mod jina_cache;

pub use spo::*;
pub use jina_api::JinaClient;
pub use jina_cache::JinaCache;
