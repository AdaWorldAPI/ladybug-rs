//! Core primitives: Fingerprints, SIMD operations, VSA algebra.

mod fingerprint;
mod simd;
mod vsa;
mod buffer;

pub use fingerprint::Fingerprint;
pub use simd::{hamming_distance, batch_hamming, HammingEngine};
pub use vsa::VsaOps;
pub use buffer::BufferPool;

/// Dense embedding vector
pub type Embedding = Vec<f32>;
