//! Core primitives: Fingerprints, SIMD operations, VSA algebra.

mod fingerprint;
pub mod simd;
mod vsa;
mod buffer;
mod scent;

pub use fingerprint::Fingerprint;
pub use simd::{hamming_distance, batch_hamming, HammingEngine};
pub use vsa::VsaOps;
pub use buffer::BufferPool;
pub use scent::*;

/// Dense embedding vector
pub type Embedding = Vec<f32>;

/// Fingerprint dimension in bits (16K = 2^14, HDC-aligned)
pub const DIM: usize = 16_384;

/// Fingerprint dimension in u64 words (16384/64 = 256, exact)
pub const DIM_U64: usize = 256;

/// Last word mask (16384 % 64 == 0, so all bits valid — no partial word)
pub const LAST_MASK: u64 = u64::MAX;
