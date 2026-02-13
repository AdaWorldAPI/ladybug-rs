//! Core primitives: Fingerprints, SIMD operations, VSA algebra.

mod buffer;
mod fingerprint;
mod scent;
pub mod simd;
pub mod vsa;

pub use buffer::BufferPool;
pub use fingerprint::Fingerprint;
pub use scent::*;
pub use simd::{HammingEngine, batch_hamming, hamming_distance};
pub use vsa::VsaOps;

/// Dense embedding vector
pub type Embedding = Vec<f32>;

/// Fingerprint dimension in bits (16K = 2^14, HDC-aligned)
pub const DIM: usize = 16_384;

/// Fingerprint dimension in u64 words (16384/64 = 256, exact)
pub const DIM_U64: usize = 256;

/// Last word mask (16384 % 64 == 0, so all bits valid — no partial word)
pub const LAST_MASK: u64 = u64::MAX;
