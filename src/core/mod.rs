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

/// Fingerprint dimension in bits (10K VSA standard)
pub const DIM: usize = 10_000;

/// Fingerprint dimension in u64 words
pub const DIM_U64: usize = 157; // ceil(10000/64)

/// Last word mask for 10K bits (only 16 bits used in last u64)
pub const LAST_MASK: u64 = (1 << 16) - 1;
