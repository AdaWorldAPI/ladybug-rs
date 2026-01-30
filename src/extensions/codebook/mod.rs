//! Codebook Extension - Multi-pass CAM with Hamming Meta-Resonance
//!
//! Massive semantic compression through learned resonance surfaces.
//!
//! # Architecture
//!
//! **Pass 1** (expensive, one-time): Collect concepts from rich corpus
//! - Books, NARS patterns, qualia mappings, SPO relations
//! - Jina embed → 10Kbit fingerprint → cluster into CAM slots
//!
//! **Pass 2** (cheap, runtime): Hamming meta-resonance lookup
//! - New text → hash fingerprint (NO API CALL)
//! - XOR scan against CAM slots
//! - ~6µs per lookup, 176K lookups/sec
//! - 157KB memory (fits L2 cache)
//!
//! # Performance
//!
//! ```text
//! 10K lookups in 56.73ms
//! Throughput: 176,274 lookups/sec
//! Memory: 157 KB (128 slots × 10Kbit)
//! ```

mod dictionary_crystal;
mod hierarchical;
mod multipass;

pub use dictionary_crystal::*;
pub use hierarchical::*;
pub use multipass::*;
