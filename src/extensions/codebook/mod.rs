//! Codebook Extension - Multi-pass CAM with Hamming Meta-Resonance
//! ~6µs per lookup, 176K lookups/sec, 157KB memory (L2 cache)

mod dictionary_crystal;
mod hierarchical;
mod multipass;

pub use dictionary_crystal::*;
pub use hierarchical::*;
pub use multipass::*;
