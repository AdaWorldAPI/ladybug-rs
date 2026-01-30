//! Hologram Extension - 4KB Holographic Crystals with Quorum ECC
//!
//! 5×5×5 quorum fields compressed to 3×10Kbit.
//! 170MB = 4096 attractor basins navigating 2^1.25M space.
//!
//! # Architecture
//!
//! ```text
//! 3 copies × 10Kbit = 30Kbit per memory
//! 5×5×5 = 125 cells × 4096 crystals = 170MB total
//! Any 2-of-3 can reconstruct (quorum consensus)
//! ```
//!
//! # Key Insight
//!
//! The crystal doesn't "store" memories—it **becomes** them.
//! Like a hologram, any shard contains the whole.
//! Damage doesn't erase; it degrades gracefully.

mod crystal4k;
mod field;
mod memory;

pub use crystal4k::*;
pub use field::*;
pub use memory::*;
