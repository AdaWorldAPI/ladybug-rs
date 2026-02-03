//! Hologram Extension - 4KB Holographic Crystals with Quorum ECC
//! 5×5×5 quorum fields, any 2-of-3 copies can reconstruct
//! Quantum crystal operations for complete quantum gate set
//! Quantum algorithms for computational tasks on crystal substrate
//! True quantum interference substrate with phase tags
//! Multi-dimensional hypercube crystals (5D, 7D, 11D, 13D, 17D)

mod crystal4k;
mod field;
mod memory;
pub mod quantum_crystal;
pub mod quantum_algorithms;
pub mod quantum_field;
pub mod quantum_5d;
pub mod quantum_7d;
pub mod quantum_11d;
pub mod quantum_13d;
pub mod quantum_17d;
pub mod bitchain_5d;
pub mod bitchain_7d;

pub use crystal4k::*;
pub use field::*;
pub use memory::*;
pub use quantum_crystal::*;
pub use quantum_algorithms::*;
pub use quantum_field::*;
pub use quantum_5d::*;
pub use quantum_7d::*;
pub use quantum_11d::*;
pub use quantum_13d::*;
pub use quantum_17d::*;
pub use bitchain_5d::*;
pub use bitchain_7d::*;
