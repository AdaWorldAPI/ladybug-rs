//! Hologram Extension - 4KB Holographic Crystals with Quorum ECC
//! 5×5×5 quorum fields, any 2-of-3 copies can reconstruct
//! Quantum crystal operations for complete quantum gate set
//! Quantum algorithms for computational tasks on crystal substrate

mod crystal4k;
mod field;
mod memory;
pub mod quantum_crystal;
pub mod quantum_algorithms;

pub use crystal4k::*;
pub use field::*;
pub use memory::*;
pub use quantum_crystal::*;
pub use quantum_algorithms::*;
