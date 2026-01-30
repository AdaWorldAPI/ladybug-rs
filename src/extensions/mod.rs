//! Optional Extensions for LadybugDB
//!
//! Enable via Cargo features: `codebook`, `hologram`, `spo`, `compress`

#[cfg(feature = "codebook")]
pub mod codebook;

#[cfg(feature = "hologram")]
pub mod hologram;

#[cfg(feature = "spo")]
pub mod spo;

#[cfg(feature = "spo")]
pub mod context_crystal;

#[cfg(feature = "spo")]
pub mod meta_resonance;

#[cfg(feature = "spo")]
pub mod nsm_substrate;

#[cfg(feature = "spo")]
pub mod codebook_training;

#[cfg(feature = "spo")]
pub mod deepnsm_integration;

#[cfg(feature = "spo")]
pub mod cognitive_codebook;

#[cfg(feature = "compress")]
pub mod compress;
