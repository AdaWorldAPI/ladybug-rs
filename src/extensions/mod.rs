//! Optional Extensions for LadybugDB
//!
//! Enable via Cargo features: `codebook`, `hologram`, `compress`
//!
//! Note: SPO modules have been promoted to `crate::spo` (top-level, always-on).

#[cfg(feature = "codebook")]
pub mod codebook;

#[cfg(feature = "hologram")]
pub mod hologram;

#[cfg(feature = "compress")]
pub mod compress;
