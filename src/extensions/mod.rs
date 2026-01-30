//! Optional Extensions for LadybugDB
//!
//! These modules provide specialized functionality that can be enabled
//! via Cargo features. The core ladybug-rs works without any extensions.
//!
//! # Available Extensions
//!
//! | Feature | Module | Description |
//! |---------|--------|-------------|
//! | `codebook` | [`codebook`] | Multi-pass CAM with Hamming meta-resonance |
//! | `hologram` | [`hologram`] | 4KB holographic crystals with quorum ECC |
//! | `spo` | [`spo`] | 3D content-addressable knowledge graph |
//! | `compress` | [`compress`] | Semantic compression via crystal dictionary |
//!
//! # Usage
//!
//! ```toml
//! [dependencies]
//! ladybug = { version = "0.1", features = ["codebook", "spo"] }
//! ```

#[cfg(feature = "codebook")]
pub mod codebook;

#[cfg(feature = "hologram")]
pub mod hologram;

#[cfg(feature = "spo")]
pub mod spo;

#[cfg(feature = "compress")]
pub mod compress;
