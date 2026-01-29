//! Cognitive Fabric - mRNA cross-pollination and butterfly detection
//!
//! This is the unified substrate where all subsystems resonate.

pub mod mrna;
pub mod butterfly;
pub mod subsystem;

pub use mrna::{MRNA, ResonanceField, CrossPollination};
pub use butterfly::{ButterflyDetector, Butterfly, ButterflyPrediction};
pub use subsystem::Subsystem;
