//! Cognitive Fabric - mRNA cross-pollination and butterfly detection
//!
//! This is the unified substrate where all subsystems resonate.

pub mod mrna;
pub mod butterfly;
pub mod subsystem;
pub mod firefly_frame;

pub use mrna::{MRNA, ResonanceField, CrossPollination, FieldSnapshot};
pub use butterfly::{ButterflyDetector, Butterfly, ButterflyPrediction};
pub use subsystem::Subsystem;
pub use firefly_frame::{
    FireflyFrame, FrameHeader, Instruction, LanguagePrefix,
    ExecutionContext, ConditionFlags, FrameBuilder,
};
