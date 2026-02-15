//! Spectroscopy Module — Classify Containers into Cognitive Coordinates
//!
//! Given an 8,192-bit Container bitpattern and a ConsciousnessSnapshot,
//! the spectroscopy detector determines:
//!
//! - **RungLevel** (0-9): semantic abstraction depth
//! - **ThinkingStyle**: which of the 12 cognitive lenses to apply
//! - **FieldModulation**: tuned parameters for the cognitive field
//!
//! # Architecture
//!
//! ```text
//!   Container ─────► features::extract() ──► SpectralFeatures
//!                                                  │
//!   ConsciousnessSnapshot ──► ContextSignal ◄──────┘
//!                                  │
//!                    ┌─────────────┼─────────────┐
//!                    │             │             │
//!                    ▼             ▼             ▼
//!              classify_rung  classify_style  compute_modulation
//!                    │             │             │
//!                    └─────────────┼─────────────┘
//!                                  │
//!                                  ▼
//!                (RungLevel, ThinkingStyle, FieldModulation)
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use ladybug::spectroscopy::detector;
//! use ladybug::container::Container;
//!
//! let c = Container::random(42);
//! let snapshot = /* obtain from layer_stack */;
//! let (rung, style, modulation) = detector::classify(&c, &snapshot);
//! ```

pub mod detector;
pub mod features;

// Re-export primary API
pub use detector::{classify, classify_rung_only, classify_style_only};
pub use features::{SpectralFeatures, extract};
