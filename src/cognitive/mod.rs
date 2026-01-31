//! Cognitive Module - Complete Cognitive Architecture
//!
//! Integrates:
//! - 12 Thinking Styles (field modulation)
//! - 4 QuadTriangles (Processing/Content/Gestalt/Crystallization)
//! - 7-Layer Consciousness Stack
//! - Collapse Gate (FLOW/HOLD/BLOCK)
//! - Rung System (0-9 meaning depth levels)
//! - Integrated Cognitive Fabric

mod thought;
mod style;
mod quad_triangle;
mod collapse_gate;
mod seven_layer;
mod rung;
mod fabric;
// TODO: Fix API mismatches before enabling
// mod grammar_engine;
// mod substrate;

pub use thought::{Thought, Concept, Belief};
pub use style::*;
pub use quad_triangle::*;
pub use collapse_gate::*;
pub use seven_layer::*;
pub use rung::*;
pub use fabric::*;
// TODO: Fix API mismatches before enabling
// pub use grammar_engine::*;
// pub use substrate::*;
