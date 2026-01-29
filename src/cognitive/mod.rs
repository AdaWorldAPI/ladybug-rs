//! Cognitive Module - Complete Cognitive Architecture
//!
//! Integrates:
//! - 12 Thinking Styles (field modulation)
//! - 4 QuadTriangles (Processing/Content/Gestalt/Crystallization)
//! - 7-Layer Consciousness Stack
//! - Collapse Gate (FLOW/HOLD/BLOCK)
//! - Integrated Cognitive Fabric
//! - Unified Cognitive Substrate
//! - Grammar-aware Cognitive Engine
//! - Unified Fabric (multi-style coordination)

mod thought;
mod style;
mod quad_triangle;
mod collapse_gate;
mod seven_layer;
mod fabric;
mod substrate;
mod grammar_engine;
mod unified_fabric;

pub use thought::{Thought, Concept, Belief};
pub use style::*;
pub use quad_triangle::*;
pub use collapse_gate::*;
pub use seven_layer::*;
pub use fabric::*;
pub use substrate::*;
pub use grammar_engine::*;
pub use unified_fabric::*;
