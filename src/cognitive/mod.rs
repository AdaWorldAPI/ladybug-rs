//! Cognitive Module - Complete Cognitive Architecture
//!
//! Integrates:
//! - 12 Thinking Styles (field modulation)
//! - 4 QuadTriangles (Processing/Content/Gestalt/Crystallization)
//! - 7-Layer Consciousness Stack
//! - Collapse Gate (FLOW/HOLD/BLOCK)
//! - Rung System (0-9 meaning depth levels)
//! - Integrated Cognitive Fabric
//! - Sigma-10 Membrane (tau/sigma/qualia -> 10K bits)

mod collapse_gate;
mod fabric;
mod grammar_engine;
pub mod membrane;
pub mod metacog;
mod quad_triangle;
pub mod recursive;
mod rung;
mod seven_layer;
mod style;
mod substrate;
mod thought;

pub use collapse_gate::*;
pub use grammar_engine::{
    GrammarCognitiveEngine, GrammarRole, GrammarTriangle, IngestResult, deserialize_state,
    process_batch, serialize_state,
};
pub use membrane::{
    ConsciousnessParams, Membrane, QUALIA_END, QUALIA_START, SIGMA_END, SIGMA_START, TAU_END,
    TAU_START, consciousness_fingerprint, decode_consciousness, encode_consciousness,
};
pub use quad_triangle::*;
pub use rung::*;
pub use seven_layer::*;
pub use style::*;
pub use substrate::*;
pub use thought::{Belief, Concept, Thought};
