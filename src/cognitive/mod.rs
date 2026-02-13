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

mod thought;
mod style;
mod quad_triangle;
mod collapse_gate;
mod seven_layer;
mod rung;
mod fabric;
mod grammar_engine;
mod substrate;
pub mod membrane;
pub mod metacog;
pub mod recursive;

pub use thought::{Thought, Concept, Belief};
pub use style::*;
pub use quad_triangle::*;
pub use collapse_gate::*;
pub use seven_layer::*;
pub use rung::*;
pub use grammar_engine::{
    GrammarRole, GrammarTriangle, GrammarCognitiveEngine,
    IngestResult, process_batch, serialize_state, deserialize_state,
};
pub use substrate::*;
pub use membrane::{
    Membrane, ConsciousnessParams,
    encode_consciousness, decode_consciousness, consciousness_fingerprint,
    TAU_START, TAU_END, SIGMA_START, SIGMA_END, QUALIA_START, QUALIA_END,
};
