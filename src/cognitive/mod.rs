//! Cognitive Module - Complete Cognitive Architecture
//!
//! Integrates:
//! - 12 Thinking Styles (field modulation)
//! - 4 QuadTriangles (Processing/Content/Gestalt/Crystallization)
//! - 10-Layer Cognitive Stack (L1:Recognition → L10:Crystallization)
//! - Satisfaction Gate (Maslow hierarchy for layer activation)
//! - 2-Stroke Engine (async layer execution with stacked awareness)
//! - Collapse Gate (FLOW/HOLD/BLOCK)
//! - Rung System (0-9 meaning depth levels)
//! - Integrated Cognitive Fabric
//! - Sigma-10 Membrane (tau/sigma/qualia -> 10K bits)
//! - Awareness Blackboard (layer stack evaluator)
//! - Cortex (XOR delta layer coordinator — Photoshop/SharePoint model)
//! - Socratic Sieve (3-gate self-modification: truth/good/necessary)
//! - DeltaLayer (ephemeral XOR diffs — ground truth stays &self forever)
//!
//! # 10-Layer Cognitive Stack
//!
//! ```text
//! L1  Recognition     — pattern matching, fingerprint encoding
//! L2  Resonance       — field binding, similarity search, association
//! L3  Appraisal       — gestalt formation, hypothesis, evaluation
//! L4  Routing         — branch selection, fan-out, template dispatch
//! L5  Execution       — active manipulation, synthesis, production
//!     ─── single agent boundary ───
//! L6  Delegation      — cognitive fan-out, multi-agent dispatch
//! L7  Contingency     — cross-branch, things could be otherwise
//! L8  Integration     — evidence merge, learning from outcomes
//! L9  Validation      — NARS revision, Brier calibration, sieve
//! L10 Crystallization — what survives becomes the system
//! ```

pub mod awareness;
pub mod cognitive_kernel;
mod collapse_gate;
pub mod cortex;
pub(crate) mod fabric;
mod grammar_engine;
pub mod layer_stack;
pub mod membrane;
pub mod metacog;
mod quad_triangle;
pub mod recursive;
mod rung;
pub mod satisfaction_gate;
mod seven_layer;
pub mod sieve;
mod style;
mod substrate;
mod thought;
pub mod two_stroke;
pub mod pattern_detector;

pub use awareness::{AwarenessBlackboard, AwarenessSnapshot, CortexResult};
pub use collapse_gate::*;
pub use cortex::{Cortex, DeltaLayer, DeltaSource};
pub use grammar_engine::{
    GrammarCognitiveEngine, GrammarRole, GrammarTriangle, IngestResult, deserialize_state,
    process_batch, serialize_state,
};
pub use layer_stack::{
    ConsciousnessSnapshot, LayerId, LayerMarker, LayerNode, LayerResult, NUM_LAYERS,
    SevenLayerNode, apply_layer_result, layer_resonance_matrix, process_all_layers_parallel,
    process_layer, process_layers_wave, snapshot_consciousness,
};
pub use membrane::{
    ConsciousnessParams, Membrane, QUALIA_END, QUALIA_START, SIGMA_END, SIGMA_START, TAU_END,
    TAU_START, consciousness_fingerprint, decode_consciousness, encode_consciousness,
};
pub use quad_triangle::*;
pub use rung::*;
pub use satisfaction_gate::LayerSatisfaction;
pub use sieve::SocraticSieve;
pub use style::*;
pub use substrate::*;
pub use thought::{Belief, Concept, Thought};
pub use two_stroke::{
    ValidationResult, build_rule_fingerprints, build_style_fingerprints, crystallize,
    process_2stroke, recover_modulation, select_rules_by_resonance, select_style_by_resonance,
    snapshot_scores, validate_l9,
};
pub use cognitive_kernel::{CognitiveKernel, CognitiveKernelResult, KernelLayerOp};
pub use pattern_detector::{MetaPattern, detect as detect_meta_pattern};
