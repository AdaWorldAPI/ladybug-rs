//! Qualia Module — Phenomenal Experience Computation
//!
//! Provides texture computation for Container fingerprints,
//! capturing the "felt-sense" of cognitive content in an 8-dimensional space.
//!
//! ## Sub-modules
//!
//! - `texture`: 8-dimensional phenomenal texture of a fingerprint
//! - `meaning_axes`: 48 canonical bipolar meaning axes (from dragonfly-vsa)
//! - `council`: Three-archetype inner council with VSA consensus
//! - `felt_traversal`: Tree walk with sibling superposition, awe triples, free energy
//! - `reflection`: NARS introspection, hydration chains, free energy semiring
//! - `volition`: Self-directed action selection via free energy + ghost resonance + council
//! - `dream_bridge`: Ghost harvesting + dream consolidation → hydration injection
//! - `mul_bridge`: MUL metacognitive state → adaptive reflection thresholds + feedback

pub mod texture;
pub mod meaning_axes;
pub mod council;
pub mod resonance;
pub mod gestalt;
pub mod felt_traversal;
pub mod reflection;
pub mod volition;
pub mod dream_bridge;
pub mod mul_bridge;

pub use texture::{GraphMetrics, Texture, compute};
pub use meaning_axes::{
    AXES, AxisActivation, AxisFamily, CodeFeeling, MeaningAxis, Viscosity,
    decode_axes, detect_viscosity, encode_axes,
};
pub use council::{Archetype, Epiphany, EpiphanyDetector, InnerCouncil};
pub use resonance::{
    AwarenessEntry, AwarenessField, AwarenessLens, AwarenessSummary,
    FocusMask, FocusedEntry, FocusedResonance, HdrResonance, TriangleCouncil,
};
pub use gestalt::{
    CollapseGate, CrossPerspective, FramedContent, GestaltFrame,
    Quadrant, RoleAtoms,
};
pub use felt_traversal::{
    AweTriple, FeltChoice, FeltPath, VERB_AWE, VERB_FELT_TRACE,
    VERB_SIBLING_BUNDLE, felt_walk, felt_wander, free_energy_landscape,
    node_free_energy,
};
pub use reflection::{
    EnergyStrategy, FreeEnergySemiring, HydrationChain, HydrationStep,
    ReflectionEntry, ReflectionOutcome, ReflectionResult, VERB_REFLECTION,
    hydrate_explorers, read_truth, reflect_and_hydrate, reflect_walk, write_truth,
};
pub use volition::{
    CouncilWeights, VolitionalAct, VolitionalAgenda, VERB_VOLITION,
    compute_agenda, focused_volition, volitional_cycle, volitional_gradient,
};
pub use dream_bridge::{
    DreamInjection, DreamReflectionConfig, DreamReflectionResult,
    GhostRecord, GHOST_RESONANCE_THRESHOLD, VERB_DREAM_GHOST,
    dream_consolidate_with_ghosts, dream_reflection_cycle,
    harvest_ghosts, ghosts_to_records,
};
pub use mul_bridge::{
    AdaptiveThresholds, MulReflectionResult,
    adaptive_thresholds, mul_council_weights, mul_reflection_feedback,
    mul_volitional_cycle, reclassify_with_thresholds, reflection_to_mul_learning,
};
