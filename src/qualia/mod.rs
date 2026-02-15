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

pub mod texture;
pub mod meaning_axes;
pub mod council;
pub mod resonance;
pub mod gestalt;

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
