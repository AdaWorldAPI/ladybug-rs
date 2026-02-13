//! NARS (Non-Axiomatic Reasoning System) implementation.
//!
//! Provides truth value management, belief revision, and inference rules
//! for reasoning under uncertainty.

pub mod adversarial;
mod context;
pub mod contradiction;
mod evidence;
mod inference;
mod truth;

pub use context::{
    AtomGate, AtomKind, CollapseModulation, InferenceContext, InferenceRuleKind, PearlMode,
    StyleWeights,
};
pub use evidence::Evidence;
pub use inference::{Abduction, Analogy, Deduction, Induction, InferenceRule};
pub use truth::TruthValue;
