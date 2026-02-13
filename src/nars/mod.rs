//! NARS (Non-Axiomatic Reasoning System) implementation.
//!
//! Provides truth value management, belief revision, and inference rules
//! for reasoning under uncertainty.

mod truth;
mod inference;
mod evidence;
mod context;
pub mod adversarial;
pub mod contradiction;

pub use truth::TruthValue;
pub use inference::{InferenceRule, Deduction, Induction, Abduction, Analogy};
pub use evidence::Evidence;
pub use context::{
    InferenceContext, StyleWeights, AtomGate, PearlMode,
    CollapseModulation, InferenceRuleKind, AtomKind,
};
