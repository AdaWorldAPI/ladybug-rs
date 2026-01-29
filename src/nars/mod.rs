//! NARS (Non-Axiomatic Reasoning System) implementation.
//!
//! Provides truth value management, belief revision, and inference rules
//! for reasoning under uncertainty.

mod truth;
mod inference;
mod evidence;

pub use truth::TruthValue;
pub use inference::{InferenceRule, Deduction, Induction, Abduction, Analogy};
pub use evidence::Evidence;
