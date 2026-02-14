//! 7-Layer Consciousness Stack — Backward-Compatibility Wrapper
//!
//! This module re-exports the 10-layer cognitive stack from `layer_stack.rs`.
//! All types and functions remain available under their original names.
//!
//! `SevenLayerNode` is a type alias for `LayerNode`.
//! `LayerId::L1` through `LayerId::L7` still work as before.
//! New layers L8 (Integration), L9 (Validation), L10 (Crystallization)
//! are available for new code.
//!
//! The wave processing function `process_layers_wave` now processes all 10
//! layers in 7 waves (L1+L2 parallel, L3+L4 parallel, L5 alone,
//! L6+L7 parallel, L8 alone, L9 alone, L10 alone).

#[allow(unused_imports)]
pub use super::layer_stack::*;
