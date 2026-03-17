//! Crystal API — crate-internal re-exports for SPO Crystal types.
//!
//! Provides a single import point for code outside `spo/` that needs
//! access to the SPO Crystal substrate:
//!
//! ```rust,ignore
//! use crate::spo::crystal_api::*;
//! ```

pub(crate) use super::spo::{OrthogonalCodebook, Qualia, SPOCrystal, Triple};
