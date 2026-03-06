// Stolen from lance-graph — hardened Cypher parser (the bouncer).
//
// This module sits OUTSIDE the door. It validates syntax, rejects garbage,
// and hands off clean resolved ASTs. It does NOT know about BindSpace,
// Containers, fingerprints, SPO, or qualia. It checks IDs at the door.
//
// REGIME BOUNDARY: No imports from crate::spo, crate::storage, crate::container,
// crate::graph::spo, or any mesh-side module. If you find yourself adding one, STOP.

pub mod ast;
pub mod error;
pub mod parser;

// Re-export the main entry point
pub use ast::*;
pub use error::{GraphError, Result};
pub use parser::parse_cypher_query;
