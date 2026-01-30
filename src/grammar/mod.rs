//! Grammar Module — The Universal Input Layer
//!
//! The Grammar Triangle transforms any text into a continuous semantic field
//! that can be encoded as a 10Kbit fingerprint for resonance operations.
//!
//! ## Architecture
//!
//! ```text
//!         CAUSALITY (agency, temporality)
//!               /\
//!              /  \
//!             /    \
//!     NSM <──⊕──> QUALIA (18D felt-sense)
//!   (65 primes)
//!         │
//!         ↓
//!    10Kbit FINGERPRINT
//! ```
//!
//! ## Components
//!
//! - **NSM** (Natural Semantic Metalanguage): 65 Wierzbicka semantic primitives
//!   found across all human languages. The irreducible building blocks of meaning.
//!
//! - **Causality**: Agency (active/passive), temporality (past/present/future),
//!   and dependency type (causal/enabling/preventing/etc.)
//!
//! - **Qualia**: 18-dimensional phenomenal field capturing the felt-sense of meaning
//!   (valence, arousal, certainty, intimacy, etc.)
//!
//! ## Usage
//!
//! ```rust
//! use ladybug::grammar::GrammarTriangle;
//!
//! // Parse text into a grammar triangle
//! let triangle = GrammarTriangle::from_text("I want to understand this deeply");
//!
//! // Get top activated NSM primitives
//! let top_nsm = triangle.top_nsm(3);
//! // e.g., [("WANT", 0.75), ("KNOW", 0.5), ("I", 0.25)]
//!
//! // Check qualia dimensions
//! let certainty = triangle.qualia("certainty").unwrap();
//! let depth = triangle.qualia("depth").unwrap();
//!
//! // Generate fingerprint for resonance
//! let fingerprint = triangle.to_fingerprint();
//!
//! // Compare two triangles
//! let other = GrammarTriangle::from_text("I desire to comprehend this thoroughly");
//! let similarity = triangle.similarity(&other);
//! ```

pub mod nsm;
pub mod causality;
pub mod qualia;
pub mod triangle;

pub use nsm::{NSMField, NSM_PRIMITIVES};
pub use causality::{CausalityFlow, DependencyType};
pub use qualia::{QualiaField, QUALIA_DIMENSIONS};
pub use triangle::{GrammarTriangle, TriangleSummary};
