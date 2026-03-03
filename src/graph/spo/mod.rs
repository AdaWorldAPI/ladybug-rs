//! SPO 3D — Three-axis content-addressable graph.
//!
//! Subject (X) × Predicate (Y) × Object (Z) stored as sparse containers
//! within a single 2KB CogRecord. Forward, reverse, and relation queries
//! are native axis scans — no join, no edge table.
//!
//! ```text
//! Forward:  "What does Jan know?"    → scan X+Y → return Z
//! Reverse:  "Who knows Ada?"         → scan Z+Y → return X
//! Relation: "How are Jan+Ada linked?"→ scan X+Z → return Y
//! Causal:   Z₁ resonates with X₂    → causal chain link
//! ```

pub mod sparse;
pub mod scent;
pub mod store;
pub mod builder;
pub mod semiring;
pub mod merkle;
pub mod case_insensitive;

// Re-exports
pub use sparse::{SparseContainer, SpoError, AxisDescriptors, pack_axes, unpack_axes, spo_err};
pub use scent::NibbleScent;
pub use store::{SpoStore, QueryHit, QueryAxis, SpoHit, TruthGate};
pub use builder::{SpoBuilder, label_fp, dn_hash};
pub use semiring::{SpoSemiring, XorBundle, HammingMin, SimilarityMax, Reachability};
pub use merkle::{SpoMerkle, MerkleHash, MerkleError, AuthenticatedResult, InclusionProof, MerkleEpoch, TrajectoryStep, TrajectoryKind};
pub use case_insensitive::CaseInsensitiveLookup;
