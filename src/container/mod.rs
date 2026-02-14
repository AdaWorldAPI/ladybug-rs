//! Container-based cognitive record architecture.
//!
//! Every record is built from aligned 8,192-bit containers (128 × u64, 1 KB).
//! Container 0 is always metadata. Containers 1..N hold content whose
//! interpretation is determined by the geometry field in metadata.
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────────┐
//! │  Container 0  (1 KB)  METADATA: identity, NARS, edges, rung  │
//! ├───────────────────────────────────────────────────────────────┤
//! │  Container 1+ (1 KB)  CONTENT: CAM / XYZ / Bridge / Tree     │
//! └───────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Type Provenance
//!
//! The canonical `Container` type lives in `ladybug_contract::container`.
//! This module re-exports it and adds database-layer operations:
//! adjacency, cache, delta, graph, search, spine, traversal, etc.

pub mod adjacency;
pub mod cache;
pub mod delta;
pub mod dn_redis;
pub mod geometry;
pub mod graph;
pub mod insert;
pub mod meta;
pub mod migrate;
pub mod record;
pub mod search;
pub mod semiring;
pub mod spine;
#[cfg(test)]
pub mod tests;
pub mod traversal;

// ============================================================================
// RE-EXPORT: Container type + constants from contract crate (single source of truth)
// ============================================================================

pub use ladybug_contract::container::{
    Container,
    CONTAINER_AVX512_ITERS,
    CONTAINER_BITS,
    CONTAINER_BYTES,
    CONTAINER_WORDS,
    EXPECTED_DISTANCE,
    MAX_CONTAINERS,
    SIGMA,
    SIGMA_APPROX,
};

// Re-export primary types from local submodules
pub use cache::ContainerCache;
pub use geometry::ContainerGeometry;
pub use meta::{MetaView, MetaViewMut};
pub use record::CogRecord;

// ============================================================================
// CONVERSIONS: Container <-> Fingerprint
// ============================================================================
//
// These live here (not in contract) because Fingerprint is defined in the
// main crate, not the contract.

impl From<&crate::core::Fingerprint> for Container {
    /// Take the first 128 words of a 256-word Fingerprint.
    fn from(fp: &crate::core::Fingerprint) -> Self {
        let mut c = Container::zero();
        c.words.copy_from_slice(&fp.as_raw()[..CONTAINER_WORDS]);
        c
    }
}

impl From<&Container> for crate::core::Fingerprint {
    /// Promote a Container to a Fingerprint (zero-extend to 256 words).
    fn from(c: &Container) -> Self {
        let mut data = [0u64; crate::FINGERPRINT_U64];
        data[..CONTAINER_WORDS].copy_from_slice(&c.words);
        crate::core::Fingerprint::from_raw(data)
    }
}
