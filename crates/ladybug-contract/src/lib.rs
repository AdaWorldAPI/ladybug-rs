//! `ladybug-contract` — Substrate types for LadybugDB.
//!
//! This crate contains the pure data types that define LadybugDB's cognitive
//! address space. It has no I/O, no storage, no network — just types, bit
//! manipulation, and serde.
//!
//! ## Usage
//!
//! ```toml
//! [dependencies]
//! ladybug-contract = { git = "https://github.com/AdaWorldAPI/ladybug-rs" }
//! ```
//!
//! ## What's included
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`container`] | 8192-bit `Container` with XOR, Hamming, bundle ops |
//! | [`geometry`] | `ContainerGeometry` — how content containers are interpreted |
//! | [`record`] | `CogRecord` — metadata + N content containers |
//! | [`meta`] | `MetaView` / `MetaViewMut` — zero-copy metadata accessors |
//! | [`address`] | `CognitiveAddress(u64)` + domain/subtype enums |
//! | [`codebook`] | `OpCategory`, `OpType`, `OpSignature` — CAM opcodes |
//! | [`index_types`] | `Key`, `Entry`, type ID constants |
//! | [`nars`] | `TruthValue` with NAL truth functions |
//! | [`qualia`] | `QualiaChannel` (8 affect dimensions) |
//! | [`layers`] | 7-layer consciousness marker constants |
//! | [`temporal`] | `Version`, `VersionDiff`, temporal error types |
//! | [`delegation`] | `DelegationRequest` / `DelegationResponse` |
//! | [`wire`] | Binary wire protocol: `CogPacket` (8+8/4096 command packets, no JSON) |
//! | [`legacy`] | V1 JSON backward compatibility (external wire format only) |
//! | [`schema`] | Unified 2×8192 CogRecord schema constants and field descriptors |

pub mod container;
pub mod wide_container;
pub mod geometry;
pub mod meta;
pub mod record;
pub mod cogrecord8k;

pub mod address;
pub mod codebook;
pub mod index_types;

pub mod nars;
pub mod qualia;
pub mod layers;
pub mod temporal;

pub mod delegation;
pub mod legacy;
pub mod wire;
pub mod schema;

// === Convenience re-exports ===
pub use container::{Container, CONTAINER_BITS, CONTAINER_BYTES, CONTAINER_WORDS};
pub use wide_container::{WideContainer, EmbeddingFormat, WIDE_BITS, WIDE_BYTES, WIDE_WORDS};
pub use geometry::ContainerGeometry;
pub use record::CogRecord;
pub use cogrecord8k::{CogRecord8K, RECORD8K_BITS, RECORD8K_BYTES, SLOT_META, SLOT_CAM, SLOT_INDEX, SLOT_EMBED};
pub use meta::{MetaView, MetaViewMut};
pub use address::{CognitiveAddress, CognitiveDomain};
pub use codebook::OpCategory;
pub use nars::TruthValue;
pub use legacy::{
    V1DataEnvelope, V1EnvelopeMetadata, V1LadybugEnvelope, V1LadybugMetadata,
    V1StepDelegationRequest, V1StepDelegationResponse, V1StepStatus, V1UnifiedStep,
};
pub use wire::{CogPacket, WireError, COG_MAGIC, WIRE_VERSION};
